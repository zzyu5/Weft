#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"

#include <string>

using namespace weft;

namespace {

std::optional<int64_t> fixedPartition(kernel::DomainType domain) {
  int64_t value = 0;
  if (!domain.getPartition().getAsInteger(10, value) && value > 0)
    return value;
  return std::nullopt;
}

class ConstrainRISCVRepresentationsPass final
    : public mlir::PassWrapper<ConstrainRISCVRepresentationsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ConstrainRISCVRepresentationsPass)

  llvm::StringRef getArgument() const final {
    return "weft-riscv-constrain-representations";
  }
  llvm::StringRef getDescription() const final {
    return "propagate SEW, LMUL, lane and ordered-control domains along SSA chains";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::Builder builder(module.getContext());
    for (riscv::ProblemOp problem :
         llvm::make_early_inc_range(module.getOps<riscv::ProblemOp>())) {
      if (problem.getStage() != "facts") {
        problem.emitError("representation constraints require a facts-stage problem");
        signalPassFailure();
        return;
      }
      kernel::KernelOp kernel =
          riscv_internal::findKernel(module, problem.getKernelAttr());
      if (!kernel) {
        problem.emitError("referenced canonical kernel does not exist");
        signalPassFailure();
        return;
      }

      llvm::SmallSet<int64_t, 4> laneAxes;
      llvm::SmallSet<int64_t, 8> cohorts;
      llvm::DenseMap<int64_t, llvm::SmallVector<kernel::DomainType, 2>>
          domainsByAxis;
      llvm::StringMap<mlir::DictionaryAttr> valuesById;
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        valuesById[*riscv_internal::string(value, "id")] = value;
      }
      auto addPartition = [&](kernel::DomainType domain) {
        if (auto partition = fixedPartition(domain)) {
          if (*partition > 1)
            cohorts.insert(*partition);
          return;
        }
        if (!domain.getPartition().starts_with("auto:"))
          return;
        llvm::StringRef name = domain.getPartition().drop_front(5);
        for (mlir::Attribute candidateAttribute : problem.getCandidates()) {
          auto candidate = mlir::cast<mlir::DictionaryAttr>(candidateAttribute);
          auto bindings = candidate.getAs<mlir::DictionaryAttr>("auto_bindings");
          if (bindings)
            if (auto value = bindings.getAs<mlir::IntegerAttr>(name);
                value && value.getInt() > 1)
              cohorts.insert(value.getInt());
        }
      };
      kernel.walk([&](kernel::DomainOp domainOperation) {
        kernel::DomainType domain = domainOperation.getResult().getType();
        domainsByAxis[domain.getAxisId()].push_back(domain);
        if (domain.getRelation() != "rows" && domain.getRelation() != "cols")
          return;
        auto partition = fixedPartition(domain);
        if ((partition && *partition > 1) ||
            domain.getPartition().starts_with("auto:")) {
          laneAxes.insert(domain.getAxisId());
          addPartition(domain);
        }
      });
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        if (riscv_internal::string(operation, "engine").value_or("") != "wide")
          continue;
        for (llvm::StringRef key : {"operands", "results"}) {
          auto ids = operation.getAs<mlir::ArrayAttr>(key);
          if (!ids)
            continue;
          for (mlir::Attribute idAttribute : ids) {
            auto found = valuesById.find(
                mlir::cast<mlir::StringAttr>(idAttribute).getValue());
            if (found == valuesById.end())
              continue;
            auto axes = found->second.getAs<mlir::DenseI64ArrayAttr>("axes");
            if (!axes)
              continue;
            for (int64_t axis : axes.asArrayRef()) {
              auto domains = domainsByAxis.find(axis);
              if (domains == domainsByAxis.end())
                continue;
              for (kernel::DomainType domain : domains->second) {
                auto partition = fixedPartition(domain);
                if (partition && *partition <= 1)
                  continue;
                laneAxes.insert(axis);
                addPartition(domain);
              }
            }
          }
        }
      }
      if (laneAxes.empty()) {
        laneAxes.insert(-1);
        cohorts.insert(1);
      }

      auto targetLMUL = problem.getTarget().getAs<mlir::DenseI64ArrayAttr>(
          "legal_lmul_eighths");
      llvm::SmallVector<mlir::Attribute> plannedValues;
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        auto type = mlir::cast<mlir::TypeAttr>(value.get("type")).getValue();
        llvm::StringRef kind =
            mlir::cast<mlir::StringAttr>(value.get("kind")).getValue();
        llvm::SmallVector<std::string> representations;
        llvm::SmallVector<int64_t> sew;
        llvm::SmallVector<int64_t> lmul;
        if (kind == "view" || kind == "slice" || kind == "encoded_value") {
          representations.push_back("memory");
        } else if (kind == "control") {
          representations.push_back("control");
        } else if (kind == "scalar") {
          representations.push_back("scalar");
          unsigned bits = riscv_internal::logicalBitWidth(type);
          if (bits)
            sew.push_back(bits);
        } else if (kind == "local_value") {
          bool hasLaneAxis = llvm::any_of(
              riscv_internal::logicalAxes(type),
              [&](int64_t axis) { return laneAxes.contains(axis); });
          representations.push_back(hasLaneAxis ? "rvv-lane" : "sequential");
          if (hasLaneAxis && riscv_internal::logicalShape(type).size() > 1)
            representations.push_back("rvv-stream");
          unsigned bits = riscv_internal::logicalBitWidth(type);
          if (bits)
            sew.push_back(std::max(8u, bits));
          if (hasLaneAxis && targetLMUL)
            lmul.append(targetLMUL.asArrayRef().begin(),
                        targetLMUL.asArrayRef().end());
        } else {
          problem.emitError("unknown canonical value kind in representation pass");
          signalPassFailure();
          return;
        }
        value = riscv_internal::set(
            value, "representation_domain",
            riscv_internal::strings(builder, representations));
        value = riscv_internal::set(value, "sew_domain",
                                    riscv_internal::integers(builder, sew));
        value = riscv_internal::set(value, "lmul_domain_eighths",
                                    riscv_internal::integers(builder, lmul));
        plannedValues.push_back(value);
      }

      llvm::SmallVector<std::string> constraints;
      for (mlir::Attribute attribute : problem.getConstraints())
        constraints.push_back(
            mlir::cast<mlir::StringAttr>(attribute).getValue().str());
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef name =
            mlir::cast<mlir::StringAttr>(operation.get("name")).getValue();
        llvm::StringRef id =
            mlir::cast<mlir::StringAttr>(operation.get("id")).getValue();
        if (name == "weft_kernel.widen")
          constraints.push_back(id.str() +
                                ": preserve-lanes; result-sew widens; lmul propagates");
        else if (name == "weft_kernel.binary" ||
                 name == "weft_kernel.unary" ||
                 name == "weft_kernel.cast")
          constraints.push_back(id.str() +
                                ": pointwise values share one lane mapping");
        else if (name == "weft_kernel.reduce" ||
                 name == "weft_kernel.fold2")
          constraints.push_back(id.str() +
                                ": closed reduction preserves remaining axes");
        else if (name == "weft_kernel.for" || name == "weft_kernel.if" ||
                 name == "weft_kernel.while")
          constraints.push_back(id.str() +
                                ": ordered-scalar; no automatic lane mapping");
      }

      llvm::SmallVector<int64_t> laneAxisValues(laneAxes.begin(), laneAxes.end());
      llvm::SmallVector<int64_t> cohortValues(cohorts.begin(), cohorts.end());
      llvm::sort(laneAxisValues);
      llvm::sort(cohortValues);
      mlir::DictionaryAttr domains = problem.getDecisionDomains();
      domains = riscv_internal::set(
          domains, "lane_axis", riscv_internal::integers(builder, laneAxisValues));
      domains = riscv_internal::set(
          domains, "cohort", riscv_internal::integers(builder, cohortValues));
      domains = riscv_internal::set(
          domains, "lmul_multiplier", riscv_internal::integers(builder, {1, 2}));
      problem.setValuesAttr(builder.getArrayAttr(plannedValues));
      problem.setConstraintsAttr(riscv_internal::strings(builder, constraints));
      problem.setDecisionDomainsAttr(domains);
      problem.setStageAttr(builder.getStringAttr("representations"));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
weft::createConstrainRISCVRepresentationsPass() {
  return std::make_unique<ConstrainRISCVRepresentationsPass>();
}
