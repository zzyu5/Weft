#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>
#include <optional>
#include <string>

using namespace weft;

namespace {

std::optional<int64_t> resolvePartition(llvm::StringRef partition,
                                        mlir::DictionaryAttr candidate) {
  int64_t fixed = 0;
  if (!partition.getAsInteger(10, fixed) && fixed > 0)
    return fixed;
  if (!partition.consume_front("auto:"))
    return std::nullopt;
  auto bindings = candidate.getAs<mlir::DictionaryAttr>("auto_bindings");
  if (!bindings)
    return std::nullopt;
  auto value = bindings.getAs<mlir::IntegerAttr>(partition);
  return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
}

bool containsAxis(mlir::DictionaryAttr value, int64_t axis) {
  auto axes = value.getAs<mlir::DenseI64ArrayAttr>("axes");
  return axes && llvm::is_contained(axes.asArrayRef(), axis);
}

int64_t laneExtent(mlir::DictionaryAttr value, int64_t axis) {
  auto axes = value.getAs<mlir::DenseI64ArrayAttr>("axes");
  auto shape = value.getAs<mlir::DenseI64ArrayAttr>("shape");
  if (!axes || !shape)
    return 1;
  for (auto [valueAxis, extent] : llvm::zip(axes.asArrayRef(), shape.asArrayRef()))
    if (valueAxis == axis)
      return std::max<int64_t>(1, extent);
  return 1;
}

std::string lmulSpelling(int64_t eighths) {
  return eighths < 8 ? "mf" + std::to_string(8 / eighths)
                     : "m" + std::to_string(eighths / 8);
}

std::optional<int64_t> chooseLMUL(int64_t lanes, int64_t sew,
                                  int64_t vlenBits,
                                  mlir::DenseI64ArrayAttr legal) {
  if (lanes <= 0 || sew <= 0 || vlenBits <= 0 || !legal)
    return std::nullopt;
  int64_t required = (lanes * sew * 8 + vlenBits - 1) / vlenBits;
  for (int64_t candidate : legal.asArrayRef())
    if (candidate >= required)
      return candidate;
  return std::nullopt;
}

std::string physicalKind(mlir::DictionaryAttr value, int64_t laneAxis,
                         int64_t physicalLanes) {
  llvm::StringRef kind = *riscv_internal::string(value, "kind");
  if (kind == "view" || kind == "slice")
    return "memory";
  if (kind == "encoded_value")
    return "encoded-record";
  if (kind == "control")
    return "control";
  if (kind == "scalar")
    return "scalar";
  if (riscv_internal::integer(value, "logical_sew").value_or(0) == 0)
    return "encoded-record";
  if (!containsAxis(value, laneAxis))
    return "sequential";
  auto shape = value.getAs<mlir::DenseI64ArrayAttr>("shape");
  return (shape && shape.size() > 1) || laneExtent(value, laneAxis) > physicalLanes
             ? "rvv-stream"
             : "rvv-lane";
}

void invalidate(riscv::ProblemOp problem, mlir::Builder &builder,
                llvm::StringRef reason) {
  problem.setResourcesAttr(riscv_internal::dictionary(
      builder, {{"invalid_reason", builder.getStringAttr(reason)}}));
  problem.setStageAttr(builder.getStringAttr("invalid"));
}

class AssignRISCVRepresentationsPass final
    : public mlir::PassWrapper<AssignRISCVRepresentationsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignRISCVRepresentationsPass)

  llvm::StringRef getArgument() const final {
    return "weft-riscv-assign-representations";
  }
  llvm::StringRef getDescription() const final {
    return "assign entity-local SEW, LMUL, vl and materialization forward";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::Builder builder(module.getContext());
    for (riscv::ProblemOp problem : module.getOps<riscv::ProblemOp>()) {
      if (problem.getStage() == "invalid")
        continue;
      if (problem.getStage() != "facts") {
        problem.emitError("representation pass requires a facts-stage candidate");
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

      llvm::StringMap<mlir::DictionaryAttr> valuesById;
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        valuesById[*riscv_internal::string(value, "id")] = value;
      }
      llvm::SmallSet<int64_t, 8> wideAxes;
      llvm::StringMap<int64_t> useCounts;
      llvm::StringMap<std::string> producers;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef name = *riscv_internal::string(operation, "name");
        if (auto operands = operation.getAs<mlir::ArrayAttr>("operands"))
          for (mlir::Attribute operand : operands)
            ++useCounts[mlir::cast<mlir::StringAttr>(operand).getValue()];
        if (auto results = operation.getAs<mlir::ArrayAttr>("results"))
          for (mlir::Attribute result : results)
            producers[mlir::cast<mlir::StringAttr>(result).getValue()] = name.str();
        if (riscv_internal::string(operation, "engine").value_or("") != "wide")
          continue;
        for (llvm::StringRef key : {"operands", "results"})
          if (auto ids = operation.getAs<mlir::ArrayAttr>(key))
            for (mlir::Attribute id : ids) {
              auto found = valuesById.find(
                  mlir::cast<mlir::StringAttr>(id).getValue());
              if (found == valuesById.end())
                continue;
              if (auto axes = found->second.getAs<mlir::DenseI64ArrayAttr>("axes"))
                for (int64_t axis : axes.asArrayRef())
                  wideAxes.insert(axis);
            }
      }

      struct AxisChoice {
        int priority = 2;
        int64_t axis = -1;
        int64_t cohort = 1;
      } selected;
      kernel.walk([&](kernel::DomainOp domainOp) {
        kernel::DomainType domain = domainOp.getResult().getType();
        if (!wideAxes.contains(domain.getAxisId()))
          return;
        auto partition = resolvePartition(domain.getPartition(), problem.getCandidate());
        if (!partition || *partition <= 1)
          return;
        int priority =
            (domain.getRelation() == "rows" || domain.getRelation() == "cols") ? 0 : 1;
        if (selected.axis < 0 || priority < selected.priority) {
          selected = {priority, domain.getAxisId(), *partition};
        } else if (domain.getAxisId() == selected.axis) {
          selected.cohort = std::max(selected.cohort, *partition);
        }
      });

      int64_t physicalLanes = 1;
      int64_t vlenBits =
          *riscv_internal::integer(problem.getTarget(), "vlen_bits");
      auto legalLMUL = problem.getTarget().getAs<mlir::DenseI64ArrayAttr>(
          "legal_lmul_eighths");
      if (selected.axis >= 0) {
        int64_t maximumSEW = 0;
        for (mlir::Attribute attribute : problem.getValues()) {
          auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
          if (containsAxis(value, selected.axis))
            maximumSEW = std::max<int64_t>(
                maximumSEW,
                std::max<int64_t>(8, riscv_internal::integer(value, "logical_sew")
                                         .value_or(0)));
        }
        if (!maximumSEW || !legalLMUL || legalLMUL.empty()) {
          invalidate(problem, builder, "lane axis has no legal target representation");
          continue;
        }
        int64_t maximumLMUL = *llvm::max_element(legalLMUL.asArrayRef());
        physicalLanes = std::min(
            selected.cohort, vlenBits * maximumLMUL / (maximumSEW * 8));
        if (physicalLanes <= 0) {
          invalidate(problem, builder, "target has no lanes for selected value widths");
          continue;
        }
      }

      llvm::SmallVector<mlir::Attribute> assignedValues;
      bool legal = true;
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        std::string id = riscv_internal::string(value, "id")->str();
        int64_t logicalSEW =
            riscv_internal::integer(value, "logical_sew").value_or(0);
        std::string kind = physicalKind(value, selected.axis, physicalLanes);
        bool lane = selected.axis >= 0 && containsAxis(value, selected.axis) &&
                    logicalSEW > 0;
        int64_t valueLanes = lane ? std::min(laneExtent(value, selected.axis),
                                             physicalLanes)
                                  : 1;
        int64_t physicalSEW = logicalSEW ? std::max<int64_t>(8, logicalSEW) : 0;
        int64_t lmul = 0;
        if (lane) {
          auto selectedLMUL =
              chooseLMUL(valueLanes, physicalSEW, vlenBits, legalLMUL);
          if (!selectedLMUL) {
            legal = false;
            break;
          }
          lmul = *selectedLMUL;
        }
        int64_t streamParts =
            lane ? std::max<int64_t>(
                       1, (laneExtent(value, selected.axis) + valueLanes - 1) /
                              valueLanes)
                 : 1;
        int64_t registerGroups = lmul ? (lmul + 7) / 8 : 0;
        bool shareAdmitted = producers.lookup(id) == "weft_kernel.admit" &&
                             useCounts.lookup(id) > 1 &&
                             registerGroups * streamParts <= 8;
        value = riscv_internal::set(value, "physical_kind",
                                    builder.getStringAttr(kind));
        value = riscv_internal::set(
            value, "physical_encoding_kind",
            builder.getStringAttr(
                riscv_internal::string(value, "encoding_kind").value_or("") ==
                        "derived_family"
                    ? "derived_instance"
                    : riscv_internal::string(value, "encoding_kind")
                          .value_or("not-applicable")));
        value = riscv_internal::set(
            value, "physical_sew", builder.getI64IntegerAttr(physicalSEW));
        value = riscv_internal::set(
            value, "lane_axis",
            builder.getI64IntegerAttr(lane ? selected.axis : 0));
        value = riscv_internal::set(
            value, "physical_lanes", builder.getI64IntegerAttr(valueLanes));
        value = riscv_internal::set(
            value, "stream_parts", builder.getI64IntegerAttr(streamParts));
        value = riscv_internal::set(
            value, "lmul_eighths", builder.getI64IntegerAttr(lmul));
        value = riscv_internal::set(
            value, "lmul",
            builder.getStringAttr(lmul ? lmulSpelling(lmul) : "none"));
        value = riscv_internal::set(
            value, "vl",
            builder.getStringAttr(lane ? "min(" + std::to_string(valueLanes) +
                                             ",remaining-axis" +
                                             std::to_string(selected.axis) + ")"
                                       : "1"));
        value = riscv_internal::set(
            value, "register_groups",
            builder.getI64IntegerAttr(registerGroups));
        value = riscv_internal::set(
            value, "storage",
            builder.getStringAttr(
                kind == "memory"          ? "pinned-memory"
                : kind == "control"       ? "control"
                : kind == "scalar"        ? "scalar-register"
                : kind == "encoded-record" ? "local-record"
                : kind == "sequential"    ? "scalar-or-rematerialized"
                                           : "vector-register"));
        value = riscv_internal::set(
            value, "materialization",
            builder.getStringAttr(producers.lookup(id) == "weft_kernel.admit"
                                      ? (shareAdmitted ? "shared-register"
                                                       : "reload-per-use")
                                      : "not-applicable"));
        value = riscv_internal::set(
            value, "decision_owner", builder.getStringAttr("value"));
        assignedValues.push_back(value);
      }
      if (!legal) {
        invalidate(problem, builder, "one value has no legal SEW/LMUL representation");
        continue;
      }
      problem.setValuesAttr(builder.getArrayAttr(assignedValues));
      problem.setStageAttr(builder.getStringAttr("representations"));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createAssignRISCVRepresentationsPass() {
  return std::make_unique<AssignRISCVRepresentationsPass>();
}
