#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>
#include <string>

using namespace weft;

namespace {

llvm::SmallVector<std::string> attributeStrings(mlir::ArrayAttr array) {
  llvm::SmallVector<std::string> result;
  if (!array)
    return result;
  for (mlir::Attribute attribute : array)
    result.push_back(mlir::cast<mlir::StringAttr>(attribute).getValue().str());
  return result;
}

class ConstrainRISCVResourcesPass final
    : public mlir::PassWrapper<ConstrainRISCVResourcesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConstrainRISCVResourcesPass)

  llvm::StringRef getArgument() const final {
    return "weft-riscv-constrain-resources";
  }
  llvm::StringRef getDescription() const final {
    return "form liveness, register, spill and loop-local schedule backedges";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::Builder builder(module.getContext());
    for (riscv::ProblemOp problem : module.getOps<riscv::ProblemOp>()) {
      if (problem.getStage() != "instructions") {
        problem.emitError(
            "resource constraints require an instructions-stage problem");
        signalPassFailure();
        return;
      }
      int64_t vectorRegisters =
          *riscv_internal::integer(problem.getTarget(), "vector_registers");
      int64_t stackBudget =
          *riscv_internal::integer(problem.getTarget(),
                                   "max_private_stack_bytes");

      llvm::StringMap<int64_t> first;
      llvm::StringMap<int64_t> last;
      std::string pipelineLevel = "not-applicable";
      for (auto [ordinal, attribute] : llvm::enumerate(problem.getOperations())) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef name =
            mlir::cast<mlir::StringAttr>(operation.get("name")).getValue();
        if (pipelineLevel == "not-applicable" &&
            (name == "weft_kernel.mac_pairs" ||
             name == "weft_kernel.mac_groups" || name == "weft_kernel.dot" ||
             name == "weft_kernel.contract" ||
             name == "weft_kernel.outer_contract")) {
          auto path = operation.getAs<mlir::ArrayAttr>("level_path");
          if (path && !path.empty())
            pipelineLevel =
                mlir::cast<mlir::StringAttr>(path[path.size() - 1])
                    .getValue()
                    .str();
        }
        for (llvm::StringRef key : {"operands", "results"}) {
          for (const std::string &id :
               attributeStrings(operation.getAs<mlir::ArrayAttr>(key))) {
            first.try_emplace(id, ordinal);
            last[id] = ordinal;
          }
        }
      }
      llvm::SmallVector<std::string> prefetchSources;
      if (pipelineLevel != "not-applicable") {
        for (mlir::Attribute attribute : problem.getOperations()) {
          auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
          llvm::StringRef name =
              mlir::cast<mlir::StringAttr>(operation.get("name")).getValue();
          if (name != "weft_kernel.field" && name != "weft_kernel.admit" &&
              name != "weft_kernel.lookup")
            continue;
          auto path = operation.getAs<mlir::ArrayAttr>("level_path");
          if (!path || path.empty() ||
              mlir::cast<mlir::StringAttr>(path[path.size() - 1]).getValue() !=
                  pipelineLevel)
            continue;
          prefetchSources.push_back(
              mlir::cast<mlir::StringAttr>(operation.get("id")).getValue().str());
        }
      }
      llvm::SmallVector<mlir::Attribute> liveValues;
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef id =
            mlir::cast<mlir::StringAttr>(value.get("id")).getValue();
        int64_t begin = first.contains(id) ? first.lookup(id) : 0;
        int64_t end = last.contains(id) ? last.lookup(id) : begin;
        value = riscv_internal::set(value, "live_start",
                                    builder.getI64IntegerAttr(begin));
        value = riscv_internal::set(value, "live_end",
                                    builder.getI64IntegerAttr(end));
        liveValues.push_back(value);
      }

      mlir::DictionaryAttr domains = problem.getDecisionDomains();
      auto laneAxes = domains.getAs<mlir::DenseI64ArrayAttr>("lane_axis");
      bool hasVectorLane = laneAxes && llvm::any_of(
                                            laneAxes.asArrayRef(),
                                            [](int64_t axis) { return axis > 0; });
      bool hasPipelineLoop = pipelineLevel != "not-applicable";
      bool hasPrefetchSource = !prefetchSources.empty();
      llvm::SmallVector<int64_t> groups;
      if (auto cohorts = domains.getAs<mlir::DenseI64ArrayAttr>("cohort")) {
        for (int64_t cohort : cohorts.asArrayRef())
          for (int64_t candidate : {1, 2, 4, 8, 16})
            if (candidate <= cohort && cohort % candidate == 0)
              groups.push_back(candidate);
      }
      llvm::sort(groups);
      groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
      if (groups.empty())
        groups.push_back(1);
      domains = riscv_internal::set(
          domains, "accumulator_groups",
          riscv_internal::integers(builder, groups));
      domains = riscv_internal::set(
          domains, "prefetch_distance",
          riscv_internal::integers(builder,
                                   hasVectorLane && hasPipelineLoop &&
                                           hasPrefetchSource
                                       ? llvm::SmallVector<int64_t>{0, 1, 2}
                                       : llvm::SmallVector<int64_t>{0}));
      domains = riscv_internal::set(
          domains, "pipeline_depth",
          riscv_internal::integers(builder,
                                   hasVectorLane && hasPipelineLoop
                                       ? llvm::SmallVector<int64_t>{1, 2}
                                       : llvm::SmallVector<int64_t>{1}));
      domains = riscv_internal::set(
          domains, "unroll",
          riscv_internal::integers(builder,
                                   hasVectorLane && hasPipelineLoop
                                       ? llvm::SmallVector<int64_t>{1, 2, 4}
                                       : llvm::SmallVector<int64_t>{1}));
      domains = riscv_internal::set(
          domains, "spill",
          riscv_internal::strings(
              builder, hasVectorLane
                           ? llvm::SmallVector<std::string>{"none", "stack"}
                           : llvm::SmallVector<std::string>{"none"}));

      mlir::DictionaryAttr resources = problem.getResourceModel();
      resources = riscv_internal::set(
          resources, "vector_registers",
          builder.getI64IntegerAttr(vectorRegisters));
      resources = riscv_internal::set(
          resources, "reserved_vector_groups", builder.getI64IntegerAttr(2));
      resources = riscv_internal::set(
          resources, "max_private_stack_bytes",
          builder.getI64IntegerAttr(stackBudget));
      resources = riscv_internal::set(resources, "liveness",
                                      builder.getStringAttr("ssa-use-interval"));
      resources = riscv_internal::set(
          resources, "pipeline_buffers",
          builder.getStringAttr("counted-at-peak"));
      resources = riscv_internal::set(resources, "pipeline_level",
                                      builder.getStringAttr(pipelineLevel));
      resources = riscv_internal::set(
          resources, "prefetch_sources",
          riscv_internal::strings(builder, prefetchSources));
      resources = riscv_internal::set(
          resources, "spill_policy",
          builder.getStringAttr("explicit-candidate-only"));
      llvm::SmallVector<std::string> constraints;
      for (mlir::Attribute attribute : problem.getConstraints())
        constraints.push_back(
            mlir::cast<mlir::StringAttr>(attribute).getValue().str());
      constraints.push_back(
          "lmul x live register-bundles + buffers <= architectural groups");
      constraints.push_back(
          "pipeline-depth and unroll feed back into operand-buffer pressure");
      constraints.push_back(
          "spill is a selected local representation, never an emitter surprise");
      problem.setValuesAttr(builder.getArrayAttr(liveValues));
      problem.setDecisionDomainsAttr(domains);
      problem.setResourceModelAttr(resources);
      problem.setConstraintsAttr(riscv_internal::strings(builder, constraints));
      problem.setStageAttr(builder.getStringAttr("resources"));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createConstrainRISCVResourcesPass() {
  return std::make_unique<ConstrainRISCVResourcesPass>();
}
