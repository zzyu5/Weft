#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
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
  auto value = bindings ? bindings.getAs<mlir::IntegerAttr>(partition)
                        : mlir::IntegerAttr();
  return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
}

class ScheduleRISCVLevelsPass final
    : public mlir::PassWrapper<ScheduleRISCVLevelsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleRISCVLevelsPass)

  llvm::StringRef getArgument() const final {
    return "weft-riscv-schedule-levels";
  }
  llvm::StringRef getDescription() const final {
    return "assign Level-local iteration, tail, unroll and pipeline schedules";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::Builder builder(module.getContext());
    for (riscv::ProblemOp problem : module.getOps<riscv::ProblemOp>()) {
      if (problem.getStage() == "invalid")
        continue;
      if (problem.getStage() != "operations") {
        problem.emitError("Level scheduling requires selected local operations");
        signalPassFailure();
        return;
      }
      llvm::StringMap<mlir::DictionaryAttr> values;
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        values[*riscv_internal::string(value, "id")] = value;
      }
      llvm::SmallVector<mlir::Attribute> operations;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        if (riscv_internal::string(operation, "name").value_or("") !=
            "weft_kernel.level") {
          operations.push_back(operation);
          continue;
        }
        auto operands = operation.getAs<mlir::ArrayAttr>("operands");
        auto domainId = operands && !operands.empty()
                            ? mlir::dyn_cast<mlir::StringAttr>(operands[0])
                            : mlir::StringAttr();
        auto domain = domainId ? values.find(domainId.getValue()) : values.end();
        if (domain == values.end()) {
          problem.emitError("Level has no canonical domain facts");
          signalPassFailure();
          return;
        }
        int64_t axis =
            riscv_internal::integer(domain->second, "domain_axis").value_or(0);
        llvm::StringRef relation =
            riscv_internal::string(domain->second, "domain_relation").value_or("");
        llvm::StringRef extent =
            riscv_internal::string(domain->second, "domain_extent").value_or("");
        llvm::StringRef partition =
            riscv_internal::string(domain->second, "domain_partition").value_or("");
        llvm::StringRef multiplicity =
            riscv_internal::string(domain->second, "domain_multiplicity").value_or("");
        llvm::StringRef tail =
            riscv_internal::string(domain->second, "domain_tail").value_or("");
        auto physicalPartition = resolvePartition(partition, problem.getCandidate());
        if (!physicalPartition) {
          problem.emitError("Level partition is not fixed by this candidate");
          signalPassFailure();
          return;
        }
        int64_t physicalLanes = 1;
        for (const auto &entry : values)
          if (riscv_internal::integer(entry.getValue(), "lane_axis").value_or(0) ==
              axis)
            physicalLanes = std::max(
                physicalLanes,
                riscv_internal::integer(entry.getValue(), "physical_lanes")
                    .value_or(1));
        bool lane = physicalLanes > 1 && *physicalPartition > 1;
        physicalLanes = lane ? std::min(physicalLanes, *physicalPartition) : 1;
        auto mapping = riscv_internal::dictionary(
            builder,
            {{"axis", builder.getI64IntegerAttr(axis)},
             {"relation", builder.getStringAttr(relation)},
             {"extent", builder.getStringAttr(extent)},
             {"partition", builder.getStringAttr(partition)},
             {"multiplicity", builder.getStringAttr(multiplicity)},
             {"logical_tail", builder.getStringAttr(tail)},
             {"physical_iteration",
              builder.getStringAttr(
                  lane ? (*physicalPartition > physicalLanes ? "rvv-stream"
                                                             : "rvv-lane")
                       : "ordered-sequential")},
             {"physical_lanes", builder.getI64IntegerAttr(physicalLanes)},
             {"active_extent",
              builder.getStringAttr(
                  lane ? "min(" + std::to_string(physicalLanes) +
                             ",remaining(" + extent.str() + "))"
                       : "clamp(" + extent.str() + "-point*" +
                             partition.str() + ",0," + partition.str() + ")")},
             {"tail_policy",
              builder.getStringAttr(
                  lane ? "strip-at-physical-lanes-and-set-vl"
                       : "explicit-multiplicity-with-parent-validity")},
             {"decision_owner", builder.getStringAttr("Level")}});
        auto schedule = riscv_internal::dictionary(
            builder,
            {{"unroll", builder.getI64IntegerAttr(1)},
             {"pipeline_depth", builder.getI64IntegerAttr(1)},
             {"prefetch_distance", builder.getI64IntegerAttr(0)},
             {"decision_owner", builder.getStringAttr("Level")}});
        operation = riscv_internal::set(operation, "level_mapping", mapping);
        operation = riscv_internal::set(operation, "schedule", schedule);
        operation = riscv_internal::set(
            operation, "realization",
            builder.getStringAttr(lane ? "wide-lane-level"
                                       : "ordered-sequential-level"));
        operations.push_back(operation);
      }
      problem.setOperationsAttr(builder.getArrayAttr(operations));
      problem.setStageAttr(builder.getStringAttr("schedule"));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createScheduleRISCVLevelsPass() {
  return std::make_unique<ScheduleRISCVLevelsPass>();
}
