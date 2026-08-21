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

mlir::DictionaryAttr scheduleForLevel(mlir::Builder &builder,
                                      mlir::DictionaryAttr candidate,
                                      llvm::StringRef levelId) {
  auto bindings = candidate.getAs<mlir::DictionaryAttr>("schedule_bindings");
  auto selected =
      bindings ? bindings.getAs<mlir::DictionaryAttr>(levelId)
               : mlir::DictionaryAttr();
  if (!selected)
    return riscv_internal::dictionary(
        builder,
        {{"unroll", builder.getI64IntegerAttr(1)},
         {"pipeline_depth", builder.getI64IntegerAttr(1)},
         {"loop_structure", builder.getStringAttr("sequential-stream")},
         {"derived_from",
          builder.getStringAttr("no schedulable local cluster in this Level")},
         {"decision_owner", builder.getStringAttr("Level")}});
  return riscv_internal::dictionary(
      builder,
      {{"unroll", selected.get("unroll")},
       {"pipeline_depth", selected.get("pipeline_depth")},
       {"loop_structure",
        builder.getStringAttr(
            riscv_internal::integer(selected, "pipeline_depth").value_or(0) == 2
                ? "cross-iteration-double-buffer"
                : "sequential-stream")},
       {"derived_from",
        builder.getStringAttr("concrete outer auto candidate for " +
                              levelId.str())},
       {"decision_owner", builder.getStringAttr("Level")}});
}

mlir::DictionaryAttr addLocalTemporaryBudget(mlir::Builder &builder,
                                             mlir::DictionaryAttr operation,
                                             mlir::DictionaryAttr schedule) {
  llvm::StringRef name =
      riscv_internal::string(operation, "name").value_or("");
  if (name != "weft_kernel.mac_pairs" && name != "weft_kernel.mac_groups")
    return operation;
  auto local = operation.getAs<mlir::DictionaryAttr>("local_operation");
  auto source = operation.getAs<mlir::DictionaryAttr>("source_attributes");
  if (!local || !source)
    return operation;
  int64_t group = riscv_internal::integer(source, "group").value_or(0);
  int64_t rawLMUL =
      riscv_internal::integer(local, "lhs_lmul_eighths").value_or(0);
  int64_t partialLMUL =
      riscv_internal::integer(local, "partial_lmul_eighths").value_or(0);
  int64_t unroll = riscv_internal::integer(schedule, "unroll").value_or(1);
  int64_t pipeline =
      riscv_internal::integer(schedule, "pipeline_depth").value_or(1);
  if (group <= 0 || rawLMUL <= 0 || partialLMUL <= 0)
    return operation;
  int64_t rawGroups = (rawLMUL + 7) / 8;
  int64_t partialGroups = (partialLMUL + 7) / 8;
  int64_t temporaryGroups = (unroll - 1) * partialGroups;
  if (pipeline > 1)
    temporaryGroups += 2 * unroll * group * rawGroups;
  local = riscv_internal::set(
      local, "temporary_vector_groups",
      builder.getI64IntegerAttr(temporaryGroups));
  local = riscv_internal::set(
      local, "temporary_groups_derived_from",
      builder.getStringAttr(
          "additional unroll partials + current/next decoded operand banks"));
  return riscv_internal::set(operation, "local_operation", local);
}

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
      llvm::StringMap<mlir::DictionaryAttr> schedulesByLevel;
      bool schedulesLegal = true;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        if (riscv_internal::string(operation, "name").value_or("") !=
            "weft_kernel.level")
          continue;
        llvm::StringRef levelId = *riscv_internal::string(operation, "id");
        mlir::DictionaryAttr schedule =
            scheduleForLevel(builder, problem.getCandidate(), levelId);
        int64_t pipeline =
            riscv_internal::integer(schedule, "pipeline_depth").value_or(0);
        if (pipeline < 1 || pipeline > 2) {
          schedulesLegal = false;
          break;
        }
        schedulesByLevel[levelId] = schedule;
      }
      if (!schedulesLegal) {
        problem.setResourcesAttr(riscv_internal::dictionary(
            builder,
            {{"invalid_reason",
              builder.getStringAttr(
                  "selected pipeline depth has no generated local schedule")}}));
        problem.setStageAttr(builder.getStringAttr("invalid"));
        continue;
      }

      llvm::SmallVector<mlir::Attribute> operations;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        if (riscv_internal::string(operation, "name").value_or("") !=
            "weft_kernel.level") {
          auto path = operation.getAs<mlir::ArrayAttr>("level_path");
          mlir::DictionaryAttr inherited;
          std::string inheritedLevel;
          if (path)
            for (mlir::Attribute level : llvm::reverse(path)) {
              llvm::StringRef id =
                  mlir::cast<mlir::StringAttr>(level).getValue();
              auto found = schedulesByLevel.find(id);
              if (found == schedulesByLevel.end())
                continue;
              inherited = found->second;
              inheritedLevel = id.str();
              break;
            }
          if (inherited &&
              riscv_internal::string(inherited, "derived_from")
                  .value_or("")
                  .starts_with("concrete outer auto candidate")) {
            operation = riscv_internal::set(operation, "schedule", inherited);
            operation = riscv_internal::set(
                operation, "schedule_level",
                builder.getStringAttr(inheritedLevel));
            operation = addLocalTemporaryBudget(builder, operation, inherited);
          }
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
        llvm::StringRef levelId = *riscv_internal::string(operation, "id");
        mlir::DictionaryAttr schedule = schedulesByLevel.lookup(levelId);
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
