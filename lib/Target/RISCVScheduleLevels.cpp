#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <functional>
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
  auto local = operation.getAs<mlir::DictionaryAttr>("local_operation");
  llvm::StringRef realization =
      riscv_internal::string(operation, "realization").value_or("");
  if (name == "weft_kernel.outer_contract" &&
      realization == "rvv.outer.encoded-widening-mac" && local) {
    int64_t maximumTerms =
        riscv_internal::integer(local, "partial_terms_max").value_or(0);
    int64_t unroll = riscv_internal::integer(schedule, "unroll").value_or(1);
    int64_t pipeline =
        riscv_internal::integer(schedule, "pipeline_depth").value_or(1);
    int64_t laneLMUL =
        riscv_internal::integer(local, "lane_lmul_eighths").value_or(0);
    int64_t partialLMUL =
        riscv_internal::integer(local, "partial_lmul_eighths").value_or(0);
    int64_t parts =
        riscv_internal::integer(local, "accumulator_parts").value_or(0);
    if (maximumTerms <= 0 || unroll <= 0 || pipeline <= 0 || laneLMUL <= 0 ||
        partialLMUL <= 0 || parts <= 0)
      return operation;
    int64_t partialTerms = std::min(maximumTerms, unroll);
    int64_t laneGroups = (laneLMUL + 7) / 8;
    int64_t partialGroups = (partialLMUL + 7) / 8;
    int64_t temporaryGroups = parts * partialGroups;
    if (pipeline > 1)
      temporaryGroups += 2 * partialTerms * laneGroups;
    local = riscv_internal::set(
        local, "partial_terms", builder.getI64IntegerAttr(partialTerms));
    local = riscv_internal::set(
        local, "operand_buffer_count",
        builder.getI64IntegerAttr(pipeline > 1 ? 2 : 1));
    local = riscv_internal::set(
        local, "temporary_vector_groups",
        builder.getI64IntegerAttr(temporaryGroups));
    local = riscv_internal::set(
        local, "temporary_groups_derived_from",
        builder.getStringAttr(
            "selected accumulator partials + scheduled encoded operand banks"));
    return riscv_internal::set(operation, "local_operation", local);
  }
  if (name != "weft_kernel.mac_pairs" && name != "weft_kernel.mac_groups")
    return operation;
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

int64_t operationOrdinal(llvm::StringRef id) {
  if (!id.consume_front("op"))
    return -1;
  int64_t ordinal = -1;
  return id.getAsInteger(10, ordinal) ? -1 : ordinal;
}

using RootSet = llvm::SmallSet<std::string, 8>;

struct LocalCluster {
  llvm::SmallVector<std::string> producers;
  llvm::SmallVector<std::string> consumers;
  llvm::SmallVector<std::string> frontier;
  int64_t bufferGroups = 0;
};

mlir::DictionaryAttr localClusterAttribute(mlir::Builder &builder,
                                           const LocalCluster &cluster,
                                           int64_t pipelineDepth) {
  return riscv_internal::dictionary(
      builder,
      {{"producer_ops", riscv_internal::strings(builder, cluster.producers)},
       {"consumer_ops", riscv_internal::strings(builder, cluster.consumers)},
       {"frontier_values", riscv_internal::strings(builder, cluster.frontier)},
       {"pipeline_depth", builder.getI64IntegerAttr(pipelineDepth)},
       {"buffer_groups",
        builder.getI64IntegerAttr(pipelineDepth > 1 ? cluster.bufferGroups : 0)},
       {"derived_from",
        builder.getStringAttr(
            "loop induction/carry dependencies + memory roots + selected local operations")},
       {"decision_owner", builder.getStringAttr("loop-local-use-def-cluster")}});
}

mlir::DictionaryAttr addClusterTemporaryBudget(
    mlir::Builder &builder, mlir::DictionaryAttr operation,
    int64_t bufferGroups) {
  if (bufferGroups <= 0)
    return operation;
  mlir::DictionaryAttr local =
      operation.getAs<mlir::DictionaryAttr>("local_operation");
  if (!local)
    local = builder.getDictionaryAttr({});
  const int64_t existing =
      riscv_internal::integer(local, "temporary_vector_groups").value_or(0);
  local = riscv_internal::set(
      local, "temporary_vector_groups",
      builder.getI64IntegerAttr(existing + bufferGroups));
  local = riscv_internal::set(
      local, "temporary_groups_derived_from",
      builder.getStringAttr(
          "selected local-operation temporaries + next-iteration frontier"));
  return riscv_internal::set(operation, "local_operation", local);
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

      llvm::StringMap<mlir::DictionaryAttr> operationsById;
      llvm::StringMap<mlir::DictionaryAttr> producersByValue;
      llvm::StringMap<llvm::SmallVector<mlir::DictionaryAttr, 2>> usersByValue;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef id = *riscv_internal::string(operation, "id");
        operationsById[id] = operation;
        if (auto results = operation.getAs<mlir::ArrayAttr>("results"))
          for (mlir::Attribute result : results)
            producersByValue[mlir::cast<mlir::StringAttr>(result).getValue()] =
                operation;
        if (auto operands = operation.getAs<mlir::ArrayAttr>("operands"))
          for (mlir::Attribute operand : operands)
            usersByValue[mlir::cast<mlir::StringAttr>(operand).getValue()]
                .push_back(operation);
      }

      llvm::StringMap<LocalCluster> clustersByLoop;
      for (const auto &loopEntry : operationsById) {
        mlir::DictionaryAttr loop = loopEntry.getValue();
        if (riscv_internal::string(loop, "name").value_or("") !=
            "weft_kernel.for")
          continue;
        const std::string loopId = loopEntry.getKey().str();
        llvm::SmallVector<std::string> expectedControl;
        if (auto outer = loop.getAs<mlir::ArrayAttr>("control_path"))
          for (mlir::Attribute attribute : outer)
            expectedControl.push_back(
                mlir::cast<mlir::StringAttr>(attribute).getValue().str());
        expectedControl.push_back(loopId);

        llvm::SmallVector<mlir::DictionaryAttr> bodyOperations;
        bool nestedControl = false;
        for (const auto &entry : operationsById) {
          mlir::DictionaryAttr operation = entry.getValue();
          auto path = operation.getAs<mlir::ArrayAttr>("control_path");
          if (!path || path.size() != expectedControl.size() ||
              operation.get("level_path") != loop.get("level_path"))
            continue;
          bool samePath = true;
          for (auto [index, attribute] : llvm::enumerate(path))
            samePath &= mlir::cast<mlir::StringAttr>(attribute).getValue() ==
                        expectedControl[index];
          if (!samePath)
            continue;
          llvm::StringRef name =
              riscv_internal::string(operation, "name").value_or("");
          nestedControl |= name == "weft_kernel.for" ||
                           name == "weft_kernel.while" ||
                           name == "weft_kernel.if" ||
                           name == "weft_kernel.level";
          bodyOperations.push_back(operation);
        }
        if (nestedControl || bodyOperations.empty())
          continue;
        llvm::sort(bodyOperations, [](mlir::DictionaryAttr lhs,
                                      mlir::DictionaryAttr rhs) {
          return operationOrdinal(*riscv_internal::string(lhs, "id")) <
                 operationOrdinal(*riscv_internal::string(rhs, "id"));
        });

        llvm::StringSet<> bodyIds;
        for (mlir::DictionaryAttr operation : bodyOperations)
          bodyIds.insert(*riscv_internal::string(operation, "id"));

        llvm::StringSet<> carryValues;
        const std::string argumentPrefix = loopId + ".region0.block0.arg";
        for (const auto &entry : values) {
          llvm::StringRef source =
              riscv_internal::string(entry.getValue(), "source").value_or("");
          if (!source.consume_front(argumentPrefix))
            continue;
          int64_t argument = -1;
          if (!source.getAsInteger(10, argument) && argument > 0)
            carryValues.insert(entry.getKey());
        }
        if (carryValues.empty())
          continue;

        llvm::StringMap<RootSet> rootCache;
        llvm::StringSet<> rootVisiting;
        std::function<RootSet(llvm::StringRef)> rootsForValue =
            [&](llvm::StringRef valueId) -> RootSet {
          if (auto cached = rootCache.find(valueId); cached != rootCache.end())
            return cached->second;
          RootSet roots;
          if (carryValues.contains(valueId)) {
            roots.insert("carry:" + valueId.str());
            rootCache[valueId] = roots;
            return roots;
          }
          if (!rootVisiting.insert(valueId).second)
            return roots;
          auto producer = producersByValue.find(valueId);
          if (producer == producersByValue.end()) {
            auto value = values.find(valueId);
            llvm::StringRef kind =
                value == values.end()
                    ? llvm::StringRef()
                    : riscv_internal::string(value->second, "kind").value_or("");
            if (kind == "view" || kind == "slice" ||
                kind == "encoded_value")
              roots.insert("memory:" + valueId.str());
          } else {
            llvm::StringRef name =
                riscv_internal::string(producer->second, "name").value_or("");
            if (name != "weft_kernel.constant" &&
                name != "weft_kernel.iota" && name != "weft_kernel.symbol" &&
                name != "weft_kernel.domain" &&
                name != "weft_kernel.root_domain")
              if (auto operands =
                      producer->second.getAs<mlir::ArrayAttr>("operands"))
                for (mlir::Attribute operand : operands) {
                  RootSet operandRoots = rootsForValue(
                      mlir::cast<mlir::StringAttr>(operand).getValue());
                  roots.insert(operandRoots.begin(), operandRoots.end());
                }
          }
          rootVisiting.erase(valueId);
          rootCache[valueId] = roots;
          return roots;
        };

        llvm::StringSet<> consumerValues;
        for (const auto &carry : carryValues)
          consumerValues.insert(carry.getKey());
        llvm::StringSet<> consumerOps;
        bool hasCompute = false;
        for (mlir::DictionaryAttr operation : bodyOperations) {
          llvm::StringRef id = *riscv_internal::string(operation, "id");
          llvm::StringRef name =
              riscv_internal::string(operation, "name").value_or("");
          bool consumer = false;
          RootSet roots;
          if (auto operands = operation.getAs<mlir::ArrayAttr>("operands"))
            for (mlir::Attribute operand : operands) {
              llvm::StringRef value =
                  mlir::cast<mlir::StringAttr>(operand).getValue();
              consumer |= consumerValues.contains(value);
              RootSet operandRoots = rootsForValue(value);
              roots.insert(operandRoots.begin(), operandRoots.end());
            }
          bool explicitCompute =
              name == "weft_kernel.dot" || name == "weft_kernel.contract" ||
              name == "weft_kernel.outer_contract" ||
              name == "weft_kernel.mac_pairs" ||
              name == "weft_kernel.mac_groups" ||
              name == "weft_kernel.reduce" || name == "weft_kernel.scan";
          if (name == "weft_kernel.binary") {
            auto source =
                operation.getAs<mlir::DictionaryAttr>("source_attributes");
            if (riscv_internal::string(source, "kind").value_or("") == "mul") {
              int64_t memoryRoots = 0;
              for (const std::string &root : roots)
                memoryRoots += llvm::StringRef(root).starts_with("memory:");
              explicitCompute |= memoryRoots >= 2;
            }
          }
          consumer |= explicitCompute;
          hasCompute |= explicitCompute;
          if (!consumer)
            continue;
          consumerOps.insert(id);
          if (auto results = operation.getAs<mlir::ArrayAttr>("results"))
            for (mlir::Attribute result : results)
              consumerValues.insert(
                  mlir::cast<mlir::StringAttr>(result).getValue());
        }
        if (!hasCompute)
          continue;

        llvm::StringSet<> frontierValues;
        for (mlir::DictionaryAttr operation : bodyOperations) {
          llvm::StringRef id = *riscv_internal::string(operation, "id");
          if (!consumerOps.contains(id))
            continue;
          if (auto operands = operation.getAs<mlir::ArrayAttr>("operands"))
            for (mlir::Attribute operand : operands) {
              llvm::StringRef value =
                  mlir::cast<mlir::StringAttr>(operand).getValue();
              auto producer = producersByValue.find(value);
              if (producer == producersByValue.end())
                continue;
              llvm::StringRef producerId =
                  *riscv_internal::string(producer->second, "id");
              if (bodyIds.contains(producerId) &&
                  !consumerOps.contains(producerId))
                frontierValues.insert(value);
            }
        }
        if (frontierValues.empty())
          continue;

        llvm::StringSet<> producerOps;
        llvm::SmallVector<std::string> pending;
        for (const auto &frontier : frontierValues)
          pending.push_back(frontier.getKey().str());
        bool hasMemoryProducer = false;
        while (!pending.empty()) {
          std::string value = std::move(pending.pop_back_val());
          auto producer = producersByValue.find(value);
          if (producer == producersByValue.end())
            continue;
          llvm::StringRef id = *riscv_internal::string(producer->second, "id");
          if (!bodyIds.contains(id) || consumerOps.contains(id) ||
              !producerOps.insert(id).second)
            continue;
          llvm::StringRef name =
              riscv_internal::string(producer->second, "name").value_or("");
          hasMemoryProducer |= name == "weft_kernel.extract" ||
                               name == "weft_kernel.lookup" ||
                               name == "weft_kernel.admit";
          if (name == "weft_kernel.commit" ||
              name == "weft_kernel.materialize" ||
              name == "weft_kernel.pack") {
            producerOps.clear();
            break;
          }
          if (auto operands = producer->second.getAs<mlir::ArrayAttr>("operands"))
            for (mlir::Attribute operand : operands)
              pending.push_back(
                  mlir::cast<mlir::StringAttr>(operand).getValue().str());
        }
        if (!hasMemoryProducer || producerOps.empty())
          continue;

        LocalCluster cluster;
        for (mlir::DictionaryAttr operation : bodyOperations) {
          llvm::StringRef id = *riscv_internal::string(operation, "id");
          if (producerOps.contains(id))
            cluster.producers.push_back(id.str());
          else if (consumerOps.contains(id) &&
                   riscv_internal::string(operation, "name").value_or("") !=
                       "weft_kernel.yield")
            cluster.consumers.push_back(id.str());
        }
        for (const auto &frontier : frontierValues) {
          cluster.frontier.push_back(frontier.getKey().str());
          auto value = values.find(frontier.getKey());
          if (value != values.end())
            cluster.bufferGroups +=
                riscv_internal::integer(value->second, "register_groups")
                    .value_or(0);
        }
        llvm::sort(cluster.frontier);
        if (!cluster.producers.empty() && !cluster.consumers.empty())
          clustersByLoop[loopId] = std::move(cluster);
      }

      llvm::StringMap<int64_t> levelLaneWidths;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        auto path = operation.getAs<mlir::ArrayAttr>("level_path");
        if (!path || path.empty())
          continue;
        auto recordValue = [&](mlir::Attribute valueAttribute) {
          auto valueId = mlir::dyn_cast<mlir::StringAttr>(valueAttribute);
          auto value = valueId ? values.find(valueId.getValue()) : values.end();
          if (value == values.end())
            return;
          int64_t laneAxis =
              riscv_internal::integer(value->second, "lane_axis").value_or(0);
          int64_t physicalLanes =
              riscv_internal::integer(value->second, "physical_lanes")
                  .value_or(1);
          if (laneAxis <= 0 || physicalLanes <= 1)
            return;
          for (mlir::Attribute levelAttribute : path) {
            auto level = mlir::cast<mlir::StringAttr>(levelAttribute);
            const std::string key = level.getValue().str() + "#" +
                                    std::to_string(laneAxis);
            levelLaneWidths[key] =
                std::max(levelLaneWidths.lookup(key), physicalLanes);
          }
        };
        if (auto operands = operation.getAs<mlir::ArrayAttr>("operands"))
          for (mlir::Attribute operand : operands)
            recordValue(operand);
        if (auto results = operation.getAs<mlir::ArrayAttr>("results"))
          for (mlir::Attribute result : results)
            recordValue(result);
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

      auto enclosingSchedule = [&](mlir::DictionaryAttr operation) {
        mlir::DictionaryAttr schedule;
        if (auto path = operation.getAs<mlir::ArrayAttr>("level_path"))
          for (mlir::Attribute level : llvm::reverse(path)) {
            llvm::StringRef id =
                mlir::cast<mlir::StringAttr>(level).getValue();
            auto found = schedulesByLevel.find(id);
            if (found == schedulesByLevel.end())
              continue;
            schedule = found->second;
            break;
          }
        return schedule;
      };
      llvm::StringMap<int64_t> clusterPipelineDepths;
      for (const auto &entry : clustersByLoop) {
        mlir::DictionaryAttr loop = operationsById.lookup(entry.getKey());
        mlir::DictionaryAttr schedule = enclosingSchedule(loop);
        clusterPipelineDepths[entry.getKey()] =
            riscv_internal::integer(schedule, "pipeline_depth").value_or(1);
      }

      llvm::SmallVector<mlir::Attribute> operations;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        if (riscv_internal::string(operation, "name").value_or("") !=
            "weft_kernel.level") {
          llvm::StringRef operationId =
              riscv_internal::string(operation, "id").value_or("");
          auto ownCluster = clustersByLoop.find(operationId);
          if (ownCluster != clustersByLoop.end()) {
            const int64_t depth = clusterPipelineDepths.lookup(operationId);
            llvm::StringRef structure =
                depth > 1 ? "cross-iteration-prologue-steady-epilogue"
                          : "sequential-local-cluster";
            mlir::DictionaryAttr cluster =
                localClusterAttribute(builder, ownCluster->second, depth);
            operation =
                riscv_internal::set(operation, "local_cluster", cluster);
            operation = riscv_internal::set(
                operation, "schedule",
                riscv_internal::dictionary(
                    builder,
                    {{"pipeline_depth", builder.getI64IntegerAttr(depth)},
                     {"loop_structure",
                      builder.getStringAttr(structure)},
                     {"decision_owner",
                      builder.getStringAttr("loop-local-use-def-cluster")}}));
            operations.push_back(operation);
            continue;
          }

          const LocalCluster *enclosingCluster = nullptr;
          std::string enclosingLoop;
          if (auto controlPath =
                  operation.getAs<mlir::ArrayAttr>("control_path"))
            for (mlir::Attribute control : llvm::reverse(controlPath)) {
              llvm::StringRef id =
                  mlir::cast<mlir::StringAttr>(control).getValue();
              auto found = clustersByLoop.find(id);
              if (found == clustersByLoop.end())
                continue;
              enclosingCluster = &found->second;
              enclosingLoop = id.str();
              break;
            }
          if (enclosingCluster) {
            const int64_t depth = clusterPipelineDepths.lookup(enclosingLoop);
            llvm::StringRef structure =
                depth > 1 ? "cross-iteration-prologue-steady-epilogue"
                          : "sequential-local-cluster";
            auto producer = llvm::find(enclosingCluster->producers,
                                       operationId.str());
            auto consumer = llvm::find(enclosingCluster->consumers,
                                       operationId.str());
            llvm::StringRef stage =
                producer != enclosingCluster->producers.end()
                    ? "producer"
                : consumer != enclosingCluster->consumers.end()
                    ? "consumer"
                    : "outside-cluster";
            int64_t order =
                stage == "producer"
                    ? std::distance(enclosingCluster->producers.begin(), producer)
                : stage == "consumer"
                    ? std::distance(enclosingCluster->consumers.begin(), consumer)
                    : -1;
            operation = riscv_internal::set(
                operation, "schedule_loop",
                builder.getStringAttr(enclosingLoop));
            operation = riscv_internal::set(
                operation, "schedule",
                riscv_internal::dictionary(
                    builder,
                    {{"cluster_stage", builder.getStringAttr(stage)},
                     {"cluster_order", builder.getI64IntegerAttr(order)},
                     {"pipeline_depth", builder.getI64IntegerAttr(depth)},
                     {"loop_structure",
                      builder.getStringAttr(structure)},
                     {"decision_owner",
                      builder.getStringAttr("loop-local-use-def-cluster")}}));
            if (depth > 1 && !enclosingCluster->consumers.empty() &&
                enclosingCluster->consumers.front() == operationId)
              operation = addClusterTemporaryBudget(
                  builder, operation, enclosingCluster->bufferGroups);
            operations.push_back(operation);
            continue;
          }

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
        llvm::StringRef levelId = *riscv_internal::string(operation, "id");
        int64_t physicalLanes = levelLaneWidths.lookup(
            levelId.str() + "#" + std::to_string(axis));
        physicalLanes = std::max<int64_t>(1, physicalLanes);
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
