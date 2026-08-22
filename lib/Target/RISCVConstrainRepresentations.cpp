#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
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
      return extent > 0 ? extent : 0;
  return 0;
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
    if (candidate >= required && candidate >= (sew + 7) / 8)
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
  int64_t extent = laneExtent(value, laneAxis);
  return (shape && shape.size() > 1) || (extent > 0 && extent > physicalLanes)
             ? "rvv-stream"
             : "rvv-lane";
}

bool propagatesLaneIdentity(llvm::StringRef name) {
  return name == "weft_kernel.admit" || name == "weft_kernel.commit" ||
         name == "weft_kernel.materialize" || name == "weft_kernel.pack" ||
         name == "weft_kernel.stage_handoff" ||
         name == "weft_kernel.extract" || name == "weft_kernel.unary" ||
         name == "weft_kernel.binary" || name == "weft_kernel.compare" ||
         name == "weft_kernel.cast" || name == "weft_kernel.widen" ||
         name == "weft_kernel.mac_pairs" ||
         name == "weft_kernel.mac_groups" || name == "weft_kernel.reduce" ||
         name == "weft_kernel.dot" || name == "weft_kernel.contract" ||
         name == "weft_kernel.outer_contract" ||
         name == "weft_kernel.lookup";
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
      llvm::StringMap<std::string> producerIds;
      llvm::StringMap<bool> singletonScalarExtractInputs;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef name = *riscv_internal::string(operation, "name");
        if (auto operands = operation.getAs<mlir::ArrayAttr>("operands"))
          for (mlir::Attribute operand : operands)
            ++useCounts[mlir::cast<mlir::StringAttr>(operand).getValue()];
        if (auto results = operation.getAs<mlir::ArrayAttr>("results"))
          for (mlir::Attribute result : results) {
            llvm::StringRef id =
                mlir::cast<mlir::StringAttr>(result).getValue();
            producers[id] = name.str();
            producerIds[id] = riscv_internal::string(operation, "id")->str();
          }
        if (name == "weft_kernel.extract") {
          auto operands = operation.getAs<mlir::ArrayAttr>("operands");
          auto results = operation.getAs<mlir::ArrayAttr>("results");
          if (operands && !operands.empty() && results && results.size() == 1) {
            llvm::StringRef inputId =
                mlir::cast<mlir::StringAttr>(operands[0]).getValue();
            llvm::StringRef resultId =
                mlir::cast<mlir::StringAttr>(results[0]).getValue();
            auto input = valuesById.find(inputId);
            auto result = valuesById.find(resultId);
            auto shape = input == valuesById.end()
                             ? mlir::DenseI64ArrayAttr()
                             : input->second.getAs<mlir::DenseI64ArrayAttr>("shape");
            bool singleton = shape && !shape.empty();
            if (singleton)
              for (int64_t extent : shape.asArrayRef())
                singleton &= extent == 1;
            if (singleton && result != valuesById.end() &&
                riscv_internal::string(result->second, "kind").value_or("") ==
                    "scalar")
              singletonScalarExtractInputs[inputId] = true;
          }
        }
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

      int64_t vlenBits =
          *riscv_internal::integer(problem.getTarget(), "vlen_bits");
      auto legalLMUL = problem.getTarget().getAs<mlir::DenseI64ArrayAttr>(
          "legal_lmul_eighths");

      llvm::SmallVector<mlir::DictionaryAttr> sourceValues;
      llvm::StringMap<unsigned> valueOrdinals;
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        valueOrdinals[*riscv_internal::string(value, "id")] =
            sourceValues.size();
        sourceValues.push_back(value);
      }
      llvm::SmallVector<unsigned> parents(sourceValues.size());
      for (unsigned index = 0; index < parents.size(); ++index)
        parents[index] = index;
      auto find = [&](unsigned value) {
        unsigned root = value;
        while (parents[root] != root)
          root = parents[root];
        while (parents[value] != value) {
          unsigned next = parents[value];
          parents[value] = root;
          value = next;
        }
        return root;
      };
      auto unite = [&](unsigned lhs, unsigned rhs) {
        unsigned lhsRoot = find(lhs);
        unsigned rhsRoot = find(rhs);
        if (lhsRoot == rhsRoot)
          return;
        if (rhsRoot < lhsRoot)
          std::swap(lhsRoot, rhsRoot);
        parents[rhsRoot] = lhsRoot;
      };

      llvm::StringMap<unsigned> handoffRoots;
      for (auto [index, value] : llvm::enumerate(sourceValues)) {
        llvm::StringRef handoff =
            riscv_internal::string(value, "handoff_class").value_or("");
        if (handoff.empty())
          continue;
        auto inserted = handoffRoots.try_emplace(handoff, index);
        if (!inserted.second)
          unite(inserted.first->second, index);
      }

      struct TransferEdge {
        std::string id;
        std::string name;
        unsigned value = 0;
      };
      llvm::SmallVector<TransferEdge> transferEdges;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef operationName =
            riscv_internal::string(operation, "name").value_or("");
        if (!propagatesLaneIdentity(operationName))
          continue;
        llvm::SmallVector<unsigned, 4> laneValues;
        for (llvm::StringRef key : {"operands", "results"})
          if (auto ids = operation.getAs<mlir::ArrayAttr>(key))
            for (mlir::Attribute idAttribute : ids) {
              llvm::StringRef id =
                  mlir::cast<mlir::StringAttr>(idAttribute).getValue();
              auto ordinal = valueOrdinals.find(id);
              if (ordinal == valueOrdinals.end())
                continue;
              mlir::DictionaryAttr value = sourceValues[ordinal->second];
              if (selected.axis >= 0 && containsAxis(value, selected.axis) &&
                  riscv_internal::integer(value, "logical_sew").value_or(0) > 0)
                laneValues.push_back(ordinal->second);
            }
        if (laneValues.size() < 2)
          continue;
        for (size_t index = 1; index < laneValues.size(); ++index)
          unite(laneValues.front(), laneValues[index]);
        transferEdges.push_back(
            {riscv_internal::string(operation, "id")->str(),
             operationName.str(),
             laneValues.front()});
      }

      struct ChainFacts {
        int64_t maximumSEW = 0;
        int64_t maximumExtent = 0;
        int64_t lanes = 1;
        llvm::SmallVector<std::string, 8> operations;
      };
      llvm::DenseMap<unsigned, ChainFacts> chains;
      for (auto [index, value] : llvm::enumerate(sourceValues)) {
        if (selected.axis < 0 || !containsAxis(value, selected.axis))
          continue;
        int64_t logicalSEW =
            riscv_internal::integer(value, "logical_sew").value_or(0);
        if (logicalSEW <= 0)
          continue;
        ChainFacts &chain = chains[find(index)];
        chain.maximumSEW =
            std::max<int64_t>(chain.maximumSEW, std::max<int64_t>(8, logicalSEW));
        chain.maximumExtent =
            std::max(chain.maximumExtent, laneExtent(value, selected.axis));
      }
      for (const TransferEdge &edge : transferEdges)
        chains[find(edge.value)].operations.push_back(edge.id + ":" + edge.name);
      if (selected.axis >= 0) {
        if (!legalLMUL || legalLMUL.empty()) {
          invalidate(problem, builder, "lane axis has no legal target LMUL set");
          continue;
        }
        int64_t maximumLMUL = *llvm::max_element(legalLMUL.asArrayRef());
        bool chainLegal = true;
        for (auto &entry : chains) {
          ChainFacts &chain = entry.second;
          int64_t extentBound =
              chain.maximumExtent > 0 ? chain.maximumExtent : selected.cohort;
          chain.lanes = std::min(
              {selected.cohort, extentBound,
               vlenBits * maximumLMUL / (chain.maximumSEW * 8)});
          if (chain.lanes <= 0)
            chainLegal = false;
          llvm::sort(chain.operations);
          chain.operations.erase(
              std::unique(chain.operations.begin(), chain.operations.end()),
              chain.operations.end());
        }
        if (!chainLegal) {
          invalidate(problem, builder,
                     "one use-def chain has no legal target representation");
          continue;
        }
      }

      llvm::SmallVector<mlir::Attribute> assignedValues;
      bool legal = true;
      for (auto [index, original] : llvm::enumerate(sourceValues)) {
        auto value = original;
        std::string id = riscv_internal::string(value, "id")->str();
        int64_t logicalSEW =
            riscv_internal::integer(value, "logical_sew").value_or(0);
        unsigned chainRoot = find(index);
        auto chainIt = chains.find(chainRoot);
        int64_t chainLanes = chainIt == chains.end() ? 1 : chainIt->second.lanes;
        bool lane = selected.axis >= 0 && containsAxis(value, selected.axis) &&
                    logicalSEW > 0;
        const bool scalarExtract = singletonScalarExtractInputs.lookup(id) &&
                                   useCounts.lookup(id) == 1;
        if (scalarExtract)
          lane = false;
        std::string kind =
            physicalKind(value, lane ? selected.axis : 0, chainLanes);
        int64_t logicalLaneExtent =
            lane ? laneExtent(value, selected.axis) : 0;
        int64_t valueLanes =
            lane ? (logicalLaneExtent > 0
                        ? std::min(logicalLaneExtent, chainLanes)
                        : chainLanes)
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
            lane ? (logicalLaneExtent > 0
                        ? std::max<int64_t>(
                              1, (logicalLaneExtent + valueLanes - 1) /
                                     valueLanes)
                        : 1)
                 : 1;
        int64_t registerGroups = lmul ? (lmul + 7) / 8 : 0;
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
                                      ? "pending-liveness"
                                      : "not-applicable"));
        value = riscv_internal::set(
            value, "representation_chain",
            builder.getStringAttr(lane ? "r" + std::to_string(chainRoot)
                                       : "not-lane-mapped"));
        value = riscv_internal::set(
            value, "sew_derived_from",
            builder.getStringAttr(
                logicalSEW > 0
                    ? (!producerIds.lookup(id).empty()
                           ? "typed result " + producerIds.lookup(id) + " (" +
                                 producers.lookup(id) + ")"
                           : "typed carried/block value " +
                                 riscv_internal::string(value, "source")
                                     .value_or("unknown-source")
                                     .str())
                    : "non-numeric canonical type"));
        value = riscv_internal::set(
            value, "lanes_derived_from",
            builder.getStringAttr(
                lane ? "axis" + std::to_string(selected.axis) +
                           " use-def chain r" + std::to_string(chainRoot) +
                           " extent/cohort/VLEN resource bound"
                : scalarExtract
                    ? "singleton local value has one scalar extract consumer"
                    : "value does not carry the selected lane axis"));
        value = riscv_internal::set(
            value, "lmul_derived_from",
            builder.getStringAttr(
                lane ? "preserve " + std::to_string(valueLanes) +
                           " logical lanes at SEW" +
                           std::to_string(physicalSEW) + " on VLEN" +
                           std::to_string(vlenBits)
                     : "not a vector value"));
        value = riscv_internal::set(
            value, "vl_derived_from",
            builder.getStringAttr(
                lane ? "remaining extent of axis" +
                           std::to_string(selected.axis) + " capped at " +
                           std::to_string(valueLanes)
                     : "scalar/control value"));
        if (lane && chainIt != chains.end())
          value = riscv_internal::set(
              value, "representation_users",
              riscv_internal::strings(builder, chainIt->second.operations));
        value = riscv_internal::set(
            value, "materialization_derived_from",
            builder.getStringAttr(producers.lookup(id) == "weft_kernel.admit"
                                      ? "deferred to live-range resource pass"
                                      : "value is not a reloadable admit result"));
        value = riscv_internal::set(
            value, "decision_owner", builder.getStringAttr("value"));
        assignedValues.push_back(value);
      }
      if (!legal) {
        invalidate(problem, builder, "one value has no legal SEW/LMUL representation");
        continue;
      }
      problem.setValuesAttr(builder.getArrayAttr(assignedValues));
      llvm::SmallVector<mlir::Attribute> assignedOperations;
      llvm::StringMap<std::string> transferDescriptions;
      for (const TransferEdge &edge : transferEdges)
        transferDescriptions[edge.id] =
            edge.name == "weft_kernel.widen" || edge.name == "weft_kernel.cast"
                ? "typed width conversion preserves logical lane identity; SEW/LMUL follow each typed value"
                : "op rule preserves the selected logical lane axis across its use-def edges";
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef id = *riscv_internal::string(operation, "id");
        auto transfer = transferDescriptions.find(id);
        if (transfer != transferDescriptions.end())
          operation = riscv_internal::set(
              operation, "representation_transfer",
              builder.getStringAttr(transfer->second));
        assignedOperations.push_back(operation);
      }
      problem.setOperationsAttr(builder.getArrayAttr(assignedOperations));
      problem.setStageAttr(builder.getStringAttr("representations"));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createAssignRISCVRepresentationsPass() {
  return std::make_unique<AssignRISCVRepresentationsPass>();
}
