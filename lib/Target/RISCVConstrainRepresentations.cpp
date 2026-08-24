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

bool sameAxisSet(mlir::DenseI64ArrayAttr lhs, mlir::DenseI64ArrayAttr rhs) {
  if (!lhs || !rhs || lhs.size() != rhs.size())
    return false;
  for (int64_t axis : lhs.asArrayRef())
    if (!llvm::is_contained(rhs.asArrayRef(), axis))
      return false;
  return true;
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
  return name == "weft_kernel.iota" || name == "weft_kernel.admit" ||
         name == "weft_kernel.commit" ||
         name == "weft_kernel.materialize" ||
         name == "weft_kernel.field" || name == "weft_kernel.extract" ||
         name == "weft_kernel.new" || name == "weft_kernel.unary" ||
         name == "weft_kernel.binary" || name == "weft_kernel.compare" ||
         name == "weft_kernel.cast" || name == "weft_kernel.widen" ||
         name == "weft_kernel.narrow" || name == "weft_kernel.fold2" ||
         name == "weft_kernel.mac_pairs" ||
         name == "weft_kernel.mac_groups" ||
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
      struct AxisFacts {
        int priority = 2;
        int64_t cohort = 1;
      };
      llvm::DenseMap<int64_t, AxisFacts> axisFacts;
      kernel.walk([&](kernel::DomainOp domainOp) {
        kernel::DomainType domain = domainOp.getResult().getType();
        auto partition =
            resolvePartition(domain.getPartition(), problem.getCandidate());
        if (!partition || *partition <= 1)
          return;
        int priority =
            (domain.getRelation() == "rows" || domain.getRelation() == "cols")
                ? 0
                : 1;
        AxisFacts &facts = axisFacts[domain.getAxisId()];
        facts.priority = std::min(facts.priority, priority);
        facts.cohort = std::max(facts.cohort, *partition);
      });

      llvm::StringMap<AxisFacts> levelAxisFacts;
      llvm::StringMap<int64_t> levelAxes;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        if (riscv_internal::string(operation, "name").value_or("") !=
            "weft_kernel.level")
          continue;
        auto operands = operation.getAs<mlir::ArrayAttr>("operands");
        if (!operands || operands.empty())
          continue;
        llvm::StringRef domainId =
            mlir::cast<mlir::StringAttr>(operands[0]).getValue();
        auto domainValue = valuesById.find(domainId);
        if (domainValue == valuesById.end())
          continue;
        auto axis = riscv_internal::integer(domainValue->second, "domain_axis");
        auto relation =
            riscv_internal::string(domainValue->second, "domain_relation");
        auto partition =
            riscv_internal::string(domainValue->second, "domain_partition");
        auto resolved = partition
                            ? resolvePartition(*partition, problem.getCandidate())
                            : std::optional<int64_t>();
        if (!axis || !relation || !resolved || *resolved <= 0)
          continue;
        llvm::StringRef levelId =
            riscv_internal::string(operation, "id").value_or("");
        levelAxes[levelId] = *axis;
        levelAxisFacts[levelId] = {
            (*relation == "rows" || *relation == "cols") ? 0 : 1,
            *resolved};
      }
      auto factsFor = [&](mlir::DictionaryAttr operation, int64_t axis) {
        if (auto path = operation.getAs<mlir::ArrayAttr>("level_path"))
          for (mlir::Attribute levelAttribute : llvm::reverse(path)) {
            llvm::StringRef level =
                mlir::cast<mlir::StringAttr>(levelAttribute).getValue();
            if (levelAxes.lookup(level) == axis)
              return levelAxisFacts.lookup(level);
          }
        return axisFacts.lookup(axis);
      };

      llvm::StringMap<int64_t> useCounts;
      llvm::StringMap<std::string> producers;
      llvm::StringMap<std::string> producerIds;
      llvm::StringMap<mlir::DictionaryAttr> producerOperations;
      llvm::StringMap<bool> singletonScalarExtractInputs;
      llvm::StringMap<int64_t> valueLaneAxes;
      llvm::StringMap<int> valueLanePriorities;
      llvm::StringMap<int64_t> valueLaneCohorts;
      llvm::StringMap<llvm::SmallVector<int64_t, 4>> valueRegisterAxes;
      llvm::DenseMap<int64_t, int64_t> explicitLocalAxisExtents;
      llvm::StringMap<bool> valueNoLaneAnchors;
      llvm::StringMap<bool> valueStrongLaneAnchors;
      llvm::StringMap<int64_t> operationLaneAxes;
      llvm::StringMap<int64_t> operationLaneCohorts;
      llvm::StringMap<int64_t> operationMemoryAffinities;
      llvm::StringMap<std::string> operationLaneReasons;
      llvm::StringMap<bool> operationNoLaneMappings;
      llvm::StringMap<std::string> operationSliceConflicts;
      llvm::StringMap<int64_t> operationAccumulatorAxes;
      llvm::StringMap<int64_t> operationEliminatedAxes;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef name = *riscv_internal::string(operation, "name");
        llvm::StringRef operationId =
            *riscv_internal::string(operation, "id");
        if (auto operands = operation.getAs<mlir::ArrayAttr>("operands"))
          for (mlir::Attribute operand : operands) {
            llvm::StringRef id =
                mlir::cast<mlir::StringAttr>(operand).getValue();
            ++useCounts[id];
          }
        if (auto results = operation.getAs<mlir::ArrayAttr>("results"))
          for (mlir::Attribute result : results) {
            llvm::StringRef id =
                mlir::cast<mlir::StringAttr>(result).getValue();
            producers[id] = name.str();
            producerIds[id] = riscv_internal::string(operation, "id")->str();
            producerOperations[id] = operation;
          }
        if (name == "weft_kernel.iota" || name == "weft_kernel.dot" ||
            name == "weft_kernel.contract" ||
            name == "weft_kernel.outer_contract")
          if (auto results = operation.getAs<mlir::ArrayAttr>("results"))
            for (mlir::Attribute result : results) {
              llvm::StringRef resultId =
                  mlir::cast<mlir::StringAttr>(result).getValue();
              if (name != "weft_kernel.iota") {
                valueStrongLaneAnchors[resultId] = true;
                continue;
              }
              auto found = valuesById.find(resultId);
              auto axes = found == valuesById.end()
                              ? mlir::DenseI64ArrayAttr()
                              : found->second.getAs<mlir::DenseI64ArrayAttr>(
                                    "axes");
              auto shape = found == valuesById.end()
                               ? mlir::DenseI64ArrayAttr()
                               : found->second.getAs<mlir::DenseI64ArrayAttr>(
                                     "shape");
              if (axes && axes.size() == 1 && shape && shape.size() == 1 &&
                  shape[0] > 1)
                explicitLocalAxisExtents[axes[0]] = shape[0];
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
      }

      llvm::StringMap<llvm::SmallVector<std::string, 8>> useDefNeighbors;
      auto connect = [&](llvm::ArrayRef<std::string> ids) {
        for (size_t lhs = 0; lhs < ids.size(); ++lhs)
          for (size_t rhs = lhs + 1; rhs < ids.size(); ++rhs) {
            useDefNeighbors[ids[lhs]].push_back(ids[rhs]);
            useDefNeighbors[ids[rhs]].push_back(ids[lhs]);
          }
      };
      llvm::StringMap<llvm::SmallVector<std::string, 8>> handoffValues;
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef handoff =
            riscv_internal::string(value, "handoff_class").value_or("");
        if (!handoff.empty())
          handoffValues[handoff].push_back(
              riscv_internal::string(value, "id")->str());
      }
      for (const auto &entry : handoffValues)
        connect(entry.getValue());
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        if (!propagatesLaneIdentity(
                riscv_internal::string(operation, "name").value_or("")))
          continue;
        llvm::SmallVector<std::string, 8> ids;
        for (llvm::StringRef key : {"operands", "results"})
          if (auto values = operation.getAs<mlir::ArrayAttr>(key))
            for (mlir::Attribute value : values)
              ids.push_back(
                  mlir::cast<mlir::StringAttr>(value).getValue().str());
        connect(ids);
      }
      auto contiguousAxisForMemory = [&](mlir::DictionaryAttr value) {
        auto axes = value.getAs<mlir::DenseI64ArrayAttr>("axes");
        if (!axes || axes.empty())
          return int64_t{0};
        if (riscv_internal::string(value, "encoding_kind").value_or("") ==
            "dense")
          return axes.asArrayRef().back();
        return int64_t{0};
      };
      auto memoryAffinityFor = [&](mlir::DictionaryAttr operation,
                                   int64_t candidateAxis) {
        llvm::SmallSet<std::string, 32> visited;
        llvm::SmallVector<std::string, 16> pending;
        for (llvm::StringRef key : {"operands", "results"})
          if (auto values = operation.getAs<mlir::ArrayAttr>(key))
            for (mlir::Attribute value : values)
              pending.push_back(
                  mlir::cast<mlir::StringAttr>(value).getValue().str());
        int64_t score = 0;
        while (!pending.empty()) {
          std::string current = std::move(pending.back());
          pending.pop_back();
          if (!visited.insert(current).second)
            continue;
          auto value = valuesById.find(current);
          if (value == valuesById.end() ||
              !containsAxis(value->second, candidateAxis))
            continue;
          llvm::StringRef kind =
              riscv_internal::string(value->second, "kind").value_or("");
          if ((kind == "view" || kind == "slice") &&
              contiguousAxisForMemory(value->second) == candidateAxis)
            ++score;
          auto neighbors = useDefNeighbors.find(current);
          if (neighbors != useDefNeighbors.end())
            pending.append(neighbors->second.begin(), neighbors->second.end());
        }
        return score;
      };

      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef name = *riscv_internal::string(operation, "name");
        llvm::StringRef operationId =
            *riscv_internal::string(operation, "id");
        int64_t operationAxis = -1;
        int operationPriority = 3;
        int64_t operationCohort = 1;
        int64_t operationMemoryAffinity = -1;
        if (name == "weft_kernel.iota") {
          auto results = operation.getAs<mlir::ArrayAttr>("results");
          if (results && results.size() == 1) {
            auto result = valuesById.find(
                mlir::cast<mlir::StringAttr>(results[0]).getValue());
            auto axes = result == valuesById.end()
                            ? mlir::DenseI64ArrayAttr()
                            : result->second.getAs<mlir::DenseI64ArrayAttr>("axes");
            auto shape = result == valuesById.end()
                             ? mlir::DenseI64ArrayAttr()
                             : result->second.getAs<mlir::DenseI64ArrayAttr>("shape");
            if (axes && axes.size() == 1 && shape && shape.size() == 1) {
              operationAxis = axes[0];
              operationPriority = -4;
              operationCohort = shape[0];
              operationMemoryAffinity = 1;
            }
          }
        }
        if (name == "weft_kernel.dot" || name == "weft_kernel.contract" ||
            name == "weft_kernel.outer_contract") {
          auto source =
              operation.getAs<mlir::DictionaryAttr>("source_attributes");
          auto over = source
                          ? source.getAs<mlir::DenseI64ArrayAttr>("over")
                          : mlir::DenseI64ArrayAttr();
          if (over && over.size() == 1)
            operationEliminatedAxes[operationId] =
                over.asArrayRef().front();
        }
        if (name == "weft_kernel.reduce") {
          auto operands = operation.getAs<mlir::ArrayAttr>("operands");
          auto source =
              operation.getAs<mlir::DictionaryAttr>("source_attributes");
          auto position = source ? source.getAs<mlir::IntegerAttr>("axis")
                                 : mlir::IntegerAttr();
          if (operands && operands.size() == 1 && position) {
            llvm::StringRef inputId =
                mlir::cast<mlir::StringAttr>(operands[0]).getValue();
            auto input = valuesById.find(inputId);
            auto axes = input == valuesById.end()
                            ? mlir::DenseI64ArrayAttr()
                            : input->second.getAs<mlir::DenseI64ArrayAttr>("axes");
            if (axes && position.getInt() >= 0 &&
                position.getInt() < static_cast<int64_t>(axes.size())) {
              const int64_t eliminatedAxis =
                  axes.asArrayRef()[position.getInt()];
              operationEliminatedAxes[operationId] = eliminatedAxis;
              const int64_t inputLane = valueLaneAxes.lookup(inputId);
              if (inputLane > 0) {
                operationAxis = inputLane;
                operationPriority = valueLanePriorities.lookup(inputId);
                operationCohort = valueLaneCohorts.lookup(inputId);
              } else {
                operationAxis = eliminatedAxis;
                operationPriority = -2;
                operationCohort = std::max<int64_t>(
                    1, factsFor(operation, operationAxis).cohort);
              }
              if (operationAxis == eliminatedAxis)
                if (auto results =
                        operation.getAs<mlir::ArrayAttr>("results"))
                  for (mlir::Attribute result : results)
                    valueNoLaneAnchors[mlir::cast<mlir::StringAttr>(result)
                                           .getValue()] = true;
            }
          }
        } else if (name == "weft_kernel.outer_contract") {
          auto source =
              operation.getAs<mlir::DictionaryAttr>("source_attributes");
          auto over = source
                          ? source.getAs<mlir::DenseI64ArrayAttr>("over")
                          : mlir::DenseI64ArrayAttr();
          auto operands = operation.getAs<mlir::ArrayAttr>("operands");
          if (over && over.size() == 1 && operands) {
            for (mlir::Attribute operandAttribute : operands) {
              llvm::StringRef valueId =
                  mlir::cast<mlir::StringAttr>(operandAttribute).getValue();
              llvm::StringRef traced = valueId;
              auto producer = producerOperations.find(traced);
              while (producer != producerOperations.end() &&
                     riscv_internal::string(producer->second, "name")
                             .value_or("") == "weft_kernel.extract") {
                auto inputs =
                    producer->second.getAs<mlir::ArrayAttr>("operands");
                if (!inputs || inputs.empty())
                  break;
                traced =
                    mlir::cast<mlir::StringAttr>(inputs[0]).getValue();
                producer = producerOperations.find(traced);
              }
              if (producer == producerOperations.end() ||
                  riscv_internal::string(producer->second, "name")
                          .value_or("") != "weft_kernel.field")
                continue;
              auto fieldOperands =
                  producer->second.getAs<mlir::ArrayAttr>("operands");
              if (!fieldOperands || fieldOperands.empty())
                continue;
              auto owner = valuesById.find(
                  mlir::cast<mlir::StringAttr>(fieldOperands[0]).getValue());
              if (owner == valuesById.end())
                continue;
              auto ownerType = owner->second.getAs<mlir::TypeAttr>("type");
              auto ownerEncoding = ownerType
                                       ? mlir::dyn_cast<kernel::EncodingType>(
                                             riscv_internal::logicalElement(
                                                 ownerType.getValue()))
                                       : kernel::EncodingType();
              const bool derivedInterleave =
                  ownerEncoding && ownerEncoding.getKind() == "derived_instance";
              if (!derivedInterleave)
                continue;
              auto operand = valuesById.find(valueId);
              auto axes = operand == valuesById.end()
                              ? mlir::DenseI64ArrayAttr()
                              : operand->second.getAs<mlir::DenseI64ArrayAttr>(
                                    "axes");
              if (!axes)
                continue;
              for (int64_t axis : axes.asArrayRef()) {
                if (axis == over.asArrayRef().front() ||
                    axisFacts.find(axis) == axisFacts.end())
                  continue;
                operationAxis = axis;
                operationPriority = -1;
                operationCohort = factsFor(operation, axis).cohort;
                break;
              }
              if (operationAxis > 0)
                break;
            }
          }
        }
        auto consider = [&](mlir::ArrayAttr ids) {
          if (!ids)
            return;
          for (mlir::Attribute idAttribute : ids) {
            llvm::StringRef id =
                mlir::cast<mlir::StringAttr>(idAttribute).getValue();
            auto found = valuesById.find(id);
            if (found == valuesById.end())
              continue;
            auto axes = found->second.getAs<mlir::DenseI64ArrayAttr>("axes");
            if (!axes)
              continue;
            for (int64_t axis : axes.asArrayRef()) {
              AxisFacts facts = factsFor(operation, axis);
              if (facts.cohort <= 1)
                continue;
              int64_t memoryAffinity = memoryAffinityFor(operation, axis);
              if (operationAxis < 0 ||
                  facts.priority < operationPriority ||
                  (facts.priority == operationPriority &&
                   (memoryAffinity > operationMemoryAffinity ||
                    (memoryAffinity == operationMemoryAffinity &&
                     facts.cohort > operationCohort)))) {
                operationAxis = axis;
                operationPriority = facts.priority;
                operationCohort = facts.cohort;
                operationMemoryAffinity = memoryAffinity;
              }
            }
          }
        };
        auto considerMappedOperands = [&](mlir::ArrayAttr ids) {
          if (!ids)
            return;
          for (mlir::Attribute idAttribute : ids) {
            llvm::StringRef id =
                mlir::cast<mlir::StringAttr>(idAttribute).getValue();
            int64_t axis = valueLaneAxes.lookup(id);
            if (axis <= 0)
              continue;
            int priority = valueLanePriorities.lookup(id);
            int64_t cohort = valueLaneCohorts.lookup(id);
            if (operationAxis < 0 || priority < operationPriority ||
                (priority == operationPriority && cohort > operationCohort)) {
              operationAxis = axis;
              operationPriority = priority;
              operationCohort = cohort;
            }
          }
        };
        if (operationAxis < 0)
          considerMappedOperands(
              operation.getAs<mlir::ArrayAttr>("operands"));
        auto anchoredNoLaneAxes = [&]() {
          llvm::SmallVector<mlir::DenseI64ArrayAttr, 2> anchors;
          for (llvm::StringRef key : {"operands", "results"})
            if (auto ids = operation.getAs<mlir::ArrayAttr>(key))
              for (mlir::Attribute idAttribute : ids) {
                llvm::StringRef id =
                    mlir::cast<mlir::StringAttr>(idAttribute).getValue();
                if (!valueNoLaneAnchors.lookup(id))
                  continue;
                auto found = valuesById.find(id);
                if (found != valuesById.end())
                  if (auto axes = found->second.getAs<mlir::DenseI64ArrayAttr>(
                          "axes"))
                    anchors.push_back(axes);
              }
          return anchors;
        }();
        bool hasStrongLaneAnchor = false;
        for (llvm::StringRef key : {"operands", "results"})
          if (auto ids = operation.getAs<mlir::ArrayAttr>(key))
            for (mlir::Attribute idAttribute : ids) {
              llvm::StringRef id =
                  mlir::cast<mlir::StringAttr>(idAttribute).getValue();
              hasStrongLaneAnchor |= valueStrongLaneAnchors.lookup(id) &&
                                     valueLaneAxes.lookup(id) > 0;
            }
        if (!anchoredNoLaneAxes.empty())
          operationSliceConflicts[operationId] =
              hasStrongLaneAnchor ? "explicit-lane-anchor-wins"
                                  : "sliced-no-lane-anchor-wins";
        // A reduce result is the parent representation with the reduction axis
        // sliced away.  It therefore acts as an explicit no-lane mapping anchor.
        // Generic pointwise propagation may adopt that sliced mapping, but it may
        // not invent a new lane merely because a surviving free axis has a local
        // cohort.  An explicit primitive/memory anchor (negative priority) still
        // wins and receives a later mechanical broadcast/conversion.
        if (name != "weft_kernel.reduce" && !anchoredNoLaneAxes.empty() &&
            !hasStrongLaneAnchor) {
          operationNoLaneMappings[operationId] = true;
          for (llvm::StringRef key : {"operands", "results"})
            if (auto ids = operation.getAs<mlir::ArrayAttr>(key))
              for (mlir::Attribute idAttribute : ids) {
                llvm::StringRef id =
                    mlir::cast<mlir::StringAttr>(idAttribute).getValue();
                auto found = valuesById.find(id);
                if (found == valuesById.end() ||
                    riscv_internal::integer(found->second, "logical_sew")
                            .value_or(0) <= 0)
                  continue;
                auto axes =
                    found->second.getAs<mlir::DenseI64ArrayAttr>("axes");
                if (!axes || !llvm::any_of(anchoredNoLaneAxes, [&](auto anchor) {
                      return sameAxisSet(anchor, axes);
                    }))
                  continue;
                valueNoLaneAnchors[id] = true;
                valueLaneAxes.erase(id);
                valueLanePriorities.erase(id);
                valueLaneCohorts.erase(id);
              }
          continue;
        }
        if (operationAxis < 0)
          consider(operation.getAs<mlir::ArrayAttr>("results"));
        if (operationAxis < 0)
          consider(operation.getAs<mlir::ArrayAttr>("operands"));
        if (operationAxis < 0)
          continue;
        if (name == "weft_kernel.outer_contract") {
          auto results = operation.getAs<mlir::ArrayAttr>("results");
          auto source =
              operation.getAs<mlir::DictionaryAttr>("source_attributes");
          auto over = source
                          ? source.getAs<mlir::DenseI64ArrayAttr>("over")
                          : mlir::DenseI64ArrayAttr();
          if (results && results.size() == 1 && over && over.size() == 1) {
            auto result = valuesById.find(
                mlir::cast<mlir::StringAttr>(results[0]).getValue());
            auto axes = result == valuesById.end()
                            ? mlir::DenseI64ArrayAttr()
                            : result->second.getAs<mlir::DenseI64ArrayAttr>("axes");
            if (axes)
              for (int64_t axis : axes.asArrayRef())
                if (axis != operationAxis &&
                    axis != over.asArrayRef().front()) {
                  llvm::StringRef operationId =
                      *riscv_internal::string(operation, "id");
                  operationAccumulatorAxes[operationId] = axis;
                  break;
                }
          }
        }
        operationLaneAxes[*riscv_internal::string(operation, "id")] =
            operationAxis;
        operationLaneCohorts[operationId] = operationCohort;
        operationMemoryAffinities[operationId] = operationMemoryAffinity;
        operationLaneReasons[operationId] =
            name == "weft_kernel.iota"
                ? "explicit logical iota axis"
            : operationEliminatedAxes.lookup(operationId) == operationAxis
                ? "explicit reduction axis eliminated by this operation"
            : operationPriority < 0
                ? "explicit packed/derived operand layout"
                : "use-def-connected memory affinity followed by innermost Level cohort";
        for (llvm::StringRef key : {"operands", "results"})
          if (auto ids = operation.getAs<mlir::ArrayAttr>(key))
            for (mlir::Attribute idAttribute : ids) {
              llvm::StringRef id =
                  mlir::cast<mlir::StringAttr>(idAttribute).getValue();
              auto found = valuesById.find(id);
              if (found == valuesById.end())
                continue;
              if (!containsAxis(found->second, operationAxis) ||
                  riscv_internal::integer(found->second, "logical_sew")
                          .value_or(0) <= 0)
                continue;
              auto previous = valueLanePriorities.find(id);
              const int64_t previousCohort = valueLaneCohorts.lookup(id);
              if (previous == valueLanePriorities.end() ||
                  operationPriority < previous->second ||
                  (operationPriority == previous->second &&
                   operationCohort > previousCohort)) {
                valueLaneAxes[id] = operationAxis;
                valueLanePriorities[id] = operationPriority;
                valueLaneCohorts[id] = operationCohort;
              }
            }
      }

      auto propagateLaneIdentities = [&]() {
        bool changed = true;
        while (changed) {
          changed = false;
          for (mlir::Attribute attribute : problem.getOperations()) {
            auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
            llvm::StringRef name =
                riscv_internal::string(operation, "name").value_or("");
            if (!propagatesLaneIdentity(name))
              continue;
            llvm::StringRef operationId =
                riscv_internal::string(operation, "id").value_or("");
            llvm::SmallVector<mlir::DenseI64ArrayAttr, 2> slicedAnchors;
            bool strongLane = false;
            for (llvm::StringRef key : {"operands", "results"})
              if (auto ids = operation.getAs<mlir::ArrayAttr>(key))
                for (mlir::Attribute idAttribute : ids) {
                  llvm::StringRef id =
                      mlir::cast<mlir::StringAttr>(idAttribute).getValue();
                  auto found = valuesById.find(id);
                  if (valueNoLaneAnchors.lookup(id) &&
                      found != valuesById.end())
                    if (auto axes = found->second.getAs<
                            mlir::DenseI64ArrayAttr>("axes"))
                      slicedAnchors.push_back(axes);
                  strongLane |= valueStrongLaneAnchors.lookup(id) &&
                                valueLaneAxes.lookup(id) > 0;
                }
            if (!slicedAnchors.empty() && !strongLane) {
              bool newlySliced = !operationNoLaneMappings.lookup(operationId);
              operationNoLaneMappings[operationId] = true;
              operationLaneAxes.erase(operationId);
              operationSliceConflicts[operationId] =
                  "sliced-no-lane-anchor-wins";
              for (llvm::StringRef key : {"operands", "results"})
                if (auto ids = operation.getAs<mlir::ArrayAttr>(key))
                  for (mlir::Attribute idAttribute : ids) {
                    llvm::StringRef id =
                        mlir::cast<mlir::StringAttr>(idAttribute).getValue();
                    auto found = valuesById.find(id);
                    auto axes =
                        found == valuesById.end()
                            ? mlir::DenseI64ArrayAttr()
                            : found->second.getAs<mlir::DenseI64ArrayAttr>(
                                  "axes");
                    if (!axes || !llvm::any_of(slicedAnchors, [&](auto anchor) {
                          return sameAxisSet(anchor, axes);
                        }))
                      continue;
                    newlySliced |= !valueNoLaneAnchors.lookup(id) ||
                                   valueLaneAxes.lookup(id) > 0;
                    valueNoLaneAnchors[id] = true;
                    valueLaneAxes.erase(id);
                    valueLanePriorities.erase(id);
                    valueLaneCohorts.erase(id);
                  }
              changed |= newlySliced;
              continue;
            }
            int64_t axis = operationLaneAxes.lookup(operationId);
            AxisFacts operationFacts =
                axis > 0 ? factsFor(operation, axis) : AxisFacts{};
            int priority = axis > 0 ? operationFacts.priority : 3;
            int64_t cohort = axis > 0 ? operationFacts.cohort : 1;
            for (llvm::StringRef key : {"results", "operands"})
              if (auto ids = operation.getAs<mlir::ArrayAttr>(key))
                for (mlir::Attribute idAttribute : ids) {
                  llvm::StringRef id =
                      mlir::cast<mlir::StringAttr>(idAttribute).getValue();
                  int64_t candidate = valueLaneAxes.lookup(id);
                  int candidatePriority = valueLanePriorities.lookup(id);
                  int64_t candidateCohort = valueLaneCohorts.lookup(id);
                  if (candidate > 0 &&
                      (axis <= 0 || candidatePriority < priority ||
                       (candidatePriority == priority &&
                        candidateCohort > cohort))) {
                    axis = candidate;
                    priority = candidatePriority;
                    cohort = candidateCohort;
                  }
                }
            if (axis <= 0)
              continue;
            operationLaneAxes[operationId] = axis;
            for (llvm::StringRef key : {"operands", "results"})
              if (auto ids = operation.getAs<mlir::ArrayAttr>(key))
                for (mlir::Attribute idAttribute : ids) {
                  llvm::StringRef id =
                      mlir::cast<mlir::StringAttr>(idAttribute).getValue();
                  auto found = valuesById.find(id);
                  if (found == valuesById.end() ||
                      valueNoLaneAnchors.lookup(id) ||
                      !containsAxis(found->second, axis) ||
                      riscv_internal::integer(found->second, "logical_sew")
                              .value_or(0) <= 0)
                    continue;
                  auto previous = valueLanePriorities.find(id);
                  const int64_t previousCohort = valueLaneCohorts.lookup(id);
                  if (previous == valueLanePriorities.end() ||
                      priority < previous->second ||
                      (priority == previous->second && cohort > previousCohort)) {
                    valueLaneAxes[id] = axis;
                    valueLanePriorities[id] = priority;
                    valueLaneCohorts[id] = cohort;
                    changed = true;
                  }
                }
          }
        }
      };
      propagateLaneIdentities();

      // A reduction slices exactly one logical axis from its input mapping.
      // The initial operation walk may see the reduction before the complete
      // iota/use-def mapping has reached its input, so refresh the slice after
      // lane propagation has stabilized.  If the eliminated axis is not the
      // input lane, the lane and all surviving free axes remain part of the
      // result representation.  Only eliminating the actual lane produces a
      // no-lane result.
      auto refreshReductionMappings = [&]() {
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        if (riscv_internal::string(operation, "name").value_or("") !=
            "weft_kernel.reduce")
          continue;
        auto operands = operation.getAs<mlir::ArrayAttr>("operands");
        auto results = operation.getAs<mlir::ArrayAttr>("results");
        llvm::StringRef operationId =
            riscv_internal::string(operation, "id").value_or("");
        const int64_t eliminatedAxis =
            operationEliminatedAxes.lookup(operationId);
        if (!operands || operands.size() != 1 || !results ||
            results.size() != 1 || eliminatedAxis <= 0)
          continue;
        llvm::StringRef inputId =
            mlir::cast<mlir::StringAttr>(operands[0]).getValue();
        llvm::StringRef resultId =
            mlir::cast<mlir::StringAttr>(results[0]).getValue();
        const int64_t inputLane = valueLaneAxes.lookup(inputId);
        auto resultValue = valuesById.find(resultId);
        if (inputLane <= 0 || resultValue == valuesById.end())
          continue;
        operationLaneAxes[operationId] = inputLane;
        operationLaneCohorts[operationId] = valueLaneCohorts.lookup(inputId);
        operationLaneReasons[operationId] =
            inputLane == eliminatedAxis
                ? "explicit reduction eliminates the input lane axis"
                : "reduce result slices one non-lane axis and preserves the input lane";
        if (inputLane == eliminatedAxis ||
            !containsAxis(resultValue->second, inputLane)) {
          valueNoLaneAnchors[resultId] = true;
          valueLaneAxes.erase(resultId);
          valueLanePriorities.erase(resultId);
          valueLaneCohorts.erase(resultId);
          continue;
        }
        operationNoLaneMappings.erase(operationId);
        operationSliceConflicts.erase(operationId);
        valueNoLaneAnchors.erase(resultId);
        valueLaneAxes[resultId] = inputLane;
        valueLanePriorities[resultId] = valueLanePriorities.lookup(inputId);
        valueLaneCohorts[resultId] = valueLaneCohorts.lookup(inputId);
      }
      };
      refreshReductionMappings();
      propagateLaneIdentities();

      struct HandoffChoice {
        int64_t axis = 0;
        int priority = 3;
        int64_t cohort = 1;
      };
      auto propagateHandoffIdentities = [&]() {
        llvm::StringMap<HandoffChoice> choices;
        for (mlir::Attribute attribute : problem.getValues()) {
          auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
          llvm::StringRef handoff =
              riscv_internal::string(value, "handoff_class").value_or("");
          llvm::StringRef id = *riscv_internal::string(value, "id");
          int64_t axis = valueLaneAxes.lookup(id);
          if (handoff.empty() || axis <= 0 || valueNoLaneAnchors.lookup(id))
            continue;
          int priority = valueLanePriorities.lookup(id);
          int64_t cohort = valueLaneCohorts.lookup(id);
          HandoffChoice &choice = choices[handoff];
          if (choice.axis <= 0 || priority < choice.priority ||
              (priority == choice.priority && cohort > choice.cohort))
            choice = {axis, priority, cohort};
        }
        bool changed = false;
        for (mlir::Attribute attribute : problem.getValues()) {
          auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
          llvm::StringRef handoff =
              riscv_internal::string(value, "handoff_class").value_or("");
          llvm::StringRef id = *riscv_internal::string(value, "id");
          auto choice = choices.find(handoff);
          if (handoff.empty() || choice == choices.end() ||
              valueNoLaneAnchors.lookup(id) ||
              !containsAxis(value, choice->second.axis) ||
              riscv_internal::integer(value, "logical_sew").value_or(0) <= 0)
            continue;
          auto previous = valueLanePriorities.find(id);
          const int64_t previousCohort = valueLaneCohorts.lookup(id);
          if (previous == valueLanePriorities.end() ||
              choice->second.priority < previous->second ||
              (choice->second.priority == previous->second &&
               choice->second.cohort > previousCohort)) {
            valueLaneAxes[id] = choice->second.axis;
            valueLanePriorities[id] = choice->second.priority;
            valueLaneCohorts[id] = choice->second.cohort;
            changed = true;
          }
        }
        return changed;
      };
      propagateHandoffIdentities();
      // Explicit iota defines the complete logical cohort.  Generic propagation
      // may copy that mapping but may never shrink it to the default cohort of
      // an axis that does not come from a Level domain.
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        if (riscv_internal::string(operation, "name").value_or("") !=
            "weft_kernel.iota")
          continue;
        auto results = operation.getAs<mlir::ArrayAttr>("results");
        if (!results || results.size() != 1)
          continue;
        llvm::StringRef id =
            mlir::cast<mlir::StringAttr>(results[0]).getValue();
        auto value = valuesById.find(id);
        auto axes = value == valuesById.end()
                        ? mlir::DenseI64ArrayAttr()
                        : value->second.getAs<mlir::DenseI64ArrayAttr>("axes");
        auto shape = value == valuesById.end()
                         ? mlir::DenseI64ArrayAttr()
                         : value->second.getAs<mlir::DenseI64ArrayAttr>("shape");
        if (!axes || axes.size() != 1 || !shape || shape.size() != 1)
          continue;
        valueLaneAxes[id] = axes[0];
        valueLanePriorities[id] = -4;
        valueLaneCohorts[id] = shape[0];
      }
      propagateLaneIdentities();
      while (propagateHandoffIdentities())
        propagateLaneIdentities();
      auto propagateSlicedHandoffs = [&]() {
        bool changed = false;
        for (const auto &entry : handoffValues) {
          llvm::SmallVector<mlir::DenseI64ArrayAttr, 2> slicedAxes;
          for (const std::string &id : entry.getValue()) {
            if (!valueNoLaneAnchors.lookup(id))
              continue;
            auto found = valuesById.find(id);
            if (found != valuesById.end())
              if (auto axes =
                      found->second.getAs<mlir::DenseI64ArrayAttr>("axes"))
                slicedAxes.push_back(axes);
          }
          if (slicedAxes.empty())
            continue;
          for (const std::string &id : entry.getValue()) {
            auto found = valuesById.find(id);
            auto axes = found == valuesById.end()
                            ? mlir::DenseI64ArrayAttr()
                            : found->second.getAs<mlir::DenseI64ArrayAttr>(
                                  "axes");
            if (!axes || valueStrongLaneAnchors.lookup(id) ||
              !llvm::any_of(slicedAxes, [&](auto anchor) {
                return sameAxisSet(anchor, axes);
              }))
              continue;
            changed |= !valueNoLaneAnchors.lookup(id) ||
                       valueLaneAxes.lookup(id) > 0;
            valueNoLaneAnchors[id] = true;
            valueLaneAxes.erase(id);
            valueLanePriorities.erase(id);
            valueLaneCohorts.erase(id);
          }
        }
        return changed;
      };
      while (propagateSlicedHandoffs())
        propagateLaneIdentities();
      bool slicedChanged = true;
      while (slicedChanged) {
        slicedChanged = false;
        for (mlir::Attribute attribute : problem.getOperations()) {
          auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
          llvm::StringRef name =
              riscv_internal::string(operation, "name").value_or("");
          if (!propagatesLaneIdentity(name))
            continue;
          llvm::SmallVector<mlir::DenseI64ArrayAttr, 2> anchors;
          bool strongLane = false;
          for (llvm::StringRef key : {"operands", "results"})
            if (auto ids = operation.getAs<mlir::ArrayAttr>(key))
              for (mlir::Attribute idAttribute : ids) {
                llvm::StringRef id =
                    mlir::cast<mlir::StringAttr>(idAttribute).getValue();
                auto found = valuesById.find(id);
                if (valueNoLaneAnchors.lookup(id) &&
                    found != valuesById.end())
                  if (auto axes = found->second.getAs<
                          mlir::DenseI64ArrayAttr>("axes"))
                    anchors.push_back(axes);
                strongLane |= valueStrongLaneAnchors.lookup(id) &&
                              valueLaneAxes.lookup(id) > 0;
              }
          if (anchors.empty() || strongLane)
            continue;
          llvm::StringRef operationId =
              riscv_internal::string(operation, "id").value_or("");
          operationNoLaneMappings[operationId] = true;
          operationLaneAxes.erase(operationId);
          operationSliceConflicts[operationId] =
              "sliced-no-lane-anchor-wins";
          for (llvm::StringRef key : {"operands", "results"})
            if (auto ids = operation.getAs<mlir::ArrayAttr>(key))
              for (mlir::Attribute idAttribute : ids) {
                llvm::StringRef id =
                    mlir::cast<mlir::StringAttr>(idAttribute).getValue();
                auto found = valuesById.find(id);
                auto axes = found == valuesById.end()
                                ? mlir::DenseI64ArrayAttr()
                                : found->second.getAs<
                                      mlir::DenseI64ArrayAttr>("axes");
                if (!axes ||
                    !llvm::any_of(anchors, [&](auto anchor) {
                      return sameAxisSet(anchor, axes);
                    }))
                  continue;
                slicedChanged |= !valueNoLaneAnchors.lookup(id) ||
                                 valueLaneAxes.lookup(id) > 0;
                valueNoLaneAnchors[id] = true;
                valueLaneAxes.erase(id);
                valueLanePriorities.erase(id);
                valueLaneCohorts.erase(id);
              }
        }
        slicedChanged |= propagateSlicedHandoffs();
      }
      refreshReductionMappings();
      propagateLaneIdentities();

      // iota establishes a logical axis, not a physical SIMD mandate.  Keep a
      // local axis in lanes only when a non-pointwise entity actually consumes
      // it as a memory, collective, contraction, or packed-compute lane.  An
      // otherwise shaped iota becomes an ordered register tuple, which avoids
      // manufacturing a vector only to immediately extract every lane again at
      // a downstream layout conflict.
      llvm::SmallSet<int64_t, 8> requiredLocalLaneAxes;
      auto anchorsLaneMapping = [](llvm::StringRef name) {
        return name == "weft_kernel.admit" ||
               name == "weft_kernel.commit" ||
               name == "weft_kernel.materialize" ||
               name == "weft_kernel.extract" ||
               name == "weft_kernel.lookup" ||
               name == "weft_kernel.reduce" ||
               name == "weft_kernel.scan" ||
               name == "weft_kernel.fold2" ||
               name == "weft_kernel.mac_pairs" ||
               name == "weft_kernel.mac_groups" ||
               name == "weft_kernel.dot" ||
               name == "weft_kernel.contract" ||
               name == "weft_kernel.outer_contract";
      };
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef id = *riscv_internal::string(operation, "id");
        llvm::StringRef name = *riscv_internal::string(operation, "name");
        int64_t axis = operationLaneAxes.lookup(id);
        if (axis > 0 && anchorsLaneMapping(name))
          requiredLocalLaneAxes.insert(axis);
      }
      for (const auto &entry : explicitLocalAxisExtents) {
        const int64_t axis = entry.first;
        if (requiredLocalLaneAxes.contains(axis))
          continue;
        for (mlir::Attribute attribute : problem.getValues()) {
          auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
          llvm::StringRef id = *riscv_internal::string(value, "id");
          if (valueLaneAxes.lookup(id) != axis ||
              valueStrongLaneAnchors.lookup(id))
            continue;
          valueLaneAxes.erase(id);
          valueLanePriorities.erase(id);
          valueLaneCohorts.erase(id);
        }
        for (mlir::Attribute attribute : problem.getOperations()) {
          auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
          llvm::StringRef id = *riscv_internal::string(operation, "id");
          llvm::StringRef name = *riscv_internal::string(operation, "name");
          if (operationLaneAxes.lookup(id) == axis &&
              !anchorsLaneMapping(name))
            operationLaneAxes.erase(id);
        }
      }

      // A value keeps every source-visible cohort axis that is not its SIMD
      // lane as an ordered register tuple.  This is the projection of the
      // logical value shape onto core-local register repetitions; it is not a
      // second source-level value decomposition.  In particular a reduce
      // result keeps all free axes even though its lane/reduction axis has
      // disappeared from the canonical result type.
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef id = *riscv_internal::string(value, "id");
        if (riscv_internal::integer(value, "logical_sew").value_or(0) <= 0)
          continue;
        const int64_t laneAxis = valueLaneAxes.lookup(id);
        auto axes = value.getAs<mlir::DenseI64ArrayAttr>("axes");
        if (!axes)
          continue;
        llvm::SmallVector<int64_t, 4> registerAxes;
        for (int64_t axis : axes.asArrayRef()) {
          auto facts = axisFacts.find(axis);
          const bool levelRegisterAxis =
              facts != axisFacts.end() && facts->second.priority == 0 &&
              facts->second.cohort > 1;
          const bool explicitLocalRegisterAxis =
              explicitLocalAxisExtents.lookup(axis) > 1;
          if (axis == laneAxis ||
              (!levelRegisterAxis && !explicitLocalRegisterAxis))
            continue;
          registerAxes.push_back(axis);
        }
        if (!registerAxes.empty())
          valueRegisterAxes[id] = std::move(registerAxes);
      }

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
        llvm::StringRef id = *riscv_internal::string(value, "id");
        llvm::StringRef handoff =
            riscv_internal::string(value, "handoff_class").value_or("");
        int64_t axis = valueLaneAxes.lookup(id);
        if (handoff.empty() || axis <= 0)
          continue;
        std::string key = handoff.str() + ":axis" + std::to_string(axis);
        auto inserted = handoffRoots.try_emplace(key, index);
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
        llvm::DenseMap<int64_t, llvm::SmallVector<unsigned, 4>> valuesByAxis;
        for (llvm::StringRef key : {"operands", "results"})
          if (auto ids = operation.getAs<mlir::ArrayAttr>(key))
            for (mlir::Attribute idAttribute : ids) {
              llvm::StringRef id =
                  mlir::cast<mlir::StringAttr>(idAttribute).getValue();
              auto ordinal = valueOrdinals.find(id);
              if (ordinal == valueOrdinals.end())
                continue;
              mlir::DictionaryAttr value = sourceValues[ordinal->second];
              int64_t axis = valueLaneAxes.lookup(id);
              if (axis > 0 && containsAxis(value, axis) &&
                  riscv_internal::integer(value, "logical_sew").value_or(0) > 0)
                valuesByAxis[axis].push_back(ordinal->second);
            }
        for (auto &entry : valuesByAxis) {
          llvm::SmallVectorImpl<unsigned> &laneValues = entry.second;
          if (laneValues.size() < 2)
            continue;
          for (size_t index = 1; index < laneValues.size(); ++index)
            unite(laneValues.front(), laneValues[index]);
          transferEdges.push_back(
              {riscv_internal::string(operation, "id")->str(),
               operationName.str(), laneValues.front()});
        }
      }

      struct ChainFacts {
        int64_t axis = 0;
        int64_t cohort = 1;
        int64_t maximumSEW = 0;
        int64_t maximumExtent = 0;
        int64_t lanes = 1;
        llvm::SmallVector<std::string, 8> operations;
      };
      llvm::DenseMap<unsigned, ChainFacts> chains;
      for (auto [index, value] : llvm::enumerate(sourceValues)) {
        llvm::StringRef id = *riscv_internal::string(value, "id");
        int64_t axis = valueLaneAxes.lookup(id);
        if (axis <= 0 || !containsAxis(value, axis))
          continue;
        int64_t logicalSEW =
            riscv_internal::integer(value, "logical_sew").value_or(0);
        if (logicalSEW <= 0)
          continue;
        ChainFacts &chain = chains[find(index)];
        chain.axis = axis;
        chain.cohort = std::max<int64_t>(chain.cohort,
                                         valueLaneCohorts.lookup(id));
        chain.maximumSEW =
            std::max<int64_t>(chain.maximumSEW, std::max<int64_t>(8, logicalSEW));
        chain.maximumExtent = std::max(chain.maximumExtent, laneExtent(value, axis));
      }
      for (const TransferEdge &edge : transferEdges)
        chains[find(edge.value)].operations.push_back(edge.id + ":" + edge.name);
      if (!chains.empty()) {
        if (!legalLMUL || legalLMUL.empty()) {
          invalidate(problem, builder, "lane axis has no legal target LMUL set");
          continue;
        }
        int64_t maximumLMUL = *llvm::max_element(legalLMUL.asArrayRef());
        bool chainLegal = true;
        for (auto &entry : chains) {
          ChainFacts &chain = entry.second;
          int64_t extentBound =
              chain.maximumExtent > 0 ? chain.maximumExtent : chain.cohort;
          chain.lanes = std::min(
              {chain.cohort, extentBound,
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
      kernel::KernelOp sourceKernel =
          module.lookupSymbol<kernel::KernelOp>(problem.getKernel());
      auto autoBindings = problem.getCandidate().getAs<mlir::DictionaryAttr>(
          "auto_bindings");
      auto resolveValueExtent = [&](int64_t extent) {
        if (extent >= 0 || !sourceKernel)
          return extent;
        const int64_t ordinal = -extent - 1;
        if (ordinal < 0 ||
            ordinal >= static_cast<int64_t>(sourceKernel.getShapeSymbols().size()))
          return int64_t{-1};
        llvm::StringRef symbol =
            mlir::cast<mlir::StringAttr>(
                sourceKernel.getShapeSymbols()[ordinal])
                .getValue();
        auto bound = autoBindings
                         ? autoBindings.getAs<mlir::IntegerAttr>(symbol)
                         : mlir::IntegerAttr();
        return bound ? bound.getInt() : int64_t{-1};
      };
      for (auto [index, original] : llvm::enumerate(sourceValues)) {
        auto value = original;
        std::string id = riscv_internal::string(value, "id")->str();
        int64_t logicalSEW =
            riscv_internal::integer(value, "logical_sew").value_or(0);
        unsigned chainRoot = find(index);
        auto chainIt = chains.find(chainRoot);
        int64_t chainLanes = chainIt == chains.end() ? 1 : chainIt->second.lanes;
        int64_t valueLaneAxis = valueLaneAxes.lookup(id);
        llvm::SmallVector<int64_t, 4> valueRegisterAxisList =
            valueRegisterAxes.lookup(id);
        bool lane = valueLaneAxis > 0 && containsAxis(value, valueLaneAxis) &&
                    logicalSEW > 0 && chainIt != chains.end();
        const bool scalarExtract = singletonScalarExtractInputs.lookup(id) &&
                                   useCounts.lookup(id) == 1;
        if (scalarExtract)
          lane = false;
        std::string kind = physicalKind(value, lane ? valueLaneAxis : 0, chainLanes);
        int64_t logicalLaneExtent =
            lane ? laneExtent(value, valueLaneAxis) : 0;
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
        int64_t registerParts = 1;
        llvm::SmallVector<int64_t, 4> registerExtents;
        auto shape = value.getAs<mlir::DenseI64ArrayAttr>("shape");
        auto axes = value.getAs<mlir::DenseI64ArrayAttr>("axes");
        for (int64_t registerAxis : valueRegisterAxisList) {
          int64_t physicalExtent = 0;
          if (shape && axes && shape.size() == axes.size())
            for (auto [extent, axis] :
                 llvm::zip(shape.asArrayRef(), axes.asArrayRef()))
              if (axis == registerAxis) {
                physicalExtent = resolveValueExtent(extent);
                break;
              }
          auto facts = axisFacts.find(registerAxis);
          if (physicalExtent <= 0 && facts != axisFacts.end())
            physicalExtent = facts->second.cohort;
          if (physicalExtent <= 0) {
            legal = false;
            break;
          }
          registerExtents.push_back(physicalExtent);
          registerParts *= physicalExtent;
        }
        if (!legal)
          break;
        if (!lane && registerParts > 1 && logicalSEW > 0)
          kind = "register-scalar-tuple";
        int64_t vectorParts = streamParts * registerParts;
        int64_t registerGroups =
            lmul ? registerParts * ((lmul + 7) / 8) : 0;
        value = riscv_internal::set(value, "physical_kind",
                                    builder.getStringAttr(kind));
        value = riscv_internal::set(
            value, "physical_encoding_kind",
            builder.getStringAttr(
                riscv_internal::string(value, "encoding_kind")
                    .value_or("not-applicable")));
        value = riscv_internal::set(
            value, "physical_sew", builder.getI64IntegerAttr(physicalSEW));
        value = riscv_internal::set(
            value, "lane_axis",
            builder.getI64IntegerAttr(lane ? valueLaneAxis : 0));
        value = riscv_internal::set(
            value, "physical_lanes", builder.getI64IntegerAttr(valueLanes));
        value = riscv_internal::set(
            value, "stream_parts", builder.getI64IntegerAttr(streamParts));
        value = riscv_internal::set(
            value, "vector_parts", builder.getI64IntegerAttr(vectorParts));
        value = riscv_internal::set(
            value, "register_parts", builder.getI64IntegerAttr(registerParts));
        value = riscv_internal::set(
            value, "register_axes",
            builder.getDenseI64ArrayAttr(valueRegisterAxisList));
        value = riscv_internal::set(
            value, "register_extents",
            builder.getDenseI64ArrayAttr(registerExtents));
        value = riscv_internal::set(
            value, "register_axis",
            builder.getI64IntegerAttr(valueRegisterAxisList.empty()
                                          ? 0
                                          : valueRegisterAxisList.front()));
        value = riscv_internal::set(
            value, "lmul_eighths", builder.getI64IntegerAttr(lmul));
        value = riscv_internal::set(
            value, "lmul",
            builder.getStringAttr(lmul ? lmulSpelling(lmul) : "none"));
        value = riscv_internal::set(
            value, "vl",
            builder.getStringAttr(lane ? "min(" + std::to_string(valueLanes) +
                                             ",remaining-axis" +
                                             std::to_string(valueLaneAxis) + ")"
                                       : "1"));
        value = riscv_internal::set(
            value, "register_groups",
            builder.getI64IntegerAttr(registerGroups));
        value = riscv_internal::set(
            value, "register_groups_per_part",
            builder.getI64IntegerAttr(lmul ? (lmul + 7) / 8 : 0));
        value = riscv_internal::set(
            value, "storage",
            builder.getStringAttr(
                kind == "memory"          ? "pinned-memory"
                : kind == "control"       ? "control"
                : kind == "scalar"        ? "scalar-register"
                : kind == "encoded-record" ? "local-record"
                : kind == "sequential"    ? "scalar-or-rematerialized"
                : kind == "register-scalar-tuple" ? "scalar-register-tuple"
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
                lane ? "axis" + std::to_string(valueLaneAxis) +
                           " use-def chain r" + std::to_string(chainRoot) +
                           " extent=" +
                           std::to_string(chainIt->second.maximumExtent) +
                           " cohort=" + std::to_string(chainIt->second.cohort) +
                           " max-sew=" +
                           std::to_string(chainIt->second.maximumSEW) +
                           " VLEN resource bound"
                : scalarExtract
                    ? "singleton local value has one scalar extract consumer"
                : valueNoLaneAnchors.lookup(id)
                    ? "operation-local sliced mapping carries no SIMD lane"
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
                           std::to_string(valueLaneAxis) + " capped at " +
                           std::to_string(valueLanes)
                     : "scalar/control value"));
        if (lane && chainIt != chains.end())
          value = riscv_internal::set(
              value, "representation_users",
              riscv_internal::strings(builder, chainIt->second.operations));
        if (lane && chainIt != chains.end()) {
          value = riscv_internal::set(
              value, "representation_chain_cohort",
              builder.getI64IntegerAttr(chainIt->second.cohort));
          value = riscv_internal::set(
              value, "representation_chain_maximum_sew",
              builder.getI64IntegerAttr(chainIt->second.maximumSEW));
          value = riscv_internal::set(
              value, "representation_chain_maximum_extent",
              builder.getI64IntegerAttr(chainIt->second.maximumExtent));
        }
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
        int64_t operationLaneAxis = operationLaneAxes.lookup(id);
        if (operationLaneAxis > 0)
          operation = riscv_internal::set(
              operation, "lane_axis",
              builder.getI64IntegerAttr(operationLaneAxis));
        if (operationLaneAxis > 0) {
          operation = riscv_internal::set(
              operation, "lane_axis_derived_from",
              builder.getStringAttr(operationLaneReasons.lookup(id)));
          operation = riscv_internal::set(
              operation, "lane_axis_memory_affinity",
              builder.getI64IntegerAttr(operationMemoryAffinities.lookup(id)));
          operation = riscv_internal::set(
              operation, "lane_axis_cohort",
              builder.getI64IntegerAttr(operationLaneCohorts.lookup(id)));
        } else if (operationNoLaneMappings.lookup(id)) {
          operation = riscv_internal::set(
              operation, "lane_axis", builder.getI64IntegerAttr(0));
          operation = riscv_internal::set(
              operation, "lane_axis_derived_from",
              builder.getStringAttr(
                  "pointwise result preserves a sliced reduction mapping"));
        }
        int64_t accumulatorAxis = operationAccumulatorAxes.lookup(id);
        if (accumulatorAxis > 0)
          operation = riscv_internal::set(
              operation, "accumulator_axis",
              builder.getI64IntegerAttr(accumulatorAxis));
        int64_t eliminatedAxis = operationEliminatedAxes.lookup(id);
        if (eliminatedAxis > 0)
          operation = riscv_internal::set(
              operation, "eliminated_axis",
              builder.getI64IntegerAttr(eliminatedAxis));
        if (auto conflict = operationSliceConflicts.find(id);
            conflict != operationSliceConflicts.end())
          operation = riscv_internal::set(
              operation, "layout_conflict_resolution",
              builder.getStringAttr(conflict->second));
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
