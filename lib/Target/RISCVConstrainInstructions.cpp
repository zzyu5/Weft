#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <optional>
#include <string>

using namespace weft;

namespace {

bool targetFlag(mlir::DictionaryAttr target, llvm::StringRef name) {
  auto value = target.getAs<mlir::BoolAttr>(name);
  return value && value.getValue();
}

std::string fieldKey(llvm::StringRef family, llvm::StringRef field) {
  return (family + "::" + field).str();
}

struct EncodingFieldFacts {
  mlir::TypeAttr type;
  mlir::DenseI64ArrayAttr shape;
  mlir::ArrayAttr layouts;
  int64_t bitOffset = 0;
  int64_t storageBits = 0;
};

struct EncodingFacts {
  struct Base {
    int64_t storageBits = 0;
    int64_t alignment = 1;
    int64_t elements = 0;
    std::string bitOrder;
    std::string byteOrder;
  };
  llvm::StringMap<EncodingFieldFacts> fields;
  llvm::StringMap<Base> bases;
  llvm::StringMap<std::string> sourceByDerived;
  llvm::StringMap<int64_t> rowsByDerived;
};

EncodingFacts collectEncodingFacts(mlir::ModuleOp module) {
  EncodingFacts result;
  for (kernel::EncodingDeclOp declaration :
       module.getOps<kernel::EncodingDeclOp>()) {
    result.bases[declaration.getSymName()] = {
        static_cast<int64_t>(declaration.getStorageBits()),
        static_cast<int64_t>(declaration.getAlignment()),
        static_cast<int64_t>(declaration.getElements()),
        declaration.getBitOrder().str(), declaration.getByteOrder().str()};
    for (size_t index = 0; index < declaration.getFieldNames().size(); ++index) {
      llvm::StringRef name =
          mlir::cast<mlir::StringAttr>(declaration.getFieldNames()[index])
              .getValue();
      result.fields[fieldKey(declaration.getSymName(), name)] = {
          mlir::cast<mlir::TypeAttr>(declaration.getFieldTypes()[index]),
          mlir::cast<mlir::DenseI64ArrayAttr>(declaration.getFieldShapes()[index]),
          mlir::cast<mlir::ArrayAttr>(declaration.getFieldLayouts()[index]),
          declaration.getFieldBitOffsets()[index],
          declaration.getFieldStorageBits()[index]};
    }
  }
  for (kernel::DeriveOp derive : module.getOps<kernel::DeriveOp>()) {
    result.sourceByDerived[derive.getResultFamily()] =
        derive.getSourceFamily().str();
    int64_t rows = 0;
    for (mlir::Attribute attribute : derive.getParameters()) {
      llvm::StringRef spelling =
          mlir::cast<mlir::StringAttr>(attribute).getValue();
      if (spelling.consume_front("rows="))
        spelling.getAsInteger(10, rows);
    }
    if (rows > 0)
      result.rowsByDerived[derive.getResultFamily()] = rows;
  }
  return result;
}

llvm::StringRef baseFamily(const EncodingFacts &facts, llvm::StringRef family) {
  llvm::SmallSet<std::string, 4> visited;
  while (true) {
    auto found = facts.sourceByDerived.find(family);
    if (found == facts.sourceByDerived.end())
      break;
    if (!visited.insert(family.str()).second)
      break;
    family = found->second;
  }
  return family;
}

llvm::StringRef firstLayoutKind(mlir::ArrayAttr layouts) {
  if (!layouts || layouts.empty())
    return {};
  auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(layouts[0]);
  auto kind = dictionary ? dictionary.getAs<mlir::StringAttr>("kind")
                         : mlir::StringAttr();
  return kind ? kind.getValue() : llvm::StringRef();
}

bool containsAxis(mlir::DictionaryAttr value, int64_t axis) {
  auto axes = value.getAs<mlir::DenseI64ArrayAttr>("axes");
  return axes && llvm::is_contained(axes.asArrayRef(), axis);
}

mlir::Type logicalElement(mlir::DictionaryAttr value) {
  auto type = value.getAs<mlir::TypeAttr>("type");
  return type ? riscv_internal::logicalElement(type.getValue()) : mlir::Type();
}

int64_t vectorParts(mlir::DictionaryAttr value) {
  return riscv_internal::integer(value, "vector_parts").value_or(1);
}

int64_t wideningPartialTerms(unsigned unsignedWidth) {
  if (unsignedWidth == 0 || unsignedWidth >= 8)
    return 0;
  const int64_t unsignedMaximum = (int64_t{1} << unsignedWidth) - 1;
  const int64_t positiveLimit = 32767 / (unsignedMaximum * 127);
  const int64_t negativeLimit = 32768 / (unsignedMaximum * 128);
  int64_t legalLimit = std::min(positiveLimit, negativeLimit);
  int64_t powerOfTwo = 1;
  while (powerOfTwo <= legalLimit / 2)
    powerOfTwo *= 2;
  return powerOfTwo;
}

const EncodingFieldFacts *fieldFactsFor(
    mlir::DictionaryAttr fieldOperation,
    const llvm::StringMap<mlir::DictionaryAttr> &values,
    const EncodingFacts &encodings) {
  if (riscv_internal::string(fieldOperation, "name").value_or("") !=
      "weft_kernel.field")
    return nullptr;
  auto source =
      fieldOperation.getAs<mlir::DictionaryAttr>("source_attributes");
  auto fieldName = source ? source.getAs<mlir::StringAttr>("name")
                          : mlir::StringAttr();
  auto operands = fieldOperation.getAs<mlir::ArrayAttr>("operands");
  if (!fieldName || !operands || operands.empty())
    return nullptr;
  auto owner = values.find(
      mlir::cast<mlir::StringAttr>(operands[0]).getValue());
  auto family = owner == values.end()
                    ? mlir::StringAttr()
                    : owner->second.getAs<mlir::StringAttr>("encoding_family");
  if (!family)
    return nullptr;
  llvm::StringRef base = baseFamily(encodings, family.getValue());
  auto field = encodings.fields.find(fieldKey(base, fieldName.getValue()));
  return field == encodings.fields.end() ? nullptr : &field->second;
}

const EncodingFieldFacts *fieldFactsForValue(
    llvm::StringRef valueId,
    const llvm::StringMap<mlir::DictionaryAttr> &producers,
    const llvm::StringMap<mlir::DictionaryAttr> &values,
    const EncodingFacts &encodings) {
  while (true) {
    auto producer = producers.find(valueId);
    if (producer == producers.end())
      return nullptr;
    llvm::StringRef name =
        riscv_internal::string(producer->second, "name").value_or("");
    if (name == "weft_kernel.field")
      return fieldFactsFor(producer->second, values, encodings);
    if (name != "weft_kernel.extract")
      return nullptr;
    auto operands = producer->second.getAs<mlir::ArrayAttr>("operands");
    if (!operands || operands.empty())
      return nullptr;
    valueId = mlir::cast<mlir::StringAttr>(operands[0]).getValue();
  }
}

std::string validityFor(mlir::DictionaryAttr operation) {
  auto path = operation.getAs<mlir::ArrayAttr>("level_path");
  if (!path || path.empty())
    return "always";
  std::string result = "intersection(";
  for (auto [index, level] : llvm::enumerate(path)) {
    if (index)
      result += ",";
    result += mlir::cast<mlir::StringAttr>(level).getValue().str() + ".active";
  }
  return result + ")";
}

std::optional<mlir::DictionaryAttr>
selectMatrix(mlir::DictionaryAttr target, mlir::DictionaryAttr operation,
             const llvm::StringMap<mlir::DictionaryAttr> &values) {
  auto operands = operation.getAs<mlir::ArrayAttr>("operands");
  auto results = operation.getAs<mlir::ArrayAttr>("results");
  if (!operands || operands.size() < 2 || !results || results.empty())
    return std::nullopt;
  auto sew = [&](mlir::Attribute id) -> int64_t {
    auto found = values.find(mlir::cast<mlir::StringAttr>(id).getValue());
    return found == values.end()
               ? 0
               : riscv_internal::integer(found->second, "logical_sew").value_or(0);
  };
  int64_t lhs = sew(operands[0]);
  int64_t rhs = sew(operands[1]);
  int64_t result = sew(results[0]);
  if (auto fragments = target.getAs<mlir::ArrayAttr>("matrix_fragments"))
    for (mlir::Attribute attribute : fragments) {
      auto fragment = mlir::cast<mlir::DictionaryAttr>(attribute);
      if (riscv_internal::integer(fragment, "lhs_sew").value_or(0) == lhs &&
          riscv_internal::integer(fragment, "rhs_sew").value_or(0) == rhs &&
          riscv_internal::integer(fragment, "accumulator_sew").value_or(0) ==
              result)
        return fragment;
    }
  return std::nullopt;
}

void invalidate(riscv::ProblemOp problem, mlir::Builder &builder,
                llvm::StringRef reason) {
  problem.setResourcesAttr(riscv_internal::dictionary(
      builder, {{"invalid_reason", builder.getStringAttr(reason)}}));
  problem.setStageAttr(builder.getStringAttr("invalid"));
}

class SelectRISCVLocalOperationsPass final
    : public mlir::PassWrapper<SelectRISCVLocalOperationsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectRISCVLocalOperationsPass)

  llvm::StringRef getArgument() const final {
    return "weft-riscv-select-local-operations";
  }
  llvm::StringRef getDescription() const final {
    return "select operation and memory-edge realizations from typed facts";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::Builder builder(module.getContext());
    EncodingFacts encodings = collectEncodingFacts(module);
    for (riscv::ProblemOp problem : module.getOps<riscv::ProblemOp>()) {
      if (problem.getStage() == "invalid")
        continue;

      if (problem.getStage() != "representations") {
        problem.emitError("operation selection requires assigned representations");
        signalPassFailure();
        return;
      }

      llvm::SmallVector<mlir::Attribute> updatedValues;
      llvm::StringMap<mlir::DictionaryAttr> values;
      for (mlir::Attribute attribute : problem.getValues()) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        auto family = value.getAs<mlir::StringAttr>("encoding_family");
        if (family) {
          auto source = encodings.sourceByDerived.find(family.getValue());
          if (source != encodings.sourceByDerived.end()) {
            auto base = encodings.bases.find(source->second);
            auto rows = encodings.rowsByDerived.find(family.getValue());
            if (base == encodings.bases.end() || rows == encodings.rowsByDerived.end()) {
              invalidate(problem, builder,
                         "derived encoding has no concrete builder layout facts");
              break;
            }
            int64_t vlen =
                *riscv_internal::integer(problem.getTarget(), "vlen_bits");
            value = riscv_internal::set(
                value, "base_encoding_family",
                builder.getStringAttr(source->second));
            value = riscv_internal::set(
                value, "interleave_rows",
                builder.getI64IntegerAttr(rows->second));
            value = riscv_internal::set(
                value, "base_record_storage_bits",
                builder.getI64IntegerAttr(base->second.storageBits));
            value = riscv_internal::set(
                value, "layout_identity",
                builder.getStringAttr(family.getValue().str() + ".rows" +
                                      std::to_string(rows->second) +
                                      ".byte-major.vlen" +
                                      std::to_string(vlen)));
          }
        }
        values[*riscv_internal::string(value, "id")] = value;
        updatedValues.push_back(value);
      }
      if (problem.getStage() == "invalid")
        continue;

      llvm::StringMap<llvm::SmallVector<std::string, 2>> valuesByHandoff;
      for (const auto &entry : values) {
        llvm::StringRef handoff =
            riscv_internal::string(entry.getValue(), "handoff_class")
                .value_or("");
        if (!handoff.empty())
          valuesByHandoff[handoff].push_back(entry.getKey().str());
      }

      llvm::StringMap<mlir::DictionaryAttr> producers;
      llvm::StringMap<llvm::SmallVector<mlir::DictionaryAttr, 2>> users;
      llvm::StringMap<int64_t> useCounts;
      struct ReductionPair {
        std::string maximum;
        std::string minimum;
      };
      llvm::StringMap<ReductionPair> reductionsByInput;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        if (auto results = operation.getAs<mlir::ArrayAttr>("results"))
          for (mlir::Attribute result : results)
            producers[mlir::cast<mlir::StringAttr>(result).getValue()] = operation;
        if (auto operands = operation.getAs<mlir::ArrayAttr>("operands"))
          for (mlir::Attribute operand : operands) {
            llvm::StringRef id =
                mlir::cast<mlir::StringAttr>(operand).getValue();
            ++useCounts[id];
            users[id].push_back(operation);
          }
        if (riscv_internal::string(operation, "name").value_or("") ==
            "weft_kernel.reduce") {
          auto operands = operation.getAs<mlir::ArrayAttr>("operands");
          auto source =
              operation.getAs<mlir::DictionaryAttr>("source_attributes");
          llvm::StringRef kind =
              riscv_internal::string(source, "kind").value_or("");
          if (operands && operands.size() == 1 &&
              (kind == "max" || kind == "min")) {
            llvm::StringRef input =
                mlir::cast<mlir::StringAttr>(operands[0]).getValue();
            std::string id = riscv_internal::string(operation, "id")->str();
            if (kind == "max")
              reductionsByInput[input].maximum = std::move(id);
            else
              reductionsByInput[input].minimum = std::move(id);
          }
        }
      }

      auto collectReachableLanes =
          [&](llvm::StringRef root, llvm::SmallSet<int64_t, 4> &axes,
              llvm::DenseMap<int64_t, int64_t> &widths) {
            llvm::SmallVector<std::string> pending{root.str()};
            llvm::StringSet<> visited;
            while (!pending.empty()) {
              std::string id = std::move(pending.pop_back_val());
              if (!visited.insert(id).second)
                continue;
              if (auto value = values.find(id); value != values.end()) {
                int64_t axis =
                    riscv_internal::integer(value->second, "lane_axis")
                        .value_or(0);
                int64_t lanes =
                    riscv_internal::integer(value->second, "physical_lanes")
                        .value_or(1);
                if (axis > 0 && lanes > 1) {
                  axes.insert(axis);
                  widths[axis] = std::max(widths.lookup(axis), lanes);
                }
                llvm::StringRef handoff =
                    riscv_internal::string(value->second, "handoff_class")
                        .value_or("");
                auto members = valuesByHandoff.find(handoff);
                if (!handoff.empty() && members != valuesByHandoff.end())
                  pending.append(members->second.begin(), members->second.end());
              }
              auto reachableUsers = users.find(id);
              if (reachableUsers == users.end())
                continue;
              for (mlir::DictionaryAttr user : reachableUsers->second)
                if (auto results = user.getAs<mlir::ArrayAttr>("results"))
                  for (mlir::Attribute result : results)
                    pending.push_back(
                        mlir::cast<mlir::StringAttr>(result).getValue().str());
            }
          };

      llvm::StringMap<std::string> coReducePartners;
      llvm::StringMap<std::string> coReduceRoles;
      for (const auto &entry : reductionsByInput) {
        const ReductionPair &pair = entry.getValue();
        if (pair.maximum.empty() || pair.minimum.empty())
          continue;
        coReducePartners[pair.maximum] = pair.minimum;
        coReducePartners[pair.minimum] = pair.maximum;
        coReduceRoles[pair.maximum] = "leader";
        coReduceRoles[pair.minimum] = "follower";
      }

      llvm::SmallVector<mlir::Attribute> operations;
      llvm::StringMap<mlir::DictionaryAttr> memoryEdgesByValue;
      bool legal = true;
      std::string illegalOperation;
      std::string illegalDetail;
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef name = *riscv_internal::string(operation, "name");
        llvm::StringRef engine =
            riscv_internal::string(operation, "engine").value_or("");
        std::string realization = "structural";
        mlir::DictionaryAttr memoryEdge;
        mlir::DictionaryAttr localOperation;

        if (name == "weft_kernel.mac_pairs" || name == "weft_kernel.mac_groups") {
          if ((engine.empty() || engine == "wide") &&
              targetFlag(problem.getTarget(), "has_widening_integer")) {
            realization = "rvv.vwmaccsu.typed";
            auto operands = operation.getAs<mlir::ArrayAttr>("operands");
            auto results = operation.getAs<mlir::ArrayAttr>("results");
            if (!operands || operands.size() < 2 || !results || results.empty()) {
              legal = false;
            } else {
              auto lhs = values.find(
                  mlir::cast<mlir::StringAttr>(operands[0]).getValue());
              auto result = values.find(
                  mlir::cast<mlir::StringAttr>(results[0]).getValue());
              const EncodingFieldFacts *lhsField = fieldFactsForValue(
                  mlir::cast<mlir::StringAttr>(operands[0]).getValue(),
                  producers, values, encodings);
              const EncodingFieldFacts *rhsField = fieldFactsForValue(
                  mlir::cast<mlir::StringAttr>(operands[1]).getValue(),
                  producers, values, encodings);
              auto grouped =
                  lhsField && lhsField->layouts.size() == 2
                      ? mlir::dyn_cast<mlir::DictionaryAttr>(lhsField->layouts[0])
                      : mlir::DictionaryAttr();
              auto layered =
                  lhsField && lhsField->layouts.size() == 2
                      ? mlir::dyn_cast<mlir::DictionaryAttr>(lhsField->layouts[1])
                      : mlir::DictionaryAttr();
              auto groupKind =
                  grouped ? grouped.getAs<mlir::StringAttr>("kind")
                          : mlir::StringAttr();
              auto layerKind =
                  layered ? layered.getAs<mlir::StringAttr>("kind")
                          : mlir::StringAttr();
              auto groupSize =
                  grouped ? grouped.getAs<mlir::IntegerAttr>("size")
                          : mlir::IntegerAttr();
              auto layerSize =
                  layered ? layered.getAs<mlir::IntegerAttr>("size")
                          : mlir::IntegerAttr();
              auto layerOrder =
                  layered ? layered.getAs<mlir::StringAttr>("order")
                          : mlir::StringAttr();
              if (lhs == values.end() || result == values.end() || !lhsField ||
                  !rhsField || firstLayoutKind(rhsField->layouts) != "natural" ||
                  !groupKind || groupKind.getValue() != "grouped" || !layerKind ||
                  layerKind.getValue() != "layered" || !groupSize || !layerSize ||
                  !layerOrder || lhsField->bitOffset % 8 ||
                  groupSize.getInt() <= 0 || layerSize.getInt() <= 0 ||
                  groupSize.getInt() % layerSize.getInt() != 0) {
                legal = false;
              } else {
                localOperation = riscv_internal::dictionary(
                    builder,
                    {{"instruction", builder.getStringAttr("rvv.vwmaccsu")},
                     {"lhs_lmul_eighths",
                      lhs->second.get("lmul_eighths")},
                     {"partial_lmul_eighths",
                      result->second.get("lmul_eighths")},
                     {"partial_sew", builder.getI64IntegerAttr(16)},
                     {"partial_layout",
                      builder.getStringAttr("group-major")},
                     {"lhs_access",
                      builder.getStringAttr(
                          "grouped-layered-constant-stride-window")},
                     {"lhs_group_size", groupSize},
                     {"lhs_layer_size", layerSize},
                     {"lhs_layer_order", layerOrder},
                     {"lhs_bit_offset",
                      builder.getI64IntegerAttr(lhsField->bitOffset)},
                     {"rhs_access",
                      builder.getStringAttr("natural-unit-stride-window")},
                     {"rhs_bit_offset",
                      builder.getI64IntegerAttr(rhsField->bitOffset)},
                     {"decision_owner", builder.getStringAttr("operation")}});
              }
            }
          } else
            legal = false;
        } else if (name == "weft_kernel.dot" || name == "weft_kernel.contract" ||
                   name == "weft_kernel.outer_contract") {
          if (engine == "matrix") {
            auto selected = selectMatrix(problem.getTarget(), operation, values);
            if (selected) {
              llvm::StringRef identity =
                  riscv_internal::string(*selected, "identity").value_or("");
              realization = "matrix." + identity.str();
              auto operands = operation.getAs<mlir::ArrayAttr>("operands");
              auto results = operation.getAs<mlir::ArrayAttr>("results");
              auto result =
                  results && !results.empty()
                      ? values.find(mlir::cast<mlir::StringAttr>(results[0])
                                        .getValue())
                      : values.end();
              const EncodingFieldFacts *lhsField =
                  operands && !operands.empty()
                      ? fieldFactsForValue(
                            mlir::cast<mlir::StringAttr>(operands[0]).getValue(),
                            producers, values, encodings)
                      : nullptr;
              if (result == values.end() || !lhsField ||
                  lhsField->layouts.size() != 2) {
                legal = false;
              } else {
                auto grouped = mlir::dyn_cast<mlir::DictionaryAttr>(
                    lhsField->layouts[0]);
                auto layered = mlir::dyn_cast<mlir::DictionaryAttr>(
                    lhsField->layouts[1]);
                auto groupKind =
                    grouped ? grouped.getAs<mlir::StringAttr>("kind")
                            : mlir::StringAttr();
                auto layerKind =
                    layered ? layered.getAs<mlir::StringAttr>("kind")
                            : mlir::StringAttr();
                auto groupSize =
                    grouped ? grouped.getAs<mlir::IntegerAttr>("size")
                            : mlir::IntegerAttr();
                auto layerSize =
                    layered ? layered.getAs<mlir::IntegerAttr>("size")
                            : mlir::IntegerAttr();
                auto layerOrder =
                    layered ? layered.getAs<mlir::StringAttr>("order")
                            : mlir::StringAttr();
                if (!groupKind || groupKind.getValue() != "grouped" ||
                    !layerKind || layerKind.getValue() != "layered" ||
                    !groupSize || !layerSize || !layerOrder ||
                    groupSize.getInt() != 2 * layerSize.getInt()) {
                  legal = false;
                } else {
                  localOperation = *selected;
                  localOperation = riscv_internal::set(
                      localOperation, "lhs_group_size", groupSize);
                  localOperation = riscv_internal::set(
                      localOperation, "lhs_layer_size", layerSize);
                  localOperation = riscv_internal::set(
                      localOperation, "lhs_layer_order", layerOrder);
                  localOperation = riscv_internal::set(
                      localOperation, "result_vector_suffix",
                      builder.getStringAttr(
                          "i" +
                          std::to_string(riscv_internal::integer(
                                             result->second, "physical_sew")
                                             .value_or(0)) +
                          riscv_internal::string(result->second, "lmul")
                              .value_or("")
                              .str()));
                  localOperation = riscv_internal::set(
                      localOperation, "interleave_rows",
                      localOperation.get("n_factor"));
                  localOperation = riscv_internal::set(
                      localOperation, "decision_owner",
                      builder.getStringAttr("operation"));
                }
              }
            } else
              legal = false;
          } else if (engine.empty() || engine == "wide") {
            bool selectedOuter = false;
            if (name == "weft_kernel.outer_contract") {
              auto operands = operation.getAs<mlir::ArrayAttr>("operands");
              auto results = operation.getAs<mlir::ArrayAttr>("results");
              auto source =
                  operation.getAs<mlir::DictionaryAttr>("source_attributes");
              auto over = source
                              ? source.getAs<mlir::DenseI64ArrayAttr>("over")
                              : mlir::DenseI64ArrayAttr();
              int64_t laneAxis =
                  riscv_internal::integer(operation, "lane_axis").value_or(0);
              int64_t accumulatorAxis =
                  riscv_internal::integer(operation, "accumulator_axis")
                      .value_or(0);
              if (operands && operands.size() == 2 && results &&
                  results.size() == 1 && over && over.size() == 1 &&
                  laneAxis > 0 && accumulatorAxis > 0) {
                llvm::StringRef lhsId =
                    mlir::cast<mlir::StringAttr>(operands[0]).getValue();
                llvm::StringRef rhsId =
                    mlir::cast<mlir::StringAttr>(operands[1]).getValue();
                llvm::StringRef resultId =
                    mlir::cast<mlir::StringAttr>(results[0]).getValue();
                auto lhs = values.find(lhsId);
                auto rhs = values.find(rhsId);
                auto result = values.find(resultId);
                const bool lhsLane =
                    lhs != values.end() && containsAxis(lhs->second, laneAxis);
                const bool rhsLane =
                    rhs != values.end() && containsAxis(rhs->second, laneAxis);
                llvm::StringRef laneId = lhsLane && !rhsLane ? lhsId : rhsId;
                llvm::StringRef repeatedId = lhsLane && !rhsLane ? rhsId : lhsId;
                auto lane = values.find(laneId);
                auto repeated = values.find(repeatedId);
                const EncodingFieldFacts *laneField = fieldFactsForValue(
                    laneId, producers, values, encodings);
                const EncodingFieldFacts *repeatedField = fieldFactsForValue(
                    repeatedId, producers, values, encodings);
                auto laneInteger =
                    lane == values.end()
                        ? mlir::IntegerType()
                        : mlir::dyn_cast<mlir::IntegerType>(
                              logicalElement(lane->second));
                auto repeatedInteger =
                    repeated == values.end()
                        ? mlir::IntegerType()
                        : mlir::dyn_cast<mlir::IntegerType>(
                              logicalElement(repeated->second));
                auto resultInteger =
                    result == values.end()
                        ? mlir::IntegerType()
                        : mlir::dyn_cast<mlir::IntegerType>(
                              logicalElement(result->second));
                const int64_t laneLMUL =
                    lane == values.end()
                        ? 0
                        : riscv_internal::integer(lane->second,
                                                  "lmul_eighths")
                              .value_or(0);
                const int64_t resultLMUL =
                    result == values.end()
                        ? 0
                        : riscv_internal::integer(result->second,
                                                  "lmul_eighths")
                              .value_or(0);
                const int64_t parts =
                    result == values.end() ? 0 : vectorParts(result->second);
                if (lhsLane != rhsLane && laneField && repeatedField &&
                    laneInteger && repeatedInteger && resultInteger &&
                    laneInteger.isUnsigned() && laneInteger.getWidth() < 8 &&
                    repeatedInteger.isSigned() &&
                    repeatedInteger.getWidth() == 8 &&
                    resultInteger.isSigned() &&
                    resultInteger.getWidth() == 32 &&
                    firstLayoutKind(repeatedField->layouts) == "natural" &&
                    laneLMUL > 0 && resultLMUL > 0 && parts > 0) {
                  const int64_t partialLMUL = laneLMUL * 2;
                  const int64_t partialTerms =
                      wideningPartialTerms(laneInteger.getWidth());
                  if (partialTerms <= 0) {
                    legal = false;
                    continue;
                  }
                  const int64_t temporaryGroups =
                      parts * ((partialLMUL + 7) / 8);
                  realization = "rvv.outer.encoded-widening-mac";
                  localOperation = riscv_internal::dictionary(
                      builder,
                      {{"instruction", builder.getStringAttr("rvv.vwmaccsu")},
                       {"lane_operand",
                        builder.getStringAttr(lhsLane ? "lhs" : "rhs")},
                       {"lane_axis", builder.getI64IntegerAttr(laneAxis)},
                       {"accumulator_axis",
                        builder.getI64IntegerAttr(accumulatorAxis)},
                       {"reduction_axis",
                        builder.getI64IntegerAttr(over.asArrayRef().front())},
                       {"lane_lmul_eighths",
                        builder.getI64IntegerAttr(laneLMUL)},
                       {"partial_lmul_eighths",
                        builder.getI64IntegerAttr(partialLMUL)},
                       {"accumulator_lmul_eighths",
                        builder.getI64IntegerAttr(resultLMUL)},
                       {"partial_sew", builder.getI64IntegerAttr(16)},
                       {"accumulator_sew", builder.getI64IntegerAttr(32)},
                       {"partial_terms_max",
                        builder.getI64IntegerAttr(partialTerms)},
                       {"accumulator_parts", builder.getI64IntegerAttr(parts)},
                       {"temporary_vector_groups",
                        builder.getI64IntegerAttr(temporaryGroups)},
                       {"decision_owner", builder.getStringAttr("operation")}});
                  llvm::StringRef laneAccess = "mapped-field-per-term";
                  if (laneField->layouts && laneField->layouts.size() >= 2) {
                    auto grouped = mlir::dyn_cast<mlir::DictionaryAttr>(
                        laneField->layouts[0]);
                    auto layered = mlir::dyn_cast<mlir::DictionaryAttr>(
                        laneField->layouts[1]);
                    auto groupedKind =
                        grouped ? grouped.getAs<mlir::StringAttr>("kind")
                                : mlir::StringAttr();
                    auto layeredKind =
                        layered ? layered.getAs<mlir::StringAttr>("kind")
                                : mlir::StringAttr();
                    auto groupSize =
                        grouped ? grouped.getAs<mlir::IntegerAttr>("size")
                                : mlir::IntegerAttr();
                    auto layerSize =
                        layered ? layered.getAs<mlir::IntegerAttr>("size")
                                : mlir::IntegerAttr();
                    auto layerOrder =
                        layered ? layered.getAs<mlir::StringAttr>("order")
                                : mlir::StringAttr();
                    if (groupedKind && groupedKind.getValue() == "grouped" &&
                        layeredKind && layeredKind.getValue() == "layered" &&
                        groupSize && layerSize && layerOrder) {
                      laneAccess =
                          "grouped-layered-constant-stride-window";
                      localOperation = riscv_internal::set(
                          localOperation, "lane_group_size", groupSize);
                      localOperation = riscv_internal::set(
                          localOperation, "lane_layer_size", layerSize);
                      localOperation = riscv_internal::set(
                          localOperation, "lane_layer_order", layerOrder);
                    }
                  }
                  localOperation = riscv_internal::set(
                      localOperation, "lane_access",
                      builder.getStringAttr(laneAccess));
                  localOperation = riscv_internal::set(
                      localOperation, "lane_bit_offset",
                      builder.getI64IntegerAttr(laneField->bitOffset));
                  localOperation = riscv_internal::set(
                      localOperation, "repeated_bit_offset",
                      builder.getI64IntegerAttr(repeatedField->bitOffset));
                  selectedOuter = true;
                } else if (lhsLane != rhsLane && laneField && laneInteger &&
                           laneInteger.isUnsigned() &&
                           laneInteger.getWidth() <= 8 && resultInteger &&
                           resultInteger.isSigned() &&
                           resultInteger.getWidth() == 32 && laneLMUL > 0 &&
                           resultLMUL > 0 && parts > 0) {
                  auto repeatedProducer = producers.find(repeatedId);
                  auto foldOperands =
                      repeatedProducer == producers.end()
                          ? mlir::ArrayAttr()
                          : repeatedProducer->second.getAs<mlir::ArrayAttr>(
                                "operands");
                  const bool folded =
                      repeatedProducer != producers.end() &&
                      riscv_internal::string(repeatedProducer->second, "name")
                              .value_or("") == "weft_kernel.fold2" &&
                      foldOperands && foldOperands.size() == 1;
                  llvm::StringRef foldInputId =
                      folded
                          ? mlir::cast<mlir::StringAttr>(foldOperands[0])
                                .getValue()
                          : llvm::StringRef();
                  const EncodingFieldFacts *foldInputField =
                      folded ? fieldFactsForValue(foldInputId, producers, values,
                                                  encodings)
                             : nullptr;
                  auto foldInput = values.find(foldInputId);
                  auto foldInputInteger =
                      foldInput == values.end()
                          ? mlir::IntegerType()
                          : mlir::dyn_cast<mlir::IntegerType>(
                                logicalElement(foldInput->second));
                  if (folded && foldInputField && foldInputInteger &&
                      foldInputInteger.isSigned() &&
                      foldInputInteger.getWidth() == 16 &&
                      firstLayoutKind(foldInputField->layouts) == "natural" &&
                      laneField->shape && !laneField->shape.empty()) {
                    realization = "rvv.outer.encoded-fold-product";
                    localOperation = riscv_internal::dictionary(
                        builder,
                        {{"instruction", builder.getStringAttr("rvv.vmacc")},
                         {"lane_operand",
                          builder.getStringAttr(lhsLane ? "lhs" : "rhs")},
                         {"lane_axis", builder.getI64IntegerAttr(laneAxis)},
                         {"accumulator_axis",
                          builder.getI64IntegerAttr(accumulatorAxis)},
                         {"reduction_axis",
                          builder.getI64IntegerAttr(over.asArrayRef().front())},
                         {"accumulator_lmul_eighths",
                          builder.getI64IntegerAttr(resultLMUL)},
                         {"accumulator_sew", builder.getI64IntegerAttr(32)},
                         {"accumulator_parts",
                          builder.getI64IntegerAttr(parts)},
                         {"group_count",
                          builder.getI64IntegerAttr(
                              laneField->shape.asArrayRef().back())},
                         {"rhs_pairs_per_group",
                          builder.getI64IntegerAttr(2)},
                         {"temporary_vector_groups",
                          builder.getI64IntegerAttr((resultLMUL + 7) / 8)},
                         {"decision_owner",
                          builder.getStringAttr("operation")}});
                    selectedOuter = true;
                  }
                }
              }
            }
            if (!selectedOuter)
              realization = "rvv.reduction-product";
            if (name == "weft_kernel.dot") {
              auto operands = operation.getAs<mlir::ArrayAttr>("operands");
              if (operands && operands.size() == 2) {
                auto lhs = producers.find(
                    mlir::cast<mlir::StringAttr>(operands[0]).getValue());
                auto rhs = producers.find(
                    mlir::cast<mlir::StringAttr>(operands[1]).getValue());
                if (lhs != producers.end() && rhs != producers.end() &&
                    riscv_internal::string(lhs->second, "name").value_or("") ==
                        "weft_kernel.field" &&
                    riscv_internal::string(rhs->second, "name").value_or("") ==
                        "weft_kernel.fold2") {
                  realization = "rvv.dot.encoded-field-folded-i16";
                  const EncodingFieldFacts *lhsField =
                      fieldFactsFor(lhs->second, values, encodings);
                  auto results = operation.getAs<mlir::ArrayAttr>("results");
                  auto result =
                      results && !results.empty()
                          ? values.find(mlir::cast<mlir::StringAttr>(
                                            results[0])
                                            .getValue())
                          : values.end();
                  if (!lhsField || !lhsField->shape ||
                      lhsField->shape.empty() || result == values.end()) {
                    legal = false;
                  } else {
                    localOperation = riscv_internal::dictionary(
                        builder,
                        {{"instruction", builder.getStringAttr("rvv.vmacc")},
                         {"accumulator_sew",
                          builder.getI64IntegerAttr(32)},
                         {"accumulator_lmul_eighths",
                          result->second.get("lmul_eighths")},
                         {"group_count",
                          builder.getI64IntegerAttr(
                              lhsField->shape.asArrayRef().back())},
                         {"rhs_pairs_per_group",
                          builder.getI64IntegerAttr(2)},
                         {"decision_owner",
                          builder.getStringAttr("operation")}});
                  }
                }
              }
            }
          } else {
            legal = false;
          }
        } else if (name == "weft_kernel.lookup") {
          if (engine == "scalar")
            realization = "scalar.lookup";
          else if ((engine.empty() || engine == "wide") &&
                   targetFlag(problem.getTarget(), "has_indexed_memory"))
            realization = "rvv.indexed-lookup";
          else
            legal = false;
        } else if (name == "weft_kernel.admit" || name == "weft_kernel.commit") {
          realization = name == "weft_kernel.admit" ? "transfer.rvv.load"
                                                     : "transfer.rvv.store";
          auto operands = operation.getAs<mlir::ArrayAttr>("operands");
          auto results = operation.getAs<mlir::ArrayAttr>("results");
          llvm::StringRef transferredId;
          if (name == "weft_kernel.admit" && results && !results.empty())
            transferredId = mlir::cast<mlir::StringAttr>(results[0]).getValue();
          else if (name == "weft_kernel.commit" && operands && !operands.empty())
            transferredId = mlir::cast<mlir::StringAttr>(operands[0]).getValue();
          auto transferred = values.find(transferredId);
          llvm::StringRef physical =
              transferred == values.end()
                  ? llvm::StringRef()
                  : riscv_internal::string(transferred->second, "physical_kind")
                        .value_or("");
          llvm::StringRef form =
              physical == "encoded-record" ? "record-address"
              : physical.starts_with("rvv") ? "runtime-strided"
                                             : "scalar-address";
          memoryEdge = riscv_internal::dictionary(
              builder,
              {{"form", builder.getStringAttr(form)},
               {"stride_source",
                builder.getStringAttr(physical.starts_with("rvv")
                                          ? "runtime View stride on selected lane axis"
                                          : "not-applicable")},
               {"derived_from",
                builder.getStringAttr(
                    "typed transfer value; concrete View stride is a runtime edge fact")},
               {"decision_owner", builder.getStringAttr("memory-edge")}});
        } else if (name == "weft_kernel.field") {
          auto source = operation.getAs<mlir::DictionaryAttr>("source_attributes");
          auto operands = operation.getAs<mlir::ArrayAttr>("operands");
          auto fieldName = source ? source.getAs<mlir::StringAttr>("name")
                                  : mlir::StringAttr();
          auto owner = operands && !operands.empty()
                           ? values.find(mlir::cast<mlir::StringAttr>(operands[0])
                                             .getValue())
                           : values.end();
          auto family = owner != values.end()
                            ? owner->second.getAs<mlir::StringAttr>(
                                  "encoding_family")
                            : mlir::StringAttr();
          if (!fieldName || !family) {
            legal = false;
          } else {
            llvm::StringRef base = baseFamily(encodings, family.getValue());
            auto field = encodings.fields.find(fieldKey(base, fieldName.getValue()));
            if (field == encodings.fields.end()) {
              legal = false;
            } else {
              llvm::StringRef kind = firstLayoutKind(field->second.layouts);
              if (kind == "natural")
                realization = "encoded-field.natural";
              else if (kind == "grouped")
                realization = "encoded-field.layered.and-shift";
              else if (kind == "joined")
                realization = "encoded-field.joined.and-shift";
              else
                legal = false;
              int64_t resultSEW = 0;
              int64_t resultLMUL = 0;
              if (auto results = operation.getAs<mlir::ArrayAttr>("results");
                  results && results.size() == 1) {
                auto result = values.find(
                    mlir::cast<mlir::StringAttr>(results[0]).getValue());
                if (result != values.end()) {
                  resultSEW = riscv_internal::integer(
                                  result->second, "physical_sew")
                                  .value_or(0);
                  resultLMUL = riscv_internal::integer(
                                   result->second, "lmul_eighths")
                                   .value_or(0);
                }
              }
              int64_t rawLMUL =
                  resultSEW > 0
                      ? std::max<int64_t>(1, resultLMUL * 8 / resultSEW)
                      : 0;
              memoryEdge = riscv_internal::dictionary(
                  builder,
                  {{"base_family", builder.getStringAttr(base)},
                   {"field", fieldName},
                   {"field_type", field->second.type},
                   {"field_shape", field->second.shape},
                   {"layout", field->second.layouts},
                   {"bit_offset",
                    builder.getI64IntegerAttr(field->second.bitOffset)},
                   {"storage_bits",
                    builder.getI64IntegerAttr(field->second.storageBits)},
                   {"unpack", builder.getStringAttr(
                                  kind == "natural" ? "none" : "and-shift")},
                   {"raw_sew", builder.getI64IntegerAttr(8)},
                   {"raw_lmul_eighths",
                    builder.getI64IntegerAttr(rawLMUL)},
                   {"form", builder.getStringAttr("field-layout-definition")},
                   {"interleave_rows",
                    builder.getI64IntegerAttr(
                        owner == values.end()
                            ? 0
                            : riscv_internal::integer(owner->second,
                                                      "interleave_rows")
                                  .value_or(0))},
                   {"derived_from",
                    builder.getStringAttr(
                        "encoding declaration; concrete access belongs to each extract")},
                   {"decision_owner", builder.getStringAttr("memory-edge")}});
            }
          }
        } else if (name == "weft_kernel.extract") {
          auto operands = operation.getAs<mlir::ArrayAttr>("operands");
          auto results = operation.getAs<mlir::ArrayAttr>("results");
          if (!operands || operands.empty() || !results || results.size() != 1) {
            legal = false;
          } else {
            llvm::StringRef input =
                mlir::cast<mlir::StringAttr>(operands[0]).getValue();
            auto inherited = memoryEdgesByValue.find(input);
            if (inherited != memoryEdgesByValue.end()) {
              memoryEdge = inherited->second;
              auto result = values.find(
                  mlir::cast<mlir::StringAttr>(results[0]).getValue());
              int64_t laneAxis =
                  result == values.end()
                      ? 0
                      : riscv_internal::integer(result->second, "lane_axis")
                            .value_or(0);
              int64_t resultSEW =
                  result == values.end()
                      ? 0
                      : riscv_internal::integer(result->second, "physical_sew")
                            .value_or(0);
              int64_t resultLMUL =
                  result == values.end()
                      ? 0
                      : riscv_internal::integer(result->second, "lmul_eighths")
                            .value_or(0);
              int64_t rows =
                  riscv_internal::integer(memoryEdge, "interleave_rows")
                      .value_or(0);
              auto layout = memoryEdge.getAs<mlir::ArrayAttr>("layout");
              llvm::StringRef layoutKind = firstLayoutKind(layout);
              llvm::StringRef form =
                  laneAxis > 0 && rows > 0 && layoutKind == "grouped"
                      ? "unit-stride-layered-unpack"
                  : laneAxis > 0 && rows > 0 && layoutKind == "joined"
                      ? "unit-stride-multi-load-join"
                  : laneAxis > 0 && rows > 0
                      ? "unit-stride-interleaved"
                  : layoutKind == "joined" ? "scalar-multi-load-join"
                                            : "scalar-indexed";
              memoryEdge = riscv_internal::set(
                  memoryEdge, "form", builder.getStringAttr(form));
              if (laneAxis > 0 && resultSEW > 0 && resultLMUL > 0)
                memoryEdge = riscv_internal::set(
                    memoryEdge, "raw_lmul_eighths",
                    builder.getI64IntegerAttr(
                        std::max<int64_t>(1, resultLMUL * 8 / resultSEW)));
              memoryEdge = riscv_internal::set(
                  memoryEdge, "access_value",
                  mlir::cast<mlir::StringAttr>(results[0]));
              memoryEdge = riscv_internal::set(
                  memoryEdge, "derived_from",
                  builder.getStringAttr(
                      "this extract selector + encoding mapping + selected lane axis"));
              realization = "encoded-access." + form.str();
            } else {
              realization = "mapped-local-extract";
            }
          }
        } else if (name == "weft_kernel.widen") {
          auto results = operation.getAs<mlir::ArrayAttr>("results");
          realization = results && results.size() == 1 &&
                                useCounts.lookup(
                                    mlir::cast<mlir::StringAttr>(results[0])
                                        .getValue()) == 1
                            ? "rvv.widen-preserve-or-defer"
                            : "rvv.widen-preserve-lanes";
        } else if (name == "weft_kernel.reduce") {
          realization = "rvv.reduce.streamed";
          llvm::StringRef id = *riscv_internal::string(operation, "id");
          auto partner = coReducePartners.find(id);
          if (partner != coReducePartners.end()) {
            realization = "rvv.co-reduce.max-min." + coReduceRoles.lookup(id);
            operation = riscv_internal::set(
                operation, "co_reduce_partner",
                builder.getStringAttr(partner->second));
            localOperation = riscv_internal::dictionary(
                builder,
                {{"instructions",
                  builder.getStringAttr("rvv.vfredmax+rvv.vfredmin")},
                 {"input_materialization",
                  builder.getStringAttr("shared-per-stream-part")},
                 {"decision_owner", builder.getStringAttr("operation-cluster")}});
          }
          auto operands = operation.getAs<mlir::ArrayAttr>("operands");
          if (partner == coReducePartners.end() && operands &&
              operands.size() == 1) {
            auto producer = producers.find(
                mlir::cast<mlir::StringAttr>(operands[0]).getValue());
            if (producer != producers.end() &&
                riscv_internal::string(producer->second, "name").value_or("") ==
                    "weft_kernel.widen") {
              llvm::StringRef kind = riscv_internal::string(
                                         operation.getAs<mlir::DictionaryAttr>(
                                             "source_attributes"),
                                         "kind")
                                         .value_or("unknown");
              realization = "rvv.widen-reduce." + kind.str();
            }
          }
        } else if (name == "weft_kernel.fold2") {
          realization = "ordered-pair-fold";
        } else if (name == "weft_kernel.binary" || name == "weft_kernel.unary" ||
                   name == "weft_kernel.cast" || name == "weft_kernel.compare") {
          realization = "mapped-pointwise";
        } else if (name == "weft_kernel.for" || name == "weft_kernel.if" ||
                   name == "weft_kernel.while") {
          realization = "ordered-scalar-control";
        } else if (name == "weft_kernel.level") {
          realization = "logical-level";
        } else if (name == "weft_kernel.materialize") {
          realization = "stage-once";
          auto operands = operation.getAs<mlir::ArrayAttr>("operands");
          auto results = operation.getAs<mlir::ArrayAttr>("results");
          llvm::SmallSet<int64_t, 4> reachableLaneAxes;
          llvm::DenseMap<int64_t, int64_t> reachableLaneWidths;
          if (results && results.size() == 1)
            collectReachableLanes(
                mlir::cast<mlir::StringAttr>(results[0]).getValue(),
                reachableLaneAxes, reachableLaneWidths);
          if (!operands || operands.size() != 1 || !results ||
              results.size() != 1 || reachableLaneAxes.size() != 1) {
            illegalDetail = "reachable lane axis count=" +
                            std::to_string(reachableLaneAxes.size());
            legal = false;
          } else {
            const int64_t laneAxis = *reachableLaneAxes.begin();
            localOperation = riscv_internal::dictionary(
                builder,
                {{"lane_axis",
                  builder.getI64IntegerAttr(laneAxis)},
                 {"decision_owner", builder.getStringAttr("operation")}});
            llvm::StringRef packId =
                mlir::cast<mlir::StringAttr>(operands[0]).getValue();
            auto packProducer = producers.find(packId);
            auto packOperands =
                packProducer == producers.end()
                    ? mlir::ArrayAttr()
                    : packProducer->second.getAs<mlir::ArrayAttr>("operands");
            if (packProducer == producers.end() || !packOperands ||
                packOperands.size() != 1 ||
                riscv_internal::string(packProducer->second, "name")
                        .value_or("") != "weft_kernel.pack") {
              illegalDetail = "materialize input is not one explicit pack request";
              legal = false;
            } else {
              llvm::StringRef sourceId =
                  mlir::cast<mlir::StringAttr>(packOperands[0]).getValue();
              auto source = values.find(sourceId);
              auto sourceAxes =
                  source == values.end()
                      ? mlir::DenseI64ArrayAttr()
                      : source->second.getAs<mlir::DenseI64ArrayAttr>("axes");
              if (!sourceAxes || sourceAxes.size() != 2 ||
                  !llvm::is_contained(sourceAxes.asArrayRef(), laneAxis)) {
                illegalDetail =
                    "local pack source is not a rank-two value carrying its lane axis";
                legal = false;
              } else {
                const int64_t recordAxis =
                    sourceAxes.asArrayRef().front() == laneAxis
                        ? sourceAxes.asArrayRef().back()
                        : sourceAxes.asArrayRef().front();
                localOperation = riscv_internal::set(
                    localOperation, "source_rank", builder.getI64IntegerAttr(2));
                localOperation = riscv_internal::set(
                    localOperation, "row_axis",
                    builder.getI64IntegerAttr(laneAxis));
                localOperation = riscv_internal::set(
                    localOperation, "record_axis",
                    builder.getI64IntegerAttr(recordAxis));
                auto family =
                    source == values.end()
                        ? mlir::StringAttr()
                        : source->second.getAs<mlir::StringAttr>(
                              "encoding_family");
                llvm::StringRef base =
                    family ? baseFamily(encodings, family.getValue())
                           : llvm::StringRef();
                auto encoding = encodings.bases.find(base);
                if (encoding != encodings.bases.end()) {
                  const int64_t rows = reachableLaneWidths.lookup(laneAxis);
                  if (rows <= 1) {
                    illegalDetail =
                        "encoded local pack has no multi-row lane cohort";
                    legal = false;
                  } else {
                    localOperation = riscv_internal::set(
                        localOperation, "packing",
                        builder.getStringAttr("encoded-record-interleave"));
                    localOperation = riscv_internal::set(
                        localOperation, "source_encoding_family",
                        builder.getStringAttr(base));
                    localOperation = riscv_internal::set(
                        localOperation, "interleave_rows",
                        builder.getI64IntegerAttr(rows));
                    localOperation = riscv_internal::set(
                        localOperation, "record_storage_bits",
                        builder.getI64IntegerAttr(encoding->second.storageBits));
                    localOperation = riscv_internal::set(
                        localOperation, "logical_elements",
                        builder.getI64IntegerAttr(encoding->second.elements));
                  }
                }
              }
            }
          }
        } else if (name == "weft_kernel.pack") {
          realization = "primitive-local-pack";
        }

        if (!legal) {
          illegalOperation =
              riscv_internal::string(operation, "id").value_or("unknown").str() +
              ":" + name.str();
          if (!illegalDetail.empty())
            illegalOperation += " (" + illegalDetail + ")";
          break;
        }
        operation = riscv_internal::set(
            operation, "source_op", operation.get("name"));
        operation = riscv_internal::set(
            operation, "realization", builder.getStringAttr(realization));
        operation = riscv_internal::set(
            operation, "validity", builder.getStringAttr(validityFor(operation)));
        operation = riscv_internal::set(
            operation, "decision_owner", builder.getStringAttr("operation"));
        if (memoryEdge)
          operation = riscv_internal::set(operation, "memory_edge", memoryEdge);
        if (localOperation)
          operation = riscv_internal::set(operation, "local_operation",
                                          localOperation);
        operations.push_back(operation);
        if (memoryEdge)
          if (auto results = operation.getAs<mlir::ArrayAttr>("results"))
            for (mlir::Attribute result : results)
              memoryEdgesByValue[mlir::cast<mlir::StringAttr>(result).getValue()] =
                  memoryEdge;
      }
      if (!legal) {
        invalidate(problem, builder,
                   "operation has no legal target-local realization: " +
                       illegalOperation);
        continue;
      }
      problem.setValuesAttr(builder.getArrayAttr(updatedValues));
      problem.setOperationsAttr(builder.getArrayAttr(operations));
      problem.setStageAttr(builder.getStringAttr("operations"));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createSelectRISCVLocalOperationsPass() {
  return std::make_unique<SelectRISCVLocalOperationsPass>();
}
