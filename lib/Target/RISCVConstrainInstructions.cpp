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

      llvm::SmallSet<int64_t, 4> candidateLaneAxes;
      for (const auto &entry : values) {
        int64_t axis =
            riscv_internal::integer(entry.getValue(), "lane_axis").value_or(0);
        if (axis > 0)
          candidateLaneAxes.insert(axis);
      }

      llvm::StringMap<mlir::DictionaryAttr> producers;
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
          for (mlir::Attribute operand : operands)
            ++useCounts[mlir::cast<mlir::StringAttr>(operand).getValue()];
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
      bool legal = true;
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
              if (lhs == values.end() || result == values.end()) {
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
          memoryEdge = riscv_internal::dictionary(
              builder,
              {{"form", builder.getStringAttr("unit-stride")},
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
                   {"decision_owner", builder.getStringAttr("memory-edge")}});
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
          if (candidateLaneAxes.size() != 1) {
            legal = false;
          } else {
            localOperation = riscv_internal::dictionary(
                builder,
                {{"lane_axis",
                  builder.getI64IntegerAttr(*candidateLaneAxes.begin())},
                 {"decision_owner", builder.getStringAttr("operation")}});
          }
        } else if (name == "weft_kernel.pack") {
          realization = "primitive-local-pack";
        }

        if (!legal)
          break;
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
      }
      if (!legal) {
        invalidate(problem, builder,
                   "one operation has no legal target-local realization");
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
