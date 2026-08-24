#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"

#include <optional>
#include <string>

using namespace weft;

namespace {

struct BaseEncoding {
  int64_t storageBits = 0;
  int64_t elements = 0;
};

struct EncodingMappings {
  llvm::StringMap<BaseEncoding> bases;
  llvm::StringMap<std::string> sourceByDerived;
  llvm::StringMap<int64_t> rowsByDerived;
};

struct StorageMapping {
  std::string baseFamily;
  std::string identity;
  int64_t rows = 0;
  int64_t recordStorageBits = 0;
  int64_t logicalElements = 0;
  int64_t rowAxis = 0;
  int64_t recordAxis = 0;

  bool operator==(const StorageMapping &other) const {
    return baseFamily == other.baseFamily && identity == other.identity &&
           rows == other.rows && recordStorageBits == other.recordStorageBits &&
           logicalElements == other.logicalElements && rowAxis == other.rowAxis &&
           recordAxis == other.recordAxis;
  }
};

EncodingMappings collectEncodingMappings(mlir::ModuleOp module) {
  EncodingMappings result;
  for (kernel::EncodingDeclOp declaration :
       module.getOps<kernel::EncodingDeclOp>())
    result.bases[declaration.getSymName()] = {
        static_cast<int64_t>(declaration.getStorageBits()),
        static_cast<int64_t>(declaration.getElements())};
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

llvm::StringRef baseFamily(const EncodingMappings &encodings,
                           llvm::StringRef family) {
  llvm::SmallSet<std::string, 4> visited;
  while (visited.insert(family.str()).second) {
    auto source = encodings.sourceByDerived.find(family);
    if (source == encodings.sourceByDerived.end())
      break;
    family = source->second;
  }
  return family;
}

int64_t axisForSymbol(kernel::KernelOp kernel, llvm::StringRef symbol) {
  for (auto [ordinal, attribute] : llvm::enumerate(kernel.getShapeSymbols()))
    if (mlir::cast<mlir::StringAttr>(attribute)
            .getValue()
            .equals_insensitive(symbol))
      return static_cast<int64_t>(ordinal) + 1;
  return 0;
}

int64_t physicalWidthOnAxis(mlir::DictionaryAttr value, int64_t axis) {
  if (riscv_internal::integer(value, "lane_axis").value_or(0) == axis)
    return riscv_internal::integer(value, "physical_lanes").value_or(1) *
           riscv_internal::integer(value, "stream_parts").value_or(1);
  auto axes = value.getAs<mlir::DenseI64ArrayAttr>("register_axes");
  auto extents = value.getAs<mlir::DenseI64ArrayAttr>("register_extents");
  if (!axes || !extents || axes.size() != extents.size())
    return 0;
  for (auto [candidate, extent] : llvm::zip(axes.asArrayRef(),
                                            extents.asArrayRef()))
    if (candidate == axis)
      return extent;
  return 0;
}

mlir::DictionaryAttr attach(mlir::Builder &builder,
                            mlir::DictionaryAttr value,
                            const StorageMapping &mapping) {
  value = riscv_internal::set(value, "base_encoding_family",
                              builder.getStringAttr(mapping.baseFamily));
  value = riscv_internal::set(value, "interleave_rows",
                              builder.getI64IntegerAttr(mapping.rows));
  value = riscv_internal::set(
      value, "base_record_storage_bits",
      builder.getI64IntegerAttr(mapping.recordStorageBits));
  value = riscv_internal::set(value, "physical_layout_identity",
                              builder.getStringAttr(mapping.identity));
  return value;
}

mlir::DictionaryAttr mappingAttribute(mlir::Builder &builder,
                                      const StorageMapping &mapping) {
  return riscv_internal::dictionary(
      builder,
      {{"packing", builder.getStringAttr("encoded-record-interleave")},
       {"source_rank", builder.getI64IntegerAttr(2)},
       {"pack_axis", builder.getI64IntegerAttr(mapping.rowAxis)},
       {"other_axis", builder.getI64IntegerAttr(mapping.recordAxis)},
       {"source_encoding_family", builder.getStringAttr(mapping.baseFamily)},
       {"interleave_rows", builder.getI64IntegerAttr(mapping.rows)},
       {"record_storage_bits",
        builder.getI64IntegerAttr(mapping.recordStorageBits)},
       {"logical_elements",
        builder.getI64IntegerAttr(mapping.logicalElements)},
       {"row_axis", builder.getI64IntegerAttr(mapping.rowAxis)},
       {"record_axis", builder.getI64IntegerAttr(mapping.recordAxis)},
       {"decision_owner", builder.getStringAttr("storage-value-use-chain")}});
}

bool propagatesStorageIdentity(llvm::StringRef name) {
  return name == "weft_kernel.slice" || name == "weft_kernel.admit" ||
         name == "weft_kernel.stage_handoff";
}

class PropagateRISCVStorageMappingsPass final
    : public mlir::PassWrapper<PropagateRISCVStorageMappingsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      PropagateRISCVStorageMappingsPass)

  llvm::StringRef getArgument() const final {
    return "weft-riscv-propagate-storage-mappings";
  }
  llvm::StringRef getDescription() const final {
    return "propagate encoded storage mappings through handoff and use-def chains";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::Builder builder(module.getContext());
    EncodingMappings encodings = collectEncodingMappings(module);
    for (riscv::ProblemOp problem : module.getOps<riscv::ProblemOp>()) {
      if (problem.getStage() == "invalid")
        continue;
      if (problem.getStage() != "conversions") {
        problem.emitError(
            "storage mapping propagation requires typed value-use conversions");
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

      llvm::SmallVector<mlir::DictionaryAttr> values;
      llvm::StringMap<size_t> valuePositions;
      llvm::StringMap<mlir::DictionaryAttr> valuesById;
      llvm::StringMap<llvm::SmallVector<std::string, 2>> valuesByHandoff;
      for (auto [position, attribute] : llvm::enumerate(problem.getValues())) {
        auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef id = *riscv_internal::string(value, "id");
        valuePositions[id] = position;
        valuesById[id] = value;
        values.push_back(value);
        llvm::StringRef handoff =
            riscv_internal::string(value, "handoff_class").value_or("");
        if (!handoff.empty())
          valuesByHandoff[handoff].push_back(id.str());
      }

      llvm::SmallVector<mlir::DictionaryAttr> operations;
      llvm::StringMap<size_t> operationPositions;
      llvm::StringMap<mlir::DictionaryAttr> producers;
      llvm::StringMap<llvm::SmallVector<mlir::DictionaryAttr, 2>> users;
      for (auto [position, attribute] :
           llvm::enumerate(problem.getOperations())) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        llvm::StringRef id = *riscv_internal::string(operation, "id");
        operationPositions[id] = position;
        operations.push_back(operation);
        if (auto results = operation.getAs<mlir::ArrayAttr>("results"))
          for (mlir::Attribute result : results)
            producers[mlir::cast<mlir::StringAttr>(result).getValue()] = operation;
        if (auto operands = operation.getAs<mlir::ArrayAttr>("operands"))
          for (mlir::Attribute operand : operands)
            users[mlir::cast<mlir::StringAttr>(operand).getValue()].push_back(
                operation);
      }

      llvm::StringMap<StorageMapping> mappings;
      auto setMapping = [&](llvm::StringRef id,
                            const StorageMapping &mapping) -> bool {
        auto existing = mappings.find(id);
        if (existing != mappings.end())
          return existing->second == mapping;
        mappings[id] = mapping;
        return true;
      };

      bool legal = true;
      std::string reason;
      const int64_t vlen =
          riscv_internal::integer(problem.getTarget(), "vlen_bits").value_or(0);

      for (const auto &entry : valuesById) {
        auto family = entry.getValue().getAs<mlir::StringAttr>("encoding_family");
        if (!family)
          continue;
        auto source = encodings.sourceByDerived.find(family.getValue());
        auto rows = encodings.rowsByDerived.find(family.getValue());
        if (source == encodings.sourceByDerived.end())
          continue;
        auto base = encodings.bases.find(source->second);
        if (rows == encodings.rowsByDerived.end() ||
            base == encodings.bases.end()) {
          legal = false;
          reason = "derived encoding has no concrete storage mapping";
          break;
        }
        setMapping(entry.getKey(),
                   {source->second, family.getValue().str() + ".rows" +
                                        std::to_string(rows->second) +
                                        ".byte-major.vlen" + std::to_string(vlen),
                    rows->second, base->second.storageBits,
                    base->second.elements, 0, 0});
      }

      if (legal) {
        for (mlir::DictionaryAttr materialize : operations) {
          if (riscv_internal::string(materialize, "name").value_or("") !=
              "weft_kernel.materialize")
            continue;
          auto materializeOperands =
              materialize.getAs<mlir::ArrayAttr>("operands");
          auto materializeResults =
              materialize.getAs<mlir::ArrayAttr>("results");
          if (!materializeOperands || materializeOperands.size() != 1 ||
              !materializeResults || materializeResults.size() != 1)
            continue;
          llvm::StringRef packId =
              mlir::cast<mlir::StringAttr>(materializeOperands[0]).getValue();
          auto pack = producers.find(packId);
          if (pack == producers.end() ||
              riscv_internal::string(pack->second, "name").value_or("") !=
                  "weft_kernel.pack")
            continue;
          auto packOperands = pack->second.getAs<mlir::ArrayAttr>("operands");
          auto packSource =
              pack->second.getAs<mlir::DictionaryAttr>("source_attributes");
          if (!packOperands || packOperands.size() != 1 || !packSource) {
            legal = false;
            reason = "encoded materialize has no explicit pack source";
            break;
          }
          llvm::StringRef sourceId =
              mlir::cast<mlir::StringAttr>(packOperands[0]).getValue();
          auto sourceValue = valuesById.find(sourceId);
          auto family = sourceValue == valuesById.end()
                            ? mlir::StringAttr()
                            : sourceValue->second.getAs<mlir::StringAttr>(
                                  "encoding_family");
          if (!family)
            continue;
          llvm::StringRef baseName = baseFamily(encodings, family.getValue());
          llvm::StringRef along =
              riscv_internal::string(packSource, "along").value_or("");
          const int64_t packAxis = axisForSymbol(kernel, along);
          auto sourceAxes = sourceValue == valuesById.end()
                                ? mlir::DenseI64ArrayAttr()
                                : sourceValue->second.getAs<
                                      mlir::DenseI64ArrayAttr>("axes");
          if (packAxis <= 0 || !sourceAxes || sourceAxes.size() != 2 ||
              !llvm::is_contained(sourceAxes.asArrayRef(), packAxis)) {
            legal = false;
            reason = "encoded pack source has no rank-two axis/storage mapping";
            break;
          }

          const int64_t recordAxis =
              sourceAxes.asArrayRef()[0] == packAxis
                  ? sourceAxes.asArrayRef()[1]
                  : sourceAxes.asArrayRef()[0];
          llvm::StringRef encodingKind =
              riscv_internal::string(sourceValue->second, "encoding_kind")
                  .value_or("");
          if (encodingKind == "dense") {
            mlir::DictionaryAttr mapping = riscv_internal::dictionary(
                builder,
                {{"source_rank", builder.getI64IntegerAttr(2)},
                 {"pack_axis", builder.getI64IntegerAttr(packAxis)},
                 {"other_axis", builder.getI64IntegerAttr(recordAxis)},
                 {"decision_owner",
                  builder.getStringAttr("storage-value-use-chain")}});
            size_t position = operationPositions.lookup(
                *riscv_internal::string(materialize, "id"));
            operations[position] = riscv_internal::set(
                materialize, "storage_mapping", mapping);
            continue;
          }

          auto base = encodings.bases.find(baseName);
          if (base == encodings.bases.end()) {
            legal = false;
            reason = "encoded pack source has no declared base storage layout";
            break;
          }

          llvm::StringRef resultId =
              mlir::cast<mlir::StringAttr>(materializeResults[0]).getValue();
          llvm::SmallVector<std::string> pending{resultId.str()};
          llvm::StringSet<> visited;
          int64_t rows = 0;
          while (!pending.empty()) {
            std::string id = std::move(pending.pop_back_val());
            if (!visited.insert(id).second)
              continue;
            if (auto value = valuesById.find(id); value != valuesById.end()) {
              rows = std::max(rows,
                              physicalWidthOnAxis(value->second, packAxis));
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
          if (rows <= 1) {
            legal = false;
            reason = "encoded local pack has no multi-record downstream use";
            break;
          }
          StorageMapping mapping{
              baseName.str(),
              "local-pack." + resultId.str() + ".rows" +
                  std::to_string(rows) + ".byte-major.vlen" +
                  std::to_string(vlen),
              rows,
              base->second.storageBits,
              base->second.elements,
              packAxis,
              recordAxis};
          setMapping(resultId, mapping);
          size_t position = operationPositions.lookup(
              *riscv_internal::string(materialize, "id"));
          operations[position] = riscv_internal::set(
              materialize, "storage_mapping", mappingAttribute(builder, mapping));
        }
      }

      bool changed = true;
      while (legal && changed) {
        changed = false;
        for (const auto &entry : valuesByHandoff) {
          std::optional<StorageMapping> mapping;
          for (const std::string &member : entry.getValue())
            if (auto found = mappings.find(member); found != mappings.end()) {
              if (mapping && !(*mapping == found->second)) {
                legal = false;
                reason = "one handoff class carries conflicting storage mappings";
                break;
              }
              mapping = found->second;
            }
          if (!legal || !mapping)
            continue;
          for (const std::string &member : entry.getValue())
            if (!mappings.contains(member)) {
              mappings[member] = *mapping;
              changed = true;
            }
        }
        for (mlir::DictionaryAttr operation : operations) {
          llvm::StringRef name =
              riscv_internal::string(operation, "name").value_or("");
          if (!propagatesStorageIdentity(name))
            continue;
          auto operands = operation.getAs<mlir::ArrayAttr>("operands");
          auto results = operation.getAs<mlir::ArrayAttr>("results");
          if (!operands || operands.empty() || !results)
            continue;
          llvm::StringRef source =
              mlir::cast<mlir::StringAttr>(operands[0]).getValue();
          auto mapping = mappings.find(source);
          if (mapping == mappings.end())
            continue;
          for (mlir::Attribute result : results) {
            llvm::StringRef id = mlir::cast<mlir::StringAttr>(result).getValue();
            if (!mappings.contains(id)) {
              mappings[id] = mapping->second;
              changed = true;
            }
          }
        }
      }

      if (!legal) {
        problem.setResourcesAttr(riscv_internal::dictionary(
            builder, {{"invalid_reason", builder.getStringAttr(reason)}}));
        problem.setStageAttr(builder.getStringAttr("invalid"));
        continue;
      }

      for (const auto &entry : mappings) {
        auto position = valuePositions.find(entry.getKey());
        if (position == valuePositions.end())
          continue;
        mlir::DictionaryAttr value =
            attach(builder, values[position->second], entry.getValue());
        values[position->second] = value;
        valuesById[entry.getKey()] = value;
      }
      llvm::SmallVector<mlir::Attribute> valueAttributes(values.begin(),
                                                         values.end());
      llvm::SmallVector<mlir::Attribute> operationAttributes(operations.begin(),
                                                             operations.end());
      problem.setValuesAttr(builder.getArrayAttr(valueAttributes));
      problem.setOperationsAttr(builder.getArrayAttr(operationAttributes));
      problem.setStageAttr(builder.getStringAttr("storage-mappings"));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
weft::createPropagateRISCVStorageMappingsPass() {
  return std::make_unique<PropagateRISCVStorageMappingsPass>();
}
