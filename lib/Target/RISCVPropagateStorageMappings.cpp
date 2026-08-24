#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"

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
  llvm::StringMap<std::string> sourceByInstance;
  llvm::StringMap<int64_t> rowsByInstance;
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
    auto yield = mlir::cast<kernel::DeriveYieldOp>(
        derive.getBody().front().getTerminator());
    auto encoding = mlir::cast<kernel::EncodingType>(
        yield.getValue().getType().getEncoding());
    llvm::StringRef identity = encoding.getLayoutIdentity();
    result.sourceByInstance[identity] = derive.getSourceFamily().str();
    int64_t rows = derive.getParameterValues().size() == 1
                       ? derive.getParameterValues()[0]
                       : 0;
    if (rows > 0)
      result.rowsByInstance[identity] = rows;
  }
  return result;
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

bool propagatesStorageIdentity(llvm::StringRef name) {
  return name == "weft_kernel.slice" || name == "weft_kernel.admit" ||
         name == "weft_kernel.materialize";
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
      for (mlir::Attribute attribute : problem.getOperations()) {
        auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
        operations.push_back(operation);
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
        auto identity = entry.getValue().getAs<mlir::StringAttr>("layout_identity");
        if (!family || !identity)
          continue;
        auto source = encodings.sourceByInstance.find(identity.getValue());
        auto rows = encodings.rowsByInstance.find(identity.getValue());
        if (source == encodings.sourceByInstance.end())
          continue;
        auto base = encodings.bases.find(source->second);
        if (rows == encodings.rowsByInstance.end() ||
            base == encodings.bases.end()) {
          legal = false;
          reason = "derived encoding has no concrete storage mapping";
          break;
        }
        setMapping(entry.getKey(),
                   {source->second, identity.getValue().str() + ".byte-major.vlen" +
                                        std::to_string(vlen),
                    rows->second, base->second.storageBits,
                    base->second.elements, 0, 0});
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
