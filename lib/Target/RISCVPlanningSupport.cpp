#include "RISCVPlanningSupport.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/raw_ostream.h"

using namespace weft;

mlir::ArrayAttr riscv_internal::strings(
    mlir::Builder &builder, llvm::ArrayRef<std::string> values) {
  llvm::SmallVector<mlir::Attribute> attributes;
  attributes.reserve(values.size());
  for (const std::string &value : values)
    attributes.push_back(builder.getStringAttr(value));
  return builder.getArrayAttr(attributes);
}

mlir::DenseI64ArrayAttr
riscv_internal::integers(mlir::Builder &builder,
                         llvm::ArrayRef<int64_t> values) {
  return builder.getDenseI64ArrayAttr(values);
}

mlir::DictionaryAttr riscv_internal::dictionary(
    mlir::Builder &builder,
    llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Attribute>> values) {
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  attributes.reserve(values.size());
  for (auto [name, value] : values)
    attributes.push_back(builder.getNamedAttr(name, value));
  return builder.getDictionaryAttr(attributes);
}

mlir::DictionaryAttr riscv_internal::set(mlir::DictionaryAttr source,
                                         llvm::StringRef name,
                                         mlir::Attribute value) {
  mlir::NamedAttrList attributes(source.getValue());
  attributes.set(name, value);
  return attributes.getDictionary(source.getContext());
}

mlir::DictionaryAttr
riscv_internal::targetFacts(mlir::Builder &builder,
                            const RISCVTargetProfile &target) {
  llvm::SmallVector<int64_t> sew(target.supportedSEW.begin(),
                                 target.supportedSEW.end());
  llvm::SmallVector<int64_t> lmul(target.legalLMULEighths.begin(),
                                 target.legalLMULEighths.end());
  llvm::SmallVector<mlir::Attribute> fragments;
  for (const RISCVFragmentCapability &fragment : target.fragmentCapabilities) {
    llvm::StringRef instruction;
    switch (fragment.instruction) {
    case RISCVFragmentInstruction::SpacemitIME1I4I8MMA:
      instruction = "spacemit-ime1-i4i8-mma";
      break;
    }
    std::string identity =
        instruction.str() + ".m" + std::to_string(fragment.mFactor) + "n" +
        std::to_string(fragment.nFactor) + "k" +
        std::to_string(fragment.kFactor);
    fragments.push_back(dictionary(
        builder,
        {{"identity", builder.getStringAttr(identity)},
         {"instruction", builder.getStringAttr(instruction)},
         {"lhs_sew", builder.getI64IntegerAttr(fragment.lhsElementBits)},
         {"rhs_sew", builder.getI64IntegerAttr(fragment.rhsElementBits)},
         {"accumulator_sew",
          builder.getI64IntegerAttr(fragment.accumulatorElementBits)},
         {"m_factor", builder.getI64IntegerAttr(fragment.mFactor)},
         {"n_factor", builder.getI64IntegerAttr(fragment.nFactor)},
         {"k_factor", builder.getI64IntegerAttr(fragment.kFactor)},
         {"fixed_resource_groups",
          builder.getI64IntegerAttr(fragment.fixedResourceGroups)}}));
  }
  return dictionary(
      builder,
      {{"triple", builder.getStringAttr(target.triple)},
       {"march", builder.getStringAttr(target.march)},
       {"abi", builder.getStringAttr(target.abi)},
       {"xlen", builder.getI64IntegerAttr(target.xlen)},
       {"little_endian", builder.getBoolAttr(target.littleEndian)},
       {"has_rvv", builder.getBoolAttr(target.hasRVV)},
       {"has_indexed_memory", builder.getBoolAttr(target.hasIndexedMemory)},
       {"has_segment_memory", builder.getBoolAttr(target.hasSegmentMemory)},
       {"has_widening_integer", builder.getBoolAttr(target.hasWideningInteger)},
       {"has_vector_f16", builder.getBoolAttr(target.hasVectorF16)},
       {"vlen_bits", builder.getI64IntegerAttr(target.vlenBits)},
       {"vector_registers",
        builder.getI64IntegerAttr(target.vectorRegisters)},
       {"max_private_stack_bytes",
        builder.getI64IntegerAttr(target.maxPrivateStackBytes)},
       {"supported_sew", integers(builder, sew)},
       {"legal_lmul_eighths", integers(builder, lmul)},
       {"matrix_fragment_count",
        builder.getI64IntegerAttr(target.fragmentCapabilities.size())},
       {"matrix_fragments", builder.getArrayAttr(fragments)}});
}

std::string riscv_internal::printType(mlir::Type type) {
  std::string result;
  llvm::raw_string_ostream output(result);
  type.print(output);
  return result;
}

std::string riscv_internal::printAttribute(mlir::Attribute attribute) {
  std::string result;
  llvm::raw_string_ostream output(result);
  attribute.print(output);
  return result;
}

mlir::Type riscv_internal::logicalElement(mlir::Type type) {
  if (auto value = mlir::dyn_cast<kernel::ValueType>(type))
    return value.getElementType();
  if (auto view = mlir::dyn_cast<kernel::ViewType>(type))
    return view.getEncoding();
  if (auto slice = mlir::dyn_cast<kernel::SliceType>(type))
    return slice.getEncoding();
  return type;
}

llvm::ArrayRef<int64_t> riscv_internal::logicalShape(mlir::Type type) {
  if (auto value = mlir::dyn_cast<kernel::ValueType>(type))
    return value.getShape().asArrayRef();
  if (auto view = mlir::dyn_cast<kernel::ViewType>(type))
    return view.getShape().asArrayRef();
  if (auto slice = mlir::dyn_cast<kernel::SliceType>(type))
    return slice.getShape().asArrayRef();
  return {};
}

llvm::ArrayRef<int64_t> riscv_internal::logicalAxes(mlir::Type type) {
  if (auto value = mlir::dyn_cast<kernel::ValueType>(type))
    return value.getAxisIds().asArrayRef();
  if (auto view = mlir::dyn_cast<kernel::ViewType>(type))
    return view.getAxisIds().asArrayRef();
  if (auto slice = mlir::dyn_cast<kernel::SliceType>(type))
    return slice.getAxisIds().asArrayRef();
  return {};
}

unsigned riscv_internal::logicalBitWidth(mlir::Type type) {
  type = logicalElement(type);
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  if (type.isIndex())
    return 64;
  return 0;
}

std::optional<int64_t>
riscv_internal::integer(mlir::DictionaryAttr dictionary,
                        llvm::StringRef name) {
  if (!dictionary)
    return std::nullopt;
  if (auto value = dictionary.getAs<mlir::IntegerAttr>(name))
    return value.getInt();
  return std::nullopt;
}

std::optional<llvm::StringRef>
riscv_internal::string(mlir::DictionaryAttr dictionary,
                       llvm::StringRef name) {
  if (!dictionary)
    return std::nullopt;
  if (auto value = dictionary.getAs<mlir::StringAttr>(name))
    return value.getValue();
  return std::nullopt;
}

mlir::ArrayAttr riscv_internal::array(mlir::DictionaryAttr dictionary,
                                      llvm::StringRef name) {
  if (!dictionary)
    return {};
  return dictionary.getAs<mlir::ArrayAttr>(name);
}

weft::kernel::KernelOp
riscv_internal::findKernel(mlir::ModuleOp module,
                           mlir::FlatSymbolRefAttr symbol) {
  return mlir::SymbolTable::lookupNearestSymbolFrom<kernel::KernelOp>(module,
                                                                     symbol);
}
