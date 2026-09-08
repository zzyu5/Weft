#include "Emitter.h"

namespace weft::riscv_emission {

std::string identifier(llvm::StringRef source) {
  std::string result;
  result.reserve(source.size() + 1);
  for (char character : source)
    result.push_back(std::isalnum(static_cast<unsigned char>(character))
                         ? character
                         : '_');
  if (result.empty() || std::isdigit(static_cast<unsigned char>(result[0])))
    result.insert(result.begin(), '_');
  return result;
}

std::string arrayString(mlir::ArrayAttr values, llvm::StringRef separator) {
  std::string result;
  for (auto [index, attribute] : llvm::enumerate(values)) {
    if (index)
      result += separator;
    result += mlir::cast<mlir::StringAttr>(attribute).getValue().str();
  }
  return result;
}

std::optional<StorageFragment>
singleStorageFragment(const EncodingField &field, llvm::StringRef logicalIndex,
                      unsigned logicalWidth) {
  if (!field.access)
    return std::nullopt;
  llvm::StringRef kind = field.access.getMapping();
  if (kind == "natural") {
    std::string bit = "(" + std::to_string(field.bitOffset) + " + (" +
                      logicalIndex.str() + ") * " +
                      std::to_string(logicalWidth) + ")";
    return StorageFragment{"(" + bit + " / 8)", "(" + bit + " % 8)",
                           logicalWidth};
  }
  if (kind != "grouped_layered")
    return std::nullopt;
  int64_t group = field.access.getGroupSize();
  int64_t layer = field.access.getLayerSize();
  llvm::StringRef order = field.access.getOrder();
  int64_t layers = group / layer;
  std::string within = "((" + logicalIndex.str() + ") % " +
                       std::to_string(group) + ")";
  std::string layerIndex = "(" + within + " / " + std::to_string(layer) + ")";
  if (order == "hi_first")
    layerIndex = "(" + std::to_string(layers - 1) + " - " + layerIndex + ")";
  std::string byte = "(" + std::to_string(field.bitOffset / 8) + " + ((" +
                     logicalIndex.str() + ") / " + std::to_string(group) + ") * " +
                     std::to_string(layer) + " + " + within + " % " +
                     std::to_string(layer) + ")";
  std::string shift = "(" + layerIndex + " * " +
                      std::to_string(logicalWidth) + ")";
  return StorageFragment{byte, shift, logicalWidth};
}

std::optional<std::string>
scalarEncodedFieldSource(const EncodingField &field,
                         llvm::StringRef elementType,
                         llvm::StringRef record,
                         llvm::StringRef logicalIndex) {
  if (field.access.getMapping() == "natural")
    return "((const " + elementType.str() + " *)((const uint8_t *)(" +
           record.str() + ") + " + std::to_string(field.bitOffset / 8) +
           "))[" + logicalIndex.str() + "]";
  if (field.access.getMapping() != "joined")
    return std::nullopt;

  auto integer = mlir::dyn_cast<mlir::IntegerType>(field.type);
  const int64_t group = field.access.getGroupSize();
  const int64_t fields = field.access.getJoinFields();
  const int64_t lowBits = field.access.getJoinLowBits();
  const int64_t role = field.access.getJoinRole();
  const unsigned logicalWidth = integer ? integer.getWidth() : 0;
  if (!integer || !integer.isUnsigned() || group <= 0 || fields <= 1 ||
      lowBits <= 0 || role < 0 || role >= fields ||
      logicalWidth <= static_cast<unsigned>(lowBits) || logicalWidth > 8 ||
      (field.access.getOrder() != "lo_first" &&
       field.access.getOrder() != "hi_first"))
    return std::nullopt;
  const int64_t physicalRole =
      field.access.getOrder() == "lo_first" ? role : fields - 1 - role;
  const uint64_t logicalMask = (uint64_t(1) << logicalWidth) - 1;
  const uint64_t lowMask = (uint64_t(1) << lowBits) - 1;
  const uint64_t highMask =
      (uint64_t(1) << (logicalWidth - lowBits)) - 1;
  const std::string index = "(" + logicalIndex.str() + ")";
  const std::string tail = "(" + index + " - " + std::to_string(group) + ")";
  const std::string headByte =
      "((const uint8_t *)(" + record.str() + "))[" +
      std::to_string(field.bitOffset / 8 + role * group) + " + " + index +
      "]";
  const std::string lowByte =
      "((const uint8_t *)(" + record.str() + "))[" +
      std::to_string(field.bitOffset / 8 + fields * group) + " + " + tail +
      "]";
  const std::string highByte =
      "((const uint8_t *)(" + record.str() + "))[" +
      std::to_string(field.bitOffset / 8 + role * group) + " + " + tail +
      "]";
  const std::string head = "((uint8_t)(" + headByte + " & " +
                           std::to_string(logicalMask) + "))";
  const std::string low =
      "((uint8_t)((" + lowByte + " >> " +
      std::to_string(physicalRole * lowBits) + ") & " +
      std::to_string(lowMask) + "))";
  const std::string high =
      "((uint8_t)((" + highByte + " >> " +
      std::to_string(logicalWidth) + ") & " +
      std::to_string(highMask) + "))";
  return "(" + index + " < " + std::to_string(group) + " ? " + head +
         " : (" + low + " | (" + high + " << " +
         std::to_string(lowBits) + ")))";
}

std::string lmulSpelling(int64_t eighths) {
  switch (eighths) {
  case 1:
    return "mf8";
  case 2:
    return "mf4";
  case 4:
    return "mf2";
  case 8:
    return "m1";
  case 16:
    return "m2";
  case 32:
    return "m4";
  case 64:
    return "m8";
  default:
    return {};
  }
}

std::string vectorSuffixFor(riscv::ValueType value) {
  char category = 'i';
  if (mlir::isa<mlir::FloatType>(value.getElementType()))
    category = 'f';
  else if (auto integer =
               mlir::dyn_cast<mlir::IntegerType>(value.getElementType()))
    category = integer.isUnsigned() ? 'u' : 'i';
  return std::string(1, category) +
         std::to_string(value.getLayout().getSew()) +
         lmulSpelling(value.getLayout().getLmulEighths());
}

std::string vectorTypeFor(riscv::ValueType value) {
  std::string suffix = vectorSuffixFor(value);
  const std::string prefix = suffix.front() == 'f'
                                 ? "vfloat"
                                 : suffix.front() == 'u' ? "vuint" : "vint";
  return prefix + suffix.substr(1) + "_t";
}

int64_t physicalLanesFor(riscv::ValueType value) {
  int64_t lanes = 1;
  for (int64_t factor : value.getLayout().getLaneFactors().asArrayRef())
    lanes *= factor;
  return std::max<int64_t>(1, lanes);
}

int64_t product(mlir::DenseI64ArrayAttr values) {
  return riscv_internal::staticProduct(values.asArrayRef()).value_or(0);
}

riscv::AccessAttr accessOf(mlir::Operation *operation) {
  return operation ? operation->getAttrOfType<riscv::AccessAttr>("access")
                   : riscv::AccessAttr();
}

riscv::LeafAttr leafOf(mlir::Operation *operation) {
  return operation ? operation->getAttrOfType<riscv::LeafAttr>("leaf")
                   : riscv::LeafAttr();
}

} // namespace weft::riscv_emission
