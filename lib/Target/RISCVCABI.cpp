#include "RISCVCABI.h"

#include "mlir/IR/BuiltinTypes.h"

#include <cctype>

namespace weft::riscv_internal {

std::string cABIIdentifier(llvm::StringRef input) {
  std::string result;
  result.reserve(input.size() + 1);
  for (char character : input) {
    unsigned char byte = static_cast<unsigned char>(character);
    result.push_back(std::isalnum(byte) || character == '_' ? character : '_');
  }
  if (result.empty() ||
      std::isdigit(static_cast<unsigned char>(result.front())))
    result.insert(result.begin(), '_');
  return result;
}

std::string cABIScalarType(mlir::Type type) {
  if (type.isIndex())
    return "size_t";
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (integer.getWidth() == 1)
      return "bool";
    if (integer.getWidth() != 8 && integer.getWidth() != 16 &&
        integer.getWidth() != 32 && integer.getWidth() != 64)
      return {};
    return std::string(integer.isUnsigned() ? "uint" : "int") +
           std::to_string(integer.getWidth()) + "_t";
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (floating.getWidth() == 16)
      return "_Float16";
    if (floating.getWidth() == 32)
      return "float";
    if (floating.getWidth() == 64)
      return "double";
  }
  return {};
}

std::string cABIPointerType(weft::kernel::PtrType pointer,
                            CPointerSpelling spelling) {
  std::string element = cABIScalarType(pointer.getElementType());
  if (element.empty())
    return {};
  std::string result =
      (pointer.getAccess() == "read" ? "const " : "") + element + " *";
  if (!pointer.getNoAlias() && !pointer.getRestrictLike())
    return result;
  if (spelling == CPointerSpelling::FunctionDefinition)
    result += " restrict";
  else if (spelling == CPointerSpelling::PublicDeclaration)
    result += " WEFT_GENERATED_RESTRICT";
  return result;
}

} // namespace weft::riscv_internal
