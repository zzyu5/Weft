#ifndef WEFT_LIB_TARGET_RISCVCABI_H
#define WEFT_LIB_TARGET_RISCVCABI_H

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace weft::riscv_internal {

enum class CPointerSpelling {
  LocalValue,
  FunctionDefinition,
  PublicDeclaration,
};

std::string cABIIdentifier(llvm::StringRef input);
std::string cABIScalarType(mlir::Type type);
std::string cABIPointerType(
    weft::kernel::PtrType pointer,
    CPointerSpelling spelling = CPointerSpelling::LocalValue);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVCABI_H
