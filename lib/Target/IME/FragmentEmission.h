#ifndef WEFT_TARGET_IME_FRAGMENT_EMISSION_H
#define WEFT_TARGET_IME_FRAGMENT_EMISSION_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace weft::riscv_emission {
struct FragmentEmission {
  llvm::function_ref<std::string(llvm::StringRef)> fresh;
  llvm::function_ref<void(llvm::StringRef)> line;
};

mlir::FailureOr<std::string> emitFragmentMMA(
    riscv::IMEFragmentMMAOp operation, llvm::StringRef lhs, llvm::StringRef rhs,
    const FragmentEmission &emission);
} // namespace weft::riscv_emission

#endif
