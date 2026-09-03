#ifndef WEFT_TARGET_RISCVCOMPILER_H
#define WEFT_TARGET_RISCVCOMPILER_H

#include "Weft/Target/RISCVTargetProfile.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <string>

namespace weft {

struct RISCVCompilerOptions {
  RISCVTargetProfile target;
  llvm::StringMap<int64_t> metaBindings;
  int64_t lmulEighths = 8;
  int64_t unroll = 1;
  int64_t pipelineDepth = 1;
  int64_t scalarLoadPrime = 0;
};

struct RISCVPhysicalizationResult {
  std::string riscvIR;
};

struct RISCVCompilationResult {
  std::string riscvIR;
  std::string intrinsicC;
};

/// Convert one instantiated Canonical Kernel IR candidate into a complete,
/// target-aware RISC-V Physical IR program.
mlir::FailureOr<RISCVPhysicalizationResult>
physicalizeRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options);

/// Translate a verified RISC-V Physical IR program mechanically to intrinsic C.
mlir::FailureOr<RISCVCompilationResult>
compileRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options);

/// Verify and mechanically translate an already-physicalized RISC-V program.
/// This is the terminal boundary used after independent Physical IR rewrites.
mlir::FailureOr<RISCVCompilationResult>
translateRISCVModule(mlir::ModuleOp module);

} // namespace weft

#endif // WEFT_TARGET_RISCVCOMPILER_H
