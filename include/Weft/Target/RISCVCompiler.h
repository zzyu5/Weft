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
};

struct RISCVPlanningResult {
  std::string assignment;
};

struct RISCVCompilationResult {
  std::string assignment;
  std::string intrinsicC;
};

/// Run transient forward RISC-V physicalization over canonical Kernel IR.
/// Candidate/assignment operations exist only for this invocation.
mlir::FailureOr<RISCVPlanningResult>
planRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options);

/// Select one resource-legal forward candidate and mechanically emit its
/// intrinsic-C program. AssignmentOp is the sole source of target choices.
mlir::FailureOr<RISCVCompilationResult>
compileRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options);

} // namespace weft

#endif // WEFT_TARGET_RISCVCOMPILER_H
