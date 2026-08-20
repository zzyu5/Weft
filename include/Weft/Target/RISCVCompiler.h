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

/// Build and solve the transient RISC-V representation problem. The input is
/// canonical Kernel IR. Physical problem/assignment operations exist only for
/// this invocation and are not accepted as source authority.
mlir::FailureOr<RISCVPlanningResult>
planRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options);

} // namespace weft

#endif // WEFT_TARGET_RISCVCOMPILER_H
