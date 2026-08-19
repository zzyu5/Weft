#ifndef WEFT_LIB_TARGET_RISCVHEADER_H
#define WEFT_LIB_TARGET_RISCVHEADER_H

#include "Weft/Target/RISCVCompiler.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class raw_ostream;
}

namespace weft::riscv_internal {

mlir::LogicalResult emitRISCVArtifactHeader(
    mlir::ModuleOp module, const RISCVCompilerOptions &options,
    llvm::raw_ostream &output);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVHEADER_H
