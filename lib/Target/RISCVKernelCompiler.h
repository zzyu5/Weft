#ifndef WEFT_LIB_TARGET_RISCVKERNELCOMPILER_H
#define WEFT_LIB_TARGET_RISCVKERNELCOMPILER_H

#include "RISCVIntrinsicC.h"
#include "Weft/Target/RISCVLowering.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class raw_ostream;
}

namespace weft::riscv_internal {

mlir::LogicalResult
compileRISCVKernelsToIntrinsicC(mlir::ModuleOp module,
                                const RISCVLoweringOptions &options,
                                llvm::raw_ostream &output,
                                SelectedIntrinsicCLeaves &selectedLeaves);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVKERNELCOMPILER_H
