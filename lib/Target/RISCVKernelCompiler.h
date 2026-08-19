#ifndef WEFT_LIB_TARGET_RISCVKERNELCOMPILER_H
#define WEFT_LIB_TARGET_RISCVKERNELCOMPILER_H

#include "RISCVIntrinsicC.h"
#include "Weft/Target/RISCVCompiler.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class raw_ostream;
}

namespace weft::riscv_internal {

mlir::LogicalResult
compileRISCVKernelsToIntrinsicC(mlir::ModuleOp module,
                                const RISCVCompilerOptions &options,
                                mlir::AnalysisManager &analysisManager,
                                llvm::raw_ostream &output,
                                SelectedLocalImplementations &selectedImplementations);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVKERNELCOMPILER_H
