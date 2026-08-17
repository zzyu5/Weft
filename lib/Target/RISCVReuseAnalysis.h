#ifndef WEFT_LIB_TARGET_RISCVREUSEANALYSIS_H
#define WEFT_LIB_TARGET_RISCVREUSEANALYSIS_H

#include "RISCVKernelFacts.h"
#include "RISCVPhysicalPlanning.h"

#include "llvm/ADT/ArrayRef.h"

namespace weft::riscv_internal {

LocalScheduleDependenceFacts analyzeLocalScheduleDependences(
    const KernelPhysicalFacts &facts,
    llvm::ArrayRef<mlir::Value> primitiveOperands, mlir::Value reductionAxis,
    mlir::Value accumulator, mlir::Value result);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVREUSEANALYSIS_H
