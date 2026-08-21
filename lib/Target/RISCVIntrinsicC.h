#ifndef WEFT_LIB_TARGET_RISCVINTRINSICC_H
#define WEFT_LIB_TARGET_RISCVINTRINSICC_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace weft {

mlir::LogicalResult emitSelectedRISCVIntrinsicC(mlir::ModuleOp module,
                                                std::string &output);

} // namespace weft

#endif // WEFT_LIB_TARGET_RISCVINTRINSICC_H
