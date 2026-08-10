#ifndef WEFT_TARGET_SOURCEEMITTER_H
#define WEFT_TARGET_SOURCEEMITTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class raw_ostream;
}

namespace weft {

/// Emit one C++ translation unit from canonical kernels and their selected
/// execution plans. The emitter consumes no algorithm facts outside the
/// canonical kernel and no physical facts outside the selected plan.
mlir::LogicalResult emitSelectedSource(mlir::ModuleOp module,
                                       llvm::raw_ostream &output);

} // namespace weft

#endif // WEFT_TARGET_SOURCEEMITTER_H
