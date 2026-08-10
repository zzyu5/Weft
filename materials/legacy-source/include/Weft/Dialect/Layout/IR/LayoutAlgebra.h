#ifndef WEFT_DIALECT_LAYOUT_IR_LAYOUTALGEBRA_H
#define WEFT_DIALECT_LAYOUT_IR_LAYOUTALGEBRA_H

#include "Weft/Dialect/Layout/IR/LayoutDialect.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace weft::layout {

BlockedLayoutAttr getBlockedLayout(mlir::MLIRContext *context, int64_t rank,
                                   llvm::ArrayRef<int64_t> order,
                                   llvm::ArrayRef<int64_t> vectorAxes);

BlockedLayoutAttr getIdentityBlockedLayout(mlir::MLIRContext *context,
                                           int64_t rank,
                                           llvm::ArrayRef<int64_t> vectorAxes);

BlockedLayoutAttr
getVectorFastestBlockedLayout(mlir::MLIRContext *context, int64_t rank,
                              llvm::ArrayRef<int64_t> vectorAxes);

mlir::FailureOr<BlockedLayoutAttr>
unifyBlockedLayouts(BlockedLayoutAttr lhs, BlockedLayoutAttr rhs);

mlir::FailureOr<BlockedLayoutAttr>
projectBlockedLayout(BlockedLayoutAttr layout,
                     llvm::ArrayRef<int64_t> removedAxes);

mlir::FailureOr<BlockedLayoutAttr>
insertBlockedAxis(BlockedLayoutAttr layout, int64_t logicalAxis,
                  int64_t traversalPosition, bool vectorized);

} // namespace weft::layout

#endif // WEFT_DIALECT_LAYOUT_IR_LAYOUTALGEBRA_H
