#ifndef WEFT_DIALECT_KERNEL_IR_KERNELDIALECT_H
#define WEFT_DIALECT_KERNEL_IR_KERNELDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Weft/Dialect/Kernel/IR/KernelOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Weft/Dialect/Kernel/IR/KernelTypes.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Kernel/IR/KernelOps.h.inc"

namespace weft::kernel {

enum class LogicalShapeKind { Scalar, Block, Region };

/// Shared canonical value queries used by the core dialect, linked extension
/// dialects, and target realizers. Extension primitives are not a second value
/// system: they consume the same scalar/block/region values and validity.
mlir::Type unwrapLogicalValidity(mlir::Type type);
mlir::Type logicalElementType(mlir::Type type);
llvm::ArrayRef<int64_t> logicalShape(mlir::Type type);
llvm::ArrayRef<int64_t> logicalAxisIds(mlir::Type type);
LogicalShapeKind logicalShapeKind(mlir::Type type);
bool isLogicalValue(mlir::Type type);
bool hasLogicalValidity(mlir::Type type);
bool isLogicalPredicate(mlir::Type type);

/// Recompute logical extent identity from canonical SSA producers. This is an
/// analysis query, not a persisted shape schema.
bool haveSameLogicalExtent(mlir::Value lhs, int64_t lhsAxis, mlir::Value rhs,
                           int64_t rhsAxis);

} // namespace weft::kernel

#endif // WEFT_DIALECT_KERNEL_IR_KERNELDIALECT_H
