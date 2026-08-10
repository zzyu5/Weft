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

/// Recompute logical extent identity from canonical SSA producers. This is an
/// analysis query, not a persisted shape schema.
bool haveSameLogicalExtent(mlir::Value lhs, int64_t lhsAxis, mlir::Value rhs,
                           int64_t rhsAxis);

} // namespace weft::kernel

#endif // WEFT_DIALECT_KERNEL_IR_KERNELDIALECT_H
