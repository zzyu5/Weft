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

#include <optional>

namespace weft::kernel {

/// A logical block-axis extent derived from canonical SSA.  Exactly one of
/// `constant` and `dynamic` is populated.  This is a recomputable fact, not a
/// second shape representation.
struct LogicalExtent {
  std::optional<int64_t> constant;
  mlir::Value dynamic;
};

std::optional<LogicalExtent> deriveLogicalExtent(mlir::Value value,
                                                 int64_t axis);
bool haveSameLogicalExtent(mlir::Value lhs, int64_t lhsAxis, mlir::Value rhs,
                           int64_t rhsAxis);

/// True when a scalar/block index value is structurally derived only from
/// canonical scalar indices, arange, expand_dims, and index pointwise
/// arithmetic in already-verified Kernel IR. This is a recomputable provenance
/// fact, not persisted IR.
bool isIndexCoordinateProvenance(mlir::Value value);

} // namespace weft::kernel

#endif // WEFT_DIALECT_KERNEL_IR_KERNELDIALECT_H
