#ifndef WEFT_DIALECT_EXECUTION_IR_EXECUTIONDIALECT_H
#define WEFT_DIALECT_EXECUTION_IR_EXECUTIONDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#include "Weft/Dialect/Execution/IR/ExecutionOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Execution/IR/ExecutionOps.h.inc"

namespace weft::execution {

inline constexpr llvm::StringLiteral kCanonicalAnchorAttr =
    "weft_kernel.anchor";

} // namespace weft::execution

#endif // WEFT_DIALECT_EXECUTION_IR_EXECUTIONDIALECT_H
