#ifndef WEFT_DIALECT_EXECUTION_IR_EXECUTIONDIALECT_H
#define WEFT_DIALECT_EXECUTION_IR_EXECUTIONDIALECT_H

#include "Weft/Dialect/Layout/IR/LayoutDialect.h"
#include "Weft/Dialect/Execution/IR/SelectedOwnerOpInterface.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#include "Weft/Dialect/Execution/IR/ExecutionOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Execution/IR/ExecutionOps.h.inc"

#endif // WEFT_DIALECT_EXECUTION_IR_EXECUTIONDIALECT_H
