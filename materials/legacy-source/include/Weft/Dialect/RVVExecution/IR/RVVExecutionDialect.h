#ifndef WEFT_DIALECT_RVVEXECUTION_IR_RVVEXECUTIONDIALECT_H
#define WEFT_DIALECT_RVVEXECUTION_IR_RVVEXECUTIONDIALECT_H

#include "Weft/Dialect/Execution/IR/SelectedOwnerOpInterface.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#include "Weft/Dialect/RVVExecution/IR/RVVExecutionOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/RVVExecution/IR/RVVExecutionOps.h.inc"

#endif // WEFT_DIALECT_RVVEXECUTION_IR_RVVEXECUTIONDIALECT_H
