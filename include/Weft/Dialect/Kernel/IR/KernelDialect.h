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

#endif // WEFT_DIALECT_KERNEL_IR_KERNELDIALECT_H
