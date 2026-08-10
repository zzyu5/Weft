#ifndef WEFT_DIALECT_EXTENSION_IR_EXTENSIONDIALECT_H
#define WEFT_DIALECT_EXTENSION_IR_EXTENSIONDIALECT_H

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Weft/Dialect/Extension/IR/ExtensionOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Extension/IR/ExtensionOps.h.inc"

#endif // WEFT_DIALECT_EXTENSION_IR_EXTENSIONDIALECT_H
