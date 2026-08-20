#ifndef WEFT_DIALECT_RISCV_IR_RISCVPLANNINGDIALECT_H
#define WEFT_DIALECT_RISCV_IR_RISCVPLANNINGDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Weft/Dialect/RISCV/IR/RISCVOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/RISCV/IR/RISCVOps.h.inc"

#endif // WEFT_DIALECT_RISCV_IR_RISCVPLANNINGDIALECT_H
