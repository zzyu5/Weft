#ifndef WEFT_DIALECT_IMEEXECUTION_IR_IMEEXECUTIONDIALECT_H
#define WEFT_DIALECT_IMEEXECUTION_IR_IMEEXECUTIONDIALECT_H

#include "Weft/Dialect/Execution/IR/SelectedOwnerOpInterface.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringRef.h"

namespace weft::ime_execution {

/// Return whether `target` contains the exact multi-letter RISC-V extension
/// token `xsmtvdotii`, optionally followed by a numeric major-p-minor version.
bool hasXsmtvdotiiExtension(llvm::StringRef target);

} // namespace weft::ime_execution

#include "Weft/Dialect/IMEExecution/IR/IMEExecutionOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/IMEExecution/IR/IMEExecutionOps.h.inc"

#endif // WEFT_DIALECT_IMEEXECUTION_IR_IMEEXECUTIONDIALECT_H
