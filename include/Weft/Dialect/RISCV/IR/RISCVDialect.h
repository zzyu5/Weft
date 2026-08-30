#ifndef WEFT_DIALECT_RISCV_IR_RISCVDIALECT_H
#define WEFT_DIALECT_RISCV_IR_RISCVDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

#include "Weft/Dialect/RISCV/IR/RISCVOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Weft/Dialect/RISCV/IR/RISCVAttrs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Weft/Dialect/RISCV/IR/RISCVTypes.h.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/RISCV/IR/RISCVOps.h.inc"

namespace weft::riscv {

/// Returns whether an RVV layout is executable under the complete target
/// register contract, including the ELEN-dependent fractional-LMUL bound.
bool supportsRVVLayout(TargetAttr target, LayoutAttr layout);

/// Projects every result part onto the packed operand address identity used by
/// grouped MAC. Equal entries denote one typed packed supply shared by those
/// result consumers.
std::optional<llvm::SmallVector<int64_t>>
groupedMacSupplyProjection(ValueType packed, ValueType result);

} // namespace weft::riscv

#endif // WEFT_DIALECT_RISCV_IR_RISCVDIALECT_H
