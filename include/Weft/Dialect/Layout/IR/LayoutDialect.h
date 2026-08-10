#ifndef WEFT_DIALECT_LAYOUT_IR_LAYOUTDIALECT_H
#define WEFT_DIALECT_LAYOUT_IR_LAYOUTDIALECT_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"

#include "Weft/Dialect/Layout/IR/LayoutOpsDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Weft/Dialect/Layout/IR/LayoutAttrs.h.inc"

#endif // WEFT_DIALECT_LAYOUT_IR_LAYOUTDIALECT_H
