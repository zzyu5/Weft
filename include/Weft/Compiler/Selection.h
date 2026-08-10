#ifndef WEFT_COMPILER_SELECTION_H
#define WEFT_COMPILER_SELECTION_H

#include "Weft/Target/RISCVTargetProfile.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

namespace weft {

struct SelectionOptions {
  RISCVTargetProfile target;
  llvm::StringMap<int64_t> metaBindings;
};

mlir::LogicalResult selectExecution(mlir::ModuleOp module,
                                    const SelectionOptions &options);

} // namespace weft

#endif // WEFT_COMPILER_SELECTION_H
