#ifndef WEFT_TARGET_IME_SELECTEDEXECUTIONIMESOURCE_H
#define WEFT_TARGET_IME_SELECTEDEXECUTIONIMESOURCE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
}

namespace weft::target::ime {

llvm::Error emitSelectedExecutionIMESource(mlir::ModuleOp module,
                                           llvm::raw_ostream &os);

} // namespace weft::target::ime

#endif // WEFT_TARGET_IME_SELECTEDEXECUTIONIMESOURCE_H

