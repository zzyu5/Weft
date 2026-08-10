#ifndef WEFT_TARGET_RVV_SELECTEDEXECUTIONRVVSOURCE_H
#define WEFT_TARGET_RVV_SELECTEDEXECUTIONRVVSOURCE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace llvm {
class raw_ostream;
}

namespace weft::target::rvv {

llvm::Error emitSelectedExecutionRVVSource(mlir::ModuleOp module,
                                           llvm::raw_ostream &os);

llvm::Error compileRVVGeneratedSourceToObject(llvm::StringRef source,
                                              llvm::raw_ostream &os);

llvm::Error compileRVVGeneratedSourceToObject(llvm::StringRef source,
                                              llvm::StringRef selectedTarget,
                                              llvm::raw_ostream &os);

} // namespace weft::target::rvv

#endif // WEFT_TARGET_RVV_SELECTEDEXECUTIONRVVSOURCE_H
