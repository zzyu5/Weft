#ifndef WEFT_TARGET_RISCVLOWERING_H
#define WEFT_TARGET_RISCVLOWERING_H

#include "Weft/Target/RISCVTargetProfile.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

namespace llvm {
class raw_ostream;
}

namespace weft {

struct RISCVBackendConfig {
  int64_t vlaLMUL = 0;
  int64_t dotLMUL = 0;
  int64_t dotKUnroll = 0;
  int64_t f16InputLMUL = 0;
  int64_t f16RowMicrotile = 0;
  int64_t narrowLMUL = 0;
  int64_t sortRadixBits = 0;
};

struct RISCVLoweringOptions {
  RISCVTargetProfile target;
  llvm::StringMap<int64_t> metaBindings;
  RISCVBackendConfig backend;
};

mlir::LogicalResult lowerToRISCVIntrinsicC(
    mlir::ModuleOp module, const RISCVLoweringOptions &options,
    llvm::raw_ostream &output);

mlir::LogicalResult emitRISCVHeader(
    mlir::ModuleOp module, const RISCVLoweringOptions &options,
    llvm::raw_ostream &output);

} // namespace weft

#endif // WEFT_TARGET_RISCVLOWERING_H
