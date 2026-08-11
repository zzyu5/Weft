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

struct RISCVLoweringOptions {
  RISCVTargetProfile target;
  llvm::StringMap<int64_t> metaBindings;
};

mlir::LogicalResult lowerToRISCVIntrinsicC(
    mlir::ModuleOp module, const RISCVLoweringOptions &options,
    llvm::raw_ostream &output);

} // namespace weft

#endif // WEFT_TARGET_RISCVLOWERING_H
