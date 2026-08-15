#include "Weft/Target/RISCVLowering.h"

#include "RISCVIntrinsicC.h"
#include "RISCVKernelCompiler.h"

#include "llvm/Support/raw_ostream.h"

#include <string>

mlir::LogicalResult weft::lowerToRISCVIntrinsicC(
    mlir::ModuleOp module, const RISCVLoweringOptions &options,
    llvm::raw_ostream &output) {
  if (!options.target.supportsFixedRVV())
    return module.emitError(
        "the intrinsic C target requires RVV with an explicit fixed VLEN; no fallback backend is installed");

  riscv_internal::SelectedIntrinsicCLeaves selectedLeaves;
  std::string body;
  llvm::raw_string_ostream bodyOutput(body);
  if (mlir::failed(riscv_internal::compileRISCVKernelsToIntrinsicC(
          module, options, bodyOutput, selectedLeaves)))
    return mlir::failure();
  bodyOutput.flush();
  riscv_internal::emitIntrinsicCPrelude(output, selectedLeaves,
                                        options.target.vlenBits);
  output << body;
  return mlir::success();
}
