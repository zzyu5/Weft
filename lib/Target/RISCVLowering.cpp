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

  riscv_internal::SelectedLocalImplementations selectedImplementations;
  std::string body;
  llvm::raw_string_ostream bodyOutput(body);
  if (mlir::failed(riscv_internal::compileRISCVKernelsToIntrinsicC(
          module, options, bodyOutput, selectedImplementations)))
    return mlir::failure();
  bodyOutput.flush();
  std::string prelude;
  llvm::raw_string_ostream preludeOutput(prelude);
  std::string unsupportedSymbol;
  if (!riscv_internal::emitIntrinsicCPrelude(
          preludeOutput, selectedImplementations, unsupportedSymbol))
    return module.emitError()
           << "RISC-V intrinsic C has no definition for selected local implementation '"
           << unsupportedSymbol << "'";
  preludeOutput.flush();
  output << prelude << body;
  return mlir::success();
}
