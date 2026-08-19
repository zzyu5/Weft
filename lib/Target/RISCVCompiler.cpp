#include "Weft/Target/RISCVCompiler.h"

#include "RISCVHeader.h"
#include "RISCVIntrinsicC.h"
#include "RISCVKernelCompiler.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <utility>

namespace {

struct RISCVArtifactState {
  weft::RISCVArtifact artifact;
  bool complete = false;
};

class CompileRISCVArtifactPass final
    : public mlir::PassWrapper<CompileRISCVArtifactPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CompileRISCVArtifactPass)

  CompileRISCVArtifactPass(
      std::shared_ptr<const weft::RISCVCompilerOptions> options,
      std::shared_ptr<RISCVArtifactState> state)
      : options(std::move(options)), state(std::move(state)) {}

  CompileRISCVArtifactPass(const CompileRISCVArtifactPass &other)
      : mlir::PassWrapper<CompileRISCVArtifactPass,
                          mlir::OperationPass<mlir::ModuleOp>>(other),
        options(other.options), state(other.state) {}

  llvm::StringRef getArgument() const final {
    return "weft-compile-riscv-artifact";
  }

  llvm::StringRef getDescription() const final {
    return "analyze canonical Weft IR and emit one RISC-V artifact";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    if (!options->target.supportsFixedRVV()) {
      module.emitError(
          "the intrinsic C target requires RVV with an explicit fixed VLEN; no fallback backend is installed");
      signalPassFailure();
      return;
    }

    weft::riscv_internal::SelectedLocalImplementations implementations;
    std::string body;
    llvm::raw_string_ostream bodyOutput(body);
    mlir::AnalysisManager analysisManager = getAnalysisManager();
    if (mlir::failed(
            weft::riscv_internal::compileRISCVKernelsToIntrinsicC(
                module, *options, analysisManager, bodyOutput,
                implementations))) {
      signalPassFailure();
      return;
    }
    bodyOutput.flush();

    std::string prelude;
    llvm::raw_string_ostream preludeOutput(prelude);
    std::string unsupportedSymbol;
    if (!weft::riscv_internal::emitIntrinsicCPrelude(
            preludeOutput, implementations, unsupportedSymbol)) {
      module.emitError()
          << "RISC-V intrinsic C has no definition for selected local implementation '"
          << unsupportedSymbol << "'";
      signalPassFailure();
      return;
    }
    preludeOutput.flush();

    std::string header;
    llvm::raw_string_ostream headerOutput(header);
    if (mlir::failed(weft::riscv_internal::emitRISCVArtifactHeader(
            module, *options, headerOutput))) {
      signalPassFailure();
      return;
    }
    headerOutput.flush();

    state->artifact.intrinsicC = std::move(prelude) + std::move(body);
    state->artifact.header = std::move(header);
    state->complete = true;
    markAllAnalysesPreserved();
  }

private:
  std::shared_ptr<const weft::RISCVCompilerOptions> options;
  std::shared_ptr<RISCVArtifactState> state;
};

} // namespace

mlir::FailureOr<weft::RISCVArtifact>
weft::compileRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options) {
  auto sharedOptions =
      std::make_shared<const RISCVCompilerOptions>(std::move(options));
  auto state = std::make_shared<RISCVArtifactState>();
  mlir::PassManager manager(module.getContext());
  manager.addPass(
      std::make_unique<CompileRISCVArtifactPass>(sharedOptions, state));
  if (mlir::failed(manager.run(module)) || !state->complete)
    return mlir::failure();
  return std::move(state->artifact);
}
