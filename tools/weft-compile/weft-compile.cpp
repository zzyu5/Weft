#include "Weft/InitWeftDialects.h"
#include "Weft/Dialect/IMEExecution/IR/IMEExecutionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Target/IME/SelectedExecutionIMESource.h"
#include "Weft/Target/RVV/SelectedExecutionRVVSource.h"
#include "Weft/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>
#include <utility>

namespace {

enum class EmissionKind { SelectedMLIR, Source, Object };

llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<canonical Weft MLIR>"),
    llvm::cl::init("-"));

llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
    llvm::cl::init("-"));

llvm::cl::opt<EmissionKind> emissionKind(
    "emit", llvm::cl::desc("Select the final Weft compiler boundary"),
    llvm::cl::values(
        clEnumValN(EmissionKind::SelectedMLIR, "selected-mlir",
                   "Persist canonical Kernel IR plus selected execution/layout"),
        clEnumValN(EmissionKind::Source, "source",
                   "Emit source through the selected execution owner"),
        clEnumValN(EmissionKind::Object, "object",
                   "Emit a RISC-V relocatable object")),
    llvm::cl::init(EmissionKind::Object));

llvm::cl::opt<std::string> target(
    "target", llvm::cl::desc("Selected RV64 target identity"),
    llvm::cl::init("rv64gcv"));

llvm::cl::opt<int64_t> vlenBits(
    "vlen-bits",
    llvm::cl::desc(
        "IME-only target VLEN in bits; xsmtvdotii currently requires 256"),
    llvm::cl::init(0));

llvm::cl::opt<int64_t> registerBudget(
    "register-budget",
    llvm::cl::desc("RVV registers available to selected blocked values"),
    llvm::cl::init(28));

llvm::cl::list<std::string> metaBindings(
    "meta",
    llvm::cl::desc(
        "Bind one canonical constexpr parameter (repeat NAME=VALUE)"),
    llvm::cl::value_desc("NAME=VALUE"), llvm::cl::ZeroOrMore);

llvm::cl::opt<int64_t> blockElements(
    "block-elements",
    llvm::cl::desc(
        "Explicit positive uniform value for every constexpr parameter"),
    llvm::cl::init(0));

int reportError(llvm::StringRef prefix, llvm::Error error) {
  llvm::errs() << prefix << ": " << llvm::toString(std::move(error)) << "\n";
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "Weft standalone RISC-V blocked/VLA kernel compiler\n");

  mlir::DialectRegistry registry;
  weft::registerAllDialects(registry);
  mlir::MLIRContext context(registry);

  std::string inputError;
  std::unique_ptr<llvm::MemoryBuffer> input =
      mlir::openInputFile(inputFilename, &inputError);
  if (!input) {
    llvm::errs() << "failed to open canonical Weft MLIR input '"
                 << inputFilename << "': " << inputError << "\n";
    return 1;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  mlir::SourceMgrDiagnosticHandler diagnosticHandler(sourceMgr, &context);
  mlir::ParserConfig parserConfig(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, parserConfig);
  if (!module)
    return 1;

  bool hasKernel = false;
  for (mlir::Operation &operation : module->getBody()->getOperations()) {
    if (llvm::isa<weft::kernel::KernelOp>(operation)) {
      hasKernel = true;
      continue;
    }
    operation.emitOpError(
        "is not accepted by the standalone canonical Weft compiler input; "
        "expected only top-level weft_kernel.kernel operations");
    return 1;
  }
  if (!hasKernel) {
    module->emitError(
        "standalone canonical Weft compiler input contains no "
        "weft_kernel.kernel");
    return 1;
  }

  bool selectsIME =
      weft::ime_execution::hasXsmtvdotiiExtension(llvm::StringRef(target));

  mlir::PassManager passManager(&context);
  passManager.enableVerifier(true);
  if (selectsIME) {
    weft::transforms::SelectIMEExecutionOptions selectionOptions;
    selectionOptions.target = target;
    selectionOptions.vlenBits = vlenBits;
    selectionOptions.metaBindings.assign(metaBindings.begin(), metaBindings.end());
    selectionOptions.blockElements = blockElements;
    passManager.addPass(weft::transforms::createSelectIMEExecutionPass(
        std::move(selectionOptions)));
  } else {
    weft::transforms::SelectRISCvExecutionOptions selectionOptions;
    selectionOptions.target = target;
    selectionOptions.registerBudget = registerBudget;
    selectionOptions.metaBindings.assign(metaBindings.begin(), metaBindings.end());
    selectionOptions.blockElements = blockElements;
    passManager.addPass(weft::transforms::createSelectRISCvExecutionPass(
        std::move(selectionOptions)));
  }
  if (mlir::failed(passManager.run(*module))) {
    llvm::errs() << "weft-compile failed while selecting owner-local RISC-V "
                    "execution\n";
    return 1;
  }

  if (selectsIME && emissionKind == EmissionKind::Object) {
    llvm::errs()
        << "selected IME object packaging is not available in this build; emit "
           "selected-mlir or source and compile it with the SpacemiT "
           "xsmtvdotii toolchain\n";
    return 1;
  }
  if (emissionKind == EmissionKind::Object && outputFilename == "-") {
    if (std::error_code error = llvm::sys::ChangeStdoutToBinary()) {
      llvm::errs() << "failed to switch stdout to binary mode: "
                   << error.message() << "\n";
      return 1;
    }
  }

  std::string outputError;
  std::unique_ptr<llvm::ToolOutputFile> output =
      mlir::openOutputFile(outputFilename, &outputError);
  if (!output) {
    llvm::errs() << "failed to open output '" << outputFilename
                 << "': " << outputError << "\n";
    return 1;
  }

  if (emissionKind == EmissionKind::SelectedMLIR) {
    module->print(output->os());
    output->os() << "\n";
  } else if (emissionKind == EmissionKind::Source) {
    llvm::Error error =
        selectsIME
            ? weft::target::ime::emitSelectedExecutionIMESource(*module,
                                                                output->os())
            : weft::target::rvv::emitSelectedExecutionRVVSource(*module,
                                                                output->os());
    if (error)
      return reportError("failed to emit selected owner source",
                         std::move(error));
  } else {
    llvm::SmallString<0> sourceStorage;
    llvm::raw_svector_ostream source(sourceStorage);
    if (llvm::Error error =
            weft::target::rvv::emitSelectedExecutionRVVSource(*module, source))
      return reportError("failed to emit selected RVV source",
                         std::move(error));
    if (llvm::Error error =
            weft::target::rvv::compileRVVGeneratedSourceToObject(
                source.str(), target, output->os()))
      return reportError("failed to package selected RVV object",
                         std::move(error));
  }

  output->keep();
  return 0;
}
