#include "Weft/Compiler/Selection.h"
#include "Weft/Dialect/Execution/IR/ExecutionDialect.h"
#include "Weft/Dialect/Extension/IR/ExtensionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Target/RISCVTargetProfile.h"
#include "Weft/Target/SourceEmitter.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace {

llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<canonical Weft MLIR>"),
    llvm::cl::init("-"));
llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output path"), llvm::cl::init("-"));
llvm::cl::opt<std::string> emitKind(
    "emit", llvm::cl::desc("canonical-mlir, selected-mlir, or source"),
    llvm::cl::init("selected-mlir"));
llvm::cl::opt<std::string> march("march", llvm::cl::desc("RISC-V ISA string"));
llvm::cl::opt<std::string> abi("abi", llvm::cl::desc("RISC-V ABI"));
llvm::cl::opt<int64_t> vlenBits(
    "vlen-bits",
    llvm::cl::desc("Fixed VLEN in bits, or zero when runtime-unknown"),
    llvm::cl::init(0));
llvm::cl::list<std::string> metaBindings(
    "meta", llvm::cl::desc("Specialization binding NAME=INTEGER"),
    llvm::cl::ZeroOrMore);

bool parseMetaBindings(llvm::StringMap<int64_t> &result) {
  for (llvm::StringRef spelling : metaBindings) {
    auto [name, valueSpelling] = spelling.split('=');
    int64_t value = 0;
    if (name.empty() || valueSpelling.empty() ||
        valueSpelling.getAsInteger(0, value)) {
      llvm::errs() << "invalid --meta binding: " << spelling << "\n";
      return false;
    }
    if (!result.try_emplace(name, value).second) {
      llvm::errs() << "duplicate --meta binding: " << name << "\n";
      return false;
    }
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Weft RISC-V kernel compiler\n");
  if (emitKind != "canonical-mlir" && emitKind != "selected-mlir" &&
      emitKind != "source") {
    llvm::errs() << "unsupported --emit value: " << emitKind << "\n";
    return 1;
  }

  auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!buffer) {
    llvm::errs() << "cannot read " << inputFilename << ": "
                 << buffer.getError().message() << "\n";
    return 1;
  }
  llvm::SourceMgr sourceManager;
  sourceManager.AddNewSourceBuffer(std::move(*buffer), llvm::SMLoc());

  mlir::DialectRegistry registry;
  registry.insert<weft::kernel::WEFTKernelDialect,
                  weft::extension::WEFTExtensionDialect,
                  weft::execution::WEFTExecutionDialect>();
  mlir::MLIRContext context(registry);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceManager, &context);
  if (!module || mlir::failed(mlir::verify(*module)))
    return 1;

  if (emitKind == "selected-mlir" || emitKind == "source") {
    weft::RISCVTargetProfile target;
    std::string error;
    if (!weft::parseRISCVTargetProfile(march, abi, vlenBits, target, error)) {
      llvm::errs() << error << "\n";
      return 1;
    }
    weft::SelectionOptions options;
    options.target = std::move(target);
    if (!parseMetaBindings(options.metaBindings) ||
        mlir::failed(weft::selectExecution(*module, options)) ||
        mlir::failed(mlir::verify(*module)))
      return 1;
  }

  std::error_code errorCode;
  llvm::ToolOutputFile output(outputFilename, errorCode,
                              llvm::sys::fs::OF_Text);
  if (errorCode) {
    llvm::errs() << "cannot open output: " << errorCode.message() << "\n";
    return 1;
  }
  if (emitKind == "source") {
    if (mlir::failed(weft::emitSelectedSource(*module, output.os())))
      return 1;
  } else {
    module->print(output.os());
    output.os() << '\n';
  }
  output.keep();
  return 0;
}
