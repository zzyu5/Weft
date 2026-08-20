#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"
#include "Weft/Target/RISCVCompiler.h"
#include "Weft/Target/RISCVTargetProfile.h"

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

#include <utility>

namespace {

llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<canonical Weft MLIR>"),
    llvm::cl::init("-"));
llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output path"), llvm::cl::init("-"));
llvm::cl::opt<std::string> emitKind(
    "emit", llvm::cl::desc("kernel-ir or physical-assignment"),
    llvm::cl::init("physical-assignment"));
llvm::cl::opt<std::string> march("march", llvm::cl::desc("RISC-V ISA string"));
llvm::cl::opt<std::string> abi("abi", llvm::cl::desc("RISC-V ABI"));
llvm::cl::opt<int64_t> vlenBits(
    "vlen-bits", llvm::cl::desc("Explicit fixed VLEN in bits"),
    llvm::cl::init(0));
llvm::cl::opt<std::string> matrixExtension(
    "matrix-extension",
    llvm::cl::desc("Target matrix extension: none or spacemit-ime1"),
    llvm::cl::init("none"));
llvm::cl::list<std::string> metaBindings(
    "meta", llvm::cl::desc("Auto specialization binding NAME=INTEGER"),
    llvm::cl::ZeroOrMore);

bool parseMetaBindings(llvm::StringMap<int64_t> &result) {
  for (llvm::StringRef spelling : metaBindings) {
    auto [name, valueSpelling] = spelling.split('=');
    int64_t value = 0;
    if (name.empty() || valueSpelling.empty() ||
        valueSpelling.getAsInteger(0, value) || value <= 0) {
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
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Weft RISC-V representation compiler\n");
  if (emitKind != "kernel-ir" && emitKind != "physical-assignment") {
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
                  weft::riscv::WEFTRISCVDialect>();
  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<weft::riscv::WEFTRISCVDialect>();
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceManager, &context);
  if (!module || mlir::failed(mlir::verify(*module)))
    return 1;

  std::error_code errorCode;
  llvm::ToolOutputFile output(outputFilename, errorCode,
                              llvm::sys::fs::OF_Text);
  if (errorCode) {
    llvm::errs() << "cannot open output: " << errorCode.message() << "\n";
    return 1;
  }
  if (emitKind == "kernel-ir") {
    bool hasTransientPlanning = false;
    module->walk([&](mlir::Operation *operation) {
      if (mlir::isa<weft::riscv::ProblemOp, weft::riscv::AssignmentOp>(operation))
        hasTransientPlanning = true;
    });
    if (hasTransientPlanning) {
      llvm::errs() << "kernel-ir input cannot contain compiler-owned weft_riscv operations\n";
      return 1;
    }
    module->print(output.os());
    output.os() << '\n';
  } else {
    weft::RISCVCompilerOptions options;
    std::string error;
    if (!weft::parseRISCVTargetProfile(march, abi, vlenBits, matrixExtension,
                                       options.target, error)) {
      llvm::errs() << error << '\n';
      return 1;
    }
    if (!parseMetaBindings(options.metaBindings))
      return 1;
    mlir::FailureOr<weft::RISCVPlanningResult> result =
        weft::planRISCVModule(*module, std::move(options));
    if (mlir::failed(result))
      return 1;
    output.os() << result->assignment;
  }
  output.keep();
  return 0;
}
