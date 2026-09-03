#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Target/RISCVCompiler.h"
#include "Weft/Target/RISCVTargetProfile.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
    llvm::cl::Positional, llvm::cl::desc("<Weft MLIR>"),
    llvm::cl::init("-"));
llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output path"), llvm::cl::init("-"));
llvm::cl::opt<std::string> emitKind(
    "emit", llvm::cl::desc("kernel-ir, riscv-ir, or intrinsic-c"),
    llvm::cl::init("riscv-ir"));
llvm::cl::opt<std::string> march("march", llvm::cl::desc("RISC-V ISA string"));
llvm::cl::opt<std::string> abi("abi", llvm::cl::desc("RISC-V ABI"));
llvm::cl::opt<int64_t> vlenBits(
    "vlen-bits", llvm::cl::desc("Explicit fixed VLEN in bits"),
    llvm::cl::init(0));
llvm::cl::opt<std::string> matrixExtension(
    "matrix-extension",
    llvm::cl::desc("Target matrix extension: none or spacemit-ime1"),
    llvm::cl::init("none"));
llvm::cl::opt<std::string> partialCombinePolicy(
    "partial-combine-policy",
    llvm::cl::desc(
        "Target partial combine priority: independent-multilevel or sequential"),
    llvm::cl::init("independent-multilevel"));
llvm::cl::opt<std::string> recordAxisPolicy(
    "record-axis-policy",
    llvm::cl::desc(
        "Target record-axis placement priority: within-record or across-records"),
    llvm::cl::init("within-record"));
llvm::cl::list<std::string> metaBindings(
    "meta", llvm::cl::desc("Auto specialization choices NAME=INTEGER[,INTEGER...]"),
    llvm::cl::ZeroOrMore);
llvm::cl::opt<int64_t> autoUnroll(
    "auto-unroll", llvm::cl::desc("Instantiated Level-local unroll binding"),
    llvm::cl::init(1));
llvm::cl::opt<int64_t> autoLMULEighths(
    "auto-lmul-eighths",
    llvm::cl::desc("Instantiated maximum RVV LMUL binding in eighths"),
    llvm::cl::init(8));
llvm::cl::opt<int64_t> autoPipelineDepth(
    "auto-pipeline-depth",
    llvm::cl::desc("Instantiated Level-local pipeline-depth binding"),
    llvm::cl::init(1));

bool parseMetaBindings(
    llvm::StringMap<int64_t> &result) {
  for (llvm::StringRef spelling : metaBindings) {
    auto [name, valueSpelling] = spelling.split('=');
    if (name.empty() || valueSpelling.empty()) {
      llvm::errs() << "invalid --meta binding: " << spelling << "\n";
      return false;
    }
    int64_t value = 0;
    if (valueSpelling.contains(',') || valueSpelling.getAsInteger(0, value) ||
        value <= 0) {
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
  if (emitKind != "kernel-ir" && emitKind != "riscv-ir" &&
      emitKind != "intrinsic-c") {
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
                  weft::riscv::WEFTRISCVDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  context.getOrLoadDialect<weft::riscv::WEFTRISCVDialect>();
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceManager, &context);
  if (!module || mlir::failed(mlir::verify(*module)))
    return 1;
  bool hasPhysicalProgram = false;
  module->walk([&](mlir::Operation *operation) {
    if (mlir::isa<weft::riscv::KernelOp>(operation))
      hasPhysicalProgram = true;
  });

  std::error_code errorCode;
  llvm::ToolOutputFile output(outputFilename, errorCode,
                              llvm::sys::fs::OF_Text);
  if (errorCode) {
    llvm::errs() << "cannot open output: " << errorCode.message() << "\n";
    return 1;
  }
  if (emitKind == "kernel-ir") {
    if (hasPhysicalProgram) {
      llvm::errs()
          << "kernel-ir input cannot contain compiler-owned weft_riscv operations\n";
      return 1;
    }
    module->print(output.os());
    output.os() << '\n';
  } else if (hasPhysicalProgram) {
    mlir::FailureOr<weft::RISCVCompilationResult> result =
        weft::translateRISCVModule(*module);
    if (mlir::failed(result))
      return 1;
    output.os() << (emitKind == "riscv-ir" ? result->riscvIR
                                           : result->intrinsicC);
  } else {
    weft::RISCVCompilerOptions options;
    std::string error;
    if (!weft::parseRISCVTargetProfile(
            march, abi, vlenBits, matrixExtension, partialCombinePolicy,
            recordAxisPolicy,
            options.target, error)) {
      llvm::errs() << error << '\n';
      return 1;
    }
    if (!options.target.supportsLMULEighths(autoLMULEighths)) {
      llvm::errs() << "--auto-lmul-eighths is not legal for this target: "
                   << autoLMULEighths << '\n';
      return 1;
    }
    if (!parseMetaBindings(options.metaBindings))
      return 1;
    if (autoUnroll <= 0 || autoPipelineDepth <= 0 || autoPipelineDepth > 2) {
      llvm::errs()
          << "--auto-unroll must be positive and --auto-pipeline-depth must "
             "be 1 or 2\n";
      return 1;
    }
    options.lmulEighths = autoLMULEighths;
    options.unroll = autoUnroll;
    options.pipelineDepth = autoPipelineDepth;
    if (emitKind == "riscv-ir") {
      mlir::FailureOr<weft::RISCVPhysicalizationResult> result =
          weft::physicalizeRISCVModule(*module, std::move(options));
      if (mlir::failed(result))
        return 1;
      output.os() << result->riscvIR;
    } else {
      mlir::FailureOr<weft::RISCVCompilationResult> result =
          weft::compileRISCVModule(*module, std::move(options));
      if (mlir::failed(result))
        return 1;
      output.os() << result->intrinsicC;
    }
  }
  output.keep();
  return 0;
}
