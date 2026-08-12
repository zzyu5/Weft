#include "Weft/Dialect/Extension/IR/ExtensionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Target/RISCVLowering.h"
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

namespace {

llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<canonical Weft MLIR>"),
    llvm::cl::init("-"));
llvm::cl::opt<std::string> outputFilename(
    "o", llvm::cl::desc("Output path"), llvm::cl::init("-"));
llvm::cl::opt<std::string> emitKind(
    "emit", llvm::cl::desc("kernel-ir or intrinsic-c"),
    llvm::cl::init("intrinsic-c"));
llvm::cl::opt<std::string> march("march", llvm::cl::desc("RISC-V ISA string"));
llvm::cl::opt<std::string> abi("abi", llvm::cl::desc("RISC-V ABI"));
llvm::cl::opt<int64_t> vlenBits(
    "vlen-bits",
    llvm::cl::desc("Fixed VLEN in bits, or zero when runtime-unknown"),
    llvm::cl::init(0));
llvm::cl::opt<std::string> matrixExtension(
    "matrix-extension",
    llvm::cl::desc("Target matrix extension: none or spacemit-ime1"),
    llvm::cl::init("none"));
llvm::cl::list<std::string> metaBindings(
    "meta", llvm::cl::desc("Specialization binding NAME=INTEGER"),
    llvm::cl::ZeroOrMore);
llvm::cl::opt<int64_t> vlaLMUL(
    "vla-lmul", llvm::cl::desc("Requested VLA data LMUL; zero selects"),
    llvm::cl::init(0));
llvm::cl::opt<int64_t> contractLMUL(
    "contract-lmul", llvm::cl::desc("Requested contract LMUL; zero selects"),
    llvm::cl::init(0));
llvm::cl::opt<int64_t> contractKUnroll(
    "contract-k-unroll",
    llvm::cl::desc("Requested contract K unroll; zero selects"),
    llvm::cl::init(0));
llvm::cl::opt<int64_t> f16InputLMUL(
    "f16-input-lmul",
    llvm::cl::desc("Requested F16 contraction input LMUL; zero selects"),
    llvm::cl::init(0));
llvm::cl::opt<int64_t> f16RowMicrotile(
    "f16-row-microtile",
    llvm::cl::desc("Requested F16 contraction row microtile; zero selects"),
    llvm::cl::init(0));
llvm::cl::opt<int64_t> narrowLMUL(
    "narrow-lmul", llvm::cl::desc("Requested f32 narrow LMUL; zero selects"),
    llvm::cl::init(0));
llvm::cl::opt<int64_t> sortRadixBits(
    "sort-radix-bits",
    llvm::cl::desc("Requested stable f32 radix width; zero selects"),
    llvm::cl::init(0));

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
  if (emitKind != "kernel-ir" && emitKind != "intrinsic-c") {
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
                  weft::extension::WEFTExtensionDialect>();
  mlir::MLIRContext context(registry);
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
  if (emitKind == "intrinsic-c") {
    weft::RISCVLoweringOptions options;
    std::string error;
    if (!weft::parseRISCVTargetProfile(march, abi, vlenBits, options.target,
                                       error)) {
      llvm::errs() << error << "\n";
      return 1;
    }
    if (matrixExtension != "none" && matrixExtension != "spacemit-ime1") {
      llvm::errs() << "unsupported --matrix-extension value: "
                   << matrixExtension << "\n";
      return 1;
    }
    options.target.matrixExtension = matrixExtension;
    options.backend.vlaLMUL = vlaLMUL;
    options.backend.contractLMUL = contractLMUL;
    options.backend.contractKUnroll = contractKUnroll;
    options.backend.f16InputLMUL = f16InputLMUL;
    options.backend.f16RowMicrotile = f16RowMicrotile;
    options.backend.narrowLMUL = narrowLMUL;
    options.backend.sortRadixBits = sortRadixBits;
    if (!parseMetaBindings(options.metaBindings) ||
        mlir::failed(
            weft::lowerToRISCVIntrinsicC(*module, options, output.os())))
      return 1;
  } else {
    module->print(output.os());
    output.os() << '\n';
  }
  output.keep();
  return 0;
}
