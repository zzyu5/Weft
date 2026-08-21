#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"
#include "Weft/Target/RISCVCompiler.h"
#include "Weft/Target/RISCVTargetProfile.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
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
    "emit", llvm::cl::desc("kernel-ir, physical-assignment, or intrinsic-c"),
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
    "meta", llvm::cl::desc("Auto specialization choices NAME=INTEGER[,INTEGER...]"),
    llvm::cl::ZeroOrMore);
llvm::cl::list<int64_t> autoUnroll(
    "auto-unroll", llvm::cl::desc("Level-local unroll choices"),
    llvm::cl::CommaSeparated, llvm::cl::ZeroOrMore);
llvm::cl::list<int64_t> autoPipelineDepth(
    "auto-pipeline-depth", llvm::cl::desc("Level-local pipeline depth choices"),
    llvm::cl::CommaSeparated, llvm::cl::ZeroOrMore);
llvm::cl::list<int64_t> autoPrefetchDistance(
    "auto-prefetch-distance", llvm::cl::desc("Level-local prefetch distance choices"),
    llvm::cl::CommaSeparated, llvm::cl::ZeroOrMore);

bool parseMetaBindings(
    llvm::StringMap<llvm::SmallVector<int64_t, 4>> &result) {
  for (llvm::StringRef spelling : metaBindings) {
    auto [name, valueSpelling] = spelling.split('=');
    if (name.empty() || valueSpelling.empty()) {
      llvm::errs() << "invalid --meta binding: " << spelling << "\n";
      return false;
    }
    llvm::SmallVector<int64_t, 4> values;
    llvm::SmallSet<int64_t, 8> seen;
    llvm::SmallVector<llvm::StringRef, 4> spellings;
    valueSpelling.split(spellings, ',', -1, true);
    for (llvm::StringRef valueText : spellings) {
      int64_t value = 0;
      if (valueText.getAsInteger(0, value) || value <= 0) {
        llvm::errs() << "invalid --meta binding: " << spelling << "\n";
        return false;
      }
      if (seen.insert(value).second)
        values.push_back(value);
    }
    if (values.empty()) {
      llvm::errs() << "invalid --meta binding: " << spelling << "\n";
      return false;
    }
    if (!result.try_emplace(name, std::move(values)).second) {
      llvm::errs() << "duplicate --meta binding: " << name << "\n";
      return false;
    }
  }
  return true;
}

bool assignChoices(llvm::cl::list<int64_t> &source,
                   llvm::SmallVectorImpl<int64_t> &destination,
                   llvm::StringRef option, bool allowZero) {
  if (source.empty())
    return true;
  llvm::SmallSet<int64_t, 8> seen;
  destination.clear();
  for (int64_t value : source) {
    if (value < 0 || (!allowZero && value == 0)) {
      llvm::errs() << "--" << option << " contains an invalid value: " << value
                   << "\n";
      return false;
    }
    if (seen.insert(value).second)
      destination.push_back(value);
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Weft RISC-V representation compiler\n");
  if (emitKind != "kernel-ir" && emitKind != "physical-assignment" &&
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
    if (!assignChoices(autoUnroll, options.unrollChoices, "auto-unroll", false) ||
        !assignChoices(autoPipelineDepth, options.pipelineDepthChoices,
                       "auto-pipeline-depth", false) ||
        !assignChoices(autoPrefetchDistance, options.prefetchDistanceChoices,
                       "auto-prefetch-distance", true))
      return 1;
    if (llvm::any_of(options.pipelineDepthChoices,
                     [](int64_t value) { return value > 2; })) {
      llvm::errs() << "--auto-pipeline-depth currently supports only 1 or 2\n";
      return 1;
    }
    if (emitKind == "physical-assignment") {
      mlir::FailureOr<weft::RISCVPlanningResult> result =
          weft::planRISCVModule(*module, std::move(options));
      if (mlir::failed(result))
        return 1;
      output.os() << result->assignment;
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
