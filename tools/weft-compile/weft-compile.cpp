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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
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
    "emit", llvm::cl::desc("kernel-ir, riscv-layout-input, riscv-ir, intrinsic-c, or artifact"),
    llvm::cl::init("riscv-ir"));
llvm::cl::opt<bool> queryNativeTarget(
    "query-native-target", llvm::cl::desc("Discover the local RISC-V execution target as JSON"),
    llvm::cl::init(false));
llvm::cl::opt<bool> resumeLayoutInput(
    "resume-layout-input",
    llvm::cl::desc("Finalize a parsed pre-resource layout input checkpoint"),
    llvm::cl::init(false));
llvm::cl::opt<std::string> march("march", llvm::cl::desc("RISC-V ISA string"));
llvm::cl::opt<std::string> abi("abi", llvm::cl::desc("RISC-V ABI"));
llvm::cl::opt<int64_t> vlenBits(
    "vlen-bits", llvm::cl::desc("Explicit fixed VLEN in bits"),
    llvm::cl::init(0));
llvm::cl::opt<std::string> matrixExtension(
    "matrix-extension",
    llvm::cl::desc("Target matrix extension: none or spacemit-ime1"),
    llvm::cl::init("none"));
llvm::cl::opt<int64_t> maxWideningCombineGroups(
    "max-widening-combine-groups",
    llvm::cl::desc(
        "Target structural limit for one widened partial-combine result"),
    llvm::cl::init(2));
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
llvm::cl::opt<int64_t> autoScalarLoadPrime(
    "auto-scalar-load-prime",
    llvm::cl::desc(
        "Instantiated target-local scalar-prime memory-leaf binding (0 or 1)"),
    llvm::cl::init(0));

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

void writeCompilationResult(weft::RISCVCompilationResult &result,
                            llvm::raw_ostream &output) {
  if (emitKind == "riscv-ir") {
    output << result.riscvIR;
    return;
  }
  if (emitKind == "intrinsic-c") {
    output << result.intrinsicC;
    return;
  }
  llvm::json::Array kernels;
  for (const auto &kernel : result.kernels) {
    llvm::json::Array arguments;
    for (const auto &parameter : kernel.arguments) {
      llvm::json::Array shape;
      for (const auto &dimension : parameter.shape)
        shape.push_back(dimension);
      arguments.push_back(llvm::json::Object{
          {"name", parameter.name}, {"c_type", parameter.cType},
          {"encoding", parameter.encoding}, {"shape", std::move(shape)},
          {"storage_bytes", parameter.storageBytes},
          {"record_elements", parameter.recordElements},
          {"alignment", parameter.alignment}, {"alias_set", parameter.aliasSet},
          {"writable", parameter.writable}});
    }
    llvm::json::Array shapeParameters;
    for (const auto &symbol : kernel.shapeParameters)
      shapeParameters.push_back(symbol);
    llvm::json::Object bindings;
    for (const auto &[name, value] : kernel.bindings)
      bindings[name] = value;
    kernels.push_back(llvm::json::Object{
        {"symbol", kernel.symbol}, {"march", kernel.march},
        {"abi", kernel.abi}, {"vlen_bits", kernel.vlenBits},
        {"arguments", std::move(arguments)},
        {"shape_parameters", std::move(shapeParameters)},
        {"bindings", std::move(bindings)}});
  }
  llvm::json::Object artifact{
      {"kind", "weft-riscv-artifact"}, {"riscv_ir", result.riscvIR},
      {"intrinsic_c", result.intrinsicC}, {"kernels", std::move(kernels)}};
  output << llvm::formatv("{0:2}", llvm::json::Value(std::move(artifact))) << '\n';
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "Weft RISC-V representation compiler\n");
  if (queryNativeTarget) {
    if (inputFilename.getNumOccurrences() || emitKind.getNumOccurrences() ||
        outputFilename.getNumOccurrences() || march.getNumOccurrences() ||
        abi.getNumOccurrences() || vlenBits.getNumOccurrences() ||
        !metaBindings.empty()) {
      llvm::errs() << "--query-native-target is a standalone discovery operation\n";
      return 1;
    }
    weft::RISCVNativeTarget native;
    std::string error;
    if (!weft::queryNativeRISCVTarget(native, error)) {
      llvm::errs() << error << '\n';
      return 1;
    }
    llvm::json::Array cpus;
    for (int cpu : native.cpus)
      cpus.push_back(cpu);
    llvm::json::Object target{
        {"march", native.profile.march}, {"abi", native.profile.abi},
        {"xlen", native.profile.xlen}, {"vlen_bits", native.profile.vlenBits},
        {"cpus", std::move(cpus)}};
    llvm::outs() << llvm::formatv("{0:2}", llvm::json::Value(std::move(target))) << '\n';
    return 0;
  }
  if (emitKind != "kernel-ir" && emitKind != "riscv-layout-input" &&
      emitKind != "riscv-ir" &&
      emitKind != "intrinsic-c" && emitKind != "artifact") {
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
  if (resumeLayoutInput &&
      (!hasPhysicalProgram ||
       (emitKind != "riscv-ir" && emitKind != "intrinsic-c" && emitKind != "artifact") ||
       march.getNumOccurrences() || abi.getNumOccurrences() ||
       vlenBits.getNumOccurrences() || matrixExtension.getNumOccurrences() ||
       maxWideningCombineGroups.getNumOccurrences() ||
       partialCombinePolicy.getNumOccurrences() ||
       recordAxisPolicy.getNumOccurrences() || !metaBindings.empty() ||
       autoUnroll.getNumOccurrences() || autoLMULEighths.getNumOccurrences() ||
       autoPipelineDepth.getNumOccurrences() ||
       autoScalarLoadPrime.getNumOccurrences())) {
    llvm::errs() << "--resume-layout-input requires physical input, final output, "
                    "and no replacement target or parameter bindings\n";
    return 1;
  }
  if (hasPhysicalProgram && emitKind == "riscv-layout-input") {
    llvm::errs() << "riscv-layout-input must be produced from Canonical Kernel IR\n";
    return 1;
  }

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
        resumeLayoutInput ? weft::completeRISCVLayoutModule(*module)
                          : weft::translateRISCVModule(*module);
    if (mlir::failed(result))
      return 1;
    writeCompilationResult(*result, output.os());
  } else {
    weft::RISCVCompilerOptions options;
    std::string error;
    if (!weft::parseRISCVTargetProfile(
            march, abi, vlenBits, matrixExtension, maxWideningCombineGroups,
            partialCombinePolicy,
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
    if (autoUnroll <= 0 || autoPipelineDepth <= 0 || autoPipelineDepth > 2 ||
        (autoScalarLoadPrime != 0 && autoScalarLoadPrime != 1)) {
      llvm::errs()
          << "--auto-unroll must be positive and --auto-pipeline-depth must "
             "be 1 or 2; --auto-scalar-load-prime must be 0 or 1\n";
      return 1;
    }
    options.lmulEighths = autoLMULEighths;
    options.unroll = autoUnroll;
    options.pipelineDepth = autoPipelineDepth;
    options.scalarLoadPrime = autoScalarLoadPrime;
    if (emitKind == "riscv-ir" || emitKind == "riscv-layout-input") {
      mlir::FailureOr<weft::RISCVPhysicalizationResult> result =
          emitKind == "riscv-layout-input"
              ? weft::prepareRISCVLayoutModule(*module, std::move(options))
              : weft::physicalizeRISCVModule(*module, std::move(options));
      if (mlir::failed(result))
        return 1;
      output.os() << result->riscvIR;
    } else {
      mlir::FailureOr<weft::RISCVCompilationResult> result =
          weft::compileRISCVModule(*module, std::move(options));
      if (mlir::failed(result))
        return 1;
      writeCompilationResult(*result, output.os());
    }
  }
  output.keep();
  return 0;
}
