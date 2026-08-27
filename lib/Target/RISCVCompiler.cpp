#include "Weft/Target/RISCVCompiler.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Target/RISCVPasses.h"
#include "RISCVIntrinsicC.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

mlir::LogicalResult runPhysicalization(mlir::ModuleOp module,
                                       weft::RISCVCompilerOptions options) {
  const int64_t lmulEighths = options.lmulEighths;
  module.getContext()->getOrLoadDialect<mlir::arith::ArithDialect>();
  module.getContext()->getOrLoadDialect<mlir::scf::SCFDialect>();
  module.getContext()->getOrLoadDialect<weft::riscv::WEFTRISCVDialect>();
  mlir::PassManager manager(module.getContext());
  manager.enableVerifier(true);
  manager.addPass(weft::createConvertWeftToRISCVPass(std::move(options)));
  manager.addPass(weft::createSelectRISCVOperationsPass());
  manager.addPass(weft::createPropagateRISCVLayoutsPass(lmulEighths));
  manager.addPass(weft::createSelectRISCVOperationsPass());
  manager.addPass(weft::createPlanRISCVMemoryPass());
  manager.addPass(weft::createCanonicalizeRISCVLayoutsPass());
  manager.addPass(weft::createSelectRISCVOperationsPass());
  manager.addPass(weft::createLowerRISCVCompositesPass());
  manager.addPass(weft::createFuseRISCVBitplanesPass());
  manager.addPass(weft::createHoistRISCVLoopInvariantsPass());
  manager.addPass(weft::createScheduleRISCVLevelsPass());
  manager.addPass(weft::createPipelineRISCVLevelsPass());
  manager.addPass(weft::createUnrollRISCVLevelsPass());
  manager.addPass(weft::createShareRISCVLayeredWindowsPass());
  manager.addPass(weft::createFinalizeRISCVLeavesPass());
  manager.addPass(weft::createMaterializeRISCVResourcesPass());
  manager.addPass(weft::createVerifyFinalRISCVPass());
  return manager.run(module);
}

std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream output(text);
  module.print(output);
  output << '\n';
  output.flush();
  return text;
}

} // namespace

mlir::FailureOr<weft::RISCVPhysicalizationResult>
weft::physicalizeRISCVModule(mlir::ModuleOp module,
                            RISCVCompilerOptions options) {
  mlir::OwningOpRef<mlir::ModuleOp> working = module.clone();
  if (mlir::failed(runPhysicalization(*working, std::move(options))))
    return mlir::failure();
  RISCVPhysicalizationResult result;
  result.riscvIR = printModule(*working);
  return result;
}

mlir::FailureOr<weft::RISCVCompilationResult>
weft::compileRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options) {
  mlir::OwningOpRef<mlir::ModuleOp> working = module.clone();
  if (mlir::failed(runPhysicalization(*working, std::move(options))))
    return mlir::failure();
  RISCVCompilationResult result;
  result.riscvIR = printModule(*working);
  if (mlir::failed(emitSelectedRISCVIntrinsicC(*working, result.intrinsicC)))
    return mlir::failure();
  return result;
}
