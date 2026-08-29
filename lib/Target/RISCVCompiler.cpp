#include "Weft/Target/RISCVCompiler.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Target/RISCVPasses.h"
#include "RISCVIntrinsicC.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
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
  manager.addPass(weft::createPlanRISCVMemoryPass());
  manager.addPass(weft::createSelectRISCVOperationsPass());
  manager.addPass(weft::createLowerRISCVCompositesPass());
  manager.addPass(weft::createCanonicalizeRISCVLayoutsPass());
  manager.addPass(weft::createFuseRISCVBitplanesPass());
  manager.addPass(weft::createHoistRISCVLoopInvariantsPass());
  manager.addPass(weft::createScheduleRISCVLevelsPass());
  manager.addPass(weft::createPipelineRISCVLevelsPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(weft::createUnrollRISCVLevelsPass());
  manager.addPass(weft::createShareRISCVLayeredWindowsPass());
  manager.addPass(weft::createPlanRISCVPartialTopologiesPass());
  manager.addPass(weft::createMaterializeRISCVPartialAccumulatorsPass());
  // Partial materialization can introduce fresh lane-to-register edges around
  // shaped iotas and other pure producers.  Canonicalize those new edges
  // before scheduling/final leaf selection so their register forms remain
  // explicit typed values instead of terminal vector extracts.
  manager.addPass(weft::createCanonicalizeRISCVLayoutsPass());
  // Partial materialization can also create a new scalar-replica projection
  // whose final access form depends on the just-canonicalized layouts.  Close
  // that physical memory edge here; terminal emission must not reconstruct a
  // gather decision for an ExtractOp that did not exist at the earlier memory
  // planning points.
  manager.addPass(weft::createPlanRISCVMemoryPass());
  manager.addPass(weft::createHoistRISCVLoopInvariantsPass());
  // Layout canonicalization and partial materialization may rewrite the
  // carrier of surviving numerical operations.  Re-select their exact
  // target-local implementation from the final typed result instead of
  // carrying an implementation chosen for a pre-rewrite layout into leaf
  // finalization.
  manager.addPass(weft::createSelectRISCVOperationsPass());
  manager.addPass(weft::createFinalizeRISCVLeavesPass());
  manager.addPass(weft::createMaterializeRISCVResourcesPass());
  manager.addPass(weft::createEliminateDeadRISCVLayoutsPass());
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
