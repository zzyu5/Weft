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

constexpr llvm::StringLiteral layoutInputAttribute = "weft.riscv.layout_input";

void addLayoutFinalization(mlir::PassManager &manager, bool scalarLoadPrime) {
  // Obsolete index paths must not count as consumers when deciding whether
  // to rematerialize the live representation at its actual use.
  manager.addPass(weft::createEliminateDeadRISCVLayoutsPass());
  manager.addPass(weft::createCanonicalizeRISCVLayoutsPass());
  manager.addPass(weft::createPlanRISCVMemoryPass());
  manager.addPass(weft::createMaterializeRISCVReplicaStorageLoadsPass());
  manager.addPass(weft::createHoistRISCVLoopInvariantsPass());
  manager.addPass(weft::createSelectRISCVOperationsPass());
  manager.addPass(weft::createFinalizeRISCVLeavesPass());
  manager.addPass(
      weft::createSelectRISCVScalarLoadPrimesPass(scalarLoadPrime));
  manager.addPass(weft::createMaterializeRISCVReadSnapshotsPass());
  manager.addPass(weft::createCloseRISCVLeafResourcesPass());
  manager.addPass(mlir::createCSEPass());
  manager.addPass(weft::createFuseRISCVPhysicalIssueLoopsPass());
  manager.addPass(weft::createEliminateDeadRISCVLayoutsPass());
  manager.addPass(weft::createMaterializeRISCVResourcesPass());
  manager.addPass(weft::createEliminateDeadRISCVLayoutsPass());
  manager.addPass(weft::createVerifyFinalRISCVPass());
}

mlir::LogicalResult runPhysicalization(mlir::ModuleOp module,
                                       weft::RISCVCompilerOptions options,
                                       bool stopBeforeLayout = false) {
  const int64_t lmulEighths = options.lmulEighths;
  const bool scalarLoadPrime = options.scalarLoadPrime != 0;
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
  manager.addPass(weft::createMaterializeRISCVProgramsPass());
  manager.addPass(weft::createVectorizeRISCVRecordLoopsPass());
  manager.addPass(weft::createCanonicalizeRISCVLayoutsPass());
  // Level materialization creates the physical points that close nested
  // domain-relative memory windows.  Plan those memory edges before bitplane
  // fusion consumes the surrounding layout conversions.
  manager.addPass(weft::createPlanRISCVNestedMemoryPass());
  manager.addPass(weft::createFuseRISCVBitplanesPass());
  manager.addPass(weft::createHoistRISCVLoopInvariantsPass());
  manager.addPass(weft::createScheduleRISCVLevelsPass());
  manager.addPass(weft::createPipelineRISCVLevelsPass());
  manager.addPass(mlir::createSCCPPass());
  manager.addPass(weft::createShareRISCVLayeredWindowsPass());
  manager.addPass(weft::createPlanRISCVPartialTopologiesPass());
  manager.addPass(weft::createMaterializeRISCVPartialAccumulatorsPass());
  // Partial topology owns issue unroll.  Preserve the original typed
  // contraction until the planner has frozen its carrier and the materializer
  // has created the issue loop; only then mechanically clone loop iterations.
  // Unrolling earlier duplicates the reduction use-def graph and makes the
  // selected full-product topology unrecognizable to its own planner.
  manager.addPass(weft::createUnrollRISCVLevelsPass());
  // Mechanical unrolling exposes complete grouped/layered issue groups created
  // by partial materialization.  Re-run the idempotent sharing owner so those
  // distinct logical offsets become explicit raw-window loads plus typed layer
  // decodes before later layout and resource passes inspect their lifetimes.
  manager.addPass(weft::createShareRISCVLayeredWindowsPass());
  // The normal compiler and parsed checkpoint share this exact suffix.  Fresh
  // rematerialization must precede memory/leaf selection and resource closure.
  if (!stopBeforeLayout)
    addLayoutFinalization(manager, scalarLoadPrime);
  if (mlir::failed(manager.run(module)))
    return mlir::failure();
  if (stopBeforeLayout) {
    mlir::Builder builder(module.getContext());
    module->setAttr(layoutInputAttribute, builder.getDictionaryAttr({
        builder.getNamedAttr("scalar_load_prime",
                             builder.getBoolAttr(scalarLoadPrime))}));
  }
  return mlir::success();
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

mlir::FailureOr<weft::RISCVPhysicalizationResult>
weft::prepareRISCVLayoutModule(mlir::ModuleOp module,
                               RISCVCompilerOptions options) {
  mlir::OwningOpRef<mlir::ModuleOp> working = module.clone();
  if (mlir::failed(runPhysicalization(*working, std::move(options), true)))
    return mlir::failure();
  return RISCVPhysicalizationResult{printModule(*working)};
}

mlir::FailureOr<weft::RISCVCompilationResult>
weft::completeRISCVLayoutModule(mlir::ModuleOp module) {
  auto checkpoint = module->getAttrOfType<mlir::DictionaryAttr>(
      layoutInputAttribute);
  auto scalarLoadPrime = checkpoint
      ? checkpoint.getAs<mlir::BoolAttr>("scalar_load_prime") : mlir::BoolAttr();
  if (!checkpoint || checkpoint.size() != 1 || !scalarLoadPrime) {
    module.emitError("layout input requires its explicit remaining binding");
    return mlir::failure();
  }
  bool foundKernel = false;
  bool closedResources = false;
  module.walk([&](riscv::KernelOp kernel) {
    foundKernel = true;
    closedResources |= kernel.getResourcesMaterialized();
  });
  if (!foundKernel || closedResources) {
    module.emitError("layout input must precede final resource materialization");
    return mlir::failure();
  }
  mlir::OwningOpRef<mlir::ModuleOp> working = module.clone();
  (*working)->removeAttr(layoutInputAttribute);
  mlir::PassManager manager(module.getContext());
  manager.enableVerifier(true);
  addLayoutFinalization(manager, scalarLoadPrime.getValue());
  if (mlir::failed(manager.run(*working)))
    return mlir::failure();
  RISCVCompilationResult result;
  result.riscvIR = printModule(*working);
  if (mlir::failed(emitSelectedRISCVIntrinsicC(*working, result.intrinsicC,
                                             result.kernels)))
    return mlir::failure();
  return result;
}

mlir::FailureOr<weft::RISCVCompilationResult>
weft::compileRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options) {
  mlir::OwningOpRef<mlir::ModuleOp> working = module.clone();
  if (mlir::failed(runPhysicalization(*working, std::move(options))))
    return mlir::failure();
  RISCVCompilationResult result;
  result.riscvIR = printModule(*working);
  if (mlir::failed(emitSelectedRISCVIntrinsicC(*working, result.intrinsicC,
                                             result.kernels)))
    return mlir::failure();
  return result;
}

mlir::FailureOr<weft::RISCVCompilationResult>
weft::translateRISCVModule(mlir::ModuleOp module) {
  mlir::OwningOpRef<mlir::ModuleOp> working = module.clone();
  mlir::PassManager manager(module.getContext());
  manager.enableVerifier(true);
  manager.addPass(weft::createVerifyFinalRISCVPass());
  if (mlir::failed(manager.run(*working)))
    return mlir::failure();
  RISCVCompilationResult result;
  result.riscvIR = printModule(*working);
  if (mlir::failed(emitSelectedRISCVIntrinsicC(*working, result.intrinsicC,
                                             result.kernels)))
    return mlir::failure();
  return result;
}
