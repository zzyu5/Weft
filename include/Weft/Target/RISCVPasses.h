#ifndef WEFT_TARGET_RISCVPASSES_H
#define WEFT_TARGET_RISCVPASSES_H

#include "Weft/Target/RISCVCompiler.h"

#include "mlir/Pass/Pass.h"

#include <memory>

namespace weft {

std::unique_ptr<mlir::Pass>
createConvertWeftToRISCVPass(RISCVCompilerOptions options);
std::unique_ptr<mlir::Pass> createSelectRISCVOperationsPass();
std::unique_ptr<mlir::Pass>
createPropagateRISCVLayoutsPass(int64_t lmulEighths);
std::unique_ptr<mlir::Pass> createPlanRISCVMemoryPass();
std::unique_ptr<mlir::Pass> createPlanRISCVNestedMemoryPass();
std::unique_ptr<mlir::Pass> createCanonicalizeRISCVLayoutsPass();
std::unique_ptr<mlir::Pass> createFuseRISCVBitplanesPass();
std::unique_ptr<mlir::Pass> createLowerRISCVCompositesPass();
std::unique_ptr<mlir::Pass> createMaterializeRISCVProgramsPass();
std::unique_ptr<mlir::Pass> createHoistRISCVLoopInvariantsPass();
std::unique_ptr<mlir::Pass> createScheduleRISCVLevelsPass();
std::unique_ptr<mlir::Pass> createPipelineRISCVLevelsPass();
std::unique_ptr<mlir::Pass> createUnrollRISCVLevelsPass();
std::unique_ptr<mlir::Pass> createShareRISCVLayeredWindowsPass();
std::unique_ptr<mlir::Pass> createMaterializeRISCVReplicaStorageLoadsPass();
std::unique_ptr<mlir::Pass> createPlanRISCVPartialTopologiesPass();
std::unique_ptr<mlir::Pass>
createMaterializeRISCVPartialAccumulatorsPass();
std::unique_ptr<mlir::Pass> createFinalizeRISCVLeavesPass();
std::unique_ptr<mlir::Pass> createMaterializeRISCVResourcesPass();
std::unique_ptr<mlir::Pass> createEliminateDeadRISCVLayoutsPass();
std::unique_ptr<mlir::Pass> createVerifyFinalRISCVPass();

} // namespace weft

#endif // WEFT_TARGET_RISCVPASSES_H
