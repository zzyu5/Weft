#ifndef WEFT_TARGET_RISCVPASSES_H
#define WEFT_TARGET_RISCVPASSES_H

#include "Weft/Target/RISCVCompiler.h"

#include "mlir/Pass/Pass.h"

#include <memory>

namespace weft {

std::unique_ptr<mlir::Pass>
createConvertWeftToRISCVPass(RISCVCompilerOptions options);
std::unique_ptr<mlir::Pass> createSelectRISCVOperationsPass();
std::unique_ptr<mlir::Pass> createPropagateRISCVLayoutsPass();
std::unique_ptr<mlir::Pass> createPlanRISCVMemoryPass();
std::unique_ptr<mlir::Pass> createCanonicalizeRISCVLayoutsPass();
std::unique_ptr<mlir::Pass> createLowerRISCVCompositesPass();
std::unique_ptr<mlir::Pass> createPipelineRISCVLevelsPass();
std::unique_ptr<mlir::Pass> createFinalizeRISCVLeavesPass();
std::unique_ptr<mlir::Pass> createMaterializeRISCVResourcesPass();
std::unique_ptr<mlir::Pass> createVerifyFinalRISCVPass();

} // namespace weft

#endif // WEFT_TARGET_RISCVPASSES_H
