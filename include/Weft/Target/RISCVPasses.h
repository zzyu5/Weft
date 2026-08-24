#ifndef WEFT_TARGET_RISCVPASSES_H
#define WEFT_TARGET_RISCVPASSES_H

#include "Weft/Target/RISCVCompiler.h"

#include "mlir/Pass/Pass.h"

#include <memory>

namespace weft {

std::unique_ptr<mlir::Pass>
createConstructRISCVProblemsPass(RISCVCompilerOptions options);
std::unique_ptr<mlir::Pass> createAssignRISCVRepresentationsPass();
std::unique_ptr<mlir::Pass> createResolveRISCVLayoutConversionsPass();
std::unique_ptr<mlir::Pass> createPropagateRISCVStorageMappingsPass();
std::unique_ptr<mlir::Pass> createSelectRISCVLocalOperationsPass();
std::unique_ptr<mlir::Pass> createScheduleRISCVLevelsPass();
std::unique_ptr<mlir::Pass> createCheckRISCVResourcesPass();
std::unique_ptr<mlir::Pass> createSelectRISCVWinnerPass();

} // namespace weft

#endif // WEFT_TARGET_RISCVPASSES_H
