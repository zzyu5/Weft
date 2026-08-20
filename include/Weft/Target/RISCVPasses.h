#ifndef WEFT_TARGET_RISCVPASSES_H
#define WEFT_TARGET_RISCVPASSES_H

#include "Weft/Target/RISCVCompiler.h"

#include "mlir/Pass/Pass.h"

#include <memory>

namespace weft {

std::unique_ptr<mlir::Pass>
createConstructRISCVProblemsPass(RISCVCompilerOptions options);
std::unique_ptr<mlir::Pass> createConstrainRISCVRepresentationsPass();
std::unique_ptr<mlir::Pass> createConstrainRISCVInstructionsPass();
std::unique_ptr<mlir::Pass> createConstrainRISCVResourcesPass();
std::unique_ptr<mlir::Pass> createSolveRISCVProblemsPass();

} // namespace weft

#endif // WEFT_TARGET_RISCVPASSES_H
