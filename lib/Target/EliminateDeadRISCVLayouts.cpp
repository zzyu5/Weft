#include "Weft/Target/RISCVPasses.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include <memory>

using namespace weft;

namespace {

class EliminateDeadRISCVLayoutsPass
    : public mlir::PassWrapper<EliminateDeadRISCVLayoutsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-eliminate-dead-layouts";
  }

  llvm::StringRef getDescription() const override {
    return "Erase unused pure layout conversions and dead physical producers";
  }

  void runOnOperation() override {
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SmallVector<riscv::ConvertLayoutOp> dead;
      getOperation().walk([&](riscv::ConvertLayoutOp conversion) {
        if (conversion.getConversion().getEffect() == "pure" &&
            conversion.getResult().use_empty())
          dead.push_back(conversion);
      });
      for (riscv::ConvertLayoutOp conversion : dead) {
        conversion.erase();
        changed = true;
      }

      llvm::SmallVector<mlir::Operation *> deadProducers;
      getOperation().walk([&](mlir::Operation *operation) {
        if (operation != getOperation() && mlir::isOpTriviallyDead(operation))
          deadProducers.push_back(operation);
      });
      for (mlir::Operation *operation : deadProducers) {
        operation->erase();
        changed = true;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createEliminateDeadRISCVLayoutsPass() {
  return std::make_unique<EliminateDeadRISCVLayoutsPass>();
}
