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

      // Resource materialization may spill a value whose reload-side slice is
      // subsequently deleted as a dead physical producer.  The spill writes a
      // compiler-owned local slot and is unobservable once that slot has no
      // reader; retaining it would turn a dead representation choice into real
      // vector stores in the emitted kernel.
      llvm::SmallVector<riscv::SpillOp> unreadSpills;
      getOperation().walk([&](riscv::SpillOp spill) {
        const bool hasReader = llvm::any_of(
            spill.getSlot().getUsers(),
            [](mlir::Operation *user) { return !mlir::isa<riscv::SpillOp>(user); });
        if (!hasReader)
          unreadSpills.push_back(spill);
      });
      for (riscv::SpillOp spill : unreadSpills) {
        spill.erase();
        changed = true;
      }
      llvm::SmallVector<riscv::LocalAllocOp> deadSpillSlots;
      getOperation().walk([&](riscv::LocalAllocOp allocation) {
        if (allocation.getResult().use_empty() &&
            allocation.getResult().getType().getPurpose() == "spill")
          deadSpillSlots.push_back(allocation);
      });
      for (riscv::LocalAllocOp allocation : deadSpillSlots) {
        allocation.erase();
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
