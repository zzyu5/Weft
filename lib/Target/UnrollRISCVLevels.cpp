#include "Weft/Target/RISCVPasses.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include <memory>

using namespace weft;

namespace {

class UnrollRISCVLevelsPass
    : public mlir::PassWrapper<UnrollRISCVLevelsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-unroll-levels";
  }
  llvm::StringRef getDescription() const override {
    return "Instantiate selected innermost Level unroll factors";
  }

  void runOnOperation() override {
    llvm::SmallVector<mlir::scf::ForOp> loops;
    getOperation().walk([&](mlir::scf::ForOp loop) {
      if (auto factor = loop->getAttrOfType<mlir::IntegerAttr>(
              "weft.riscv.unroll_factor");
          factor && factor.getInt() > 1)
        loops.push_back(loop);
    });

    bool failed = false;
    for (mlir::scf::ForOp loop : loops) {
      auto factor = loop->getAttrOfType<mlir::IntegerAttr>(
          "weft.riscv.unroll_factor");
      loop->removeAttr("weft.riscv.unroll_factor");
      if (mlir::failed(mlir::loopUnrollByFactor(loop, factor.getInt()))) {
        loop.emitError("selected Level unroll factor is not structurally legal");
        failed = true;
      }
    }
    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createUnrollRISCVLevelsPass() {
  return std::make_unique<UnrollRISCVLevelsPass>();
}
