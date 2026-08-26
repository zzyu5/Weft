#include "Weft/Target/RISCVPasses.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include <memory>

namespace {

class HoistRISCVLoopInvariantsPass
    : public mlir::PassWrapper<HoistRISCVLoopInvariantsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-hoist-loop-invariants";
  }
  llvm::StringRef getDescription() const override {
    return "Hoist side-effect-free physical coordinate and value expressions";
  }

  void runOnOperation() override {
    llvm::SmallVector<mlir::LoopLikeOpInterface> loops;
    getOperation().walk<mlir::WalkOrder::PostOrder>(
        [&](mlir::LoopLikeOpInterface loop) { loops.push_back(loop); });
    for (mlir::LoopLikeOpInterface loop : loops) {
      auto regions = loop.getLoopRegions();
      mlir::moveLoopInvariantCode(
          regions,
          [&](mlir::Value value, mlir::Region *) {
            return loop.isDefinedOutsideOfLoop(value);
          },
          [&](mlir::Operation *operation, mlir::Region *) {
            if (operation->getDialect() &&
                operation->getDialect()->getNamespace() == "arith")
              return mlir::isMemoryEffectFree(operation);
            return mlir::isa<weft::riscv::ConstantOp, weft::riscv::IotaOp,
                             weft::riscv::UnaryOp, weft::riscv::BinaryOp,
                             weft::riscv::CompareOp, weft::riscv::CastOp,
                             weft::riscv::NarrowOp, weft::riscv::WidenOp>(
                       operation) &&
                   mlir::isMemoryEffectFree(operation);
          },
          [&](mlir::Operation *operation, mlir::Region *) {
            loop.moveOutOfLoop(operation);
          });
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createHoistRISCVLoopInvariantsPass() {
  return std::make_unique<HoistRISCVLoopInvariantsPass>();
}
