#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

#include <memory>

using namespace weft;

namespace {

bool rematerializable(mlir::Operation *operation) {
  return mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp,
                   riscv::CastOp, riscv::NarrowOp, riscv::WidenOp>(operation) &&
         operation->getNumResults() == 1;
}

bool isPureRepresentationConversion(riscv::ConvertLayoutOp conversion) {
  return conversion.getConversion().getEffect() == "pure";
}

class CanonicalizeRISCVLayoutsPass
    : public mlir::PassWrapper<CanonicalizeRISCVLayoutsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-canonicalize-layouts";
  }
  llvm::StringRef getDescription() const override {
    return "Eliminate and rematerialize explicit physical layout conversions";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    bool changed = true;
    while (changed) {
      changed = false;
      llvm::SmallVector<riscv::ConvertLayoutOp> conversions;
      getOperation().walk(
          [&](riscv::ConvertLayoutOp operation) { conversions.push_back(operation); });
      for (riscv::ConvertLayoutOp conversion : conversions) {
        if (!conversion)
          continue;
        if (auto previous =
                conversion.getInput().getDefiningOp<riscv::ConvertLayoutOp>()) {
          if (isPureRepresentationConversion(previous) &&
              isPureRepresentationConversion(conversion) &&
              previous.getInput().getType() == conversion.getResult().getType()) {
            conversion.getResult().replaceAllUsesWith(previous.getInput());
            rewriter.eraseOp(conversion);
            if (previous.getResult().use_empty())
              rewriter.eraseOp(previous);
            changed = true;
            continue;
          }
        }
        if (!isPureRepresentationConversion(conversion))
          continue;
        mlir::Operation *producer = conversion.getInput().getDefiningOp();
        if (!producer || !rematerializable(producer) ||
            !conversion.getInput().hasOneUse())
          continue;
        auto targetType = conversion.getResult().getType();
        rewriter.setInsertionPoint(conversion);
        llvm::SmallVector<mlir::Value> operands;
        for (mlir::Value operand : producer->getOperands()) {
          auto operandType = mlir::dyn_cast<riscv::ValueType>(operand.getType());
          if (!operandType) {
            operands.push_back(operand);
            continue;
          }
          auto kernel = conversion->getParentOfType<riscv::KernelOp>();
          riscv::LayoutAttr required = riscv_internal::projectLayout(
              rewriter, operandType, targetType.getLayout(), kernel.getTarget());
          if (!required) {
            conversion.emitError(
                "rematerialized producer has no legal operand layout projection");
            signalPassFailure();
            return;
          }
          mlir::Type requiredType = riscv_internal::withLayout(operandType, required);
          if (requiredType == operand.getType()) {
            operands.push_back(operand);
            continue;
          }
          auto edge = rewriter.create<riscv::ConvertLayoutOp>(
              conversion.getLoc(), requiredType, operand,
              riscv_internal::layoutConversion(rewriter,
                                               operandType.getLayout(), required),
              riscv::AccessAttr(),
              riscv_internal::unselectedLeaf(rewriter));
          operands.push_back(edge.getResult());
        }
        mlir::OperationState state(producer->getLoc(), producer->getName());
        state.addOperands(operands);
        state.addTypes(targetType);
        state.addAttributes(producer->getAttrs());
        mlir::Operation *rematerialized = rewriter.create(state);
        conversion.getResult().replaceAllUsesWith(rematerialized->getResult(0));
        rewriter.eraseOp(conversion);
        if (producer->use_empty())
          rewriter.eraseOp(producer);
        changed = true;
      }
    }

    // Dominating identical conversions share one SSA result.
    getOperation().walk([&](mlir::Block *block) {
      llvm::DenseMap<std::pair<mlir::Value, mlir::Type>, riscv::ConvertLayoutOp>
          available;
      for (mlir::Operation &operation : llvm::make_early_inc_range(*block)) {
        auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation);
        if (!conversion || !isPureRepresentationConversion(conversion))
          continue;
        auto key = std::make_pair(conversion.getInput(),
                                  conversion.getResult().getType());
        auto found = available.find(key);
        if (found == available.end()) {
          available.try_emplace(key, conversion);
          continue;
        }
        conversion.getResult().replaceAllUsesWith(found->second.getResult());
        rewriter.eraseOp(conversion);
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createCanonicalizeRISCVLayoutsPass() {
  return std::make_unique<CanonicalizeRISCVLayoutsPass>();
}
