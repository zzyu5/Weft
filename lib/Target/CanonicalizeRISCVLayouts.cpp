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
  return mlir::isa<riscv::IotaOp, riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp,
                   riscv::CastOp, riscv::NarrowOp, riscv::WidenOp>(operation) &&
         operation->getNumResults() == 1;
}

bool isPureRepresentationConversion(riscv::ConvertLayoutOp conversion) {
  return conversion.getConversion().getEffect() == "pure";
}

bool hasFinalResourceContract(riscv::ConvertLayoutOp conversion) {
  auto kernel = conversion->getParentOfType<riscv::KernelOp>();
  return kernel && kernel.getResourcesMaterialized();
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
        // Resource materialization freezes the final physical program.  A
        // backward-rematerialization after that point would change layouts,
        // selected leaves, and the measured live-set without rerunning their
        // owners.  Keep only the dominance-safe identical-conversion CSE below
        // available when this pass is replayed on independently parsed final
        // RISC-V IR.
        if (hasFinalResourceContract(conversion))
          continue;
        if (auto previous =
                conversion.getInput().getDefiningOp<riscv::ConvertLayoutOp>()) {
          if (isPureRepresentationConversion(previous) &&
              isPureRepresentationConversion(conversion)) {
            if (previous.getInput().getType() ==
                conversion.getResult().getType()) {
              conversion.getResult().replaceAllUsesWith(previous.getInput());
              rewriter.eraseOp(conversion);
              if (previous.getResult().use_empty())
                rewriter.eraseOp(previous);
              changed = true;
              continue;
            }
            // Compose two pure, layout-only edges before attempting backward
            // rematerialization.  Without this, an iota or pointwise producer
            // can remain hidden behind an intermediate representation and the
            // terminal emitter would be forced to interpret a conversion
            // chain.  Memory-carrying conversions keep their explicit edges.
            if (!previous->hasAttr("source_access") &&
                !conversion->hasAttr("source_access")) {
              auto source = mlir::cast<riscv::ValueType>(
                  previous.getInput().getType());
              auto target = mlir::cast<riscv::ValueType>(
                  conversion.getResult().getType());
              rewriter.setInsertionPoint(conversion);
              auto composed = rewriter.create<riscv::ConvertLayoutOp>(
                  conversion.getLoc(), target, previous.getInput(),
                  riscv_internal::layoutConversion(
                      rewriter, source.getLayout(), target.getLayout()),
                  riscv::AccessAttr(),
                  riscv_internal::unselectedLeaf(rewriter));
              conversion.getResult().replaceAllUsesWith(composed.getResult());
              rewriter.eraseOp(conversion);
              if (previous.getResult().use_empty())
                rewriter.eraseOp(previous);
              changed = true;
              continue;
            }
          }
        }
        if (!isPureRepresentationConversion(conversion))
          continue;
        mlir::Operation *producer = conversion.getInput().getDefiningOp();
        if (auto extract = mlir::dyn_cast_or_null<riscv::ExtractOp>(producer)) {
          if (!conversion.getInput().hasOneUse() ||
              !extract->hasAttr("index_pattern"))
            continue;
          rewriter.setInsertionPoint(conversion);
          auto rematerialized = rewriter.create<riscv::ExtractOp>(
              extract.getLoc(), conversion.getResult().getType(),
              extract.getInput(), extract.getIndices(), extract.getSelectors(),
              extract.getAccess(), extract.getLeaf());
          for (auto attribute : extract->getAttrs())
            if (attribute.getName() != "operandSegmentSizes")
              rematerialized->setAttr(attribute.getName(), attribute.getValue());
          conversion.getResult().replaceAllUsesWith(rematerialized.getResult());
          rewriter.eraseOp(conversion);
          rewriter.eraseOp(extract);
          changed = true;
          continue;
        }
        if (!producer || !rematerializable(producer) ||
            !conversion.getInput().hasOneUse())
          continue;
        auto targetType = conversion.getResult().getType();
        auto targetValue = mlir::cast<riscv::ValueType>(targetType);
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
          mlir::Value projected = operand;
          if (requiredType != operand.getType()) {
            auto edge = rewriter.create<riscv::ConvertLayoutOp>(
                conversion.getLoc(), requiredType, operand,
                riscv_internal::layoutConversion(
                    rewriter, operandType.getLayout(), required),
                riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
            riscv_internal::copyOrigin(producer, edge);
            projected = edge.getResult();
            operandType = mlir::cast<riscv::ValueType>(requiredType);
          }

          bool properSubdomain =
              operandType.getAxisIds().size() < targetValue.getAxisIds().size() &&
              operandType.getLayout().getCarrier() == "rvv" &&
              targetValue.getLayout().getCarrier() == "rvv" &&
              operandType.getLayout().getSew() == targetValue.getLayout().getSew();
          for (auto [position, axis] :
               llvm::enumerate(operandType.getAxisIds().asArrayRef())) {
            auto found = llvm::find(targetValue.getAxisIds().asArrayRef(), axis);
            properSubdomain &=
                found != targetValue.getAxisIds().asArrayRef().end() &&
                operandType.getShape()[position] ==
                    targetValue.getShape()[static_cast<size_t>(
                        found - targetValue.getAxisIds().asArrayRef().begin())];
          }
          auto inputLanes = riscv_internal::staticProduct(
              operandType.getLayout().getLaneFactors().asArrayRef());
          auto targetLanes = riscv_internal::staticProduct(
              targetValue.getLayout().getLaneFactors().asArrayRef());
          if (properSubdomain && inputLanes && targetLanes &&
              *targetLanes > *inputLanes) {
            auto broadcastType = riscv::ValueType::get(
                rewriter.getContext(), operandType.getElementType(),
                targetValue.getShape(), targetValue.getAxisIds(),
                targetValue.getLayout());
            auto broadcast = rewriter.create<riscv::RVVAxisBroadcastOp>(
                conversion.getLoc(), broadcastType, projected,
                riscv_internal::leaf(
                    rewriter, "rvv", "axis-broadcast",
                    "rvv.axis-broadcast", "rvv.axis-broadcast",
                    operandType.getLayout().getRegisterGroups(),
                    targetValue.getLayout().getRegisterGroups(), 1, 0, "none",
                    "exact", {*inputLanes, *targetLanes}));
            riscv_internal::copyOrigin(producer, broadcast);
            projected = broadcast.getResult();
          }
          operands.push_back(projected);
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
      llvm::SmallVector<riscv::ConvertLayoutOp> available;
      for (mlir::Operation &operation : llvm::make_early_inc_range(*block)) {
        auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation);
        if (!conversion || !isPureRepresentationConversion(conversion))
          continue;
        auto found = llvm::find_if(available, [&](riscv::ConvertLayoutOp prior) {
          return prior.getInput() == conversion.getInput() &&
                 prior.getResult().getType() ==
                     conversion.getResult().getType() &&
                 prior->getAttr("conversion") ==
                     conversion->getAttr("conversion") &&
                 prior->getAttr("source_access") ==
                     conversion->getAttr("source_access") &&
                 prior->getAttr("leaf") == conversion->getAttr("leaf");
        });
        if (found == available.end()) {
          available.push_back(conversion);
          continue;
        }
        conversion.getResult().replaceAllUsesWith(found->getResult());
        rewriter.eraseOp(conversion);
      }
    });

  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createCanonicalizeRISCVLayoutsPass() {
  return std::make_unique<CanonicalizeRISCVLayoutsPass>();
}
