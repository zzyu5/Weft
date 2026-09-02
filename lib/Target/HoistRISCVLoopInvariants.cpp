#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

#include "llvm/ADT/SmallPtrSet.h"

#include <memory>

namespace {

bool dependsOnDeferredField(mlir::Value value,
                            llvm::SmallPtrSetImpl<mlir::Operation *> &visited) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition ||
      mlir::isa<weft::riscv::RegisterMaterializeOp>(definition))
    return false;
  if (mlir::isa<weft::riscv::FieldOp>(definition))
    return true;
  if (!mlir::isMemoryEffectFree(definition) || !visited.insert(definition).second)
    return false;
  return llvm::any_of(definition->getOperands(), [&](mlir::Value operand) {
    return dependsOnDeferredField(operand, visited);
  });
}

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
      bool hasWriteEffect = false;
      for (mlir::Region *region : loop.getLoopRegions())
        region->walk([&](mlir::Operation *operation) {
          if (operation == loop.getOperation())
            return;
          auto effects =
              mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
          if (!effects)
            return;
          llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
          effects.getEffects(instances);
          hasWriteEffect |= llvm::any_of(instances, [](const auto &instance) {
            return !mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
          });
        });
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
            if (auto lookup =
                    mlir::dyn_cast<weft::riscv::LookupOp>(operation))
              return mlir::isa<weft::riscv::ValueType>(
                  lookup.getTable().getType());
            if (mlir::isa<weft::riscv::RVVReplicaStorageLoadOp>(operation))
              return !hasWriteEffect;
            if (auto conversion =
                    mlir::dyn_cast<weft::riscv::ConvertLayoutOp>(operation))
              return conversion.getConversion().getEffect() == "pure";
            return mlir::isa<weft::riscv::ConstantOp,
                             weft::riscv::IotaOp,
                             weft::riscv::UnaryOp, weft::riscv::BinaryOp,
                             weft::riscv::CompareOp, weft::riscv::CastOp,
                             weft::riscv::NarrowOp, weft::riscv::WidenOp,
                             weft::riscv::LookupOp, weft::riscv::FieldOp,
                             weft::riscv::ExtractOp>(operation) &&
                   mlir::isMemoryEffectFree(operation);
          },
          [&](mlir::Operation *operation, mlir::Region *) {
            loop.moveOutOfLoop(operation);
          });
    }
    llvm::SmallVector<mlir::Operation *> operations;
    getOperation().walk(
        [&](mlir::Operation *operation) { operations.push_back(operation); });
    for (mlir::Operation *operation : operations) {
      for (mlir::Value result : operation->getResults()) {
        mlir::Type type = result.getType();
        if (!type.isIndex() &&
            !mlir::isa<mlir::IntegerType, mlir::FloatType>(type))
          continue;
        llvm::SmallPtrSet<mlir::Operation *, 8> visited;
        if (!dependsOnDeferredField(result, visited))
          continue;
        llvm::SmallVector<mlir::OpOperand *> nestedUses;
        for (mlir::OpOperand &use : result.getUses()) {
          mlir::Operation *ancestor = use.getOwner();
          while (ancestor && ancestor->getBlock() != operation->getBlock())
            ancestor = ancestor->getParentOp();
          if (ancestor && ancestor != use.getOwner())
            nestedUses.push_back(&use);
        }
        if (nestedUses.empty())
          continue;
        mlir::OpBuilder builder(operation);
        builder.setInsertionPointAfter(operation);
        auto materialized = builder.create<weft::riscv::RegisterMaterializeOp>(
            operation->getLoc(), type, result, -1, -1, -1,
            "physical-share");
        weft::riscv_internal::copyOrigin(operation,
                                         materialized.getOperation());
        for (mlir::OpOperand *use : nestedUses)
          use->set(materialized.getResult());
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createHoistRISCVLoopInvariantsPass() {
  return std::make_unique<HoistRISCVLoopInvariantsPass>();
}
