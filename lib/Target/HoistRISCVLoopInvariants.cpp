#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

bool hasOnlyReadEffects(mlir::Operation *operation) {
  // Physical points are SSA coordinate constructors.  They intentionally do
  // not carry the generic Pure trait because their iteration identity must
  // remain explicit, but they do not read or write memory.
  if (mlir::isa<weft::riscv::PhysicalPointOp>(operation))
    return true;
  if (mlir::isMemoryEffectFree(operation))
    return true;
  auto interface =
      mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!interface)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  return llvm::all_of(effects, [](const auto &effect) {
    return mlir::isa<mlir::MemoryEffects::Read>(effect.getEffect());
  });
}

bool hasOnlyReadEffects(mlir::LoopLikeOpInterface loop) {
  bool onlyReads = true;
  for (mlir::Region *region : loop.getLoopRegions())
    region->walk([&](mlir::Operation *operation) {
      if (operation != loop.getOperation() &&
          !hasOnlyReadEffects(operation))
        onlyReads = false;
    });
  return onlyReads;
}

bool hasAtLeastTwoIterations(mlir::LoopLikeOpInterface loop) {
  auto forLoop = mlir::dyn_cast<mlir::scf::ForOp>(loop.getOperation());
  if (!forLoop)
    return false;
  auto lower =
      forLoop.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
  auto upper =
      forLoop.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
  auto step = forLoop.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
  return lower && upper && step && step.value() > 0 &&
         upper.value() - lower.value() > step.value();
}

void materializeDeferredScalarFieldsBeforeReadLoops(
    mlir::ModuleOp module, mlir::IRRewriter &rewriter) {
  llvm::SmallVector<weft::riscv::FieldOp> fields;
  module.walk([&](weft::riscv::FieldOp field) { fields.push_back(field); });
  for (weft::riscv::FieldOp field : fields) {
    mlir::Type type = field.getResult().getType();
    if ((!type.isIndex() &&
         !mlir::isa<mlir::IntegerType, mlir::FloatType>(type)) ||
        llvm::any_of(field.getResult().getUsers(), [](mlir::Operation *user) {
          return mlir::isa<weft::riscv::RegisterMaterializeOp>(user);
        }))
      continue;

    mlir::Operation *target = nullptr;
    for (mlir::Operation *cursor = field->getPrevNode(); cursor;
         cursor = cursor->getPrevNode()) {
      if (cursor == field.getOwner().getDefiningOp())
        break;
      auto loop = mlir::dyn_cast<mlir::LoopLikeOpInterface>(cursor);
      if (loop) {
        if (!loop.isDefinedOutsideOfLoop(field.getOwner()) ||
            !hasAtLeastTwoIterations(loop) || !hasOnlyReadEffects(loop))
          break;
        target = cursor;
        continue;
      }
      if (!hasOnlyReadEffects(cursor))
        break;
    }
    if (!target)
      continue;

    field->moveBefore(target);
    rewriter.setInsertionPointAfter(field);
    auto materialized = rewriter.create<weft::riscv::RegisterMaterializeOp>(
        field.getLoc(), type, field.getResult(), -1, -1, -1,
        "physical-share");
    weft::riscv_internal::copyOrigin(field, materialized);
    field.getResult().replaceAllUsesExcept(materialized.getResult(),
                                           materialized.getOperation());
  }
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
    mlir::IRRewriter rewriter(&getContext());
    materializeDeferredScalarFieldsBeforeReadLoops(getOperation(), rewriter);
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
