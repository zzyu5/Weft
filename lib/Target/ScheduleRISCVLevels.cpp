#include "Weft/Target/RISCVPasses.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

#include <memory>

using namespace weft;

namespace {

constexpr llvm::StringLiteral kStageAttr = "weft.riscv.pipeline_stage";
constexpr llvm::StringLiteral kOrderAttr = "weft.riscv.pipeline_order";

bool hasWriteOrUnknownEffect(mlir::Operation *operation) {
  if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation)) {
    llvm::StringRef effect = conversion.getConversion().getEffect();
    return effect != "pure" && effect != "read";
  }
  if (mlir::isMemoryEffectFree(operation))
    return false;
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return true;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  for (const auto &instance : instances)
    if (!mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect()))
      return true;
  return false;
}

bool hasNestedRegion(mlir::Operation *operation) {
  return operation->getNumRegions() != 0;
}

bool dependsOnStageOne(mlir::Operation *operation,
                       llvm::DenseSet<mlir::Operation *> &stageOne,
                       mlir::Block *body) {
  for (mlir::Value operand : operation->getOperands()) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
      if (argument.getOwner() == body && argument.getArgNumber() > 0)
        return true;
      continue;
    }
    if (mlir::Operation *definition = operand.getDefiningOp();
        definition && stageOne.contains(definition))
      return true;
  }
  return false;
}

class ScheduleRISCVLevelsPass
    : public mlir::PassWrapper<ScheduleRISCVLevelsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-schedule-levels";
  }
  llvm::StringRef getDescription() const override {
    return "Assign dependency- and effect-derived stages to physical Level operations";
  }

  void runOnOperation() override {
    bool failed = false;
    llvm::SmallVector<mlir::scf::ForOp> loops;
    getOperation().walk<mlir::WalkOrder::PostOrder>(
        [&](mlir::scf::ForOp loop) {
          if (loop->hasAttr("weft.riscv.schedule"))
            loops.push_back(loop);
        });

    for (mlir::scf::ForOp loop : loops) {
      auto schedule = loop->getAttrOfType<riscv::ScheduleAttr>(
          "weft.riscv.schedule");
      if (!schedule) {
        loop.emitError("scheduled physical loop lost its typed schedule");
        failed = true;
        continue;
      }
      if (schedule.getPipelineDepth() == 1)
        continue;
      if (schedule.getPipelineDepth() != 2 ||
          schedule.getBufferCount() != 2) {
        loop.emitError(
            "physical scheduler currently implements one-iteration-distance, two-stage pipelines");
        failed = true;
        continue;
      }
      bool nestedScheduledLoop = false;
      loop.getBody()->walk([&](mlir::scf::ForOp nested) {
        if (nested != loop && nested->hasAttr("weft.riscv.schedule"))
          nestedScheduledLoop = true;
      });
      if (nestedScheduledLoop) {
        loop->removeAttr("weft.riscv.schedule");
        continue;
      }
      if (loop.getInitArgs().empty()) {
        loop.emitError(
            "pipelined physical Level has no loop-carried result to consume a staged producer");
        failed = true;
        continue;
      }

      mlir::Block *body = loop.getBody();
      llvm::SmallVector<mlir::Operation *> operations;
      for (mlir::Operation &operation : body->without_terminator()) {
        if (hasNestedRegion(&operation)) {
          operation.emitError(
              "nested-region operation has no physical pipeline predication contract");
          failed = true;
          operations.clear();
          break;
        }
        if (hasWriteOrUnknownEffect(&operation)) {
          operation.emitError(
              "effectful operation has no physical pipeline ordering or predication contract");
          failed = true;
          operations.clear();
          break;
        }
        operations.push_back(&operation);
      }
      if (operations.empty())
        continue;

      llvm::DenseSet<mlir::Operation *> stageOne;
      for (mlir::Operation *operation : operations)
        if (dependsOnStageOne(operation, stageOne, body))
          stageOne.insert(operation);

      auto yield = mlir::cast<mlir::scf::YieldOp>(body->getTerminator());
      for (mlir::Value value : yield.getResults())
        if (mlir::Operation *definition = value.getDefiningOp();
            definition && definition->getBlock() == body)
          stageOne.insert(definition);

      bool changed = true;
      while (changed) {
        changed = false;
        for (mlir::Operation *operation : operations)
          if (!stageOne.contains(operation) &&
              dependsOnStageOne(operation, stageOne, body)) {
            stageOne.insert(operation);
            changed = true;
          }
      }

      bool hasStageZero = false;
      bool hasStageOne = false;
      bool crossesStage = false;
      int64_t stageZeroOrder = 0;
      int64_t stageOneOrder = 0;
      for (mlir::Operation *operation : operations) {
        const bool second = stageOne.contains(operation);
        hasStageZero |= !second;
        hasStageOne |= second;
        operation->setAttr(kStageAttr,
                           mlir::IntegerAttr::get(
                               mlir::IntegerType::get(&getContext(), 64),
                               second ? 1 : 0));
        operation->setAttr(kOrderAttr,
                           mlir::IntegerAttr::get(
                               mlir::IntegerType::get(&getContext(), 64),
                               second ? stageOneOrder++ : stageZeroOrder++));
        if (!second)
          for (mlir::Value result : operation->getResults())
            for (mlir::OpOperand &use : result.getUses())
              if (use.getOwner()->getBlock() == body &&
                  (use.getOwner() == yield.getOperation() ||
                   stageOne.contains(use.getOwner())))
                crossesStage = true;
      }
      if (!hasStageZero || !hasStageOne || !crossesStage) {
        for (mlir::Operation *operation : operations) {
          operation->removeAttr(kStageAttr);
          operation->removeAttr(kOrderAttr);
        }
        loop.emitError(
            "physical Level has no dependency-derived producer/consumer cluster for the requested pipeline depth");
        failed = true;
      }
    }

    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createScheduleRISCVLevelsPass() {
  return std::make_unique<ScheduleRISCVLevelsPass>();
}
