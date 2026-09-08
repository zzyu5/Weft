#include "Weft/Target/RISCVPasses.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

#include <memory>
#include <optional>
#include <algorithm>

using namespace weft;

namespace {

constexpr llvm::StringLiteral kStageAttr = "weft.riscv.pipeline_stage";
constexpr llvm::StringLiteral kOrderAttr = "weft.riscv.pipeline_order";

std::optional<int64_t> stageOf(mlir::Operation *operation) {
  auto stage = operation->getAttrOfType<mlir::IntegerAttr>(kStageAttr);
  if (!stage)
    return std::nullopt;
  return stage.getInt();
}

std::optional<int64_t> orderOf(mlir::Operation *operation) {
  auto order = operation->getAttrOfType<mlir::IntegerAttr>(kOrderAttr);
  if (!order || order.getInt() < 0)
    return std::nullopt;
  return order.getInt();
}

bool recursivelyUses(mlir::Operation *operation, mlir::Value value) {
  bool found = false;
  operation->walk([&](mlir::Operation *nested) {
    if (!found && llvm::is_contained(nested->getOperands(), value))
      found = true;
  });
  return found;
}

mlir::Operation *bodyOwner(mlir::Operation *operation, mlir::Block *body) {
  while (operation && operation->getBlock() != body)
    operation = operation->getParentOp();
  return operation;
}

mlir::Operation *cloneScheduled(mlir::IRRewriter &rewriter,
                                mlir::Operation *source,
                                mlir::IRMapping &mapping) {
  mlir::Operation *clone = rewriter.clone(*source, mapping);
  clone->removeAttr(kStageAttr);
  clone->removeAttr(kOrderAttr);
  return clone;
}

void copyLoopAttrs(mlir::Operation *source, mlir::Operation *target,
                   bool preserveLevelIdentity) {
  for (mlir::NamedAttribute attribute : source->getAttrs()) {
    // The guard owns the original Level identity, but only the steady-state
    // loop can consume the selected unroll plan.
    if (!mlir::isa<mlir::scf::ForOp>(target) &&
        (attribute.getName() == "weft.riscv.unroll_factor" ||
         attribute.getName() == "weft.riscv.unroll_order"))
      continue;
    if (attribute.getName() != "weft.riscv.schedule" &&
        (preserveLevelIdentity || attribute.getName() != "weft.riscv.level"))
      target->setAttr(attribute.getName(), attribute.getValue());
  }
  if (!preserveLevelIdentity &&
      !target->hasAttr("weft.riscv.direction"))
    if (auto level =
            source->getAttrOfType<riscv::LevelAttr>("weft.riscv.level"))
      target->setAttr("weft.riscv.direction",
                      mlir::StringAttr::get(target->getContext(),
                                            level.getDirection()));
}

class PipelineRISCVLevelsPass
    : public mlir::PassWrapper<PipelineRISCVLevelsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-pipeline-levels";
  }
  llvm::StringRef getDescription() const override {
    return "Expand scheduled physical Levels into SSA-versioned prologue, steady state, and epilogue";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
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
      if (schedule.getPipelineDepth() == 1) {
        loop->removeAttr("weft.riscv.schedule");
        continue;
      }
      if (schedule.getPipelineDepth() != 2 ||
          schedule.getBufferCount() != 2) {
        loop.emitError(
            "physical expander currently implements one-iteration-distance, two-stage pipelines");
        failed = true;
        continue;
      }

      mlir::Block *body = loop.getBody();
      llvm::SmallVector<mlir::Operation *> stageZero;
      llvm::SmallVector<mlir::Operation *> stageOne;
      llvm::DenseSet<int64_t> stageZeroOrders;
      llvm::DenseSet<int64_t> stageOneOrders;
      bool missingStage = false;
      for (mlir::Operation &operation : body->without_terminator()) {
        std::optional<int64_t> stage = stageOf(&operation);
        std::optional<int64_t> order = orderOf(&operation);
        if (!stage || (*stage != 0 && *stage != 1) || !order) {
          operation.emitError(
              "pipeline expander received an operation without a valid scheduled stage/order");
          missingStage = true;
          break;
        }
        llvm::DenseSet<int64_t> &orders =
            *stage == 0 ? stageZeroOrders : stageOneOrders;
        if (!orders.insert(*order).second) {
          operation.emitError(
              "pipeline expander received duplicate order within one stage");
          missingStage = true;
          break;
        }
        (*stage == 0 ? stageZero : stageOne).push_back(&operation);
      }
      if (missingStage || stageZero.empty() || stageOne.empty()) {
        failed = true;
        continue;
      }
      auto byScheduledOrder = [](mlir::Operation *lhs, mlir::Operation *rhs) {
        return *orderOf(lhs) < *orderOf(rhs);
      };
      std::stable_sort(stageZero.begin(), stageZero.end(), byScheduledOrder);
      std::stable_sort(stageOne.begin(), stageOne.end(), byScheduledOrder);

      auto yield = mlir::cast<mlir::scf::YieldOp>(body->getTerminator());
      llvm::DenseSet<mlir::Operation *> stageOneSet(stageOne.begin(),
                                                    stageOne.end());
      bool needsBufferedIV = false;
      for (mlir::Operation *operation : stageOne)
        needsBufferedIV |= recursivelyUses(operation, loop.getInductionVar());
      llvm::SmallVector<mlir::Value> crossStageValues;
      llvm::DenseSet<mlir::Value> seenCrossStage;
      for (mlir::Operation *operation : stageZero)
        for (mlir::Value result : operation->getResults()) {
          bool crosses = false;
          for (mlir::OpOperand &use : result.getUses())
            if (mlir::Operation *owner = bodyOwner(use.getOwner(), body);
                owner == yield.getOperation() || stageOneSet.contains(owner)) {
              crosses = true;
              break;
            }
          if (crosses && seenCrossStage.insert(result).second)
            crossStageValues.push_back(result);
        }
      if (crossStageValues.empty()) {
        loop.emitError(
            "pipeline schedule has no SSA value crossing from producer to consumer stage");
        failed = true;
        continue;
      }

      rewriter.setInsertionPoint(loop);
      mlir::Value nonEmpty = rewriter.create<mlir::arith::CmpIOp>(
          loop.getLoc(), mlir::arith::CmpIPredicate::ult,
          loop.getLowerBound(), loop.getUpperBound());
      auto pipeline = rewriter.create<mlir::scf::IfOp>(
          loop.getLoc(), loop.getResultTypes(), nonEmpty, true);
      copyLoopAttrs(loop, pipeline, true);

      rewriter.setInsertionPointToStart(&pipeline.getThenRegion().front());
      mlir::IRMapping prologue;
      prologue.map(loop.getInductionVar(), loop.getLowerBound());
      for (auto [argument, initial] :
           llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs()))
        prologue.map(argument, initial);
      for (mlir::Operation *operation : stageZero)
        cloneScheduled(rewriter, operation, prologue);

      llvm::SmallVector<mlir::Value> steadyInitial(loop.getInitArgs());
      if (needsBufferedIV)
        steadyInitial.push_back(loop.getLowerBound());
      for (mlir::Value value : crossStageValues)
        steadyInitial.push_back(prologue.lookup(value));

      mlir::Value steadyLower = rewriter.create<mlir::arith::AddIOp>(
          loop.getLoc(), loop.getLowerBound(), loop.getStep());
      auto steady = rewriter.create<mlir::scf::ForOp>(
          loop.getLoc(), steadyLower, loop.getUpperBound(), loop.getStep(),
          steadyInitial);
      copyLoopAttrs(loop, steady, false);
      if (!steady.getBody()->empty())
        if (auto oldYield =
                mlir::dyn_cast<mlir::scf::YieldOp>(steady.getBody()->back()))
          rewriter.eraseOp(oldYield);
      rewriter.setInsertionPointToStart(steady.getBody());

      const unsigned carryCount = loop.getInitArgs().size();
      mlir::IRMapping currentProducer;
      currentProducer.map(loop.getInductionVar(), steady.getInductionVar());
      for (auto [argument, carried] :
           llvm::zip(loop.getRegionIterArgs(),
                     steady.getRegionIterArgs().take_front(carryCount)))
        currentProducer.map(argument, carried);
      for (mlir::Operation *operation : stageZero)
        cloneScheduled(rewriter, operation, currentProducer);

      mlir::IRMapping previousConsumer;
      if (needsBufferedIV)
        previousConsumer.map(loop.getInductionVar(),
                             steady.getRegionIterArg(carryCount));
      for (auto [argument, carried] :
           llvm::zip(loop.getRegionIterArgs(),
                     steady.getRegionIterArgs().take_front(carryCount)))
        previousConsumer.map(argument, carried);
      for (auto [index, value] : llvm::enumerate(crossStageValues))
        previousConsumer.map(value,
                             steady.getRegionIterArg(
                                 carryCount + (needsBufferedIV ? 1 : 0) + index));
      for (mlir::Operation *operation : stageOne)
        cloneScheduled(rewriter, operation, previousConsumer);

      llvm::SmallVector<mlir::Value> steadyYield;
      for (mlir::Value value : yield.getResults())
        steadyYield.push_back(previousConsumer.lookupOrDefault(value));
      if (needsBufferedIV)
        steadyYield.push_back(steady.getInductionVar());
      for (mlir::Value value : crossStageValues)
        steadyYield.push_back(currentProducer.lookup(value));
      rewriter.setInsertionPointToEnd(steady.getBody());
      rewriter.create<mlir::scf::YieldOp>(loop.getLoc(), steadyYield);

      rewriter.setInsertionPointAfter(steady);
      mlir::IRMapping epilogue;
      if (needsBufferedIV)
        epilogue.map(loop.getInductionVar(), steady.getResult(carryCount));
      for (auto [argument, carried] :
           llvm::zip(loop.getRegionIterArgs(),
                     steady.getResults().take_front(carryCount)))
        epilogue.map(argument, carried);
      for (auto [index, value] : llvm::enumerate(crossStageValues))
        epilogue.map(value,
                     steady.getResult(carryCount +
                                      (needsBufferedIV ? 1 : 0) + index));
      for (mlir::Operation *operation : stageOne)
        cloneScheduled(rewriter, operation, epilogue);
      llvm::SmallVector<mlir::Value> finalValues;
      for (mlir::Value value : yield.getResults())
        finalValues.push_back(epilogue.lookupOrDefault(value));
      rewriter.create<mlir::scf::YieldOp>(loop.getLoc(), finalValues);

      rewriter.setInsertionPointToStart(&pipeline.getElseRegion().front());
      rewriter.create<mlir::scf::YieldOp>(loop.getLoc(), loop.getInitArgs());
      loop.getResults().replaceAllUsesWith(pipeline.getResults());
      rewriter.eraseOp(loop);
    }

    getOperation().walk([&](mlir::Operation *operation) {
      if (operation->hasAttr("weft.riscv.schedule") ||
          operation->hasAttr(kStageAttr) || operation->hasAttr(kOrderAttr)) {
        operation->emitError(
            "physical schedule survived without structural pipeline expansion");
        failed = true;
      }
      if (operation->hasAttr("schedule")) {
        operation->emitError(
            "operation-local schedule survived without an explicit physical loop");
        failed = true;
      }
    });
    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createPipelineRISCVLevelsPass() {
  return std::make_unique<PipelineRISCVLevelsPass>();
}
