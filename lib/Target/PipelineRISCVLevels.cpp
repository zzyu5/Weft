#include "Weft/Target/RISCVPasses.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include <memory>

using namespace weft;

namespace {

bool isWindowLoad(mlir::Operation *operation) {
  return mlir::isa<riscv::RVVGroupedMacLoadOp,
                   riscv::RVVEncodedDotLoadOp>(operation);
}

bool isWindowStep(mlir::Operation *operation) {
  return mlir::isa<riscv::RVVGroupedMacStepOp,
                   riscv::RVVEncodedDotStepOp>(operation);
}

mlir::Value windowIndex(mlir::Operation *operation) {
  if (auto load = mlir::dyn_cast<riscv::RVVGroupedMacLoadOp>(operation))
    return load.getGroupIndex();
  if (auto load = mlir::dyn_cast<riscv::RVVEncodedDotLoadOp>(operation))
    return load.getGroupIndex();
  return {};
}

mlir::Value cloneLoad(mlir::IRRewriter &rewriter, mlir::Operation *source,
                      mlir::Value index) {
  mlir::IRMapping mapping;
  mapping.map(windowIndex(source), index);
  return rewriter.clone(*source, mapping)->getResult(0);
}

mlir::Value cloneStep(mlir::IRRewriter &rewriter, mlir::Operation *source,
                      mlir::Value window, mlir::Value accumulator) {
  mlir::IRMapping mapping;
  mapping.map(source->getOperand(0), window);
  mapping.map(source->getOperand(1), accumulator);
  return rewriter.clone(*source, mapping)->getResult(0);
}

class PipelineRISCVLevelsPass
    : public mlir::PassWrapper<PipelineRISCVLevelsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-pipeline-levels";
  }
  llvm::StringRef getDescription() const override {
    return "Materialize explicit prologue/steady/epilogue physical window pipelines";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    bool failed = false;
    llvm::SmallVector<mlir::scf::ForOp> loops;
    getOperation().walk([&](mlir::scf::ForOp loop) {
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
          schedule.getBufferCount() != 2 || loop.getInitArgs().size() != 1) {
        loop.emitError(
            "physical scheduler currently supports a two-stage, two-buffer, single-carry pipeline");
        failed = true;
        continue;
      }
      mlir::Block *body = loop.getBody();
      llvm::SmallVector<mlir::Operation *> operations;
      for (mlir::Operation &operation : body->without_terminator())
        operations.push_back(&operation);
      if (operations.size() != 2 || !isWindowLoad(operations[0]) ||
          !isWindowStep(operations[1]) ||
          operations[1]->getOperand(0) != operations[0]->getResult(0) ||
          operations[1]->getOperand(1) != loop.getRegionIterArg(0)) {
        loop.emitError(
            "pipeline schedule has no explicit load-window/compute-window local cluster");
        failed = true;
        continue;
      }

      mlir::Operation *load = operations[0];
      mlir::Operation *step = operations[1];
      rewriter.setInsertionPoint(loop);
      mlir::Value nonEmpty = rewriter.create<mlir::arith::CmpIOp>(
          loop.getLoc(), mlir::arith::CmpIPredicate::ult,
          loop.getLowerBound(), loop.getUpperBound());
      auto pipeline = rewriter.create<mlir::scf::IfOp>(
          loop.getLoc(), mlir::TypeRange{loop.getResult(0).getType()}, nonEmpty,
          true);
      rewriter.setInsertionPointToStart(&pipeline.getThenRegion().front());
      mlir::Value firstWindow = cloneLoad(rewriter, load, loop.getLowerBound());
      mlir::Value steadyLower = rewriter.create<mlir::arith::AddIOp>(
          loop.getLoc(), loop.getLowerBound(), loop.getStep());
      auto steady = rewriter.create<mlir::scf::ForOp>(
          loop.getLoc(), steadyLower, loop.getUpperBound(), loop.getStep(),
          mlir::ValueRange{loop.getInitArgs().front(), firstWindow});
      steady->setAttr("weft.riscv.direction",
                      rewriter.getStringAttr("ascending"));
      rewriter.setInsertionPointToStart(steady.getBody());
      mlir::Value nextWindow =
          cloneLoad(rewriter, load, steady.getInductionVar());
      mlir::Value nextAccumulator = cloneStep(
          rewriter, step, steady.getRegionIterArg(1),
          steady.getRegionIterArg(0));
      rewriter.setInsertionPointToEnd(steady.getBody());
      rewriter.create<mlir::scf::YieldOp>(
          loop.getLoc(), mlir::ValueRange{nextAccumulator, nextWindow});

      rewriter.setInsertionPointAfter(steady);
      mlir::Value finalAccumulator =
          cloneStep(rewriter, step, steady.getResult(1), steady.getResult(0));
      rewriter.create<mlir::scf::YieldOp>(loop.getLoc(), finalAccumulator);
      rewriter.setInsertionPointToStart(&pipeline.getElseRegion().front());
      rewriter.create<mlir::scf::YieldOp>(loop.getLoc(), loop.getInitArgs().front());
      loop.getResult(0).replaceAllUsesWith(pipeline.getResult(0));
      rewriter.eraseOp(loop);
    }

    // Schedule attributes are inputs to structural rewriting, not terminal
    // authority. Operations without a physical loop have no local pipeline
    // dimension; consuming their attribute is exact rather than a downgrade.
    getOperation().walk([&](mlir::Operation *operation) {
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
