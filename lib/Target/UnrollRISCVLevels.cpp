#include "Weft/Target/RISCVPasses.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <memory>
#include <tuple>

using namespace weft;

namespace {

constexpr llvm::StringLiteral kIssueOrderAttr =
    "weft.riscv.issue_operation_order";
constexpr llvm::StringLiteral kIssueIterationAttr =
    "weft.riscv.issue_iteration";

bool isReadOrPure(mlir::Operation *operation) {
  if (mlir::isMemoryEffectFree(operation))
    return true;
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return llvm::all_of(instances, [](const auto &instance) {
    return mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
  });
}

mlir::LogicalResult scheduleOperationMajor(
    mlir::Operation *diagnosticOwner, mlir::Block *body,
    llvm::ArrayRef<mlir::Operation *> operations,
    mlir::Operation *insertionAnchor) {
  llvm::DenseMap<mlir::Operation *, unsigned> originalPosition;
  for (auto [position, operation] : llvm::enumerate(operations)) {
    if (operation->getNumRegions() != 0 || !isReadOrPure(operation))
      return diagnosticOwner->emitError(
          "operation-major issue unroll requires one flat read-only or pure body");
    originalPosition[operation] = position;
  }
  if (operations.empty())
    return diagnosticOwner->emitError(
        "operation-major issue unroll produced no schedulable operations");

  llvm::DenseMap<mlir::Operation *, unsigned> indegree;
  llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Operation *>> users;
  for (mlir::Operation *operation : operations)
    indegree[operation] = 0;
  for (mlir::Operation *operation : operations) {
    llvm::DenseSet<mlir::Operation *> dependencies;
    for (mlir::Value operand : operation->getOperands()) {
      mlir::Operation *definition = operand.getDefiningOp();
      if (definition && definition->getBlock() == body &&
          indegree.count(definition))
        dependencies.insert(definition);
    }
    indegree[operation] = dependencies.size();
    for (mlir::Operation *dependency : dependencies)
      users[dependency].push_back(operation);
  }

  auto scheduleKey = [&](mlir::Operation *operation) {
    auto order = operation->getAttrOfType<mlir::IntegerAttr>(kIssueOrderAttr);
    auto iteration =
        operation->getAttrOfType<mlir::IntegerAttr>(kIssueIterationAttr);
    // The SCF unroller may insert induction-offset arithmetic that is not a
    // clone of a source body operation.  It has no side effects and must be
    // available before the first cloned operation that consumes it.
    const int64_t operationOrder = order ? order.getInt() : -1;
    const int64_t issueIteration = iteration ? iteration.getInt() : -1;
    return std::tuple<int64_t, int64_t, unsigned>(
        operationOrder, issueIteration, originalPosition.lookup(operation));
  };

  llvm::SmallVector<mlir::Operation *> ready;
  for (mlir::Operation *operation : operations)
    if (indegree.lookup(operation) == 0)
      ready.push_back(operation);
  llvm::SmallVector<mlir::Operation *> scheduled;
  scheduled.reserve(operations.size());
  while (!ready.empty()) {
    auto selected = llvm::min_element(
        ready, [&](mlir::Operation *lhs, mlir::Operation *rhs) {
          return scheduleKey(lhs) < scheduleKey(rhs);
        });
    mlir::Operation *operation = *selected;
    ready.erase(selected);
    scheduled.push_back(operation);
    for (mlir::Operation *user : users[operation]) {
      unsigned &degree = indegree[user];
      if (--degree == 0)
        ready.push_back(user);
    }
  }
  if (scheduled.size() != operations.size())
    return diagnosticOwner->emitError(
        "operation-major issue unroll has a cyclic physical dependency");

  for (mlir::Operation *operation : scheduled) {
    operation->removeAttr(kIssueOrderAttr);
    operation->removeAttr(kIssueIterationAttr);
    operation->moveBefore(insertionAnchor);
  }
  return mlir::success();
}

mlir::LogicalResult scheduleOperationMajor(mlir::scf::ForOp loop) {
  llvm::SmallVector<mlir::Operation *> operations;
  for (mlir::Operation &operation : loop.getBody()->without_terminator())
    operations.push_back(&operation);
  return scheduleOperationMajor(loop, loop.getBody(), operations,
                                loop.getBody()->getTerminator());
}

mlir::LogicalResult scheduleFullyUnrolledOperationMajor(
    mlir::Operation *diagnosticOwner, mlir::Block *body) {
  mlir::Operation *first = nullptr;
  mlir::Operation *last = nullptr;
  for (mlir::Operation &operation : *body) {
    if (!operation.hasAttr(kIssueIterationAttr))
      continue;
    if (!first)
      first = &operation;
    last = &operation;
  }
  if (!first || !last || !last->getNextNode())
    return diagnosticOwner->emitError(
        "fully unrolled operation-major issue has no closed cloned region");
  llvm::SmallVector<mlir::Operation *> operations;
  for (mlir::Operation *operation = first;;
       operation = operation->getNextNode()) {
    operations.push_back(operation);
    if (operation == last)
      break;
  }
  return scheduleOperationMajor(diagnosticOwner, body, operations,
                                last->getNextNode());
}

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
    bool changed = false;
    for (mlir::scf::ForOp loop : loops) {
      auto factor = loop->getAttrOfType<mlir::IntegerAttr>(
          "weft.riscv.unroll_factor");
      auto order = loop->getAttrOfType<mlir::StringAttr>(
          "weft.riscv.unroll_order");
      loop->removeAttr("weft.riscv.unroll_factor");
      loop->removeAttr("weft.riscv.unroll_order");
      const bool operationMajor = order && order.getValue() == "operation-major";
      if (order && !operationMajor) {
        loop.emitError("selected issue unroll has an unknown operation order");
        failed = true;
        continue;
      }
      if (operationMajor) {
        int64_t sourceOrder = 0;
        for (mlir::Operation &operation : loop.getBody()->without_terminator())
          operation.setAttr(kIssueOrderAttr,
                            mlir::IntegerAttr::get(
                                mlir::IntegerType::get(&getContext(), 64),
                                sourceOrder++));
      }
      mlir::Block *unrollParent = loop->getBlock();
      auto unrolled = mlir::loopUnrollByFactor(
          loop, factor.getInt(),
          operationMajor
              ? mlir::function_ref<void(unsigned, mlir::Operation *,
                                        mlir::OpBuilder)>(
                    [&](unsigned iteration, mlir::Operation *operation,
                        mlir::OpBuilder builder) {
                      operation->setAttr(
                          kIssueIterationAttr,
                          builder.getI64IntegerAttr(iteration));
                    })
              : nullptr);
      if (mlir::failed(unrolled)) {
        loop.emitError("selected Level unroll factor is not structurally legal");
        failed = true;
      } else {
        changed = true;
        if (operationMajor) {
          auto scheduleResult =
              (*unrolled).mainLoopOp
                  ? scheduleOperationMajor(*(*unrolled).mainLoopOp)
                  : scheduleFullyUnrolledOperationMajor(getOperation(),
                                                        unrollParent);
          if (mlir::failed(scheduleResult))
            failed = true;
          if ((*unrolled).epilogueLoopOp)
            (*unrolled).epilogueLoopOp->walk([&](mlir::Operation *operation) {
              operation->removeAttr(kIssueOrderAttr);
              operation->removeAttr(kIssueIterationAttr);
            });
        }
      }
    }
    // MLIR's generic unroller preserves operation attributes verbatim.  A
    // partial birth is a target-local SSA lifetime identity, so cloned partial
    // definitions in one owner domain must receive fresh identities just as
    // cloned SSA definitions receive fresh results.
    if (changed && !failed) {
      llvm::DenseMap<int64_t, int64_t> maximumBirth;
      getOperation().walk([&](mlir::Operation *operation) {
        if (auto partial = mlir::dyn_cast<riscv::RVVPartialSetOp>(operation))
          maximumBirth[partial.getOwnerDomainId()] =
              std::max(maximumBirth.lookup(partial.getOwnerDomainId()),
                       static_cast<int64_t>(partial.getBirthId()));
        if (auto capture =
                mlir::dyn_cast<riscv::RVVPartialCaptureOp>(operation))
          maximumBirth[capture.getOwnerDomainId()] =
              std::max(maximumBirth.lookup(capture.getOwnerDomainId()),
                       static_cast<int64_t>(capture.getBirthId()));
        if (auto collect =
                mlir::dyn_cast<riscv::RVVPartialCollectOp>(operation))
          maximumBirth[collect.getOwnerDomainId()] =
              std::max(maximumBirth.lookup(collect.getOwnerDomainId()),
                       static_cast<int64_t>(collect.getBirthId()));
      });
      llvm::DenseSet<std::pair<int64_t, int64_t>> seen;
      mlir::Builder builder(&getContext());
      auto freshen = [&](mlir::Operation *operation, int64_t owner,
                         int64_t birth) {
        if (seen.insert({owner, birth}).second)
          return;
        const int64_t replacement = ++maximumBirth[owner];
        operation->setAttr("birth_id",
                           builder.getI64IntegerAttr(replacement));
        seen.insert({owner, replacement});
      };
      getOperation().walk([&](mlir::Operation *operation) {
        if (auto partial = mlir::dyn_cast<riscv::RVVPartialSetOp>(operation))
          freshen(operation, partial.getOwnerDomainId(), partial.getBirthId());
        if (auto capture =
                mlir::dyn_cast<riscv::RVVPartialCaptureOp>(operation))
          freshen(operation, capture.getOwnerDomainId(), capture.getBirthId());
        if (auto collect =
                mlir::dyn_cast<riscv::RVVPartialCollectOp>(operation))
          freshen(operation, collect.getOwnerDomainId(), collect.getBirthId());
      });
    }
    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createUnrollRISCVLevelsPass() {
  return std::make_unique<UnrollRISCVLevelsPass>();
}
