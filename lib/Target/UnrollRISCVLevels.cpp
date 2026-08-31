#include "Weft/Target/RISCVPasses.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
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
    bool changed = false;
    for (mlir::scf::ForOp loop : loops) {
      auto factor = loop->getAttrOfType<mlir::IntegerAttr>(
          "weft.riscv.unroll_factor");
      loop->removeAttr("weft.riscv.unroll_factor");
      if (mlir::failed(mlir::loopUnrollByFactor(loop, factor.getInt()))) {
        loop.emitError("selected Level unroll factor is not structurally legal");
        failed = true;
      } else {
        changed = true;
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
