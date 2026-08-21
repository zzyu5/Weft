#include "Weft/Target/RISCVPasses.h"

#include "RISCVPlanningSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <limits>
#include <string>

using namespace weft;

namespace {

class SelectRISCVWinnerPass final
    : public mlir::PassWrapper<SelectRISCVWinnerPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SelectRISCVWinnerPass)

  llvm::StringRef getArgument() const final {
    return "weft-riscv-select-winner";
  }
  llvm::StringRef getDescription() const final {
    return "select the cheapest complete forward-physicalized candidate";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());
    llvm::StringMap<riscv::ProblemOp> winners;
    llvm::StringMap<int64_t> costs;
    llvm::StringMap<bool> seen;
    llvm::StringMap<llvm::SmallVector<std::string>> invalidReasons;

    for (riscv::ProblemOp problem : module.getOps<riscv::ProblemOp>()) {
      std::string kernel = problem.getKernel().str();
      seen[kernel] = true;
      if (problem.getStage() == "invalid") {
        invalidReasons[kernel].push_back(
            riscv_internal::string(problem.getResources(), "invalid_reason")
                .value_or("candidate invalid without a reason")
                .str());
        continue;
      }
      if (problem.getStage() != "resources") {
        problem.emitError("winner selection requires resource-checked candidates");
        signalPassFailure();
        return;
      }
      int64_t cost =
          riscv_internal::integer(problem.getResources(), "cost")
              .value_or(std::numeric_limits<int64_t>::max());
      if (!winners.contains(kernel) || cost < costs.lookup(kernel)) {
        winners[kernel] = problem;
        costs[kernel] = cost;
      }
    }

    for (const auto &entry : seen)
      if (!winners.contains(entry.getKey())) {
        mlir::InFlightDiagnostic diagnostic = module.emitError()
            << "RISC-V physicalization has no legal candidate for @"
            << entry.getKey();
        for (const std::string &reason : invalidReasons.lookup(entry.getKey()))
          diagnostic << "\n  - " << reason;
        signalPassFailure();
        return;
      }

    builder.setInsertionPointToEnd(module.getBody());
    unsigned assignmentIndex = 0;
    for (const auto &entry : winners) {
      riscv::ProblemOp winner = entry.getValue();
      mlir::OperationState state(winner.getLoc(),
                                 riscv::AssignmentOp::getOperationName());
      state.addAttribute("sym_name", builder.getStringAttr(
                                         "__weft_assignment_" +
                                         std::to_string(assignmentIndex++)));
      state.addAttribute("kernel", winner.getKernelAttr());
      state.addAttribute("target", winner.getTarget());
      state.addAttribute("candidate", winner.getCandidate());
      state.addAttribute("values", winner.getValues());
      state.addAttribute("operations", winner.getOperations());
      state.addAttribute("resources", winner.getResources());
      state.addAttribute("status", builder.getStringAttr("complete"));
      builder.create(state);
    }
    for (riscv::ProblemOp problem :
         llvm::make_early_inc_range(module.getOps<riscv::ProblemOp>()))
      problem.erase();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createSelectRISCVWinnerPass() {
  return std::make_unique<SelectRISCVWinnerPass>();
}
