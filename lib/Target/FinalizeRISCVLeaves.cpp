#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include <memory>
#include <string>

using namespace weft;

namespace {

class FinalizeRISCVLeavesPass
    : public mlir::PassWrapper<FinalizeRISCVLeavesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-finalize-leaves";
  }
  llvm::StringRef getDescription() const override {
    return "Finalize exact terminal leaf instructions from solved physical types";
  }

  void runOnOperation() override {
    mlir::Builder builder(&getContext());
    bool failed = false;
    getOperation().walk([&](mlir::Operation *operation) {
      // Layout canonicalization, scheduling, and partial materialization can
      // all create conversions after composite lowering.  Close conversions
      // here, after the last physical rewrite, so every conversion in the
      // final program receives its leaf from the same typed ConversionAttr.
      if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(operation)) {
        llvm::StringRef kind = conversion.getConversion().getKind();
        llvm::StringRef instruction =
            kind == "splat"          ? "rvv.splat"
            : kind == "extract"      ? "rvv.extract"
            : kind == "local_load"   ? "rvv.local-load"
            : kind == "local_store"  ? "rvv.local-store"
            : kind == "tuple"        ? "rvv.tuple-convert"
            : kind == "register_to_lane" ? "rvv.register-to-lane"
            : kind == "time_to_lane" ? "rvv.time-to-lane"
            : kind == "lane_to_register" ? "rvv.lane-to-register"
            : kind == "reshape"      ? "rvv.layout-reshape"
                                      : llvm::StringRef();
        if (instruction.empty()) {
          conversion.emitError()
              << "no terminal leaf implements typed layout conversion kind '"
              << kind << "'";
          failed = true;
          return;
        }
        riscv::LeafAttr oldLeaf = conversion.getLeaf();
        conversion.setLeafAttr(riscv_internal::leaf(
            builder, "rvv", "layout-conversion", instruction, instruction,
            oldLeaf.getOperandGroups(), oldLeaf.getResultGroups(),
            conversion.getConversion().getTemporaryGroups(), 0));
        conversion->removeAttr("implementation");
        return;
      }
      auto implementation =
          operation->getAttrOfType<riscv::ImplementationAttr>("implementation");
      if (!implementation)
        return;
      std::string instruction;
      int64_t temporaries = 0;
      llvm::StringRef engine = implementation.getEngine();
      if (!operation->getResultTypes().empty() &&
          !mlir::isa<riscv::ReduceOp>(operation)) {
        llvm::StringRef carrier =
            riscv_internal::layoutOf(operation->getResult(0).getType())
                ? riscv_internal::layoutOf(operation->getResult(0).getType())
                      .getCarrier()
                : llvm::StringRef("scalar");
        if ((implementation.getEngine() == "rvv" && carrier != "rvv") ||
            (implementation.getEngine() == "scalar" && carrier != "scalar")) {
          operation->emitError()
              << "selected " << operation->getName() << " engine "
              << implementation.getEngine()
              << " disagrees with propagated result carrier " << carrier;
          failed = true;
          return;
        }
      }
      if (auto iota = mlir::dyn_cast<riscv::IotaOp>(operation)) {
        instruction = riscv_internal::terminalInstruction(iota);
      } else if (auto unary = mlir::dyn_cast<riscv::UnaryOp>(operation)) {
        instruction = riscv_internal::terminalInstruction(unary);
        temporaries = instruction == "rvv.exp-approx-f32" ? 8 : 0;
      } else if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(operation)) {
        instruction = riscv_internal::terminalInstruction(binary);
      } else if (auto compare = mlir::dyn_cast<riscv::CompareOp>(operation)) {
        instruction = riscv_internal::terminalInstruction(compare);
      } else if (mlir::isa<riscv::CastOp, riscv::NarrowOp,
                           riscv::WidenOp>(operation)) {
        instruction = riscv_internal::terminalInstruction(operation);
      } else if (auto reduce = mlir::dyn_cast<riscv::ReduceOp>(operation)) {
        instruction = riscv_internal::terminalInstruction(reduce);
      } else if (mlir::isa<riscv::Fold2Op>(operation)) {
        instruction = riscv_internal::terminalInstruction(operation);
      } else if (auto update = mlir::dyn_cast<riscv::UpdateOp>(operation)) {
        auto layout = riscv_internal::layoutOf(update.getResult().getType());
        if (layout && layout.getCarrier() == "local") {
          instruction = "local.update";
          engine = "scalar";
        } else {
          instruction = implementation.getOperation().str();
        }
      } else if (mlir::isa<riscv::LocalLoadOp,
                           riscv::LocalStoreOp>(operation)) {
        instruction = implementation.getOperation().str();
        engine = "transfer";
      } else {
        operation->emitError(
            "structural implementation was not lowered to a terminal physical operation");
        failed = true;
        return;
      }
      if (instruction.empty()) {
        operation->emitError()
            << "no exact terminal instruction implements the solved physical representation for "
            << operation->getName() << " with implementation " << implementation
            << ", operands " << operation->getOperandTypes() << ", results "
            << operation->getResultTypes();
        failed = true;
        return;
      }
      operation->setAttr(
          "leaf", riscv_internal::leaf(
                      builder, engine, implementation.getFamily(), instruction,
                      instruction, 0, 0, temporaries, 0, "none", "agnostic",
                      implementation.getParameters().asArrayRef()));
      operation->removeAttr("implementation");
    });
    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createFinalizeRISCVLeavesPass() {
  return std::make_unique<FinalizeRISCVLeavesPass>();
}
