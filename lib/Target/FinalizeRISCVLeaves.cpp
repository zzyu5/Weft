#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

using namespace weft;

namespace {

bool isIntegerZero(mlir::Value value, unsigned &remaining) {
  if (!remaining ||
      !mlir::isa<mlir::IntegerType>(
          riscv_internal::logicalElement(value.getType())))
    return false;
  --remaining;
  mlir::Attribute attribute;
  if (auto constant = value.getDefiningOp<riscv::ConstantOp>())
    attribute = constant.getValue();
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    attribute = constant.getValue();
  if (auto integer = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attribute))
    return integer.getValue().isZero();
  auto *producer = value.getDefiningOp();
  if (!producer)
    return false;
  if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(producer))
    return conversion.getConversion().getEffect() == "pure" &&
           isIntegerZero(conversion.getInput(), remaining);
  if (mlir::isa<riscv::CastOp, riscv::NarrowOp, riscv::WidenOp,
                riscv::RVVSplatOp, riscv::RVVAxisBroadcastOp>(producer))
    return isIntegerZero(producer->getOperand(0), remaining);
  if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(producer))
    if (binary.getKind() == "mul")
      return isIntegerZero(binary.getLhs(), remaining) ||
             isIntegerZero(binary.getRhs(), remaining);
  return false;
}

void foldIntegerIdentities(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  module.walk([&](riscv::BinaryOp operation) {
    auto kernel = operation->getParentOfType<riscv::KernelOp>();
    auto result = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
    if (!kernel || kernel.getResourcesMaterialized() ||
        !result || result.getLayout().getCarrier() != "rvv" ||
        (operation.getKind() != "add" && operation.getKind() != "sub"))
      return;
    auto tryIdentity = [&](mlir::Value input, mlir::Value zero) {
      // Equality includes all axes, validity, and carrier factors. Removing an
      // integer identity must not silently remove an author's broadcast axis.
      unsigned remaining = 32;
      if (input.getType() != operation.getResult().getType() ||
          !isIntegerZero(zero, remaining))
        return false;
      rewriter.replaceOp(operation, input);
      return true;
    };
    if (tryIdentity(operation.getLhs(), operation.getRhs()))
      return;
    if (operation.getKind() == "add")
      tryIdentity(operation.getRhs(), operation.getLhs());
  });
}

void foldUnsignedPowerOfTwoArithmetic(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  unsigned quotients = 0, remainders = 0;
  llvm::SmallVector<riscv::BinaryOp> candidates;
  module.walk([&](riscv::BinaryOp op) { candidates.push_back(op); });
  for (auto operation : candidates) {
    auto kernel = operation->getParentOfType<riscv::KernelOp>();
    auto result = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
    auto element = result
        ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
        : mlir::IntegerType();
    if (!kernel || kernel.getResourcesMaterialized() || !element ||
        !element.isUnsigned() || result.getLayout().getCarrier() != "rvv" ||
        operation.getLhs().getType() != result ||
        operation.getRhs().getType() != element ||
        (operation.getKind() != "div" && operation.getKind() != "mod"))
      continue;
    mlir::Attribute attribute;
    if (auto constant = operation.getRhs().getDefiningOp<riscv::ConstantOp>())
      attribute = constant.getValue();
    if (auto constant = operation.getRhs().getDefiningOp<mlir::arith::ConstantOp>())
      attribute = constant.getValue();
    auto divisor = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attribute);
    if (!divisor || !divisor.getValue().isPowerOf2())
      continue;
    const bool quotient = operation.getKind() == "div";
    llvm::APInt immediate = quotient
        ? llvm::APInt(element.getWidth(), divisor.getValue().logBase2())
        : divisor.getValue() - 1;
    rewriter.setInsertionPoint(operation);
    auto constant = rewriter.create<riscv::ConstantOp>(
        operation.getLoc(), element, rewriter.getIntegerAttr(element, immediate));
    llvm::StringRef instruction = quotient ? "rvv.vsrl.vx" : "rvv.vand.vx";
    auto replacement = rewriter.create<riscv::BinaryOp>(
        operation.getLoc(), result, operation.getLhs(), constant.getResult(),
        quotient ? "shr" : "and",
        riscv_internal::leaf(
            rewriter, "rvv", "pointwise", instruction, instruction,
            result.getLayout().getRegisterGroups(),
            result.getLayout().getRegisterGroups(), 0, 0, "none", "agnostic"));
    riscv_internal::copyOrigin(operation, replacement);
    rewriter.replaceOp(operation, replacement.getResult());
    quotient ? ++quotients : ++remainders;
  }
  if (quotients || remainders)
    llvm::errs() << "weft-constant-div: power-of-two-quotients=" << quotients
                 << " power-of-two-remainders=" << remainders << '\n';
}

void selectUnsignedMultiplyHigh(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  llvm::SmallVector<riscv::NarrowOp> candidates;
  module.walk([&](riscv::NarrowOp op) { candidates.push_back(op); });
  for (auto narrow : candidates) {
    auto kernel = narrow->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized() || narrow.getSaturate() ||
        narrow.getRounding() != "rtz")
      continue;
    auto shift = narrow.getInput().getDefiningOp<riscv::BinaryOp>();
    if (!shift || shift.getKind() != "shr" ||
        !shift.getResult().hasOneUse())
      continue;
    auto multiply =
        shift.getLhs().getDefiningOp<riscv::RVVWidenScalarMultiplyOp>();
    if (!multiply || !multiply.getResult().hasOneUse())
      continue;
    auto input = multiply.getLhs().getType();
    auto element = mlir::dyn_cast<mlir::IntegerType>(input.getElementType());
    mlir::Attribute amount;
    if (auto constant = shift.getRhs().getDefiningOp<riscv::ConstantOp>())
      amount = constant.getValue();
    if (auto constant = shift.getRhs().getDefiningOp<mlir::arith::ConstantOp>())
      amount = constant.getValue();
    auto bits = mlir::dyn_cast_or_null<mlir::IntegerAttr>(amount);
    if (!element || !element.isUnsigned() ||
        (element.getWidth() != 8 && element.getWidth() != 16) ||
        narrow.getResult().getType() != input ||
        shift.getResult().getType() != multiply.getResult().getType() ||
        multiply.getLeaf().getInstruction() != "rvv.vwmulu.vx" ||
        !bits || bits.getValue().getLimitedValue() != element.getWidth())
      continue;
    // The doubled-width unsigned product cannot overflow; taking its upper
    // half needs neither rounding nor saturation. The preceding byte wrap,
    // every logical axis, and every later accumulation remain unchanged.
    rewriter.setInsertionPoint(narrow);
    auto high = rewriter.create<riscv::RVVMultiplyHighScalarOp>(
        narrow.getLoc(), input, multiply.getLhs(), multiply.getRhs(),
        riscv_internal::leaf(
            rewriter, "rvv", "multiply-high-scalar", "rvv.vmulhu.vx",
            "rvv.vmulhu.vx", input.getLayout().getRegisterGroups(),
            input.getLayout().getRegisterGroups()));
    riscv_internal::copyOrigin(narrow, high);
    rewriter.replaceOp(narrow, high.getResult());
    rewriter.eraseOp(shift);
    rewriter.eraseOp(multiply);
  }
}

mlir::LogicalResult materializeFieldReads(mlir::ModuleOp module) {
  mlir::IRRewriter rewriter(module.getContext());
  llvm::SmallVector<riscv::ConvertLayoutOp> candidates;
  module.walk([&](riscv::ConvertLayoutOp op) { candidates.push_back(op); });
  for (auto conversion : candidates) {
    auto kernel = conversion->getParentOfType<riscv::KernelOp>();
    auto kind = conversion.getConversion().getKind();
    auto input = conversion.getInput();
    if (!kernel || kernel.getResourcesMaterialized() ||
        !conversion.getSourceAccess())
      continue;
    // Only storage projections are reads. A computed numeric SSA value must
    // retain its explicit representation conversion, never reload its source.
    llvm::SmallVector<mlir::Value> indices;
    if (!riscv::fieldReadProjection(input, &indices))
      continue;
    const bool resupply = kind == "local_load" || kind == "time_to_lane";
    auto readType = resupply ? conversion.getResult().getType() : input.getType();
    if (readType.getLayout().getCarrier() != "rvv" &&
        readType.getLayout().getCarrier() != "scalar")
      continue;
    auto temporaries = riscv::fieldReadTemporaryGroups(
        readType, *conversion.getSourceAccess(), indices);
    if (!temporaries)
      return conversion.emitError("encoded read has no closed local resource contract");
    rewriter.setInsertionPoint(conversion);
    auto read = rewriter.create<riscv::FieldReadOp>(
        conversion.getLoc(), readType, input, indices,
        *conversion.getSourceAccess(),
        riscv_internal::leaf(
            rewriter, "transfer", "field-read", "encoded.field-read",
            "encoded.field-read", 0, 0,
            *temporaries, 0, "none", "exact"));
    riscv_internal::copyOrigin(conversion, read);
    if (resupply)
      rewriter.replaceOp(conversion, read.getResult());
    else {
      conversion.getInputMutable().assign(read.getResult());
      conversion.removeSourceAccessAttr();
    }
  }
  llvm::SmallVector<riscv::ExtractOp> extracts;
  module.walk([&](riscv::ExtractOp extract) { extracts.push_back(extract); });
  for (auto extract : extracts) {
    auto kernel = extract->getParentOfType<riscv::KernelOp>();
    llvm::SmallVector<mlir::Value> indices;
    if (!kernel || kernel.getResourcesMaterialized() ||
        extract.getAccess().getForm() != "register" ||
        !riscv::fieldReadProjection(extract.getInput(), &indices))
      continue;
    auto input = extract.getInput();
    auto access = input.getDefiningOp()->getAttrOfType<riscv::AccessAttr>("access");
    auto temporaries = riscv::fieldReadTemporaryGroups(input.getType(), access, indices);
    if (!access || !temporaries)
      return extract.emitError("register extract source has no closed encoded read");
    rewriter.setInsertionPoint(extract);
    auto read = rewriter.create<riscv::FieldReadOp>(
        extract.getLoc(), input.getType(), input, indices, access,
        riscv_internal::leaf(rewriter, "transfer", "field-read",
                             "encoded.field-read", "encoded.field-read",
                             0, 0, *temporaries, 0, "none", "exact"));
    riscv_internal::copyOrigin(input.getDefiningOp(), read);
    extract.getInputMutable().assign(read.getResult());
  }
  return mlir::success();
}

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
    foldIntegerIdentities(getOperation());
    foldUnsignedPowerOfTwoArithmetic(getOperation());
    selectUnsignedMultiplyHigh(getOperation());
    if (mlir::failed(materializeFieldReads(getOperation()))) {
      signalPassFailure();
      return;
    }
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
