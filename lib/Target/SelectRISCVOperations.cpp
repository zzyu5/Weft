#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSwitch.h"

#include <memory>
using namespace weft;

namespace {

bool isShaped(mlir::Type type) {
  return mlir::isa<riscv::ValueType>(type);
}

bool isTrivialShaped(mlir::Type type) {
  auto value = mlir::dyn_cast<riscv::ValueType>(type);
  return value && llvm::all_of(value.getShape().asArrayRef(),
                               [](int64_t extent) { return extent == 1; });
}

unsigned width(mlir::Type type) {
  return riscv_internal::logicalBitWidth(type);
}

bool isFloat(mlir::Type type) {
  return mlir::isa<mlir::FloatType>(riscv_internal::logicalElement(type));
}

bool isUnsigned(mlir::Type type) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(
      riscv_internal::logicalElement(type));
  return integer && integer.isUnsigned();
}

bool isSigned(mlir::Type type) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(
      riscv_internal::logicalElement(type));
  return integer && integer.isSigned();
}

bool supportsRVVType(riscv::TargetAttr target, mlir::Type type) {
  auto value = mlir::dyn_cast<riscv::ValueType>(type);
  if (!value)
    return true;
  mlir::Type element = value.getElementType();
  if (mlir::isa<kernel::EncodingType>(element))
    return true;
  const unsigned sew = std::max<unsigned>(8, width(type));
  if (!target.getHasRVV() || target.getVlenBits() <= 0 ||
      target.getVectorRegisters() <= 0 ||
      !llvm::is_contained(target.getSupportedSEW().asArrayRef(), sew))
    return false;
  auto layout = value.getLayout();
  if (layout.getCarrier() == "rvv" &&
      (layout.getSew() != sew ||
       !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(),
                           layout.getLmulEighths())))
    return false;
  return !element.isF16() || target.getHasVectorF16();
}

bool supportsRVVOperation(mlir::Operation *operation) {
  auto kernel = operation->getParentOfType<riscv::KernelOp>();
  if (!kernel)
    return false;
  for (mlir::Type type : operation->getOperandTypes())
    if (!supportsRVVType(kernel.getTarget(), type))
      return false;
  for (mlir::Type type : operation->getResultTypes())
    if (!supportsRVVType(kernel.getTarget(), type))
      return false;
  return true;
}

bool hasEncodedFieldOrigin(mlir::Value value) {
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (mlir::Operation *definition = value.getDefiningOp()) {
    if (!visited.insert(definition).second)
      return false;
    if (mlir::isa<riscv::FieldOp>(definition))
      return true;
    if (!mlir::isa<riscv::Fold2Op, riscv::CastOp, riscv::NarrowOp,
                   riscv::WidenOp, riscv::ConvertLayoutOp,
                   riscv::MaterializeOp, riscv::RegisterMaterializeOp>(
            definition) ||
        definition->getNumOperands() == 0)
      return false;
    value = definition->getOperand(0);
  }
  return false;
}

riscv::FragmentCapabilityAttr matchingFragment(mlir::Operation *operation,
                                               mlir::Value lhs,
                                               mlir::Value rhs,
                                               mlir::Type result) {
  auto signednessMatches = [](mlir::Type type, llvm::StringRef expected) {
    auto integer =
        mlir::dyn_cast<mlir::IntegerType>(riscv_internal::logicalElement(type));
    return integer && !integer.isSignless() &&
           ((expected == "signed" && integer.isSigned()) ||
            (expected == "unsigned" && integer.isUnsigned()));
  };
  auto kernel = operation->getParentOfType<riscv::KernelOp>();
  for (mlir::Attribute attribute : kernel.getTarget().getFragments()) {
    auto fragment = mlir::cast<riscv::FragmentCapabilityAttr>(attribute);
    auto lhsAxes = riscv_internal::logicalAxes(lhs.getType());
    auto rhsAxes = riscv_internal::logicalAxes(rhs.getType());
    auto resultAxes = riscv_internal::logicalAxes(result);
    if (fragment.getLhsBits() == width(lhs.getType()) &&
        fragment.getRhsBits() == width(rhs.getType()) &&
        signednessMatches(lhs.getType(), fragment.getLhsSignedness()) &&
        signednessMatches(rhs.getType(), fragment.getRhsSignedness()) &&
        fragment.getAccumulatorBits() == width(result) &&
        lhsAxes.size() == 2 && rhsAxes.size() == 2 && resultAxes.size() == 2 &&
        riscv_internal::physicalExtent(lhs, lhsAxes[0]) == fragment.getMFactor() &&
        riscv_internal::physicalExtent(lhs, lhsAxes[1]) == fragment.getKFactor() &&
        riscv_internal::physicalExtent(rhs, rhsAxes[0]) == fragment.getKFactor() &&
        riscv_internal::physicalExtent(rhs, rhsAxes[1]) == fragment.getNFactor() &&
        riscv_internal::physicalExtent(operation->getResult(0), resultAxes[0]) ==
            fragment.getMFactor() &&
        riscv_internal::physicalExtent(operation->getResult(0), resultAxes[1]) ==
            fragment.getNFactor() &&
        lhsAxes[1] == rhsAxes[0] && lhsAxes[0] == resultAxes[0] &&
        rhsAxes[1] == resultAxes[1])
      return fragment;
  }
  return {};
}

riscv::ImplementationAttr scalarImplementation(mlir::Builder &builder,
                                               llvm::StringRef family,
                                               llvm::StringRef operation) {
  return riscv_internal::implementation(builder, "scalar", family, operation);
}

riscv::ImplementationAttr rvvImplementation(
    mlir::Builder &builder, llvm::StringRef family,
    llvm::StringRef operation, llvm::ArrayRef<int64_t> parameters = {}) {
  return riscv_internal::implementation(builder, "rvv", family, operation,
                                        parameters);
}

void selectLoopCarriedWidenProducts(mlir::ModuleOp module,
                                    mlir::IRRewriter &rewriter) {
  module.walk([&](mlir::scf::ForOp loop) {
    auto kernel = loop->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized())
      return;
    auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [index, argument] : llvm::enumerate(loop.getRegionIterArgs())) {
      auto type = mlir::dyn_cast<riscv::ValueType>(argument.getType());
      auto element = type ? mlir::dyn_cast<mlir::IntegerType>(type.getElementType())
                          : mlir::IntegerType();
      if (!type || !element || !element.isSigned() || element.getWidth() != 16 ||
          type.getShape().size() != 1 || type.getShape()[0] <= 1)
        continue;
      auto layout = type.getLayout();
      if (layout.getCarrier() != "rvv" || layout.getValidity() != "full" ||
          layout.getTimeFactors()[0] != 1 ||
          layout.getLaneFactors()[0] != type.getShape()[0] ||
          layout.getReplicaFactors()[0] != 1 ||
          layout.getFragmentFactors()[0] != 1 ||
          layout.getLocalFactors()[0] != 1)
        continue;

      llvm::SmallVector<std::pair<riscv::BinaryOp, riscv::RVVWidenMultiplyOp>> chain;
      mlir::Value current = argument;
      while (current.hasOneUse()) {
        mlir::Operation *user = *current.getUsers().begin();
        if (user == yield.getOperation())
          break;
        auto add = mlir::dyn_cast<riscv::BinaryOp>(user);
        if (!add || add.getKind() != "add" ||
            add->getBlock() != loop.getBody() || add.getResult().getType() != type)
          break;
        mlir::Value other = add.getLhs() == current ? add.getRhs() : add.getLhs();
        auto product = other.getDefiningOp<riscv::RVVWidenMultiplyOp>();
        if (!product || !other.hasOneUse() || other.getType() != type ||
            product->getBlock() != loop.getBody())
          break;
        auto lhs = product.getLhs().getType();
        auto rhs = product.getRhs().getType();
        if (width(lhs) > 8 || width(rhs) > 8 ||
            (!isSigned(lhs) && !isSigned(rhs)) ||
            lhs.getLayout().getValidity() != "full" ||
            rhs.getLayout().getValidity() != "full")
          break;
        chain.emplace_back(add, product);
        current = add.getResult();
      }
      // Select the entire carried chain together; a prefix would not satisfy
      // the existing partial operation's closed-chain contract.
      if (chain.empty() || !current.hasOneUse() ||
          *current.getUsers().begin() != yield.getOperation() ||
          yield.getOperand(index) != current)
        continue;
      for (auto [add, product] : chain) {
        mlir::Value accumulator = add.getLhs() == product.getResult()
                                      ? add.getRhs() : add.getLhs();
        auto lhs = product.getLhs().getType().getLayout();
        auto rhs = product.getRhs().getType().getLayout();
        rewriter.setInsertionPoint(add);
        auto accumulate = rewriter.create<riscv::RVVWidenAccumulateOp>(
            add.getLoc(), type, product.getLhs(), product.getRhs(), accumulator,
            type.getAxisIds(),
            riscv_internal::leaf(
                rewriter, "rvv", "widen-accumulate", "rvv.vwmacc.partial",
                "rvv.vwmacc.partial", lhs.getRegisterGroups() +
                    rhs.getRegisterGroups() + layout.getRegisterGroups(),
                layout.getRegisterGroups(), 0, 0, "none", "agnostic"));
        riscv_internal::copyOrigin(add, accumulate);
        rewriter.replaceOp(add, accumulate.getResult());
        rewriter.eraseOp(product);
      }
    }
  });
}

void selectZeroSeedProducts(mlir::ModuleOp module,
                            mlir::IRRewriter &rewriter) {
  // Seed a full byte-product chain without keeping a zero vector live.
  // Other partial representations retain their selected accumulator form.
  llvm::SmallVector<riscv::RVVWidenAccumulateOp> accumulates;
  module.walk([&](riscv::RVVWidenAccumulateOp accumulate) {
    accumulates.push_back(accumulate);
  });
  for (riscv::RVVWidenAccumulateOp accumulate : accumulates) {
    auto kernel = accumulate->getParentOfType<riscv::KernelOp>();
    if (!kernel || kernel.getResourcesMaterialized())
      continue;
    auto seed = accumulate.getAccumulator().getDefiningOp<riscv::RVVSplatOp>();
    auto constant = seed ? seed.getScalar().getDefiningOp<riscv::ConstantOp>()
                         : riscv::ConstantOp();
    auto zero = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                         : mlir::IntegerAttr();
    if (!zero || !zero.getValue().isZero())
      continue;
    auto lhs = accumulate.getLhs().getType();
    auto rhs = accumulate.getRhs().getType();
    auto result = accumulate.getResult().getType();
    auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
    auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
    auto narrow = lhs.getLayout();
    auto wide = result.getLayout();
    auto next = accumulate.getResult().hasOneUse()
        ? mlir::dyn_cast<riscv::RVVWidenAccumulateOp>(
              *accumulate.getResult().getUsers().begin())
        : riscv::RVVWidenAccumulateOp();
    if (!lhsElement || !rhsElement || lhsElement.isSignless() ||
        rhsElement.isSignless() ||
        (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
        lhsElement.getWidth() > 8 || rhsElement.getWidth() > 8 ||
        narrow.getSew() != 8 || wide.getSew() != 16 ||
        narrow.getValidity() != "full" || wide.getValidity() != "full" ||
        !next || next.getAccumulator() != accumulate.getResult() ||
        next.getReductionAxes() != accumulate.getReductionAxes() ||
        next->getBlock() != accumulate->getBlock() ||
        lhs.getShape() != rhs.getShape() ||
        lhs.getShape() != result.getShape() ||
        lhs.getAxisIds() != rhs.getAxisIds() ||
        lhs.getAxisIds() != result.getAxisIds() || narrow != rhs.getLayout() ||
        narrow.getTimeFactors() != wide.getTimeFactors() ||
        narrow.getLaneFactors() != wide.getLaneFactors() ||
        narrow.getReplicaFactors() != wide.getReplicaFactors() ||
        narrow.getFragmentFactors() != wide.getFragmentFactors() ||
        narrow.getLocalFactors() != wide.getLocalFactors())
      continue;
    llvm::StringRef instruction = lhsElement.isSigned() && rhsElement.isSigned()
        ? "rvv.vwmul.vv"
        : lhsElement.isSigned() ? "rvv.vwmulsu.vv"
                                : "rvv.vwmulsu.vv.swap";
    rewriter.setInsertionPoint(accumulate);
    auto product = rewriter.create<riscv::RVVWidenMultiplyOp>(
        accumulate.getLoc(), result, accumulate.getLhs(), accumulate.getRhs(),
        riscv_internal::leaf(
            rewriter, "rvv", "widen-multiply", instruction, instruction,
            narrow.getRegisterGroups() + rhs.getLayout().getRegisterGroups(),
            wide.getRegisterGroups(), 0, 0, "none", "exact"));
    riscv_internal::copyOrigin(accumulate, product);
    accumulate.getResult().replaceAllUsesWith(product.getResult());
    rewriter.eraseOp(accumulate);
    if (seed.getResult().use_empty())
      rewriter.eraseOp(seed);
  }
}

class SelectRISCVOperationsPass
    : public mlir::PassWrapper<SelectRISCVOperationsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-select-operations";
  }
  llvm::StringRef getDescription() const override {
    return "Select target-local operation families from typed numerical facts";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    selectLoopCarriedWidenProducts(getOperation(), rewriter);
    selectZeroSeedProducts(getOperation(), rewriter);
    mlir::Builder builder(&getContext());
    bool failed = false;
    getOperation().walk([&](mlir::Operation *operation) {
      auto leaf = operation->getAttrOfType<riscv::LeafAttr>("leaf");
      if (failed || !leaf || leaf.getEngine() != "unselected")
        return;
      // Memory forms and their terminal transfer leaves have one producer:
      // PlanRISCVMemory, after layouts are known.  Selection establishes only
      // numerical/structured operation anchors.
      if (mlir::isa<riscv::LoadOp, riscv::StoreOp, riscv::FieldOp,
                    riscv::ExtractOp, riscv::LookupOp,
                    riscv::RVVIndexedEntryLoadOp,
                    riscv::RVVUnitEntryWindowLoadOp,
                    riscv::RVVBitmaskWindowLoadOp,
                    riscv::ConvertLayoutOp>(operation))
        return;
      riscv::ImplementationAttr selected = select(operation, builder);
      if (!selected) {
        auto diagnostic = operation->emitError()
                          << "SelectRISCVOperations has no legal target-local operation for '"
                          << operation->getName() << "'";
        if (operation->getNumResults() == 1)
          diagnostic << " with result " << operation->getResult(0).getType();
        failed = true;
        return;
      }
      operation->setAttr("implementation", selected);
    });
    if (failed)
      signalPassFailure();
  }

private:
  riscv::ImplementationAttr select(mlir::Operation *operation,
                                   mlir::Builder &builder) {
    auto chooseScalarOrRVV = [&](llvm::StringRef family,
                                 llvm::StringRef scalarInstruction,
                                 llvm::StringRef vectorInstruction,
                                 mlir::Type result) {
      auto value = mlir::dyn_cast<riscv::ValueType>(result);
      if (!value || isTrivialShaped(result) ||
          value.getLayout().getCarrier() == "scalar")
        return scalarImplementation(builder, family, scalarInstruction);
      if (value.getLayout().getCarrier() != "unassigned" &&
          value.getLayout().getCarrier() != "rvv")
        return riscv::ImplementationAttr();
      return supportsRVVOperation(operation)
                 ? rvvImplementation(builder, family, vectorInstruction)
                 : riscv::ImplementationAttr();
    };

    if (auto op = mlir::dyn_cast<riscv::IotaOp>(operation)) {
      auto result = mlir::cast<riscv::ValueType>(op.getResult().getType());
      if (isTrivialShaped(result) || result.getLayout().getCarrier() == "scalar")
        return scalarImplementation(builder, "iota", "register.iota");
      if (result.getLayout().getCarrier() != "unassigned" &&
          result.getLayout().getCarrier() != "rvv")
        return {};
      return supportsRVVOperation(operation)
                 ? rvvImplementation(builder, "iota", "rvv.vid")
                 : riscv::ImplementationAttr();
    }
    if (auto op = mlir::dyn_cast<riscv::UpdateOp>(operation))
      return chooseScalarOrRVV("state-update", "scalar.update", "rvv.merge",
                               op.getResult().getType());
    if (auto op = mlir::dyn_cast<riscv::UnaryOp>(operation)) {
      std::string scalar = ("scalar." + op.getKind()).str();
      std::string vector = ("rvv.v" + op.getKind()).str();
      auto result = mlir::dyn_cast<riscv::ValueType>(op.getResult().getType());
      if (op.getKind() == "exp" && result &&
          result.getLayout().getCarrier() != "scalar" &&
          riscv_internal::logicalElement(op.getResult().getType()).isF32())
        return rvvImplementation(builder, "exp-approx",
                                 "rvv.exp-approx-f32");
      return chooseScalarOrRVV("unary", scalar, vector,
                               op.getResult().getType());
    }
    if (auto op = mlir::dyn_cast<riscv::BinaryOp>(operation)) {
      std::string scalar = ("scalar." + op.getKind()).str();
      std::string vector = ("rvv.v" + op.getKind()).str();
      return chooseScalarOrRVV("pointwise", scalar, vector,
                               op.getResult().getType());
    }
    if (auto op = mlir::dyn_cast<riscv::CompareOp>(operation)) {
      std::string scalar = ("scalar.cmp." + op.getPredicate()).str();
      std::string vector = ("rvv.vm" + op.getPredicate()).str();
      return chooseScalarOrRVV("compare", scalar, vector,
                               op.getResult().getType());
    }
    if (auto op = mlir::dyn_cast<riscv::CastOp>(operation))
      return chooseScalarOrRVV("cast", "scalar.cast", "rvv.convert",
                               op.getResult().getType());
    if (auto op = mlir::dyn_cast<riscv::NarrowOp>(operation))
      return chooseScalarOrRVV("narrow", "scalar.narrow", "rvv.vnclip",
                               op.getResult().getType());
    if (auto op = mlir::dyn_cast<riscv::WidenOp>(operation))
      return chooseScalarOrRVV("widen", "scalar.widen", "rvv.widen",
                               op.getResult().getType());
    if (auto op = mlir::dyn_cast<riscv::MacGroupsOp>(operation)) {
      auto target = operation->getParentOfType<riscv::KernelOp>().getTarget();
      if (!supportsRVVOperation(operation))
        return {};
      const bool mixedUnsignedSigned =
          (isUnsigned(op.getLhs().getType()) &&
           isSigned(op.getRhs().getType())) ||
          (isSigned(op.getLhs().getType()) &&
           isUnsigned(op.getRhs().getType()));
      if (mixedUnsignedSigned && width(op.getLhs().getType()) <= 8 &&
          width(op.getRhs().getType()) <= 8 &&
          target.getHasWideningInteger())
        return rvvImplementation(
            builder, "grouped-mac", "rvv.vwmaccsu.typed",
            {static_cast<int64_t>(op.getGroup())});
      return {};
    }
    if (auto op = mlir::dyn_cast<riscv::ReduceOp>(operation)) {
      auto input = mlir::dyn_cast<riscv::ValueType>(op.getInput().getType());
      if (!input)
        return {};
      if (input.getLayout().getCarrier() == "scalar") {
        std::string instruction =
            ("scalar.register-reduce." + op.getKind()).str();
        return riscv_internal::implementation(
            builder, "scalar", "reduce", instruction,
            {static_cast<int64_t>(op.getAxis())});
      }
      if (!supportsRVVOperation(operation))
        return {};
      std::string instruction =
          isFloat(op.getResult().getType())
              ? ("rvv.vfred" + op.getKind()).str()
              : ("rvv.vred" + op.getKind()).str();
      return rvvImplementation(builder, "reduce", instruction,
                               {static_cast<int64_t>(op.getAxis())});
    }
    if (auto op = mlir::dyn_cast<riscv::Fold2Op>(operation))
      return supportsRVVOperation(operation)
                 ? rvvImplementation(builder, "fold2", "rvv.pair-fold-add")
                 : riscv::ImplementationAttr();
    auto selectContract = [&](mlir::Value lhs, mlir::Value rhs,
                              mlir::Type result, bool permitIME,
                              bool permitWidenDot,
                              llvm::ArrayRef<int64_t> over)
        -> riscv::ImplementationAttr {
      if (permitIME)
        if (auto fragment = matchingFragment(operation, lhs, rhs, result))
          return riscv_internal::implementation(
              builder, "ime", "fragment-mma", fragment.getInstruction(),
              {fragment.getMFactor(), fragment.getNFactor(),
               fragment.getKFactor()});
      if (!supportsRVVOperation(operation))
        return {};
      auto lhsInteger = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(lhs.getType()));
      auto rhsInteger = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(rhs.getType()));
      auto resultInteger = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(result));
      auto lhsFloat = mlir::dyn_cast<mlir::FloatType>(
          riscv_internal::logicalElement(lhs.getType()));
      auto rhsFloat = mlir::dyn_cast<mlir::FloatType>(
          riscv_internal::logicalElement(rhs.getType()));
      auto resultFloat = mlir::dyn_cast<mlir::FloatType>(
          riscv_internal::logicalElement(result));
      auto target = operation->getParentOfType<riscv::KernelOp>().getTarget();
      auto wideningDotFitsRegisterMicrotile = [&]() {
        if (!mlir::isa<riscv::OuterContractOp>(operation))
          return true;
        auto value = mlir::dyn_cast<riscv::ValueType>(result);
        if (!value)
          return false;
        int64_t replicas = 1;
        for (int64_t axis : value.getAxisIds().asArrayRef()) {
          int64_t extent = riscv_internal::physicalExtent(
              operation->getResult(0), axis);
          if (extent <= 0 ||
              replicas > target.getVectorRegisters() / extent)
            return false;
          replicas *= extent;
        }
        // An 8/16-bit widening product needs at least two register groups per
        // accumulator at LMUL=m1, plus one group for each operand.  This is the
        // minimum resource proof available before layout propagation; the
        // exact selected LMUL is checked again by LowerRISCVComposites.
        return replicas > 0 &&
               replicas * 2 + 2 <= target.getVectorRegisters();
      };
      // The widening-dot family follows from the typed numerical relation and
      // target capability.  It must be selected before layout propagation so
      // its reduction axis can anchor the operand layouts.  The concrete
      // doubled-LMUL legality is checked after those layouts exist, when the
      // composite is lowered.
      if (permitWidenDot && lhsInteger && rhsInteger && resultInteger &&
          lhsInteger.getWidth() <= 16 && rhsInteger.getWidth() <= 16 &&
          std::max<unsigned>(8, lhsInteger.getWidth()) ==
              std::max<unsigned>(8, rhsInteger.getWidth()) &&
          (lhsInteger.isSigned() || rhsInteger.isSigned()) &&
          resultInteger.isSigned() && resultInteger.getWidth() == 32 &&
          !over.empty() && target.getHasWideningInteger() &&
          wideningDotFitsRegisterMicrotile())
        return rvvImplementation(builder, "widen-dot",
                                 "rvv.vwmul-vwredsum", over);
      if (lhsInteger && rhsInteger && lhsInteger.getWidth() <= 8 &&
          rhsInteger.getWidth() <= 8 &&
          ((lhsInteger.isUnsigned() && rhsInteger.isSigned()) ||
           (lhsInteger.isSigned() && rhsInteger.isUnsigned())) &&
          width(result) == 32 && hasEncodedFieldOrigin(lhs) &&
          hasEncodedFieldOrigin(rhs))
        return rvvImplementation(builder, "encoded-contract",
                                 "rvv.vmacc.decoded-u8-s8", over);
      if (lhsFloat && rhsFloat && resultFloat && lhsFloat.isF16() &&
          rhsFloat.isF16() && resultFloat.isF32() &&
          target.getHasVectorF16())
        return rvvImplementation(builder, "widen-float-contract",
                                 "rvv.vfwmacc", over);
      llvm::StringRef instruction = isFloat(result) ? "rvv.vfmacc"
                                                    : "rvv.vmacc";
      return rvvImplementation(builder, "contract", instruction, over);
    };
    if (auto op = mlir::dyn_cast<riscv::DotOp>(operation))
      return selectContract(op.getLhs(), op.getRhs(), op.getResult().getType(),
                            false, true, op.getOver());
    if (auto op = mlir::dyn_cast<riscv::ContractOp>(operation))
      return selectContract(op.getLhs(), op.getRhs(), op.getResult().getType(),
                            true, true, op.getOver());
    if (auto op = mlir::dyn_cast<riscv::OuterContractOp>(operation))
      return selectContract(op.getLhs(), op.getRhs(), op.getResult().getType(),
                            true, true, op.getOver());
    return {};
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createSelectRISCVOperationsPass() {
  return std::make_unique<SelectRISCVOperationsPass>();
}
