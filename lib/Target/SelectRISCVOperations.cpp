#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/IR/BuiltinTypes.h"
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
                    riscv::ExtractOp, riscv::LookupOp>(operation))
        return;
      riscv::ImplementationAttr selected = select(operation, builder);
      if (!selected) {
        operation->emitError(
            "SelectRISCVOperations has no legal target-local operation for the typed primitive");
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
      if (!isShaped(result) || isTrivialShaped(result))
        return scalarImplementation(builder, family, scalarInstruction);
      return supportsRVVOperation(operation)
                 ? rvvImplementation(builder, family, vectorInstruction)
                 : riscv::ImplementationAttr();
    };

    if (auto op = mlir::dyn_cast<riscv::IotaOp>(operation)) {
      if (isTrivialShaped(op.getResult().getType()))
        return scalarImplementation(builder, "iota", "register.iota");
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
      if (op.getKind() == "exp" && isShaped(op.getResult().getType()) &&
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
      if (isUnsigned(op.getLhs().getType()) && isSigned(op.getRhs().getType()) &&
          width(op.getLhs().getType()) <= 8 && width(op.getRhs().getType()) <= 8 &&
          target.getHasWideningInteger())
        return rvvImplementation(
            builder, "grouped-mac", "rvv.vwmaccsu.typed",
            {static_cast<int64_t>(op.getGroup())});
      return {};
    }
    if (auto op = mlir::dyn_cast<riscv::ReduceOp>(operation)) {
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
      if (lhsInteger && rhsInteger && lhsInteger.getWidth() <= 8 &&
          rhsInteger.getWidth() <= 8 &&
          ((lhsInteger.isUnsigned() && rhsInteger.isSigned()) ||
           (lhsInteger.isSigned() && rhsInteger.isUnsigned())) &&
          width(result) == 32 && hasEncodedFieldOrigin(lhs) &&
          hasEncodedFieldOrigin(rhs))
        return rvvImplementation(builder, "encoded-contract",
                                 "rvv.vmacc.decoded-u8-s8", over);
      llvm::StringRef instruction = isFloat(result) ? "rvv.vfmacc"
                                                    : "rvv.vmacc";
      return rvvImplementation(builder, "contract", instruction, over);
    };
    if (auto op = mlir::dyn_cast<riscv::DotOp>(operation))
      return selectContract(op.getLhs(), op.getRhs(), op.getResult().getType(),
                            false, op.getOver());
    if (auto op = mlir::dyn_cast<riscv::ContractOp>(operation))
      return selectContract(op.getLhs(), op.getRhs(), op.getResult().getType(),
                            true, op.getOver());
    if (auto op = mlir::dyn_cast<riscv::OuterContractOp>(operation))
      return selectContract(op.getLhs(), op.getRhs(), op.getResult().getType(),
                            true, op.getOver());
    return {};
  }
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createSelectRISCVOperationsPass() {
  return std::make_unique<SelectRISCVOperationsPass>();
}
