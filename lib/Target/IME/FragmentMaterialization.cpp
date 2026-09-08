#include "FragmentMaterialization.h"
#include "Weft/Dialect/RISCV/IR/Fragment.h"
#include "../RISCVPhysicalSupport.h"
#include "Weft/Target/RISCVFragment.h"

using namespace weft;
using riscv_internal::accessOf;

namespace {
riscv::LayoutAttr fragmentLayout(mlir::Builder &builder, riscv::ValueType value,
                                 llvm::ArrayRef<int64_t> physicalShape) {
  auto axes = value.getAxisIds().asArrayRef();
  llvm::SmallVector<int64_t> time(axes.size(), 1);
  llvm::SmallVector<int64_t> one(axes.size(), 1);
  llvm::SmallVector<int64_t> fragment(physicalShape.begin(), physicalShape.end());
  return riscv::LayoutAttr::get(
      builder.getContext(), "ime", value.getAxisIds(),
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, fragment),
      riscv_internal::integers(builder, one),
      std::max<int64_t>(8, riscv_internal::logicalBitWidth(value)), 0, 1, 0,
      value.getLayout().getValidity());
}

} // namespace

mlir::FailureOr<mlir::Value> weft::riscv_internal::materializeFragmentProduct(
    mlir::IRRewriter &rewriter, mlir::Operation *operation,
    riscv::ImplementationAttr implementation, riscv::ValueType resultType) {

  auto parameters = implementation.getParameters().asArrayRef();
  riscv::FragmentCapabilityAttr capability;
  for (mlir::Attribute candidate :
       operation->getParentOfType<riscv::KernelOp>()
           .getTarget()
           .getFragments()) {
    auto fragment = mlir::cast<riscv::FragmentCapabilityAttr>(candidate);
    if (fragment.getInstruction() == implementation.getOperation()) {
      capability = fragment;
      break;
    }
  }
  if (parameters.size() != 3 || !capability) {
    operation->emitError("IME selection has no fragment shape/resources");
    return mlir::failure();
  }
  const auto *contract = findRISCVFragmentCapability(capability.getInstruction());
  if (!contract) {
    operation->emitError("selected fragment has no registered local contract");
    return mlir::failure();
  }
  auto lhsType =
      mlir::cast<riscv::ValueType>(operation->getOperand(0).getType());
  auto rhsType =
      mlir::cast<riscv::ValueType>(operation->getOperand(1).getType());
  llvm::SmallVector<int64_t, 2> lhsShape = {
      capability.getMFactor(), capability.getKFactor()};
  llvm::SmallVector<int64_t, 2> rhsShape = {
      capability.getKFactor(), capability.getNFactor()};
  llvm::SmallVector<int64_t, 2> resultShape = {
      capability.getMFactor(), capability.getNFactor()};
  auto makeFragment = [&](riscv::ValueType value, llvm::StringRef role,
                          riscv::FragmentPackingAttr packing,
                          llvm::ArrayRef<int64_t> shape) {
    const int64_t resources =
        role == "lhs" ? capability.getLhsResourceGroups()
        : role == "rhs" ? capability.getRhsResourceGroups()
                        : capability.getAccumulatorResourceGroups();
    return riscv::FragmentType::get(
        value.getContext(), implementation.getOperation(),
        role, packing, value.getElementType(),
        riscv_internal::integers(rewriter, shape), value.getAxisIds(),
        fragmentLayout(rewriter, value, shape), resources);
  };
  riscv::FragmentType lhsFragment =
      makeFragment(lhsType, "lhs", capability.getLhsPacking(), lhsShape);
  riscv::FragmentType rhsFragment =
      makeFragment(rhsType, "rhs", capability.getRhsPacking(), rhsShape);
  riscv::FragmentType resultFragment = makeFragment(
      resultType, "accumulator", capability.getAccumulatorPacking(),
      resultShape);
  auto packLeaf = [&](llvm::StringRef role) {
    return riscv_internal::leaf(
        rewriter, "ime", "fragment-pack", ("ime.pack." + role).str(),
        ("ime.pack." + role).str(), 0, 0, contract->packTemporaryGroups, 0,
        "none", "exact", parameters,
        role == "lhs" ? contract->lhsPackLocalBytes : contract->rhsPackLocalBytes);
  };
  riscv::AccessAttr lhsAccess = accessOf(operation->getOperand(0));
  riscv::AccessAttr rhsAccess = accessOf(operation->getOperand(1));
  if (!lhsAccess || !rhsAccess) {
    operation->emitError("IME operands have no typed memory access contract");
    return mlir::failure();
  }
  auto lhsPack = rewriter.create<riscv::IMEPackOp>(
      operation->getLoc(), lhsFragment, operation->getOperand(0), "lhs",
      capability.getLhsPacking(), lhsAccess,
      packLeaf("lhs"));
  auto rhsPack = rewriter.create<riscv::IMEPackOp>(
      operation->getLoc(), rhsFragment, operation->getOperand(1), "rhs",
      capability.getRhsPacking(), rhsAccess,
      packLeaf("rhs"));
  auto mmaLeaf = riscv_internal::leaf(
      rewriter, "ime", "fragment-mma", implementation.getOperation(),
      implementation.getOperation(), 0, 0, 0,
      riscv::fragmentMMAAdditionalGroups(capability), "none", "exact",
      parameters, contract->mmaLocalBytes);
  auto mma = rewriter.create<riscv::IMEFragmentMMAOp>(
      operation->getLoc(), resultFragment, lhsPack.getResult(),
      rhsPack.getResult(), capability.getMmaGroups(),
      capability.getMmaChunks(), capability.getClobbers(),
      capability.getVolatileAsm(), capability.getMemoryClobber(), mmaLeaf);
  auto unpackLeaf = riscv_internal::leaf(
      rewriter, "ime", "fragment-unpack", "ime.unpack.rvv",
      "ime.unpack.rvv", 0, 0, contract->unpackTemporaryGroups, 0);
  auto unpack = rewriter.create<riscv::IMEUnpackOp>(
      operation->getLoc(), resultType, mma.getResult(),
      riscv_internal::conversion(rewriter, "fragment_to_rvv", "handoff",
                                 contract->unpackTemporaryGroups),
      unpackLeaf);
  riscv_internal::copyOrigin(operation, mma);
  riscv_internal::copyOrigin(operation, unpack);
  if (auto canonical = operation->getAttr("canonical_op")) {
    mma->setAttr("canonical_op", canonical);
    unpack->setAttr("canonical_op", canonical);
  }
  return unpack.getResult();
}
