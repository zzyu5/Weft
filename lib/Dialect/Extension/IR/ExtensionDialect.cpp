#include "Weft/Dialect/Extension/IR/ExtensionDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

using namespace weft::extension;

#include "Weft/Dialect/Extension/IR/ExtensionOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Extension/IR/ExtensionOps.cpp.inc"

namespace {

mlir::Type elementTypeOf(mlir::Type type) {
  if (auto masked = mlir::dyn_cast<weft::kernel::MaskedType>(type))
    type = masked.getValueType();
  if (auto block = mlir::dyn_cast<weft::kernel::BlockType>(type))
    return block.getElementType();
  if (auto region = mlir::dyn_cast<weft::kernel::RegionType>(type))
    return region.getElementType();
  return type;
}

} // namespace

mlir::LogicalResult BlockScaledContractOp::verify() {
  mlir::Type lhsType = getLhs().getType();
  mlir::Type rhsType = getRhs().getType();
  if (auto masked = mlir::dyn_cast<weft::kernel::MaskedType>(lhsType))
    lhsType = masked.getValueType();
  if (auto masked = mlir::dyn_cast<weft::kernel::MaskedType>(rhsType))
    rhsType = masked.getValueType();
  auto lhs = mlir::dyn_cast<weft::kernel::BlockType>(lhsType);
  auto rhs = mlir::dyn_cast<weft::kernel::BlockType>(rhsType);
  if (!lhs || !rhs)
    return emitOpError("lhs and rhs must be logical blocks");
  if (lhs.getShape().empty() || rhs.getShape().empty() ||
      lhs.getShape().back() != rhs.getShape().front())
    return emitOpError(
        "last lhs and first rhs dimensions must form the dot axis");
  if (lhs.getShape().back() == -1 &&
      !weft::kernel::haveSameLogicalExtent(
          getLhs(), static_cast<int64_t>(lhs.getShape().size()) - 1, getRhs(),
          0))
    return emitOpError("cannot prove block-scaled dot extent identity");
  if (getRounding() != "rne")
    return emitOpError("this semantic primitive requires rounding=\"rne\"");
  if (!getSaturation())
    return emitOpError("this semantic primitive requires saturation=true");
  if (elementTypeOf(getScaleA().getType()) !=
          elementTypeOf(getScaleB().getType()) ||
      elementTypeOf(getInit().getType()) != elementTypeOf(getResult().getType()) ||
      getInit().getType() != getResult().getType())
    return emitOpError("scale and accumulator/result element types must match");
  llvm::SmallVector<int64_t> outputShape(lhs.getShape().drop_back());
  outputShape.append(rhs.getShape().drop_front().begin(),
                     rhs.getShape().drop_front().end());
  if (outputShape.empty()) {
    if (mlir::isa<weft::kernel::BlockType>(getResult().getType()))
      return emitOpError("fully contracted result must be scalar");
  } else {
    auto result = mlir::dyn_cast<weft::kernel::BlockType>(getResult().getType());
    if (!result || result.getShape() != llvm::ArrayRef<int64_t>(outputShape))
      return emitOpError("result shape must match the implicit dot relation");
  }
  if (!getOperation()->getParentOfType<weft::kernel::KernelOp>())
    return emitOpError("must be nested in a canonical Weft kernel");
  return mlir::success();
}

void WEFTExtensionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/Extension/IR/ExtensionOps.cpp.inc"
      >();
}
