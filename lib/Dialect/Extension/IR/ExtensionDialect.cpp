#include "Weft/Dialect/Extension/IR/ExtensionDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

using namespace weft::extension;

#include "Weft/Dialect/Extension/IR/ExtensionOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Extension/IR/ExtensionOps.cpp.inc"

mlir::LogicalResult AffineI4I8ContractOp::verify() {
  auto blockType = [](mlir::Type type) {
    if (auto masked = mlir::dyn_cast<weft::kernel::MaskedType>(type))
      type = masked.getValueType();
    return mlir::dyn_cast<weft::kernel::BlockType>(type);
  };
  auto activation = blockType(getActivation().getType());
  auto packedWeight = blockType(getPackedWeight().getType());
  auto weightScale = blockType(getWeightScale().getType());
  auto weightZeroPoint = blockType(getWeightZeroPoint().getType());
  auto init = blockType(getInit().getType());
  auto result = blockType(getResult().getType());
  if (!activation || activation.getShape() != llvm::ArrayRef<int64_t>({32}) ||
      !activation.getElementType().isSignedInteger(8))
    return emitOpError("activation must be a signed i8 block<32>");
  if (!packedWeight ||
      packedWeight.getShape() != llvm::ArrayRef<int64_t>({16, 16}) ||
      !packedWeight.getElementType().isUnsignedInteger(8))
    return emitOpError("packed_weight must be a u8 block<16x16>");
  if (!getActivationScale().getType().isF32())
    return emitOpError("activation_scale must be scalar f32");
  if (!weightScale ||
      weightScale.getShape() != llvm::ArrayRef<int64_t>({16}) ||
      !weightScale.getElementType().isF32())
    return emitOpError("weight_scale must be an f32 block<16>");
  if (!weightZeroPoint ||
      weightZeroPoint.getShape() != llvm::ArrayRef<int64_t>({16}) ||
      !weightZeroPoint.getElementType().isUnsignedInteger(8))
    return emitOpError("weight_zero_point must be a u8 block<16>");
  if (!init || !result || init.getShape() != llvm::ArrayRef<int64_t>({16}) ||
      init.getElementType() != mlir::Float32Type::get(getContext()) ||
      getInit().getType() != getResult().getType())
    return emitOpError("init and result must be the same f32 block<16>");
  if (!getOperation()->getParentOfType<weft::kernel::KernelOp>())
    return emitOpError("must be nested in a canonical Weft kernel");
  return mlir::success();
}

mlir::LogicalResult GroupedAffineI4I8DotOp::verify() {
  auto blockType = [](mlir::Type type) {
    if (auto masked = mlir::dyn_cast<weft::kernel::MaskedType>(type))
      type = masked.getValueType();
    return mlir::dyn_cast<weft::kernel::BlockType>(type);
  };
  auto packedWeight = blockType(getPackedWeight().getType());
  auto scaleMin = blockType(getScaleMin().getType());
  auto activation = blockType(getActivation().getType());
  auto activationSums = blockType(getActivationSumBytes().getType());
  if (!packedWeight ||
      packedWeight.getShape() != llvm::ArrayRef<int64_t>({128}) ||
      !packedWeight.getElementType().isUnsignedInteger(8))
    return emitOpError("packed_weight must be a u8 block<128>");
  if (!scaleMin || scaleMin.getShape() != llvm::ArrayRef<int64_t>({12}) ||
      !scaleMin.getElementType().isUnsignedInteger(8))
    return emitOpError("scale_min must be a u8 block<12>");
  if (!activation ||
      activation.getShape() != llvm::ArrayRef<int64_t>({256}) ||
      !activation.getElementType().isSignedInteger(8))
    return emitOpError("activation must be a signed i8 block<256>");
  if (!activationSums ||
      activationSums.getShape() != llvm::ArrayRef<int64_t>({32}) ||
      !activationSums.getElementType().isUnsignedInteger(8))
    return emitOpError("activation_sum_bytes must be a u8 block<32>");
  if (!getDotScale().getType().isF32() ||
      !getMinimumScale().getType().isF32())
    return emitOpError("dot_scale and minimum_scale must be scalar f32");
  if (!getInit().getType().isF32() || getInit().getType() != getResult().getType())
    return emitOpError("init and result must be scalar f32");
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
