#include "Weft/Dialect/Extension/IR/ExtensionDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"

using namespace weft::extension;

#include "Weft/Dialect/Extension/IR/ExtensionOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Extension/IR/ExtensionOps.cpp.inc"

namespace {
mlir::LogicalResult requireUnmaskedLocalValue(mlir::Operation *op,
                                               mlir::Value value,
                                               llvm::StringRef name) {
  if (!weft::kernel::isLogicalValue(value.getType()))
    return op->emitError() << name << " must be a canonical logical value";
  if (weft::kernel::hasLogicalValidity(value.getType()))
    return op->emitError()
           << name
           << " carries logical validity; fill it before this extension primitive";
  return mlir::success();
}

weft::kernel::BlockType localBlock(mlir::Operation *op, mlir::Value value,
                                   llvm::StringRef name) {
  if (mlir::failed(requireUnmaskedLocalValue(op, value, name)))
    return {};
  return mlir::dyn_cast<weft::kernel::BlockType>(
      weft::kernel::unwrapLogicalValidity(value.getType()));
}

mlir::LogicalResult requireScalar(mlir::Operation *op, mlir::Value value,
                                  mlir::Type expected,
                                  llvm::StringRef name) {
  if (mlir::failed(requireUnmaskedLocalValue(op, value, name)))
    return mlir::failure();
  if (weft::kernel::logicalShapeKind(value.getType()) !=
          weft::kernel::LogicalShapeKind::Scalar ||
      weft::kernel::logicalElementType(value.getType()) != expected)
    return op->emitError() << name << " must be scalar " << expected;
  return mlir::success();
}

mlir::LogicalResult requireReadableU8Pointer(mlir::Operation *op,
                                             mlir::Value value,
                                             llvm::StringRef name) {
  auto pointer = mlir::dyn_cast<weft::kernel::PtrType>(value.getType());
  if (!pointer || !pointer.getElementType().isUnsignedInteger(8) ||
      pointer.getAccess() == "write")
    return op->emitError() << name << " must be a readable scalar u8 pointer";
  return mlir::success();
}

mlir::LogicalResult requirePersistentU8Pointer(
    mlir::Operation *op, mlir::Value value, llvm::StringRef name,
    llvm::StringRef storageFormat) {
  if (mlir::failed(requireReadableU8Pointer(op, value, name)))
    return mlir::failure();
  auto pointer = mlir::cast<weft::kernel::PtrType>(value.getType());
  if (pointer.getStorageClass() != "persistent" ||
      pointer.getStorageFormat() != storageFormat)
    return op->emitError()
           << name << " must use persistent format " << storageFormat;
  return mlir::success();
}

mlir::LogicalResult verifyScalarDotResult(mlir::Operation *op,
                                          mlir::Value init,
                                          mlir::Value result) {
  mlir::Type f32 = mlir::Float32Type::get(op->getContext());
  if (mlir::failed(requireScalar(op, init, f32, "init")) ||
      mlir::failed(requireScalar(op, result, f32, "result")))
    return mlir::failure();
  if (init.getType() != result.getType())
    return op->emitError("init and result must be scalar f32");
  if (!op->getParentOfType<weft::kernel::KernelOp>())
    return op->emitError("must be nested in a canonical Weft kernel");
  return mlir::success();
}
} // namespace

mlir::LogicalResult LoadF16LEOp::verify() {
  if (mlir::failed(requireReadableU8Pointer(*this, getBase(), "base")))
    return mlir::failure();
  if (weft::kernel::logicalShapeKind(getResult().getType()) !=
          weft::kernel::LogicalShapeKind::Scalar ||
      !weft::kernel::logicalElementType(getResult().getType()).isF32())
    return emitOpError("result must be scalar f32");
  if (!getOperation()->getParentOfType<weft::kernel::KernelOp>())
    return emitOpError("must be nested in a canonical Weft kernel");
  return mlir::success();
}

mlir::LogicalResult AffineI4I8DotOp::verify() {
  auto activation = localBlock(*this, getActivation(), "activation");
  auto init = localBlock(*this, getInit(), "init");
  auto result = localBlock(*this, getResult(), "result");
  if (!activation || activation.getShape() != llvm::ArrayRef<int64_t>({32}) ||
      !activation.getElementType().isSignedInteger(8))
    return emitOpError("activation must be a signed i8 block<32>");
  if (mlir::failed(requirePersistentU8Pointer(
          *this, getPackedBase(), "packed_base", "q4_k_n16_k32_304b")))
    return mlir::failure();
  if (mlir::failed(requireScalar(*this, getActivationScale(),
                                 mlir::Float32Type::get(getContext()),
                                 "activation_scale")))
    return emitOpError("activation_scale must be scalar f32");
  if (!init || !result || init.getShape() != llvm::ArrayRef<int64_t>({16}) ||
      !init.getElementType().isF32() ||
      getInit().getType() != getResult().getType())
    return emitOpError("init and result must be the same f32 block<16>");
  if (!getOperation()->getParentOfType<weft::kernel::KernelOp>())
    return emitOpError("must be nested in a canonical Weft kernel");
  return mlir::success();
}

mlir::LogicalResult SymmetricI4I8DotOp::verify() {
  auto activation = localBlock(*this, getActivation(), "activation");
  auto init = localBlock(*this, getInit(), "init");
  auto result = localBlock(*this, getResult(), "result");
  if (!activation || activation.getShape() != llvm::ArrayRef<int64_t>({32}) ||
      !activation.getElementType().isSignedInteger(8))
    return emitOpError("activation must be a signed i8 block<32>");
  if (mlir::failed(requirePersistentU8Pointer(
          *this, getPackedBase(), "packed_base", "q4_0_n16_k32_288b")))
    return mlir::failure();
  if (mlir::failed(requireScalar(*this, getActivationScale(),
                                 mlir::Float32Type::get(getContext()),
                                 "activation_scale")))
    return emitOpError("activation_scale must be scalar f32");
  if (!init || !result || init.getShape() != llvm::ArrayRef<int64_t>({16}) ||
      !init.getElementType().isF32() ||
      getInit().getType() != getResult().getType())
    return emitOpError("init and result must be the same f32 block<16>");
  if (!getOperation()->getParentOfType<weft::kernel::KernelOp>())
    return emitOpError("must be nested in a canonical Weft kernel");
  return mlir::success();
}

mlir::LogicalResult GroupedAffineI4I8DotOp::verify() {
  auto packedWeight = localBlock(*this, getPackedWeight(), "packed_weight");
  auto scaleMin = localBlock(*this, getScaleMin(), "scale_min");
  auto activation = localBlock(*this, getActivation(), "activation");
  auto activationSums =
      localBlock(*this, getActivationSumBytes(), "activation_sum_bytes");
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
  mlir::Type f32 = mlir::Float32Type::get(getContext());
  if (mlir::failed(requireScalar(*this, getDotScale(), f32, "dot_scale")) ||
      mlir::failed(requireScalar(*this, getMinimumScale(), f32,
                                 "minimum_scale")))
    return emitOpError("dot_scale and minimum_scale must be scalar f32");
  return verifyScalarDotResult(*this, getInit(), getResult());
}

mlir::LogicalResult SignBitI8DotOp::verify() {
  auto signBits = localBlock(*this, getSignBits(), "sign_bits");
  auto activation = localBlock(*this, getActivation(), "activation");
  if (!signBits || signBits.getShape() != llvm::ArrayRef<int64_t>({4}) ||
      !signBits.getElementType().isUnsignedInteger(8))
    return emitOpError("sign_bits must be a u8 block<4>");
  if (!activation || activation.getShape() != llvm::ArrayRef<int64_t>({32}) ||
      !activation.getElementType().isSignedInteger(8))
    return emitOpError("activation must be a signed i8 block<32>");
  mlir::Type f32 = mlir::Float32Type::get(getContext());
  if (mlir::failed(requireScalar(*this, getActivationScale(), f32,
                                 "activation_scale")) ||
      mlir::failed(requireScalar(*this, getSignScale(), f32, "sign_scale")))
    return emitOpError("activation_scale and sign_scale must be scalar f32");
  return verifyScalarDotResult(*this, getInit(), getResult());
}

mlir::LogicalResult E2M1E8M0I8DotOp::verify() {
  auto packedCodes = localBlock(*this, getPackedCodes(), "packed_codes");
  auto activation = localBlock(*this, getActivation(), "activation");
  if (!packedCodes ||
      packedCodes.getShape() != llvm::ArrayRef<int64_t>({16}) ||
      !packedCodes.getElementType().isUnsignedInteger(8))
    return emitOpError("packed_codes must be a u8 block<16>");
  if (mlir::failed(requireScalar(*this, getExponent(),
                                 mlir::IntegerType::get(
                                     getContext(), 8,
                                     mlir::IntegerType::Unsigned),
                                 "exponent")))
    return emitOpError("exponent must be scalar u8");
  if (!activation || activation.getShape() != llvm::ArrayRef<int64_t>({32}) ||
      !activation.getElementType().isSignedInteger(8))
    return emitOpError("activation must be a signed i8 block<32>");
  if (mlir::failed(requireScalar(*this, getActivationScale(),
                                 mlir::Float32Type::get(getContext()),
                                 "activation_scale")))
    return emitOpError("activation_scale must be scalar f32");
  return verifyScalarDotResult(*this, getInit(), getResult());
}

mlir::LogicalResult verifyByteBlock(mlir::Operation *op, mlir::Value value,
                                    int64_t extent, bool signedElement,
                                    llvm::StringRef name);

mlir::LogicalResult PackedI5I8DotOp::verify() {
  if (mlir::failed(verifyByteBlock(*this, getLowBits(), 16, false,
                                  "low_bits")) ||
      mlir::failed(verifyByteBlock(*this, getHighBits(), 4, false,
                                  "high_bits")) ||
      mlir::failed(verifyByteBlock(*this, getActivation(), 32, true,
                                  "activation")))
    return mlir::failure();
  if (mlir::failed(requireScalar(*this, getZeroPoint(),
                                 mlir::IntegerType::get(
                                     getContext(), 32,
                                     mlir::IntegerType::Signed),
                                 "zero_point")))
    return emitOpError("zero_point must be scalar i32");
  mlir::Type f32 = mlir::Float32Type::get(getContext());
  if (mlir::failed(requireScalar(*this, getDotScale(), f32, "dot_scale")) ||
      mlir::failed(requireScalar(*this, getAdditiveBias(), f32,
                                 "additive_bias")))
    return emitOpError("dot_scale and additive_bias must be scalar f32");
  return verifyScalarDotResult(*this, getInit(), getResult());
}

mlir::LogicalResult verifyByteBlock(mlir::Operation *op, mlir::Value value,
                                    int64_t extent, bool signedElement,
                                    llvm::StringRef name) {
  if (mlir::failed(requireUnmaskedLocalValue(op, value, name)))
    return mlir::failure();
  mlir::Type type = weft::kernel::unwrapLogicalValidity(value.getType());
  auto block = mlir::dyn_cast<weft::kernel::BlockType>(type);
  if (!block || block.getShape() != llvm::ArrayRef<int64_t>({extent}) ||
      (signedElement ? !block.getElementType().isSignedInteger(8)
                     : !block.getElementType().isUnsignedInteger(8)))
    return op->emitError() << name << " must be a "
                           << (signedElement ? "signed i8" : "u8")
                           << " block<" << extent << ">";
  return mlir::success();
}

mlir::LogicalResult IQ2SI8DotOp::verify() {
  if (mlir::failed(verifyByteBlock(*this, getCodes(), 32, false, "codes")) ||
      mlir::failed(verifyByteBlock(*this, getHighBits(), 8, false, "high_bits")) ||
      mlir::failed(verifyByteBlock(*this, getSignBits(), 32, false, "sign_bits")) ||
      mlir::failed(verifyByteBlock(*this, getScales(), 8, false, "scales")) ||
      mlir::failed(verifyByteBlock(*this, getActivation(), 256, true, "activation")))
    return mlir::failure();
  mlir::Type f32 = mlir::Float32Type::get(getContext());
  if (mlir::failed(requireScalar(*this, getWeightScale(), f32,
                                 "weight_scale")) ||
      mlir::failed(requireScalar(*this, getActivationScale(), f32,
                                 "activation_scale")))
    return emitOpError("weight_scale and activation_scale must be scalar f32");
  return verifyScalarDotResult(*this, getInit(), getResult());
}

mlir::LogicalResult IQ3SI8DotOp::verify() {
  if (mlir::failed(verifyByteBlock(*this, getCodes(), 64, false, "codes")) ||
      mlir::failed(verifyByteBlock(*this, getHighBits(), 8, false, "high_bits")) ||
      mlir::failed(verifyByteBlock(*this, getSignBits(), 32, false, "sign_bits")) ||
      mlir::failed(verifyByteBlock(*this, getScales(), 4, false, "scales")) ||
      mlir::failed(verifyByteBlock(*this, getActivation(), 256, true, "activation")))
    return mlir::failure();
  mlir::Type f32 = mlir::Float32Type::get(getContext());
  if (mlir::failed(requireScalar(*this, getWeightScale(), f32,
                                 "weight_scale")) ||
      mlir::failed(requireScalar(*this, getActivationScale(), f32,
                                 "activation_scale")))
    return emitOpError("weight_scale and activation_scale must be scalar f32");
  return verifyScalarDotResult(*this, getInit(), getResult());
}

mlir::LogicalResult IQ1MI8DotOp::verify() {
  if (mlir::failed(verifyByteBlock(*this, getCodes(), 32, false, "codes")) ||
      mlir::failed(verifyByteBlock(*this, getHighDeltaBits(), 16, false,
                                  "high_delta_bits")) ||
      mlir::failed(verifyByteBlock(*this, getScales(), 8, false, "scales")) ||
      mlir::failed(verifyByteBlock(*this, getActivation(), 256, true, "activation")))
    return mlir::failure();
  if (mlir::failed(requireScalar(*this, getActivationScale(),
                                 mlir::Float32Type::get(getContext()),
                                 "activation_scale")))
    return emitOpError("activation_scale must be scalar f32");
  return verifyScalarDotResult(*this, getInit(), getResult());
}

mlir::LogicalResult Q6KI8DotOp::verify() {
  if (mlir::failed(verifyByteBlock(*this, getLowBits(), 128, false, "low_bits")) ||
      mlir::failed(verifyByteBlock(*this, getHighBits(), 64, false, "high_bits")) ||
      mlir::failed(verifyByteBlock(*this, getGroupScales(), 16, true,
                                  "group_scales")) ||
      mlir::failed(verifyByteBlock(*this, getActivation(), 256, true, "activation")))
    return mlir::failure();
  mlir::Type f32 = mlir::Float32Type::get(getContext());
  if (mlir::failed(requireScalar(*this, getWeightScale(), f32,
                                 "weight_scale")) ||
      mlir::failed(requireScalar(*this, getActivationScale(), f32,
                                 "activation_scale")))
    return emitOpError("weight_scale and activation_scale must be scalar f32");
  return verifyScalarDotResult(*this, getInit(), getResult());
}

void WEFTExtensionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/Extension/IR/ExtensionOps.cpp.inc"
      >();
}
