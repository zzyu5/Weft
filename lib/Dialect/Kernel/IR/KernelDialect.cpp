#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace weft::kernel;

#include "Weft/Dialect/Kernel/IR/KernelOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Weft/Dialect/Kernel/IR/KernelTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Kernel/IR/KernelOps.cpp.inc"

namespace {

bool isScalarType(mlir::Type type) {
  return type.isIndex() ||
         mlir::isa<mlir::IntegerType, mlir::FloatType>(type);
}

bool isIntegerLike(mlir::Type type) {
  return type.isIndex() || mlir::isa<mlir::IntegerType>(type);
}

std::optional<unsigned> fixedBitWidth(mlir::Type type) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  return std::nullopt;
}

mlir::Type unwrapMasked(mlir::Type type) {
  if (auto masked = mlir::dyn_cast<MaskedType>(type))
    return masked.getValueType();
  return type;
}

bool isMasked(mlir::Type type) { return mlir::isa<MaskedType>(type); }

mlir::Type elementTypeOf(mlir::Type type) {
  type = unwrapMasked(type);
  if (auto block = mlir::dyn_cast<BlockType>(type))
    return block.getElementType();
  if (auto region = mlir::dyn_cast<RegionType>(type))
    return region.getElementType();
  return type;
}

llvm::ArrayRef<int64_t> staticShapeOf(mlir::Type type) {
  type = unwrapMasked(type);
  if (auto block = mlir::dyn_cast<BlockType>(type))
    return block.getShape();
  if (auto region = mlir::dyn_cast<RegionType>(type))
    return region.getShape();
  return {};
}

enum class ShapeKind { Scalar, Block, Region };

ShapeKind shapeKindOf(mlir::Type type) {
  type = unwrapMasked(type);
  if (mlir::isa<RegionType>(type))
    return ShapeKind::Region;
  if (mlir::isa<BlockType>(type))
    return ShapeKind::Block;
  return ShapeKind::Scalar;
}

bool isLogicalValueType(mlir::Type type) {
  type = unwrapMasked(type);
  return isScalarType(type) || mlir::isa<BlockType, RegionType, TupleType>(type);
}

bool isPredicateType(mlir::Type type) {
  if (isMasked(type))
    return false;
  auto integer = mlir::dyn_cast<mlir::IntegerType>(elementTypeOf(type));
  return integer && integer.getWidth() == 1;
}

bool isPointwiseValueType(mlir::Type type) {
  type = unwrapMasked(type);
  if (auto block = mlir::dyn_cast<BlockType>(type))
    return isScalarType(block.getElementType());
  if (auto region = mlir::dyn_cast<RegionType>(type))
    return isScalarType(region.getElementType());
  return isScalarType(type);
}

bool isPointerValue(mlir::Type type) {
  type = unwrapMasked(type);
  return mlir::isa<PtrType>(elementTypeOf(type));
}

PtrType pointerTypeOf(mlir::Type type) {
  return mlir::dyn_cast<PtrType>(elementTypeOf(type));
}

bool containsRegionValue(mlir::Type type) {
  if (auto masked = mlir::dyn_cast<MaskedType>(type))
    return containsRegionValue(masked.getValueType());
  if (mlir::isa<RegionType>(type))
    return true;
  if (auto tuple = mlir::dyn_cast<TupleType>(type))
    return llvm::any_of(tuple.getTypes(), containsRegionValue);
  return false;
}

bool validDimension(int64_t dimension) {
  return dimension == -1 || dimension > 0;
}

std::optional<llvm::SmallVector<int64_t>>
broadcastStaticShape(mlir::Type lhsType, mlir::Type rhsType) {
  ShapeKind lhsKind = shapeKindOf(lhsType);
  ShapeKind rhsKind = shapeKindOf(rhsType);
  if (lhsKind == ShapeKind::Scalar)
    return llvm::SmallVector<int64_t>(staticShapeOf(rhsType));
  if (rhsKind == ShapeKind::Scalar)
    return llvm::SmallVector<int64_t>(staticShapeOf(lhsType));

  llvm::ArrayRef<int64_t> lhs = staticShapeOf(lhsType);
  llvm::ArrayRef<int64_t> rhs = staticShapeOf(rhsType);
  if (lhs.size() != rhs.size())
    return std::nullopt;
  llvm::SmallVector<int64_t> result;
  result.reserve(lhs.size());
  for (auto [a, b] : llvm::zip(lhs, rhs)) {
    if (a == b) {
      result.push_back(a);
    } else if (a == 1) {
      result.push_back(b);
    } else if (b == 1) {
      result.push_back(a);
    } else {
      return std::nullopt;
    }
  }
  return result;
}

ShapeKind broadcastKind(mlir::Type lhs, mlir::Type rhs) {
  ShapeKind lhsKind = shapeKindOf(lhs);
  ShapeKind rhsKind = shapeKindOf(rhs);
  if (lhsKind == ShapeKind::Region || rhsKind == ShapeKind::Region)
    return ShapeKind::Region;
  if (lhsKind == ShapeKind::Block || rhsKind == ShapeKind::Block)
    return ShapeKind::Block;
  return ShapeKind::Scalar;
}

mlir::LogicalResult requireKernelAncestor(mlir::Operation *operation) {
  if (operation->getParentOfType<KernelOp>())
    return mlir::success();
  return operation->emitOpError("must be nested in weft_kernel.kernel");
}

mlir::LogicalResult verifyResultShape(mlir::Operation *operation,
                                      mlir::Type resultType,
                                      ShapeKind expectedKind,
                                      llvm::ArrayRef<int64_t> expectedShape) {
  mlir::Type bare = unwrapMasked(resultType);
  if (shapeKindOf(bare) != expectedKind)
    return operation->emitOpError("result has the wrong logical shape kind");
  if (expectedKind != ShapeKind::Scalar &&
      staticShapeOf(bare) != expectedShape)
    return operation->emitOpError("result has the wrong logical shape");
  return mlir::success();
}

bool haveSameExtent(mlir::Value lhs, int64_t lhsAxis, mlir::Value rhs,
                    int64_t rhsAxis);

mlir::LogicalResult verifyPointwiseResult(mlir::Operation *operation,
                                          mlir::Value lhs, mlir::Value rhs,
                                          mlir::Type result,
                                          bool predicateResult = false) {
  mlir::Type lhsType = lhs.getType();
  mlir::Type rhsType = rhs.getType();
  if (!isPointwiseValueType(lhsType) || !isPointwiseValueType(rhsType) ||
      !isPointwiseValueType(result))
    return operation->emitOpError(
        "pointwise operands/results must be scalar, block, or VLA values");
  auto shape = broadcastStaticShape(lhsType, rhsType);
  if (!shape)
    return operation->emitOpError(
        "operands are not logical broadcast-compatible");
  if (mlir::failed(verifyResultShape(operation, result,
                                     broadcastKind(lhsType, rhsType), *shape)))
    return mlir::failure();
  auto lhsShape = staticShapeOf(lhsType);
  auto rhsShape = staticShapeOf(rhsType);
  if (shapeKindOf(lhsType) != ShapeKind::Scalar &&
      shapeKindOf(rhsType) != ShapeKind::Scalar) {
    for (int64_t axis = 0; axis < static_cast<int64_t>(lhsShape.size()); ++axis)
      if (lhsShape[axis] == -1 && rhsShape[axis] == -1 &&
          !haveSameExtent(lhs, axis, rhs, axis))
        return operation->emitOpError()
               << "cannot prove pointwise dynamic extent identity at axis "
               << axis;
  }
  bool expectedMasked = isMasked(lhsType) || isMasked(rhsType);
  if (isMasked(result) != expectedMasked)
    return operation->emitOpError(
        "result must preserve pointwise logical validity");
  mlir::Type lhsElement = elementTypeOf(lhsType);
  mlir::Type rhsElement = elementTypeOf(rhsType);
  mlir::Type resultElement = elementTypeOf(result);
  if (lhsElement != rhsElement)
    return operation->emitOpError("operand element types must match");
  if (predicateResult) {
    if (!isPredicateType(result))
      return operation->emitOpError("comparison result must contain i1");
  } else if (resultElement != lhsElement) {
    return operation->emitOpError("result element type must match operands");
  }
  return mlir::success();
}

mlir::LogicalResult verifyYieldBlock(mlir::Operation *owner, mlir::Block &block,
                                     mlir::TypeRange expectedTypes) {
  auto yield = mlir::dyn_cast<YieldOp>(block.getTerminator());
  if (!yield)
    return owner->emitOpError("region must terminate with weft_kernel.yield");
  if (yield->getNumOperands() != expectedTypes.size())
    return owner->emitOpError("region yield count does not match results");
  for (auto [value, type] : llvm::zip(yield->getOperands(), expectedTypes))
    if (value.getType() != type)
      return owner->emitOpError("region yield types do not match results");
  return mlir::success();
}

std::optional<int64_t> constantIndex(mlir::Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  if (!integer)
    return std::nullopt;
  return integer.getInt();
}

struct LogicalExtent {
  std::optional<int64_t> constant;
  mlir::Value identity;
};

bool sameExtent(const LogicalExtent &lhs, const LogicalExtent &rhs) {
  if (lhs.constant && rhs.constant)
    return lhs.constant == rhs.constant;
  return lhs.identity && rhs.identity && lhs.identity == rhs.identity;
}

std::optional<LogicalExtent>
deriveExtentImpl(mlir::Value value, int64_t axis,
                 llvm::DenseSet<mlir::Value> &visited);

std::optional<LogicalExtent>
deriveBroadcastExtent(mlir::Value lhs, mlir::Value rhs, int64_t axis,
                      llvm::DenseSet<mlir::Value> &visited) {
  ShapeKind lhsKind = shapeKindOf(lhs.getType());
  ShapeKind rhsKind = shapeKindOf(rhs.getType());
  if (lhsKind == ShapeKind::Scalar)
    return deriveExtentImpl(rhs, axis, visited);
  if (rhsKind == ShapeKind::Scalar)
    return deriveExtentImpl(lhs, axis, visited);
  auto lhsShape = staticShapeOf(lhs.getType());
  auto rhsShape = staticShapeOf(rhs.getType());
  if (axis < 0 || axis >= static_cast<int64_t>(lhsShape.size()) ||
      axis >= static_cast<int64_t>(rhsShape.size()))
    return std::nullopt;
  if (lhsShape[axis] == 1)
    return deriveExtentImpl(rhs, axis, visited);
  if (rhsShape[axis] == 1)
    return deriveExtentImpl(lhs, axis, visited);
  llvm::DenseSet<mlir::Value> lhsVisited = visited;
  llvm::DenseSet<mlir::Value> rhsVisited = visited;
  auto lhsExtent = deriveExtentImpl(lhs, axis, lhsVisited);
  auto rhsExtent = deriveExtentImpl(rhs, axis, rhsVisited);
  if (!lhsExtent || !rhsExtent || !sameExtent(*lhsExtent, *rhsExtent))
    return std::nullopt;
  return lhsExtent;
}

std::optional<LogicalExtent>
deriveExtentImpl(mlir::Value value, int64_t axis,
                 llvm::DenseSet<mlir::Value> &visited) {
  mlir::Type type = unwrapMasked(value.getType());
  auto shape = staticShapeOf(type);
  if (shapeKindOf(type) == ShapeKind::Scalar || axis < 0 ||
      axis >= static_cast<int64_t>(shape.size()) ||
      !visited.insert(value).second)
    return std::nullopt;
  if (shape[axis] != -1)
    return LogicalExtent{shape[axis], {}};

  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Operation *parent = argument.getOwner()->getParentOp();
    if (auto loop = mlir::dyn_cast_or_null<ForOp>(parent)) {
      if (argument.getArgNumber() == 0)
        return std::nullopt;
      auto init = loop.getInitArgs();
      unsigned carried = argument.getArgNumber() - 1;
      if (carried < init.size())
        return deriveExtentImpl(init[carried], axis, visited);
    }
    // The VLA coordinate itself is the canonical identity of its lexical
    // logical domain. Pointwise users preserve it through their producers.
    return LogicalExtent{std::nullopt, value};
  }

  mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  if (auto blockAxis = mlir::dyn_cast<BlockAxisOp>(definition))
    return axis == 0
               ? std::optional<LogicalExtent>(LogicalExtent{
                     constantIndex(blockAxis.getExtent()), blockAxis.getExtent()})
               : std::nullopt;
  if (auto full = mlir::dyn_cast<FullOp>(definition)) {
    if (axis >= static_cast<int64_t>(full.getExtents().size()))
      return std::nullopt;
    mlir::Value extent = full.getExtents()[axis];
    return LogicalExtent{constantIndex(extent), extent};
  }
  if (auto expand = mlir::dyn_cast<ExpandDimsOp>(definition)) {
    int64_t inserted = expand.getAxis();
    if (axis == inserted)
      return LogicalExtent{1, {}};
    return deriveExtentImpl(expand.getInput(),
                            axis < inserted ? axis : axis - 1, visited);
  }
  if (auto broadcast = mlir::dyn_cast<BroadcastToOp>(definition)) {
    if (axis >= static_cast<int64_t>(broadcast.getExtents().size()))
      return std::nullopt;
    mlir::Value extent = broadcast.getExtents()[axis];
    return LogicalExtent{constantIndex(extent), extent};
  }
  if (auto reshape = mlir::dyn_cast<ReshapeOp>(definition)) {
    if (axis >= static_cast<int64_t>(reshape.getExtents().size()))
      return std::nullopt;
    mlir::Value extent = reshape.getExtents()[axis];
    return LogicalExtent{constantIndex(extent), extent};
  }
  if (auto transpose = mlir::dyn_cast<TransposeOp>(definition))
    return deriveExtentImpl(transpose.getInput(), transpose.getPermutation()[axis],
                            visited);
  if (auto permute = mlir::dyn_cast<PermuteOp>(definition))
    return deriveExtentImpl(permute.getInput(), permute.getPermutation()[axis],
                            visited);
  if (auto unary = mlir::dyn_cast<UnaryOp>(definition))
    return deriveExtentImpl(unary.getInput(), axis, visited);
  if (auto cast = mlir::dyn_cast<CastOp>(definition))
    return deriveExtentImpl(cast.getInput(), axis, visited);
  if (auto bitcast = mlir::dyn_cast<BitcastOp>(definition))
    return deriveExtentImpl(bitcast.getInput(), axis, visited);
  if (auto widen = mlir::dyn_cast<WidenOp>(definition))
    return deriveExtentImpl(widen.getInput(), axis, visited);
  if (auto narrow = mlir::dyn_cast<NarrowOp>(definition))
    return deriveExtentImpl(narrow.getInput(), axis, visited);
  if (auto valid = mlir::dyn_cast<ValidOp>(definition))
    return deriveExtentImpl(valid.getInput(), axis, visited);
  if (auto fill = mlir::dyn_cast<FillOp>(definition))
    return deriveExtentImpl(fill.getInput(), axis, visited);
  if (auto load = mlir::dyn_cast<LoadOp>(definition))
    return deriveExtentImpl(load.getPointer(), axis, visited);
  if (auto scan = mlir::dyn_cast<ScanOp>(definition))
    return deriveExtentImpl(scan.getInput(), axis, visited);
  if (auto dot = mlir::dyn_cast<DotOp>(definition)) {
    bool lhsRegion = mlir::isa<RegionType>(unwrapMasked(dot.getLhs().getType()));
    bool rhsRegion = mlir::isa<RegionType>(unwrapMasked(dot.getRhs().getType()));
    if (lhsRegion || rhsRegion) {
      if (axis == 0)
        return deriveExtentImpl(lhsRegion ? dot.getLhs() : dot.getRhs(), 0,
                                visited);
      --axis;
    }
    if (!lhsRegion && axis == 0)
      return deriveExtentImpl(dot.getLhs(), 0, visited);
    return std::nullopt;
  }
  if (auto matmul = mlir::dyn_cast<MatmulOp>(definition)) {
    if (axis == 0)
      return deriveExtentImpl(matmul.getLhs(), 0, visited);
    if (axis == 1)
      return deriveExtentImpl(matmul.getRhs(), 1, visited);
    return std::nullopt;
  }
  if (auto binary = mlir::dyn_cast<BinaryOp>(definition))
    return deriveBroadcastExtent(binary.getLhs(), binary.getRhs(), axis,
                                 visited);
  if (auto compare = mlir::dyn_cast<CompareOp>(definition))
    return deriveBroadcastExtent(compare.getLhs(), compare.getRhs(), axis,
                                 visited);
  if (auto pointerAdd = mlir::dyn_cast<PtrAddOp>(definition))
    return deriveBroadcastExtent(pointerAdd.getBase(), pointerAdd.getOffset(),
                                 axis, visited);
  if (auto select = mlir::dyn_cast<SelectOp>(definition))
    return deriveBroadcastExtent(select.getTrueValue(), select.getFalseValue(),
                                 axis, visited);
  if (auto loop = mlir::dyn_cast<ForOp>(definition)) {
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    if (result && result.getResultNumber() < loop.getInitArgs().size())
      return deriveExtentImpl(loop.getInitArgs()[result.getResultNumber()], axis,
                              visited);
  }
  return std::nullopt;
}

std::optional<LogicalExtent> deriveExtent(mlir::Value value, int64_t axis) {
  llvm::DenseSet<mlir::Value> visited;
  return deriveExtentImpl(value, axis, visited);
}

bool haveSameExtent(mlir::Value lhs, int64_t lhsAxis, mlir::Value rhs,
                    int64_t rhsAxis) {
  auto lhsExtent = deriveExtent(lhs, lhsAxis);
  auto rhsExtent = deriveExtent(rhs, rhsAxis);
  return lhsExtent && rhsExtent && sameExtent(*lhsExtent, *rhsExtent);
}

mlir::LogicalResult verifyFootprint(mlir::Operation *operation,
                                    mlir::Value value, mlir::Value footprint,
                                    llvm::StringRef role) {
  if (shapeKindOf(value.getType()) == ShapeKind::Scalar)
    return mlir::success();
  auto broadcast = broadcastStaticShape(value.getType(), footprint.getType());
  auto valueShape = staticShapeOf(value.getType());
  auto footprintShape = staticShapeOf(footprint.getType());
  if (!broadcast ||
      broadcastKind(value.getType(), footprint.getType()) !=
          shapeKindOf(footprint.getType()) ||
      llvm::ArrayRef<int64_t>(*broadcast) != footprintShape)
    return operation->emitOpError() << role << " does not broadcast to the pointer";
  for (int64_t axis = 0; axis < static_cast<int64_t>(valueShape.size()); ++axis) {
    if (valueShape[axis] == -1 &&
        !haveSameExtent(value, axis, footprint, axis))
      return operation->emitOpError()
             << "cannot prove " << role << " dynamic extent identity at axis "
             << axis;
  }
  return mlir::success();
}

bool validOrder(llvm::StringRef order) {
  return llvm::StringSwitch<bool>(order)
      .Cases("ordered", "preserve", "relaxed", true)
      .Default(false);
}

bool validMath(llvm::StringRef mode) {
  return llvm::StringSwitch<bool>(mode)
      .Cases("strict", "native", "fast", true)
      .Default(false);
}

bool validRounding(llvm::StringRef mode) {
  return llvm::StringSwitch<bool>(mode)
      .Cases("rne", "rtz", "rdn", "rup", "rmm", true)
      .Default(false);
}

bool validAtomicOrder(llvm::StringRef order) {
  return llvm::StringSwitch<bool>(order)
      .Cases("relaxed", "acquire", "release", "acq_rel", "seq_cst", true)
      .Default(false);
}

} // namespace

bool weft::kernel::haveSameLogicalExtent(mlir::Value lhs, int64_t lhsAxis,
                                         mlir::Value rhs, int64_t rhsAxis) {
  return haveSameExtent(lhs, lhsAxis, rhs, rhsAxis);
}

mlir::LogicalResult PtrType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type elementType, llvm::StringRef addressSpace,
    llvm::StringRef access, bool, int64_t alignment, bool) {
  if (!isScalarType(elementType) || elementType.isIndex())
    return emitError() << "pointer element type must be a non-index scalar";
  if (addressSpace.empty())
    return emitError() << "pointer address space must not be empty";
  if (access != "read" && access != "write" && access != "readwrite")
    return emitError() << "pointer access must be read, write, or readwrite";
  if (alignment < 0 ||
      (alignment != 0 && !llvm::isPowerOf2_64(static_cast<uint64_t>(alignment))))
    return emitError() << "pointer alignment must be zero or a power of two";
  return mlir::success();
}

mlir::LogicalResult ConstexprType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type valueType) {
  if (!isScalarType(valueType))
    return emitError() << "constexpr must wrap a scalar or index type";
  return mlir::success();
}

mlir::LogicalResult BlockType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
  if (shape.empty())
    return emitError() << "logical block must have positive rank";
  if (llvm::any_of(shape, [](int64_t d) { return !validDimension(d); }))
    return emitError() << "block dimensions must be positive or -1";
  if (!isScalarType(elementType) && !mlir::isa<PtrType>(elementType))
    return emitError() << "block element must be scalar or pointer";
  return mlir::success();
}

mlir::LogicalResult RegionType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
  if (shape.empty() || shape.front() != -1)
    return emitError()
           << "region shape must begin with the active VLA dimension -1";
  if (llvm::any_of(shape, [](int64_t d) { return !validDimension(d); }))
    return emitError() << "region block dimensions must be positive or -1";
  if (!isScalarType(elementType) && !mlir::isa<PtrType>(elementType))
    return emitError() << "region element must be scalar or pointer";
  return mlir::success();
}

mlir::LogicalResult MaskedType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type valueType) {
  if (mlir::isa<MaskedType, TupleType, PtrType, ConstexprType>(valueType) ||
      !isLogicalValueType(valueType))
    return emitError() << "masked must wrap one scalar, block, or VLA value";
  return mlir::success();
}

mlir::LogicalResult TupleType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<mlir::Type> types) {
  if (types.empty())
    return emitError() << "state tuple must not be empty";
  if (llvm::any_of(types, [](mlir::Type type) {
        return mlir::isa<PtrType, ConstexprType>(type);
      }))
    return emitError() << "state tuple cannot contain ABI-only types";
  return mlir::success();
}

mlir::LogicalResult KernelOp::verify() {
  mlir::Block &entry = getBody().front();
  if (getArgNames().size() != entry.getNumArguments() ||
      getArgKinds().size() != entry.getNumArguments())
    return emitOpError("arg_names and arg_kinds must match entry arguments");
  llvm::DenseSet<llvm::StringRef> seen;
  for (auto [nameAttr, kindAttr, argument] :
       llvm::zip(getArgNames(), getArgKinds(), entry.getArguments())) {
    auto name = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
    auto kind = mlir::dyn_cast<mlir::StringAttr>(kindAttr);
    if (!name || name.getValue().empty() || !kind)
      return emitOpError("argument metadata must contain non-empty strings");
    if (!seen.insert(name.getValue()).second)
      return emitOpError("argument names must be unique");
    mlir::Type type = argument.getType();
    bool valid = kind.getValue() == "pointer" ? mlir::isa<PtrType>(type)
                 : kind.getValue() == "scalar" ? isScalarType(type)
                 : kind.getValue() == "constexpr"
                     ? mlir::isa<ConstexprType>(type)
                     : false;
    if (!valid)
      return emitOpError("argument kind is incompatible with its canonical type");
  }
  if (!mlir::isa<ReturnOp>(entry.getTerminator()))
    return emitOpError("entry must terminate with weft_kernel.return");
  mlir::Type returnType = getReturnType();
  if (!mlir::isa<mlir::NoneType>(returnType) && !isScalarType(returnType))
    return emitOpError("return_type must be none or one scalar type");
  auto returnOp = mlir::cast<ReturnOp>(entry.getTerminator());
  if (mlir::isa<mlir::NoneType>(returnType)) {
    if (!returnOp.getValues().empty())
      return emitOpError("void kernel cannot return a value");
  } else if (returnOp.getValues().size() != 1 ||
             returnOp.getValues().front().getType() != returnType) {
    return emitOpError("kernel return must match return_type");
  }
  return mlir::success();
}

mlir::LogicalResult ReturnOp::verify() {
  auto kernel =
      mlir::dyn_cast_or_null<KernelOp>(getOperation()->getBlock()->getParentOp());
  if (!kernel)
    return emitOpError("must terminate a kernel entry");
  mlir::Type returnType = kernel.getReturnType();
  if (mlir::isa<mlir::NoneType>(returnType)) {
    if (!getValues().empty())
      return emitOpError("void kernel cannot return a value");
  } else if (getValues().size() != 1 ||
             getValues().front().getType() != returnType) {
    return emitOpError("return value must match the kernel return_type");
  }
  return mlir::success();
}

mlir::LogicalResult YieldOp::verify() {
  mlir::Operation *parent = getOperation()->getBlock()->getParentOp();
  if (!parent ||
      !mlir::isa<IfOp, ForOp, WhileOp, VLAOp, SummaryFoldOp>(parent))
    return emitOpError("must terminate a structured Weft region");
  return mlir::success();
}

mlir::LogicalResult ConditionOp::verify() {
  mlir::Operation *parent = getOperation()->getBlock()->getParentOp();
  if (!mlir::isa_and_nonnull<WhileOp>(parent))
    return emitOpError("must terminate a weft_kernel.while condition region");
  return mlir::success();
}

mlir::LogicalResult ConstantOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  auto typed = mlir::dyn_cast<mlir::TypedAttr>(getValue());
  if (!typed || typed.getType() != getResult().getType() ||
      !isScalarType(getResult().getType()))
    return emitOpError("typed scalar value must match the result type");
  return mlir::success();
}

mlir::LogicalResult MetaValueOp::verify() {
  if (getInput().getType().getValueType() != getResult().getType())
    return emitOpError("result must match the constexpr value type");
  return mlir::success();
}

mlir::LogicalResult IfOp::verify() {
  if (getThenRegion().front().getNumArguments() != 0 ||
      getElseRegion().front().getNumArguments() != 0)
    return emitOpError("if regions cannot have block arguments");
  if (mlir::failed(verifyYieldBlock(getOperation(), getThenRegion().front(),
                                    getResultTypes())) ||
      mlir::failed(verifyYieldBlock(getOperation(), getElseRegion().front(),
                                    getResultTypes())))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult ForOp::verify() {
  auto init = getInitArgs();
  if (init.size() != getNumResults())
    return emitOpError("init count must match result count");
  for (auto [value, result] : llvm::zip(init, getResults()))
    if (value.getType() != result.getType())
      return emitOpError("init and result types must match");
  mlir::Block &body = getBody().front();
  if (body.getNumArguments() != init.size() + 1 ||
      !body.getArgument(0).getType().isIndex())
    return emitOpError("body arguments must be index then carried values");
  for (auto [argument, value] :
       llvm::zip(body.getArguments().drop_front(), init))
    if (argument.getType() != value.getType())
      return emitOpError("carried block arguments must match init types");
  if (mlir::failed(verifyYieldBlock(getOperation(), body, getResultTypes())))
    return mlir::failure();
  if (auto step = constantIndex(getStep()); step && *step == 0)
    return emitOpError("step must not be zero");
  return mlir::success();
}

mlir::LogicalResult WhileOp::verify() {
  auto init = getInitArgs();
  if (init.size() != getNumResults())
    return emitOpError("init count must match result count");
  for (auto [value, result] : llvm::zip(init, getResults()))
    if (value.getType() != result.getType())
      return emitOpError("init and result types must match");
  mlir::Block &condition = getConditionRegion().front();
  mlir::Block &body = getBodyRegion().front();
  if (condition.getNumArguments() != init.size() ||
      body.getNumArguments() != init.size())
    return emitOpError("while regions must receive every carried value");
  for (unsigned i = 0; i < init.size(); ++i)
    if (condition.getArgument(i).getType() != init[i].getType() ||
        body.getArgument(i).getType() != init[i].getType())
      return emitOpError("while carried argument types must match init types");
  auto terminator = mlir::dyn_cast<ConditionOp>(condition.getTerminator());
  if (!terminator || terminator.getValues().size() != init.size())
    return emitOpError("condition region must forward every carried value");
  for (auto [value, result] : llvm::zip(terminator.getValues(), getResults()))
    if (value.getType() != result.getType())
      return emitOpError("condition forwarded types must match results");
  return verifyYieldBlock(getOperation(), body, getResultTypes());
}

mlir::LogicalResult VLAOp::verify() {
  if (getOperation()->getParentOfType<VLAOp>())
    return emitOpError("VLA regions cannot be nested");
  mlir::Block &body = getBody().front();
  if (body.getNumArguments() != 1)
    return emitOpError("VLA body must receive exactly one logical coordinate");
  auto coordinate = mlir::dyn_cast<RegionType>(body.getArgument(0).getType());
  if (!coordinate || coordinate.getShape().size() != 1 ||
      coordinate.getShape().front() != -1 ||
      !coordinate.getElementType().isIndex())
    return emitOpError(
        "VLA coordinate must have !weft_kernel.region<[-1], index>");
  if (llvm::any_of(getResultTypes(), containsRegionValue))
    return emitOpError("values retaining the active VLA axis cannot escape");
  return verifyYieldBlock(getOperation(), body, getResultTypes());
}

mlir::LogicalResult BlockAxisOp::verify() {
  auto result = getResult().getType();
  if (result.getShape().size() != 1 || !result.getElementType().isIndex())
    return emitOpError("result must be a rank-one logical index block");
  if (auto extent = constantIndex(getExtent())) {
    if (*extent <= 0)
      return emitOpError("extent must be positive");
    if (result.getShape().front() != *extent)
      return emitOpError("static result extent must match the operand");
  } else if (result.getShape().front() != -1) {
    return emitOpError("non-constant extent requires -1 in the block type");
  }
  return mlir::success();
}

mlir::LogicalResult FullOp::verify() {
  auto result = getResult().getType();
  if (getShape().size() != result.getShape().size() ||
      getExtents().size() != result.getShape().size())
    return emitOpError("shape, extents and result rank must match");
  if (getShape() != result.getShape())
    return emitOpError("shape attribute must match result type");
  if (getValue().getType() != result.getElementType())
    return emitOpError("fill value must match result element type");
  return mlir::success();
}

mlir::LogicalResult ExpandDimsOp::verify() {
  ShapeKind kind = shapeKindOf(getInput().getType());
  if (kind == ShapeKind::Scalar)
    return emitOpError("input must be a block or VLA value");
  int64_t axis = getAxis();
  auto inputShape = staticShapeOf(getInput().getType());
  if (axis < 0 || axis > static_cast<int64_t>(inputShape.size()))
    return emitOpError("axis is outside [0, rank]");
  llvm::SmallVector<int64_t> expected(inputShape);
  expected.insert(expected.begin() + axis, 1);
  if (mlir::failed(verifyResultShape(getOperation(), getResult().getType(), kind,
                                     expected)) ||
      elementTypeOf(getInput().getType()) != elementTypeOf(getResult().getType()) ||
      isMasked(getInput().getType()) != isMasked(getResult().getType()))
    return emitOpError("expand_dims must only insert one singleton axis");
  return mlir::success();
}

mlir::LogicalResult BroadcastToOp::verify() {
  if (getShape().size() != getExtents().size() ||
      getShape() != staticShapeOf(getResult().getType()))
    return emitOpError("shape operands, attribute and result must agree");
  if (shapeKindOf(getInput().getType()) != shapeKindOf(getResult().getType()) ||
      elementTypeOf(getInput().getType()) != elementTypeOf(getResult().getType()) ||
      isMasked(getInput().getType()) != isMasked(getResult().getType()))
    return emitOpError("broadcast_to must preserve value kind, element and validity");
  return mlir::success();
}

mlir::LogicalResult ReshapeOp::verify() {
  if (getShape().size() != getExtents().size() ||
      getShape() != staticShapeOf(getResult().getType()))
    return emitOpError("shape operands, attribute and result must agree");
  if (shapeKindOf(getInput().getType()) != shapeKindOf(getResult().getType()) ||
      elementTypeOf(getInput().getType()) != elementTypeOf(getResult().getType()) ||
      isMasked(getInput().getType()) != isMasked(getResult().getType()))
    return emitOpError("reshape must preserve value kind, element and validity");
  return mlir::success();
}

static mlir::LogicalResult verifyPermutation(mlir::Operation *operation,
                                             mlir::Value input,
                                             mlir::Value result,
                                             llvm::ArrayRef<int64_t> permutation) {
  auto shape = staticShapeOf(input.getType());
  if (permutation.size() != shape.size())
    return operation->emitOpError("permutation length must match logical rank");
  llvm::SmallVector<int64_t> expected;
  llvm::DenseSet<int64_t> seen;
  for (int64_t axis : permutation) {
    if (axis < 0 || axis >= static_cast<int64_t>(shape.size()) ||
        !seen.insert(axis).second)
      return operation->emitOpError("permutation must contain every axis once");
    expected.push_back(shape[axis]);
  }
  if (mlir::failed(verifyResultShape(operation, result.getType(),
                                     shapeKindOf(input.getType()), expected)) ||
      elementTypeOf(input.getType()) != elementTypeOf(result.getType()) ||
      isMasked(input.getType()) != isMasked(result.getType()))
    return operation->emitOpError("permutation result type is inconsistent");
  return mlir::success();
}

mlir::LogicalResult TransposeOp::verify() {
  return verifyPermutation(getOperation(), getInput(), getResult(),
                           getPermutation());
}

mlir::LogicalResult PtrAddOp::verify() {
  if (!isPointerValue(getBase().getType()) ||
      !isIntegerLike(elementTypeOf(getOffset().getType())) ||
      !isPointerValue(getResult().getType()))
    return emitOpError("requires pointer base, index offset and pointer result");
  auto shape = broadcastStaticShape(getBase().getType(), getOffset().getType());
  if (!shape ||
      mlir::failed(verifyResultShape(getOperation(), getResult().getType(),
                                     broadcastKind(getBase().getType(),
                                                   getOffset().getType()),
                                     *shape)) ||
      pointerTypeOf(getBase().getType()) != pointerTypeOf(getResult().getType()))
    return emitOpError("pointer result must preserve pointer facts and joined shape");
  return mlir::success();
}

mlir::LogicalResult UnaryOp::verify() {
  if (!llvm::StringSwitch<bool>(getKind())
           .Cases("neg", "exp", "exp2", "log", "sqrt", "rsqrt", "sin", true)
           .Cases("cos", "floor", true)
           .Default(false))
    return emitOpError("unsupported unary kind");
  if (!validMath(getMath()))
    return emitOpError("math must be strict, native, or fast");
  if (getInput().getType() != getResult().getType())
    return emitOpError("unary operation must preserve type and validity");
  if (getKind() != "neg" && !mlir::isa<mlir::FloatType>(elementTypeOf(getInput().getType())))
    return emitOpError("transcendental operation requires floating elements");
  return mlir::success();
}

mlir::LogicalResult BinaryOp::verify() {
  if (!llvm::StringSwitch<bool>(getKind())
           .Cases("add", "sub", "mul", "div", "mod", true)
           .Cases("and", "or", "xor", "shl", "shr", "max", "min", true)
           .Default(false))
    return emitOpError("unsupported binary kind");
  if ((getKind() == "and" || getKind() == "or" || getKind() == "xor" ||
       getKind() == "shl" || getKind() == "shr") &&
      !isIntegerLike(elementTypeOf(getLhs().getType())))
    return emitOpError("bitwise operations require integer elements");
  return verifyPointwiseResult(getOperation(), getLhs(), getRhs(),
                               getResult().getType());
}

mlir::LogicalResult CompareOp::verify() {
  if (!llvm::StringSwitch<bool>(getPredicate())
           .Cases("eq", "ne", "lt", "le", "gt", "ge", true)
           .Default(false))
    return emitOpError("unsupported comparison predicate");
  return verifyPointwiseResult(getOperation(), getLhs(), getRhs(),
                               getResult().getType(), true);
}

mlir::LogicalResult CastOp::verify() {
  if (shapeKindOf(getInput().getType()) != shapeKindOf(getResult().getType()) ||
      staticShapeOf(getInput().getType()) != staticShapeOf(getResult().getType()) ||
      isMasked(getInput().getType()) != isMasked(getResult().getType()) ||
      !isScalarType(elementTypeOf(getInput().getType())) ||
      !isScalarType(elementTypeOf(getResult().getType())))
    return emitOpError("cast must preserve logical shape and validity");
  return mlir::success();
}

mlir::LogicalResult BitcastOp::verify() {
  if (shapeKindOf(getInput().getType()) != shapeKindOf(getResult().getType()) ||
      staticShapeOf(getInput().getType()) != staticShapeOf(getResult().getType()) ||
      isMasked(getInput().getType()) != isMasked(getResult().getType()))
    return emitOpError("bitcast must preserve logical shape and validity");
  auto inputWidth = fixedBitWidth(elementTypeOf(getInput().getType()));
  auto resultWidth = fixedBitWidth(elementTypeOf(getResult().getType()));
  if (!inputWidth || !resultWidth || inputWidth != resultWidth)
    return emitOpError("bitcast element types must have equal fixed width");
  return mlir::success();
}

mlir::LogicalResult SelectOp::verify() {
  if (!isPredicateType(getPredicate().getType()))
    return emitOpError("predicate must contain i1");
  if (mlir::failed(verifyPointwiseResult(getOperation(),
                                         getTrueValue(), getFalseValue(),
                                         getResult().getType())))
    return mlir::failure();
  return verifyFootprint(getOperation(), getPredicate(), getResult(),
                         "predicate");
}

mlir::LogicalResult TupleOp::verify() {
  if (getValues().empty() || getValues().size() != getResult().getType().getTypes().size())
    return emitOpError("tuple values and result fields must match");
  for (auto [value, type] :
       llvm::zip(getValues(), getResult().getType().getTypes()))
    if (value.getType() != type)
      return emitOpError("tuple field type mismatch");
  return mlir::success();
}

mlir::LogicalResult TupleGetOp::verify() {
  int64_t index = getIndex();
  auto types = getInput().getType().getTypes();
  if (index < 0 || index >= static_cast<int64_t>(types.size()) ||
      getResult().getType() != types[index])
    return emitOpError("tuple index or result type is invalid");
  return mlir::success();
}

mlir::LogicalResult SpecialValueOp::verify() {
  if (getKind() != "neg_inf" ||
      !mlir::isa<mlir::FloatType>(getResult().getType()))
    return emitOpError("only floating neg_inf is supported");
  return mlir::success();
}

mlir::LogicalResult LoadOp::verify() {
  if (!isPointerValue(getPointer().getType()) ||
      !isPredicateType(getWhere().getType()))
    return emitOpError("load requires a pointer and logical predicate");
  PtrType pointer = pointerTypeOf(getPointer().getType());
  if (pointer.getAccess() == "write")
    return emitOpError("cannot load through a writeonly pointer");
  auto bareResult = unwrapMasked(getResult().getType());
  if (elementTypeOf(bareResult) != pointer.getElementType() ||
      shapeKindOf(bareResult) != shapeKindOf(getPointer().getType()) ||
      staticShapeOf(bareResult) != staticShapeOf(getPointer().getType()))
    return emitOpError("result must follow the pointer footprint and element type");
  bool noOther = mlir::isa<mlir::NoneType>(getOther().getType());
  if (noOther != isMasked(getResult().getType()))
    return emitOpError("omitted other must produce a validity-carrying result");
  if (!noOther && elementTypeOf(getOther().getType()) != pointer.getElementType())
    return emitOpError("other element type must match the pointer");
  if (mlir::failed(verifyFootprint(getOperation(), getWhere(), getPointer(),
                                   "where")) ||
      (!noOther &&
       mlir::failed(verifyFootprint(getOperation(), getOther(), getPointer(),
                                    "other"))))
    return mlir::failure();
  if (getAlignment() < 0)
    return emitOpError("alignment must be non-negative");
  return mlir::success();
}

mlir::LogicalResult StoreOp::verify() {
  if (!isPointerValue(getPointer().getType()) ||
      !isPredicateType(getWhere().getType()))
    return emitOpError("store requires a pointer and logical predicate");
  PtrType pointer = pointerTypeOf(getPointer().getType());
  if (pointer.getAccess() == "read")
    return emitOpError("cannot store through a readonly pointer");
  if (elementTypeOf(getValue().getType()) != pointer.getElementType())
    return emitOpError("stored element type must match the pointer");
  if (mlir::failed(verifyFootprint(getOperation(), getWhere(), getPointer(),
                                   "where")) ||
      mlir::failed(verifyFootprint(getOperation(), getValue(), getPointer(),
                                   "value")))
    return mlir::failure();
  if (getAlignment() < 0)
    return emitOpError("alignment must be non-negative");
  return mlir::success();
}

mlir::LogicalResult SortIndicesOp::verify() {
  PtrType input = getInput().getType();
  PtrType output = getOutput().getType();
  if (!input.getElementType().isF32() || input.getAccess() == "write")
    return emitOpError("input must be a readable f32 pointer");
  if (!output.getElementType().isUnsignedInteger(32) ||
      output.getAccess() == "read")
    return emitOpError("output must be a writable u32 pointer");
  if (getOrder() != "ascending" && getOrder() != "descending")
    return emitOpError("order must be ascending or descending");
  if (getNan() != "last")
    return emitOpError("nan must be last");
  if (getTie() != "index_ascending")
    return emitOpError("tie must be index_ascending");
  return mlir::success();
}

mlir::LogicalResult PrefetchOp::verify() {
  if (!isPointerValue(getPointer().getType()) ||
      !isPredicateType(getWhere().getType()))
    return emitOpError("prefetch requires a pointer and predicate");
  if (!llvm::StringSwitch<bool>(getLocality())
           .Cases("default", "low", "moderate", "high", true)
           .Default(false))
    return emitOpError("unsupported locality");
  if (mlir::failed(verifyFootprint(getOperation(), getWhere(), getPointer(),
                                   "where")))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult AtomicAddOp::verify() {
  if (!isPointerValue(getPointer().getType()) ||
      elementTypeOf(getValue().getType()) != pointerTypeOf(getPointer().getType()).getElementType() ||
      !isPredicateType(getWhere().getType()) ||
      elementTypeOf(getResult().getType()) != elementTypeOf(getValue().getType()) ||
      !validAtomicOrder(getOrder()))
    return emitOpError("invalid atomic_add types or memory order");
  if (mlir::failed(verifyFootprint(getOperation(), getWhere(), getPointer(),
                                   "where")) ||
      mlir::failed(verifyFootprint(getOperation(), getValue(), getPointer(),
                                   "value")))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult FenceOp::verify() {
  if (!validAtomicOrder(getOrder()) || getOrder() == "relaxed")
    return emitOpError("fence order must be acquire, release, acq_rel, or seq_cst");
  return mlir::success();
}

mlir::LogicalResult ValidOp::verify() {
  mlir::Type value = getInput().getType().getValueType();
  mlir::Type result = getResult().getType();
  if (!isPredicateType(result) || shapeKindOf(value) != shapeKindOf(result) ||
      staticShapeOf(value) != staticShapeOf(result))
    return emitOpError("valid must return an i1 predicate over the same shape");
  return mlir::success();
}

mlir::LogicalResult FillOp::verify() {
  mlir::Type value = getInput().getType().getValueType();
  if (getResult().getType() != value ||
      elementTypeOf(getFillValue().getType()) != elementTypeOf(value))
    return emitOpError("fill must remove validity without changing value type");
  return mlir::success();
}

static mlir::Type stateInputType(mlir::Type type) {
  return unwrapMasked(type);
}

mlir::LogicalResult ReduceOp::verify() {
  if (!validOrder(getOrder()))
    return emitOpError("invalid reduction order");
  if (!llvm::StringSwitch<bool>(getKind())
           .Cases("add", "max", "min", "mul", "and", "or", true)
           .Default(false))
    return emitOpError("unsupported reduction operation");
  mlir::Type input = stateInputType(getInput().getType());
  mlir::Type accType = getAccDtype();
  if (!isScalarType(accType) || getIdentity().getType() != accType ||
      !isPredicateType(getWhere().getType()))
    return emitOpError("identity/accumulator/predicate types are inconsistent");
  ShapeKind kind = shapeKindOf(input);
  auto shape = staticShapeOf(input);
  int64_t axis = getAxis();
  ShapeKind resultKind;
  llvm::SmallVector<int64_t> resultShape(shape);
  if (kind == ShapeKind::Region && axis == -1) {
    resultShape.erase(resultShape.begin());
    resultKind = resultShape.empty() ? ShapeKind::Scalar : ShapeKind::Block;
  } else if (kind == ShapeKind::Block && axis >= 0 &&
             axis < static_cast<int64_t>(shape.size())) {
    resultShape.erase(resultShape.begin() + axis);
    resultKind = resultShape.empty() ? ShapeKind::Scalar : ShapeKind::Block;
  } else {
    return emitOpError("axis does not select the active VLA or a block axis");
  }
  if (mlir::failed(verifyResultShape(getOperation(), getResult().getType(),
                                     resultKind, resultShape)) ||
      elementTypeOf(getResult().getType()) != accType ||
      isMasked(getResult().getType()))
    return emitOpError("reduction result type is inconsistent");
  if (mlir::failed(verifyFootprint(getOperation(), getWhere(), getInput(),
                                   "where")))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult ScanOp::verify() {
  if (!validOrder(getOrder()) || !isPredicateType(getWhere().getType()) ||
      getIdentity().getType() != getAccDtype())
    return emitOpError("invalid scan order, identity, or predicate");
  mlir::Type input = stateInputType(getInput().getType());
  if (shapeKindOf(input) == ShapeKind::Scalar ||
      shapeKindOf(getResult().getType()) != shapeKindOf(input) ||
      staticShapeOf(getResult().getType()) != staticShapeOf(input) ||
      elementTypeOf(getResult().getType()) != getAccDtype())
    return emitOpError("scan must produce prefixes over the input domain");
  if (!mlir::isa<mlir::NoneType>(getSegmentStart().getType()) &&
      !isPredicateType(getSegmentStart().getType()))
    return emitOpError("segment_start must be omitted or a predicate");
  if (mlir::failed(verifyFootprint(getOperation(), getWhere(), getInput(),
                                   "where")) ||
      (!mlir::isa<mlir::NoneType>(getSegmentStart().getType()) &&
       mlir::failed(verifyFootprint(getOperation(), getSegmentStart(),
                                    getInput(), "segment_start"))))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult SummaryFoldOp::verify() {
  if (!validOrder(getOrder()) || !isPredicateType(getWhere().getType()))
    return emitOpError("invalid summary order or predicate");
  mlir::Type input = stateInputType(getInput().getType());
  if (shapeKindOf(input) == ShapeKind::Scalar)
    return emitOpError("summary_fold input must have a logical domain");
  if (mlir::failed(verifyFootprint(getOperation(), getWhere(), getInput(),
                                   "where")))
    return mlir::failure();
  bool hasCoordinate =
      !mlir::isa<mlir::NoneType>(getCoordinate().getType());
  if (hasCoordinate &&
      mlir::failed(verifyFootprint(getOperation(), getCoordinate(), getInput(),
                                   "coordinate")))
    return mlir::failure();
  mlir::Type element = elementTypeOf(input);
  mlir::Type state = getIdentity().getType();
  mlir::Block &lift = getLift().front();
  mlir::Block &merge = getMerge().front();
  mlir::Block &finalize = getFinalize().front();
  if (lift.getNumArguments() != (hasCoordinate ? 2U : 1U) ||
      lift.getArgument(0).getType() != element ||
      (hasCoordinate && lift.getArgument(1).getType() !=
                            elementTypeOf(getCoordinate().getType())) ||
      merge.getNumArguments() != 2 ||
      merge.getArgument(0).getType() != state ||
      merge.getArgument(1).getType() != state ||
      finalize.getNumArguments() != 1 ||
      finalize.getArgument(0).getType() != state)
    return emitOpError("lift/merge/finalize argument types are inconsistent");
  if (mlir::failed(verifyYieldBlock(getOperation(), lift, {state})) ||
      mlir::failed(verifyYieldBlock(getOperation(), merge, {state})) ||
      mlir::failed(verifyYieldBlock(getOperation(), finalize,
                                    getOperation()->getResultTypes())))
    return mlir::failure();
  for (mlir::Region *region : {&getLift(), &getMerge(), &getFinalize()}) {
    for (mlir::Operation &nested : region->front().getOperations()) {
      if (mlir::isa<YieldOp>(nested))
        continue;
      if (!mlir::isMemoryEffectFree(&nested))
        return emitOpError(
            "lift/merge/finalize regions must contain only pure operations");
    }
  }
  return mlir::success();
}

mlir::LogicalResult ArgMaxOp::verify() {
  if (shapeKindOf(getInput().getType()) == ShapeKind::Scalar ||
      elementTypeOf(getInput().getType()).isF32() == false ||
      elementTypeOf(getCoordinate().getType()).isIndex() == false ||
      shapeKindOf(getInput().getType()) !=
          shapeKindOf(getCoordinate().getType()) ||
      staticShapeOf(getInput().getType()) !=
          staticShapeOf(getCoordinate().getType()))
    return emitOpError("input and coordinate must be matching f32/index domains");
  auto result = getResult().getType().getTypes();
  if (result.size() != 2 || !result[0].isF32() || !result[1].isIndex())
    return emitOpError("result must be tuple<f32,index>");
  if (getTie() != "lowest_coordinate" || getOrder() != "relaxed")
    return emitOpError(
        "argmax requires lowest-coordinate tie and relaxed order");
  return mlir::success();
}

mlir::LogicalResult OnlineSoftmaxSummaryOp::verify() {
  if (shapeKindOf(getInput().getType()) == ShapeKind::Scalar ||
      !elementTypeOf(getInput().getType()).isF32())
    return emitOpError("input must be a logical f32 domain");
  auto result = getResult().getType().getTypes();
  if (result.size() != 2 || !result[0].isF32() || !result[1].isF32())
    return emitOpError("result must be tuple<f32,f32>");
  if (getMath() != "native" || getOrder() != "preserve")
    return emitOpError("online softmax summary requires native math and preserve order");
  return mlir::success();
}

static mlir::LogicalResult verifyProductResult(
    mlir::Operation *operation, mlir::Value lhs, mlir::Value rhs,
    mlir::Value init, mlir::Value result, mlir::Type accType,
    llvm::StringRef order, llvm::StringRef math,
    llvm::ArrayRef<int64_t> outputShape, ShapeKind resultKind,
    int64_t lhsReductionAxis, int64_t rhsReductionAxis) {
  if (!validOrder(order) || !validMath(math))
    return operation->emitOpError("invalid structured-product order or math mode");
  mlir::Type lhsElement = elementTypeOf(unwrapMasked(lhs.getType()));
  mlir::Type rhsElement = elementTypeOf(unwrapMasked(rhs.getType()));
  if (!mlir::isa<mlir::FloatType>(lhsElement) || lhsElement != rhsElement ||
      !mlir::isa<mlir::FloatType>(accType))
    return operation->emitOpError(
        "dot/matmul operands must share a floating element type and use a floating accumulator");
  auto lhsShape = staticShapeOf(unwrapMasked(lhs.getType()));
  auto rhsShape = staticShapeOf(unwrapMasked(rhs.getType()));
  if (lhsShape[lhsReductionAxis] != rhsShape[rhsReductionAxis] ||
      (lhsShape[lhsReductionAxis] == -1 &&
       !haveSameExtent(lhs, lhsReductionAxis, rhs, rhsReductionAxis)))
    return operation->emitOpError(
        "dot/matmul reduction extents must have the same logical identity");
  mlir::Type bareInit = unwrapMasked(init.getType());
  bool scalarInit = shapeKindOf(bareInit) == ShapeKind::Scalar;
  bool shapedInit = shapeKindOf(bareInit) == resultKind &&
                    staticShapeOf(bareInit) == outputShape;
  if (mlir::failed(verifyResultShape(operation, result.getType(), resultKind,
                                     outputShape)) ||
      elementTypeOf(result.getType()) != accType ||
      elementTypeOf(init.getType()) != accType || (!scalarInit && !shapedInit))
    return operation->emitOpError(
        "dot/matmul result and explicit accumulator init are inconsistent");
  return mlir::success();
}

mlir::LogicalResult DotOp::verify() {
  mlir::Type lhsType = unwrapMasked(getLhs().getType());
  mlir::Type rhsType = unwrapMasked(getRhs().getType());
  auto lhsBlock = mlir::dyn_cast<BlockType>(lhsType);
  auto lhsRegion = mlir::dyn_cast<RegionType>(lhsType);
  auto rhsBlock = mlir::dyn_cast<BlockType>(rhsType);
  auto rhsRegion = mlir::dyn_cast<RegionType>(rhsType);
  bool rowVector = (lhsBlock || lhsRegion) && rhsBlock &&
                   staticShapeOf(lhsType).size() == 2 &&
                   staticShapeOf(rhsType).size() == 1;
  bool rowBank = lhsBlock && rhsRegion &&
                 staticShapeOf(lhsType).size() == 2 &&
                 staticShapeOf(rhsType).size() == 2;
  if (!rowVector && !rowBank)
    return emitOpError(
        "dot requires [R,K] x [K], [VLA,K] x [K], or [R,K] x [VLA,K]");
  llvm::SmallVector<int64_t> outputShape;
  ShapeKind resultKind;
  if (lhsRegion || rhsRegion) {
    outputShape.push_back(-1);
    if (lhsBlock)
      outputShape.push_back(lhsBlock.getShape()[0]);
    resultKind = ShapeKind::Region;
  } else {
    outputShape.push_back(lhsBlock.getShape()[0]);
    resultKind = ShapeKind::Block;
  }
  return verifyProductResult(getOperation(), getLhs(), getRhs(), getInit(),
                             getResult(), getAccDtype(), getOrder(), getMath(),
                             outputShape, resultKind, 1,
                             staticShapeOf(rhsType).size() - 1);
}

mlir::LogicalResult MatmulOp::verify() {
  auto lhs = mlir::dyn_cast<BlockType>(unwrapMasked(getLhs().getType()));
  auto rhs = mlir::dyn_cast<BlockType>(unwrapMasked(getRhs().getType()));
  if (!lhs || !rhs || lhs.getShape().size() != 2 || rhs.getShape().size() != 2)
    return emitOpError("matmul requires local rank-two [M,K] x [K,N] blocks");
  llvm::SmallVector<int64_t> outputShape{lhs.getShape()[0], rhs.getShape()[1]};
  return verifyProductResult(getOperation(), getLhs(), getRhs(), getInit(),
                             getResult(), getAccDtype(), getOrder(), getMath(),
                             outputShape, ShapeKind::Block, 1, 0);
}

mlir::LogicalResult PermuteOp::verify() {
  return verifyPermutation(getOperation(), getInput(), getResult(),
                           getPermutation());
}

mlir::LogicalResult LookupOp::verify() {
  if (!mlir::isa<BlockType>(unwrapMasked(getTable().getType())) ||
      !isIntegerLike(elementTypeOf(getIndices().getType())) ||
      !isPredicateType(getWhere().getType()) ||
      shapeKindOf(getResult().getType()) != shapeKindOf(getIndices().getType()) ||
      staticShapeOf(getResult().getType()) != staticShapeOf(getIndices().getType()))
    return emitOpError("lookup indices, predicate, and result footprint are invalid");
  if (mlir::failed(verifyFootprint(getOperation(), getWhere(), getIndices(),
                                   "where")))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult DecodeOp::verify() {
  if (!mlir::isa<BlockType>(unwrapMasked(getTable().getType())) ||
      !isIntegerLike(elementTypeOf(getCodes().getType())) ||
      !isPredicateType(getWhere().getType()) ||
      elementTypeOf(getResult().getType()) != getOutDtype() ||
      shapeKindOf(getResult().getType()) != shapeKindOf(getCodes().getType()) ||
      staticShapeOf(getResult().getType()) != staticShapeOf(getCodes().getType()))
    return emitOpError("decode codes, predicate, output dtype, and shape are invalid");
  if (mlir::failed(verifyFootprint(getOperation(), getWhere(), getCodes(),
                                   "where")))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult WidenOp::verify() {
  if (!isPointwiseValueType(getInput().getType()) ||
      !isPointwiseValueType(getResult().getType()) ||
      shapeKindOf(getInput().getType()) != shapeKindOf(getResult().getType()) ||
      staticShapeOf(getInput().getType()) != staticShapeOf(getResult().getType()) ||
      isMasked(getInput().getType()) != isMasked(getResult().getType()))
    return emitOpError("widen must preserve shape and validity");
  return mlir::success();
}

mlir::LogicalResult NarrowOp::verify() {
  if (!validRounding(getRounding()) ||
      !isPointwiseValueType(getInput().getType()) ||
      !isPointwiseValueType(getResult().getType()) ||
      shapeKindOf(getInput().getType()) != shapeKindOf(getResult().getType()) ||
      staticShapeOf(getInput().getType()) != staticShapeOf(getResult().getType()) ||
      isMasked(getInput().getType()) != isMasked(getResult().getType()))
    return emitOpError("narrow must preserve shape/validity and use a known rounding mode");
  return mlir::success();
}

void WEFTKernelDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Weft/Dialect/Kernel/IR/KernelTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/Kernel/IR/KernelOps.cpp.inc"
      >();
}
