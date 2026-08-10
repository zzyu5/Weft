#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"

#include <optional>

using namespace weft::kernel;

#include "Weft/Dialect/Kernel/IR/KernelOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Weft/Dialect/Kernel/IR/KernelTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Kernel/IR/KernelOps.cpp.inc"

namespace {

bool isScalarDataType(mlir::Type type) {
  return type.isIndex() || mlir::isa<mlir::IntegerType, mlir::FloatType>(type);
}

mlir::LogicalResult requireKernelAncestor(mlir::Operation *operation) {
  if (operation->getParentOfType<KernelOp>())
    return mlir::success();
  return operation->emitOpError("must be nested in a weft_kernel.kernel");
}

bool isIntegerLike(mlir::Type type) {
  return type.isIndex() || mlir::isa<mlir::IntegerType>(type);
}

mlir::Type elementTypeOf(mlir::Type type) {
  if (auto block = mlir::dyn_cast<BlockType>(type))
    return block.getElementType();
  return type;
}

llvm::ArrayRef<int64_t> shapeOf(mlir::Type type) {
  if (auto block = mlir::dyn_cast<BlockType>(type))
    return block.getShape();
  return {};
}

bool sameShape(mlir::Type lhs, mlir::Type rhs) {
  auto lhsBlock = mlir::dyn_cast<BlockType>(lhs);
  auto rhsBlock = mlir::dyn_cast<BlockType>(rhs);
  if (!lhsBlock || !rhsBlock)
    return true;
  return lhsBlock.getShape() == rhsBlock.getShape();
}

std::optional<llvm::SmallVector<int64_t, 4>>
broadcastShape(mlir::Type lhs, mlir::Type rhs) {
  auto lhsBlock = mlir::dyn_cast<BlockType>(lhs);
  auto rhsBlock = mlir::dyn_cast<BlockType>(rhs);
  if (!lhsBlock && !rhsBlock)
    return llvm::SmallVector<int64_t, 4>{};
  if (!lhsBlock)
    return llvm::SmallVector<int64_t, 4>(rhsBlock.getShape());
  if (!rhsBlock)
    return llvm::SmallVector<int64_t, 4>(lhsBlock.getShape());
  if (lhsBlock.getShape().size() != rhsBlock.getShape().size())
    return std::nullopt;

  llvm::SmallVector<int64_t, 4> result;
  for (auto [lhsDimension, rhsDimension] :
       llvm::zip(lhsBlock.getShape(), rhsBlock.getShape())) {
    if (lhsDimension == rhsDimension) {
      result.push_back(lhsDimension);
      continue;
    }
    if (lhsDimension == 1) {
      result.push_back(rhsDimension);
      continue;
    }
    if (rhsDimension == 1) {
      result.push_back(lhsDimension);
      continue;
    }
    return std::nullopt;
  }
  return result;
}

std::optional<int64_t> indexConstantValue(mlir::Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  return integer ? std::optional<int64_t>(integer.getInt()) : std::nullopt;
}

LogicalExtent extentFromIndex(mlir::Value value) {
  if (auto constant = indexConstantValue(value))
    return LogicalExtent{constant, {}};
  return LogicalExtent{std::nullopt, value};
}

bool sameLogicalExtent(const LogicalExtent &lhs, const LogicalExtent &rhs) {
  if (lhs.constant && rhs.constant)
    return lhs.constant == rhs.constant;
  return lhs.dynamic && rhs.dynamic && lhs.dynamic == rhs.dynamic;
}

std::optional<LogicalExtent>
deriveLogicalExtentImpl(mlir::Value value, int64_t axis,
                        llvm::DenseSet<mlir::Value> &visited);

std::optional<LogicalExtent>
deriveBroadcastExtent(mlir::Value lhs, mlir::Value rhs, int64_t axis,
                      llvm::DenseSet<mlir::Value> &visited) {
  auto lhsBlock = mlir::dyn_cast<BlockType>(lhs.getType());
  auto rhsBlock = mlir::dyn_cast<BlockType>(rhs.getType());
  if (!lhsBlock)
    return deriveLogicalExtentImpl(rhs, axis, visited);
  if (!rhsBlock)
    return deriveLogicalExtentImpl(lhs, axis, visited);
  if (axis < 0 || axis >= static_cast<int64_t>(lhsBlock.getShape().size()) ||
      axis >= static_cast<int64_t>(rhsBlock.getShape().size()))
    return std::nullopt;
  if (lhsBlock.getShape()[axis] == 1)
    return deriveLogicalExtentImpl(rhs, axis, visited);
  if (rhsBlock.getShape()[axis] == 1)
    return deriveLogicalExtentImpl(lhs, axis, visited);

  llvm::DenseSet<mlir::Value> lhsVisited = visited;
  llvm::DenseSet<mlir::Value> rhsVisited = visited;
  auto lhsExtent = deriveLogicalExtentImpl(lhs, axis, lhsVisited);
  auto rhsExtent = deriveLogicalExtentImpl(rhs, axis, rhsVisited);
  if (!lhsExtent || !rhsExtent || !sameLogicalExtent(*lhsExtent, *rhsExtent))
    return std::nullopt;
  return lhsExtent;
}

std::optional<LogicalExtent>
deriveLogicalExtentImpl(mlir::Value value, int64_t axis,
                        llvm::DenseSet<mlir::Value> &visited) {
  auto block = mlir::dyn_cast<BlockType>(value.getType());
  if (!block || axis < 0 ||
      axis >= static_cast<int64_t>(block.getShape().size()) ||
      !visited.insert(value).second)
    return std::nullopt;
  int64_t dimension = block.getShape()[axis];
  if (dimension != -1)
    return LogicalExtent{dimension, {}};

  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto loop = llvm::dyn_cast_or_null<ForOp>(argument.getOwner()->getParentOp());
    if (!loop || argument.getArgNumber() == 0)
      return std::nullopt;
    mlir::OperandRange initArgs = loop->getOperands().drop_front(3);
    unsigned carried = argument.getArgNumber() - 1;
    if (carried >= initArgs.size())
      return std::nullopt;
    return deriveLogicalExtentImpl(initArgs[carried], axis, visited);
  }

  mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  if (auto arange = llvm::dyn_cast<ArangeOp>(definition))
    return axis == 0 ? std::optional<LogicalExtent>(
                           extentFromIndex(arange.getExtent()))
                     : std::nullopt;
  if (auto expand = llvm::dyn_cast<ExpandDimsOp>(definition)) {
    int64_t inserted = expand.getAxisAttr().getInt();
    if (axis == inserted)
      return LogicalExtent{1, {}};
    return deriveLogicalExtentImpl(expand.getInput(),
                                   axis < inserted ? axis : axis - 1, visited);
  }
  if (auto splat = llvm::dyn_cast<SplatOp>(definition))
    return deriveLogicalExtentImpl(splat.getShapeLike(), axis, visited);
  if (auto unary = llvm::dyn_cast<UnaryOp>(definition))
    return deriveLogicalExtentImpl(unary.getInput(), axis, visited);
  if (auto cast = llvm::dyn_cast<CastOp>(definition))
    return deriveLogicalExtentImpl(cast.getInput(), axis, visited);
  if (auto binary = llvm::dyn_cast<BinaryOp>(definition))
    return deriveBroadcastExtent(binary.getLhs(), binary.getRhs(), axis,
                                 visited);
  if (auto compare = llvm::dyn_cast<CompareOp>(definition))
    return deriveBroadcastExtent(compare.getLhs(), compare.getRhs(), axis,
                                 visited);
  if (auto pointerAdd = llvm::dyn_cast<PtrAddOp>(definition))
    return deriveBroadcastExtent(pointerAdd.getBase(), pointerAdd.getOffset(),
                                 axis, visited);
  if (auto load = llvm::dyn_cast<LoadOp>(definition))
    return deriveLogicalExtentImpl(load.getPointer(), axis, visited);
  if (auto reduction = llvm::dyn_cast<ReduceOp>(definition)) {
    int64_t reduced = reduction.getAxisAttr().getInt();
    return deriveLogicalExtentImpl(reduction.getInput(),
                                   axis < reduced ? axis : axis + 1, visited);
  }
  if (auto contract = llvm::dyn_cast<ContractOp>(definition)) {
    auto lhs = mlir::cast<BlockType>(contract.getLhs().getType());
    llvm::DenseSet<int64_t> lhsContracted(contract.getLhsAxes().begin(),
                                          contract.getLhsAxes().end());
    llvm::DenseSet<int64_t> rhsContracted(contract.getRhsAxes().begin(),
                                          contract.getRhsAxes().end());
    int64_t resultAxis = 0;
    for (int64_t lhsAxis = 0;
         lhsAxis < static_cast<int64_t>(lhs.getShape().size()); ++lhsAxis) {
      if (lhsContracted.contains(lhsAxis))
        continue;
      if (resultAxis++ == axis)
        return deriveLogicalExtentImpl(contract.getLhs(), lhsAxis, visited);
    }
    auto rhs = mlir::cast<BlockType>(contract.getRhs().getType());
    for (int64_t rhsAxis = 0;
         rhsAxis < static_cast<int64_t>(rhs.getShape().size()); ++rhsAxis) {
      if (rhsContracted.contains(rhsAxis))
        continue;
      if (resultAxis++ == axis)
        return deriveLogicalExtentImpl(contract.getRhs(), rhsAxis, visited);
    }
    return std::nullopt;
  }
  if (auto loop = llvm::dyn_cast<ForOp>(definition)) {
    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    if (!result)
      return std::nullopt;
    mlir::OperandRange initArgs = loop->getOperands().drop_front(3);
    if (result.getResultNumber() >= initArgs.size())
      return std::nullopt;
    return deriveLogicalExtentImpl(initArgs[result.getResultNumber()], axis,
                                   visited);
  }
  return std::nullopt;
}

mlir::LogicalResult verifyDynamicBroadcastExtents(mlir::Operation *operation,
                                                  mlir::Value lhs,
                                                  mlir::Value rhs) {
  auto lhsBlock = mlir::dyn_cast<BlockType>(lhs.getType());
  auto rhsBlock = mlir::dyn_cast<BlockType>(rhs.getType());
  if (!lhsBlock || !rhsBlock)
    return mlir::success();
  for (int64_t axis = 0;
       axis < static_cast<int64_t>(lhsBlock.getShape().size()); ++axis) {
    int64_t lhsDimension = lhsBlock.getShape()[axis];
    int64_t rhsDimension = rhsBlock.getShape()[axis];
    if (lhsDimension == 1 || rhsDimension == 1 ||
        (lhsDimension != -1 && rhsDimension != -1))
      continue;
    if (!haveSameLogicalExtent(lhs, axis, rhs, axis))
      return operation->emitOpError()
             << "cannot prove that dynamic broadcast axis " << axis
             << " has one canonical runtime extent";
  }
  return mlir::success();
}

bool broadcastsIntoMemoryShape(mlir::Type value, mlir::Type pointer) {
  auto pointerBlock = mlir::dyn_cast<BlockType>(pointer);
  auto valueBlock = mlir::dyn_cast<BlockType>(value);
  if (!pointerBlock)
    return !valueBlock;
  if (!valueBlock)
    return true;
  auto joined = broadcastShape(value, pointer);
  return joined &&
         llvm::ArrayRef<int64_t>(*joined) == pointerBlock.getShape();
}

bool isPointerValue(mlir::Type type) {
  if (mlir::isa<PtrType>(type))
    return true;
  auto block = mlir::dyn_cast<BlockType>(type);
  return block && mlir::isa<PtrType>(block.getElementType());
}

mlir::Type pointerElementType(mlir::Type type) {
  if (auto pointer = mlir::dyn_cast<PtrType>(type))
    return pointer.getElementType();
  if (auto block = mlir::dyn_cast<BlockType>(type))
    if (auto pointer = mlir::dyn_cast<PtrType>(block.getElementType()))
      return pointer.getElementType();
  return {};
}

bool isPredicateType(mlir::Type type) {
  mlir::Type element = elementTypeOf(type);
  auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
  return integer && integer.getWidth() == 1;
}

bool closedUnaryKind(llvm::StringRef kind) {
  return llvm::StringSwitch<bool>(kind)
      .Cases("neg", "exp", "exp2", "log", "rsqrt", true)
      .Default(false);
}

bool closedBinaryKind(llvm::StringRef kind) {
  return llvm::StringSwitch<bool>(kind)
      .Cases("add", "sub", "mul", "div", "mod", true)
      .Cases("and", "or", "xor", "max", "min", true)
      .Default(false);
}

bool closedPredicate(llvm::StringRef predicate) {
  return llvm::StringSwitch<bool>(predicate)
      .Cases("eq", "ne", "lt", "le", "gt", "ge", true)
      .Default(false);
}

bool closedReduction(llvm::StringRef kind) {
  return llvm::StringSwitch<bool>(kind)
      .Cases("sum", "max", "min", true)
      .Default(false);
}

bool isIndexCoordinateProvenanceImpl(mlir::Value value,
                                     llvm::DenseSet<mlir::Value> &visited) {
  auto block = mlir::dyn_cast<BlockType>(value.getType());
  if (!block)
    return value.getType().isIndex();
  if (!block.getElementType().isIndex() || !visited.insert(value).second)
    return false;

  mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  if (llvm::isa<ArangeOp>(definition))
    return true;
  if (auto expand = llvm::dyn_cast<ExpandDimsOp>(definition))
    return isIndexCoordinateProvenanceImpl(expand.getInput(), visited);
  if (auto binary = llvm::dyn_cast<BinaryOp>(definition)) {
    llvm::DenseSet<mlir::Value> lhsVisited = visited;
    llvm::DenseSet<mlir::Value> rhsVisited = visited;
    return isIndexCoordinateProvenanceImpl(binary.getLhs(), lhsVisited) &&
           isIndexCoordinateProvenanceImpl(binary.getRhs(), rhsVisited);
  }
  return false;
}

mlir::LogicalResult verifyBroadcastResult(mlir::Operation *operation,
                                          mlir::Value lhs,
                                          mlir::Value rhs,
                                          mlir::Type result) {
  auto shape = broadcastShape(lhs.getType(), rhs.getType());
  if (!shape)
    return operation->emitOpError("requires axis-wise broadcast-compatible shapes");
  if (mlir::failed(verifyDynamicBroadcastExtents(operation, lhs, rhs)))
    return mlir::failure();
  auto lhsBlock = mlir::dyn_cast<BlockType>(lhs.getType());
  auto rhsBlock = mlir::dyn_cast<BlockType>(rhs.getType());
  auto resultBlock = mlir::dyn_cast<BlockType>(result);
  if ((lhsBlock || rhsBlock) != static_cast<bool>(resultBlock))
    return operation->emitOpError(
        "requires a block result exactly when either operand is blocked");
  if (resultBlock) {
    if (resultBlock.getShape() != llvm::ArrayRef<int64_t>(*shape))
      return operation->emitOpError(
          "result block shape does not match the broadcast join");
  }
  return mlir::success();
}

} // namespace

std::optional<LogicalExtent>
weft::kernel::deriveLogicalExtent(mlir::Value value, int64_t axis) {
  llvm::DenseSet<mlir::Value> visited;
  return deriveLogicalExtentImpl(value, axis, visited);
}

bool weft::kernel::haveSameLogicalExtent(mlir::Value lhs, int64_t lhsAxis,
                                         mlir::Value rhs, int64_t rhsAxis) {
  auto lhsExtent = deriveLogicalExtent(lhs, lhsAxis);
  auto rhsExtent = deriveLogicalExtent(rhs, rhsAxis);
  return lhsExtent && rhsExtent && sameLogicalExtent(*lhsExtent, *rhsExtent);
}

bool weft::kernel::isIndexCoordinateProvenance(mlir::Value value) {
  llvm::DenseSet<mlir::Value> visited;
  return isIndexCoordinateProvenanceImpl(value, visited);
}

mlir::LogicalResult PtrType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type elementType, llvm::StringRef addressSpace) {
  if (!isScalarDataType(elementType))
    return emitError() << "pointer element type must be a scalar data type";
  if (addressSpace.empty())
    return emitError() << "pointer address space must not be empty";
  return mlir::success();
}

mlir::LogicalResult BlockType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, mlir::Type elementType) {
  if (shape.empty())
    return emitError() << "block shape must have positive rank";
  for (int64_t dimension : shape)
    if (dimension != -1 && dimension <= 0)
      return emitError()
             << "block dimensions must be positive or -1 for dynamic extent";
  if (!isScalarDataType(elementType) && !mlir::isa<PtrType>(elementType))
    return emitError()
           << "block element type must be scalar data or a typed pointer";
  return mlir::success();
}

mlir::LogicalResult ConstexprType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type valueType) {
  if (!isScalarDataType(valueType))
    return emitError() << "constexpr value type must be a scalar data type";
  return mlir::success();
}

mlir::LogicalResult KernelOp::verify() {
  int64_t gridRank = getGridRankAttr().getInt();
  if (gridRank <= 0)
    return emitOpError("requires a positive grid_rank");
  mlir::Block &entry = getBody().front();
  auto names = getArgNames();
  auto kinds = getArgKinds();
  if (names.size() != entry.getNumArguments() ||
      kinds.size() != entry.getNumArguments())
    return emitOpError(
        "requires arg_names and arg_kinds to match entry block arguments");
  llvm::DenseSet<llvm::StringRef> seenNames;
  for (auto [index, values] :
       llvm::enumerate(llvm::zip(names, kinds, entry.getArguments()))) {
    auto [nameAttr, kindAttr, argument] = values;
    auto name = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
    auto kind = mlir::dyn_cast<mlir::StringAttr>(kindAttr);
    if (!name || name.getValue().empty() || !kind)
      return emitOpError("requires non-empty string arg_names and arg_kinds");
    if (!seenNames.insert(name.getValue()).second)
      return emitOpError() << "duplicates entry argument name '"
                           << name.getValue() << "'";
    mlir::Type type = argument.getType();
    llvm::StringRef spelling = kind.getValue();
    bool valid = spelling == "pointer"   ? mlir::isa<PtrType>(type)
                 : spelling == "scalar" ? isScalarDataType(type)
                 : spelling == "constexpr"
                     ? mlir::isa<ConstexprType>(type)
                     : false;
    if (!valid)
      return emitOpError() << "argument " << index << " named '"
                           << name.getValue() << "' has kind '" << spelling
                           << "' incompatible with type " << type;
  }
  if (!mlir::isa<ReturnOp>(entry.getTerminator()))
    return emitOpError("entry block must terminate with weft_kernel.return");
  return mlir::success();
}

mlir::LogicalResult ReturnOp::verify() {
  if (!mlir::isa_and_nonnull<KernelOp>(
          getOperation()->getBlock()->getParentOp()))
    return emitOpError("must terminate a weft_kernel.kernel entry");
  return mlir::success();
}

mlir::LogicalResult YieldOp::verify() {
  if (!mlir::isa_and_nonnull<ForOp>(
          getOperation()->getBlock()->getParentOp()))
    return emitOpError("must terminate a weft_kernel.for body");
  return mlir::success();
}

mlir::LogicalResult ConstantOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  auto value = mlir::dyn_cast<mlir::TypedAttr>(getValue());
  if (!value || value.getType() != getResult().getType())
    return emitOpError("value attribute type must match result type");
  if (!isScalarDataType(getResult().getType()))
    return emitOpError("supports only scalar data constants");
  return mlir::success();
}

mlir::LogicalResult MetaValueOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  auto input = mlir::dyn_cast<ConstexprType>(getInput().getType());
  if (!input)
    return emitOpError("input must have !weft_kernel.constexpr type");
  if (input.getValueType() != getResult().getType())
    return emitOpError("result type must match constexpr value type");
  return mlir::success();
}

mlir::LogicalResult TaskIdOp::verify() {
  int64_t axis = getAxisAttr().getInt();
  if (axis < 0)
    return emitOpError("axis must be non-negative");
  KernelOp kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError("must be nested in a weft_kernel.kernel");
  if (axis >= kernel.getGridRankAttr().getInt())
    return emitOpError("axis must be smaller than enclosing grid_rank");
  return mlir::success();
}

mlir::LogicalResult ArangeOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  auto result = mlir::dyn_cast<BlockType>(getResult().getType());
  if (!result || result.getShape().size() != 1 ||
      !result.getElementType().isIndex())
    return emitOpError("result must be a rank-one index block");
  int64_t resultExtent = result.getShape().front();
  if (auto constant = getExtent().getDefiningOp<ConstantOp>()) {
    auto value = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    if (!value || value.getInt() <= 0)
      return emitOpError("constant extent must be positive");
    if (resultExtent != -1 && resultExtent != value.getInt())
      return emitOpError("static result extent must match constant extent");
  } else if (resultExtent != -1) {
    return emitOpError("dynamic extent requires -1 in the result block shape");
  }
  return mlir::success();
}

mlir::LogicalResult ExpandDimsOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  auto input = mlir::dyn_cast<BlockType>(getInput().getType());
  auto result = mlir::dyn_cast<BlockType>(getResult().getType());
  if (!input || !result)
    return emitOpError("input and result must be logical blocks");
  int64_t axis = getAxisAttr().getInt();
  if (axis < 0 || axis > static_cast<int64_t>(input.getShape().size()))
    return emitOpError("axis must be in [0, input rank]");
  if (input.getElementType() != result.getElementType())
    return emitOpError("must preserve the block element type");
  llvm::SmallVector<int64_t, 4> expected(input.getShape());
  expected.insert(expected.begin() + axis, 1);
  if (result.getShape() != llvm::ArrayRef<int64_t>(expected))
    return emitOpError("result shape must insert one singleton axis");
  return mlir::success();
}

mlir::LogicalResult SplatOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  if (!isScalarDataType(getValue().getType()))
    return emitOpError("value must be scalar data");
  auto shapeLike = mlir::dyn_cast<BlockType>(getShapeLike().getType());
  auto result = mlir::dyn_cast<BlockType>(getResult().getType());
  if (!shapeLike || !result)
    return emitOpError("shape_like and result must be logical blocks");
  if (result.getShape() != shapeLike.getShape())
    return emitOpError("result must preserve the shape_like logical shape");
  if (result.getElementType() != getValue().getType())
    return emitOpError("result element type must match the scalar value");
  return mlir::success();
}

mlir::LogicalResult PtrAddOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  if (!isPointerValue(getBase().getType()))
    return emitOpError("base must be a pointer or block of pointers");
  if (!isIntegerLike(elementTypeOf(getOffset().getType())))
    return emitOpError("offset must be an integer/index scalar or block");
  if (!isPointerValue(getResult().getType()))
    return emitOpError("result must be a pointer or block of pointers");
  if (pointerElementType(getBase().getType()) !=
      pointerElementType(getResult().getType()))
    return emitOpError("result must preserve pointer element type");
  if (mlir::isa<BlockType>(getBase().getType()) ||
      mlir::isa<BlockType>(getOffset().getType())) {
    auto result = mlir::dyn_cast<BlockType>(getResult().getType());
    if (!result)
      return emitOpError("blocked pointer arithmetic requires a block result");
    auto expected = broadcastShape(getBase().getType(), getOffset().getType());
    if (!expected)
      return emitOpError(
          "blocked base and offset shapes must be broadcast-compatible");
    if (mlir::failed(verifyDynamicBroadcastExtents(
            getOperation(), getBase(), getOffset())))
      return mlir::failure();
    if (result.getShape() != llvm::ArrayRef<int64_t>(*expected))
      return emitOpError("result block shape must match the broadcast join");
  } else if (mlir::isa<BlockType>(getResult().getType())) {
    return emitOpError("scalar pointer arithmetic requires a scalar pointer result");
  }
  return mlir::success();
}

mlir::LogicalResult UnaryOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  if (!closedUnaryKind(getKind()))
    return emitOpError("kind must be one of neg, exp, exp2, log, or rsqrt");
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result types must match");
  mlir::Type element = elementTypeOf(getInput().getType());
  if (!isScalarDataType(element))
    return emitOpError("requires numeric scalar or block input");
  if (getKind() != "neg" && !mlir::isa<mlir::FloatType>(element))
    return emitOpError("transcendental unary operations require float elements");
  return mlir::success();
}

mlir::LogicalResult BinaryOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  if (!closedBinaryKind(getKind()))
    return emitOpError("kind is not a supported pointwise binary operation");
  if (mlir::failed(verifyBroadcastResult(getOperation(), getLhs(), getRhs(),
                                         getResult().getType())))
    return mlir::failure();
  mlir::Type lhsElement = elementTypeOf(getLhs().getType());
  mlir::Type rhsElement = elementTypeOf(getRhs().getType());
  mlir::Type resultElement = elementTypeOf(getResult().getType());
  if (lhsElement != rhsElement || lhsElement != resultElement ||
      !isScalarDataType(lhsElement))
    return emitOpError("operand and result element types must be equal numeric types");
  if ((getKind() == "div") && !mlir::isa<mlir::FloatType>(lhsElement))
    return emitOpError("div requires float elements in the first slice");
  if ((getKind() == "mod" || getKind() == "and" || getKind() == "or" ||
       getKind() == "xor") &&
      !isIntegerLike(lhsElement))
    return emitOpError("mod and bitwise operations require integer/index elements");
  bool bitwise = getKind() == "and" || getKind() == "or" ||
                 getKind() == "xor";
  if (isPredicateType(lhsElement) && !bitwise)
    return emitOpError(
        "boolean elements support only and, or, and xor binary operations");
  return mlir::success();
}

mlir::LogicalResult CompareOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  if (!closedPredicate(getPredicate()))
    return emitOpError("predicate must be eq, ne, lt, le, gt, or ge");
  if (mlir::failed(verifyBroadcastResult(getOperation(), getLhs(), getRhs(),
                                         getResult().getType())))
    return mlir::failure();
  if (elementTypeOf(getLhs().getType()) != elementTypeOf(getRhs().getType()))
    return emitOpError("operand element types must match");
  if (!isScalarDataType(elementTypeOf(getLhs().getType())))
    return emitOpError("comparison operands must contain scalar data");
  if (!isPredicateType(getResult().getType()))
    return emitOpError("result must be i1 or a block of i1");
  return mlir::success();
}

mlir::LogicalResult CastOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  if (!sameShape(getInput().getType(), getResult().getType()))
    return emitOpError("cast must preserve block shape");
  if (mlir::isa<BlockType>(getInput().getType()) !=
      mlir::isa<BlockType>(getResult().getType()))
    return emitOpError("cast cannot change scalar/block rank");
  if (!isScalarDataType(elementTypeOf(getInput().getType())) ||
      !isScalarDataType(elementTypeOf(getResult().getType())))
    return emitOpError("cast requires scalar data element types");
  return mlir::success();
}

mlir::LogicalResult LoadOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  if (!isPointerValue(getPointer().getType()))
    return emitOpError("pointer must be a pointer or block of pointers");
  if (!isPredicateType(getMask().getType()))
    return emitOpError("mask must be i1 or a block of i1");
  if (!broadcastsIntoMemoryShape(getMask().getType(),
                                 getPointer().getType()))
    return emitOpError("mask must broadcast exactly into the pointer footprint");
  if (mlir::failed(verifyDynamicBroadcastExtents(
          getOperation(), getPointer(), getMask())))
    return mlir::failure();
  mlir::Type expectedElement = pointerElementType(getPointer().getType());
  if (elementTypeOf(getResult().getType()) != expectedElement ||
      elementTypeOf(getOther().getType()) != expectedElement)
    return emitOpError("pointer, other and result element types must match");
  if (!broadcastsIntoMemoryShape(getOther().getType(),
                                 getPointer().getType()))
    return emitOpError(
        "other must broadcast exactly into the pointer footprint");
  if (mlir::failed(verifyDynamicBroadcastExtents(
          getOperation(), getPointer(), getOther())))
    return mlir::failure();
  bool blockedPointer = mlir::isa<BlockType>(getPointer().getType());
  if (blockedPointer != mlir::isa<BlockType>(getResult().getType()))
    return emitOpError("result shape must follow pointer shape");
  if (blockedPointer &&
      shapeOf(getPointer().getType()) != shapeOf(getResult().getType()))
    return emitOpError("result block shape must match pointer block shape");
  return mlir::success();
}

mlir::LogicalResult StoreOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  if (!isPointerValue(getPointer().getType()))
    return emitOpError("pointer must be a pointer or block of pointers");
  if (!isPredicateType(getMask().getType()))
    return emitOpError("mask must be i1 or a block of i1");
  if (!broadcastsIntoMemoryShape(getMask().getType(),
                                 getPointer().getType()) ||
      !broadcastsIntoMemoryShape(getValue().getType(),
                                 getPointer().getType()))
    return emitOpError(
        "value and mask must broadcast exactly into the pointer footprint");
  if (mlir::failed(verifyDynamicBroadcastExtents(
          getOperation(), getPointer(), getMask())) ||
      mlir::failed(verifyDynamicBroadcastExtents(
          getOperation(), getPointer(), getValue())))
    return mlir::failure();
  if (pointerElementType(getPointer().getType()) !=
      elementTypeOf(getValue().getType()))
    return emitOpError("pointer and value element types must match");
  return mlir::success();
}

mlir::LogicalResult ContractOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  auto lhs = mlir::dyn_cast<BlockType>(getLhs().getType());
  auto rhs = mlir::dyn_cast<BlockType>(getRhs().getType());
  if (!lhs || !rhs)
    return emitOpError("lhs and rhs must be logical blocks");
  llvm::ArrayRef<int64_t> lhsAxes = getLhsAxes();
  llvm::ArrayRef<int64_t> rhsAxes = getRhsAxes();
  if (lhsAxes.empty() || lhsAxes.size() != rhsAxes.size())
    return emitOpError(
        "requires equally sized, non-empty lhs_axes and rhs_axes");

  llvm::DenseSet<int64_t> contractedLhs;
  llvm::DenseSet<int64_t> contractedRhs;
  for (auto [lhsAxis, rhsAxis] : llvm::zip(lhsAxes, rhsAxes)) {
    if (lhsAxis < 0 ||
        lhsAxis >= static_cast<int64_t>(lhs.getShape().size()) ||
        !contractedLhs.insert(lhsAxis).second)
      return emitOpError("lhs_axes must be unique axes inside lhs rank");
    if (rhsAxis < 0 ||
        rhsAxis >= static_cast<int64_t>(rhs.getShape().size()) ||
        !contractedRhs.insert(rhsAxis).second)
      return emitOpError("rhs_axes must be unique axes inside rhs rank");
    if (lhs.getShape()[lhsAxis] != rhs.getShape()[rhsAxis])
      return emitOpError(
          "paired contraction dimensions must have identical logical extents");
    if (lhs.getShape()[lhsAxis] == -1 &&
        !haveSameLogicalExtent(getLhs(), lhsAxis, getRhs(), rhsAxis))
      return emitOpError()
             << "cannot prove that paired dynamic contraction axes "
             << lhsAxis << " and " << rhsAxis
             << " have one canonical runtime extent";
  }

  mlir::Type element = lhs.getElementType();
  if (element != rhs.getElementType() || !isScalarDataType(element) ||
      isPredicateType(element))
    return emitOpError(
        "lhs and rhs must have the same non-predicate numeric element type");

  llvm::SmallVector<int64_t, 4> resultShape;
  for (auto [axis, dimension] : llvm::enumerate(lhs.getShape()))
    if (!contractedLhs.contains(axis))
      resultShape.push_back(dimension);
  for (auto [axis, dimension] : llvm::enumerate(rhs.getShape()))
    if (!contractedRhs.contains(axis))
      resultShape.push_back(dimension);

  mlir::Type initElement = elementTypeOf(getInit().getType());
  auto lhsInteger = mlir::dyn_cast<mlir::IntegerType>(element);
  auto initInteger = mlir::dyn_cast<mlir::IntegerType>(initElement);
  bool signedI8ToI32 = lhsInteger && initInteger && lhsInteger.isSigned() &&
                       lhsInteger.getWidth() == 8 && initInteger.isSigned() &&
                       initInteger.getWidth() == 32;
  if (initElement != element && !signedI8ToI32)
    return emitOpError(
        "init element type must match lhs/rhs or use the validated signed "
        "i8 x signed i8 to signed i32 widening form");
  if (auto init = mlir::dyn_cast<BlockType>(getInit().getType())) {
    if (resultShape.empty() ||
        init.getShape() != llvm::ArrayRef<int64_t>(resultShape))
      return emitOpError(
          "blocked init shape must exactly match the contraction result");
  } else if (!isScalarDataType(getInit().getType()) ||
             isPredicateType(getInit().getType())) {
    return emitOpError("init must be a numeric scalar or result-shaped block");
  }

  if (resultShape.empty()) {
    if (getResult().getType() != initElement)
      return emitOpError("fully contracted result must be a scalar");
  } else {
    auto result = mlir::dyn_cast<BlockType>(getResult().getType());
    if (!result || result.getElementType() != initElement ||
        result.getShape() != llvm::ArrayRef<int64_t>(resultShape))
      return emitOpError(
          "result shape must be non-contracted lhs axes followed by "
          "non-contracted rhs axes");
  }
  return mlir::success();
}

mlir::LogicalResult ReduceOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  auto input = mlir::dyn_cast<BlockType>(getInput().getType());
  if (!input)
    return emitOpError("input must be a logical block");
  int64_t axis = getAxisAttr().getInt();
  if (axis < 0 || static_cast<size_t>(axis) >= input.getShape().size())
    return emitOpError("axis is outside the input block rank");
  if (!closedReduction(getKind()))
    return emitOpError("kind must be sum, max, or min");
  if (mlir::isa<BlockType>(getInit().getType()) ||
      input.getElementType() != getInit().getType())
    return emitOpError(
        "requires a scalar init matching the input element type");
  llvm::SmallVector<int64_t, 4> projectedShape(input.getShape());
  projectedShape.erase(projectedShape.begin() + axis);
  if (projectedShape.empty()) {
    if (getResult().getType() != input.getElementType())
      return emitOpError("rank-one reduction result must be scalar");
  } else {
    auto result = mlir::dyn_cast<BlockType>(getResult().getType());
    if (!result || result.getElementType() != input.getElementType() ||
        result.getShape() != llvm::ArrayRef<int64_t>(projectedShape))
      return emitOpError(
          "rank-N reduction result must remove exactly the selected axis");
  }
  return mlir::success();
}

mlir::LogicalResult ForOp::verify() {
  if (mlir::failed(requireKernelAncestor(getOperation())))
    return mlir::failure();
  mlir::Operation *operation = getOperation();
  mlir::OperandRange operands = operation->getOperands();
  mlir::OperandRange initArgs = operands.drop_front(3);
  if (initArgs.size() != operation->getNumResults())
    return emitOpError("init argument count must match result count");
  for (auto [init, result] : llvm::zip(initArgs, operation->getResults()))
    if (init.getType() != result.getType())
      return emitOpError("init and result types must match");

  mlir::Block &body = getBody().front();
  if (body.getNumArguments() != initArgs.size() + 1 ||
      !body.getArgument(0).getType().isIndex())
    return emitOpError(
        "body arguments must be induction index followed by carried values");
  for (auto [argument, init] :
       llvm::zip(body.getArguments().drop_front(), initArgs))
    if (argument.getType() != init.getType())
      return emitOpError("carried body argument types must match init values");

  auto yield = mlir::dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield->getNumOperands() != operation->getNumResults())
    return emitOpError("body must yield one value for every loop result");
  for (auto [value, result] :
       llvm::zip(yield->getOperands(), operation->getResults()))
    if (value.getType() != result.getType())
      return emitOpError("yield and loop result types must match");
  for (auto [init, yielded] : llvm::zip(initArgs, yield->getOperands())) {
    auto block = mlir::dyn_cast<BlockType>(init.getType());
    if (!block)
      continue;
    for (int64_t axis = 0;
         axis < static_cast<int64_t>(block.getShape().size()); ++axis)
      if (block.getShape()[axis] == -1 &&
          !haveSameLogicalExtent(init, axis, yielded, axis))
        return emitOpError()
               << "cannot prove loop-carried dynamic extent invariance at "
               << "axis " << axis;
  }
  if (auto constant = getStep().getDefiningOp<ConstantOp>()) {
    auto value = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    if (value && value.getValue().isZero())
      return emitOpError("step must not be statically zero");
  }
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
