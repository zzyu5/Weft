#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace weft::kernel;

#include "Weft/Dialect/Kernel/IR/KernelOpsDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Weft/Dialect/Kernel/IR/KernelTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/Kernel/IR/KernelOps.cpp.inc"

namespace {

bool isScalar(mlir::Type type) {
  return type.isIndex() || mlir::isa<mlir::IntegerType, mlir::FloatType>(type);
}

bool isEncoding(mlir::Type type) { return mlir::isa<EncodingType>(type); }

bool isViewLike(mlir::Type type) {
  return mlir::isa<ViewType, SliceType>(type);
}

bool isLocalValue(mlir::Type type) {
  return isScalar(type) || isEncoding(type) || mlir::isa<ValueType>(type);
}

mlir::Type logicalElement(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getElementType();
  if (auto view = mlir::dyn_cast<ViewType>(type))
    return view.getEncoding();
  if (auto slice = mlir::dyn_cast<SliceType>(type))
    return slice.getEncoding();
  return type;
}

llvm::ArrayRef<int64_t> logicalShape(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getShape().asArrayRef();
  if (auto view = mlir::dyn_cast<ViewType>(type))
    return view.getShape().asArrayRef();
  if (auto slice = mlir::dyn_cast<SliceType>(type))
    return slice.getShape().asArrayRef();
  return {};
}

llvm::ArrayRef<int64_t> logicalAxes(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getAxisIds().asArrayRef();
  if (auto view = mlir::dyn_cast<ViewType>(type))
    return view.getAxisIds().asArrayRef();
  if (auto slice = mlir::dyn_cast<SliceType>(type))
    return slice.getAxisIds().asArrayRef();
  return {};
}

bool sameDomain(mlir::Type lhs, mlir::Type rhs) {
  return logicalShape(lhs) == logicalShape(rhs) &&
         logicalAxes(lhs) == logicalAxes(rhs);
}

bool isNumeric(mlir::Type type) {
  type = logicalElement(type);
  return mlir::isa<mlir::IntegerType, mlir::FloatType>(type) || type.isIndex();
}

std::optional<unsigned> bitWidth(mlir::Type type) {
  type = logicalElement(type);
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  return std::nullopt;
}

mlir::LogicalResult verifyShape(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> axes,
    bool allowEmpty) {
  if (!allowEmpty && shape.empty())
    return emitError() << "logical Value must have positive rank";
  if (shape.size() != axes.size())
    return emitError() << "logical shape and axis identity rank must match";
  llvm::DenseSet<int64_t> seen;
  for (auto [extent, axis] : llvm::zip(shape, axes)) {
    if (extent == 0)
      return emitError() << "logical extents must be positive or symbolic";
    if (axis <= 0)
      return emitError() << "logical axis identities must be positive";
    if (!seen.insert(axis).second)
      return emitError() << "one logical axis identity cannot occur twice";
  }
  return mlir::success();
}

mlir::LogicalResult verifyStringArray(mlir::Operation *operation,
                                      mlir::ArrayAttr values,
                                      llvm::StringRef name) {
  if (llvm::any_of(values, [](mlir::Attribute value) {
        return !mlir::isa<mlir::StringAttr>(value);
      }))
    return operation->emitOpError() << name << " must contain only strings";
  return mlir::success();
}

mlir::LogicalResult verifyPointwise(mlir::Operation *operation,
                                    mlir::Type lhs, mlir::Type rhs,
                                    mlir::Type result) {
  if (!isLocalValue(lhs) || !isLocalValue(rhs) || !isLocalValue(result))
    return operation->emitOpError("pointwise operands must be local Values");
  if (logicalElement(lhs) != logicalElement(rhs))
    return operation->emitOpError("pointwise element types must match");
  if (logicalElement(result) != logicalElement(lhs))
    return operation->emitOpError("pointwise result element type must match");
  llvm::DenseSet<int64_t> axes;
  for (int64_t axis : logicalAxes(lhs))
    axes.insert(axis);
  for (int64_t axis : logicalAxes(rhs))
    axes.insert(axis);
  if (axes.size() != logicalAxes(result).size())
    return operation->emitOpError("pointwise result must preserve the union of axes");
  for (int64_t axis : logicalAxes(result))
    if (!axes.contains(axis))
      return operation->emitOpError("pointwise result contains an unrelated axis");
  for (auto [axis, extent] : llvm::zip(logicalAxes(result), logicalShape(result))) {
    for (mlir::Type operand : {lhs, rhs}) {
      auto operandAxes = logicalAxes(operand);
      auto found = llvm::find(operandAxes, axis);
      if (found == operandAxes.end())
        continue;
      size_t position = static_cast<size_t>(found - operandAxes.begin());
      if (logicalShape(operand)[position] != extent)
        return operation->emitOpError(
            "pointwise operands disagree on a shared logical axis extent");
    }
  }
  return mlir::success();
}

mlir::LogicalResult verifyControlYield(mlir::Operation *owner,
                                       mlir::Block &block,
                                       mlir::TypeRange expected) {
  auto yield = mlir::dyn_cast<YieldOp>(block.getTerminator());
  if (!yield)
    return owner->emitOpError("control region must end in weft_kernel.yield");
  if (yield.getValues().getTypes() != expected)
    return owner->emitOpError("control yield types must match operation results");
  return mlir::success();
}

mlir::LogicalResult verifyContractLike(mlir::Operation *operation,
                                       mlir::Value lhs, mlir::Value rhs,
                                       llvm::ArrayRef<int64_t> over,
                                       std::optional<mlir::Type> accType,
                                       mlir::Type result) {
  if (!isLocalValue(lhs.getType()) || !isLocalValue(rhs.getType()) ||
      !isLocalValue(result))
    return operation->emitOpError("contraction operands must be local Values");
  if (over.empty())
    return operation->emitOpError("contraction requires at least one reduction axis");
  llvm::DenseSet<int64_t> reduction;
  for (int64_t axis : over) {
    if (axis <= 0 || !llvm::is_contained(logicalAxes(lhs.getType()), axis) ||
        !llvm::is_contained(logicalAxes(rhs.getType()), axis) ||
        !reduction.insert(axis).second)
      return operation->emitOpError(
          "each contraction axis must occur once in both operands");
  }
  llvm::DenseSet<int64_t> expected;
  for (int64_t axis : logicalAxes(lhs.getType()))
    if (!reduction.contains(axis))
      expected.insert(axis);
  for (int64_t axis : logicalAxes(rhs.getType()))
    if (!reduction.contains(axis))
      expected.insert(axis);
  if (expected.size() != logicalAxes(result).size())
    return operation->emitOpError("contraction result has the wrong free axes");
  for (int64_t axis : logicalAxes(result))
    if (!expected.contains(axis))
      return operation->emitOpError("contraction result contains an unrelated axis");
  if (accType && logicalElement(result) != *accType)
    return operation->emitOpError("acc_type must equal the result element type");
  return mlir::success();
}

} // namespace

mlir::LogicalResult EncodingType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef family, llvm::StringRef kind,
    llvm::StringRef layoutIdentity) {
  if (family.empty())
    return emitError() << "encoding family must not be empty";
  if (kind != "base" && kind != "dense" && kind != "derived_family" &&
      kind != "derived_instance")
    return emitError() << "unknown encoding kind";
  if ((kind == "base" || kind == "dense" || kind == "derived_instance") &&
      layoutIdentity.empty())
    return emitError() << "concrete encoding requires a layout identity";
  if (kind == "derived_family" && !layoutIdentity.empty())
    return emitError() << "abstract derived encoding cannot carry a layout identity";
  return mlir::success();
}

mlir::LogicalResult ViewType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type encoding, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds) {
  if (!isEncoding(encoding))
    return emitError() << "View encoding must be an Encoding type";
  return verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef(), true);
}

mlir::LogicalResult ValueType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type elementType, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds) {
  if (!isScalar(elementType) && !isEncoding(elementType))
    return emitError() << "Value element must be numeric or encoded";
  return verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef(), false);
}

mlir::LogicalResult SliceType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type encoding, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds) {
  if (!isEncoding(encoding))
    return emitError() << "slice encoding must be an Encoding type";
  return verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef(), true);
}

mlir::LogicalResult DomainType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef axisName, int64_t axisId, llvm::StringRef relation,
    llvm::StringRef extent, llvm::StringRef partition,
    llvm::StringRef multiplicity, llvm::StringRef tail) {
  if (axisName.empty() || extent.empty() || partition.empty() ||
      multiplicity.empty())
    return emitError() << "domain fields must not be empty";
  if (relation != "root" && relation != "rows" && relation != "cols" &&
      relation != "tiles" && relation != "blocks" && relation != "subs")
    return emitError() << "unknown level relation";
  if ((relation == "root") != (axisId == 0))
    return emitError() << "only the root domain has axis identity zero";
  if (tail != "exact" && tail != "tail")
    return emitError() << "domain tail must be exact or tail";
  return mlir::success();
}

mlir::LogicalResult PointType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    DomainType domain) {
  if (!domain)
    return emitError() << "point must reference a domain";
  return mlir::success();
}

mlir::LogicalResult EncodingDeclOp::verify() {
  if (getKind() != "base")
    return emitOpError("encoding declaration must define a base layout");
  if (getBitOrder() != "lsb_first" && getBitOrder() != "msb_first")
    return emitOpError("bit_order must be lsb_first or msb_first");
  if (getByteOrder() != "little" && getByteOrder() != "big")
    return emitOpError("byte_order must be little or big");
  if (getAlignment() <= 0 || getElements() <= 0 || getStorageBits() <= 0)
    return emitOpError("alignment, elements, and storage_bits must be positive");
  if (failed(verifyStringArray(*this, getFieldNames(), "field_names")))
    return mlir::failure();
  const size_t count = getFieldNames().size();
  if (getFieldTypes().size() != count || getFieldShapes().size() != count ||
      getFieldLayouts().size() != count ||
      getFieldBitOffsets().size() != count ||
      getFieldStorageBits().size() != count)
    return emitOpError("all encoding field arrays must have the same length");
  int64_t previousEnd = 0;
  int64_t joinedOffset = -1;
  int64_t joinedWidth = 0;
  int64_t joinedFields = 0;
  int64_t joinedNextRole = 0;
  llvm::SmallVector<std::pair<int64_t, int64_t>> spans;
  for (size_t index = 0; index < count; ++index) {
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(getFieldTypes()[index]);
    auto shapeAttr =
        mlir::dyn_cast<mlir::DenseI64ArrayAttr>(getFieldShapes()[index]);
    auto layouts = mlir::dyn_cast<mlir::ArrayAttr>(getFieldLayouts()[index]);
    if (!typeAttr || !shapeAttr || !layouts || layouts.empty())
      return emitOpError("field_types and field_shapes have invalid elements");
    auto kind = [](mlir::Attribute attribute) -> llvm::StringRef {
      auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(attribute);
      auto value = dictionary ? dictionary.getAs<mlir::StringAttr>("kind")
                              : mlir::StringAttr();
      return value ? value.getValue() : llvm::StringRef();
    };
    bool sharedJoined = false;
    int64_t joinedRole = -1;
    if (kind(layouts[0]) == "natural") {
      if (layouts.size() != 1)
        return emitOpError("natural field layout cannot be combined");
    } else if (kind(layouts[0]) == "joined") {
      if (layouts.size() != 1)
        return emitOpError("joined field layout cannot be combined");
      auto joined = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
      auto group = joined.getAs<mlir::IntegerAttr>("size");
      auto fields = joined.getAs<mlir::IntegerAttr>("fields");
      auto lowBits = joined.getAs<mlir::IntegerAttr>("low_bits");
      auto role = joined.getAs<mlir::IntegerAttr>("role");
      auto order = joined.getAs<mlir::StringAttr>("order");
      auto integer = mlir::dyn_cast<mlir::IntegerType>(typeAttr.getValue());
      if (!group || !fields || !lowBits || !role || !order || !integer ||
          shapeAttr.size() != 1 || group.getInt() <= 0 || fields.getInt() < 2 ||
          role.getInt() < 0 || role.getInt() >= fields.getInt() ||
          shapeAttr[0] != 2 * group.getInt() || lowBits.getInt() <= 0 ||
          lowBits.getInt() >= integer.getWidth() ||
          lowBits.getInt() * fields.getInt() != 8 ||
          integer.getWidth() + integer.getWidth() - lowBits.getInt() != 8 ||
          (order.getValue() != "lo_first" && order.getValue() != "hi_first"))
        return emitOpError("invalid joined field layout");
      joinedRole = role.getInt();
      sharedJoined = true;
    } else {
      if (layouts.size() != 2 || kind(layouts[0]) != "grouped" ||
          kind(layouts[1]) != "layered")
        return emitOpError(
            "field layout must be natural or grouped followed by layered");
      auto grouped = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
      auto layered = mlir::cast<mlir::DictionaryAttr>(layouts[1]);
      auto group = grouped.getAs<mlir::IntegerAttr>("size");
      auto layer = layered.getAs<mlir::IntegerAttr>("size");
      auto order = layered.getAs<mlir::StringAttr>("order");
      if (!group || !layer || !order || group.getInt() <= 0 ||
          layer.getInt() <= 0 || group.getInt() % layer.getInt() ||
          (order.getValue() != "lo_first" &&
           order.getValue() != "hi_first"))
        return emitOpError("invalid grouped/layered field layout");
      auto integer = mlir::dyn_cast<mlir::IntegerType>(typeAttr.getValue());
      if (!integer || shapeAttr.size() != 1 ||
          shapeAttr[0] % group.getInt() ||
          integer.getWidth() * (group.getInt() / layer.getInt()) != 8)
        return emitOpError(
            "grouped/layered field must form one byte per layer position");
    }
    int64_t offset = getFieldBitOffsets()[index];
    int64_t width = getFieldStorageBits()[index];
    if (width <= 0 || offset + width > getStorageBits())
      return emitOpError("encoding fields overlap or exceed storage_bits");
    if (sharedJoined) {
      auto joined = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
      int64_t fields = joined.getAs<mlir::IntegerAttr>("fields").getInt();
      if (joinedRole == 0) {
        if (offset < previousEnd)
          return emitOpError("joined storage overlaps a previous field");
        joinedOffset = offset;
        joinedWidth = width;
        joinedFields = fields;
        joinedNextRole = 1;
        previousEnd = offset + width;
        spans.emplace_back(offset, offset + width);
      } else if (offset != joinedOffset || width != joinedWidth ||
                 fields != joinedFields || joinedRole != joinedNextRole++) {
        return emitOpError(
            "joined logical fields must be consecutive roles over one storage span");
      }
    } else {
      if (joinedOffset >= 0 && joinedNextRole != joinedFields)
        return emitOpError("joined storage is missing one or more logical fields");
      joinedOffset = -1;
      joinedNextRole = 0;
      if (offset < previousEnd)
        return emitOpError("encoding fields overlap");
      previousEnd = offset + width;
      spans.emplace_back(offset, offset + width);
    }
  }
  if (joinedOffset >= 0 && joinedNextRole != joinedFields)
    return emitOpError("joined storage is missing one or more logical fields");
  if (getPadding().size() % 3)
    return emitOpError("padding is a sequence of offset, width, fill triples");
  for (size_t index = 0; index < getPadding().size(); index += 3) {
    if (getPadding()[index] < 0 || getPadding()[index + 1] <= 0 ||
        getPadding()[index] + getPadding()[index + 1] > getStorageBits() ||
        getPadding()[index + 2] < 0 || getPadding()[index + 2] > 255)
      return emitOpError("padding triple is outside the declared layout");
    if (getPadding()[index] % 8 || getPadding()[index + 1] % 8)
      return emitOpError("padding byte ranges must be byte aligned");
    spans.emplace_back(getPadding()[index],
                       getPadding()[index] + getPadding()[index + 1]);
  }
  llvm::sort(spans);
  int64_t cursor = 0;
  for (auto [begin, end] : spans) {
    if (begin != cursor)
      return emitOpError("layout gaps must be represented by explicit padding");
    cursor = end;
  }
  if (cursor != getStorageBits())
    return emitOpError("fields and padding must cover storage_bits exactly");
  return mlir::success();
}

mlir::LogicalResult DeriveOp::verify() {
  if (getSourceFamily().empty() || getResultFamily().empty() ||
      getSourceFamily() == getResultFamily())
    return emitOpError("derive requires distinct source and result families");
  if (failed(verifyStringArray(*this, getParameters(), "parameters")))
    return mlir::failure();
  mlir::Block &body = getBody().front();
  if (body.getNumArguments() != 1 || !mlir::isa<ViewType>(body.getArgument(0).getType()))
    return emitOpError("derive body takes one source View");
  auto source = mlir::cast<ViewType>(body.getArgument(0).getType()).getEncoding();
  if (mlir::cast<EncodingType>(source).getFamily() != getSourceFamily())
    return emitOpError("derive source View family does not match source_family");
  auto yield = mlir::dyn_cast<DeriveYieldOp>(body.getTerminator());
  if (!yield)
    return emitOpError("derive body must end in derive_yield");
  auto resultEncoding = yield.getValue().getType().getEncoding();
  auto result = mlir::cast<EncodingType>(resultEncoding);
  if (result.getKind() != "derived_instance" ||
      result.getFamily() != getResultFamily() || result.getLayoutIdentity().empty())
    return emitOpError(
        "derive result must be one concrete instance of its derived encoding family");
  return mlir::success();
}

mlir::LogicalResult DeriveYieldOp::verify() {
  if (!mlir::isa_and_nonnull<DeriveOp>(getOperation()->getParentOp()))
    return emitOpError("derive_yield terminates only a derive body");
  return mlir::success();
}

mlir::LogicalResult KernelOp::verify() {
  mlir::Block &body = getBody().front();
  if (getArgNames().size() != body.getNumArguments())
    return emitOpError("arg_names must match kernel entry arguments");
  if (failed(verifyStringArray(*this, getArgNames(), "arg_names")) ||
      failed(verifyStringArray(*this, getShapeSymbols(), "shape_symbols")))
    return mlir::failure();
  if (!mlir::isa<ReturnOp>(body.getTerminator()))
    return emitOpError("kernel body must end in weft_kernel.return");
  return mlir::success();
}

mlir::LogicalResult ReturnOp::verify() {
  if (!mlir::isa_and_nonnull<KernelOp>(getOperation()->getParentOp()))
    return emitOpError("return terminates only a kernel body");
  if (!getValues().empty())
    return emitOpError("current kernels return through commit, not SSA results");
  return mlir::success();
}

mlir::LogicalResult RootDomainOp::verify() {
  DomainType type = getResult().getType();
  if (type.getRelation() != "root" || type.getAxisId() != 0)
    return emitOpError("root_domain must produce the canonical root domain");
  return mlir::success();
}

mlir::LogicalResult SymbolOp::verify() {
  if (getName().empty() || getKind() != "shape")
    return emitOpError("symbol currently represents one named shape value");
  return mlir::success();
}

mlir::LogicalResult DomainOp::verify() {
  DomainType child = getResult().getType();
  if (child.getRelation() == "root")
    return emitOpError("child domain cannot be another root");
  return mlir::success();
}

mlir::LogicalResult LevelOp::verify() {
  if (getCarried().size() != getResults().size())
    return emitOpError("level carried operands and results must match");
  for (auto [operand, result] : llvm::zip(getCarried(), getResults()))
    if (operand.getType() != result.getType())
      return emitOpError("level carried operand/result types must match");
  if (failed(verifyStringArray(*this, getStateNames(), "state_names")) ||
      failed(verifyStringArray(*this, getStagedNames(), "staged_names")))
    return mlir::failure();

  DomainType domain = getDomain().getType();
  PointType point = PointType::get(getContext(), domain);
  mlir::Block &state = getStateBirths().front();
  mlir::Block &staged = getStagedBirths().front();
  mlir::Block &body = getBody().front();
  if (state.getNumArguments() != 1 || state.getArgument(0).getType() != point ||
      staged.getNumArguments() != 1 || staged.getArgument(0).getType() != point)
    return emitOpError("both births regions take the current level point");
  auto stateYield = mlir::dyn_cast<BirthsYieldOp>(state.getTerminator());
  auto stagedYield = mlir::dyn_cast<BirthsYieldOp>(staged.getTerminator());
  if (!stateYield || !stagedYield)
    return emitOpError("births regions must end in births_yield");
  if (stateYield.getValues().size() != getStateNames().size() ||
      stagedYield.getValues().size() != getStagedNames().size())
    return emitOpError("birth names must match births_yield values");

  const size_t expectedArguments = 1 + getCarried().size() +
                                   stateYield.getValues().size() +
                                   stagedYield.getValues().size();
  if (body.getNumArguments() != expectedArguments ||
      body.getArgument(0).getType() != point)
    return emitOpError("level body signature does not match domain and births");
  size_t cursor = 1;
  for (mlir::Value carried : getCarried())
    if (body.getArgument(cursor++).getType() != carried.getType())
      return emitOpError("level body carried argument type mismatch");
  for (mlir::Value birth : stateYield.getValues())
    if (body.getArgument(cursor++).getType() != birth.getType())
      return emitOpError("state birth type mismatch");
  for (mlir::Value birth : stagedYield.getValues())
    if (body.getArgument(cursor++).getType() != birth.getType())
      return emitOpError("staged birth type mismatch");
  auto handoff = mlir::dyn_cast<HandoffOp>(body.getTerminator());
  if (!handoff || handoff.getValues().getTypes() != getResults().getTypes())
    return emitOpError("level body handoff must match level results");
  if (handoff.getNames().size() != getResults().size() ||
      failed(verifyStringArray(handoff, handoff.getNames(), "names")))
    return emitOpError("handoff names must match level results");
  return mlir::success();
}

mlir::LogicalResult BirthsYieldOp::verify() {
  auto level = mlir::dyn_cast_or_null<LevelOp>(getOperation()->getParentOp());
  if (!level || (getOperation()->getParentRegion() != &level.getStateBirths() &&
                 getOperation()->getParentRegion() != &level.getStagedBirths()))
    return emitOpError("births_yield terminates a level births region");
  return mlir::success();
}

mlir::LogicalResult HandoffOp::verify() {
  auto level = mlir::dyn_cast_or_null<LevelOp>(getOperation()->getParentOp());
  if (!level || getOperation()->getParentRegion() != &level.getBody())
    return emitOpError("handoff terminates a level body");
  if (getNames().size() != getValues().size())
    return emitOpError("handoff names and values must match");
  return verifyStringArray(*this, getNames(), "names");
}

mlir::LogicalResult YieldOp::verify() {
  mlir::Operation *parent = getOperation()->getParentOp();
  if (!mlir::isa_and_nonnull<ForOp, IfOp, WhileOp>(parent))
    return emitOpError("yield terminates ordinary for, if, or while control");
  return mlir::success();
}

mlir::LogicalResult ConditionOp::verify() {
  auto loop = mlir::dyn_cast_or_null<WhileOp>(getOperation()->getParentOp());
  if (!loop || getOperation()->getParentRegion() != &loop.getConditionRegion())
    return emitOpError("condition terminates a while condition region");
  if (getValues().getTypes() != loop.getResults().getTypes())
    return emitOpError("while condition carry must match while results");
  return mlir::success();
}

mlir::LogicalResult ForOp::verify() {
  if (getInitArgs().getTypes() != getResults().getTypes())
    return emitOpError("for init and result types must match");
  mlir::Block &body = getBody().front();
  if (body.getNumArguments() != 1 + getInitArgs().size() ||
      !body.getArgument(0).getType().isIndex())
    return emitOpError("for body takes index followed by carried values");
  for (auto [argument, initial] :
       llvm::zip(body.getArguments().drop_front(), getInitArgs()))
    if (argument.getType() != initial.getType())
      return emitOpError("for carried argument type mismatch");
  return verifyControlYield(*this, body, getResults().getTypes());
}

mlir::LogicalResult IfOp::verify() {
  mlir::Block &thenBlock = getThenRegion().front();
  mlir::Block &elseBlock = getElseRegion().front();
  if (thenBlock.getNumArguments() || elseBlock.getNumArguments())
    return emitOpError("ordinary if regions do not take implicit arguments");
  if (failed(verifyControlYield(*this, thenBlock, getResults().getTypes())))
    return mlir::failure();
  return verifyControlYield(*this, elseBlock, getResults().getTypes());
}

mlir::LogicalResult WhileOp::verify() {
  if (getInitArgs().getTypes() != getResults().getTypes())
    return emitOpError("while init and result types must match");
  mlir::Block &condition = getConditionRegion().front();
  mlir::Block &body = getBodyRegion().front();
  if (condition.getArgumentTypes() != getResults().getTypes() ||
      body.getArgumentTypes() != getResults().getTypes())
    return emitOpError("while regions take the carried values");
  auto conditionTerminator = mlir::dyn_cast<ConditionOp>(condition.getTerminator());
  if (!conditionTerminator)
    return emitOpError("while condition region must end in condition");
  if (conditionTerminator.getValues().getTypes() != getResults().getTypes())
    return emitOpError("while condition carry must match while results");
  return verifyControlYield(*this, body, getResults().getTypes());
}

mlir::LogicalResult ConstantOp::verify() {
  if (!isScalar(getResult().getType()))
    return emitOpError("constant result must be a scalar or index");
  if (auto typed = mlir::dyn_cast<mlir::TypedAttr>(getValue());
      typed && typed.getType() != getResult().getType())
    return emitOpError("constant attribute and result types must match");
  return mlir::success();
}

mlir::LogicalResult IotaOp::verify() {
  auto result = getResult().getType();
  auto element = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  if (!element || element.isSigned() || result.getShape().size() != 1 ||
      result.getAxisIds().size() != 1)
    return emitOpError("iota produces one unsigned integer logical axis");
  if (getEnd() <= getStart() || result.getShape()[0] != getEnd() - getStart())
    return emitOpError("iota shape must equal its positive [start, end) extent");
  return mlir::success();
}

mlir::LogicalResult NewOp::verify() {
  if (!isLocalValue(getResult().getType()))
    return emitOpError("new produces a local Value");
  if (getInitialized() != static_cast<bool>(getInitial()))
    return emitOpError("initialized must agree with the optional initial value");
  if (!getInitial())
    return mlir::success();
  mlir::Type initial = getInitial().getType();
  mlir::Type result = getResult().getType();
  if (initial == result)
    return mlir::success();
  if (!logicalShape(initial).empty() || logicalElement(initial) != logicalElement(result))
    return emitOpError("new initial value must match or scalar-broadcast to the result");
  return mlir::success();
}

mlir::LogicalResult MaterializeOp::verify() {
  if (getInput().getType() != getResult().getType())
    return emitOpError("materialize preserves its logical value type");
  return mlir::success();
}

mlir::LogicalResult AdmitOp::verify() {
  if (!isViewLike(getRegion().getType()) || !isLocalValue(getResult().getType()))
    return emitOpError("admit maps a View region to a local Value");
  mlir::Type resultElement = logicalElement(getResult().getType());
  if (auto encoded = mlir::dyn_cast<EncodingType>(resultElement)) {
    auto regionEncoding =
        mlir::cast<EncodingType>(logicalElement(getRegion().getType()));
    if (encoded.getFamily() != regionEncoding.getFamily() ||
        logicalShape(getResult().getType()).size() >
            logicalShape(getRegion().getType()).size() ||
        !llvm::equal(logicalAxes(getResult().getType()),
                     logicalAxes(getRegion().getType()).take_front(
                         logicalAxes(getResult().getType()).size())))
      return emitOpError("encoded admit may remove only its record axis");
  } else if (!sameDomain(getRegion().getType(), getResult().getType())) {
    return emitOpError("dense admit must preserve the logical domain");
  }
  return mlir::success();
}

mlir::LogicalResult CommitOp::verify() {
  if (!isLocalValue(getValue().getType()) || !isViewLike(getRegion().getType()))
    return emitOpError("commit maps a local Value to a View region");
  if (!sameDomain(getValue().getType(), getRegion().getType()))
    return emitOpError("commit value and destination must have the same domain");
  if (!isNumeric(getValue().getType()) &&
      logicalElement(getValue().getType()) != logicalElement(getRegion().getType()))
    return emitOpError("encoded commit requires identical encoding");
  return mlir::success();
}

mlir::LogicalResult SliceOp::verify() {
  if (!isViewLike(getBase().getType()))
    return emitOpError("slice base must be a View or slice");
  if (failed(verifyStringArray(*this, getSelectors(), "selectors")))
    return mlir::failure();
  size_t indexed = 0;
  for (mlir::Attribute attribute : getSelectors()) {
    llvm::StringRef selector = mlir::cast<mlir::StringAttr>(attribute).getValue();
    if (selector != "all" && selector != "domain" &&
        selector != "group_index" && selector != "index" &&
        selector != "gather")
      return emitOpError("unknown slice selector");
    if (selector != "all")
      ++indexed;
  }
  if (indexed != getIndices().size())
    return emitOpError("slice selectors and index operands disagree");
  return mlir::success();
}

mlir::LogicalResult FieldOp::verify() {
  mlir::Type owner = getOwner().getType();
  mlir::Type result = getResult().getType();
  if ((!isLocalValue(owner) && !isViewLike(owner)) ||
      !isEncoding(logicalElement(owner)) || getName().empty())
    return emitOpError(
        "field access requires a named field of an encoded Value or View region");
  if (isLocalValue(owner)) {
    if (!isLocalValue(result) || !isNumeric(result))
      return emitOpError("encoded Value field access produces a numeric local Value");
    return mlir::success();
  }
  auto resultSlice = mlir::dyn_cast<SliceType>(result);
  auto resultEncoding = resultSlice
                            ? mlir::dyn_cast<EncodingType>(resultSlice.getEncoding())
                            : EncodingType();
  if (!resultEncoding || resultEncoding.getKind() != "dense")
    return emitOpError("encoded View field access produces a dense numeric slice");
  return mlir::success();
}

mlir::LogicalResult ExtractOp::verify() {
  if (!mlir::isa<ValueType>(getInput().getType()) ||
      failed(verifyStringArray(*this, getSelectors(), "selectors")))
    return emitOpError("extract requires a shaped local Value and selectors");
  size_t cursor = 0;
  for (mlir::Attribute attribute : getSelectors()) {
    llvm::StringRef selector = mlir::cast<mlir::StringAttr>(attribute).getValue();
    if (selector == "all")
      continue;
    if (cursor >= getIndices().size())
      return emitOpError("extract selector has no corresponding index");
    mlir::Type indexType = getIndices()[cursor++].getType();
    if (selector != "gather")
      continue;
    auto value = mlir::dyn_cast<ValueType>(indexType);
    auto integer = value
                       ? mlir::dyn_cast<mlir::IntegerType>(value.getElementType())
                       : mlir::IntegerType();
    if (!integer || integer.isSigned())
      return emitOpError("gather selector requires shaped unsigned indices");
  }
  if (cursor != getIndices().size())
    return emitOpError("extract selectors and index operands disagree");
  return mlir::success();
}

mlir::LogicalResult UpdateOp::verify() {
  if (!mlir::isa<ValueType>(getInput().getType()) ||
      getResult().getType() != getInput().getType())
    return emitOpError("update preserves one shaped state Value");
  if (failed(verifyStringArray(*this, getSelectors(), "selectors")))
    return mlir::failure();
  if (logicalElement(getValue().getType()) !=
      mlir::cast<ValueType>(getInput().getType()).getElementType())
    return emitOpError("updated element type must match the state Value");
  return mlir::success();
}

mlir::LogicalResult UnaryOp::verify() {
  if (getInput().getType() != getResult().getType() ||
      !isNumeric(getInput().getType()))
    return emitOpError("unary operation preserves its logical type");
  if (getKind() != "neg" && getKind() != "abs" && getKind() != "exp")
    return emitOpError("unknown unary numeric operation");
  return mlir::success();
}

mlir::LogicalResult BinaryOp::verify() {
  if (failed(verifyPointwise(*this, getLhs().getType(), getRhs().getType(),
                             getResult().getType())))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult CompareOp::verify() {
  if (!isLocalValue(getLhs().getType()) || !isLocalValue(getRhs().getType()) ||
      logicalElement(getLhs().getType()) != logicalElement(getRhs().getType()))
    return emitOpError("compare operands must have matching element types");
  auto predicate = mlir::dyn_cast<mlir::IntegerType>(logicalElement(getResult().getType()));
  if (!predicate || predicate.getWidth() != 1)
    return emitOpError("compare result must have i1 elements");
  llvm::DenseSet<int64_t> expected;
  for (int64_t axis : logicalAxes(getLhs().getType()))
    expected.insert(axis);
  for (int64_t axis : logicalAxes(getRhs().getType()))
    expected.insert(axis);
  if (expected.size() != logicalAxes(getResult().getType()).size())
    return emitOpError("compare result must preserve the union of operand axes");
  for (auto [axis, extent] :
       llvm::zip(logicalAxes(getResult().getType()),
                 logicalShape(getResult().getType()))) {
    if (!expected.contains(axis))
      return emitOpError("compare result contains an unrelated axis");
    for (mlir::Type operand : {getLhs().getType(), getRhs().getType()}) {
      auto operandAxes = logicalAxes(operand);
      auto found = llvm::find(operandAxes, axis);
      if (found != operandAxes.end() &&
          logicalShape(operand)[static_cast<size_t>(found - operandAxes.begin())] !=
              extent)
        return emitOpError(
            "compare operands disagree on a shared logical axis extent");
    }
  }
  return mlir::success();
}

mlir::LogicalResult CastOp::verify() {
  if (!isNumeric(getInput().getType()) || !isNumeric(getResult().getType()) ||
      !sameDomain(getInput().getType(), getResult().getType()))
    return emitOpError("cast converts numeric elements and preserves the logical domain");
  auto rounding = (*this)->getAttrOfType<mlir::StringAttr>("rounding");
  auto saturate = (*this)->getAttrOfType<mlir::BoolAttr>("saturate");
  if (static_cast<bool>(rounding) != static_cast<bool>(saturate))
    return emitOpError("rounded narrowing requires both rounding and saturate attributes");
  if (rounding) {
    if (rounding.getValue() != "rne" && rounding.getValue() != "rtz" &&
        rounding.getValue() != "rdn" && rounding.getValue() != "rup")
      return emitOpError("unknown numeric narrowing rounding mode");
    if (!mlir::isa<mlir::FloatType>(logicalElement(getInput().getType())) ||
        !mlir::isa<mlir::IntegerType>(logicalElement(getResult().getType())))
      return emitOpError("rounded narrowing converts floating values to integers");
  }
  return mlir::success();
}

mlir::LogicalResult MacPairsOp::verify() {
  if (getGroup() != 2 || getOverflow() != "wrap")
    return emitOpError("mac_pairs is a wrapping group of exactly two elements");
  return mlir::success();
}

mlir::LogicalResult MacGroupsOp::verify() {
  if (getGroup() <= 0 || getOverflow() != "wrap")
    return emitOpError("mac_groups requires positive group and wrap overflow");
  return mlir::success();
}

mlir::LogicalResult WidenOp::verify() {
  if (!sameDomain(getInput().getType(), getResult().getType()))
    return emitOpError("widen preserves the logical domain");
  auto inputWidth = bitWidth(getInput().getType());
  auto resultWidth = bitWidth(getResult().getType());
  mlir::Type inputElement = logicalElement(getInput().getType());
  mlir::Type resultElement = logicalElement(getResult().getType());
  const bool integerToFloat = mlir::isa<mlir::IntegerType>(inputElement) &&
                              mlir::isa<mlir::FloatType>(resultElement);
  if (!inputWidth || !resultWidth ||
      (integerToFloat ? *resultWidth < *inputWidth
                      : *resultWidth <= *inputWidth))
    return emitOpError(
        "widen must increase width or convert an integer to no-narrower float");
  return mlir::success();
}

mlir::LogicalResult ReduceOp::verify() {
  if (!mlir::isa<ValueType>(getInput().getType()) ||
      !isLocalValue(getResult().getType()))
    return emitOpError("reduce consumes a shaped Value");
  auto input = mlir::cast<ValueType>(getInput().getType());
  auto inputShape = input.getShape().asArrayRef();
  auto inputAxes = input.getAxisIds().asArrayRef();
  auto resultShape = logicalShape(getResult().getType());
  auto resultAxes = logicalAxes(getResult().getType());
  if (getAxis() < 0 || static_cast<size_t>(getAxis()) >= inputShape.size() ||
      resultShape.size() + 1 != inputShape.size() ||
      logicalElement(getResult().getType()) != input.getElementType())
    return emitOpError("reduce result must remove exactly its selected axis");
  size_t resultIndex = 0;
  for (size_t inputIndex = 0; inputIndex < inputShape.size(); ++inputIndex) {
    if (inputIndex == static_cast<size_t>(getAxis()))
      continue;
    if (resultShape[resultIndex] != inputShape[inputIndex] ||
        resultAxes[resultIndex] != inputAxes[inputIndex])
      return emitOpError("reduce must preserve every non-reduced logical axis");
    ++resultIndex;
  }
  return mlir::success();
}

mlir::LogicalResult Fold2Op::verify() {
  if (!mlir::isa<ValueType>(getInput().getType()) ||
      !mlir::isa<ValueType>(getResult().getType()))
    return emitOpError("fold2 consumes and produces shaped Values");
  auto input = mlir::cast<ValueType>(getInput().getType());
  auto result = mlir::cast<ValueType>(getResult().getType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  llvm::ArrayRef<int64_t> inputShape = input.getShape().asArrayRef();
  llvm::ArrayRef<int64_t> resultShape = result.getShape().asArrayRef();
  if (inputShape.empty() || inputShape.size() != resultShape.size() ||
      inputShape.back() <= 0 || inputShape.back() % 2 ||
      resultShape.back() != inputShape.back() / 2 ||
      !resultElement || resultElement.getWidth() != 32 ||
      input.getAxisIds() != result.getAxisIds() ||
      !llvm::equal(inputShape.drop_back(), resultShape.drop_back()))
    return emitOpError("fold2 halves an even final logical extent");
  return mlir::success();
}

mlir::LogicalResult DotOp::verify() {
  return verifyContractLike(*this, getLhs(), getRhs(), getOver(), getAccType(),
                            getResult().getType());
}

mlir::LogicalResult ContractOp::verify() {
  return verifyContractLike(*this, getLhs(), getRhs(), getOver(), getAccType(),
                            getResult().getType());
}

mlir::LogicalResult OuterContractOp::verify() {
  return verifyContractLike(*this, getLhs(), getRhs(), getOver(), getAccType(),
                            getResult().getType());
}

mlir::LogicalResult LookupOp::verify() {
  auto index = mlir::dyn_cast<mlir::IntegerType>(logicalElement(getIndices().getType()));
  if (!isLocalValue(getTable().getType()) || !isNumeric(getTable().getType()) ||
      !index || index.isSigned())
    return emitOpError(
        "lookup requires a numeric local table and unsigned integer indices");
  if (!sameDomain(getIndices().getType(), getResult().getType()))
    return emitOpError("lookup result must follow the index domain");
  if (logicalElement(getTable().getType()) != logicalElement(getResult().getType()))
    return emitOpError("lookup result element type must match the table element type");
  return mlir::success();
}

mlir::LogicalResult InterleaveOp::verify() {
  if (getRows() <= 0 || !sameDomain(getView().getType(), getResult().getType()))
    return emitOpError("interleave requires positive rows and preserves the View domain");
  if (mlir::cast<EncodingType>(getResult().getType().getEncoding()).getKind() !=
      "derived_instance")
    return emitOpError("interleave in derive produces a concrete derived instance");
  if (!mlir::isa_and_nonnull<DeriveOp>(getOperation()->getParentOp()))
    return emitOpError("interleave is valid only inside a derive body");
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
