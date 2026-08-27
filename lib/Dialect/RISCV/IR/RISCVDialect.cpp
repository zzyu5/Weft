#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <optional>

using namespace weft::riscv;

#include "Weft/Dialect/RISCV/IR/RISCVOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Weft/Dialect/RISCV/IR/RISCVAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Weft/Dialect/RISCV/IR/RISCVTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/RISCV/IR/RISCVOps.cpp.inc"

namespace {

mlir::LogicalResult verifyShape(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> axes,
    bool allowEmpty = false) {
  if (!allowEmpty && shape.empty())
    return emitError() << "physical shaped value must have positive rank";
  if (shape.size() != axes.size())
    return emitError() << "logical shape and axis identity rank must match";
  llvm::DenseSet<int64_t> seen;
  for (auto [extent, axis] : llvm::zip(shape, axes)) {
    if (extent == 0)
      return emitError() << "logical extents are positive or symbolic";
    if (axis <= 0 || !seen.insert(axis).second)
      return emitError() << "logical axis identities are positive and unique";
  }
  return mlir::success();
}

bool isScalar(mlir::Type type) {
  return type.isIndex() || mlir::isa<mlir::IntegerType, mlir::FloatType>(type);
}

bool isPhysicalValue(mlir::Type type) {
  return isScalar(type) || mlir::isa<ValueType, FragmentType>(type);
}

llvm::ArrayRef<int64_t> shapeOf(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getShape().asArrayRef();
  if (auto memory = mlir::dyn_cast<MemDescType>(type))
    return memory.getShape().asArrayRef();
  if (auto local = mlir::dyn_cast<LocalType>(type))
    return local.getShape().asArrayRef();
  if (auto fragment = mlir::dyn_cast<FragmentType>(type))
    return fragment.getShape().asArrayRef();
  return {};
}

llvm::ArrayRef<int64_t> axesOf(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getAxisIds().asArrayRef();
  if (auto memory = mlir::dyn_cast<MemDescType>(type))
    return memory.getAxisIds().asArrayRef();
  if (auto local = mlir::dyn_cast<LocalType>(type))
    return local.getAxisIds().asArrayRef();
  if (auto fragment = mlir::dyn_cast<FragmentType>(type))
    return fragment.getAxisIds().asArrayRef();
  return {};
}

mlir::Type elementOf(mlir::Type type) {
  if (auto value = mlir::dyn_cast<ValueType>(type))
    return value.getElementType();
  if (auto fragment = mlir::dyn_cast<FragmentType>(type))
    return fragment.getElementType();
  if (auto local = mlir::dyn_cast<LocalType>(type))
    return local.getElementType();
  return type;
}

bool sameLogicalDomain(mlir::Type lhs, mlir::Type rhs) {
  return shapeOf(lhs) == shapeOf(rhs) && axesOf(lhs) == axesOf(rhs);
}

bool canRead(llvm::StringRef access) {
  return access == "read" || access == "readwrite";
}

bool canWrite(llvm::StringRef access) {
  return access == "write" || access == "readwrite";
}

std::optional<int64_t> checkedPositiveProduct(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values) {
    if (value <= 0 || result > std::numeric_limits<int64_t>::max() / value)
      return std::nullopt;
    result *= value;
  }
  return result;
}

mlir::LogicalResult verifyStaticRepresentation(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::ArrayRef<int64_t> shape, LayoutAttr layout) {
  auto time = layout.getTimeFactors().asArrayRef();
  auto lane = layout.getLaneFactors().asArrayRef();
  auto replica = layout.getReplicaFactors().asArrayRef();
  auto fragment = layout.getFragmentFactors().asArrayRef();
  auto local = layout.getLocalFactors().asArrayRef();
  if (shape.size() != time.size())
    return emitError() << "physical representation rank does not match its shape";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (shape[index] < 0)
      continue;
    int64_t factors[] = {time[index], lane[index], replica[index],
                         fragment[index], local[index]};
    auto represented = checkedPositiveProduct(factors);
    if (!represented || *represented != shape[index])
      return emitError() << "physical factors overflow or do not represent "
                         << shape[index] << " coordinates at dimension "
                         << index;
  }
  return mlir::success();
}

bool sameAxisMapping(ValueType lhs, size_t lhsIndex, ValueType rhs,
                     size_t rhsIndex) {
  auto left = lhs.getLayout();
  auto right = rhs.getLayout();
  return left.getTimeFactors()[lhsIndex] == right.getTimeFactors()[rhsIndex] &&
         left.getLaneFactors()[lhsIndex] == right.getLaneFactors()[rhsIndex] &&
         left.getReplicaFactors()[lhsIndex] ==
             right.getReplicaFactors()[rhsIndex] &&
         left.getFragmentFactors()[lhsIndex] ==
             right.getFragmentFactors()[rhsIndex] &&
         left.getLocalFactors()[lhsIndex] == right.getLocalFactors()[rhsIndex];
}

std::optional<int64_t> physicalPartCount(ValueType value) {
  auto time =
      checkedPositiveProduct(value.getLayout().getTimeFactors().asArrayRef());
  auto replicas = checkedPositiveProduct(
      value.getLayout().getReplicaFactors().asArrayRef());
  if (!time || !replicas ||
      *time > std::numeric_limits<int64_t>::max() / *replicas)
    return std::nullopt;
  return *time * *replicas;
}

bool isNumericPhysical(mlir::Type type) {
  mlir::Type element = elementOf(type);
  return element.isIndex() ||
         mlir::isa<mlir::IntegerType, mlir::FloatType>(element);
}

FieldOp sourceField(mlir::Value value) {
  while (mlir::Operation *definition = value.getDefiningOp()) {
    if (auto field = mlir::dyn_cast<FieldOp>(definition))
      return field;
    if (auto extract = mlir::dyn_cast<ExtractOp>(definition)) {
      value = extract.getInput();
      continue;
    }
    if (auto conversion = mlir::dyn_cast<ConvertLayoutOp>(definition)) {
      value = conversion.getInput();
      continue;
    }
    if (auto materialize = mlir::dyn_cast<RegisterMaterializeOp>(definition)) {
      value = materialize.getInput();
      continue;
    }
    break;
  }
  return {};
}

LoadOp sourceLoad(mlir::Value value) {
  while (mlir::Operation *definition = value.getDefiningOp()) {
    if (auto load = mlir::dyn_cast<LoadOp>(definition))
      return load;
    if (auto conversion = mlir::dyn_cast<ConvertLayoutOp>(definition)) {
      value = conversion.getInput();
      continue;
    }
    if (auto materialize = mlir::dyn_cast<RegisterMaterializeOp>(definition)) {
      value = materialize.getInput();
      continue;
    }
    break;
  }
  return {};
}

mlir::Value sourcePoint(mlir::Value value, int64_t axis) {
  llvm::SmallVector<mlir::Value> worklist{value};
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (auto point = mlir::dyn_cast<PointType>(current.getType());
        point && point.getDomain().getAxisId() == axis)
      return current;
    mlir::Operation *definition = current.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      continue;
    llvm::append_range(worklist, definition->getOperands());
  }
  return {};
}

bool supportsLayout(TargetAttr target, LayoutAttr layout) {
  int64_t elen = 0;
  for (int64_t supported : target.getSupportedSEW().asArrayRef())
    elen = std::max(elen, supported);
  const bool legalFractionalLMUL =
      layout && elen > 0 && layout.getLmulEighths() * elen >=
                                8 * layout.getSew();
  return layout && layout.getCarrier() == "rvv" && target.getHasRVV() &&
         target.getVlenBits() > 0 && target.getVectorRegisters() > 0 &&
         llvm::is_contained(target.getSupportedSEW().asArrayRef(),
                            layout.getSew()) &&
         llvm::is_contained(target.getLegalLMULEighths().asArrayRef(),
                            layout.getLmulEighths()) &&
         legalFractionalLMUL;
}

std::optional<int64_t> doubledPositive(int64_t value) {
  if (value <= 0 || value > std::numeric_limits<int64_t>::max() / 2)
    return std::nullopt;
  return value * 2;
}

bool hasGroupedMacOperandGeometry(mlir::Operation *operation, mlir::Value lhs,
                                  mlir::Value rhs, int64_t reductionAxis,
                                  AccessAttr lhsAccess, AccessAttr rhsAccess,
                                  int64_t group, LayoutAttr partialLayout) {
  auto lhsType = mlir::dyn_cast<ValueType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<ValueType>(rhs.getType());
  auto lhsInteger = lhsType
                        ? mlir::dyn_cast<mlir::IntegerType>(
                              lhsType.getElementType())
                        : mlir::IntegerType();
  auto rhsInteger = rhsType
                        ? mlir::dyn_cast<mlir::IntegerType>(
                              rhsType.getElementType())
                        : mlir::IntegerType();
  FieldOp lhsField = sourceField(lhs);
  FieldOp rhsField = sourceField(rhs);
  LoadOp lhsLoad = lhsField ? sourceLoad(lhsField.getOwner()) : LoadOp();
  LoadOp rhsLoad = rhsField ? sourceLoad(rhsField.getOwner()) : LoadOp();
  auto lhsMemory = lhsLoad ? lhsLoad.getRegion().getType() : MemDescType();
  mlir::Value lhsPoint = sourcePoint(lhs, reductionAxis);
  mlir::Value rhsPoint = sourcePoint(rhs, reductionAxis);
  auto kernel = operation->getParentOfType<KernelOp>();
  if (!kernel || !lhsType || !rhsType || !lhsInteger || !rhsInteger ||
      !lhsInteger.isUnsigned() || lhsInteger.getWidth() >= 8 ||
      !rhsInteger.isSigned() || rhsInteger.getWidth() != 8 || !lhsField ||
      !rhsField || !lhsLoad || !rhsLoad || !lhsPoint || !rhsPoint ||
      lhsPoint != rhsPoint || lhsAccess.getMapping() != "grouped_layered" ||
      rhsAccess.getMapping() != "natural" || group <= 0 ||
      lhsAccess.getGroupSize() <= 0 || lhsAccess.getLayerSize() <= 0 ||
      lhsAccess.getGroupSize() % lhsAccess.getLayerSize() ||
      lhsAccess.getLayerSize() % group || lhsAccess.getBitOffset() % 8 ||
      rhsAccess.getBitOffset() % 8 ||
      (lhsAccess.getOrder() != "lo_first" &&
       lhsAccess.getOrder() != "hi_first") ||
      !supportsLayout(kernel.getTarget(), lhsType.getLayout()) ||
      !supportsLayout(kernel.getTarget(), partialLayout) ||
      !kernel.getTarget().getHasWideningInteger())
    return false;
  int64_t laneAxis = 0;
  for (auto [axis, factor] :
       llvm::zip(lhsType.getAxisIds().asArrayRef(),
                 lhsType.getLayout().getLaneFactors().asArrayRef()))
    if (axis != reductionAxis && factor > 1) {
      laneAxis = axis;
      break;
    }
  bool hasCohortStride = false;
  if (lhsMemory && laneAxis > 0)
    for (auto [axis, stride] :
         llvm::zip(lhsMemory.getAxisIds().asArrayRef(),
                   lhsMemory.getStrides().asArrayRef()))
      if (axis == laneAxis && stride != 0)
        hasCohortStride = true;
  return lhsMemory && lhsMemory.getElements() > 0 &&
         (lhsMemory.getInterleaveRows() > 0 || hasCohortStride);
}

mlir::LogicalResult verifyBroadcastDomain(mlir::Operation *operation,
                                          mlir::Type lhs, mlir::Type rhs,
                                          mlir::Type result) {
  llvm::SmallVector<int64_t> expectedAxes(axesOf(lhs));
  for (int64_t axis : axesOf(rhs))
    if (!llvm::is_contained(expectedAxes, axis))
      expectedAxes.push_back(axis);
  if (!llvm::equal(expectedAxes, axesOf(result)))
    return operation->emitOpError(
        "pointwise result axes must follow lhs order then rhs-only axes");
  for (auto [axis, extent] : llvm::zip(axesOf(result), shapeOf(result))) {
    std::optional<int64_t> expectedExtent;
    for (mlir::Type operand : {lhs, rhs}) {
      auto axes = axesOf(operand);
      auto found = llvm::find(axes, axis);
      if (found == axes.end())
        continue;
      int64_t operandExtent =
          shapeOf(operand)[static_cast<size_t>(found - axes.begin())];
      if (!expectedExtent || *expectedExtent == 1)
        expectedExtent = operandExtent;
      else if (operandExtent != 1 && operandExtent != *expectedExtent)
        return operation->emitOpError(
            "pointwise operands disagree on a shared logical axis extent");
    }
    if (!expectedExtent || extent != *expectedExtent)
      return operation->emitOpError(
          "pointwise result extent must equal its broadcast operand extent");
  }
  auto resultValue = mlir::dyn_cast<ValueType>(result);
  if (!resultValue)
    return mlir::success();
  for (mlir::Type operandType : {lhs, rhs}) {
    auto operand = mlir::dyn_cast<ValueType>(operandType);
    if (!operand)
      continue;
    for (auto [position, axis] :
         llvm::enumerate(operand.getAxisIds().asArrayRef())) {
      auto found = llvm::find(resultValue.getAxisIds().asArrayRef(), axis);
      if (found == resultValue.getAxisIds().asArrayRef().end())
        return operation->emitOpError(
            "pointwise physical operand axis is absent from its result");
      size_t resultPosition = static_cast<size_t>(
          found - resultValue.getAxisIds().asArrayRef().begin());
      const int64_t operandExtent = operand.getShape()[position];
      const int64_t resultExtent = resultValue.getShape()[resultPosition];
      if (operandExtent == 1 && resultExtent != 1)
        continue;
      if (!sameAxisMapping(operand, position, resultValue, resultPosition))
        return operation->emitOpError(
            "pointwise non-broadcast axis requires an explicit layout conversion");
    }
  }
  return mlir::success();
}

mlir::LogicalResult verifyProjection(mlir::Operation *operation,
                                     mlir::Type base,
                                     mlir::ValueRange indices,
                                     mlir::ArrayAttr selectors,
                                     mlir::Type projected,
                                     bool allowGather) {
  if (selectors.size() > axesOf(base).size())
    return operation->emitOpError("projection has more selectors than input axes");
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  size_t indexCursor = 0;
  constexpr int64_t dynamicProjection = std::numeric_limits<int64_t>::min();
  for (size_t position = 0; position < axesOf(base).size(); ++position) {
    llvm::StringRef selector =
        position < selectors.size()
            ? mlir::cast<mlir::StringAttr>(selectors[position]).getValue()
            : llvm::StringRef("all");
    if (selector == "all") {
      expectedAxes.push_back(axesOf(base)[position]);
      expectedShape.push_back(shapeOf(base)[position]);
      continue;
    }
    if (indexCursor >= indices.size())
      return operation->emitOpError(
          "projection selector has no corresponding index operand");
    mlir::Value index = indices[indexCursor++];
    if (selector == "domain" || selector == "group_index") {
      auto point = mlir::dyn_cast<PointType>(index.getType());
      if (!point || point.getDomain().getAxisId() != axesOf(base)[position])
        return operation->emitOpError(
            "Level projection point must select the corresponding logical axis");
      if (selector == "domain") {
        expectedAxes.push_back(axesOf(base)[position]);
        expectedShape.push_back(dynamicProjection);
      }
      continue;
    }
    if (selector == "index") {
      if (!index.getType().isIndex())
        return operation->emitOpError(
            "scalar projection index must have index type");
      continue;
    }
    if (selector != "gather" || !allowGather)
      return operation->emitOpError("unknown or disallowed projection selector");
    auto gather = mlir::dyn_cast<ValueType>(index.getType());
    auto integer = gather
                       ? mlir::dyn_cast<mlir::IntegerType>(gather.getElementType())
                       : mlir::IntegerType();
    if (!integer || integer.isSigned())
      return operation->emitOpError(
          "gather projection requires shaped unsigned indices");
    for (auto [axis, extent] : llvm::zip(gather.getAxisIds().asArrayRef(),
                                        gather.getShape().asArrayRef())) {
      auto found = llvm::find(expectedAxes, axis);
      if (found == expectedAxes.end()) {
        expectedAxes.push_back(axis);
        expectedShape.push_back(extent);
      } else if (expectedShape[static_cast<size_t>(found - expectedAxes.begin())] !=
                 extent) {
        return operation->emitOpError(
            "gather projection disagrees on a shared logical axis extent");
      }
    }
  }
  if (indexCursor != indices.size() ||
      !llvm::equal(expectedAxes, axesOf(projected)) ||
      expectedShape.size() != shapeOf(projected).size())
    return operation->emitOpError(
        "projection operands and result logical axes do not match");
  for (auto [expected, actual] : llvm::zip(expectedShape, shapeOf(projected)))
    if (expected != dynamicProjection && expected != actual)
      return operation->emitOpError(
          "projection result has an incorrect logical extent");
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

mlir::LogicalResult verifyLeafOperation(mlir::Operation *operation) {
  auto leaf = operation->getAttrOfType<LeafAttr>("leaf");
  if (!leaf)
    return operation->emitOpError("target-local operation requires a leaf contract");
  return mlir::success();
}

bool exactLeaf(LeafAttr leaf, llvm::StringRef engine,
               llvm::StringRef family, llvm::StringRef instruction,
               llvm::StringRef mask, llvm::StringRef tail) {
  return leaf && leaf.getEngine() == engine && leaf.getFamily() == family &&
         leaf.getInstruction() == instruction &&
         leaf.getSpelling() == instruction && leaf.getMask() == mask &&
         leaf.getTail() == tail;
}

mlir::LogicalResult verifyConversion(mlir::Operation *operation,
                                     mlir::Type input, mlir::Type result) {
  if (!isPhysicalValue(input) || !isPhysicalValue(result) ||
      !sameLogicalDomain(input, result))
    return operation->emitOpError(
        "physical conversion preserves the logical value domain");
  return verifyLeafOperation(operation);
}

FragmentCapabilityAttr fragmentCapability(mlir::Operation *operation,
                                          llvm::StringRef instruction) {
  auto kernel = operation->getParentOfType<KernelOp>();
  if (!kernel)
    return {};
  for (mlir::Attribute attribute : kernel.getTarget().getFragments()) {
    auto capability = mlir::cast<FragmentCapabilityAttr>(attribute);
    if (capability.getInstruction() == instruction)
      return capability;
  }
  return {};
}

unsigned elementBitWidth(mlir::Type type) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  return 0;
}

bool integerSignednessMatches(mlir::Type type, llvm::StringRef signedness) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
  return integer && !integer.isSignless() &&
         ((signedness == "signed" && integer.isSigned()) ||
          (signedness == "unsigned" && integer.isUnsigned()));
}

EncodingDeclOp findEncodingDeclaration(mlir::Operation *operation,
                                       llvm::StringRef family) {
  EncodingDeclOp result;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](EncodingDeclOp declaration) {
      if (!result && declaration.getSymName() == family)
        result = declaration;
    });
  return result;
}

llvm::StringRef baseEncodingFamily(mlir::Operation *operation,
                                   weft::kernel::EncodingType encoding) {
  if (encoding.getKind() == "base" || encoding.getKind() == "dense")
    return encoding.getFamily();
  llvm::StringRef result;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](DerivedEncodingOp derived) {
      if (result.empty() && derived.getResultFamily() == encoding.getFamily() &&
          derived.getLayoutIdentity() == encoding.getLayoutIdentity() &&
          derived.getParameterValues() == encoding.getParameters().asArrayRef())
        result = derived.getSourceFamily();
    });
  return result;
}

} // namespace

mlir::LogicalResult LayoutAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef carrier, mlir::DenseI64ArrayAttr axisIds,
    mlir::DenseI64ArrayAttr timeFactors,
    mlir::DenseI64ArrayAttr laneFactors,
    mlir::DenseI64ArrayAttr replicaFactors,
    mlir::DenseI64ArrayAttr fragmentFactors,
    mlir::DenseI64ArrayAttr localFactors, int64_t sew,
    int64_t lmulEighths, int64_t vl, int64_t registerGroups,
    llvm::StringRef validity) {
  if (carrier != "unassigned" && carrier != "scalar" && carrier != "rvv" &&
      carrier != "ime" && carrier != "local")
    return emitError() << "unknown physical carrier";
  const size_t rank = axisIds.size();
  if (timeFactors.size() != rank || laneFactors.size() != rank ||
      replicaFactors.size() != rank || fragmentFactors.size() != rank ||
      localFactors.size() != rank)
    return emitError() << "every logical axis requires all five physical factors";
  llvm::DenseSet<int64_t> seen;
  for (int64_t axis : axisIds.asArrayRef())
    if (axis <= 0 || !seen.insert(axis).second)
      return emitError() << "layout axis identities are positive and unique";
  auto validFactors = [&](mlir::DenseI64ArrayAttr factors) {
    return llvm::all_of(factors.asArrayRef(), [&](int64_t factor) {
      return carrier == "unassigned" ? factor >= 0 : factor > 0;
    });
  };
  if (!validFactors(timeFactors) || !validFactors(laneFactors) ||
      !validFactors(replicaFactors) || !validFactors(fragmentFactors) ||
      !validFactors(localFactors))
    return emitError() << "physical factors must be positive after layout assignment";
  if (carrier == "unassigned") {
    if (sew != 0 || lmulEighths != 0 || vl != 0 || registerGroups != 0 ||
        validity != "unassigned")
      return emitError() << "unassigned layout cannot carry physical decisions";
    return mlir::success();
  }
  for (auto factors : {timeFactors, laneFactors, replicaFactors,
                       fragmentFactors, localFactors})
    if (!checkedPositiveProduct(factors.asArrayRef()))
      return emitError()
             << "physical factor product overflows the representable program geometry";
  if (sew <= 0 || vl <= 0 || registerGroups < 0)
    return emitError() << "assigned layout requires positive SEW/vl and resources";
  if (carrier == "rvv" && lmulEighths <= 0)
    return emitError() << "RVV layout requires a legal positive LMUL";
  if (carrier != "rvv" && lmulEighths != 0)
    return emitError() << "only an RVV layout carries LMUL";
  if (validity != "full" && validity != "tail" && validity != "mask")
    return emitError() << "assigned layout has unknown validity semantics";
  return mlir::success();
}

mlir::LogicalResult FragmentPackingAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef schema, mlir::DenseI64ArrayAttr axisOrder,
    int64_t rowsPerTile, int64_t columnsPerTile,
    llvm::StringRef tileOrder, llvm::StringRef elementOrder,
    int64_t storageBits, int64_t alignment) {
  if (schema.empty() || rowsPerTile <= 0 || columnsPerTile <= 0 ||
      !axisOrder || axisOrder.size() != 2 || axisOrder[0] == axisOrder[1] ||
      llvm::any_of(axisOrder.asArrayRef(),
                   [](int64_t axis) { return axis < 0 || axis > 1; }) ||
      storageBits <= 0 || storageBits % 8 || alignment <= 0)
    return emitError()
           << "fragment packing requires one rank-two byte-addressable tiled storage mapping";
  auto validOrder = [](llvm::StringRef order) {
    return order == "row_major" || order == "column_major";
  };
  if (!validOrder(tileOrder) || !validOrder(elementOrder))
    return emitError()
           << "fragment packing requires explicit tile and in-tile orders";
  return mlir::success();
}

mlir::LogicalResult FragmentCapabilityAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef instruction, llvm::StringRef lhsSignedness,
    llvm::StringRef rhsSignedness, int64_t lhsBits, int64_t rhsBits,
    int64_t accumulatorBits, int64_t mFactor, int64_t nFactor,
    int64_t kFactor, int64_t lhsResourceGroups, int64_t rhsResourceGroups,
    int64_t accumulatorResourceGroups, FragmentPackingAttr lhsPacking,
    FragmentPackingAttr rhsPacking, FragmentPackingAttr accumulatorPacking,
    int64_t mmaGroups, int64_t mmaChunks, mlir::ArrayAttr clobbers,
    bool volatileAsm, bool memoryClobber) {
  if (instruction.empty() || lhsBits <= 0 || rhsBits <= 0 ||
      accumulatorBits <= 0 || mFactor <= 0 || nFactor <= 0 || kFactor <= 0 ||
      (lhsSignedness != "signed" && lhsSignedness != "unsigned") ||
      (rhsSignedness != "signed" && rhsSignedness != "unsigned") ||
      lhsResourceGroups <= 0 || rhsResourceGroups <= 0 ||
      accumulatorResourceGroups <= 0 || !lhsPacking || !rhsPacking ||
      !accumulatorPacking ||
      mmaGroups <= 0 || mmaChunks <= 0 || !volatileAsm || !memoryClobber ||
      clobbers.empty() ||
      llvm::any_of(clobbers, [](mlir::Attribute attribute) {
        auto value = mlir::dyn_cast<mlir::StringAttr>(attribute);
        return !value || value.getValue().empty();
      }))
    return emitError() << "fragment capability fields must be positive and complete";
  if (lhsPacking.getRowsPerTile() * lhsPacking.getColumnsPerTile() *
              lhsPacking.getStorageBits() / 8 <=
          0 ||
      rhsPacking.getRowsPerTile() * rhsPacking.getColumnsPerTile() *
              rhsPacking.getStorageBits() / 8 <=
          0 ||
      accumulatorPacking.getRowsPerTile() *
              accumulatorPacking.getColumnsPerTile() *
              accumulatorPacking.getStorageBits() / 8 <=
          0)
    return emitError() << "fragment packing tile geometry is incomplete";
  if (mFactor % lhsPacking.getRowsPerTile() ||
      kFactor % lhsPacking.getColumnsPerTile() ||
      nFactor % rhsPacking.getRowsPerTile() ||
      kFactor % rhsPacking.getColumnsPerTile() ||
      mFactor % accumulatorPacking.getRowsPerTile() ||
      nFactor % accumulatorPacking.getColumnsPerTile() ||
      lhsPacking.getStorageBits() < lhsBits ||
      rhsPacking.getStorageBits() < rhsBits ||
      accumulatorPacking.getStorageBits() < accumulatorBits)
    return emitError()
           << "fragment packing geometry does not cover the capability operands";
  return mlir::success();
}

mlir::LogicalResult TargetAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef triple, llvm::StringRef march, llvm::StringRef abi,
    bool hasRVV, bool hasVectorF16, bool hasIndexedMemory,
    bool hasSegmentMemory, bool hasWideningInteger, bool hasWideningFloat,
    int64_t vlenBits, int64_t vectorRegisters,
    int64_t maxPrivateStackBytes, mlir::DenseI64ArrayAttr supportedSEW,
    mlir::DenseI64ArrayAttr legalLMULEighths, mlir::ArrayAttr fragments) {
  if (triple.empty() || march.empty() || abi.empty() || vlenBits <= 0 ||
      vectorRegisters <= 0 || maxPrivateStackBytes < 0 || supportedSEW.empty() ||
      legalLMULEighths.empty())
    return emitError() << "RISC-V target facts must be complete";
  if (!hasRVV || (!hasWideningInteger && !hasWideningFloat))
    return emitError()
           << "current RISC-V physical dialect requires RVV and at least one widening class";
  (void)hasVectorF16;
  (void)hasIndexedMemory;
  (void)hasSegmentMemory;
  if (llvm::any_of(supportedSEW.asArrayRef(), [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(legalLMULEighths.asArrayRef(),
                   [](int64_t value) { return value <= 0; }) ||
      llvm::any_of(fragments, [](mlir::Attribute value) {
        return !mlir::isa<FragmentCapabilityAttr>(value);
      }))
    return emitError() << "target capability domains are malformed";
  return mlir::success();
}

mlir::LogicalResult AccessAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef form, llvm::StringRef mapping, int64_t alignment,
    int64_t indexSEW, int64_t segmentFields, int64_t groupSize,
    int64_t layerSize, int64_t joinFields, int64_t joinLowBits,
    int64_t joinRole, int64_t bitOffset, int64_t storageBits,
    llvm::StringRef order) {
  if (form != "unassigned" && form != "unit" && form != "strided" &&
      form != "indexed" && form != "segment" && form != "register" &&
      form != "local")
    return emitError() << "unknown physical memory form";
  if (mapping != "dense" && mapping != "natural" &&
      mapping != "grouped_layered" && mapping != "joined" &&
      mapping != "opaque")
    return emitError() << "unknown storage mapping";
  if (alignment <= 0 || indexSEW < 0 || segmentFields < 0 || groupSize < 0 ||
      layerSize < 0 || joinFields < 0 || joinLowBits < 0 || joinRole < 0 ||
      bitOffset < 0 || storageBits < 0)
    return emitError() << "memory access parameters are invalid";
  if (order != "none" && order != "lo_first" && order != "hi_first")
    return emitError() << "unknown bit-layer order";
  if (form == "indexed" && indexSEW == 0)
    return emitError() << "indexed access requires an index SEW";
  if (form == "segment" && segmentFields < 2)
    return emitError() << "segment access requires at least two fields";
  if (mapping == "joined" &&
      (groupSize <= 0 || joinFields <= 1 || joinLowBits <= 0 ||
       joinRole >= joinFields))
    return emitError() << "joined storage mapping requires complete field geometry";
  if (mapping == "grouped_layered" &&
      (groupSize <= 0 || layerSize <= 0 || groupSize % layerSize ||
       storageBits <= 0 || (order != "lo_first" && order != "hi_first")))
    return emitError()
           << "grouped/layered storage mapping requires complete group, layer, bit, and order geometry";
  return mlir::success();
}

mlir::LogicalResult ScheduleAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef kind, int64_t unroll, int64_t pipelineDepth,
    int64_t bufferCount, int64_t prefetchDistance) {
  if (kind != "sequential" && kind != "pipelined")
    return emitError() << "unknown physical schedule kind";
  if (unroll <= 0 || pipelineDepth <= 0 || bufferCount <= 0 ||
      prefetchDistance < 0)
    return emitError() << "schedule parameters must be positive";
  if ((pipelineDepth == 1) != (kind == "sequential") ||
      bufferCount < pipelineDepth)
    return emitError() << "schedule kind, depth, and buffer count disagree";
  return mlir::success();
}

mlir::LogicalResult LevelAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    int64_t domainId, int64_t axisId, llvm::StringRef relation,
    int64_t stateBirthCount, int64_t stagedBirthCount, int64_t handoffCount,
    llvm::StringRef direction) {
  if (domainId <= 0 || axisId <= 0)
    return emitError() << "physical Level requires positive domain and axis identities";
  if (relation != "rows" && relation != "cols" && relation != "tiles" &&
      relation != "blocks" && relation != "subs")
    return emitError() << "unknown physical Level relation";
  if (stateBirthCount < 0 || stagedBirthCount < 0 || handoffCount < 0)
    return emitError() << "physical Level counts cannot be negative";
  if (direction != "ascending" && direction != "descending")
    return emitError() << "physical Level requires one ordered direction";
  return mlir::success();
}

mlir::LogicalResult ImplementationAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef engine, llvm::StringRef family,
    llvm::StringRef operation, mlir::DenseI64ArrayAttr parameters) {
  if (engine != "scalar" && engine != "rvv" && engine != "ime" &&
      engine != "transfer")
    return emitError() << "unknown physical implementation engine";
  if (family.empty() || operation.empty())
    return emitError() << "physical implementation anchor must be complete";
  if (llvm::any_of(parameters.asArrayRef(),
                   [](int64_t value) { return value < 0; }))
    return emitError() << "physical implementation parameters cannot be negative";
  return mlir::success();
}

mlir::LogicalResult LeafAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef engine, llvm::StringRef family,
    llvm::StringRef instruction, llvm::StringRef spelling,
    int64_t operandGroups, int64_t resultGroups, int64_t temporaryGroups,
    int64_t fragmentGroups, llvm::StringRef mask, llvm::StringRef tail,
    mlir::DenseI64ArrayAttr parameters, int64_t localBytes) {
  if (engine != "unselected" && engine != "scalar" && engine != "rvv" &&
      engine != "ime" && engine != "transfer")
    return emitError() << "unknown physical engine";
  if (operandGroups < 0 || resultGroups < 0 || temporaryGroups < 0 ||
      fragmentGroups < 0 || localBytes < 0)
    return emitError() << "leaf resource counts cannot be negative";
  if (mask != "none" && mask != "explicit" && mask != "tail")
    return emitError() << "unknown leaf mask policy";
  if (tail != "exact" && tail != "agnostic")
    return emitError() << "unknown leaf tail policy";
  if (engine == "unselected") {
    if (!family.empty() || !instruction.empty() || !spelling.empty())
      return emitError() << "unselected leaf cannot claim a target operation";
  } else if (family.empty() || instruction.empty() || spelling.empty()) {
    return emitError() << "selected leaf contract must be complete";
  }
  if (llvm::any_of(parameters.asArrayRef(), [](int64_t value) { return value < 0; }))
    return emitError() << "leaf parameters cannot be negative";
  return mlir::success();
}

mlir::LogicalResult ConversionAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef kind, llvm::StringRef effect, int64_t temporaryGroups) {
  if (kind != "splat" && kind != "extract" && kind != "reshape" &&
      kind != "tuple" && kind != "register_to_lane" &&
      kind != "time_to_lane" && kind != "lane_to_register" &&
      kind != "rvv_to_fragment" &&
      kind != "fragment_to_rvv" && kind != "local_load" &&
      kind != "local_store")
    return emitError() << "unknown physical conversion kind";
  if (effect != "pure" && effect != "read" && effect != "write" &&
      effect != "handoff")
    return emitError() << "unknown physical conversion effect";
  if (temporaryGroups < 0)
    return emitError() << "conversion temporary resources cannot be negative";
  return mlir::success();
}

mlir::LogicalResult ValueType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type elementType, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds, LayoutAttr layout) {
  if (!elementType || !layout)
    return emitError() << "physical value requires element and layout types";
  if (failed(verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef())))
    return mlir::failure();
  if (layout.getAxisIds() != axisIds)
    return emitError() << "physical layout must preserve every logical axis identity";
  if (layout.getCarrier() != "unassigned" &&
      failed(verifyStaticRepresentation(emitError, shape.asArrayRef(), layout)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult MemDescType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type encoding, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds, mlir::DenseI64ArrayAttr strides,
    mlir::DenseI64ArrayAttr origins, int64_t alignment,
    llvm::StringRef addressClass, llvm::StringRef access, int64_t aliasSet,
    llvm::StringRef layoutIdentity, int64_t storageBits, int64_t elements,
    int64_t interleaveRows) {
  if (!mlir::isa<weft::kernel::EncodingType>(encoding))
    return emitError() << "memory descriptor requires a canonical Encoding identity";
  if (failed(verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef(), true)))
    return mlir::failure();
  if (strides.size() != shape.size() || origins.size() != shape.size())
    return emitError() << "descriptor stride/origin rank must match logical rank";
  if (alignment <= 0 || storageBits <= 0 || elements <= 0 || interleaveRows < 0)
    return emitError() << "descriptor storage facts are invalid";
  if (addressClass != "pinned" && addressClass != "slice" &&
      addressClass != "local")
    return emitError() << "unknown descriptor address class";
  if (access != "none" && access != "read" && access != "write" &&
      access != "readwrite")
    return emitError() << "unknown descriptor access mode";
  if (addressClass == "pinned" && (aliasSet < 0 || layoutIdentity.empty()))
    return emitError() << "pinned descriptor requires alias and layout identity";
  return mlir::success();
}

mlir::LogicalResult LocalType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type elementType, mlir::DenseI64ArrayAttr shape,
    mlir::DenseI64ArrayAttr axisIds, int64_t sizeBytes, int64_t alignment,
    int64_t aliasSet, llvm::StringRef purpose, int64_t ownerDomainId,
    int64_t birthId, int64_t lifetimeEndDomainId, llvm::StringRef schema) {
  if (!elementType || failed(verifyShape(emitError, shape.asArrayRef(),
                                         axisIds.asArrayRef(), true)))
    return mlir::failure();
  if ((sizeBytes == 0 || sizeBytes < -1) || alignment <= 0 || aliasSet < 0)
    return emitError()
           << "local storage size is positive or dynamic (-1), and alignment/alias are positive";
  if (purpose != "spill" && purpose != "pack" && purpose != "pipeline" &&
      purpose != "handoff")
    return emitError() << "unknown local storage purpose";
  if (ownerDomainId < 0 || birthId < 0 || lifetimeEndDomainId < ownerDomainId ||
      schema.empty())
    return emitError() << "local storage requires explicit owner, birth, lifetime, and schema";
  if (sizeBytes >= 0) {
    int64_t logicalElements = 1;
    for (int64_t extent : shape.asArrayRef()) {
      if (extent <= 0)
        return emitError()
               << "static local storage cannot carry a symbolic logical shape";
      if (logicalElements > std::numeric_limits<int64_t>::max() / extent)
        return emitError() << "local storage logical shape overflows";
      logicalElements *= extent;
    }
    const unsigned bits = elementBitWidth(elementType);
    if (!bits)
      return emitError() << "local storage element type has no fixed width";
    if (logicalElements >
        (std::numeric_limits<int64_t>::max() - 7) / bits)
      return emitError() << "local storage byte size overflows";
    const int64_t minimumBytes = (logicalElements * bits + 7) / 8;
    if (sizeBytes < minimumBytes)
      return emitError()
             << "local storage does not cover its complete logical value";
  }
  return mlir::success();
}

mlir::LogicalResult FragmentType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef family, llvm::StringRef role, FragmentPackingAttr packing,
    mlir::Type elementType,
    mlir::DenseI64ArrayAttr shape, mlir::DenseI64ArrayAttr axisIds,
    LayoutAttr layout, int64_t resourceGroups) {
  if (family.empty() ||
      (role != "lhs" && role != "rhs" && role != "accumulator") ||
      !packing || !elementType || resourceGroups <= 0 || !layout ||
      layout.getCarrier() != "ime")
    return emitError() << "fragment requires one complete IME representation";
  if (failed(verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef())) ||
      layout.getAxisIds() != axisIds)
    return mlir::failure();
  if (llvm::any_of(shape.asArrayRef(), [](int64_t extent) {
        return extent <= 0;
      }))
    return emitError() << "extension fragment shape must be fully static";
  const unsigned bits = elementBitWidth(elementType);
  if (!bits || packing.getStorageBits() < bits)
    return emitError()
           << "fragment packing storage width must cover its logical element";
  auto order = packing.getAxisOrder().asArrayRef();
  if (shape.size() != 2 || order.size() != 2 ||
      shape[order[0]] % packing.getRowsPerTile() ||
      shape[order[1]] % packing.getColumnsPerTile())
    return emitError()
           << "fragment logical shape must tile exactly under its physical axis order";
  if (failed(verifyStaticRepresentation(emitError, shape.asArrayRef(), layout)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult WindowType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef family, mlir::Type lhsType, mlir::Type rhsType,
    mlir::Type resultType,
    int64_t reductionAxis, int64_t slots, int64_t termsPerSlot,
    int64_t resultParts, LayoutAttr partialLayout, LayoutAttr resultLayout,
    int64_t resourceGroups) {
  if (family != "grouped-mac" && family != "encoded-dot" &&
      family != "contract")
    return emitError() << "unknown physical operand-window family";
  auto lhs = mlir::dyn_cast<ValueType>(lhsType);
  auto rhs = mlir::dyn_cast<ValueType>(rhsType);
  auto result = mlir::dyn_cast<ValueType>(resultType);
  if (!lhs || !rhs || !result ||
      reductionAxis <= 0 || slots <= 0 ||
      termsPerSlot <= 0 || resultParts <= 0 || !partialLayout || !resultLayout ||
      partialLayout.getCarrier() != "rvv" || resultLayout.getCarrier() != "rvv" ||
      result.getLayout() != resultLayout || resourceGroups <= 0)
    return emitError() << "physical operand window requires complete typed resources";
  if (!llvm::is_contained(lhs.getAxisIds().asArrayRef(), reductionAxis) ||
      !llvm::is_contained(rhs.getAxisIds().asArrayRef(), reductionAxis) ||
      llvm::is_contained(result.getAxisIds().asArrayRef(), reductionAxis))
    return emitError()
           << "operand window must consume one shared reduction axis and preserve only free axes";
  auto extentFor = [&](ValueType value, int64_t axis) {
    auto found = llvm::find(value.getAxisIds().asArrayRef(), axis);
    return found == value.getAxisIds().asArrayRef().end()
               ? int64_t(0)
               : value.getShape()[static_cast<size_t>(
                     found - value.getAxisIds().asArrayRef().begin())];
  };
  const int64_t lhsReduction = extentFor(lhs, reductionAxis);
  const int64_t rhsReduction = extentFor(rhs, reductionAxis);
  if (lhsReduction > 0 && rhsReduction > 0 && lhsReduction != rhsReduction)
    return emitError()
           << "operand window reduction-axis extents must agree";
  llvm::SmallVector<int64_t> expectedFreeAxes;
  llvm::SmallVector<int64_t> expectedFreeShape;
  auto appendFree = [&](ValueType value) -> mlir::LogicalResult {
    for (auto [axis, extent] : llvm::zip(value.getAxisIds().asArrayRef(),
                                        value.getShape().asArrayRef())) {
      if (axis == reductionAxis)
        continue;
      auto found = llvm::find(expectedFreeAxes, axis);
      if (found == expectedFreeAxes.end()) {
        expectedFreeAxes.push_back(axis);
        expectedFreeShape.push_back(extent);
        continue;
      }
      size_t position =
          static_cast<size_t>(found - expectedFreeAxes.begin());
      if (expectedFreeShape[position] != extent)
        return emitError()
               << "operand window operands disagree on a shared free-axis extent";
    }
    return mlir::success();
  };
  if (failed(appendFree(lhs)) || failed(appendFree(rhs)))
    return mlir::failure();
  if (!llvm::equal(expectedFreeAxes, result.getAxisIds().asArrayRef()) ||
      !llvm::equal(expectedFreeShape, result.getShape().asArrayRef()))
    return emitError()
           << "operand window result must preserve ordered free logical axes and extents";
  if (partialLayout.getAxisIds() != lhs.getAxisIds() &&
      partialLayout.getAxisIds() != result.getAxisIds())
    return emitError()
           << "operand window partial layout must describe its lhs or result domain";
  auto timeParts =
      checkedPositiveProduct(resultLayout.getTimeFactors().asArrayRef());
  auto registerParts =
      checkedPositiveProduct(resultLayout.getReplicaFactors().asArrayRef());
  const bool partsOverflow =
      !timeParts || !registerParts ||
      *timeParts > std::numeric_limits<int64_t>::max() / *registerParts;
  const int64_t expectedParts =
      partsOverflow ? -1 : *timeParts * *registerParts;
  if (resultParts != expectedParts)
    return emitError()
           << "physical operand window result parts disagree with its layout";
  return mlir::success();
}

mlir::LogicalResult LayeredWindowType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ValueType fieldType, ValueType resultType, int64_t reductionAxis,
    int64_t layers, int64_t windowsPerLayer, int64_t resourceGroups) {
  if (!fieldType || !resultType || reductionAxis <= 0 || layers <= 1 ||
      windowsPerLayer <= 0 || resourceGroups <= 0 ||
      fieldType.getElementType() != resultType.getElementType() ||
      resultType.getLayout().getCarrier() != "rvv")
    return emitError()
           << "layered storage window requires typed field/result values and positive geometry";
  auto fieldAxis = llvm::find(fieldType.getAxisIds().asArrayRef(), reductionAxis);
  auto resultAxis = llvm::find(resultType.getAxisIds().asArrayRef(), reductionAxis);
  if (fieldAxis == fieldType.getAxisIds().asArrayRef().end() ||
      resultAxis == resultType.getAxisIds().asArrayRef().end())
    return emitError()
           << "layered storage window must preserve its reduction-axis identity";
  const size_t resultPosition = static_cast<size_t>(
      resultAxis - resultType.getAxisIds().asArrayRef().begin());
  const size_t fieldPosition = static_cast<size_t>(
      fieldAxis - fieldType.getAxisIds().asArrayRef().begin());
  if (resultType.getLayout().getTimeFactors()[resultPosition] != 1 ||
      resultType.getLayout().getLaneFactors()[resultPosition] <= 1 ||
      resultType.getShape()[resultPosition] !=
          resultType.getLayout().getLaneFactors()[resultPosition] ||
      resultType.getShape()[resultPosition] > fieldType.getShape()[fieldPosition] ||
      resultType.getLayout().getVl() < resultType.getShape()[resultPosition] ||
      resourceGroups != resultType.getLayout().getRegisterGroups())
    return emitError()
           << "layered storage window result must be one complete, resource-exact RVV lane window";
  if (fieldType.getAxisIds() != resultType.getAxisIds())
    return emitError()
           << "layered storage window must preserve every logical axis identity";
  for (size_t position = 0; position < resultType.getShape().size(); ++position) {
    if (position == resultPosition)
      continue;
    if (fieldType.getShape()[position] != resultType.getShape()[position] ||
        resultType.getLayout().getTimeFactors()[position] != 1 ||
        resultType.getLayout().getLaneFactors()[position] != 1 ||
        (resultType.getShape()[position] > 0
             ? resultType.getLayout().getReplicaFactors()[position] !=
                   resultType.getShape()[position]
             : resultType.getLayout().getReplicaFactors()[position] <= 0) ||
        resultType.getLayout().getFragmentFactors()[position] != 1 ||
        resultType.getLayout().getLocalFactors()[position] != 1)
      return emitError()
             << "layered storage window free axes must be preserved as register replicas";
  }
  return mlir::success();
}

mlir::LogicalResult DomainType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef axisName, int64_t domainId, int64_t parentDomainId,
    int64_t axisId, llvm::StringRef relation, llvm::StringRef tail) {
  if (axisName.empty() ||
      (relation != "root" && relation != "rows" && relation != "cols" &&
       relation != "tiles" && relation != "blocks" && relation != "subs") ||
      (tail != "exact" && tail != "tail"))
    return emitError() << "invalid physical domain identity";
  if (relation == "root") {
    if (domainId != 0 || parentDomainId != -1 || axisId != 0)
      return emitError() << "root domain identity is fixed";
  } else if (domainId <= 0 || parentDomainId < 0 || axisId <= 0) {
    return emitError() << "child domain identities must be positive";
  }
  return mlir::success();
}

mlir::LogicalResult PointType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    DomainType domain) {
  if (!domain)
    return emitError() << "point requires one physical domain";
  return mlir::success();
}

mlir::LogicalResult EncodingDeclOp::verify() {
  const size_t count = getFieldNames().size();
  if (getKind() != "base" || getLayoutIdentity() != getSymName() ||
      getAlignment() <= 0 || getElements() <= 0 || getStorageBits() <= 0 ||
      count == 0 || getFieldTypes().size() != count ||
      getFieldShapes().size() != count || getFieldLayouts().size() != count ||
      getFieldBitOffsets().size() != count ||
      getFieldStorageBits().size() != count)
    return emitOpError("physical Encoding declaration is incomplete");
  if (failed(verifyStringArray(*this, getFieldNames(), "field_names")))
    return mlir::failure();
  int64_t covered = 0;
  for (auto [offset, width] : llvm::zip(getFieldBitOffsets(),
                                       getFieldStorageBits())) {
    if (offset < 0 || width <= 0 || offset + width > getStorageBits())
      return emitOpError("Encoding field storage span is invalid");
    covered = std::max(covered, offset + width);
  }
  for (size_t index = 0; index < getPadding().size(); index += 3) {
    if (index + 2 >= getPadding().size())
      return emitOpError("padding is an offset,width,fill sequence");
    covered = std::max(covered, getPadding()[index] + getPadding()[index + 1]);
  }
  if (covered != getStorageBits())
    return emitOpError("Encoding fields and padding must cover the record");
  return mlir::success();
}

mlir::LogicalResult DerivedEncodingOp::verify() {
  if (getSourceFamily().empty() || getResultFamily().empty() ||
      getSourceFamily() == getResultFamily() || getLayoutIdentity().empty() ||
      getParameterNames().empty() ||
      getParameterNames().size() != getParameterValues().size() ||
      getInterleaveRows() <= 0)
    return emitOpError("derived Encoding declaration is incomplete");
  if (failed(verifyStringArray(*this, getParameterNames(), "parameter_names")))
    return mlir::failure();
  llvm::DenseSet<llvm::StringRef> names;
  for (auto [nameAttr, value] :
       llvm::zip(getParameterNames(), getParameterValues())) {
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(nameAttr).getValue();
    if (name.empty() || !names.insert(name).second || value <= 0)
      return emitOpError(
          "derived Encoding parameters must be named uniquely and be positive");
  }
  if (getParameterValues().size() != 1 ||
      getParameterValues()[0] != getInterleaveRows())
    return emitOpError(
        "the supported derived Encoding is exactly one parameterized interleave");
  auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
  unsigned sourceDeclarations = 0;
  unsigned identicalDerivations = 0;
  if (module)
    module.walk([&](mlir::Operation *operation) {
      if (auto declaration = mlir::dyn_cast<EncodingDeclOp>(operation)) {
        if (declaration.getSymName() == getSourceFamily())
          ++sourceDeclarations;
        return;
      }
      auto derived = mlir::dyn_cast<DerivedEncodingOp>(operation);
      if (derived && derived.getResultFamily() == getResultFamily() &&
          derived.getLayoutIdentity() == getLayoutIdentity() &&
          derived.getParameterValues() == getParameterValues())
        ++identicalDerivations;
    });
  if (sourceDeclarations != 1 || identicalDerivations != 1)
    return emitOpError(
        "derived Encoding must uniquely reference one base declaration and one concrete identity");
  return mlir::success();
}

mlir::LogicalResult ArtifactPackOp::verify() {
  if (getSourceFamily().empty() || getResultFamily().empty() ||
      getSourceFamily() == getResultFamily() || getLayoutIdentity().empty() ||
      getRecordBytes() <= 0 || getElements() <= 0 || getInterleaveRows() <= 0)
    return emitOpError("artifact pack identity and storage geometry must be complete");
  mlir::Block &size = getSizeBody().front();
  if (size.getNumArguments() != 2 ||
      !llvm::all_of(size.getArgumentTypes(),
                    [](mlir::Type type) { return type.isIndex(); }) ||
      !mlir::isa<ArtifactSizeYieldOp>(size.getTerminator()))
    return emitOpError("artifact packed-size region takes M/K and yields one index");
  mlir::Block &pack = getPackBody().front();
  if (pack.getNumArguments() != 4 ||
      !mlir::isa<MemDescType>(pack.getArgument(0).getType()) ||
      !mlir::isa<MemDescType>(pack.getArgument(1).getType()) ||
      !pack.getArgument(2).getType().isIndex() ||
      !pack.getArgument(3).getType().isIndex() ||
      !mlir::isa<ArtifactReturnOp>(pack.getTerminator()))
    return emitOpError(
        "artifact pack region takes source/target descriptors and M/K indices");
  auto source = mlir::cast<MemDescType>(pack.getArgument(0).getType());
  auto target = mlir::cast<MemDescType>(pack.getArgument(1).getType());
  auto sourceEncoding = mlir::cast<weft::kernel::EncodingType>(source.getEncoding());
  auto targetEncoding = mlir::cast<weft::kernel::EncodingType>(target.getEncoding());
  if (sourceEncoding.getFamily() != getSourceFamily() ||
      targetEncoding.getFamily() != getResultFamily() ||
      targetEncoding.getLayoutIdentity() != getLayoutIdentity() ||
      target.getInterleaveRows() != getInterleaveRows())
    return emitOpError("artifact pack region descriptors disagree with its Encoding identity");
  return mlir::success();
}

mlir::LogicalResult ArtifactSizeYieldOp::verify() {
  auto parent = getOperation()->getParentOfType<ArtifactPackOp>();
  if (!parent || getOperation()->getParentRegion() != &parent.getSizeBody())
    return emitOpError("artifact_size_yield terminates only an artifact size region");
  return mlir::success();
}

mlir::LogicalResult ArtifactReturnOp::verify() {
  auto parent = getOperation()->getParentOfType<ArtifactPackOp>();
  if (!parent || getOperation()->getParentRegion() != &parent.getPackBody())
    return emitOpError("artifact_return terminates only an artifact pack region");
  return mlir::success();
}

mlir::LogicalResult StorageLoadOp::verify() {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(getResult().getType());
  if (!integer || integer.getWidth() != 8 || !integer.isSignless())
    return emitOpError("raw artifact storage load produces one signless storage byte");
  if (!getOperation()->getParentOfType<ArtifactPackOp>() ||
      !canRead(getSource().getType().getAccess()) ||
      !exactLeaf(getLeaf(), "transfer", "artifact-storage",
                 "scalar.storage.load.u8", "none", "exact"))
    return emitOpError("raw storage load is private to a typed artifact builder");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult StorageStoreOp::verify() {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(getValue().getType());
  if (!integer || integer.getWidth() != 8 || !integer.isSignless())
    return emitOpError("raw artifact storage store consumes one signless storage byte");
  if (!getOperation()->getParentOfType<ArtifactPackOp>() ||
      !canWrite(getDestination().getType().getAccess()) ||
      !exactLeaf(getLeaf(), "transfer", "artifact-storage",
                 "scalar.storage.store.u8", "none", "exact"))
    return emitOpError("raw storage store is private to a typed artifact builder");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult KernelOp::verify() {
  mlir::Block &body = getBody().front();
  if (getArgNames().size() != body.getNumArguments() ||
      getArgAccess().size() != body.getNumArguments() ||
      getArgAliasSets().size() != body.getNumArguments() ||
      getParameterNames().size() != getParameterValues().size())
    return emitOpError("physical kernel ABI and parameter bindings disagree");
  if (failed(verifyStringArray(*this, getArgNames(), "arg_names")) ||
      failed(verifyStringArray(*this, getArgAccess(), "arg_access")) ||
      failed(verifyStringArray(*this, getShapeSymbols(), "shape_symbols")) ||
      failed(verifyStringArray(*this, getParameterNames(), "parameter_names")))
    return mlir::failure();
  for (mlir::BlockArgument argument : body.getArguments())
    if (!mlir::isa<MemDescType>(argument.getType()))
      return emitOpError("physical kernel pointer arguments are typed memdescs");
  if (!mlir::isa<ReturnOp>(body.getTerminator()))
    return emitOpError("physical kernel body must end in weft_riscv.return");
  if (getVectorRegisterPeak() < 0 || getFragmentRegisterPeak() < 0 ||
      getLocalStorageBytes() < 0)
    return emitOpError("physical resource summary cannot be negative");
  return mlir::success();
}

mlir::LogicalResult ReturnOp::verify() {
  if (!mlir::isa_and_nonnull<KernelOp>(getOperation()->getParentOp()) ||
      !getValues().empty())
    return emitOpError("physical kernel returns only through explicit stores");
  return mlir::success();
}

mlir::LogicalResult RootDomainOp::verify() {
  return getResult().getType().getRelation() == "root"
             ? mlir::success()
             : emitOpError("root_domain must produce the root domain");
}

mlir::LogicalResult SymbolOp::verify() {
  return getName().empty() ? emitOpError("shape symbol name must not be empty")
                           : mlir::success();
}

mlir::LogicalResult DomainOp::verify() {
  DomainType parent = getParent().getType();
  DomainType child = getResult().getType();
  if (child.getRelation() == "root" ||
      child.getParentDomainId() != parent.getDomainId())
    return emitOpError("physical child domain must preserve parent identity");
  return mlir::success();
}

mlir::LogicalResult LoopOp::verify() {
  if (getStateBirthCount() < 0 || getStagedBirthCount() < 0 ||
      getHandoffCount() != static_cast<int64_t>(getResults().size()))
    return emitOpError("physical loop birth/handoff identity is incomplete");
  if (getCarried().size() != getResults().size())
    return emitOpError("physical loop carried operands and results must match");
  if (getDomain().getType().getParentDomainId() !=
      getParentPoint().getType().getDomain().getDomainId())
    return emitOpError(
        "physical loop must consume the explicit point of its parent domain");
  for (auto [operand, result] : llvm::zip(getCarried(), getResults()))
    if (operand.getType() != result.getType())
      return emitOpError("physical loop carry types must be stable");
  mlir::Block &body = getBody().front();
  if (body.getNumArguments() != 1 + getCarried().size() ||
      body.getArgument(0).getType() !=
          PointType::get(getContext(), getDomain().getType()))
    return emitOpError("physical loop body signature is malformed");
  for (auto [argument, carried] :
       llvm::zip(body.getArguments().drop_front(), getCarried()))
    if (argument.getType() != carried.getType())
      return emitOpError("physical loop body carry type mismatch");
  auto yield = mlir::dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getValues().getTypes() != getResults().getTypes())
    return emitOpError("physical loop yield must match loop results");
  int64_t stateBirths = 0;
  int64_t stagedBirths = 0;
  const int64_t owner = getDomain().getType().getDomainId();
  for (mlir::Operation &operation : body.without_terminator()) {
    if (auto state = mlir::dyn_cast<NewOp>(operation);
        state && state.getOwnerDomainId() == owner)
      ++stateBirths;
    if (auto staged = mlir::dyn_cast<MaterializeOp>(operation);
        staged && staged.getOwnerDomainId() == owner)
      ++stagedBirths;
  }
  if (stateBirths != getStateBirthCount() ||
      stagedBirths != getStagedBirthCount())
    return emitOpError(
        "physical loop birth identities disagree with its typed body");
  return mlir::success();
}

mlir::LogicalResult YieldOp::verify() {
  return mlir::isa_and_nonnull<LoopOp>(getOperation()->getParentOp())
             ? mlir::success()
             : emitOpError("weft_riscv.yield terminates a physical loop");
}

mlir::LogicalResult RootPointOp::verify() {
  if (getDomain().getType().getRelation() != "root" ||
      getResult().getType().getDomain() != getDomain().getType())
    return emitOpError("root_point must preserve the root domain identity");
  return mlir::success();
}

mlir::LogicalResult PhysicalPointOp::verify() {
  auto parent = getParent().getType().getDomain();
  auto result = getResult().getType().getDomain();
  if (result.getDomainId() <= 0 ||
      result.getParentDomainId() != parent.getDomainId())
    return emitOpError("physical point requires a non-root domain identity");
  return mlir::success();
}

mlir::LogicalResult ConstantOp::verify() {
  auto typed = mlir::dyn_cast<mlir::TypedAttr>(getValue());
  if (!typed || typed.getType() != getResult().getType() ||
      !isScalar(getResult().getType()))
    return emitOpError("physical constant attribute and scalar result must match");
  return mlir::success();
}

mlir::LogicalResult IotaOp::verify() {
  mlir::Type element = elementOf(getResult().getType());
  auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
  if (getEnd() <= getStart() || getResult().getType().getShape().size() != 1 ||
      getResult().getType().getShape()[0] != getEnd() - getStart() ||
      (!element.isIndex() && (!integer || integer.isSigned())))
    return emitOpError(
        "physical iota must match one unsigned-integer or index domain");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult NewOp::verify() {
  if (!isPhysicalValue(getResult().getType()) ||
      getInitialized() != static_cast<bool>(getInitial()) ||
      !getInitialized() || !getInitial() ||
      (getPlacement() != "scalar" && getPlacement() != "register" &&
       getPlacement() != "local"))
    return emitOpError("physical state birth has an invalid placement or initial value");
  if (getOwnerDomainId() < 0 || getBirthId() < 0 ||
      getLifetimeEndDomainId() < getOwnerDomainId())
    return emitOpError("physical state birth requires explicit owner and lifetime");
  if (auto result = mlir::dyn_cast<ValueType>(getResult().getType())) {
    if (auto initial = mlir::dyn_cast<ValueType>(getInitial().getType())) {
      if (!sameLogicalDomain(initial, result) ||
          initial.getElementType() != result.getElementType() ||
          initial.getLayout() != result.getLayout())
        return emitOpError(
            "physical shaped state initializer requires an explicit closed layout conversion");
    } else if (!isScalar(getInitial().getType()) ||
               elementOf(getInitial().getType()) != result.getElementType()) {
      return emitOpError(
          "physical state initializer must be layout-identical or scalar-broadcastable");
    }
    return mlir::success();
  }
  if (!isScalar(getResult().getType()) ||
      getInitial().getType() != getResult().getType())
    return emitOpError(
        "scalar physical state initializer and result types must match exactly");
  return mlir::success();
}

mlir::LogicalResult MaterializeOp::verify() {
  if (!isPhysicalValue(getInput().getType()) ||
      !isPhysicalValue(getResult().getType()) ||
      !sameLogicalDomain(getInput().getType(), getResult().getType()) ||
      elementOf(getInput().getType()) != elementOf(getResult().getType()) ||
      (getPlacement() != "shared" && getPlacement() != "register" &&
       getPlacement() != "local" && getPlacement() != "reload"))
    return emitOpError("physical materialize preserves value identity and has one placement");
  if (getOwnerDomainId() < 0 || getBirthId() < 0 ||
      getLifetimeEndDomainId() < getOwnerDomainId() || getSchema().empty())
    return emitOpError("physical staged birth requires explicit owner, lifetime, and schema");
  return mlir::success();
}

mlir::LogicalResult StagedViewOp::verify() {
  if (getSource().getType() != getResult().getType() || getSchema().empty() ||
      getOwnerDomainId() < 0 || getBirthId() < 0 ||
      getLifetimeEndDomainId() < getOwnerDomainId())
    return emitOpError(
        "reload staged view must preserve one typed descriptor and carry its owner/lifetime identity");
  return mlir::success();
}

mlir::LogicalResult MemoryViewOp::verify() {
  const int64_t rank = getBase().getType().getShape().size();
  if (getBase().getType() != getResult().getType() ||
      static_cast<int64_t>(getExtents().size()) != rank ||
      static_cast<int64_t>(getStrides().size()) != rank ||
      static_cast<int64_t>(getOrigins().size()) != rank)
    return emitOpError(
        "memory view requires one explicit extent, stride, and origin per descriptor axis");
  return mlir::success();
}

mlir::LogicalResult SliceOp::verify() {
  if (failed(verifyStringArray(*this, getSelectors(), "selectors")) ||
      getBase().getType().getEncoding() != getResult().getType().getEncoding())
    return emitOpError("physical slice preserves pinned Encoding identity");
  auto base = getBase().getType();
  auto result = getResult().getType();
  if (result.getAddressClass() != "slice" ||
      result.getAccess() != base.getAccess() ||
      result.getAliasSet() != base.getAliasSet() ||
      result.getLayoutIdentity() != base.getLayoutIdentity() ||
      result.getAlignment() != base.getAlignment() ||
      result.getStorageBits() != base.getStorageBits() ||
      result.getElements() != base.getElements() ||
      result.getInterleaveRows() != base.getInterleaveRows())
    return emitOpError(
        "physical slice must preserve its base storage, alias, and access facts");
  if (getSelectors().size() > base.getShape().size())
    return emitOpError("physical slice has more selectors than base axes");
  if (failed(verifyProjection(*this, base, getIndices(), getSelectors(), result,
                              false)))
    return mlir::failure();
  int64_t indexOperands = 0;
  llvm::SmallVector<int64_t> expectedStrides;
  llvm::SmallVector<int64_t> expectedOrigins;
  for (size_t dimension = 0; dimension < base.getShape().size(); ++dimension) {
    llvm::StringRef kind = "all";
    if (dimension < getSelectors().size())
      kind = mlir::cast<mlir::StringAttr>(getSelectors()[dimension]).getValue();
    if (kind != "all")
      ++indexOperands;
    if (kind == "index" || kind == "group_index")
      continue;
    if (kind != "all" && kind != "domain")
      return emitOpError("physical memory slice selector is not addressable");
    expectedStrides.push_back(base.getStrides()[dimension]);
    expectedOrigins.push_back(base.getOrigins()[dimension]);
  }
  if (indexOperands != static_cast<int64_t>(getIndices().size()) ||
      expectedStrides.size() != result.getShape().size() ||
      result.getStrides().asArrayRef() != llvm::ArrayRef(expectedStrides) ||
      result.getOrigins().asArrayRef() != llvm::ArrayRef(expectedOrigins))
    return emitOpError(
        "physical slice indices and projected descriptor strides disagree");
  return mlir::success();
}

mlir::LogicalResult LoadOp::verify() {
  if (!canRead(getRegion().getType().getAccess()) ||
      !isPhysicalValue(getResult().getType()) ||
      !sameLogicalDomain(getRegion().getType(), getResult().getType()))
    return emitOpError("physical load produces a scalar or shaped physical value");
  auto encoding =
      mlir::cast<weft::kernel::EncodingType>(getRegion().getType().getEncoding());
  mlir::Type resultElement = elementOf(getResult().getType());
  if (encoding.getKind() == "dense") {
    if (elementBitWidth(resultElement) != getRegion().getType().getStorageBits())
      return emitOpError(
          "dense physical load result width must match its memory Encoding");
  } else if (resultElement != encoding) {
    return emitOpError(
        "encoded physical load must preserve the pinned Encoding identity");
  }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult StoreOp::verify() {
  if (!canWrite(getRegion().getType().getAccess()))
    return emitOpError("physical store descriptor is not writable");
  if (!isPhysicalValue(getValue().getType()) ||
      !sameLogicalDomain(getValue().getType(), getRegion().getType()))
    return emitOpError("physical store value and descriptor domains must match");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult FieldOp::verify() {
  mlir::Type ownerType = getOwner().getType();
  mlir::Type encodingType;
  if (auto memory = mlir::dyn_cast<MemDescType>(ownerType))
    encodingType = memory.getEncoding();
  else if (auto value = mlir::dyn_cast<ValueType>(ownerType))
    encodingType = value.getElementType();
  auto encoding = mlir::dyn_cast<weft::kernel::EncodingType>(encodingType);
  if (getName().empty() || !encoding ||
      (!isPhysicalValue(getResult().getType()) &&
       !mlir::isa<MemDescType>(getResult().getType())))
    return emitOpError("physical field access requires a named physical result");
  llvm::StringRef family = baseEncodingFamily(*this, encoding);
  auto declaration = findEncodingDeclaration(*this, family);
  if (!declaration)
    return emitOpError("physical field owner has no matching Encoding declaration");
  size_t fieldIndex = declaration.getFieldNames().size();
  for (auto [index, name] : llvm::enumerate(declaration.getFieldNames()))
    if (mlir::cast<mlir::StringAttr>(name).getValue() == getName()) {
      fieldIndex = index;
      break;
    }
  if (fieldIndex == declaration.getFieldNames().size())
    return emitOpError("physical field name is absent from its Encoding");
  mlir::Type fieldType =
      mlir::cast<mlir::TypeAttr>(declaration.getFieldTypes()[fieldIndex]).getValue();
  auto fieldShape =
      mlir::cast<mlir::DenseI64ArrayAttr>(declaration.getFieldShapes()[fieldIndex])
          .asArrayRef();
  if (shapeOf(ownerType).empty() ||
      shapeOf(ownerType).back() != declaration.getElements())
    return emitOpError(
        "physical encoded field owner must end in one complete record axis");
  llvm::SmallVector<int64_t> expectedShape(shapeOf(ownerType).drop_back());
  expectedShape.append(fieldShape.begin(), fieldShape.end());
  auto resultShape = shapeOf(getResult().getType());
  auto ownerAxes = axesOf(ownerType);
  auto resultAxes = axesOf(getResult().getType());
  if (!llvm::equal(expectedShape, resultShape) ||
      resultAxes.size() < ownerAxes.size() - 1 ||
      !llvm::equal(resultAxes.take_front(ownerAxes.size() - 1),
                   ownerAxes.drop_back()) ||
      (!fieldShape.empty() &&
       (resultAxes.empty() || resultAxes.back() != ownerAxes.back())))
    return emitOpError(
        "physical field result must replace the record axis with its declared field shape");
  if (auto result = mlir::dyn_cast<ValueType>(getResult().getType())) {
    if (result.getElementType() != fieldType)
      return emitOpError(
          "physical field result element must match its Encoding declaration");
  } else if (isScalar(getResult().getType())) {
    if (!fieldShape.empty() || getResult().getType() != fieldType)
      return emitOpError(
          "scalar physical field result must match one scalar Encoding field");
  } else {
    auto memory = mlir::cast<MemDescType>(getResult().getType());
    auto dense = mlir::cast<weft::kernel::EncodingType>(memory.getEncoding());
    if (dense.getKind() != "dense" ||
        memory.getStorageBits() != elementBitWidth(fieldType))
      return emitOpError(
          "physical field descriptor must use the declared dense element width");
  }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult ExtractOp::verify() {
  if (failed(verifyStringArray(*this, getSelectors(), "selectors")) ||
      !isPhysicalValue(getResult().getType()) ||
      elementOf(getInput().getType()) != elementOf(getResult().getType()))
    return emitOpError("physical extract requires typed selectors and result");
  auto pattern = getOperation()->getAttrOfType<mlir::DenseI64ArrayAttr>(
      "index_pattern");
  if (pattern) {
    auto input = getInput().getType();
    auto result = mlir::dyn_cast<ValueType>(getResult().getType());
    size_t regularDimension = getSelectors().size();
    unsigned regularCount = 0;
    for (auto [dimension, selectorAttribute] :
         llvm::enumerate(getSelectors())) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "regular") {
        ++regularCount;
        regularDimension = dimension;
      } else if (selector != "all") {
        return emitOpError(
            "regular physical extract currently permits only all/regular selectors");
      }
    }
    if (!result || pattern.size() != 3 || regularCount != 1 ||
        !getIndices().empty() || getAccess().getForm() != "indexed" ||
        input.getShape().size() != result.getShape().size() ||
        input.getAxisIds() != result.getAxisIds() ||
        regularDimension >= input.getShape().size() || pattern[0] < 0 ||
        pattern[1] <= 0 || pattern[2] <= 0 ||
        result.getShape()[regularDimension] <= 0)
      return emitOpError(
          "regular physical extract requires one closed base/stride/repeat mapping");
    for (size_t dimension = 0; dimension < input.getShape().size(); ++dimension)
      if (dimension != regularDimension &&
          input.getShape()[dimension] != result.getShape()[dimension])
        return emitOpError(
            "regular physical extract must preserve every non-indexed axis");
    const int64_t steps =
        (result.getShape()[regularDimension] - 1) / pattern[2];
    if (steps > (std::numeric_limits<int64_t>::max() - pattern[0]) /
                    pattern[1])
      return emitOpError("regular physical extract index mapping overflows");
    const int64_t last = pattern[0] + steps * pattern[1];
    if (last < pattern[0] || last >= input.getShape()[regularDimension])
      return emitOpError(
          "regular physical extract index mapping exceeds its source axis");
  } else if (failed(verifyProjection(*this, getInput().getType(), getIndices(),
                                     getSelectors(), getResult().getType(),
                                     true))) {
    return mlir::failure();
  }
  if (getAccess().getForm() == "register") {
    auto result = mlir::dyn_cast<ValueType>(getResult().getType());
    auto input = getInput().getType();
    unsigned gatherCount = 0;
    mlir::Value gatherIndex;
    size_t cursor = 0;
    for (mlir::Attribute selectorAttribute : getSelectors()) {
      llvm::StringRef selector =
          mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
      if (selector == "all")
        continue;
      if (cursor >= getIndices().size())
        return emitOpError("register gather selector has no index operand");
      if (selector == "gather") {
        ++gatherCount;
        gatherIndex = getIndices()[cursor];
      }
      ++cursor;
    }
    auto indices = gatherIndex
                       ? mlir::dyn_cast<ValueType>(gatherIndex.getType())
                       : ValueType();
    auto time = checkedPositiveProduct(input.getLayout().getTimeFactors().asArrayRef());
    auto replicas =
        checkedPositiveProduct(input.getLayout().getReplicaFactors().asArrayRef());
    if (!result || !indices || gatherCount != 1 ||
        input.getLayout().getCarrier() != "rvv" ||
        result.getLayout().getCarrier() != "rvv" ||
        indices.getLayout().getCarrier() != "rvv" || !time || *time != 1 ||
        !replicas || *replicas != 1 ||
        input.getLayout().getSew() != result.getLayout().getSew() ||
        input.getLayout().getLmulEighths() !=
            result.getLayout().getLmulEighths() ||
        indices.getLayout().getSew() != result.getLayout().getSew() ||
        indices.getLayout().getLmulEighths() !=
            result.getLayout().getLmulEighths() ||
        !exactLeaf(getLeaf(), "rvv", "extract", "rvv.extract.vrgather",
                   "none", "exact"))
      return emitOpError(
                 "register extract requires one complete RVV source, index, result, and exact gather leaf; input=")
             << input << ", index="
             << (gatherIndex ? gatherIndex.getType() : mlir::Type())
             << ", result=" << getResult().getType();
  }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult UpdateOp::verify() {
  if (getInput().getType() != getResult().getType() ||
      failed(verifyStringArray(*this, getSelectors(), "selectors")))
    return emitOpError()
           << "physical update preserves its shaped state representation; input="
           << getInput().getType() << ", result=" << getResult().getType();
  if (elementOf(getValue().getType()) != getInput().getType().getElementType())
    return emitOpError("physical update value element type must match its state");
  if (failed(verifyProjection(*this, getInput().getType(), getIndices(),
                              getSelectors(), getValue().getType(), true)))
    return mlir::failure();
  return verifyLeafOperation(*this);
}

mlir::LogicalResult UnaryOp::verify() {
  if (getInput().getType() != getResult().getType() ||
      !isNumericPhysical(getInput().getType()) ||
      (getKind() != "neg" && getKind() != "abs" && getKind() != "exp") ||
      (getKind() == "exp" &&
       !mlir::isa<mlir::FloatType>(elementOf(getInput().getType()))))
    return emitOpError("physical unary operation preserves its value type");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult BinaryOp::verify() {
  constexpr llvm::StringLiteral allowed[] = {
      "add", "sub", "mul", "div", "mod", "and", "or",
      "xor", "shl", "shr", "max", "min"};
  if (!isPhysicalValue(getLhs().getType()) || !isPhysicalValue(getRhs().getType()) ||
      !isPhysicalValue(getResult().getType()) ||
      !isNumericPhysical(getLhs().getType()) ||
      !isNumericPhysical(getRhs().getType()) ||
      !isNumericPhysical(getResult().getType()) ||
      !llvm::is_contained(allowed, getKind()) ||
      elementOf(getLhs().getType()) != elementOf(getRhs().getType()) ||
      elementOf(getLhs().getType()) != elementOf(getResult().getType()))
    return emitOpError("physical binary operation requires physical values");
  if (failed(verifyBroadcastDomain(*this, getLhs().getType(),
                                   getRhs().getType(), getResult().getType())))
    return mlir::failure();
  return verifyLeafOperation(*this);
}

mlir::LogicalResult CompareOp::verify() {
  constexpr llvm::StringLiteral predicates[] = {"eq", "ne", "lt",
                                                 "le", "gt", "ge"};
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
      elementOf(getResult().getType()));
  if (!isPhysicalValue(getLhs().getType()) || !isPhysicalValue(getRhs().getType()) ||
      !isPhysicalValue(getResult().getType()) ||
      !isNumericPhysical(getLhs().getType()) ||
      !isNumericPhysical(getRhs().getType()) ||
      elementOf(getLhs().getType()) != elementOf(getRhs().getType()) ||
      !resultElement || resultElement.getWidth() != 1 ||
      !llvm::is_contained(predicates, getPredicate()))
    return emitOpError("physical compare requires physical values");
  if (failed(verifyBroadcastDomain(*this, getLhs().getType(),
                                   getRhs().getType(), getResult().getType())))
    return mlir::failure();
  return verifyLeafOperation(*this);
}

mlir::LogicalResult CastOp::verify() {
  return verifyConversion(*this, getInput().getType(), getResult().getType());
}

mlir::LogicalResult NarrowOp::verify() {
  return verifyConversion(*this, getInput().getType(), getResult().getType());
}

mlir::LogicalResult WidenOp::verify() {
  return verifyConversion(*this, getInput().getType(), getResult().getType());
}

mlir::LogicalResult MacGroupsOp::verify() {
  if (getGroup() == 0 ||
      getGroup() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      getOverflow() != "wrap")
    return emitOpError("physical mac_groups requires positive group and wrap semantics");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult ReduceOp::verify() {
  auto input = mlir::dyn_cast<ValueType>(getInput().getType());
  if (!input || !isPhysicalValue(getResult().getType()) ||
      (getKind() != "add" && getKind() != "max" && getKind() != "min") ||
      getAxis() < 0 || static_cast<size_t>(getAxis()) >= input.getShape().size() ||
      elementOf(getInput().getType()) != elementOf(getResult().getType()))
    return emitOpError("physical reduce requires physical values");
  llvm::SmallVector<int64_t> expectedShape(input.getShape().asArrayRef());
  llvm::SmallVector<int64_t> expectedAxes(input.getAxisIds().asArrayRef());
  expectedShape.erase(expectedShape.begin() + getAxis());
  expectedAxes.erase(expectedAxes.begin() + getAxis());
  if (shapeOf(getResult().getType()) != llvm::ArrayRef<int64_t>(expectedShape) ||
      axesOf(getResult().getType()) != llvm::ArrayRef<int64_t>(expectedAxes))
    return emitOpError(
        "physical reduce must remove exactly its selected axis and preserve every free axis");
  if (auto result = mlir::dyn_cast<ValueType>(getResult().getType()))
    for (auto [resultPosition, axis] :
         llvm::enumerate(result.getAxisIds().asArrayRef())) {
      auto found = llvm::find(input.getAxisIds().asArrayRef(), axis);
      if (found == input.getAxisIds().asArrayRef().end() ||
          !sameAxisMapping(input,
                           static_cast<size_t>(
                               found - input.getAxisIds().asArrayRef().begin()),
                           result, resultPosition))
        return emitOpError(
            "physical reduce must preserve every free-axis representation");
    }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult Fold2Op::verify() {
  auto input = mlir::dyn_cast<ValueType>(getInput().getType());
  auto result = mlir::dyn_cast<ValueType>(getResult().getType());
  if (!input || !result)
    return emitOpError("physical fold2 consumes and produces shaped values");
  auto inputElement = mlir::dyn_cast<mlir::IntegerType>(input.getElementType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto inputShape = input.getShape().asArrayRef();
  auto resultShape = result.getShape().asArrayRef();
  if (inputShape.empty() || inputShape.size() != resultShape.size() ||
      inputShape.back() <= 0 || inputShape.back() % 2 ||
      resultShape.back() != inputShape.back() / 2 ||
      !inputElement || !resultElement || !resultElement.isSigned() ||
      resultElement.getWidth() != 32 ||
      input.getAxisIds() != result.getAxisIds() ||
      !llvm::equal(inputShape.drop_back(), resultShape.drop_back()))
    return emitOpError(
        "physical fold2 must halve the final axis and produce signed i32");
  if ((input.getLayout().getCarrier() != "unassigned" &&
       input.getLayout().getCarrier() != "local") ||
      (result.getLayout().getCarrier() != "unassigned" &&
       result.getLayout().getCarrier() != "rvv") ||
      (getAccess().getForm() != "unassigned" &&
       (getAccess().getMapping() != "natural" ||
        getAccess().getBitOffset() % 8)))
    return emitOpError()
           << "physical fold2 requires a closed natural-field to RVV realization"
           << " (input=" << input.getLayout().getCarrier()
           << ", result=" << result.getLayout().getCarrier()
           << ", access=" << getAccess().getForm() << "/"
           << getAccess().getMapping() << ", bit_offset="
           << getAccess().getBitOffset() << ")";
  return verifyLeafOperation(*this);
}

static mlir::LogicalResult verifyContractOp(mlir::Operation *operation) {
  auto lhs = mlir::dyn_cast<ValueType>(operation->getOperand(0).getType());
  auto rhs = mlir::dyn_cast<ValueType>(operation->getOperand(1).getType());
  auto result = mlir::dyn_cast<ValueType>(operation->getResult(0).getType());
  auto over = operation->getAttrOfType<mlir::DenseI64ArrayAttr>("over");
  if (!lhs || !rhs || !over || over.empty())
    return operation->emitOpError(
        "physical contract requires shaped operands and reduction axes");
  llvm::DenseSet<int64_t> reduction;
  for (int64_t axis : over.asArrayRef())
    if (axis <= 0 || !reduction.insert(axis).second ||
        !llvm::is_contained(lhs.getAxisIds().asArrayRef(), axis) ||
        !llvm::is_contained(rhs.getAxisIds().asArrayRef(), axis))
      return operation->emitOpError(
          "physical contract reduction axes must be positive, unique, and shared");
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  auto appendFree = [&](ValueType value) -> mlir::LogicalResult {
    for (auto [axis, extent] : llvm::zip(value.getAxisIds().asArrayRef(),
                                        value.getShape().asArrayRef())) {
      if (reduction.contains(axis))
        continue;
      auto found = llvm::find(expectedAxes, axis);
      if (found == expectedAxes.end()) {
        expectedAxes.push_back(axis);
        expectedShape.push_back(extent);
      } else if (expectedShape[static_cast<size_t>(found - expectedAxes.begin())] !=
                 extent) {
        return operation->emitOpError(
            "physical contract operands disagree on a shared free-axis extent");
      }
    }
    return mlir::success();
  };
  if (failed(appendFree(lhs)) || failed(appendFree(rhs)))
    return mlir::failure();
  for (int64_t axis : over.asArrayRef()) {
    auto lhsIt = llvm::find(lhs.getAxisIds().asArrayRef(), axis);
    auto rhsIt = llvm::find(rhs.getAxisIds().asArrayRef(), axis);
    int64_t lhsExtent = lhs.getShape()[static_cast<size_t>(
        lhsIt - lhs.getAxisIds().asArrayRef().begin())];
    int64_t rhsExtent = rhs.getShape()[static_cast<size_t>(
        rhsIt - rhs.getAxisIds().asArrayRef().begin())];
    if (lhsExtent > 0 && rhsExtent > 0 && lhsExtent != rhsExtent)
      return operation->emitOpError(
          "physical contract reduction-axis extents disagree");
  }
  if (expectedAxes.empty()) {
    if (result || !isNumericPhysical(operation->getResult(0).getType()))
      return operation->emitOpError(
          "physical scalar contract must eliminate every logical axis");
  } else if (!result ||
             !llvm::equal(expectedAxes, result.getAxisIds().asArrayRef()) ||
             !llvm::equal(expectedShape, result.getShape().asArrayRef())) {
    return operation->emitOpError(
        "physical contract result must preserve ordered free axes and extents");
  }
  if (auto accumulator = operation->getAttrOfType<mlir::TypeAttr>("acc_type");
      accumulator && accumulator.getValue() !=
                         elementOf(operation->getResult(0).getType()))
    return operation->emitOpError(
        "physical contract accumulator type must equal its result element type");
  auto laneOperand = operation->getAttrOfType<mlir::StringAttr>("lane_operand");
  auto laneForm = operation->getAttrOfType<mlir::StringAttr>("lane_memory_form");
  if (!laneOperand || !laneForm ||
      (laneOperand.getValue() != "unassigned" &&
       laneOperand.getValue() != "lhs" && laneOperand.getValue() != "rhs") ||
      (laneForm.getValue() != "unassigned" && laneForm.getValue() != "unit" &&
       laneForm.getValue() != "strided" && laneForm.getValue() != "indexed" &&
       laneForm.getValue() != "segment" && laneForm.getValue() != "register"))
    return operation->emitOpError(
        "physical contract lane access must be unassigned or explicitly closed");
  return verifyLeafOperation(operation);
}

mlir::LogicalResult DotOp::verify() { return verifyContractOp(*this); }
mlir::LogicalResult ContractOp::verify() { return verifyContractOp(*this); }
mlir::LogicalResult OuterContractOp::verify() { return verifyContractOp(*this); }

mlir::LogicalResult LookupOp::verify() {
  auto indexElement = mlir::dyn_cast<mlir::IntegerType>(
      elementOf(getIndices().getType()));
  if (getBounds() != "in_bounds" ||
      (!elementOf(getIndices().getType()).isIndex() &&
       (!indexElement || indexElement.isSigned())) ||
      !sameLogicalDomain(getIndices().getType(), getResult().getType()) ||
      !isNumericPhysical(getResult().getType()) ||
      (!mlir::isa<ValueType, MemDescType>(getTable().getType())))
    return emitOpError("physical lookup currently requires in-bounds semantics");
  if (auto table = mlir::dyn_cast<ValueType>(getTable().getType());
      table && table.getElementType() != elementOf(getResult().getType()))
    return emitOpError("physical lookup table and result element types disagree");
  if (auto table = mlir::dyn_cast<MemDescType>(getTable().getType())) {
    auto encoding =
        mlir::cast<weft::kernel::EncodingType>(table.getEncoding());
    if (encoding.getKind() != "dense" ||
        table.getStorageBits() != elementBitWidth(elementOf(getResult().getType())))
      return emitOpError(
          "physical lookup descriptor must match the result element width");
  }
  if (auto indices = mlir::dyn_cast<ValueType>(getIndices().getType()))
    if (auto result = mlir::dyn_cast<ValueType>(getResult().getType()))
      for (size_t position = 0; position < indices.getAxisIds().size(); ++position)
        if (!sameAxisMapping(indices, position, result, position))
          return emitOpError(
              "physical lookup indices/result require an explicit layout conversion");
  llvm::StringRef carrier = "scalar";
  if (auto indices = mlir::dyn_cast<ValueType>(getIndices().getType()))
    carrier = indices.getLayout().getCarrier();
  auto tableValue = mlir::dyn_cast<ValueType>(getTable().getType());
  auto resultValue = mlir::dyn_cast<ValueType>(getResult().getType());
  // ConvertWeftToRISCV creates a typed lookup before layout propagation and
  // memory planning have assigned its access form and terminal leaf.  Keep
  // that transient state verifiable only while both decisions are explicitly
  // unassigned; PlanRISCVMemory must close them together before final
  // verification.
  if (getAccess().getForm() == "unassigned" &&
      getLeaf().getEngine() == "unselected")
    return verifyLeafOperation(*this);
  if (tableValue) {
    auto tableLayout = tableValue.getLayout();
    auto indexValue = mlir::dyn_cast<ValueType>(getIndices().getType());
    if (!indexValue || !resultValue || carrier != "rvv" ||
        tableLayout.getCarrier() != "rvv" ||
        resultValue.getLayout().getCarrier() != "rvv" ||
        tableLayout.getSew() != resultValue.getLayout().getSew() ||
        tableLayout.getLmulEighths() != resultValue.getLayout().getLmulEighths() ||
        indexValue.getLayout().getSew() != resultValue.getLayout().getSew() ||
        indexValue.getLayout().getLmulEighths() !=
            resultValue.getLayout().getLmulEighths() ||
        getAccess().getForm() != "register" ||
        !exactLeaf(getLeaf(), "rvv", "lookup", "rvv.vrgather", "none",
                   "exact"))
      return emitOpError(
          "register lookup requires one layout-compatible RVV table, index, result, and exact gather leaf");
    auto tableParts = checkedPositiveProduct(tableLayout.getTimeFactors().asArrayRef());
    auto tableReplicas =
        checkedPositiveProduct(tableLayout.getReplicaFactors().asArrayRef());
    if (!tableParts || !tableReplicas || *tableParts != 1 ||
        *tableReplicas != 1)
      return emitOpError(
          "register lookup table must fit one complete RVV register value");
    return verifyLeafOperation(*this);
  }
  if ((carrier == "scalar" &&
       (getAccess().getForm() != "unit" ||
        !exactLeaf(getLeaf(), "scalar", "lookup", "scalar.lookup", "none",
                   "exact"))) ||
      (carrier == "rvv" &&
       (getAccess().getForm() != "indexed" ||
        !exactLeaf(getLeaf(), "rvv", "lookup", "rvv.vluxei", "none",
                   "exact"))) ||
      (carrier != "scalar" && carrier != "rvv"))
    return emitOpError(
        "physical lookup access, index carrier, and exact leaf must agree");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult ConvertLayoutOp::verify() {
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  if (input.getElementType() != result.getElementType() ||
      input.getShape() != result.getShape() || input.getAxisIds() != result.getAxisIds() ||
      input.getLayout() == result.getLayout())
    return emitOpError("convert_layout changes only a non-identical physical representation");
  llvm::StringRef source = input.getLayout().getCarrier();
  llvm::StringRef target = result.getLayout().getCarrier();
  llvm::StringRef kind = getConversion().getKind();
  llvm::StringRef effect = getConversion().getEffect();
  if ((source == "scalar" && target == "rvv" && kind != "splat" &&
       kind != "register_to_lane" && kind != "time_to_lane") ||
      (source == "rvv" && target == "scalar" && kind != "extract") ||
      (source == "local" && target != "local" && kind != "local_load") ||
      (source != "local" && target == "local" && kind != "local_store") ||
      (source == target && kind != "reshape" && kind != "tuple" &&
       kind != "register_to_lane" && kind != "lane_to_register") ||
      (kind == "local_load" && effect != "read") ||
      (kind == "local_store" && effect != "write") ||
      (kind != "local_load" && kind != "local_store" && effect != "pure") ||
      kind == "rvv_to_fragment" || kind == "fragment_to_rvv")
    return emitOpError() << "convert_layout kind '" << kind
                         << "' does not match " << source << " -> " << target;
  if (source == "scalar" && target == "rvv" &&
      kind == "register_to_lane") {
    auto sourceLayout = input.getLayout();
    auto targetLayout = result.getLayout();
    auto sourceTime = sourceLayout.getTimeFactors().asArrayRef();
    auto sourceLane = sourceLayout.getLaneFactors().asArrayRef();
    auto sourceReplica = sourceLayout.getReplicaFactors().asArrayRef();
    auto sourceFragment = sourceLayout.getFragmentFactors().asArrayRef();
    auto sourceLocal = sourceLayout.getLocalFactors().asArrayRef();
    auto targetTime = targetLayout.getTimeFactors().asArrayRef();
    auto targetLane = targetLayout.getLaneFactors().asArrayRef();
    auto targetReplica = targetLayout.getReplicaFactors().asArrayRef();
    auto targetFragment = targetLayout.getFragmentFactors().asArrayRef();
    auto targetLocal = targetLayout.getLocalFactors().asArrayRef();
    std::optional<size_t> laneDimension;
    for (size_t dimension = 0; dimension < targetLane.size(); ++dimension) {
      if (targetLane[dimension] > 1) {
        if (laneDimension)
          return emitOpError(
              "scalar register-to-lane conversion requires exactly one lane axis");
        laneDimension = dimension;
      }
      if (sourceTime[dimension] != 1 || sourceLane[dimension] != 1 ||
          sourceFragment[dimension] != 1 || sourceLocal[dimension] != 1 ||
          targetFragment[dimension] != 1 || targetLocal[dimension] != 1)
        return emitOpError(
            "scalar register-to-lane conversion only moves register replicas into RVV lanes");
    }
    if (!laneDimension)
      return emitOpError(
          "scalar register-to-lane conversion requires one non-unit target lane axis");
    for (size_t dimension = 0; dimension < targetLane.size(); ++dimension) {
      if (dimension == *laneDimension) {
        auto represented = checkedPositiveProduct(
            {targetTime[dimension], targetLane[dimension]});
        if (!represented || sourceReplica[dimension] != *represented ||
            targetReplica[dimension] != 1)
          return emitOpError(
              "scalar register-to-lane conversion must preserve the complete moved axis");
        continue;
      }
      if (targetTime[dimension] != 1 || targetLane[dimension] != 1 ||
          sourceReplica[dimension] != targetReplica[dimension])
        return emitOpError(
            "scalar register-to-lane conversion cannot remap another logical axis");
    }
  }
  if (source == "scalar" && target == "rvv" && kind == "time_to_lane") {
    auto sourceLayout = input.getLayout();
    auto targetLayout = result.getLayout();
    auto sourceTime = sourceLayout.getTimeFactors().asArrayRef();
    auto sourceLane = sourceLayout.getLaneFactors().asArrayRef();
    auto sourceReplica = sourceLayout.getReplicaFactors().asArrayRef();
    auto sourceFragment = sourceLayout.getFragmentFactors().asArrayRef();
    auto sourceLocal = sourceLayout.getLocalFactors().asArrayRef();
    auto targetTime = targetLayout.getTimeFactors().asArrayRef();
    auto targetLane = targetLayout.getLaneFactors().asArrayRef();
    auto targetReplica = targetLayout.getReplicaFactors().asArrayRef();
    auto targetFragment = targetLayout.getFragmentFactors().asArrayRef();
    auto targetLocal = targetLayout.getLocalFactors().asArrayRef();
    std::optional<size_t> laneDimension;
    for (size_t dimension = 0; dimension < targetLane.size(); ++dimension) {
      if (targetLane[dimension] > 1) {
        if (laneDimension)
          return emitOpError(
              "scalar time-to-lane conversion requires exactly one lane axis");
        laneDimension = dimension;
      }
      if (sourceLane[dimension] != 1 ||
          sourceFragment[dimension] != 1 || sourceLocal[dimension] != 1 ||
          targetFragment[dimension] != 1 || targetLocal[dimension] != 1)
        return emitOpError(
            "scalar time-to-lane conversion only moves issue-time parts into RVV lanes");
    }
    if (!laneDimension)
      return emitOpError(
          "scalar time-to-lane conversion requires one non-unit target lane axis");
    for (size_t dimension = 0; dimension < targetLane.size(); ++dimension) {
      if (dimension == *laneDimension) {
        auto represented = checkedPositiveProduct(
            {targetTime[dimension], targetLane[dimension]});
        if (!represented || sourceTime[dimension] != *represented ||
            sourceReplica[dimension] != targetReplica[dimension] ||
            sourceTime[dimension] <= targetTime[dimension])
          return emitOpError(
              "scalar time-to-lane conversion must preserve the complete moved axis");
        continue;
      }
      if (targetLane[dimension] != 1 ||
          sourceTime[dimension] != targetTime[dimension] ||
          sourceReplica[dimension] != targetReplica[dimension])
        return emitOpError(
            "scalar time-to-lane conversion cannot remap another logical axis");
    }
  }
  if (auto access = (*this)->getAttrOfType<AccessAttr>("source_access")) {
    if (effect != "pure" && effect != "read")
      return emitOpError(
          "source_access is only legal on a read-only representation conversion");
    if (access.getForm() != "unit" && access.getForm() != "strided" &&
        access.getForm() != "indexed" && access.getForm() != "segment")
      return emitOpError(
          "source_access must name one exact target memory form");
  }
  if (getLeaf().getEngine() != "unselected") {
    llvm::StringRef expected =
        kind == "splat"          ? "rvv.splat"
        : kind == "extract"      ? "rvv.extract"
        : kind == "local_load"   ? "rvv.local-load"
        : kind == "local_store"  ? "rvv.local-store"
        : kind == "tuple"        ? "rvv.tuple-convert"
        : kind == "register_to_lane" ? "rvv.register-to-lane"
        : kind == "time_to_lane" ? "rvv.time-to-lane"
        : kind == "lane_to_register" ? "rvv.lane-to-register"
                                      : "rvv.layout-reshape";
    if (getLeaf().getEngine() != "rvv" ||
        getLeaf().getFamily() != "layout-conversion" ||
        getLeaf().getInstruction() != expected)
      return emitOpError(
          "convert_layout leaf does not implement its typed conversion kind");
  }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult LocalAllocOp::verify() {
  LocalType type = getResult().getType();
  if ((type.getSizeBytes() == 0 || type.getSizeBytes() < -1) ||
      type.getAlignment() <= 0 ||
      type.getPurpose().empty() || type.getSchema().empty())
    return emitOpError("local allocation requires size, alignment, purpose, and schema");
  if (type.getSizeBytes() >= 0) {
    auto constant = getSizeBytes().getDefiningOp<mlir::arith::ConstantIndexOp>();
    if (!constant || constant.value() != type.getSizeBytes())
      return emitOpError(
          "static local allocation size SSA value must equal its LocalType size");
  }
  return mlir::success();
}

mlir::LogicalResult LocalCapacityGuardOp::verify() {
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel || getLimitBytes() <= 0 || getBirthId() < 0 ||
      getLimitBytes() > kernel.getTarget().getMaxPrivateStackBytes() ||
      !exactLeaf(getLeaf(), "scalar", "local-capacity",
                 "scalar.local-capacity-guard", "none", "exact"))
    return emitOpError(
        "dynamic local capacity guard requires one target-bounded scalar trap contract");
  auto next = std::next(getOperation()->getIterator());
  if (next == getOperation()->getBlock()->end())
    return emitOpError("dynamic local capacity guard must immediately precede its allocation");
  auto allocation = mlir::dyn_cast<LocalAllocOp>(&*next);
  if (!allocation || allocation.getSizeBytes() != getSizeBytes() ||
      allocation.getResult().getType().getSizeBytes() != -1 ||
      allocation.getResult().getType().getBirthId() != getBirthId())
    return emitOpError(
        "dynamic local capacity guard must uniquely guard the following allocation");
  return mlir::success();
}

mlir::LogicalResult LocalBindOp::verify() {
  LocalType storage = getStorage().getType();
  ValueType result = getResult().getType();
  if (result.getLayout().getCarrier() != "local" ||
      storage.getElementType() != result.getElementType() ||
      storage.getShape() != result.getShape() ||
      storage.getAxisIds() != result.getAxisIds())
    return emitOpError(
        "local bind must expose exactly the logical domain owned by its storage");
  if (storage.getSizeBytes() >= 0) {
    auto elements = getElements().getDefiningOp<mlir::arith::ConstantIndexOp>();
    int64_t expectedElements = 1;
    for (int64_t extent : result.getShape().asArrayRef()) {
      if (extent <= 0)
        return emitOpError(
            "static local storage cannot bind a symbolic logical extent");
      expectedElements *= extent;
    }
    if (!elements || elements.value() != expectedElements)
      return emitOpError(
          "static local binding element count must equal its logical shape");
  }
  return mlir::success();
}

mlir::LogicalResult LocalLoadOp::verify() {
  if (getSource().getType().getLayout().getCarrier() != "local" ||
      elementOf(getResult().getType()) != getSource().getType().getElementType() ||
      !isScalar(getResult().getType()) ||
      !exactLeaf(getLeaf(), "transfer", "local-load", "local.load.element",
                 "none", "exact"))
    return emitOpError("local load reads one scalar element from a local value");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult LocalStoreOp::verify() {
  if (getDestination().getType().getLayout().getCarrier() != "local" ||
      elementOf(getValue().getType()) !=
          getDestination().getType().getElementType() ||
      !isScalar(getValue().getType()) ||
      !exactLeaf(getLeaf(), "transfer", "local-store",
                 "local.store.element", "none", "exact"))
    return emitOpError("local store writes one scalar element into a local value");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult SpillOp::verify() {
  auto value = mlir::dyn_cast<ValueType>(getInput().getType());
  LocalType slot = getSlot().getType();
  if (!value || slot.getPurpose() != "spill" ||
      value.getElementType() != slot.getElementType() ||
      value.getShape() != slot.getShape() || value.getAxisIds() != slot.getAxisIds() ||
      !exactLeaf(getLeaf(), "transfer", "spill", "rvv.spill", "none",
                 "exact") ||
      getLeaf().getTemporaryGroups() != 0 ||
      getLeaf().getFragmentGroups() != 0 || getLeaf().getLocalBytes() != 0)
    return emitOpError("spill slot must exactly preserve its physical value domain");
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  const int64_t requiredBytes =
      value.getLayout().getRegisterGroups() * kernel.getTarget().getVlenBits() / 8;
  if (slot.getSizeBytes() != requiredBytes)
    return emitOpError("spill slot size must equal the selected RVV register bytes");
  return verifyLeafOperation(*this);
}
mlir::LogicalResult ReloadOp::verify() {
  auto value = mlir::dyn_cast<ValueType>(getResult().getType());
  LocalType slot = getSlot().getType();
  if (!value || slot.getPurpose() != "spill" ||
      value.getElementType() != slot.getElementType() ||
      value.getShape() != slot.getShape() || value.getAxisIds() != slot.getAxisIds() ||
      !exactLeaf(getLeaf(), "transfer", "reload", "rvv.reload", "none",
                 "exact") ||
      getLeaf().getTemporaryGroups() != 0 ||
      getLeaf().getFragmentGroups() != 0 || getLeaf().getLocalBytes() != 0)
    return emitOpError("reload must restore the exact spilled logical domain");
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  const int64_t requiredBytes =
      value.getLayout().getRegisterGroups() * kernel.getTarget().getVlenBits() / 8;
  if (slot.getSizeBytes() != requiredBytes)
    return emitOpError("reload slot size must equal the selected RVV register bytes");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult IMEFragmentMMAOp::verify() {
  FragmentType lhs = getLhs().getType();
  FragmentType rhs = getRhs().getType();
  FragmentType result = getResult().getType();
  auto capability = fragmentCapability(*this, getLeaf().getInstruction());
  if (!capability ||
      capability.getInstruction() != "spacemit-ime1-i8-mma" ||
      !exactLeaf(getLeaf(), "ime", "fragment-mma",
                 capability.getInstruction(), "none", "exact") ||
      lhs.getFamily() != rhs.getFamily() || result.getFamily() != lhs.getFamily() ||
      lhs.getFamily() != capability.getInstruction() ||
      lhs.getRole() != "lhs" || rhs.getRole() != "rhs" ||
      result.getRole() != "accumulator" ||
      lhs.getPacking() != capability.getLhsPacking() ||
      rhs.getPacking() != capability.getRhsPacking() ||
      result.getPacking() != capability.getAccumulatorPacking() ||
      elementBitWidth(lhs.getElementType()) != capability.getLhsBits() ||
      elementBitWidth(rhs.getElementType()) != capability.getRhsBits() ||
      !integerSignednessMatches(lhs.getElementType(),
                                capability.getLhsSignedness()) ||
      !integerSignednessMatches(rhs.getElementType(),
                                capability.getRhsSignedness()) ||
      elementBitWidth(result.getElementType()) != capability.getAccumulatorBits() ||
      getLeaf().getFragmentGroups() != 0 ||
      getGroups() != capability.getMmaGroups() ||
      getChunks() != capability.getMmaChunks() ||
      getAsmClobbers() != capability.getClobbers() ||
      getVolatileAsm() != capability.getVolatileAsm() ||
      getMemoryClobber() != capability.getMemoryClobber() ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({capability.getMFactor(),
                                   capability.getNFactor(),
                                   capability.getKFactor()}) ||
      lhs.getShape().size() != 2 || rhs.getShape().size() != 2 ||
      result.getShape().size() != 2 ||
      getLeaf().getLocalBytes() !=
          result.getShape()[0] * result.getShape()[1] *
              std::max<int64_t>(
                  1, (elementBitWidth(result.getElementType()) + 7) / 8))
    return emitOpError("IME fragment MMA requires an IME leaf contract");
  if (lhs.getShape().size() != 2 || rhs.getShape().size() != 2 ||
      result.getShape().size() != 2 ||
      lhs.getShape()[0] != capability.getMFactor() ||
      lhs.getShape()[1] != capability.getKFactor() ||
      rhs.getShape()[0] != capability.getKFactor() ||
      rhs.getShape()[1] != capability.getNFactor() ||
      result.getShape()[0] != capability.getMFactor() ||
      result.getShape()[1] != capability.getNFactor() ||
      lhs.getAxisIds()[1] != rhs.getAxisIds()[0] ||
      lhs.getAxisIds()[0] != result.getAxisIds()[0] ||
      rhs.getAxisIds()[1] != result.getAxisIds()[1])
    return emitOpError(
        "IME fragment MMA axes and shapes do not match the selected M4xN4xK8 capability");
  if (lhs.getResourceGroups() != capability.getLhsResourceGroups() ||
      rhs.getResourceGroups() != capability.getRhsResourceGroups() ||
      result.getResourceGroups() != capability.getAccumulatorResourceGroups())
    return emitOpError(
        "IME fragment resources do not match the target capability");
  return mlir::success();
}

mlir::LogicalResult IMEPackOp::verify() {
  FragmentType result = getResult().getType();
  auto input = mlir::dyn_cast<ValueType>(getInput().getType());
  auto capability = fragmentCapability(*this, result.getFamily());
  if (!input || !capability ||
      capability.getInstruction() != "spacemit-ime1-i8-mma" ||
      (getRole() != "lhs" && getRole() != "rhs") ||
      !exactLeaf(getLeaf(), "ime", "fragment-pack",
                 getRole() == "lhs" ? "ime.pack.lhs" : "ime.pack.rhs",
                 "none", "exact") ||
      getPacking() != (getRole() == "lhs" ? capability.getLhsPacking()
                                           : capability.getRhsPacking()) ||
      result.getRole() != getRole() || result.getPacking() != getPacking() ||
      result.getResourceGroups() !=
          (getRole() == "lhs" ? capability.getLhsResourceGroups()
                              : capability.getRhsResourceGroups()) ||
      getPacking().getStorageBits() != 8 ||
      result.getLayout().getCarrier() != "ime" ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({capability.getMFactor(),
                                   capability.getNFactor(),
                                   capability.getKFactor()}))
    return emitOpError("IME pack requires one typed fragment operand contract");
  const bool lhs = getRole() == "lhs";
  auto packing = getPacking();
  const int64_t expectedBits = lhs ? capability.getLhsBits()
                                   : capability.getRhsBits();
  const llvm::StringRef expectedSignedness =
      lhs ? capability.getLhsSignedness() : capability.getRhsSignedness();
  if (elementBitWidth(result.getElementType()) != expectedBits ||
      elementBitWidth(input.getElementType()) != expectedBits ||
      !integerSignednessMatches(input.getElementType(), expectedSignedness) ||
      input.getAxisIds() != result.getAxisIds())
    return emitOpError("IME pack input and fragment logical domains must agree");
  auto layout = input.getLayout();
  int64_t laneAxes = 0;
  if (layout.getCarrier() != "rvv" || layout.getSew() != 8 ||
      input.getShape().size() != layout.getTimeFactors().size() ||
      input.getShape().size() != layout.getLaneFactors().size() ||
      input.getShape().size() != layout.getReplicaFactors().size() ||
      input.getShape().size() != layout.getFragmentFactors().size() ||
      input.getShape().size() != layout.getLocalFactors().size())
    return emitOpError("IME pack input requires one complete eight-bit RVV mapping");
  for (auto [extent, time, lane, replica, fragment, local] :
       llvm::zip(result.getShape().asArrayRef(),
                 layout.getTimeFactors().asArrayRef(),
                 layout.getLaneFactors().asArrayRef(),
                 layout.getReplicaFactors().asArrayRef(),
                 layout.getFragmentFactors().asArrayRef(),
                 layout.getLocalFactors().asArrayRef())) {
    if (extent <= 0 || time <= 0 || lane <= 0 || replica <= 0 ||
        fragment <= 0 || local <= 0 ||
        time * lane * replica * fragment * local != extent)
      return emitOpError("IME pack input time/lane/register factors are incomplete");
    laneAxes += lane > 1;
  }
  if (laneAxes != 1)
    return emitOpError("IME pack input requires exactly one SIMD lane axis");
  if ((lhs && (result.getShape().size() != 2 ||
               result.getShape()[0] != capability.getMFactor() ||
               result.getShape()[1] != capability.getKFactor())) ||
      (!lhs && (result.getShape().size() != 2 ||
                result.getShape()[0] != capability.getKFactor() ||
                result.getShape()[1] != capability.getNFactor())))
    return emitOpError("IME pack logical shape disagrees with the target fragment");
  auto axisOrder = packing.getAxisOrder().asArrayRef();
  const int64_t rows = result.getShape()[axisOrder[0]];
  const int64_t columns = result.getShape()[axisOrder[1]];
  if (rows % packing.getRowsPerTile() ||
      columns % packing.getColumnsPerTile() ||
      packing.getStorageBits() < expectedBits)
    return emitOpError(
        "IME operand shape and element width do not fit the selected tiled packing");
  const int64_t logicalBytes =
      rows * columns *
      std::max<int64_t>(1, (elementBitWidth(input.getElementType()) + 7) / 8);
  const int64_t packedBytes =
      rows * columns * packing.getStorageBits() / 8;
  if (getLeaf().getLocalBytes() != logicalBytes + packedBytes)
    return emitOpError(
        "IME pack leaf must account for every primitive-private byte");
  if (getAccess().getMapping() == "opaque" || getAccess().getForm() == "opaque")
    return emitOpError("IME operand has no selected physical access form");
  return mlir::success();
}

mlir::LogicalResult IMEUnpackOp::verify() {
  auto capability = fragmentCapability(*this, getInput().getType().getFamily());
  if (!capability ||
      capability.getInstruction() != "spacemit-ime1-i8-mma" ||
      !exactLeaf(getLeaf(), "ime", "fragment-unpack", "ime.unpack.rvv",
                 "none", "exact") ||
      getConversion().getKind() != "fragment_to_rvv" ||
      getConversion().getEffect() != "handoff" ||
      getInput().getType().getRole() != "accumulator" ||
      getInput().getType().getPacking() !=
          capability.getAccumulatorPacking() ||
      getResult().getType().getLayout().getCarrier() != "rvv" ||
      getInput().getType().getAxisIds() != getResult().getType().getAxisIds() ||
      getInput().getType().getElementType() !=
          getResult().getType().getElementType() ||
      elementBitWidth(getResult().getType().getElementType()) !=
          capability.getAccumulatorBits() ||
      getLeaf().getLocalBytes() != 0)
    return emitOpError("IME unpack must preserve a fragment logical domain into RVV");
  auto fragment = getInput().getType();
  auto layout = getResult().getType().getLayout();
  for (auto [extent, time, lane, replica, fragmentFactor, local] :
       llvm::zip(fragment.getShape().asArrayRef(),
                 layout.getTimeFactors().asArrayRef(),
                 layout.getLaneFactors().asArrayRef(),
                 layout.getReplicaFactors().asArrayRef(),
                 layout.getFragmentFactors().asArrayRef(),
                 layout.getLocalFactors().asArrayRef()))
    if (time * lane * replica * fragmentFactor * local != extent)
      return emitOpError(
          "IME unpack result layout does not cover the complete fragment domain");
  return mlir::success();
}

mlir::LogicalResult RegisterMaterializeOp::verify() {
  if (!isPhysicalValue(getInput().getType()) ||
      getInput().getType() != getResult().getType() ||
      getOwnerDomainId() < 0 || getBirthId() < 0 ||
      getLifetimeEndDomainId() < getOwnerDomainId() ||
      (getRealization() != "share" && getRealization() != "reload" &&
       getRealization() != "rematerialize"))
    return emitOpError("register materialize requires stable value identity and lifetime");
  return mlir::success();
}

mlir::LogicalResult RVVBitplaneMergeOp::verify() {
  ValueType low = getLow().getType();
  ValueType plane = getPlane().getType();
  ValueType result = getResult().getType();
  auto lowElement = mlir::dyn_cast<mlir::IntegerType>(low.getElementType());
  auto planeElement =
      mlir::dyn_cast<mlir::IntegerType>(plane.getElementType());
  auto field = getPlane().getDefiningOp<FieldOp>();
  const bool logicalPlane =
      planeElement && !planeElement.isSigned() && planeElement.getWidth() == 1 &&
      plane.getShape() == low.getShape() && plane.getAxisIds() == low.getAxisIds() &&
      field && field.getAccess().getForm() == "indexed" &&
      field.getAccess().getMapping() == "grouped_layered" &&
      field.getAccess().getGroupSize() == 8 &&
      field.getAccess().getLayerSize() == 1;
  const bool bytePlane =
      planeElement && !planeElement.isSigned() && planeElement.getWidth() == 8 &&
      plane.getLayout().getCarrier() == "local";
  if (!lowElement || lowElement.isSigned() || lowElement.getWidth() != 8 ||
      (!logicalPlane && !bytePlane) || low != result ||
      low.getLayout().getCarrier() != "rvv" || getInsertBit() <= 0 ||
      getInsertBit() >= 8)
    return emitOpError(
        "RVV bitplane merge requires one unsigned-eight-bit RVV value and one typed logical-bit or local-byte plane");
  if (low.getShape().size() != plane.getShape().size() ||
      low.getAxisIds() != plane.getAxisIds())
    return emitOpError(
        "RVV bitplane merge requires matching logical axes for values and storage plane");
  int64_t bitplaneAxes = 0;
  int64_t bitplaneAxis = 0;
  if (bytePlane) {
    for (auto [axis, lowExtent, planeExtent] :
         llvm::zip(low.getAxisIds().asArrayRef(), low.getShape().asArrayRef(),
                   plane.getShape().asArrayRef())) {
      if (lowExtent == planeExtent)
        continue;
      if (planeExtent <= 0 || lowExtent != planeExtent * 8)
        return emitOpError(
            "RVV bitplane merge plane must pack exactly eight logical values per byte");
      ++bitplaneAxes;
      bitplaneAxis = axis;
    }
    if (bitplaneAxes != 1)
      return emitOpError("RVV bitplane merge requires one byte-packed logical axis");
  }
  int64_t laneAxis = 0;
  int64_t laneAxes = 0;
  for (auto [axis, lane] :
       llvm::zip(low.getLayout().getAxisIds().asArrayRef(),
                 low.getLayout().getLaneFactors().asArrayRef()))
    if (lane > 1) {
      laneAxis = axis;
      ++laneAxes;
    }
  if (logicalPlane) {
    if (laneAxes != 1)
      return emitOpError(
          "RVV logical bitplane merge must map one encoded logical axis to lanes");
    bitplaneAxis = laneAxis;
  }
  auto packed = llvm::find(low.getAxisIds().asArrayRef(), bitplaneAxis);
  const size_t packedPosition = static_cast<size_t>(
      packed - low.getAxisIds().asArrayRef().begin());
  const bool laneMask =
      getLeaf().getInstruction() == "rvv.bitplane-merge.mask" &&
      laneAxes == 1 && laneAxis == bitplaneAxis &&
      low.getLayout().getVl() > 0 && low.getLayout().getVl() % 8 == 0;
  const bool stridedByte =
      getLeaf().getInstruction() == "rvv.bitplane-merge.strided" &&
      laneAxes == 1 && laneAxis != bitplaneAxis &&
      packed != low.getAxisIds().asArrayRef().end() &&
      low.getLayout().getLaneFactors()[packedPosition] == 1 &&
      low.getLayout().getReplicaFactors()[packedPosition] == 1 &&
      low.getLayout().getFragmentFactors()[packedPosition] == 1 &&
      low.getLayout().getLocalFactors()[packedPosition] == 1 &&
      low.getLayout().getTimeFactors()[packedPosition] ==
          low.getShape()[packedPosition];
  bool otherTime = false;
  for (auto [axis, time] :
       llvm::zip(low.getAxisIds().asArrayRef(),
                 low.getLayout().getTimeFactors().asArrayRef()))
    otherTime |= axis != bitplaneAxis && time != 1;
  if (!laneMask && !(stridedByte && !otherTime))
    return emitOpError(
        "RVV bitplane merge layout does not match its selected lane-mask or strided-byte realization");
  if (!field ||
      ((!logicalPlane && field.getAccess().getMapping() != "natural") ||
       (logicalPlane && field.getAccess().getMapping() != "grouped_layered")) ||
      field.getAccess().getBitOffset() % 8)
    return emitOpError(
        "RVV bitplane merge requires one byte-aligned encoded bitplane field");
  if ((!exactLeaf(getLeaf(), "rvv", "bitplane-merge",
                  "rvv.bitplane-merge.mask", "none", "agnostic") &&
       !exactLeaf(getLeaf(), "rvv", "bitplane-merge",
                  "rvv.bitplane-merge.strided", "none", "agnostic")) ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({static_cast<int64_t>(getInsertBit())}) ||
      getLeaf().getLocalBytes() != 0)
    return emitOpError("RVV bitplane merge has no exact selected leaf");
  return mlir::success();
}

mlir::LogicalResult RVVBitmaskDecodeOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto fieldElement = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto origin = getOrigin().getDefiningOp<PhysicalPointOp>();
  auto point = getPoint().getDefiningOp<PhysicalPointOp>();
  const int64_t axis = getPoint().getType().getDomain().getAxisId();
  auto position = llvm::find(result.getAxisIds().asArrayRef(), axis);
  const bool hasAxis = position != result.getAxisIds().asArrayRef().end();
  const size_t axisPosition = hasAxis
                                  ? static_cast<size_t>(
                                        position -
                                        result.getAxisIds().asArrayRef().begin())
                                  : 0;
  const int64_t extent = hasAxis ? result.getShape()[axisPosition] : 0;
  auto partition = point
                       ? point.getPartition().getDefiningOp<
                             mlir::arith::ConstantIndexOp>()
                       : mlir::arith::ConstantIndexOp();
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  bool compatibleShape = field.getAxisIds() == result.getAxisIds() &&
                         field.getShape().size() == result.getShape().size();
  if (compatibleShape)
    for (size_t index = 0; index < field.getShape().size(); ++index) {
      if (index == axisPosition)
        compatibleShape &= field.getShape()[index] >= extent;
      else
        compatibleShape &= field.getShape()[index] == result.getShape()[index];
    }
  bool onlyPointAxisIsStreamed = hasAxis;
  if (onlyPointAxisIsStreamed)
    for (size_t index = 0; index < result.getShape().size(); ++index)
      if (index != axisPosition)
        onlyPointAxisIsStreamed &=
            result.getLayout().getTimeFactors()[index] == 1 &&
            result.getLayout().getLaneFactors()[index] == 1;

  bool ownerAnchoredAtOrigin = false;
  if (sourceField && origin) {
    if (auto owner = sourceField.getOwner().getDefiningOp<ExtractOp>()) {
      auto ownerInput = mlir::dyn_cast<ValueType>(owner.getInput().getType());
      size_t cursor = 0;
      for (auto [position, selectorAttribute] :
           llvm::enumerate(owner.getSelectors())) {
        llvm::StringRef selector =
            mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
        if (selector == "all")
          continue;
        if (cursor >= owner.getIndices().size())
          break;
        mlir::Value index = owner.getIndices()[cursor++];
        if (ownerInput && position < ownerInput.getAxisIds().size() &&
            ownerInput.getAxisIds()[position] == axis && selector == "domain" &&
            index == getOrigin())
          ownerAnchoredAtOrigin = true;
      }
    }
  }
  if (!fieldElement || fieldElement.isSigned() || fieldElement.getWidth() != 1 ||
      !resultElement || resultElement.isSigned() ||
      resultElement.getWidth() != 1 || !sourceField || !origin || !point ||
      !partition || !hasAxis || extent <= 0 || partition.value() != extent ||
      point.getParent() != getOrigin() ||
      point.getResult().getType().getDomain().getAxisId() !=
          origin.getResult().getType().getDomain().getAxisId() ||
      !ownerAnchoredAtOrigin || !compatibleShape || !onlyPointAxisIsStreamed ||
      getAccess() != sourceField.getAccess() ||
      getAccess().getForm() != "indexed" ||
      getAccess().getMapping() != "grouped_layered" ||
      getAccess().getGroupSize() <= 0 || getAccess().getLayerSize() <= 0 ||
      getAccess().getGroupSize() != getAccess().getLayerSize() * 8 ||
      getAccess().getOrder() != "lo_first" ||
      getAccess().getBitOffset() % 8 ||
      extent % getAccess().getGroupSize() ||
      result.getLayout().getCarrier() != "rvv" ||
      result.getLayout().getLaneFactors()[axisPosition] %
          getAccess().getGroupSize() ||
      result.getLayout().getTimeFactors()[axisPosition] *
              result.getLayout().getLaneFactors()[axisPosition] !=
          extent ||
      (validity != "full" && validity != "tail") ||
      !exactLeaf(getLeaf(), "rvv", "bitmask-decode", "rvv.bitmask-decode",
                 "none", tail))
    return emitOpError(
        "RVV bitmask decode requires a byte-aligned contiguous logical-u1 field, "
        "an explicitly anchored sub-Level point, and a complete time/lane result");
  return mlir::success();
}

mlir::LogicalResult RVVGroupedMacReduceOp::verify() {
  auto lhs = mlir::dyn_cast<ValueType>(getLhs().getType());
  auto resultParts = physicalPartCount(getResult().getType());
  if (!lhs || lhs.getLayout().getCarrier() != "rvv")
    return emitOpError("grouped MAC reduction lhs must use an RVV representation");
  if (!resultParts || *resultParts != 1 ||
      getResult().getType().getLayout().getCarrier() != "rvv")
    return emitOpError("grouped MAC reduction result must be one RVV part");
  if (getGroup() <= 0 || getUnroll() <= 0 || getReductionAxis() <= 0)
    return emitOpError("grouped MAC reduction has invalid structural parameters");
  if (getPartialLayout().getCarrier() != "rvv" ||
      getPartialLayout().getSew() != 16)
    return emitOpError("grouped MAC reduction requires a 16-bit RVV partial layout");
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError(
        "grouped MAC reduction must be nested in one target kernel");
  auto target = kernel.getTarget();
  if (!hasGroupedMacOperandGeometry(
          getOperation(), getLhs(), getRhs(), getReductionAxis(),
          getLhsAccess(), getRhsAccess(), getGroup(), getPartialLayout()) ||
      !supportsLayout(target, getResult().getType().getLayout()))
    return emitOpError(
        "grouped MAC reduction has no complete typed field, cohort-storage, "
        "RVV-layout, or shared reduction-point realization");
  if (getLhsAccess().getForm() == "unassigned" ||
      getRhsAccess().getForm() == "unassigned")
    return emitOpError("grouped MAC reduction operands require selected access forms");
  if (!exactLeaf(getLeaf(), "rvv", "grouped-mac-reduce",
                 "rvv.grouped-mac-reduce.u8-s8", "none", "agnostic") &&
      !exactLeaf(getLeaf(), "rvv", "grouped-mac-reduce",
                 "rvv.grouped-mac-reduce.u8-s8", "none", "exact"))
    return emitOpError("grouped MAC reduction leaf is not the selected RVV operation");
  if (getLeaf().getParameters().asArrayRef() !=
      llvm::ArrayRef<int64_t>({static_cast<int64_t>(getGroup()),
                               static_cast<int64_t>(getUnroll()),
                               static_cast<int64_t>(getReductionAxis())}))
    return emitOpError("grouped MAC reduction leaf parameters disagree with its schedule");
  return mlir::success();
}

mlir::LogicalResult RVVGroupedMacLoadOp::verify() {
  auto window = getResult().getType();
  auto lhs = mlir::dyn_cast<ValueType>(getLhs().getType());
  if (!lhs || lhs.getLayout().getCarrier() != "rvv")
    return emitOpError("grouped MAC lhs must use an RVV representation");
  if (window.getFamily() != "grouped-mac" || getGroup() <= 0 ||
      getUnroll() <= 0 || getReductionAxis() <= 0)
    return emitOpError("grouped MAC window has invalid structural parameters");
  if (window.getReductionAxis() != getReductionAxis() ||
      window.getSlots() != getUnroll() ||
      window.getTermsPerSlot() != getGroup())
    return emitOpError("grouped MAC window shape disagrees with its operation");
  if (window.getLhsType() != getLhs().getType() ||
      window.getRhsType() != getRhs().getType() ||
      window.getPartialLayout() != getPartialLayout())
    return emitOpError("grouped MAC window type disagrees with its operands");
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError("grouped MAC load must be nested in one target kernel");
  auto target = kernel.getTarget();
  auto termsPerWindow = checkedPositiveProduct(
      {static_cast<int64_t>(getGroup()), static_cast<int64_t>(getUnroll())});
  if ((getWindowForm() != "compact" && getWindowForm() != "fragmented") ||
      !termsPerWindow ||
      (getWindowForm() == "compact" &&
       getLhsAccess().getLayerSize() % *termsPerWindow != 0))
    return emitOpError(
        "grouped MAC load has no closed compact/fragmented window form");
  if (!hasGroupedMacOperandGeometry(
          getOperation(), getLhs(), getRhs(), getReductionAxis(),
          getLhsAccess(), getRhsAccess(), getGroup(), getPartialLayout()) ||
      !supportsLayout(target, window.getResultLayout()))
    return emitOpError(
        "grouped MAC load has no complete typed field, cohort-storage, "
        "RVV-layout, or shared reduction-point realization");
  if (getLhsAccess().getForm() == "unassigned" ||
      getRhsAccess().getForm() == "unassigned")
    return emitOpError("grouped MAC operands require selected access forms");
  if ((!exactLeaf(getLeaf(), "rvv", "grouped-mac-load",
                  "rvv.grouped-mac-load.u8-s8", "none", "agnostic") &&
       !exactLeaf(getLeaf(), "rvv", "grouped-mac-load",
                  "rvv.grouped-mac-load.u8-s8", "none", "exact")) ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>(
              {static_cast<int64_t>(getGroup()),
               static_cast<int64_t>(getUnroll()),
               static_cast<int64_t>(getReductionAxis())}))
    return emitOpError("grouped MAC load leaf disagrees with its window");
  return mlir::success();
}

mlir::LogicalResult RVVGroupedMacStepOp::verify() {
  auto window = getWindow().getType();
  if (window.getFamily() != "grouped-mac" ||
      getAccumulator().getType() != getResult().getType() ||
      window.getResultType() != getResult().getType() ||
      window.getResultLayout() != getResult().getType().getLayout() ||
      window.getResultLayout().getCarrier() != "rvv" ||
      (!exactLeaf(getLeaf(), "rvv", "grouped-mac-step",
                  "rvv.vwmaccsu.vx.grouped", "none", "agnostic") &&
       !exactLeaf(getLeaf(), "rvv", "grouped-mac-step",
                  "rvv.vwmaccsu.vx.grouped", "none", "exact")) ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({window.getTermsPerSlot(), window.getSlots(),
                                   window.getReductionAxis()}))
    return emitOpError("grouped MAC compute step contract is incomplete");
  return mlir::success();
}

mlir::LogicalResult RVVEncodedDotLoadOp::verify() {
  auto window = getResult().getType();
  auto lhs = mlir::dyn_cast<ValueType>(getLhs().getType());
  auto lhsParts = lhs ? physicalPartCount(lhs) : std::nullopt;
  if (!lhs || lhs.getLayout().getCarrier() != "rvv" || !lhsParts ||
      *lhsParts % window.getResultParts() ||
      window.getFamily() != "encoded-dot" || getUnroll() <= 0 ||
      getRhsPairsPerGroup() <= 0 || getReductionAxis() <= 0 ||
      window.getReductionAxis() != getReductionAxis() ||
      window.getSlots() != getUnroll() ||
      window.getTermsPerSlot() != getRhsPairsPerGroup() ||
      window.getLhsType() != getLhs().getType() ||
      window.getRhsType() != getRhs().getType() ||
      getLhsAccess().getForm() == "unassigned" ||
      getRhsAccess().getForm() == "unassigned" ||
      !exactLeaf(getLeaf(), "rvv", "encoded-dot-load",
                 "rvv.encoded-dot-load.i16-pairs", "none", "agnostic") ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>(
              {static_cast<int64_t>(getReductionAxis()),
               static_cast<int64_t>(getUnroll()),
               static_cast<int64_t>(getRhsPairsPerGroup())}))
    return emitOpError("encoded dot load window contract is incomplete");
  return mlir::success();
}

mlir::LogicalResult RVVEncodedDotStepOp::verify() {
  auto window = getWindow().getType();
  if (window.getFamily() != "encoded-dot" ||
      getAccumulator().getType() != getResult().getType() ||
      window.getResultType() != getResult().getType() ||
      window.getResultLayout() != getResult().getType().getLayout() ||
      !exactLeaf(getLeaf(), "rvv", "encoded-dot-step",
                 "rvv.vzext-vmacc.vx", "none", "agnostic") ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({window.getReductionAxis(), window.getSlots(),
                                   window.getTermsPerSlot()}))
    return emitOpError("encoded dot compute step contract is incomplete");
  return mlir::success();
}

mlir::LogicalResult RVVWidenDotOp::verify() {
  ValueType lhs = getLhs().getType();
  ValueType rhs = getRhs().getType();
  auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
  auto shapedResult = mlir::dyn_cast<ValueType>(getResult().getType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
      shapedResult ? shapedResult.getElementType() : getResult().getType());
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError("RVV widening dot must be nested in one target kernel");
  auto target = kernel.getTarget();
  auto partialLMULValue = doubledPositive(lhs.getLayout().getLmulEighths());
  const int64_t partialLMUL = partialLMULValue.value_or(-1);
  llvm::SmallVector<int64_t> freeAxes;
  for (int64_t axis : lhs.getAxisIds().asArrayRef())
    if (!llvm::is_contained(getOver(), axis))
      freeAxes.push_back(axis);
  for (int64_t axis : rhs.getAxisIds().asArrayRef())
    if (!llvm::is_contained(getOver(), axis) &&
        !llvm::is_contained(freeAxes, axis))
      freeAxes.push_back(axis);
  const bool legalResult =
      (!shapedResult && isScalar(getResult().getType()) && freeAxes.empty()) ||
      (shapedResult && shapedResult.getLayout().getCarrier() == "scalar" &&
       shapedResult.getAxisIds().asArrayRef() ==
           llvm::ArrayRef<int64_t>(freeAxes));
  auto factorFor = [](LayoutAttr layout, mlir::DenseI64ArrayAttr factors,
                      int64_t axis) {
    auto found = llvm::find(layout.getAxisIds().asArrayRef(), axis);
    if (found == layout.getAxisIds().asArrayRef().end())
      return int64_t{0};
    return factors[static_cast<size_t>(
        found - layout.getAxisIds().asArrayRef().begin())];
  };
  const int64_t reductionAxis = getOver().empty() ? 0 : getOver()[0];
  const bool compatibleReductionMapping =
      reductionAxis && lhs.getLayout().getCarrier() == "rvv" &&
      rhs.getLayout().getCarrier() == "rvv" &&
      lhs.getLayout().getSew() == rhs.getLayout().getSew() &&
      lhs.getLayout().getLmulEighths() ==
          rhs.getLayout().getLmulEighths() &&
      lhs.getLayout().getVl() == rhs.getLayout().getVl() &&
      factorFor(lhs.getLayout(), lhs.getLayout().getLaneFactors(),
                reductionAxis) ==
          factorFor(rhs.getLayout(), rhs.getLayout().getLaneFactors(),
                    reductionAxis) &&
      factorFor(lhs.getLayout(), lhs.getLayout().getTimeFactors(),
                reductionAxis) ==
          factorFor(rhs.getLayout(), rhs.getLayout().getTimeFactors(),
                    reductionAxis);
  if ((getStreamReduction() != "fused" &&
       getStreamReduction() != "per_stream") ||
      !lhsElement || !rhsElement || !resultElement ||
      lhsElement.getWidth() > 16 || rhsElement.getWidth() > 16 ||
      std::max<unsigned>(8, lhsElement.getWidth()) !=
          std::max<unsigned>(8, rhsElement.getWidth()) ||
      (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
      resultElement.getWidth() != 32 || !legalResult ||
      getOver().size() != 1 ||
      !llvm::is_contained(lhs.getAxisIds().asArrayRef(), getOver()[0]) ||
      !llvm::is_contained(rhs.getAxisIds().asArrayRef(), getOver()[0]) ||
      !compatibleReductionMapping ||
      !target.getHasWideningInteger() ||
      !supportsLayout(target, lhs.getLayout()) ||
      !supportsLayout(target, rhs.getLayout()) ||
      !partialLMULValue ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), partialLMUL) ||
      !exactLeaf(getLeaf(), "rvv", "widen-dot",
                 "rvv.vwmul-vwredsum", "none", "exact"))
    return emitOpError()
           << "RVV widening dot requires matching <=16-bit integer vectors with "
              "at least one signed operand, an explicit fused/per-stream "
              "reduction policy, one shared "
              "reduction axis, a signed-i32 scalar result, and a legal doubled "
              "LMUL; lhs="
           << lhs << ", rhs=" << rhs << ", result=" << getResult().getType()
           << ", over=" << getOver() << ", partial_lmul=" << partialLMUL
           << ", leaf=" << getLeaf();
  return mlir::success();
}

mlir::LogicalResult RVVStorageWindowOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  const int64_t axis = getReductionAxis();
  auto fieldAxis = llvm::find(field.getAxisIds().asArrayRef(), axis);
  auto resultAxis = llvm::find(result.getAxisIds().asArrayRef(), axis);
  auto pointDomain = getOrigin().getType().getDomain();
  auto pointAxis = pointDomain.getAxisId();
  auto timeParts = checkedPositiveProduct(
      result.getLayout().getTimeFactors().asArrayRef());
  llvm::StringRef mapping = getAccess().getMapping();
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  if (!sourceField || getAccess() != sourceField.getAccess() ||
      !getLogicalOffset().getType().isIndex() || axis <= 0 || pointAxis != axis ||
      field.getElementType() != result.getElementType() ||
      field.getAxisIds() != result.getAxisIds() ||
      fieldAxis == field.getAxisIds().asArrayRef().end() ||
      resultAxis == result.getAxisIds().asArrayRef().end() ||
      pointDomain.getTail() != "exact" ||
      result.getLayout().getCarrier() != "rvv" || !timeParts || *timeParts != 1 ||
      (mapping != "natural" && mapping != "grouped_layered") ||
      (validity != "full" && validity != "tail") ||
      !exactLeaf(getLeaf(), "rvv", "storage-window", "rvv.storage-window",
                 "none", tail))
    return emitOpError(
        "RVV storage window requires one typed field/point/offset edge and one complete lane result");
  const size_t fieldPosition = static_cast<size_t>(
      fieldAxis - field.getAxisIds().asArrayRef().begin());
  const size_t resultPosition = static_cast<size_t>(
      resultAxis - result.getAxisIds().asArrayRef().begin());
  if (result.getShape()[resultPosition] <= 0 ||
      field.getShape()[fieldPosition] < result.getShape()[resultPosition] ||
      result.getShape()[resultPosition] !=
          result.getLayout().getLaneFactors()[resultPosition] ||
      result.getLayout().getVl() < result.getShape()[resultPosition])
    return emitOpError(
        "RVV storage window result must cover one legal lane-sized subrange");
  for (size_t position = 0; position < result.getShape().size(); ++position) {
    if (position == resultPosition)
      continue;
    if (field.getShape()[position] != result.getShape()[position] ||
        result.getLayout().getLaneFactors()[position] != 1 ||
        result.getLayout().getTimeFactors()[position] != 1 ||
        (result.getShape()[position] > 0
             ? result.getLayout().getReplicaFactors()[position] !=
                   result.getShape()[position]
             : result.getLayout().getReplicaFactors()[position] <= 0) ||
        result.getLayout().getFragmentFactors()[position] != 1 ||
        result.getLayout().getLocalFactors()[position] != 1)
      return emitOpError(
          "RVV storage window must preserve every free axis as register replicas");
  }
  return mlir::success();
}

mlir::LogicalResult RVVLayeredStorageLoadOp::verify() {
  ValueType field = getField().getType();
  LayeredWindowType window = getResult().getType();
  ValueType result = window.getResultType();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  const int64_t group = getAccess().getGroupSize();
  const int64_t layer = getAccess().getLayerSize();
  const int64_t layers = layer > 0 ? group / layer : 0;
  auto resultAxis = llvm::find(result.getAxisIds().asArrayRef(),
                               getReductionAxis());
  const int64_t lanes =
      resultAxis == result.getAxisIds().asArrayRef().end()
          ? 0
          : result.getLayout().getLaneFactors()[static_cast<size_t>(
                resultAxis - result.getAxisIds().asArrayRef().begin())];
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  if (!sourceField || getAccess() != sourceField.getAccess() || !integer ||
      integer.isSigned() || getOrigin().getType().getDomain().getAxisId() !=
                                getReductionAxis() ||
      getOrigin().getType().getDomain().getTail() != "exact" ||
      !getWindowIndex().getType().isIndex() ||
      getAccess().getForm() != "indexed" ||
      getAccess().getMapping() != "grouped_layered" || group <= 0 ||
      layer <= 0 || group % layer || layers <= 1 || lanes <= 1 ||
      layer % lanes || integer.getWidth() * layers != 8 ||
      getAccess().getBitOffset() % 8 ||
      (getAccess().getOrder() != "lo_first" &&
       getAccess().getOrder() != "hi_first") ||
      window.getFieldType() != field || window.getReductionAxis() !=
                                            getReductionAxis() ||
      window.getLayers() != layers ||
      window.getWindowsPerLayer() != layer / lanes ||
      !exactLeaf(getLeaf(), "rvv", "layered-storage-load",
                 "rvv.layered-storage-load", "none", tail))
    return emitOpError(
        "layered storage load requires one grouped/layered field and closed byte-window geometry");
  return mlir::success();
}

mlir::LogicalResult RVVLayeredStorageDecodeOp::verify() {
  LayeredWindowType window = getWindow().getType();
  ValueType result = getResult().getType();
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  if (getLayer() < 0 || getLayer() >= window.getLayers() ||
      result != window.getResultType() ||
      !exactLeaf(getLeaf(), "rvv", "layered-storage-decode",
                 "rvv.layered-storage-decode", "none", tail))
    return emitOpError(
        "layered storage decode requires one selected layer of its typed window");
  return mlir::success();
}

mlir::LogicalResult RVVWidenAccumulateOp::verify() {
  ValueType lhs = getLhs().getType();
  ValueType rhs = getRhs().getType();
  ValueType accumulator = getAccumulator().getType();
  ValueType result = getResult().getType();
  auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
  auto partialElement =
      mlir::dyn_cast<mlir::IntegerType>(accumulator.getElementType());
  auto lhsAxis = llvm::find(lhs.getAxisIds().asArrayRef(), getReductionAxis());
  auto rhsAxis = llvm::find(rhs.getAxisIds().asArrayRef(), getReductionAxis());
  auto partialAxis = llvm::find(accumulator.getAxisIds().asArrayRef(),
                                getReductionAxis());
  auto oneTimePart = [](ValueType value) {
    return checkedPositiveProduct(value.getLayout().getTimeFactors().asArrayRef()) ==
           std::optional<int64_t>(1);
  };
  auto reductionPosition = [&](ValueType value) -> std::optional<size_t> {
    auto found = llvm::find(value.getAxisIds().asArrayRef(), getReductionAxis());
    if (found == value.getAxisIds().asArrayRef().end())
      return std::nullopt;
    return static_cast<size_t>(found - value.getAxisIds().asArrayRef().begin());
  };
  auto reductionLane = [&](ValueType value) -> int64_t {
    auto position = reductionPosition(value);
    return position ? value.getLayout().getLaneFactors()[*position] : 0;
  };
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  auto appendFree = [&](ValueType value) -> bool {
    for (auto [axis, extent] : llvm::zip(value.getAxisIds().asArrayRef(),
                                        value.getShape().asArrayRef())) {
      if (axis == getReductionAxis())
        continue;
      auto found = llvm::find(expectedAxes, axis);
      if (found == expectedAxes.end()) {
        expectedAxes.push_back(axis);
        expectedShape.push_back(extent);
      } else if (expectedShape[static_cast<size_t>(found - expectedAxes.begin())] !=
                 extent) {
        return false;
      }
    }
    return true;
  };
  const bool compatibleFreeAxes = appendFree(lhs) && appendFree(rhs);
  expectedAxes.push_back(getReductionAxis());
  expectedShape.push_back(reductionLane(accumulator));
  auto operandMapsToPartial = [&](ValueType operand) {
    for (auto [position, axis] :
         llvm::enumerate(operand.getAxisIds().asArrayRef())) {
      auto partialPosition = llvm::find(accumulator.getAxisIds().asArrayRef(), axis);
      if (partialPosition == accumulator.getAxisIds().asArrayRef().end())
        return false;
      const size_t target = static_cast<size_t>(
          partialPosition - accumulator.getAxisIds().asArrayRef().begin());
      if (axis == getReductionAxis()) {
        if (operand.getLayout().getLaneFactors()[position] !=
                accumulator.getLayout().getLaneFactors()[target] ||
            operand.getShape()[position] != accumulator.getShape()[target])
          return false;
        continue;
      }
      if (operand.getLayout().getTimeFactors()[position] != 1 ||
          operand.getLayout().getLaneFactors()[position] != 1 ||
          operand.getLayout().getReplicaFactors()[position] !=
              accumulator.getLayout().getReplicaFactors()[target] ||
          operand.getLayout().getFragmentFactors()[position] != 1 ||
          operand.getLayout().getLocalFactors()[position] != 1)
        return false;
    }
    return true;
  };
  auto loop = getOperation()->getParentOfType<mlir::scf::ForOp>();
  bool loopCarried = loop && getOperation()->getBlock() == loop.getBody();
  unsigned carryIndex = 0;
  mlir::Value chain = getAccumulator();
  while (loopCarried) {
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(chain)) {
      if (argument.getOwner() != loop.getBody() || argument.getArgNumber() == 0) {
        loopCarried = false;
        break;
      }
      carryIndex = argument.getArgNumber() - 1;
      break;
    }
    auto previous = chain.getDefiningOp<RVVWidenAccumulateOp>();
    if (!previous || previous.getOperation()->getBlock() != loop.getBody() ||
        previous.getReductionAxis() != getReductionAxis()) {
      loopCarried = false;
      break;
    }
    chain = previous.getAccumulator();
  }
  if (loopCarried) {
    mlir::Value current = getResult();
    while (true) {
      if (!current.hasOneUse()) {
        loopCarried = false;
        break;
      }
      mlir::Operation *user = *current.getUsers().begin();
      if (auto next = mlir::dyn_cast<RVVWidenAccumulateOp>(user)) {
        if (next.getAccumulator() != current ||
            next.getReductionAxis() != getReductionAxis()) {
          loopCarried = false;
          break;
        }
        current = next.getResult();
        continue;
      }
      auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(user);
      loopCarried = yield && carryIndex < yield.getNumOperands() &&
                    yield.getOperand(carryIndex) == current;
      break;
    }
  }
  if (!lhsElement || !rhsElement || !partialElement ||
      lhsElement.isSignless() || rhsElement.isSignless() ||
      !partialElement.isSigned() ||
      std::max<unsigned>(8, lhsElement.getWidth()) !=
          std::max<unsigned>(8, rhsElement.getWidth()) ||
      partialElement.getWidth() !=
          2 * std::max<unsigned>(8, lhsElement.getWidth()) ||
      (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
      accumulator != result || getReductionAxis() <= 0 ||
      lhsAxis == lhs.getAxisIds().asArrayRef().end() ||
      rhsAxis == rhs.getAxisIds().asArrayRef().end() ||
      partialAxis == accumulator.getAxisIds().asArrayRef().end() ||
      !oneTimePart(lhs) || !oneTimePart(rhs) || !oneTimePart(accumulator) ||
      lhs.getLayout().getCarrier() != "rvv" ||
      rhs.getLayout().getCarrier() != "rvv" ||
      accumulator.getLayout().getCarrier() != "rvv" ||
      lhs.getLayout().getVl() != rhs.getLayout().getVl() ||
      lhs.getLayout().getVl() != accumulator.getLayout().getVl() ||
      lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
      accumulator.getLayout().getSew() != 2 * lhs.getLayout().getSew() ||
      lhs.getLayout().getLmulEighths() != rhs.getLayout().getLmulEighths() ||
      accumulator.getLayout().getLmulEighths() !=
          2 * lhs.getLayout().getLmulEighths() ||
      !compatibleFreeAxes ||
      accumulator.getAxisIds().asArrayRef() !=
          llvm::ArrayRef<int64_t>(expectedAxes) ||
      accumulator.getShape().asArrayRef() !=
          llvm::ArrayRef<int64_t>(expectedShape) ||
      !operandMapsToPartial(lhs) || !operandMapsToPartial(rhs) || !loopCarried ||
      !exactLeaf(getLeaf(), "rvv", "widen-accumulate",
                 "rvv.vwmacc.partial", "none", "agnostic"))
    return emitOpError(
        "RVV widened accumulation requires matching single-window operands and one loop-carried partial");
  return mlir::success();
}

mlir::LogicalResult RVVFinalizeWidenDotOp::verify() {
  ValueType partial = getPartial().getType();
  auto partialElement =
      mlir::dyn_cast<mlir::IntegerType>(partial.getElementType());
  auto resultValue = mlir::dyn_cast<ValueType>(getResult().getType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
      resultValue ? resultValue.getElementType() : getResult().getType());
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  for (auto [axis, extent] : llvm::zip(partial.getAxisIds().asArrayRef(),
                                      partial.getShape().asArrayRef()))
    if (axis != getReductionAxis()) {
      expectedAxes.push_back(axis);
      expectedShape.push_back(extent);
    }
  const bool legalResult =
      (!resultValue && expectedAxes.empty() && isScalar(getResult().getType())) ||
      (resultValue && resultValue.getLayout().getCarrier() == "scalar" &&
       llvm::equal(expectedAxes, resultValue.getAxisIds().asArrayRef()) &&
       llvm::equal(expectedShape, resultValue.getShape().asArrayRef()));
  auto loop = getPartial().getDefiningOp<mlir::scf::ForOp>();
  auto reduction = llvm::find(partial.getAxisIds().asArrayRef(),
                              getReductionAxis());
  const size_t reductionPosition =
      reduction == partial.getAxisIds().asArrayRef().end()
          ? 0
          : static_cast<size_t>(reduction -
                                partial.getAxisIds().asArrayRef().begin());
  bool completeMapping = partial.getLayout().getCarrier() == "rvv" && loop &&
                         reduction != partial.getAxisIds().asArrayRef().end();
  if (completeMapping)
    for (size_t position = 0; position < partial.getShape().size(); ++position) {
      const bool reductionDimension = position == reductionPosition;
      completeMapping &= partial.getLayout().getTimeFactors()[position] == 1 &&
                         partial.getLayout().getFragmentFactors()[position] == 1 &&
                         partial.getLayout().getLocalFactors()[position] == 1 &&
                         (reductionDimension
                              ? partial.getLayout().getLaneFactors()[position] ==
                                        partial.getShape()[position] &&
                                    partial.getLayout().getReplicaFactors()[position] == 1
                              : partial.getLayout().getLaneFactors()[position] == 1 &&
                                    (partial.getShape()[position] > 0
                                         ? partial.getLayout().getReplicaFactors()[position] ==
                                               partial.getShape()[position]
                                         : partial.getLayout().getReplicaFactors()[position] > 0));
    }
  if (completeMapping && resultValue)
    for (size_t position = 0; position < resultValue.getShape().size(); ++position)
      completeMapping &=
          resultValue.getLayout().getTimeFactors()[position] == 1 &&
          resultValue.getLayout().getLaneFactors()[position] == 1 &&
          resultValue.getLayout().getReplicaFactors()[position] ==
              partial.getLayout().getReplicaFactors()[position] &&
          resultValue.getLayout().getFragmentFactors()[position] == 1 &&
          resultValue.getLayout().getLocalFactors()[position] == 1;
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!partialElement || !partialElement.isSigned() ||
      partialElement.getWidth() != 16 || !resultElement ||
      !resultElement.isSigned() || resultElement.getWidth() != 32 ||
      !completeMapping || !legalResult || !kernel ||
      !supportsLayout(kernel.getTarget(), partial.getLayout()) ||
      !exactLeaf(getLeaf(), "rvv", "finalize-widen-dot",
                 "rvv.vwredsum.partial", "none", "exact"))
    return emitOpError(
        "final widened dot reduction must remove one lane axis and preserve every free replica");
  return mlir::success();
}

mlir::LogicalResult RVVWidenReduceOp::verify() {
  auto input = getInput().getType();
  auto inputElement =
      mlir::dyn_cast<mlir::IntegerType>(input.getElementType());
  auto resultElement =
      mlir::dyn_cast<mlir::IntegerType>(getResult().getType());
  auto parts = physicalPartCount(input);
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError(
        "RVV widening reduction must be nested in one target kernel");
  auto target = kernel.getTarget();
  if (!inputElement || inputElement.isSignless() || !resultElement ||
      resultElement.isSignless() || getKind() != "add" ||
      resultElement.getWidth() != inputElement.getWidth() * 2 ||
      inputElement.getWidth() < 8 || inputElement.getWidth() > 16 ||
      !isScalar(getResult().getType()) || getAxis() < 0 ||
      static_cast<size_t>(getAxis()) >= input.getShape().size() ||
      input.getLayout().getCarrier() != "rvv" ||
      checkedPositiveProduct(input.getLayout().getTimeFactors().asArrayRef()) !=
          std::optional<int64_t>(1) || !parts || *parts != 1 ||
      !target.getHasRVV() || !target.getHasWideningInteger() ||
      !llvm::is_contained(target.getSupportedSEW().asArrayRef(),
                          input.getLayout().getSew()) ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(),
                          input.getLayout().getLmulEighths()) ||
      !exactLeaf(getLeaf(), "rvv", "widen-reduce", "rvv.vwredsum", "none",
                 "exact"))
    return emitOpError(
        "RVV widening reduction requires one signedness-qualified integer "
        "vector, an add reduction axis, and a double-width scalar result");
  return mlir::success();
}

mlir::LogicalResult RVVPartitionedWidenReduceStoreOp::verify() {
  auto input = getInput().getType();
  auto inputElement =
      mlir::dyn_cast<mlir::IntegerType>(input.getElementType());
  auto destination = getDestination().getType();
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(destination.getEncoding());
  auto destinationField = getDestination().getDefiningOp<FieldOp>();
  auto parts = physicalPartCount(input);
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError(
        "partitioned widening reduction store must be nested in one target kernel");
  auto target = kernel.getTarget();
  std::string destinationFamily;
  if (inputElement)
    destinationFamily =
        std::string(inputElement.isUnsigned() ? "u" : "i") +
        std::to_string(inputElement.getWidth() * 2);
  auto axis = llvm::find(destination.getAxisIds().asArrayRef(),
                         getDestinationAxis());
  if (!inputElement || !inputElement.isSigned() || inputElement.getWidth() != 8 ||
      input.getShape().size() != 1 || getPartition() != 16 || getCount() <= 1 ||
      input.getShape()[0] != getPartition() * getCount() ||
      input.getLayout().getCarrier() != "rvv" ||
      input.getLayout().getValidity() != "full" || !parts ||
      getCount() % *parts || !encoding || !destinationField ||
      destinationField.getAccess().getMapping() != "natural" ||
      destinationField.getAccess().getBitOffset() % 8 ||
      encoding.getKind() != "dense" ||
      encoding.getFamily() != destinationFamily ||
      axis == destination.getAxisIds().asArrayRef().end() ||
      getAccess().getForm() != "unit" || getAccess().getMapping() != "dense" ||
      !target.getHasRVV() || !target.getHasWideningInteger() ||
      !llvm::is_contained(target.getSupportedSEW().asArrayRef(),
                          input.getLayout().getSew()) ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(),
                          input.getLayout().getLmulEighths()) ||
      !exactLeaf(getLeaf(), "rvv", "partitioned-widen-reduce-store",
                 "rvv.partitioned-vwredsum-store", "none", "exact"))
    return emitOpError()
           << "partitioned widening reduction store requires one full signed-i8 "
              "RVV axis split into 16-element partitions, an aligned natural "
              "i16 field, and a closed unit destination edge on a "
              "widening-capable target; input="
           << input << ", physical_parts=" << (parts ? *parts : -1)
           << ", partition=" << getPartition() << ", count=" << getCount()
           << ", destination=" << destination << ", field_access="
           << (destinationField ? destinationField.getAccess() : AccessAttr())
           << ", store_access=" << getAccess() << ", target=" << target;
  return mlir::success();
}

mlir::LogicalResult RVVLayeredWindowOp::verify() {
  auto field = getField().getType();
  auto first = getFirst().getType();
  auto second = getSecond().getType();
  auto element = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  const int64_t group = getAccess().getGroupSize();
  const int64_t layer = getAccess().getLayerSize();
  const int64_t axis = getPoint().getType().getDomain().getAxisId();
  auto resultAxis = llvm::find(first.getAxisIds().asArrayRef(), axis);
  const bool hasLayerAxis =
      resultAxis != first.getAxisIds().asArrayRef().end() &&
      first.getShape()[static_cast<size_t>(
          resultAxis - first.getAxisIds().asArrayRef().begin())] == layer;
  auto physicalPoint = getPoint().getDefiningOp<PhysicalPointOp>();
  auto partition = physicalPoint
                       ? physicalPoint.getPartition().getDefiningOp<
                             mlir::arith::ConstantIndexOp>()
                       : mlir::arith::ConstantIndexOp();
  llvm::StringRef validity = first.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  if (!element || element.isSigned() || field.getShape().size() != first.getShape().size() ||
      first != second || first.getElementType() != field.getElementType() ||
      first.getAxisIds() != field.getAxisIds() || !hasLayerAxis ||
      first.getLayout().getCarrier() != "rvv" ||
      (validity != "full" && validity != "tail") ||
      getAccess().getForm() != "indexed" ||
      getAccess().getMapping() != "grouped_layered" || group <= 0 ||
      layer <= 0 || group != layer * 2 || element.getWidth() * 2 > 8 ||
      (getAccess().getOrder() != "lo_first" &&
       getAccess().getOrder() != "hi_first") ||
      !partition || partition.value() != layer ||
      !exactLeaf(getLeaf(), "rvv", "layered-window", "rvv.layered-window",
                 "none", tail))
    return emitOpError(
        "RVV layered window requires two equal full/tail layer results, one "
        "two-layer packed field, and an identical exact layer-sized point");
  return mlir::success();
}

mlir::LogicalResult RVVLayeredStreamOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto element = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  auto sourceField = getField().getDefiningOp<FieldOp>();
  const int64_t group = getAccess().getGroupSize();
  const int64_t layer = getAccess().getLayerSize();
  const int64_t layers = layer > 0 ? group / layer : 0;
  bool hasStreamAxis = false;
  bool onlyOneStreamAxis = true;
  for (size_t index = 0; index < field.getShape().size(); ++index) {
    const int64_t extent = field.getShape()[index];
    const int64_t time = field.getLayout().getTimeFactors()[index];
    const int64_t lanes = field.getLayout().getLaneFactors()[index];
    const bool streamAxis =
        group > 0 && layer > 0 && extent > 0 && extent % group == 0 &&
        lanes > 1 && layer % lanes == 0 && time > 1 &&
        time * lanes == extent;
    if (streamAxis) {
      onlyOneStreamAxis &= !hasStreamAxis;
      hasStreamAxis = true;
    } else {
      onlyOneStreamAxis &= time == 1 && lanes == 1;
    }
  }
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  if (!element || element.isSigned() || !sourceField ||
      getAccess() != sourceField.getAccess() || field != result ||
      field.getLayout().getCarrier() != "rvv" || !hasStreamAxis ||
      !onlyOneStreamAxis ||
      (validity != "full" && validity != "tail") ||
      getAccess().getForm() != "indexed" ||
      getAccess().getMapping() != "grouped_layered" || group <= 0 ||
      layer <= 0 || group % layer || layers <= 1 ||
      element.getWidth() * layers > 8 || getAccess().getBitOffset() % 8 ||
      (getAccess().getOrder() != "lo_first" &&
       getAccess().getOrder() != "hi_first") ||
      !exactLeaf(getLeaf(), "rvv", "layered-stream", "rvv.layered-stream",
                 "none", tail))
    return emitOpError(
        "RVV layered stream requires one byte-aligned packed field whose "
        "grouped/layered geometry is represented by one complete time/lane axis");
  return mlir::success();
}

mlir::LogicalResult RVVStreamReduceOp::verify() {
  auto source = mlir::dyn_cast<MemDescType>(getSource().getType());
  auto encoding = source
                      ? mlir::dyn_cast<kernel::EncodingType>(source.getEncoding())
                      : kernel::EncodingType();
  auto layout = getInputLayout();
  auto streams = checkedPositiveProduct(
      layout.getTimeFactors().asArrayRef());
  if (!source || !encoding || encoding.getKind() != "dense" ||
      encoding.getFamily() != "f32" || layout.getSew() != 32 || !streams ||
      *streams <= 1 || layout.getCarrier() != "rvv" ||
      layout.getValidity() != "full" || layout.getAxisIds().size() != 1 ||
      layout.getReplicaFactors()[0] != 1 || layout.getLaneFactors()[0] <= 1 ||
      layout.getLaneFactors()[0] != layout.getVl() || getAxis() != 0 ||
      getKinds().size() != getNumResults() || getNumResults() == 0 ||
      (getAccess().getForm() != "unit" && getAccess().getForm() != "strided") ||
      getAccess().getMapping() != "dense" ||
      !exactLeaf(getLeaf(), "rvv", "stream-reduce",
                 "rvv.stream-reduce", "none", "exact"))
    return emitOpError(
        "RVV stream reduction requires a full multi-stream f32 value, one "
        "valid axis, matching scalar results, and a selected memory edge");
  for (auto [kind, result] : llvm::zip(getKinds(), getResults())) {
    auto name = mlir::dyn_cast<mlir::StringAttr>(kind);
    if (!name || (name.getValue() != "add" && name.getValue() != "max" &&
                  name.getValue() != "min") ||
        !result.getType().isF32())
      return emitOpError(
          "RVV stream reduction kinds and scalar result types disagree");
  }
  return mlir::success();
}

mlir::LogicalResult RVVStreamDotOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto lhsEncoding =
      mlir::dyn_cast<kernel::EncodingType>(lhs.getEncoding());
  auto rhsEncoding =
      mlir::dyn_cast<kernel::EncodingType>(rhs.getEncoding());
  auto layout = getInputLayout();
  const bool lhsMemory = getLhsAccess().getForm() == "unit" ||
                         getLhsAccess().getForm() == "strided";
  const bool rhsMemory = getRhsAccess().getForm() == "unit" ||
                         getRhsAccess().getForm() == "strided";
  if (!lhsEncoding || !rhsEncoding || lhsEncoding.getKind() != "dense" ||
      rhsEncoding.getKind() != "dense" || lhsEncoding.getFamily() != "f32" ||
      rhsEncoding.getFamily() != "f32" || lhs.getShape() != rhs.getShape() ||
      lhs.getAxisIds() != rhs.getAxisIds() || layout.getSew() != 32 ||
      layout.getCarrier() != "rvv" ||
      (layout.getValidity() != "full" && layout.getValidity() != "tail") ||
      layout.getAxisIds().size() != 1 ||
      layout.getReplicaFactors()[0] != 1 || layout.getLaneFactors()[0] <= 1 ||
      layout.getLaneFactors()[0] != layout.getVl() || getAxis() != 0 ||
      !lhsMemory || !rhsMemory || getLhsAccess().getMapping() != "dense" ||
      getRhsAccess().getMapping() != "dense" || !getResult().getType().isF32() ||
      !exactLeaf(getLeaf(), "rvv", "stream-dot", "rvv.stream-dot", "none",
                 "exact"))
    return emitOpError(
        "RVV stream dot requires matching dense f32 slices, one full "
        "multi-stream RVV reduction axis, two selected memory edges, and a "
        "scalar f32 result");
  return mlir::success();
}

mlir::LogicalResult RVVStreamContractOp::verify() {
  auto lhs = getLhs().getType();
  auto rhs = getRhs().getType();
  auto result = getResult().getType();
  auto lhsEncoding =
      mlir::dyn_cast<kernel::EncodingType>(lhs.getEncoding());
  auto rhsEncoding =
      mlir::dyn_cast<kernel::EncodingType>(rhs.getEncoding());
  auto operand = getOperandLayout();
  auto accumulator = getAccumulatorLayout();
  const bool lhsMemory = getLhsAccess().getForm() == "unit" ||
                         getLhsAccess().getForm() == "strided";
  const bool rhsMemory = getRhsAccess().getForm() == "unit" ||
                         getRhsAccess().getForm() == "strided";
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  auto appendFreeAxes = [&](MemDescType operand) -> mlir::LogicalResult {
    bool foundReduction = false;
    for (auto [shape, axis] : llvm::zip(operand.getShape().asArrayRef(),
                                        operand.getAxisIds().asArrayRef())) {
      if (axis == getReductionAxis()) {
        foundReduction = true;
        continue;
      }
      auto found = llvm::find(expectedAxes, axis);
      if (found == expectedAxes.end()) {
        expectedAxes.push_back(axis);
        expectedShape.push_back(shape);
      } else if (expectedShape[found - expectedAxes.begin()] != shape) {
        return emitOpError(
            "stream contract operands disagree on a free-axis shape");
      }
    }
    return mlir::success(foundReduction);
  };
  if (mlir::failed(appendFreeAxes(lhs)) || mlir::failed(appendFreeAxes(rhs)))
    return mlir::failure();
  llvm::SmallVector<int64_t> expectedAccumulatorAxes(expectedAxes.begin(),
                                                     expectedAxes.end());
  expectedAccumulatorAxes.push_back(getReductionAxis());
  auto resultLanes = checkedPositiveProduct(
      result.getLayout().getLaneFactors().asArrayRef());
  const bool ordinaryF32 = lhsEncoding && rhsEncoding &&
                           lhsEncoding.getFamily() == "f32" &&
                           rhsEncoding.getFamily() == "f32";
  const bool wideningF16 = lhsEncoding && rhsEncoding &&
                           lhsEncoding.getFamily() == "f16" &&
                           rhsEncoding.getFamily() == "f16";
  const bool exactInstruction =
      ordinaryF32
          ? exactLeaf(getLeaf(), "rvv", "stream-contract",
                      "rvv.stream-contract", "none", "exact")
          : wideningF16 &&
                exactLeaf(getLeaf(), "rvv", "stream-contract",
                          "rvv.stream-widen-contract", "none", "exact");
  if (!lhsEncoding || !rhsEncoding || lhsEncoding.getKind() != "dense" ||
      rhsEncoding.getKind() != "dense" ||
      (!ordinaryF32 && !wideningF16) || getReductionAxis() <= 0 ||
      result.getElementType() != mlir::Float32Type::get(getContext()) ||
      result.getAxisIds().asArrayRef() != llvm::ArrayRef(expectedAxes) ||
      result.getShape().asArrayRef() != llvm::ArrayRef(expectedShape) ||
      result.getLayout().getCarrier() != "scalar" ||
      !resultLanes || *resultLanes != 1 ||
      operand.getCarrier() != "rvv" ||
      operand.getAxisIds().size() != 1 ||
      operand.getAxisIds()[0] != static_cast<int64_t>(getReductionAxis()) ||
      operand.getLaneFactors().size() != 1 ||
      operand.getLaneFactors()[0] != operand.getVl() || operand.getVl() <= 1 ||
      operand.getSew() != (wideningF16 ? 16 : 32) ||
      accumulator.getCarrier() != "rvv" || accumulator.getSew() != 32 ||
      accumulator.getAxisIds().asArrayRef() !=
          llvm::ArrayRef(expectedAccumulatorAxes) ||
      accumulator.getLaneFactors().asArrayRef().back() != accumulator.getVl() ||
      accumulator.getVl() <= 1 || accumulator.getLmulEighths() <= 0 ||
      getStationaryOperand() != "lhs" && getStationaryOperand() != "rhs" ||
      getUnroll() <= 0 ||
      (wideningF16
           ? accumulator.getLmulEighths() != operand.getLmulEighths() * 2
           : accumulator.getLmulEighths() != operand.getLmulEighths()) ||
      accumulator.getVl() != operand.getVl() || !lhsMemory || !rhsMemory ||
      getLhsAccess().getMapping() != "dense" ||
      getRhsAccess().getMapping() != "dense" || !exactInstruction)
    return emitOpError(
        "RVV stream contract requires two matching dense floating slices, one shared "
        "reduction lane, scalar free-axis replicas, selected memory edges, "
        "and a closed local schedule");
  for (size_t index = 0; index < expectedAxes.size(); ++index) {
    if (accumulator.getTimeFactors()[index] != 1 ||
        accumulator.getLaneFactors()[index] != 1 ||
        accumulator.getReplicaFactors()[index] !=
            result.getLayout().getReplicaFactors()[index] ||
        accumulator.getFragmentFactors()[index] != 1 ||
        accumulator.getLocalFactors()[index] != 1)
      return emitOpError(
          "stream contract accumulator does not preserve its free-axis replicas");
  }
  const size_t reductionPosition = expectedAxes.size();
  if (accumulator.getTimeFactors()[reductionPosition] != 1 ||
      accumulator.getReplicaFactors()[reductionPosition] != 1 ||
      accumulator.getFragmentFactors()[reductionPosition] != 1 ||
      accumulator.getLocalFactors()[reductionPosition] != 1)
    return emitOpError(
        "stream contract reduction axis has a non-lane physical factor");
  return mlir::success();
}

mlir::LogicalResult RVVSplatOp::verify() {
  if (!isScalar(getScalar().getType()) ||
      !exactLeaf(getLeaf(), "rvv", "splat", "rvv.splat", "none", "exact") ||
      getResult().getType().getLayout().getCarrier() != "rvv" ||
      elementOf(getScalar().getType()) !=
          getResult().getType().getElementType())
    return emitOpError()
           << "RVV splat requires a matching scalar and RVV result; scalar="
           << getScalar().getType() << ", result=" << getResult().getType()
           << ", leaf engine=" << getLeaf().getEngine();
  return mlir::success();
}

mlir::LogicalResult ProjectReductionOperandOp::verify() {
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  llvm::SmallVector<int64_t> expectedShape;
  llvm::SmallVector<int64_t> expectedAxes;
  bool found = false;
  for (auto [shape, axis] :
       llvm::zip(input.getShape().asArrayRef(), input.getAxisIds().asArrayRef())) {
    if (axis == getReductionAxis()) {
      if (found)
        return emitOpError("reduction operand axis identity is not unique");
      found = true;
      continue;
    }
    expectedShape.push_back(shape);
    expectedAxes.push_back(axis);
  }
  if (!found || input.getElementType() != result.getElementType() ||
      result.getShape().asArrayRef() != llvm::ArrayRef(expectedShape) ||
      result.getAxisIds().asArrayRef() != llvm::ArrayRef(expectedAxes) ||
      result.getLayout().getCarrier() == "ime" ||
      result.getLayout().getCarrier() == "local" ||
      !exactLeaf(getLeaf(), "transfer", "reduction-projection",
                 "rvv.project-reduction-operand", "none", "exact"))
    return emitOpError(
        "reduction operand projection must remove exactly one logical axis");
  auto inputLayout = input.getLayout();
  auto resultLayout = result.getLayout();
  for (auto [position, axis] :
       llvm::enumerate(result.getAxisIds().asArrayRef())) {
    auto found = llvm::find(input.getAxisIds().asArrayRef(), axis);
    if (found == input.getAxisIds().asArrayRef().end())
      return emitOpError("reduction projection introduced a new free axis");
    size_t sourcePosition =
        static_cast<size_t>(found - input.getAxisIds().asArrayRef().begin());
    if (resultLayout.getTimeFactors()[position] !=
            inputLayout.getTimeFactors()[sourcePosition] ||
        resultLayout.getLaneFactors()[position] !=
            inputLayout.getLaneFactors()[sourcePosition] ||
        resultLayout.getReplicaFactors()[position] !=
            inputLayout.getReplicaFactors()[sourcePosition] ||
        resultLayout.getFragmentFactors()[position] !=
            inputLayout.getFragmentFactors()[sourcePosition] ||
        resultLayout.getLocalFactors()[position] !=
            inputLayout.getLocalFactors()[sourcePosition])
      return emitOpError(
          "reduction projection must preserve every free-axis physical coordinate");
  }
  return mlir::success();
}

mlir::LogicalResult RVVContractStepOp::verify() {
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  auto appendFree = [&](mlir::Type type) -> mlir::LogicalResult {
    for (auto [axis, extent] : llvm::zip(axesOf(type), shapeOf(type))) {
      if (axis == getReductionAxis())
        continue;
      auto found = llvm::find(expectedAxes, axis);
      if (found == expectedAxes.end()) {
        expectedAxes.push_back(axis);
        expectedShape.push_back(extent);
      } else if (expectedShape[static_cast<size_t>(found - expectedAxes.begin())] !=
                 extent) {
        return emitOpError(
            "RVV contract step operands disagree on a free-axis extent");
      }
    }
    return mlir::success();
  };
  if (failed(appendFree(getLhs().getType())) ||
      failed(appendFree(getRhs().getType())))
    return mlir::failure();
  mlir::Value laneValue = getLaneOperand() == "lhs" ? getLhs() : getRhs();
  auto laneType = mlir::dyn_cast<ValueType>(laneValue.getType());
  if (getAccumulator().getType() != getResult().getType() ||
      getReductionAxis() <= 0 ||
      llvm::is_contained(getResult().getType().getAxisIds().asArrayRef(),
                         getReductionAxis()) ||
      !llvm::equal(expectedAxes, getResult().getType().getAxisIds().asArrayRef()) ||
      !llvm::equal(expectedShape, getResult().getType().getShape().asArrayRef()) ||
      (getLaneOperand() != "lhs" && getLaneOperand() != "rhs") ||
      (getLaneMemoryForm() != "unit" &&
       getLaneMemoryForm() != "strided" &&
      getLaneMemoryForm() != "indexed" &&
       getLaneMemoryForm() != "register") ||
      !laneType || laneType.getLayout() != getLaneLoadLayout() ||
      getLaneLoadLayout().getCarrier() != "rvv" ||
      getLeaf().getEngine() != "rvv" ||
      getLeaf().getFamily() != "contract-step" ||
      (getLeaf().getInstruction() != "rvv.vfmacc.vf" &&
       getLeaf().getInstruction() != "rvv.vfwmacc.vf" &&
       getLeaf().getInstruction() != "rvv.vmacc.vx") ||
      getLeaf().getSpelling() != getLeaf().getInstruction() ||
      getLeaf().getMask() != "none" || getLeaf().getTail() != "agnostic" ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>(
              {static_cast<int64_t>(getReductionAxis())}) ||
      (getLeaf().getInstruction() == "rvv.vfmacc.vf" &&
       (elementOf(getLhs().getType()) != getResult().getType().getElementType() ||
        elementOf(getRhs().getType()) !=
            getResult().getType().getElementType())) ||
      (getLeaf().getInstruction() == "rvv.vfwmacc.vf" &&
       (!elementOf(getLhs().getType()).isF16() ||
        !elementOf(getRhs().getType()).isF16() ||
        !getResult().getType().getElementType().isF32())))
    return emitOpError()
           << "RVV contract step has an incomplete typed contract; accumulator="
           << getAccumulator().getType() << ", result=" << getResult().getType()
           << ", reduction_axis=" << getReductionAxis()
           << ", lane_operand=" << getLaneOperand()
           << ", lane_memory_form=" << getLaneMemoryForm()
           << ", lane_load_layout=" << getLaneLoadLayout()
           << ", leaf=" << getLeaf();
  return mlir::success();
}

mlir::LogicalResult RVVEncodedContractStepOp::verify() {
  auto lhs = mlir::dyn_cast<ValueType>(getLhs().getType());
  auto rhs = mlir::dyn_cast<ValueType>(getRhs().getType());
  auto lhsInteger = lhs ? mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType())
                        : mlir::IntegerType();
  auto rhsInteger = rhs ? mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType())
                        : mlir::IntegerType();
  const bool mixedNarrow =
      lhsInteger && rhsInteger && lhsInteger.getWidth() <= 8 &&
      rhsInteger.getWidth() <= 8 &&
      ((lhsInteger.isUnsigned() && rhsInteger.isSigned()) ||
       (lhsInteger.isSigned() && rhsInteger.isUnsigned()));
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  auto appendFree = [&](ValueType value) -> mlir::LogicalResult {
    for (auto [axis, extent] : llvm::zip(value.getAxisIds().asArrayRef(),
                                        value.getShape().asArrayRef())) {
      if (axis == getReductionAxis())
        continue;
      auto found = llvm::find(expectedAxes, axis);
      if (found == expectedAxes.end()) {
        expectedAxes.push_back(axis);
        expectedShape.push_back(extent);
      } else if (expectedShape[static_cast<size_t>(found - expectedAxes.begin())] !=
                 extent) {
        return emitOpError(
            "encoded contract operands disagree on a free-axis extent");
      }
    }
    return mlir::success();
  };
  if (lhs && rhs &&
      (failed(appendFree(lhs)) || failed(appendFree(rhs))))
    return mlir::failure();
  ValueType laneType = getLaneOperand() == "lhs" ? lhs : rhs;
  if (!mixedNarrow || getAccumulator().getType() != getResult().getType() ||
      getReductionAxis() <= 0 ||
      llvm::is_contained(getResult().getType().getAxisIds().asArrayRef(),
                         getReductionAxis()) ||
      !llvm::equal(expectedAxes, getResult().getType().getAxisIds().asArrayRef()) ||
      !llvm::equal(expectedShape, getResult().getType().getShape().asArrayRef()) ||
      (getLaneOperand() != "lhs" && getLaneOperand() != "rhs") ||
      getLhsAccess().getForm() == "unassigned" ||
      getRhsAccess().getForm() == "unassigned" ||
      getLhsAccess().getMapping() == "opaque" ||
      getRhsAccess().getMapping() == "opaque" ||
      !laneType || laneType.getAxisIds() != getLaneLoadLayout().getAxisIds() ||
      getLaneLoadLayout().getCarrier() != "rvv" ||
      getLaneLoadLayout().getSew() != 8 ||
      !exactLeaf(getLeaf(), "rvv", "encoded-contract-step",
                 "rvv.vmacc.decoded-u8-s8", "none", "agnostic") ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>(
              {static_cast<int64_t>(getReductionAxis())}))
    return emitOpError()
           << "encoded contract step requires mixed narrow integers, typed "
              "accesses, an i32 RVV accumulator, and one closed decode-MAC "
              "leaf; lhs="
           << getLhs().getType() << ", rhs=" << getRhs().getType()
           << ", accumulator=" << getAccumulator().getType()
           << ", result=" << getResult().getType()
           << ", lane_operand=" << getLaneOperand()
           << ", lane_load_layout=" << getLaneLoadLayout()
           << ", lhs_access=" << getLhsAccess()
           << ", rhs_access=" << getRhsAccess() << ", leaf=" << getLeaf()
           << ", mixed=" << mixedNarrow
           << ", same_accumulator="
           << (getAccumulator().getType() == getResult().getType())
           << ", result_has_reduction_axis="
           << llvm::is_contained(
                  getResult().getType().getAxisIds().asArrayRef(),
                  getReductionAxis())
           << ", lane_name_valid="
           << (getLaneOperand() == "lhs" || getLaneOperand() == "rhs")
           << ", accesses_valid="
           << (getLhsAccess().getForm() != "unassigned" &&
               getRhsAccess().getForm() != "unassigned" &&
               getLhsAccess().getMapping() != "opaque" &&
               getRhsAccess().getMapping() != "opaque")
           << ", axes_match="
           << llvm::equal(expectedAxes,
                          getResult().getType().getAxisIds().asArrayRef())
           << ", shape_match="
           << llvm::equal(expectedShape,
                          getResult().getType().getShape().asArrayRef())
           << ", lane_axes_match="
           << (laneType &&
               laneType.getAxisIds() == getLaneLoadLayout().getAxisIds())
           << ", lane_carrier=" << getLaneLoadLayout().getCarrier()
           << ", lane_sew=" << getLaneLoadLayout().getSew()
           << ", leaf_match="
           << exactLeaf(getLeaf(), "rvv", "encoded-contract-step",
                        "rvv.vmacc.decoded-u8-s8", "none", "agnostic")
           << ", parameters_match="
           << (getLeaf().getParameters().asArrayRef() ==
               llvm::ArrayRef<int64_t>(
                   {static_cast<int64_t>(getReductionAxis())}));
  auto accumulatorElement = getResult().getType().getElementType();
  auto accumulatorInteger =
      mlir::dyn_cast<mlir::IntegerType>(accumulatorElement);
  if (!accumulatorInteger || !accumulatorInteger.isSigned() ||
      accumulatorInteger.getWidth() != 32)
    return emitOpError("encoded contract step accumulator must be signed i32");
  return mlir::success();
}

void WEFTRISCVDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Weft/Dialect/RISCV/IR/RISCVAttrs.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "Weft/Dialect/RISCV/IR/RISCVTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/RISCV/IR/RISCVOps.cpp.inc"
      >();
}
