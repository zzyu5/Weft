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
#include "mlir/Interfaces/SideEffectInterfaces.h"
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

bool weft::riscv::supportsRVVLayout(TargetAttr target, LayoutAttr layout) {
  int64_t elen = 0;
  for (int64_t supported : target.getSupportedSEW().asArrayRef())
    elen = std::max(elen, supported);
  const bool legalFractionalLMUL =
      layout && elen > 0 &&
      layout.getLmulEighths() * elen >= 8 * layout.getSew();
  return layout && layout.getCarrier() == "rvv" && target.getHasRVV() &&
         target.getVlenBits() > 0 && target.getVectorRegisters() > 0 &&
         llvm::is_contained(target.getSupportedSEW().asArrayRef(),
                            layout.getSew()) &&
         llvm::is_contained(target.getLegalLMULEighths().asArrayRef(),
                            layout.getLmulEighths()) &&
         legalFractionalLMUL;
}

std::optional<llvm::SmallVector<int64_t>>
weft::riscv::groupedMacSupplyProjection(ValueType packed, ValueType result) {
  if (!packed || !result || packed.getLayout().getCarrier() != "rvv" ||
      result.getLayout().getCarrier() != "rvv")
    return std::nullopt;
  auto product = [](llvm::ArrayRef<int64_t> factors) -> std::optional<int64_t> {
    int64_t value = 1;
    for (int64_t factor : factors) {
      if (factor <= 0 ||
          value > std::numeric_limits<int64_t>::max() / factor)
        return std::nullopt;
      value *= factor;
    }
    return value;
  };
  auto timeParts = product(result.getLayout().getTimeFactors().asArrayRef());
  auto registerParts = product(result.getLayout().getReplicaFactors().asArrayRef());
  if (!timeParts || !registerParts || *timeParts <= 0 || *registerParts <= 0 ||
      *registerParts > std::numeric_limits<int64_t>::max() / *timeParts)
    return std::nullopt;

  int64_t laneAxis = 0;
  for (auto [axis, factor] :
       llvm::zip(result.getAxisIds().asArrayRef(),
                 result.getLayout().getLaneFactors().asArrayRef())) {
    if (factor <= 1)
      continue;
    if (laneAxis != 0)
      return std::nullopt;
    laneAxis = axis;
  }
  if (laneAxis <= 0)
    return std::nullopt;

  llvm::SmallVector<int64_t> mapping;
  llvm::SmallVector<llvm::SmallVector<int64_t>> supplyKeys;
  const int64_t resultParts = *timeParts * *registerParts;
  mapping.reserve(static_cast<size_t>(resultParts));
  for (int64_t part = 0; part < resultParts; ++part) {
    int64_t stream = part % *timeParts;
    int64_t replica = part / *timeParts;
    llvm::SmallVector<int64_t> timeCoordinates(result.getAxisIds().size(), 0);
    llvm::SmallVector<int64_t> replicaCoordinates(result.getAxisIds().size(), 0);
    for (int64_t position = static_cast<int64_t>(result.getAxisIds().size()) - 1;
         position >= 0; --position) {
      const int64_t time = result.getLayout().getTimeFactors()[position];
      const int64_t copies = result.getLayout().getReplicaFactors()[position];
      if (time <= 0 || copies <= 0)
        return std::nullopt;
      timeCoordinates[position] = stream % time;
      stream /= time;
      replicaCoordinates[position] = replica % copies;
      replica /= copies;
    }
    if (stream != 0 || replica != 0)
      return std::nullopt;

    llvm::SmallVector<int64_t> key;
    for (size_t position = 0; position < result.getAxisIds().size(); ++position) {
      const int64_t axis = result.getAxisIds()[position];
      if (!llvm::is_contained(packed.getAxisIds().asArrayRef(), axis))
        continue;
      key.push_back(replicaCoordinates[position]);
      if (axis == laneAxis)
        key.push_back(timeCoordinates[position]);
    }
    auto found = llvm::find(supplyKeys, key);
    if (found == supplyKeys.end()) {
      mapping.push_back(static_cast<int64_t>(supplyKeys.size()));
      supplyKeys.push_back(std::move(key));
    } else {
      mapping.push_back(static_cast<int64_t>(found - supplyKeys.begin()));
    }
  }
  if (supplyKeys.empty())
    return std::nullopt;
  return mapping;
}

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

std::optional<int64_t> checkedProduct(llvm::ArrayRef<int64_t> factors) {
  int64_t result = 1;
  for (int64_t factor : factors) {
    if (factor <= 0 ||
        result > std::numeric_limits<int64_t>::max() / factor)
      return std::nullopt;
    result *= factor;
  }
  return result;
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

bool sameStorageGeometry(AccessAttr lhs, AccessAttr rhs) {
  return lhs && rhs && lhs.getMapping() == rhs.getMapping() &&
         lhs.getGroupSize() == rhs.getGroupSize() &&
         lhs.getLayerSize() == rhs.getLayerSize() &&
         lhs.getJoinFields() == rhs.getJoinFields() &&
         lhs.getJoinLowBits() == rhs.getJoinLowBits() &&
         lhs.getJoinRole() == rhs.getJoinRole() &&
         lhs.getBitOffset() == rhs.getBitOffset() &&
         lhs.getStorageBits() == rhs.getStorageBits() &&
         lhs.getOrder() == rhs.getOrder();
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
    if (auto extract = mlir::dyn_cast<ExtractOp>(definition)) {
      value = extract.getInput();
      continue;
    }
    if (auto pack = mlir::dyn_cast<EncodedLocalPackOp>(definition)) {
      value = pack.getInput();
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

std::optional<int64_t> doubledPositive(int64_t value) {
  if (value <= 0 || value > std::numeric_limits<int64_t>::max() / 2)
    return std::nullopt;
  return value * 2;
}

int64_t localPackInterleaveRows(mlir::Value owner);

bool hasGroupedMacOperandGeometry(mlir::Operation *operation, mlir::Value lhs,
                                  mlir::Value rhs, int64_t reductionAxis,
                                  GroupedMacPlanAttr plan,
                                  LayoutAttr loadLayout,
                                  LayoutAttr partialLayout,
                                  LayoutAttr resultLayout) {
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
  auto rhsMemory = rhsLoad ? rhsLoad.getRegion().getType() : MemDescType();
  int64_t lhsInterleaveRows =
      lhsField ? localPackInterleaveRows(lhsField.getOwner()) : 0;
  if (lhsInterleaveRows <= 0 && lhsMemory)
    lhsInterleaveRows = lhsMemory.getInterleaveRows();
  AccessAttr lhsAccess = lhsField ? lhsField.getAccess() : AccessAttr();
  AccessAttr rhsAccess = rhsField ? rhsField.getAccess() : AccessAttr();
  mlir::Value lhsPoint = sourcePoint(lhs, reductionAxis);
  mlir::Value rhsPoint = sourcePoint(rhs, reductionAxis);
  auto kernel = operation->getParentOfType<KernelOp>();
  if (!kernel || !lhsType || !rhsType || !lhsInteger || !rhsInteger ||
      !lhsInteger.isUnsigned() || lhsInteger.getWidth() >= 8 ||
      !rhsInteger.isSigned() || rhsInteger.getWidth() != 8 || !lhsField ||
      !rhsField || !lhsLoad || !rhsLoad || !lhsPoint || !rhsPoint ||
      lhsPoint != rhsPoint || !plan ||
      lhsAccess.getMapping() != "grouped_layered" ||
      rhsAccess.getMapping() != "natural" || plan.getGroup() <= 0 ||
      lhsAccess.getGroupSize() <= 0 || lhsAccess.getLayerSize() <= 0 ||
      lhsAccess.getGroupSize() % lhsAccess.getLayerSize() ||
      lhsAccess.getLayerSize() % plan.getGroup() || lhsAccess.getBitOffset() % 8 ||
      rhsAccess.getBitOffset() % 8 ||
      (lhsAccess.getOrder() != "lo_first" &&
       lhsAccess.getOrder() != "hi_first") ||
      !supportsRVVLayout(kernel.getTarget(), lhsType.getLayout()) ||
      !supportsRVVLayout(kernel.getTarget(), loadLayout) ||
      !supportsRVVLayout(kernel.getTarget(), partialLayout) ||
      !kernel.getTarget().getHasWideningInteger())
    return false;
  if (plan.getReductionAxis() != reductionAxis ||
      plan.getStorageGroup() != lhsAccess.getGroupSize() ||
      plan.getStorageLayer() != lhsAccess.getLayerSize() ||
      plan.getLogicalWidth() != lhsInteger.getWidth() ||
      plan.getLhsBitOffsetBytes() != lhsAccess.getBitOffset() / 8 ||
      plan.getRhsBitOffsetBytes() != rhsAccess.getBitOffset() / 8 ||
      plan.getRecordElements() != lhsMemory.getElements() ||
      plan.getRecordElements() != rhsMemory.getElements() ||
      plan.getInterleaveRows() != lhsInterleaveRows ||
      plan.getTermByteStride() !=
          (lhsInterleaveRows > 0 ? lhsInterleaveRows : 1))
    return false;
  int64_t laneAxis = 0;
  bool reductionLane = false;
  for (auto [axis, factor] :
       llvm::zip(lhsType.getAxisIds().asArrayRef(),
                 lhsType.getLayout().getLaneFactors().asArrayRef())) {
    if (axis == reductionAxis && factor > 1)
      reductionLane = true;
    if (axis != reductionAxis && factor > 1) {
      laneAxis = axis;
      break;
    }
  }
  bool hasCohortStride = false;
  if (lhsMemory && laneAxis > 0)
    for (auto [axis, stride] :
         llvm::zip(lhsMemory.getAxisIds().asArrayRef(),
                   lhsMemory.getStrides().asArrayRef()))
      if (axis == laneAxis && stride != 0)
        hasCohortStride = true;
  const bool result =
      lhsMemory && ((plan.getLoadForm() == "unit" &&
                     lhsInterleaveRows > 0) ||
                    (plan.getLoadForm() == "strided" && hasCohortStride &&
                     plan.getRowStrideAxis() == laneAxis));
  (void)reductionLane;
  auto sameCoordinates = [](LayoutAttr lhsLayout, LayoutAttr rhsLayout) {
    return lhsLayout.getAxisIds() == rhsLayout.getAxisIds() &&
           lhsLayout.getTimeFactors() == rhsLayout.getTimeFactors() &&
           lhsLayout.getLaneFactors() == rhsLayout.getLaneFactors() &&
           lhsLayout.getReplicaFactors() == rhsLayout.getReplicaFactors() &&
           lhsLayout.getFragmentFactors() == rhsLayout.getFragmentFactors() &&
           lhsLayout.getLocalFactors() == rhsLayout.getLocalFactors() &&
           lhsLayout.getVl() == rhsLayout.getVl() &&
           lhsLayout.getValidity() == rhsLayout.getValidity();
  };
  auto resultType = mlir::dyn_cast<ValueType>(
      mlir::isa<RVVGroupedMacReduceOp>(operation)
          ? mlir::cast<RVVGroupedMacReduceOp>(operation).getResult().getType()
          : mlir::cast<RVVGroupedMacLoadOp>(operation)
                .getResult()
                .getType()
                .getResultType());
  if (!resultType || resultType.getLayout() != resultLayout ||
      resultLayout.getSew() != 32 ||
      resultLayout.getLmulEighths() !=
          2 * partialLayout.getLmulEighths())
    return false;
  auto expectedGroups = [](LayoutAttr layout) {
    auto replicas = checkedPositiveProduct(
        layout.getReplicaFactors().asArrayRef());
    return replicas ? ((layout.getLmulEighths() + 7) / 8) * *replicas : -1;
  };
  if (loadLayout.getRegisterGroups() != expectedGroups(loadLayout) ||
      partialLayout.getRegisterGroups() != expectedGroups(partialLayout) ||
      resultLayout.getRegisterGroups() != expectedGroups(resultLayout))
    return false;
  for (auto [position, axis] :
       llvm::enumerate(lhsType.getAxisIds().asArrayRef())) {
    auto resultAxis = llvm::find(resultLayout.getAxisIds().asArrayRef(), axis);
    if (resultAxis == resultLayout.getAxisIds().asArrayRef().end()) {
      if (axis != reductionAxis ||
          loadLayout.getTimeFactors()[position] !=
              std::max<int64_t>(1, lhsType.getShape()[position]) ||
          loadLayout.getLaneFactors()[position] != 1 ||
          loadLayout.getReplicaFactors()[position] != 1 ||
          loadLayout.getFragmentFactors()[position] != 1 ||
          loadLayout.getLocalFactors()[position] != 1)
        return false;
      continue;
    }
    const size_t resultPosition = static_cast<size_t>(
        resultAxis - resultLayout.getAxisIds().asArrayRef().begin());
    if (loadLayout.getTimeFactors()[position] !=
            resultLayout.getTimeFactors()[resultPosition] ||
        loadLayout.getLaneFactors()[position] !=
            resultLayout.getLaneFactors()[resultPosition] ||
        loadLayout.getReplicaFactors()[position] !=
            resultLayout.getReplicaFactors()[resultPosition] ||
        loadLayout.getFragmentFactors()[position] !=
            resultLayout.getFragmentFactors()[resultPosition] ||
        loadLayout.getLocalFactors()[position] !=
            resultLayout.getLocalFactors()[resultPosition])
      return false;
  }
  for (auto [position, axis] :
       llvm::enumerate(rhsType.getAxisIds().asArrayRef())) {
    if (axis == reductionAxis)
      continue;
    auto resultAxis = llvm::find(resultLayout.getAxisIds().asArrayRef(), axis);
    if (resultAxis == resultLayout.getAxisIds().asArrayRef().end())
      return false;
    const size_t resultPosition = static_cast<size_t>(
        resultAxis - resultLayout.getAxisIds().asArrayRef().begin());
    auto rhsLayout = rhsType.getLayout();
    if (rhsLayout.getTimeFactors()[position] !=
            resultLayout.getTimeFactors()[resultPosition] ||
        rhsLayout.getLaneFactors()[position] !=
            resultLayout.getLaneFactors()[resultPosition] ||
        rhsLayout.getReplicaFactors()[position] !=
            resultLayout.getReplicaFactors()[resultPosition] ||
        rhsLayout.getFragmentFactors()[position] !=
            resultLayout.getFragmentFactors()[resultPosition] ||
        rhsLayout.getLocalFactors()[position] !=
            resultLayout.getLocalFactors()[resultPosition])
      return false;
  }
  for (int64_t axis : resultLayout.getAxisIds().asArrayRef())
    if (!llvm::is_contained(lhsType.getAxisIds().asArrayRef(), axis) &&
        !llvm::is_contained(rhsType.getAxisIds().asArrayRef(), axis))
      return false;
  return result && loadLayout.getAxisIds() == lhsType.getAxisIds() &&
         partialLayout.getAxisIds() == lhsType.getAxisIds() &&
         sameCoordinates(loadLayout, partialLayout) &&
         loadLayout.getSew() == 8 && partialLayout.getSew() == 16 &&
         partialLayout.getLmulEighths() ==
             2 * loadLayout.getLmulEighths() &&
         loadLayout.getVl() == resultLayout.getVl();
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

int64_t derivedInterleaveRows(mlir::Operation *operation,
                              weft::kernel::EncodingType encoding) {
  if (!encoding || encoding.getKind() != "derived_instance")
    return 0;
  int64_t rows = 0;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](DerivedEncodingOp derived) {
      if (derived.getResultFamily() == encoding.getFamily() &&
          derived.getLayoutIdentity() == encoding.getLayoutIdentity() &&
          derived.getParameterValues() == encoding.getParameters().asArrayRef())
        rows = derived.getInterleaveRows();
    });
  return rows;
}

int64_t localPackInterleaveRows(mlir::Value owner) {
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (owner) {
    mlir::Operation *definition = owner.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      return 0;
    if (auto pack = mlir::dyn_cast<EncodedLocalPackOp>(definition))
      return pack.getPlan().getInterleaveRows();
    if (auto extract = mlir::dyn_cast<ExtractOp>(definition)) {
      owner = extract.getInput();
      continue;
    }
    if (auto convert = mlir::dyn_cast<ConvertLayoutOp>(definition)) {
      owner = convert.getInput();
      continue;
    }
    if (auto materialize = mlir::dyn_cast<RegisterMaterializeOp>(definition)) {
      owner = materialize.getInput();
      continue;
    }
    return 0;
  }
  return 0;
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

mlir::LogicalResult PartialTopologyAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef kind, int64_t rootOperand,
    mlir::DenseI64ArrayAttr partialAxes, mlir::DenseI64ArrayAttr outputAxes,
    int64_t sourceSlots, int64_t partialSlots, int64_t outputReplicas,
    int64_t laneSplit, int64_t combineArity,
    mlir::DenseI64ArrayAttr slotOrder, int64_t resourceGroups) {
  if (kind != "unassigned" && kind != "sequential_fused" &&
      kind != "sequential_per_stream" && kind != "independent" &&
      kind != "replica_reduced" &&
      kind != "scaled" && kind != "reduced_scaled" &&
      kind != "level_scaled" &&
      kind != "merged" && kind != "layered")
    return emitError() << "unknown partial topology kind";
  if (kind == "unassigned") {
    if (rootOperand != -1 || !partialAxes.empty() || !outputAxes.empty() ||
        sourceSlots != 0 || partialSlots != 0 || outputReplicas != 0 ||
        laneSplit != 0 || combineArity != 0 || !slotOrder.empty() ||
        resourceGroups != 0)
      return emitError()
             << "unassigned partial topology cannot carry physical decisions";
    return mlir::success();
  }
  if (rootOperand < -1 || sourceSlots <= 0 || partialSlots <= 0 ||
      outputReplicas <= 0 || laneSplit <= 0 || combineArity <= 0 ||
      resourceGroups <= 0 ||
      slotOrder.size() != static_cast<size_t>(partialSlots))
    return emitError() << "selected partial topology has incomplete geometry";
  llvm::SmallVector<int64_t> ordered(slotOrder.asArrayRef());
  llvm::sort(ordered);
  for (auto [index, slot] : llvm::enumerate(ordered))
    if (slot != static_cast<int64_t>(index))
      return emitError() << "partial topology slot order is not a permutation";
  llvm::SmallVector<int64_t> uniquePartialAxes(partialAxes.asArrayRef());
  llvm::SmallVector<int64_t> uniqueOutputAxes(outputAxes.asArrayRef());
  llvm::sort(uniquePartialAxes);
  llvm::sort(uniqueOutputAxes);
  if (std::adjacent_find(uniquePartialAxes.begin(), uniquePartialAxes.end()) !=
          uniquePartialAxes.end() ||
      std::adjacent_find(uniqueOutputAxes.begin(), uniqueOutputAxes.end()) !=
          uniqueOutputAxes.end())
    return emitError() << "partial topology axes must be unique";
  if (llvm::any_of(partialAxes.asArrayRef(), [&](int64_t axis) {
        return axis == 0 || llvm::is_contained(outputAxes.asArrayRef(), axis);
      }) ||
      llvm::any_of(outputAxes.asArrayRef(),
                   [](int64_t axis) { return axis == 0; }))
    return emitError()
           << "partial and output axes must be nonzero and disjoint";
  if (laneSplit > 1 &&
      (kind != "scaled" || sourceSlots * laneSplit != partialSlots))
    return emitError()
           << "lane-split topology must close scaled source and partial slots";
  if ((kind == "independent" || kind == "replica_reduced" ||
       kind == "scaled" ||
       kind == "reduced_scaled" || kind == "level_scaled" ||
       kind == "layered") &&
      (combineArity > partialSlots || partialSlots % combineArity))
    return emitError()
           << "partial topology combine arity must divide its partial slots";
  const bool ownsRoot = kind == "layered";
  if (ownsRoot != (rootOperand >= 0))
    return emitError()
           << "only a layered topology may own one root operand";
  return mlir::success();
}

mlir::LogicalResult PackedPlaneMergePlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    int64_t logicalAxis, int64_t logicalElements, int64_t lowBits,
    int64_t highBits, int64_t groupSize, int64_t lowLayerBytes,
    int64_t highLayerBytes, int64_t lowByteOffset, int64_t highByteOffset,
    int64_t insertBit) {
  if (logicalAxis <= 0 || logicalElements <= 0 || logicalElements % 4 ||
      lowBits <= 0 || highBits <= 0 || lowBits >= 8 || highBits >= 8 ||
      groupSize != logicalElements || lowLayerBytes <= 0 ||
      highLayerBytes <= 0 || lowLayerBytes % 4 || highLayerBytes % 4 ||
      groupSize * lowBits != lowLayerBytes * 8 ||
      groupSize * highBits != highLayerBytes * 8 || lowByteOffset < 0 ||
      highByteOffset < 0 || insertBit < lowBits || insertBit + highBits > 8)
    return emitError()
           << "packed plane merge plan requires two byte-parallel sub-byte layers over one four-element-aligned logical group";
  return mlir::success();
}

mlir::LogicalResult PartialLayoutPlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::Type sourceLhsType, mlir::Type sourceRhsType,
    mlir::Type sourceSlotType, mlir::Type splitSourceSlotType,
    mlir::Type partialSlotType, mlir::Type widenedSlotType,
    mlir::Type reducedSlotType, mlir::Type vectorSlotType,
    mlir::Type sourceSetType, mlir::Type repackedSetType,
    mlir::Type partialSetType, mlir::Type scaledSetType,
    mlir::Type directSetType, mlir::Type reducedSetType,
    mlir::Type scaleCombinedSetType, mlir::Type fullScaledSetType,
    mlir::DenseI64ArrayAttr sourceLhsParts,
    mlir::DenseI64ArrayAttr sourceRhsParts,
    mlir::DenseI64ArrayAttr sourceLhsOffsets,
    mlir::DenseI64ArrayAttr sourceRhsOffsets) {
  auto sourceLhs = mlir::dyn_cast<weft::riscv::ValueType>(sourceLhsType);
  auto sourceRhs = mlir::dyn_cast<weft::riscv::ValueType>(sourceRhsType);
  auto source = mlir::dyn_cast<weft::riscv::ValueType>(sourceSlotType);
  auto split = mlir::dyn_cast<weft::riscv::ValueType>(splitSourceSlotType);
  auto partial = mlir::dyn_cast<weft::riscv::ValueType>(partialSlotType);
  auto widened = mlir::dyn_cast<weft::riscv::ValueType>(widenedSlotType);
  auto reduced = mlir::dyn_cast<weft::riscv::ValueType>(reducedSlotType);
  auto vector = mlir::dyn_cast<weft::riscv::ValueType>(vectorSlotType);
  auto sourceSet = mlir::dyn_cast<weft::riscv::PartialSetType>(sourceSetType);
  auto repackedSet =
      mlir::dyn_cast<weft::riscv::PartialSetType>(repackedSetType);
  auto partialSet = mlir::dyn_cast<weft::riscv::PartialSetType>(partialSetType);
  auto scaledSet = mlir::dyn_cast<weft::riscv::PartialSetType>(scaledSetType);
  auto directSet = mlir::dyn_cast<weft::riscv::PartialSetType>(directSetType);
  auto reducedSet = mlir::dyn_cast<weft::riscv::PartialSetType>(reducedSetType);
  auto scaleCombinedSet =
      mlir::dyn_cast<weft::riscv::PartialSetType>(scaleCombinedSetType);
  auto fullScaledSet =
      mlir::dyn_cast<weft::riscv::PartialSetType>(fullScaledSetType);
  auto optionalPhysicalValue = [](mlir::Type type) {
    return mlir::isa<weft::riscv::ValueType, mlir::NoneType>(type);
  };
  auto optionalPartialSet = [](mlir::Type type) {
    return mlir::isa<weft::riscv::PartialSetType, mlir::NoneType>(type);
  };
  if (!partial || !optionalPhysicalValue(sourceLhsType) ||
      !optionalPhysicalValue(sourceRhsType) ||
      !optionalPhysicalValue(sourceSlotType) ||
      !optionalPhysicalValue(splitSourceSlotType) ||
      !optionalPhysicalValue(widenedSlotType) ||
      !optionalPhysicalValue(reducedSlotType) ||
      !optionalPhysicalValue(vectorSlotType) ||
      !optionalPartialSet(sourceSetType) ||
      !optionalPartialSet(repackedSetType) ||
      !optionalPartialSet(partialSetType) ||
      !optionalPartialSet(scaledSetType) ||
      !optionalPartialSet(directSetType) ||
      !optionalPartialSet(reducedSetType) ||
      !optionalPartialSet(scaleCombinedSetType) ||
      !optionalPartialSet(fullScaledSetType))
    return emitError()
           << "partial layout plan requires one concrete partial slot and typed optional slots";
  if (partial.getLayout().getCarrier() != "rvv" ||
      (widened && widened.getLayout().getCarrier() != "rvv") ||
      (reduced && reduced.getLayout().getCarrier() != "rvv") ||
      (vector && vector.getLayout().getCarrier() != "rvv"))
    return emitError() << "partial layout plan values must use the RVV carrier";
  const bool hasSourcePlan = static_cast<bool>(source) || static_cast<bool>(split) ||
                             static_cast<bool>(sourceLhs) ||
                             static_cast<bool>(sourceRhs);
  if (hasSourcePlan != (static_cast<bool>(source) && static_cast<bool>(split) &&
                        static_cast<bool>(sourceLhs) &&
                        static_cast<bool>(sourceRhs) && sourceSet &&
                        repackedSet && reducedSet && scaleCombinedSet) ||
      (source && (sourceLhs.getLayout().getCarrier() != "rvv" ||
                  sourceRhs.getLayout().getCarrier() != "rvv" ||
                  source.getLayout().getCarrier() != "rvv" ||
                  split.getLayout().getCarrier() != "rvv")))
    return emitError() << "partial source layout plan is incomplete";
  if (!hasSourcePlan &&
      (sourceSet || repackedSet ||
       !sourceLhsParts.empty() || !sourceRhsParts.empty() ||
       !sourceLhsOffsets.empty() || !sourceRhsOffsets.empty()))
    return emitError()
           << "partial layout plan without source coalescing cannot carry source maps";
  if (hasSourcePlan &&
      (sourceLhsParts.empty() ||
       sourceLhsParts.size() != sourceRhsParts.size() ||
       sourceLhsParts.size() != sourceLhsOffsets.size() ||
       sourceLhsParts.size() != sourceRhsOffsets.size() ||
       llvm::any_of(sourceLhsParts.asArrayRef(),
                    [](int64_t value) { return value < 0; }) ||
       llvm::any_of(sourceRhsParts.asArrayRef(),
                    [](int64_t value) { return value < 0; }) ||
       llvm::any_of(sourceLhsOffsets.asArrayRef(),
                    [](int64_t value) { return value < 0; }) ||
       llvm::any_of(sourceRhsOffsets.asArrayRef(),
                    [](int64_t value) { return value < 0; })))
    return emitError()
           << "partial layout plan requires one complete non-negative source map";
  if (hasSourcePlan &&
      (sourceSet.getPartialType() != source ||
       repackedSet.getPartialType() != partial ||
       reducedSet.getPartialType() != reduced ||
       scaleCombinedSet.getPartialType() != reduced ||
       sourceSet.getReductionAxis() != repackedSet.getReductionAxis() ||
       sourceSet.getReductionAxis() != reducedSet.getReductionAxis() ||
       sourceSet.getReductionAxis() != scaleCombinedSet.getReductionAxis() ||
       sourceSet.getSlots() <= 0 || repackedSet.getSlots() <= 0 ||
       repackedSet.getSlots() % sourceSet.getSlots() ||
       reducedSet.getSlots() != repackedSet.getSlots() ||
       scaleCombinedSet.getSlots() != sourceSet.getSlots()))
    return emitError()
           << "partial source layout plan has inconsistent typed set geometry";
  auto partialMatches = [](weft::riscv::PartialSetType set,
                           weft::riscv::ValueType value) {
    return !set || (value && set.getPartialType() == value &&
                    set.getSlots() > 0 && set.getTermsPerSlot() > 0 &&
                    set.getResourceGroups() > 0);
  };
  if (!partialMatches(partialSet, partial) ||
      !partialMatches(scaledSet, widened) ||
      !partialMatches(directSet, partial) ||
      !partialMatches(reducedSet, reduced) ||
      !partialMatches(scaleCombinedSet, reduced) ||
      !partialMatches(fullScaledSet, widened))
    return emitError()
           << "partial layout plan contains a set with a mismatched slot type";
  return mlir::success();
}

mlir::LogicalResult LayeredStreamGeometryAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, int64_t axis,
    int64_t groupSize, int64_t layerSize, int64_t laneCount,
    int64_t streamCount, int64_t replicaCount,
    mlir::DenseI64ArrayAttr windowForStream,
    mlir::DenseI64ArrayAttr representativeStreamForWindow,
    mlir::DenseI64ArrayAttr physicalLayerForStream,
    mlir::DenseI64ArrayAttr physicalLayerForLogicalLayer,
    mlir::DenseI64ArrayAttr shiftAmountForLogicalLayer,
    mlir::DenseI64ArrayAttr maskValueForLogicalLayer,
    mlir::DenseI64ArrayAttr shiftAmountForStream,
    mlir::DenseI64ArrayAttr maskValueForStream,
    mlir::DenseI64ArrayAttr groupForWindow,
    mlir::DenseI64ArrayAttr withinForWindow) {
  if (axis <= 0 || groupSize <= 0 || layerSize <= 0 ||
      groupSize % layerSize || laneCount <= 0 || streamCount <= 0 ||
      replicaCount <= 0 ||
      windowForStream.size() != static_cast<size_t>(streamCount) ||
      representativeStreamForWindow.size() != groupForWindow.size() ||
      physicalLayerForStream.size() != static_cast<size_t>(streamCount) ||
      shiftAmountForStream.size() != static_cast<size_t>(streamCount) ||
      maskValueForStream.size() != static_cast<size_t>(streamCount) ||
      groupForWindow.empty() || groupForWindow.size() != withinForWindow.size())
    return emitError() << "layered stream geometry is incomplete";
  const int64_t layers = groupSize / layerSize;
  if (physicalLayerForLogicalLayer.size() != static_cast<size_t>(layers) ||
      shiftAmountForLogicalLayer.size() != static_cast<size_t>(layers) ||
      maskValueForLogicalLayer.size() != static_cast<size_t>(layers))
    return emitError()
           << "layered stream geometry requires one complete decode plan per logical layer";
  llvm::SmallVector<int64_t> logicalLayerMap(
      physicalLayerForLogicalLayer.asArrayRef());
  llvm::sort(logicalLayerMap);
  for (auto [index, layer] : llvm::enumerate(logicalLayerMap))
    if (layer != static_cast<int64_t>(index))
      return emitError()
             << "layered stream logical-to-physical layer map must be a permutation";
  for (auto [layer, shiftAmount, maskValue] :
       llvm::zip(physicalLayerForLogicalLayer.asArrayRef(),
                 shiftAmountForLogicalLayer.asArrayRef(),
                 maskValueForLogicalLayer.asArrayRef()))
    if (shiftAmount < 0 || maskValue < 0 ||
        (layer == 0) != (shiftAmount == 0))
      return emitError()
             << "layered stream logical decode plan disagrees with its physical layer";
  for (int64_t window : windowForStream.asArrayRef())
    if (window < 0 || window >= static_cast<int64_t>(groupForWindow.size()))
      return emitError() << "layered stream references an absent raw window";
  for (auto [window, representative] :
       llvm::enumerate(representativeStreamForWindow.asArrayRef()))
    if (representative < 0 || representative >= streamCount ||
        windowForStream[static_cast<size_t>(representative)] !=
            static_cast<int64_t>(window))
      return emitError()
             << "layered stream raw window lacks an exact representative stream";
  for (int64_t layer : physicalLayerForStream.asArrayRef())
    if (layer < 0 || layer >= layers)
      return emitError() << "layered stream physical layer is out of range";
  for (auto [layer, shiftAmount, maskValue] :
       llvm::zip(physicalLayerForStream.asArrayRef(),
                 shiftAmountForStream.asArrayRef(),
                 maskValueForStream.asArrayRef())) {
    auto logical = llvm::find(physicalLayerForLogicalLayer.asArrayRef(), layer);
    if (logical == physicalLayerForLogicalLayer.asArrayRef().end())
      return emitError()
             << "layered stream physical layer has no logical decode owner";
    const size_t logicalIndex = static_cast<size_t>(
        logical - physicalLayerForLogicalLayer.asArrayRef().begin());
    if (shiftAmount != shiftAmountForLogicalLayer[logicalIndex] ||
        maskValue != maskValueForLogicalLayer[logicalIndex])
      return emitError()
             << "layered stream decode plan disagrees with its logical layer";
  }
  for (int64_t within : withinForWindow.asArrayRef())
    if (within < 0 || within + laneCount > layerSize)
      return emitError() << "layered raw window crosses a storage layer";
  return mlir::success();
}

mlir::LogicalResult StorageWindowPlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef kind, int64_t reductionAxis, int64_t projectionBase,
    int64_t projectionStride, int64_t projectionRepeat,
    int64_t projectionExtent, int64_t offsetAlignment,
    int64_t recordElements, int64_t byteOffset, int64_t elementBits,
    int64_t groupSize, int64_t layerSize, int64_t physicalLayerBase,
    int64_t physicalLayerStep, int64_t shiftBase, int64_t shiftStep,
    int64_t maskValue) {
  if (reductionAxis <= 0 || projectionBase < 0 || projectionStride <= 0 ||
      projectionRepeat <= 0 || projectionExtent <= 0 || offsetAlignment <= 0 ||
      recordElements <= 0 || byteOffset < 0 || elementBits <= 0)
    return emitError() << "storage window plan has incomplete logical or storage geometry";
  if (kind == "unit" || kind == "strided" || kind == "repeat") {
    const bool exactKind =
        (kind == "unit" && projectionRepeat == 1 && projectionStride == 1) ||
        (kind == "strided" && projectionRepeat == 1 && projectionStride > 1) ||
        (kind == "repeat" && projectionRepeat > 1);
    if (!exactKind || groupSize != 0 || layerSize != 0 ||
        physicalLayerBase != 0 || physicalLayerStep != 0 || shiftBase != 0 ||
        shiftStep != 0 || maskValue != 0)
      return emitError()
             << "natural storage window plan disagrees with its selected load form";
    return mlir::success();
  }
  if (kind == "interleaved_natural" || kind == "interleaved_joined") {
    if (projectionBase != 0 || projectionStride != 1 ||
        projectionRepeat != 1 || groupSize != 0 || layerSize != 0 ||
        physicalLayerBase != 0 || physicalLayerStep != 0 || shiftBase != 0 ||
        shiftStep != 0 || maskValue != 0)
      return emitError()
             << "interleaved record projection disagrees with its selected load form";
    return mlir::success();
  }
  if (kind != "layered")
    return emitError() << "unknown storage window implementation kind";
  const int64_t layers =
      layerSize > 0 && groupSize > 0 ? groupSize / layerSize : 0;
  const bool exactLayerMap =
      layers > 1 &&
      ((physicalLayerBase == 0 && physicalLayerStep == 1) ||
       (physicalLayerBase == layers - 1 && physicalLayerStep == -1));
  if (projectionBase != 0 || projectionStride != 1 ||
      projectionRepeat != 1 || groupSize <= 0 || layerSize <= 0 ||
      groupSize % layerSize || elementBits >= 8 || !exactLayerMap ||
      shiftBase != physicalLayerBase * elementBits ||
      shiftStep != physicalLayerStep * elementBits ||
      maskValue != ((int64_t{1} << elementBits) - 1))
    return emitError()
           << "layered storage window plan has incomplete address or decode geometry";
  return mlir::success();
}

mlir::LogicalResult LocalPackPlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef kind, int64_t rowAxis, int64_t recordAxis,
    int64_t interleaveRows, int64_t recordBytes, int64_t recordElements,
    int64_t localBytes) {
  auto interleavedRecordBytes =
      checkedPositiveProduct({interleaveRows, recordBytes});
  if (kind != "interleave" || rowAxis <= 0 || recordAxis <= 0 ||
      rowAxis == recordAxis || interleaveRows <= 1 || recordBytes <= 0 ||
      recordElements <= 0 || localBytes <= 0 ||
      !interleavedRecordBytes || localBytes % *interleavedRecordBytes != 0)
    return emitError()
           << "local pack plan requires one closed record-interleave geometry";
  return mlir::success();
}

mlir::LogicalResult LayeredPartialPlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    int64_t rootIndex, mlir::Type decodedWindowType,
    mlir::Type storageWindowType, mlir::Type accumulatorType,
    mlir::ArrayAttr rootWindowTypes, mlir::ArrayAttr rootStoragePlans,
    LayeredStreamGeometryAttr geometry) {
  auto decoded =
      mlir::dyn_cast<weft::riscv::ValueType>(decodedWindowType);
  auto storage =
      mlir::dyn_cast<weft::riscv::LayeredWindowType>(storageWindowType);
  auto accumulator =
      mlir::dyn_cast<weft::riscv::ValueType>(accumulatorType);
  if (rootIndex < 0 || rootIndex >= static_cast<int64_t>(rootWindowTypes.size()) ||
      !decoded || !storage || !accumulator || !geometry ||
      rootWindowTypes.empty() || rootStoragePlans.size() != rootWindowTypes.size() ||
      decoded.getLayout().getCarrier() != "rvv" ||
      accumulator.getLayout().getCarrier() != "rvv" ||
      storage.getResultType() != decoded ||
      storage.getReductionAxis() != geometry.getAxis() ||
      storage.getLayers() != geometry.getGroupSize() / geometry.getLayerSize() ||
      storage.getWindowsPerLayer() * geometry.getLaneCount() !=
          geometry.getLayerSize())
    return emitError()
           << "layered partial plan requires one closed typed storage, decode, accumulator, and root plan";
  for (size_t index = 0; index < rootWindowTypes.size(); ++index) {
    mlir::Attribute typeValue = rootWindowTypes[index];
    mlir::Attribute planValue = rootStoragePlans[index];
    auto typeAttribute = mlir::dyn_cast<mlir::TypeAttr>(typeValue);
    auto type = typeAttribute
                    ? mlir::dyn_cast<weft::riscv::ValueType>(
                          typeAttribute.getValue())
                    : weft::riscv::ValueType();
    auto plan = mlir::dyn_cast<weft::riscv::StorageWindowPlanAttr>(planValue);
    if (!type || type.getLayout().getCarrier() != "rvv" || !plan ||
        plan.getReductionAxis() != geometry.getAxis() ||
        (static_cast<int64_t>(index) == rootIndex &&
         (plan.getKind() != "layered" ||
          plan.getGroupSize() != geometry.getGroupSize() ||
          plan.getLayerSize() != geometry.getLayerSize())))
      return emitError()
             << "layered partial root requires one typed RVV window and storage plan";
  }
  return mlir::success();
}

mlir::LogicalResult GroupedMacPlanAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    llvm::StringRef kind, int64_t group, int64_t unroll,
    int64_t reductionAxis, int64_t storageGroup, int64_t storageLayer,
    int64_t logicalWidth, int64_t lhsBitOffsetBytes,
    int64_t rhsBitOffsetBytes, int64_t recordElements,
    int64_t interleaveRows, llvm::StringRef loadForm,
    int64_t rowStrideAxis, int64_t termByteStride,
    int64_t physicalLayerBase, int64_t physicalLayerStep,
    int64_t shiftBase, int64_t shiftStep, int64_t maskValue,
    mlir::DenseI64ArrayAttr termOrder) {
  const int64_t layers =
      storageLayer > 0 && storageGroup > 0 ? storageGroup / storageLayer : 0;
  const bool exactLayerMap =
      layers > 0 &&
      ((physicalLayerBase == 0 && physicalLayerStep == 1) ||
       (physicalLayerBase == layers - 1 && physicalLayerStep == -1));
  if ((kind != "reduce" && kind != "compact" && kind != "fragmented") ||
      group <= 0 || unroll <= 0 || reductionAxis <= 0 || storageGroup <= 0 ||
      storageLayer <= 0 || storageGroup % storageLayer ||
      storageLayer % group || logicalWidth <= 0 || logicalWidth >= 8 ||
      lhsBitOffsetBytes < 0 || rhsBitOffsetBytes < 0 || recordElements <= 0 ||
      interleaveRows < 0 || termByteStride <= 0 || !exactLayerMap ||
      shiftBase != physicalLayerBase * logicalWidth ||
      shiftStep != physicalLayerStep * logicalWidth ||
      (loadForm != "unit" && loadForm != "strided") ||
      (loadForm == "unit" &&
       (interleaveRows <= 0 || rowStrideAxis != 0 ||
        termByteStride != interleaveRows)) ||
      (loadForm == "strided" &&
       (interleaveRows != 0 || rowStrideAxis <= 0 || termByteStride != 1)) ||
      maskValue != ((int64_t{1} << logicalWidth) - 1))
    return emitError() << "grouped MAC plan has incomplete typed storage geometry";
  auto terms = checkedPositiveProduct({group, unroll});
  if (!terms || termOrder.size() != static_cast<size_t>(*terms))
    return emitError() << "grouped MAC plan has no complete term order";
  llvm::SmallVector<int64_t> sorted(termOrder.asArrayRef());
  llvm::sort(sorted);
  for (auto [index, term] : llvm::enumerate(sorted))
    if (term != static_cast<int64_t>(index))
      return emitError() << "grouped MAC term order must be a permutation";
  for (int64_t slot = 0; slot < unroll; ++slot) {
    llvm::SmallVector<int64_t> slotTerms;
    slotTerms.reserve(static_cast<size_t>(group));
    for (int64_t term = 0; term < group; ++term)
      slotTerms.push_back(termOrder[static_cast<size_t>(slot * group + term)]);
    llvm::sort(slotTerms);
    for (int64_t term = 0; term < group; ++term)
      if (slotTerms[static_cast<size_t>(term)] != slot * group + term)
        return emitError()
               << "grouped MAC term order must preserve each unrolled slot";
  }
  if (kind == "compact" && storageLayer % *terms)
    return emitError()
           << "compact grouped MAC plan crosses a selected storage layer";
  if (kind == "fragmented" && storageLayer % *terms == 0)
    return emitError()
           << "fragmented grouped MAC plan duplicates one compact storage window";
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
  // A reduction-operand projection may eliminate its final logical axis while
  // still requiring an explicit physical carrier (for example, one scalar
  // register tuple selected from RVV lanes).  Such a value has rank zero in
  // the logical program, but its LayoutAttr remains meaningful and must round
  // trip through textual physical IR.  Operations that require a shaped
  // domain verify positive rank locally.
  if (failed(
          verifyShape(emitError, shape.asArrayRef(), axisIds.asArrayRef(), true)))
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
    bool fullyStaticShape = true;
    for (int64_t extent : shape.asArrayRef()) {
      if (extent < 0) {
        fullyStaticShape = false;
        continue;
      }
      if (logicalElements > std::numeric_limits<int64_t>::max() / extent)
        return emitError() << "local storage logical shape overflows";
      logicalElements *= extent;
    }
    if (fullyStaticShape) {
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

mlir::LogicalResult PartialSetType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    ValueType partialType, int64_t reductionAxis, int64_t slots,
    int64_t termsPerSlot, int64_t resourceGroups) {
  if (!partialType || reductionAxis <= 0 || slots <= 0 || termsPerSlot <= 0 ||
      resourceGroups <= 0 || partialType.getLayout().getCarrier() != "rvv")
    return emitError()
           << "partial set requires one RVV partial type and positive topology";
  auto axis = llvm::find(partialType.getAxisIds().asArrayRef(), reductionAxis);
  if (axis == partialType.getAxisIds().asArrayRef().end())
    return emitError() << "partial set type does not retain its reduction axis";
  const size_t position = static_cast<size_t>(
      axis - partialType.getAxisIds().asArrayRef().begin());
  if (partialType.getLayout().getTimeFactors()[position] != 1 ||
      partialType.getLayout().getLaneFactors()[position] !=
          partialType.getShape()[position] ||
      partialType.getLayout().getReplicaFactors()[position] != 1 ||
      partialType.getLayout().getFragmentFactors()[position] != 1 ||
      partialType.getLayout().getLocalFactors()[position] != 1)
    return emitError()
           << "partial set reduction axis must be one complete lane partial";
  for (size_t other = 0; other < partialType.getShape().size(); ++other) {
    if (other == position)
      continue;
    const int64_t lanes = partialType.getLayout().getLaneFactors()[other];
    const int64_t replicas =
        partialType.getLayout().getReplicaFactors()[other];
    if (partialType.getLayout().getTimeFactors()[other] != 1 || lanes <= 0 ||
        replicas <= 0 ||
        (partialType.getShape()[other] > 0 &&
         lanes * replicas != partialType.getShape()[other]) ||
        partialType.getLayout().getFragmentFactors()[other] != 1 ||
        partialType.getLayout().getLocalFactors()[other] != 1)
      return emitError()
             << "partial-set free axes must remain exact lane/register coordinates";
  }
  if (partialType.getLayout().getRegisterGroups() <= 0 ||
      slots > std::numeric_limits<int64_t>::max() /
                  partialType.getLayout().getRegisterGroups() ||
      resourceGroups !=
          slots * partialType.getLayout().getRegisterGroups())
    return emitError()
           << "partial set resource groups disagree with its independent slots";
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
  if (getLocalPackPlan() && getPlacement() != "local")
    return emitOpError("a selected local pack plan requires local placement");
  if (getLocalPackPlan() &&
      !mlir::isa<kernel::EncodingType>(elementOf(getResult().getType())))
    return emitOpError("a local encoded pack plan requires an encoded value");
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
    auto resultTime = result ? checkedPositiveProduct(
                                   result.getLayout().getTimeFactors().asArrayRef())
                             : std::optional<int64_t>();
    auto resultReplicas =
        result ? checkedPositiveProduct(
                     result.getLayout().getReplicaFactors().asArrayRef())
               : std::optional<int64_t>();
    auto indexTime = indices ? checkedPositiveProduct(
                                   indices.getLayout().getTimeFactors().asArrayRef())
                             : std::optional<int64_t>();
    auto indexReplicas =
        indices ? checkedPositiveProduct(
                      indices.getLayout().getReplicaFactors().asArrayRef())
                : std::optional<int64_t>();
    if (result && indices && input.getLayout().getCarrier() == "rvv" &&
        result.getLayout().getCarrier() == "scalar" &&
        indices.getLayout().getCarrier() == "scalar") {
      auto sourceLanes = checkedPositiveProduct(
          input.getLayout().getLaneFactors().asArrayRef());
      auto resultLanes = checkedPositiveProduct(
          result.getLayout().getLaneFactors().asArrayRef());
      auto indexLanes = checkedPositiveProduct(
          indices.getLayout().getLaneFactors().asArrayRef());
      auto laneCount = getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "lane_gather_count");
      auto sourceParts =
          getOperation()->getAttrOfType<mlir::DenseI64ArrayAttr>(
              "lane_gather_source_parts");
      auto indexParts =
          getOperation()->getAttrOfType<mlir::DenseI64ArrayAttr>(
              "lane_gather_index_parts");
      if (gatherCount != 1 || !time || *time != 1 || !sourceLanes ||
          *sourceLanes <= 1 || !replicas || *replicas <= 0 || !resultTime ||
          *resultTime != 1 || !resultLanes || *resultLanes != 1 ||
          !resultReplicas || *resultReplicas <= 0 || !indexTime ||
          *indexTime != 1 || !indexLanes || *indexLanes != 1 ||
          !indexReplicas || *indexReplicas <= 0 || !laneCount ||
          laneCount.getInt() != *sourceLanes || !sourceParts || !indexParts ||
          sourceParts.size() != static_cast<size_t>(*resultReplicas) ||
          indexParts.size() != static_cast<size_t>(*resultReplicas) ||
          !exactLeaf(getLeaf(), "rvv", "extract",
                     "rvv.extract.lane-to-replica", "none", "exact"))
        return emitOpError(
            "lane-to-replica extract requires one complete RVV source and a typed source/index map per result replica");
      for (auto [sourcePart, indexPart] :
           llvm::zip(sourceParts.asArrayRef(), indexParts.asArrayRef()))
        if (sourcePart < 0 || sourcePart >= *replicas || indexPart < 0 ||
            indexPart >= *indexReplicas)
          return emitOpError(
              "lane-to-replica extract references an absent source or index part");
      return verifyLeafOperation(*this);
    }
    if (result && indices && input.getLayout().getCarrier() == "scalar" &&
        result.getLayout().getCarrier() == "scalar" &&
        indices.getLayout().getCarrier() == "scalar") {
      auto arity = getOperation()->getAttrOfType<mlir::IntegerAttr>(
          "replica_gather_arity");
      auto candidates = getOperation()->getAttrOfType<mlir::DenseI64ArrayAttr>(
          "replica_gather_candidates");
      auto keys = getOperation()->getAttrOfType<mlir::DenseI64ArrayAttr>(
          "replica_gather_keys");
      if (gatherCount != 1 || !time || *time != 1 || !resultTime ||
          *resultTime != 1 || !indexTime || *indexTime != 1 || !replicas ||
          !resultReplicas || !indexReplicas || *resultReplicas <= 1 ||
          *indexReplicas <= 1 || *resultReplicas % *indexReplicas || !arity ||
          arity.getInt() <= 0 ||
          !candidates ||
          candidates.size() !=
              static_cast<size_t>(*resultReplicas * arity.getInt()) ||
          !keys || keys.size() != candidates.size() ||
          !exactLeaf(getLeaf(), "scalar", "extract",
                     "scalar.replica-gather", "none", "exact"))
        return emitOpError(
            "scalar register extract requires one typed replica candidate matrix and exact gather leaf");
      for (int64_t resultPart = 0; resultPart < *resultReplicas; ++resultPart) {
        llvm::DenseSet<int64_t> row;
        llvm::DenseSet<int64_t> keyRow;
        for (int64_t candidate = 0; candidate < arity.getInt(); ++candidate) {
          const int64_t offset = resultPart * arity.getInt() + candidate;
          int64_t source = candidates[offset];
          int64_t key = keys[offset];
          if (source < 0 || source >= *replicas || !row.insert(source).second ||
              key < 0 || !keyRow.insert(key).second)
            return emitOpError(
                "scalar register extract rows require distinct non-negative keys and distinct in-range source replicas");
        }
      }
      return verifyLeafOperation(*this);
    }
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
        return emitOpError()
               << "physical reduce must preserve every free-axis representation; input="
               << input << ", result=" << result
               << ", free_axis=" << axis;
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
          return emitOpError()
                 << "physical lookup indices/result require an explicit layout conversion; indices="
                 << indices << ", result=" << result;
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

mlir::LogicalResult RVVIndexedEntryLoadOp::verify() {
  auto entry = getEntryIndices().getType();
  auto result = getResult().getType();
  auto indexElement = mlir::dyn_cast<mlir::IntegerType>(entry.getElementType());
  const bool scalarEntries = entry.getLayout().getCarrier() == "scalar";
  const bool vectorEntries = entry.getLayout().getCarrier() == "rvv";
  if ((!entry.getElementType().isIndex() &&
       (!indexElement || indexElement.isSigned())) ||
      (!scalarEntries && !vectorEntries) ||
      result.getLayout().getCarrier() != "rvv")
    return emitOpError(
        "indexed entry load requires unsigned scalar or RVV entry indices and an RVV result");
  if (getPayloadExtent() <= 0 || getEntryStride() < getPayloadExtent())
    return emitOpError(
        "indexed entry load requires one positive contiguous payload within each entry stride");

  llvm::SmallVector<int64_t> sourceAxes;
  llvm::SmallVector<int64_t> sourceShape;
  if (auto descriptor = mlir::dyn_cast<MemDescType>(getSource().getType())) {
    auto encoding =
        mlir::cast<weft::kernel::EncodingType>(descriptor.getEncoding());
    if (encoding.getKind() != "dense" || descriptor.getShape().size() != 1 ||
        descriptor.getStorageBits() != elementBitWidth(result.getElementType()))
      return emitOpError(
          "indexed entry load requires a dense descriptor matching the payload element width");
    sourceAxes.append(descriptor.getAxisIds().asArrayRef().begin(),
                      descriptor.getAxisIds().asArrayRef().end());
    sourceShape.append(descriptor.getShape().asArrayRef().begin(),
                       descriptor.getShape().asArrayRef().end());
  } else if (auto value = mlir::dyn_cast<ValueType>(getSource().getType())) {
    if (value.getLayout().getCarrier() != "local" ||
        value.getElementType() != result.getElementType())
      return emitOpError(
          "indexed entry value source must be one local payload with matching element type");
    sourceAxes.append(value.getAxisIds().asArrayRef().begin(),
                      value.getAxisIds().asArrayRef().end());
    sourceShape.append(value.getShape().asArrayRef().begin(),
                       value.getShape().asArrayRef().end());
  } else {
    return emitOpError(
        "indexed entry load source must be a dense descriptor or local value");
  }
  auto sourceAxis = llvm::find(sourceAxes, getSourceAxis());
  if (sourceAxis == sourceAxes.end())
    return emitOpError("indexed entry load source axis is absent from its source");
  const size_t sourcePosition =
      static_cast<size_t>(sourceAxis - sourceAxes.begin());

  auto entryAxes = entry.getAxisIds().asArrayRef();
  auto entryShape = entry.getShape().asArrayRef();
  auto resultAxes = result.getAxisIds().asArrayRef();
  auto resultShape = result.getShape().asArrayRef();
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  for (size_t position = 0; position < sourceAxes.size(); ++position) {
    if (position == sourcePosition)
      continue;
    expectedAxes.push_back(sourceAxes[position]);
    expectedShape.push_back(sourceShape[position]);
  }
  const size_t retainedSourceAxes = expectedAxes.size();
  for (auto [axis, extent] : llvm::zip(entryAxes, entryShape)) {
    if (llvm::is_contained(expectedAxes, axis) || axis == getPayloadAxis())
      return emitOpError(
          "indexed entry axes must be disjoint from retained source and payload axes");
    expectedAxes.push_back(axis);
    expectedShape.push_back(extent);
  }
  expectedAxes.push_back(getPayloadAxis());
  expectedShape.push_back(getPayloadExtent());
  if (!llvm::equal(expectedAxes, resultAxes) ||
      !llvm::equal(expectedShape, resultShape))
    return emitOpError(
        "indexed entry load result must retain source axes and append entry plus payload axes");
  for (size_t position = 0; position < entryAxes.size(); ++position)
    if (!sameAxisMapping(entry, position, result,
                         retainedSourceAxes + position))
      return emitOpError(
          "indexed entry load must preserve every outer entry-axis representation");

  const size_t payloadPosition = resultAxes.size() - 1;
  auto layout = result.getLayout();
  const unsigned payloadBits = elementBitWidth(result.getElementType());
  if (layout.getTimeFactors()[payloadPosition] != 1 ||
      layout.getLaneFactors()[payloadPosition] != getPayloadExtent() ||
      layout.getReplicaFactors()[payloadPosition] != 1 ||
      layout.getFragmentFactors()[payloadPosition] != 1 ||
      layout.getLocalFactors()[payloadPosition] != 1)
    return emitOpError(
        "indexed entry payload axis must be one complete RVV lane dimension");

  if (getAccess().getMapping() != "natural")
    return emitOpError(
        "indexed entry load requires one natural entry-payload mapping");
  if (scalarEntries) {
    if (getAccess().getForm() != "unit" ||
        !exactLeaf(getLeaf(), "rvv", "indexed-entry-load",
                   "rvv.indexed-entry-load", "none", "exact"))
      return emitOpError(
          "scalar indexed entries require one exact unit-payload RVV leaf");
  } else {
    if (!indexElement ||
        (indexElement.getWidth() != 16 && indexElement.getWidth() != 32 &&
         indexElement.getWidth() != 64) ||
        payloadBits == 0 ||
        payloadBits * static_cast<uint64_t>(getPayloadExtent()) != 64 ||
        getEntryStride() != getPayloadExtent() ||
        getAccess().getForm() != "indexed" ||
        !exactLeaf(getLeaf(), "rvv", "indexed-entry-gather",
                   "rvv.indexed-entry-gather", "none", "exact"))
      return emitOpError(
          "RVV indexed entries require one exact contiguous-entry gather leaf")
             << "; entry_stride=" << getEntryStride()
             << ", payload_extent=" << getPayloadExtent()
             << ", index_type=" << entry << ", result_type=" << result;
  }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RVVUnitEntryWindowLoadOp::verify() {
  ValueType source = getSource().getType();
  ValueType result = getResult().getType();
  auto field = getSource().getDefiningOp<FieldOp>();
  auto baseValue = mlir::dyn_cast<ValueType>(getEntryBase().getType());
  mlir::Type baseElement = baseValue ? baseValue.getElementType()
                                     : getEntryBase().getType();
  auto baseInteger = mlir::dyn_cast<mlir::IntegerType>(baseElement);
  bool scalarBase = baseElement.isIndex() ||
                    (baseInteger && !baseInteger.isSigned());
  if (baseValue) {
    auto layout = baseValue.getLayout();
    scalarBase = scalarBase && layout.getCarrier() == "scalar" &&
                 llvm::all_of(layout.getTimeFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getLaneFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getReplicaFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getFragmentFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getLocalFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; });
  }
  if (!field || source.getLayout().getCarrier() != "local" || !scalarBase ||
      result.getLayout().getCarrier() != "rvv" ||
      source.getElementType() != result.getElementType() ||
      getEntryAxes().empty() ||
      getEntryAxes().size() != getEntryExtents().size() ||
      llvm::any_of(getEntryExtents(),
                   [](int64_t extent) { return extent <= 0; }) ||
      getPayloadExtent() <= 0 ||
      getEntryStride() != getPayloadExtent() ||
      llvm::is_contained(getEntryAxes(), getPayloadAxis()) ||
      getAccess().getForm() != "unit" ||
      getAccess().getMapping() != "natural" ||
      !exactLeaf(getLeaf(), "rvv", "unit-entry-window-load",
                 "rvv.unit-entry-window-load", "none", "exact"))
    return emitOpError(
        "unit entry window load requires one byte-contiguous typed local field, scalar base, and exact RVV leaf");

  llvm::DenseSet<int64_t> entryAxes;
  for (int64_t axis : getEntryAxes())
    if (!entryAxes.insert(axis).second)
      return emitOpError("unit entry window entry axes must be unique");

  auto sourceAxes = source.getAxisIds().asArrayRef();
  auto sourceShape = source.getShape().asArrayRef();
  auto sourceAxis = llvm::find(sourceAxes, getSourceAxis());
  if (sourceAxis == sourceAxes.end() ||
      llvm::any_of(getEntryAxes(), [&](int64_t axis) {
        return llvm::is_contained(sourceAxes, axis);
      }) ||
      llvm::is_contained(sourceAxes, getPayloadAxis()))
    return emitOpError(
        "unit entry window source, entry, and payload axes must be disjoint");
  const size_t sourcePosition =
      static_cast<size_t>(sourceAxis - sourceAxes.begin());
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  for (size_t position = 0; position < sourceAxes.size(); ++position) {
    if (position == sourcePosition)
      continue;
    expectedAxes.push_back(sourceAxes[position]);
    expectedShape.push_back(sourceShape[position]);
  }
  const size_t retainedAxes = expectedAxes.size();
  expectedAxes.append(getEntryAxes().begin(), getEntryAxes().end());
  expectedShape.append(getEntryExtents().begin(), getEntryExtents().end());
  expectedAxes.push_back(getPayloadAxis());
  expectedShape.push_back(getPayloadExtent());
  if (!llvm::equal(expectedAxes, result.getAxisIds().asArrayRef()) ||
      !llvm::equal(expectedShape, result.getShape().asArrayRef()))
    return emitOpError(
        "unit entry window result must retain source axes and append entry plus payload axes");
  auto layout = result.getLayout();
  for (size_t position = 0; position < retainedAxes; ++position)
    if (layout.getTimeFactors()[position] != 1 ||
        layout.getLaneFactors()[position] != 1 ||
        layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1)
      return emitOpError(
          "unit entry window retained axes must remain register coordinates");
  llvm::SmallVector<int64_t> windowExtents(getEntryExtents().begin(),
                                           getEntryExtents().end());
  windowExtents.push_back(getPayloadExtent());
  for (size_t offset = 0; offset < windowExtents.size(); ++offset) {
    const size_t position = retainedAxes + offset;
    const int64_t extent = windowExtents[offset];
    if (layout.getTimeFactors()[position] != 1 ||
        layout.getLaneFactors()[position] != extent ||
        layout.getReplicaFactors()[position] != 1 ||
        layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1)
      return emitOpError(
                 "unit entry and payload axes must form one complete RVV lane window")
             << "; entry_axes=" << getEntryAxes()
             << ", payload_axis=" << getPayloadAxis()
             << ", result_type=" << result;
  }
  return verifyLeafOperation(*this);
}

void ConvertLayoutOp::getEffects(
    llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
        mlir::MemoryEffects::Effect>> &effects) {
  llvm::StringRef effect = getConversion().getEffect();
  if (effect == "read") {
    effects.emplace_back(mlir::MemoryEffects::Read::get());
    return;
  }
  if (effect == "write") {
    effects.emplace_back(mlir::MemoryEffects::Write::get());
    return;
  }
  if (effect == "handoff") {
    effects.emplace_back(mlir::MemoryEffects::Read::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get());
  }
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
      if (sourceLane[dimension] != 1 ||
          sourceFragment[dimension] != 1 || sourceLocal[dimension] != 1 ||
          targetFragment[dimension] != 1 || targetLocal[dimension] != 1)
        return emitOpError(
            "scalar register-to-lane conversion only uses scalar time/replica coordinates and RVV lanes");
    }
    if (!laneDimension)
      return emitOpError(
          "scalar register-to-lane conversion requires one non-unit target lane axis");
    for (size_t dimension = 0; dimension < targetLane.size(); ++dimension) {
      if (dimension == *laneDimension) {
        auto represented = checkedPositiveProduct(
            {targetTime[dimension], targetLane[dimension]});
        if (!represented || sourceTime[dimension] != 1 ||
            sourceReplica[dimension] != *represented ||
            targetReplica[dimension] != 1)
          return emitOpError(
              "scalar register-to-lane conversion must preserve the complete moved axis");
        continue;
      }
      auto sourceRepresented = checkedPositiveProduct(
          {sourceTime[dimension], sourceReplica[dimension]});
      auto targetRepresented = checkedPositiveProduct(
          {targetTime[dimension], targetReplica[dimension]});
      if (targetLane[dimension] != 1 || !sourceRepresented ||
          !targetRepresented || *sourceRepresented != *targetRepresented)
        return emitOpError(
            "scalar register-to-lane conversion must preserve every non-lane logical axis");
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
      auto sourceRepresented = checkedPositiveProduct(
          {sourceTime[dimension], sourceReplica[dimension]});
      auto targetRepresented = checkedPositiveProduct(
          {targetTime[dimension], targetReplica[dimension]});
      if (targetLane[dimension] != 1 || !sourceRepresented ||
          !targetRepresented || *sourceRepresented != *targetRepresented)
        return emitOpError(
            "scalar time-to-lane conversion must preserve every non-lane logical axis");
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
    bool symbolic = false;
    for (int64_t extent : result.getShape().asArrayRef()) {
      if (extent < 0) {
        symbolic = true;
        continue;
      }
      if (extent == 0)
        return emitOpError("static local storage has a zero logical extent");
      expectedElements *= extent;
    }
    if (symbolic) {
      auto physicalElements = checkedPositiveProduct(
          result.getLayout().getLocalFactors().asArrayRef());
      if (!physicalElements)
        return emitOpError(
            "symbolic local binding requires one fixed local-coordinate extent");
      expectedElements = *physicalElements;
    }
    if (!elements || elements.value() != expectedElements)
      return emitOpError(
          "static local binding element count must equal its represented physical shape");
    const unsigned bits = elementBitWidth(result.getElementType());
    if (!bits || bits % 8 ||
        expectedElements >
            std::numeric_limits<int64_t>::max() / (bits / 8) ||
        storage.getSizeBytes() != expectedElements * (bits / 8))
      return emitOpError(
          "static local binding byte size must exactly preserve its physical elements");
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

mlir::LogicalResult RVVLocalMaterializeOp::verify() {
  ValueType input = getInput().getType();
  ValueType destination = getDestination().getType();
  auto localElements = checkedPositiveProduct(
      destination.getLayout().getLocalFactors().asArrayRef());
  const unsigned bits = elementBitWidth(input.getElementType());
  int64_t laneDimensions = 0;
  bool mappingClosed = input.getShape().size() == destination.getShape().size();
  for (size_t dimension = 0; mappingClosed && dimension < input.getShape().size();
       ++dimension) {
    const int64_t lane = input.getLayout().getLaneFactors()[dimension];
    const int64_t replica = input.getLayout().getReplicaFactors()[dimension];
    const int64_t local = destination.getLayout().getLocalFactors()[dimension];
    if (lane > 1) {
      ++laneDimensions;
      mappingClosed &= dimension + 1 == input.getShape().size() && lane == local &&
                       replica == 1;
    } else {
      mappingClosed &= replica == local;
    }
    mappingClosed &= input.getLayout().getTimeFactors()[dimension] == 1 &&
                     input.getLayout().getFragmentFactors()[dimension] == 1 &&
                     input.getLayout().getLocalFactors()[dimension] == 1;
  }
  const int64_t localBytes =
      localElements && bits
          ? (*localElements * static_cast<int64_t>(bits) + 7) / 8
          : -1;
  if (input.getLayout().getCarrier() != "rvv" ||
      destination.getLayout().getCarrier() != "local" ||
      !sameLogicalDomain(input, destination) ||
      input.getElementType() != destination.getElementType() ||
      laneDimensions != 1 || !mappingClosed || localBytes <= 0 ||
      !getDestination().getDefiningOp<LocalBindOp>() ||
      !exactLeaf(getLeaf(), "transfer", "local-materialize",
                 "rvv.local-materialize", "none", "exact") ||
      getLeaf().getOperandGroups() != input.getLayout().getRegisterGroups() ||
      getLeaf().getResultGroups() != 0 || getLeaf().getTemporaryGroups() != 0 ||
      getLeaf().getFragmentGroups() != 0 ||
      getLeaf().getLocalBytes() != localBytes)
    return emitOpError(
        "RVV local materialize requires one complete innermost-lane value and an exact local representation");
  return mlir::success();
}

mlir::LogicalResult EncodedLocalPackOp::verify() {
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  LocalType storage = getStorage().getType();
  LocalPackPlanAttr plan = getPlan();
  LayoutAttr transfer = getTransferLayout();
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(input.getElementType());
  auto row = getRowPoint().getType().getDomain();
  auto record = getRecordPoint().getType().getDomain();
  auto rowPosition = llvm::find(result.getAxisIds().asArrayRef(), plan.getRowAxis());
  auto recordPosition =
      llvm::find(result.getAxisIds().asArrayRef(), plan.getRecordAxis());
  auto representedExtent = [&](decltype(rowPosition) position)
      -> std::optional<int64_t> {
    if (position == result.getAxisIds().asArrayRef().end())
      return std::nullopt;
    const size_t axis = static_cast<size_t>(
        position - result.getAxisIds().asArrayRef().begin());
    int64_t extent = 1;
    for (mlir::DenseI64ArrayAttr factors : {
             result.getLayout().getTimeFactors(),
             result.getLayout().getLaneFactors(),
             result.getLayout().getReplicaFactors(),
             result.getLayout().getFragmentFactors(),
             result.getLayout().getLocalFactors()}) {
      const int64_t factor = factors[axis];
      if (factor <= 0 ||
          extent > std::numeric_limits<int64_t>::max() / factor)
        return std::nullopt;
      extent *= factor;
    }
    return extent;
  };
  auto rowExtent = representedExtent(rowPosition);
  auto recordExtent = representedExtent(recordPosition);
  std::optional<int64_t> expectedLocalBytes;
  if (rowExtent && recordExtent && *recordExtent % plan.getRecordElements() == 0) {
    auto records = checkedPositiveProduct(
        {*rowExtent, *recordExtent / plan.getRecordElements()});
    if (records)
      expectedLocalBytes =
          checkedPositiveProduct({*records, plan.getRecordBytes()});
  }
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  const bool transferLayout =
      kernel && transfer && transfer.getCarrier() == "rvv" &&
      transfer.getAxisIds().asArrayRef() ==
          llvm::ArrayRef<int64_t>({plan.getRowAxis()}) &&
      transfer.getTimeFactors().asArrayRef() == llvm::ArrayRef<int64_t>({1}) &&
      transfer.getLaneFactors().asArrayRef() ==
          llvm::ArrayRef<int64_t>({plan.getInterleaveRows()}) &&
      transfer.getReplicaFactors().asArrayRef() ==
          llvm::ArrayRef<int64_t>({1}) &&
      transfer.getFragmentFactors().asArrayRef() ==
          llvm::ArrayRef<int64_t>({1}) &&
      transfer.getLocalFactors().asArrayRef() ==
          llvm::ArrayRef<int64_t>({1}) &&
      transfer.getSew() == 8 && transfer.getLmulEighths() == 8 &&
      transfer.getVl() == plan.getInterleaveRows() &&
      transfer.getRegisterGroups() == 1 &&
      supportsRVVLayout(kernel.getTarget(), transfer);
  if (!encoding || encoding.getKind() != "base" ||
      !sameLogicalDomain(input, result) ||
      input.getElementType() != result.getElementType() ||
      result.getLayout().getCarrier() != "local" ||
      storage.getPurpose() != "pack" ||
      storage.getElementType() != result.getElementType() ||
      storage.getShape() != result.getShape() ||
      storage.getAxisIds() != result.getAxisIds() ||
      storage.getSizeBytes() != plan.getLocalBytes() ||
      !expectedLocalBytes || *expectedLocalBytes != plan.getLocalBytes() ||
      row.getAxisId() != plan.getRowAxis() ||
      record.getAxisId() != plan.getRecordAxis() ||
      !rowExtent || !recordExtent || *rowExtent % plan.getInterleaveRows() ||
      *recordExtent % plan.getRecordElements() ||
      !getStorage().getDefiningOp<LocalAllocOp>() ||
      !transferLayout ||
      !exactLeaf(getLeaf(), "transfer", "local-pack",
                 "rvv.local-pack.interleave", "none", "exact") ||
      getLeaf().getOperandGroups() != 0 || getLeaf().getResultGroups() != 0 ||
      getLeaf().getTemporaryGroups() != transfer.getRegisterGroups() ||
      getLeaf().getFragmentGroups() != 0 || getLeaf().getLocalBytes() != 0)
    return emitOpError(
        "encoded local pack requires one allocated, lifetime-owned interleaved record view");
  return verifyLeafOperation(*this);
}

mlir::LogicalResult SpillOp::verify() {
  auto value = mlir::dyn_cast<ValueType>(getInput().getType());
  LocalType slot = getSlot().getType();
  if (!value || slot.getPurpose() != "spill" ||
      value.getElementType() != slot.getElementType() ||
      value.getShape() != slot.getShape() || value.getAxisIds() != slot.getAxisIds() ||
      getLeaf().getTemporaryGroups() != 0 ||
      getLeaf().getFragmentGroups() != 0 || getLeaf().getLocalBytes() != 0)
    return emitOpError("spill slot must exactly preserve its physical value domain");
  if (value.getLayout().getCarrier() == "scalar") {
    auto timeParts =
        checkedProduct(value.getLayout().getTimeFactors().asArrayRef());
    auto replicaParts =
        checkedProduct(value.getLayout().getReplicaFactors().asArrayRef());
    const unsigned bits = elementBitWidth(value.getElementType());
    if (!exactLeaf(getLeaf(), "transfer", "spill", "scalar.tuple-spill",
                   "none", "exact") ||
        !timeParts || *timeParts != 1 || !replicaParts || *replicaParts <= 0 ||
        !bits || bits % 8 ||
        *replicaParts >
            std::numeric_limits<int64_t>::max() / (bits / 8) ||
        slot.getSizeBytes() != *replicaParts * (bits / 8))
      return emitOpError(
          "scalar tuple spill requires one exact compact local representation");
    return verifyLeafOperation(*this);
  }
  if (!exactLeaf(getLeaf(), "transfer", "spill", "rvv.spill", "none",
                 "exact"))
    return emitOpError("non-scalar spill requires the exact RVV spill leaf");
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  auto streamParts =
      checkedProduct(value.getLayout().getTimeFactors().asArrayRef());
  const int64_t groups = value.getLayout().getRegisterGroups();
  const int64_t vlenBytes = kernel ? kernel.getTarget().getVlenBits() / 8 : 0;
  if (!kernel || !streamParts || *streamParts <= 0 || groups <= 0 ||
      vlenBytes <= 0 || groups > std::numeric_limits<int64_t>::max() / *streamParts ||
      groups * *streamParts >
          std::numeric_limits<int64_t>::max() / vlenBytes ||
      slot.getSizeBytes() != groups * *streamParts * vlenBytes)
    return emitOpError(
        "spill slot size must preserve every selected RVV time part");
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
  auto streamParts =
      checkedProduct(value.getLayout().getTimeFactors().asArrayRef());
  const int64_t groups = value.getLayout().getRegisterGroups();
  const int64_t vlenBytes = kernel ? kernel.getTarget().getVlenBits() / 8 : 0;
  if (!kernel || !streamParts || *streamParts <= 0 || groups <= 0 ||
      vlenBytes <= 0 || groups > std::numeric_limits<int64_t>::max() / *streamParts ||
      groups * *streamParts >
          std::numeric_limits<int64_t>::max() / vlenBytes ||
      slot.getSizeBytes() != groups * *streamParts * vlenBytes)
    return emitOpError(
        "reload slot size must preserve every selected RVV time part");
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

mlir::LogicalResult PackedPlaneMergeOp::verify() {
  ValueType low = getLowField().getType();
  ValueType high = getHighField().getType();
  ValueType result = getResult().getType();
  auto lowElement = mlir::dyn_cast<mlir::IntegerType>(low.getElementType());
  auto highElement = mlir::dyn_cast<mlir::IntegerType>(high.getElementType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto lowField = getLowField().getDefiningOp<FieldOp>();
  auto highField = getHighField().getDefiningOp<FieldOp>();
  auto plan = getPlan();
  auto axis = llvm::find(result.getAxisIds().asArrayRef(), plan.getLogicalAxis());
  if (axis == result.getAxisIds().asArrayRef().end())
    return emitOpError("packed plane merge result has no planned logical axis");
  const size_t axisPosition = static_cast<size_t>(
      axis - result.getAxisIds().asArrayRef().begin());
  if (!lowElement || lowElement.isSigned() || !highElement ||
      highElement.isSigned() || !resultElement || resultElement.isSigned() ||
      resultElement.getWidth() != 8 || !lowField || !highField ||
      lowField.getOwner() != highField.getOwner() || low.getShape() != result.getShape() ||
      high.getShape() != result.getShape() || low.getAxisIds() != result.getAxisIds() ||
      high.getAxisIds() != result.getAxisIds() ||
      lowElement.getWidth() != plan.getLowBits() ||
      highElement.getWidth() != plan.getHighBits() ||
      lowField.getAccess().getMapping() != "grouped_layered" ||
      highField.getAccess().getMapping() != "grouped_layered" ||
      lowField.getAccess().getOrder() != "lo_first" ||
      highField.getAccess().getOrder() != "lo_first" ||
      lowField.getAccess().getGroupSize() != plan.getGroupSize() ||
      highField.getAccess().getGroupSize() != plan.getGroupSize() ||
      lowField.getAccess().getLayerSize() != plan.getLowLayerBytes() ||
      highField.getAccess().getLayerSize() != plan.getHighLayerBytes() ||
      lowField.getAccess().getBitOffset() / 8 != plan.getLowByteOffset() ||
      highField.getAccess().getBitOffset() / 8 != plan.getHighByteOffset() ||
      lowField.getAccess().getBitOffset() % 8 ||
      highField.getAccess().getBitOffset() % 8 ||
      result.getLayout().getCarrier() != "scalar" ||
      result.getShape()[axisPosition] != plan.getLogicalElements() ||
      result.getLayout().getTimeFactors()[axisPosition] != 1 ||
      result.getLayout().getLaneFactors()[axisPosition] != 1 ||
      result.getLayout().getReplicaFactors()[axisPosition] !=
          plan.getLogicalElements())
    return emitOpError(
        "packed plane merge requires two byte-aligned grouped/layered fields and one scalar logical tuple");
  for (size_t index = 0; index < result.getShape().size(); ++index)
    if (index != axisPosition &&
        (result.getLayout().getTimeFactors()[index] != 1 ||
         result.getLayout().getLaneFactors()[index] != 1 ||
         result.getLayout().getFragmentFactors()[index] != 1 ||
         result.getLayout().getLocalFactors()[index] != 1))
      return emitOpError(
          "packed plane merge may only preserve non-scale axes as scalar register replicas");
  const bool scalarWords =
      exactLeaf(getLeaf(), "scalar", "packed-plane-merge",
                "scalar.packed-plane-merge.words", "none", "exact") &&
      getLeaf().getOperandGroups() == 0 && getLeaf().getResultGroups() == 0 &&
      getLeaf().getTemporaryGroups() == 0 && getLeaf().getFragmentGroups() == 0 &&
      getLeaf().getLocalBytes() == 0;
  if (!scalarWords ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({plan.getLogicalAxis(),
                                   plan.getLogicalElements(),
                                   plan.getLowBits(), plan.getHighBits(),
                                   plan.getInsertBit()}))
    return emitOpError("packed plane merge has no exact selected leaf; leaf=")
           << getLeaf() << ", plan=" << plan;
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

mlir::LogicalResult RVVBitmaskWindowLoadOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto fieldElement = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  auto resultElement = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto baseValue = mlir::dyn_cast<ValueType>(getByteBase().getType());
  mlir::Type baseElement =
      baseValue ? baseValue.getElementType() : getByteBase().getType();
  auto baseInteger = mlir::dyn_cast<mlir::IntegerType>(baseElement);
  bool scalarBase = baseElement.isIndex() ||
                    (baseInteger && !baseInteger.isSigned());
  if (baseValue) {
    auto layout = baseValue.getLayout();
    scalarBase = scalarBase && layout.getCarrier() == "scalar" &&
                 llvm::all_of(layout.getTimeFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getLaneFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getReplicaFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getFragmentFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; }) &&
                 llvm::all_of(layout.getLocalFactors().asArrayRef(),
                              [](int64_t factor) { return factor == 1; });
  }
  auto sourceAccess = sourceField ? sourceField.getAccess() : AccessAttr();
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  if (!sourceField || !fieldElement || fieldElement.isSigned() ||
      fieldElement.getWidth() != 1 || !resultElement ||
      resultElement.isSigned() || resultElement.getWidth() != 1 ||
      field.getLayout().getCarrier() != "local" ||
      result.getLayout().getCarrier() != "rvv" || !scalarBase ||
      getWindowAxes().empty() ||
      getWindowAxes().size() != getWindowExtents().size() ||
      llvm::any_of(getWindowExtents(),
                   [](int64_t extent) { return extent <= 0; }) ||
      getAccess().getForm() != "unit" ||
      getAccess().getMapping() != "grouped_layered" ||
      getAccess().getGroupSize() <= 0 || getAccess().getLayerSize() <= 0 ||
      getAccess().getGroupSize() != getAccess().getLayerSize() * 8 ||
      getAccess().getOrder() != "lo_first" || getAccess().getBitOffset() % 8 ||
      sourceAccess.getMapping() != getAccess().getMapping() ||
      sourceAccess.getGroupSize() != getAccess().getGroupSize() ||
      sourceAccess.getLayerSize() != getAccess().getLayerSize() ||
      sourceAccess.getOrder() != getAccess().getOrder() ||
      sourceAccess.getBitOffset() != getAccess().getBitOffset() ||
      (validity != "full" && validity != "tail") ||
      !exactLeaf(getLeaf(), "rvv", "bitmask-window-load",
                 "rvv.bitmask-window-load", "none", tail))
    return emitOpError()
           << "RVV bitmask window load requires one byte-aligned logical-u1 field, "
              "one scalar byte base, complete lane window axes, and the exact RVV leaf; "
              "field_def="
           << (getField().getDefiningOp()
                   ? getField().getDefiningOp()->getName().getStringRef()
                   : llvm::StringRef("block-argument"))
           << ", field=" << field << ", byte_base=" << getByteBase().getType()
           << ", scalar_base=" << scalarBase << ", result=" << result
           << ", access=" << getAccess();

  llvm::DenseSet<int64_t> windowAxes;
  for (int64_t axis : getWindowAxes())
    if (!windowAxes.insert(axis).second)
      return emitOpError("RVV bitmask window axes must be unique");
  auto windowElements =
      checkedPositiveProduct(getWindowExtents());
  if (!windowElements || *windowElements % getAccess().getGroupSize())
    return emitOpError(
        "RVV bitmask window extent must cover complete packed groups");

  auto fieldAxes = field.getAxisIds().asArrayRef();
  auto fieldShape = field.getShape().asArrayRef();
  auto sourceAxis = llvm::find(fieldAxes, getSourceAxis());
  if (sourceAxis == fieldAxes.end() ||
      llvm::any_of(getWindowAxes(), [&](int64_t axis) {
        return llvm::is_contained(fieldAxes, axis);
      }))
    return emitOpError(
        "RVV bitmask source and window axes must be disjoint");
  const size_t sourcePosition =
      static_cast<size_t>(sourceAxis - fieldAxes.begin());
  llvm::SmallVector<int64_t> expectedAxes;
  llvm::SmallVector<int64_t> expectedShape;
  for (size_t position = 0; position < fieldAxes.size(); ++position) {
    if (position == sourcePosition)
      continue;
    expectedAxes.push_back(fieldAxes[position]);
    expectedShape.push_back(fieldShape[position]);
  }
  const size_t retainedAxes = expectedAxes.size();
  expectedAxes.append(getWindowAxes().begin(), getWindowAxes().end());
  expectedShape.append(getWindowExtents().begin(), getWindowExtents().end());
  if (!llvm::equal(expectedAxes, result.getAxisIds().asArrayRef()) ||
      !llvm::equal(expectedShape, result.getShape().asArrayRef()))
    return emitOpError(
        "RVV bitmask window result must retain source axes and append its logical window axes");

  auto layout = result.getLayout();
  for (size_t position = 0; position < retainedAxes; ++position)
    if (layout.getTimeFactors()[position] != 1 ||
        layout.getLaneFactors()[position] != 1 ||
        layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1)
      return emitOpError(
          "RVV bitmask window retained axes must remain register coordinates");
  for (size_t offset = 0; offset < getWindowExtents().size(); ++offset) {
    const size_t position = retainedAxes + offset;
    if (layout.getTimeFactors()[position] != 1 ||
        layout.getLaneFactors()[position] != getWindowExtents()[offset] ||
        layout.getReplicaFactors()[position] != 1 ||
        layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1)
      return emitOpError(
          "RVV bitmask window axes must form one complete RVV lane window");
  }
  return verifyLeafOperation(*this);
}

mlir::LogicalResult RVVGroupedMacReduceOp::verify() {
  auto lhs = mlir::dyn_cast<ValueType>(getLhs().getType());
  auto resultParts = physicalPartCount(getResult().getType());
  if (!lhs || lhs.getLayout().getCarrier() != "rvv")
    return emitOpError("grouped MAC reduction lhs must use an RVV representation");
  if (!resultParts || *resultParts <= 0 ||
      getResult().getType().getLayout().getCarrier() != "rvv")
    return emitOpError(
        "grouped MAC reduction result must have one or more RVV output parts");
  if (!getPlan() || getPlan().getKind() != "reduce")
    return emitOpError("grouped MAC reduction has invalid structural parameters");
  auto supplyPlan = groupedMacSupplyProjection(lhs, getResult().getType());
  if (!supplyPlan ||
      getPackedSupplyForResult() != llvm::ArrayRef<int64_t>(*supplyPlan))
    return emitOpError(
        "grouped MAC reduction packed-supply mapping disagrees with its typed axes and layout");
  if (getPartialLayout().getCarrier() != "rvv" ||
      getPartialLayout().getSew() != 16)
    return emitOpError("grouped MAC reduction requires a 16-bit RVV partial layout");
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError(
        "grouped MAC reduction must be nested in one target kernel");
  auto target = kernel.getTarget();
  const bool operandGeometry = hasGroupedMacOperandGeometry(
      getOperation(), getLhs(), getRhs(), getPlan().getReductionAxis(),
      getPlan(), getLoadLayout(), getPartialLayout(),
      getResult().getType().getLayout());
  const bool resultLayout =
      supportsRVVLayout(target, getResult().getType().getLayout());
  if (!operandGeometry || !resultLayout)
    return emitOpError()
           << "grouped MAC reduction has no complete typed field, cohort-storage, "
              "RVV-layout, or shared reduction-point realization; operand_geometry="
           << operandGeometry << ", result_layout=" << resultLayout
           << ", lhs=" << getLhs().getType() << ", rhs=" << getRhs().getType()
           << ", result=" << getResult().getType()
           << ", plan=" << getPlan()
           << ", load_layout=" << getLoadLayout()
           << ", partial_layout=" << getPartialLayout();
  if (!exactLeaf(getLeaf(), "rvv", "grouped-mac-reduce",
                 "rvv.grouped-mac-reduce.u8-s8", "none", "agnostic") &&
      !exactLeaf(getLeaf(), "rvv", "grouped-mac-reduce",
                 "rvv.grouped-mac-reduce.u8-s8", "none", "exact"))
    return emitOpError("grouped MAC reduction leaf is not the selected RVV operation");
  if (getLeaf().getParameters().asArrayRef() !=
      llvm::ArrayRef<int64_t>({getPlan().getGroup(), getPlan().getUnroll(),
                               getPlan().getReductionAxis()}))
    return emitOpError("grouped MAC reduction leaf parameters disagree with its schedule");
  auto issueParts = checkedPositiveProduct(
      getResult().getType().getLayout().getTimeFactors().asArrayRef());
  auto expectedTemporaries =
      issueParts
          ? checkedPositiveProduct(
                {getPartialLayout().getRegisterGroups(), getPlan().getUnroll(),
                 *issueParts})
          : std::nullopt;
  if (!expectedTemporaries ||
      getLeaf().getOperandGroups() !=
          (getLoadLayout().getLmulEighths() + 7) / 8 ||
      getLeaf().getResultGroups() !=
          getResult().getType().getLayout().getRegisterGroups() ||
      getLeaf().getTemporaryGroups() != *expectedTemporaries)
    return emitOpError(
        "grouped MAC reduction leaf resources disagree with its typed layouts");
  return mlir::success();
}

mlir::LogicalResult RVVGroupedMacLoadOp::verify() {
  auto window = getResult().getType();
  auto lhs = mlir::dyn_cast<ValueType>(getLhs().getType());
  if (!lhs || lhs.getLayout().getCarrier() != "rvv")
    return emitOpError("grouped MAC lhs must use an RVV representation");
  if (window.getFamily() != "grouped-mac" || !getPlan() ||
      (getPlan().getKind() != "compact" &&
       getPlan().getKind() != "fragmented"))
    return emitOpError("grouped MAC window has invalid structural parameters");
  if (window.getReductionAxis() != getPlan().getReductionAxis() ||
      window.getSlots() != getPlan().getUnroll() ||
      window.getTermsPerSlot() != getPlan().getGroup())
    return emitOpError("grouped MAC window shape disagrees with its operation");
  if (window.getLhsType() != getLhs().getType() ||
      window.getRhsType() != getRhs().getType() ||
      window.getPartialLayout() != getPartialLayout())
    return emitOpError("grouped MAC window type disagrees with its operands");
  auto windowResult = mlir::dyn_cast<ValueType>(window.getResultType());
  auto supplyPlan = groupedMacSupplyProjection(lhs, windowResult);
  auto issueParts = checkedPositiveProduct(
      window.getResultLayout().getTimeFactors().asArrayRef());
  auto replicaParts = checkedPositiveProduct(
      window.getResultLayout().getReplicaFactors().asArrayRef());
  if (!supplyPlan ||
      getPackedSupplyForResult() != llvm::ArrayRef<int64_t>(*supplyPlan) ||
      !issueParts || *issueParts != 1 || !replicaParts ||
      *replicaParts != window.getResultParts())
    return emitOpError(
        "grouped MAC window requires one typed packed-supply mapping over register replicas");
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError("grouped MAC load must be nested in one target kernel");
  auto target = kernel.getTarget();
  if (!hasGroupedMacOperandGeometry(
          getOperation(), getLhs(), getRhs(), getPlan().getReductionAxis(),
          getPlan(), getLoadLayout(), getPartialLayout(),
          window.getResultLayout()) ||
      !supportsRVVLayout(target, window.getResultLayout()))
    return emitOpError(
        "grouped MAC load has no complete typed field, cohort-storage, "
        "RVV-layout, or shared reduction-point realization");
  if ((!exactLeaf(getLeaf(), "rvv", "grouped-mac-load",
                  "rvv.grouped-mac-load.u8-s8", "none", "agnostic") &&
       !exactLeaf(getLeaf(), "rvv", "grouped-mac-load",
                  "rvv.grouped-mac-load.u8-s8", "none", "exact")) ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>(
              {getPlan().getGroup(), getPlan().getUnroll(),
               getPlan().getReductionAxis()}))
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
  auto partialLMULValue = doubledPositive(getSliceLmulEighths());
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
  const int64_t outputParts =
      shapedResult ? physicalPartCount(shapedResult).value_or(-1) : 1;
  auto laneProduct = [](ValueType value,
                        llvm::ArrayRef<int64_t> selectedAxes) {
    int64_t result = 1;
    for (auto [axis, factor] :
         llvm::zip(value.getAxisIds().asArrayRef(),
                   value.getLayout().getLaneFactors().asArrayRef())) {
      if (!llvm::is_contained(selectedAxes, axis))
        continue;
      if (factor <= 0 || result > std::numeric_limits<int64_t>::max() / factor)
        return int64_t{-1};
      result *= factor;
    }
    return result;
  };
  auto allLanes = [](ValueType value) {
    return checkedPositiveProduct(
               value.getLayout().getLaneFactors().asArrayRef())
        .value_or(-1);
  };
  const int64_t lhsReductionLanes = laneProduct(lhs, getOver());
  const int64_t rhsReductionLanes = laneProduct(rhs, getOver());
  const int64_t lhsLanes = allLanes(lhs);
  const int64_t rhsLanes = allLanes(rhs);
  auto validOffsets = [&](llvm::ArrayRef<int64_t> offsets, int64_t lanes) {
    if (offsets.size() != static_cast<size_t>(outputParts) || lanes <= 0 ||
        getReductionLanes() <= 0)
      return false;
    return llvm::all_of(offsets, [&](int64_t offset) {
      return offset >= 0 && offset % getReductionLanes() == 0 &&
             offset <= lanes - getReductionLanes();
    });
  };
  auto lhsParts = physicalPartCount(lhs);
  auto rhsParts = physicalPartCount(rhs);
  auto validParts = [&](llvm::ArrayRef<int64_t> parts,
                        std::optional<int64_t> available) {
    if (!available || getReductionStreams() <= 0 ||
        outputParts > std::numeric_limits<int64_t>::max() /
                          getReductionStreams() ||
        parts.size() != static_cast<size_t>(outputParts * getReductionStreams()))
      return false;
    return llvm::all_of(parts, [&](int64_t part) {
      return part >= 0 && part < *available;
    });
  };
  const bool closedLaneSlices =
      outputParts > 0 && getReductionStreams() > 0 &&
      getReductionLanes() == lhsReductionLanes &&
      getReductionLanes() == rhsReductionLanes && lhsLanes > 0 &&
      rhsLanes > 0 &&
      getSliceLmulEighths() <= lhs.getLayout().getLmulEighths() &&
      getSliceLmulEighths() <= rhs.getLayout().getLmulEighths() &&
      lhs.getLayout().getLmulEighths() * getReductionLanes() <=
          getSliceLmulEighths() * lhsLanes &&
      rhs.getLayout().getLmulEighths() * getReductionLanes() <=
          getSliceLmulEighths() * rhsLanes &&
      validOffsets(getLhsLaneOffsets(), lhsLanes) &&
      validOffsets(getRhsLaneOffsets(), rhsLanes) &&
      validParts(getLhsParts(), lhsParts) &&
      validParts(getRhsParts(), rhsParts);
  auto factorFor = [](LayoutAttr layout, mlir::DenseI64ArrayAttr factors,
                      int64_t axis) {
    auto found = llvm::find(layout.getAxisIds().asArrayRef(), axis);
    if (found == layout.getAxisIds().asArrayRef().end())
      return int64_t{0};
    return factors[static_cast<size_t>(
        found - layout.getAxisIds().asArrayRef().begin())];
  };
  bool compatibleReductionMapping =
      !getOver().empty() && lhs.getLayout().getCarrier() == "rvv" &&
      rhs.getLayout().getCarrier() == "rvv" &&
      lhs.getLayout().getSew() == rhs.getLayout().getSew();
  for (int64_t axis : getOver()) {
    auto completeMappedAxis = [&](ValueType value) {
      auto found = llvm::find(value.getAxisIds().asArrayRef(), axis);
      if (found == value.getAxisIds().asArrayRef().end())
        return false;
      const size_t position = static_cast<size_t>(
          found - value.getAxisIds().asArrayRef().begin());
      const int64_t extent = value.getShape()[position];
      auto layout = value.getLayout();
      return extent > 0 &&
             layout.getTimeFactors()[position] *
                     layout.getLaneFactors()[position] ==
                 extent &&
             layout.getReplicaFactors()[position] == 1 &&
             layout.getFragmentFactors()[position] == 1 &&
             layout.getLocalFactors()[position] == 1;
    };
    compatibleReductionMapping &=
        completeMappedAxis(lhs) && completeMappedAxis(rhs) &&
        factorFor(lhs.getLayout(), lhs.getLayout().getLaneFactors(), axis) ==
            factorFor(rhs.getLayout(), rhs.getLayout().getLaneFactors(), axis) &&
        factorFor(lhs.getLayout(), lhs.getLayout().getTimeFactors(), axis) ==
            factorFor(rhs.getLayout(), rhs.getLayout().getTimeFactors(), axis);
  }
  llvm::StringRef topologyKind = getPartialTopology().getKind();
  auto partialLayoutPlan = getPartialLayoutPlanAttr();
  auto layeredPartialPlan = getLayeredPartialPlanAttr();
  const bool topologyAssigned = topologyKind != "unassigned";
  bool layoutPlanClosed =
      topologyAssigned == static_cast<bool>(partialLayoutPlan) &&
      (topologyKind == "layered") == static_cast<bool>(layeredPartialPlan);
  if (partialLayoutPlan) {
    auto partialSlot = mlir::dyn_cast<ValueType>(
        partialLayoutPlan.getPartialSlotType());
    int64_t plannedLanes = 1;
    bool completePartialAxes =
        partialSlot && partialSlot.getLayout().getCarrier() == "rvv" &&
        partialSlot.getAxisIds().asArrayRef() == getOver() &&
        partialSlot.getShape().size() == getOver().size();
    if (completePartialAxes)
      for (auto [extent, lane] :
           llvm::zip(partialSlot.getShape().asArrayRef(),
                     partialSlot.getLayout().getLaneFactors().asArrayRef())) {
        if (extent <= 0 || extent != lane ||
            plannedLanes > getReductionLanes() / extent) {
          completePartialAxes = false;
          break;
        }
        plannedLanes *= extent;
      }
    layoutPlanClosed &=
        completePartialAxes && plannedLanes == getReductionLanes();
    const int64_t sourceSlots = getPartialTopology().getSourceSlots();
    const int64_t split = getPartialTopology().getLaneSplit();
    const int64_t expected =
        getPartialTopology().getOutputReplicas() * sourceSlots;
    const int64_t sourceLanes = split * getReductionLanes();
    auto sourceLhs = mlir::dyn_cast<ValueType>(
        partialLayoutPlan.getSourceLhsType());
    auto sourceRhs = mlir::dyn_cast<ValueType>(
        partialLayoutPlan.getSourceRhsType());
    auto mapComplete = [&](mlir::DenseI64ArrayAttr values) {
      return expected > 0 &&
             values.size() == static_cast<size_t>(expected);
    };
    const bool sourcePlanRequired = split > 1;
    layoutPlanClosed &=
        sourcePlanRequired == static_cast<bool>(sourceLhs) &&
        sourcePlanRequired == static_cast<bool>(sourceRhs) &&
        sourcePlanRequired == mlir::isa<ValueType>(
                                  partialLayoutPlan.getSourceSlotType()) &&
        sourcePlanRequired == mlir::isa<ValueType>(
                                  partialLayoutPlan.getSplitSourceSlotType());
    auto sameLogicalValue = [](ValueType source, ValueType planned) {
      return source && planned &&
             source.getElementType() == planned.getElementType() &&
             source.getShape() == planned.getShape() &&
             source.getAxisIds() == planned.getAxisIds();
    };
    if (sourcePlanRequired)
      layoutPlanClosed &=
          sourceSlots > 0 && sourceLanes > 0 &&
          sameLogicalValue(lhs, sourceLhs) &&
          sameLogicalValue(rhs, sourceRhs) &&
          sourceLhs.getLayout().getCarrier() == "rvv" &&
          sourceRhs.getLayout().getCarrier() == "rvv" &&
          supportsRVVLayout(target, sourceLhs.getLayout()) &&
          supportsRVVLayout(target, sourceRhs.getLayout()) &&
          mapComplete(partialLayoutPlan.getSourceLhsParts()) &&
          mapComplete(partialLayoutPlan.getSourceRhsParts()) &&
          mapComplete(partialLayoutPlan.getSourceLhsOffsets()) &&
          mapComplete(partialLayoutPlan.getSourceRhsOffsets());
    if (layoutPlanClosed && sourcePlanRequired) {
      auto plannedLhsParts = physicalPartCount(sourceLhs);
      auto plannedRhsParts = physicalPartCount(sourceRhs);
      auto plannedLhsLanes = allLanes(sourceLhs);
      auto plannedRhsLanes = allLanes(sourceRhs);
      if (!plannedLhsParts || !plannedRhsParts ||
          plannedLhsLanes < sourceLanes || plannedRhsLanes < sourceLanes) {
        layoutPlanClosed = false;
      } else {
        for (auto [lhsPart, rhsPart, lhsOffset, rhsOffset] : llvm::zip(
                 partialLayoutPlan.getSourceLhsParts().asArrayRef(),
                 partialLayoutPlan.getSourceRhsParts().asArrayRef(),
                 partialLayoutPlan.getSourceLhsOffsets().asArrayRef(),
                 partialLayoutPlan.getSourceRhsOffsets().asArrayRef()))
          layoutPlanClosed &= lhsPart >= 0 && lhsPart < *plannedLhsParts &&
                              rhsPart >= 0 && rhsPart < *plannedRhsParts &&
                              lhsOffset >= 0 && rhsOffset >= 0 &&
                              lhsOffset % sourceLanes == 0 &&
                              rhsOffset % sourceLanes == 0 &&
                              lhsOffset <= plannedLhsLanes - sourceLanes &&
                              rhsOffset <= plannedRhsLanes - sourceLanes;
      }
    }
  }
  if (layeredPartialPlan)
    layoutPlanClosed &= getPartialTopology().getRootOperand() >= 0 &&
                        getOver().size() == 1 &&
                        layeredPartialPlan.getGeometry().getAxis() == getOver()[0];
  const bool validTopology =
      topologyKind == "unassigned" || topologyKind == "sequential_fused" ||
      topologyKind == "sequential_per_stream" ||
      topologyKind == "independent" || topologyKind == "replica_reduced" ||
      topologyKind == "scaled" ||
      topologyKind == "reduced_scaled" ||
      topologyKind == "level_scaled" || topologyKind == "merged" ||
      topologyKind == "layered";
  if (!validTopology || !layoutPlanClosed || getPartialUnroll() <= 0 ||
      getSliceLmulEighths() <= 0 || !closedLaneSlices ||
      !lhsElement || !rhsElement || !resultElement ||
      lhsElement.getWidth() > 16 || rhsElement.getWidth() > 16 ||
      std::max<unsigned>(8, lhsElement.getWidth()) !=
          std::max<unsigned>(8, rhsElement.getWidth()) ||
      (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
      resultElement.getWidth() != 32 || !legalResult ||
      getOver().empty() ||
      !llvm::is_contained(lhs.getAxisIds().asArrayRef(), getOver()[0]) ||
      !llvm::is_contained(rhs.getAxisIds().asArrayRef(), getOver()[0]) ||
      !compatibleReductionMapping ||
      !target.getHasWideningInteger() ||
      !supportsRVVLayout(target, lhs.getLayout()) ||
      !supportsRVVLayout(target, rhs.getLayout()) ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(),
                          getSliceLmulEighths()) ||
      !partialLMULValue ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), partialLMUL) ||
      !exactLeaf(getLeaf(), "rvv", "widen-dot",
                 "rvv.vwmul-vwredsum", "none", "exact"))
    return emitOpError()
           << "RVV widening dot requires matching <=16-bit integer vectors with "
              "at least one signed operand, one selected partial topology, "
              "one or more mapped reduction axes, a signed-i32 scalar result, "
              "and a legal doubled "
              "LMUL; lhs="
           << lhs << ", rhs=" << rhs << ", result=" << getResult().getType()
           << ", over=" << getOver() << ", partial_lmul=" << partialLMUL
           << ", leaf=" << getLeaf();
  return mlir::success();
}

mlir::LogicalResult RVVWidenMultiplyOp::verify() {
  ValueType lhs = getLhs().getType();
  ValueType rhs = getRhs().getType();
  ValueType result = getResult().getType();
  auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
  auto resultElement =
      mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  llvm::StringRef instruction;
  if (lhsElement && rhsElement) {
    if (lhsElement.isUnsigned() && rhsElement.isUnsigned())
      instruction = "rvv.vwmulu.vv";
    else if (lhsElement.isSigned() && rhsElement.isSigned())
      instruction = "rvv.vwmul.vv";
    else if (lhsElement.isSigned())
      instruction = "rvv.vwmulsu.vv";
    else
      instruction = "rvv.vwmulsu.vv.swap";
  }
  const bool sameCoordinates =
      lhs.getShape() == rhs.getShape() && lhs.getShape() == result.getShape() &&
      lhs.getAxisIds() == rhs.getAxisIds() &&
      lhs.getAxisIds() == result.getAxisIds() &&
      lhs.getLayout().getCarrier() == "rvv" &&
      rhs.getLayout().getCarrier() == "rvv" &&
      result.getLayout().getCarrier() == "rvv" &&
      lhs.getLayout().getAxisIds() == rhs.getLayout().getAxisIds() &&
      lhs.getLayout().getAxisIds() == result.getLayout().getAxisIds() &&
      lhs.getLayout().getTimeFactors() == rhs.getLayout().getTimeFactors() &&
      lhs.getLayout().getTimeFactors() == result.getLayout().getTimeFactors() &&
      lhs.getLayout().getLaneFactors() == rhs.getLayout().getLaneFactors() &&
      lhs.getLayout().getLaneFactors() == result.getLayout().getLaneFactors() &&
      lhs.getLayout().getReplicaFactors() ==
          rhs.getLayout().getReplicaFactors() &&
      lhs.getLayout().getReplicaFactors() ==
          result.getLayout().getReplicaFactors() &&
      lhs.getLayout().getFragmentFactors() ==
          rhs.getLayout().getFragmentFactors() &&
      lhs.getLayout().getFragmentFactors() ==
          result.getLayout().getFragmentFactors() &&
      lhs.getLayout().getLocalFactors() == rhs.getLayout().getLocalFactors() &&
      lhs.getLayout().getLocalFactors() == result.getLayout().getLocalFactors();
  if (!kernel || !lhsElement || !rhsElement || !resultElement ||
      lhsElement.isSignless() || rhsElement.isSignless() ||
      resultElement.isSignless() || lhsElement.getWidth() > 16 ||
      rhsElement.getWidth() > 16 ||
      lhsElement.getWidth() > lhs.getLayout().getSew() ||
      rhsElement.getWidth() > rhs.getLayout().getSew() ||
      resultElement.getWidth() != 2 * lhs.getLayout().getSew() ||
      !sameCoordinates ||
      lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
      result.getLayout().getSew() !=
          static_cast<int64_t>(resultElement.getWidth()) ||
      rhs.getLayout().getLmulEighths() !=
          lhs.getLayout().getLmulEighths() ||
      result.getLayout().getLmulEighths() !=
          2 * lhs.getLayout().getLmulEighths() ||
      lhs.getLayout().getVl() != rhs.getLayout().getVl() ||
      lhs.getLayout().getVl() != result.getLayout().getVl() ||
      !kernel.getTarget().getHasWideningInteger() ||
      !supportsRVVLayout(kernel.getTarget(), lhs.getLayout()) ||
      !supportsRVVLayout(kernel.getTarget(), rhs.getLayout()) ||
      !supportsRVVLayout(kernel.getTarget(), result.getLayout()) ||
      !exactLeaf(getLeaf(), "rvv", "widen-multiply", instruction, "none",
                 "exact"))
    return emitOpError(
        "RVV widening multiply requires coordinate-identical integer operands and one exact doubled-width result layout");
  return mlir::success();
}

mlir::LogicalResult RVVWidenScalarMultiplyOp::verify() {
  ValueType lhs = getLhs().getType();
  ValueType result = getResult().getType();
  auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(getRhs().getType());
  auto resultElement =
      mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  llvm::StringRef instruction;
  if (lhsElement && rhsElement) {
    if (lhsElement.isUnsigned() && rhsElement.isUnsigned())
      instruction = "rvv.vwmulu.vx";
    else if (lhsElement.isSigned() && rhsElement.isSigned())
      instruction = "rvv.vwmul.vx";
    else if (lhsElement.isSigned())
      instruction = "rvv.vwmulsu.vx";
  }
  const bool sameCoordinates =
      lhs.getShape() == result.getShape() &&
      lhs.getAxisIds() == result.getAxisIds() &&
      lhs.getLayout().getCarrier() == "rvv" &&
      result.getLayout().getCarrier() == "rvv" &&
      lhs.getLayout().getAxisIds() == result.getLayout().getAxisIds() &&
      lhs.getLayout().getTimeFactors() ==
          result.getLayout().getTimeFactors() &&
      lhs.getLayout().getLaneFactors() ==
          result.getLayout().getLaneFactors() &&
      lhs.getLayout().getReplicaFactors() ==
          result.getLayout().getReplicaFactors() &&
      lhs.getLayout().getFragmentFactors() ==
          result.getLayout().getFragmentFactors() &&
      lhs.getLayout().getLocalFactors() ==
          result.getLayout().getLocalFactors();
  if (!kernel || !lhsElement || !rhsElement || !resultElement ||
      lhsElement.isSignless() || rhsElement.isSignless() ||
      resultElement.isSignless() || !isScalar(getRhs().getType()) ||
      lhsElement != rhsElement || lhsElement.getWidth() > 16 ||
      lhsElement.getWidth() != lhs.getLayout().getSew() ||
      resultElement.getWidth() != 2 * lhsElement.getWidth() ||
      result.getLayout().getSew() !=
          static_cast<int64_t>(resultElement.getWidth()) ||
      result.getLayout().getLmulEighths() !=
          2 * lhs.getLayout().getLmulEighths() ||
      lhs.getLayout().getVl() != result.getLayout().getVl() ||
      !sameCoordinates || !kernel.getTarget().getHasWideningInteger() ||
      !supportsRVVLayout(kernel.getTarget(), lhs.getLayout()) ||
      !supportsRVVLayout(kernel.getTarget(), result.getLayout()) ||
      !exactLeaf(getLeaf(), "rvv", "widen-scalar-multiply", instruction,
                 "none", "exact"))
    return emitOpError(
        "RVV widening scalar multiply requires one narrow scalar and one exact doubled-width result layout");
  return mlir::success();
}

mlir::LogicalResult RVVRegularRepeatIndexOp::verify() {
  if (getReductionAxis() <= 0 || getRepeat() <= 1 || getResults().empty() ||
      getPartBases().size() != getResults().size())
    return emitOpError(
        "regular-repeat index requires one reduction axis, repeat factor, and base array per result");
  const bool powerOfTwo = (getRepeat() & (getRepeat() - 1)) == 0;
  llvm::StringRef instruction = powerOfTwo ? "rvv.regular-repeat-index.pow2"
                                           : "rvv.regular-repeat-index.div";
  if (!exactLeaf(getLeaf(), "rvv", "regular-repeat-index", instruction,
                 "none", "exact"))
    return emitOpError(
        "regular-repeat index leaf disagrees with its repeat geometry");
  mlir::Type firstType = getResults().front().getType();
  for (auto [resultValue, basesAttribute] :
       llvm::zip(getResults(), getPartBases())) {
    ValueType result = mlir::cast<ValueType>(resultValue.getType());
    auto element = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
    auto bases = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(basesAttribute);
    auto axis = llvm::find(result.getAxisIds().asArrayRef(), getReductionAxis());
    auto timeParts = checkedPositiveProduct(
        result.getLayout().getTimeFactors().asArrayRef());
    if (resultValue.getType() != firstType || !element || !element.isUnsigned() ||
        result.getLayout().getCarrier() != "rvv" ||
        result.getLayout().getSew() != static_cast<int64_t>(element.getWidth()) ||
        axis == result.getAxisIds().asArrayRef().end() || !bases || !timeParts ||
        bases.size() != static_cast<size_t>(*timeParts))
      return emitOpError(
          "regular-repeat index results must share one unsigned RVV coordinate type");
    const size_t position = static_cast<size_t>(
        axis - result.getAxisIds().asArrayRef().begin());
    const int64_t lanes = result.getLayout().getLaneFactors()[position];
    if (lanes <= getRepeat() || result.getShape()[position] !=
                                    lanes * result.getLayout().getTimeFactors()[position])
      return emitOpError(
          "regular-repeat index is only valid for a repeated source spanning multiple lanes");
    for (int64_t base : bases.asArrayRef())
      if (base < 0)
        return emitOpError("regular-repeat index base must be non-negative");
  }
  return mlir::success();
}

mlir::LogicalResult RVVRegularRepeatGatherOp::verify() {
  ValueType field = getField().getType();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto fieldElement = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  const int64_t axis = getReductionAxis();
  auto fieldAxis = llvm::find(field.getAxisIds().asArrayRef(), axis);
  if (!sourceField || !fieldElement || fieldElement.isSignless() || axis <= 0 ||
      fieldAxis == field.getAxisIds().asArrayRef().end() ||
      getAccess() != sourceField.getAccess() ||
      getAccess().getMapping() != "natural" ||
      getAccess().getForm() != "unit" || getAccess().getBitOffset() % 8 ||
      (fieldElement.getWidth() != 8 && fieldElement.getWidth() != 16 &&
       fieldElement.getWidth() != 32) ||
      getSourceBase() < 0 || getSourceCount() <= 0 || getRepeat() <= 1 ||
      getResults().empty() || getPartBases().size() != getResults().size())
    return emitOpError()
           << "regular-repeat gather requires one natural encoded field and closed load/gather geometry; field="
           << field << ", source-field=" << static_cast<bool>(sourceField)
           << ", element=" << field.getElementType() << ", axis=" << axis
           << ", access=" << getAccess() << ", source-base="
           << getSourceBase() << ", source-count=" << getSourceCount()
           << ", repeat=" << getRepeat() << ", results=" << getResults().size()
           << ", indices=" << getIndices().size();
  const bool powerOfTwo =
      getRepeat() > 0 && (getRepeat() & (getRepeat() - 1)) == 0;
  auto firstResult = mlir::cast<ValueType>(getResults().front().getType());
  auto firstAxis = llvm::find(firstResult.getAxisIds().asArrayRef(), axis);
  const int64_t firstLanes =
      firstAxis == firstResult.getAxisIds().asArrayRef().end()
          ? 0
          : firstResult.getLayout().getLaneFactors()[static_cast<size_t>(
                firstAxis - firstResult.getAxisIds().asArrayRef().begin())];
  llvm::StringRef instruction =
      firstLanes <= getRepeat()
          ? "rvv.regular-repeat-broadcast"
          : powerOfTwo ? "rvv.regular-repeat-gather.pow2"
                       : "rvv.regular-repeat-gather.div";
  if (!exactLeaf(getLeaf(), "rvv", "regular-repeat-gather", instruction,
                 "none", "exact"))
    return emitOpError(
        "regular-repeat gather leaf disagrees with its repeat-index geometry");

  const size_t fieldPosition = static_cast<size_t>(
      fieldAxis - field.getAxisIds().asArrayRef().begin());
  if (field.getShape()[fieldPosition] <= 0 ||
      getSourceBase() > field.getShape()[fieldPosition] - getSourceCount())
    return emitOpError("regular-repeat gather source window is out of bounds");

  if ((firstLanes <= getRepeat() && !getIndices().empty()) ||
      (firstLanes > getRepeat() &&
       getIndices().size() != getResults().size()))
    return emitOpError(
        "regular-repeat gather must carry typed indices exactly when lanes span repeated source elements");
  for (auto [resultNumber, resultValue] : llvm::enumerate(getResults())) {
    ValueType result = mlir::cast<ValueType>(resultValue.getType());
    auto resultAxis = llvm::find(result.getAxisIds().asArrayRef(), axis);
    auto bases = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(
        getPartBases()[resultNumber]);
    auto timeParts = checkedPositiveProduct(
        result.getLayout().getTimeFactors().asArrayRef());
    auto registerParts = checkedPositiveProduct(
        result.getLayout().getReplicaFactors().asArrayRef());
    if (!bases || !timeParts || !registerParts ||
        bases.size() != static_cast<size_t>(*timeParts) ||
        resultAxis == result.getAxisIds().asArrayRef().end() ||
        result.getElementType() != field.getElementType() ||
        result.getLayout().getCarrier() != "rvv" ||
        result.getLayout().getSew() != fieldElement.getWidth() ||
        *registerParts <= 0)
      return emitOpError(
          "regular-repeat gather result coordinates are incomplete");
    const size_t resultPosition = static_cast<size_t>(
        resultAxis - result.getAxisIds().asArrayRef().begin());
    const int64_t lanes = result.getLayout().getLaneFactors()[resultPosition];
    if (lanes <= 1 || result.getLayout().getVl() < lanes ||
        result.getShape()[resultPosition] !=
            lanes * result.getLayout().getTimeFactors()[resultPosition] ||
        (getRepeat() % lanes != 0 && lanes % getRepeat() != 0))
      return emitOpError(
          "regular-repeat gather result does not have one complete lane/time projection");
    for (int64_t base : bases.asArrayRef())
      if (base < 0 ||
          base + ((lanes - 1) / getRepeat()) >= getSourceCount())
        return emitOpError(
            "regular-repeat gather index addresses outside its selected source window");
    if (firstLanes <= getRepeat())
      continue;
    mlir::Value index = getIndices()[resultNumber];
    ValueType indexType = mlir::cast<ValueType>(index.getType());
    auto indexElement =
        mlir::dyn_cast<mlir::IntegerType>(indexType.getElementType());
    auto indexOp = index.getDefiningOp<RVVRegularRepeatIndexOp>();
    if (!indexElement || !indexElement.isUnsigned() ||
        indexType.getShape() != result.getShape() ||
        indexType.getAxisIds() != result.getAxisIds() ||
        indexType.getLayout() != result.getLayout() || !indexOp ||
        indexOp.getReductionAxis() != axis ||
        indexOp.getRepeat() != getRepeat())
      return emitOpError(
          "regular-repeat gather typed index does not match its result coordinates");
    auto indexBases = mlir::cast<mlir::DenseI64ArrayAttr>(
        indexOp.getPartBases()[
            mlir::cast<mlir::OpResult>(index).getResultNumber()]);
    if (indexBases != bases)
      return emitOpError(
          "regular-repeat gather index bases disagree with the selected source window");
  }
  return mlir::success();
}

mlir::LogicalResult RVVStorageWindowOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  StorageWindowPlanAttr plan = getPlan();
  const int64_t axis = plan.getReductionAxis();
  auto fieldAxis = llvm::find(field.getAxisIds().asArrayRef(), axis);
  auto resultAxis = llvm::find(result.getAxisIds().asArrayRef(), axis);
  auto pointDomain = getOrigin().getType().getDomain();
  auto pointAxis = pointDomain.getAxisId();
  LoadOp load = sourceField ? sourceLoad(sourceField.getOwner()) : LoadOp();
  MemDescType memory = load ? load.getRegion().getType() : MemDescType();
  auto timeParts = checkedPositiveProduct(
      result.getLayout().getTimeFactors().asArrayRef());
  llvm::StringRef mapping = getAccess().getMapping();
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  llvm::StringRef instruction =
      mapping == "grouped_layered" ? "rvv.storage-window.layered"
                                    : "rvv.storage-window.natural";
  if (!sourceField || !memory ||
      !sameStorageGeometry(getAccess(), sourceField.getAccess()) ||
      !getLogicalOffset().getType().isIndex() || axis <= 0 || pointAxis != axis ||
      field.getElementType() != result.getElementType() ||
      field.getAxisIds() != result.getAxisIds() ||
      fieldAxis == field.getAxisIds().asArrayRef().end() ||
      resultAxis == result.getAxisIds().asArrayRef().end() ||
      pointDomain.getTail() != "exact" ||
      result.getLayout().getCarrier() != "rvv" || !timeParts || *timeParts <= 0 ||
      (mapping != "natural" && mapping != "grouped_layered") ||
      (validity != "full" && validity != "tail") ||
      plan.getRecordElements() != memory.getElements() ||
      getAccess().getBitOffset() % 8 ||
      plan.getByteOffset() != getAccess().getBitOffset() / 8 ||
      plan.getElementBits() !=
          static_cast<int64_t>(elementBitWidth(field.getElementType())) ||
      !exactLeaf(getLeaf(), "rvv", "storage-window", instruction,
                 "none", tail))
    return emitOpError()
           << "RVV storage window requires one typed field/point/offset edge and "
              "one complete lane result; field="
           << field << ", result=" << result << ", axis=" << axis
           << ", point-axis=" << pointAxis << ", access=" << getAccess()
           << ", field-access="
           << (sourceField ? sourceField.getAccess() : AccessAttr())
           << ", point-tail=" << pointDomain.getTail() << ", leaf=" << getLeaf()
           << ", plan=" << plan;
  const size_t fieldPosition = static_cast<size_t>(
      fieldAxis - field.getAxisIds().asArrayRef().begin());
  const size_t resultPosition = static_cast<size_t>(
      resultAxis - result.getAxisIds().asArrayRef().begin());
  const int64_t lanes =
      result.getLayout().getLaneFactors()[resultPosition];
  const int64_t projectionBase = plan.getProjectionBase();
  const int64_t projectionStride = plan.getProjectionStride();
  const int64_t projectionRepeat = plan.getProjectionRepeat();
  const int64_t projectionExtent = plan.getProjectionExtent();
  const int64_t offsetAlignment = plan.getOffsetAlignment();
  const bool projectionBounds =
      projectionBase >= 0 && projectionStride > 0 && projectionRepeat > 0 &&
      projectionExtent > 0 && field.getShape()[fieldPosition] > 0 &&
      projectionBase <= field.getShape()[fieldPosition] - 1 &&
      (projectionExtent - 1) / projectionRepeat <=
          (field.getShape()[fieldPosition] - 1 - projectionBase) /
              projectionStride;
  if (lanes <= 0 || !projectionBounds ||
      projectionExtent < lanes * *timeParts ||
      offsetAlignment != lanes ||
      (projectionRepeat % lanes != 0 && lanes % projectionRepeat != 0) ||
      result.getShape()[resultPosition] !=
          result.getLayout().getLaneFactors()[resultPosition] *
              result.getLayout().getTimeFactors()[resultPosition] ||
      result.getLayout().getVl() < lanes)
    return emitOpError()
           << "RVV storage window result must cover one bounded, lane-aligned "
              "projected subrange; field="
           << field << ", result=" << result << ", lanes=" << lanes
           << ", projection=<" << projectionBase << "," << projectionStride
           << "," << projectionRepeat << "," << projectionExtent
           << ">, offset-alignment=" << offsetAlignment;
  if (mapping == "grouped_layered") {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
    const int64_t group = getAccess().getGroupSize();
    const int64_t layer = getAccess().getLayerSize();
    const int64_t layers = layer > 0 ? group / layer : 0;
    const int64_t physicalLayerBase =
        getAccess().getOrder() == "lo_first" ? 0 : layers - 1;
    const int64_t physicalLayerStep =
        getAccess().getOrder() == "lo_first" ? 1 : -1;
    if (!integer || integer.isSigned() || integer.getWidth() <= 0 ||
        integer.getWidth() >= 8 || group <= 0 || layer <= 0 || group % layer ||
        layer % lanes || getAccess().getBitOffset() % 8 ||
        plan.getKind() != "layered" || plan.getGroupSize() != group ||
        plan.getLayerSize() != layer ||
        plan.getPhysicalLayerBase() != physicalLayerBase ||
        plan.getPhysicalLayerStep() != physicalLayerStep ||
        plan.getShiftBase() != physicalLayerBase * integer.getWidth() ||
        plan.getShiftStep() != physicalLayerStep * integer.getWidth() ||
        plan.getMaskValue() != (int64_t{1} << integer.getWidth()) - 1 ||
        projectionBase != 0 || projectionStride != 1 || projectionRepeat != 1 ||
        projectionExtent != lanes * *timeParts)
      return emitOpError(
          "layered RVV storage window requires one byte-aligned contiguous lane window inside a storage layer");
  } else {
    llvm::StringRef expectedKind =
        projectionRepeat > 1
            ? "repeat"
            : projectionStride == 1 ? "unit" : "strided";
    if (plan.getKind() != expectedKind)
      return emitOpError(
          "natural RVV storage window plan disagrees with its selected access form");
  }
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

mlir::LogicalResult RVVLayeredRecordLoadOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto physicalPoint = getOrigin().getDefiningOp<PhysicalPointOp>();
  auto pointPartition =
      physicalPoint
          ? physicalPoint.getPartition().getDefiningOp<mlir::arith::ConstantIndexOp>()
          : mlir::arith::ConstantIndexOp();
  StorageWindowPlanAttr plan = getPlan();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  LoadOp load = sourceField ? sourceLoad(sourceField.getOwner()) : LoadOp();
  MemDescType memory = load ? load.getRegion().getType() : MemDescType();
  const int64_t reductionAxis = plan ? plan.getReductionAxis() : 0;
  auto fieldReduction =
      llvm::find(field.getAxisIds().asArrayRef(), reductionAxis);
  auto resultReduction =
      llvm::find(result.getAxisIds().asArrayRef(), reductionAxis);
  int64_t laneAxis = 0;
  int64_t lanes = 0;
  for (auto [axis, factor] : llvm::zip(
           result.getAxisIds().asArrayRef(),
           result.getLayout().getLaneFactors().asArrayRef())) {
    if (factor <= 1)
      continue;
    if (laneAxis != 0)
      return emitOpError(
          "layered record load result must have exactly one SIMD lane axis");
    laneAxis = axis;
    lanes = factor;
  }
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  if (!sourceField || !physicalPoint || !pointPartition || !plan || !memory ||
      !integer || integer.isSigned() || integer.getWidth() <= 0 ||
      integer.getWidth() >= 8 || !getLogicalOffset().getType().isIndex() ||
      !sameStorageGeometry(getAccess(), sourceField.getAccess()) ||
      getAccess().getMapping() != "grouped_layered" ||
      (getAccess().getForm() != "strided" &&
       getAccess().getForm() != "indexed") ||
      getOrigin().getType().getDomain().getAxisId() != reductionAxis ||
      getOrigin().getType().getDomain().getTail() != "exact" ||
      fieldReduction == field.getAxisIds().asArrayRef().end() ||
      resultReduction != result.getAxisIds().asArrayRef().end() ||
      field.getElementType() != result.getElementType() ||
      field.getShape().size() != result.getShape().size() + 1 ||
      result.getLayout().getCarrier() != "rvv" || laneAxis <= 0 || lanes <= 1 ||
      result.getLayout().getSew() != 8 ||
      (validity != "full" && validity != "tail") ||
      plan.getKind() != "layered" || plan.getProjectionBase() != 0 ||
      plan.getProjectionStride() != 1 || plan.getProjectionRepeat() != 1 ||
      plan.getProjectionExtent() <= 0 || plan.getOffsetAlignment() != 1 ||
      pointPartition.value() != plan.getProjectionExtent() ||
      plan.getRecordElements() != memory.getElements() ||
      plan.getByteOffset() != getAccess().getBitOffset() / 8 ||
      plan.getElementBits() != integer.getWidth() ||
      plan.getGroupSize() != getAccess().getGroupSize() ||
      plan.getLayerSize() != getAccess().getLayerSize() ||
      plan.getGroupSize() <= 0 || plan.getLayerSize() <= 0 ||
      plan.getGroupSize() % plan.getLayerSize() ||
      integer.getWidth() * (plan.getGroupSize() / plan.getLayerSize()) != 8 ||
      getAccess().getBitOffset() % 8 ||
      !exactLeaf(getLeaf(), "rvv", "layered-record-load",
                 "rvv.layered-record-load", "none", tail))
    return emitOpError(
        "layered record load requires one typed reduction subview and one independent record-lane result");

  const size_t reductionPosition = static_cast<size_t>(
      fieldReduction - field.getAxisIds().asArrayRef().begin());
  if (field.getShape()[reductionPosition] < plan.getProjectionExtent())
    return emitOpError(
        "layered record load projection exceeds the encoded field extent");
  size_t resultPosition = 0;
  for (size_t fieldPosition = 0; fieldPosition < field.getShape().size();
       ++fieldPosition) {
    if (fieldPosition == reductionPosition)
      continue;
    if (resultPosition >= result.getShape().size() ||
        field.getAxisIds()[fieldPosition] !=
            result.getAxisIds()[resultPosition] ||
        field.getShape()[fieldPosition] != result.getShape()[resultPosition])
      return emitOpError(
          "layered record load must preserve every non-reduction logical axis");
    ++resultPosition;
  }
  if (resultPosition != result.getShape().size())
    return emitOpError(
        "layered record load result has an extra logical axis");
  auto lane = llvm::find(result.getAxisIds().asArrayRef(), laneAxis);
  const size_t lanePosition = static_cast<size_t>(
      lane - result.getAxisIds().asArrayRef().begin());
  if (result.getShape()[lanePosition] !=
          result.getLayout().getTimeFactors()[lanePosition] * lanes ||
      result.getLayout().getReplicaFactors()[lanePosition] != 1 ||
      result.getLayout().getFragmentFactors()[lanePosition] != 1 ||
      result.getLayout().getLocalFactors()[lanePosition] != 1 ||
      result.getLayout().getVl() < lanes)
    return emitOpError(
        "layered record load lane axis must cover its result through time and SIMD lanes");
  return mlir::success();
}

mlir::LogicalResult RVVLayeredStorageLoadOp::verify() {
  ValueType field = getField().getType();
  LayeredWindowType window = getResult().getType();
  ValueType result = window.getResultType();
  StorageWindowPlanAttr plan = getPlan();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  const int64_t group = plan ? plan.getGroupSize() : 0;
  const int64_t layer = plan ? plan.getLayerSize() : 0;
  const int64_t layers = layer > 0 ? group / layer : 0;
  auto physicalPoint = getOrigin().getDefiningOp<PhysicalPointOp>();
  auto pointPartition =
      physicalPoint
          ? physicalPoint.getPartition().getDefiningOp<mlir::arith::ConstantIndexOp>()
          : mlir::arith::ConstantIndexOp();
  auto resultAxis = llvm::find(result.getAxisIds().asArrayRef(),
                               plan ? plan.getReductionAxis() : 0);
  const int64_t lanes =
      resultAxis == result.getAxisIds().asArrayRef().end()
          ? 0
          : result.getLayout().getLaneFactors()[static_cast<size_t>(
                resultAxis - result.getAxisIds().asArrayRef().begin())];
  auto fieldAxis = llvm::find(field.getAxisIds().asArrayRef(),
                              plan ? plan.getReductionAxis() : 0);
  const int64_t fieldExtent =
      fieldAxis == field.getAxisIds().asArrayRef().end()
          ? 0
          : field.getShape()[static_cast<size_t>(
                fieldAxis - field.getAxisIds().asArrayRef().begin())];
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  if (!sourceField || getAccess() != sourceField.getAccess() || !plan || !integer ||
      integer.isSigned() || getOrigin().getType().getDomain().getAxisId() !=
                                plan.getReductionAxis() ||
      getOrigin().getType().getDomain().getTail() != "exact" ||
      !getWindowIndex().getType().isIndex() ||
      getAccess().getForm() != "indexed" ||
      getAccess().getMapping() != "grouped_layered" || group <= 0 ||
      layer <= 0 || group % layer || layers <= 1 || lanes <= 1 ||
      !pointPartition || pointPartition.value() % group ||
      layer % lanes || integer.getWidth() * layers != 8 ||
      plan.getKind() != "layered" ||
      plan.getByteOffset() != getAccess().getBitOffset() / 8 ||
      plan.getElementBits() != integer.getWidth() ||
      plan.getGroupSize() != getAccess().getGroupSize() ||
      plan.getLayerSize() != getAccess().getLayerSize() ||
      plan.getProjectionBase() < 0 || plan.getProjectionExtent() <= 0 ||
      plan.getProjectionBase() % group || plan.getProjectionExtent() % group ||
      fieldExtent <= 0 || plan.getProjectionBase() > fieldExtent ||
      plan.getProjectionExtent() > fieldExtent - plan.getProjectionBase() ||
      getAccess().getBitOffset() % 8 ||
      (getAccess().getOrder() != "lo_first" &&
       getAccess().getOrder() != "hi_first") ||
      window.getFieldType() != field || window.getReductionAxis() !=
                                            plan.getReductionAxis() ||
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
  auto load = getWindow().getDefiningOp<RVVLayeredStorageLoadOp>();
  StorageWindowPlanAttr plan = load ? load.getPlan() : StorageWindowPlanAttr();
  const int64_t expectedPhysicalLayer =
      plan ? plan.getPhysicalLayerBase() +
                 getLayer() * plan.getPhysicalLayerStep()
           : -1;
  auto integer =
      mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  const int64_t width = integer ? integer.getWidth() : 0;
  const int64_t expectedShift =
      plan ? plan.getShiftBase() + getLayer() * plan.getShiftStep() : -1;
  const int64_t expectedMask =
      integer && (expectedPhysicalLayer + 1) * width != 8
          ? (int64_t(1) << width) - 1
          : 0;
  llvm::StringRef expectedInstruction =
      expectedShift > 0
          ? (expectedMask > 0 ? "rvv.layered-storage-decode.shift-mask"
                              : "rvv.layered-storage-decode.shift")
          : (expectedMask > 0 ? "rvv.layered-storage-decode.mask"
                              : "rvv.layered-storage-decode.identity");
  if (getLayer() < 0 || getLayer() >= window.getLayers() || !load || !plan ||
      plan.getKind() != "layered" ||
      getPhysicalLayer() != expectedPhysicalLayer ||
      !integer || integer.isSigned() || width <= 0 || width > 8 ||
      width != plan.getElementBits() ||
      getShiftAmount() != expectedShift || getMaskValue() != expectedMask ||
      result != window.getResultType() ||
      !exactLeaf(getLeaf(), "rvv", "layered-storage-decode",
                 expectedInstruction, "none", tail))
    return emitOpError(
        "layered storage decode requires one selected layer of its typed window");
  return mlir::success();
}

mlir::LogicalResult RVVReplicaStorageLoadOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto sourceField = getField().getDefiningOp<FieldOp>();
  auto storageLoad = sourceField ? sourceLoad(sourceField.getOwner()) : LoadOp();
  auto memory = storageLoad ? storageLoad.getRegion().getType() : MemDescType();
  mlir::Type fieldElement = field.getElementType();
  auto fieldInteger = mlir::dyn_cast<mlir::IntegerType>(fieldElement);
  const int64_t fieldBits = elementBitWidth(fieldElement);
  mlir::Type baseType = getLogicalBase().getType();
  const bool scalarBase = baseType.isIndex() || mlir::isa<mlir::IntegerType>(baseType);
  StorageWindowPlanAttr plan = getPlan();
  const int64_t laneAxis = plan.getReductionAxis();
  const int64_t logicalSpan = plan.getProjectionExtent();
  const int64_t logicalBaseMultiple = plan.getOffsetAlignment();
  const int64_t recordRank = getRecordRank();
  auto resultAxis = llvm::find(result.getAxisIds().asArrayRef(), laneAxis);
  auto parts = physicalPartCount(result);
  const bool natural = plan.getKind() == "unit";
  const bool layered = plan.getKind() == "layered";
  const bool interleavedNatural = plan.getKind() == "interleaved_natural";
  const bool interleavedJoined = plan.getKind() == "interleaved_joined";
  const bool interleaved = interleavedNatural || interleavedJoined;
  const int64_t group = plan.getGroupSize();
  const int64_t layer = plan.getLayerSize();
  const int64_t layers = layered && layer > 0 ? group / layer : 1;
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  llvm::StringRef instruction = natural
                                    ? "rvv.replica-storage-load.natural"
                                : layered
                                    ? "rvv.replica-storage-load.layered"
                                : interleavedNatural
                                    ? "rvv.replica-storage-load.interleaved-natural"
                                    : "rvv.replica-storage-load.interleaved-joined";
  auto ownerEncoding = mlir::dyn_cast<weft::kernel::EncodingType>(
      sourceField ? elementOf(sourceField.getOwner().getType()) : mlir::Type());
  int64_t interleave =
      sourceField ? localPackInterleaveRows(sourceField.getOwner()) : 0;
  if (interleave <= 0 && sourceField)
    interleave = derivedInterleaveRows(*this, ownerEncoding);
  if (!sourceField || !storageLoad || !memory ||
      !sameStorageGeometry(getAccess(), sourceField.getAccess()) ||
      fieldBits <= 0 || (fieldInteger && fieldInteger.isSignless()) || !scalarBase ||
      laneAxis <= 0 || plan.getProjectionBase() != 0 ||
      plan.getProjectionStride() != 1 || plan.getProjectionRepeat() != 1 ||
      plan.getRecordElements() != memory.getElements() ||
      plan.getByteOffset() != getAccess().getBitOffset() / 8 ||
      plan.getElementBits() != fieldBits ||
      (natural && getAccess().getMapping() != "natural") ||
      (layered && getAccess().getMapping() != "grouped_layered") ||
      (interleavedNatural && getAccess().getMapping() != "natural") ||
      (interleavedJoined && getAccess().getMapping() != "joined") ||
      resultAxis == result.getAxisIds().asArrayRef().end() ||
      field.getElementType() != result.getElementType() ||
      result.getLayout().getCarrier() != "rvv" || !parts || *parts <= 0 ||
      logicalBaseMultiple <= 0 || logicalSpan <= 0 ||
      getWindowOffsets().empty() ||
      recordRank < 0 || recordRank > static_cast<int64_t>(field.getShape().size()) ||
      getRecordCoordinatesForWindow().size() !=
          getWindowOffsets().size() * static_cast<size_t>(recordRank) ||
      getWindowForPart().size() != static_cast<size_t>(*parts) ||
      getLayerForPart().size() != static_cast<size_t>(*parts) ||
      getPhysicalLayerForPart().size() != static_cast<size_t>(*parts) ||
      getShiftOffsetForPart().size() != static_cast<size_t>(*parts) ||
      getShiftBaseFactorForPart().size() != static_cast<size_t>(*parts) ||
      getMaskValueForPart().size() != static_cast<size_t>(*parts) ||
      (!natural && !layered && !interleaved) ||
      (validity != "full" && validity != "tail") ||
      !exactLeaf(getLeaf(), "rvv", "replica-storage-load", instruction, "none",
                 tail))
    return emitOpError()
           << "replica storage load requires one scalar base and a closed field-to-register window map; field="
           << field << ", result=" << result << ", source_field="
           << static_cast<bool>(sourceField) << ", access=" << getAccess()
           << ", source_access="
           << (sourceField ? sourceField.getAccess() : AccessAttr())
           << ", storage_load=" << static_cast<bool>(storageLoad)
           << ", memory=" << memory << ", interleave=" << interleave
           << ", result_axis="
           << (resultAxis == result.getAxisIds().asArrayRef().end()
                   ? -1
                   : static_cast<int64_t>(resultAxis -
                                          result.getAxisIds().asArrayRef().begin()))
           << ", plan=" << plan
           << ", window_offsets=" << getWindowOffsets()
           << ", record_rank=" << recordRank
           << ", record_coordinates_for_window="
           << getRecordCoordinatesForWindow()
           << ", window_for_part=" << getWindowForPart()
           << ", layer_for_part=" << getLayerForPart()
           << ", physical_layer_for_part=" << getPhysicalLayerForPart()
           << ", shift_offset_for_part=" << getShiftOffsetForPart()
           << ", shift_base_factor_for_part=" << getShiftBaseFactorForPart()
           << ", mask_value_for_part=" << getMaskValueForPart()
           << ", parts=" << (parts ? *parts : -1)
           << ", base_type=" << baseType << ", leaf=" << getLeaf();

  const size_t lanePosition = static_cast<size_t>(
      resultAxis - result.getAxisIds().asArrayRef().begin());
  const int64_t primaryLanes =
      result.getLayout().getLaneFactors()[lanePosition];
  auto physicalLanes =
      checkedPositiveProduct(result.getLayout().getLaneFactors().asArrayRef());
  if (primaryLanes <= 1 || !physicalLanes || *physicalLanes <= 1 ||
      result.getShape()[lanePosition] !=
          primaryLanes * result.getLayout().getTimeFactors()[lanePosition] ||
      result.getLayout().getReplicaFactors()[lanePosition] != 1 ||
      result.getLayout().getFragmentFactors()[lanePosition] != 1 ||
      result.getLayout().getLocalFactors()[lanePosition] != 1)
    return emitOpError(
        "replica storage load lane axis must be one complete SIMD window");
  for (size_t position = 0; position < result.getShape().size(); ++position) {
    const int64_t time = result.getLayout().getTimeFactors()[position];
    const int64_t lane = result.getLayout().getLaneFactors()[position];
    const int64_t replica = result.getLayout().getReplicaFactors()[position];
    const int64_t axis = result.getAxisIds()[position];
    auto fieldAxis = llvm::find(field.getAxisIds().asArrayRef(), axis);
    const bool preservedFieldAxis =
        fieldAxis != field.getAxisIds().asArrayRef().end() &&
        field.getShape()[static_cast<size_t>(
            fieldAxis - field.getAxisIds().asArrayRef().begin())] ==
            result.getShape()[position];
    int64_t factors[] = {time, lane, replica};
    auto represented = checkedPositiveProduct(factors);
    if ((lane > 1 && replica != 1) ||
        (time > 1 && replica > 1) || !represented ||
        (result.getShape()[position] > 0 &&
         *represented != result.getShape()[position]) ||
        (result.getShape()[position] <= 0 &&
         (validity != "tail" ||
          (*represented != 1 && !preservedFieldAxis))) ||
        result.getLayout().getFragmentFactors()[position] != 1 ||
        result.getLayout().getLocalFactors()[position] != 1)
      return emitOpError(
          "replica storage load must preserve every logical axis as lane, issue time, or register replicas");
  }
  auto streams = checkedPositiveProduct(
      result.getLayout().getTimeFactors().asArrayRef());
  if (!streams || *streams <= 0)
    return emitOpError(
        "replica storage load must define a positive issue-time decomposition");
  if (*parts % *streams)
    return emitOpError(
        "replica storage load has an incomplete stream/replica product");
  auto loadLanes = checkedPositiveProduct(
      result.getLayout().getLaneFactors().asArrayRef());
  if (!loadLanes || *loadLanes <= 1)
    return emitOpError(
        "replica storage load has no positive typed load-lane extent");
  for (size_t window = 0; window < getWindowOffsets().size(); ++window) {
    for (int64_t position = 0; position < recordRank; ++position) {
      const int64_t coordinate = getRecordCoordinatesForWindow()[
          window * static_cast<size_t>(recordRank) + static_cast<size_t>(position)];
      const int64_t extent = field.getShape()[static_cast<size_t>(position)];
      int64_t width = 1;
      auto mapped = llvm::find(result.getAxisIds().asArrayRef(),
                               field.getAxisIds()[static_cast<size_t>(position)]);
      if (mapped != result.getAxisIds().asArrayRef().end())
        width = result.getLayout().getLaneFactors()[static_cast<size_t>(
            mapped - result.getAxisIds().asArrayRef().begin())];
      if (coordinate < 0 || width <= 0 ||
          (extent > 0 && coordinate > extent - width))
        return emitOpError()
               << "replica storage load window has an out-of-range record coordinate; window="
               << window << ", position=" << position
               << ", coordinate=" << coordinate << ", width=" << width
               << ", extent=" << extent << ", record_rank=" << recordRank
               << ", field=" << field << ", result=" << result;
    }
  }
  for (int64_t window : getWindowForPart())
    if (window < 0 ||
        window >= static_cast<int64_t>(getWindowOffsets().size()))
      return emitOpError(
          "replica storage load part references an invalid source window");

  if (interleaved) {
    if (interleave <= 0 || recordRank != 1 ||
        result.getAxisIds()[lanePosition] != field.getAxisIds()[0] ||
        result.getShape()[lanePosition] > interleave ||
        logicalSpan != result.getShape()[lanePosition] ||
        (getAccess().getForm() != "strided" &&
         getAccess().getForm() != "indexed") ||
        (interleavedNatural && fieldBits != 8 && fieldBits != 16 &&
         fieldBits != 32) ||
        (interleavedJoined &&
         (!fieldInteger || !fieldInteger.isUnsigned() || fieldBits > 8 ||
          getAccess().getGroupSize() <= 0 || getAccess().getJoinFields() <= 1 ||
          getAccess().getJoinLowBits() <= 0 ||
          getAccess().getJoinLowBits() >= fieldBits ||
          getAccess().getJoinRole() < 0 ||
          getAccess().getJoinRole() >= getAccess().getJoinFields() ||
          (getAccess().getOrder() != "lo_first" &&
           getAccess().getOrder() != "hi_first"))))
      return emitOpError(
          "interleaved replica storage load requires one explicit record-axis projection and closed byte assembly");
  } else if (natural) {
    if (getAccess().getForm() != "indexed" || getAccess().getBitOffset() % 8 ||
        !fieldInteger ||
        (fieldBits != 8 && fieldBits != 16 && fieldBits != 32) ||
        result.getLayout().getSew() != fieldBits)
      return emitOpError(
          "natural replica storage load requires byte-addressable indexed source elements");
  } else {
    if (getAccess().getForm() != "indexed" || group <= 0 || layer <= 0 ||
        group % layer || layers <= 1 || layer % *physicalLanes ||
        !fieldInteger || fieldInteger.isSigned() || fieldBits * layers > 8 ||
        result.getLayout().getSew() != 8 || getAccess().getBitOffset() % 8 ||
        (getAccess().getOrder() != "lo_first" &&
         getAccess().getOrder() != "hi_first") ||
        group % logicalSpan)
      return emitOpError(
          "layered replica storage load requires closed grouped/layered byte geometry");
  }


  for (int64_t offset : getWindowOffsets()) {
    const bool bounded = natural || interleaved
                             ? offset >= 0 &&
                                   offset <= logicalSpan - *loadLanes
                             : offset >= 0 &&
                                   offset <= layer - *physicalLanes;
    if (!bounded)
      return emitOpError("replica storage load window offset is out of bounds");
  }
  for (auto [window, storageLayer] :
       llvm::zip(getWindowForPart(), getLayerForPart())) {
    if (window < 0 || window >= static_cast<int64_t>(getWindowOffsets().size()) ||
        storageLayer < 0 || storageLayer >= layers ||
        ((natural || interleaved) && storageLayer != 0))
      return emitOpError(
          "replica storage load part map references an invalid window or layer");
  }
  const int64_t elementBits = fieldBits;
  const int64_t fullMask = (int64_t{1} << elementBits) - 1;
  const bool dynamicBase =
      layered && logicalBaseMultiple % group != 0;
  for (auto [logicalLayer, physicalLayer, shiftOffset, shiftBaseFactor,
             maskValue] :
       llvm::zip(getLayerForPart(), getPhysicalLayerForPart(),
                 getShiftOffsetForPart(), getShiftBaseFactorForPart(),
                 getMaskValueForPart())) {
    if (natural || interleaved) {
      if (physicalLayer != 0 || shiftOffset != 0 || shiftBaseFactor != 0 ||
          maskValue != 0)
        return emitOpError(
            "natural replica storage load cannot carry a physical layer");
      continue;
    }
    const int64_t expectedPhysical =
        dynamicBase ? 0
                    : (getAccess().getOrder() == "lo_first"
                           ? logicalLayer
                           : layers - 1 - logicalLayer);
    const int64_t expectedOffset =
        dynamicBase
            ? (getAccess().getOrder() == "lo_first"
                   ? logicalLayer * elementBits
                   : (layers - 1 - logicalLayer) * elementBits)
            : expectedPhysical * elementBits;
    const int64_t expectedFactor =
        dynamicBase
            ? (getAccess().getOrder() == "lo_first" ? elementBits
                                                      : -elementBits)
            : 0;
    const int64_t expectedMask =
        dynamicBase || (expectedPhysical + 1) * elementBits != 8 ? fullMask : 0;
    if (logicalLayer < 0 || logicalLayer >= layers ||
        physicalLayer != expectedPhysical || shiftOffset != expectedOffset ||
        shiftBaseFactor != expectedFactor || maskValue != expectedMask)
      return emitOpError(
          "replica storage physical layer disagrees with the selected base proof");
  }
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
  bool explicitChain = false;
  if (!loopCarried) {
    mlir::Value first = getAccumulator();
    while (auto previous = first.getDefiningOp<RVVWidenAccumulateOp>()) {
      if (previous.getOperation()->getBlock() != getOperation()->getBlock() ||
          previous.getReductionAxis() != getReductionAxis()) {
        first = {};
        break;
      }
      first = previous.getAccumulator();
    }
    auto splatSeed = first ? first.getDefiningOp<RVVSplatOp>() : RVVSplatOp();
    auto productSeed =
        first ? first.getDefiningOp<RVVWidenMultiplyOp>()
              : RVVWidenMultiplyOp();
    explicitChain =
        (splatSeed && splatSeed.getOperation()->getBlock() ==
                          getOperation()->getBlock()) ||
        (productSeed && productSeed.getOperation()->getBlock() ==
                            getOperation()->getBlock());
    mlir::Value current = getResult();
    while (explicitChain) {
      if (!current.hasOneUse()) {
        explicitChain = false;
        break;
      }
      mlir::Operation *user = *current.getUsers().begin();
      if (auto next = mlir::dyn_cast<RVVWidenAccumulateOp>(user)) {
        if (next.getAccumulator() != current ||
            next.getReductionAxis() != getReductionAxis() ||
            next.getOperation()->getBlock() != getOperation()->getBlock()) {
          explicitChain = false;
          break;
        }
        current = next.getResult();
        continue;
      }
      if (auto capture = mlir::dyn_cast<RVVPartialCaptureOp>(user))
        explicitChain = capture.getInput() == current;
      else if (auto finalize = mlir::dyn_cast<RVVFinalizeWidenDotOp>(user))
        explicitChain = finalize.getPartial() == current &&
                        finalize.getReductionAxis() == getReductionAxis();
      else
        explicitChain = false;
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
      !operandMapsToPartial(lhs) || !operandMapsToPartial(rhs) ||
      (!loopCarried && !explicitChain) ||
      !exactLeaf(getLeaf(), "rvv", "widen-accumulate",
                 "rvv.vwmacc.partial", "none", "agnostic"))
    return emitOpError(
        "RVV widened accumulation requires matching single-window operands and one closed partial chain");
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
  auto explicitChain = getPartial().getDefiningOp<RVVWidenAccumulateOp>();
  auto reduction = llvm::find(partial.getAxisIds().asArrayRef(),
                              getReductionAxis());
  const size_t reductionPosition =
      reduction == partial.getAxisIds().asArrayRef().end()
          ? 0
          : static_cast<size_t>(reduction -
                                partial.getAxisIds().asArrayRef().begin());
  bool completeMapping = partial.getLayout().getCarrier() == "rvv" &&
                         (loop || explicitChain) &&
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
  llvm::StringRef instruction =
      partialElement && partialElement.getWidth() == 16
          ? llvm::StringRef("rvv.vwredsum.partial")
          : llvm::StringRef("rvv.vredsum.partial");
  if (!partialElement || !partialElement.isSigned() ||
      (partialElement.getWidth() != 16 && partialElement.getWidth() != 32) ||
      !resultElement ||
      !resultElement.isSigned() || resultElement.getWidth() != 32 ||
      !completeMapping || !legalResult || !kernel ||
      !supportsRVVLayout(kernel.getTarget(), partial.getLayout()) ||
      !exactLeaf(getLeaf(), "rvv", "finalize-widen-dot",
                 instruction, "none", "exact"))
    return emitOpError(
        "final widened dot reduction must remove one lane axis and preserve every free replica");
  return mlir::success();
}

mlir::LogicalResult RVVPartialSetOp::verify() {
  PartialSetType result = getResult().getType();
  ValueType partial = result.getPartialType();
  auto partialElement =
      mlir::dyn_cast<mlir::IntegerType>(partial.getElementType());
  auto positionOf = [&](ValueType value) -> std::optional<size_t> {
    auto found = llvm::find(value.getAxisIds().asArrayRef(), getReductionAxis());
    if (found == value.getAxisIds().asArrayRef().end())
      return std::nullopt;
    return static_cast<size_t>(found - value.getAxisIds().asArrayRef().begin());
  };
  auto partialAxis = positionOf(partial);
  if (getLhs().empty() || getRhs().empty() || !partialElement ||
      !partialElement.isSigned() || !partialAxis ||
      getLhsOperands().size() != static_cast<size_t>(result.getSlots()) ||
      getRhsOperands().size() != static_cast<size_t>(result.getSlots()) ||
      getLhsParts().size() != static_cast<size_t>(result.getSlots()) ||
      getRhsParts().size() != static_cast<size_t>(result.getSlots()) ||
      getLhsLaneOffsets().size() != static_cast<size_t>(result.getSlots()) ||
      getRhsLaneOffsets().size() != static_cast<size_t>(result.getSlots()) ||
      getReductionAxis() != result.getReductionAxis() ||
      getOwnerDomainId() < 0 || getBirthId() < 0 ||
      getLifetimeEndDomainId() < getOwnerDomainId() ||
      !exactLeaf(getLeaf(), "rvv", "partial-set", "rvv.partial-set",
                 "none", "exact"))
    return emitOpError(
        "RVV partial set requires paired stream operands and one closed result topology");
  auto preservesPartialAxes = [&](ValueType operand) {
    if (partial.getAxisIds().size() == 1)
      return true;
    if (partial.getAxisIds() != operand.getAxisIds() ||
        partial.getShape().size() != operand.getShape().size())
      return false;
    for (size_t position = 0; position < partial.getShape().size(); ++position) {
      if (partial.getAxisIds()[position] == getReductionAxis()) {
        if (partial.getShape()[position] !=
                operand.getLayout().getLaneFactors()[position] ||
            partial.getLayout().getTimeFactors()[position] != 1 ||
            partial.getLayout().getLaneFactors()[position] !=
                operand.getLayout().getLaneFactors()[position] ||
            partial.getLayout().getReplicaFactors()[position] != 1)
          return false;
        continue;
      }
      if (partial.getShape()[position] != operand.getShape()[position] ||
          operand.getLayout().getTimeFactors()[position] != 1 ||
          partial.getLayout().getTimeFactors()[position] != 1 ||
          partial.getLayout().getLaneFactors()[position] !=
              operand.getLayout().getLaneFactors()[position] ||
          partial.getLayout().getReplicaFactors()[position] !=
              operand.getLayout().getReplicaFactors()[position])
        return false;
    }
    return checkedPositiveProduct(
               partial.getLayout().getLaneFactors().asArrayRef()) ==
           checkedPositiveProduct(
               operand.getLayout().getLaneFactors().asArrayRef());
  };
  for (int64_t slot = 0; slot < result.getSlots(); ++slot) {
    const int64_t lhsOperand = getLhsOperands()[slot];
    const int64_t rhsOperand = getRhsOperands()[slot];
    if (lhsOperand < 0 || lhsOperand >= static_cast<int64_t>(getLhs().size()) ||
        rhsOperand < 0 || rhsOperand >= static_cast<int64_t>(getRhs().size()))
      return emitOpError(
          "RVV partial set slot references an absent operand");
    ValueType lhs = mlir::cast<ValueType>(getLhs()[lhsOperand].getType());
    ValueType rhs = mlir::cast<ValueType>(getRhs()[rhsOperand].getType());
    auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
    auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
    auto lhsAxis = positionOf(lhs);
    auto rhsAxis = positionOf(rhs);
    auto lhsParts = physicalPartCount(lhs);
    auto rhsParts = physicalPartCount(rhs);
    auto lhsLanes = checkedPositiveProduct(
        lhs.getLayout().getLaneFactors().asArrayRef());
    auto rhsLanes = checkedPositiveProduct(
        rhs.getLayout().getLaneFactors().asArrayRef());
    const int64_t sliceLanes =
        partial.getLayout().getLaneFactors()[*partialAxis];
    int64_t lhsSliceLMUL =
        lhsLanes && *lhsLanes > 0 &&
                (lhs.getLayout().getLmulEighths() * sliceLanes) % *lhsLanes == 0
            ? lhs.getLayout().getLmulEighths() * sliceLanes / *lhsLanes
            : -1;
    int64_t rhsSliceLMUL =
        rhsLanes && *rhsLanes > 0 &&
                (rhs.getLayout().getLmulEighths() * sliceLanes) % *rhsLanes == 0
            ? rhs.getLayout().getLmulEighths() * sliceLanes / *rhsLanes
            : -1;
    // RVV group extraction cannot name a fractional destination register.
    // A sub-register slice therefore uses an m1 carrier and records its true
    // lane offset; this is the same typed contract used by RVVWidenDotOp.
    if (lhsSliceLMUL > 0 && lhsSliceLMUL < 8 &&
        lhs.getLayout().getLmulEighths() > lhsSliceLMUL)
      lhsSliceLMUL = 8;
    if (rhsSliceLMUL > 0 && rhsSliceLMUL < 8 &&
        rhs.getLayout().getLmulEighths() > rhsSliceLMUL)
      rhsSliceLMUL = 8;
    const llvm::StringRef instruction = getMultiplyInstruction();
    const bool exactMultiply =
        (instruction == "rvv.vwmul.vv" && lhsElement && rhsElement &&
         lhsElement.isSigned() && rhsElement.isSigned()) ||
        (instruction == "rvv.vwmul.vv.reinterpret-rhs" && lhsElement &&
         rhsElement && lhsElement.isSigned() && rhsElement.isUnsigned() &&
         rhsElement.getWidth() < rhs.getLayout().getSew()) ||
        (instruction == "rvv.vwmul.vv.reinterpret-lhs" && lhsElement &&
         rhsElement && lhsElement.isUnsigned() && rhsElement.isSigned() &&
         lhsElement.getWidth() < lhs.getLayout().getSew()) ||
        (instruction == "rvv.vwmulsu.vv" && lhsElement && rhsElement &&
         lhsElement.isSigned() && rhsElement.isUnsigned()) ||
        (instruction == "rvv.vwmulsu.vv.swap" && lhsElement && rhsElement &&
         lhsElement.isUnsigned() && rhsElement.isSigned());
    if (!lhsElement || !rhsElement || lhsElement.isSignless() ||
        rhsElement.isSignless() ||
        (!lhsElement.isSigned() && !rhsElement.isSigned()) || !lhsAxis ||
        !rhsAxis || lhs.getLayout().getCarrier() != "rvv" ||
        rhs.getLayout().getCarrier() != "rvv" ||
        std::max<unsigned>(8, lhsElement.getWidth()) !=
            std::max<unsigned>(8, rhsElement.getWidth()) ||
        partialElement.getWidth() !=
            2 * std::max<unsigned>(8, lhsElement.getWidth()) ||
        lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
        lhs.getLayout().getTimeFactors()[*lhsAxis] <= 0 ||
        rhs.getLayout().getTimeFactors()[*rhsAxis] <= 0 ||
        sliceLanes <= 0 || !lhsLanes || !rhsLanes ||
        sliceLanes > *lhsLanes || sliceLanes > *rhsLanes ||
        getLhsLaneOffsets()[slot] < 0 || getRhsLaneOffsets()[slot] < 0 ||
        getLhsLaneOffsets()[slot] % sliceLanes ||
        getRhsLaneOffsets()[slot] % sliceLanes ||
        getLhsLaneOffsets()[slot] > *lhsLanes - sliceLanes ||
        getRhsLaneOffsets()[slot] > *rhsLanes - sliceLanes ||
        partial.getLayout().getSew() != 2 * lhs.getLayout().getSew() ||
        lhsSliceLMUL <= 0 || lhsSliceLMUL != rhsSliceLMUL ||
        partial.getLayout().getLmulEighths() != 2 * lhsSliceLMUL ||
        result.getTermsPerSlot() !=
            partial.getLayout().getLaneFactors()[*partialAxis] ||
        !preservesPartialAxes(lhs) || !preservesPartialAxes(rhs) ||
        !lhsParts || !rhsParts || getLhsParts()[slot] < 0 ||
        getRhsParts()[slot] < 0 || getLhsParts()[slot] >= *lhsParts ||
        getRhsParts()[slot] >= *rhsParts || !exactMultiply)
      return emitOpError(
          "RVV partial set operand pair does not produce the declared widened slots");
  }
  return mlir::success();
}

mlir::LogicalResult RVVPartialCaptureOp::verify() {
  ValueType input = getInput().getType();
  PartialSetType result = getResult().getType();
  auto axis = llvm::find(input.getAxisIds().asArrayRef(),
                         result.getReductionAxis());
  auto parts = physicalPartCount(input);
  llvm::StringRef tail = input.getLayout().getValidity() == "tail"
                             ? llvm::StringRef("agnostic")
                             : llvm::StringRef("exact");
  if (input.getLayout().getCarrier() != "rvv" || !parts || *parts != 1 ||
      axis == input.getAxisIds().asArrayRef().end() ||
      result.getPartialType() != input || result.getSlots() != 1 ||
      result.getTermsPerSlot() !=
          input.getLayout().getLaneFactors()[static_cast<size_t>(
              axis - input.getAxisIds().asArrayRef().begin())] ||
      result.getResourceGroups() != input.getLayout().getRegisterGroups() ||
      getOwnerDomainId() < 0 || getBirthId() < 0 ||
      getLifetimeEndDomainId() < getOwnerDomainId() ||
      getLeaf().getOperandGroups() != input.getLayout().getRegisterGroups() ||
      getLeaf().getResultGroups() != result.getResourceGroups() ||
      !exactLeaf(getLeaf(), "rvv", "partial-capture",
                 "rvv.partial-capture", "none", tail))
    return emitOpError(
        "RVV partial capture requires one complete register-resident lane partial");
  return mlir::success();
}

mlir::LogicalResult RVVPartialRepackOp::verify() {
  PartialSetType input = getInput().getType();
  PartialSetType result = getResult().getType();
  ValueType source = input.getPartialType();
  ValueType target = result.getPartialType();
  auto sourceElement =
      mlir::dyn_cast<mlir::IntegerType>(source.getElementType());
  auto targetElement =
      mlir::dyn_cast<mlir::IntegerType>(target.getElementType());
  auto sourceAxis = llvm::find(source.getAxisIds().asArrayRef(),
                               input.getReductionAxis());
  auto targetAxis = llvm::find(target.getAxisIds().asArrayRef(),
                               input.getReductionAxis());
  const int64_t split = static_cast<int64_t>(getSplitFactor());
  bool compatibleAxes = source.getAxisIds() == target.getAxisIds() &&
                        source.getShape().size() == target.getShape().size();
  if (compatibleAxes && sourceAxis != source.getAxisIds().asArrayRef().end() &&
      targetAxis != target.getAxisIds().asArrayRef().end()) {
    const size_t sourcePosition = static_cast<size_t>(
        sourceAxis - source.getAxisIds().asArrayRef().begin());
    const size_t targetPosition = static_cast<size_t>(
        targetAxis - target.getAxisIds().asArrayRef().begin());
    compatibleAxes &= sourcePosition == targetPosition;
    for (size_t position = 0; position < source.getShape().size(); ++position) {
      if (position == sourcePosition) {
        compatibleAxes &= source.getShape()[position] ==
                              target.getShape()[position] * split &&
                          source.getLayout().getLaneFactors()[position] ==
                              target.getLayout().getLaneFactors()[position] * split &&
                          source.getLayout().getTimeFactors()[position] == 1 &&
                          target.getLayout().getTimeFactors()[position] == 1 &&
                          source.getLayout().getReplicaFactors()[position] == 1 &&
                          target.getLayout().getReplicaFactors()[position] == 1;
      } else {
        compatibleAxes &= source.getShape()[position] == target.getShape()[position] &&
                          source.getLayout().getTimeFactors()[position] ==
                              target.getLayout().getTimeFactors()[position] &&
                          source.getLayout().getLaneFactors()[position] ==
                              target.getLayout().getLaneFactors()[position] &&
                          source.getLayout().getReplicaFactors()[position] ==
                              target.getLayout().getReplicaFactors()[position];
      }
    }
  } else {
    compatibleAxes = false;
  }
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  if (split <= 1 || !sourceElement || !targetElement ||
      sourceElement != targetElement || !compatibleAxes ||
      source.getLayout().getCarrier() != "rvv" ||
      target.getLayout().getCarrier() != "rvv" ||
      source.getLayout().getSew() != target.getLayout().getSew() ||
      source.getLayout().getLmulEighths() !=
          target.getLayout().getLmulEighths() * split ||
      source.getLayout().getVl() != target.getLayout().getVl() * split ||
      result.getReductionAxis() != input.getReductionAxis() ||
      result.getSlots() != input.getSlots() * split ||
      result.getTermsPerSlot() * split != input.getTermsPerSlot() ||
      result.getResourceGroups() != input.getResourceGroups() || !kernel ||
      !supportsRVVLayout(kernel.getTarget(), source.getLayout()) ||
      !supportsRVVLayout(kernel.getTarget(), target.getLayout()) ||
      !exactLeaf(getLeaf(), "rvv", "partial-repack",
                 "rvv.partial-repack.split", "none", "exact") ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({input.getReductionAxis(), split}))
    return emitOpError(
        "RVV partial repack requires one exact lane/LMUL split preserving all logical terms and resources");
  return mlir::success();
}

mlir::LogicalResult RVVPartialMergeOp::verify() {
  PartialSetType result = getResult().getType();
  if (getInputs().size() < 2)
    return emitOpError("RVV partial merge requires at least two input sets");
  int64_t slots = 0;
  int64_t terms = 0;
  int64_t resources = 0;
  for (mlir::Value inputValue : getInputs()) {
    PartialSetType input = mlir::cast<PartialSetType>(inputValue.getType());
    if (input.getPartialType() != result.getPartialType() ||
        input.getReductionAxis() != result.getReductionAxis())
      return emitOpError(
          "RVV partial merge inputs must share one partial layout and reduction axis");
    slots += input.getSlots();
    terms += input.getSlots() * input.getTermsPerSlot();
    resources += input.getResourceGroups();
  }
  const int64_t resultGroups =
      result.getPartialType().getLayout().getRegisterGroups();
  if (getTopology() != "pairwise" || result.getSlots() != 1 ||
      result.getTermsPerSlot() != terms ||
      result.getResourceGroups() != resultGroups ||
      getLeaf().getOperandGroups() != resources ||
      getLeaf().getResultGroups() != resultGroups ||
      !exactLeaf(getLeaf(), "rvv", "partial-merge", "rvv.partial-merge",
                 "none", "exact") ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({result.getReductionAxis(),
                                   static_cast<int64_t>(getInputs().size()),
                                   slots, terms}))
    return emitOpError(
        "RVV partial merge must pairwise combine every compatible slot into one typed partial");
  return mlir::success();
}

mlir::LogicalResult RVVPartialReduceOp::verify() {
  PartialSetType input = getInput().getType();
  PartialSetType result = getResult().getType();
  ValueType inputPartial = input.getPartialType();
  ValueType resultPartial = result.getPartialType();
  auto inputElement =
      mlir::dyn_cast<mlir::IntegerType>(inputPartial.getElementType());
  auto resultElement =
      mlir::dyn_cast<mlir::IntegerType>(resultPartial.getElementType());
  auto resultAxis = llvm::find(resultPartial.getAxisIds().asArrayRef(),
                               input.getReductionAxis());
  if (!inputElement || !resultElement || !inputElement.isSigned() ||
      !resultElement.isSigned() ||
      (inputElement.getWidth() != 16 && inputElement.getWidth() != 32) ||
      resultElement.getWidth() != 32 ||
      result.getReductionAxis() != input.getReductionAxis() ||
      result.getSlots() != input.getSlots() ||
      result.getTermsPerSlot() != input.getTermsPerSlot() ||
      resultAxis == resultPartial.getAxisIds().asArrayRef().end())
    return emitOpError(
        "RVV partial reduction requires one topology-preserving i16/i32 to i32 reduction");
  const size_t position = static_cast<size_t>(
      resultAxis - resultPartial.getAxisIds().asArrayRef().begin());
  llvm::StringRef instruction = inputElement.getWidth() == 16
                                    ? "rvv.partial-reduce.widen"
                                    : "rvv.partial-reduce";
  if (resultPartial.getShape().size() != 1 ||
      resultPartial.getShape()[position] != 1 ||
      resultPartial.getLayout().getLaneFactors()[position] != 1 ||
      resultPartial.getLayout().getSew() != 32 ||
      resultPartial.getLayout().getLmulEighths() != 8 ||
      resultPartial.getLayout().getRegisterGroups() != 1 ||
      !exactLeaf(getLeaf(), "rvv", "partial-reduce", instruction, "none",
                 "exact"))
    return emitOpError(
        "RVV partial reduction result must retain a singleton typed reduction slot; result=")
           << resultPartial << ", instruction=" << instruction
           << ", leaf=" << getLeaf();
  return mlir::success();
}

mlir::LogicalResult RVVPartialScaleCombineOp::verify() {
  PartialSetType input = getInput().getType();
  PartialSetType result = getResult().getType();
  ValueType partial = input.getPartialType();
  auto element = mlir::dyn_cast<mlir::IntegerType>(partial.getElementType());
  if (!element || !element.isSigned() || element.getWidth() != 32 ||
      getScales().size() != static_cast<size_t>(input.getSlots()) ||
      getScaleReplicas().size() != static_cast<size_t>(input.getSlots()) ||
      getSlotOrder().size() != static_cast<size_t>(input.getSlots()) ||
      result.getPartialType() != partial ||
      result.getReductionAxis() != input.getReductionAxis() ||
      result.getSlots() <= 0 || input.getSlots() % result.getSlots() ||
      result.getTermsPerSlot() !=
          input.getTermsPerSlot() * input.getSlots() / result.getSlots() ||
      !exactLeaf(getLeaf(), "rvv", "partial-scale-combine",
                 "rvv.partial-scale-combine", "none", "exact"))
    return emitOpError(
        "RVV scaled partial combine requires one scale per slot and one closed fanout");
  llvm::SmallVector<bool> seen(input.getSlots(), false);
  for (int64_t slot : getSlotOrder()) {
    if (slot < 0 || slot >= input.getSlots() || seen[slot])
      return emitOpError(
          "RVV scaled partial combine slot order must be a permutation");
    seen[slot] = true;
  }
  for (auto [scale, replica] :
       llvm::zip(getScales(), getScaleReplicas())) {
    ValueType type = mlir::cast<ValueType>(scale.getType());
    auto scaleElement = mlir::dyn_cast<mlir::IntegerType>(type.getElementType());
    auto replicas = checkedPositiveProduct(
        type.getLayout().getReplicaFactors().asArrayRef());
    if (!scaleElement || !scaleElement.isSigned() ||
        scaleElement.getWidth() != 32 ||
        type.getLayout().getCarrier() != "scalar" || !replicas ||
        replica < 0 || replica >= *replicas ||
        !llvm::all_of(type.getLayout().getTimeFactors().asArrayRef(),
                      [](int64_t factor) { return factor == 1; }) ||
        !llvm::all_of(type.getLayout().getLaneFactors().asArrayRef(),
                      [](int64_t factor) { return factor == 1; }))
      return emitOpError(
          "RVV scaled partial combine scales require in-bounds signed-i32 scalar replicas");
  }
  return mlir::success();
}

mlir::LogicalResult RVVPartialWidenScaleOp::verify() {
  PartialSetType input = getInput().getType();
  PartialSetType result = getResult().getType();
  ValueType inputPartial = input.getPartialType();
  ValueType resultPartial = result.getPartialType();
  auto inputElement =
      mlir::dyn_cast<mlir::IntegerType>(inputPartial.getElementType());
  auto resultElement =
      mlir::dyn_cast<mlir::IntegerType>(resultPartial.getElementType());
  auto kernel = getOperation()->getParentOfType<KernelOp>();
  const bool sameCoordinates =
      inputPartial.getAxisIds() == resultPartial.getAxisIds() &&
      inputPartial.getShape() == resultPartial.getShape() &&
      inputPartial.getLayout().getAxisIds() ==
          resultPartial.getLayout().getAxisIds() &&
      inputPartial.getLayout().getTimeFactors() ==
          resultPartial.getLayout().getTimeFactors() &&
      inputPartial.getLayout().getLaneFactors() ==
          resultPartial.getLayout().getLaneFactors() &&
      inputPartial.getLayout().getReplicaFactors() ==
          resultPartial.getLayout().getReplicaFactors() &&
      inputPartial.getLayout().getFragmentFactors() ==
          resultPartial.getLayout().getFragmentFactors() &&
      inputPartial.getLayout().getLocalFactors() ==
          resultPartial.getLayout().getLocalFactors();
  if (!inputElement || !inputElement.isSigned() ||
      inputElement.getWidth() != 16 || !resultElement ||
      !resultElement.isSigned() || resultElement.getWidth() != 32 ||
      inputPartial.getLayout().getCarrier() != "rvv" ||
      resultPartial.getLayout().getCarrier() != "rvv" || !sameCoordinates ||
      getCombineArity() <= 0 || input.getSlots() % getCombineArity() ||
      result.getReductionAxis() != input.getReductionAxis() ||
      result.getSlots() != input.getSlots() / getCombineArity() ||
      result.getTermsPerSlot() !=
          input.getTermsPerSlot() * getCombineArity() ||
      resultPartial.getLayout().getSew() !=
          2 * inputPartial.getLayout().getSew() ||
      resultPartial.getLayout().getLmulEighths() !=
          2 * inputPartial.getLayout().getLmulEighths() ||
      resultPartial.getLayout().getVl() != inputPartial.getLayout().getVl() ||
      resultPartial.getLayout().getRegisterGroups() !=
          std::max<int64_t>(
              1, (resultPartial.getLayout().getLmulEighths() + 7) / 8) ||
      result.getResourceGroups() !=
          result.getSlots() *
              resultPartial.getLayout().getRegisterGroups() ||
      getScales().size() != static_cast<size_t>(input.getSlots()) ||
      getScaleReplicas().size() != static_cast<size_t>(input.getSlots()) ||
      !kernel || !kernel.getTarget().getHasWideningInteger() ||
      !supportsRVVLayout(kernel.getTarget(), inputPartial.getLayout()) ||
      !supportsRVVLayout(kernel.getTarget(), resultPartial.getLayout()) ||
      !exactLeaf(getLeaf(), "rvv", "partial-widen-scale",
                 "rvv.partial-widen-scale", "none", "exact") ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>(
              {input.getReductionAxis(), input.getSlots(),
               static_cast<int64_t>(getCombineArity())}))
    return emitOpError(
        "RVV partial widen-scale requires one signed narrow scalar per i16 slot and an exact i32 topology");
  for (auto [scale, replica] :
       llvm::zip(getScales(), getScaleReplicas())) {
    ValueType type = mlir::cast<ValueType>(scale.getType());
    auto scaleElement =
        mlir::dyn_cast<mlir::IntegerType>(type.getElementType());
    auto replicas = checkedPositiveProduct(
        type.getLayout().getReplicaFactors().asArrayRef());
    if (!scaleElement || !scaleElement.isSigned() ||
        scaleElement.getWidth() <= 0 || scaleElement.getWidth() > 16 ||
        type.getLayout().getCarrier() != "scalar" || !replicas ||
        replica < 0 || replica >= *replicas ||
        !llvm::all_of(type.getLayout().getTimeFactors().asArrayRef(),
                      [](int64_t factor) { return factor == 1; }) ||
        !llvm::all_of(type.getLayout().getLaneFactors().asArrayRef(),
                      [](int64_t factor) { return factor == 1; }))
      return emitOpError(
          "RVV partial widen-scale operands require in-bounds signed narrow scalar replicas");
  }
  return mlir::success();
}

mlir::LogicalResult RVVPartialCombineOp::verify() {
  PartialSetType input = getInput().getType();
  PartialSetType result = getResult().getType();
  if (getArity() <= 1 || getTopology() != "pairwise" ||
      input.getSlots() % getArity() ||
      result.getPartialType() != input.getPartialType() ||
      result.getReductionAxis() != input.getReductionAxis() ||
      result.getSlots() != input.getSlots() / getArity() ||
      result.getTermsPerSlot() != input.getTermsPerSlot() * getArity() ||
      !exactLeaf(getLeaf(), "rvv", "partial-combine",
                 "rvv.partial-combine", "none", "exact") ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({input.getReductionAxis(),
                                   static_cast<int64_t>(getArity()),
                                   result.getSlots()}))
    return emitOpError(
        "RVV partial combine requires an exact pairwise divisive tree level over one partial set");
  return mlir::success();
}

mlir::LogicalResult RVVPartialFinalizeOp::verify() {
  PartialSetType input = getInput().getType();
  auto result = mlir::dyn_cast<mlir::IntegerType>(getResult().getType());
  auto resultValue = mlir::dyn_cast<ValueType>(getResult().getType());
  auto partial = mlir::dyn_cast<mlir::IntegerType>(
      input.getPartialType().getElementType());
  auto axis = llvm::find(input.getPartialType().getAxisIds().asArrayRef(),
                         getReductionAxis());
  if ((!result && !resultValue) || !partial || !partial.isSigned() ||
      (partial.getWidth() != 16 && partial.getWidth() != 32) ||
      getReductionAxis() != input.getReductionAxis() ||
      axis == input.getPartialType().getAxisIds().asArrayRef().end())
    return emitOpError(
        "RVV partial finalize requires one signed widened set and an i32 result");
  const size_t position = static_cast<size_t>(
      axis - input.getPartialType().getAxisIds().asArrayRef().begin());
  if (resultValue) {
    auto resultElement =
        mlir::dyn_cast<mlir::IntegerType>(resultValue.getElementType());
    bool preservesFreeAxes =
        resultElement && resultElement.isSigned() &&
        resultElement.getWidth() == 32 && partial.getWidth() == 16 &&
        input.getSlots() == 1 &&
        input.getPartialType().getShape()[position] == 1 &&
        input.getPartialType().getLayout().getTimeFactors()[position] == 1 &&
        input.getPartialType().getLayout().getLaneFactors()[position] == 1 &&
        input.getPartialType().getLayout().getReplicaFactors()[position] == 1 &&
        resultValue.getLayout().getCarrier() == "rvv" &&
        resultValue.getAxisIds().size() + 1 ==
            input.getPartialType().getAxisIds().size() &&
        resultValue.getLayout().getSew() == 32 &&
        resultValue.getLayout().getLmulEighths() ==
            input.getPartialType().getLayout().getLmulEighths() * 2 &&
        resultValue.getLayout().getVl() ==
            input.getPartialType().getLayout().getVl();
    size_t resultPosition = 0;
    for (size_t partialPosition = 0;
         preservesFreeAxes &&
         partialPosition < input.getPartialType().getAxisIds().size();
         ++partialPosition) {
      if (partialPosition == position)
        continue;
      preservesFreeAxes &=
          resultPosition < resultValue.getAxisIds().size() &&
          resultValue.getAxisIds()[resultPosition] ==
              input.getPartialType().getAxisIds()[partialPosition] &&
          resultValue.getShape()[resultPosition] ==
              input.getPartialType().getShape()[partialPosition] &&
          resultValue.getLayout().getTimeFactors()[resultPosition] ==
              input.getPartialType().getLayout().getTimeFactors()[partialPosition] &&
          resultValue.getLayout().getLaneFactors()[resultPosition] ==
              input.getPartialType().getLayout().getLaneFactors()[partialPosition] &&
          resultValue.getLayout().getReplicaFactors()[resultPosition] ==
              input.getPartialType().getLayout().getReplicaFactors()[partialPosition] &&
          resultValue.getLayout().getFragmentFactors()[resultPosition] ==
              input.getPartialType().getLayout().getFragmentFactors()[partialPosition] &&
          resultValue.getLayout().getLocalFactors()[resultPosition] ==
              input.getPartialType().getLayout().getLocalFactors()[partialPosition];
      ++resultPosition;
    }
    if (!preservesFreeAxes ||
        !exactLeaf(getLeaf(), "rvv", "partial-finalize",
                   "rvv.partial-finalize.widen", "none", "exact"))
      return emitOpError(
          "RVV vector partial finalize must widen one singleton reduction coordinate and preserve every free physical axis");
    return mlir::success();
  }
  if (!result || !result.isSigned() || result.getWidth() != 32)
    return emitOpError(
        "RVV scalar partial finalize requires one signed i32 result");
  llvm::StringRef instruction =
      input.getPartialType().getLayout().getLaneFactors()[position] == 1
          ? "rvv.partial-finalize.extract"
          : "rvv.partial-finalize.reduce";
  if (!exactLeaf(getLeaf(), "rvv", "partial-finalize", instruction, "none",
                 "exact"))
    return emitOpError()
           << "RVV partial finalize leaf disagrees with its typed lane topology; "
              "expected_instruction="
           << instruction << ", reduction_axis=" << getReductionAxis()
           << ", partial=" << input.getPartialType() << ", leaf=" << getLeaf();
  return mlir::success();
}

mlir::LogicalResult RVVAssembleReplicasOp::verify() {
  ValueType result = getResult().getType();
  auto parts = checkedPositiveProduct(
      result.getLayout().getReplicaFactors().asArrayRef());
  if (result.getLayout().getCarrier() != "scalar" || !parts ||
      *parts != static_cast<int64_t>(getValues().size()) ||
      llvm::any_of(getValues(), [&](mlir::Value value) {
        return value.getType() != result.getElementType();
      }) ||
      !exactLeaf(getLeaf(), "scalar", "assemble-replicas",
                 "scalar.assemble-replicas", "none", "exact"))
    return emitOpError(
        "scalar replica assembly requires one scalar value per result replica");
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
  auto plan = getPlan();
  const int64_t group = plan.getGroupSize();
  const int64_t layer = plan.getLayerSize();
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
      getAccess().getGroupSize() != group || getAccess().getLayerSize() != layer ||
      plan.getKind() != "layered" || plan.getReductionAxis() != axis ||
      plan.getProjectionBase() != 0 || plan.getProjectionStride() != 1 ||
      plan.getProjectionRepeat() != 1 || plan.getProjectionExtent() != layer ||
      layer <= 0 || group != layer * 2 || element.getWidth() * 2 > 8 ||
      (getAccess().getOrder() != "lo_first" &&
       getAccess().getOrder() != "hi_first") ||
      getPhysicalLayerForResult().size() != 2 ||
      getShiftForResult().size() != 2 || getMaskForResult().size() != 2 ||
      !partition || partition.value() != layer ||
      !exactLeaf(getLeaf(), "rvv", "layered-window", "rvv.layered-window",
                 "none", tail))
    return emitOpError(
        "RVV layered window requires two equal full/tail layer results, one "
        "two-layer packed field, and an identical exact layer-sized point");
  llvm::SmallVector<int64_t> ordered(getPhysicalLayerForResult());
  llvm::sort(ordered);
  if (ordered != llvm::SmallVector<int64_t>({0, 1}))
    return emitOpError(
        "RVV layered window physical result layers must form one permutation");
  for (size_t index = 0; index < getPhysicalLayerForResult().size(); ++index) {
    const int64_t physicalLayer = getPhysicalLayerForResult()[index];
    const int64_t shift = getShiftForResult()[index];
    const int64_t mask = getMaskForResult()[index];
    if (physicalLayer !=
            plan.getPhysicalLayerBase() +
                static_cast<int64_t>(index) * plan.getPhysicalLayerStep() ||
        shift != physicalLayer * static_cast<int64_t>(element.getWidth()) ||
        mask != ((physicalLayer + 1) *
                            static_cast<int64_t>(element.getWidth()) !=
                        8
                    ? (int64_t{1} << element.getWidth()) - 1
                    : 0))
      return emitOpError(
          "RVV layered window result decode plan disagrees with its typed layer");
  }
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
  auto geometry = getGeometry();
  bool exactLayerOps =
      geometry.getPhysicalLayerForStream().size() ==
          geometry.getShiftAmountForStream().size() &&
      geometry.getPhysicalLayerForStream().size() ==
          geometry.getMaskValueForStream().size();
  if (exactLayerOps && element)
    for (auto [physicalLayer, shiftAmount, maskValue] :
         llvm::zip(geometry.getPhysicalLayerForStream().asArrayRef(),
                   geometry.getShiftAmountForStream().asArrayRef(),
                   geometry.getMaskValueForStream().asArrayRef()))
      exactLayerOps &=
          shiftAmount == physicalLayer * element.getWidth() &&
          maskValue ==
              ((physicalLayer + 1) * element.getWidth() != 8
                   ? (int64_t(1) << element.getWidth()) - 1
                   : 0);
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
      geometry.getGroupSize() != group || geometry.getLayerSize() != layer ||
      geometry.getStreamCount() !=
          checkedPositiveProduct(
              result.getLayout().getTimeFactors().asArrayRef()).value_or(-1) ||
      geometry.getReplicaCount() !=
          checkedPositiveProduct(
              result.getLayout().getReplicaFactors().asArrayRef()).value_or(-1) ||
      !exactLayerOps ||
      !exactLeaf(getLeaf(), "rvv", "layered-stream", "rvv.layered-stream",
                 "none", tail))
    return emitOpError(
        "RVV layered stream requires one byte-aligned packed field whose "
        "grouped/layered geometry is represented by one complete time/lane axis");
  return mlir::success();
}

mlir::LogicalResult RVVProjectedLayeredStreamOp::verify() {
  ValueType field = getField().getType();
  ValueType result = getResult().getType();
  auto element = mlir::dyn_cast<mlir::IntegerType>(field.getElementType());
  auto sourceField = getField().getDefiningOp<FieldOp>();
  const int64_t axis = getReductionAxis();
  auto fieldAxis = llvm::find(field.getAxisIds().asArrayRef(), axis);
  auto resultAxis = llvm::find(result.getAxisIds().asArrayRef(), axis);
  const int64_t group = getAccess().getGroupSize();
  const int64_t layer = getAccess().getLayerSize();
  const int64_t layers = layer > 0 ? group / layer : 0;
  auto physicalPoint = getOrigin().getDefiningOp<PhysicalPointOp>();
  auto pointPartition =
      physicalPoint
          ? physicalPoint.getPartition().getDefiningOp<mlir::arith::ConstantIndexOp>()
          : mlir::arith::ConstantIndexOp();
  const int64_t fieldExtent =
      fieldAxis == field.getAxisIds().asArrayRef().end()
          ? 0
          : field.getShape()[static_cast<size_t>(
                fieldAxis - field.getAxisIds().asArrayRef().begin())];
  const int64_t resultExtent =
      resultAxis == result.getAxisIds().asArrayRef().end()
          ? 0
          : result.getShape()[static_cast<size_t>(
                resultAxis - result.getAxisIds().asArrayRef().begin())];
  const int64_t lanes =
      resultAxis == result.getAxisIds().asArrayRef().end()
          ? 0
          : result.getLayout().getLaneFactors()[static_cast<size_t>(
                resultAxis - result.getAxisIds().asArrayRef().begin())];
  const int64_t time =
      resultAxis == result.getAxisIds().asArrayRef().end()
          ? 0
          : result.getLayout().getTimeFactors()[static_cast<size_t>(
                resultAxis - result.getAxisIds().asArrayRef().begin())];
  llvm::StringRef validity = result.getLayout().getValidity();
  llvm::StringRef tail = validity == "tail" ? "agnostic" : "exact";
  auto geometry = getGeometry();
  bool exactLayerOps =
      geometry.getPhysicalLayerForStream().size() ==
          geometry.getShiftAmountForStream().size() &&
      geometry.getPhysicalLayerForStream().size() ==
          geometry.getMaskValueForStream().size();
  if (exactLayerOps && element)
    for (auto [physicalLayer, shiftAmount, maskValue] :
         llvm::zip(geometry.getPhysicalLayerForStream().asArrayRef(),
                   geometry.getShiftAmountForStream().asArrayRef(),
                   geometry.getMaskValueForStream().asArrayRef()))
      exactLayerOps &=
          shiftAmount == physicalLayer * element.getWidth() &&
          maskValue ==
              ((physicalLayer + 1) * element.getWidth() != 8
                   ? (int64_t(1) << element.getWidth()) - 1
                   : 0);
  bool preservesFreeAxes = field.getAxisIds() == result.getAxisIds() &&
                          field.getShape().size() == result.getShape().size();
  if (preservesFreeAxes)
    for (size_t position = 0; position < result.getShape().size(); ++position) {
      if (result.getAxisIds()[position] == axis)
        continue;
      preservesFreeAxes &=
          field.getShape()[position] == result.getShape()[position] &&
          result.getLayout().getTimeFactors()[position] == 1 &&
          result.getLayout().getLaneFactors()[position] == 1 &&
          (result.getShape()[position] > 0
               ? result.getLayout().getReplicaFactors()[position] ==
                     result.getShape()[position]
               : result.getLayout().getReplicaFactors()[position] > 0) &&
          result.getLayout().getFragmentFactors()[position] == 1 &&
          result.getLayout().getLocalFactors()[position] == 1;
    }
  if (!element || element.isSigned() || !sourceField ||
      getAccess() != sourceField.getAccess() || axis <= 0 ||
      getOrigin().getType().getDomain().getAxisId() != axis ||
      getOrigin().getType().getDomain().getTail() != "exact" ||
      fieldAxis == field.getAxisIds().asArrayRef().end() ||
      resultAxis == result.getAxisIds().asArrayRef().end() ||
      field.getElementType() != result.getElementType() || !preservesFreeAxes ||
      getProjectionBase() < 0 || getProjectionStride() != 1 ||
      getProjectionRepeat() != 1 || getProjectionExtent() != resultExtent ||
      resultExtent <= 0 || fieldExtent <= 0 ||
      getProjectionBase() > fieldExtent ||
      resultExtent > fieldExtent - getProjectionBase() ||
      group <= 0 || layer <= 0 || group % layer || layers <= 1 ||
      getProjectionBase() % group || resultExtent % group ||
      !pointPartition || pointPartition.value() % group ||
      result.getLayout().getCarrier() != "rvv" || lanes <= 1 || time <= 1 ||
      time > std::numeric_limits<int64_t>::max() / lanes ||
      time * lanes != resultExtent || layer % lanes ||
      element.getWidth() * layers > 8 || getAccess().getBitOffset() % 8 ||
      getAccess().getForm() != "indexed" ||
      getAccess().getMapping() != "grouped_layered" ||
      (getAccess().getOrder() != "lo_first" &&
       getAccess().getOrder() != "hi_first") ||
      geometry.getAxis() != axis || geometry.getGroupSize() != group ||
      geometry.getLayerSize() != layer || geometry.getLaneCount() != lanes ||
      geometry.getStreamCount() !=
          checkedPositiveProduct(
              result.getLayout().getTimeFactors().asArrayRef()).value_or(-1) ||
      geometry.getReplicaCount() !=
          checkedPositiveProduct(
              result.getLayout().getReplicaFactors().asArrayRef()).value_or(-1) ||
      !exactLayerOps ||
      (validity != "full" && validity != "tail") ||
      !exactLeaf(getLeaf(), "rvv", "projected-layered-stream",
                 "rvv.projected-layered-stream", "none", tail))
    return emitOpError(
        "projected layered stream requires one typed field/point projection and one closed grouped/layered time-lane mapping");
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

mlir::LogicalResult RVVAxisBroadcastOp::verify() {
  ValueType input = getInput().getType();
  ValueType result = getResult().getType();
  auto inputLayout = input.getLayout();
  auto resultLayout = result.getLayout();
  auto inputLanes = checkedProduct(inputLayout.getLaneFactors().asArrayRef());
  auto resultLanes = checkedProduct(resultLayout.getLaneFactors().asArrayRef());
  bool preservesSourceDomain = input.getElementType() == result.getElementType();
  for (auto [position, axis] :
       llvm::enumerate(input.getAxisIds().asArrayRef())) {
    auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
    if (found == result.getAxisIds().asArrayRef().end()) {
      preservesSourceDomain = false;
      break;
    }
    size_t resultPosition = static_cast<size_t>(
        found - result.getAxisIds().asArrayRef().begin());
    preservesSourceDomain &=
        input.getShape()[position] == result.getShape()[resultPosition];
  }
  if (!preservesSourceDomain || input.getAxisIds().size() >=
                                    result.getAxisIds().size() ||
      inputLayout.getCarrier() != "rvv" ||
      resultLayout.getCarrier() != "rvv" || !inputLanes || !resultLanes ||
      *inputLanes <= 0 || *resultLanes <= *inputLanes ||
      inputLayout.getSew() != resultLayout.getSew() ||
      inputLayout.getLmulEighths() > resultLayout.getLmulEighths() ||
      !exactLeaf(getLeaf(), "rvv", "axis-broadcast",
                 "rvv.axis-broadcast", "none", "exact") ||
      getLeaf().getParameters().asArrayRef() !=
          llvm::ArrayRef<int64_t>({*inputLanes, *resultLanes}))
    return emitOpError(
        "RVV axis broadcast must embed one proper logical sub-domain in a wider typed lane layout");
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
