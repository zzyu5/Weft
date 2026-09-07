#include "RISCVPhysicalSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <algorithm>
#include <cstdint>
#include <numeric>

using namespace weft;

namespace {

std::optional<int64_t> resolvedPhysicalIndex(mlir::Value value,
                                             mlir::Operation *scope) {
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return integer.getInt();
  if (auto symbol = value.getDefiningOp<riscv::SymbolOp>()) {
    auto kernel = scope->getParentOfType<riscv::KernelOp>();
    if (!kernel)
      return std::nullopt;
    for (auto [name, binding] :
         llvm::zip(kernel.getParameterNames(), kernel.getParameterValues()))
      if (mlir::cast<mlir::StringAttr>(name).getValue() == symbol.getName())
        return binding;
  }
  return std::nullopt;
}

int64_t domainPartition(riscv::DomainType domain, mlir::Operation *scope) {
  int64_t result = 0;
  auto module = scope->getParentOfType<mlir::ModuleOp>();
  module.walk([&](riscv::DomainOp operation) {
    if (operation.getResult().getType().getDomainId() == domain.getDomainId())
      if (auto value = resolvedPhysicalIndex(operation.getPartition(), scope))
        result = *value;
  });
  return result;
}

bool checkedRegularAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedRegularSubtract(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs) ||
      (rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs))
    return false;
  result = lhs - rhs;
  return true;
}

bool checkedRegularScale(int64_t value, int64_t factor, int64_t &result) {
  if (factor <= 0 || value > std::numeric_limits<int64_t>::max() / factor ||
      value < std::numeric_limits<int64_t>::min() / factor)
    return false;
  result = value * factor;
  return true;
}

std::optional<riscv_internal::RegularIndexRelation>
analyzeRegularIndexRelationImpl(mlir::Value value, unsigned depth) {
  if (depth > 16)
    return std::nullopt;
  if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>())
    return analyzeRegularIndexRelationImpl(conversion.getInput(), depth + 1);
  if (auto iota = value.getDefiningOp<riscv::IotaOp>()) {
    if (iota.getStart() >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return std::nullopt;
    return riscv_internal::RegularIndexRelation{
        {}, static_cast<int64_t>(iota.getStart()), 1, 1};
  }
  auto binary = value.getDefiningOp<riscv::BinaryOp>();
  if (!binary)
    return std::nullopt;

  auto lhs = analyzeRegularIndexRelationImpl(binary.getLhs(), depth + 1);
  auto rhs = analyzeRegularIndexRelationImpl(binary.getRhs(), depth + 1);
  auto lhsConstant = riscv_internal::constantInt(binary.getLhs());
  auto rhsConstant = riscv_internal::constantInt(binary.getRhs());
  auto scalarCoordinate = [](mlir::Value candidate) {
    mlir::Type type = candidate.getType();
    if (type.isIndex())
      return true;
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
      return !integer.isSigned();
    auto physical = mlir::dyn_cast<riscv::ValueType>(type);
    if (!physical || physical.getLayout().getCarrier() != "scalar")
      return false;
    auto productIsOne = [](llvm::ArrayRef<int64_t> factors) {
      auto product = riscv_internal::staticProduct(factors);
      return product && *product == 1;
    };
    return productIsOne(
               physical.getLayout().getTimeFactors().asArrayRef()) &&
           productIsOne(
               physical.getLayout().getLaneFactors().asArrayRef()) &&
           productIsOne(
               physical.getLayout().getReplicaFactors().asArrayRef()) &&
           productIsOne(
               physical.getLayout().getFragmentFactors().asArrayRef()) &&
           productIsOne(
               physical.getLayout().getLocalFactors().asArrayRef());
  };
  if (binary.getKind() == "add") {
    if (lhs && rhsConstant)
      return checkedRegularAdd(lhs->base, *rhsConstant, lhs->base)
                 ? lhs
                 : std::nullopt;
    if (rhs && lhsConstant)
      return checkedRegularAdd(rhs->base, *lhsConstant, rhs->base)
                 ? rhs
                 : std::nullopt;
    if (lhs && !lhs->dynamicBase && scalarCoordinate(binary.getRhs())) {
      lhs->dynamicBase = binary.getRhs();
      return lhs;
    }
    if (rhs && !rhs->dynamicBase && scalarCoordinate(binary.getLhs())) {
      rhs->dynamicBase = binary.getLhs();
      return rhs;
    }
  }
  if (binary.getKind() == "sub" && lhs && rhsConstant)
    return checkedRegularSubtract(lhs->base, *rhsConstant, lhs->base)
               ? lhs
               : std::nullopt;
  if (binary.getKind() == "mul") {
    auto scale = [&](riscv_internal::RegularIndexRelation pattern,
                     int64_t factor)
        -> std::optional<riscv_internal::RegularIndexRelation> {
      if (!checkedRegularScale(pattern.base, factor, pattern.base) ||
          !checkedRegularScale(pattern.stride, factor, pattern.stride))
        return std::nullopt;
      return pattern;
    };
    if (lhs && rhsConstant)
      return scale(*lhs, *rhsConstant);
    if (rhs && lhsConstant)
      return scale(*rhs, *lhsConstant);
  }
  if (binary.getKind() == "div" && lhs && rhsConstant && *rhsConstant > 0 &&
      lhs->base % *rhsConstant == 0 && lhs->stride == 1 &&
      lhs->repeat <= std::numeric_limits<int64_t>::max() / *rhsConstant) {
    lhs->base /= *rhsConstant;
    lhs->repeat *= *rhsConstant;
    return lhs;
  }
  return std::nullopt;
}

} // namespace

mlir::DenseI64ArrayAttr
riscv_internal::integers(mlir::Builder &builder,
                         llvm::ArrayRef<int64_t> values) {
  return builder.getDenseI64ArrayAttr(values);
}

std::optional<riscv_internal::RegularIndexRelation>
riscv_internal::analyzeRegularIndexRelation(mlir::Value value,
                                            unsigned depth) {
  return analyzeRegularIndexRelationImpl(value, depth);
}

int64_t riscv_internal::physicalExtent(mlir::Value value, int64_t axis) {
  auto axes = logicalAxes(value.getType());
  auto shape = logicalShape(value.getType());
  auto found = llvm::find(axes, axis);
  if (found == axes.end())
    return 1;
  const int64_t direct = shape[static_cast<size_t>(found - axes.begin())];
  if (direct > 0)
    return direct;
  for (mlir::Operation *parent = value.getParentRegion()->getParentOp(); parent;
       parent = parent->getParentOp())
    if (auto loop = mlir::dyn_cast<riscv::LoopOp>(parent)) {
      riscv::DomainType domain = loop.getDomain().getType();
      if (domain.getAxisId() == axis)
        return domainPartition(domain, parent);
    }
  return 0;
}

llvm::StringRef riscv_internal::layeredStorageDecodeInstruction(
    int64_t shiftAmount, int64_t maskValue) {
  if (shiftAmount > 0)
    return maskValue > 0 ? "rvv.layered-storage-decode.shift-mask"
                         : "rvv.layered-storage-decode.shift";
  return maskValue > 0 ? "rvv.layered-storage-decode.mask"
                       : "rvv.layered-storage-decode.identity";
}

std::optional<riscv::StorageWindowPlanAttr>
riscv_internal::storageWindowPlan(
    mlir::Builder &builder, riscv::FieldOp field, int64_t reductionAxis,
    int64_t projectionBase, int64_t projectionStride,
    int64_t projectionRepeat, int64_t projectionExtent,
    int64_t offsetAlignment, bool recordProjection) {
  riscv::LoadOp load = sourceLoad(field.getOwner());
  auto memory = load ? load.getRegion().getType() : riscv::MemDescType();
  auto fieldType =
      mlir::dyn_cast<riscv::ValueType>(field.getResult().getType());
  auto integer = fieldType
                     ? mlir::dyn_cast<mlir::IntegerType>(
                           fieldType.getElementType())
                     : mlir::IntegerType();
  auto floating = fieldType
                      ? mlir::dyn_cast<mlir::FloatType>(
                            fieldType.getElementType())
                      : mlir::FloatType();
  const int64_t elementBits =
      integer ? integer.getWidth() : floating ? floating.getWidth() : 0;
  riscv::AccessAttr access = field.getAccess();
  if (!memory || !fieldType || !access || reductionAxis <= 0 ||
      projectionBase < 0 || projectionStride <= 0 || projectionRepeat <= 0 ||
      projectionExtent <= 0 || offsetAlignment <= 0 ||
      memory.getElements() <= 0 || access.getBitOffset() < 0 ||
      access.getBitOffset() % 8 || elementBits <= 0)
    return std::nullopt;

  llvm::StringRef kind;
  int64_t group = 0;
  int64_t layer = 0;
  int64_t physicalLayerBase = 0;
  int64_t physicalLayerStep = 0;
  int64_t shiftBase = 0;
  int64_t shiftStep = 0;
  int64_t maskValue = 0;
  auto ownerElement = logicalElement(field.getOwner().getType());
  auto ownerEncoding = mlir::dyn_cast<kernel::EncodingType>(ownerElement);
  const int64_t recordInterleave =
      ownerEncoding ? interleaveRows(field, ownerEncoding) : 0;
  if (recordProjection && recordInterleave > 0 &&
      access.getMapping() == "natural") {
    kind = "interleaved_natural";
  } else if (recordProjection && recordInterleave > 0 &&
             access.getMapping() == "joined") {
    kind = "interleaved_joined";
  } else if (access.getMapping() == "natural") {
    kind = projectionRepeat > 1
               ? "repeat"
               : projectionStride == 1 ? "unit" : "strided";
  } else if (access.getMapping() == "grouped_layered") {
    group = access.getGroupSize();
    layer = access.getLayerSize();
    const int64_t layers = layer > 0 ? group / layer : 0;
    if (!integer || integer.isSigned() || elementBits >= 8 || group <= 0 ||
        layer <= 0 || group % layer || layers <= 1 ||
        (access.getOrder() != "lo_first" &&
         access.getOrder() != "hi_first"))
      return std::nullopt;
    kind = "layered";
    physicalLayerBase = access.getOrder() == "lo_first" ? 0 : layers - 1;
    physicalLayerStep = access.getOrder() == "lo_first" ? 1 : -1;
    shiftBase = physicalLayerBase * elementBits;
    shiftStep = physicalLayerStep * elementBits;
    maskValue = (int64_t{1} << elementBits) - 1;
  } else {
    return std::nullopt;
  }
  return riscv::StorageWindowPlanAttr::get(
      builder.getContext(), kind, reductionAxis, projectionBase,
      projectionStride, projectionRepeat, projectionExtent, offsetAlignment,
      memory.getElements(), access.getBitOffset() / 8, elementBits, group,
      layer, physicalLayerBase, physicalLayerStep, shiftBase, shiftStep,
      maskValue);
}

std::optional<riscv_internal::WidenDotLaneSlicePlan>
riscv_internal::planWidenDotLaneSlices(
    riscv::ValueType operand, riscv::ValueType result,
    llvm::ArrayRef<int64_t> reductionAxes, riscv::TargetAttr target) {
  auto layout = operand.getLayout();
  auto lanes = staticProduct(layout.getLaneFactors().asArrayRef());
  if (!lanes || *lanes <= 0)
    return std::nullopt;

  int64_t reductionLanes = 1;
  bool sawFreeLane = false;
  for (int64_t position = static_cast<int64_t>(operand.getAxisIds().size()) - 1;
       position >= 0; --position) {
    const int64_t lane = layout.getLaneFactors()[position];
    if (lane <= 1)
      continue;
    const int64_t axis = operand.getAxisIds()[position];
    if (llvm::is_contained(reductionAxes, axis)) {
      if (sawFreeLane ||
          reductionLanes > std::numeric_limits<int64_t>::max() / lane)
        return std::nullopt;
      reductionLanes *= lane;
    } else {
      sawFreeLane = true;
    }
  }
  if (reductionLanes <= 0 || *lanes % reductionLanes)
    return std::nullopt;

  auto sliceLmul =
      riscv::rvvLaneSliceLMULEighths(target, operand, reductionLanes);
  if (!sliceLmul ||
      !llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), *sliceLmul))
    return std::nullopt;

  int64_t outputParts = 1;
  if (result) {
    if (result.getLayout().getCarrier() != "scalar")
      return std::nullopt;
    outputParts = staticProduct(result.getLayout().getReplicaFactors().asArrayRef())
                      .value_or(-1);
    if (outputParts <= 0)
      return std::nullopt;
  }

  WidenDotLaneSlicePlan plan;
  plan.reductionLanes = reductionLanes;
  plan.sliceLmulEighths = *sliceLmul;
  llvm::SmallVector<int64_t> reductionTimeExtents;
  for (int64_t axis : reductionAxes) {
    auto found = llvm::find(operand.getAxisIds().asArrayRef(), axis);
    if (found == operand.getAxisIds().asArrayRef().end())
      return std::nullopt;
    const size_t position = static_cast<size_t>(
        found - operand.getAxisIds().asArrayRef().begin());
    const int64_t extent = layout.getTimeFactors()[position];
    if (extent <= 0)
      return std::nullopt;
    reductionTimeExtents.push_back(extent);
  }
  plan.reductionStreams = staticProduct(reductionTimeExtents).value_or(-1);
  const int64_t totalStreams =
      staticProduct(layout.getTimeFactors().asArrayRef()).value_or(-1);
  const int64_t registerParts =
      staticProduct(layout.getReplicaFactors().asArrayRef()).value_or(-1);
  if (plan.reductionStreams <= 0 || totalStreams <= 0 || registerParts <= 0 ||
      registerParts > std::numeric_limits<int64_t>::max() / totalStreams)
    return std::nullopt;
  const int64_t totalParts = registerParts * totalStreams;

  for (int64_t outputPart = 0; outputPart < outputParts; ++outputPart) {
    llvm::SmallVector<int64_t> resultCoordinates(
        result ? result.getAxisIds().size() : 0, 0);
    int64_t remaining = outputPart;
    if (result) {
      for (int64_t position = static_cast<int64_t>(result.getAxisIds().size()) - 1;
           position >= 0; --position) {
        const int64_t replica = result.getLayout().getReplicaFactors()[position];
        if (replica <= 0)
          return std::nullopt;
        resultCoordinates[position] = remaining % replica;
        remaining /= replica;
      }
      if (remaining)
        return std::nullopt;
    }

    auto resultCoordinateForAxis = [&](int64_t axis) -> std::optional<int64_t> {
      if (!result)
        return std::nullopt;
      auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
      if (found == result.getAxisIds().asArrayRef().end())
        return std::nullopt;
      return resultCoordinates[static_cast<size_t>(
          found - result.getAxisIds().asArrayRef().begin())];
    };

    int64_t laneOffset = 0;
    int64_t laneStride = 1;
    for (int64_t position = static_cast<int64_t>(operand.getAxisIds().size()) - 1;
         position >= 0; --position) {
      const int64_t lane = layout.getLaneFactors()[position];
      if (lane <= 1)
        continue;
      const int64_t axis = operand.getAxisIds()[position];
      int64_t coordinate = 0;
      if (!llvm::is_contained(reductionAxes, axis)) {
        auto logicalCoordinate = resultCoordinateForAxis(axis);
        if (!logicalCoordinate || *logicalCoordinate < 0)
          return std::nullopt;
        coordinate = *logicalCoordinate % lane;
      }
      if (coordinate > 0 &&
          laneOffset > std::numeric_limits<int64_t>::max() -
                           coordinate * laneStride)
        return std::nullopt;
      laneOffset += coordinate * laneStride;
      if (laneStride > std::numeric_limits<int64_t>::max() / lane)
        return std::nullopt;
      laneStride *= lane;
    }
    if (laneOffset < 0 || laneOffset % reductionLanes ||
        laneOffset > *lanes - reductionLanes)
      return std::nullopt;
    plan.offsets.push_back(laneOffset);

    for (int64_t reductionStream = 0;
         reductionStream < plan.reductionStreams; ++reductionStream) {
      llvm::SmallVector<int64_t> reductionCoordinates(reductionAxes.size(), 0);
      int64_t remainingReduction = reductionStream;
      for (int64_t index = static_cast<int64_t>(reductionAxes.size()) - 1;
           index >= 0; --index) {
        reductionCoordinates[index] =
            remainingReduction % reductionTimeExtents[index];
        remainingReduction /= reductionTimeExtents[index];
      }
      if (remainingReduction)
        return std::nullopt;

      int64_t streamPart = 0;
      int64_t registerPart = 0;
      for (size_t position = 0; position < operand.getAxisIds().size();
           ++position) {
        const int64_t axis = operand.getAxisIds()[position];
        const int64_t time = layout.getTimeFactors()[position];
        const int64_t lane = layout.getLaneFactors()[position];
        const int64_t replica = layout.getReplicaFactors()[position];
        int64_t timeCoordinate = 0;
        int64_t replicaCoordinate = 0;
        if (auto reduction = llvm::find(reductionAxes, axis);
            reduction != reductionAxes.end()) {
          timeCoordinate = reductionCoordinates[static_cast<size_t>(
              reduction - reductionAxes.begin())];
        } else {
          auto logicalCoordinate = resultCoordinateForAxis(axis);
          if (!logicalCoordinate)
            return std::nullopt;
          if (lane > 1) {
            if (replica != 1 || *logicalCoordinate >= time * lane)
              return std::nullopt;
            timeCoordinate = *logicalCoordinate / lane;
          } else {
            if (*logicalCoordinate >= time * replica)
              return std::nullopt;
            timeCoordinate = *logicalCoordinate / replica;
            replicaCoordinate = *logicalCoordinate % replica;
          }
        }
        streamPart = streamPart * time + timeCoordinate;
        registerPart = registerPart * replica + replicaCoordinate;
      }
      const int64_t physicalPart = registerPart * totalStreams + streamPart;
      if (physicalPart < 0 || physicalPart >= totalParts)
        return std::nullopt;
      plan.parts.push_back(physicalPart);
    }
  }
  return plan;
}

mlir::ArrayAttr riscv_internal::strings(
    mlir::Builder &builder, llvm::ArrayRef<std::string> values) {
  llvm::SmallVector<mlir::Attribute> attributes;
  for (const std::string &value : values)
    attributes.push_back(builder.getStringAttr(value));
  return builder.getArrayAttr(attributes);
}

mlir::DictionaryAttr riscv_internal::dictionary(
    mlir::Builder &builder,
    llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Attribute>> values) {
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  for (const auto &[name, value] : values)
    if (value)
      attributes.emplace_back(builder.getStringAttr(name), value);
  return builder.getDictionaryAttr(attributes);
}

std::optional<int64_t>
riscv_internal::integer(mlir::DictionaryAttr dictionary,
                        llvm::StringRef name) {
  if (!dictionary)
    return std::nullopt;
  if (auto value = dictionary.getAs<mlir::IntegerAttr>(name))
    return value.getInt();
  return std::nullopt;
}

std::optional<llvm::StringRef>
riscv_internal::string(mlir::DictionaryAttr dictionary, llvm::StringRef name) {
  if (!dictionary)
    return std::nullopt;
  if (auto value = dictionary.getAs<mlir::StringAttr>(name))
    return value.getValue();
  return std::nullopt;
}

mlir::ArrayAttr riscv_internal::array(mlir::DictionaryAttr dictionary,
                                      llvm::StringRef name) {
  return dictionary ? dictionary.getAs<mlir::ArrayAttr>(name)
                    : mlir::ArrayAttr();
}

riscv::LayoutAttr
riscv_internal::unassignedLayout(mlir::Builder &builder,
                                 llvm::ArrayRef<int64_t> axes) {
  llvm::SmallVector<int64_t> zero(axes.size(), 0);
  return riscv::LayoutAttr::get(builder.getContext(), "unassigned",
                                integers(builder, axes), integers(builder, zero),
                                integers(builder, zero), integers(builder, zero),
                                integers(builder, zero), integers(builder, zero),
                                0, 0, 0, 0, "unassigned");
}

namespace {

unsigned bitWidth(mlir::Type type) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  if (type.isIndex())
    return 64;
  return 8;
}

} // namespace

riscv::LayoutAttr riscv_internal::scalarLayout(
    mlir::Builder &builder, mlir::Type element, llvm::ArrayRef<int64_t> shape,
    llvm::ArrayRef<int64_t> axes, llvm::StringRef validity) {
  llvm::SmallVector<int64_t> time;
  llvm::SmallVector<int64_t> one(axes.size(), 1);
  for (int64_t extent : shape)
    time.push_back(extent > 0 ? extent : 1);
  return riscv::LayoutAttr::get(
      builder.getContext(), "scalar", integers(builder, axes),
      integers(builder, time), integers(builder, one), integers(builder, one),
      integers(builder, one), integers(builder, one), bitWidth(element), 0, 1,
      0, validity);
}

riscv::LayoutAttr riscv_internal::projectLayout(
    mlir::Builder &builder, riscv::ValueType source,
    riscv::LayoutAttr reference, riscv::TargetAttr target) {
  auto axes = source.getAxisIds().asArrayRef();
  auto referenceAxes = reference.getAxisIds().asArrayRef();
  auto project = [&](mlir::DenseI64ArrayAttr factors,
                     bool sequentialWhenAbsent) {
    llvm::SmallVector<int64_t> result;
    for (auto [index, axis] : llvm::enumerate(axes)) {
      auto found = llvm::find(referenceAxes, axis);
      result.push_back(
          found == referenceAxes.end()
              ? (sequentialWhenAbsent
                     ? std::max<int64_t>(1, source.getShape()[index])
                     : 1)
              : factors[static_cast<size_t>(found - referenceAxes.begin())]);
    }
    return result;
  };
  auto time = project(reference.getTimeFactors(), true);
  auto lane = project(reference.getLaneFactors(), false);
  auto replica = project(reference.getReplicaFactors(), false);
  auto fragment = project(reference.getFragmentFactors(), false);
  auto local = project(reference.getLocalFactors(), false);
  int64_t laneCount = 1;
  for (int64_t factor : lane)
    laneCount *= factor;
  llvm::StringRef carrier = reference.getCarrier();
  if (carrier == "rvv" && laneCount <= 1)
    carrier = "scalar";
  int64_t sew = std::max<int64_t>(8, logicalBitWidth(source));
  int64_t lmul = 0;
  if (carrier == "rvv") {
    int64_t elen = 0;
    for (int64_t supported : target.getSupportedSEW().asArrayRef())
      elen = std::max(elen, supported);
    if (elen <= 0)
      return {};
    int64_t requested = std::max<int64_t>(
        1, (laneCount * sew * 8 + target.getVlenBits() - 1) /
               target.getVlenBits());
    requested = std::max(requested, (8 * sew + elen - 1) / elen);
    for (int64_t legal : target.getLegalLMULEighths().asArrayRef())
      if (legal >= requested) {
        lmul = legal;
        break;
      }
    if (!lmul)
      return {};
  }
  int64_t replicaCount = 1;
  for (int64_t factor : replica)
    replicaCount *= factor;
  int64_t groups =
      carrier == "rvv" ? ((lmul + 7) / 8) * replicaCount : 0;
  return riscv::LayoutAttr::get(
      builder.getContext(), carrier, source.getAxisIds(),
      integers(builder, time), integers(builder, lane),
      integers(builder, replica), integers(builder, fragment),
      integers(builder, local), sew, lmul, carrier == "rvv" ? laneCount : 1,
      groups, reference.getValidity());
}

riscv::AccessAttr
riscv_internal::unassignedAccess(mlir::Builder &builder) {
  return riscv::AccessAttr::get(builder.getContext(), "unassigned", "opaque",
                                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, "none");
}

riscv::AccessAttr riscv_internal::denseAccess(mlir::Builder &builder,
                                              riscv::MemDescType memory,
                                              mlir::Type physicalValue) {
  riscv::LayoutAttr layout = layoutOf(physicalValue);
  int64_t mappedAxis = 0;
  if (layout) {
    for (auto [axis, factor] : llvm::zip(
             layout.getAxisIds().asArrayRef(),
             layout.getLaneFactors().asArrayRef()))
      if (factor > 1) {
        mappedAxis = axis;
        break;
      }
    if (!mappedAxis)
      for (auto [axis, factor] : llvm::zip(
               layout.getAxisIds().asArrayRef(),
               layout.getReplicaFactors().asArrayRef()))
        if (factor > 1) {
          mappedAxis = axis;
          break;
        }
  }
  llvm::StringRef form = "unit";
  if (mappedAxis) {
    auto axes = memory.getAxisIds().asArrayRef();
    auto found = llvm::find(axes, mappedAxis);
    if (found == axes.end()) {
      form = "indexed";
    } else {
      const size_t dimension = static_cast<size_t>(found - axes.begin());
      form = dimension < memory.getStrides().size() &&
                     memory.getStrides()[dimension] == 1
                 ? "unit"
                 : "strided";
    }
  }
  return riscv::AccessAttr::get(
      builder.getContext(), form, "dense",
      std::max<int64_t>(1, memory.getAlignment()),
      form == "indexed" ? 32 : 0, 0, 0, 0, 0, 0, 0, 0, 0, "none");
}

riscv::ConversionAttr
riscv_internal::conversion(mlir::Builder &builder, llvm::StringRef kind,
                           llvm::StringRef effect,
                           int64_t temporaryGroups) {
  return riscv::ConversionAttr::get(builder.getContext(), kind, effect,
                                    temporaryGroups);
}

riscv::ConversionAttr
riscv_internal::layoutConversion(mlir::Builder &builder,
                                 riscv::LayoutAttr source,
                                 riscv::LayoutAttr target) {
  llvm::StringRef kind = "reshape";
  llvm::StringRef effect = "pure";
  if (source.getCarrier() == target.getCarrier()) {
    bool registerToLane = false;
    bool timeToLane = false;
    bool laneToRegister = false;
    auto sourceAxes = source.getAxisIds().asArrayRef();
    auto targetAxes = target.getAxisIds().asArrayRef();
    for (auto [index, axis] : llvm::enumerate(sourceAxes)) {
      auto found = llvm::find(targetAxes, axis);
      if (found == targetAxes.end())
        continue;
      size_t targetIndex = static_cast<size_t>(found - targetAxes.begin());
      const int64_t sourceLane = source.getLaneFactors()[index];
      const int64_t targetLane = target.getLaneFactors()[targetIndex];
      registerToLane |=
          targetLane > sourceLane &&
          source.getReplicaFactors()[index] >
              target.getReplicaFactors()[targetIndex];
      timeToLane |= targetLane > sourceLane &&
                    source.getTimeFactors()[index] >
                        target.getTimeFactors()[targetIndex];
      laneToRegister |=
          sourceLane > targetLane &&
          target.getReplicaFactors()[targetIndex] >
              source.getReplicaFactors()[index];
    }
    kind = registerToLane ? "register_to_lane"
           : timeToLane   ? "time_to_lane"
           : laneToRegister ? "lane_to_register"
                            : "tuple";
  } else if (source.getCarrier() == "scalar" &&
             target.getCarrier() == "rvv") {
    bool registerToLane = false;
    bool timeToLane = false;
    auto sourceAxes = source.getAxisIds().asArrayRef();
    auto targetAxes = target.getAxisIds().asArrayRef();
    for (auto [index, axis] : llvm::enumerate(sourceAxes)) {
      auto found = llvm::find(targetAxes, axis);
      if (found == targetAxes.end())
        continue;
      size_t targetIndex = static_cast<size_t>(found - targetAxes.begin());
      if (target.getLaneFactors()[targetIndex] <= 1)
        continue;
      registerToLane |= source.getReplicaFactors()[index] >
                        target.getReplicaFactors()[targetIndex];
      timeToLane |= source.getTimeFactors()[index] >
                    target.getTimeFactors()[targetIndex];
    }
    // Moving both issue-time and register factors into one lane axis needs a
    // distinct typed conversion.  Keep it illegal instead of assigning either
    // existing kind and relying on the emitter to reinterpret the layout.
    kind = registerToLane && timeToLane
               ? "reshape"
           : registerToLane ? "register_to_lane"
           : timeToLane     ? "time_to_lane"
                            : "splat";
  } else if (source.getCarrier() == "rvv" &&
             target.getCarrier() == "scalar") {
    kind = "extract";
  } else if (source.getCarrier() == "local") {
    kind = "local_load";
    effect = "read";
  } else if (target.getCarrier() == "local") {
    kind = "local_store";
    effect = "write";
  }
  return conversion(builder, kind, effect);
}

riscv::ScheduleAttr riscv_internal::schedule(mlir::Builder &builder,
                                             int64_t unroll,
                                             int64_t pipelineDepth,
                                             int64_t prefetchDistance) {
  return riscv::ScheduleAttr::get(
      builder.getContext(), pipelineDepth > 1 ? "pipelined" : "sequential",
      unroll, pipelineDepth, pipelineDepth, prefetchDistance);
}

riscv::ImplementationAttr riscv_internal::implementation(
    mlir::Builder &builder, llvm::StringRef engine, llvm::StringRef family,
    llvm::StringRef operation, llvm::ArrayRef<int64_t> parameters) {
  return riscv::ImplementationAttr::get(builder.getContext(), engine, family,
                                        operation, integers(builder, parameters));
}

riscv::LeafAttr riscv_internal::unselectedLeaf(mlir::Builder &builder) {
  return riscv::LeafAttr::get(builder.getContext(), "unselected", "", "", "",
                              0, 0, 0, 0, "none", "exact",
                              integers(builder, {}), 0);
}

riscv::LeafAttr riscv_internal::leaf(
    mlir::Builder &builder, llvm::StringRef engine, llvm::StringRef family,
    llvm::StringRef instruction, llvm::StringRef spelling,
    int64_t operandGroups, int64_t resultGroups, int64_t temporaryGroups,
    int64_t fragmentGroups, llvm::StringRef mask, llvm::StringRef tail,
    llvm::ArrayRef<int64_t> parameters, int64_t localBytes) {
  return riscv::LeafAttr::get(
      builder.getContext(), engine, family, instruction, spelling, operandGroups,
      resultGroups, temporaryGroups, fragmentGroups, mask, tail,
      integers(builder, parameters), localBytes);
}

riscv::LeafAttr riscv_internal::readSnapshotLeaf(
    mlir::Builder &builder, riscv::TargetAttr target, int64_t bytes) {
  if (target.getHasRVV()) {
    int64_t selectedLMUL = 8;
    int64_t chunkBytes = target.getVlenBits() / 8;
    int64_t transfers = bytes / chunkBytes + (bytes % chunkBytes != 0);
    for (int64_t lmul : {16, 32}) {
      if (!llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), lmul))
        continue;
      const int64_t capacity = target.getVlenBits() * lmul / 64;
      const int64_t count = bytes / capacity + (bytes % capacity != 0);
      if (count < transfers) {
        selectedLMUL = lmul;
        chunkBytes = capacity;
        transfers = count;
      }
    }
    return leaf(builder, "transfer", "load", "rvv.read-snapshot",
                "rvv.read-snapshot", 0, 0, selectedLMUL / 8, 0, "none", "exact",
                {selectedLMUL, chunkBytes});
  }
  return leaf(builder, "transfer", "load", "scalar.read-snapshot",
              "scalar.read-snapshot", 0, 0);
}

riscv::TargetAttr
riscv_internal::target(mlir::Builder &builder,
                       const RISCVTargetProfile &profile) {
  llvm::SmallVector<int64_t> sews;
  llvm::SmallVector<int64_t> lmuls;
  llvm::SmallVector<mlir::Attribute> fragments;
  for (unsigned sew : profile.supportedSEW)
    sews.push_back(sew);
  for (int lmul : profile.legalLMULEighths)
    lmuls.push_back(lmul);
  llvm::StringRef partialCombinePolicy;
  switch (profile.partialCombinePolicy) {
  case RISCVPartialCombinePolicy::IndependentMultilevel:
    partialCombinePolicy = "independent-multilevel";
    break;
  case RISCVPartialCombinePolicy::Sequential:
    partialCombinePolicy = "sequential";
    break;
  }
  llvm::StringRef recordAxisPolicy;
  switch (profile.recordAxisPolicy) {
  case RISCVRecordAxisPolicy::WithinRecord:
    recordAxisPolicy = "within-record";
    break;
  case RISCVRecordAxisPolicy::AcrossRecords:
    recordAxisPolicy = "across-records";
    break;
  }
  for (const RISCVFragmentCapability &fragment : profile.fragmentCapabilities) {
    llvm::StringRef instruction;
    riscv::FragmentPackingAttr lhsPacking;
    riscv::FragmentPackingAttr rhsPacking;
    riscv::FragmentPackingAttr accumulatorPacking;
    int64_t mmaGroups = 0;
    int64_t mmaChunks = 0;
    mlir::ArrayAttr clobbers;
    bool volatileAsm = false;
    bool memoryClobber = false;
    switch (fragment.instruction) {
    case RISCVFragmentInstruction::SpacemitIME1I8MMA:
      instruction = "spacemit-ime1-i8-mma";
      lhsPacking = riscv::FragmentPackingAttr::get(
          builder.getContext(), "ime1-i8-m4-k8",
          integers(builder, {0, 1}), 4, 8, "row_major", "row_major", 8, 32);
      rhsPacking = riscv::FragmentPackingAttr::get(
          builder.getContext(), "ime1-i8-n4-k8-transposed",
          integers(builder, {1, 0}), 4, 8, "row_major", "row_major", 8, 32);
      accumulatorPacking = riscv::FragmentPackingAttr::get(
          builder.getContext(), "ime1-i32-m4-n4",
          integers(builder, {0, 1}), 4, 4, "row_major", "row_major", 32, 32);
      mmaGroups = 1;
      mmaChunks = 1;
      clobbers = builder.getArrayAttr(
          {builder.getStringAttr("t0"), builder.getStringAttr("v0"),
           builder.getStringAttr("v1"), builder.getStringAttr("v2"),
           builder.getStringAttr("v3")});
      volatileAsm = true;
      memoryClobber = true;
      break;
    }
    auto signedness = [](RISCVFragmentSignedness value) -> llvm::StringRef {
      return value == RISCVFragmentSignedness::Signed ? "signed" : "unsigned";
    };
    fragments.push_back(riscv::FragmentCapabilityAttr::get(
        builder.getContext(), instruction, signedness(fragment.lhsSignedness),
        signedness(fragment.rhsSignedness), fragment.lhsElementBits,
        fragment.rhsElementBits, fragment.accumulatorElementBits,
        fragment.mFactor, fragment.nFactor, fragment.kFactor,
        fragment.lhsResourceGroups, fragment.rhsResourceGroups,
        fragment.accumulatorResourceGroups, lhsPacking, rhsPacking,
        accumulatorPacking, mmaGroups, mmaChunks, clobbers, volatileAsm,
        memoryClobber));
  }
  return riscv::TargetAttr::get(
      builder.getContext(), profile.triple, profile.march, profile.abi,
      profile.hasRVV, profile.hasVectorF16, profile.hasIndexedMemory,
      profile.hasSegmentMemory, profile.hasWideningInteger,
      profile.hasWideningFloat,
      profile.vlenBits, profile.vectorRegisters, profile.maxPrivateStackBytes,
      integers(builder, sews), integers(builder, lmuls),
      profile.maxWideningCombineGroups, partialCombinePolicy, recordAxisPolicy,
      builder.getArrayAttr(fragments));
}

mlir::Type riscv_internal::logicalElement(mlir::Type type) {
  if (auto value = mlir::dyn_cast<kernel::ValueType>(type))
    return value.getElementType();
  if (auto view = mlir::dyn_cast<kernel::ViewType>(type))
    return view.getEncoding();
  if (auto slice = mlir::dyn_cast<kernel::SliceType>(type))
    return slice.getEncoding();
  if (auto value = mlir::dyn_cast<riscv::ValueType>(type))
    return value.getElementType();
  if (auto memory = mlir::dyn_cast<riscv::MemDescType>(type))
    return memory.getEncoding();
  if (auto local = mlir::dyn_cast<riscv::LocalType>(type))
    return local.getElementType();
  if (auto fragment = mlir::dyn_cast<riscv::FragmentType>(type))
    return fragment.getElementType();
  return type;
}

llvm::ArrayRef<int64_t> riscv_internal::logicalShape(mlir::Type type) {
  if (auto value = mlir::dyn_cast<kernel::ValueType>(type))
    return value.getShape().asArrayRef();
  if (auto view = mlir::dyn_cast<kernel::ViewType>(type))
    return view.getShape().asArrayRef();
  if (auto slice = mlir::dyn_cast<kernel::SliceType>(type))
    return slice.getShape().asArrayRef();
  if (auto value = mlir::dyn_cast<riscv::ValueType>(type))
    return value.getShape().asArrayRef();
  if (auto memory = mlir::dyn_cast<riscv::MemDescType>(type))
    return memory.getShape().asArrayRef();
  if (auto local = mlir::dyn_cast<riscv::LocalType>(type))
    return local.getShape().asArrayRef();
  if (auto fragment = mlir::dyn_cast<riscv::FragmentType>(type))
    return fragment.getShape().asArrayRef();
  return {};
}

llvm::ArrayRef<int64_t> riscv_internal::logicalAxes(mlir::Type type) {
  if (auto value = mlir::dyn_cast<kernel::ValueType>(type))
    return value.getAxisIds().asArrayRef();
  if (auto view = mlir::dyn_cast<kernel::ViewType>(type))
    return view.getAxisIds().asArrayRef();
  if (auto slice = mlir::dyn_cast<kernel::SliceType>(type))
    return slice.getAxisIds().asArrayRef();
  if (auto value = mlir::dyn_cast<riscv::ValueType>(type))
    return value.getAxisIds().asArrayRef();
  if (auto memory = mlir::dyn_cast<riscv::MemDescType>(type))
    return memory.getAxisIds().asArrayRef();
  if (auto local = mlir::dyn_cast<riscv::LocalType>(type))
    return local.getAxisIds().asArrayRef();
  if (auto fragment = mlir::dyn_cast<riscv::FragmentType>(type))
    return fragment.getAxisIds().asArrayRef();
  return {};
}

unsigned riscv_internal::logicalBitWidth(mlir::Type type) {
  return bitWidth(logicalElement(type));
}

riscv::LayoutAttr riscv_internal::layoutOf(mlir::Type type) {
  if (auto value = mlir::dyn_cast<riscv::ValueType>(type))
    return value.getLayout();
  if (auto fragment = mlir::dyn_cast<riscv::FragmentType>(type))
    return fragment.getLayout();
  return {};
}

mlir::Type riscv_internal::withLayout(mlir::Type type,
                                      riscv::LayoutAttr layout) {
  if (auto value = mlir::dyn_cast<riscv::ValueType>(type))
    return riscv::ValueType::get(type.getContext(), value.getElementType(),
                                 value.getShape(), value.getAxisIds(), layout);
  if (auto fragment = mlir::dyn_cast<riscv::FragmentType>(type))
    return riscv::FragmentType::get(
        type.getContext(), fragment.getFamily(), fragment.getRole(),
        fragment.getPacking(), fragment.getElementType(), fragment.getShape(),
        fragment.getAxisIds(), layout, fragment.getResourceGroups());
  return type;
}

std::string
riscv_internal::terminalInstruction(mlir::Operation *operation) {
  auto carrierOf = [](mlir::Type type) -> llvm::StringRef {
    if (auto value = mlir::dyn_cast<riscv::ValueType>(type))
      return value.getLayout().getCarrier();
    if (mlir::isa<riscv::FragmentType>(type))
      return "ime";
    return "scalar";
  };
  auto isFloat = [&](mlir::Type type) {
    return mlir::isa<mlir::FloatType>(logicalElement(type));
  };
  auto isUnsigned = [&](mlir::Type type) {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(logicalElement(type));
    return integer && integer.isUnsigned();
  };

  if (auto iota = mlir::dyn_cast<riscv::IotaOp>(operation)) {
    auto layout = layoutOf(iota.getResult().getType());
    if (layout && layout.getCarrier() == "rvv")
      return "rvv.iota";
    auto time = layout ? staticProduct(layout.getTimeFactors().asArrayRef())
                       : std::optional<int64_t>();
    auto replicas = layout
                        ? staticProduct(layout.getReplicaFactors().asArrayRef())
                        : std::optional<int64_t>();
    const bool tuple = time && replicas && *time > 0 && *replicas > 0 &&
                       *time <= std::numeric_limits<int64_t>::max() / *replicas &&
                       *time * *replicas > 1;
    return tuple ? "register.iota" : "scalar.iota";
  }
  if (auto unary = mlir::dyn_cast<riscv::UnaryOp>(operation)) {
    if (carrierOf(unary.getResult().getType()) != "rvv")
      return ("scalar." + unary.getKind()).str();
    if (unary.getKind() == "exp" &&
        logicalElement(unary.getResult().getType()).isF32())
      return "rvv.exp-approx-f32";
    if (unary.getKind() == "neg")
      return isFloat(unary.getResult().getType()) ? "rvv.vfneg.v"
                                                  : "rvv.vneg.v";
    if (unary.getKind() == "abs" &&
        isFloat(unary.getResult().getType()))
      return "rvv.vfabs.v";
    return {};
  }
  if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(operation)) {
    const bool floating = isFloat(binary.getResult().getType());
    const bool unsignedInteger = isUnsigned(binary.getResult().getType());
    std::string stem =
        binary.getKind() == "add" ? (floating ? "vfadd" : "vadd")
      : binary.getKind() == "sub" ? (floating ? "vfsub" : "vsub")
      : binary.getKind() == "mul" ? (floating ? "vfmul" : "vmul")
      : binary.getKind() == "div" ? (floating ? "vfdiv" : unsignedInteger ? "vdivu" : "vdiv")
      : binary.getKind() == "mod" ? (unsignedInteger ? "vremu" : "vrem")
      : binary.getKind() == "and" ? "vand"
      : binary.getKind() == "or"  ? "vor"
      : binary.getKind() == "xor" ? "vxor"
      : binary.getKind() == "shl" ? "vsll"
      : binary.getKind() == "shr" ? (unsignedInteger ? "vsrl" : "vsra")
      : binary.getKind() == "max" ? (floating ? "vfmax" : unsignedInteger ? "vmaxu" : "vmax")
      : binary.getKind() == "min" ? (floating ? "vfmin" : unsignedInteger ? "vminu" : "vmin")
                                     : std::string();
    if (stem.empty())
      return {};
    if (carrierOf(binary.getResult().getType()) != "rvv")
      return ("scalar." + binary.getKind()).str();
    const bool lhsVector = carrierOf(binary.getLhs().getType()) == "rvv";
    const bool rhsVector = carrierOf(binary.getRhs().getType()) == "rvv";
    if (lhsVector && rhsVector)
      return "rvv." + stem + ".vv";
    if (lhsVector != rhsVector) {
      const bool scalarOnLeft = !lhsVector;
      const bool commutative =
          binary.getKind() == "add" || binary.getKind() == "mul" ||
          binary.getKind() == "and" || binary.getKind() == "or" ||
          binary.getKind() == "xor" || binary.getKind() == "max" ||
          binary.getKind() == "min";
      if (scalarOnLeft && !commutative)
        return "rvv." + stem + ".vv.splat-lhs";
      return "rvv." + stem + (floating ? ".vf" : ".vx") +
             (scalarOnLeft ? ".swap" : "");
    }
    return {};
  }
  if (auto compare = mlir::dyn_cast<riscv::CompareOp>(operation))
    return (carrierOf(compare.getResult().getType()) == "rvv" ? "rvv.cmp."
                                                               : "scalar.cmp.") +
           compare.getPredicate().str();
  if (mlir::isa<riscv::CastOp, riscv::NarrowOp,
                riscv::WidenOp>(operation)) {
    mlir::Type source = logicalElement(operation->getOperand(0).getType());
    mlir::Type target = logicalElement(operation->getResult(0).getType());
    if (carrierOf(operation->getResult(0).getType()) != "rvv")
      return mlir::isa<riscv::WidenOp>(operation) ? "scalar.widen"
           : mlir::isa<riscv::NarrowOp>(operation) ? "scalar.narrow"
                                                    : "scalar.cast";
    if (source == target)
      return "rvv.identity";
    auto sourceInteger = mlir::dyn_cast<mlir::IntegerType>(source);
    auto targetInteger = mlir::dyn_cast<mlir::IntegerType>(target);
    if (sourceInteger && targetInteger &&
        sourceInteger.getWidth() == targetInteger.getWidth())
      return "rvv.reinterpret";
    if (sourceInteger && targetInteger && sourceInteger.isUnsigned() &&
        sourceInteger.getWidth() < targetInteger.getWidth() &&
        std::max<unsigned>(8, sourceInteger.getWidth()) ==
            std::max<unsigned>(8, targetInteger.getWidth()))
      return targetInteger.isSigned() ? "rvv.reinterpret" : "rvv.identity";
    if (source.isF16() && target.isF32())
      return "rvv.fwiden.f16-f32";
    if (source.isF32() && target.isF16())
      return "rvv.fnarrow.f32-f16";
    if (source.isInteger(32) && target.isF32())
      return "rvv.convert.i32-f32";
    if (sourceInteger && targetInteger &&
        targetInteger.getWidth() >
            std::max<unsigned>(8, sourceInteger.getWidth()) &&
        targetInteger.getWidth() %
                std::max<unsigned>(8, sourceInteger.getWidth()) ==
            0) {
      unsigned factor = targetInteger.getWidth() /
                        std::max<unsigned>(8, sourceInteger.getWidth());
      if (factor == 2 || factor == 4 || factor == 8)
        return std::string(sourceInteger.isSigned() ? "rvv.sext.vf"
                                                    : "rvv.zext.vf") +
               std::to_string(factor);
    }
    if (auto narrow = mlir::dyn_cast<riscv::NarrowOp>(operation);
        narrow && sourceInteger && targetInteger && !narrow.getSaturate() &&
        sourceInteger.getWidth() == targetInteger.getWidth() * 2 &&
        targetInteger.getWidth() >= 8)
      return "rvv.narrow.int.vf2";
    if (auto narrow = mlir::dyn_cast<riscv::NarrowOp>(operation);
        narrow && source.isF32() && targetInteger &&
        targetInteger.getWidth() == 8)
      return ("rvv.f32-i8-" +
              llvm::StringRef(narrow.getSaturate() ? "saturate." : "nonsaturating.") +
              narrow.getRounding())
          .str();
    return {};
  }
  if (auto reduce = mlir::dyn_cast<riscv::ReduceOp>(operation)) {
    auto input = mlir::dyn_cast<riscv::ValueType>(reduce.getInput().getType());
    if (!input || reduce.getAxis() < 0 ||
        reduce.getAxis() >= static_cast<int64_t>(input.getAxisIds().size()))
      return {};
    int64_t axis = input.getAxisIds()[reduce.getAxis()];
    bool registerAxis = false;
    for (auto [identity, factor] : llvm::zip(
             input.getAxisIds().asArrayRef(),
             input.getLayout().getReplicaFactors().asArrayRef()))
      registerAxis |= identity == axis && factor > 1;
    llvm::StringRef carrier = input.getLayout().getCarrier();
    if (carrier == "scalar")
      return registerAxis
                 ? "scalar.register-reduce." + reduce.getKind().str()
                 : std::string();
    if (carrier != "rvv")
      return {};
    return std::string(registerAxis ? "rvv.register-reduce."
                                    : "rvv.lane-reduce.") +
           reduce.getKind().str();
  }
  if (mlir::isa<riscv::Fold2Op>(operation))
    return "rvv.pair-fold-add";
  return {};
}

std::optional<int64_t>
riscv_internal::staticProduct(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values) {
    if (value <= 0 || result > std::numeric_limits<int64_t>::max() / value)
      return std::nullopt;
    result *= value;
  }
  return result;
}

std::string riscv_internal::printType(mlir::Type type) {
  std::string result;
  llvm::raw_string_ostream output(result);
  type.print(output);
  return result;
}

riscv::EncodingDeclOp
riscv_internal::findEncoding(mlir::Operation *operation,
                             llvm::StringRef family) {
  riscv::EncodingDeclOp result;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](riscv::EncodingDeclOp declaration) {
      if (!result && declaration.getSymName() == family)
        result = declaration;
    });
  return result;
}

llvm::StringRef riscv_internal::baseEncodingFamily(
    mlir::Operation *operation, kernel::EncodingType encoding) {
  if (encoding.getKind() == "base" || encoding.getKind() == "dense")
    return encoding.getFamily();
  llvm::StringRef result;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](riscv::DerivedEncodingOp derived) {
      if (result.empty() && derived.getResultFamily() == encoding.getFamily() &&
          derived.getLayoutIdentity() == encoding.getLayoutIdentity() &&
          derived.getParameterValues() == encoding.getParameters().asArrayRef())
        result = derived.getSourceFamily();
    });
  return result;
}

int64_t riscv_internal::interleaveRows(mlir::Operation *operation,
                                       kernel::EncodingType encoding) {
  mlir::Value owner;
  if (auto field = mlir::dyn_cast<riscv::FieldOp>(operation))
    owner = field.getOwner();
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  while (owner) {
    mlir::Operation *definition = owner.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      break;
    if (auto pack = mlir::dyn_cast<riscv::EncodedLocalBindOp>(definition))
      return pack.getPlan().getInterleaveRows();
    if (auto pack = mlir::dyn_cast<riscv::EncodedLocalPackOp>(definition))
      return pack.getPlan().getInterleaveRows();
    if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(definition)) {
      owner = extract.getInput();
      continue;
    }
    if (auto materialize =
            mlir::dyn_cast<riscv::RegisterMaterializeOp>(definition)) {
      owner = materialize.getInput();
      continue;
    }
    if (auto convert = mlir::dyn_cast<riscv::ConvertLayoutOp>(definition)) {
      owner = convert.getInput();
      continue;
    }
    break;
  }
  if (encoding.getKind() != "derived_instance")
    return 0;
  int64_t result = 0;
  if (auto module = operation->getParentOfType<mlir::ModuleOp>())
    module.walk([&](riscv::DerivedEncodingOp derived) {
      if (derived.getResultFamily() == encoding.getFamily() &&
          derived.getLayoutIdentity() == encoding.getLayoutIdentity() &&
          derived.getParameterValues() == encoding.getParameters().asArrayRef())
        result = derived.getInterleaveRows();
    });
  return result;
}

riscv_internal::FieldFacts
riscv_internal::fieldFacts(riscv::FieldOp operation) {
  auto ownerElement = logicalElement(operation.getOwner().getType());
  auto encoding = mlir::dyn_cast<kernel::EncodingType>(ownerElement);
  if (!encoding)
    return {};
  llvm::StringRef base = baseEncodingFamily(operation, encoding);
  riscv::EncodingDeclOp declaration = findEncoding(operation, base);
  if (!declaration)
    return {};
  size_t index = declaration.getFieldNames().size();
  for (auto [position, name] : llvm::enumerate(declaration.getFieldNames()))
    if (mlir::cast<mlir::StringAttr>(name).getValue() == operation.getName()) {
      index = position;
      break;
    }
  if (index == declaration.getFieldNames().size())
    return {};

  FieldFacts result;
  auto fieldShape =
      mlir::cast<mlir::DenseI64ArrayAttr>(declaration.getFieldShapes()[index]);
  result.scalarPerRecord = fieldShape.empty();
  result.logicalRank = fieldShape.size();
  result.bitOffset = declaration.getFieldBitOffsets()[index];
  result.storageBits = declaration.getFieldStorageBits()[index];
  if (result.bitOffset % 8 == 0)
    result.alignment = std::gcd<int64_t>(declaration.getAlignment(),
                                         result.bitOffset / 8);
  auto layouts =
      mlir::cast<mlir::ArrayAttr>(declaration.getFieldLayouts()[index]);
  auto layoutKind = [](mlir::Attribute attribute) -> llvm::StringRef {
    auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(attribute);
    auto kind = dictionary ? dictionary.getAs<mlir::StringAttr>("kind")
                           : mlir::StringAttr();
    return kind ? kind.getValue() : llvm::StringRef();
  };
  llvm::StringRef kind =
      layouts.empty() ? llvm::StringRef() : layoutKind(layouts[0]);
  if (kind == "natural") {
    result.mapping = "natural";
  } else if (kind == "joined") {
    result.mapping = "joined";
    auto joined = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
    result.group = joined.getAs<mlir::IntegerAttr>("size").getInt();
    result.joinFields = joined.getAs<mlir::IntegerAttr>("fields").getInt();
    result.joinLowBits = joined.getAs<mlir::IntegerAttr>("low_bits").getInt();
    result.joinRole = joined.getAs<mlir::IntegerAttr>("role").getInt();
    result.order = joined.getAs<mlir::StringAttr>("order").getValue();
  } else if (kind == "grouped" && layouts.size() == 2 &&
             layoutKind(layouts[1]) == "layered") {
    result.mapping = "grouped_layered";
    auto grouped = mlir::cast<mlir::DictionaryAttr>(layouts[0]);
    auto layered = mlir::cast<mlir::DictionaryAttr>(layouts[1]);
    result.group = grouped.getAs<mlir::IntegerAttr>("size").getInt();
    result.layer = layered.getAs<mlir::IntegerAttr>("size").getInt();
    result.order = layered.getAs<mlir::StringAttr>("order").getValue();
  }
  return result;
}

std::optional<int64_t> riscv_internal::constantInt(mlir::Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<riscv::CastOp>()) {
      value = cast.getInput();
      continue;
    }
    if (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
      value = cast.getIn();
      continue;
    }
    if (auto cast = value.getDefiningOp<mlir::arith::IndexCastUIOp>()) {
      value = cast.getIn();
      continue;
    }
    break;
  }
  if (auto constant = value.getDefiningOp<riscv::ConstantOp>())
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return integer.getInt();
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return integer.getInt();
  return std::nullopt;
}

mlir::Value riscv_internal::stripRepresentationConversions(
    mlir::Value value,
    llvm::SmallVectorImpl<riscv::ConvertLayoutOp> *conversions) {
  while (true) {
    if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
      auto input = mlir::dyn_cast<riscv::ValueType>(conversion.getInput().getType());
      auto result = mlir::dyn_cast<riscv::ValueType>(conversion.getResult().getType());
      llvm::StringRef effect = conversion.getConversion().getEffect();
      if (!input || !result || input.getShape() != result.getShape() ||
          input.getAxisIds() != result.getAxisIds() ||
          (effect != "pure" && effect != "read"))
        break;
      if (conversions)
        conversions->push_back(conversion);
      value = conversion.getInput();
      continue;
    }
    if (auto materialize =
            value.getDefiningOp<riscv::RegisterMaterializeOp>()) {
      value = materialize.getInput();
      continue;
    }
    break;
  }
  return value;
}

bool riscv_internal::knownMultipleOf(mlir::Value value, int64_t divisor,
                                     unsigned depth) {
  if (divisor <= 0 || depth > 24)
    return false;
  if (auto constant = riscv_internal::constantInt(value))
    return *constant % divisor == 0;
  if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>())
    return riscv_internal::knownMultipleOf(conversion.getInput(), divisor,
                                           depth + 1);
  if (auto cast = value.getDefiningOp<riscv::CastOp>())
    return riscv_internal::knownMultipleOf(cast.getInput(), divisor, depth + 1);
  if (auto widen = value.getDefiningOp<riscv::WidenOp>())
    return riscv_internal::knownMultipleOf(widen.getInput(), divisor,
                                           depth + 1);
  auto binary = value.getDefiningOp<riscv::BinaryOp>();
  if (!binary)
    return false;
  if (binary.getKind() == "add" || binary.getKind() == "sub")
    return riscv_internal::knownMultipleOf(binary.getLhs(), divisor,
                                           depth + 1) &&
           riscv_internal::knownMultipleOf(binary.getRhs(), divisor,
                                           depth + 1);
  if (binary.getKind() != "mul")
    return false;
  if (auto lhs = riscv_internal::constantInt(binary.getLhs());
      lhs && *lhs % divisor == 0)
    return true;
  if (auto rhs = riscv_internal::constantInt(binary.getRhs());
      rhs && *rhs % divisor == 0)
    return true;
  return riscv_internal::knownMultipleOf(binary.getLhs(), divisor, depth + 1) ||
         riscv_internal::knownMultipleOf(binary.getRhs(), divisor, depth + 1);
}

namespace {

mlir::Value stripIndexWidthConversions(mlir::Value value) {
  while (true) {
    if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
      value = conversion.getInput();
      continue;
    }
    if (auto cast = value.getDefiningOp<riscv::CastOp>()) {
      value = cast.getInput();
      continue;
    }
    if (auto widen = value.getDefiningOp<riscv::WidenOp>()) {
      value = widen.getInput();
      continue;
    }
    return value;
  }
}

} // namespace

std::optional<riscv_internal::IndexedEntryRelation>
riscv_internal::analyzeIndexedEntryRelation(
    mlir::Value fullIndices, riscv::ValueType result,
    llvm::ArrayRef<int64_t> retainedAxes,
    llvm::ArrayRef<int64_t> retainedShape) {
  auto indices = mlir::dyn_cast<riscv::ValueType>(fullIndices.getType());
  auto resultAxes = result.getAxisIds().asArrayRef();
  auto resultShape = result.getShape().asArrayRef();
  if (!indices || retainedAxes.size() != retainedShape.size() ||
      resultAxes.size() != retainedAxes.size() + indices.getAxisIds().size() ||
      resultShape.size() != retainedShape.size() + indices.getShape().size() ||
      !llvm::equal(retainedAxes, resultAxes.take_front(retainedAxes.size())) ||
      !llvm::equal(retainedShape, resultShape.take_front(retainedShape.size())) ||
      !llvm::equal(indices.getAxisIds().asArrayRef(),
                   resultAxes.drop_front(retainedAxes.size())) ||
      !llvm::equal(indices.getShape().asArrayRef(),
                   resultShape.drop_front(retainedShape.size())))
    return std::nullopt;

  mlir::Value indexValue = stripRepresentationConversions(fullIndices);
  auto add = indexValue.getDefiningOp<riscv::BinaryOp>();
  if (!add || add.getKind() != "add")
    return std::nullopt;
  auto match = [&](mlir::Value entryTerm,
                   mlir::Value payloadTerm)
      -> std::optional<IndexedEntryRelation> {
    auto stripAxisBroadcast = [](mlir::Value value) {
      while (auto broadcast =
                 value.getDefiningOp<riscv::RVVAxisBroadcastOp>())
        value = broadcast.getInput();
      return value;
    };
    mlir::Value payloadSource =
        stripIndexWidthConversions(stripAxisBroadcast(payloadTerm));
    auto payloadIota = payloadSource.getDefiningOp<riscv::IotaOp>();
    auto payloadType = payloadIota
                           ? mlir::dyn_cast<riscv::ValueType>(
                                 payloadIota.getResult().getType())
                           : riscv::ValueType();
    if (!payloadIota || !payloadType || payloadType.getAxisIds().size() != 1 ||
        payloadIota.getStart() != 0 ||
        payloadIota.getEnd() !=
            static_cast<uint64_t>(payloadType.getShape()[0]) ||
        payloadType.getShape()[0] <= 0)
      return std::nullopt;
    const int64_t payloadAxis = payloadType.getAxisIds()[0];
    const int64_t payloadExtent = payloadType.getShape()[0];
    if (resultAxes.empty() || resultAxes.back() != payloadAxis ||
        resultShape.back() != payloadExtent)
      return std::nullopt;

    mlir::Value scaledEntry =
        stripRepresentationConversions(stripAxisBroadcast(entryTerm));
    auto multiply = scaledEntry.getDefiningOp<riscv::BinaryOp>();
    mlir::Value entryIndices;
    std::optional<int64_t> stride;
    int64_t indexDivisor = 1;
    if (multiply && multiply.getKind() == "mul") {
      if ((stride = constantInt(multiply.getRhs())))
        entryIndices = multiply.getLhs();
      else if ((stride = constantInt(multiply.getLhs())))
        entryIndices = multiply.getRhs();
    }
    if (!entryIndices || !stride || *stride < payloadExtent) {
      if (!riscv_internal::knownMultipleOf(scaledEntry, payloadExtent))
        return std::nullopt;
      entryIndices = scaledEntry;
      stride = payloadExtent;
      indexDivisor = payloadExtent;
    }
    entryIndices = stripRepresentationConversions(entryIndices);
    auto entryType = mlir::dyn_cast<riscv::ValueType>(entryIndices.getType());
    if (entryType) {
      if ((entryType.getLayout().getCarrier() != "unassigned" &&
           entryType.getLayout().getCarrier() != "scalar" &&
           entryType.getLayout().getCarrier() != "rvv") ||
          indices.getAxisIds().size() != entryType.getAxisIds().size() + 1 ||
          !llvm::equal(entryType.getAxisIds().asArrayRef(),
                       indices.getAxisIds().asArrayRef().drop_back()) ||
          !llvm::equal(entryType.getShape().asArrayRef(),
                       indices.getShape().asArrayRef().drop_back()))
        return std::nullopt;
    } else {
      auto integer = mlir::dyn_cast<mlir::IntegerType>(entryIndices.getType());
      if ((!entryIndices.getType().isIndex() &&
           (!integer || integer.isSigned())) ||
          indices.getAxisIds().size() != 1 || indices.getShape().size() != 1)
        return std::nullopt;
    }
    return IndexedEntryRelation{entryIndices, payloadAxis, payloadExtent,
                                *stride, indexDivisor};
  };
  if (auto relation = match(add.getLhs(), add.getRhs()))
    return relation;
  return match(add.getRhs(), add.getLhs());
}

std::optional<riscv_internal::IntegerRange>
riscv_internal::integerRange(mlir::Value value, unsigned depth) {
  if (depth > 16)
    return std::nullopt;
  mlir::Value source = stripRepresentationConversions(value);
  if (source != value)
    return integerRange(source, depth + 1);
  auto type = mlir::dyn_cast<mlir::IntegerType>(logicalElement(value.getType()));
  if (!type || type.getWidth() == 0 || type.getWidth() >= 63)
    return std::nullopt;
  const __int128 typeMinimum =
      type.isSigned() ? -(__int128{1} << (type.getWidth() - 1)) : 0;
  const __int128 typeMaximum =
      type.isSigned() ? (__int128{1} << (type.getWidth() - 1)) - 1
                      : (__int128{1} << type.getWidth()) - 1;
  auto checkedRange = [&](const __int128 minimum,
                          const __int128 maximum)
      -> std::optional<IntegerRange> {
    if (minimum < typeMinimum || maximum > typeMaximum || minimum > maximum)
      return std::nullopt;
    return IntegerRange{static_cast<int64_t>(minimum),
                        static_cast<int64_t>(maximum)};
  };
  auto constantRange = [](mlir::Attribute attribute)
      -> std::optional<IntegerRange> {
    auto integer = mlir::dyn_cast_or_null<mlir::IntegerAttr>(attribute);
    if (!integer)
      return std::nullopt;
    const int64_t constant = integer.getInt();
    return IntegerRange{constant, constant};
  };
  if (auto constant = value.getDefiningOp<riscv::ConstantOp>())
    if (auto range = constantRange(constant.getValue()))
      return range;
  if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto range = constantRange(constant.getValue()))
      return range;
  if (auto widen = value.getDefiningOp<riscv::WidenOp>())
    if (auto range = integerRange(widen.getInput(), depth + 1))
      return range;
  if (auto cast = value.getDefiningOp<riscv::CastOp>())
    if (auto range = integerRange(cast.getInput(), depth + 1))
      if (auto checked = checkedRange(range->minimum, range->maximum))
        return checked;
  if (auto narrow = value.getDefiningOp<riscv::NarrowOp>())
    if (!narrow.getSaturate())
      if (auto range = integerRange(narrow.getInput(), depth + 1))
        if (auto checked = checkedRange(range->minimum, range->maximum))
          return checked;
  // A typed logical extract selects elements without changing their numeric
  // interpretation.  Its result range is therefore a subset of the input
  // range, independent of the selected coordinates or physical layout.
  if (auto extract = value.getDefiningOp<riscv::ExtractOp>())
    if (auto range = integerRange(extract.getInput(), depth + 1))
      if (auto checked = checkedRange(range->minimum, range->maximum))
        return checked;
  if (auto binary = value.getDefiningOp<riscv::BinaryOp>()) {
    auto lhs = integerRange(binary.getLhs(), depth + 1);
    auto rhs = integerRange(binary.getRhs(), depth + 1);
    if (lhs && rhs) {
      if (binary.getKind() == "add")
        if (auto range = checkedRange(
                static_cast<__int128>(lhs->minimum) + rhs->minimum,
                static_cast<__int128>(lhs->maximum) + rhs->maximum))
          return range;
      if (binary.getKind() == "sub")
        if (auto range = checkedRange(
                static_cast<__int128>(lhs->minimum) - rhs->maximum,
                static_cast<__int128>(lhs->maximum) - rhs->minimum))
          return range;
      if (binary.getKind() == "mul") {
        const __int128 candidates[] = {
            static_cast<__int128>(lhs->minimum) * rhs->minimum,
            static_cast<__int128>(lhs->minimum) * rhs->maximum,
            static_cast<__int128>(lhs->maximum) * rhs->minimum,
            static_cast<__int128>(lhs->maximum) * rhs->maximum};
        if (auto range = checkedRange(
                *std::min_element(std::begin(candidates), std::end(candidates)),
                *std::max_element(std::begin(candidates), std::end(candidates))))
          return range;
      }
      if (binary.getKind() == "and") {
        const IntegerRange *mask =
            rhs->minimum == rhs->maximum && rhs->minimum >= 0 ? &*rhs
            : lhs->minimum == lhs->maximum && lhs->minimum >= 0 ? &*lhs
                                                                  : nullptr;
        if (mask)
          if (auto range = checkedRange(0, mask->maximum))
            return range;
      }
      if (binary.getKind() == "or" && lhs->minimum >= 0 &&
          rhs->minimum >= 0) {
        auto coveringMask = [](int64_t maximum) {
          uint64_t value = static_cast<uint64_t>(maximum);
          value |= value >> 1;
          value |= value >> 2;
          value |= value >> 4;
          value |= value >> 8;
          value |= value >> 16;
          value |= value >> 32;
          return value;
        };
        const __int128 maximum =
            static_cast<__int128>(coveringMask(lhs->maximum) |
                                  coveringMask(rhs->maximum));
        if (auto range = checkedRange(0, maximum))
          return range;
      }
      if (binary.getKind() == "shl" && lhs->minimum >= 0 &&
          rhs->minimum == rhs->maximum && rhs->minimum >= 0 &&
          rhs->minimum < static_cast<int64_t>(type.getWidth()))
        if (auto range = checkedRange(
                static_cast<__int128>(lhs->minimum) << rhs->minimum,
                static_cast<__int128>(lhs->maximum) << rhs->minimum))
          return range;
      if (binary.getKind() == "shr" && lhs->minimum >= 0 &&
          rhs->minimum == rhs->maximum && rhs->minimum >= 0 &&
          rhs->minimum < static_cast<int64_t>(type.getWidth()))
        if (auto range = checkedRange(lhs->minimum >> rhs->minimum,
                                      lhs->maximum >> rhs->minimum))
          return range;
    }
  }
  if (type.isSigned()) {
    const int64_t bound = int64_t{1} << (type.getWidth() - 1);
    return IntegerRange{-bound, bound - 1};
  }
  return IntegerRange{0, (int64_t{1} << type.getWidth()) - 1};
}

std::optional<int64_t>
riscv_internal::maximumMagnitude(mlir::Value value) {
  auto range = integerRange(value);
  if (!range)
    return std::nullopt;
  return std::max(range->maximum,
                  range->minimum < 0 ? -range->minimum : range->minimum);
}

riscv::FieldOp riscv_internal::sourceField(mlir::Value value) {
  while (mlir::Operation *definition = value.getDefiningOp()) {
    value = stripRepresentationConversions(value);
    definition = value.getDefiningOp();
    if (auto field = mlir::dyn_cast_or_null<riscv::FieldOp>(definition))
      return field;
    if (auto extract = mlir::dyn_cast_or_null<riscv::ExtractOp>(definition)) {
      value = extract.getInput();
      continue;
    }
    break;
  }
  return {};
}

riscv::LoadOp riscv_internal::sourceLoad(mlir::Value value) {
  while (mlir::Operation *definition = value.getDefiningOp()) {
    value = stripRepresentationConversions(value);
    definition = value.getDefiningOp();
    if (auto load = mlir::dyn_cast_or_null<riscv::LoadOp>(definition))
      return load;
    if (auto extract = mlir::dyn_cast_or_null<riscv::ExtractOp>(definition)) {
      value = extract.getInput();
      continue;
    }
    if (auto pack =
            mlir::dyn_cast_or_null<riscv::EncodedLocalBindOp>(definition)) {
      value = pack.getInput();
      continue;
    }
    if (auto pack =
            mlir::dyn_cast_or_null<riscv::EncodedLocalPackOp>(definition)) {
      value = pack.getInput();
      continue;
    }
    if (auto materialize =
            mlir::dyn_cast_or_null<riscv::RegisterMaterializeOp>(definition)) {
      value = materialize.getInput();
      continue;
    }
    break;
  }
  return {};
}

riscv::AccessAttr riscv_internal::accessOf(mlir::Value value) {
  while (mlir::Operation *definition = value.getDefiningOp()) {
    if (auto access = definition->getAttrOfType<riscv::AccessAttr>("access"))
      if (!(access.getForm() == "local" && access.getMapping() == "dense"))
        return access;
    value = stripRepresentationConversions(value);
    definition = value.getDefiningOp();
    if (auto extract = mlir::dyn_cast_or_null<riscv::ExtractOp>(definition)) {
      value = extract.getInput();
      continue;
    }
    break;
  }
  return {};
}

riscv::PhysicalPointOp riscv_internal::originPoint(mlir::Value value,
                                                   int64_t axis) {
  llvm::SmallVector<mlir::Value> worklist{value};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    current = stripRepresentationConversions(current);
    if (auto point = current.getDefiningOp<riscv::PhysicalPointOp>();
        point && point.getResult().getType().getDomain().getAxisId() == axis)
      return point;
    mlir::Operation *definition = current.getDefiningOp();
    if (!definition)
      continue;
    if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(definition)) {
      for (mlir::Value index : extract.getIndices())
        worklist.push_back(index);
      worklist.push_back(extract.getInput());
      continue;
    }
    if (auto slice = mlir::dyn_cast<riscv::SliceOp>(definition)) {
      for (mlir::Value index : slice.getIndices())
        worklist.push_back(index);
      worklist.push_back(slice.getBase());
      continue;
    }
    if (auto field = mlir::dyn_cast<riscv::FieldOp>(definition)) {
      worklist.push_back(field.getOwner());
      continue;
    }
    if (auto materialize =
            mlir::dyn_cast<riscv::RegisterMaterializeOp>(definition))
      worklist.push_back(materialize.getInput());
  }
  return {};
}

void riscv_internal::copyOrigin(mlir::Operation *source,
                                mlir::Operation *target) {
  if (mlir::Attribute origin = source->getAttr("source_origin")) {
    target->setAttr("source_origin", origin);
    return;
  }
  std::string spelling;
  llvm::raw_string_ostream output(spelling);
  source->getLoc().print(output);
  target->setAttr("source_origin",
                  mlir::StringAttr::get(target->getContext(), output.str()));
}
