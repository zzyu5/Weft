#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>

using namespace weft;

namespace {

int64_t product(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values)
    result *= value;
  return result;
}

bool hasAxis(riscv::ValueType value, int64_t axis) {
  return llvm::is_contained(value.getAxisIds().asArrayRef(), axis);
}

bool hasTopology(riscv::RVVWidenDotOp dot, llvm::StringRef kind) {
  return dot.getPartialTopology().getKind() == kind;
}

int64_t ownerDomain(mlir::Operation *operation) {
  for (mlir::Operation *parent = operation; parent;
       parent = parent->getParentOp()) {
    if (auto level =
            parent->getAttrOfType<riscv::LevelAttr>("weft.riscv.level"))
      return level.getDomainId();
  }
  return 0;
}

bool mayMoveReadAcross(mlir::Operation *operation) {
  if (mlir::isMemoryEffectFree(operation))
    return true;
  auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(operation);
  if (!effects)
    return false;
  llvm::SmallVector<mlir::MemoryEffects::EffectInstance> instances;
  effects.getEffects(instances);
  return llvm::all_of(instances, [](const auto &instance) {
    return mlir::isa<mlir::MemoryEffects::Read>(instance.getEffect());
  });
}

void sinkReplicaSupplies(riscv::RVVPartialSetOp partial) {
  llvm::SmallVector<riscv::RVVReplicaStorageLoadOp> supplies;
  for (mlir::Value operand : partial->getOperands()) {
    auto supply = operand.getDefiningOp<riscv::RVVReplicaStorageLoadOp>();
    if (!supply || !supply.getResult().hasOneUse() ||
        supply->getBlock() != partial->getBlock() ||
        llvm::is_contained(supplies, supply))
      continue;
    bool movable = true;
    for (mlir::Operation *cursor = supply->getNextNode();
         cursor && cursor != partial.getOperation(); cursor = cursor->getNextNode())
      if (!mayMoveReadAcross(cursor)) {
        movable = false;
        break;
      }
    if (movable)
      supplies.push_back(supply);
  }
  for (riscv::RVVReplicaStorageLoadOp supply : supplies)
    supply->moveBefore(partial);
}

std::optional<size_t> axisPosition(riscv::ValueType value, int64_t axis) {
  auto found = llvm::find(value.getAxisIds().asArrayRef(), axis);
  if (found == value.getAxisIds().asArrayRef().end())
    return std::nullopt;
  return static_cast<size_t>(found - value.getAxisIds().asArrayRef().begin());
}

riscv::ValueType projectOneWindow(mlir::Builder &builder, riscv::ValueType value,
                                  int64_t axis) {
  auto position = axisPosition(value, axis);
  if (!position)
    return {};
  llvm::SmallVector<int64_t> shape(value.getShape().asArrayRef());
  llvm::SmallVector<int64_t> time(value.getLayout().getTimeFactors().asArrayRef());
  const int64_t lanes = value.getLayout().getLaneFactors()[*position];
  if (lanes <= 1 || time[*position] <= 1)
    return {};
  shape[*position] = lanes;
  time[*position] = 1;
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), value.getLayout().getCarrier(),
      value.getLayout().getAxisIds(), riscv_internal::integers(builder, time),
      value.getLayout().getLaneFactors(), value.getLayout().getReplicaFactors(),
      value.getLayout().getFragmentFactors(), value.getLayout().getLocalFactors(),
      value.getLayout().getSew(), value.getLayout().getLmulEighths(),
      value.getLayout().getVl(), value.getLayout().getRegisterGroups(),
      value.getLayout().getValidity());
  return riscv::ValueType::get(builder.getContext(), value.getElementType(),
                               riscv_internal::integers(builder, shape),
                               value.getAxisIds(), layout);
}

struct ProjectedRoot {
  mlir::Value value;
  riscv::FieldOp field;
  riscv::PhysicalPointOp origin;
  riscv::ValueType type;
  riscv::AccessAttr access;
  int64_t base = 0;
  int64_t stride = 1;
  int64_t repeat = 1;
  int64_t extent = 0;
  bool layered = false;
  riscv::LayeredStreamGeometryAttr geometry;
};

std::optional<ProjectedRoot> projectedRoot(mlir::Value value, int64_t axis) {
  if (auto stream =
          value.getDefiningOp<riscv::RVVProjectedLayeredStreamOp>()) {
    return ProjectedRoot{stream.getResult(),
                         stream.getField().getDefiningOp<riscv::FieldOp>(),
                         stream.getOrigin().getDefiningOp<riscv::PhysicalPointOp>(),
                         stream.getResult().getType(),
                         stream.getAccess(),
                         static_cast<int64_t>(stream.getProjectionBase()),
                         static_cast<int64_t>(stream.getProjectionStride()),
                         static_cast<int64_t>(stream.getProjectionRepeat()),
                         static_cast<int64_t>(stream.getProjectionExtent()),
                         true,
                         stream.getGeometry()};
  }
  if (auto stream = value.getDefiningOp<riscv::RVVLayeredStreamOp>()) {
    auto field = stream.getField().getDefiningOp<riscv::FieldOp>();
    auto type = stream.getResult().getType();
    auto position = axisPosition(type, axis);
    if (!field || !position)
      return std::nullopt;
    return ProjectedRoot{stream.getResult(),
                         field,
                         riscv_internal::originPoint(field.getOwner(), axis),
                         type,
                         stream.getAccess(),
                         0,
                         1,
                         1,
                         type.getShape()[*position],
                         true,
                         stream.getGeometry()};
  }
  if (auto extract = value.getDefiningOp<riscv::ExtractOp>()) {
    auto field = extract.getInput().getDefiningOp<riscv::FieldOp>();
    auto type = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
    auto position = type ? axisPosition(type, axis) : std::optional<size_t>();
    if (!field || !type || !position)
      return std::nullopt;
    if (extract.getIndices().size() == 1) {
      auto point = riscv_internal::stripRepresentationConversions(
                       extract.getIndices().front())
                       .getDefiningOp<riscv::PhysicalPointOp>();
      int64_t domainSelectors = 0;
      bool selectorsLegal = true;
      for (mlir::Attribute selector : extract.getSelectors()) {
        llvm::StringRef name =
            mlir::cast<mlir::StringAttr>(selector).getValue();
        if (name == "domain")
          ++domainSelectors;
        else if (name != "all")
          selectorsLegal = false;
      }
      if (point && selectorsLegal && domainSelectors == 1 &&
          point.getResult().getType().getDomain().getAxisId() == axis)
        return ProjectedRoot{
            extract.getResult(), field, point, type, extract.getAccess(), 0, 1,
            1, type.getShape()[*position],
            extract.getAccess().getMapping() == "grouped_layered"};
    }
    auto pattern =
        extract->getAttrOfType<mlir::DenseI64ArrayAttr>("index_pattern");
    if (!pattern || pattern.size() != 3)
      return std::nullopt;
    size_t regular = extract.getSelectors().size();
    bool hasRegular = false;
    for (auto [index, selector] : llvm::enumerate(extract.getSelectors())) {
      llvm::StringRef name = mlir::cast<mlir::StringAttr>(selector).getValue();
      if (name == "regular") {
        if (regular != extract.getSelectors().size())
          return std::nullopt;
        regular = index;
        hasRegular = true;
      } else if (name != "all") {
        return std::nullopt;
      }
    }
    if (!hasRegular || regular != *position || !extract.getIndices().empty())
      return std::nullopt;
    return ProjectedRoot{extract.getResult(),
                         field,
                         riscv_internal::originPoint(field.getOwner(), axis),
                         type,
                         extract.getAccess(),
                         pattern[0],
                         pattern[1],
                         pattern[2],
                         type.getShape()[*position],
                         extract.getAccess().getMapping() == "grouped_layered"};
  }
  if (auto field = value.getDefiningOp<riscv::FieldOp>()) {
    auto type = mlir::dyn_cast<riscv::ValueType>(field.getResult().getType());
    auto position = type ? axisPosition(type, axis) : std::optional<size_t>();
    if (!type || !position)
      return std::nullopt;
    return ProjectedRoot{field.getResult(),
                         field,
                         riscv_internal::originPoint(field.getOwner(), axis),
                         type,
                         field.getAccess(),
                         0,
                         1,
                         1,
                         type.getShape()[*position],
                         field.getAccess().getMapping() == "grouped_layered"};
  }
  return std::nullopt;
}

void collectProjectedRoots(mlir::Value value, int64_t axis,
                           llvm::DenseSet<mlir::Operation *> &visited,
                           llvm::SmallVectorImpl<ProjectedRoot> &roots) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return;
  if (auto root = projectedRoot(value, axis)) {
    roots.push_back(*root);
    return;
  }
  if (!mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                 riscv::NarrowOp, riscv::WidenOp, riscv::ConvertLayoutOp,
                 riscv::LookupOp,
                 riscv::RVVWidenMultiplyOp,
                 riscv::RVVWidenScalarMultiplyOp,
                 riscv::RegisterMaterializeOp>(definition))
    return;
  for (mlir::Value operand : definition->getOperands())
    if (mlir::isa<riscv::ValueType>(operand.getType()))
      collectProjectedRoots(operand, axis, visited, roots);
}

bool dependsOnAny(mlir::Value value,
                  const llvm::DenseMap<mlir::Value, mlir::Value> &replacements,
                  llvm::DenseMap<mlir::Value, bool> &cache) {
  if (replacements.contains(value))
    return true;
  if (auto found = cache.find(value); found != cache.end())
    return found->second;
  bool dependent = false;
  if (mlir::Operation *definition = value.getDefiningOp())
    if (mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                  riscv::NarrowOp, riscv::WidenOp, riscv::ConvertLayoutOp,
                  riscv::LookupOp,
                  riscv::RVVWidenMultiplyOp,
                  riscv::RVVWidenScalarMultiplyOp,
                  riscv::RegisterMaterializeOp>(definition))
      for (mlir::Value operand : definition->getOperands())
        if (mlir::isa<riscv::ValueType>(operand.getType()))
          dependent |= dependsOnAny(operand, replacements, cache);
  cache[value] = dependent;
  return dependent;
}

bool canProjectWindowSlice(
    mlir::Value value,
    const llvm::DenseMap<mlir::Value, mlir::Value> &replacements, int64_t axis,
    mlir::Builder &builder, llvm::DenseMap<mlir::Value, bool> &dependence,
    llvm::DenseMap<mlir::Value, bool> &cache) {
  if (replacements.contains(value))
    return true;
  if (auto found = cache.find(value); found != cache.end())
    return found->second;
  auto type = mlir::dyn_cast<riscv::ValueType>(value.getType());
  auto position = type ? axisPosition(type, axis) : std::optional<size_t>();
  if (!type || !position) {
    cache[value] = true;
    return true;
  }
  if (!dependsOnAny(value, replacements, dependence)) {
    const bool alreadyOneWindow =
        type.getLayout().getTimeFactors()[*position] == 1;
    cache[value] = alreadyOneWindow;
    return alreadyOneWindow;
  }
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition ||
      !mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                 riscv::NarrowOp, riscv::WidenOp, riscv::ConvertLayoutOp,
                 riscv::LookupOp, riscv::RVVWidenMultiplyOp,
                 riscv::RVVWidenScalarMultiplyOp,
                 riscv::RegisterMaterializeOp>(definition) ||
      !projectOneWindow(builder, type, axis)) {
    cache[value] = false;
    return false;
  }
  bool projectable = true;
  for (mlir::Value operand : definition->getOperands())
    if (mlir::isa<riscv::ValueType>(operand.getType()))
      projectable &= canProjectWindowSlice(operand, replacements, axis, builder,
                                           dependence, cache);
  cache[value] = projectable;
  return projectable;
}

mlir::FailureOr<mlir::Value> cloneWindowSlice(
    mlir::Value value,
    const llvm::DenseMap<mlir::Value, mlir::Value> &replacements, int64_t axis,
    mlir::IRRewriter &rewriter, llvm::DenseMap<mlir::Value, mlir::Value> &clones,
    llvm::DenseMap<mlir::Value, bool> &dependence) {
  if (auto found = replacements.find(value); found != replacements.end())
    return found->second;
  if (!dependsOnAny(value, replacements, dependence))
    return value;
  if (auto found = clones.find(value); found != clones.end())
    return found->second;
  auto sourceType = mlir::dyn_cast<riscv::ValueType>(value.getType());
  auto resultType = sourceType ? projectOneWindow(rewriter, sourceType, axis)
                               : riscv::ValueType();
  mlir::Operation *definition = value.getDefiningOp();
  if (!resultType || !definition)
    return mlir::failure();

  auto cloneOperand = [&](mlir::Value operand) -> mlir::FailureOr<mlir::Value> {
    if (!mlir::isa<riscv::ValueType>(operand.getType()))
      return operand;
    return cloneWindowSlice(operand, replacements, axis, rewriter, clones,
                            dependence);
  };
  mlir::Value cloned;
  if (auto unary = mlir::dyn_cast<riscv::UnaryOp>(definition)) {
    auto input = cloneOperand(unary.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::UnaryOp>(unary.getLoc(), resultType, *input,
                                         unary.getKind(), unary.getLeaf())
                 .getResult();
  } else if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(definition)) {
    auto lhs = cloneOperand(binary.getLhs());
    auto rhs = cloneOperand(binary.getRhs());
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::BinaryOp>(binary.getLoc(), resultType, *lhs, *rhs,
                                          binary.getKind(), binary.getLeaf())
                 .getResult();
  } else if (auto cast = mlir::dyn_cast<riscv::CastOp>(definition)) {
    auto input = cloneOperand(cast.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::CastOp>(cast.getLoc(), resultType, *input,
                                        cast.getLeaf())
                 .getResult();
  } else if (auto narrow = mlir::dyn_cast<riscv::NarrowOp>(definition)) {
    auto input = cloneOperand(narrow.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::NarrowOp>(narrow.getLoc(), resultType, *input,
                                          narrow.getRounding(),
                                          narrow.getSaturate(), narrow.getLeaf())
                 .getResult();
  } else if (auto widen = mlir::dyn_cast<riscv::WidenOp>(definition)) {
    auto input = cloneOperand(widen.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::WidenOp>(widen.getLoc(), resultType, *input,
                                         widen.getLeaf())
                 .getResult();
  } else if (auto lookup = mlir::dyn_cast<riscv::LookupOp>(definition)) {
    auto indices = cloneOperand(lookup.getIndices());
    if (mlir::failed(indices))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::LookupOp>(
                     lookup.getLoc(), resultType, lookup.getTable(), *indices,
                     lookup.getBounds(), lookup.getAccess(), lookup.getLeaf())
                 .getResult();
  } else if (auto multiply =
                 mlir::dyn_cast<riscv::RVVWidenMultiplyOp>(definition)) {
    auto lhs = cloneOperand(multiply.getLhs());
    auto rhs = cloneOperand(multiply.getRhs());
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::RVVWidenMultiplyOp>(
                     multiply.getLoc(), resultType, *lhs, *rhs,
                     multiply.getLeaf())
                 .getResult();
  } else if (auto multiply =
                 mlir::dyn_cast<riscv::RVVWidenScalarMultiplyOp>(definition)) {
    auto lhs = cloneOperand(multiply.getLhs());
    if (mlir::failed(lhs))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::RVVWidenScalarMultiplyOp>(
                     multiply.getLoc(), resultType, *lhs, multiply.getRhs(),
                     multiply.getLeaf())
                 .getResult();
  } else if (auto conversion =
                 mlir::dyn_cast<riscv::ConvertLayoutOp>(definition)) {
    auto input = cloneOperand(conversion.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::ConvertLayoutOp>(
                     conversion.getLoc(), resultType, *input,
                     conversion.getConversion(), conversion.getSourceAccessAttr(),
                     conversion.getLeaf())
                 .getResult();
  } else if (auto materialize =
                 mlir::dyn_cast<riscv::RegisterMaterializeOp>(definition)) {
    auto input = cloneOperand(materialize.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::RegisterMaterializeOp>(
                     materialize.getLoc(), resultType, *input,
                     materialize.getOwnerDomainId(), materialize.getBirthId(),
                     materialize.getLifetimeEndDomainId(),
                     materialize.getRealization())
                 .getResult();
  } else {
    return mlir::failure();
  }
  riscv_internal::copyOrigin(definition, cloned.getDefiningOp());
  if (auto implementation =
          definition->getAttrOfType<riscv::ImplementationAttr>("implementation"))
    cloned.getDefiningOp()->setAttr("implementation", implementation);
  clones[value] = cloned;
  return cloned;
}

riscv::ValueType partialType(mlir::Builder &builder, riscv::ValueType lhs,
                             mlir::Type resultType, int64_t reductionAxis) {
  auto result = mlir::dyn_cast<riscv::ValueType>(resultType);
  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> replicas;
  llvm::StringRef validity = lhs.getLayout().getValidity();
  if (result) {
    if (result.getLayout().getCarrier() != "scalar")
      return {};
    axes.assign(result.getAxisIds().asArrayRef().begin(),
                result.getAxisIds().asArrayRef().end());
    shape.assign(result.getShape().asArrayRef().begin(),
                 result.getShape().asArrayRef().end());
    replicas.assign(result.getLayout().getReplicaFactors().asArrayRef().begin(),
                    result.getLayout().getReplicaFactors().asArrayRef().end());
    for (size_t position = 0; position < result.getShape().size(); ++position)
      if (result.getLayout().getTimeFactors()[position] != 1 ||
          result.getLayout().getLaneFactors()[position] != 1 ||
          (result.getShape()[position] > 0
               ? result.getLayout().getReplicaFactors()[position] !=
                     result.getShape()[position]
               : result.getLayout().getReplicaFactors()[position] <= 0) ||
          result.getLayout().getFragmentFactors()[position] != 1 ||
          result.getLayout().getLocalFactors()[position] != 1)
        return {};
    validity = result.getLayout().getValidity();
  }
  auto position = axisPosition(lhs, reductionAxis);
  if (!position)
    return {};
  auto inputElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  if (!inputElement || inputElement.isSignless() || inputElement.getWidth() > 16)
    return {};
  const int64_t partialWidth =
      2 * std::max<int64_t>(8, inputElement.getWidth());
  const int64_t lanes = lhs.getLayout().getLaneFactors()[*position];
  const int64_t partialLMUL = lhs.getLayout().getLmulEighths() * 2;
  axes.push_back(reductionAxis);
  shape.push_back(lanes);
  replicas.push_back(1);
  llvm::SmallVector<int64_t> time(axes.size(), 1);
  llvm::SmallVector<int64_t> lane(axes.size(), 1);
  llvm::SmallVector<int64_t> one(axes.size(), 1);
  lane.back() = lanes;
  const int64_t groupsPerVector = std::max<int64_t>(1, (partialLMUL + 7) / 8);
  const int64_t groups = groupsPerVector * product(replicas);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", riscv_internal::integers(builder, axes),
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replicas),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one), partialWidth, partialLMUL,
      lhs.getLayout().getVl(), groups, validity);
  auto element = mlir::IntegerType::get(builder.getContext(), partialWidth,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), element,
                               riscv_internal::integers(builder, shape),
                               riscv_internal::integers(builder, axes), layout);
}

riscv::ValueType projectReplicaReductionOperandType(
    mlir::Builder &builder, riscv::ValueType input, int64_t removedAxis) {
  auto removed = axisPosition(input, removedAxis);
  if (!removed || input.getLayout().getCarrier() != "rvv" ||
      input.getLayout().getTimeFactors()[*removed] <= 1 ||
      input.getLayout().getLaneFactors()[*removed] != 1 ||
      input.getLayout().getReplicaFactors()[*removed] != 1 ||
      input.getLayout().getFragmentFactors()[*removed] != 1 ||
      input.getLayout().getLocalFactors()[*removed] != 1)
    return {};

  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<int64_t> time;
  llvm::SmallVector<int64_t> lane;
  llvm::SmallVector<int64_t> replica;
  llvm::SmallVector<int64_t> fragment;
  llvm::SmallVector<int64_t> local;
  auto source = input.getLayout();
  for (size_t position = 0; position < input.getAxisIds().size(); ++position) {
    if (position == *removed)
      continue;
    shape.push_back(input.getShape()[position]);
    axes.push_back(input.getAxisIds()[position]);
    time.push_back(source.getTimeFactors()[position]);
    lane.push_back(source.getLaneFactors()[position]);
    replica.push_back(source.getReplicaFactors()[position]);
    fragment.push_back(source.getFragmentFactors()[position]);
    local.push_back(source.getLocalFactors()[position]);
  }
  const int64_t laneCount = product(lane);
  const int64_t replicaCount = product(replica);
  if (laneCount <= 1 || replicaCount <= 0 || product(time) != 1 ||
      product(fragment) != 1 || product(local) != 1)
    return {};
  const int64_t groupsPerVector =
      std::max<int64_t>(1, (source.getLmulEighths() + 7) / 8);
  auto axisIds = riscv_internal::integers(builder, axes);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds,
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replica),
      riscv_internal::integers(builder, fragment),
      riscv_internal::integers(builder, local), source.getSew(),
      source.getLmulEighths(), laneCount, groupsPerVector * replicaCount,
      source.getValidity());
  return riscv::ValueType::get(builder.getContext(), input.getElementType(),
                               riscv_internal::integers(builder, shape), axisIds,
                               layout);
}

riscv::ValueType replicaReducedAccumulatorType(
    mlir::Builder &builder, riscv::ValueType lhs, riscv::ValueType rhs,
    int64_t reductionAxis) {
  auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
  auto lhsReduction = axisPosition(lhs, reductionAxis);
  auto rhsReduction = axisPosition(rhs, reductionAxis);
  if (!lhsElement || !rhsElement || lhsElement.isSignless() ||
      rhsElement.isSignless() ||
      (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
      std::max<unsigned>(8, lhsElement.getWidth()) !=
          std::max<unsigned>(8, rhsElement.getWidth()) ||
      !lhsReduction || !rhsReduction ||
      lhs.getLayout().getCarrier() != "rvv" ||
      rhs.getLayout().getCarrier() != "rvv" ||
      lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
      lhs.getLayout().getLmulEighths() != rhs.getLayout().getLmulEighths() ||
      lhs.getLayout().getVl() != rhs.getLayout().getVl())
    return {};

  const int64_t reductionLanes =
      lhs.getLayout().getLaneFactors()[*lhsReduction];
  if (reductionLanes <= 1 ||
      reductionLanes != rhs.getLayout().getLaneFactors()[*rhsReduction] ||
      lhs.getShape()[*lhsReduction] != rhs.getShape()[*rhsReduction])
    return {};

  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> replicas;
  auto appendFree = [&](riscv::ValueType operand) -> bool {
    for (size_t position = 0; position < operand.getAxisIds().size(); ++position) {
      const int64_t axis = operand.getAxisIds()[position];
      if (axis == reductionAxis)
        continue;
      if (operand.getLayout().getTimeFactors()[position] != 1 ||
          operand.getLayout().getLaneFactors()[position] != 1 ||
          operand.getLayout().getReplicaFactors()[position] <= 0 ||
          operand.getLayout().getFragmentFactors()[position] != 1 ||
          operand.getLayout().getLocalFactors()[position] != 1)
        return false;
      auto found = llvm::find(axes, axis);
      if (found == axes.end()) {
        axes.push_back(axis);
        shape.push_back(operand.getShape()[position]);
        replicas.push_back(operand.getLayout().getReplicaFactors()[position]);
        continue;
      }
      const size_t target = static_cast<size_t>(found - axes.begin());
      if (shape[target] != operand.getShape()[position] ||
          replicas[target] != operand.getLayout().getReplicaFactors()[position])
        return false;
    }
    return true;
  };
  if (!appendFree(lhs) || !appendFree(rhs))
    return {};

  axes.push_back(reductionAxis);
  shape.push_back(lhs.getShape()[*lhsReduction]);
  replicas.push_back(1);
  llvm::SmallVector<int64_t> time(axes.size(), 1);
  llvm::SmallVector<int64_t> lane(axes.size(), 1);
  llvm::SmallVector<int64_t> one(axes.size(), 1);
  lane.back() = reductionLanes;
  const int64_t partialWidth =
      2 * std::max<unsigned>(8, lhsElement.getWidth());
  const int64_t partialLMUL = 2 * lhs.getLayout().getLmulEighths();
  const int64_t groupsPerVector =
      std::max<int64_t>(1, (partialLMUL + 7) / 8);
  const int64_t replicaCount = product(replicas);
  if (partialLMUL <= 0 || replicaCount <= 0)
    return {};
  auto axisIds = riscv_internal::integers(builder, axes);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds,
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replicas),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one), partialWidth, partialLMUL,
      reductionLanes, groupsPerVector * replicaCount,
      lhs.getLayout().getValidity());
  auto element = mlir::IntegerType::get(builder.getContext(), partialWidth,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), element,
                               riscv_internal::integers(builder, shape), axisIds,
                               layout);
}

riscv::ValueType partialSlotType(mlir::Builder &builder, riscv::ValueType operand,
                                 int64_t reductionAxis) {
  auto position = axisPosition(operand, reductionAxis);
  auto inputElement =
      mlir::dyn_cast<mlir::IntegerType>(operand.getElementType());
  if (!position || !inputElement || inputElement.isSignless() ||
      inputElement.getWidth() > 16)
    return {};
  const int64_t lanes = operand.getLayout().getLaneFactors()[*position];
  const int64_t partialWidth =
      2 * std::max<int64_t>(8, inputElement.getWidth());
  const int64_t partialLMUL = operand.getLayout().getLmulEighths() * 2;
  const int64_t groups = std::max<int64_t>(1, (partialLMUL + 7) / 8);
  auto axisIds = riscv_internal::integers(builder, {reductionAxis});
  auto one = riscv_internal::integers(builder, {1});
  auto lane = riscv_internal::integers(builder, {lanes});
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds, one, lane, one, one, one,
      partialWidth, partialLMUL, operand.getLayout().getVl(), groups,
      operand.getLayout().getValidity());
  auto element = mlir::IntegerType::get(builder.getContext(), partialWidth,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), element, lane, axisIds,
                               layout);
}

riscv::ValueType partialSlotType(mlir::Builder &builder,
                                 riscv::RVVWidenDotOp dot) {
  if (dot.getOver().empty() || dot.getReductionLanes() <= 0 ||
      dot.getSliceLmulEighths() <= 0)
    return {};
  auto inputElement = mlir::dyn_cast<mlir::IntegerType>(
      dot.getLhs().getType().getElementType());
  if (!inputElement || inputElement.isSignless() ||
      inputElement.getWidth() > 16)
    return {};
  const int64_t partialWidth =
      2 * std::max<int64_t>(8, inputElement.getWidth());
  const int64_t partialLMUL = 2 * dot.getSliceLmulEighths();
  const int64_t groups = std::max<int64_t>(1, (partialLMUL + 7) / 8);
  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<int64_t> lanes;
  int64_t laneProduct = 1;
  for (int64_t axis : dot.getOver()) {
    auto position = axisPosition(dot.getLhs().getType(), axis);
    if (!position)
      return {};
    const int64_t lane =
        dot.getLhs().getType().getLayout().getLaneFactors()[*position];
    if (lane <= 0 || laneProduct > dot.getReductionLanes() / lane)
      return {};
    axes.push_back(axis);
    lanes.push_back(lane);
    laneProduct *= lane;
  }
  if (laneProduct != dot.getReductionLanes())
    return {};
  llvm::SmallVector<int64_t> one(axes.size(), 1);
  auto axisIds = riscv_internal::integers(builder, axes);
  auto oneAttr = riscv_internal::integers(builder, one);
  auto lanesAttr = riscv_internal::integers(builder, lanes);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds, oneAttr, lanesAttr, oneAttr,
      oneAttr, oneAttr,
      partialWidth, partialLMUL, dot.getReductionLanes(), groups,
      dot.getLhs().getType().getLayout().getValidity());
  auto element = mlir::IntegerType::get(builder.getContext(), partialWidth,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), element, lanesAttr, axisIds,
                               layout);
}

riscv::ValueType groupedLanePartialSlotType(mlir::Builder &builder,
                                            riscv::ValueType lhs,
                                            riscv::ValueType rhs,
                                            int64_t reductionAxis,
                                            int64_t lanes) {
  auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(
      lhs.getElementType());
  auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(
      rhs.getElementType());
  if (lanes <= 0 || !lhsElement || !rhsElement ||
      lhsElement.isSignless() || rhsElement.isSignless() ||
      lhsElement.getWidth() != rhsElement.getWidth() ||
      lhsElement.getWidth() > 16)
    return {};
  const int64_t partialWidth =
      2 * std::max<int64_t>(8, lhsElement.getWidth());
  auto lhsLanes = riscv_internal::staticProduct(
      lhs.getLayout().getLaneFactors().asArrayRef());
  auto rhsLanes = riscv_internal::staticProduct(
      rhs.getLayout().getLaneFactors().asArrayRef());
  if (!lhsLanes || !rhsLanes || *lhsLanes <= 0 || *rhsLanes <= 0 ||
      lanes > *lhsLanes || lanes > *rhsLanes ||
      (lhs.getLayout().getLmulEighths() * lanes) %
          *lhsLanes ||
      (rhs.getLayout().getLmulEighths() * lanes) %
          *rhsLanes)
    return {};
  int64_t lhsSliceLMUL =
      lhs.getLayout().getLmulEighths() * lanes / *lhsLanes;
  int64_t rhsSliceLMUL =
      rhs.getLayout().getLmulEighths() * lanes / *rhsLanes;
  // A hardware sub-register slice is carried in m1, matching the typed
  // RVVPartialSet contract.  Otherwise the widened product has exactly twice
  // the LMUL of the actual source lane slice, not twice an independently
  // reconstructed per-dot factor.
  if (lhsSliceLMUL > 0 && lhsSliceLMUL < 8 &&
      lhs.getLayout().getLmulEighths() > lhsSliceLMUL)
    lhsSliceLMUL = 8;
  if (rhsSliceLMUL > 0 && rhsSliceLMUL < 8 &&
      rhs.getLayout().getLmulEighths() > rhsSliceLMUL)
    rhsSliceLMUL = 8;
  if (lhsSliceLMUL <= 0 || lhsSliceLMUL != rhsSliceLMUL ||
      lhsSliceLMUL > std::numeric_limits<int64_t>::max() / 2)
    return {};
  const int64_t partialLMUL = 2 * lhsSliceLMUL;
  auto axisIds = riscv_internal::integers(builder, {reductionAxis});
  auto one = riscv_internal::integers(builder, {1});
  auto lane = riscv_internal::integers(builder, {lanes});
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds, one, lane, one, one, one,
      partialWidth, partialLMUL, lanes,
      std::max<int64_t>(1, (partialLMUL + 7) / 8),
      lhs.getLayout().getValidity());
  auto element = mlir::IntegerType::get(builder.getContext(), partialWidth,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), element, lane, axisIds,
                               layout);
}

riscv::ValueType splitPartialSlotType(mlir::Builder &builder,
                                      riscv::ValueType source,
                                      int64_t reductionAxis, int64_t split) {
  auto position = axisPosition(source, reductionAxis);
  if (!position || split <= 1 ||
      source.getLayout().getLaneFactors()[*position] % split ||
      source.getLayout().getLmulEighths() % split ||
      source.getLayout().getVl() % split)
    return {};
  const int64_t lanes =
      source.getLayout().getLaneFactors()[*position] / split;
  const int64_t lmul = source.getLayout().getLmulEighths() / split;
  auto axisIds = riscv_internal::integers(builder, {reductionAxis});
  auto one = riscv_internal::integers(builder, {1});
  auto lane = riscv_internal::integers(builder, {lanes});
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds, one, lane, one, one, one,
      source.getLayout().getSew(), lmul, source.getLayout().getVl() / split,
      std::max<int64_t>(1, (lmul + 7) / 8),
      source.getLayout().getValidity());
  return riscv::ValueType::get(builder.getContext(), source.getElementType(),
                               lane, axisIds, layout);
}

riscv::ValueType widenPartialSlotType(mlir::Builder &builder,
                                      riscv::ValueType source) {
  auto element = mlir::dyn_cast<mlir::IntegerType>(source.getElementType());
  if (!element || !element.isSigned() || element.getWidth() != 16 ||
      source.getLayout().getCarrier() != "rvv")
    return {};
  const int64_t lmul = source.getLayout().getLmulEighths() * 2;
  const int64_t groups = std::max<int64_t>(1, (lmul + 7) / 8);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", source.getLayout().getAxisIds(),
      source.getLayout().getTimeFactors(), source.getLayout().getLaneFactors(),
      source.getLayout().getReplicaFactors(),
      source.getLayout().getFragmentFactors(),
      source.getLayout().getLocalFactors(), 32, lmul,
      source.getLayout().getVl(), groups, source.getLayout().getValidity());
  auto widened = mlir::IntegerType::get(builder.getContext(), 32,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), widened, source.getShape(),
                               source.getAxisIds(), layout);
}

riscv::ValueType vectorPartialSlotType(
    mlir::Builder &builder, riscv::ValueType operand,
    riscv::ValueType result, int64_t reductionAxis) {
  auto reductionPosition = axisPosition(operand, reductionAxis);
  auto inputElement =
      mlir::dyn_cast<mlir::IntegerType>(operand.getElementType());
  auto resultElement =
      mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  if (!reductionPosition || !inputElement || inputElement.isSignless() ||
      inputElement.getWidth() > 16 || !resultElement ||
      !resultElement.isSigned() || resultElement.getWidth() != 32 ||
      result.getLayout().getCarrier() != "rvv")
    return {};

  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> time;
  llvm::SmallVector<int64_t> lane;
  llvm::SmallVector<int64_t> replicas;
  llvm::SmallVector<int64_t> one(operand.getShape().size(), 1);
  for (size_t position = 0; position < operand.getShape().size(); ++position) {
    const int64_t axis = operand.getAxisIds()[position];
    if (axis == reductionAxis) {
      const int64_t lanes =
          operand.getLayout().getLaneFactors()[position];
      if (lanes <= 0)
        return {};
      shape.push_back(lanes);
      time.push_back(1);
      lane.push_back(lanes);
      replicas.push_back(1);
      continue;
    }
    auto resultPosition = axisPosition(result, axis);
    if (!resultPosition || result.getShape()[*resultPosition] !=
                               operand.getShape()[position] ||
        result.getLayout().getTimeFactors()[*resultPosition] != 1 ||
        result.getLayout().getReplicaFactors()[*resultPosition] != 1 ||
        result.getLayout().getFragmentFactors()[*resultPosition] != 1 ||
        result.getLayout().getLocalFactors()[*resultPosition] != 1 ||
        result.getLayout().getLaneFactors()[*resultPosition] !=
            operand.getLayout().getLaneFactors()[position])
      return {};
    shape.push_back(result.getShape()[*resultPosition]);
    time.push_back(1);
    lane.push_back(result.getLayout().getLaneFactors()[*resultPosition]);
    replicas.push_back(1);
  }
  if (result.getAxisIds().size() + 1 != operand.getAxisIds().size())
    return {};

  const int64_t partialWidth =
      2 * std::max<int64_t>(8, inputElement.getWidth());
  const int64_t partialLMUL = operand.getLayout().getLmulEighths() * 2;
  if (result.getLayout().getSew() != partialWidth * 2 ||
      result.getLayout().getLmulEighths() != partialLMUL * 2)
    return {};
  const int64_t groups = std::max<int64_t>(1, (partialLMUL + 7) / 8);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", operand.getAxisIds(),
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replicas),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one), partialWidth, partialLMUL,
      operand.getLayout().getVl(), groups,
      operand.getLayout().getValidity());
  auto element = mlir::IntegerType::get(builder.getContext(), partialWidth,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), element,
                               riscv_internal::integers(builder, shape),
                               operand.getAxisIds(), layout);
}

std::optional<std::string>
partialMultiplyInstruction(riscv::ValueType lhs, riscv::ValueType rhs) {
  auto lhsElement =
      mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement =
      mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
  if (!lhsElement || !rhsElement || lhsElement.isSignless() ||
      rhsElement.isSignless())
    return std::nullopt;
  if (lhsElement.isSigned() && rhsElement.isSigned())
    return std::string("rvv.vwmul.vv");
  if (lhsElement.isSigned() && rhsElement.isUnsigned())
    return rhsElement.getWidth() < rhs.getLayout().getSew()
               ? std::string("rvv.vwmul.vv.reinterpret-rhs")
               : std::string("rvv.vwmulsu.vv");
  if (lhsElement.isUnsigned() && rhsElement.isSigned())
    return lhsElement.getWidth() < lhs.getLayout().getSew()
               ? std::string("rvv.vwmul.vv.reinterpret-lhs")
               : std::string("rvv.vwmulsu.vv.swap");
  return std::nullopt;
}

riscv::ValueType reducedPartialSlotType(mlir::Builder &builder,
                                        riscv::ValueType partial,
                                        int64_t reductionAxis) {
  auto position = axisPosition(partial, reductionAxis);
  auto element = mlir::dyn_cast<mlir::IntegerType>(partial.getElementType());
  if (!position || !element || !element.isSigned() ||
      (element.getWidth() != 16 && element.getWidth() != 32))
    return {};
  auto axisIds = riscv_internal::integers(builder, {reductionAxis});
  auto one = riscv_internal::integers(builder, {1});
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds, one, one, one, one, one, 32, 8, 1,
      1, "full");
  auto resultElement = mlir::IntegerType::get(
      builder.getContext(), 32, mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), resultElement, one, axisIds,
                               layout);
}

std::optional<llvm::StringRef>
partialFinalizeInstruction(riscv::PartialSetType input,
                           int64_t reductionAxis) {
  auto partial = input.getPartialType();
  auto axis = llvm::find(partial.getAxisIds().asArrayRef(), reductionAxis);
  if (axis == partial.getAxisIds().asArrayRef().end())
    return std::nullopt;
  const size_t position = static_cast<size_t>(
      axis - partial.getAxisIds().asArrayRef().begin());
  return partial.getLayout().getLaneFactors()[position] == 1
             ? llvm::StringRef("rvv.partial-finalize.extract")
             : llvm::StringRef("rvv.partial-finalize.reduce");
}

bool isIntegerZero(mlir::Value value) {
  auto constant = value.getDefiningOp<riscv::ConstantOp>();
  if (!constant)
    return false;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  return integer && integer.getValue().isZero();
}

struct PartialAddLeaf {
  riscv::RVVPartialFinalizeOp finalize;
  riscv::RVVWidenDotOp dot;
};

bool collectPartialAddTree(
    mlir::Value value, riscv::BinaryOp root,
    llvm::SmallVectorImpl<riscv::BinaryOp> &adds,
    llvm::SmallVectorImpl<PartialAddLeaf> &leaves) {
  if (auto add = value.getDefiningOp<riscv::BinaryOp>()) {
    if (add.getKind() != "add" ||
        (add != root && !add.getResult().hasOneUse()))
      return false;
    adds.push_back(add);
    return collectPartialAddTree(add.getLhs(), root, adds, leaves) &&
           collectPartialAddTree(add.getRhs(), root, adds, leaves);
  }
  if (isIntegerZero(value))
    return true;
  auto finalize = value.getDefiningOp<riscv::RVVPartialFinalizeOp>();
  if (finalize) {
    auto input = finalize.getInput().getType();
    if (!finalize.getResult().hasOneUse() ||
        finalize.getResult().getType() != root.getResult().getType() ||
        input.getSlots() != 1 ||
        input.getReductionAxis() != finalize.getReductionAxis())
      return false;
    leaves.push_back(PartialAddLeaf{finalize, {}});
    return true;
  }
  auto dot = value.getDefiningOp<riscv::RVVWidenDotOp>();
  if (!dot || !dot.getResult().hasOneUse() ||
      dot.getResult().getType() != root.getResult().getType() ||
      dot.getOver().size() != 1)
    return false;
  leaves.push_back(PartialAddLeaf{{}, dot});
  return true;
}

std::optional<riscv::PartialSetType>
partialTypeForAddLeaf(mlir::Builder &builder, PartialAddLeaf leaf) {
  if (leaf.finalize)
    return leaf.finalize.getInput().getType();
  if (!leaf.dot)
    return std::nullopt;
  const int64_t reductionAxis = leaf.dot.getOver()[0];
  riscv::ValueType lhs = leaf.dot.getLhs().getType();
  riscv::ValueType rhs = leaf.dot.getRhs().getType();
  auto lhsStreams = riscv_internal::staticProduct(
      lhs.getLayout().getTimeFactors().asArrayRef());
  auto rhsStreams = riscv_internal::staticProduct(
      rhs.getLayout().getTimeFactors().asArrayRef());
  auto layoutPlan = leaf.dot.getPartialLayoutPlanAttr();
  auto slot = layoutPlan
                  ? mlir::dyn_cast<riscv::ValueType>(
                        layoutPlan.getPartialSlotType())
                  : riscv::ValueType();
  auto result = mlir::dyn_cast<mlir::IntegerType>(leaf.dot.getResult().getType());
  if (!lhsStreams || !rhsStreams || *lhsStreams != 1 || *rhsStreams != 1 ||
      !slot || !result || !result.isSigned() ||
      result.getWidth() != 32 || !partialMultiplyInstruction(lhs, rhs))
    return std::nullopt;
  return riscv::PartialSetType::get(
      builder.getContext(), slot, reductionAxis, 1,
      slot.getShape()[*axisPosition(slot, reductionAxis)],
      slot.getLayout().getRegisterGroups());
}

std::optional<int64_t> partialSplitFactor(riscv::PartialSetType source,
                                          riscv::PartialSetType target) {
  if (source.getSlots() != 1 || target.getSlots() != 1 ||
      source.getReductionAxis() != target.getReductionAxis() ||
      source.getPartialType().getElementType() !=
          target.getPartialType().getElementType() ||
      source.getPartialType().getAxisIds() !=
          target.getPartialType().getAxisIds())
    return std::nullopt;
  riscv::ValueType sourcePartial = source.getPartialType();
  riscv::ValueType targetPartial = target.getPartialType();
  auto sourcePosition =
      axisPosition(sourcePartial, source.getReductionAxis());
  auto targetPosition =
      axisPosition(targetPartial, target.getReductionAxis());
  if (!sourcePosition || !targetPosition ||
      *sourcePosition != *targetPosition ||
      sourcePartial.getShape().size() != targetPartial.getShape().size() ||
      sourcePartial.getLayout().getCarrier() != "rvv" ||
      targetPartial.getLayout().getCarrier() != "rvv" ||
      sourcePartial.getLayout().getSew() !=
          targetPartial.getLayout().getSew())
    return std::nullopt;
  const int64_t sourceLanes =
      sourcePartial.getLayout().getLaneFactors()[*sourcePosition];
  const int64_t targetLanes =
      targetPartial.getLayout().getLaneFactors()[*targetPosition];
  if (sourceLanes <= 0 || targetLanes <= 0 || sourceLanes % targetLanes)
    return std::nullopt;
  const int64_t split = sourceLanes / targetLanes;
  if (split <= 0 || source.getTermsPerSlot() % split ||
      sourcePartial.getLayout().getLmulEighths() !=
          targetPartial.getLayout().getLmulEighths() * split ||
      sourcePartial.getLayout().getVl() !=
          targetPartial.getLayout().getVl() * split)
    return std::nullopt;
  for (size_t position = 0; position < sourcePartial.getShape().size();
       ++position) {
    if (position == *sourcePosition) {
      if (sourcePartial.getShape()[position] !=
              targetPartial.getShape()[position] * split ||
          sourcePartial.getLayout().getTimeFactors()[position] != 1 ||
          targetPartial.getLayout().getTimeFactors()[position] != 1 ||
          sourcePartial.getLayout().getReplicaFactors()[position] != 1 ||
          targetPartial.getLayout().getReplicaFactors()[position] != 1)
        return std::nullopt;
      continue;
    }
    if (sourcePartial.getShape()[position] !=
            targetPartial.getShape()[position] ||
        sourcePartial.getLayout().getTimeFactors()[position] !=
            targetPartial.getLayout().getTimeFactors()[position] ||
        sourcePartial.getLayout().getLaneFactors()[position] !=
            targetPartial.getLayout().getLaneFactors()[position] ||
        sourcePartial.getLayout().getReplicaFactors()[position] !=
            targetPartial.getLayout().getReplicaFactors()[position] ||
        sourcePartial.getLayout().getFragmentFactors()[position] !=
            targetPartial.getLayout().getFragmentFactors()[position] ||
        sourcePartial.getLayout().getLocalFactors()[position] !=
            targetPartial.getLayout().getLocalFactors()[position])
      return std::nullopt;
  }
  return split;
}

std::optional<int64_t> projectReplica(riscv::ValueType source,
                                      riscv::ValueType result,
                                      int64_t resultPart) {
  auto resultParts = riscv_internal::staticProduct(
      result.getLayout().getReplicaFactors().asArrayRef());
  auto sourceParts = riscv_internal::staticProduct(
      source.getLayout().getReplicaFactors().asArrayRef());
  if (!resultParts || !sourceParts || resultPart < 0 ||
      resultPart >= *resultParts)
    return std::nullopt;
  llvm::DenseMap<int64_t, int64_t> coordinates;
  int64_t remaining = resultPart;
  for (int64_t position = static_cast<int64_t>(result.getAxisIds().size()) - 1;
       position >= 0; --position) {
    const int64_t factor = result.getLayout().getReplicaFactors()[position];
    if (factor <= 0)
      return std::nullopt;
    coordinates[result.getAxisIds()[position]] = remaining % factor;
    remaining /= factor;
  }
  int64_t sourcePart = 0;
  for (auto [axis, factor] :
       llvm::zip(source.getAxisIds().asArrayRef(),
                 source.getLayout().getReplicaFactors().asArrayRef())) {
    if (factor <= 0)
      return std::nullopt;
    int64_t coordinate = 0;
    if (factor > 1) {
      auto found = coordinates.find(axis);
      if (found == coordinates.end() || found->second >= factor)
        return std::nullopt;
      coordinate = found->second;
    }
    sourcePart = sourcePart * factor + coordinate;
  }
  return sourcePart < *sourceParts ? std::optional<int64_t>(sourcePart)
                                   : std::nullopt;
}

std::optional<int64_t> composeReplicaPart(
    riscv::ValueType value, llvm::ArrayRef<int64_t> outerAxes,
    int64_t outerPart, llvm::ArrayRef<int64_t> innerAxes, int64_t innerPart) {
  llvm::DenseMap<int64_t, int64_t> coordinates;
  auto decode = [&](llvm::ArrayRef<int64_t> axes,
                    int64_t part) -> mlir::LogicalResult {
    for (int64_t position = static_cast<int64_t>(axes.size()) - 1;
         position >= 0; --position) {
      auto axis = axisPosition(value, axes[position]);
      if (!axis)
        return mlir::failure();
      const int64_t factor = value.getLayout().getReplicaFactors()[*axis];
      if (factor <= 0)
        return mlir::failure();
      coordinates[axes[position]] = part % factor;
      part /= factor;
    }
    return part == 0 ? mlir::success() : mlir::failure();
  };
  if (mlir::failed(decode(outerAxes, outerPart)) ||
      mlir::failed(decode(innerAxes, innerPart)))
    return std::nullopt;
  int64_t result = 0;
  for (auto [axis, factor] :
       llvm::zip(value.getAxisIds().asArrayRef(),
                 value.getLayout().getReplicaFactors().asArrayRef())) {
    auto found = coordinates.find(axis);
    const int64_t coordinate = found == coordinates.end() ? 0 : found->second;
    if (factor <= 0 || coordinate < 0 || coordinate >= factor)
      return std::nullopt;
    result = result * factor + coordinate;
  }
  return result;
}

void eraseDeadChain(mlir::Value value,
                    const llvm::DenseSet<mlir::Value> &stops,
                    llvm::DenseSet<mlir::Operation *> &visited,
                    mlir::IRRewriter &rewriter) {
  if (stops.contains(value))
    return;
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return;
  llvm::SmallVector<mlir::Value> operands(definition->getOperands());
  if (llvm::all_of(definition->getResults(), [](mlir::Value result) {
        return result.use_empty();
      }) &&
      mlir::isa<riscv::RVVLayeredStreamOp,
                riscv::RVVProjectedLayeredStreamOp, riscv::ExtractOp,
                riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                riscv::NarrowOp, riscv::WidenOp,
                riscv::RVVWidenMultiplyOp,
                riscv::RVVWidenScalarMultiplyOp, riscv::RVVWidenDotOp,
                riscv::ConvertLayoutOp,
                riscv::RegisterMaterializeOp>(definition)) {
    rewriter.eraseOp(definition);
    for (mlir::Value operand : operands)
      eraseDeadChain(operand, stops, visited, rewriter);
  }
}

riscv::ValueType scalarReplicaType(mlir::Builder &builder,
                                   riscv::ValueType source) {
  llvm::SmallVector<int64_t> one(source.getShape().size(), 1);
  llvm::SmallVector<int64_t> replicas;
  for (auto [position, extent] :
       llvm::enumerate(source.getShape().asArrayRef())) {
    const int64_t replicasForAxis =
        extent > 0 ? extent
                   : source.getLayout().getReplicaFactors()[position];
    if (replicasForAxis <= 0)
      return {};
    replicas.push_back(replicasForAxis);
  }
  mlir::Type element = source.getElementType();
  unsigned width = 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element))
    width = integer.getWidth();
  else if (auto floating = mlir::dyn_cast<mlir::FloatType>(element))
    width = floating.getWidth();
  if (!width)
    return {};
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "scalar", source.getAxisIds(),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, replicas),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one), std::max<unsigned>(8, width), 0, 1,
      0, source.getLayout().getValidity());
  return riscv::ValueType::get(builder.getContext(), element, source.getShape(),
                               source.getAxisIds(), layout);
}

mlir::FailureOr<mlir::Value> rematerializeScalarReplicas(
    mlir::Value value, mlir::IRRewriter &rewriter,
    llvm::DenseMap<mlir::Value, mlir::Value> &memo) {
  auto sourceType = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!sourceType)
    return value;
  auto found = memo.find(value);
  if (found != memo.end())
    return found->second;
  bool alreadyScalar = sourceType.getLayout().getCarrier() == "scalar";
  for (int64_t factor :
       sourceType.getLayout().getTimeFactors().asArrayRef())
    alreadyScalar &= factor == 1;
  for (int64_t factor :
       sourceType.getLayout().getLaneFactors().asArrayRef())
    alreadyScalar &= factor == 1;
  if (alreadyScalar) {
    memo.try_emplace(value, value);
    return value;
  }

  if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
    if (conversion.getConversion().getEffect() != "pure")
      return mlir::failure();
    auto rematerialized = rematerializeScalarReplicas(conversion.getInput(),
                                                      rewriter, memo);
    if (mlir::failed(rematerialized))
      return mlir::failure();
    memo.try_emplace(value, *rematerialized);
    return *rematerialized;
  }

  mlir::Operation *producer = value.getDefiningOp();
  if (!producer || !mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp,
                              riscv::CastOp, riscv::NarrowOp,
                              riscv::WidenOp>(producer) ||
      producer->getNumResults() != 1)
    return mlir::failure();
  riscv::ValueType targetType = scalarReplicaType(rewriter, sourceType);
  if (!targetType)
    return mlir::failure();
  llvm::SmallVector<mlir::Value> operands;
  for (mlir::Value operand : producer->getOperands()) {
    auto rematerialized = rematerializeScalarReplicas(operand, rewriter, memo);
    if (mlir::failed(rematerialized))
      return mlir::failure();
    operands.push_back(*rematerialized);
  }
  mlir::OperationState state(producer->getLoc(), producer->getName());
  state.addOperands(operands);
  state.addTypes(targetType);
  for (mlir::NamedAttribute attribute : producer->getAttrs())
    if (attribute.getName() != "leaf")
      state.addAttribute(attribute.getName(), attribute.getValue());
  state.addAttribute("leaf", riscv_internal::unselectedLeaf(rewriter));
  mlir::Operation *clone = rewriter.create(state);
  mlir::Value result = clone->getResult(0);
  memo.try_emplace(value, result);
  return result;
}

std::optional<mlir::Value> narrowScaleSource(mlir::Value value) {
  mlir::Value cursor = value;
  while (auto conversion = cursor.getDefiningOp<riscv::ConvertLayoutOp>()) {
    if (conversion.getConversion().getEffect() != "pure")
      return std::nullopt;
    cursor = conversion.getInput();
  }
  auto widen = cursor.getDefiningOp<riscv::WidenOp>();
  if (!widen)
    return std::nullopt;
  auto source = mlir::dyn_cast<riscv::ValueType>(widen.getInput().getType());
  auto result = mlir::dyn_cast<riscv::ValueType>(widen.getResult().getType());
  auto sourceElement =
      source ? mlir::dyn_cast<mlir::IntegerType>(source.getElementType())
             : mlir::IntegerType();
  auto resultElement =
      result ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
             : mlir::IntegerType();
  if (!source || !result || !sourceElement || !sourceElement.isSigned() ||
      sourceElement.getWidth() <= 0 || sourceElement.getWidth() > 16 ||
      !resultElement || !resultElement.isSigned() ||
      resultElement.getWidth() != 32 ||
      source.getLayout().getCarrier() != "scalar" ||
      result.getLayout().getCarrier() != "scalar" ||
      source.getShape() != result.getShape() ||
      source.getAxisIds() != result.getAxisIds() ||
      source.getLayout().getTimeFactors() !=
          result.getLayout().getTimeFactors() ||
      source.getLayout().getLaneFactors() !=
          result.getLayout().getLaneFactors() ||
      source.getLayout().getReplicaFactors() !=
          result.getLayout().getReplicaFactors())
    return std::nullopt;
  return widen.getInput();
}

std::optional<mlir::Value>
materializeExactNarrowScale(mlir::Value value, mlir::IRRewriter &rewriter) {
  if (auto existing = narrowScaleSource(value))
    return existing;
  auto source = mlir::dyn_cast<riscv::ValueType>(value.getType());
  auto range = riscv_internal::integerRange(value);
  if (!source || source.getLayout().getCarrier() != "scalar" || !range ||
      range->minimum < std::numeric_limits<int16_t>::min() ||
      range->maximum > std::numeric_limits<int16_t>::max())
    return std::nullopt;
  auto element = mlir::IntegerType::get(rewriter.getContext(), 16,
                                        mlir::IntegerType::Signed);
  auto layout = riscv::LayoutAttr::get(
      rewriter.getContext(), "scalar", source.getLayout().getAxisIds(),
      source.getLayout().getTimeFactors(), source.getLayout().getLaneFactors(),
      source.getLayout().getReplicaFactors(),
      source.getLayout().getFragmentFactors(),
      source.getLayout().getLocalFactors(), 16, 0, 1, 0,
      source.getLayout().getValidity());
  auto target = riscv::ValueType::get(rewriter.getContext(), element,
                                      source.getShape(), source.getAxisIds(),
                                      layout);
  auto narrow = rewriter.create<riscv::NarrowOp>(
      value.getLoc(), target, value, "none", false,
      riscv_internal::unselectedLeaf(rewriter));
  if (mlir::Operation *definition = value.getDefiningOp())
    riscv_internal::copyOrigin(definition, narrow);
  return narrow.getResult();
}

struct ScaledPartialContribution {
  riscv::RVVWidenDotOp dot;
  riscv::BinaryOp scaleMultiply;
  riscv::BinaryOp carryAdd;
  mlir::Value scale;
  mlir::Value previous;
};

std::optional<ScaledPartialContribution>
matchScaledPartialContribution(mlir::Value value) {
  auto add = value.getDefiningOp<riscv::BinaryOp>();
  if (!add || add.getKind() != "add" || !add.getResult().hasOneUse())
    return std::nullopt;

  auto matchSide = [&](mlir::Value contribution,
                       mlir::Value previous)
      -> std::optional<ScaledPartialContribution> {
    auto multiply = contribution.getDefiningOp<riscv::BinaryOp>();
    if (!multiply || multiply.getKind() != "mul")
      return std::nullopt;
    auto lhsDot = multiply.getLhs().getDefiningOp<riscv::RVVWidenDotOp>();
    auto rhsDot = multiply.getRhs().getDefiningOp<riscv::RVVWidenDotOp>();
    if (static_cast<bool>(lhsDot) == static_cast<bool>(rhsDot))
      return std::nullopt;
    riscv::RVVWidenDotOp dot = lhsDot ? lhsDot : rhsDot;
    mlir::Value scale = lhsDot ? multiply.getRhs() : multiply.getLhs();
    auto scaleType = mlir::dyn_cast<riscv::ValueType>(scale.getType());
    auto scaleParts = scaleType
                          ? riscv_internal::staticProduct(
                                scaleType.getLayout()
                                    .getReplicaFactors()
                                    .asArrayRef())
                          : std::optional<int64_t>();
    auto scaleElement =
        scaleType
            ? mlir::dyn_cast<mlir::IntegerType>(scaleType.getElementType())
            : mlir::IntegerType();
    if (!scaleType || scaleType.getLayout().getCarrier() != "scalar" ||
        !scaleParts || *scaleParts <= 0 || !scaleElement ||
        !scaleElement.isSigned() || scaleElement.getWidth() != 32 ||
        dot.getPartialUnroll() <= 1 || dot.getOver().size() != 1 ||
        !dot.getResult().hasOneUse() || !multiply.getResult().hasOneUse())
      return std::nullopt;
    return ScaledPartialContribution{dot, multiply, add, scale, previous};
  };

  auto lhs = matchSide(add.getLhs(), add.getRhs());
  auto rhs = matchSide(add.getRhs(), add.getLhs());
  if (static_cast<bool>(lhs) == static_cast<bool>(rhs))
    return std::nullopt;
  return lhs ? lhs : rhs;
}

std::optional<llvm::SmallVector<ScaledPartialContribution>>
matchLevelScaledLoop(mlir::scf::ForOp loop) {
  if (loop.getInitArgs().size() != 1 || loop.getNumResults() != 1)
    return std::nullopt;
  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return std::nullopt;
  mlir::Value cursor = yield.getOperand(0);
  auto first = matchScaledPartialContribution(cursor);
  if (!first)
    return std::nullopt;
  const int64_t slots = first->dot.getPartialUnroll();
  if (slots <= 1)
    return std::nullopt;
  llvm::SmallVector<ScaledPartialContribution> contributions;
  while (static_cast<int64_t>(contributions.size()) < slots) {
    auto contribution = matchScaledPartialContribution(cursor);
    if (!contribution)
      break;
    contributions.push_back(*contribution);
    cursor = contribution->previous;
  }
  if (static_cast<int64_t>(contributions.size()) != slots ||
      cursor != loop.getRegionIterArg(0))
    return std::nullopt;
  std::reverse(contributions.begin(), contributions.end());
  return contributions;
}

struct ReplicaScaledDotReduction {
  llvm::SmallVector<riscv::ReduceOp> reductions;
  llvm::SmallVector<riscv::ConvertLayoutOp> reductionConversions;
  llvm::SmallVector<int64_t> reducedAxes;
  riscv::BinaryOp scaleMultiply;
  riscv::RVVWidenDotOp dot;
  mlir::Value dotSide;
  mlir::Value scale;
};

std::optional<ReplicaScaledDotReduction>
matchReplicaScaledDotReduction(riscv::ReduceOp reduce) {
  llvm::SmallVector<riscv::ReduceOp> reductions;
  llvm::SmallVector<riscv::ConvertLayoutOp> reductionConversions;
  llvm::SmallVector<int64_t> reducedAxes;
  auto stripPureConversions = [&](mlir::Value value) {
    while (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
      if (conversion.getConversion().getEffect() != "pure" ||
          !conversion.getResult().hasOneUse())
        break;
      reductionConversions.push_back(conversion);
      value = conversion.getInput();
    }
    return value;
  };
  riscv::ReduceOp current = reduce;
  mlir::Value multipliedInput;
  while (current) {
    if (current.getKind() != "add")
      return std::nullopt;
    auto inputType = mlir::dyn_cast<riscv::ValueType>(current.getInput().getType());
    if (!inputType || current.getAxis() < 0 ||
        static_cast<size_t>(current.getAxis()) >= inputType.getAxisIds().size())
      return std::nullopt;
    reductions.push_back(current);
    reducedAxes.push_back(inputType.getAxisIds()[current.getAxis()]);
    mlir::Value reducedInput = stripPureConversions(current.getInput());
    auto inner = reducedInput.getDefiningOp<riscv::ReduceOp>();
    if (!inner) {
      multipliedInput = reducedInput;
      break;
    }
    if (!reducedInput.hasOneUse())
      return std::nullopt;
    current = inner;
  }
  auto multiply = multipliedInput.getDefiningOp<riscv::BinaryOp>();
  if (!multiply || multiply.getKind() != "mul" ||
      !multiply.getResult().hasOneUse())
    return std::nullopt;
  auto findDot = [](mlir::Value value) -> riscv::RVVWidenDotOp {
    while (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
      if (conversion.getConversion().getEffect() != "pure")
        return {};
      value = conversion.getInput();
    }
    return value.getDefiningOp<riscv::RVVWidenDotOp>();
  };
  auto lhsDot = findDot(multiply.getLhs());
  auto rhsDot = findDot(multiply.getRhs());
  if (static_cast<bool>(lhsDot) == static_cast<bool>(rhsDot))
    return std::nullopt;
  riscv::RVVWidenDotOp dot = lhsDot ? lhsDot : rhsDot;
  mlir::Value dotSide = lhsDot ? multiply.getLhs() : multiply.getRhs();
  mlir::Value scale = lhsDot ? multiply.getRhs() : multiply.getLhs();
  if (dot.getOver().size() != 1)
    return std::nullopt;
  return ReplicaScaledDotReduction{std::move(reductions),
                                   std::move(reductionConversions),
                                   std::move(reducedAxes), multiply, dot, dotSide,
                                   scale};
}

int64_t replicaProductForAxes(riscv::ValueType value,
                              llvm::ArrayRef<int64_t> selectedAxes) {
  int64_t result = 1;
  for (auto [axis, factor] :
       llvm::zip(value.getAxisIds().asArrayRef(),
                 value.getLayout().getReplicaFactors().asArrayRef())) {
    if (!llvm::is_contained(selectedAxes, axis))
      continue;
    if (factor <= 0 || result > std::numeric_limits<int64_t>::max() / factor)
      return -1;
    result *= factor;
  }
  return result;
}

int64_t replicaProduct(riscv::ValueType value) {
  return replicaProductForAxes(value, value.getAxisIds().asArrayRef());
}

llvm::SmallVector<int64_t>
remainingAxes(riscv::ValueType value, llvm::ArrayRef<int64_t> removed) {
  llvm::SmallVector<int64_t> result;
  for (int64_t axis : value.getAxisIds().asArrayRef())
    if (!llvm::is_contained(removed, axis))
      result.push_back(axis);
  return result;
}

riscv::PartialTopologyAttr makePartialTopology(
    mlir::Builder &builder, riscv::RVVWidenDotOp dot, llvm::StringRef kind,
    int64_t rootOperand, llvm::ArrayRef<int64_t> partialAxes,
    llvm::ArrayRef<int64_t> outputAxes, int64_t sourceSlots,
    int64_t partialSlots, int64_t outputReplicas, int64_t laneSplit,
    int64_t combineArity, int64_t resourceGroups) {
  llvm::SmallVector<int64_t> slotOrder;
  const bool chainOrdered =
      kind == "reduced_scaled" || (kind == "scaled" && laneSplit > 1);
  if (chainOrdered && laneSplit > 1 &&
      sourceSlots * laneSplit == partialSlots &&
      combineArity == sourceSlots)
    for (int64_t source = 0; source < sourceSlots; ++source)
      for (int64_t lane = 0; lane < laneSplit; ++lane)
        slotOrder.push_back(source * laneSplit + lane);
  else if (chainOrdered)
    for (int64_t chain = 0; chain < combineArity; ++chain)
      for (int64_t slot = chain; slot < partialSlots; slot += combineArity)
        slotOrder.push_back(slot);
  else
    for (int64_t slot = 0; slot < partialSlots; ++slot)
      slotOrder.push_back(slot);
  return riscv::PartialTopologyAttr::get(
      builder.getContext(), kind, rootOperand,
      builder.getDenseI64ArrayAttr(partialAxes),
      builder.getDenseI64ArrayAttr(outputAxes), sourceSlots, partialSlots,
      outputReplicas, laneSplit, combineArity,
      builder.getDenseI64ArrayAttr(slotOrder), resourceGroups);
}

struct LayeredTopologyFacts {
  int64_t reductionAxis = 0;
  int64_t rootOperand = -1;
  int64_t streams = 0;
  int64_t group = 0;
  int64_t layerExtent = 0;
  int64_t layers = 0;
  int64_t lanes = 0;
  int64_t windowsPerLayer = 0;
  int64_t windowCount = 0;
  riscv::ValueType lhsType;
  riscv::ValueType rhsType;
  ProjectedRoot layered;
  llvm::SmallVector<ProjectedRoot> roots;
  riscv::ValueType layeredWindowType;
  llvm::DenseMap<mlir::Value, riscv::ValueType> windowTypes;
  riscv::ValueType accumulatorType;
  mlir::IntegerType partialInteger;
};

std::optional<LayeredTopologyFacts>
analyzeLayeredTopology(mlir::Builder &builder, riscv::RVVWidenDotOp dot) {
  if (dot.getOver().size() != 1)
    return std::nullopt;
  LayeredTopologyFacts facts;
  facts.reductionAxis = dot.getOver()[0];
  facts.lhsType = dot.getLhs().getType();
  facts.rhsType = dot.getRhs().getType();
  auto lhsPosition = axisPosition(facts.lhsType, facts.reductionAxis);
  auto rhsPosition = axisPosition(facts.rhsType, facts.reductionAxis);
  if (!lhsPosition || !rhsPosition)
    return std::nullopt;
  facts.streams =
      facts.lhsType.getLayout().getTimeFactors()[*lhsPosition];
  if (facts.streams <= 1 ||
      facts.streams !=
          facts.rhsType.getLayout().getTimeFactors()[*rhsPosition])
    return std::nullopt;

  llvm::DenseSet<mlir::Operation *> lhsVisited;
  llvm::DenseSet<mlir::Operation *> rhsVisited;
  llvm::SmallVector<ProjectedRoot> lhsRoots;
  llvm::SmallVector<ProjectedRoot> rhsRoots;
  collectProjectedRoots(dot.getLhs(), facts.reductionAxis, lhsVisited, lhsRoots);
  collectProjectedRoots(dot.getRhs(), facts.reductionAxis, rhsVisited, rhsRoots);
  const int64_t lhsLayered = llvm::count_if(
      lhsRoots, [](const ProjectedRoot &root) { return root.layered; });
  const int64_t rhsLayered = llvm::count_if(
      rhsRoots, [](const ProjectedRoot &root) { return root.layered; });
  if (lhsLayered + rhsLayered != 1)
    return std::nullopt;
  facts.rootOperand = lhsLayered == 1 ? 0 : 1;
  facts.roots.assign(lhsRoots.begin(), lhsRoots.end());
  facts.roots.append(rhsRoots.begin(), rhsRoots.end());
  auto layeredIt = llvm::find_if(
      facts.roots, [](const ProjectedRoot &root) { return root.layered; });
  if (layeredIt == facts.roots.end())
    return std::nullopt;
  facts.layered = *layeredIt;
  if (!facts.layered.field || !facts.layered.origin)
    return std::nullopt;

  riscv::AccessAttr access = facts.layered.access;
  riscv::LayeredStreamGeometryAttr geometry = facts.layered.geometry;
  if (!geometry || geometry.getAxis() != facts.reductionAxis ||
      geometry.getStreamCount() != facts.streams)
    return std::nullopt;
  facts.group = geometry.getGroupSize();
  facts.layerExtent = geometry.getLayerSize();
  facts.layers = facts.layerExtent > 0 ? facts.group / facts.layerExtent : 0;
  auto layeredInteger = mlir::dyn_cast<mlir::IntegerType>(
      mlir::cast<riscv::ValueType>(facts.layered.field.getResult().getType())
          .getElementType());
  auto layeredPosition = axisPosition(facts.layered.type, facts.reductionAxis);
  if (!layeredPosition)
    return std::nullopt;
  facts.lanes = geometry.getLaneCount();
  if (!layeredInteger || layeredInteger.isSigned() || facts.group <= 0 ||
      facts.layerExtent <= 0 || facts.group % facts.layerExtent ||
      facts.layers <= 1 || layeredInteger.getWidth() * facts.layers != 8 ||
      access.getBitOffset() % 8 ||
      (access.getOrder() != "lo_first" && access.getOrder() != "hi_first") ||
      facts.layered.origin.getResult().getType().getDomain().getTail() !=
          "exact" ||
      facts.lanes <= 1 ||
      facts.layered.type.getLayout().getLaneFactors()[*layeredPosition] !=
          facts.lanes ||
      facts.layerExtent % facts.lanes ||
      facts.streams % facts.layers || facts.streams < facts.layers)
    return std::nullopt;
  facts.windowsPerLayer = facts.layerExtent / facts.lanes;
  facts.windowCount = geometry.getGroupForWindow().size();
  if (facts.windowCount <= 0 ||
      facts.windowCount != facts.streams / facts.layers)
    return std::nullopt;
  facts.layeredWindowType =
      projectOneWindow(builder, facts.layered.type, facts.reductionAxis);
  bool completeRoots = static_cast<bool>(facts.layeredWindowType);
  for (ProjectedRoot &root : facts.roots) {
    auto position = axisPosition(root.type, facts.reductionAxis);
    auto fieldType =
        root.field
            ? mlir::dyn_cast<riscv::ValueType>(root.field.getResult().getType())
            : riscv::ValueType();
    auto fieldPosition = fieldType
                             ? axisPosition(fieldType, facts.reductionAxis)
                             : std::optional<size_t>();
    auto integer = fieldType
                       ? mlir::dyn_cast<mlir::IntegerType>(
                             fieldType.getElementType())
                       : mlir::IntegerType();
    auto window = projectOneWindow(builder, root.type, facts.reductionAxis);
    const bool bounded =
        position && fieldPosition && root.base >= 0 && root.stride > 0 &&
        root.repeat > 0 && root.extent == root.type.getShape()[*position] &&
        root.extent > 0 && fieldType.getShape()[*fieldPosition] > 0 &&
        root.base <= fieldType.getShape()[*fieldPosition] - 1 &&
        (root.extent - 1) / root.repeat <=
            (fieldType.getShape()[*fieldPosition] - 1 - root.base) /
                root.stride;
    completeRoots &=
        root.field && root.origin && integer && window && bounded &&
        root.origin.getResult().getType().getDomain().getTail() == "exact" &&
        root.type.getLayout().getTimeFactors()[*position] == facts.streams &&
        root.type.getLayout().getLaneFactors()[*position] == facts.lanes &&
        root.extent == facts.layered.extent &&
        (root.repeat % facts.lanes == 0 || facts.lanes % root.repeat == 0) &&
        (root.layered ||
         (root.access.getMapping() == "natural" &&
          root.access.getBitOffset() % 8 == 0 && integer.getWidth() >= 8 &&
          integer.getWidth() <= 32 && integer.getWidth() % 8 == 0));
    if (window)
      facts.windowTypes[root.value] = window;
  }
  facts.accumulatorType = partialType(builder, facts.lhsType,
                                      dot.getResult().getType(),
                                      facts.reductionAxis);
  facts.partialInteger =
      facts.accumulatorType
          ? mlir::dyn_cast<mlir::IntegerType>(
                facts.accumulatorType.getElementType())
          : mlir::IntegerType();
  llvm::DenseMap<mlir::Value, mlir::Value> projectedRoots;
  for (const ProjectedRoot &root : facts.roots)
    projectedRoots[root.value] = root.value;
  llvm::DenseMap<mlir::Value, bool> dependence;
  llvm::DenseMap<mlir::Value, bool> projectable;
  const bool completeSlice =
      canProjectWindowSlice(dot.getLhs(), projectedRoots, facts.reductionAxis,
                            builder, dependence, projectable) &&
      canProjectWindowSlice(dot.getRhs(), projectedRoots, facts.reductionAxis,
                            builder, dependence, projectable);
  if (!completeRoots || !completeSlice || !facts.accumulatorType ||
      !facts.partialInteger ||
      (facts.partialInteger.getWidth() != 16 &&
       facts.partialInteger.getWidth() != 32))
    return std::nullopt;
  return facts;
}

struct ScaledTopologyFacts {
  llvm::SmallVector<int64_t> partialAxes;
  llvm::SmallVector<int64_t> outputAxes;
  int64_t partialSlots = 0;
  int64_t outputReplicas = 0;
};

struct ScaledSourceLaneGeometry {
  riscv::ValueType sourceSlot;
  riscv::ValueType splitSourceSlot;
  riscv::ValueType partialSlot;
  riscv::ValueType reducedSlot;
  int64_t sourceSetGroups = 0;
  int64_t reducedGroups = 0;
};

struct SourceLanePlan {
  int64_t sourceSlots = 0;
  int64_t laneSplit = 1;
  riscv::ValueType lhsType;
  riscv::ValueType rhsType;
  ScaledSourceLaneGeometry geometry;
  llvm::SmallVector<int64_t> lhsParts;
  llvm::SmallVector<int64_t> rhsParts;
  llvm::SmallVector<int64_t> lhsOffsets;
  llvm::SmallVector<int64_t> rhsOffsets;
};

std::optional<riscv::ValueType>
coalescePartialOperand(mlir::Builder &builder, riscv::ValueType source,
                       llvm::ArrayRef<int64_t> partialAxes,
                       int64_t requiredLanes, riscv::TargetAttr target) {
  auto currentLanes = riscv_internal::staticProduct(
      source.getLayout().getLaneFactors().asArrayRef());
  if (!currentLanes || *currentLanes <= 0 || requiredLanes <= 0)
    return std::nullopt;
  if (*currentLanes >= requiredLanes)
    return source;
  if (requiredLanes % *currentLanes)
    return std::nullopt;
  const int64_t factor = requiredLanes / *currentLanes;
  if (factor <= 1 || (factor & (factor - 1)))
    return std::nullopt;

  llvm::SmallVector<int64_t> time(
      source.getLayout().getTimeFactors().asArrayRef());
  llvm::SmallVector<int64_t> lane(
      source.getLayout().getLaneFactors().asArrayRef());
  llvm::SmallVector<int64_t> replica(
      source.getLayout().getReplicaFactors().asArrayRef());
  std::optional<size_t> movedPosition;
  for (int64_t axis : partialAxes) {
    auto position = axisPosition(source, axis);
    if (!position || replica[*position] < factor ||
        replica[*position] % factor)
      continue;
    if (movedPosition)
      return std::nullopt;
    movedPosition = *position;
  }
  if (!movedPosition)
    return std::nullopt;
  const size_t position = *movedPosition;
  const int64_t remainingReplica = replica[position] / factor;
  if (remainingReplica <= 0 ||
      time[position] > std::numeric_limits<int64_t>::max() /
                           remainingReplica ||
      lane[position] > std::numeric_limits<int64_t>::max() / factor ||
      source.getLayout().getLmulEighths() >
          std::numeric_limits<int64_t>::max() / factor ||
      source.getLayout().getVl() >
          std::numeric_limits<int64_t>::max() / factor)
    return std::nullopt;
  time[position] *= remainingReplica;
  lane[position] *= factor;
  replica[position] = 1;
  const int64_t lmul = source.getLayout().getLmulEighths() * factor;
  const int64_t vl = source.getLayout().getVl() * factor;
  if (!llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), lmul))
    return std::nullopt;
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", source.getAxisIds(),
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replica),
      source.getLayout().getFragmentFactors(),
      source.getLayout().getLocalFactors(), source.getLayout().getSew(), lmul,
      vl, std::max<int64_t>(1, (lmul + 7) / 8),
      source.getLayout().getValidity());
  auto result = riscv::ValueType::get(builder.getContext(),
                                      source.getElementType(), source.getShape(),
                                      source.getAxisIds(), layout);
  if (!riscv::supportsRVVLayout(target, layout))
    return std::nullopt;
  return result;
}

std::optional<ScaledSourceLaneGeometry>
deriveScaledSourceLaneGeometry(mlir::Builder &builder,
                               riscv::RVVWidenDotOp dot,
                               riscv::ValueType lhsSource,
                               riscv::ValueType rhsSource,
                               int64_t partialSlots, int64_t sourceSlots,
                               int64_t laneSplit) {
  if (dot.getOver().size() != 1 || partialSlots <= 0 || sourceSlots <= 0 ||
      laneSplit <= 1 || sourceSlots * laneSplit != partialSlots)
    return std::nullopt;
  const int64_t reductionAxis = dot.getOver()[0];
  const int64_t sourceLanes = dot.getReductionLanes() * laneSplit;
  auto sourceSlot = groupedLanePartialSlotType(
      builder, lhsSource, rhsSource, reductionAxis, sourceLanes);
  auto splitSourceSlot = sourceSlot
                             ? splitPartialSlotType(builder, sourceSlot,
                                                    reductionAxis, laneSplit)
                             : riscv::ValueType();
  auto partialSlot = splitSourceSlot;
  auto reducedSlot =
      partialSlot
          ? reducedPartialSlotType(builder, partialSlot, reductionAxis)
          : riscv::ValueType();
  auto kernel = dot->getParentOfType<riscv::KernelOp>();
  if (!kernel || !sourceSlot || !splitSourceSlot || !partialSlot || !reducedSlot)
    return std::nullopt;
  const auto legalLMUL =
      kernel.getTarget().getLegalLMULEighths().asArrayRef();
  if (!llvm::is_contained(legalLMUL,
                          sourceSlot.getLayout().getLmulEighths()) ||
      !llvm::is_contained(legalLMUL,
                          splitSourceSlot.getLayout().getLmulEighths()) ||
      !llvm::is_contained(legalLMUL,
                          reducedSlot.getLayout().getLmulEighths()))
    return std::nullopt;
  const int64_t sourceSetGroups =
      sourceSlots * sourceSlot.getLayout().getRegisterGroups();
  const int64_t reducedGroups =
      partialSlots * reducedSlot.getLayout().getRegisterGroups();
  // RVVPartialRepack is a typed register-group view: its verifier requires the
  // source and split result to carry the same aggregate resource groups.  It
  // therefore extends one resource identity instead of allocating a second
  // set.  Cross-value pressure is checked later from the materialized SSA live
  // intervals; this local legality check only rejects an individually
  // impossible aggregate.
  if (std::max(sourceSetGroups, reducedGroups) >
      kernel.getTarget().getVectorRegisters())
    return std::nullopt;
  return ScaledSourceLaneGeometry{sourceSlot, splitSourceSlot, partialSlot,
                                  reducedSlot, sourceSetGroups, reducedGroups};
}

std::optional<SourceLanePlan>
planScaledSourceLanes(mlir::Builder &builder, riscv::RVVWidenDotOp dot,
                      const ScaledTopologyFacts &facts) {
  auto result = mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
  auto kernel = dot->getParentOfType<riscv::KernelOp>();
  if (!result || !kernel || facts.partialSlots <= 1 ||
      facts.outputReplicas <= 0 || dot.getReductionLanes() <= 0 ||
      dot.getSliceLmulEighths() <= 0)
    return std::nullopt;

  for (int64_t split = facts.partialSlots; split > 1; --split) {
    if (facts.partialSlots % split)
      continue;
    const int64_t sourceLanes = split * dot.getReductionLanes();
    auto lhsSource = coalescePartialOperand(
        builder, dot.getLhs().getType(), facts.partialAxes, sourceLanes,
        kernel.getTarget());
    auto rhsSource = coalescePartialOperand(
        builder, dot.getRhs().getType(), facts.partialAxes, sourceLanes,
        kernel.getTarget());
    auto lhsSlices =
        lhsSource ? riscv_internal::planWidenDotLaneSlices(
                        *lhsSource, result, dot.getOver(), kernel.getTarget())
                  : std::optional<riscv_internal::WidenDotLaneSlicePlan>();
    auto rhsSlices =
        rhsSource ? riscv_internal::planWidenDotLaneSlices(
                        *rhsSource, result, dot.getOver(), kernel.getTarget())
                  : std::optional<riscv_internal::WidenDotLaneSlicePlan>();
    if (!lhsSource || !rhsSource || !lhsSlices || !rhsSlices ||
        sourceLanes > lhsSource->getLayout().getVl() ||
        sourceLanes > rhsSource->getLayout().getVl() ||
        lhsSlices->reductionLanes != dot.getReductionLanes() ||
        rhsSlices->reductionLanes != dot.getReductionLanes() ||
        lhsSlices->reductionStreams != dot.getReductionStreams() ||
        rhsSlices->reductionStreams != dot.getReductionStreams() ||
        lhsSlices->parts.size() != rhsSlices->parts.size() ||
        lhsSlices->offsets.size() != rhsSlices->offsets.size())
      continue;
    bool complete = true;
    llvm::SmallVector<int64_t> sourceLhsParts;
    llvm::SmallVector<int64_t> sourceRhsParts;
    llvm::SmallVector<int64_t> sourceLhsOffsets;
    llvm::SmallVector<int64_t> sourceRhsOffsets;
    for (int64_t outputPart = 0; outputPart < facts.outputReplicas && complete;
         ++outputPart) {
      for (int64_t first = 0; first < facts.partialSlots && complete;
           first += split) {
        std::optional<int64_t> firstLhsPart;
        std::optional<int64_t> firstRhsPart;
        std::optional<int64_t> firstLhsOffset;
        std::optional<int64_t> firstRhsOffset;
        for (int64_t inner = 0; inner < split; ++inner) {
          auto resultPart = composeReplicaPart(
              result, facts.outputAxes, outputPart, facts.partialAxes,
              first + inner);
          const int64_t planned =
              resultPart ? *resultPart * dot.getReductionStreams() : -1;
          if (!resultPart || planned < 0 ||
              planned >= static_cast<int64_t>(lhsSlices->parts.size()) ||
              planned >= static_cast<int64_t>(rhsSlices->parts.size()) ||
              *resultPart >=
                  static_cast<int64_t>(lhsSlices->offsets.size()) ||
              *resultPart >=
                  static_cast<int64_t>(rhsSlices->offsets.size())) {
            complete = false;
            break;
          }
          const int64_t lhsPart = lhsSlices->parts[planned];
          const int64_t rhsPart = rhsSlices->parts[planned];
          const int64_t lhsOffset = lhsSlices->offsets[*resultPart];
          const int64_t rhsOffset = rhsSlices->offsets[*resultPart];
          if (!firstLhsPart) {
            firstLhsPart = lhsPart;
            firstRhsPart = rhsPart;
            firstLhsOffset = lhsOffset;
            firstRhsOffset = rhsOffset;
          }
          const int64_t expectedLhs =
              *firstLhsOffset + inner * dot.getReductionLanes();
          const int64_t expectedRhs =
              *firstRhsOffset + inner * dot.getReductionLanes();
          if (lhsPart != *firstLhsPart || rhsPart != *firstRhsPart ||
              lhsOffset != expectedLhs || rhsOffset != expectedRhs) {
            complete = false;
            break;
          }
        }
        complete &= firstLhsOffset && firstRhsOffset && sourceLanes > 0 &&
                    *firstLhsOffset % sourceLanes == 0 &&
                    *firstRhsOffset % sourceLanes == 0;
        if (complete) {
          sourceLhsParts.push_back(*firstLhsPart);
          sourceRhsParts.push_back(*firstRhsPart);
          sourceLhsOffsets.push_back(*firstLhsOffset);
          sourceRhsOffsets.push_back(*firstRhsOffset);
        }
      }
    }
    const int64_t sourceSlots = facts.partialSlots / split;
    complete &= sourceLhsParts.size() ==
                    static_cast<size_t>(facts.outputReplicas * sourceSlots) &&
                sourceRhsParts.size() == sourceLhsParts.size() &&
                sourceLhsOffsets.size() == sourceLhsParts.size() &&
                sourceRhsOffsets.size() == sourceLhsParts.size();
    auto geometry = complete
                        ? deriveScaledSourceLaneGeometry(
                              builder, dot, *lhsSource, *rhsSource,
                              facts.partialSlots, sourceSlots, split)
                        : std::optional<ScaledSourceLaneGeometry>();
    if (geometry)
      return SourceLanePlan{sourceSlots,
                            split,
                            *lhsSource,
                            *rhsSource,
                            *geometry,
                            std::move(sourceLhsParts),
                            std::move(sourceRhsParts),
                            std::move(sourceLhsOffsets),
                            std::move(sourceRhsOffsets)};
  }
  return std::nullopt;
}

class PlanRISCVPartialTopologiesPass
    : public mlir::PassWrapper<PlanRISCVPartialTopologiesPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-plan-partial-topologies";
  }

  llvm::StringRef getDescription() const override {
    return "Select one typed widened-partial topology from operands, uses, and target resources";
  }

  void runOnOperation() override {
    mlir::Builder builder(&getContext());
    llvm::DenseMap<mlir::Operation *, ScaledTopologyFacts> scaledFacts;
    llvm::DenseMap<mlir::Operation *, riscv::ReduceOp> replicaReductions;
    llvm::DenseSet<mlir::Operation *> levelScaledDots;
    llvm::SmallVector<riscv::ReduceOp> reductions;
    getOperation().walk(
        [&](riscv::ReduceOp reduce) { reductions.push_back(reduce); });
    for (riscv::ReduceOp reduce : reductions) {
      auto input = mlir::dyn_cast<riscv::ValueType>(reduce.getInput().getType());
      auto result = mlir::dyn_cast<riscv::ValueType>(reduce.getResult().getType());
      auto dot = reduce.getInput().getDefiningOp<riscv::RVVWidenDotOp>();
      auto resultElement =
          result ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
                 : mlir::IntegerType();
      if (reduce.getKind() == "add" && input && result && dot &&
          dot.getResult().hasOneUse() && dot.getOver().size() == 1 &&
          reduce.getAxis() >= 0 &&
          static_cast<size_t>(reduce.getAxis()) < input.getAxisIds().size() &&
          input.getLayout().getCarrier() == "scalar" &&
          result.getLayout().getCarrier() == "scalar" && resultElement &&
          resultElement.isSigned() && resultElement.getWidth() == 32) {
        const size_t position = static_cast<size_t>(reduce.getAxis());
        const int64_t reducedAxis = input.getAxisIds()[position];
        const int64_t replicas =
            input.getLayout().getReplicaFactors()[position];
        if (reducedAxis != dot.getOver()[0] && replicas > 1 &&
            input.getLayout().getTimeFactors()[position] == 1 &&
            input.getLayout().getLaneFactors()[position] == 1 &&
            input.getLayout().getFragmentFactors()[position] == 1 &&
            input.getLayout().getLocalFactors()[position] == 1)
          replicaReductions[dot.getOperation()] = reduce;
      }
      auto match = matchReplicaScaledDotReduction(reduce);
      if (!match)
        continue;
      auto dotResultType =
          mlir::dyn_cast<riscv::ValueType>(match->dot.getResult().getType());
      if (!dotResultType)
        continue;
      ScaledTopologyFacts facts;
      facts.partialAxes = match->reducedAxes;
      llvm::sort(facts.partialAxes);
      facts.partialAxes.erase(
          std::unique(facts.partialAxes.begin(), facts.partialAxes.end()),
          facts.partialAxes.end());
      facts.outputAxes = remainingAxes(dotResultType, facts.partialAxes);
      facts.partialSlots =
          replicaProductForAxes(dotResultType, facts.partialAxes);
      facts.outputReplicas =
          replicaProductForAxes(dotResultType, facts.outputAxes);
      if (facts.partialSlots <= 1 || facts.outputReplicas <= 0)
        continue;
      auto found = scaledFacts.find(match->dot.getOperation());
      if (found == scaledFacts.end() ||
          found->second.partialAxes.size() < facts.partialAxes.size())
        scaledFacts[match->dot.getOperation()] = std::move(facts);
    }
    getOperation().walk([&](mlir::scf::ForOp loop) {
      auto contributions = matchLevelScaledLoop(loop);
      if (!contributions)
        return;
      for (ScaledPartialContribution &contribution : *contributions)
        levelScaledDots.insert(contribution.dot.getOperation());
    });

    llvm::SmallVector<riscv::RVVWidenDotOp> dots;
    getOperation().walk(
        [&](riscv::RVVWidenDotOp dot) { dots.push_back(dot); });
    for (riscv::RVVWidenDotOp dot : dots) {
      dot->removeAttr("partial_layout_plan");
      dot->removeAttr("layered_partial_plan");
      auto result = mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
      llvm::SmallVector<int64_t> outputAxes =
          result ? llvm::SmallVector<int64_t>(result.getAxisIds().asArrayRef())
                 : llvm::SmallVector<int64_t>();
      const int64_t outputReplicas = result ? replicaProduct(result) : 1;
      const int64_t partialGroups =
          std::max<int64_t>(1, (2 * dot.getSliceLmulEighths() + 7) / 8);
      const int64_t streams = dot.getReductionStreams();
      const int64_t operandGroups =
          dot.getLhs().getType().getLayout().getRegisterGroups() +
          dot.getRhs().getType().getLayout().getRegisterGroups();
      auto kernel = dot->getParentOfType<riscv::KernelOp>();
      const int64_t available =
          kernel ? kernel.getTarget().getVectorRegisters() - operandGroups : -1;
      const bool fused = dot.getFusedStreamsLegal();

      llvm::StringRef kind;
      int64_t rootOperand = -1;
      llvm::SmallVector<int64_t> partialAxes(dot.getOver());
      int64_t sourceSlots = streams;
      int64_t partialSlots = fused ? 1 : streams;
      int64_t replicas = outputReplicas;
      int64_t laneSplit = 1;
      int64_t combineArity = fused ? streams : 1;
      int64_t resources = std::max<int64_t>(1, partialSlots * partialGroups + 2);
      std::optional<SourceLanePlan> sourcePlan;
      std::optional<LayeredTopologyFacts> layeredPlan;
      riscv::ValueType replicaAccumulator;

      if (auto found = replicaReductions.find(dot.getOperation());
          found != replicaReductions.end()) {
        riscv::ReduceOp reduce = found->second;
        auto input = mlir::cast<riscv::ValueType>(reduce.getInput().getType());
        auto reduced = mlir::cast<riscv::ValueType>(reduce.getResult().getType());
        const size_t position = static_cast<size_t>(reduce.getAxis());
        const int64_t reducedAxis = input.getAxisIds()[position];
        const int64_t reducedReplicas =
            input.getLayout().getReplicaFactors()[position];
        const int64_t survivingReplicas = replicaProduct(reduced);
        auto projectedLhs = projectReplicaReductionOperandType(
            builder, dot.getLhs().getType(), reducedAxis);
        auto projectedRhs = projectReplicaReductionOperandType(
            builder, dot.getRhs().getType(), reducedAxis);
        replicaAccumulator =
            projectedLhs && projectedRhs
                ? replicaReducedAccumulatorType(builder, projectedLhs,
                                                projectedRhs, dot.getOver()[0])
                : riscv::ValueType();
        const int64_t accumulatorGroups =
            replicaAccumulator
                ? replicaAccumulator.getLayout().getRegisterGroups()
                : 0;
        if (reducedReplicas > 1 && survivingReplicas > 0 && kernel &&
            replicaAccumulator && accumulatorGroups <= available &&
            riscv::supportsRVVLayout(kernel.getTarget(),
                                     replicaAccumulator.getLayout())) {
          kind = "replica_reduced";
          partialAxes.assign(1, reducedAxis);
          outputAxes.assign(reduced.getAxisIds().asArrayRef().begin(),
                            reduced.getAxisIds().asArrayRef().end());
          sourceSlots = reducedReplicas;
          partialSlots = reducedReplicas;
          replicas = survivingReplicas;
          combineArity = reducedReplicas;
          resources = std::max<int64_t>(1, operandGroups + accumulatorGroups + 2);
        }
      }
      if (kind.empty() && fused)
        if (auto layered = analyzeLayeredTopology(builder, dot)) {
          kind = "layered";
          rootOperand = layered->rootOperand;
          partialSlots = layered->streams;
          combineArity = layered->streams;
          resources =
              std::max<int64_t>(1, layered->streams * partialGroups + 2);
          layeredPlan = std::move(*layered);
        }
      if (kind.empty())
        if (auto found = scaledFacts.find(dot.getOperation());
            found != scaledFacts.end()) {
          const unsigned operandBits =
              riscv_internal::logicalBitWidth(dot.getLhs().getType().getElementType());
          const bool oneScaleGroupFillsVector =
              kernel && operandBits && dot.getReductionLanes() > 0 &&
              dot.getReductionLanes() * static_cast<int64_t>(operandBits) >=
                  kernel.getTarget().getVlenBits();
          // If one scale group already fills a target vector, reducing before
          // scaling avoids widening a full lane set only to immediately reduce
          // it.  Otherwise the physical partial stays in lanes, is widened and
          // scaled there, and is reduced only after the selected combine tree.
          // Scale dominance across a source Level is a legality fact, not a
          // reason to force either topology.
          const bool reduceBeforeScale = oneScaleGroupFillsVector;
          kind = reduceBeforeScale ? "reduced_scaled" : "scaled";
          partialAxes = found->second.partialAxes;
          outputAxes = found->second.outputAxes;
          partialSlots = found->second.partialSlots;
          replicas = found->second.outputReplicas;
          combineArity = partialSlots;
          resources =
              std::max<int64_t>(1, partialSlots * partialGroups + 2);
          if (auto source = planScaledSourceLanes(builder, dot, found->second)) {
            if (!reduceBeforeScale)
              kind = "scaled";
            sourceSlots = source->sourceSlots;
            laneSplit = source->laneSplit;
            // One wide source partial owns every lane slice produced by its
            // repack.  Keep those slices in one scale-combine chain instead
            // of rebuilding one scalar chain per reduced slot.
            combineArity = sourceSlots;
            sourcePlan = std::move(*source);
          }
        }
      if (kind.empty() && levelScaledDots.contains(dot.getOperation())) {
        kind = "level_scaled";
        partialSlots = dot.getPartialUnroll();
        combineArity = partialSlots % 2 == 0 ? 2 : partialSlots;
        resources =
            std::max<int64_t>(1, partialSlots * partialGroups + 2);
      }
      // Two slots have no intermediate tree level: replacing two independent
      // reductions by two multiplies, one vector add, and one reduction adds a
      // dependency without exposing parallel combine depth.  Existing SG2044
      // measurements confirm the direct per-stream form is faster.  Four or
      // more slots expose at least one real pairwise level and remain eligible.
      if (kind.empty() && streams >= 4 &&
          (streams & (streams - 1)) == 0 &&
          streams * partialGroups <= available) {
        kind = "independent";
        partialSlots = streams;
        combineArity = 2;
        resources = std::max<int64_t>(1, streams * partialGroups + 2);
      }
      if (kind.empty())
        kind = fused ? "sequential_fused" : "sequential_per_stream";

      dot->setAttr("partial_topology",
                   makePartialTopology(builder, dot, kind, rootOperand,
                                       partialAxes, outputAxes, sourceSlots,
                                       partialSlots, replicas, laneSplit,
                                       combineArity,
                                       resources));

      // The topology owner also freezes every intermediate physical value
      // layout used by materialization.  The following pass may instantiate
      // PartialSet containers from these types, but it must not reconstruct
      // lane/time/replica placement from factors a second time.
      riscv::ValueType partialSlot =
          sourcePlan ? sourcePlan->geometry.partialSlot
                     : partialSlotType(builder, dot);
      riscv::ValueType sourceSlot =
          sourcePlan ? sourcePlan->geometry.sourceSlot : riscv::ValueType();
      riscv::ValueType splitSourceSlot =
          sourcePlan ? sourcePlan->geometry.splitSourceSlot
                     : riscv::ValueType();
      riscv::ValueType widenedSlot =
          partialSlot ? widenPartialSlotType(builder, partialSlot)
                      : riscv::ValueType();
      riscv::ValueType reducedSlot =
          sourcePlan
              ? sourcePlan->geometry.reducedSlot
              : (partialSlot && dot.getOver().size() == 1
                     ? reducedPartialSlotType(builder, partialSlot,
                                              dot.getOver()[0])
                     : riscv::ValueType());
      auto vectorResult =
          mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
      if (dot.getResult().hasOneUse()) {
        auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(
            *dot.getResult().getUsers().begin());
        if (conversion && conversion.getConversion().getEffect() == "pure" &&
            conversion.getConversion().getKind() == "register_to_lane")
          vectorResult =
              mlir::dyn_cast<riscv::ValueType>(conversion.getResult().getType());
      }
      riscv::ValueType vectorSlot = kind == "replica_reduced"
                                        ? replicaAccumulator
                                    : vectorResult && dot.getOver().size() == 1
                                        ? vectorPartialSlotType(
                                              builder, dot.getLhs().getType(),
                                              vectorResult, dot.getOver()[0])
                                        : riscv::ValueType();
      if (kind != "replica_reduced" && vectorSlot &&
          vectorPartialSlotType(builder, dot.getRhs().getType(), vectorResult,
                                dot.getOver()[0]) != vectorSlot)
        vectorSlot = {};
      if (!partialSlot) {
        dot.emitError("selected partial topology has no typed partial layout");
        signalPassFailure();
        return;
      }
      riscv::PartialSetType sourceSetType;
      riscv::PartialSetType repackedSetType;
      riscv::PartialSetType partialSetType;
      riscv::PartialSetType scaledSetType;
      riscv::PartialSetType directSetType;
      riscv::PartialSetType reducedSetType;
      riscv::PartialSetType scaleCombinedSetType;
      riscv::PartialSetType fullScaledSetType;
      const int64_t reductionAxis =
          dot.getOver().empty() ? 0 : dot.getOver()[0];
      const bool completeSetGeometry =
          dot.getOver().size() == 1 && reductionAxis > 0 &&
          combineArity > 0 && partialSlots > 0 &&
          partialSlots % combineArity == 0 && !partialSlot.getShape().empty();
      if (completeSetGeometry) {
        const int64_t partialGroups =
            combineArity * partialSlot.getLayout().getRegisterGroups();
        partialSetType = riscv::PartialSetType::get(
            builder.getContext(), partialSlot, reductionAxis, combineArity,
            partialSlot.getShape()[0], partialGroups);
        if (widenedSlot) {
          const int64_t widenedGroups =
              widenedSlot.getLayout().getRegisterGroups();
          scaledSetType = riscv::PartialSetType::get(
              builder.getContext(), widenedSlot, reductionAxis, 1,
              partialSetType.getTermsPerSlot() * combineArity, widenedGroups);
          fullScaledSetType = riscv::PartialSetType::get(
              builder.getContext(), widenedSlot, reductionAxis, 1,
              partialSlot.getShape()[0] * partialSlots, widenedGroups);
        }
        if (kind == "reduced_scaled")
          directSetType = riscv::PartialSetType::get(
              builder.getContext(), partialSlot, reductionAxis, partialSlots,
              partialSlot.getShape()[0],
              partialSlots * partialSlot.getLayout().getRegisterGroups());
      }
      if (sourcePlan && completeSetGeometry) {
        const int64_t chunkPartialSlots = sourceSlots * laneSplit;
        const int64_t sourceSetGroups =
            sourceSlots * sourceSlot.getLayout().getRegisterGroups();
        const int64_t reducedSetGroups =
            chunkPartialSlots * reducedSlot.getLayout().getRegisterGroups();
        sourceSetType = riscv::PartialSetType::get(
            builder.getContext(), sourceSlot, reductionAxis, sourceSlots,
            sourceSlot.getShape()[0], sourceSetGroups);
        repackedSetType = riscv::PartialSetType::get(
            builder.getContext(), partialSlot, reductionAxis, chunkPartialSlots,
            sourceSetType.getTermsPerSlot() / laneSplit, sourceSetGroups);
        reducedSetType = riscv::PartialSetType::get(
            builder.getContext(), reducedSlot, reductionAxis, chunkPartialSlots,
            repackedSetType.getTermsPerSlot(), reducedSetGroups);
        scaleCombinedSetType = riscv::PartialSetType::get(
            builder.getContext(), reducedSlot, reductionAxis, sourceSlots,
            reducedSetType.getTermsPerSlot() * laneSplit,
            sourceSlots * reducedSlot.getLayout().getRegisterGroups());
      } else if (completeSetGeometry && reducedSlot) {
        reducedSetType = riscv::PartialSetType::get(
            builder.getContext(), reducedSlot, reductionAxis, partialSlots,
            partialSlot.getShape()[0],
            partialSlots * reducedSlot.getLayout().getRegisterGroups());
        const int64_t fanout = partialSlots / combineArity;
        scaleCombinedSetType = riscv::PartialSetType::get(
            builder.getContext(), reducedSlot, reductionAxis, combineArity,
            reducedSetType.getTermsPerSlot() * fanout,
            combineArity * reducedSlot.getLayout().getRegisterGroups());
      }
      auto optionalType = [&](riscv::ValueType type) -> mlir::Type {
        return type ? mlir::Type(type)
                    : mlir::Type(mlir::NoneType::get(builder.getContext()));
      };
      auto optionalSetType = [&](riscv::PartialSetType type) -> mlir::Type {
        return type ? mlir::Type(type)
                    : mlir::Type(mlir::NoneType::get(builder.getContext()));
      };
      llvm::ArrayRef<int64_t> empty;
      dot->setAttr(
          "partial_layout_plan",
          riscv::PartialLayoutPlanAttr::get(
              builder.getContext(),
              optionalType(sourcePlan ? sourcePlan->lhsType
                                      : riscv::ValueType()),
              optionalType(sourcePlan ? sourcePlan->rhsType
                                      : riscv::ValueType()),
              optionalType(sourceSlot),
              optionalType(splitSourceSlot), partialSlot,
              optionalType(widenedSlot), optionalType(reducedSlot),
              optionalType(vectorSlot),
              optionalSetType(sourceSetType), optionalSetType(repackedSetType),
              optionalSetType(partialSetType), optionalSetType(scaledSetType),
              optionalSetType(directSetType), optionalSetType(reducedSetType),
              optionalSetType(scaleCombinedSetType),
              optionalSetType(fullScaledSetType),
              builder.getDenseI64ArrayAttr(sourcePlan ? sourcePlan->lhsParts
                                                      : empty),
              builder.getDenseI64ArrayAttr(sourcePlan ? sourcePlan->rhsParts
                                                      : empty),
              builder.getDenseI64ArrayAttr(sourcePlan ? sourcePlan->lhsOffsets
                                                      : empty),
              builder.getDenseI64ArrayAttr(sourcePlan ? sourcePlan->rhsOffsets
                                                      : empty)));

      if (kind == "layered") {
        if (!layeredPlan) {
          dot.emitError("selected layered topology has no typed physical plan");
          signalPassFailure();
          return;
        }
        auto root = llvm::find_if(
            layeredPlan->roots, [&](const ProjectedRoot &candidate) {
              return candidate.value == layeredPlan->layered.value;
            });
        if (root == layeredPlan->roots.end()) {
          dot.emitError("selected layered topology lost its typed root identity");
          signalPassFailure();
          return;
        }
        const int64_t rootIndex = static_cast<int64_t>(
            root - layeredPlan->roots.begin());
        llvm::SmallVector<mlir::Attribute> rootWindowTypes;
        llvm::SmallVector<mlir::Attribute> rootStoragePlans;
        rootWindowTypes.reserve(layeredPlan->roots.size());
        rootStoragePlans.reserve(layeredPlan->roots.size());
        for (const ProjectedRoot &candidate : layeredPlan->roots) {
          auto windowType = layeredPlan->windowTypes.lookup(candidate.value);
          if (!windowType) {
            dot.emitError(
                "selected layered topology has an untyped projected root");
            signalPassFailure();
            return;
          }
          auto storagePlan = riscv_internal::storageWindowPlan(
              builder, candidate.field, layeredPlan->reductionAxis,
              candidate.base, candidate.stride, candidate.repeat,
              candidate.extent, layeredPlan->layered.geometry.getLaneCount());
          if (!storagePlan) {
            dot.emitError(
                "selected layered topology has an unplanned projected storage root");
            signalPassFailure();
            return;
          }
          rootWindowTypes.push_back(mlir::TypeAttr::get(windowType));
          rootStoragePlans.push_back(*storagePlan);
        }
        const int64_t rawGroups =
            layeredPlan->layeredWindowType.getLayout().getRegisterGroups();
        auto storageType = riscv::LayeredWindowType::get(
            builder.getContext(),
            mlir::cast<riscv::ValueType>(
                layeredPlan->layered.field.getResult().getType()),
            layeredPlan->layeredWindowType, layeredPlan->reductionAxis,
            layeredPlan->layers, layeredPlan->windowsPerLayer, rawGroups);
        dot->setAttr(
            "layered_partial_plan",
            riscv::LayeredPartialPlanAttr::get(
                builder.getContext(), rootIndex,
                layeredPlan->layeredWindowType, storageType,
                layeredPlan->accumulatorType,
                builder.getArrayAttr(rootWindowTypes),
                builder.getArrayAttr(rootStoragePlans),
                layeredPlan->layered.geometry));
      }
    }

  }
};

class MaterializeRISCVPartialAccumulatorsPass
    : public mlir::PassWrapper<MaterializeRISCVPartialAccumulatorsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  llvm::StringRef getArgument() const override {
    return "weft-riscv-materialize-partial-accumulators";
  }

  llvm::StringRef getDescription() const override {
    return "Materialize sequential storage windows and loop-carried widened partial accumulators";
  }

  void runOnOperation() override {
    mlir::IRRewriter rewriter(&getContext());
    bool failed = false;
    int64_t nextPartialBirthId = 0;
    getOperation().walk([&](mlir::Operation *operation) {
      if (auto partial = mlir::dyn_cast<riscv::RVVPartialSetOp>(operation))
        nextPartialBirthId =
            std::max(nextPartialBirthId,
                     static_cast<int64_t>(partial.getBirthId()) + 1);
      if (auto capture = mlir::dyn_cast<riscv::RVVPartialCaptureOp>(operation))
        nextPartialBirthId =
            std::max(nextPartialBirthId,
                     static_cast<int64_t>(capture.getBirthId()) + 1);
    });

    llvm::SmallVector<riscv::ExtractOp> extracts;
    getOperation().walk(
        [&](riscv::ExtractOp extract) { extracts.push_back(extract); });
    for (riscv::ExtractOp extract : extracts) {
      auto field = extract.getInput().getDefiningOp<riscv::FieldOp>();
      auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
      if (!field || !result || extract.getAccess().getMapping() !=
                                   "grouped_layered" ||
          extract.getIndices().size() != 1)
        continue;
      auto point = extract.getIndices().front().getDefiningOp<riscv::PhysicalPointOp>();
      if (!point)
        continue;
      if (point.getResult().getType().getDomain().getTail() != "exact")
        continue;
      const int64_t axis = point.getResult().getType().getDomain().getAxisId();
      auto position = axisPosition(result, axis);
      if (!position || result.getLayout().getLaneFactors()[*position] <= 1 ||
          result.getShape()[*position] !=
              result.getLayout().getLaneFactors()[*position] *
                  result.getLayout().getTimeFactors()[*position])
        continue;
      rewriter.setInsertionPoint(extract);
      mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(
          extract.getLoc(), 0);
      llvm::StringRef tail = result.getLayout().getValidity() == "tail"
                                 ? "agnostic"
                                 : "exact";
      auto plan = riscv_internal::storageWindowPlan(
          rewriter, field, axis, 0, 1, 1, result.getShape()[*position],
          result.getLayout().getLaneFactors()[*position]);
      if (!plan) {
        extract.emitError(
            "grouped/layered extract has no closed typed storage-window plan");
        failed = true;
        continue;
      }
      auto window = rewriter.create<riscv::RVVStorageWindowOp>(
          extract.getLoc(), result, field.getResult(), point.getResult(), zero,
          *plan, extract.getAccess(),
          riscv_internal::leaf(rewriter, "rvv", "storage-window",
                               "rvv.storage-window.layered",
                               "rvv.storage-window.layered",
                               mlir::cast<riscv::ValueType>(field.getResult().getType())
                                   .getLayout()
                                   .getRegisterGroups(),
                               result.getLayout().getRegisterGroups(), 0, 0,
                               "none", tail));
      riscv_internal::copyOrigin(extract, window);
      extract.getResult().replaceAllUsesWith(window.getResult());
      rewriter.eraseOp(extract);
    }

    llvm::SmallVector<riscv::BinaryOp> wideningProducts;
    getOperation().walk([&](riscv::BinaryOp binary) {
      if (binary.getKind() == "mul")
        wideningProducts.push_back(binary);
    });
    for (riscv::BinaryOp binary : wideningProducts) {
      auto lhsWiden = binary.getLhs().getDefiningOp<riscv::WidenOp>();
      auto rhsWiden = binary.getRhs().getDefiningOp<riscv::WidenOp>();
      auto lhs = lhsWiden
                     ? mlir::dyn_cast<riscv::ValueType>(
                           lhsWiden.getInput().getType())
                     : riscv::ValueType();
      auto rhs = rhsWiden
                     ? mlir::dyn_cast<riscv::ValueType>(
                           rhsWiden.getInput().getType())
                     : riscv::ValueType();
      auto result =
          mlir::dyn_cast<riscv::ValueType>(binary.getResult().getType());
      auto lhsElement =
          lhs ? mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType())
              : mlir::IntegerType();
      auto rhsElement =
          rhs ? mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType())
              : mlir::IntegerType();
      auto resultElement =
          result ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
                 : mlir::IntegerType();
      auto kernel = binary->getParentOfType<riscv::KernelOp>();
      if (!lhs || !rhs || !result || !lhsElement || !rhsElement ||
          !resultElement || lhsElement.isSignless() || rhsElement.isSignless() ||
          resultElement.isSignless() ||
          lhsElement.getWidth() > lhs.getLayout().getSew() ||
          rhsElement.getWidth() > rhs.getLayout().getSew() ||
          lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
          resultElement.getWidth() != 2 * lhs.getLayout().getSew() ||
          lhs.getShape() != rhs.getShape() || lhs.getShape() != result.getShape() ||
          lhs.getAxisIds() != rhs.getAxisIds() ||
          lhs.getAxisIds() != result.getAxisIds() ||
          lhs.getLayout().getTimeFactors() !=
              rhs.getLayout().getTimeFactors() ||
          lhs.getLayout().getTimeFactors() !=
              result.getLayout().getTimeFactors() ||
          lhs.getLayout().getLaneFactors() !=
              rhs.getLayout().getLaneFactors() ||
          lhs.getLayout().getLaneFactors() !=
              result.getLayout().getLaneFactors() ||
          lhs.getLayout().getReplicaFactors() !=
              rhs.getLayout().getReplicaFactors() ||
          lhs.getLayout().getReplicaFactors() !=
              result.getLayout().getReplicaFactors() ||
          lhs.getLayout().getFragmentFactors() !=
              rhs.getLayout().getFragmentFactors() ||
          lhs.getLayout().getFragmentFactors() !=
              result.getLayout().getFragmentFactors() ||
          lhs.getLayout().getLocalFactors() !=
              rhs.getLayout().getLocalFactors() ||
          lhs.getLayout().getLocalFactors() !=
              result.getLayout().getLocalFactors() ||
          lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
          result.getLayout().getSew() != resultElement.getWidth() ||
          lhs.getLayout().getLmulEighths() !=
              rhs.getLayout().getLmulEighths() ||
          result.getLayout().getLmulEighths() !=
              2 * lhs.getLayout().getLmulEighths() ||
          lhs.getLayout().getVl() != rhs.getLayout().getVl() ||
          lhs.getLayout().getVl() != result.getLayout().getVl() || !kernel ||
          !kernel.getTarget().getHasWideningInteger() ||
          !llvm::is_contained(
              kernel.getTarget().getLegalLMULEighths().asArrayRef(),
              result.getLayout().getLmulEighths()))
        continue;
      llvm::StringRef instruction;
      if (lhsElement.isUnsigned() && rhsElement.isUnsigned())
        instruction = "rvv.vwmulu.vv";
      else if (lhsElement.isSigned() && rhsElement.isSigned())
        instruction = "rvv.vwmul.vv";
      else if (lhsElement.isSigned())
        instruction = "rvv.vwmulsu.vv";
      else
        instruction = "rvv.vwmulsu.vv.swap";
      rewriter.setInsertionPoint(binary);
      auto fused = rewriter.create<riscv::RVVWidenMultiplyOp>(
          binary.getLoc(), result, lhsWiden.getInput(), rhsWiden.getInput(),
          riscv_internal::leaf(
              rewriter, "rvv", "widen-multiply", instruction, instruction,
              lhs.getLayout().getRegisterGroups() +
                  rhs.getLayout().getRegisterGroups(),
              result.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
      riscv_internal::copyOrigin(binary, fused);
      binary.getResult().replaceAllUsesWith(fused.getResult());
      rewriter.eraseOp(binary);
      if (lhsWiden.getResult().use_empty())
        rewriter.eraseOp(lhsWiden);
      if (rhsWiden && rhsWiden != lhsWiden && rhsWiden.getResult().use_empty())
        rewriter.eraseOp(rhsWiden);
    }

    llvm::SmallVector<riscv::BinaryOp> scalarWideningProducts;
    getOperation().walk([&](riscv::BinaryOp binary) {
      if (binary.getKind() == "mul")
        scalarWideningProducts.push_back(binary);
    });
    for (riscv::BinaryOp binary : scalarWideningProducts) {
      riscv::WidenOp widen =
          binary.getLhs().getDefiningOp<riscv::WidenOp>();
      mlir::Value scalar = binary.getRhs();
      if (!widen) {
        widen = binary.getRhs().getDefiningOp<riscv::WidenOp>();
        scalar = binary.getLhs();
      }
      auto input = widen ? mlir::dyn_cast<riscv::ValueType>(
                               widen.getInput().getType())
                         : riscv::ValueType();
      auto widened = widen ? mlir::dyn_cast<riscv::ValueType>(
                                 widen.getResult().getType())
                           : riscv::ValueType();
      auto result =
          mlir::dyn_cast<riscv::ValueType>(binary.getResult().getType());
      auto constant = scalar.getDefiningOp<riscv::ConstantOp>();
      auto integer = constant
                         ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                         : mlir::IntegerAttr();
      auto inputElement =
          input ? mlir::dyn_cast<mlir::IntegerType>(input.getElementType())
                : mlir::IntegerType();
      auto scalarElement = mlir::dyn_cast<mlir::IntegerType>(scalar.getType());
      auto resultElement =
          result ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
                 : mlir::IntegerType();
      bool scalarFits = false;
      if (integer && inputElement)
        scalarFits = inputElement.isSigned()
                         ? integer.getValue().isSignedIntN(inputElement.getWidth())
                         : integer.getValue().isIntN(inputElement.getWidth());
      if (!input || !widened || !result || !constant || !integer ||
          !inputElement || !scalarElement || !resultElement || !scalarFits ||
          inputElement.isSignless() || scalarElement.isSignless() ||
          resultElement.isSignless() ||
          scalarElement != widened.getElementType() ||
          resultElement != widened.getElementType() ||
          result.getShape() != input.getShape() ||
          result.getAxisIds() != input.getAxisIds() ||
          result.getLayout().getTimeFactors() !=
              input.getLayout().getTimeFactors() ||
          result.getLayout().getLaneFactors() !=
              input.getLayout().getLaneFactors() ||
          result.getLayout().getReplicaFactors() !=
              input.getLayout().getReplicaFactors() ||
          result.getLayout().getFragmentFactors() !=
              input.getLayout().getFragmentFactors() ||
          result.getLayout().getLocalFactors() !=
              input.getLayout().getLocalFactors() ||
          resultElement.getWidth() != 2 * inputElement.getWidth() ||
          (inputElement.isUnsigned() && scalarElement.isSigned()))
        continue;
      llvm::StringRef instruction;
      if (inputElement.isUnsigned())
        instruction = "rvv.vwmulu.vx";
      else if (scalarElement.isSigned())
        instruction = "rvv.vwmul.vx";
      else
        instruction = "rvv.vwmulsu.vx";
      rewriter.setInsertionPoint(binary);
      auto narrowConstant = rewriter.create<riscv::ConstantOp>(
          binary.getLoc(), inputElement,
          rewriter.getIntegerAttr(inputElement, integer.getInt()));
      auto fused = rewriter.create<riscv::RVVWidenScalarMultiplyOp>(
          binary.getLoc(), result, widen.getInput(), narrowConstant.getResult(),
          riscv_internal::leaf(
              rewriter, "rvv", "widen-scalar-multiply", instruction,
              instruction, input.getLayout().getRegisterGroups(),
              result.getLayout().getRegisterGroups(), 0, 0, "none", "exact"));
      riscv_internal::copyOrigin(binary, fused);
      binary.getResult().replaceAllUsesWith(fused.getResult());
      rewriter.eraseOp(binary);
      if (widen.getResult().use_empty())
        rewriter.eraseOp(widen);
      if (constant.getResult().use_empty())
        rewriter.eraseOp(constant);
    }

    llvm::SmallVector<riscv::RVVWidenAccumulateOp> zeroSeededAccumulates;
    getOperation().walk([&](riscv::RVVWidenAccumulateOp accumulate) {
      zeroSeededAccumulates.push_back(accumulate);
    });
    for (riscv::RVVWidenAccumulateOp accumulate : zeroSeededAccumulates) {
      auto splat =
          accumulate.getAccumulator().getDefiningOp<riscv::RVVSplatOp>();
      auto constant = splat
                          ? splat.getScalar().getDefiningOp<riscv::ConstantOp>()
                          : riscv::ConstantOp();
      auto integer = constant
                         ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                         : mlir::IntegerAttr();
      if (!splat || !constant || !integer || integer.getInt() != 0 ||
          !splat.getResult().hasOneUse())
        continue;
      auto lhs = accumulate.getLhs().getType();
      auto rhs = accumulate.getRhs().getType();
      auto result = accumulate.getResult().getType();
      auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
      auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
      auto resultElement =
          mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
      auto kernel = accumulate->getParentOfType<riscv::KernelOp>();
      if (!lhsElement || !rhsElement || lhsElement.isSignless() ||
          rhsElement.isSignless() || !resultElement ||
          resultElement.isSignless() ||
          (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
          lhs.getShape() != rhs.getShape() || lhs.getShape() != result.getShape() ||
          lhs.getAxisIds() != rhs.getAxisIds() ||
          lhs.getAxisIds() != result.getAxisIds() ||
          lhs.getLayout().getTimeFactors() !=
              rhs.getLayout().getTimeFactors() ||
          lhs.getLayout().getTimeFactors() !=
              result.getLayout().getTimeFactors() ||
          lhs.getLayout().getLaneFactors() !=
              rhs.getLayout().getLaneFactors() ||
          lhs.getLayout().getLaneFactors() !=
              result.getLayout().getLaneFactors() ||
          lhs.getLayout().getReplicaFactors() !=
              rhs.getLayout().getReplicaFactors() ||
          lhs.getLayout().getReplicaFactors() !=
              result.getLayout().getReplicaFactors() ||
          lhs.getLayout().getFragmentFactors() !=
              rhs.getLayout().getFragmentFactors() ||
          lhs.getLayout().getFragmentFactors() !=
              result.getLayout().getFragmentFactors() ||
          lhs.getLayout().getLocalFactors() !=
              rhs.getLayout().getLocalFactors() ||
          lhs.getLayout().getLocalFactors() !=
              result.getLayout().getLocalFactors() ||
          lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
          resultElement.getWidth() != 2 * lhs.getLayout().getSew() ||
          result.getLayout().getSew() != resultElement.getWidth() ||
          lhs.getLayout().getLmulEighths() !=
              rhs.getLayout().getLmulEighths() ||
          result.getLayout().getLmulEighths() !=
              2 * lhs.getLayout().getLmulEighths() ||
          lhs.getLayout().getVl() != rhs.getLayout().getVl() ||
          lhs.getLayout().getVl() != result.getLayout().getVl() || !kernel ||
          !kernel.getTarget().getHasWideningInteger() ||
          !llvm::is_contained(
              kernel.getTarget().getLegalLMULEighths().asArrayRef(),
              result.getLayout().getLmulEighths()))
        continue;
      llvm::StringRef instruction;
      if (lhsElement.isSigned() && rhsElement.isSigned())
        instruction = "rvv.vwmul.vv";
      else if (lhsElement.isSigned())
        instruction = "rvv.vwmulsu.vv";
      else
        instruction = "rvv.vwmulsu.vv.swap";
      rewriter.setInsertionPoint(accumulate);
      auto product = rewriter.create<riscv::RVVWidenMultiplyOp>(
          accumulate.getLoc(), accumulate.getResult().getType(),
          accumulate.getLhs(), accumulate.getRhs(),
          riscv_internal::leaf(
              rewriter, "rvv", "widen-multiply", instruction, instruction,
              lhs.getLayout().getRegisterGroups() +
                  rhs.getLayout().getRegisterGroups(),
              accumulate.getResult().getType().getLayout().getRegisterGroups(),
              0, 0, "none", "exact"));
      riscv_internal::copyOrigin(accumulate, product);
      accumulate.getResult().replaceAllUsesWith(product.getResult());
      rewriter.eraseOp(accumulate);
      if (splat.getResult().use_empty())
        rewriter.eraseOp(splat);
      if (constant.getResult().use_empty())
        rewriter.eraseOp(constant);
    }

    llvm::SmallVector<riscv::ReduceOp> replicaPartialReductions;
    llvm::SmallVector<mlir::Value> replicaCleanupRoots;
    getOperation().walk([&](riscv::ReduceOp reduce) {
      if (mlir::isa<mlir::IntegerType>(
              riscv_internal::logicalElement(reduce.getResult().getType())))
        replicaPartialReductions.push_back(reduce);
    });
    for (riscv::ReduceOp reduce : replicaPartialReductions) {
      auto match = matchReplicaScaledDotReduction(reduce);
      if (!match)
        continue;
      riscv::RVVWidenDotOp dot = match->dot;
      const bool reduceBeforeScale = hasTopology(dot, "reduced_scaled");
      if (!hasTopology(dot, "scaled") && !reduceBeforeScale)
        continue;
      llvm::SmallVector<int64_t> matchedAxes(match->reducedAxes);
      llvm::sort(matchedAxes);
      matchedAxes.erase(std::unique(matchedAxes.begin(), matchedAxes.end()),
                        matchedAxes.end());
      if (matchedAxes != llvm::SmallVector<int64_t>(
                             dot.getPartialTopology().getPartialAxes().asArrayRef()))
        continue;
      riscv::ValueType lhs = dot.getLhs().getType();
      riscv::ValueType rhs = dot.getRhs().getType();
      auto dotResult = mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
      auto scaledType =
          mlir::dyn_cast<riscv::ValueType>(match->scaleMultiply.getResult().getType());
      auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(reduce.getResult().getType()));
      const int64_t reductionAxis = dot.getOver()[0];
      auto lhsReduction = axisPosition(lhs, reductionAxis);
      auto rhsReduction = axisPosition(rhs, reductionAxis);
      const int64_t slots = dot.getPartialTopology().getPartialSlots();
      const int64_t outputParts =
          dot.getPartialTopology().getOutputReplicas();
      auto finalValue =
          mlir::dyn_cast<riscv::ValueType>(reduce.getResult().getType());
      llvm::SmallVector<int64_t> partialAxes(
          dot.getPartialTopology().getPartialAxes().asArrayRef());
      llvm::SmallVector<int64_t> outputAxes(
          dot.getPartialTopology().getOutputAxes().asArrayRef());
      auto layoutPlan = dot.getPartialLayoutPlanAttr();
      auto slotType =
          layoutPlan ? mlir::dyn_cast<riscv::ValueType>(
                           layoutPlan.getPartialSlotType())
                     : riscv::ValueType();
      bool complete = layoutPlan && dotResult && scaledType && resultElement &&
                      resultElement.isSigned() && resultElement.getWidth() == 32 &&
                      lhsReduction && rhsReduction && slotType && slots >= 2 &&
                      ((slots & (slots - 1)) == 0) && outputParts > 0 &&
                      dotResult.getLayout().getCarrier() == "scalar" &&
                      dotResult.getShape() == scaledType.getShape() &&
                      dotResult.getAxisIds() == scaledType.getAxisIds() &&
                      llvm::all_of(dotResult.getAxisIds().asArrayRef(),
                                   [&](int64_t axis) {
                                     return llvm::is_contained(partialAxes, axis) ||
                                            llvm::is_contained(outputAxes, axis);
                                   }) &&
                      replicaProductForAxes(dotResult, partialAxes) == slots &&
                      replicaProductForAxes(dotResult, outputAxes) == outputParts &&
                      ((!finalValue && outputAxes.empty() && outputParts == 1) ||
                       (finalValue &&
                        finalValue.getAxisIds().asArrayRef() ==
                            llvm::ArrayRef<int64_t>(outputAxes))) &&
                      lhs.getLayout().getTimeFactors()[*lhsReduction] == 1 &&
                      rhs.getLayout().getTimeFactors()[*rhsReduction] == 1;
      if (complete) {
        for (size_t position = 0; position < dotResult.getShape().size(); ++position)
          complete &= dotResult.getLayout().getTimeFactors()[position] == 1 &&
                      dotResult.getLayout().getLaneFactors()[position] == 1 &&
                      dotResult.getLayout().getReplicaFactors()[position] > 0;
      }
      auto multiplyInstruction =
          complete ? partialMultiplyInstruction(lhs, rhs) : std::nullopt;
      if (!complete || !multiplyInstruction)
        continue;

      rewriter.setInsertionPoint(reduce);
      llvm::DenseMap<mlir::Value, mlir::Value> rematerialized;
      auto rematerializedScale =
          rematerializeScalarReplicas(match->scale, rewriter, rematerialized);
      mlir::Value scalarScale;
      if (mlir::succeeded(rematerializedScale)) {
        scalarScale = *rematerializedScale;
      } else if (auto source =
                     mlir::dyn_cast<riscv::ValueType>(match->scale.getType())) {
        auto target = scalarReplicaType(rewriter, source);
        if (target) {
          auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
              reduce.getLoc(), target, match->scale,
              riscv_internal::layoutConversion(rewriter, source.getLayout(),
                                               target.getLayout()),
              riscv::AccessAttr(),
              riscv_internal::unselectedLeaf(rewriter));
          if (mlir::Operation *definition = match->scale.getDefiningOp())
            riscv_internal::copyOrigin(definition, conversion);
          scalarScale = conversion.getResult();
        }
      }
      auto scalarScaleType =
          scalarScale
              ? mlir::dyn_cast<riscv::ValueType>(scalarScale.getType())
              : riscv::ValueType();
      auto scaleParts =
          scalarScaleType
              ? riscv_internal::staticProduct(
                    scalarScaleType.getLayout()
                        .getReplicaFactors()
                        .asArrayRef())
              : std::optional<int64_t>();
      auto scaleElement =
          scalarScaleType
              ? mlir::dyn_cast<mlir::IntegerType>(
                    scalarScaleType.getElementType())
              : mlir::IntegerType();
      if (!scalarScaleType ||
          scalarScaleType.getLayout().getCarrier() != "scalar" ||
          !scaleParts || *scaleParts < slots || !scaleElement ||
          !scaleElement.isSigned() || scaleElement.getWidth() != 32)
        continue;

      auto kernel = dot->getParentOfType<riscv::KernelOp>();
      auto widenedSlot = mlir::dyn_cast<riscv::ValueType>(
          layoutPlan.getWidenedSlotType());
      auto narrowScale = materializeExactNarrowScale(scalarScale, rewriter);
      auto narrowScaleType =
          narrowScale
              ? mlir::dyn_cast<riscv::ValueType>((*narrowScale).getType())
              : riscv::ValueType();
      auto narrowScaleParts =
          narrowScaleType
              ? riscv_internal::staticProduct(
                    narrowScaleType.getLayout().getReplicaFactors().asArrayRef())
              : std::optional<int64_t>();
      const int64_t combineArity =
          dot.getPartialTopology().getCombineArity();
      const int64_t laneSplit = dot.getPartialTopology().getLaneSplit();
      const int64_t sourceSlots = dot.getPartialTopology().getSourceSlots();
      auto plannedSourceSet = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getSourceSetType());
      auto plannedRepackedSet = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getRepackedSetType());
      auto plannedReducedSet = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getReducedSetType());
      auto plannedScaleCombinedSet = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getScaleCombinedSetType());
      std::optional<ScaledSourceLaneGeometry> sourceGeometry;
      if (laneSplit > 1) {
        auto sourceSlot =
            mlir::dyn_cast<riscv::ValueType>(layoutPlan.getSourceSlotType());
        auto splitSourceSlot = mlir::dyn_cast<riscv::ValueType>(
            layoutPlan.getSplitSourceSlotType());
        auto partialSlot =
            mlir::dyn_cast<riscv::ValueType>(layoutPlan.getPartialSlotType());
        auto reducedSlot =
            mlir::dyn_cast<riscv::ValueType>(layoutPlan.getReducedSlotType());
        if (sourceSlot && splitSourceSlot && partialSlot && reducedSlot &&
            plannedSourceSet && plannedRepackedSet && plannedReducedSet &&
            plannedScaleCombinedSet)
          sourceGeometry = ScaledSourceLaneGeometry{
              sourceSlot, splitSourceSlot, partialSlot, reducedSlot,
              plannedSourceSet.getResourceGroups(),
              plannedReducedSet.getResourceGroups()};
      }
      if (sourceGeometry) {
        slotType = sourceGeometry->partialSlot;
      }
      const int64_t scaleChunkCount =
          combineArity > 0 && slots % combineArity == 0
              ? slots / combineArity
              : 0;
      const int64_t scaleChains = combineArity;
      const int64_t scaleFanout =
          scaleChains > 0 ? slots / scaleChains : 0;
      auto setType = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getPartialSetType());
      auto scaledSetType = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getScaledSetType());
      auto directSetType = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getDirectSetType());
      auto reducedSetType = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getReducedSetType());
      auto scaleCombinedSetType = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getScaleCombinedSetType());
      auto fullScaledSetType = mlir::dyn_cast<riscv::PartialSetType>(
          layoutPlan.getFullScaledSetType());
      const int64_t setGroups =
          setType ? setType.getResourceGroups() : 0;
      const int64_t scaledGroups =
          scaledSetType ? scaledSetType.getResourceGroups() : 0;
      const int64_t narrowScaleGroups =
          narrowScaleType ? narrowScaleType.getLayout().getRegisterGroups() : 0;
      auto sourceSlot = sourceGeometry ? sourceGeometry->sourceSlot
                                       : riscv::ValueType();
      auto sourceSetType = sourceGeometry ? plannedSourceSet
                                          : riscv::PartialSetType();
      auto repackedSetType = sourceGeometry ? plannedRepackedSet
                                            : riscv::PartialSetType();
      auto reducedSlot = mlir::dyn_cast<riscv::ValueType>(
          layoutPlan.getReducedSlotType());
      const bool wideSourceLegal =
          laneSplit > 1
              ? (sourceGeometry && sourceSetType && repackedSetType &&
                 reducedSlot && reducedSetType && scaleCombinedSetType &&
                 sourceSetType.getTermsPerSlot() % laneSplit == 0)
              : !reduceBeforeScale ||
                    (directSetType && reducedSetType && scaleCombinedSetType);
      const bool narrowScaleLegal =
          reduceBeforeScale || laneSplit > 1 ||
          (widenedSlot && narrowScale && narrowScaleType && narrowScaleParts &&
           *narrowScaleParts >= slots && scaleChunkCount > 0 && setGroups > 0 &&
           scaledGroups > 0 && scaledSetType &&
           fullScaledSetType &&
           setGroups + scaleChunkCount * scaledGroups +
                   narrowScaleGroups <=
               kernel.getTarget().getVectorRegisters());
      if (!kernel || !wideSourceLegal || !narrowScaleLegal)
        continue;

      rewriter.setInsertionPoint(reduce);
      mlir::Value sourceLhs = dot.getLhs();
      mlir::Value sourceRhs = dot.getRhs();
      if (laneSplit > 1) {
        auto plannedLhs = mlir::dyn_cast<riscv::ValueType>(
            layoutPlan.getSourceLhsType());
        auto plannedRhs = mlir::dyn_cast<riscv::ValueType>(
            layoutPlan.getSourceRhsType());
        if (!plannedLhs || !plannedRhs)
          continue;
        auto materializeOperandLayout = [&](mlir::Value input,
                                            riscv::ValueType planned)
            -> mlir::Value {
          if (input.getType() == planned)
            return input;
          auto source = mlir::cast<riscv::ValueType>(input.getType());
          auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
              reduce.getLoc(), planned, input,
              riscv_internal::layoutConversion(rewriter, source.getLayout(),
                                               planned.getLayout()),
              riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
          if (mlir::Operation *definition = input.getDefiningOp())
            riscv_internal::copyOrigin(definition, conversion);
          return conversion.getResult();
        };
        sourceLhs = materializeOperandLayout(sourceLhs, plannedLhs);
        sourceRhs = materializeOperandLayout(sourceRhs, plannedRhs);
      }
      llvm::SmallVector<mlir::Value> scalarParts;
      bool outputComplete = true;
      for (int64_t outputPart = 0; outputPart < outputParts; ++outputPart) {
        llvm::SmallVector<int64_t> lhsParts;
        llvm::SmallVector<int64_t> rhsParts;
        llvm::SmallVector<int64_t> lhsLaneOffsets;
        llvm::SmallVector<int64_t> rhsLaneOffsets;
        llvm::SmallVector<int64_t> scaleReplicas;
        for (int64_t slot = 0; slot < slots; ++slot) {
          auto resultPart = composeReplicaPart(dotResult, outputAxes, outputPart,
                                               partialAxes, slot);
          const int64_t planned =
              resultPart ? *resultPart * dot.getReductionStreams() : -1;
          auto lhsPart =
              planned >= 0 &&
                      planned < static_cast<int64_t>(dot.getLhsParts().size())
                  ? std::optional<int64_t>(dot.getLhsParts()[planned])
                  : std::nullopt;
          auto rhsPart =
              planned >= 0 &&
                      planned < static_cast<int64_t>(dot.getRhsParts().size())
                  ? std::optional<int64_t>(dot.getRhsParts()[planned])
                  : std::nullopt;
          auto scalePart = resultPart
                               ? projectReplica(scalarScaleType, dotResult,
                                                *resultPart)
                               : std::optional<int64_t>();
          auto lhsLaneOffset =
              resultPart &&
                      *resultPart <
                          static_cast<int64_t>(dot.getLhsLaneOffsets().size())
                  ? std::optional<int64_t>(
                        dot.getLhsLaneOffsets()[*resultPart])
                  : std::nullopt;
          auto rhsLaneOffset =
              resultPart &&
                      *resultPart <
                          static_cast<int64_t>(dot.getRhsLaneOffsets().size())
                  ? std::optional<int64_t>(
                        dot.getRhsLaneOffsets()[*resultPart])
                  : std::nullopt;
          if (!lhsPart || !rhsPart || !scalePart || !lhsLaneOffset ||
              !rhsLaneOffset) {
            outputComplete = false;
            break;
          }
          lhsParts.push_back(*lhsPart);
          rhsParts.push_back(*rhsPart);
          lhsLaneOffsets.push_back(*lhsLaneOffset);
          rhsLaneOffsets.push_back(*rhsLaneOffset);
          scaleReplicas.push_back(*scalePart);
        }
        if (!outputComplete)
          break;

        auto combineReducedPartials = [&](mlir::Value reducedPartials) {
          llvm::SmallVector<mlir::Value> scales(slots, scalarScale);
          auto combined = rewriter.create<riscv::RVVPartialScaleCombineOp>(
              reduce.getLoc(), scaleCombinedSetType, reducedPartials, scales,
              rewriter.getDenseI64ArrayAttr(scaleReplicas),
              dot.getPartialTopology().getSlotOrder(),
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-scale-combine",
                  "rvv.partial-scale-combine", "rvv.partial-scale-combine",
                  reducedSetType.getResourceGroups(),
                  scaleCombinedSetType.getResourceGroups(), 1, 0, "none",
                  "exact",
                  {reductionAxis, slots, scaleChains, scaleFanout}));
          riscv_internal::copyOrigin(dot, combined);
          return combined.getResult();
        };

        mlir::Value partials;
        if (laneSplit > 1) {
          if (!layoutPlan) {
            outputComplete = false;
            break;
          }
          const int64_t begin = outputPart * sourceSlots;
          auto sourceLhsParts = layoutPlan.getSourceLhsParts()
                                    .asArrayRef()
                                    .slice(begin, sourceSlots);
          auto sourceRhsParts = layoutPlan.getSourceRhsParts()
                                    .asArrayRef()
                                    .slice(begin, sourceSlots);
          auto sourceLhsOffsets = layoutPlan.getSourceLhsOffsets()
                                      .asArrayRef()
                                      .slice(begin, sourceSlots);
          auto sourceRhsOffsets = layoutPlan.getSourceRhsOffsets()
                                      .asArrayRef()
                                      .slice(begin, sourceSlots);
          llvm::SmallVector<int64_t> sourceOperands(sourceSlots, 0);
          auto sourceSet = rewriter.create<riscv::RVVPartialSetOp>(
              reduce.getLoc(), sourceSetType,
              mlir::ValueRange{sourceLhs}, mlir::ValueRange{sourceRhs},
              rewriter.getDenseI64ArrayAttr(sourceOperands),
              rewriter.getDenseI64ArrayAttr(sourceOperands),
              rewriter.getDenseI64ArrayAttr(sourceLhsParts),
              rewriter.getDenseI64ArrayAttr(sourceRhsParts),
              rewriter.getDenseI64ArrayAttr(sourceLhsOffsets),
              rewriter.getDenseI64ArrayAttr(sourceRhsOffsets), reductionAxis,
              *multiplyInstruction,
              ownerDomain(reduce), nextPartialBirthId++, ownerDomain(reduce),
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-set", "rvv.partial-set",
                  "rvv.partial-set", 0, sourceSetType.getResourceGroups(), 0,
                  0, "none", "exact", {reductionAxis, sourceSlots}));
          riscv_internal::copyOrigin(dot, sourceSet);
          sinkReplicaSupplies(sourceSet);
          auto repacked = rewriter.create<riscv::RVVPartialRepackOp>(
              reduce.getLoc(), repackedSetType, sourceSet.getResult(), laneSplit,
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-repack",
                  "rvv.partial-repack.split", "rvv.partial-repack.split",
                  sourceSetType.getResourceGroups(),
                  repackedSetType.getResourceGroups(), 0, 0, "none", "exact",
                  {reductionAxis, laneSplit}));
          riscv_internal::copyOrigin(dot, repacked);
          auto reduced = rewriter.create<riscv::RVVPartialReduceOp>(
              reduce.getLoc(), reducedSetType, repacked.getResult(),
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-reduce",
                  "rvv.partial-reduce.widen", "rvv.partial-reduce.widen",
                  repackedSetType.getResourceGroups(),
                  reducedSetType.getResourceGroups(), 1, 0, "none", "exact",
                  {reductionAxis, slots}));
          riscv_internal::copyOrigin(dot, reduced);
          partials = combineReducedPartials(reduced.getResult());
        } else if (reduceBeforeScale) {
          llvm::SmallVector<int64_t> operandSlots(slots, 0);
          auto set = rewriter.create<riscv::RVVPartialSetOp>(
              reduce.getLoc(), directSetType, mlir::ValueRange{dot.getLhs()},
              mlir::ValueRange{dot.getRhs()},
              rewriter.getDenseI64ArrayAttr(operandSlots),
              rewriter.getDenseI64ArrayAttr(operandSlots),
              rewriter.getDenseI64ArrayAttr(lhsParts),
              rewriter.getDenseI64ArrayAttr(rhsParts),
              rewriter.getDenseI64ArrayAttr(lhsLaneOffsets),
              rewriter.getDenseI64ArrayAttr(rhsLaneOffsets), reductionAxis,
              *multiplyInstruction,
              ownerDomain(reduce), nextPartialBirthId++, ownerDomain(reduce),
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-set", "rvv.partial-set",
                  "rvv.partial-set", 0, directSetType.getResourceGroups(), 0,
                  0, "none", "exact", {reductionAxis, slots}));
          riscv_internal::copyOrigin(dot, set);
          sinkReplicaSupplies(set);
          auto reduced = rewriter.create<riscv::RVVPartialReduceOp>(
              reduce.getLoc(), reducedSetType, set.getResult(),
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-reduce",
                  "rvv.partial-reduce.widen", "rvv.partial-reduce.widen",
                  directSetType.getResourceGroups(),
                  reducedSetType.getResourceGroups(), 1, 0, "none", "exact",
                  {reductionAxis, slots}));
          riscv_internal::copyOrigin(dot, reduced);
          partials = combineReducedPartials(reduced.getResult());
        } else {
          llvm::SmallVector<mlir::Value> chunkPartials;
          for (int64_t first = 0; first < slots; first += combineArity) {
            llvm::SmallVector<int64_t> operandSlots(combineArity, 0);
            auto slice = [&](llvm::ArrayRef<int64_t> values) {
              return rewriter.getDenseI64ArrayAttr(
                  values.slice(first, combineArity));
            };
            auto set = rewriter.create<riscv::RVVPartialSetOp>(
                reduce.getLoc(), setType, mlir::ValueRange{dot.getLhs()},
                mlir::ValueRange{dot.getRhs()},
                rewriter.getDenseI64ArrayAttr(operandSlots),
                rewriter.getDenseI64ArrayAttr(operandSlots), slice(lhsParts),
                slice(rhsParts), slice(lhsLaneOffsets), slice(rhsLaneOffsets),
                reductionAxis, *multiplyInstruction,
                ownerDomain(reduce), nextPartialBirthId++, ownerDomain(reduce),
                riscv_internal::leaf(
                    rewriter, "rvv", "partial-set", "rvv.partial-set",
                    "rvv.partial-set", 0, setGroups, 0, 0, "none", "exact",
                    {reductionAxis, combineArity}));
            riscv_internal::copyOrigin(dot, set);
            sinkReplicaSupplies(set);
            llvm::SmallVector<mlir::Value> scales(combineArity, *narrowScale);
            auto scaled = rewriter.create<riscv::RVVPartialWidenScaleOp>(
                reduce.getLoc(), scaledSetType, set.getResult(), scales,
                slice(scaleReplicas), combineArity,
                riscv_internal::leaf(
                    rewriter, "rvv", "partial-widen-scale",
                    "rvv.partial-widen-scale", "rvv.partial-widen-scale",
                    setGroups + narrowScaleGroups, scaledGroups, 0, 0, "none",
                    "exact", {reductionAxis, combineArity, combineArity}));
            riscv_internal::copyOrigin(dot, scaled);
            chunkPartials.push_back(scaled.getResult());
          }

          partials = chunkPartials.front();
          if (chunkPartials.size() > 1) {
            auto merged = rewriter.create<riscv::RVVPartialMergeOp>(
                reduce.getLoc(), fullScaledSetType, chunkPartials, "pairwise",
                riscv_internal::leaf(
                    rewriter, "rvv", "partial-merge", "rvv.partial-merge",
                    "rvv.partial-merge", scaleChunkCount * scaledGroups,
                    scaledGroups, 0, 0, "none", "exact",
                    {reductionAxis, scaleChunkCount, scaleChunkCount,
                     slots * setType.getTermsPerSlot()}));
            riscv_internal::copyOrigin(dot, merged);
            partials = merged.getResult();
          }
        }
        auto finalType = mlir::cast<riscv::PartialSetType>(partials.getType());
        auto instruction = partialFinalizeInstruction(finalType, reductionAxis);
        if (!instruction) {
          outputComplete = false;
          break;
        }
        auto finalized = rewriter.create<riscv::RVVPartialFinalizeOp>(
            reduce.getLoc(), resultElement, partials, reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-finalize", *instruction,
                *instruction, finalType.getResourceGroups(), 0, 0, 0, "none",
                "exact", {reductionAxis, 1}));
        riscv_internal::copyOrigin(dot, finalized);
        scalarParts.push_back(finalized.getResult());
      }
      if (!outputComplete ||
          scalarParts.size() != static_cast<size_t>(outputParts))
        continue;
      mlir::Value replacement;
      if (finalValue) {
        auto assembled = rewriter.create<riscv::RVVAssembleReplicasOp>(
            reduce.getLoc(), finalValue, scalarParts,
            riscv_internal::leaf(rewriter, "scalar", "assemble-replicas",
                                 "scalar.assemble-replicas",
                                 "scalar.assemble-replicas", 0, 0));
        riscv_internal::copyOrigin(dot, assembled);
        replacement = assembled.getResult();
      } else {
        replacement = scalarParts.front();
      }
      reduce.getResult().replaceAllUsesWith(replacement);

      mlir::Value oldScale = match->scale;
      mlir::Value oldDotSide = match->dotSide;
      llvm::SmallVector<mlir::Operation *> replacedChain;
      llvm::DenseSet<mlir::Operation *> seenChain;
      auto rememberChainOp = [&](mlir::Operation *operation) {
        if (operation && seenChain.insert(operation).second)
          replacedChain.push_back(operation);
      };
      for (riscv::ReduceOp reduction : match->reductions)
        rememberChainOp(reduction);
      for (riscv::ConvertLayoutOp conversion : match->reductionConversions)
        rememberChainOp(conversion);
      rememberChainOp(match->scaleMultiply);
      bool erasedAny = false;
      do {
        erasedAny = false;
        for (mlir::Operation *&operation : replacedChain) {
          if (!operation ||
              !llvm::all_of(operation->getResults(), [](mlir::Value result) {
                return result.use_empty();
              }))
            continue;
          rewriter.eraseOp(operation);
          operation = nullptr;
          erasedAny = true;
        }
      } while (erasedAny);
      if (llvm::any_of(replacedChain,
                       [](mlir::Operation *operation) { return operation; })) {
        reduce.emitError(
            "scaled partial rewrite left a live reduction-chain operation");
        failed = true;
        break;
      }
      replicaCleanupRoots.push_back(oldScale);
      replicaCleanupRoots.push_back(oldDotSide);
      continue;

    }
    llvm::DenseSet<mlir::Operation *> replicaCleanupVisited;
    llvm::DenseSet<mlir::Value> replicaCleanupStops;
    for (mlir::Value root : replicaCleanupRoots)
      eraseDeadChain(root, replicaCleanupStops, replicaCleanupVisited,
                     rewriter);

    llvm::SmallVector<mlir::scf::ForOp> partialLoops;
    getOperation().walk(
        [&](mlir::scf::ForOp loop) { partialLoops.push_back(loop); });
    for (mlir::scf::ForOp loop : partialLoops) {
      auto matchedContributions = matchLevelScaledLoop(loop);
      if (!matchedContributions ||
          !llvm::all_of(*matchedContributions,
                        [](const ScaledPartialContribution &contribution) {
                          return hasTopology(contribution.dot, "level_scaled");
                        }))
        continue;
      llvm::SmallVector<ScaledPartialContribution> contributions =
          std::move(*matchedContributions);
      const int64_t slots = contributions.front().dot.getPartialUnroll();
      auto yield =
          mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());

      const int64_t reductionAxis = contributions.front().dot.getOver()[0];
      auto carryType =
          mlir::dyn_cast<riscv::ValueType>(loop.getRegionIterArg(0).getType());
      auto carryParts =
          carryType
              ? riscv_internal::staticProduct(
                    carryType.getLayout().getReplicaFactors().asArrayRef())
              : std::optional<int64_t>();
      auto firstLayoutPlan =
          contributions.front().dot.getPartialLayoutPlanAttr();
      auto firstTopology = contributions.front().dot.getPartialTopology();
      auto slotType =
          firstLayoutPlan
              ? mlir::dyn_cast<riscv::ValueType>(
                    firstLayoutPlan.getPartialSlotType())
              : riscv::ValueType();
      bool complete = firstLayoutPlan && carryType &&
                      carryType.getLayout().getCarrier() == "scalar" &&
                      carryParts && *carryParts > 0 && slotType;
      for (ScaledPartialContribution &contribution : contributions) {
        riscv::ValueType lhs = contribution.dot.getLhs().getType();
        riscv::ValueType rhs = contribution.dot.getRhs().getType();
        auto lhsPosition = axisPosition(lhs, reductionAxis);
        auto rhsPosition = axisPosition(rhs, reductionAxis);
        complete &= contribution.dot.getPartialUnroll() == slots &&
                    contribution.dot.getOver().size() == 1 &&
                    contribution.dot.getOver()[0] == reductionAxis &&
                    contribution.dot.getPartialTopology() == firstTopology &&
                    contribution.dot.getResult().getType() == carryType &&
                    contribution.scaleMultiply.getResult().getType() == carryType &&
                    contribution.carryAdd.getResult().getType() == carryType &&
                    lhsPosition && rhsPosition &&
                    lhs.getLayout().getTimeFactors()[*lhsPosition] == 1 &&
                    rhs.getLayout().getTimeFactors()[*rhsPosition] == 1 &&
                    contribution.dot.getPartialLayoutPlanAttr() &&
                    mlir::dyn_cast<riscv::ValueType>(
                        contribution.dot.getPartialLayoutPlanAttr()
                            .getPartialSlotType()) == slotType;
      }
      if (!complete)
        continue;

      struct OutputPartialOperands {
        llvm::SmallVector<mlir::Value> lhs;
        llvm::SmallVector<mlir::Value> rhs;
        llvm::SmallVector<mlir::Value> scales;
        llvm::SmallVector<int64_t> lhsReplicas;
        llvm::SmallVector<int64_t> rhsReplicas;
        llvm::SmallVector<int64_t> scaleReplicas;
      };
      llvm::SmallVector<OutputPartialOperands, 1> outputOperands(*carryParts);
      std::string multiplyInstruction;
      for (int64_t outputPart = 0; outputPart < *carryParts; ++outputPart) {
        OutputPartialOperands &operands = outputOperands[outputPart];
        for (ScaledPartialContribution &contribution : contributions) {
          riscv::ValueType lhs = contribution.dot.getLhs().getType();
          riscv::ValueType rhs = contribution.dot.getRhs().getType();
          riscv::ValueType scale =
              mlir::cast<riscv::ValueType>(contribution.scale.getType());
          auto lhsReplica = projectReplica(lhs, carryType, outputPart);
          auto rhsReplica = projectReplica(rhs, carryType, outputPart);
          auto scaleReplica = projectReplica(scale, carryType, outputPart);
          if (!lhsReplica || !rhsReplica || !scaleReplica) {
            complete = false;
            break;
          }
          auto selected = partialMultiplyInstruction(lhs, rhs);
          if (!selected) {
            complete = false;
            break;
          }
          if (multiplyInstruction.empty())
            multiplyInstruction = *selected;
          else if (multiplyInstruction != *selected) {
            complete = false;
            break;
          }
          operands.lhs.push_back(contribution.dot.getLhs());
          operands.rhs.push_back(contribution.dot.getRhs());
          operands.scales.push_back(contribution.scale);
          operands.lhsReplicas.push_back(*lhsReplica);
          operands.rhsReplicas.push_back(*rhsReplica);
          operands.scaleReplicas.push_back(*scaleReplica);
        }
        if (!complete)
          break;
      }
      if (!complete)
        continue;

      auto reducedSlot = mlir::dyn_cast<riscv::ValueType>(
          firstLayoutPlan.getReducedSlotType());
      if (!reducedSlot)
        continue;
      auto setType = riscv::PartialSetType::get(
          rewriter.getContext(), slotType, reductionAxis, slots,
          slotType.getShape()[0],
          slots * slotType.getLayout().getRegisterGroups());
      auto reducedType = riscv::PartialSetType::get(
          rewriter.getContext(), reducedSlot, reductionAxis, slots,
          setType.getTermsPerSlot(),
          slots * reducedSlot.getLayout().getRegisterGroups());
      const int64_t chains = firstTopology.getCombineArity();
      if (chains <= 0 || slots % chains ||
          firstTopology.getSlotOrder().size() != static_cast<size_t>(slots))
        continue;
      const int64_t fanout = slots / chains;
      auto combinedType = riscv::PartialSetType::get(
          rewriter.getContext(), reducedSlot, reductionAxis, chains,
          reducedType.getTermsPerSlot() * fanout,
          chains * reducedSlot.getLayout().getRegisterGroups());

      rewriter.setInsertionPoint(yield);
      llvm::SmallVector<mlir::Value> scalarParts;
      for (OutputPartialOperands &operands : outputOperands) {
        llvm::SmallVector<int64_t> operandSlots;
        llvm::SmallVector<int64_t> laneOffsets(slots, 0);
        for (int64_t slot = 0; slot < slots; ++slot)
          operandSlots.push_back(slot);
        auto set = rewriter.create<riscv::RVVPartialSetOp>(
            yield.getLoc(), setType, operands.lhs, operands.rhs,
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(operands.lhsReplicas),
            rewriter.getDenseI64ArrayAttr(operands.rhsReplicas),
            rewriter.getDenseI64ArrayAttr(laneOffsets),
            rewriter.getDenseI64ArrayAttr(laneOffsets), reductionAxis,
            multiplyInstruction,
            ownerDomain(yield), nextPartialBirthId++, ownerDomain(yield),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-set", "rvv.partial-set",
                "rvv.partial-set", 0, setType.getResourceGroups(), 0, 0,
                "none", "exact", {reductionAxis, slots}));
        riscv_internal::copyOrigin(contributions.front().dot, set);
        sinkReplicaSupplies(set);
        auto reduced = rewriter.create<riscv::RVVPartialReduceOp>(
            yield.getLoc(), reducedType, set.getResult(),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-reduce",
                mlir::cast<mlir::IntegerType>(slotType.getElementType())
                            .getWidth() == 16
                    ? "rvv.partial-reduce.widen"
                    : "rvv.partial-reduce",
                mlir::cast<mlir::IntegerType>(slotType.getElementType())
                            .getWidth() == 16
                    ? "rvv.partial-reduce.widen"
                    : "rvv.partial-reduce",
                setType.getResourceGroups(), reducedType.getResourceGroups(), 1,
                0, "none", "exact", {reductionAxis, slots}));
        riscv_internal::copyOrigin(contributions.front().dot, reduced);
        auto combined = rewriter.create<riscv::RVVPartialScaleCombineOp>(
            yield.getLoc(), combinedType, reduced.getResult(), operands.scales,
            rewriter.getDenseI64ArrayAttr(operands.scaleReplicas),
            firstTopology.getSlotOrder(),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-scale-combine",
                "rvv.partial-scale-combine", "rvv.partial-scale-combine",
                reducedType.getResourceGroups(),
                combinedType.getResourceGroups(), 1, 0, "none", "exact",
                {reductionAxis, slots, chains, fanout}));
        riscv_internal::copyOrigin(contributions.front().dot, combined);
        auto finalized = rewriter.create<riscv::RVVPartialFinalizeOp>(
            yield.getLoc(), carryType.getElementType(), combined.getResult(),
            reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-finalize",
                *partialFinalizeInstruction(combinedType, reductionAxis),
                *partialFinalizeInstruction(combinedType, reductionAxis),
                combinedType.getResourceGroups(), 0, 0, 0, "none", "exact",
                {reductionAxis, chains}));
        riscv_internal::copyOrigin(contributions.front().dot, finalized);
        scalarParts.push_back(finalized.getResult());
      }
      auto assembled = rewriter.create<riscv::RVVAssembleReplicasOp>(
          yield.getLoc(), carryType, scalarParts,
          riscv_internal::leaf(rewriter, "scalar", "assemble-replicas",
                               "scalar.assemble-replicas",
                               "scalar.assemble-replicas", 0, 0));
      riscv_internal::copyOrigin(contributions.front().dot, assembled);
      riscv::BinaryOp finalAdd = contributions.back().carryAdd;
      finalAdd->moveBefore(yield);
      rewriter.modifyOpInPlace(finalAdd, [&] {
        finalAdd->setOperand(0, loop.getRegionIterArg(0));
        finalAdd->setOperand(1, assembled.getResult());
      });

      for (ScaledPartialContribution &contribution :
           llvm::reverse(contributions)) {
        if (contribution.carryAdd != finalAdd)
          rewriter.eraseOp(contribution.carryAdd);
        rewriter.eraseOp(contribution.scaleMultiply);
        rewriter.eraseOp(contribution.dot);
      }
    }

    // A canonical reduction over one surviving replica axis of a widened dot
    // is one register-local accumulation chain followed by one lane reduction.
    // The topology planner freezes the accumulator layout; this pass only
    // projects each replica contribution and instantiates that closed chain.
    llvm::SmallVector<riscv::ReduceOp> replicaReductions;
    getOperation().walk([&](riscv::ReduceOp reduce) {
      auto dot = reduce.getInput().getDefiningOp<riscv::RVVWidenDotOp>();
      if (dot && hasTopology(dot, "replica_reduced"))
        replicaReductions.push_back(reduce);
    });
    for (riscv::ReduceOp reduce : replicaReductions) {
      auto dot = reduce.getInput().getDefiningOp<riscv::RVVWidenDotOp>();
      auto result = mlir::dyn_cast<riscv::ValueType>(reduce.getResult().getType());
      auto resultElement =
          result ? mlir::dyn_cast<mlir::IntegerType>(result.getElementType())
                 : mlir::IntegerType();
      auto topology = dot ? dot.getPartialTopologyAttr()
                          : riscv::PartialTopologyAttr();
      auto layoutPlan = dot ? dot.getPartialLayoutPlanAttr()
                            : riscv::PartialLayoutPlanAttr();
      auto accumulatorType =
          layoutPlan ? mlir::dyn_cast<riscv::ValueType>(
                           layoutPlan.getVectorSlotType())
                     : riscv::ValueType();
      if (!dot || !result || !resultElement ||
          !resultElement.isSigned() || resultElement.getWidth() != 32 ||
          !topology || topology.getPartialAxes().size() != 1 ||
          !accumulatorType || dot.getReductionStreams() != 1)
        continue;
      const int64_t replicaAxis = topology.getPartialAxes()[0];
      auto projectedLhs = projectReplicaReductionOperandType(
          rewriter, dot.getLhs().getType(), replicaAxis);
      auto projectedRhs = projectReplicaReductionOperandType(
          rewriter, dot.getRhs().getType(), replicaAxis);
      if (!projectedLhs || !projectedRhs ||
          replicaReducedAccumulatorType(rewriter, projectedLhs, projectedRhs,
                                        dot.getOver()[0]) != accumulatorType)
        continue;

      rewriter.setInsertionPoint(reduce);
      const int64_t reductionAxis = dot.getOver()[0];
      auto partialElement =
          mlir::cast<mlir::IntegerType>(accumulatorType.getElementType());
      auto zero = rewriter.create<riscv::ConstantOp>(
          reduce.getLoc(), partialElement,
          rewriter.getIntegerAttr(partialElement, 0));
      auto initial = rewriter.create<riscv::RVVSplatOp>(
          reduce.getLoc(), accumulatorType, zero.getResult(),
          riscv_internal::leaf(
              rewriter, "rvv", "splat", "rvv.splat", "rvv.splat", 0,
              accumulatorType.getLayout().getRegisterGroups()));
      mlir::Value carried = initial.getResult();
      for (int64_t replica = 0; replica < topology.getSourceSlots(); ++replica) {
        auto index = rewriter.create<mlir::arith::ConstantIndexOp>(
            reduce.getLoc(), replica);
        auto lhs = rewriter.create<riscv::ProjectReductionOperandOp>(
            reduce.getLoc(), projectedLhs, dot.getLhs(), index, replicaAxis,
            riscv_internal::leaf(
                rewriter, "transfer", "reduction-projection",
                "rvv.project-reduction-operand",
                "rvv.project-reduction-operand", 0, 0, 1, 0));
        auto rhs = rewriter.create<riscv::ProjectReductionOperandOp>(
            reduce.getLoc(), projectedRhs, dot.getRhs(), index, replicaAxis,
            riscv_internal::leaf(
                rewriter, "transfer", "reduction-projection",
                "rvv.project-reduction-operand",
                "rvv.project-reduction-operand", 0, 0, 1, 0));
        auto accumulate = rewriter.create<riscv::RVVWidenAccumulateOp>(
            reduce.getLoc(), accumulatorType, lhs.getResult(), rhs.getResult(),
            carried, reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "widen-accumulate", "rvv.vwmacc.partial",
                "rvv.vwmacc.partial",
                projectedLhs.getLayout().getRegisterGroups() +
                    projectedRhs.getLayout().getRegisterGroups() +
                    accumulatorType.getLayout().getRegisterGroups(),
                accumulatorType.getLayout().getRegisterGroups(), 0, 0, "none",
                "agnostic"));
        riscv_internal::copyOrigin(dot, lhs);
        riscv_internal::copyOrigin(dot, rhs);
        riscv_internal::copyOrigin(dot, accumulate);
        carried = accumulate.getResult();
      }
      auto finalized = rewriter.create<riscv::RVVFinalizeWidenDotOp>(
          reduce.getLoc(), result, carried, reductionAxis,
          riscv_internal::leaf(
              rewriter, "rvv", "finalize-widen-dot",
              partialElement.getWidth() == 16 ? "rvv.vwredsum.partial"
                                              : "rvv.vredsum.partial",
              partialElement.getWidth() == 16 ? "rvv.vwredsum.partial"
                                              : "rvv.vredsum.partial",
              accumulatorType.getLayout().getRegisterGroups(), 0, 2, 0,
              "none", "exact", {reductionAxis}));
      riscv_internal::copyOrigin(reduce, finalized);
      reduce.getResult().replaceAllUsesWith(finalized.getResult());
      rewriter.eraseOp(reduce);
      rewriter.eraseOp(dot);
    }

    llvm::SmallVector<riscv::RVVWidenDotOp> independentDots;
    getOperation().walk([&](riscv::RVVWidenDotOp dot) {
      independentDots.push_back(dot);
    });
    for (riscv::RVVWidenDotOp dot : independentDots) {
      if (!hasTopology(dot, "independent") || dot.getOver().empty())
        continue;
      const int64_t reductionAxis = dot.getOver()[0];
      riscv::ValueType lhs = dot.getLhs().getType();
      riscv::ValueType rhs = dot.getRhs().getType();
      auto lhsPosition = axisPosition(lhs, reductionAxis);
      auto rhsPosition = axisPosition(rhs, reductionAxis);
      if (!lhsPosition || !rhsPosition)
        continue;
      auto lhsStreams = riscv_internal::staticProduct(
          lhs.getLayout().getTimeFactors().asArrayRef());
      auto rhsStreams = riscv_internal::staticProduct(
          rhs.getLayout().getTimeFactors().asArrayRef());
      if (!lhsStreams || !rhsStreams || *lhsStreams != *rhsStreams)
        continue;
      const int64_t slots = *lhsStreams;
      if (slots < 2 || (slots & (slots - 1)) != 0)
        continue;
      auto layoutPlan = dot.getPartialLayoutPlanAttr();
      auto slotType =
          layoutPlan ? mlir::dyn_cast<riscv::ValueType>(
                           layoutPlan.getPartialSlotType())
                     : riscv::ValueType();
      auto resultValue =
          mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
      auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
          resultValue ? resultValue.getElementType() : dot.getResult().getType());
      auto outputParts = resultValue
                             ? riscv_internal::staticProduct(
                                   resultValue.getLayout()
                                       .getReplicaFactors()
                                       .asArrayRef())
                             : std::optional<int64_t>(1);
      if (!slotType || !resultElement || !resultElement.isSigned() ||
          resultElement.getWidth() != 32 || !outputParts || *outputParts <= 0)
        continue;
      auto multiplyInstruction = partialMultiplyInstruction(lhs, rhs);
      if (!multiplyInstruction)
        continue;

      riscv::ConvertLayoutOp vectorConsumer;
      riscv::ValueType vectorResult;
      if (dot.getResult().hasOneUse()) {
        vectorConsumer = mlir::dyn_cast<riscv::ConvertLayoutOp>(
            *dot.getResult().getUsers().begin());
        if (vectorConsumer &&
            vectorConsumer.getConversion().getEffect() == "pure" &&
            vectorConsumer.getConversion().getKind() == "register_to_lane")
          vectorResult =
              mlir::dyn_cast<riscv::ValueType>(vectorConsumer.getResult().getType());
      }
      auto vectorSlotType =
          layoutPlan ? mlir::dyn_cast<riscv::ValueType>(
                           layoutPlan.getVectorSlotType())
                     : riscv::ValueType();
      auto lhsReplicas = riscv_internal::staticProduct(
          lhs.getLayout().getReplicaFactors().asArrayRef());
      auto rhsReplicas = riscv_internal::staticProduct(
          rhs.getLayout().getReplicaFactors().asArrayRef());
      auto vectorResultStreams =
          vectorResult
              ? riscv_internal::staticProduct(
                    vectorResult.getLayout().getTimeFactors().asArrayRef())
              : std::optional<int64_t>();
      auto vectorResultReplicas =
          vectorResult
              ? riscv_internal::staticProduct(
                    vectorResult.getLayout().getReplicaFactors().asArrayRef())
              : std::optional<int64_t>();
      const bool preservesFreeLane =
          vectorConsumer && vectorResult && vectorSlotType && lhsStreams && rhsStreams &&
          *lhsStreams == slots && *rhsStreams == slots && lhsReplicas &&
          rhsReplicas && *lhsReplicas == 1 && *rhsReplicas == 1 &&
          vectorResultStreams && vectorResultReplicas &&
          *vectorResultStreams == 1 && *vectorResultReplicas == 1;
      if (preservesFreeLane) {
        const int64_t slotGroups =
            vectorSlotType.getLayout().getRegisterGroups();
        const int64_t setGroups =
            slots * slotGroups;
        const int64_t operandGroups = lhs.getLayout().getRegisterGroups() +
                                      rhs.getLayout().getRegisterGroups();
        auto kernel = dot->getParentOfType<riscv::KernelOp>();
        if (!kernel || setGroups <= 0 || operandGroups < 0 ||
            setGroups >
                kernel.getTarget().getVectorRegisters() - operandGroups)
          continue;

        rewriter.setInsertionPoint(dot);
        auto partialType = riscv::PartialSetType::get(
            rewriter.getContext(), vectorSlotType, reductionAxis, slots,
            vectorSlotType.getShape()[*axisPosition(vectorSlotType,
                                                     reductionAxis)],
            setGroups);
        llvm::SmallVector<int64_t> operandSlots(slots, 0);
        llvm::SmallVector<int64_t> operandParts;
        for (int64_t stream = 0; stream < slots; ++stream)
          operandParts.push_back(stream);
        mlir::Value partials;
        int64_t remainingSlots = slots;
        int64_t termsPerSlot = partialType.getTermsPerSlot();
        auto set = rewriter.create<riscv::RVVPartialSetOp>(
            dot.getLoc(), partialType, mlir::ValueRange{dot.getLhs()},
            mlir::ValueRange{dot.getRhs()},
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(operandParts),
            rewriter.getDenseI64ArrayAttr(operandParts),
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(operandSlots), reductionAxis,
            *multiplyInstruction,
            ownerDomain(dot), nextPartialBirthId++, ownerDomain(dot),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-set", "rvv.partial-set",
                "rvv.partial-set", operandGroups, setGroups, 0, 0, "none",
                "exact", {reductionAxis, slots}));
        riscv_internal::copyOrigin(dot, set);
        sinkReplicaSupplies(set);
        partials = set.getResult();
        while (remainingSlots > 1) {
          if (remainingSlots % 2)
            break;
          remainingSlots /= 2;
          termsPerSlot *= 2;
          auto combinedType = riscv::PartialSetType::get(
              rewriter.getContext(), vectorSlotType, reductionAxis,
              remainingSlots, termsPerSlot,
              remainingSlots *
                  vectorSlotType.getLayout().getRegisterGroups());
          auto combine = rewriter.create<riscv::RVVPartialCombineOp>(
              dot.getLoc(), combinedType, partials, 2, "pairwise",
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-combine",
                  "rvv.partial-combine", "rvv.partial-combine",
                  mlir::cast<riscv::PartialSetType>(partials.getType())
                      .getResourceGroups(),
                  combinedType.getResourceGroups(), 0, 0, "none", "exact",
                  {reductionAxis, 2, remainingSlots}));
          riscv_internal::copyOrigin(dot, combine);
          partials = combine.getResult();
        }
        if (remainingSlots != 1)
          continue;
        auto finalize = rewriter.create<riscv::RVVPartialFinalizeOp>(
            dot.getLoc(), vectorResult, partials, reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-finalize",
                "rvv.partial-finalize.widen",
                "rvv.partial-finalize.widen",
                mlir::cast<riscv::PartialSetType>(partials.getType())
                    .getResourceGroups(),
                vectorResult.getLayout().getRegisterGroups(), 1, 0, "none",
                "exact", {reductionAxis, slots}));
        riscv_internal::copyOrigin(dot, finalize);
        vectorConsumer.getResult().replaceAllUsesWith(finalize.getResult());
        rewriter.eraseOp(vectorConsumer);
        rewriter.eraseOp(dot);
        continue;
      }

      const int64_t slotGroups = slotType.getLayout().getRegisterGroups();
      const int64_t setGroups = slots * slotGroups;
      const int64_t operandGroups = lhs.getLayout().getRegisterGroups() +
                                    rhs.getLayout().getRegisterGroups();
      auto kernel = dot->getParentOfType<riscv::KernelOp>();
      if (!kernel || setGroups <= 0 || operandGroups < 0 ||
          setGroups >
              kernel.getTarget().getVectorRegisters() - operandGroups)
        continue;

      rewriter.setInsertionPoint(dot);
      llvm::SmallVector<mlir::Value> scalarParts;
      bool complete = true;
      for (int64_t outputPart = 0; outputPart < *outputParts; ++outputPart) {
        auto lhsReplica = resultValue
                              ? projectReplica(lhs, resultValue, outputPart)
                              : std::optional<int64_t>(0);
        auto rhsReplica = resultValue
                              ? projectReplica(rhs, resultValue, outputPart)
                              : std::optional<int64_t>(0);
        if (!lhsReplica || !rhsReplica) {
          complete = false;
          break;
        }
        auto partialType = riscv::PartialSetType::get(
            rewriter.getContext(), slotType, reductionAxis, slots,
            slotType.getShape()[0], setGroups);
        llvm::SmallVector<int64_t> operandSlots(slots, 0);
        llvm::SmallVector<int64_t> lhsParts;
        llvm::SmallVector<int64_t> rhsParts;
        for (int64_t stream = 0; stream < slots; ++stream) {
          lhsParts.push_back(*lhsReplica * slots + stream);
          rhsParts.push_back(*rhsReplica * slots + stream);
        }
        mlir::Value partials;
        int64_t remainingSlots = slots;
        int64_t termsPerSlot = partialType.getTermsPerSlot();
        auto set = rewriter.create<riscv::RVVPartialSetOp>(
            dot.getLoc(), partialType, mlir::ValueRange{dot.getLhs()},
            mlir::ValueRange{dot.getRhs()},
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(lhsParts),
            rewriter.getDenseI64ArrayAttr(rhsParts),
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(operandSlots), reductionAxis,
            *multiplyInstruction,
            ownerDomain(dot), nextPartialBirthId++, ownerDomain(dot),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-set", "rvv.partial-set",
                "rvv.partial-set", operandGroups, setGroups, 0, 0, "none",
                "exact", {reductionAxis, slots}));
        riscv_internal::copyOrigin(dot, set);
        sinkReplicaSupplies(set);
        partials = set.getResult();
        while (remainingSlots > 1) {
          const int64_t arity = remainingSlots % 2 == 0 ? 2 : remainingSlots;
          remainingSlots /= arity;
          termsPerSlot *= arity;
          auto combinedType = riscv::PartialSetType::get(
              rewriter.getContext(), slotType, reductionAxis, remainingSlots,
              termsPerSlot,
              remainingSlots * slotType.getLayout().getRegisterGroups());
          auto combine = rewriter.create<riscv::RVVPartialCombineOp>(
              dot.getLoc(), combinedType, partials, arity, "pairwise",
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-combine",
                  "rvv.partial-combine", "rvv.partial-combine",
                  mlir::cast<riscv::PartialSetType>(partials.getType())
                      .getResourceGroups(),
                  combinedType.getResourceGroups(), 0, 0, "none", "exact",
                  {reductionAxis, arity, remainingSlots}));
          riscv_internal::copyOrigin(dot, combine);
          partials = combine.getResult();
        }
        if (!complete)
          break;
        auto finalizeInstruction = partialFinalizeInstruction(
            mlir::cast<riscv::PartialSetType>(partials.getType()),
            reductionAxis);
        if (!finalizeInstruction) {
          complete = false;
          break;
        }
        auto finalize = rewriter.create<riscv::RVVPartialFinalizeOp>(
            dot.getLoc(), resultElement, partials, reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-finalize",
                *finalizeInstruction, *finalizeInstruction,
                mlir::cast<riscv::PartialSetType>(partials.getType())
                    .getResourceGroups(),
                0, 2, 0, "none", "exact",
                {reductionAxis, remainingSlots}));
        riscv_internal::copyOrigin(dot, finalize);
        scalarParts.push_back(finalize.getResult());
      }
      if (!complete)
        continue;
      mlir::Value replacement;
      if (resultValue) {
        auto assembled = rewriter.create<riscv::RVVAssembleReplicasOp>(
            dot.getLoc(), resultValue, scalarParts,
            riscv_internal::leaf(rewriter, "scalar", "assemble-replicas",
                                 "scalar.assemble-replicas",
                                 "scalar.assemble-replicas", 0, 0));
        riscv_internal::copyOrigin(dot, assembled);
        replacement = assembled.getResult();
      } else {
        replacement = scalarParts.front();
      }
      dot.getResult().replaceAllUsesWith(replacement);
      rewriter.eraseOp(dot);
    }

    // Scalar add trees can expose independently materialized widened partials
    // that share one logical reduction axis but use different legal RVV lane
    // widths. Normalize the wider partials to the widest existing common
    // representation, join their slots, and reduce once.
    llvm::SmallVector<riscv::BinaryOp> partialAddRoots;
    getOperation().walk([&](riscv::BinaryOp add) {
      if (add.getKind() == "add" &&
          mlir::isa<mlir::IntegerType>(add.getResult().getType()))
        partialAddRoots.push_back(add);
    });
    for (riscv::BinaryOp root : partialAddRoots) {
      if (!root || !root->getBlock())
        continue;
      llvm::SmallVector<riscv::BinaryOp> adds;
      llvm::SmallVector<PartialAddLeaf> leaves;
      if (!collectPartialAddTree(root.getResult(), root, adds, leaves) ||
          leaves.size() < 2)
        continue;

      llvm::SmallVector<riscv::PartialSetType> leafTypes;
      bool completeTypes = true;
      for (PartialAddLeaf leaf : leaves) {
        auto type = partialTypeForAddLeaf(rewriter, leaf);
        if (!type) {
          completeTypes = false;
          break;
        }
        leafTypes.push_back(*type);
      }
      if (!completeTypes)
        continue;

      riscv::PartialSetType targetType;
      llvm::SmallVector<int64_t> splits;
      llvm::SmallVector<size_t> order(leaves.size());
      std::iota(order.begin(), order.end(), 0);
      llvm::sort(order, [&](size_t lhs, size_t rhs) {
        auto lhsSet = leafTypes[lhs];
        auto rhsSet = leafTypes[rhs];
        auto lhsPosition = axisPosition(lhsSet.getPartialType(),
                                        lhsSet.getReductionAxis());
        auto rhsPosition = axisPosition(rhsSet.getPartialType(),
                                        rhsSet.getReductionAxis());
        return lhsSet.getPartialType().getLayout().getLaneFactors()
                   [*lhsPosition] >
               rhsSet.getPartialType().getLayout().getLaneFactors()
                   [*rhsPosition];
      });
      for (size_t targetIndex : order) {
        auto candidate = leafTypes[targetIndex];
        llvm::SmallVector<int64_t> candidateSplits;
        bool compatible = true;
        for (riscv::PartialSetType source : leafTypes) {
          auto split = partialSplitFactor(source, candidate);
          if (!split || *split > 2) {
            compatible = false;
            break;
          }
          candidateSplits.push_back(*split);
        }
        if (compatible) {
          targetType = candidate;
          splits = std::move(candidateSplits);
          break;
        }
      }
      if (!targetType)
        continue;

      rewriter.setInsertionPoint(root);
      llvm::SmallVector<mlir::Value> normalized;
      int64_t totalSlots = 0;
      int64_t totalTerms = 0;
      int64_t totalResources = 0;
      for (auto [leaf, declaredType, split] :
           llvm::zip(leaves, leafTypes, splits)) {
        auto sourceType = declaredType;
        mlir::Value value;
        mlir::Operation *origin = nullptr;
        if (leaf.finalize) {
          value = leaf.finalize.getInput();
          origin = leaf.finalize.getOperation();
        } else {
          auto multiplyInstruction = partialMultiplyInstruction(
              leaf.dot.getLhs().getType(), leaf.dot.getRhs().getType());
          if (!multiplyInstruction)
            break;
          auto set = rewriter.create<riscv::RVVPartialSetOp>(
              root.getLoc(), sourceType,
              mlir::ValueRange{leaf.dot.getLhs()},
              mlir::ValueRange{leaf.dot.getRhs()},
              rewriter.getDenseI64ArrayAttr({0}),
              rewriter.getDenseI64ArrayAttr({0}),
              rewriter.getDenseI64ArrayAttr({0}),
              rewriter.getDenseI64ArrayAttr({0}),
              rewriter.getDenseI64ArrayAttr({0}),
              rewriter.getDenseI64ArrayAttr({0}),
              sourceType.getReductionAxis(), *multiplyInstruction,
              ownerDomain(root), nextPartialBirthId++, ownerDomain(root),
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-set", "rvv.partial-set",
                  "rvv.partial-set", 0, sourceType.getResourceGroups(), 0, 0,
                  "none", "exact", {sourceType.getReductionAxis(), 1}));
          riscv_internal::copyOrigin(leaf.dot, set);
          sinkReplicaSupplies(set);
          value = set.getResult();
          origin = leaf.dot.getOperation();
        }
        if (split > 1) {
          auto repackedType = riscv::PartialSetType::get(
              rewriter.getContext(), targetType.getPartialType(),
              targetType.getReductionAxis(), sourceType.getSlots() * split,
              sourceType.getTermsPerSlot() / split,
              sourceType.getResourceGroups());
          auto repack = rewriter.create<riscv::RVVPartialRepackOp>(
              root.getLoc(), repackedType, value, split,
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-repack",
                  "rvv.partial-repack.split", "rvv.partial-repack.split",
                  sourceType.getResourceGroups(),
                  repackedType.getResourceGroups(), 0, 0, "none", "exact",
                  {sourceType.getReductionAxis(), split}));
          riscv_internal::copyOrigin(origin, repack);
          value = repack.getResult();
          sourceType = repackedType;
        }
        normalized.push_back(value);
        totalSlots += sourceType.getSlots();
        totalTerms += sourceType.getSlots() * sourceType.getTermsPerSlot();
        totalResources += sourceType.getResourceGroups();
      }
      if (normalized.size() != leaves.size())
        continue;
      auto combinedType = riscv::PartialSetType::get(
          rewriter.getContext(), targetType.getPartialType(),
          targetType.getReductionAxis(), 1, totalTerms,
          targetType.getPartialType().getLayout().getRegisterGroups());
      auto combine = rewriter.create<riscv::RVVPartialMergeOp>(
          root.getLoc(), combinedType, normalized, "pairwise",
          riscv_internal::leaf(
              rewriter, "rvv", "partial-merge", "rvv.partial-merge",
              "rvv.partial-merge", totalResources,
              combinedType.getResourceGroups(), 0, 0, "none", "exact",
              {targetType.getReductionAxis(),
               static_cast<int64_t>(normalized.size()), totalSlots,
               totalTerms}));
      riscv_internal::copyOrigin(root, combine);
      auto instruction = partialFinalizeInstruction(
          combinedType, targetType.getReductionAxis());
      if (!instruction)
        continue;
      auto finalized = rewriter.create<riscv::RVVPartialFinalizeOp>(
          root.getLoc(), root.getResult().getType(), combine.getResult(),
          targetType.getReductionAxis(),
          riscv_internal::leaf(
              rewriter, "rvv", "partial-finalize", *instruction,
              *instruction, combinedType.getResourceGroups(), 0, 2, 0,
              "none", "exact", {targetType.getReductionAxis(), 1}));
      riscv_internal::copyOrigin(root, finalized);

      root.getResult().replaceAllUsesWith(finalized.getResult());
      for (riscv::BinaryOp add : adds)
        if (add && add->getBlock() && add.getResult().use_empty())
          rewriter.eraseOp(add);
      for (PartialAddLeaf leaf : leaves) {
        if (leaf.finalize && leaf.finalize->getBlock() &&
            leaf.finalize.getResult().use_empty())
          rewriter.eraseOp(leaf.finalize);
        if (leaf.dot && leaf.dot->getBlock() && leaf.dot.getResult().use_empty())
          rewriter.eraseOp(leaf.dot);
      }
    }

    llvm::SmallVector<riscv::RVVWidenDotOp> dots;
    getOperation().walk(
        [&](riscv::RVVWidenDotOp dot) { dots.push_back(dot); });
    for (riscv::RVVWidenDotOp dot : dots) {
      if (!hasTopology(dot, "layered") || dot.getOver().size() != 1)
        continue;
      auto plan = dot.getLayeredPartialPlanAttr();
      llvm::DenseSet<mlir::Operation *> lhsVisited;
      llvm::DenseSet<mlir::Operation *> rhsVisited;
      llvm::SmallVector<ProjectedRoot> roots;
      collectProjectedRoots(dot.getLhs(), dot.getOver()[0], lhsVisited, roots);
      collectProjectedRoots(dot.getRhs(), dot.getOver()[0], rhsVisited, roots);
      if (!plan || plan.getRootIndex() < 0 ||
          plan.getRootIndex() >= static_cast<int64_t>(roots.size()) ||
          plan.getRootWindowTypes().size() != roots.size() ||
          plan.getRootStoragePlans().size() != roots.size()) {
        dot.emitError(
            "selected layered partial topology no longer satisfies its typed plan");
        failed = true;
        continue;
      }
      const int64_t reductionAxis = plan.getGeometry().getAxis();
      ProjectedRoot layered = roots[static_cast<size_t>(plan.getRootIndex())];
      if (!layered.layered || !layered.field || !layered.origin ||
          layered.geometry != plan.getGeometry()) {
        dot.emitError(
            "selected layered partial root disagrees with its frozen typed geometry");
        failed = true;
        continue;
      }
      llvm::DenseMap<mlir::Value, riscv::ValueType> windowTypes;
      llvm::DenseMap<mlir::Value, riscv::StorageWindowPlanAttr> storagePlans;
      bool completeRootTypes = true;
      for (auto [root, typeAttributeValue, storagePlanValue] :
           llvm::zip(roots, plan.getRootWindowTypes(),
                     plan.getRootStoragePlans())) {
        auto typeAttribute = mlir::dyn_cast<mlir::TypeAttr>(typeAttributeValue);
        auto windowType =
            typeAttribute
                ? mlir::dyn_cast<riscv::ValueType>(typeAttribute.getValue())
                : riscv::ValueType();
        auto storagePlan =
            mlir::dyn_cast<riscv::StorageWindowPlanAttr>(storagePlanValue);
        completeRootTypes &= static_cast<bool>(windowType) &&
                             static_cast<bool>(storagePlan);
        if (windowType && storagePlan) {
          windowTypes[root.value] = windowType;
          storagePlans[root.value] = storagePlan;
        }
      }
      if (!completeRootTypes) {
        dot.emitError("selected layered partial roots lost their typed windows");
        failed = true;
        continue;
      }
      riscv::AccessAttr access = layered.access;
      const int64_t group = plan.getGeometry().getGroupSize();
      const int64_t layerExtent = plan.getGeometry().getLayerSize();
      auto storageType =
          mlir::cast<riscv::LayeredWindowType>(plan.getStorageWindowType());
      const int64_t layers = storageType.getLayers();
      const int64_t lanes = plan.getGeometry().getLaneCount();
      const int64_t windowsPerLayer = storageType.getWindowsPerLayer();
      const int64_t windowCount =
          static_cast<int64_t>(plan.getGeometry().getGroupForWindow().size());
      auto layeredWindowType =
          mlir::cast<riscv::ValueType>(plan.getDecodedWindowType());
      auto accumulatorType =
          mlir::cast<riscv::ValueType>(plan.getAccumulatorType());
      auto partialInteger = mlir::cast<mlir::IntegerType>(
          accumulatorType.getElementType());

      rewriter.setInsertionPoint(dot);
      auto partialElement =
          mlir::cast<mlir::IntegerType>(accumulatorType.getElementType());
      auto zeroAttr = rewriter.getIntegerAttr(partialElement, 0);
      auto zero = rewriter.create<riscv::ConstantOp>(dot.getLoc(), partialElement,
                                                      zeroAttr);
      auto initial = rewriter.create<riscv::RVVSplatOp>(
          dot.getLoc(), accumulatorType, zero.getResult(),
          riscv_internal::leaf(
              rewriter, "rvv", "splat", "rvv.splat", "rvv.splat", 0,
              accumulatorType.getLayout().getRegisterGroups()));
      mlir::Value lower = rewriter.create<mlir::arith::ConstantIndexOp>(
          dot.getLoc(), 0);
      mlir::Value upper = rewriter.create<mlir::arith::ConstantIndexOp>(
          dot.getLoc(), windowCount);
      mlir::Value step = rewriter.create<mlir::arith::ConstantIndexOp>(
          dot.getLoc(), 1);
      auto loop = rewriter.create<mlir::scf::ForOp>(
          dot.getLoc(), lower, upper, step, mlir::ValueRange{initial.getResult()});
      loop->setAttr("weft.riscv.direction",
                    rewriter.getStringAttr("ascending"));
      loop->setAttr("weft.riscv.system_unroll",
                    rewriter.getStringAttr("disable"));
      rewriter.setInsertionPointToStart(loop.getBody());
      mlir::Value windowIndex = loop.getInductionVar();
      const int64_t rawGroups =
          layeredWindowType.getLayout().getRegisterGroups();
      auto storagePlan = storagePlans.lookup(layered.value);
      if (!storagePlan) {
        dot.emitError(
            "selected layered partial root has no closed storage-load plan");
        failed = true;
        continue;
      }
      auto storage = rewriter.create<riscv::RVVLayeredStorageLoadOp>(
          dot.getLoc(), storageType, layered.field.getResult(),
          layered.origin.getResult(), windowIndex, storagePlan, access,
          riscv_internal::leaf(
                rewriter, "rvv", "layered-storage-load",
                "rvv.layered-storage-load", "rvv.layered-storage-load",
                mlir::cast<riscv::ValueType>(layered.field.getResult().getType())
                    .getLayout()
                    .getRegisterGroups(),
                rawGroups, rawGroups, 0, "none",
              layeredWindowType.getLayout().getValidity() == "tail" ? "agnostic"
                                                                     : "exact"));

      mlir::Value windowsPerLayerValue =
          rewriter.create<mlir::arith::ConstantIndexOp>(dot.getLoc(),
                                                        windowsPerLayer);
      mlir::Value groupValue = rewriter.create<mlir::arith::ConstantIndexOp>(
          dot.getLoc(), group);
      mlir::Value laneValue = rewriter.create<mlir::arith::ConstantIndexOp>(
          dot.getLoc(), lanes);
      mlir::Value groupIndex = rewriter.create<mlir::arith::DivUIOp>(
          dot.getLoc(), windowIndex, windowsPerLayerValue);
      mlir::Value withinWindow = rewriter.create<mlir::arith::RemUIOp>(
          dot.getLoc(), windowIndex, windowsPerLayerValue);
      mlir::Value groupBase = rewriter.create<mlir::arith::MulIOp>(
          dot.getLoc(), groupIndex, groupValue);
      mlir::Value withinBase = rewriter.create<mlir::arith::MulIOp>(
          dot.getLoc(), withinWindow, laneValue);

      mlir::Value carried = loop.getRegionIterArg(0);
      for (int64_t layer = 0; layer < layers; ++layer) {
        const int64_t physicalLayer =
            plan.getGeometry().getPhysicalLayerForLogicalLayer()[
                static_cast<size_t>(layer)];
        const int64_t shiftAmount =
            plan.getGeometry().getShiftAmountForLogicalLayer()[
                static_cast<size_t>(layer)];
        const int64_t maskValue =
            plan.getGeometry().getMaskValueForLogicalLayer()[
                static_cast<size_t>(layer)];
        llvm::StringRef instruction =
            riscv_internal::layeredStorageDecodeInstruction(shiftAmount,
                                                            maskValue);
        auto decoded = rewriter.create<riscv::RVVLayeredStorageDecodeOp>(
            dot.getLoc(), layeredWindowType, storage.getResult(), layer,
            physicalLayer, shiftAmount, maskValue,
            riscv_internal::leaf(
                rewriter, "rvv", "layered-storage-decode",
                instruction, instruction,
                rawGroups, layeredWindowType.getLayout().getRegisterGroups(), 1,
                0, "none",
                layeredWindowType.getLayout().getValidity() == "tail"
                    ? "agnostic"
                    : "exact"));
        mlir::Value layerValue = rewriter.create<mlir::arith::ConstantIndexOp>(
            dot.getLoc(), layer * layerExtent);
        mlir::Value layerBase = rewriter.create<mlir::arith::AddIOp>(
            dot.getLoc(), groupBase, layerValue);
        mlir::Value logicalOffset = rewriter.create<mlir::arith::AddIOp>(
            dot.getLoc(), layerBase, withinBase);

        llvm::DenseMap<mlir::Value, mlir::Value> replacements;
        replacements[layered.value] = decoded.getResult();
        for (ProjectedRoot &root : roots) {
          if (root.value == layered.value)
            continue;
          riscv::ValueType windowType = windowTypes.lookup(root.value);
          llvm::StringRef rootTail =
              windowType.getLayout().getValidity() == "tail" ? "agnostic"
                                                               : "exact";
          auto rootPlan = storagePlans.lookup(root.value);
          if (!rootPlan) {
            dot.emitError(
                "projected partial root has no closed typed storage-window plan");
            failed = true;
            break;
          }
          auto window = rewriter.create<riscv::RVVStorageWindowOp>(
              dot.getLoc(), windowType, root.field.getResult(),
              root.origin.getResult(), logicalOffset, rootPlan,
              root.field.getAccess(),
              riscv_internal::leaf(
                  rewriter, "rvv", "storage-window",
                  root.field.getAccess().getMapping() == "grouped_layered"
                      ? "rvv.storage-window.layered"
                      : "rvv.storage-window.natural",
                  root.field.getAccess().getMapping() == "grouped_layered"
                      ? "rvv.storage-window.layered"
                      : "rvv.storage-window.natural",
                  mlir::cast<riscv::ValueType>(root.field.getResult().getType())
                      .getLayout()
                      .getRegisterGroups(),
                  windowType.getLayout().getRegisterGroups(), 1, 0, "none",
                  rootTail));
          replacements[root.value] = window.getResult();
        }

        llvm::DenseMap<mlir::Value, mlir::Value> clones;
        llvm::DenseMap<mlir::Value, bool> dependence;
        auto lhsSlice = cloneWindowSlice(dot.getLhs(), replacements,
                                         reductionAxis, rewriter, clones,
                                         dependence);
        auto rhsSlice = cloneWindowSlice(dot.getRhs(), replacements,
                                         reductionAxis, rewriter, clones,
                                         dependence);
        if (mlir::failed(lhsSlice) || mlir::failed(rhsSlice)) {
          failed = true;
          break;
        }

        auto accumulate = rewriter.create<riscv::RVVWidenAccumulateOp>(
            dot.getLoc(), accumulatorType, *lhsSlice, *rhsSlice, carried,
            reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "widen-accumulate", "rvv.vwmacc.partial",
                "rvv.vwmacc.partial",
                mlir::cast<riscv::ValueType>(lhsSlice->getType())
                        .getLayout()
                        .getRegisterGroups() +
                    mlir::cast<riscv::ValueType>(rhsSlice->getType())
                        .getLayout()
                        .getRegisterGroups() +
                    accumulatorType.getLayout().getRegisterGroups(),
                accumulatorType.getLayout().getRegisterGroups(), 0, 0, "none",
                "agnostic"));
        carried = accumulate.getResult();
      }
      if (failed) {
        rewriter.eraseOp(loop);
        break;
      }
      rewriter.create<mlir::scf::YieldOp>(dot.getLoc(), carried);
      rewriter.setInsertionPointAfter(loop);
      auto finalized = rewriter.create<riscv::RVVFinalizeWidenDotOp>(
          dot.getLoc(), dot.getResult().getType(), loop.getResult(0),
          reductionAxis,
          riscv_internal::leaf(rewriter, "rvv", "finalize-widen-dot",
                               partialInteger.getWidth() == 16
                                   ? "rvv.vwredsum.partial"
                                   : "rvv.vredsum.partial",
                               partialInteger.getWidth() == 16
                                   ? "rvv.vwredsum.partial"
                                   : "rvv.vredsum.partial",
                               accumulatorType.getLayout().getRegisterGroups(),
                               0, 2));
      riscv_internal::copyOrigin(dot, finalized);
      mlir::Value oldLhs = dot.getLhs();
      mlir::Value oldRhs = dot.getRhs();
      dot.getResult().replaceAllUsesWith(finalized.getResult());
      rewriter.eraseOp(dot);
      llvm::DenseSet<mlir::Value> stops;
      for (ProjectedRoot &root : roots)
        stops.insert(root.field.getResult());
      llvm::DenseSet<mlir::Operation *> erased;
      eraseDeadChain(oldLhs, stops, erased, rewriter);
      eraseDeadChain(oldRhs, stops, erased, rewriter);
    }

    llvm::SmallVector<riscv::RVVPartialSetOp> partialSets;
    getOperation().walk(
        [&](riscv::RVVPartialSetOp partial) { partialSets.push_back(partial); });
    for (riscv::RVVPartialSetOp partial : partialSets)
      sinkReplicaSupplies(partial);

    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
weft::createPlanRISCVPartialTopologiesPass() {
  return std::make_unique<PlanRISCVPartialTopologiesPass>();
}

std::unique_ptr<mlir::Pass>
weft::createMaterializeRISCVPartialAccumulatorsPass() {
  return std::make_unique<MaterializeRISCVPartialAccumulatorsPass>();
}
