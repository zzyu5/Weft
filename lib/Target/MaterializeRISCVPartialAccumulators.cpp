#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"
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
#include <utility>

using namespace weft;

namespace {

int64_t product(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values)
    result *= value;
  return result;
}

std::optional<int64_t> checkedProduct(llvm::ArrayRef<int64_t> values) {
  int64_t result = 1;
  for (int64_t value : values) {
    if (value <= 0 || result > std::numeric_limits<int64_t>::max() / value)
      return std::nullopt;
    result *= value;
  }
  return result;
}

bool reachesReductionThroughPureOps(mlir::Value root) {
  llvm::SmallVector<mlir::Value> worklist{root};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (mlir::Operation *consumer : value.getUsers()) {
      if (mlir::isa<riscv::ReduceOp>(consumer))
        return true;
      if (consumer->getNumRegions() == 0 && consumer->getNumResults() == 1 &&
          mlir::isMemoryEffectFree(consumer))
        worklist.push_back(consumer->getResult(0));
    }
  }
  return false;
}

bool checkedSubtract(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs) ||
      (rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs))
    return false;
  result = lhs - rhs;
  return true;
}

bool checkedScale(int64_t value, int64_t factor, int64_t &result) {
  if (factor < 0 ||
      (factor > 0 &&
       (value > std::numeric_limits<int64_t>::max() / factor ||
        value < std::numeric_limits<int64_t>::min() / factor)))
    return false;
  result = value * factor;
  return true;
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

std::optional<int64_t>
laneProductForAxes(riscv::ValueType value, llvm::ArrayRef<int64_t> axes) {
  int64_t result = 1;
  for (int64_t axis : axes) {
    auto position = axisPosition(value, axis);
    if (!position)
      return std::nullopt;
    const int64_t lanes = value.getLayout().getLaneFactors()[*position];
    if (lanes <= 0 || result > std::numeric_limits<int64_t>::max() / lanes)
      return std::nullopt;
    result *= lanes;
  }
  return result;
}

std::optional<int64_t>
primaryReductionLaneAxis(riscv::ValueType partial,
                         llvm::ArrayRef<int64_t> reductionAxes) {
  if (!partial || reductionAxes.empty())
    return std::nullopt;
  std::optional<int64_t> primary;
  for (int64_t axis : reductionAxes) {
    auto position = axisPosition(partial, axis);
    if (!position)
      return std::nullopt;
    if (partial.getLayout().getLaneFactors()[*position] > 1)
      primary = axis;
  }
  // Layout propagation coalesces reduction coordinates in logical order and
  // makes the last lane-bearing axis the physical anchor. A scalar product has
  // no non-unit lane, but still needs one deterministic retained identity.
  return primary ? primary
                 : std::optional<int64_t>(reductionAxes.back());
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
    auto inputType = mlir::dyn_cast<riscv::ValueType>((*input).getType());
    const bool sameWidth =
        inputType &&
        riscv_internal::logicalBitWidth(inputType.getElementType()) ==
            riscv_internal::logicalBitWidth(resultType.getElementType());
    if (sameWidth && inputType.getLayout() != resultType.getLayout()) {
      auto castType = riscv::ValueType::get(
          rewriter.getContext(), resultType.getElementType(),
          inputType.getShape(), inputType.getAxisIds(), inputType.getLayout());
      auto localCast = rewriter.create<riscv::CastOp>(
          cast.getLoc(), castType, *input,
          riscv_internal::unselectedLeaf(rewriter));
      auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
          cast.getLoc(), resultType, localCast.getResult(),
          riscv_internal::layoutConversion(rewriter, inputType.getLayout(),
                                           resultType.getLayout()),
          riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
      riscv_internal::copyOrigin(cast, localCast);
      riscv_internal::copyOrigin(cast, conversion);
      cloned = conversion.getResult();
    } else {
      cloned = rewriter
                   .create<riscv::CastOp>(cast.getLoc(), resultType, *input,
                                          cast.getLeaf())
                   .getResult();
    }
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
    if (mlir::failed(indices)) {
      lookup.emitError("issue-window projection could not project lookup indices");
      return mlir::failure();
    }
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

// Materialize one issue-time window of a shaped axis.  This differs from
// projectOneWindow: an upstream producer may carry a wider lane group than its
// downstream consumer and therefore needs to be sliced to the consumer's
// selected window extent, not merely to its own current lane factor.
riscv::ValueType issueWindowType(mlir::Builder &builder,
                                 riscv::TargetAttr target,
                                 riscv::ValueType source, int64_t axis,
                                 int64_t windowExtent) {
  auto position = axisPosition(source, axis);
  if (!position || windowExtent <= 0 ||
      source.getShape()[*position] < windowExtent ||
      source.getShape()[*position] % windowExtent)
    return {};

  llvm::SmallVector<int64_t> shape(source.getShape().asArrayRef());
  llvm::SmallVector<int64_t> time(
      source.getLayout().getTimeFactors().asArrayRef());
  llvm::SmallVector<int64_t> lane(
      source.getLayout().getLaneFactors().asArrayRef());
  llvm::SmallVector<int64_t> replica(
      source.getLayout().getReplicaFactors().asArrayRef());
  llvm::SmallVector<int64_t> fragment(
      source.getLayout().getFragmentFactors().asArrayRef());
  llvm::SmallVector<int64_t> local(
      source.getLayout().getLocalFactors().asArrayRef());
  if (fragment[*position] != 1 || local[*position] != 1)
    return {};
  shape[*position] = windowExtent;
  time[*position] = 1;

  int64_t lmul = source.getLayout().getLmulEighths();
  int64_t vl = source.getLayout().getVl();
  int64_t groups = source.getLayout().getRegisterGroups();
  if (source.getLayout().getCarrier() == "scalar") {
    lane[*position] = 1;
    replica[*position] = windowExtent;
    lmul = 0;
    vl = 1;
    groups = 0;
  } else if (source.getLayout().getCarrier() == "rvv") {
    const int64_t oldLanes = product(source.getLayout().getLaneFactors());
    if (replica[*position] > 1) {
      if (time[*position] != 1 || lane[*position] != 1 ||
          source.getShape()[*position] != replica[*position] ||
          replica[*position] % windowExtent)
        return {};
      replica[*position] = windowExtent;
    } else {
      lane[*position] = windowExtent;
    }
    const int64_t newLanes = product(lane);
    const int64_t replicas = product(replica);
    if (oldLanes <= 0 || newLanes <= 0 || replicas <= 0 || lmul <= 0 ||
        lmul > std::numeric_limits<int64_t>::max() / newLanes)
      return {};
    const int64_t scaledLmul = lmul * newLanes;
    const int64_t requestedLmul =
        scaledLmul / oldLanes + (scaledLmul % oldLanes != 0);
    int64_t elen = 0;
    for (int64_t supported : target.getSupportedSEW().asArrayRef())
      elen = std::max(elen, supported);
    lmul = 0;
    for (int64_t legal : target.getLegalLMULEighths().asArrayRef()) {
      if (legal >= requestedLmul && elen > 0 &&
          legal * elen >= 8 * source.getLayout().getSew() &&
          (lmul == 0 || legal < lmul))
        lmul = legal;
    }
    if (lmul <= 0)
      return {};
    vl = newLanes;
    groups = product({std::max<int64_t>(1, (lmul + 7) / 8), replicas});
  } else {
    return {};
  }

  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), source.getLayout().getCarrier(),
      source.getLayout().getAxisIds(), riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replica),
      riscv_internal::integers(builder, fragment),
      riscv_internal::integers(builder, local), source.getLayout().getSew(),
      lmul, vl, groups, source.getLayout().getValidity());
  if (layout.getCarrier() == "rvv" && !riscv::supportsRVVLayout(target, layout))
    return {};
  return riscv::ValueType::get(
      builder.getContext(), source.getElementType(),
      riscv_internal::integers(builder, shape), source.getAxisIds(), layout);
}

struct IssueStorageWindowCandidate {
  riscv::FieldOp field;
  riscv::PhysicalPointOp point;
  riscv::ValueType resultType;
  riscv::StorageWindowPlanAttr plan;
  int64_t axis;
};

std::optional<IssueStorageWindowCandidate>
analyzeIssueStorageWindowCandidate(mlir::Builder &builder,
                                   riscv::ExtractOp extract) {
  auto field = extract.getInput().getDefiningOp<riscv::FieldOp>();
  auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
  if (!field || !result ||
      extract.getAccess().getMapping() != "grouped_layered" ||
      extract.getIndices().size() != 1)
    return std::nullopt;
  auto point =
      extract.getIndices().front().getDefiningOp<riscv::PhysicalPointOp>();
  if (!point || point.getResult().getType().getDomain().getTail() != "exact")
    return std::nullopt;
  const int64_t axis = point.getResult().getType().getDomain().getAxisId();
  auto position = axisPosition(result, axis);
  if (!position || result.getLayout().getLaneFactors()[*position] <= 1 ||
      result.getShape()[*position] !=
          result.getLayout().getLaneFactors()[*position] *
              result.getLayout().getTimeFactors()[*position])
    return std::nullopt;
  auto plan = riscv_internal::storageWindowPlan(
      builder, field, axis, 0, 1, 1, result.getShape()[*position],
      result.getLayout().getLaneFactors()[*position]);
  if (!plan)
    return std::nullopt;
  return IssueStorageWindowCandidate{field, point, result, *plan, axis};
}

struct IssueStorageSupplyFacts {
  bool closed = false;
  bool hasStorageWindow = false;
};

IssueStorageSupplyFacts analyzeIssueStorageSupply(
    mlir::Builder &builder, mlir::Value value, int64_t axis,
    int64_t windowExtent, riscv::TargetAttr target,
    llvm::DenseMap<mlir::Value, IssueStorageSupplyFacts> &memo) {
  if (auto found = memo.find(value); found != memo.end())
    return found->second;
  auto sourceType = mlir::dyn_cast<riscv::ValueType>(value.getType());
  auto position = sourceType ? axisPosition(sourceType, axis)
                             : std::optional<size_t>();
  if (!sourceType || !position)
    return memo[value] = {true, false};
  if (!issueWindowType(builder, target, sourceType, axis, windowExtent))
    return memo[value] = {};
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    return memo[value] = {};

  if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(definition)) {
    auto candidate = analyzeIssueStorageWindowCandidate(builder, extract);
    return memo[value] = {
               static_cast<bool>(candidate) && candidate->axis == axis,
               static_cast<bool>(candidate) && candidate->axis == axis};
  }
  if (auto window = mlir::dyn_cast<riscv::RVVStorageWindowOp>(definition)) {
    auto plan = window.getPlan();
    const int64_t lanes =
        sourceType.getLayout().getLaneFactors()[*position];
    const bool closed = plan && plan.getReductionAxis() == axis && lanes > 0 &&
                        sourceType.getShape()[*position] ==
                            sourceType.getLayout().getTimeFactors()[*position] *
                                lanes;
    return memo[value] = {closed, closed};
  }

  auto analyzeOperand = [&](mlir::Value operand) {
    return analyzeIssueStorageSupply(builder, operand, axis, windowExtent,
                                     target, memo);
  };
  if (auto unary = mlir::dyn_cast<riscv::UnaryOp>(definition))
    return memo[value] = analyzeOperand(unary.getInput());
  if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(definition)) {
    IssueStorageSupplyFacts lhs = analyzeOperand(binary.getLhs());
    IssueStorageSupplyFacts rhs = analyzeOperand(binary.getRhs());
    return memo[value] = {lhs.closed && rhs.closed,
                          lhs.hasStorageWindow || rhs.hasStorageWindow};
  }
  if (auto cast = mlir::dyn_cast<riscv::CastOp>(definition))
    return memo[value] = analyzeOperand(cast.getInput());
  if (auto narrow = mlir::dyn_cast<riscv::NarrowOp>(definition))
    return memo[value] = analyzeOperand(narrow.getInput());
  if (auto widen = mlir::dyn_cast<riscv::WidenOp>(definition))
    return memo[value] = analyzeOperand(widen.getInput());
  if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(definition))
    return memo[value] = analyzeOperand(conversion.getInput());
  return memo[value] = {};
}

bool supportsIssueStorageRematerialization(mlir::Builder &builder,
                                           mlir::Value value, int64_t axis,
                                           int64_t windowExtent,
                                           riscv::TargetAttr target) {
  llvm::DenseMap<mlir::Value, IssueStorageSupplyFacts> memo;
  IssueStorageSupplyFacts facts = analyzeIssueStorageSupply(
      builder, value, axis, windowExtent, target, memo);
  return facts.closed && facts.hasStorageWindow;
}

mlir::FailureOr<mlir::Value> cloneIssueWindow(
    mlir::Value value, int64_t axis, int64_t windowExtent,
    mlir::Value windowIndex, riscv::TargetAttr target,
    mlir::IRRewriter &rewriter,
    llvm::DenseMap<mlir::Value, mlir::Value> &clones) {
  if (auto found = clones.find(value); found != clones.end())
    return found->second;
  auto sourceType = mlir::dyn_cast<riscv::ValueType>(value.getType());
  auto position = sourceType ? axisPosition(sourceType, axis)
                             : std::optional<size_t>();
  if (!sourceType || !position)
    return value;
  auto resultType =
      issueWindowType(rewriter, target, sourceType, axis, windowExtent);
  mlir::Operation *definition = value.getDefiningOp();
  if (!resultType || !definition) {
    if (definition)
      definition->emitError("issue-window projection has no legal projected type; source=")
          << sourceType << ", axis=" << axis
          << ", window_extent=" << windowExtent;
    return mlir::failure();
  }

  auto cloneOperand = [&](mlir::Value operand) -> mlir::FailureOr<mlir::Value> {
    if (!mlir::isa<riscv::ValueType>(operand.getType()))
      return operand;
    return cloneIssueWindow(operand, axis, windowExtent, windowIndex, target,
                            rewriter, clones);
  };
  auto remember = [&](mlir::Value cloned) -> mlir::Value {
    riscv_internal::copyOrigin(definition, cloned.getDefiningOp());
    clones[value] = cloned;
    return cloned;
  };

  if (auto iota = mlir::dyn_cast<riscv::IotaOp>(definition)) {
    if (iota.getStart() != 0 ||
        iota.getEnd() - iota.getStart() != sourceType.getShape()[*position])
      return mlir::failure();
    auto local = rewriter.create<riscv::IotaOp>(
        iota.getLoc(), resultType, 0, windowExtent,
        riscv_internal::unselectedLeaf(rewriter));
    auto element = mlir::dyn_cast<mlir::IntegerType>(resultType.getElementType());
    if (!element)
      return mlir::failure();
    auto castIndex = rewriter.create<riscv::CastOp>(
        iota.getLoc(), element, windowIndex,
        riscv_internal::unselectedLeaf(rewriter));
    auto extent = rewriter.create<riscv::ConstantOp>(
        iota.getLoc(), element, rewriter.getIntegerAttr(element, windowExtent));
    auto offset = rewriter.create<riscv::BinaryOp>(
        iota.getLoc(), element, castIndex.getResult(), extent.getResult(), "mul",
        riscv_internal::unselectedLeaf(rewriter));
    auto shifted = rewriter.create<riscv::BinaryOp>(
        iota.getLoc(), resultType, local.getResult(), offset.getResult(), "add",
        riscv_internal::unselectedLeaf(rewriter));
    return remember(shifted.getResult());
  }

  if (auto load =
          mlir::dyn_cast<riscv::RVVUnitEntryWindowLoadOp>(definition)) {
    llvm::SmallVector<int64_t> axes(load.getEntryAxes());
    llvm::SmallVector<int64_t> extents(load.getEntryExtents());
    auto found = llvm::find(axes, axis);
    if (found == axes.end() || axes.size() != extents.size())
      return mlir::failure();
    const size_t entryPosition = static_cast<size_t>(found - axes.begin());
    int64_t suffix = 1;
    for (size_t index = entryPosition + 1; index < extents.size(); ++index) {
      if (extents[index] <= 0 ||
          suffix > std::numeric_limits<int64_t>::max() / extents[index])
        return mlir::failure();
      suffix *= extents[index];
    }
    if (windowExtent > std::numeric_limits<int64_t>::max() / suffix)
      return mlir::failure();
    const int64_t coordinateSpan = windowExtent * suffix;
    extents[entryPosition] = windowExtent;
    mlir::Value base = load.getEntryBase();
    mlir::Value windowBase;
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(base.getType())) {
      auto typedIndex = rewriter.create<riscv::CastOp>(
          load.getLoc(), integer, windowIndex,
          riscv_internal::unselectedLeaf(rewriter));
      auto span = rewriter.create<riscv::ConstantOp>(
          load.getLoc(), integer,
          rewriter.getIntegerAttr(integer, coordinateSpan));
      auto offset = rewriter.create<riscv::BinaryOp>(
          load.getLoc(), integer, typedIndex.getResult(), span.getResult(), "mul",
          riscv_internal::unselectedLeaf(rewriter));
      windowBase = rewriter
                       .create<riscv::BinaryOp>(
                           load.getLoc(), integer, base, offset.getResult(),
                           "add", riscv_internal::unselectedLeaf(rewriter))
                       .getResult();
    } else if (base.getType().isIndex()) {
      auto span = rewriter.create<mlir::arith::ConstantIndexOp>(
          load.getLoc(), coordinateSpan);
      auto offset = rewriter.create<mlir::arith::MulIOp>(
          load.getLoc(), windowIndex, span.getResult());
      windowBase = rewriter
                       .create<mlir::arith::AddIOp>(load.getLoc(), base,
                                                   offset.getResult())
                       .getResult();
    } else {
      return mlir::failure();
    }
    auto cloned = rewriter.create<riscv::RVVUnitEntryWindowLoadOp>(
        load.getLoc(), resultType, load.getSource(), windowBase,
        load.getSourceAxis(), rewriter.getDenseI64ArrayAttr(axes),
        rewriter.getDenseI64ArrayAttr(extents), load.getPayloadAxis(),
        load.getPayloadExtent(), load.getEntryStride(), load.getAccess(),
        riscv_internal::leaf(
            rewriter, "rvv", "unit-entry-window-load",
            "rvv.unit-entry-window-load", "rvv.unit-entry-window-load", 0,
            resultType.getLayout().getRegisterGroups()));
    return remember(cloned.getResult());
  }

  if (auto load =
          mlir::dyn_cast<riscv::RVVBitmaskWindowLoadOp>(definition)) {
    llvm::SmallVector<int64_t> axes(load.getWindowAxes());
    llvm::SmallVector<int64_t> extents(load.getWindowExtents());
    auto found = llvm::find(axes, axis);
    if (found == axes.end() || axes.size() != extents.size())
      return mlir::failure();
    const size_t windowPosition = static_cast<size_t>(found - axes.begin());
    int64_t suffix = 1;
    for (size_t index = windowPosition + 1; index < extents.size(); ++index) {
      if (extents[index] <= 0 ||
          suffix > std::numeric_limits<int64_t>::max() / extents[index])
        return mlir::failure();
      suffix *= extents[index];
    }
    if (windowExtent > std::numeric_limits<int64_t>::max() / suffix)
      return mlir::failure();
    const int64_t windowBits = windowExtent * suffix;
    if (windowBits <= 0 || windowBits % 8)
      return mlir::failure();
    extents[windowPosition] = windowExtent;

    mlir::Value base = load.getByteBase();
    mlir::Value windowBase;
    const int64_t windowBytes = windowBits / 8;
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(base.getType())) {
      auto typedIndex = rewriter.create<riscv::CastOp>(
          load.getLoc(), integer, windowIndex,
          riscv_internal::unselectedLeaf(rewriter));
      auto span = rewriter.create<riscv::ConstantOp>(
          load.getLoc(), integer,
          rewriter.getIntegerAttr(integer, windowBytes));
      auto offset = rewriter.create<riscv::BinaryOp>(
          load.getLoc(), integer, typedIndex.getResult(), span.getResult(),
          "mul", riscv_internal::unselectedLeaf(rewriter));
      windowBase = rewriter
                       .create<riscv::BinaryOp>(
                           load.getLoc(), integer, base, offset.getResult(),
                           "add", riscv_internal::unselectedLeaf(rewriter))
                       .getResult();
    } else if (base.getType().isIndex()) {
      auto span = rewriter.create<mlir::arith::ConstantIndexOp>(
          load.getLoc(), windowBytes);
      auto offset = rewriter.create<mlir::arith::MulIOp>(
          load.getLoc(), windowIndex, span.getResult());
      windowBase = rewriter
                       .create<mlir::arith::AddIOp>(load.getLoc(), base,
                                                   offset.getResult())
                       .getResult();
    } else {
      return mlir::failure();
    }

    auto partBitOffsets =
        riscv::bitmaskWindowPartOffsets(resultType, axes, extents);
    if (!partBitOffsets)
      return mlir::failure();
    auto selected = load.getLeaf();
    const int64_t temporaryGroups = std::max<int64_t>(
        1, (resultType.getLayout().getLmulEighths() + 7) / 8);
    auto cloned = rewriter.create<riscv::RVVBitmaskWindowLoadOp>(
        load.getLoc(), resultType, load.getField(), windowBase,
        load.getSourceAxis(), rewriter.getDenseI64ArrayAttr(axes),
        rewriter.getDenseI64ArrayAttr(extents),
        rewriter.getDenseI64ArrayAttr(*partBitOffsets), load.getAccess(),
        riscv_internal::leaf(
            rewriter, selected.getEngine(), selected.getFamily(),
            selected.getInstruction(), selected.getSpelling(),
            selected.getOperandGroups(),
            resultType.getLayout().getRegisterGroups(), temporaryGroups,
            selected.getFragmentGroups(), selected.getMask(),
            selected.getTail(), selected.getParameters(),
            selected.getLocalBytes()));
    return remember(cloned.getResult());
  }

  if (auto window = mlir::dyn_cast<riscv::RVVStorageWindowOp>(definition)) {
    auto sourcePlan = window.getPlan();
    auto resultPosition = axisPosition(resultType, axis);
    if (!resultPosition)
      return mlir::failure();
    const int64_t lanes =
        resultType.getLayout().getLaneFactors()[*resultPosition];
    if (!sourcePlan || sourcePlan.getReductionAxis() != axis || lanes <= 0 ||
        windowExtent != lanes ||
        resultType.getLayout().getTimeFactors()[*resultPosition] != 1)
      return mlir::failure();

    auto extent = rewriter.create<mlir::arith::ConstantIndexOp>(
        window.getLoc(), windowExtent);
    auto dynamicOffset = rewriter.create<mlir::arith::MulIOp>(
        window.getLoc(), windowIndex, extent.getResult());
    mlir::Value logicalOffset = rewriter.create<mlir::arith::AddIOp>(
        window.getLoc(), window.getLogicalOffset(), dynamicOffset.getResult());
    auto projectedPlan = riscv::StorageWindowPlanAttr::get(
        rewriter.getContext(), sourcePlan.getKind(),
        sourcePlan.getReductionAxis(), sourcePlan.getProjectionBase(),
        sourcePlan.getProjectionStride(), sourcePlan.getProjectionRepeat(),
        windowExtent, lanes, sourcePlan.getRecordElements(),
        sourcePlan.getByteOffset(), sourcePlan.getElementBits(),
        sourcePlan.getGroupSize(), sourcePlan.getLayerSize(),
        sourcePlan.getPhysicalLayerBase(), sourcePlan.getPhysicalLayerStep(),
        sourcePlan.getShiftBase(), sourcePlan.getShiftStep(),
        sourcePlan.getMaskValue());
    auto selected = window.getLeaf();
    auto projected = rewriter.create<riscv::RVVStorageWindowOp>(
        window.getLoc(), resultType, window.getField(), window.getOrigin(),
        logicalOffset, projectedPlan, window.getAccess(),
        riscv_internal::leaf(
            rewriter, selected.getEngine(), selected.getFamily(),
            selected.getInstruction(), selected.getSpelling(),
            selected.getOperandGroups(),
            resultType.getLayout().getRegisterGroups(),
            selected.getTemporaryGroups(), selected.getFragmentGroups(),
            selected.getMask(),
            resultType.getLayout().getValidity() == "tail" ? "agnostic"
                                                            : "exact",
            selected.getParameters(), selected.getLocalBytes()));
    return remember(projected.getResult());
  }

  if (auto load =
          mlir::dyn_cast<riscv::RVVReplicaStorageLoadOp>(definition)) {
    auto field = load.getField().getDefiningOp<riscv::FieldOp>();
    auto sourcePlan = load.getPlan();
    auto issueTimes =
        checkedProduct(resultType.getLayout().getTimeFactors().asArrayRef());
    auto issueReplicas =
        checkedProduct(resultType.getLayout().getReplicaFactors().asArrayRef());
    int64_t issueParts = 0;
    if (!issueTimes || !issueReplicas ||
        !checkedScale(*issueTimes, *issueReplicas, issueParts))
      return mlir::failure();
    if (!field || !sourcePlan || sourcePlan.getKind() != "unit")
      return mlir::failure();

    // A preplanned storage value may be split in issue time along an axis
    // orthogonal to its load-lane axis.  When that split is a translation of
    // one otherwise identical physical window, project it by advancing the
    // scalar base and retaining the selected one-part load.  The storage form
    // and all decode facts remain owned by the original plan.
    if (sourcePlan.getReductionAxis() != axis) {
      auto issuePosition = axisPosition(sourceType, axis);
      auto sourceTimes =
          checkedProduct(sourceType.getLayout().getTimeFactors().asArrayRef());
      auto sourceReplicas = checkedProduct(
          sourceType.getLayout().getReplicaFactors().asArrayRef());
      int64_t sourceParts = 0;
      int64_t representedIssueExtent = 0;
      const int64_t issueCount =
          issuePosition
              ? sourceType.getLayout().getTimeFactors()[*issuePosition]
              : 0;
      const bool oneTimeAxis =
          issuePosition && sourceTimes && sourceReplicas &&
          checkedScale(*sourceTimes, *sourceReplicas, sourceParts) &&
          checkedScale(issueCount, windowExtent, representedIssueExtent) &&
          issueCount > 1 && issueParts == 1 &&
          sourceParts == issueCount &&
          sourceType.getLayout().getLaneFactors()[*issuePosition] ==
              windowExtent &&
          sourceType.getShape()[*issuePosition] == representedIssueExtent &&
          llvm::all_of(sourceType.getLayout().getReplicaFactors().asArrayRef(),
                       [](int64_t factor) { return factor == 1; });
      if (!oneTimeAxis ||
          load.getWindowForPart().size() != static_cast<size_t>(issueCount) ||
          load.getLayerForPart().size() != static_cast<size_t>(issueCount) ||
          load.getPhysicalLayerForPart().size() !=
              static_cast<size_t>(issueCount) ||
          load.getShiftOffsetForPart().size() !=
              static_cast<size_t>(issueCount) ||
          load.getShiftBaseFactorForPart().size() !=
              static_cast<size_t>(issueCount) ||
          load.getMaskValueForPart().size() !=
              static_cast<size_t>(issueCount))
        return mlir::failure();

      const int64_t firstWindow = load.getWindowForPart()[0];
      if (firstWindow < 0 ||
          firstWindow >= static_cast<int64_t>(load.getWindowOffsets().size()))
        return mlir::failure();
      const int64_t firstOffset = load.getWindowOffsets()[firstWindow];
      int64_t issueStride = 0;
      if (issueCount > 1) {
        const int64_t secondWindow = load.getWindowForPart()[1];
        if (secondWindow < 0 ||
            secondWindow >= static_cast<int64_t>(load.getWindowOffsets().size()) ||
            !checkedSubtract(load.getWindowOffsets()[secondWindow], firstOffset,
                             issueStride) ||
            issueStride <= 0)
          return mlir::failure();
      }
      const int64_t recordRank = load.getRecordRank();
      for (int64_t issue = 0; issue < issueCount; ++issue) {
        const int64_t window = load.getWindowForPart()[issue];
        int64_t expectedDelta = 0;
        int64_t actualDelta = 0;
        if (window < 0 ||
            window >= static_cast<int64_t>(load.getWindowOffsets().size()) ||
            !checkedScale(issue, issueStride, expectedDelta) ||
            !checkedSubtract(load.getWindowOffsets()[window], firstOffset,
                             actualDelta) ||
            actualDelta != expectedDelta ||
            load.getLayerForPart()[issue] != load.getLayerForPart()[0] ||
            load.getPhysicalLayerForPart()[issue] !=
                load.getPhysicalLayerForPart()[0] ||
            load.getShiftOffsetForPart()[issue] !=
                load.getShiftOffsetForPart()[0] ||
            load.getShiftBaseFactorForPart()[issue] !=
                load.getShiftBaseFactorForPart()[0] ||
            load.getMaskValueForPart()[issue] !=
                load.getMaskValueForPart()[0])
          return mlir::failure();
        for (int64_t coordinate = 0; coordinate < recordRank; ++coordinate)
          if (load.getRecordCoordinatesForWindow()[static_cast<size_t>(window) *
                                                       recordRank +
                                                   coordinate] !=
              load.getRecordCoordinatesForWindow()[
                  static_cast<size_t>(firstWindow) * recordRank + coordinate])
            return mlir::failure();
      }

      mlir::Value issueBase = load.getLogicalBase();
      const int64_t constantOffset = firstOffset;
      if (auto integer =
              mlir::dyn_cast<mlir::IntegerType>(issueBase.getType())) {
        auto typedIndex = rewriter.create<riscv::CastOp>(
            load.getLoc(), integer, windowIndex,
            riscv_internal::unselectedLeaf(rewriter));
        auto stride = rewriter.create<riscv::ConstantOp>(
            load.getLoc(), integer,
            rewriter.getIntegerAttr(integer, issueStride));
        auto dynamicOffset = rewriter.create<riscv::BinaryOp>(
            load.getLoc(), integer, typedIndex.getResult(), stride.getResult(),
            "mul", riscv_internal::unselectedLeaf(rewriter));
        mlir::Value totalOffset = dynamicOffset.getResult();
        if (constantOffset) {
          auto constant = rewriter.create<riscv::ConstantOp>(
              load.getLoc(), integer,
              rewriter.getIntegerAttr(integer, constantOffset));
          totalOffset = rewriter
                            .create<riscv::BinaryOp>(
                                load.getLoc(), integer, totalOffset,
                                constant.getResult(), "add",
                                riscv_internal::unselectedLeaf(rewriter))
                            .getResult();
        }
        issueBase = rewriter
                        .create<riscv::BinaryOp>(
                            load.getLoc(), integer, issueBase, totalOffset,
                            "add", riscv_internal::unselectedLeaf(rewriter))
                        .getResult();
      } else if (issueBase.getType().isIndex()) {
        auto stride = rewriter.create<mlir::arith::ConstantIndexOp>(
            load.getLoc(), issueStride);
        mlir::Value totalOffset =
            rewriter
                .create<mlir::arith::MulIOp>(load.getLoc(), windowIndex,
                                             stride.getResult())
                .getResult();
        if (constantOffset) {
          auto constant = rewriter.create<mlir::arith::ConstantIndexOp>(
              load.getLoc(), constantOffset);
          totalOffset =
              rewriter
                  .create<mlir::arith::AddIOp>(load.getLoc(), totalOffset,
                                              constant.getResult())
                  .getResult();
        }
        issueBase =
            rewriter
                .create<mlir::arith::AddIOp>(load.getLoc(), issueBase,
                                            totalOffset)
                .getResult();
      } else {
        return mlir::failure();
      }

      int64_t projectedAlignment = sourcePlan.getOffsetAlignment();
      projectedAlignment = std::gcd(projectedAlignment, issueStride);
      if (constantOffset)
        projectedAlignment = std::gcd(projectedAlignment, constantOffset);
      projectedAlignment = std::max<int64_t>(1, projectedAlignment);
      auto projectedPlan = riscv::StorageWindowPlanAttr::get(
          rewriter.getContext(), sourcePlan.getKind(),
          sourcePlan.getReductionAxis(), sourcePlan.getProjectionBase(),
          sourcePlan.getProjectionStride(), sourcePlan.getProjectionRepeat(),
          sourcePlan.getProjectionExtent(), projectedAlignment,
          sourcePlan.getRecordElements(), sourcePlan.getByteOffset(),
          sourcePlan.getElementBits(), sourcePlan.getGroupSize(),
          sourcePlan.getLayerSize(), sourcePlan.getPhysicalLayerBase(),
          sourcePlan.getPhysicalLayerStep(), sourcePlan.getShiftBase(),
          sourcePlan.getShiftStep(), sourcePlan.getMaskValue());
      llvm::SmallVector<int64_t> recordCoordinates;
      for (int64_t coordinate = 0; coordinate < recordRank; ++coordinate)
        recordCoordinates.push_back(load.getRecordCoordinatesForWindow()[
            static_cast<size_t>(firstWindow) * recordRank + coordinate]);
      auto singleton = [&](int64_t value) {
        return rewriter.getDenseI64ArrayAttr({value});
      };
      auto selected = load.getLeaf();
      const int64_t temporaryGroups = std::max<int64_t>(
          1, (resultType.getLayout().getLmulEighths() + 7) / 8);
      auto cloned = rewriter.create<riscv::RVVReplicaStorageLoadOp>(
          load.getLoc(), resultType, load.getField(), issueBase, projectedPlan,
          recordRank, singleton(0),
          rewriter.getDenseI64ArrayAttr(recordCoordinates), singleton(0),
          singleton(load.getLayerForPart()[0]),
          singleton(load.getPhysicalLayerForPart()[0]),
          singleton(load.getShiftOffsetForPart()[0]),
          singleton(load.getShiftBaseFactorForPart()[0]),
          singleton(load.getMaskValueForPart()[0]), load.getAccess(),
          riscv_internal::leaf(
              rewriter, selected.getEngine(), selected.getFamily(),
              selected.getInstruction(), selected.getSpelling(),
              selected.getOperandGroups(),
              resultType.getLayout().getRegisterGroups(), temporaryGroups,
              selected.getFragmentGroups(), selected.getMask(),
              resultType.getLayout().getValidity() == "tail" ? "agnostic"
                                                              : "exact",
              selected.getParameters(), selected.getLocalBytes()));
      return remember(cloned.getResult());
    }

    if (load.getRecordRank() != 0 || issueParts != 1)
      return mlir::failure();

    mlir::Value base = load.getLogicalBase();
    mlir::Value issueBase;
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(base.getType())) {
      auto typedIndex = rewriter.create<riscv::CastOp>(
          load.getLoc(), integer, windowIndex,
          riscv_internal::unselectedLeaf(rewriter));
      auto extent = rewriter.create<riscv::ConstantOp>(
          load.getLoc(), integer,
          rewriter.getIntegerAttr(integer, windowExtent));
      auto offset = rewriter.create<riscv::BinaryOp>(
          load.getLoc(), integer, typedIndex.getResult(), extent.getResult(),
          "mul", riscv_internal::unselectedLeaf(rewriter));
      issueBase = rewriter
                      .create<riscv::BinaryOp>(
                          load.getLoc(), integer, base, offset.getResult(),
                          "add", riscv_internal::unselectedLeaf(rewriter))
                      .getResult();
    } else if (base.getType().isIndex()) {
      auto extent = rewriter.create<mlir::arith::ConstantIndexOp>(
          load.getLoc(), windowExtent);
      auto offset = rewriter.create<mlir::arith::MulIOp>(
          load.getLoc(), windowIndex, extent.getResult());
      issueBase = rewriter
                      .create<mlir::arith::AddIOp>(
                          load.getLoc(), base, offset.getResult())
                      .getResult();
    } else {
      return mlir::failure();
    }

    // The storage-load owner has already selected the physical memory form.
    // Issue-window materialization only projects that closed unit-load plan to
    // one smaller carrier; it must not re-run storage-form selection.
    const int64_t projectedAlignment =
        std::gcd(sourcePlan.getOffsetAlignment(), windowExtent);
    auto projectedPlan = riscv::StorageWindowPlanAttr::get(
        rewriter.getContext(), sourcePlan.getKind(),
        sourcePlan.getReductionAxis(), sourcePlan.getProjectionBase(),
        sourcePlan.getProjectionStride(), sourcePlan.getProjectionRepeat(),
        windowExtent, std::max<int64_t>(1, projectedAlignment),
        sourcePlan.getRecordElements(), sourcePlan.getByteOffset(),
        sourcePlan.getElementBits(), sourcePlan.getGroupSize(),
        sourcePlan.getLayerSize(), sourcePlan.getPhysicalLayerBase(),
        sourcePlan.getPhysicalLayerStep(), sourcePlan.getShiftBase(),
        sourcePlan.getShiftStep(), sourcePlan.getMaskValue());
    auto zero = rewriter.getDenseI64ArrayAttr({0});
    auto empty = rewriter.getDenseI64ArrayAttr({});
    auto selected = load.getLeaf();
    const int64_t temporaryGroups = std::max<int64_t>(
        1, (resultType.getLayout().getLmulEighths() + 7) / 8);
    auto cloned = rewriter.create<riscv::RVVReplicaStorageLoadOp>(
        load.getLoc(), resultType, load.getField(), issueBase, projectedPlan, 0,
        zero, empty, zero, zero, zero, zero, zero, zero, load.getAccess(),
        riscv_internal::leaf(
            rewriter, selected.getEngine(), selected.getFamily(),
            selected.getInstruction(), selected.getSpelling(),
            selected.getOperandGroups(),
            resultType.getLayout().getRegisterGroups(), temporaryGroups,
            selected.getFragmentGroups(), selected.getMask(),
            resultType.getLayout().getValidity() == "tail" ? "agnostic"
                                                           : "exact",
            selected.getParameters(), selected.getLocalBytes()));
    return remember(cloned.getResult());
  }

  if (auto load = mlir::dyn_cast<riscv::RVVIndexedEntryLoadOp>(definition)) {
    auto offsets = cloneOperand(load.getEntryOffsets());
    if (mlir::failed(offsets)) {
      load.emitError(
          "issue-window projection could not project indexed-entry byte offsets; offset_type=")
          << load.getEntryOffsets().getType() << ", offset_def="
          << (load.getEntryOffsets().getDefiningOp()
                  ? load.getEntryOffsets()
                        .getDefiningOp()
                        ->getName()
                        .getStringRef()
                  : llvm::StringRef("<block-argument>"));
      return mlir::failure();
    }
    auto offsetType = mlir::dyn_cast<riscv::ValueType>((*offsets).getType());
    auto selected = load.getLeaf();
    const int64_t operandGroups =
        offsetType && offsetType.getLayout().getCarrier() == "rvv"
            ? offsetType.getLayout().getRegisterGroups()
            : 0;
    auto cloned = rewriter.create<riscv::RVVIndexedEntryLoadOp>(
        load.getLoc(), resultType, load.getSource(), *offsets,
        load.getSourceAxis(), load.getPayloadAxis(), load.getPayloadExtent(),
        load.getEntryByteStride(), load.getAccess(),
        riscv_internal::leaf(
            rewriter, selected.getEngine(), selected.getFamily(),
            selected.getInstruction(), selected.getSpelling(), operandGroups,
            resultType.getLayout().getRegisterGroups(),
            selected.getTemporaryGroups(), selected.getFragmentGroups(),
            selected.getMask(), selected.getTail(), selected.getParameters(),
            selected.getLocalBytes()));
    return remember(cloned.getResult());
  }

  if (auto index =
          mlir::dyn_cast<riscv::RVVRegularRepeatIndexOp>(definition)) {
    auto selected = llvm::find(index.getResults(), value);
    if (selected == index.getResults().end())
      return mlir::failure();
    auto bases = rewriter.getDenseI64ArrayAttr({0});
    auto cloned = rewriter.create<riscv::RVVRegularRepeatIndexOp>(
        index.getLoc(), mlir::TypeRange{resultType}, index.getReductionAxis(),
        index.getRepeat(), rewriter.getArrayAttr({bases}),
        riscv_internal::leaf(
            rewriter, "rvv", "regular-repeat-index",
            (index.getRepeat() & (index.getRepeat() - 1)) == 0
                ? "rvv.regular-repeat-index.pow2"
                : "rvv.regular-repeat-index.div",
            (index.getRepeat() & (index.getRepeat() - 1)) == 0
                ? "rvv.regular-repeat-index.pow2"
                : "rvv.regular-repeat-index.div",
            0, resultType.getLayout().getRegisterGroups(), 1, 0, "none",
            "exact", {static_cast<int64_t>(index.getReductionAxis()),
                       static_cast<int64_t>(index.getRepeat())}));
    return remember(cloned.getResults().front());
  }

  if (auto gather =
          mlir::dyn_cast<riscv::RVVRegularRepeatGatherOp>(definition)) {
    auto selected = llvm::find(gather.getResults(), value);
    if (selected == gather.getResults().end() ||
        !gather.getSourceBase().getType().isIndex())
      return mlir::failure();
    const size_t resultNumber =
        static_cast<size_t>(selected - gather.getResults().begin());
    auto axis = axisPosition(resultType, gather.getReductionAxis());
    const int64_t lanes =
        axis ? resultType.getLayout().getLaneFactors()[*axis] : 0;
    const int64_t repeat = gather.getRepeat();
    if (lanes <= 1 || repeat <= 1 ||
        (lanes % repeat != 0 && repeat % lanes != 0))
      return mlir::failure();
    const int64_t sourceCount = lanes <= repeat ? 1 : lanes / repeat;
    auto stride = rewriter.create<mlir::arith::ConstantIndexOp>(
        gather.getLoc(), sourceCount);
    auto offset = rewriter.create<mlir::arith::MulIOp>(
        gather.getLoc(), windowIndex, stride.getResult());
    auto sourceBase = rewriter.create<mlir::arith::AddIOp>(
        gather.getLoc(), gather.getSourceBase(), offset.getResult());

    llvm::SmallVector<mlir::Value> indices;
    if (lanes > repeat) {
      if (resultNumber >= gather.getIndices().size())
        return mlir::failure();
      auto projectedIndex = cloneOperand(gather.getIndices()[resultNumber]);
      if (mlir::failed(projectedIndex))
        return mlir::failure();
      indices.push_back(*projectedIndex);
    }
    auto bases = rewriter.getDenseI64ArrayAttr({0});
    const bool powerOfTwo = (repeat & (repeat - 1)) == 0;
    llvm::StringRef instruction =
        lanes <= repeat ? "rvv.regular-repeat-broadcast"
                        : powerOfTwo ? "rvv.regular-repeat-gather.pow2"
                                     : "rvv.regular-repeat-gather.div";
    const int64_t temporaryGroups =
        2 * resultType.getLayout().getRegisterGroups();
    auto cloned = rewriter.create<riscv::RVVRegularRepeatGatherOp>(
        gather.getLoc(), mlir::TypeRange{resultType}, gather.getField(), indices,
        gather.getSourceAxis(), gather.getReductionAxis(),
        sourceBase.getResult(), sourceCount, repeat,
        rewriter.getArrayAttr({bases}), gather.getAccess(),
        riscv_internal::leaf(
            rewriter, "rvv", "regular-repeat-gather", instruction,
            instruction,
            gather.getField().getType().getLayout().getRegisterGroups(), 0,
            temporaryGroups, 0, "none", "exact",
            {static_cast<int64_t>(gather.getSourceAxis()),
             static_cast<int64_t>(gather.getReductionAxis()), sourceCount,
             repeat}));
    return remember(cloned.getResults().front());
  }

  if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(definition)) {
    llvm::SmallVector<mlir::Value> indices;
    for (mlir::Value index : extract.getIndices()) {
      auto cloned = cloneOperand(index);
      if (mlir::failed(cloned)) {
        auto diagnostic = extract.emitError(
            "issue-window projection could not project extract coordinates; index=");
        diagnostic << index;
        if (mlir::Operation *producer = index.getDefiningOp())
          diagnostic << ", producer=" << producer->getName().getStringRef();
        return mlir::failure();
      }
      indices.push_back(*cloned);
    }
    auto cloned = rewriter.create<riscv::ExtractOp>(
        extract.getLoc(), resultType, extract.getInput(), indices,
        extract.getSelectors(), extract.getAccess(),
        riscv_internal::unselectedLeaf(rewriter));
    if (auto pattern =
            extract->getAttrOfType<mlir::DenseI64ArrayAttr>("index_pattern"))
      cloned->setAttr("index_pattern", pattern);
    return remember(cloned.getResult());
  }

  mlir::Value cloned;
  if (auto unary = mlir::dyn_cast<riscv::UnaryOp>(definition)) {
    auto input = cloneOperand(unary.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::UnaryOp>(unary.getLoc(), resultType, *input,
                                         unary.getKind(),
                                         riscv_internal::unselectedLeaf(rewriter))
                 .getResult();
  } else if (auto binary = mlir::dyn_cast<riscv::BinaryOp>(definition)) {
    auto lhs = cloneOperand(binary.getLhs());
    auto rhs = cloneOperand(binary.getRhs());
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::BinaryOp>(
                     binary.getLoc(), resultType, *lhs, *rhs, binary.getKind(),
                     riscv_internal::unselectedLeaf(rewriter))
                 .getResult();
  } else if (auto cast = mlir::dyn_cast<riscv::CastOp>(definition)) {
    auto input = cloneOperand(cast.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    auto inputType = mlir::dyn_cast<riscv::ValueType>((*input).getType());
    const bool sameWidth =
        inputType &&
        riscv_internal::logicalBitWidth(inputType.getElementType()) ==
            riscv_internal::logicalBitWidth(resultType.getElementType());
    if (sameWidth && inputType.getLayout() != resultType.getLayout()) {
      auto castType = riscv::ValueType::get(
          rewriter.getContext(), resultType.getElementType(),
          inputType.getShape(), inputType.getAxisIds(), inputType.getLayout());
      auto localCast = rewriter.create<riscv::CastOp>(
          cast.getLoc(), castType, *input,
          riscv_internal::unselectedLeaf(rewriter));
      auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
          cast.getLoc(), resultType, localCast.getResult(),
          riscv_internal::layoutConversion(rewriter, inputType.getLayout(),
                                           resultType.getLayout()),
          riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
      riscv_internal::copyOrigin(cast, localCast);
      riscv_internal::copyOrigin(cast, conversion);
      cloned = conversion.getResult();
    } else {
      cloned = rewriter
                   .create<riscv::CastOp>(
                       cast.getLoc(), resultType, *input,
                       riscv_internal::unselectedLeaf(rewriter))
                   .getResult();
    }
  } else if (auto narrow = mlir::dyn_cast<riscv::NarrowOp>(definition)) {
    auto input = cloneOperand(narrow.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::NarrowOp>(
                     narrow.getLoc(), resultType, *input, narrow.getRounding(),
                     narrow.getSaturate(),
                     riscv_internal::unselectedLeaf(rewriter))
                 .getResult();
  } else if (auto widen = mlir::dyn_cast<riscv::WidenOp>(definition)) {
    auto input = cloneOperand(widen.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::WidenOp>(widen.getLoc(), resultType, *input,
                                         riscv_internal::unselectedLeaf(rewriter))
                 .getResult();
  } else if (auto conversion =
                 mlir::dyn_cast<riscv::ConvertLayoutOp>(definition)) {
    auto input = cloneOperand(conversion.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    if ((*input).getType() == resultType) {
      cloned = *input;
    } else {
      auto inputType = mlir::cast<riscv::ValueType>((*input).getType());
      cloned = rewriter
                   .create<riscv::ConvertLayoutOp>(
                       conversion.getLoc(), resultType, *input,
                       riscv_internal::layoutConversion(
                           rewriter, inputType.getLayout(), resultType.getLayout()),
                       conversion.getSourceAccessAttr(),
                       riscv_internal::unselectedLeaf(rewriter))
                   .getResult();
    }
  } else if (auto broadcast =
                 mlir::dyn_cast<riscv::RVVAxisBroadcastOp>(definition)) {
    auto input = cloneOperand(broadcast.getInput());
    auto inputType = mlir::succeeded(input)
                         ? mlir::dyn_cast<riscv::ValueType>((*input).getType())
                         : riscv::ValueType();
    const int64_t inputLanes =
        inputType ? product(inputType.getLayout().getLaneFactors()) : 0;
    const int64_t resultLanes = product(resultType.getLayout().getLaneFactors());
    if (mlir::failed(input) || !inputType || inputLanes <= 0 ||
        resultLanes < inputLanes)
      return mlir::failure();
    cloned = rewriter
                 .create<riscv::RVVAxisBroadcastOp>(
                     broadcast.getLoc(), resultType, *input,
                     riscv_internal::leaf(
                         rewriter, "rvv", "axis-broadcast",
                         "rvv.axis-broadcast", "rvv.axis-broadcast",
                         inputType.getLayout().getRegisterGroups(),
                         resultType.getLayout().getRegisterGroups(), 1, 0,
                         "none", "exact", {inputLanes, resultLanes}))
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
    definition->emitError("issue-window projection does not support producer ")
        << definition->getName().getStringRef() << "; source=" << sourceType;
    return mlir::failure();
  }
  return remember(cloned);
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
    llvm::ArrayRef<int64_t> reductionAxes) {
  auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
  if (reductionAxes.empty() || !lhsElement || !rhsElement ||
      lhsElement.isSignless() || rhsElement.isSignless() ||
      (!lhsElement.isSigned() && !rhsElement.isSigned()) ||
      std::max<unsigned>(8, lhsElement.getWidth()) !=
          std::max<unsigned>(8, rhsElement.getWidth()) ||
      lhs.getLayout().getCarrier() != "rvv" ||
      rhs.getLayout().getCarrier() != "rvv" ||
      lhs.getLayout().getSew() != rhs.getLayout().getSew() ||
      lhs.getLayout().getLmulEighths() != rhs.getLayout().getLmulEighths() ||
      lhs.getLayout().getVl() != rhs.getLayout().getVl())
    return {};

  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> replicas;
  auto appendFree = [&](riscv::ValueType operand) -> bool {
    for (size_t position = 0; position < operand.getAxisIds().size(); ++position) {
      const int64_t axis = operand.getAxisIds()[position];
      if (llvm::is_contained(reductionAxes, axis))
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

  llvm::SmallVector<int64_t> reductionLanes;
  int64_t totalReductionLanes = 1;
  for (int64_t reductionAxis : reductionAxes) {
    auto lhsReduction = axisPosition(lhs, reductionAxis);
    auto rhsReduction = axisPosition(rhs, reductionAxis);
    if (!lhsReduction || !rhsReduction)
      return {};
    const int64_t lanes =
        lhs.getLayout().getLaneFactors()[*lhsReduction];
    if (lanes <= 0 ||
        lanes != rhs.getLayout().getLaneFactors()[*rhsReduction] ||
        lhs.getShape()[*lhsReduction] != rhs.getShape()[*rhsReduction] ||
        lhs.getLayout().getTimeFactors()[*lhsReduction] != 1 ||
        rhs.getLayout().getTimeFactors()[*rhsReduction] != 1 ||
        lhs.getLayout().getReplicaFactors()[*lhsReduction] != 1 ||
        rhs.getLayout().getReplicaFactors()[*rhsReduction] != 1 ||
        lhs.getLayout().getFragmentFactors()[*lhsReduction] != 1 ||
        rhs.getLayout().getFragmentFactors()[*rhsReduction] != 1 ||
        lhs.getLayout().getLocalFactors()[*lhsReduction] != 1 ||
        rhs.getLayout().getLocalFactors()[*rhsReduction] != 1 ||
        totalReductionLanes > std::numeric_limits<int64_t>::max() / lanes)
      return {};
    axes.push_back(reductionAxis);
    shape.push_back(lhs.getShape()[*lhsReduction]);
    replicas.push_back(1);
    reductionLanes.push_back(lanes);
    totalReductionLanes *= lanes;
  }
  llvm::SmallVector<int64_t> time(axes.size(), 1);
  llvm::SmallVector<int64_t> lane(axes.size(), 1);
  llvm::SmallVector<int64_t> one(axes.size(), 1);
  for (size_t position = 0; position < reductionLanes.size(); ++position)
    lane[lane.size() - reductionLanes.size() + position] =
        reductionLanes[position];
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
      totalReductionLanes, groupsPerVector * replicaCount,
      lhs.getLayout().getValidity());
  auto element = mlir::IntegerType::get(builder.getContext(), partialWidth,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), element,
                               riscv_internal::integers(builder, shape), axisIds,
                               layout);
}

riscv::ValueType replicaReducedAccumulatorType(
    mlir::Builder &builder, riscv::ValueType lhs, riscv::ValueType rhs,
    int64_t reductionAxis) {
  return replicaReducedAccumulatorType(
      builder, lhs, rhs, llvm::ArrayRef<int64_t>(reductionAxis));
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

riscv::ValueType issueSliceType(mlir::Builder &builder,
                                riscv::ValueType source,
                                llvm::ArrayRef<int64_t> reductionAxes) {
  if (reductionAxes.empty() || source.getLayout().getCarrier() != "rvv")
    return {};
  llvm::SmallVector<int64_t> shape(source.getShape().asArrayRef());
  llvm::SmallVector<int64_t> timeFactors(
      source.getLayout().getTimeFactors().asArrayRef());
  bool sliced = false;
  for (int64_t reductionAxis : reductionAxes) {
    auto position = axisPosition(source, reductionAxis);
    if (!position)
      return {};
    const int64_t time = source.getLayout().getTimeFactors()[*position];
    const int64_t lane = source.getLayout().getLaneFactors()[*position];
    const int64_t replica = source.getLayout().getReplicaFactors()[*position];
    if (time <= 0 || lane <= 0 || replica != 1 ||
        source.getShape()[*position] != time * lane ||
        source.getLayout().getFragmentFactors()[*position] != 1 ||
        source.getLayout().getLocalFactors()[*position] != 1)
      return {};
    if (time > 1) {
      shape[*position] = lane;
      timeFactors[*position] = 1;
      sliced = true;
    }
  }
  if (!sliced)
    return {};
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", source.getAxisIds(),
      builder.getDenseI64ArrayAttr(timeFactors),
      source.getLayout().getLaneFactors(),
      source.getLayout().getReplicaFactors(),
      source.getLayout().getFragmentFactors(),
      source.getLayout().getLocalFactors(), source.getLayout().getSew(),
      source.getLayout().getLmulEighths(), source.getLayout().getVl(),
      source.getLayout().getRegisterGroups(),
      source.getLayout().getValidity());
  return riscv::ValueType::get(builder.getContext(), source.getElementType(),
                               builder.getDenseI64ArrayAttr(shape),
                               source.getAxisIds(), layout);
}

riscv::ValueType sequentialIssueType(mlir::Builder &builder,
                                     riscv::RVVWidenDotOp dot,
                                     riscv::ValueType source) {
  if (dot.getOver().empty() || dot.getReductionLanes() <= 0 ||
      dot.getSliceLmulEighths() <= 0 ||
      source.getLayout().getCarrier() != "rvv" ||
      source.getLayout().getLmulEighths() < dot.getSliceLmulEighths() ||
      source.getLayout().getLmulEighths() % dot.getSliceLmulEighths())
    return {};
  llvm::SmallVector<int64_t> shape;
  llvm::SmallVector<int64_t> lanes;
  int64_t laneProduct = 1;
  for (int64_t reductionAxis : dot.getOver()) {
    auto position = axisPosition(source, reductionAxis);
    if (!position)
      return {};
    const int64_t time = source.getLayout().getTimeFactors()[*position];
    const int64_t lane = source.getLayout().getLaneFactors()[*position];
    if (time <= 0 || lane <= 0 ||
        source.getShape()[*position] != time * lane ||
        source.getLayout().getReplicaFactors()[*position] != 1 ||
        source.getLayout().getFragmentFactors()[*position] != 1 ||
        source.getLayout().getLocalFactors()[*position] != 1 ||
        laneProduct > dot.getReductionLanes() / lane)
      return {};
    shape.push_back(lane);
    lanes.push_back(lane);
    laneProduct *= lane;
  }
  for (size_t position = 0; position < source.getAxisIds().size(); ++position)
    if (!llvm::is_contained(dot.getOver(), source.getAxisIds()[position]) &&
        (source.getLayout().getFragmentFactors()[position] != 1 ||
         source.getLayout().getLocalFactors()[position] != 1))
      return {};
  if (laneProduct != dot.getReductionLanes())
    return {};
  llvm::SmallVector<int64_t> one(dot.getOver().size(), 1);
  auto axes = builder.getDenseI64ArrayAttr(dot.getOver());
  auto oneAttr = builder.getDenseI64ArrayAttr(one);
  auto laneAttr = builder.getDenseI64ArrayAttr(lanes);
  const int64_t groups =
      std::max<int64_t>(1, (dot.getSliceLmulEighths() + 7) / 8);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axes, oneAttr, laneAttr, oneAttr, oneAttr,
      oneAttr, source.getLayout().getSew(), dot.getSliceLmulEighths(),
      dot.getReductionLanes(), groups, source.getLayout().getValidity());
  return riscv::ValueType::get(builder.getContext(), source.getElementType(),
                               builder.getDenseI64ArrayAttr(shape), axes, layout);
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
  auto lhsSliceLMUL = riscv::rvvLaneSliceLMULEighths(lhs, lanes);
  auto rhsSliceLMUL = riscv::rvvLaneSliceLMULEighths(rhs, lanes);
  if (!lhsSliceLMUL || !rhsSliceLMUL || *lhsSliceLMUL != *rhsSliceLMUL ||
      *lhsSliceLMUL > std::numeric_limits<int64_t>::max() / 2)
    return {};
  const int64_t partialLMUL = 2 * *lhsSliceLMUL;
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

std::optional<std::string>
sequentialMultiplyInstruction(riscv::ValueType lhs, riscv::ValueType rhs) {
  if (!lhs || !rhs)
    return std::nullopt;
  auto lhsElement =
      mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
  auto rhsElement =
      mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
  if (!lhsElement || !rhsElement || lhsElement.isSignless() ||
      rhsElement.isSignless())
    return std::nullopt;
  if (lhsElement.isSigned() && rhsElement.isSigned())
    return std::string("rvv.vwmul.vv");
  if (lhsElement.isSigned())
    return std::string("rvv.vwmulsu.vv");
  if (rhsElement.isSigned())
    return std::string("rvv.vwmulsu.vv.swap");
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
  auto lanes = riscv::rvvLaneCount(partial);
  if (!lanes)
    return std::nullopt;
  return *lanes == 1
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
      result.getWidth() != 32)
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
                riscv::RVVUnitEntryWindowLoadOp,
                riscv::RVVIndexedEntryLoadOp, riscv::IotaOp,
                riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                riscv::NarrowOp, riscv::WidenOp,
                riscv::RVVWidenMultiplyOp,
                riscv::RVVWidenScalarMultiplyOp, riscv::RVVWidenDotOp,
                riscv::RVVRegularRepeatIndexOp,
                riscv::RVVRegularRepeatGatherOp,
                riscv::ConvertLayoutOp,
                riscv::RegisterMaterializeOp>(definition)) {
    rewriter.eraseOp(definition);
    for (mlir::Value operand : operands)
      eraseDeadChain(operand, stops, visited, rewriter);
  }
}

bool isDeadChainCandidate(mlir::Operation *operation) {
  return mlir::isa<riscv::RVVLayeredStreamOp,
                   riscv::RVVProjectedLayeredStreamOp, riscv::ExtractOp,
                   riscv::RVVUnitEntryWindowLoadOp,
                   riscv::RVVIndexedEntryLoadOp, riscv::IotaOp,
                   riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                   riscv::NarrowOp, riscv::WidenOp,
                   riscv::ReduceOp,
                   riscv::RVVWidenMultiplyOp,
                   riscv::RVVWidenScalarMultiplyOp, riscv::RVVWidenDotOp,
                   riscv::ConvertLayoutOp, riscv::RVVAxisBroadcastOp,
                   riscv::RVVRegularRepeatIndexOp,
                   riscv::RVVRegularRepeatGatherOp,
                   riscv::RegisterMaterializeOp>(operation);
}

void collectDeadChainCandidates(
    mlir::Value value, const llvm::DenseSet<mlir::Value> &stops,
    llvm::DenseSet<mlir::Operation *> &candidates) {
  if (stops.contains(value))
    return;
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !isDeadChainCandidate(definition) ||
      !candidates.insert(definition).second)
    return;
  for (mlir::Value operand : definition->getOperands())
    collectDeadChainCandidates(operand, stops, candidates);
}

void sweepDeadChainCandidates(
    llvm::DenseSet<mlir::Operation *> &candidates,
    mlir::IRRewriter &rewriter) {
  while (true) {
    mlir::Operation *dead = nullptr;
    for (mlir::Operation *candidate : candidates) {
      if (llvm::all_of(candidate->getResults(),
                       [](mlir::Value result) { return result.use_empty(); })) {
        dead = candidate;
        break;
      }
    }
    if (!dead)
      return;
    candidates.erase(dead);
    rewriter.eraseOp(dead);
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
  if (auto gather =
          mlir::dyn_cast_or_null<riscv::RVVRegularRepeatGatherOp>(producer)) {
    auto targetType = scalarReplicaType(rewriter, sourceType);
    auto replicas =
        targetType
            ? riscv_internal::staticProduct(
                  targetType.getLayout().getReplicaFactors().asArrayRef())
            : std::optional<int64_t>();
    auto bases = gather.getPartBases().size() == 1
                     ? mlir::dyn_cast<mlir::DenseI64ArrayAttr>(
                           gather.getPartBases()[0])
                     : mlir::DenseI64ArrayAttr();
    if (!targetType || !replicas || gather.getResults().size() != 1 ||
        gather.getIndices().size() != 1 || !bases || bases.size() != 1 ||
        bases[0] != 0 || *replicas != gather.getSourceCount() * gather.getRepeat())
      return mlir::failure();
    llvm::SmallVector<int64_t> partBases;
    partBases.reserve(*replicas);
    for (int64_t replica = 0; replica < *replicas; ++replica)
      partBases.push_back(replica / gather.getRepeat());
    auto scalarLoad = rewriter.create<riscv::RVVRegularRepeatScalarLoadOp>(
        gather.getLoc(), targetType, gather.getField(), gather.getSourceAxis(),
        gather.getReductionAxis(), gather.getSourceBase(),
        gather.getSourceCount(), gather.getRepeat(),
        rewriter.getDenseI64ArrayAttr(partBases), gather.getAccess(),
        riscv_internal::leaf(
            rewriter, "scalar", "regular-repeat-scalar-load",
            "scalar.regular-repeat-load", "scalar.regular-repeat-load", 0, 0,
            0, 0, "none", "exact",
            {static_cast<int64_t>(gather.getSourceAxis()),
             static_cast<int64_t>(gather.getReductionAxis()),
             static_cast<int64_t>(gather.getSourceCount()),
             static_cast<int64_t>(gather.getRepeat())}));
    riscv_internal::copyOrigin(gather, scalarLoad);
    memo.try_emplace(value, scalarLoad.getResult());
    return scalarLoad.getResult();
  }
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

bool supportsScalarReplicaRematerialization(
    mlir::Value value, llvm::DenseSet<mlir::Operation *> &visited) {
  auto sourceType = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!sourceType)
    return true;
  bool alreadyScalar = sourceType.getLayout().getCarrier() == "scalar";
  for (int64_t factor :
       sourceType.getLayout().getTimeFactors().asArrayRef())
    alreadyScalar &= factor == 1;
  for (int64_t factor :
       sourceType.getLayout().getLaneFactors().asArrayRef())
    alreadyScalar &= factor == 1;
  if (alreadyScalar)
    return true;

  mlir::Operation *producer = value.getDefiningOp();
  if (!producer || !visited.insert(producer).second)
    return false;
  if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(producer))
    return conversion.getConversion().getEffect() == "pure" &&
           supportsScalarReplicaRematerialization(conversion.getInput(),
                                                  visited);
  if (auto gather = mlir::dyn_cast<riscv::RVVRegularRepeatGatherOp>(producer)) {
    auto bases = gather.getPartBases().size() == 1
                     ? mlir::dyn_cast<mlir::DenseI64ArrayAttr>(
                           gather.getPartBases()[0])
                     : mlir::DenseI64ArrayAttr();
    return gather.getResults().size() == 1 && gather.getIndices().size() == 1 &&
           bases && bases.size() == 1 && bases[0] == 0;
  }
  if (!mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp,
                 riscv::CastOp, riscv::NarrowOp, riscv::WidenOp>(producer) ||
      producer->getNumResults() != 1)
    return false;
  return llvm::all_of(producer->getOperands(), [&](mlir::Value operand) {
    return supportsScalarReplicaRematerialization(operand, visited);
  });
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

riscv::ValueType plannedNarrowScaleType(mlir::Builder &builder,
                                        mlir::Value value,
                                        riscv::ValueType scalarScaleType) {
  if (!scalarScaleType ||
      scalarScaleType.getLayout().getCarrier() != "scalar")
    return {};
  unsigned width = 0;
  if (auto existing = narrowScaleSource(value)) {
    auto existingType = mlir::dyn_cast<riscv::ValueType>(existing->getType());
    auto element = existingType
                       ? mlir::dyn_cast<mlir::IntegerType>(
                             existingType.getElementType())
                       : mlir::IntegerType();
    if (element && element.isSigned() && element.getWidth() <= 16)
      width = element.getWidth();
  } else if (auto range = riscv_internal::integerRange(value);
             range && range->minimum >= std::numeric_limits<int16_t>::min() &&
             range->maximum <= std::numeric_limits<int16_t>::max()) {
    width = 16;
  }
  if (!width)
    return {};
  auto element = mlir::IntegerType::get(builder.getContext(), width,
                                        mlir::IntegerType::Signed);
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "scalar", scalarScaleType.getAxisIds(),
      scalarScaleType.getLayout().getTimeFactors(),
      scalarScaleType.getLayout().getLaneFactors(),
      scalarScaleType.getLayout().getReplicaFactors(),
      scalarScaleType.getLayout().getFragmentFactors(),
      scalarScaleType.getLayout().getLocalFactors(),
      std::max<unsigned>(8, width), 0, 1, 0,
      scalarScaleType.getLayout().getValidity());
  return riscv::ValueType::get(builder.getContext(), element,
                               scalarScaleType.getShape(),
                               scalarScaleType.getAxisIds(), layout);
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
    auto scalarInteger = mlir::dyn_cast<mlir::IntegerType>(scale.getType());
    auto scaleParts = scaleType
                          ? riscv_internal::staticProduct(
                                scaleType.getLayout()
                                    .getReplicaFactors()
                                    .asArrayRef())
                          : scalarInteger && scalarInteger.isSigned() &&
                                    scalarInteger.getWidth() == 32
                                ? std::optional<int64_t>(1)
                                : std::optional<int64_t>();
    auto scaleElement =
        scaleType
            ? mlir::dyn_cast<mlir::IntegerType>(scaleType.getElementType())
            : scalarInteger;
    const bool scalarCarrier =
        scaleType ? scaleType.getLayout().getCarrier() == "scalar"
                  : static_cast<bool>(scalarInteger);
    if (!scalarCarrier ||
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

std::optional<ScaledPartialContribution>
matchLevelScaledLoopSeed(mlir::scf::ForOp loop) {
  if (loop.getInitArgs().size() != 1 || loop.getNumResults() != 1)
    return std::nullopt;
  auto yield =
      mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return std::nullopt;
  auto contribution = matchScaledPartialContribution(yield.getOperand(0));
  if (!contribution || contribution->previous != loop.getRegionIterArg(0))
    return std::nullopt;
  auto factor = loop->getAttrOfType<mlir::IntegerAttr>(
      "weft.riscv.unroll_factor");
  if (!factor || factor.getInt() != contribution->dot.getPartialUnroll() ||
      factor.getInt() <= 1)
    return std::nullopt;
  return contribution;
}

mlir::FailureOr<mlir::scf::ForOp>
expandPlannedLevelScaledIssueLoop(mlir::scf::ForOp loop, int64_t factor,
                                  mlir::IRRewriter &rewriter) {
  auto lower = loop.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
  auto upper = loop.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
  auto step = loop.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (!lower || !upper || !step || factor <= 1 || step.value() <= 0 ||
      upper.value() < lower.value())
    return mlir::failure();
  const int64_t distance = upper.value() - lower.value();
  if (distance % step.value())
    return mlir::failure();
  const int64_t iterations = distance / step.value();
  if (iterations <= 0 || iterations % factor)
    return mlir::failure();

  loop->removeAttr("weft.riscv.unroll_factor");
  loop->removeAttr("weft.riscv.unroll_order");

  // The generic SCF unroller erases the loop when the selected factor equals
  // its exact trip count.  A level-scaled partial plan still needs one region
  // to own the cloned issue cohort until the typed PartialSet program replaces
  // it.  Keep that owner as a single-iteration loop and clone the remaining
  // issues into its body, threading the original iter_args mechanically.
  if (iterations == factor) {
    auto yield = mlir::dyn_cast<mlir::scf::YieldOp>(
        loop.getBody()->getTerminator());
    if (!yield || yield.getNumOperands() != loop.getInitArgs().size())
      return mlir::failure();

    llvm::SmallVector<mlir::Operation *> templateOperations;
    for (mlir::Operation &operation : loop.getBody()->without_terminator())
      templateOperations.push_back(&operation);
    llvm::SmallVector<mlir::Value> current(yield.getOperands().begin(),
                                           yield.getOperands().end());
    rewriter.setInsertionPoint(yield);
    for (int64_t iteration = 1; iteration < factor; ++iteration) {
      mlir::IRMapping mapping;
      auto induction = rewriter.create<mlir::arith::ConstantIndexOp>(
          loop.getLoc(), lower.value() + iteration * step.value());
      mapping.map(loop.getInductionVar(), induction.getResult());
      for (auto [regionArgument, value] :
           llvm::zip(loop.getRegionIterArgs(), current))
        mapping.map(regionArgument, value);
      for (mlir::Operation *operation : templateOperations)
        rewriter.clone(*operation, mapping);

      llvm::SmallVector<mlir::Value> next;
      next.reserve(yield.getNumOperands());
      for (mlir::Value value : yield.getOperands())
        next.push_back(mapping.lookupOrDefault(value));
      current = std::move(next);
    }
    rewriter.setInsertionPointToStart(loop.getBody());
    auto firstInduction = rewriter.create<mlir::arith::ConstantIndexOp>(
        loop.getLoc(), lower.value());
    loop.getInductionVar().replaceAllUsesWith(firstInduction.getResult());
    rewriter.modifyOpInPlace(yield, [&] { yield->setOperands(current); });
    rewriter.setInsertionPoint(loop);
    auto singleIterationUpper = rewriter.create<mlir::arith::ConstantIndexOp>(
        loop.getLoc(), lower.value() + step.value());
    rewriter.modifyOpInPlace(loop, [&] {
      loop.getUpperBoundMutable().assign(singleIterationUpper.getResult());
    });
    return loop;
  }

  auto unrolled = mlir::loopUnrollByFactor(loop, factor);
  if (mlir::failed(unrolled) || !(*unrolled).mainLoopOp)
    return mlir::failure();
  return *(*unrolled).mainLoopOp;
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
  riscv::ValueType lhsIssueType;
  riscv::ValueType rhsIssueType;
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
  facts.lhsIssueType =
      projectOneWindow(builder, facts.lhsType, facts.reductionAxis);
  facts.rhsIssueType =
      projectOneWindow(builder, facts.rhsType, facts.reductionAxis);
  facts.accumulatorType = partialType(builder, facts.lhsType,
                                      dot.getResult().getType(),
                                      facts.reductionAxis);
  auto expectedAccumulator =
      facts.lhsIssueType && facts.rhsIssueType
          ? replicaReducedAccumulatorType(builder, facts.lhsIssueType,
                                          facts.rhsIssueType,
                                          facts.reductionAxis)
          : riscv::ValueType();
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
  if (!completeRoots || !completeSlice || !facts.lhsIssueType ||
      !facts.rhsIssueType || !facts.accumulatorType ||
      facts.accumulatorType != expectedAccumulator ||
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

bool hasGappedIndexedWindowRoot(mlir::Value value, int64_t windowAxis,
                                llvm::DenseSet<mlir::Operation *> &visited) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return false;
  if (auto load = mlir::dyn_cast<riscv::RVVIndexedEntryLoadOp>(definition)) {
    auto result = load.getResult().getType();
    auto position = axisPosition(result, windowAxis);
    const unsigned elementBits =
        riscv_internal::logicalBitWidth(result.getElementType());
    const int64_t payloadBytes =
        elementBits && elementBits % 8 == 0
            ? load.getPayloadExtent() * static_cast<int64_t>(elementBits / 8)
            : 0;
    if (position && payloadBytes > 0 &&
        load.getEntryByteStride() > payloadBytes &&
        result.getLayout().getTimeFactors()[*position] > 0 &&
        result.getLayout().getLaneFactors()[*position] > 1 &&
        result.getLayout().getReplicaFactors()[*position] == 1 &&
        result.getLayout().getFragmentFactors()[*position] == 1 &&
        result.getLayout().getLocalFactors()[*position] == 1)
      return true;
  }
  if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(definition)) {
    auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
    if (result && extract.getIndices().size() == 1 &&
        riscv_internal::sourceField(extract.getInput())) {
      auto relation = riscv_internal::analyzeIndexedEntryRelation(
          extract.getIndices()[0], result, {}, {});
      auto position = axisPosition(result, windowAxis);
      if (relation && position &&
          relation->entryStride > relation->payloadExtent &&
          result.getLayout().getTimeFactors()[*position] > 0 &&
          result.getLayout().getLaneFactors()[*position] > 1 &&
          result.getLayout().getReplicaFactors()[*position] == 1 &&
          result.getLayout().getFragmentFactors()[*position] == 1 &&
          result.getLayout().getLocalFactors()[*position] == 1)
        return true;
    }
  }
  if (!mayMoveReadAcross(definition))
    return false;
  return llvm::any_of(definition->getOperands(), [&](mlir::Value operand) {
    return hasGappedIndexedWindowRoot(operand, windowAxis, visited);
  });
}

bool hasGappedIndexedWindowRoot(mlir::Value value, int64_t windowAxis) {
  llvm::DenseSet<mlir::Operation *> visited;
  return hasGappedIndexedWindowRoot(value, windowAxis, visited);
}

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

std::optional<riscv::NestedPartialPlanAttr> planNestedPartialCarrier(
    mlir::Builder &builder, riscv::RVVWidenDotOp dot,
    const ReplicaScaledDotReduction &match, int64_t windowAxis,
    int64_t issueStreams, int64_t windowExtent) {
  auto kernel = dot->getParentOfType<riscv::KernelOp>();
  auto dotResult = mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
  mlir::Value finalResult = match.reductions.empty()
                                ? mlir::Value()
                                : match.reductions.front()->getResult(0);
  mlir::Type finalType = finalResult ? finalResult.getType() : mlir::Type();
  auto finalValue = mlir::dyn_cast<riscv::ValueType>(finalType);
  llvm::SmallVector<int64_t> outputAxes =
      finalValue
          ? llvm::SmallVector<int64_t>(finalValue.getAxisIds().asArrayRef())
          : llvm::SmallVector<int64_t>();
  auto outputReplicaCount =
      finalValue ? riscv_internal::staticProduct(
                       finalValue.getLayout().getReplicaFactors().asArrayRef())
                 : std::optional<int64_t>(1);
  const bool terminalOutputCohort =
      finalResult && outputReplicaCount && *outputReplicaCount > 1 &&
      !reachesReductionThroughPureOps(finalResult);
  auto scaleType = mlir::dyn_cast<riscv::ValueType>(match.scale.getType());
  const bool finalCarrierClosed =
      !finalValue ||
      (finalValue.getLayout().getCarrier() == "scalar" &&
       llvm::all_of(finalValue.getLayout().getTimeFactors().asArrayRef(),
                    [](int64_t factor) { return factor == 1; }) &&
       llvm::all_of(finalValue.getLayout().getLaneFactors().asArrayRef(),
                    [](int64_t factor) { return factor == 1; }) &&
       llvm::all_of(finalValue.getLayout().getFragmentFactors().asArrayRef(),
                    [](int64_t factor) { return factor == 1; }) &&
       llvm::all_of(finalValue.getLayout().getLocalFactors().asArrayRef(),
                    [](int64_t factor) { return factor == 1; }));
  if (!kernel || !dotResult || dot.getOver().empty() || issueStreams <= 0 ||
      windowExtent <= 0 || !scaleType || !outputReplicaCount ||
      *outputReplicaCount <= 0 || !finalCarrierClosed)
    return std::nullopt;

  auto coalesceReductionTime = [&](riscv::ValueType source) {
    if (!source || source.getLayout().getCarrier() != "rvv")
      return riscv::ValueType();
    llvm::SmallVector<int64_t> time(
        source.getLayout().getTimeFactors().asArrayRef());
    llvm::SmallVector<int64_t> lane(
        source.getLayout().getLaneFactors().asArrayRef());
    auto replica = source.getLayout().getReplicaFactors();
    auto fragment = source.getLayout().getFragmentFactors();
    auto local = source.getLayout().getLocalFactors();
    auto oldLanes = riscv_internal::staticProduct(lane);
    if (!oldLanes || *oldLanes <= 0)
      return riscv::ValueType();
    for (int64_t axis : dot.getOver()) {
      auto position = axisPosition(source, axis);
      if (!position || time[*position] <= 0 || lane[*position] <= 0 ||
          replica[*position] != 1 || fragment[*position] != 1 ||
          local[*position] != 1 ||
          lane[*position] >
              std::numeric_limits<int64_t>::max() / time[*position])
        return riscv::ValueType();
      lane[*position] *= time[*position];
      time[*position] = 1;
    }
    auto newLanes = riscv_internal::staticProduct(lane);
    auto replicas = riscv_internal::staticProduct(replica.asArrayRef());
    if (!newLanes || !replicas || *newLanes <= 0 || *replicas <= 0 ||
        source.getLayout().getLmulEighths() <= 0 ||
        source.getLayout().getLmulEighths() >
            std::numeric_limits<int64_t>::max() / *newLanes)
      return riscv::ValueType();
    const int64_t scaled =
        source.getLayout().getLmulEighths() * *newLanes;
    const int64_t requested = scaled / *oldLanes + (scaled % *oldLanes != 0);
    int64_t selectedLMUL = 0;
    for (int64_t legal : kernel.getTarget().getLegalLMULEighths().asArrayRef())
      if (legal >= requested &&
          (selectedLMUL == 0 || legal < selectedLMUL))
        selectedLMUL = legal;
    if (selectedLMUL <= 0)
      return riscv::ValueType();
    auto layout = riscv::LayoutAttr::get(
        builder.getContext(), "rvv", source.getAxisIds(),
        riscv_internal::integers(builder, time),
        riscv_internal::integers(builder, lane), replica, fragment, local,
        source.getLayout().getSew(), selectedLMUL, *newLanes,
        std::max<int64_t>(1, (selectedLMUL + 7) / 8) * *replicas,
        source.getLayout().getValidity());
    if (!riscv::supportsRVVLayout(kernel.getTarget(), layout))
      return riscv::ValueType();
    return riscv::ValueType::get(builder.getContext(), source.getElementType(),
                                 source.getShape(), source.getAxisIds(), layout);
  };

  auto issueLhs = issueWindowType(builder, kernel.getTarget(),
                                  dot.getLhs().getType(), windowAxis,
                                  windowExtent);
  auto issueRhs = issueWindowType(builder, kernel.getTarget(),
                                  dot.getRhs().getType(), windowAxis,
                                  windowExtent);
  // A multi-output contraction can amortize one complete reduction carrier
  // across its output replicas.  For a scalar result the existing nested
  // topology is already closed; widening it here would rewrite unrelated
  // index/storage producers without reducing consumer work.
  if (terminalOutputCohort) {
    issueLhs = coalesceReductionTime(issueLhs);
    issueRhs = coalesceReductionTime(issueRhs);
  }
  auto issueScale = issueWindowType(builder, kernel.getTarget(), scaleType,
                                    windowAxis, windowExtent);
  auto scaleReplicaType =
      issueScale ? scalarReplicaType(builder, issueScale) : riscv::ValueType();
  auto scaleElement =
      scaleReplicaType
          ? mlir::dyn_cast<mlir::IntegerType>(scaleReplicaType.getElementType())
          : mlir::IntegerType();
  auto lhsLanes = issueLhs ? riscv_internal::staticProduct(
                                 issueLhs.getLayout().getLaneFactors().asArrayRef())
                           : std::optional<int64_t>();
  auto rhsLanes = issueRhs ? riscv_internal::staticProduct(
                                 issueRhs.getLayout().getLaneFactors().asArrayRef())
                           : std::optional<int64_t>();
  auto lhsTime = issueLhs ? riscv_internal::staticProduct(
                                issueLhs.getLayout().getTimeFactors().asArrayRef())
                          : std::optional<int64_t>();
  auto rhsTime = issueRhs ? riscv_internal::staticProduct(
                                issueRhs.getLayout().getTimeFactors().asArrayRef())
                          : std::optional<int64_t>();
  auto lhsReplicas =
      issueLhs ? riscv_internal::staticProduct(
                     issueLhs.getLayout().getReplicaFactors().asArrayRef())
               : std::optional<int64_t>();
  auto rhsReplicas =
      issueRhs ? riscv_internal::staticProduct(
                     issueRhs.getLayout().getReplicaFactors().asArrayRef())
               : std::optional<int64_t>();
  llvm::SmallVector<int64_t> partialAxes(match.reducedAxes);
  llvm::SmallVector<int64_t> partialAxisExtents;
  int64_t partialSlots = 1;
  for (int64_t axis : partialAxes) {
    auto lhsExtent = laneProductForAxes(issueLhs, {axis});
    auto rhsExtent = laneProductForAxes(issueRhs, {axis});
    if (!lhsExtent || !rhsExtent || *lhsExtent <= 0 ||
        *lhsExtent != *rhsExtent ||
        partialSlots > std::numeric_limits<int64_t>::max() / *lhsExtent) {
      partialSlots = 0;
      break;
    }
    partialAxisExtents.push_back(*lhsExtent);
    partialSlots *= *lhsExtent;
  }
  const int64_t sourceLanes =
      dot.getReductionLanes() * partialSlots;
  if (!issueLhs || !issueRhs || !issueScale || !scaleReplicaType || !scaleElement ||
      !scaleElement.isSigned() || scaleElement.getWidth() != 32 || !lhsLanes ||
      !rhsLanes || !lhsTime || !rhsTime || !lhsReplicas || !rhsReplicas ||
      partialSlots < windowExtent || partialSlots % windowExtent ||
      *lhsLanes != sourceLanes || *rhsLanes != sourceLanes || *lhsTime != 1 ||
      *rhsTime != 1 || *lhsReplicas <= 0 || *rhsReplicas <= 0) {
    return std::nullopt;
  }

  llvm::SmallVector<int64_t> lhsSourceParts;
  llvm::SmallVector<int64_t> rhsSourceParts;
  llvm::SmallVector<int64_t> lhsLaneOffsets;
  llvm::SmallVector<int64_t> rhsLaneOffsets;
  llvm::SmallVector<int64_t> scaleReplicaParts;
  llvm::SmallVector<int64_t> windowAxes(partialAxes);
  if (!terminalOutputCohort) {
    auto availableScales = riscv_internal::staticProduct(
        scaleReplicaType.getLayout().getReplicaFactors().asArrayRef());
    if (*outputReplicaCount != 1 || !availableScales ||
        *availableScales < partialSlots)
      return std::nullopt;
    lhsSourceParts.push_back(0);
    rhsSourceParts.push_back(0);
    lhsLaneOffsets.push_back(0);
    rhsLaneOffsets.push_back(0);
    for (int64_t window = 0; window < partialSlots; ++window)
      scaleReplicaParts.push_back(window);
  }
  for (int64_t output = 0; terminalOutputCohort &&
                                   output < *outputReplicaCount;
       ++output) {
    auto firstResultPart = composeReplicaPart(
        dotResult, outputAxes, output, windowAxes, 0);
    auto lhsPart = firstResultPart
                       ? projectReplica(issueLhs, dotResult, *firstResultPart)
                       : std::optional<int64_t>();
    auto rhsPart = firstResultPart
                       ? projectReplica(issueRhs, dotResult, *firstResultPart)
                       : std::optional<int64_t>();
    if (!lhsPart || !rhsPart)
      return std::nullopt;
    lhsSourceParts.push_back(*lhsPart);
    rhsSourceParts.push_back(*rhsPart);
    lhsLaneOffsets.push_back(0);
    rhsLaneOffsets.push_back(0);
    for (int64_t window = 0; window < partialSlots; ++window) {
      auto resultPart = composeReplicaPart(dotResult, outputAxes, output,
                                           windowAxes, window);
      auto scalePart = resultPart
                           ? projectReplica(scaleReplicaType, dotResult,
                                            *resultPart)
                           : std::optional<int64_t>();
      if (!scalePart)
        return std::nullopt;
      scaleReplicaParts.push_back(*scalePart);
    }
  }

  const int64_t reductionAxis = dot.getOver()[0];
  auto sourceSlot = groupedLanePartialSlotType(
      builder, issueLhs, issueRhs, reductionAxis, sourceLanes);
  auto splitSlot =
      partialSlots == 1
          ? sourceSlot
          : sourceSlot
                ? splitPartialSlotType(builder, sourceSlot, reductionAxis,
                                       partialSlots)
                : riscv::ValueType();
  auto reducedSlot = splitSlot
                         ? reducedPartialSlotType(builder, splitSlot,
                                                  reductionAxis)
                         : riscv::ValueType();
  auto multiplyInstruction = partialMultiplyInstruction(issueLhs, issueRhs);
  const auto legal = kernel.getTarget().getLegalLMULEighths().asArrayRef();
  if (!sourceSlot || !splitSlot || !reducedSlot || !multiplyInstruction ||
      !llvm::is_contained(legal, sourceSlot.getLayout().getLmulEighths()) ||
      !llvm::is_contained(legal, splitSlot.getLayout().getLmulEighths()) ||
      !llvm::is_contained(legal, reducedSlot.getLayout().getLmulEighths())) {
    return std::nullopt;
  }
  auto sourceSet = riscv::PartialSetType::get(
      builder.getContext(), sourceSlot, reductionAxis, 1,
      sourceSlot.getShape()[0], sourceSlot.getLayout().getRegisterGroups());
  auto repackedSet =
      partialSlots == 1
          ? sourceSet
          : riscv::PartialSetType::get(
                builder.getContext(), splitSlot, reductionAxis, partialSlots,
                sourceSet.getTermsPerSlot() / partialSlots,
                sourceSet.getResourceGroups());
  auto reducedSet = riscv::PartialSetType::get(
      builder.getContext(), reducedSlot, reductionAxis, partialSlots,
      repackedSet.getTermsPerSlot(),
      partialSlots * reducedSlot.getLayout().getRegisterGroups());
  const int64_t scaleCombineArity =
      partialAxisExtents.empty() ? 0 : partialAxisExtents.back();
  if (scaleCombineArity <= 0 || partialSlots % scaleCombineArity)
    return std::nullopt;
  int64_t remainingSlots = partialSlots / scaleCombineArity;
  int64_t termsPerSlot =
      repackedSet.getTermsPerSlot() * scaleCombineArity;
  auto scaleCombinedSet = riscv::PartialSetType::get(
      builder.getContext(), reducedSlot, reductionAxis, remainingSlots,
      termsPerSlot,
      remainingSlots * reducedSlot.getLayout().getRegisterGroups());
  llvm::SmallVector<mlir::Attribute> combineSetTypes;
  llvm::SmallVector<int64_t> combineAxes;
  llvm::SmallVector<int64_t> combineArities;
  for (int64_t stage = static_cast<int64_t>(partialAxes.size()) - 2;
       stage >= 0; --stage) {
    const int64_t arity = partialAxisExtents[stage];
    if (arity <= 0 || remainingSlots % arity)
      return std::nullopt;
    remainingSlots /= arity;
    termsPerSlot *= arity;
    auto stageSet = riscv::PartialSetType::get(
        builder.getContext(), reducedSlot, reductionAxis, remainingSlots,
        termsPerSlot,
        remainingSlots * reducedSlot.getLayout().getRegisterGroups());
    combineSetTypes.push_back(mlir::TypeAttr::get(stageSet));
    combineAxes.push_back(partialAxes[stage]);
    combineArities.push_back(arity);
  }
  if (remainingSlots != 1)
    return std::nullopt;
  auto finalSet = combineSetTypes.empty()
                      ? scaleCombinedSet
                      : mlir::cast<riscv::PartialSetType>(
                            mlir::cast<mlir::TypeAttr>(combineSetTypes.back())
                                .getValue());
  const int64_t operandAndProduct =
      issueLhs.getLayout().getRegisterGroups() +
      issueRhs.getLayout().getRegisterGroups() + sourceSet.getResourceGroups();
  const int64_t repackAndReduce =
      sourceSet.getResourceGroups() + reducedSet.getResourceGroups();
  const int64_t reduceAndScale =
      reducedSet.getResourceGroups() +
      issueScale.getLayout().getRegisterGroups() +
      scaleCombinedSet.getResourceGroups();
  int64_t combineResources = scaleCombinedSet.getResourceGroups();
  riscv::PartialSetType previousSet = scaleCombinedSet;
  for (mlir::Attribute typeAttr : combineSetTypes) {
    auto nextSet = mlir::cast<riscv::PartialSetType>(
        mlir::cast<mlir::TypeAttr>(typeAttr).getValue());
    combineResources =
        std::max(combineResources, previousSet.getResourceGroups() +
                                       nextSet.getResourceGroups());
    previousSet = nextSet;
  }
  const int64_t resources = 1 + std::max(
      {operandAndProduct, repackAndReduce, reduceAndScale, combineResources});
  if (resources > kernel.getTarget().getVectorRegisters())
    return std::nullopt;
  const int64_t issueUnroll =
      std::min<int64_t>(dot.getPartialUnroll(), issueStreams);
  auto scaleParts = riscv_internal::staticProduct(
      scaleReplicaType.getLayout().getReplicaFactors().asArrayRef());
  llvm::DenseSet<mlir::Operation *> visited;
  const llvm::StringRef scaleSupply =
      scaleParts && *scaleParts >= 8 &&
              supportsScalarReplicaRematerialization(match.scale, visited)
          ? "scalar-rematerialize"
          : "vector-convert";
  auto finalizeInstruction = partialFinalizeInstruction(finalSet, reductionAxis);
  if (!finalizeInstruction)
    return std::nullopt;
  return riscv::NestedPartialPlanAttr::get(
      builder.getContext(), windowAxis, issueStreams, windowExtent,
      riscv_internal::integers(builder, partialAxes),
      riscv_internal::integers(builder, partialAxisExtents), partialSlots,
      issueUnroll,
      riscv_internal::integers(builder, outputAxes), *outputReplicaCount,
      riscv_internal::integers(builder, lhsSourceParts),
      riscv_internal::integers(builder, rhsSourceParts),
      riscv_internal::integers(builder, lhsLaneOffsets),
      riscv_internal::integers(builder, rhsLaneOffsets),
      riscv_internal::integers(builder, scaleReplicaParts), issueLhs, issueRhs,
      sourceSlot, splitSlot, reducedSlot, scaleReplicaType,
      sourceSet, repackedSet, reducedSet,
      scaleCombinedSet, builder.getArrayAttr(combineSetTypes),
      riscv_internal::integers(builder, combineAxes),
      riscv_internal::integers(builder, combineArities), scaleSupply,
      *multiplyInstruction, "rvv.partial-reduce.widen", *finalizeInstruction,
      resources);
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
    llvm::DenseMap<mlir::Operation *, riscv::ReduceOp> scaledReductions;
    llvm::DenseMap<mlir::Operation *, riscv::ReduceOp> replicaReductions;
    llvm::DenseMap<mlir::Operation *, riscv::ReduceOp> nestedScaledReductions;
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
      if (match->dot.getOver().size() > 1) {
        auto found = nestedScaledReductions.find(match->dot.getOperation());
        auto previous =
            found == nestedScaledReductions.end()
                ? std::optional<ReplicaScaledDotReduction>()
                : matchReplicaScaledDotReduction(found->second);
        if (!match->reducedAxes.empty() &&
            (!previous || previous->reducedAxes.size() <
                              match->reducedAxes.size()))
          nestedScaledReductions[match->dot.getOperation()] = reduce;
        continue;
      }
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
      scaledReductions[match->dot.getOperation()] = reduce;
    }
    getOperation().walk([&](mlir::scf::ForOp loop) {
      auto contributions = matchLevelScaledLoop(loop);
      if (contributions) {
        for (ScaledPartialContribution &contribution : *contributions)
          levelScaledDots.insert(contribution.dot.getOperation());
        return;
      }
      if (auto seed = matchLevelScaledLoopSeed(loop))
        levelScaledDots.insert(seed->dot.getOperation());
    });

    llvm::SmallVector<riscv::RVVWidenDotOp> dots;
    getOperation().walk(
        [&](riscv::RVVWidenDotOp dot) { dots.push_back(dot); });
    for (riscv::RVVWidenDotOp dot : dots) {
      dot->removeAttr("partial_layout_plan");
      dot->removeAttr("nested_partial_plan");
      dot->removeAttr("sequential_partial_plan");
      dot->removeAttr("scaled_partial_plan");
      dot->removeAttr("layered_partial_plan");
      dot->removeAttr("partial_combine_plan");
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
      std::optional<riscv::NestedPartialPlanAttr> nestedPlan;
      std::optional<LayeredTopologyFacts> layeredPlan;
      riscv::ValueType replicaAccumulator;
      riscv::ValueType sequentialLhsIssue;
      riscv::ValueType sequentialRhsIssue;
      riscv::ValueType sequentialAccumulator;
      std::optional<std::string> sequentialMultiply;
      llvm::StringRef sequentialLhsSupply = "slice";
      llvm::StringRef sequentialRhsSupply = "slice";
      bool sequentialPlanClosed = false;

      if (auto found = nestedScaledReductions.find(dot.getOperation());
          found != nestedScaledReductions.end()) {
        auto match = matchReplicaScaledDotReduction(found->second);
        int64_t windowAxis = 0;
        if (match)
          for (int64_t axis : match->reducedAxes) {
            auto candidateLhs = axisPosition(dot.getLhs().getType(), axis);
            auto candidateRhs = axisPosition(dot.getRhs().getType(), axis);
            auto candidateResult = result ? axisPosition(result, axis)
                                          : std::optional<size_t>();
            if (!candidateLhs || !candidateRhs || !candidateResult)
              continue;
            const int64_t lhsAxisTime =
                dot.getLhs().getType().getLayout().getTimeFactors()[*candidateLhs];
            const int64_t rhsAxisTime =
                dot.getRhs().getType().getLayout().getTimeFactors()[*candidateRhs];
            const int64_t lhsAxisLane =
                dot.getLhs().getType().getLayout().getLaneFactors()[*candidateLhs];
            const int64_t rhsAxisLane =
                dot.getRhs().getType().getLayout().getLaneFactors()[*candidateRhs];
            const int64_t axisExtent = result.getShape()[*candidateResult];
            if (lhsAxisTime <= 0 || lhsAxisTime != rhsAxisTime ||
                lhsAxisLane <= 0 || lhsAxisLane != rhsAxisLane ||
                axisExtent != lhsAxisTime * lhsAxisLane)
              continue;
            if (!windowAxis || lhsAxisTime > 1) {
              windowAxis = axis;
              if (lhsAxisTime > 1)
                break;
            }
          }
        auto lhsPosition = axisPosition(dot.getLhs().getType(), windowAxis);
        auto rhsPosition = axisPosition(dot.getRhs().getType(), windowAxis);
        auto resultPosition = result ? axisPosition(result, windowAxis)
                                     : std::optional<size_t>();
        const int64_t lhsTime = lhsPosition
                                    ? dot.getLhs()
                                          .getType()
                                          .getLayout()
                                          .getTimeFactors()[*lhsPosition]
                                    : 0;
        const int64_t rhsTime = rhsPosition
                                    ? dot.getRhs()
                                          .getType()
                                          .getLayout()
                                          .getTimeFactors()[*rhsPosition]
                                    : 0;
        const int64_t lhsWindow = lhsPosition
                                      ? dot.getLhs()
                                            .getType()
                                            .getLayout()
                                            .getLaneFactors()[*lhsPosition]
                                      : 0;
        const int64_t rhsWindow = rhsPosition
                                      ? dot.getRhs()
                                            .getType()
                                            .getLayout()
                                            .getLaneFactors()[*rhsPosition]
                                      : 0;
        const int64_t logicalExtent =
            resultPosition ? result.getShape()[*resultPosition] : 0;
        mlir::Type finalType = found->second.getResult().getType();
        auto finalElement = mlir::dyn_cast<mlir::IntegerType>(
            riscv_internal::logicalElement(finalType));
        auto finalValue = mlir::dyn_cast<riscv::ValueType>(finalType);
        const int64_t finalReplicaCount =
            finalValue
                ? product(finalValue.getLayout().getReplicaFactors())
                : 1;
        const bool closedScalarFinal =
            !finalValue ||
            (finalValue.getLayout().getCarrier() == "scalar" &&
             product(finalValue.getLayout().getTimeFactors()) == 1 &&
             product(finalValue.getLayout().getLaneFactors()) == 1 &&
             (finalReplicaCount == 1 ||
              (finalReplicaCount > 1 &&
               !reachesReductionThroughPureOps(found->second.getResult()))) &&
             product(finalValue.getLayout().getFragmentFactors()) == 1 &&
             product(finalValue.getLayout().getLocalFactors()) == 1);
        if (match && windowAxis > 0 && lhsPosition && rhsPosition &&
            resultPosition && lhsTime >= 1 && lhsTime == rhsTime &&
            lhsWindow > 1 && lhsWindow == rhsWindow &&
            logicalExtent == lhsTime * lhsWindow && finalElement &&
            finalElement.isSigned() && finalElement.getWidth() == 32 &&
            closedScalarFinal && dot.getPartialUnroll() > 0) {
          const bool issueOnlyWindow =
              hasGappedIndexedWindowRoot(dot.getLhs(), windowAxis) ||
              hasGappedIndexedWindowRoot(dot.getRhs(), windowAxis);
          const int64_t plannedStreams =
              issueOnlyWindow ? logicalExtent : lhsTime;
          const int64_t plannedWindow = issueOnlyWindow ? 1 : lhsWindow;
          nestedPlan = planNestedPartialCarrier(
              builder, dot, *match, windowAxis, plannedStreams, plannedWindow);
          if (!nestedPlan) {
            dot.emitError(
                "nested scaled contraction has no legal full-product carrier under the target resource contract; lhs=")
                << dot.getLhs().getType() << ", rhs=" << dot.getRhs().getType()
                << ", scale=" << match->scale.getType()
                 << ", window_axis=" << windowAxis << ", issue_streams="
                << lhsTime << ", window_extent=" << lhsWindow
                << ", reduction_lanes=" << dot.getReductionLanes()
                << ", reduction_streams=" << dot.getReductionStreams()
                << ", final_type=" << found->second.getResult().getType();
            signalPassFailure();
            return;
          }
          kind = "nested_scaled_stream";
          partialAxes.assign((*nestedPlan).getPartialAxes().asArrayRef().begin(),
                             (*nestedPlan).getPartialAxes().asArrayRef().end());
          outputAxes.assign((*nestedPlan).getOutputAxes().asArrayRef().begin(),
                            (*nestedPlan).getOutputAxes().asArrayRef().end());
          sourceSlots = plannedStreams;
          partialSlots =
              plannedStreams * (*nestedPlan).getPartialSlots();
          replicas = (*nestedPlan).getOutputReplicas();
          laneSplit = (*nestedPlan).getPartialSlots();
          combineArity = (*nestedPlan).getPartialSlots();
          resources = (*nestedPlan).getResourceGroups();
        }
      }

      if (kind.empty())
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
          int64_t rootGroups = 0;
          for (const ProjectedRoot &root : layered->roots)
            rootGroups +=
                layered->windowTypes.lookup(root.value)
                    .getLayout()
                    .getRegisterGroups();
          const int64_t layeredResources =
              layered->accumulatorType.getLayout().getRegisterGroups() +
              layered->layeredWindowType.getLayout().getRegisterGroups() +
              rootGroups + 1;
          if (kernel &&
              layeredResources <= kernel.getTarget().getVectorRegisters()) {
            kind = "layered";
            rootOperand = layered->rootOperand;
            partialSlots = layered->streams;
            combineArity = layered->streams;
            resources = layeredResources;
            layeredPlan = std::move(*layered);
          }
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
      if (kind.empty() && outputReplicas == 1 &&
          levelScaledDots.contains(dot.getOperation())) {
        kind = "level_scaled";
        partialSlots = dot.getPartialUnroll();
        combineArity = partialSlots % 2 == 0 ? 2 : partialSlots;
        resources =
            std::max<int64_t>(1, partialSlots * partialGroups + 2);
        if (!kernel || resources > kernel.getTarget().getVectorRegisters()) {
          dot.emitError(
              "selected level-scaled issue cohort exceeds the target vector-register budget; requested_slots=")
              << partialSlots << ", resource_groups=" << resources
              << ", available="
              << (kernel ? kernel.getTarget().getVectorRegisters() : int64_t{-1});
          signalPassFailure();
          return;
        }
      }
      // Independent partials are a structural preference supplied by the
      // target profile, not a universal property of RVV.  The generic planner
      // only proves that the selected multilevel tree is geometrically and
      // resource legal.
      if (kind.empty() && kernel &&
          kernel.getTarget().getPartialCombinePolicy() ==
              "independent-multilevel" &&
          streams >= 4 &&
          (streams & (streams - 1)) == 0 &&
          available >= 2 && partialGroups > 0 &&
          streams <= (available - 2) / partialGroups) {
        kind = "independent";
        partialSlots = streams;
        combineArity = 2;
        resources = std::max<int64_t>(1, streams * partialGroups + 2);
      }
      if (kind.empty())
        kind = fused ? "sequential_fused" : "sequential_per_stream";

      // A sequential contraction is only real when its complete issue slice,
      // accumulator carrier, and simultaneous resource contract are selected
      // before materialization.  Fused and per-stream realizations share the
      // same carrier; they differ only in whether the carrier or the finalized
      // i32 value crosses an issue boundary.
      if (kind == "sequential_fused" || kind == "sequential_per_stream" ||
          kind == "level_scaled") {
        sequentialLhsIssue =
            sequentialIssueType(builder, dot, dot.getLhs().getType());
        sequentialRhsIssue =
            sequentialIssueType(builder, dot, dot.getRhs().getType());
        sequentialAccumulator = partialSlotType(builder, dot);
        sequentialMultiply = sequentialMultiplyInstruction(
            sequentialLhsIssue, sequentialRhsIssue);
        if (kind == "level_scaled" && kernel && dot.getOver().size() == 1 &&
            sequentialLhsIssue && sequentialRhsIssue) {
          const int64_t reductionAxis = dot.getOver()[0];
          auto lhsPosition = axisPosition(sequentialLhsIssue, reductionAxis);
          auto rhsPosition = axisPosition(sequentialRhsIssue, reductionAxis);
          if (lhsPosition &&
              supportsIssueStorageRematerialization(
                  builder, dot.getLhs(), reductionAxis,
                  sequentialLhsIssue.getShape()[*lhsPosition],
                  kernel.getTarget()))
            sequentialLhsSupply = "storage-rematerialize";
          if (rhsPosition &&
              supportsIssueStorageRematerialization(
                  builder, dot.getRhs(), reductionAxis,
                  sequentialRhsIssue.getShape()[*rhsPosition],
                  kernel.getTarget()))
            sequentialRhsSupply = "storage-rematerialize";
        }
        auto lhsParts = riscv_internal::staticProduct(
            sequentialLhsIssue
                ? sequentialLhsIssue.getLayout().getReplicaFactors().asArrayRef()
                : llvm::ArrayRef<int64_t>());
        auto rhsParts = riscv_internal::staticProduct(
            sequentialRhsIssue
                ? sequentialRhsIssue.getLayout().getReplicaFactors().asArrayRef()
                : llvm::ArrayRef<int64_t>());
        const int64_t plannedResources =
            sequentialLhsIssue && sequentialRhsIssue && sequentialAccumulator
                ? std::max<int64_t>(
                      sequentialLhsIssue.getLayout().getRegisterGroups() +
                          sequentialRhsIssue.getLayout().getRegisterGroups() +
                          sequentialAccumulator.getLayout().getRegisterGroups(),
                      sequentialAccumulator.getLayout().getRegisterGroups() + 2)
                : 0;
        sequentialPlanClosed =
            sequentialLhsIssue && sequentialRhsIssue && sequentialAccumulator &&
            sequentialMultiply &&
            lhsParts && rhsParts && *lhsParts == 1 && *rhsParts == 1 &&
            outputReplicas > 0 &&
            dot.getLhsParts().size() ==
                static_cast<size_t>(streams * outputReplicas) &&
            dot.getRhsParts().size() ==
                static_cast<size_t>(streams * outputReplicas) &&
            dot.getLhsLaneOffsets().size() ==
                static_cast<size_t>(outputReplicas) &&
            dot.getRhsLaneOffsets().size() ==
                static_cast<size_t>(outputReplicas) &&
            kernel && plannedResources <= kernel.getTarget().getVectorRegisters();
        if (!sequentialPlanClosed) {
          dot.emitError(
              "selected sequential topology has no closed issue-slice, free-axis, and accumulator carrier; lhs_issue=")
              << sequentialLhsIssue << ", rhs_issue=" << sequentialRhsIssue
              << ", accumulator=" << sequentialAccumulator
              << ", output_replicas=" << outputReplicas
              << ", lhs_issue_parts="
              << (lhsParts ? *lhsParts : int64_t{-1})
              << ", rhs_issue_parts="
              << (rhsParts ? *rhsParts : int64_t{-1})
              << ", lhs_plan_parts=" << dot.getLhsParts().size()
              << ", rhs_plan_parts=" << dot.getRhsParts().size()
              << ", streams=" << streams
              << ", resources=" << plannedResources;
          signalPassFailure();
          return;
        }
        if (kind != "level_scaled")
          resources = plannedResources;
      }

      dot->setAttr("partial_topology",
                   makePartialTopology(builder, dot, kind, rootOperand,
                                       partialAxes, outputAxes, sourceSlots,
                                       partialSlots, replicas, laneSplit,
                                       combineArity,
                                       resources));
      if (nestedPlan)
        dot->setAttr("nested_partial_plan", *nestedPlan);

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

      if (kind == "sequential_fused" || kind == "sequential_per_stream" ||
          kind == "level_scaled") {
        auto lhsIssue = sequentialLhsIssue;
        auto rhsIssue = sequentialRhsIssue;
        auto accumulator = sequentialAccumulator;
        auto accumulatorElement = accumulator
                                      ? mlir::dyn_cast<mlir::IntegerType>(
                                            accumulator.getElementType())
                                      : mlir::IntegerType();
        const int64_t issueCount = dot.getReductionStreams();
        llvm::SmallVector<int64_t> lhsLaneOffsets;
        llvm::SmallVector<int64_t> rhsLaneOffsets;
        lhsLaneOffsets.reserve(static_cast<size_t>(issueCount * outputReplicas));
        rhsLaneOffsets.reserve(static_cast<size_t>(issueCount * outputReplicas));
        for (int64_t output = 0; output < outputReplicas; ++output)
          for (int64_t issue = 0; issue < issueCount; ++issue) {
            lhsLaneOffsets.push_back(dot.getLhsLaneOffsets()[output]);
            rhsLaneOffsets.push_back(dot.getRhsLaneOffsets()[output]);
          }
        const int64_t plannedResources =
            lhsIssue && rhsIssue && accumulator
                ? std::max<int64_t>(
                      lhsIssue.getLayout().getRegisterGroups() +
                          rhsIssue.getLayout().getRegisterGroups() +
                          accumulator.getLayout().getRegisterGroups(),
                      accumulator.getLayout().getRegisterGroups() + 2)
                : 0;
        if (!sequentialPlanClosed || !lhsIssue || !rhsIssue || !accumulator ||
            !accumulatorElement ||
            (kind != "level_scaled" && plannedResources != resources) || !kernel ||
            plannedResources > kernel.getTarget().getVectorRegisters()) {
          dot.emitError(
              "selected sequential topology has no closed issue-slice and accumulator plan");
          signalPassFailure();
          return;
        }
        const llvm::StringRef realization =
            kind == "level_scaled"
                ? llvm::StringRef("fused_partial")
                : kind == "sequential_fused" ? llvm::StringRef("fused")
                                              : llvm::StringRef("per_stream");
        const llvm::StringRef finalizeInstruction =
            kind == "level_scaled"
                ? llvm::StringRef("none")
                : accumulatorElement.getWidth() == 16
                      ? llvm::StringRef("rvv.vwredsum.partial")
                      : llvm::StringRef("rvv.vredsum.partial");
        dot->setAttr(
            "sequential_partial_plan",
            riscv::SequentialPartialPlanAttr::get(
                builder.getContext(), realization,
                issueCount,
                builder.getDenseI64ArrayAttr(dot.getOver()), lhsIssue,
                rhsIssue, accumulator, sequentialLhsSupply,
                sequentialRhsSupply,
                builder.getDenseI64ArrayAttr(dot.getLhsParts()),
                builder.getDenseI64ArrayAttr(dot.getRhsParts()),
                builder.getDenseI64ArrayAttr(lhsLaneOffsets),
                builder.getDenseI64ArrayAttr(rhsLaneOffsets),
                *sequentialMultiply, "rvv.vwmacc.partial", finalizeInstruction,
                plannedResources));
      }

      if (kind == "scaled" || kind == "reduced_scaled") {
        auto found = scaledReductions.find(dot.getOperation());
        auto match = found != scaledReductions.end()
                         ? matchReplicaScaledDotReduction(found->second)
                         : std::optional<ReplicaScaledDotReduction>();
        auto scaleSource =
            match ? mlir::dyn_cast<riscv::ValueType>(match->scale.getType())
                  : riscv::ValueType();
        auto scalarScale =
            scaleSource ? scalarReplicaType(builder, scaleSource)
                        : riscv::ValueType();
        llvm::DenseSet<mlir::Operation *> rematerializationVisited;
        const bool canRematerialize =
            match && supportsScalarReplicaRematerialization(
                         match->scale, rematerializationVisited);
        const llvm::StringRef scaleSupply =
            canRematerialize ? "scalar-rematerialize" : "vector-convert";
        auto narrowScale =
            kind == "scaled" && laneSplit == 1 && match
                ? plannedNarrowScaleType(builder, match->scale, scalarScale)
                : riscv::ValueType();
        auto multiplyInstruction =
            partialMultiplyInstruction(dot.getLhs().getType(),
                                       dot.getRhs().getType());
        auto finalSet = laneSplit > 1 || kind == "reduced_scaled"
                            ? scaleCombinedSetType
                            : fullScaledSetType;
        auto finalizeInstruction =
            finalSet ? partialFinalizeInstruction(finalSet, reductionAxis)
                     : std::optional<llvm::StringRef>();
        const llvm::StringRef reduceInstruction =
            laneSplit > 1 || kind == "reduced_scaled"
                ? llvm::StringRef("rvv.partial-reduce.widen")
                : llvm::StringRef("none");
        int64_t plannedResources = resources;
        if (kind == "scaled" && laneSplit == 1 && partialSetType &&
            scaledSetType && combineArity > 0 &&
            partialSlots % combineArity == 0) {
          const int64_t chunks = partialSlots / combineArity;
          plannedResources = std::max<int64_t>(
              plannedResources,
              partialSetType.getResourceGroups() +
                  chunks * scaledSetType.getResourceGroups() +
                  (narrowScale
                       ? narrowScale.getLayout().getRegisterGroups()
                       : 0));
        }
        if (!match || !scalarScale || !multiplyInstruction || !finalSet ||
            !finalizeInstruction ||
            (kind == "scaled" && laneSplit == 1 && !narrowScale) || !kernel ||
            plannedResources > kernel.getTarget().getVectorRegisters()) {
          dot.emitError(
              "selected scaled topology has no closed scale-supply and reduction plan");
          signalPassFailure();
          return;
        }
        if (plannedResources != resources) {
          auto topology = dot.getPartialTopology();
          dot->setAttr(
              "partial_topology",
              riscv::PartialTopologyAttr::get(
                  builder.getContext(), topology.getKind(),
                  topology.getRootOperand(), topology.getPartialAxes(),
                  topology.getOutputAxes(), topology.getSourceSlots(),
                  topology.getPartialSlots(), topology.getOutputReplicas(),
                  topology.getLaneSplit(), topology.getCombineArity(),
                  topology.getSlotOrder(), plannedResources));
        }
        dot->setAttr(
            "scaled_partial_plan",
            riscv::ScaledPartialPlanAttr::get(
                builder.getContext(), kind, scalarScale,
                narrowScale
                    ? mlir::Type(narrowScale)
                    : mlir::Type(mlir::NoneType::get(builder.getContext())),
                scaleSupply, *multiplyInstruction, reduceInstruction,
                *finalizeInstruction, plannedResources));
      }

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
        llvm::SmallVector<mlir::Attribute> rootWindowInstructions;
        rootWindowTypes.reserve(layeredPlan->roots.size());
        rootStoragePlans.reserve(layeredPlan->roots.size());
        rootWindowInstructions.reserve(layeredPlan->roots.size());
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
          rootWindowInstructions.push_back(builder.getStringAttr(
              candidate.access.getMapping() == "grouped_layered"
                  ? "rvv.storage-window.layered"
                  : "rvv.storage-window.natural"));
        }
        const int64_t rawGroups =
            layeredPlan->layeredWindowType.getLayout().getRegisterGroups();
        auto storageType = riscv::LayeredWindowType::get(
            builder.getContext(),
            mlir::cast<riscv::ValueType>(
                layeredPlan->layered.field.getResult().getType()),
            layeredPlan->layeredWindowType, layeredPlan->reductionAxis,
            layeredPlan->layers, layeredPlan->windowsPerLayer, rawGroups);
        llvm::SmallVector<mlir::Attribute> decodeInstructions;
        for (auto [shift, mask] : llvm::zip(
                 layeredPlan->layered.geometry
                     .getShiftAmountForLogicalLayer()
                     .asArrayRef(),
                 layeredPlan->layered.geometry
                     .getMaskValueForLogicalLayer()
                     .asArrayRef()))
          decodeInstructions.push_back(builder.getStringAttr(
              riscv_internal::layeredStorageDecodeInstruction(shift, mask)));
        auto accumulatorElement = mlir::dyn_cast<mlir::IntegerType>(
            layeredPlan->accumulatorType.getElementType());
        if (!accumulatorElement ||
            (accumulatorElement.getWidth() != 16 &&
             accumulatorElement.getWidth() != 32)) {
          dot.emitError(
              "selected layered topology has no typed final reduction leaf");
          signalPassFailure();
          return;
        }
        const llvm::StringRef finalizeInstruction =
            accumulatorElement.getWidth() == 16 ? "rvv.vwredsum.partial"
                                                : "rvv.vredsum.partial";
        dot->setAttr(
            "layered_partial_plan",
            riscv::LayeredPartialPlanAttr::get(
                builder.getContext(), rootIndex,
                layeredPlan->layeredWindowType, storageType,
                layeredPlan->lhsIssueType, layeredPlan->rhsIssueType,
                layeredPlan->accumulatorType,
                builder.getArrayAttr(rootWindowTypes),
                builder.getArrayAttr(rootStoragePlans),
                builder.getArrayAttr(rootWindowInstructions),
                builder.getArrayAttr(decodeInstructions), finalizeInstruction,
                layeredPlan->layered.geometry, resources));
      }
    }

    // A nested partial plan owns the contraction's issue unroll.  Lowering
    // records the same requested factor on the original issue loop so
    // non-partial contractions can still use generic loop unrolling.  Once a
    // nested plan has been selected, consume that loop attribute here: the
    // materializer will create the planned issue loop and attach the frozen
    // NestedPartialPlanAttr::issue_unroll exactly once.  A mixed loop would
    // otherwise have two incompatible schedule owners, so reject it instead
    // of silently unrolling the whole canonical contraction as well.
    llvm::SmallVector<mlir::scf::ForOp> issueLoops;
    getOperation().walk([&](mlir::scf::ForOp loop) {
      if (loop->hasAttr("weft.riscv.unroll_factor"))
        issueLoops.push_back(loop);
    });
    for (mlir::scf::ForOp loop : issueLoops) {
      llvm::SmallVector<riscv::RVVWidenDotOp> ownedDots;
      loop.walk([&](riscv::RVVWidenDotOp dot) {
        if (dot->getParentOfType<mlir::scf::ForOp>() == loop)
          ownedDots.push_back(dot);
      });
      const bool consumesIssueUnroll = llvm::any_of(
          ownedDots, [](riscv::RVVWidenDotOp dot) {
            return static_cast<bool>(dot.getNestedPartialPlanAttr());
          });
      if (!consumesIssueUnroll)
        continue;
      if (ownedDots.empty() ||
          !llvm::all_of(ownedDots, [](riscv::RVVWidenDotOp dot) {
            return static_cast<bool>(dot.getNestedPartialPlanAttr());
          })) {
        loop.emitError(
            "one physical issue loop mixes planner-owned and generic unroll contracts");
        signalPassFailure();
        return;
      }
      const int64_t requested =
          loop
              ->getAttrOfType<mlir::IntegerAttr>(
                  "weft.riscv.unroll_factor")
              .getInt();
      for (riscv::RVVWidenDotOp dot : ownedDots) {
        auto plan = dot.getNestedPartialPlanAttr();
        if (dot.getPartialUnroll() != requested ||
            plan.getIssueUnroll() !=
                std::min<int64_t>(requested, plan.getIssueStreams())) {
          dot.emitError(
              "nested partial plan did not consume the issue-loop unroll contract exactly once");
          signalPassFailure();
          return;
        }
      }
      loop->removeAttr("weft.riscv.unroll_factor");
    }

    // Freeze the complete combine program after every dot has its selected
    // topology and typed layout plan.  The following materialization pass may
    // validate these facts against the still-present SSA graph, but it must not
    // choose a vector/scalar realization or rebuild intermediate set types.
    for (riscv::RVVWidenDotOp dot : dots) {
      if (!dot || !dot->getBlock() ||
          dot.getPartialTopology().getKind() != "independent")
        continue;
      riscv::ValueType lhs = dot.getLhs().getType();
      riscv::ValueType rhs = dot.getRhs().getType();
      auto lhsStreams = riscv_internal::staticProduct(
          lhs.getLayout().getTimeFactors().asArrayRef());
      auto rhsStreams = riscv_internal::staticProduct(
          rhs.getLayout().getTimeFactors().asArrayRef());
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
      auto multiplyInstruction = partialMultiplyInstruction(lhs, rhs);
      if (!lhsStreams || !rhsStreams || *lhsStreams != *rhsStreams ||
          *lhsStreams < 2 || !layoutPlan || !slotType || !resultElement ||
          !resultElement.isSigned() || resultElement.getWidth() != 32 ||
          !outputParts || *outputParts <= 0 || !multiplyInstruction) {
        dot.emitError(
            "selected independent topology has no complete typed combine inputs");
        signalPassFailure();
        return;
      }
      const int64_t slots = *lhsStreams;
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
          mlir::dyn_cast<riscv::ValueType>(layoutPlan.getVectorSlotType());
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
          vectorConsumer && vectorResult && vectorSlotType && lhsReplicas &&
          rhsReplicas && *lhsReplicas == 1 && *rhsReplicas == 1 &&
          vectorResultStreams && vectorResultReplicas &&
          *vectorResultStreams == 1 && *vectorResultReplicas == 1;
      riscv::ValueType selectedSlot =
          preservesFreeLane ? vectorSlotType : slotType;
      auto reductionAxis =
          primaryReductionLaneAxis(selectedSlot, dot.getOver());
      auto termsPerSourceSlot = laneProductForAxes(selectedSlot, dot.getOver());
      const int64_t slotGroups = selectedSlot.getLayout().getRegisterGroups();
      const int64_t setGroups = slots * slotGroups;
      const int64_t operandGroups = lhs.getLayout().getRegisterGroups() +
                                    rhs.getLayout().getRegisterGroups();
      auto kernel = dot->getParentOfType<riscv::KernelOp>();
      if (!reductionAxis || !termsPerSourceSlot || !kernel || setGroups <= 0 ||
          operandGroups < 0 ||
          kernel.getTarget().getVectorRegisters() < operandGroups + 2 ||
          setGroups >
              kernel.getTarget().getVectorRegisters() - operandGroups - 2) {
        dot.emitError(
            "selected independent topology exceeds its typed source-set resource contract");
        signalPassFailure();
        return;
      }
      const auto reductionPosition =
          axisPosition(selectedSlot, *reductionAxis);
      if (!reductionPosition) {
        dot.emitError(
            "selected independent topology lost its reduction axis in the partial carrier");
        signalPassFailure();
        return;
      }
      auto sourceSet = riscv::PartialSetType::get(
          builder.getContext(), selectedSlot, *reductionAxis, slots,
          *termsPerSourceSlot, setGroups);
      llvm::SmallVector<mlir::Attribute> combineTypes;
      llvm::SmallVector<int64_t> combineArities;
      riscv::PartialSetType finalSet = sourceSet;
      int64_t remainingSlots = slots;
      int64_t termsPerSlot = sourceSet.getTermsPerSlot();
      while (remainingSlots > 1) {
        const int64_t arity = remainingSlots % 2 == 0 ? 2 : remainingSlots;
        remainingSlots /= arity;
        termsPerSlot *= arity;
        finalSet = riscv::PartialSetType::get(
            builder.getContext(), selectedSlot, *reductionAxis, remainingSlots,
            termsPerSlot, remainingSlots * slotGroups);
        combineTypes.push_back(mlir::TypeAttr::get(finalSet));
        combineArities.push_back(arity);
      }
      llvm::SmallVector<int64_t> lhsParts;
      llvm::SmallVector<int64_t> rhsParts;
      const int64_t plannedOutputs = preservesFreeLane ? 1 : *outputParts;
      for (int64_t outputPart = 0; outputPart < plannedOutputs; ++outputPart) {
        int64_t lhsBase = 0;
        int64_t rhsBase = 0;
        if (!preservesFreeLane) {
          auto lhsReplica = resultValue
                                ? projectReplica(lhs, resultValue, outputPart)
                                : std::optional<int64_t>(0);
          auto rhsReplica = resultValue
                                ? projectReplica(rhs, resultValue, outputPart)
                                : std::optional<int64_t>(0);
          if (!lhsReplica || !rhsReplica) {
            dot.emitError(
                "selected independent topology has no complete output-replica map");
            signalPassFailure();
            return;
          }
          lhsBase = *lhsReplica * slots;
          rhsBase = *rhsReplica * slots;
        }
        for (int64_t stream = 0; stream < slots; ++stream) {
          lhsParts.push_back(preservesFreeLane ? stream : lhsBase + stream);
          rhsParts.push_back(preservesFreeLane ? stream : rhsBase + stream);
        }
      }
      auto finalizeInstruction =
          preservesFreeLane
              ? std::optional<llvm::StringRef>("rvv.partial-finalize.widen")
              : partialFinalizeInstruction(finalSet, *reductionAxis);
      if (!finalizeInstruction) {
        dot.emitError(
            "selected independent topology has no typed finalization leaf");
        signalPassFailure();
        return;
      }
      llvm::ArrayRef<int64_t> empty;
      dot->setAttr(
          "partial_combine_plan",
          riscv::PartialCombinePlanAttr::get(
              builder.getContext(),
              preservesFreeLane ? "independent_vector"
                                : "independent_scalar",
              sourceSet, mlir::NoneType::get(builder.getContext()), finalSet,
              builder.getArrayAttr(combineTypes),
              builder.getDenseI64ArrayAttr(combineArities),
              builder.getDenseI64ArrayAttr(lhsParts),
              builder.getDenseI64ArrayAttr(rhsParts),
              builder.getDenseI64ArrayAttr(empty), *multiplyInstruction,
              *finalizeInstruction, plannedOutputs,
              dot.getPartialTopology().getResourceGroups()));
    }

    llvm::SmallVector<mlir::scf::ForOp> plannedLevelLoops;
    getOperation().walk(
        [&](mlir::scf::ForOp loop) { plannedLevelLoops.push_back(loop); });
    for (mlir::scf::ForOp loop : plannedLevelLoops) {
      auto matched = matchLevelScaledLoop(loop);
      if (!matched)
        if (auto seed = matchLevelScaledLoopSeed(loop))
          matched = llvm::SmallVector<ScaledPartialContribution>{*seed};
      if (!matched ||
          !llvm::all_of(*matched, [](const ScaledPartialContribution &item) {
            return hasTopology(item.dot, "level_scaled");
          }))
        continue;
      mlir::Type carry = loop.getRegionIterArg(0).getType();
      auto carryType = mlir::dyn_cast<riscv::ValueType>(carry);
      auto carryInteger = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(carry));
      auto outputParts = carryType
                             ? riscv_internal::staticProduct(
                                   carryType.getLayout()
                                       .getReplicaFactors()
                                       .asArrayRef())
                             : carryInteger && carryInteger.isSigned() &&
                                       carryInteger.getWidth() == 32
                                   ? std::optional<int64_t>(1)
                                   : std::optional<int64_t>();
      auto firstLayout = matched->front().dot.getPartialLayoutPlanAttr();
      auto firstTopology = matched->front().dot.getPartialTopology();
      auto slotType =
          firstLayout ? mlir::dyn_cast<riscv::ValueType>(
                            firstLayout.getPartialSlotType())
                      : riscv::ValueType();
      auto reducedSet =
          firstLayout ? mlir::dyn_cast<riscv::PartialSetType>(
                            firstLayout.getReducedSetType())
                      : riscv::PartialSetType();
      auto finalSet =
          firstLayout ? mlir::dyn_cast<riscv::PartialSetType>(
                            firstLayout.getScaleCombinedSetType())
                      : riscv::PartialSetType();
      const int64_t slots = firstTopology.getPartialSlots();
      const int64_t reductionAxis = matched->front().dot.getOver()[0];
      const bool scalarCarry =
          carryType ? carryType.getLayout().getCarrier() == "scalar"
                    : static_cast<bool>(carryInteger);
      if (!scalarCarry || !outputParts || *outputParts <= 0 || !slotType ||
          !reducedSet || !finalSet || slots <= 0) {
        loop.emitError(
            "selected level-scaled topology has no complete typed combine carrier");
        signalPassFailure();
        return;
      }
      auto sourceSet = riscv::PartialSetType::get(
          builder.getContext(), slotType, reductionAxis, slots,
          slotType.getShape()[0],
          slots * slotType.getLayout().getRegisterGroups());
      auto finalizeInstruction =
          partialFinalizeInstruction(finalSet, reductionAxis);
      std::optional<std::string> commonMultiply;
      for (ScaledPartialContribution &contribution : *matched) {
        auto selected = partialMultiplyInstruction(contribution.dot.getLhs().getType(),
                                                   contribution.dot.getRhs().getType());
        if (!selected || (commonMultiply && *commonMultiply != *selected)) {
          loop.emitError(
              "selected level-scaled topology has inconsistent multiply leaves");
          signalPassFailure();
          return;
        }
        commonMultiply = *selected;
      }
      if (!commonMultiply || !finalizeInstruction) {
        loop.emitError(
            "selected level-scaled topology has no complete leaf selection");
        signalPassFailure();
        return;
      }
      llvm::ArrayRef<int64_t> empty;
      for (ScaledPartialContribution &contribution : *matched) {
        llvm::SmallVector<int64_t> lhsParts;
        llvm::SmallVector<int64_t> rhsParts;
        llvm::SmallVector<int64_t> scaleParts;
        for (int64_t outputPart = 0; outputPart < *outputParts; ++outputPart) {
          auto singletonPart = [](riscv::ValueType value)
              -> std::optional<int64_t> {
            auto parts = riscv_internal::staticProduct(
                value.getLayout().getReplicaFactors().asArrayRef());
            return parts && *parts == 1 ? std::optional<int64_t>(0)
                                        : std::nullopt;
          };
          auto lhsPart = carryType
                             ? projectReplica(contribution.dot.getLhs().getType(),
                                              carryType, outputPart)
                             : singletonPart(contribution.dot.getLhs().getType());
          auto rhsPart = carryType
                             ? projectReplica(contribution.dot.getRhs().getType(),
                                              carryType, outputPart)
                             : singletonPart(contribution.dot.getRhs().getType());
          auto scaleType =
              mlir::dyn_cast<riscv::ValueType>(contribution.scale.getType());
          auto scalePart =
              carryType && scaleType
                  ? projectReplica(scaleType, carryType, outputPart)
                  : !carryType && !scaleType
                        ? std::optional<int64_t>(0)
                        : scaleType ? singletonPart(scaleType) : std::nullopt;
          if (!lhsPart || !rhsPart || !scalePart) {
            loop.emitError(
                "selected level-scaled topology has no complete output-replica map");
            signalPassFailure();
            return;
          }
          lhsParts.push_back(*lhsPart);
          rhsParts.push_back(*rhsPart);
          scaleParts.push_back(*scalePart);
        }
        contribution.dot->setAttr(
            "partial_combine_plan",
            riscv::PartialCombinePlanAttr::get(
                builder.getContext(), "level_scaled", sourceSet, reducedSet,
                finalSet, builder.getArrayAttr({}),
                builder.getDenseI64ArrayAttr(empty),
                builder.getDenseI64ArrayAttr(lhsParts),
                builder.getDenseI64ArrayAttr(rhsParts),
                builder.getDenseI64ArrayAttr(scaleParts), *commonMultiply,
                *finalizeInstruction, *outputParts,
                firstTopology.getResourceGroups()));
      }
    }

    // Scalar add trees combine multiple already-planned partial leaves.  Plan
    // the common carrier and every repack before materialization so equivalent
    // add spelling cannot trigger a second layout search in the next pass.
    llvm::SmallVector<riscv::BinaryOp> partialAddRoots;
    getOperation().walk([&](riscv::BinaryOp add) {
      add->removeAttr("partial_add_tree_plan");
      if (add.getKind() == "add" &&
          mlir::isa<mlir::IntegerType>(add.getResult().getType()))
        partialAddRoots.push_back(add);
    });
    for (riscv::BinaryOp root : partialAddRoots) {
      llvm::SmallVector<riscv::BinaryOp> adds;
      llvm::SmallVector<PartialAddLeaf> leaves;
      if (!collectPartialAddTree(root.getResult(), root, adds, leaves) ||
          leaves.size() < 2)
        continue;
      llvm::SmallVector<riscv::PartialSetType> leafTypes;
      bool complete = true;
      for (PartialAddLeaf leaf : leaves) {
        auto type = partialTypeForAddLeaf(builder, leaf);
        if (!type) {
          complete = false;
          break;
        }
        leafTypes.push_back(*type);
      }
      if (!complete)
        continue;
      riscv::PartialSetType targetType;
      llvm::SmallVector<int64_t> splits;
      llvm::SmallVector<size_t> order(leaves.size());
      std::iota(order.begin(), order.end(), 0);
      llvm::sort(order, [&](size_t lhsIndex, size_t rhsIndex) {
        auto lhsSet = leafTypes[lhsIndex];
        auto rhsSet = leafTypes[rhsIndex];
        auto lhsPosition = axisPosition(lhsSet.getPartialType(),
                                        lhsSet.getReductionAxis());
        auto rhsPosition = axisPosition(rhsSet.getPartialType(),
                                        rhsSet.getReductionAxis());
        return lhsSet.getPartialType().getLayout().getLaneFactors()[*lhsPosition] >
               rhsSet.getPartialType().getLayout().getLaneFactors()[*rhsPosition];
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
      llvm::SmallVector<mlir::Attribute> leafTypeAttrs;
      llvm::SmallVector<mlir::Attribute> normalizedTypeAttrs;
      llvm::SmallVector<mlir::Attribute> multiplyAttrs;
      int64_t totalSlots = 0;
      int64_t totalTerms = 0;
      int64_t totalResources = 0;
      for (auto [leaf, source, split] : llvm::zip(leaves, leafTypes, splits)) {
        auto normalized =
            split == 1
                ? source
                : riscv::PartialSetType::get(
                      builder.getContext(), targetType.getPartialType(),
                      targetType.getReductionAxis(), source.getSlots() * split,
                      source.getTermsPerSlot() / split,
                      source.getResourceGroups());
        leafTypeAttrs.push_back(mlir::TypeAttr::get(source));
        normalizedTypeAttrs.push_back(mlir::TypeAttr::get(normalized));
        if (leaf.dot) {
          auto multiply = partialMultiplyInstruction(leaf.dot.getLhs().getType(),
                                                     leaf.dot.getRhs().getType());
          if (!multiply) {
            complete = false;
            break;
          }
          multiplyAttrs.push_back(builder.getStringAttr(*multiply));
        } else {
          multiplyAttrs.push_back(builder.getStringAttr(""));
        }
        totalSlots += normalized.getSlots();
        totalTerms += normalized.getSlots() * normalized.getTermsPerSlot();
        totalResources += normalized.getResourceGroups();
      }
      if (!complete)
        continue;
      auto mergedType = riscv::PartialSetType::get(
          builder.getContext(), targetType.getPartialType(),
          targetType.getReductionAxis(), 1, totalTerms,
          targetType.getPartialType().getLayout().getRegisterGroups());
      auto finalizeInstruction =
          partialFinalizeInstruction(mergedType, targetType.getReductionAxis());
      if (!finalizeInstruction)
        continue;
      root->setAttr(
          "partial_add_tree_plan",
          riscv::PartialAddTreePlanAttr::get(
              builder.getContext(), builder.getArrayAttr(leafTypeAttrs),
              builder.getArrayAttr(normalizedTypeAttrs),
              builder.getArrayAttr(multiplyAttrs),
              builder.getDenseI64ArrayAttr(splits), mergedType,
              *finalizeInstruction, totalSlots, totalTerms, totalResources));
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
      if (auto collect = mlir::dyn_cast<riscv::RVVPartialCollectOp>(operation))
        nextPartialBirthId =
            std::max(nextPartialBirthId,
                     static_cast<int64_t>(collect.getBirthId()) + 1);
    });

    llvm::SmallVector<riscv::ExtractOp> extracts;
    getOperation().walk(
        [&](riscv::ExtractOp extract) { extracts.push_back(extract); });
    for (riscv::ExtractOp extract : extracts) {
      auto candidate = analyzeIssueStorageWindowCandidate(rewriter, extract);
      if (!candidate)
        continue;
      rewriter.setInsertionPoint(extract);
      mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(
          extract.getLoc(), 0);
      llvm::StringRef tail = candidate->resultType.getLayout().getValidity() == "tail"
                                 ? "agnostic"
                                 : "exact";
      auto window = rewriter.create<riscv::RVVStorageWindowOp>(
          extract.getLoc(), candidate->resultType, candidate->field.getResult(),
          candidate->point.getResult(), zero, candidate->plan,
          extract.getAccess(),
          riscv_internal::leaf(rewriter, "rvv", "storage-window",
                               "rvv.storage-window.layered",
                               "rvv.storage-window.layered",
                               mlir::cast<riscv::ValueType>(
                                   candidate->field.getResult().getType())
                                   .getLayout()
                                   .getRegisterGroups(),
                               candidate->resultType.getLayout().getRegisterGroups(),
                               0, 0,
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

    llvm::SmallVector<riscv::ReduceOp> nestedScaledReductions;
    llvm::DenseSet<mlir::Operation *> matchedNestedDots;
    getOperation().walk([&](riscv::ReduceOp reduce) {
      auto match = matchReplicaScaledDotReduction(reduce);
      auto plan = match ? match->dot.getNestedPartialPlanAttr()
                        : riscv::NestedPartialPlanAttr();
      if (match && plan && hasTopology(match->dot, "nested_scaled_stream") &&
          match->reducedAxes == llvm::SmallVector<int64_t>(
                                     plan.getPartialAxes().asArrayRef())) {
        nestedScaledReductions.push_back(reduce);
        matchedNestedDots.insert(match->dot.getOperation());
      }
    });
    getOperation().walk([&](riscv::RVVWidenDotOp dot) {
      if (!hasTopology(dot, "nested_scaled_stream") ||
          matchedNestedDots.contains(dot.getOperation()))
        return;
      auto diagnostic = dot.emitError(
          "nested topology lost its scale/reduction chain before materialization; users=");
      for (mlir::Operation *user : dot.getResult().getUsers())
        diagnostic << user->getName().getStringRef() << " ";
    });
    llvm::SmallVector<mlir::Value> nestedDeadRoots;
    for (riscv::ReduceOp reduce : nestedScaledReductions) {
      if (!reduce || !reduce->getBlock())
        continue;
      auto match = matchReplicaScaledDotReduction(reduce);
      if (!match || match->reducedAxes.empty() || match->dot.getOver().size() < 2)
        continue;
      riscv::RVVWidenDotOp dot = match->dot;
      auto nestedPlan = dot.getNestedPartialPlanAttr();
      const int64_t windowAxis =
          nestedPlan ? nestedPlan.getWindowAxis() : int64_t{0};
      const int64_t streams =
          nestedPlan ? nestedPlan.getIssueStreams() : int64_t{0};
      const int64_t windowExtent =
          nestedPlan ? nestedPlan.getWindowExtent() : int64_t{0};
      const int64_t partialSlots =
          nestedPlan ? nestedPlan.getPartialSlots() : int64_t{0};
      auto partialAxes = nestedPlan ? nestedPlan.getPartialAxes()
                                    : mlir::DenseI64ArrayAttr();
      auto partialAxisExtents =
          nestedPlan ? nestedPlan.getPartialAxisExtents()
                     : mlir::DenseI64ArrayAttr();
      auto combineSetTypes = nestedPlan ? nestedPlan.getCombineSetTypes()
                                        : mlir::ArrayAttr();
      auto combineAxes = nestedPlan ? nestedPlan.getCombineAxes()
                                    : mlir::DenseI64ArrayAttr();
      auto combineArities = nestedPlan ? nestedPlan.getCombineArities()
                                       : mlir::DenseI64ArrayAttr();
      auto issueLhsType =
          nestedPlan ? mlir::dyn_cast<riscv::ValueType>(
                           nestedPlan.getIssueLhsType())
                     : riscv::ValueType();
      auto issueRhsType =
          nestedPlan ? mlir::dyn_cast<riscv::ValueType>(
                           nestedPlan.getIssueRhsType())
                     : riscv::ValueType();
      auto scaleReplicaType =
          nestedPlan ? mlir::dyn_cast<riscv::ValueType>(
                           nestedPlan.getScaleReplicaType())
                     : riscv::ValueType();
      auto sourceSetType =
          nestedPlan ? mlir::dyn_cast<riscv::PartialSetType>(
                           nestedPlan.getSourceSetType())
                     : riscv::PartialSetType();
      auto repackedSetType =
          nestedPlan ? mlir::dyn_cast<riscv::PartialSetType>(
                           nestedPlan.getRepackedSetType())
                     : riscv::PartialSetType();
      auto reducedSetType =
          nestedPlan ? mlir::dyn_cast<riscv::PartialSetType>(
                           nestedPlan.getReducedSetType())
                     : riscv::PartialSetType();
      auto scaleCombinedSetType =
          nestedPlan ? mlir::dyn_cast<riscv::PartialSetType>(
                           nestedPlan.getScaleCombinedSetType())
                     : riscv::PartialSetType();
      mlir::Type finalType = reduce.getResult().getType();
      auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(finalType));
      auto finalValue = mlir::dyn_cast<riscv::ValueType>(finalType);
      const int64_t outputReplicas =
          nestedPlan ? nestedPlan.getOutputReplicas() : int64_t{0};
      auto finalReplicas =
          finalValue ? riscv_internal::staticProduct(
                           finalValue.getLayout().getReplicaFactors().asArrayRef())
                     : std::optional<int64_t>(1);
      const bool closedScalarFinal =
          !finalValue ||
          (finalValue.getLayout().getCarrier() == "scalar" &&
           product(finalValue.getLayout().getTimeFactors()) == 1 &&
           product(finalValue.getLayout().getLaneFactors()) == 1 &&
           product(finalValue.getLayout().getReplicaFactors()) > 0 &&
           product(finalValue.getLayout().getFragmentFactors()) == 1 &&
           product(finalValue.getLayout().getLocalFactors()) == 1);
      if (!nestedPlan || windowAxis <= 0 || streams <= 0 || windowExtent <= 0 ||
          partialSlots < windowExtent || partialSlots % windowExtent ||
          outputReplicas <= 0 || !finalReplicas ||
          *finalReplicas != outputReplicas ||
          nestedPlan.getLhsSourceParts().size() !=
              static_cast<size_t>(outputReplicas) ||
          nestedPlan.getRhsSourceParts().size() !=
              static_cast<size_t>(outputReplicas) ||
          nestedPlan.getLhsLaneOffsets().size() !=
              static_cast<size_t>(outputReplicas) ||
          nestedPlan.getRhsLaneOffsets().size() !=
              static_cast<size_t>(outputReplicas) ||
          nestedPlan.getScaleReplicas().size() !=
              static_cast<size_t>(outputReplicas * partialSlots) ||
          !partialAxes || !partialAxisExtents || !combineSetTypes ||
          !combineAxes || !combineArities ||
          partialAxes.size() != partialAxisExtents.size() ||
          combineSetTypes.size() != combineAxes.size() ||
          combineSetTypes.size() != combineArities.size() ||
          !issueLhsType || !issueRhsType || !scaleReplicaType ||
          !sourceSetType || !repackedSetType || !reducedSetType ||
          !scaleCombinedSetType || !resultElement || !resultElement.isSigned() ||
          resultElement.getWidth() != 32 || !closedScalarFinal) {
        dot.emitError(
            "nested topology reached materialization without its complete typed carrier plan");
        failed = true;
        continue;
      }

      rewriter.setInsertionPoint(reduce);
      auto zero = rewriter.create<riscv::ConstantOp>(
          reduce.getLoc(), resultElement,
          rewriter.getIntegerAttr(resultElement, 0));
      llvm::SmallVector<mlir::Value> initial(
          static_cast<size_t>(outputReplicas), zero.getResult());
      auto lower = rewriter.create<mlir::arith::ConstantIndexOp>(
          reduce.getLoc(), 0);
      auto upper = rewriter.create<mlir::arith::ConstantIndexOp>(
          reduce.getLoc(), streams);
      auto step = rewriter.create<mlir::arith::ConstantIndexOp>(
          reduce.getLoc(), 1);
      auto loop = rewriter.create<mlir::scf::ForOp>(
          reduce.getLoc(), lower, upper, step, initial,
          [&](mlir::OpBuilder &builder, mlir::Location location,
              mlir::Value, mlir::ValueRange carried) {
            builder.create<mlir::scf::YieldOp>(location, carried);
          });
      loop->setAttr("weft.riscv.direction",
                    rewriter.getStringAttr("ascending"));
      loop->setAttr("weft.riscv.system_unroll",
                    rewriter.getStringAttr("disable"));
      if (nestedPlan.getIssueUnroll() > 1) {
        loop->setAttr("weft.riscv.unroll_factor",
                      rewriter.getI64IntegerAttr(nestedPlan.getIssueUnroll()));
        auto sourceSet =
            mlir::dyn_cast<riscv::PartialSetType>(nestedPlan.getSourceSetType());
        auto kernel = dot->getParentOfType<riscv::KernelOp>();
        // Operation-major ordering is closed only when every issue owns one
        // complete product carrier.  Reordering a multi-lane issue window
        // extends several partial lifetimes at once instead of merely grouping
        // equivalent target states.
        if (sourceSet && kernel && nestedPlan.getWindowExtent() == 1 &&
            nestedPlan.getIssueUnroll() * sourceSet.getResourceGroups() <
                kernel.getTarget().getVectorRegisters())
          loop->setAttr("weft.riscv.unroll_order",
                        rewriter.getStringAttr("operation-major"));
      }
      if (auto defaultYield = mlir::dyn_cast<mlir::scf::YieldOp>(
              loop.getBody()->getTerminator()))
        rewriter.eraseOp(defaultYield);
      rewriter.setInsertionPointToStart(loop.getBody());

      llvm::DenseMap<mlir::Value, mlir::Value> clones;
      auto target = dot->getParentOfType<riscv::KernelOp>().getTarget();
      auto materializeIssueOperand = [&](mlir::Value source,
                                         riscv::ValueType planned)
          -> mlir::FailureOr<mlir::Value> {
        auto sourceType = mlir::dyn_cast<riscv::ValueType>(source.getType());
        // One issue with an output cohort already consumes the operand's
        // complete logical domain.  Keep that SSA supply and let the explicit
        // layout edge below move its output replicas into the planned carrier;
        // cloning its internal storage windows would project coordinates that
        // are not being iterated.
        if (streams == 1 && outputReplicas > 1 && sourceType &&
            sourceType.getShape() == planned.getShape() &&
            sourceType.getAxisIds() == planned.getAxisIds())
          return source;
        return cloneIssueWindow(source, windowAxis, windowExtent,
                                loop.getInductionVar(), target, rewriter,
                                clones);
      };
      auto lhsSlice = materializeIssueOperand(dot.getLhs(), issueLhsType);
      auto rhsSlice = materializeIssueOperand(dot.getRhs(), issueRhsType);
      auto lhsType = mlir::succeeded(lhsSlice)
                         ? mlir::dyn_cast<riscv::ValueType>((*lhsSlice).getType())
                         : riscv::ValueType();
      auto rhsType = mlir::succeeded(rhsSlice)
                         ? mlir::dyn_cast<riscv::ValueType>((*rhsSlice).getType())
                         : riscv::ValueType();
      if (mlir::failed(lhsSlice) || mlir::failed(rhsSlice) || !lhsType ||
          !rhsType) {
        dot.emitError(
            "nested issue-window cloning did not produce typed RVV operands; lhs=")
            << lhsType << ", planned_lhs=" << issueLhsType
            << ", rhs=" << rhsType << ", planned_rhs=" << issueRhsType;
        rewriter.eraseOp(loop);
        failed = true;
        continue;
      }
      auto materializeIssueLayout = [&](mlir::Value value,
                                        riscv::ValueType source,
                                        riscv::ValueType planned) -> mlir::Value {
        if (source == planned)
          return value;
        auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
            dot.getLoc(), planned, value,
            riscv_internal::layoutConversion(rewriter, source.getLayout(),
                                             planned.getLayout()),
            riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
        if (mlir::Operation *definition = value.getDefiningOp())
          riscv_internal::copyOrigin(definition, conversion);
        return conversion.getResult();
      };
      mlir::Value issueLhs =
          materializeIssueLayout(*lhsSlice, lhsType, issueLhsType);
      mlir::Value issueRhs =
          materializeIssueLayout(*rhsSlice, rhsType, issueRhsType);
      auto scale = cloneIssueWindow(match->scale, windowAxis, windowExtent,
                                    loop.getInductionVar(), target, rewriter,
                                    clones);
      auto scaleSourceType =
          mlir::succeeded(scale)
              ? mlir::dyn_cast<riscv::ValueType>((*scale).getType())
              : riscv::ValueType();
      if (mlir::failed(scale) || !scaleSourceType) {
        dot.emitError(
            "nested issue-window materialization rejected its planned scale supply");
        rewriter.eraseOp(loop);
        failed = true;
        continue;
      }
      mlir::Value scalarScale;
      if (nestedPlan.getScaleSupply() == "scalar-rematerialize") {
        llvm::DenseMap<mlir::Value, mlir::Value> rematerialized;
        auto rematerializedScale =
            rematerializeScalarReplicas(*scale, rewriter, rematerialized);
        if (mlir::failed(rematerializedScale) ||
            (*rematerializedScale).getType() != scaleReplicaType) {
          dot.emitError(
              "nested carrier materialization disagrees with its selected scalar scale supply");
          rewriter.eraseOp(loop);
          failed = true;
          continue;
        }
        scalarScale = *rematerializedScale;
        llvm::DenseSet<mlir::Value> stops;
        // A pure layout conversion may rematerialize to an existing scalar
        // producer from the old scale chain.  Keep that selected SSA owner
        // alive while deleting the now-dead vector representation; otherwise
        // the scale-combine op below would receive a dangling Value.
        stops.insert(scalarScale);
        llvm::DenseSet<mlir::Operation *> candidates;
        collectDeadChainCandidates(*scale, stops, candidates);
        sweepDeadChainCandidates(candidates, rewriter);
      } else if (scaleSourceType == scaleReplicaType) {
        scalarScale = *scale;
      } else {
        auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
            reduce.getLoc(), scaleReplicaType, *scale,
            riscv_internal::layoutConversion(rewriter,
                                             scaleSourceType.getLayout(),
                                             scaleReplicaType.getLayout()),
            riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
        if (mlir::Operation *definition = (*scale).getDefiningOp())
          riscv_internal::copyOrigin(definition, conversion);
        scalarScale = conversion.getResult();
      }
      const int64_t reductionAxis = sourceSetType.getReductionAxis();
      auto zeroOperand = rewriter.getDenseI64ArrayAttr({0});
      llvm::SmallVector<int64_t> windowOrder;
      windowOrder.reserve(static_cast<size_t>(partialSlots));
      for (int64_t window = 0; window < partialSlots; ++window)
        windowOrder.push_back(window);
      llvm::SmallVector<mlir::Value> accumulatedOutputs;
      accumulatedOutputs.reserve(static_cast<size_t>(outputReplicas));
      for (int64_t output = 0; output < outputReplicas; ++output) {
        auto lhsPart = rewriter.getDenseI64ArrayAttr(
            nestedPlan.getLhsSourceParts().asArrayRef().slice(output, 1));
        auto rhsPart = rewriter.getDenseI64ArrayAttr(
            nestedPlan.getRhsSourceParts().asArrayRef().slice(output, 1));
        auto lhsOffset = rewriter.getDenseI64ArrayAttr(
            nestedPlan.getLhsLaneOffsets().asArrayRef().slice(output, 1));
        auto rhsOffset = rewriter.getDenseI64ArrayAttr(
            nestedPlan.getRhsLaneOffsets().asArrayRef().slice(output, 1));
        auto sourceSet = rewriter.create<riscv::RVVPartialSetOp>(
            dot.getLoc(), sourceSetType, mlir::ValueRange{issueLhs},
            mlir::ValueRange{issueRhs}, zeroOperand, zeroOperand, lhsPart,
            rhsPart, lhsOffset, rhsOffset, reductionAxis,
            nestedPlan.getMultiplyInstruction(), ownerDomain(reduce),
            nextPartialBirthId++, ownerDomain(reduce),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-set", "rvv.partial-set",
                "rvv.partial-set",
                issueLhsType.getLayout().getRegisterGroups() +
                    issueRhsType.getLayout().getRegisterGroups(),
                sourceSetType.getResourceGroups(), 0, 0, "none", "exact",
                {reductionAxis, 1}));
        riscv_internal::copyOrigin(dot, sourceSet);
        sinkReplicaSupplies(sourceSet);
        mlir::Value reductionInput = sourceSet.getResult();
        if (partialSlots > 1) {
          auto repacked = rewriter.create<riscv::RVVPartialRepackOp>(
              dot.getLoc(), repackedSetType, sourceSet.getResult(), partialSlots,
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-repack",
                  "rvv.partial-repack.split", "rvv.partial-repack.split",
                  sourceSetType.getResourceGroups(),
                  repackedSetType.getResourceGroups(), 0, 0, "none", "exact",
                  {reductionAxis, partialSlots}));
          riscv_internal::copyOrigin(dot, repacked);
          reductionInput = repacked.getResult();
        }
        auto scaleParts = rewriter.getDenseI64ArrayAttr(
            nestedPlan.getScaleReplicas().asArrayRef().slice(
                output * partialSlots, partialSlots));
        auto partialReduced = rewriter.create<riscv::RVVPartialReduceOp>(
            dot.getLoc(), reducedSetType, reductionInput,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-reduce",
                nestedPlan.getReduceInstruction(),
                nestedPlan.getReduceInstruction(),
                repackedSetType.getResourceGroups(),
                reducedSetType.getResourceGroups(), 1, 0, "none", "exact",
                {reductionAxis, partialSlots}));
        riscv_internal::copyOrigin(dot, partialReduced);
        llvm::SmallVector<mlir::Value> scales(
            static_cast<size_t>(partialSlots), scalarScale);
        auto combined = rewriter.create<riscv::RVVPartialScaleCombineOp>(
            reduce.getLoc(), scaleCombinedSetType, partialReduced.getResult(),
            scales, scaleParts, rewriter.getDenseI64ArrayAttr(windowOrder),
            rewriter.getDenseI64ArrayAttr({partialAxes.asArrayRef().back()}),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-scale-combine",
                "rvv.partial-scale-combine", "rvv.partial-scale-combine",
                reducedSetType.getResourceGroups(),
                scaleCombinedSetType.getResourceGroups(), 1, 0, "none",
                "exact",
                {reductionAxis, partialSlots, scaleCombinedSetType.getSlots(),
                 partialSlots / scaleCombinedSetType.getSlots()}));
        riscv_internal::copyOrigin(dot, combined);
        mlir::Value combinedPartials = combined.getResult();
        for (auto [typeAttr, logicalAxis, arity] :
             llvm::zip(combineSetTypes, combineAxes.asArrayRef(),
                       combineArities.asArrayRef())) {
          auto combinedType = mlir::cast<riscv::PartialSetType>(
              mlir::cast<mlir::TypeAttr>(typeAttr).getValue());
          auto topologyCombined =
              rewriter.create<riscv::RVVPartialCombineOp>(
                  reduce.getLoc(), combinedType, combinedPartials, arity,
                  "pairwise", rewriter.getDenseI64ArrayAttr({logicalAxis}),
                  riscv_internal::leaf(
                      rewriter, "rvv", "partial-combine",
                      "rvv.partial-combine", "rvv.partial-combine",
                      mlir::cast<riscv::PartialSetType>(
                          combinedPartials.getType())
                          .getResourceGroups(),
                      combinedType.getResourceGroups(), 0, 0, "none", "exact",
                      {reductionAxis, arity, combinedType.getSlots()}));
          riscv_internal::copyOrigin(dot, topologyCombined);
          combinedPartials = topologyCombined.getResult();
        }
        auto finalSetType =
            mlir::cast<riscv::PartialSetType>(combinedPartials.getType());
        auto finalized = rewriter.create<riscv::RVVPartialFinalizeOp>(
            reduce.getLoc(), resultElement, combinedPartials, reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-finalize",
                nestedPlan.getFinalizeInstruction(),
                nestedPlan.getFinalizeInstruction(),
                finalSetType.getResourceGroups(), 0, 1, 0, "none",
                "exact", {reductionAxis, 1}));
        riscv_internal::copyOrigin(dot, finalized);
        auto accumulated = rewriter.create<riscv::BinaryOp>(
            reduce.getLoc(), resultElement, loop.getRegionIterArg(output),
            finalized.getResult(), "add",
            riscv_internal::unselectedLeaf(rewriter));
        accumulatedOutputs.push_back(accumulated.getResult());
      }
      if (failed) {
        rewriter.eraseOp(loop);
        continue;
      }
      rewriter.create<mlir::scf::YieldOp>(reduce.getLoc(), accumulatedOutputs);

      rewriter.setInsertionPointAfter(loop);
      nestedDeadRoots.push_back(reduce.getInput());
      mlir::Value replacement;
      if (finalValue) {
        auto assembled = rewriter.create<riscv::RVVAssembleReplicasOp>(
            reduce.getLoc(), finalValue, loop.getResults(),
            riscv_internal::leaf(rewriter, "scalar", "assemble-replicas",
                                 "scalar.assemble-replicas",
                                 "scalar.assemble-replicas", 0, 0));
        riscv_internal::copyOrigin(reduce, assembled);
        replacement = assembled.getResult();
      } else {
        replacement = loop.getResult(0);
      }
      reduce.getResult().replaceAllUsesWith(replacement);
      rewriter.eraseOp(reduce);
    }
    llvm::DenseSet<mlir::Value> nestedCleanupStops;
    llvm::DenseSet<mlir::Operation *> nestedCleanupCandidates;
    for (mlir::Value root : nestedDeadRoots)
      collectDeadChainCandidates(root, nestedCleanupStops,
                                 nestedCleanupCandidates);
    sweepDeadChainCandidates(nestedCleanupCandidates, rewriter);

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
      auto scalePlan = dot.getScaledPartialPlanAttr();
      auto slotType =
          layoutPlan ? mlir::dyn_cast<riscv::ValueType>(
                           layoutPlan.getPartialSlotType())
                     : riscv::ValueType();
      bool complete = layoutPlan && scalePlan &&
                      scalePlan.getRealization() ==
                          dot.getPartialTopology().getKind() &&
                      dotResult && scaledType && resultElement &&
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
      if (!complete)
        continue;

      rewriter.setInsertionPoint(reduce);
      mlir::Value scalarScale;
      auto plannedScalarScale = mlir::dyn_cast<riscv::ValueType>(
          scalePlan.getScalarScaleType());
      if (scalePlan.getScaleSupply() == "scalar-rematerialize") {
        llvm::DenseMap<mlir::Value, mlir::Value> rematerialized;
        auto rematerializedScale = rematerializeScalarReplicas(
            match->scale, rewriter, rematerialized);
        if (mlir::succeeded(rematerializedScale) &&
            (*rematerializedScale).getType() == plannedScalarScale)
          scalarScale = *rematerializedScale;
      } else if (scalePlan.getScaleSupply() == "vector-convert") {
        auto source =
            mlir::dyn_cast<riscv::ValueType>(match->scale.getType());
        if (source && plannedScalarScale) {
          if (source == plannedScalarScale) {
            scalarScale = match->scale;
          } else {
          auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
              reduce.getLoc(), plannedScalarScale, match->scale,
              riscv_internal::layoutConversion(rewriter, source.getLayout(),
                                               plannedScalarScale.getLayout()),
              riscv::AccessAttr(),
              riscv_internal::unselectedLeaf(rewriter));
          if (mlir::Operation *definition = match->scale.getDefiningOp())
            riscv_internal::copyOrigin(definition, conversion);
          scalarScale = conversion.getResult();
          }
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
      if (!scalarScaleType || scalarScaleType != plannedScalarScale ||
          scalarScaleType.getLayout().getCarrier() != "scalar" ||
          !scaleParts || *scaleParts < slots || !scaleElement ||
          !scaleElement.isSigned() || scaleElement.getWidth() != 32)
        continue;

      auto kernel = dot->getParentOfType<riscv::KernelOp>();
      auto widenedSlot = mlir::dyn_cast<riscv::ValueType>(
          layoutPlan.getWidenedSlotType());
      auto plannedNarrowScale = mlir::dyn_cast<riscv::ValueType>(
          scalePlan.getNarrowScaleType());
      auto narrowScale = plannedNarrowScale
                             ? materializeExactNarrowScale(scalarScale, rewriter)
                             : std::optional<mlir::Value>();
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
          (widenedSlot && narrowScale && narrowScaleType == plannedNarrowScale &&
           narrowScaleParts &&
           *narrowScaleParts >= slots && scaleChunkCount > 0 && setGroups > 0 &&
           scaledGroups > 0 && scaledSetType && fullScaledSetType);
      if (!kernel || !wideSourceLegal || !narrowScaleLegal ||
          scalePlan.getResourceGroups() >
              kernel.getTarget().getVectorRegisters())
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
              rewriter.getDenseI64ArrayAttr(partialAxes),
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
              scalePlan.getMultiplyInstruction(),
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
                  scalePlan.getReduceInstruction(),
                  scalePlan.getReduceInstruction(),
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
              scalePlan.getMultiplyInstruction(),
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
                  scalePlan.getReduceInstruction(),
                  scalePlan.getReduceInstruction(),
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
                reductionAxis, scalePlan.getMultiplyInstruction(),
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
                rewriter.getDenseI64ArrayAttr(partialAxes),
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
        auto finalized = rewriter.create<riscv::RVVPartialFinalizeOp>(
            reduce.getLoc(), resultElement, partials, reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-finalize",
                scalePlan.getFinalizeInstruction(),
                scalePlan.getFinalizeInstruction(),
                finalType.getResourceGroups(), 0, 0, 0, "none",
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

    // The planner sees one unreplicated loop-carried contribution and freezes
    // the complete issue cohort before any cloning.  Expand that exact cohort
    // here so the materializer consumes its typed plan rather than asking the
    // generic Level unroller to duplicate an unplanned reduction graph.
    llvm::SmallVector<mlir::scf::ForOp> levelScaledSeeds;
    getOperation().walk([&](mlir::scf::ForOp loop) {
      auto seed = matchLevelScaledLoopSeed(loop);
      if (seed && hasTopology(seed->dot, "level_scaled"))
        levelScaledSeeds.push_back(loop);
    });
    for (mlir::scf::ForOp loop : levelScaledSeeds) {
      auto seed = matchLevelScaledLoopSeed(loop);
      const int64_t factor = seed ? seed->dot.getPartialTopology().getPartialSlots()
                                  : int64_t{0};
      auto expanded = seed ? expandPlannedLevelScaledIssueLoop(loop, factor,
                                                               rewriter)
                           : mlir::FailureOr<mlir::scf::ForOp>(mlir::failure());
      if (mlir::failed(expanded)) {
        loop.emitError(
            "selected level-scaled topology cannot mechanically expand its exact issue cohort");
        failed = true;
      }
    }
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
      mlir::Type carry = loop.getRegionIterArg(0).getType();
      auto carryType = mlir::dyn_cast<riscv::ValueType>(carry);
      auto carryElement = mlir::dyn_cast<mlir::IntegerType>(
          riscv_internal::logicalElement(carry));
      auto carryParts =
          carryType
              ? riscv_internal::staticProduct(
                    carryType.getLayout().getReplicaFactors().asArrayRef())
              : carryElement && carryElement.isSigned() &&
                        carryElement.getWidth() == 32
                    ? std::optional<int64_t>(1)
                    : std::optional<int64_t>();
      auto firstLayoutPlan =
          contributions.front().dot.getPartialLayoutPlanAttr();
      auto firstTopology = contributions.front().dot.getPartialTopology();
      auto firstCombinePlan =
          contributions.front().dot->getAttrOfType<riscv::PartialCombinePlanAttr>(
              "partial_combine_plan");
      auto slotType =
          firstLayoutPlan
              ? mlir::dyn_cast<riscv::ValueType>(
                    firstLayoutPlan.getPartialSlotType())
              : riscv::ValueType();
      const bool scalarCarry =
          carryType ? carryType.getLayout().getCarrier() == "scalar"
                    : static_cast<bool>(carryElement);
      bool complete = firstLayoutPlan && firstCombinePlan && scalarCarry &&
                      carryParts && *carryParts == 1 && carryElement &&
                      carryElement.isSigned() && carryElement.getWidth() == 32 &&
                      firstCombinePlan.getRealization() == "level_scaled" &&
                      firstCombinePlan.getOutputParts() == *carryParts &&
                      slotType;
      for (ScaledPartialContribution &contribution : contributions) {
        auto combinePlan = contribution.dot->getAttrOfType<
            riscv::PartialCombinePlanAttr>("partial_combine_plan");
        auto issuePlan = contribution.dot.getSequentialPartialPlanAttr();
        auto issueAccumulator =
            issuePlan
                ? mlir::dyn_cast<riscv::ValueType>(issuePlan.getAccumulatorType())
                : riscv::ValueType();
        complete &= contribution.dot.getPartialUnroll() == slots &&
                    contribution.dot.getOver().size() == 1 &&
                    contribution.dot.getOver()[0] == reductionAxis &&
                    contribution.dot.getPartialTopology() == firstTopology &&
                    contribution.dot.getResult().getType() == carry &&
                    contribution.scaleMultiply.getResult().getType() == carry &&
                    contribution.carryAdd.getResult().getType() == carry &&
                    issuePlan &&
                    issuePlan.getRealization() == "fused_partial" &&
                    issuePlan.getIssueCount() > 0 &&
                    issueAccumulator == slotType &&
                    contribution.dot.getPartialLayoutPlanAttr() &&
                    combinePlan &&
                    combinePlan.getRealization() == "level_scaled" &&
                    combinePlan.getSourceSetType() ==
                        firstCombinePlan.getSourceSetType() &&
                    combinePlan.getReducedSetType() ==
                        firstCombinePlan.getReducedSetType() &&
                    combinePlan.getFinalSetType() ==
                        firstCombinePlan.getFinalSetType() &&
                    combinePlan.getMultiplyInstruction() ==
                        firstCombinePlan.getMultiplyInstruction() &&
                    combinePlan.getFinalizeInstruction() ==
                        firstCombinePlan.getFinalizeInstruction() &&
                    combinePlan.getLhsParts().size() ==
                        static_cast<size_t>(*carryParts) &&
                    combinePlan.getRhsParts().size() ==
                        static_cast<size_t>(*carryParts) &&
                    combinePlan.getScaleParts().size() ==
                        static_cast<size_t>(*carryParts) &&
                    mlir::dyn_cast<riscv::ValueType>(
                        contribution.dot.getPartialLayoutPlanAttr()
                            .getPartialSlotType()) == slotType;
      }
      if (!complete)
        continue;

      struct OutputPartialOperands {
        llvm::SmallVector<mlir::Value> partials;
        llvm::SmallVector<mlir::Value> scales;
        llvm::SmallVector<int64_t> scaleReplicas;
      };
      llvm::SmallVector<OutputPartialOperands, 1> outputOperands(*carryParts);
      auto setType = mlir::cast<riscv::PartialSetType>(
          firstCombinePlan.getSourceSetType());
      auto reducedType = mlir::cast<riscv::PartialSetType>(
          firstCombinePlan.getReducedSetType());
      auto combinedType = mlir::cast<riscv::PartialSetType>(
          firstCombinePlan.getFinalSetType());
      const int64_t chains = firstTopology.getCombineArity();
      if (chains <= 0 || slots % chains ||
          firstTopology.getSlotOrder().size() != static_cast<size_t>(slots))
        continue;
      const int64_t fanout = slots / chains;

      rewriter.setInsertionPoint(yield);
      for (ScaledPartialContribution &contribution : contributions) {
        auto combinePlan = contribution.dot->getAttrOfType<
            riscv::PartialCombinePlanAttr>("partial_combine_plan");
        auto issuePlan = contribution.dot.getSequentialPartialPlanAttr();
        auto lhsIssue = mlir::cast<riscv::ValueType>(issuePlan.getLhsIssueType());
        auto rhsIssue = mlir::cast<riscv::ValueType>(issuePlan.getRhsIssueType());
        const int64_t issueCount = issuePlan.getIssueCount();
        auto kernel = contribution.dot->getParentOfType<riscv::KernelOp>();
        auto materializeIssueOperand =
            [&](mlir::Value source, riscv::ValueType issueType,
                llvm::StringRef supply, mlir::DenseI64ArrayAttr sourceParts,
                mlir::DenseI64ArrayAttr laneOffsets, size_t index,
                int64_t issue) -> mlir::FailureOr<mlir::Value> {
          if (supply == "slice") {
            auto slice = rewriter.create<riscv::RVVIssueSliceOp>(
                yield.getLoc(), issueType, source,
                rewriter.getDenseI64ArrayAttr({sourceParts[index]}),
                rewriter.getDenseI64ArrayAttr({laneOffsets[index]}),
                riscv_internal::leaf(
                    rewriter, "transfer", "issue-slice", "rvv.issue-slice",
                    "rvv.issue-slice", 0,
                    issueType.getLayout().getRegisterGroups(), 0, 0, "none",
                    "exact"));
            riscv_internal::copyOrigin(contribution.dot, slice);
            return slice.getResult();
          }
          auto position = axisPosition(issueType, reductionAxis);
          if (supply != "storage-rematerialize" || !kernel || !position)
            return mlir::failure();
          auto issueIndex = rewriter.create<mlir::arith::ConstantIndexOp>(
              yield.getLoc(), issue);
          llvm::DenseMap<mlir::Value, mlir::Value> clones;
          auto projected = cloneIssueWindow(
              source, reductionAxis, issueType.getShape()[*position],
              issueIndex.getResult(), kernel.getTarget(), rewriter, clones);
          if (mlir::failed(projected) || (*projected).getType() != issueType)
            return mlir::failure();
          return *projected;
        };
        for (int64_t outputPart = 0; outputPart < *carryParts; ++outputPart) {
          mlir::Value partial;
          for (int64_t issue = 0; issue < issueCount; ++issue) {
            const size_t index =
                static_cast<size_t>(outputPart * issueCount + issue);
            auto lhsSlice = materializeIssueOperand(
                contribution.dot.getLhs(), lhsIssue,
                issuePlan.getLhsIssueSupply(),
                issuePlan.getLhsSourceParts(), issuePlan.getLhsLaneOffsets(),
                index, issue);
            auto rhsSlice = materializeIssueOperand(
                contribution.dot.getRhs(), rhsIssue,
                issuePlan.getRhsIssueSupply(),
                issuePlan.getRhsSourceParts(), issuePlan.getRhsLaneOffsets(),
                index, issue);
            if (mlir::failed(lhsSlice) || mlir::failed(rhsSlice)) {
              contribution.dot.emitError(
                  "selected issue supply cannot materialize its frozen storage/view program");
              failed = true;
              break;
            }
            if (!partial) {
              auto product = rewriter.create<riscv::RVVWidenMultiplyOp>(
                  yield.getLoc(), slotType, *lhsSlice, *rhsSlice,
                  riscv_internal::leaf(
                      rewriter, "rvv", "widen-multiply",
                      issuePlan.getMultiplyInstruction(),
                      issuePlan.getMultiplyInstruction(),
                      lhsIssue.getLayout().getRegisterGroups() +
                          rhsIssue.getLayout().getRegisterGroups(),
                      slotType.getLayout().getRegisterGroups(), 0, 0, "none",
                      "exact"));
              riscv_internal::copyOrigin(contribution.dot, product);
              partial = product.getResult();
            } else {
              auto accumulate = rewriter.create<riscv::RVVWidenAccumulateOp>(
                  yield.getLoc(), slotType, *lhsSlice, *rhsSlice, partial,
                  issuePlan.getReductionAxes(),
                  riscv_internal::leaf(
                      rewriter, "rvv", "widen-accumulate",
                      issuePlan.getAccumulateInstruction(),
                      issuePlan.getAccumulateInstruction(),
                      lhsIssue.getLayout().getRegisterGroups() +
                          rhsIssue.getLayout().getRegisterGroups() +
                          slotType.getLayout().getRegisterGroups(),
                      slotType.getLayout().getRegisterGroups(), 0, 0, "none",
                      "agnostic"));
              riscv_internal::copyOrigin(contribution.dot, accumulate);
              partial = accumulate.getResult();
            }
          }
          if (failed)
            break;
          OutputPartialOperands &operands = outputOperands[outputPart];
          operands.partials.push_back(partial);
          operands.scales.push_back(contribution.scale);
          operands.scaleReplicas.push_back(
              combinePlan.getScaleParts()[outputPart]);
        }
        if (failed)
          break;
      }
      if (failed)
        continue;

      llvm::SmallVector<mlir::Value> scalarParts;
      for (OutputPartialOperands &operands : outputOperands) {
        auto set = rewriter.create<riscv::RVVPartialCollectOp>(
            yield.getLoc(), setType, operands.partials,
            ownerDomain(yield), nextPartialBirthId++, ownerDomain(yield),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-collect", "rvv.partial-collect",
                "rvv.partial-collect", setType.getResourceGroups(),
                setType.getResourceGroups(), 0, 0, "none",
                slotType.getLayout().getValidity() == "tail" ? "agnostic"
                                                               : "exact",
                {reductionAxis, slots}));
        riscv_internal::copyOrigin(contributions.front().dot, set);
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
            firstTopology.getPartialAxes(),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-scale-combine",
                "rvv.partial-scale-combine", "rvv.partial-scale-combine",
                reducedType.getResourceGroups(),
                combinedType.getResourceGroups(), 1, 0, "none", "exact",
                {reductionAxis, slots, chains, fanout}));
        riscv_internal::copyOrigin(contributions.front().dot, combined);
        auto finalized = rewriter.create<riscv::RVVPartialFinalizeOp>(
            yield.getLoc(), carryElement, combined.getResult(),
            reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-finalize",
                firstCombinePlan.getFinalizeInstruction(),
                firstCombinePlan.getFinalizeInstruction(),
                combinedType.getResourceGroups(), 0, 0, 0, "none", "exact",
                {reductionAxis, chains}));
        riscv_internal::copyOrigin(contributions.front().dot, finalized);
        scalarParts.push_back(finalized.getResult());
      }
      mlir::Value materializedCarry;
      if (carryType) {
        auto assembled = rewriter.create<riscv::RVVAssembleReplicasOp>(
            yield.getLoc(), carryType, scalarParts,
            riscv_internal::leaf(rewriter, "scalar", "assemble-replicas",
                                 "scalar.assemble-replicas",
                                 "scalar.assemble-replicas", 0, 0));
        riscv_internal::copyOrigin(contributions.front().dot, assembled);
        materializedCarry = assembled.getResult();
      } else {
        materializedCarry = scalarParts.front();
      }
      riscv::BinaryOp finalAdd = contributions.back().carryAdd;
      finalAdd->moveBefore(yield);
      rewriter.modifyOpInPlace(finalAdd, [&] {
        finalAdd->setOperand(0, loop.getRegionIterArg(0));
        finalAdd->setOperand(1, materializedCarry);
      });

      llvm::SmallVector<mlir::Value> issueCleanupRoots;
      for (ScaledPartialContribution &contribution :
           llvm::reverse(contributions)) {
        issueCleanupRoots.push_back(contribution.dot.getLhs());
        issueCleanupRoots.push_back(contribution.dot.getRhs());
        if (contribution.carryAdd != finalAdd)
          rewriter.eraseOp(contribution.carryAdd);
        rewriter.eraseOp(contribution.scaleMultiply);
        rewriter.eraseOp(contribution.dot);
      }
      llvm::DenseSet<mlir::Operation *> issueCleanupVisited;
      llvm::DenseSet<mlir::Value> issueCleanupStops;
      for (mlir::Value root : issueCleanupRoots)
        eraseDeadChain(root, issueCleanupStops, issueCleanupVisited, rewriter);
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
            carried, rewriter.getDenseI64ArrayAttr({reductionAxis}),
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
          reduce.getLoc(), result, carried,
          rewriter.getDenseI64ArrayAttr({reductionAxis}),
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

    // Instantiate the already-selected sequential program.  Each issue slice
    // is a typed projection of an existing vector part.  A fused realization
    // carries one widened accumulator across issues; a per-stream realization
    // finalizes each issue and combines the explicit i32 SSA results.
    llvm::SmallVector<riscv::RVVWidenDotOp> sequentialDots;
    getOperation().walk([&](riscv::RVVWidenDotOp dot) {
      auto plan = dot.getSequentialPartialPlanAttr();
      if (plan && plan.getRealization() != "fused_partial")
        sequentialDots.push_back(dot);
    });
    for (riscv::RVVWidenDotOp dot : sequentialDots) {
      auto plan = dot.getSequentialPartialPlanAttr();
      auto lhsIssue = mlir::dyn_cast<riscv::ValueType>(plan.getLhsIssueType());
      auto rhsIssue = mlir::dyn_cast<riscv::ValueType>(plan.getRhsIssueType());
      auto accumulator =
          mlir::dyn_cast<riscv::ValueType>(plan.getAccumulatorType());
      auto partialElement =
          accumulator
              ? mlir::dyn_cast<mlir::IntegerType>(accumulator.getElementType())
              : mlir::IntegerType();
      const int64_t issueCount = plan.getIssueCount();
      const int64_t outputParts =
          issueCount > 0
              ? static_cast<int64_t>(plan.getLhsSourceParts().size()) /
                    issueCount
              : 0;
      auto resultValue =
          mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
      auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
          resultValue ? resultValue.getElementType() : dot.getResult().getType());
      const bool fused = plan.getRealization() == "fused";
      const bool perStream = plan.getRealization() == "per_stream";
      const bool matchingTopology =
          (fused && hasTopology(dot, "sequential_fused")) ||
          (perStream && hasTopology(dot, "sequential_per_stream"));
      if (!lhsIssue || !rhsIssue || !accumulator || !partialElement ||
          !resultElement || !resultElement.isSigned() ||
          resultElement.getWidth() != 32 || !matchingTopology ||
          issueCount <= 0 || outputParts <= 0 ||
          plan.getLhsSourceParts().size() !=
              plan.getRhsSourceParts().size() ||
          plan.getLhsSourceParts().size() != plan.getLhsLaneOffsets().size() ||
          plan.getRhsSourceParts().size() != plan.getRhsLaneOffsets().size()) {
        dot.emitError(
            "selected sequential topology lost its closed issue program");
        failed = true;
        continue;
      }

      rewriter.setInsertionPoint(dot);
      auto materializeIssueOperands = [&](int64_t output, int64_t issue) {
        const size_t index = static_cast<size_t>(output * issueCount + issue);
        auto lhsParts = rewriter.getDenseI64ArrayAttr(
            {plan.getLhsSourceParts()[index]});
        auto rhsParts = rewriter.getDenseI64ArrayAttr(
            {plan.getRhsSourceParts()[index]});
        auto lhsOffsets = rewriter.getDenseI64ArrayAttr(
            {plan.getLhsLaneOffsets()[index]});
        auto rhsOffsets = rewriter.getDenseI64ArrayAttr(
            {plan.getRhsLaneOffsets()[index]});
        auto lhsSlice = rewriter.create<riscv::RVVIssueSliceOp>(
            dot.getLoc(), lhsIssue, dot.getLhs(), lhsParts, lhsOffsets,
            riscv_internal::leaf(rewriter, "transfer", "issue-slice",
                                 "rvv.issue-slice", "rvv.issue-slice", 0,
                                 lhsIssue.getLayout().getRegisterGroups(), 0, 0,
                                 "none", "exact"));
        auto rhsSlice = rewriter.create<riscv::RVVIssueSliceOp>(
            dot.getLoc(), rhsIssue, dot.getRhs(), rhsParts, rhsOffsets,
            riscv_internal::leaf(rewriter, "transfer", "issue-slice",
                                 "rvv.issue-slice", "rvv.issue-slice", 0,
                                 rhsIssue.getLayout().getRegisterGroups(), 0, 0,
                                 "none", "exact"));
        riscv_internal::copyOrigin(dot, lhsSlice);
        riscv_internal::copyOrigin(dot, rhsSlice);
        return std::make_pair(lhsSlice.getResult(), rhsSlice.getResult());
      };
      auto materializeIssue = [&](int64_t output, int64_t issue,
                                  mlir::Value carried) -> mlir::Value {
        auto [lhsSlice, rhsSlice] = materializeIssueOperands(output, issue);
        auto accumulate = rewriter.create<riscv::RVVWidenAccumulateOp>(
            dot.getLoc(), accumulator, lhsSlice, rhsSlice, carried,
            plan.getReductionAxes(),
            riscv_internal::leaf(
                rewriter, "rvv", "widen-accumulate",
                plan.getAccumulateInstruction(),
                plan.getAccumulateInstruction(),
                lhsIssue.getLayout().getRegisterGroups() +
                    rhsIssue.getLayout().getRegisterGroups() +
                    accumulator.getLayout().getRegisterGroups(),
                accumulator.getLayout().getRegisterGroups(), 0, 0, "none",
                "agnostic"));
        riscv_internal::copyOrigin(dot, accumulate);
        return accumulate.getResult();
      };
      auto makeInitial = [&]() {
        auto zero = rewriter.create<riscv::ConstantOp>(
            dot.getLoc(), partialElement,
            rewriter.getIntegerAttr(partialElement, 0));
        auto initial = rewriter.create<riscv::RVVSplatOp>(
            dot.getLoc(), accumulator, zero.getResult(),
            riscv_internal::leaf(
                rewriter, "rvv", "splat", "rvv.splat", "rvv.splat", 0,
                accumulator.getLayout().getRegisterGroups()));
        riscv_internal::copyOrigin(dot, initial);
        return initial.getResult();
      };
      auto finalize = [&](mlir::Value partial) {
        auto finalized = rewriter.create<riscv::RVVFinalizeWidenDotOp>(
            dot.getLoc(), resultElement, partial,
            plan.getReductionAxes(),
            riscv_internal::leaf(
                rewriter, "rvv", "finalize-widen-dot",
                plan.getFinalizeInstruction(), plan.getFinalizeInstruction(),
                accumulator.getLayout().getRegisterGroups(), 0, 2, 0, "none",
                "exact", plan.getReductionAxes().asArrayRef()));
        riscv_internal::copyOrigin(dot, finalized);
        return finalized.getResult();
      };

      llvm::SmallVector<mlir::Value> materializedOutputs;
      for (int64_t output = 0; output < outputParts; ++output) {
        mlir::Value outputValue;
        if (issueCount == 1) {
          auto [lhsSlice, rhsSlice] = materializeIssueOperands(output, 0);
          auto product = rewriter.create<riscv::RVVWidenMultiplyOp>(
              dot.getLoc(), accumulator, lhsSlice, rhsSlice,
              riscv_internal::leaf(
                  rewriter, "rvv", "widen-multiply",
                  plan.getMultiplyInstruction(),
                  plan.getMultiplyInstruction(),
                  lhsIssue.getLayout().getRegisterGroups() +
                      rhsIssue.getLayout().getRegisterGroups(),
                  accumulator.getLayout().getRegisterGroups(), 0, 0, "none",
                  "exact"));
          riscv_internal::copyOrigin(dot, product);
          outputValue = finalize(product.getResult());
        } else if (fused) {
          mlir::Value carried = makeInitial();
          for (int64_t issue = 0; issue < issueCount; ++issue)
            carried = materializeIssue(output, issue, carried);
          outputValue = finalize(carried);
        } else {
          for (int64_t issue = 0; issue < issueCount; ++issue) {
            mlir::Value contribution = finalize(
                materializeIssue(output, issue, makeInitial()));
            if (!outputValue) {
              outputValue = contribution;
              continue;
            }
            auto combined = rewriter.create<riscv::BinaryOp>(
                dot.getLoc(), resultElement, outputValue, contribution, "add",
                riscv_internal::unselectedLeaf(rewriter));
            riscv_internal::copyOrigin(dot, combined);
            outputValue = combined.getResult();
          }
        }
        materializedOutputs.push_back(outputValue);
      }
      mlir::Value replacement;
      if (resultValue) {
        auto assembled = rewriter.create<riscv::RVVAssembleReplicasOp>(
            dot.getLoc(), resultValue, materializedOutputs,
            riscv_internal::leaf(rewriter, "scalar", "assemble-replicas",
                                 "scalar.assemble-replicas",
                                 "scalar.assemble-replicas", 0, 0));
        riscv_internal::copyOrigin(dot, assembled);
        replacement = assembled.getResult();
      } else {
        replacement = materializedOutputs.front();
      }
      dot.getResult().replaceAllUsesWith(replacement);
      rewriter.eraseOp(dot);
    }

    llvm::SmallVector<riscv::RVVWidenDotOp> independentDots;
    getOperation().walk([&](riscv::RVVWidenDotOp dot) {
      independentDots.push_back(dot);
    });
    for (riscv::RVVWidenDotOp dot : independentDots) {
      if (!hasTopology(dot, "independent") || dot.getOver().empty())
        continue;
      riscv::ValueType lhs = dot.getLhs().getType();
      riscv::ValueType rhs = dot.getRhs().getType();
      auto plan = dot->getAttrOfType<riscv::PartialCombinePlanAttr>(
          "partial_combine_plan");
      auto sourceSet =
          plan ? mlir::dyn_cast<riscv::PartialSetType>(plan.getSourceSetType())
               : riscv::PartialSetType();
      const int64_t reductionAxis =
          sourceSet ? sourceSet.getReductionAxis() : int64_t{0};
      auto finalSet =
          plan ? mlir::dyn_cast<riscv::PartialSetType>(plan.getFinalSetType())
               : riscv::PartialSetType();
      auto resultValue =
          mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
      auto resultElement = mlir::dyn_cast<mlir::IntegerType>(
          resultValue ? resultValue.getElementType() : dot.getResult().getType());
      if (!plan || !sourceSet || !finalSet || reductionAxis <= 0 ||
          !resultElement ||
          !resultElement.isSigned() || resultElement.getWidth() != 32 ||
          plan.getOutputParts() <= 0 ||
          plan.getLhsParts().size() != plan.getRhsParts().size() ||
          plan.getLhsParts().size() !=
              static_cast<size_t>(plan.getOutputParts() * sourceSet.getSlots()) ||
          plan.getCombineSetTypes().size() !=
              plan.getCombineArities().size()) {
        dot.emitError(
            "selected independent topology reached materialization without a closed combine plan");
        failed = true;
        continue;
      }
      riscv::ConvertLayoutOp vectorConsumer;
      riscv::ValueType vectorResult;
      if (plan.getRealization() == "independent_vector" &&
          dot.getResult().hasOneUse()) {
        vectorConsumer = mlir::dyn_cast<riscv::ConvertLayoutOp>(
            *dot.getResult().getUsers().begin());
        if (vectorConsumer &&
            vectorConsumer.getConversion().getEffect() == "pure" &&
            vectorConsumer.getConversion().getKind() == "register_to_lane")
          vectorResult =
              mlir::dyn_cast<riscv::ValueType>(vectorConsumer.getResult().getType());
      }
      const int64_t operandGroups = lhs.getLayout().getRegisterGroups() +
                                    rhs.getLayout().getRegisterGroups();
      if ((plan.getRealization() == "independent_vector") !=
              static_cast<bool>(vectorResult) ||
          (plan.getRealization() != "independent_vector" &&
           plan.getRealization() != "independent_scalar")) {
        dot.emitError(
            "selected independent realization no longer matches its consumer");
        failed = true;
        continue;
      }
      rewriter.setInsertionPoint(dot);
      llvm::SmallVector<mlir::Value> materializedParts;
      const int64_t slots = sourceSet.getSlots();
      for (int64_t outputPart = 0; outputPart < plan.getOutputParts();
           ++outputPart) {
        llvm::SmallVector<int64_t> operandSlots(slots, 0);
        auto lhsParts = plan.getLhsParts().asArrayRef().slice(
            static_cast<size_t>(outputPart * slots), static_cast<size_t>(slots));
        auto rhsParts = plan.getRhsParts().asArrayRef().slice(
            static_cast<size_t>(outputPart * slots), static_cast<size_t>(slots));
        auto set = rewriter.create<riscv::RVVPartialSetOp>(
            dot.getLoc(), sourceSet, mlir::ValueRange{dot.getLhs()},
            mlir::ValueRange{dot.getRhs()},
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(lhsParts),
            rewriter.getDenseI64ArrayAttr(rhsParts),
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(operandSlots), reductionAxis,
            plan.getMultiplyInstruction(),
            ownerDomain(dot), nextPartialBirthId++, ownerDomain(dot),
            riscv_internal::leaf(
                rewriter, "rvv", "partial-set", "rvv.partial-set",
                "rvv.partial-set", operandGroups,
                sourceSet.getResourceGroups(), 0, 0, "none", "exact",
                {reductionAxis, slots}));
        riscv_internal::copyOrigin(dot, set);
        sinkReplicaSupplies(set);
        mlir::Value partials = set.getResult();
        for (auto [typeAttr, arity] :
             llvm::zip(plan.getCombineSetTypes(),
                       plan.getCombineArities().asArrayRef())) {
          auto combinedType = mlir::cast<riscv::PartialSetType>(
              mlir::cast<mlir::TypeAttr>(typeAttr).getValue());
          auto combine = rewriter.create<riscv::RVVPartialCombineOp>(
              dot.getLoc(), combinedType, partials, arity, "pairwise",
              dot.getPartialTopology().getPartialAxes(),
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-combine",
                  "rvv.partial-combine", "rvv.partial-combine",
                  mlir::cast<riscv::PartialSetType>(partials.getType())
                      .getResourceGroups(),
                  combinedType.getResourceGroups(), 0, 0, "none", "exact",
                  {reductionAxis, arity, combinedType.getSlots()}));
          riscv_internal::copyOrigin(dot, combine);
          partials = combine.getResult();
        }
        mlir::Type finalizedType = vectorResult
                                       ? mlir::Type(vectorResult)
                                       : mlir::Type(resultElement);
        auto finalize = rewriter.create<riscv::RVVPartialFinalizeOp>(
            dot.getLoc(), finalizedType, partials, reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-finalize",
                plan.getFinalizeInstruction(), plan.getFinalizeInstruction(),
                mlir::cast<riscv::PartialSetType>(partials.getType())
                    .getResourceGroups(),
                vectorResult ? vectorResult.getLayout().getRegisterGroups() : 0,
                2, 0, "none", "exact", {reductionAxis, slots}));
        riscv_internal::copyOrigin(dot, finalize);
        materializedParts.push_back(finalize.getResult());
      }
      mlir::Value replacement;
      if (vectorResult) {
        replacement = materializedParts.front();
        vectorConsumer.getResult().replaceAllUsesWith(replacement);
        rewriter.eraseOp(vectorConsumer);
      } else if (resultValue) {
        auto assembled = rewriter.create<riscv::RVVAssembleReplicasOp>(
            dot.getLoc(), resultValue, materializedParts,
            riscv_internal::leaf(rewriter, "scalar", "assemble-replicas",
                                 "scalar.assemble-replicas",
                                 "scalar.assemble-replicas", 0, 0));
        riscv_internal::copyOrigin(dot, assembled);
        replacement = assembled.getResult();
      } else {
        replacement = materializedParts.front();
      }
      if (!vectorResult)
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
      auto plan = root->getAttrOfType<riscv::PartialAddTreePlanAttr>(
          "partial_add_tree_plan");
      if (!plan)
        continue;
      llvm::SmallVector<riscv::BinaryOp> adds;
      llvm::SmallVector<PartialAddLeaf> leaves;
      if (!collectPartialAddTree(root.getResult(), root, adds, leaves) ||
          leaves.size() != plan.getLeafSetTypes().size() ||
          leaves.size() != plan.getNormalizedSetTypes().size() ||
          leaves.size() != plan.getLeafMultiplyInstructions().size() ||
          leaves.size() != plan.getSplits().size()) {
        root.emitError(
            "partial add tree no longer matches its frozen typed leaf plan");
        failed = true;
        continue;
      }

      rewriter.setInsertionPoint(root);
      llvm::SmallVector<mlir::Value> normalized;
      bool complete = true;
      for (size_t index = 0; index < leaves.size(); ++index) {
        PartialAddLeaf leaf = leaves[index];
        auto sourceType = mlir::cast<riscv::PartialSetType>(
            mlir::cast<mlir::TypeAttr>(plan.getLeafSetTypes()[index])
                .getValue());
        auto normalizedType = mlir::cast<riscv::PartialSetType>(
            mlir::cast<mlir::TypeAttr>(plan.getNormalizedSetTypes()[index])
                .getValue());
        auto currentType = partialTypeForAddLeaf(rewriter, leaf);
        const int64_t split = plan.getSplits()[index];
        auto multiply = mlir::cast<mlir::StringAttr>(
            plan.getLeafMultiplyInstructions()[index]);
        if (!currentType || *currentType != sourceType) {
          complete = false;
          break;
        }
        mlir::Value value;
        mlir::Operation *origin = nullptr;
        if (leaf.finalize) {
          if (!multiply.getValue().empty()) {
            complete = false;
            break;
          }
          value = leaf.finalize.getInput();
          origin = leaf.finalize.getOperation();
        } else {
          if (multiply.getValue().empty()) {
            complete = false;
            break;
          }
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
              sourceType.getReductionAxis(), multiply.getValue(),
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
          auto repack = rewriter.create<riscv::RVVPartialRepackOp>(
              root.getLoc(), normalizedType, value, split,
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-repack",
                  "rvv.partial-repack.split", "rvv.partial-repack.split",
                  sourceType.getResourceGroups(),
                  normalizedType.getResourceGroups(), 0, 0, "none", "exact",
                  {sourceType.getReductionAxis(), split}));
          riscv_internal::copyOrigin(origin, repack);
          value = repack.getResult();
        } else if (normalizedType != sourceType) {
          complete = false;
          break;
        }
        normalized.push_back(value);
      }
      if (!complete || normalized.size() != leaves.size()) {
        root.emitError(
            "partial add tree materialization disagrees with its frozen typed plan");
        failed = true;
        continue;
      }
      auto combinedType =
          mlir::cast<riscv::PartialSetType>(plan.getMergedSetType());
      auto combine = rewriter.create<riscv::RVVPartialMergeOp>(
          root.getLoc(), combinedType, normalized, "pairwise",
          riscv_internal::leaf(
              rewriter, "rvv", "partial-merge", "rvv.partial-merge",
              "rvv.partial-merge", plan.getTotalResourceGroups(),
              combinedType.getResourceGroups(), 0, 0, "none", "exact",
              {combinedType.getReductionAxis(),
               static_cast<int64_t>(normalized.size()), plan.getTotalSlots(),
               plan.getTotalTerms()}));
      riscv_internal::copyOrigin(root, combine);
      auto finalized = rewriter.create<riscv::RVVPartialFinalizeOp>(
          root.getLoc(), root.getResult().getType(), combine.getResult(),
          combinedType.getReductionAxis(),
          riscv_internal::leaf(
              rewriter, "rvv", "partial-finalize",
              plan.getFinalizeInstruction(), plan.getFinalizeInstruction(),
              combinedType.getResourceGroups(), 0, 2, 0, "none", "exact",
              {combinedType.getReductionAxis(), 1}));
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
          plan.getRootStoragePlans().size() != roots.size() ||
          plan.getRootWindowInstructions().size() != roots.size()) {
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
      llvm::DenseMap<mlir::Value, mlir::StringAttr> windowInstructions;
      bool completeRootTypes = true;
      for (auto [root, typeAttributeValue, storagePlanValue,
                 instructionValue] :
           llvm::zip(roots, plan.getRootWindowTypes(),
                     plan.getRootStoragePlans(),
                     plan.getRootWindowInstructions())) {
        auto typeAttribute = mlir::dyn_cast<mlir::TypeAttr>(typeAttributeValue);
        auto windowType =
            typeAttribute
                ? mlir::dyn_cast<riscv::ValueType>(typeAttribute.getValue())
                : riscv::ValueType();
        auto storagePlan =
            mlir::dyn_cast<riscv::StorageWindowPlanAttr>(storagePlanValue);
        auto instruction = mlir::dyn_cast<mlir::StringAttr>(instructionValue);
        completeRootTypes &= static_cast<bool>(windowType) &&
                             static_cast<bool>(storagePlan) &&
                             static_cast<bool>(instruction);
        if (windowType && storagePlan && instruction) {
          windowTypes[root.value] = windowType;
          storagePlans[root.value] = storagePlan;
          windowInstructions[root.value] = instruction;
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
      if (plan.getDecodeInstructions().size() != static_cast<size_t>(layers)) {
        dot.emitError(
            "selected layered partial plan lost its decode instruction sequence");
        failed = true;
        continue;
      }
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
        auto instructionAttr = mlir::dyn_cast<mlir::StringAttr>(
            plan.getDecodeInstructions()[static_cast<size_t>(layer)]);
        if (!instructionAttr) {
          dot.emitError(
              "selected layered partial plan has an untyped decode instruction");
          failed = true;
          break;
        }
        llvm::StringRef instruction = instructionAttr.getValue();
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
          auto rootInstruction = windowInstructions.lookup(root.value);
          if (!rootPlan || !rootInstruction) {
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
                  rootInstruction.getValue(), rootInstruction.getValue(),
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
            rewriter.getDenseI64ArrayAttr({reductionAxis}),
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
          rewriter.getDenseI64ArrayAttr({reductionAxis}),
          riscv_internal::leaf(rewriter, "rvv", "finalize-widen-dot",
                               plan.getFinalizeInstruction(),
                               plan.getFinalizeInstruction(),
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

    // Every selected widening-dot must now be an explicit issue/product/
    // reduction program.  Leaving the composite here would make the terminal
    // emitter reconstruct hidden SSA values and their resource lifetime.
    getOperation().walk([&](riscv::RVVWidenDotOp dot) {
      dot.emitError(
          "selected widening dot was not materialized into the final physical program; topology=")
          << dot.getPartialTopology();
      failed = true;
    });

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
