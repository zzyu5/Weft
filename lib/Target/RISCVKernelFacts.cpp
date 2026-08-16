#include "RISCVKernelFacts.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/DenseSet.h"

#include <algorithm>
#include <optional>

namespace weft::riscv_internal {
namespace {

using namespace weft::kernel;

mlir::Type elementType(mlir::Type type) { return logicalElementType(type); }

bool isRegionValue(mlir::Type type) {
  return logicalShapeKind(type) == LogicalShapeKind::Region;
}

std::optional<int64_t> integerConstantValue(mlir::Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto integer = constant
                     ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                     : mlir::IntegerAttr{};
  return integer ? std::optional<int64_t>(integer.getInt()) : std::nullopt;
}

LaneRelation combineLaneRelations(LaneRelation lhs, LaneRelation rhs) {
  if (lhs == LaneRelation::NonAffine || rhs == LaneRelation::NonAffine)
    return LaneRelation::NonAffine;
  if (lhs == LaneRelation::Independent)
    return rhs;
  if (rhs == LaneRelation::Independent)
    return lhs;
  if (lhs == LaneRelation::Indexed || rhs == LaneRelation::Indexed)
    return lhs == rhs ? LaneRelation::Indexed : LaneRelation::NonAffine;
  return LaneRelation::Strided;
}

AffineScalarExpression constantExpression(int64_t value) {
  AffineScalarExpression expression;
  expression.kind = AffineScalarExpressionKind::Constant;
  expression.constant = value;
  return expression;
}

AffineScalarExpression valueExpression(mlir::Value value) {
  AffineScalarExpression expression;
  expression.kind = AffineScalarExpressionKind::Value;
  expression.value = value;
  return expression;
}

AffineScalarExpression binaryExpression(AffineScalarExpressionKind kind,
                                        AffineScalarExpression lhs,
                                        AffineScalarExpression rhs) {
  if (kind == AffineScalarExpressionKind::Add &&
      lhs.kind == AffineScalarExpressionKind::Constant && lhs.constant == 0)
    return rhs;
  if ((kind == AffineScalarExpressionKind::Add ||
       kind == AffineScalarExpressionKind::Subtract) &&
      rhs.kind == AffineScalarExpressionKind::Constant && rhs.constant == 0)
    return lhs;
  if (kind == AffineScalarExpressionKind::Multiply) {
    if ((lhs.kind == AffineScalarExpressionKind::Constant && lhs.constant == 0) ||
        (rhs.kind == AffineScalarExpressionKind::Constant && rhs.constant == 0))
      return constantExpression(0);
    if (lhs.kind == AffineScalarExpressionKind::Constant && lhs.constant == 1)
      return rhs;
    if (rhs.kind == AffineScalarExpressionKind::Constant && rhs.constant == 1)
      return lhs;
  }
  AffineScalarExpression expression;
  expression.kind = kind;
  expression.lhs = std::make_shared<const AffineScalarExpression>(std::move(lhs));
  expression.rhs = std::make_shared<const AffineScalarExpression>(std::move(rhs));
  return expression;
}

bool dependsOn(mlir::Value value, mlir::Value target,
               llvm::DenseSet<mlir::Value> &visited) {
  if (value == target)
    return true;
  if (!value || !visited.insert(value).second)
    return false;
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  return llvm::any_of(definition->getOperands(), [&](mlir::Value operand) {
    return dependsOn(operand, target, visited);
  });
}

bool dependsOn(mlir::Value value, mlir::Value target) {
  llvm::DenseSet<mlir::Value> visited;
  return dependsOn(value, target, visited);
}

std::optional<AffineScalarExpression>
deriveLaneStride(mlir::Value value, mlir::Value coordinate) {
  if (value == coordinate)
    return constantExpression(1);
  if (!dependsOn(value, coordinate))
    return constantExpression(0);
  if (auto pointer = value.getDefiningOp<PtrAddOp>()) {
    std::optional<AffineScalarExpression> base =
        deriveLaneStride(pointer.getBase(), coordinate);
    std::optional<AffineScalarExpression> offset =
        deriveLaneStride(pointer.getOffset(), coordinate);
    if (!base || !offset)
      return std::nullopt;
    return binaryExpression(AffineScalarExpressionKind::Add, std::move(*base),
                            std::move(*offset));
  }
  if (auto binary = value.getDefiningOp<BinaryOp>()) {
    if (binary.getKind() == "add" || binary.getKind() == "sub") {
      std::optional<AffineScalarExpression> lhs =
          deriveLaneStride(binary.getLhs(), coordinate);
      std::optional<AffineScalarExpression> rhs =
          deriveLaneStride(binary.getRhs(), coordinate);
      if (!lhs || !rhs)
        return std::nullopt;
      return binaryExpression(binary.getKind() == "add"
                                  ? AffineScalarExpressionKind::Add
                                  : AffineScalarExpressionKind::Subtract,
                              std::move(*lhs), std::move(*rhs));
    }
    if (binary.getKind() == "mul") {
      const bool lhsDependent = dependsOn(binary.getLhs(), coordinate);
      const bool rhsDependent = dependsOn(binary.getRhs(), coordinate);
      if (lhsDependent == rhsDependent)
        return std::nullopt;
      mlir::Value dependent =
          lhsDependent ? binary.getLhs() : binary.getRhs();
      mlir::Value factor = lhsDependent ? binary.getRhs() : binary.getLhs();
      std::optional<AffineScalarExpression> stride =
          deriveLaneStride(dependent, coordinate);
      if (!stride)
        return std::nullopt;
      AffineScalarExpression factorExpression =
          integerConstantValue(factor)
              ? constantExpression(*integerConstantValue(factor))
              : valueExpression(factor);
      return binaryExpression(AffineScalarExpressionKind::Multiply,
                              std::move(*stride),
                              std::move(factorExpression));
    }
    return std::nullopt;
  }
  if (auto cast = value.getDefiningOp<CastOp>())
    return deriveLaneStride(cast.getInput(), coordinate);
  if (auto expand = value.getDefiningOp<ExpandDimsOp>())
    return deriveLaneStride(expand.getInput(), coordinate);
  return std::nullopt;
}

mlir::Value pointerRoot(mlir::Value value) {
  if (mlir::isa<PtrType>(value.getType()))
    return value;
  auto pointer = value.getDefiningOp<PtrAddOp>();
  return pointer ? pointerRoot(pointer.getBase()) : mlir::Value{};
}

mlir::Value containingVLACoordinate(mlir::Value value) {
  mlir::Operation *anchor = value.getDefiningOp();
  if (!anchor) {
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    anchor = argument ? argument.getOwner()->getParentOp() : nullptr;
  }
  if (!anchor)
    return {};
  VLAOp vla = mlir::dyn_cast<VLAOp>(anchor);
  if (!vla)
    vla = anchor->getParentOfType<VLAOp>();
  return vla ? vla.getBody().front().getArgument(0) : mlir::Value{};
}

llvm::SmallVector<mlir::Value>
logicalAxesForValue(mlir::Value value, const KernelPhysicalFacts &facts) {
  llvm::SmallVector<mlir::Value> axes;
  mlir::Type logicalType = unwrapLogicalValidity(value.getType());
  bool regionValue = mlir::isa<RegionType>(logicalType);
  for (int64_t identity : logicalAxisIds(logicalType)) {
    mlir::Value axis;
    if (identity > 0) {
      auto found = facts.blockAxes.find(identity);
      if (found != facts.blockAxes.end())
        axis = found->second;
    } else if (regionValue && identity < 0) {
      axis = containingVLACoordinate(value);
    }
    if (axis && !llvm::is_contained(axes, axis))
      axes.push_back(axis);
  }
  return axes;
}

} // namespace

LaneRelation classifyLaneRelation(mlir::Value value, mlir::Value coordinate) {
  if (value == coordinate)
    return LaneRelation::UnitStride;
  if (value.getDefiningOp<BlockIndexOp>())
    return LaneRelation::Independent;
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Operation *parent = argument.getOwner()->getParentOp();
    if (mlir::isa_and_nonnull<VLAOp>(parent) && argument.getArgNumber() == 0)
      return LaneRelation::Independent;
  }
  if (auto pointer = value.getDefiningOp<PtrAddOp>())
    return combineLaneRelations(
        classifyLaneRelation(pointer.getBase(), coordinate),
        classifyLaneRelation(pointer.getOffset(), coordinate));
  if (auto binary = value.getDefiningOp<BinaryOp>()) {
    LaneRelation lhs = classifyLaneRelation(binary.getLhs(), coordinate);
    LaneRelation rhs = classifyLaneRelation(binary.getRhs(), coordinate);
    if (binary.getKind() == "add" || binary.getKind() == "sub")
      return combineLaneRelations(lhs, rhs);
    if (binary.getKind() == "mul") {
      if (lhs != LaneRelation::Independent && rhs != LaneRelation::Independent)
        return LaneRelation::NonAffine;
      LaneRelation dependent =
          lhs == LaneRelation::Independent ? rhs : lhs;
      mlir::Value scale =
          lhs == LaneRelation::Independent ? binary.getLhs() : binary.getRhs();
      if (dependent == LaneRelation::Independent)
        return LaneRelation::Independent;
      if (std::optional<int64_t> constant = integerConstantValue(scale)) {
        if (*constant == 0)
          return LaneRelation::Independent;
        if (*constant == 1)
          return dependent;
      }
      if (dependent == LaneRelation::Indexed)
        return LaneRelation::Indexed;
      return dependent == LaneRelation::NonAffine ? LaneRelation::NonAffine
                                                   : LaneRelation::Strided;
    }
    if (elementType(binary.getResult().getType()).isIndex() &&
        (lhs != LaneRelation::Independent || rhs != LaneRelation::Independent))
      return LaneRelation::Indexed;
  }
  if (auto select = value.getDefiningOp<SelectOp>())
    if (isRegionValue(select.getResult().getType()) &&
        elementType(select.getResult().getType()).isIndex())
      return LaneRelation::Indexed;
  if (auto expand = value.getDefiningOp<ExpandDimsOp>())
    return classifyLaneRelation(expand.getInput(), coordinate);
  if (auto cast = value.getDefiningOp<CastOp>()) {
    mlir::Type source = elementType(cast.getInput().getType());
    mlir::Type result = elementType(cast.getResult().getType());
    if (source.isUnsignedInteger(32) && result.isIndex() &&
        isRegionValue(cast.getResult().getType()))
      return LaneRelation::Indexed;
    return classifyLaneRelation(cast.getInput(), coordinate);
  }
  return isRegionValue(value.getType()) ? LaneRelation::NonAffine
                                        : LaneRelation::Independent;
}

mlir::Value findIndexedOffset(mlir::Value pointer, mlir::Value coordinate) {
  auto add = pointer.getDefiningOp<PtrAddOp>();
  if (!add)
    return {};
  LaneRelation base = classifyLaneRelation(add.getBase(), coordinate);
  LaneRelation offset = classifyLaneRelation(add.getOffset(), coordinate);
  if (base == LaneRelation::Independent && offset == LaneRelation::Indexed)
    return add.getOffset();
  if (base == LaneRelation::Indexed && offset == LaneRelation::Independent)
    return findIndexedOffset(add.getBase(), coordinate);
  return {};
}

mlir::LogicalResult analyzeKernelPhysicalFacts(KernelOp kernel,
                                               KernelPhysicalFacts &facts) {
  facts.blockAxes.clear();
  facts.axes.clear();
  facts.values.clear();
  facts.memory.clear();
  facts.operationOrdinals.clear();
  unsigned ordinal = 0;
  kernel.walk([&](mlir::Operation *operation) {
    if (operation != kernel.getOperation())
      facts.operationOrdinals.try_emplace(operation, ordinal++);
  });

  bool failed = false;
  kernel.walk([&](BlockIndexOp axis) {
    if (failed)
      return;
    LogicalAxisFact fact;
    fact.coordinate = axis.getResult();
    fact.lowerBound = axis.getOffset();
    fact.extent = axis.getExtent();
    fact.identity = axis.getAxis();
    for (mlir::Operation *parent = axis->getParentOp();
         parent && parent != kernel.getOperation(); parent = parent->getParentOp())
      if (mlir::isa<ForOp, WhileOp, IfOp>(parent))
        fact.orderedParents.push_back(parent);
    if (!facts.blockAxes.try_emplace(fact.identity, fact.coordinate).second ||
        !facts.axes.try_emplace(fact.coordinate, std::move(fact)).second) {
      axis.emitError("logical block axis has multiple physical fact owners");
      failed = true;
    }
  });
  kernel.walk([&](VLAOp vla) {
    if (failed)
      return;
    LogicalAxisFact fact;
    fact.coordinate = vla.getBody().front().getArgument(0);
    fact.lowerBound = vla.getBegin();
    fact.upperBound = vla.getEnd();
    fact.vla = true;
    for (mlir::Operation *parent = vla->getParentOp();
         parent && parent != kernel.getOperation(); parent = parent->getParentOp())
      if (mlir::isa<ForOp, WhileOp, IfOp>(parent))
        fact.orderedParents.push_back(parent);
    if (!facts.axes.try_emplace(fact.coordinate, std::move(fact)).second) {
      vla.emitError("VLA axis has multiple physical fact owners");
      failed = true;
    }
  });
  if (failed)
    return mlir::failure();

  auto analyzeValue = [&](mlir::Value value) {
    if (!value || facts.values.contains(value))
      return;
    ValueUseFact fact;
    fact.value = value;
    fact.logicalAxes = logicalAxesForValue(value, facts);
    mlir::Value reloadValue = value;
    while (auto cast = reloadValue.getDefiningOp<CastOp>()) {
      if (cast.getInput().getType() != cast.getResult().getType())
        break;
      reloadValue = cast.getInput();
    }
    if (auto load = reloadValue.getDefiningOp<LoadOp>())
      fact.reloadSource = load.getOperation();
    mlir::Block *definitionBlock = nullptr;
    if (mlir::Operation *definition = value.getDefiningOp()) {
      definitionBlock = definition->getBlock();
      fact.definitionOrdinal = facts.operationOrdinals.lookup(definition);
    } else if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      definitionBlock = argument.getOwner();
      mlir::Operation *parent = definitionBlock->getParentOp();
      fact.definitionOrdinal = parent ? facts.operationOrdinals.lookup(parent) : 0;
      fact.controlCarried = parent && mlir::isa<ForOp, WhileOp, IfOp>(parent);
    }
    fact.lastUseOrdinal = fact.definitionOrdinal;
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *consumer = use.getOwner();
      if (!llvm::is_contained(fact.consumers, consumer))
        fact.consumers.push_back(consumer);
      fact.lastUseOrdinal =
          std::max(fact.lastUseOrdinal,
                   facts.operationOrdinals.lookup(consumer));
      fact.crossesRegion |= definitionBlock && consumer->getBlock() != definitionBlock;
      fact.controlCarried |= mlir::isa<YieldOp, ConditionOp>(consumer);
    }
    fact.multipleConsumers = fact.consumers.size() > 1;
    if (auto result = mlir::dyn_cast<mlir::OpResult>(value))
      fact.controlCarried |= mlir::isa<ForOp, WhileOp, IfOp>(result.getOwner());
    for (const auto &axis : facts.axes)
      if (dependsOn(value, axis.first) ||
          llvm::is_contained(fact.logicalAxes, axis.first))
        fact.axisDependencies.push_back(axis.first);
    facts.values.try_emplace(value, std::move(fact));
  };

  for (mlir::BlockArgument argument : kernel.getBody().front().getArguments())
    analyzeValue(argument);
  kernel.walk([&](mlir::Operation *operation) {
    for (mlir::Value result : operation->getResults())
      analyzeValue(result);
    for (mlir::Region &region : operation->getRegions())
      for (mlir::Block &block : region)
        for (mlir::BlockArgument argument : block.getArguments())
          analyzeValue(argument);
  });

  kernel.walk([&](mlir::Operation *operation) {
    mlir::Value pointer;
    mlir::Value predicate;
    mlir::Type accessedType;
    bool write = false;
    if (auto load = mlir::dyn_cast<LoadOp>(operation)) {
      pointer = load.getPointer();
      predicate = load.getWhere();
      accessedType = elementType(load.getResult().getType());
    } else if (auto store = mlir::dyn_cast<StoreOp>(operation)) {
      pointer = store.getPointer();
      predicate = store.getWhere();
      accessedType = elementType(store.getValue().getType());
      write = true;
    } else {
      return;
    }
    MemoryAccessFact fact;
    fact.operation = operation;
    fact.pointer = pointer;
    fact.root = pointerRoot(pointer);
    fact.predicate = predicate;
    fact.elementType = accessedType;
    fact.write = write;
    for (const auto &axis : facts.axes) {
      MemoryAxisFact axisFact;
      axisFact.relation = classifyLaneRelation(pointer, axis.first);
      if (axisFact.relation == LaneRelation::UnitStride ||
          axisFact.relation == LaneRelation::Strided)
        axisFact.laneStride = deriveLaneStride(pointer, axis.first);
      if (axisFact.relation == LaneRelation::Indexed)
        axisFact.indexedOffset = findIndexedOffset(pointer, axis.first);
      fact.axes.try_emplace(axis.first, std::move(axisFact));
    }
    facts.memory.try_emplace(operation, std::move(fact));
  });
  return mlir::success();
}

} // namespace weft::riscv_internal
