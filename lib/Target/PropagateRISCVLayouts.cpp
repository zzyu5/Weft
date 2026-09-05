#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"
#include "RISCVResidencyPlanning.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <optional>

using namespace weft;

namespace {

struct Roles {
  int64_t laneAxis = 0;
  // Some target operations consume a cartesian logical tile in one RVV
  // register group.  laneAxis is the instruction's primary lane coordinate
  // (for example a contraction reduction axis); coalescedLaneAxes are
  // surviving logical coordinates that are linearized beside it.  Keeping the
  // axes distinct here lets later typed operations recover each logical
  // coordinate while the physical lane count is their product.
  llvm::SmallSet<int64_t, 4> coalescedLaneAxes;
  llvm::SmallSet<int64_t, 4> replicaAxes;
  llvm::SmallSet<int64_t, 4> sequentialAxes;
  bool ime = false;
  bool local = false;
  bool anchored = false;
  bool fullLaneExtent = false;
  bool registerTuple = false;
  // A typed affine extract may have a storage-contiguous axis that conflicts
  // with a downstream compute layout.  Keep that memory anchor intact and let
  // insertUseConversions materialize the representation change on the SSA use
  // edge.  Propagating the compute layout back into the extract would turn a
  // provably contiguous window into an indexed gather.
  bool memoryAnchored = false;
  llvm::SmallVector<int64_t, 4> fragmentParameters;
};

struct AffineAxes {
  llvm::DenseMap<int64_t, int64_t> coefficients;
  bool valid = true;
};

bool checkedMultiply(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs == 0 || rhs == 0) {
    result = 0;
    return true;
  }
  if (lhs > 0) {
    if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() / rhs) ||
        (rhs < 0 && rhs < std::numeric_limits<int64_t>::min() / lhs))
      return false;
  } else {
    if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() / rhs) ||
        (rhs < 0 && lhs < std::numeric_limits<int64_t>::max() / rhs))
      return false;
  }
  result = lhs * rhs;
  return true;
}

void addAxisCoefficient(AffineAxes &result, int64_t axis,
                        int64_t coefficient) {
  if (!result.valid)
    return;
  int64_t next = result.coefficients.lookup(axis);
  if ((coefficient > 0 &&
       next > std::numeric_limits<int64_t>::max() - coefficient) ||
      (coefficient < 0 &&
       next < std::numeric_limits<int64_t>::min() - coefficient)) {
    result.valid = false;
    return;
  }
  next += coefficient;
  if (next)
    result.coefficients[axis] = next;
  else
    result.coefficients.erase(axis);
}

void decomposeAffineAxes(mlir::Value value, int64_t coefficient,
                         AffineAxes &result, unsigned depth = 0) {
  if (!result.valid || depth > 24)
    return result.valid = false, void();
  if (riscv_internal::constantInt(value))
    return;
  if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
    if (conversion.getConversion().getEffect() != "pure")
      return result.valid = false, void();
    decomposeAffineAxes(conversion.getInput(), coefficient, result, depth + 1);
    return;
  }
  if (auto cast = value.getDefiningOp<riscv::CastOp>()) {
    decomposeAffineAxes(cast.getInput(), coefficient, result, depth + 1);
    return;
  }
  if (auto iota = value.getDefiningOp<riscv::IotaOp>()) {
    auto type = iota.getResult().getType();
    if (type.getShape().size() != 1 || type.getAxisIds().size() != 1 ||
        iota.getEnd() - iota.getStart() != type.getShape()[0])
      return result.valid = false, void();
    addAxisCoefficient(result, type.getAxisIds()[0], coefficient);
    return;
  }
  if (auto binary = value.getDefiningOp<riscv::BinaryOp>()) {
    llvm::StringRef kind = binary.getKind();
    if (kind == "add" || kind == "sub") {
      decomposeAffineAxes(binary.getLhs(), coefficient, result, depth + 1);
      int64_t rhsCoefficient = coefficient;
      if (kind == "sub" &&
          !checkedMultiply(coefficient, -1, rhsCoefficient))
        return result.valid = false, void();
      decomposeAffineAxes(binary.getRhs(), rhsCoefficient, result, depth + 1);
      return;
    }
    if (kind == "mul") {
      if (auto lhs = riscv_internal::constantInt(binary.getLhs())) {
        int64_t scaled = 0;
        if (!checkedMultiply(coefficient, *lhs, scaled))
          return result.valid = false, void();
        decomposeAffineAxes(binary.getRhs(), scaled, result, depth + 1);
        return;
      }
      if (auto rhs = riscv_internal::constantInt(binary.getRhs())) {
        int64_t scaled = 0;
        if (!checkedMultiply(coefficient, *rhs, scaled))
          return result.valid = false, void();
        decomposeAffineAxes(binary.getLhs(), scaled, result, depth + 1);
        return;
      }
    }
  }
  // Scalar loop/Level bases do not affect which shaped axis is contiguous.
  if (!mlir::isa<riscv::ValueType>(value.getType()))
    return;
  result.valid = false;
}

std::optional<Roles> storageRolesFor(mlir::Value value,
                                     bool requireWideningConsumer = true) {
  auto extract = value.getDefiningOp<riscv::ExtractOp>();
  auto result = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!extract || !result)
    return std::nullopt;
  llvm::SmallVector<mlir::Value> worklist{value};
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;
  bool feedsWideningContraction = !requireWideningConsumer;
  while (!worklist.empty() && !feedsWideningContraction) {
    mlir::Value current = worklist.pop_back_val();
    for (mlir::Operation *user : current.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      if (mlir::isa<riscv::DotOp, riscv::ContractOp,
                    riscv::OuterContractOp>(user)) {
        auto implementation =
            user->getAttrOfType<riscv::ImplementationAttr>("implementation");
        feedsWideningContraction =
            implementation && implementation.getFamily() == "widen-dot" &&
            (user->getOperand(0) == current || user->getOperand(1) == current);
        continue;
      }
      if (!mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp,
                     riscv::CastOp, riscv::NarrowOp, riscv::WidenOp,
                     riscv::ConvertLayoutOp>(user) ||
          user->getNumResults() != 1)
        continue;
      auto next = mlir::dyn_cast<riscv::ValueType>(user->getResult(0).getType());
      if (next && next.getShape() == result.getShape() &&
          next.getAxisIds() == result.getAxisIds())
        worklist.push_back(user->getResult(0));
    }
  }
  if (!feedsWideningContraction)
    return std::nullopt;
  auto field = riscv_internal::sourceField(extract.getInput());
  if (!field)
    return std::nullopt;
  riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(field);
  if (facts.mapping != "natural" && facts.mapping != "grouped_layered")
    return std::nullopt;

  size_t indexCursor = 0;
  mlir::Value gatherIndex;
  for (mlir::Attribute selectorAttribute : extract.getSelectors()) {
    llvm::StringRef selector =
        mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
    if (selector == "all")
      continue;
    if (indexCursor >= extract.getIndices().size())
      return std::nullopt;
    mlir::Value index = extract.getIndices()[indexCursor++];
    if (selector == "gather") {
      if (gatherIndex)
        return std::nullopt;
      gatherIndex = index;
    }
  }
  if (!gatherIndex || indexCursor != extract.getIndices().size())
    return std::nullopt;

  AffineAxes affine;
  decomposeAffineAxes(gatherIndex, 1, affine);
  if (!affine.valid)
    return std::nullopt;
  int64_t contiguousAxis = 0;
  for (auto [axis, extent] :
       llvm::zip(result.getAxisIds().asArrayRef(), result.getShape().asArrayRef())) {
    if (affine.coefficients.lookup(axis) != 1 || extent <= 1)
      continue;
    if (facts.mapping == "grouped_layered" &&
        (facts.layer <= 0 || extent > facts.layer))
      continue;
    if (contiguousAxis)
      return std::nullopt;
    contiguousAxis = axis;
  }
  if (!contiguousAxis)
    return std::nullopt;

  Roles roles;
  roles.laneAxis = contiguousAxis;
  if (facts.mapping == "natural" || facts.mapping == "grouped_layered") {
    int64_t span = 0;
    for (auto [axis, extent] : llvm::zip(result.getAxisIds().asArrayRef(),
                                         result.getShape().asArrayRef()))
      if (axis == contiguousAxis)
        span = extent;
    if (span <= 0)
      return std::nullopt;
    llvm::SmallSet<int64_t, 4> consumed{contiguousAxis};
    while (true) {
      int64_t nextAxis = 0;
      int64_t nextExtent = 0;
      for (auto [axis, extent] : llvm::zip(
               result.getAxisIds().asArrayRef(), result.getShape().asArrayRef())) {
        if (consumed.contains(axis) || extent <= 1 ||
            affine.coefficients.lookup(axis) != span)
          continue;
        if (nextAxis)
          return std::nullopt;
        nextAxis = axis;
        nextExtent = extent;
      }
      if (!nextAxis)
        break;
      int64_t nextSpan = 0;
      if (!checkedMultiply(span, nextExtent, nextSpan))
        return std::nullopt;
      // A grouped/layered field is contiguous only inside one physical layer.
      // Crossing the layer boundary changes which packed bits supply the
      // logical value, so that axis must remain a replica/time coordinate.
      if (facts.mapping == "grouped_layered" &&
          (facts.layer <= 0 || nextSpan > facts.layer))
        break;
      roles.coalescedLaneAxes.insert(nextAxis);
      consumed.insert(nextAxis);
      span = nextSpan;
    }
  }
  roles.anchored = true;
  roles.fullLaneExtent = true;
  roles.memoryAnchored = true;
  for (int64_t axis : riscv_internal::logicalAxes(value.getType())) {
    const int64_t extent = riscv_internal::physicalExtent(value, axis);
    if (axis != contiguousAxis &&
        !roles.coalescedLaneAxes.contains(axis) && extent > 0 && extent <= 16)
      roles.replicaAxes.insert(axis);
  }
  return roles;
}

bool containsAxis(mlir::Type type, int64_t axis) {
  return llvm::is_contained(riscv_internal::logicalAxes(type), axis);
}

void addSmallReplicas(mlir::Value value, Roles &roles, int64_t except = 0) {
  for (int64_t axis : riscv_internal::logicalAxes(value.getType()))
    if (int64_t extent = riscv_internal::physicalExtent(value, axis);
        axis != except && !roles.coalescedLaneAxes.contains(axis) &&
        !roles.sequentialAxes.contains(axis) && extent > 0 &&
        extent <= 16)
      roles.replicaAxes.insert(axis);
}

bool layoutPreservingPointwise(mlir::Operation *operation) {
  return mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp,
                   riscv::CastOp, riscv::NarrowOp, riscv::WidenOp,
                   riscv::ConvertLayoutOp, riscv::MaterializeOp>(operation);
}

bool preservesExistingAxes(mlir::Type sourceType, mlir::Type resultType) {
  auto source = mlir::dyn_cast<riscv::ValueType>(sourceType);
  auto result = mlir::dyn_cast<riscv::ValueType>(resultType);
  if (!source || !result)
    return false;
  for (auto [position, axis] :
       llvm::enumerate(source.getAxisIds().asArrayRef())) {
    auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
    if (found == result.getAxisIds().asArrayRef().end() ||
        source.getShape()[position] !=
            result.getShape()[static_cast<size_t>(
                found - result.getAxisIds().asArrayRef().begin())])
      return false;
  }
  return true;
}

bool feedsExpandedIndexedLookup(mlir::Value root) {
  auto rootType = mlir::dyn_cast<riscv::ValueType>(root.getType());
  if (!rootType)
    return false;
  int64_t nonSingletonAxes = 0;
  for (int64_t axis : rootType.getAxisIds().asArrayRef()) {
    const int64_t extent = riscv_internal::physicalExtent(root, axis);
    if (extent <= 0)
      return false;
    nonSingletonAxes += extent > 1;
  }
  if (nonSingletonAxes != 1)
    return false;
  llvm::SmallVector<mlir::Value> worklist{root};
  llvm::SmallPtrSet<mlir::Operation *, 32> visited;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (auto lookup = mlir::dyn_cast<riscv::LookupOp>(user)) {
        auto indices = mlir::dyn_cast<riscv::ValueType>(
            lookup.getIndices().getType());
        if (use.getOperandNumber() == 1 && indices &&
            indices.getAxisIds().size() > rootType.getAxisIds().size() &&
            preservesExistingAxes(root.getType(), indices))
          return true;
        continue;
      }
      if (!layoutPreservingPointwise(user) || user->getNumResults() != 1 ||
          !visited.insert(user).second ||
          !preservesExistingAxes(value.getType(), user->getResult(0).getType()))
        continue;
      worklist.push_back(user->getResult(0));
    }
  }
  return false;
}

std::optional<int64_t>
downstreamImmediateReducedFreeAxis(mlir::Operation *contraction,
                                   llvm::ArrayRef<int64_t> reductionAxes) {
  if (!contraction || contraction->getNumResults() != 1)
    return std::nullopt;
  llvm::SmallVector<mlir::Value> worklist{contraction->getResult(0)};
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;
  std::optional<int64_t> reducedAxis;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    auto source = mlir::dyn_cast<riscv::ValueType>(value.getType());
    if (!source)
      continue;
    for (mlir::Operation *user : value.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      if (auto reduce = mlir::dyn_cast<riscv::ReduceOp>(user)) {
        if (reduce.getAxis() < 0 ||
            reduce.getAxis() >= static_cast<int64_t>(source.getAxisIds().size()))
          continue;
        const int64_t axis = source.getAxisIds()[reduce.getAxis()];
        if (llvm::is_contained(reductionAxes, axis))
          continue;
        if (reducedAxis && *reducedAxis != axis)
          return std::nullopt;
        reducedAxis = axis;
        // Preserve the explicit nested reduction tree.  Only the first
        // reduction outside the contraction may join its lane coordinate;
        // outer reductions remain independent partial/register coordinates.
        continue;
      }
      if (!layoutPreservingPointwise(user) || user->getNumResults() != 1)
        continue;
      auto result = mlir::dyn_cast<riscv::ValueType>(user->getResult(0).getType());
      if (!result ||
          !preservesExistingAxes(value.getType(), user->getResult(0).getType()))
        continue;
      worklist.push_back(user->getResult(0));
    }
  }
  return reducedAxis;
}

llvm::SmallVector<int64_t>
downstreamReducedFreeAxes(mlir::Operation *contraction,
                          llvm::ArrayRef<int64_t> contractionAxes) {
  llvm::SmallVector<int64_t> reducedAxes;
  if (!contraction || contraction->getNumResults() != 1)
    return reducedAxes;

  llvm::SmallVector<mlir::Value> worklist{contraction->getResult(0)};
  llvm::SmallPtrSet<mlir::Operation *, 32> visited;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    auto source = mlir::dyn_cast<riscv::ValueType>(value.getType());
    if (!source)
      continue;
    for (mlir::Operation *user : value.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      if (auto reduce = mlir::dyn_cast<riscv::ReduceOp>(user)) {
        if (reduce.getAxis() < 0 ||
            reduce.getAxis() >= static_cast<int64_t>(source.getAxisIds().size()))
          continue;
        const int64_t axis = source.getAxisIds()[reduce.getAxis()];
        if (!llvm::is_contained(contractionAxes, axis) &&
            !llvm::is_contained(reducedAxes, axis))
          reducedAxes.push_back(axis);
        worklist.push_back(reduce.getResult());
        continue;
      }
      if (!layoutPreservingPointwise(user) || user->getNumResults() != 1)
        continue;
      auto result =
          mlir::dyn_cast<riscv::ValueType>(user->getResult(0).getType());
      if (!result ||
          !preservesExistingAxes(value.getType(), user->getResult(0).getType()))
        continue;
      worklist.push_back(user->getResult(0));
    }
  }
  return reducedAxes;
}

llvm::SmallVector<int64_t> downstreamContractionAxes(mlir::Value root) {
  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<mlir::Value> worklist{root};
  llvm::SmallPtrSet<mlir::Operation *, 32> visited;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    auto source = mlir::dyn_cast<riscv::ValueType>(value.getType());
    if (!source)
      continue;
    for (mlir::Operation *user : value.getUsers()) {
      if (!visited.insert(user).second)
        continue;
      if (auto contract = mlir::dyn_cast<riscv::ContractOp>(user)) {
        auto implementation =
            contract->getAttrOfType<riscv::ImplementationAttr>("implementation");
        if (!implementation || implementation.getFamily() != "widen-dot" ||
            (contract.getLhs() != value && contract.getRhs() != value))
          continue;
        for (int64_t axis : contract.getOver())
          if (!llvm::is_contained(axes, axis))
            axes.push_back(axis);
        for (int64_t axis : downstreamReducedFreeAxes(
                 contract.getOperation(), contract.getOver()))
          if (containsAxis(root.getType(), axis) &&
              !llvm::is_contained(axes, axis))
            axes.push_back(axis);
        continue;
      }
      if (!layoutPreservingPointwise(user) || user->getNumResults() != 1)
        continue;
      auto result =
          mlir::dyn_cast<riscv::ValueType>(user->getResult(0).getType());
      if (!result || result.getShape() != source.getShape() ||
          result.getAxisIds() != source.getAxisIds())
        continue;
      worklist.push_back(user->getResult(0));
    }
  }
  return axes;
}

bool supportsPackedIndexedEntryLookup(riscv::LookupOp lookup,
                                      const riscv_internal::IndexedEntryRelation &relation,
                                      riscv::ValueType result) {
  const int64_t packedBits =
      riscv_internal::logicalBitWidth(result.getElementType()) *
      relation.payloadExtent;
  if (packedBits != 32 && packedBits != 64)
    return false;
  mlir::Value table =
      riscv_internal::stripRepresentationConversions(lookup.getTable());
  mlir::Value descriptor = table;
  if (auto load = table.getDefiningOp<riscv::LoadOp>())
    descriptor = load.getRegion();
  int64_t alignment = 1;
  int64_t bitOffset = 0;
  if (auto memory = mlir::dyn_cast<riscv::MemDescType>(descriptor.getType()))
    alignment = memory.getAlignment();
  if (auto field = descriptor.getDefiningOp<riscv::FieldOp>()) {
    auto facts = riscv_internal::fieldFacts(field);
    alignment = facts.alignment;
    bitOffset = facts.bitOffset;
  }
  return alignment >= packedBits / 8 && bitOffset % packedBits == 0;
}

bool constrainIndexedEntryLookup(riscv::LookupOp lookup, mlir::Value value,
                                 Roles &roles) {
  auto result = mlir::dyn_cast<riscv::ValueType>(lookup.getResult().getType());
  if (!result)
    return false;
  auto relation = riscv_internal::analyzeIndexedEntryRelation(
      lookup.getIndices(), result, {}, {});
  if (!relation || !containsAxis(value.getType(), relation->payloadAxis))
    return false;
  // The payload inside one table entry owns the primary memory-facing lane
  // coordinate. Entry coordinates consumed by the same contraction can join
  // that lane domain; surviving axes remain independent replicas.
  roles.anchored = true;
  roles.fullLaneExtent = true;
  roles.memoryAnchored = true;
  roles.laneAxis = relation->payloadAxis;
  if (supportsPackedIndexedEntryLookup(lookup, *relation, result)) {
    llvm::SmallVector<int64_t> contractionAxes =
        downstreamContractionAxes(lookup.getResult());
    auto entryType =
        mlir::dyn_cast<riscv::ValueType>(relation->entryIndices.getType());
    if (entryType)
      for (int64_t axis : entryType.getAxisIds().asArrayRef())
        if (llvm::is_contained(contractionAxes, axis))
          roles.coalescedLaneAxes.insert(axis);
  }
  addSmallReplicas(value, roles, roles.laneAxis);
  return true;
}

bool storageSupportsLaneAxes(mlir::Value root,
                             llvm::ArrayRef<int64_t> requestedAxes) {
  llvm::SmallVector<mlir::Value> worklist{root};
  llvm::SmallPtrSet<mlir::Operation *, 32> visited;
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      continue;
    if (mlir::isa<riscv::ExtractOp>(definition)) {
      auto storage = storageRolesFor(value);
      if (!storage || !storage->memoryAnchored)
        continue;
      for (int64_t axis : requestedAxes)
        if (containsAxis(value.getType(), axis) &&
            axis != storage->laneAxis &&
            !storage->coalescedLaneAxes.contains(axis))
          return false;
      continue;
    }
    if (!layoutPreservingPointwise(definition))
      continue;
    for (mlir::Value operand : definition->getOperands()) {
      auto operandType = mlir::dyn_cast<riscv::ValueType>(operand.getType());
      auto resultType = mlir::dyn_cast<riscv::ValueType>(value.getType());
      if (operandType && resultType &&
          operandType.getShape() == resultType.getShape() &&
          operandType.getAxisIds() == resultType.getAxisIds())
        worklist.push_back(operand);
    }
  }
  return true;
}

int64_t freeAxisSharedByOneOperand(mlir::Value lhs, mlir::Value rhs) {
  auto lhsAxes = riscv_internal::logicalAxes(lhs.getType());
  auto rhsAxes = riscv_internal::logicalAxes(rhs.getType());
  for (int64_t axis : lhsAxes)
    if (!llvm::is_contained(rhsAxes, axis))
      return axis;
  for (int64_t axis : rhsAxes)
    if (!llvm::is_contained(lhsAxes, axis))
      return axis;
  return 0;
}

bool hasOperandLocalFreeAxis(mlir::Value value, mlir::Value other,
                             llvm::ArrayRef<int64_t> reductionAxes) {
  auto otherAxes = riscv_internal::logicalAxes(other.getType());
  return llvm::any_of(riscv_internal::logicalAxes(value.getType()),
                      [&](int64_t axis) {
                        return !llvm::is_contained(reductionAxes, axis) &&
                               !llvm::is_contained(otherAxes, axis);
                      });
}

void constrainGroupedMac(riscv::MacGroupsOp operation, mlir::Value value,
                         Roles &roles) {
  roles.anchored = true;
  roles.fullLaneExtent = true;
  // A grouped MAC consumes one shared reduction coordinate.  If one operand
  // also carries an output/cohort axis, that free axis is the reusable vector
  // direction.  Falling back to the final reduction axis is only valid for a
  // scalar-output vec-dot.  This is an axis relation, not a source-shape
  // matcher: canonical and derived encodings use the same rule.
  int64_t groupedAxis = 0;
  auto lhsAxes = riscv_internal::logicalAxes(operation.getLhs().getType());
  auto rhsAxes = riscv_internal::logicalAxes(operation.getRhs().getType());
  if (!lhsAxes.empty() && !rhsAxes.empty() && lhsAxes.back() == rhsAxes.back())
    groupedAxis = lhsAxes.back();
  auto integerOf = [](mlir::Value operand) {
    return mlir::dyn_cast<mlir::IntegerType>(
        riscv_internal::logicalElement(operand.getType()));
  };
  auto lhsInteger = integerOf(operation.getLhs());
  auto rhsInteger = integerOf(operation.getRhs());
  mlir::Value packedOperand;
  if (lhsInteger && rhsInteger) {
    if (lhsInteger.isUnsigned() && rhsInteger.isSigned())
      packedOperand = operation.getLhs();
    if (rhsInteger.isUnsigned() && lhsInteger.isSigned())
      packedOperand = operation.getRhs();
  }
  int64_t lane = 0;
  if (packedOperand)
    for (int64_t axis : riscv_internal::logicalAxes(packedOperand.getType()))
      if (axis != groupedAxis) {
        lane = axis;
        break;
      }
  if (!lane)
    lane = freeAxisSharedByOneOperand(operation.getLhs(), operation.getRhs());
  if (!lane) {
    lane = groupedAxis;
  }
  // When a free output axis occupies lanes, the final common axis remains the
  // grouped reduction coordinate.  It is issued in time; treating its small
  // extent as register replicas would keep every term live simultaneously and
  // multiply the partial resource count by the group size.
  if (groupedAxis && lane != groupedAxis &&
      containsAxis(value.getType(), groupedAxis))
    roles.sequentialAxes.insert(groupedAxis);
  if (lane && containsAxis(value.getType(), lane))
    roles.laneAxis = lane;
  addSmallReplicas(value, roles, roles.laneAxis);
}

int64_t rhsFreeLane(mlir::Value lhs, mlir::Value rhs,
                    llvm::ArrayRef<int64_t> reduction) {
  for (auto it = riscv_internal::logicalAxes(rhs.getType()).rbegin();
       it != riscv_internal::logicalAxes(rhs.getType()).rend(); ++it)
    if (!llvm::is_contained(reduction, *it) &&
        !llvm::is_contained(riscv_internal::logicalAxes(lhs.getType()), *it))
      return *it;
  for (auto it = riscv_internal::logicalAxes(lhs.getType()).rbegin();
       it != riscv_internal::logicalAxes(lhs.getType()).rend(); ++it)
    if (!llvm::is_contained(reduction, *it) &&
        !llvm::is_contained(riscv_internal::logicalAxes(rhs.getType()), *it))
      return *it;
  return 0;
}

std::optional<Roles>
reductionSupplyRoles(mlir::Value value,
                     llvm::ArrayRef<int64_t> reductionAxes) {
  llvm::SmallVector<mlir::Value> worklist{value};
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;
  std::optional<Roles> result;
  auto mergeCandidate = [&](const Roles &candidate) {
    if (!candidate.laneAxis ||
        !llvm::is_contained(reductionAxes, candidate.laneAxis))
      return true;
    for (int64_t axis : candidate.coalescedLaneAxes)
      if (!llvm::is_contained(reductionAxes, axis))
        return true;
    if (!result) {
      result = candidate;
      return true;
    }
    if (result->laneAxis != candidate.laneAxis)
      return false;
    for (int64_t axis : candidate.coalescedLaneAxes)
      result->coalescedLaneAxes.insert(axis);
    result->fullLaneExtent |= candidate.fullLaneExtent;
    result->memoryAnchored |= candidate.memoryAnchored;
    return true;
  };

  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    mlir::Operation *definition = current.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      continue;
    if (auto lookup = mlir::dyn_cast<riscv::LookupOp>(definition)) {
      Roles candidate;
      if (constrainIndexedEntryLookup(lookup, current, candidate) &&
          !mergeCandidate(candidate))
        return std::nullopt;
      continue;
    }
    if (mlir::isa<riscv::ExtractOp>(definition)) {
      auto candidate = storageRolesFor(current, false);
      if (candidate && !mergeCandidate(*candidate))
        return std::nullopt;
      continue;
    }
    if (!layoutPreservingPointwise(definition) ||
        definition->getNumResults() != 1)
      continue;
    auto currentType = mlir::dyn_cast<riscv::ValueType>(current.getType());
    if (!currentType)
      continue;
    for (mlir::Value operand : definition->getOperands()) {
      auto operandType = mlir::dyn_cast<riscv::ValueType>(operand.getType());
      if (operandType && operandType.getShape() == currentType.getShape() &&
          operandType.getAxisIds() == currentType.getAxisIds())
        worklist.push_back(operand);
    }
  }
  return result;
}

template <typename Contract>
void constrainOuterContractOperand(Contract operation, mlir::Value value,
                                   Roles &roles) {
  roles.anchored = true;
  auto reduction = operation.getOver();
  int64_t lane = rhsFreeLane(operation.getLhs(), operation.getRhs(), reduction);
  auto implementation = operation->template getAttrOfType<
      riscv::ImplementationAttr>("implementation");
  if (implementation &&
      (implementation.getFamily() == "widen-float-contract" ||
       implementation.getFamily() == "widen-dot" ||
       implementation.getOperation() == "rvv.vfmacc")) {
    roles.fullLaneExtent = true;
    for (int64_t axis : reduction)
      if (containsAxis(value.getType(), axis)) {
        roles.laneAxis = axis;
        break;
      }
    if (!roles.laneAxis)
      roles.registerTuple = true;
    addSmallReplicas(value, roles, roles.laneAxis);
    return;
  }
  if (implementation && implementation.getEngine() == "ime") {
    auto parameters = implementation.getParameters().asArrayRef();
    roles.fragmentParameters.assign(parameters.begin(), parameters.end());
    // Canonical values remain ordinary physical values.  IME fragments are
    // separate typed temporaries introduced by LowerRISCVComposites, followed
    // by an explicit fragment-to-RVV handoff.
  }
  if (lane && containsAxis(value.getType(), lane))
    roles.laneAxis = lane;
  else
    for (int64_t axis : reduction)
      if (containsAxis(value.getType(), axis)) {
        roles.laneAxis = axis;
        break;
      }
  for (int64_t axis : riscv_internal::logicalAxes(value.getType()))
    if (!llvm::is_contained(reduction, axis) && axis != roles.laneAxis &&
        riscv_internal::physicalExtent(value, axis) > 0 && riscv_internal::physicalExtent(value, axis) <= 16)
      roles.replicaAxes.insert(axis);
}

template <typename Contract>
void constrainReductionContractOperand(Contract operation, mlir::Value value,
                                       Roles &roles) {
  auto implementation = operation->template getAttrOfType<
      riscv::ImplementationAttr>("implementation");
  if (implementation && implementation.getOperation() == "rvv.vfmacc" &&
      hasOperandLocalFreeAxis(operation.getLhs(), operation.getRhs(),
                              operation.getOver()) &&
      hasOperandLocalFreeAxis(operation.getRhs(), operation.getLhs(),
                              operation.getOver())) {
    // A contract whose two operands contribute independent output axes is the
    // same physical outer-product relation regardless of which canonical
    // spelling produced it.  For an RVV FMA stream, keep K in the lane carrier
    // and both surviving output axes as register replicas.
    constrainOuterContractOperand(operation, value, roles);
    return;
  }
  roles.anchored = true;
  roles.fullLaneExtent = true;
  auto reduction = operation.getOver();
  int64_t reductionElements = 1;
  bool completeReduction = !reduction.empty();
  for (int64_t axis : reduction) {
    if (!containsAxis(value.getType(), axis)) {
      completeReduction = false;
      break;
    }
    const int64_t extent = riscv_internal::physicalExtent(value, axis);
    if (extent <= 0) {
      completeReduction = false;
      break;
    }
    reductionElements *= extent;
  }
  auto kernel = value.getParentRegion()->getParentOfType<riscv::KernelOp>();
  const int64_t sew =
      std::max<int64_t>(8, riscv_internal::logicalBitWidth(value.getType()));
  const int64_t baseLanes =
      kernel && sew > 0 ? kernel.getTarget().getVlenBits() / sew : 0;
  // A contraction shorter than one base RVV vector whose operand already
  // carries a free axis at least as wide as the reduction is an outer-product
  // issue: keep that free axis in lanes and advance the reduction in time.
  // A smaller free axis does not amortize serializing the larger reduction;
  // keep the reduction in lanes and preserve the free axis as replicas instead.
  // This is derived from the two axis extents and the target base vector, not
  // from an operator or encoding family.
  if (completeReduction && baseLanes > 1 && reductionElements < baseLanes)
    for (int64_t axis : riscv_internal::logicalAxes(value.getType())) {
      const int64_t extent = riscv_internal::physicalExtent(value, axis);
      if (!llvm::is_contained(reduction, axis) &&
          extent > reductionElements && extent <= baseLanes) {
        roles.laneAxis = axis;
        for (int64_t reductionAxis : reduction)
          if (containsAxis(value.getType(), reductionAxis))
            roles.sequentialAxes.insert(reductionAxis);
        return;
      }
    }
  const bool isOperand =
      value == operation.getLhs() || value == operation.getRhs();
  const bool cartesianOutput =
      hasOperandLocalFreeAxis(operation.getLhs(), operation.getRhs(),
                              reduction) &&
      hasOperandLocalFreeAxis(operation.getRhs(), operation.getLhs(), reduction);
  const auto lhsAxes = riscv_internal::logicalAxes(operation.getLhs().getType());
  const auto rhsAxes = riscv_internal::logicalAxes(operation.getRhs().getType());
  const bool sharedFreeAxis = llvm::any_of(lhsAxes, [&](int64_t axis) {
    return !llvm::is_contained(reduction, axis) &&
           llvm::is_contained(rhsAxes, axis);
  });
  if (!isOperand && cartesianOutput && !sharedFreeAxis) {
    // Each operand contributes an independently surviving output coordinate.
    // With no shared free coordinate, the reduction carrier feeds their final
    // Cartesian product, so neither output axis is a SIMD reduction lane.
    // Preserve every output coordinate as a register-replica slot.
    roles.registerTuple = true;
    addSmallReplicas(value, roles);
    return;
  }
  if (isOperand)
    if (auto supply = reductionSupplyRoles(value, reduction)) {
      roles.laneAxis = supply->laneAxis;
      roles.coalescedLaneAxes = supply->coalescedLaneAxes;
      roles.memoryAnchored = supply->memoryAnchored;
    }
  if (!roles.laneAxis)
    for (int64_t axis : reduction)
      if (containsAxis(value.getType(), axis)) {
        if (!roles.laneAxis)
          roles.laneAxis = axis;
        else
          roles.coalescedLaneAxes.insert(axis);
      }
  // A contraction result and its operands deliberately have different physical
  // layouts.  The result preserves every later reduction coordinate as an
  // independent register partial.  An operand may instead carry those same
  // coordinates beside K in one wide lane window, from which the typed partial
  // program takes narrow slices.  This is the RVV analogue of a Triton
  // DotOperandEncoding whose parent is the accumulator encoding: it changes the
  // operand representation, not the accumulator topology.
  auto reducedFreeAxes =
      downstreamReducedFreeAxes(operation.getOperation(), reduction);
  if (roles.laneAxis && isOperand)
    for (int64_t freeAxis : reducedFreeAxes)
      if (freeAxis != roles.laneAxis && containsAxis(value.getType(), freeAxis) &&
          storageSupportsLaneAxes(value, llvm::ArrayRef<int64_t>(freeAxis))) {
        roles.coalescedLaneAxes.insert(freeAxis);
        roles.replicaAxes.erase(freeAxis);
      }
  // A shaped contraction result no longer contains the eliminated reduction
  // axis.  When exactly one operand contributes a surviving free axis, that
  // axis is the contraction's reusable output cohort and preserves the same
  // lane organization already chosen on that operand.  A scalar-output dot has
  // no such axis and remains a register tuple.
  if (!roles.laneAxis) {
    if (auto reducedFreeAxis = downstreamImmediateReducedFreeAxis(
            operation.getOperation(), reduction);
        reducedFreeAxis && containsAxis(value.getType(), *reducedFreeAxis)) {
      roles.registerTuple = true;
      addSmallReplicas(value, roles);
      return;
    }
    const int64_t freeLane =
        freeAxisSharedByOneOperand(operation.getLhs(), operation.getRhs());
    if (freeLane && containsAxis(value.getType(), freeLane)) {
      roles.laneAxis = freeLane;
      addSmallReplicas(value, roles, freeLane);
      return;
    }
    roles.registerTuple = true;
    addSmallReplicas(value, roles);
    return;
  }
  for (int64_t axis : riscv_internal::logicalAxes(value.getType()))
    if (!llvm::is_contained(reduction, axis) && axis != roles.laneAxis &&
        !roles.coalescedLaneAxes.contains(axis) &&
        riscv_internal::physicalExtent(value, axis) > 0 &&
        riscv_internal::physicalExtent(value, axis) <= 16)
      roles.replicaAxes.insert(axis);
}

Roles rolesFor(mlir::Value value) {
  Roles roles;
  auto element = riscv_internal::logicalElement(value.getType());
  if (mlir::isa<kernel::EncodingType>(element))
    return roles;

  // A storage-facing carrier is authoritative when this exact shaped value
  // supplies either a widening contraction or an indexed lookup.  The latter
  // may add broadcast axes while preserving every source axis: those existing
  // axes still have to arrive in lanes so the lookup index is not rebuilt from
  // a scalar tuple at every issue.  This remains narrower than treating every
  // affine scale/index value as a contraction anchor.
  if (auto storage =
          storageRolesFor(value, !feedsExpandedIndexedLookup(value)))
    return *storage;

  if (value.getDefiningOp<riscv::FieldOp>()) {
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *owner = use.getOwner();
      auto implementation = owner->getAttrOfType<riscv::ImplementationAttr>(
          "implementation");
      if (!implementation || implementation.getFamily() != "widen-dot")
        continue;
      if (auto contract = mlir::dyn_cast<riscv::DotOp>(owner))
        constrainReductionContractOperand(contract, value, roles);
      else if (auto contract = mlir::dyn_cast<riscv::ContractOp>(owner))
        constrainReductionContractOperand(contract, value, roles);
      else if (auto contract = mlir::dyn_cast<riscv::OuterContractOp>(owner))
        constrainOuterContractOperand(contract, value, roles);
      if (roles.anchored)
        return roles;
    }
    const bool addressableField =
        !value.use_empty() && llvm::all_of(value.getUsers(), [](mlir::Operation *user) {
          return mlir::isa<riscv::ExtractOp, riscv::SliceOp,
                           riscv::Fold2Op>(user);
        });
    if (addressableField) {
      roles.local = true;
      return roles;
    }
  }

  if (auto state = value.getDefiningOp<riscv::NewOp>();
      state && state.getOwnerDomainId() == 0 &&
      llvm::any_of(riscv_internal::logicalShape(value.getType()),
                   [](int64_t extent) { return extent < 0; })) {
    roles.local = true;
    return roles;
  }

  if (auto materialize = value.getDefiningOp<riscv::MaterializeOp>()) {
    bool groupedProjection = !value.use_empty();
    for (mlir::Operation *user : value.getUsers()) {
      auto extract = mlir::dyn_cast<riscv::ExtractOp>(user);
      bool hasGroupIndex = false;
      if (extract && extract.getInput() == value)
        hasGroupIndex = llvm::any_of(
            extract.getSelectors(), [](mlir::Attribute selector) {
              return mlir::cast<mlir::StringAttr>(selector).getValue() ==
                     "group_index";
            });
      groupedProjection &= static_cast<bool>(extract) && hasGroupIndex;
    }
    if (materialize.getPlacement() == "local" || groupedProjection) {
      roles.local = true;
      return roles;
    }
  }
  if (auto reshape = value.getDefiningOp<riscv::ReshapeOp>()) {
    auto axes = reshape.getResult().getType().getAxisIds().asArrayRef();
    if (!axes.empty()) {
      roles.laneAxis = axes.back();
      roles.anchored = true;
      roles.fullLaneExtent = true;
    }
    return roles;
  }
  if (value.getDefiningOp<riscv::ReduceOp>()) {
    // A reduction result preserves every free logical axis.  Small free axes
    // are register replicas; the eliminated axis is anchored on the producer,
    // not manufactured on the result.
    roles.anchored = true;
    addSmallReplicas(value, roles);
    return roles;
  }
  if (value.getDefiningOp<riscv::Fold2Op>()) {
    roles.anchored = true;
    roles.fullLaneExtent = true;
    auto axes = riscv_internal::logicalAxes(value.getType());
    if (!axes.empty())
      roles.laneAxis = axes.back();
    addSmallReplicas(value, roles, roles.laneAxis);
    return roles;
  }
  if (auto lookup = value.getDefiningOp<riscv::LookupOp>();
      lookup && constrainIndexedEntryLookup(lookup, value, roles))
    return roles;
  if (auto operation = value.getDefiningOp<riscv::MacGroupsOp>()) {
    constrainGroupedMac(operation, value, roles);
  } else if (auto operation = value.getDefiningOp<riscv::DotOp>()) {
    constrainReductionContractOperand(operation, value, roles);
  } else if (auto operation = value.getDefiningOp<riscv::ContractOp>()) {
    constrainReductionContractOperand(operation, value, roles);
  } else if (auto operation = value.getDefiningOp<riscv::OuterContractOp>()) {
    constrainOuterContractOperand(operation, value, roles);
  }

  for (mlir::OpOperand &use : value.getUses()) {
    mlir::Operation *owner = use.getOwner();
    if (auto reduce = mlir::dyn_cast<riscv::ReduceOp>(owner)) {
      auto axes = riscv_internal::logicalAxes(value.getType());
      if (reduce.getAxis() >= 0 &&
          reduce.getAxis() < static_cast<int64_t>(axes.size())) {
        // The eliminated axis is only the fallback lane anchor.  A producer or
        // loop-carried value may already have a surviving lane axis, in which
        // case this is a register-axis reduction.  Defer that choice until the
        // strong producer/control constraints have propagated.
        roles.anchored = true;
      }
    } else if (auto mac = mlir::dyn_cast<riscv::MacGroupsOp>(owner)) {
      constrainGroupedMac(mac, value, roles);
    } else if (auto contract = mlir::dyn_cast<riscv::DotOp>(owner)) {
      constrainReductionContractOperand(contract, value, roles);
    } else if (auto contract = mlir::dyn_cast<riscv::ContractOp>(owner)) {
      constrainReductionContractOperand(contract, value, roles);
    } else if (auto contract = mlir::dyn_cast<riscv::OuterContractOp>(owner)) {
      constrainOuterContractOperand(contract, value, roles);
    } else if (auto lookup = mlir::dyn_cast<riscv::LookupOp>(owner)) {
      if (!constrainIndexedEntryLookup(lookup, value, roles)) {
        auto axes = riscv_internal::logicalAxes(lookup.getIndices().getType());
        if (!axes.empty() && containsAxis(value.getType(), axes.back()))
          roles.laneAxis = axes.back();
        addSmallReplicas(value, roles, roles.laneAxis);
      }
    } else if (auto reshape = mlir::dyn_cast<riscv::ReshapeOp>(owner)) {
      if (reshape.getInput() != value || reshape.getOrder().empty())
        continue;
      roles.laneAxis = reshape.getOrder().back();
      roles.coalescedLaneAxes.clear();
      roles.replicaAxes.clear();
      roles.sequentialAxes.clear();
      for (int64_t axis : reshape.getOrder().drop_back())
        roles.coalescedLaneAxes.insert(axis);
      roles.anchored = true;
      roles.fullLaneExtent = true;
    }
  }

  return roles;
}

bool mergeRoles(const Roles &source, mlir::Value target, Roles &destination,
                bool preserveCarrier = false) {
  bool changed = false;
  const bool destinationWasAnchored = destination.anchored;
  auto axes = riscv_internal::logicalAxes(target.getType());
  if (preserveCarrier && source.local && !destination.local) {
    destination.local = true;
    changed = true;
  }
  if (preserveCarrier && source.ime && !destination.ime) {
    destination.ime = true;
    destination.fragmentParameters = source.fragmentParameters;
    changed = true;
  }
  if (source.registerTuple && !destination.registerTuple &&
      !(destination.anchored && destination.laneAxis)) {
    destination.registerTuple = true;
    destination.laneAxis = 0;
    addSmallReplicas(target, destination);
    changed = true;
  }
  if (source.laneAxis && llvm::is_contained(axes, source.laneAxis) &&
      destination.laneAxis == 0 && !destination.registerTuple) {
    destination.laneAxis = source.laneAxis;
    destination.replicaAxes.erase(source.laneAxis);
    changed = true;
  }
  for (int64_t axis : source.replicaAxes)
    if (axis != destination.laneAxis && llvm::is_contained(axes, axis) &&
        !destination.coalescedLaneAxes.contains(axis) &&
        !destination.sequentialAxes.contains(axis) &&
        destination.replicaAxes.insert(axis).second)
      changed = true;
  for (int64_t axis : source.coalescedLaneAxes)
    if (axis != destination.laneAxis && llvm::is_contained(axes, axis) &&
        !destination.memoryAnchored &&
        !destination.sequentialAxes.contains(axis) &&
        destination.coalescedLaneAxes.insert(axis).second) {
      destination.replicaAxes.erase(axis);
      changed = true;
    }
  for (int64_t axis : source.sequentialAxes)
    if (llvm::is_contained(axes, axis) &&
        destination.sequentialAxes.insert(axis).second) {
      destination.replicaAxes.erase(axis);
      changed = true;
    }
  if (source.anchored && !destination.anchored) {
    destination.anchored = true;
    changed = true;
  }
  if (source.fullLaneExtent && !destination.fullLaneExtent) {
    destination.fullLaneExtent = true;
    changed = true;
  }
  if (source.memoryAnchored && !destinationWasAnchored &&
      !destination.memoryAnchored) {
    destination.memoryAnchored = true;
    changed = true;
  }
  return changed;
}

void completeRoles(mlir::Value value, Roles &roles) {
  if (roles.local || roles.ime)
    return;
  if (roles.registerTuple) {
    roles.laneAxis = 0;
    roles.coalescedLaneAxes.clear();
    addSmallReplicas(value, roles);
    return;
  }
  auto axes = riscv_internal::logicalAxes(value.getType());
  // A coalesced lane coordinate cannot exist without a primary lane carrier.
  // Pointwise propagation can legitimately retain only a surviving coalesced
  // axis after its original primary axis was projected away.  Promote the
  // innermost such axis so the resulting type never claims a scalar carrier
  // while still containing non-unit lane factors.
  if (!roles.laneAxis && !roles.coalescedLaneAxes.empty())
    for (auto axis = axes.rbegin(); axis != axes.rend(); ++axis)
      if (roles.coalescedLaneAxes.erase(*axis)) {
        roles.laneAxis = *axis;
        roles.anchored = true;
        break;
      }
  if (!roles.anchored && roles.laneAxis == 0 && !axes.empty()) {
    // A shaped Value already carries an explicit logical domain.  Unlike an
    // ordinary scalar for/while iteration, that domain is legal input to the
    // target mapping.  The target's base structural rule maps the innermost
    // logical axis to RVV; memory and consumer anchors above may replace this
    // with a more constrained axis.  Root dynamic state is handled earlier and
    // deliberately remains local (Top-K is the canonical example).
    roles.laneAxis = axes.back();
    roles.anchored = true;
  }
  addSmallReplicas(value, roles, roles.laneAxis);
  if (roles.laneAxis)
    roles.replicaAxes.erase(roles.laneAxis);
  for (int64_t axis : roles.coalescedLaneAxes)
    roles.replicaAxes.erase(axis);
}

int64_t roundLegalLMUL(riscv::TargetAttr target, int64_t requested,
                       int64_t sew) {
  int64_t elen = 0;
  for (int64_t supported : target.getSupportedSEW().asArrayRef())
    elen = std::max(elen, supported);
  if (elen <= 0)
    return 0;
  requested = std::max(requested, (8 * sew + elen - 1) / elen);
  for (int64_t legal : target.getLegalLMULEighths().asArrayRef())
    if (legal >= requested)
      return legal;
  return 0;
}

riscv::LayoutAttr buildLayout(mlir::Builder &builder, mlir::Value value,
                              Roles roles, int64_t connectedLaneSpan,
                              int64_t lmulEighths) {
  auto type = mlir::cast<riscv::ValueType>(value.getType());
  auto axes = type.getAxisIds().asArrayRef();
  auto element = type.getElementType();
  auto kernel = value.getParentRegion()->getParentOfType<riscv::KernelOp>();
  riscv::TargetAttr target = kernel.getTarget();
  const int64_t sew = std::max<int64_t>(8, riscv_internal::logicalBitWidth(type));

  llvm::SmallVector<int64_t> time;
  llvm::SmallVector<int64_t> lane(axes.size(), 1);
  llvm::SmallVector<int64_t> replica(axes.size(), 1);
  llvm::SmallVector<int64_t> fragment(axes.size(), 1);
  llvm::SmallVector<int64_t> local(axes.size(), 1);

  if (mlir::isa<kernel::EncodingType>(element)) {
    for (int64_t axis : axes)
      time.push_back(std::max<int64_t>(1, riscv_internal::physicalExtent(value, axis)));
    return riscv::LayoutAttr::get(
        builder.getContext(), "local", type.getAxisIds(),
        riscv_internal::integers(builder, time),
        riscv_internal::integers(builder, lane),
        riscv_internal::integers(builder, replica),
        riscv_internal::integers(builder, fragment),
        riscv_internal::integers(builder, local), 8, 0, 1, 0, "full");
  }

  if (roles.local) {
    for (size_t index = 0; index < axes.size(); ++index) {
      int64_t extent =
          std::max<int64_t>(1, riscv_internal::physicalExtent(value, axes[index]));
      time.push_back(1);
      local[index] = extent;
    }
    return riscv::LayoutAttr::get(
        builder.getContext(), "local", type.getAxisIds(),
        riscv_internal::integers(builder, time),
        riscv_internal::integers(builder, lane),
        riscv_internal::integers(builder, replica),
        riscv_internal::integers(builder, fragment),
        riscv_internal::integers(builder, local), sew, 0, 1, 0, "full");
  }

  if (roles.ime) {
    auto parameters = roles.fragmentParameters;
    llvm::SmallVector<int64_t> freeAxes;
    for (int64_t axis : axes)
      freeAxes.push_back(axis);
    for (size_t index = 0; index < axes.size(); ++index) {
      int64_t extent = std::max<int64_t>(1, riscv_internal::physicalExtent(value, axes[index]));
      int64_t factor = 1;
      if (!parameters.empty()) {
        if (index == 0)
          factor = std::min(extent, parameters[0]);
        else if (index == 1 && parameters.size() > 1)
          factor = std::min(extent, parameters[1]);
        else if (parameters.size() > 2)
          factor = std::min(extent, parameters[2]);
      }
      fragment[index] = std::max<int64_t>(1, factor);
      time.push_back((extent + fragment[index] - 1) / fragment[index]);
    }
    return riscv::LayoutAttr::get(
        builder.getContext(), "ime", type.getAxisIds(),
        riscv_internal::integers(builder, time),
        riscv_internal::integers(builder, lane),
        riscv_internal::integers(builder, replica),
        riscv_internal::integers(builder, fragment),
        riscv_internal::integers(builder, local), sew, 0, 1, 0, "full");
  }

  int64_t desiredLanes = 1;
  const int64_t laneCapacity =
      std::max<int64_t>(1, target.getVlenBits() * lmulEighths / (8 * sew));
  if (roles.laneAxis) {
    int64_t logicalExtent = riscv_internal::physicalExtent(value, roles.laneAxis);
    int64_t representationExtent =
        logicalExtent > 0 ? logicalExtent
                          : std::max<int64_t>(1, connectedLaneSpan);
    // The instantiated LMUL binding is a maximum lane capacity, not a new
    // logical axis.  Smaller values still use the smallest legal LMUL that
    // covers their extent; larger values become explicit issue-time strips.
    desiredLanes = std::min(
        representationExtent,
        std::max<int64_t>(1, std::min(laneCapacity, connectedLaneSpan)));
    if (logicalExtent > 0 && logicalExtent % desiredLanes != 0) {
      int64_t divisor = 1;
      while (divisor <= desiredLanes / 2)
        divisor *= 2;
      while (divisor > 1 && logicalExtent % divisor != 0)
        divisor /= 2;
      desiredLanes = divisor;
    }
  }
  int64_t totalDesiredLanes = desiredLanes;
  for (int64_t position = static_cast<int64_t>(axes.size()) - 1;
       position >= 0; --position) {
    const size_t index = static_cast<size_t>(position);
    const int64_t axis = axes[index];
    if (!roles.coalescedLaneAxes.contains(axis))
      continue;
    const int64_t extent =
        std::max<int64_t>(1, riscv_internal::physicalExtent(value, axis));
    int64_t available = std::max<int64_t>(1, laneCapacity / totalDesiredLanes);
    int64_t factor = std::min(extent, available);
    while (factor > 1 && extent % factor)
      --factor;
    lane[index] = std::max<int64_t>(1, factor);
    totalDesiredLanes *= lane[index];
  }
  int64_t requestedLMUL =
      roles.laneAxis
          ? std::max<int64_t>(1, (totalDesiredLanes * sew * 8 +
                                  target.getVlenBits() - 1) /
                                     target.getVlenBits())
          : 0;
  int64_t lmul =
      roles.laneAxis ? roundLegalLMUL(target, requestedLMUL, sew) : 0;
  if (roles.laneAxis && lmul == 0)
    return {};
  int64_t actualLanes = roles.laneAxis ? totalDesiredLanes : 1;

  int64_t replicas = 1;
  for (size_t index = 0; index < axes.size(); ++index) {
    int64_t logicalExtent = riscv_internal::physicalExtent(value, axes[index]);
    int64_t extent = logicalExtent > 0 ? logicalExtent
                     : axes[index] == roles.laneAxis
                         ? std::max<int64_t>(1, connectedLaneSpan)
                         : 1;
    if (axes[index] == roles.laneAxis)
      lane[index] = std::min(extent, desiredLanes);
    if (roles.replicaAxes.contains(axes[index])) {
      replica[index] = extent;
      replicas *= extent;
    }
    int64_t mapped = lane[index] * replica[index];
    time.push_back((extent + mapped - 1) / mapped);
  }
  // A logical axis of extent one does not constitute an RVV representation.
  // Keep the axis identity in the type, but use the scalar carrier so later
  // memory and operation passes do not invent a one-lane vector operation.
  const bool usesRVV = roles.laneAxis && actualLanes > 1;
  llvm::StringRef carrier = usesRVV ? "rvv" : "scalar";
  if (!usesRVV)
    lmul = 0;
  int64_t groups = usesRVV ? ((lmul + 7) / 8) * replicas : 0;
  llvm::StringRef validity = "full";
  if (roles.laneAxis && riscv_internal::physicalExtent(value, roles.laneAxis) <= 0)
    validity = "tail";
  for (mlir::Operation *parent = value.getParentRegion()->getParentOp(); parent;
       parent = parent->getParentOp())
    if (auto loop = mlir::dyn_cast<riscv::LoopOp>(parent))
      if (loop.getDomain().getType().getTail() == "tail" &&
          llvm::is_contained(axes,
                             loop.getDomain().getType().getAxisId()))
        validity = "tail";
  return riscv::LayoutAttr::get(
      builder.getContext(), carrier, type.getAxisIds(),
      riscv_internal::integers(builder, time),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, replica),
      riscv_internal::integers(builder, fragment),
      riscv_internal::integers(builder, local), sew, lmul,
      usesRVV ? actualLanes : 1, groups, validity);
}

bool pointwiseLike(mlir::Operation *operation) {
  return mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp,
                   riscv::CastOp, riscv::NarrowOp, riscv::WidenOp,
                   riscv::UpdateOp, riscv::MaterializeOp>(operation);
}

std::optional<int64_t> encodedLaneLimit(mlir::Value value, int64_t laneAxis) {
  llvm::SmallPtrSet<mlir::Operation *, 8> visited;
  std::function<std::optional<int64_t>(mlir::Value)> visit =
      [&](mlir::Value current) -> std::optional<int64_t> {
    mlir::Operation *definition = current.getDefiningOp();
    if (!definition || !visited.insert(definition).second)
      return std::nullopt;
    if (auto field = mlir::dyn_cast<riscv::FieldOp>(definition)) {
      riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(field);
      auto type = mlir::dyn_cast<riscv::ValueType>(field.getResult().getType());
      if (!type || facts.logicalRank <= 0 ||
          facts.logicalRank > static_cast<int64_t>(type.getAxisIds().size()))
        return std::nullopt;
      auto logicalFieldAxes = type.getAxisIds().asArrayRef().take_back(
          static_cast<size_t>(facts.logicalRank));
      if (facts.mapping == "grouped_layered" && facts.layer > 0 &&
          llvm::is_contained(logicalFieldAxes, laneAxis))
        return facts.layer;
      return std::nullopt;
    }
    if (auto conversion =
            mlir::dyn_cast<riscv::ConvertLayoutOp>(definition)) {
      auto input =
          mlir::dyn_cast<riscv::ValueType>(conversion.getInput().getType());
      auto result =
          mlir::dyn_cast<riscv::ValueType>(conversion.getResult().getType());
      llvm::StringRef effect = conversion.getConversion().getEffect();
      if (!input || !result || input.getShape() != result.getShape() ||
          input.getAxisIds() != result.getAxisIds() ||
          (effect != "pure" && effect != "read"))
        return std::nullopt;
      return visit(conversion.getInput());
    }
    if (auto materialize = mlir::dyn_cast<riscv::MaterializeOp>(definition))
      return visit(materialize.getInput());
    if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(definition))
      return visit(extract.getInput());
    if (auto lookup = mlir::dyn_cast<riscv::LookupOp>(definition))
      return visit(lookup.getIndices());
    // Storage-layer width constrains the raw field and representation-only
    // casts, not a newly computed logical value.  In particular, combining
    // low/high bit layers creates a full logical vector that may be
    // re-partitioned through an explicit convert_layout.
    if (!mlir::isa<riscv::UnaryOp, riscv::CastOp, riscv::NarrowOp,
                   riscv::WidenOp>(definition))
      return std::nullopt;
    std::optional<int64_t> limit;
    for (mlir::Value operand : definition->getOperands())
      if (auto operandLimit = visit(operand))
        limit = limit ? std::min(*limit, *operandLimit) : operandLimit;
    return limit;
  };
  return visit(value);
}

void setValueLayout(mlir::Value value, riscv::LayoutAttr layout) {
  if (mlir::isa<riscv::ValueType>(value.getType()))
    value.setType(riscv_internal::withLayout(value.getType(), layout));
}

riscv::LayoutAttr withRegisterWidth(mlir::Builder &builder,
                                    riscv::LayoutAttr layout,
                                    int64_t lmulEighths) {
  int64_t replicas = 1;
  for (int64_t factor : layout.getReplicaFactors().asArrayRef())
    replicas *= factor;
  return riscv::LayoutAttr::get(
      builder.getContext(), layout.getCarrier(), layout.getAxisIds(),
      layout.getTimeFactors(), layout.getLaneFactors(),
      layout.getReplicaFactors(), layout.getFragmentFactors(),
      layout.getLocalFactors(), layout.getSew(), lmulEighths, layout.getVl(),
      ((lmulEighths + 7) / 8) * replicas, layout.getValidity());
}

riscv::LayoutAttr registerGatherSourceLayout(mlir::Builder &builder,
                                             riscv::ValueType source,
                                             riscv::LayoutAttr resultLayout,
                                             riscv::TargetAttr target) {
  auto shape = source.getShape().asArrayRef();
  if (shape.size() != 1 || shape.front() <= 0 ||
      resultLayout.getCarrier() != "rvv" ||
      resultLayout.getSew() !=
          std::max<int64_t>(8, riscv_internal::logicalBitWidth(source)))
    return {};
  const int64_t capacity = target.getVlenBits() *
                           resultLayout.getLmulEighths() /
                           (8 * resultLayout.getSew());
  if (shape.front() > capacity)
    return {};
  llvm::SmallVector<int64_t> one{1};
  llvm::SmallVector<int64_t> lane{shape.front()};
  return riscv::LayoutAttr::get(
      builder.getContext(), "rvv", source.getAxisIds(),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, lane),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one),
      riscv_internal::integers(builder, one), resultLayout.getSew(),
      resultLayout.getLmulEighths(), shape.front(),
      (resultLayout.getLmulEighths() + 7) / 8, "full");
}

class PropagateRISCVLayoutsPass
    : public mlir::PassWrapper<PropagateRISCVLayoutsPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  explicit PropagateRISCVLayoutsPass(int64_t lmulEighths)
      : lmulEighths(lmulEighths) {}
  llvm::StringRef getArgument() const override {
    return "weft-riscv-propagate-layouts";
  }
  llvm::StringRef getDescription() const override {
    return "Propagate logical-axis preserving physical layouts and insert conversions";
  }

  void runOnOperation() override {
    mlir::OpBuilder builder(&getContext());
    bool failed = false;
    llvm::SmallVector<mlir::Value> values;
    getOperation().walk([&](mlir::Operation *operation) {
      for (mlir::Value result : operation->getResults())
        if (mlir::isa<riscv::ValueType>(result.getType()))
          values.push_back(result);
      for (mlir::Region &region : operation->getRegions())
        for (mlir::Block &block : region)
          for (mlir::BlockArgument argument : block.getArguments())
            if (mlir::isa<riscv::ValueType>(argument.getType()))
              values.push_back(argument);
    });

    llvm::DenseMap<mlir::Value, Roles> roles;
    for (mlir::Value value : values)
      roles.try_emplace(value, rolesFor(value));

    // Pointwise values and loop-carried state share one logical distribution.
    // Anchors come from selected target operations; propagation never creates a
    // new logical axis and never crosses a local-storage boundary.
    auto propagateRoles = [&]() {
      bool changed = true;
      while (changed) {
        changed = false;
      getOperation().walk([&](mlir::Operation *operation) {
        if (!pointwiseLike(operation) || operation->getNumResults() != 1)
          return;
        mlir::Value result = operation->getResult(0);
        if (!mlir::isa<riscv::ValueType>(result.getType()))
          return;
        for (mlir::Value operand : operation->getOperands()) {
          if (!mlir::isa<riscv::ValueType>(operand.getType()))
            continue;
          if (mlir::isa<riscv::UpdateOp>(operation) &&
              (roles[result].local || roles[operand].local)) {
            Roles local;
            local.local = true;
            changed |= mergeRoles(local, result, roles[result]);
            changed |= mergeRoles(local, operand, roles[operand]);
            continue;
          }
          // A local-storage value used by ordinary pointwise code requires a
          // typed local_load conversion; it does not force the consumer's
          // entire value chain into local storage.  Conversely, a vector
          // consumer does not rewrite the source object's placement.
          if (roles[result].local || roles[operand].local)
            continue;
          changed |= mergeRoles(roles[operand], result, roles[result]);
          changed |= mergeRoles(roles[result], operand, roles[operand]);
        }
      });
      getOperation().walk([&](riscv::ConvertLayoutOp conversion) {
        mlir::Value input = conversion.getInput();
        mlir::Value result = conversion.getResult();
        auto inputType = mlir::dyn_cast<riscv::ValueType>(input.getType());
        auto resultType = mlir::dyn_cast<riscv::ValueType>(result.getType());
        llvm::StringRef effect = conversion.getConversion().getEffect();
        if (!inputType || !resultType ||
            inputType.getElementType() != resultType.getElementType() ||
            inputType.getShape() != resultType.getShape() ||
            inputType.getAxisIds() != resultType.getAxisIds() ||
            (effect != "pure" && effect != "read"))
          return;
        changed |= mergeRoles(roles[input], result, roles[result]);
        changed |= mergeRoles(roles[result], input, roles[input]);
      });
      getOperation().walk([&](riscv::LoopOp loop) {
        auto yield = mlir::cast<riscv::YieldOp>(loop.getBody().front().getTerminator());
        for (auto [index, initial] : llvm::enumerate(loop.getCarried())) {
          mlir::Value argument = loop.getBody().front().getArgument(index + 1);
          mlir::Value next = yield.getValues()[index];
          mlir::Value result = loop.getResult(index);
          mlir::Value valuesToMerge[] = {initial, argument, next, result};
          for (mlir::Value source : valuesToMerge)
            for (mlir::Value target : valuesToMerge)
              if (mlir::isa<riscv::ValueType>(source.getType()) &&
                  mlir::isa<riscv::ValueType>(target.getType()) &&
                  !roles[source].local && !roles[target].local)
                changed |=
                    mergeRoles(roles[source], target, roles[target], true);
        }
      });
      getOperation().walk([&](mlir::Operation *operation) {
        if (!mlir::isa<mlir::scf::ForOp, mlir::scf::IfOp,
                       mlir::scf::WhileOp>(operation))
          return;
        llvm::SmallVector<mlir::Value> related;
        for (mlir::Value operand : operation->getOperands())
          if (mlir::isa<riscv::ValueType>(operand.getType()))
            related.push_back(operand);
        for (mlir::Value result : operation->getResults())
          if (mlir::isa<riscv::ValueType>(result.getType()))
            related.push_back(result);
        for (mlir::Region &region : operation->getRegions())
          for (mlir::Block &block : region) {
            for (mlir::BlockArgument argument : block.getArguments())
              if (mlir::isa<riscv::ValueType>(argument.getType()))
                related.push_back(argument);
            if (mlir::Operation *terminator = block.getTerminator())
              for (mlir::Value operand : terminator->getOperands())
                if (mlir::isa<riscv::ValueType>(operand.getType()))
                  related.push_back(operand);
          }
        for (mlir::Value source : related)
          for (mlir::Value target : related) {
            auto sourceType = mlir::cast<riscv::ValueType>(source.getType());
            auto targetType = mlir::cast<riscv::ValueType>(target.getType());
            if (sourceType.getElementType() == targetType.getElementType() &&
                sourceType.getShape() == targetType.getShape() &&
                sourceType.getAxisIds() == targetType.getAxisIds())
              changed |=
                  mergeRoles(roles[source], target, roles[target], true);
          }
      });
      // Reduction is a projection, not a layout-equivalence edge.  Its result
      // is projected from the selected input representation after all values
      // receive layouts; consumer conflicts are represented by convert_layout.
      getOperation().walk([&](riscv::ExtractOp extract) {
        mlir::Value input = extract.getInput();
        mlir::Value result = extract.getResult();
        if (!mlir::isa<riscv::ValueType>(input.getType()) ||
            !mlir::isa<riscv::ValueType>(result.getType()) ||
            roles[input].local || roles[result].local)
          return;
        // Extract removes axes selected by a point/index and preserves every
        // remaining logical axis.  Representation roles on those surviving
        // axes therefore propagate in both directions; consumed axes are
        // filtered by mergeRoles rather than rediscovered from source shape.
        changed |= mergeRoles(roles[input], result, roles[result]);
        Roles projectedResult = roles[result];
        const bool shapedGather = llvm::any_of(
            extract.getSelectors(), [](mlir::Attribute selector) {
              return mlir::cast<mlir::StringAttr>(selector).getValue() ==
                     "gather";
            });
        if (projectedResult.registerTuple && !shapedGather) {
          auto inputAxes = riscv_internal::logicalAxes(input.getType());
          auto resultAxes = riscv_internal::logicalAxes(result.getType());
          if (llvm::any_of(inputAxes, [&](int64_t axis) {
                return !llvm::is_contained(resultAxes, axis);
              })) {
            projectedResult.registerTuple = false;
            projectedResult.anchored = false;
          }
        }
        changed |= mergeRoles(projectedResult, input, roles[input]);
      });
      }
    };
    // First propagate operation and consumer anchors.  Only values that remain
    // unanchored then receive the target's base innermost-lane mapping.  That
    // default is itself a physical fact needed by downstream reductions and
    // control-flow handoffs, so run the same propagation to a second fixed
    // point instead of leaving it stranded on the defining value.
    propagateRoles();
    getOperation().walk([&](riscv::ReduceOp reduce) {
      mlir::Value input = reduce.getInput();
      auto inputType = mlir::dyn_cast<riscv::ValueType>(input.getType());
      auto axes = riscv_internal::logicalAxes(input.getType());
      if (!inputType ||
          reduce.getAxis() < 0 ||
          reduce.getAxis() >= static_cast<int64_t>(axes.size()))
        return;
      const int64_t eliminated = axes[reduce.getAxis()];
      Roles &inputRoles = roles[input];
      // A structured-product result deliberately remains a scalar register
      // tuple over its surviving free axes.  A following reduction requires a
      // different lane representation of that same logical value; preserve the
      // producer mapping here and insert the typed use-edge conversion below.
      // Ordinary shaped producers still anchor the eliminated axis directly.
      if (!inputRoles.laneAxis && !inputRoles.registerTuple)
        inputRoles.laneAxis = eliminated;
      inputRoles.anchored = true;
      if (inputRoles.laneAxis) {
        inputRoles.replicaAxes.erase(inputRoles.laneAxis);
        addSmallReplicas(input, inputRoles, inputRoles.laneAxis);
        inputRoles.replicaAxes.erase(eliminated);
      }
    });
    propagateRoles();
    for (mlir::Value value : values)
      completeRoles(value, roles[value]);

    // A widening contraction has one physical issue topology over its reduction
    // coordinates.  Operand-local storage propagation is only a proposal for
    // how to supply that common carrier: it cannot give the two operands
    // different entry/payload decompositions and leave the terminal operation
    // to reconcile them.  When both storage proposals agree, preserve them.  If
    // they disagree, use the canonical logical reduction order: the last
    // declared reduction axis is primary and the preceding reduction axes
    // coalesce outside it.  Memory planning may then realize either operand with
    // a direct load or an explicit representation conversion.  Unique output
    // axes remain owned by their respective operand.
    auto normalizeWidenDotReductionCarrier =
        [&](mlir::Operation *operation) -> bool {
      llvm::SmallVector<int64_t> reductionAxes;
      if (auto dot = mlir::dyn_cast<riscv::DotOp>(operation))
        reductionAxes.assign(dot.getOver().begin(), dot.getOver().end());
      else if (auto contract = mlir::dyn_cast<riscv::ContractOp>(operation))
        reductionAxes.assign(contract.getOver().begin(), contract.getOver().end());
      if (reductionAxes.empty())
        return false;
      auto implementation = operation->getAttrOfType<
          riscv::ImplementationAttr>("implementation");
      if (!implementation || implementation.getFamily() != "widen-dot")
        return false;
      mlir::Value lhs = operation->getOperand(0);
      mlir::Value rhs = operation->getOperand(1);
      // A common carrier is required when both operands contribute an
      // independently surviving free-axis domain, as in a blocked outer
      // contraction.  A scalar-output dot has no such relation: forcing its
      // storage/index producers through this anchor can only create an
      // unrelated lane/time repartition before the lookup or decode leaf.
      if (!hasOperandLocalFreeAxis(lhs, rhs, reductionAxes) ||
          !hasOperandLocalFreeAxis(rhs, lhs, reductionAxes))
        return false;
      auto lhsSupply = reductionSupplyRoles(lhs, reductionAxes);
      auto rhsSupply = reductionSupplyRoles(rhs, reductionAxes);
      Roles commonReductionCarrier;
      const Roles *anchor = nullptr;
      if (lhsSupply && rhsSupply &&
          lhsSupply->laneAxis == rhsSupply->laneAxis &&
          lhsSupply->coalescedLaneAxes == rhsSupply->coalescedLaneAxes)
        anchor = &*lhsSupply;
      else if (lhsSupply && !rhsSupply)
        anchor = &*lhsSupply;
      else if (!lhsSupply && rhsSupply)
        anchor = &*rhsSupply;
      if (!anchor) {
        auto lhsType = mlir::dyn_cast<riscv::ValueType>(lhs.getType());
        auto rhsType = mlir::dyn_cast<riscv::ValueType>(rhs.getType());
        bool commonDomain = lhsType && rhsType;
        for (int64_t axis : reductionAxes) {
          auto lhsAxis = lhsType
                             ? llvm::find(lhsType.getAxisIds().asArrayRef(), axis)
                             : llvm::ArrayRef<int64_t>::iterator();
          auto rhsAxis = rhsType
                             ? llvm::find(rhsType.getAxisIds().asArrayRef(), axis)
                             : llvm::ArrayRef<int64_t>::iterator();
          commonDomain &= lhsType && rhsType &&
                          lhsAxis != lhsType.getAxisIds().asArrayRef().end() &&
                          rhsAxis != rhsType.getAxisIds().asArrayRef().end();
          if (!commonDomain)
            break;
          const size_t lhsPosition = static_cast<size_t>(
              lhsAxis - lhsType.getAxisIds().asArrayRef().begin());
          const size_t rhsPosition = static_cast<size_t>(
              rhsAxis - rhsType.getAxisIds().asArrayRef().begin());
          commonDomain &= lhsType.getShape()[lhsPosition] > 0 &&
                          lhsType.getShape()[lhsPosition] ==
                              rhsType.getShape()[rhsPosition];
        }
        if (!commonDomain)
          return true;
        commonReductionCarrier.anchored = true;
        commonReductionCarrier.fullLaneExtent = true;
        commonReductionCarrier.laneAxis = reductionAxes.back();
        for (int64_t axis :
             llvm::ArrayRef<int64_t>(reductionAxes).drop_back())
          commonReductionCarrier.coalescedLaneAxes.insert(axis);
        anchor = &commonReductionCarrier;
      }
      for (mlir::Value operand : {lhs, rhs}) {
        Roles &operandRoles = roles[operand];
        for (int64_t axis : reductionAxes) {
          operandRoles.replicaAxes.erase(axis);
          operandRoles.coalescedLaneAxes.erase(axis);
          operandRoles.sequentialAxes.erase(axis);
        }
        operandRoles.laneAxis = anchor->laneAxis;
        for (int64_t axis : anchor->coalescedLaneAxes)
          operandRoles.coalescedLaneAxes.insert(axis);
        operandRoles.registerTuple = false;
        operandRoles.anchored = true;
        operandRoles.fullLaneExtent = anchor->fullLaneExtent;
        operandRoles.memoryAnchored = anchor->memoryAnchored;
      }
      return true;
    };
    getOperation().walk([&](mlir::Operation *operation) {
      if (normalizeWidenDotReductionCarrier(operation))
        return;
      // Only an outer contraction has two independently surviving free-axis
      // domains whose shared partial coordinates require one issue mapping.
      // Dot/contract and grouped-MAC reductions already anchor their single
      // surviving domain through the reduction result and must retain it.
      if (!mlir::isa<riscv::OuterContractOp>(operation))
        return;
      auto implementation =
          operation->getAttrOfType<riscv::ImplementationAttr>("implementation");
      if (!implementation || implementation.getFamily() != "widen-dot")
        return;
      mlir::Value lhs = operation->getOperand(0);
      mlir::Value rhs = operation->getOperand(1);
      auto lhsType = mlir::dyn_cast<riscv::ValueType>(lhs.getType());
      auto rhsType = mlir::dyn_cast<riscv::ValueType>(rhs.getType());
      if (!lhsType || !rhsType)
        return;
      llvm::SmallVector<int64_t> sharedAxes;
      for (auto [lhsPosition, axis] :
           llvm::enumerate(lhsType.getAxisIds().asArrayRef())) {
        auto rhsAxis = llvm::find(rhsType.getAxisIds().asArrayRef(), axis);
        if (rhsAxis == rhsType.getAxisIds().asArrayRef().end())
          continue;
        const size_t rhsPosition = static_cast<size_t>(
            rhsAxis - rhsType.getAxisIds().asArrayRef().begin());
        if (lhsType.getShape()[lhsPosition] != rhsType.getShape()[rhsPosition])
          return;
        sharedAxes.push_back(axis);
      }
      const bool lhsHasUniqueFreeAxis = llvm::any_of(
          lhsType.getAxisIds().asArrayRef(),
          [&](int64_t axis) { return !llvm::is_contained(sharedAxes, axis); });
      const bool rhsHasUniqueFreeAxis = llvm::any_of(
          rhsType.getAxisIds().asArrayRef(),
          [&](int64_t axis) { return !llvm::is_contained(sharedAxes, axis); });
      if (sharedAxes.empty() || !lhsHasUniqueFreeAxis ||
          !rhsHasUniqueFreeAxis || roles[lhs].laneAxis == 0 ||
          roles[lhs].laneAxis != roles[rhs].laneAxis ||
          !llvm::is_contained(sharedAxes, roles[lhs].laneAxis))
        return;
      auto sharedReplicaProduct = [&](mlir::Value value) {
        int64_t result = 1;
        for (int64_t axis : sharedAxes)
          if (roles[value].replicaAxes.contains(axis)) {
            const int64_t extent = riscv_internal::physicalExtent(value, axis);
            if (extent <= 0 || !checkedMultiply(result, extent, result))
              return int64_t{-1};
          }
        return result;
      };
      const int64_t lhsReplicas = sharedReplicaProduct(lhs);
      const int64_t rhsReplicas = sharedReplicaProduct(rhs);
      if (lhsReplicas <= 0 || rhsReplicas <= 0 || lhsReplicas == rhsReplicas)
        return;
      mlir::Value anchor = lhsReplicas < rhsReplicas ? lhs : rhs;
      mlir::Value target = lhsReplicas < rhsReplicas ? rhs : lhs;
      Roles &anchorRoles = roles[anchor];
      Roles &targetRoles = roles[target];
      for (int64_t axis : sharedAxes) {
        targetRoles.replicaAxes.erase(axis);
        targetRoles.coalescedLaneAxes.erase(axis);
        targetRoles.sequentialAxes.erase(axis);
        if (anchorRoles.replicaAxes.contains(axis))
          targetRoles.replicaAxes.insert(axis);
        if (anchorRoles.coalescedLaneAxes.contains(axis))
          targetRoles.coalescedLaneAxes.insert(axis);
        if (anchorRoles.sequentialAxes.contains(axis))
          targetRoles.sequentialAxes.insert(axis);
      }
      targetRoles.laneAxis = anchorRoles.laneAxis;
      targetRoles.registerTuple = false;
      targetRoles.anchored = true;
      targetRoles.fullLaneExtent = anchorRoles.fullLaneExtent;
    });
    propagateRoles();
    // The fixed-point propagation above carries the selected operand relation
    // through producers, but storage-facing proposals may also reintroduce a
    // reduction axis as issue time or a replica.  Reapply the same operation
    // anchor once, after propagation, so the final operand types consumed by
    // LowerRISCVComposites have one closed lane/time decomposition.  This is the
    // same owner and rule, not a second topology decision.
    getOperation().walk([&](mlir::Operation *operation) {
      normalizeWidenDotReductionCarrier(operation);
    });

    // The command-line LMUL binding fixes the base physical parameter, but a
    // width-changing use-def chain has one additional legality relation: a
    // widening value that preserves its logical lane span needs proportionally
    // more register width.  Freeze that derived width per SSA value and carry it
    // through the selected widening contraction.  This is not a second LMUL
    // candidate; it is the unique width required to keep the same logical lanes.
    llvm::DenseMap<mlir::Value, int64_t> valueLMULBounds;
    for (mlir::Value value : values)
      valueLMULBounds[value] = lmulEighths;
    bool widthChanged = true;
    while (widthChanged) {
      widthChanged = false;
      getOperation().walk([&](mlir::Operation *operation) {
        if (mlir::isa<riscv::CastOp, riscv::NarrowOp, riscv::WidenOp>(operation) &&
            operation->getNumOperands() == 1 && operation->getNumResults() == 1) {
          mlir::Value input = operation->getOperand(0);
          mlir::Value result = operation->getResult(0);
          auto inputType = mlir::dyn_cast<riscv::ValueType>(input.getType());
          auto resultType = mlir::dyn_cast<riscv::ValueType>(result.getType());
          if (!inputType || !resultType ||
              inputType.getShape() != resultType.getShape() ||
              inputType.getAxisIds() != resultType.getAxisIds() ||
              roles[input].laneAxis == 0 ||
              roles[input].laneAxis != roles[result].laneAxis ||
              roles[input].local || roles[result].local || roles[input].ime ||
              roles[result].ime)
            return;
          auto kernel = operation->getParentOfType<riscv::KernelOp>();
          if (!kernel)
            return;
          const int64_t inputSEW = std::max<int64_t>(
              8, riscv_internal::logicalBitWidth(inputType));
          const int64_t resultSEW = std::max<int64_t>(
              8, riscv_internal::logicalBitWidth(resultType));
          auto propagateWidth = [&](mlir::Value from, int64_t fromSEW,
                                    mlir::Value to, int64_t toSEW) {
            const int64_t fromWidth = valueLMULBounds.lookup(from);
            const int64_t requested =
                (fromWidth * toSEW + fromSEW - 1) / fromSEW;
            const int64_t legal =
                roundLegalLMUL(kernel.getTarget(), requested, toSEW);
            if (legal && legal > valueLMULBounds.lookup(to)) {
              valueLMULBounds[to] = legal;
              widthChanged = true;
            }
          };
          propagateWidth(input, inputSEW, result, resultSEW);
          propagateWidth(result, resultSEW, input, inputSEW);
          return;
        }

        if (!mlir::isa<riscv::DotOp, riscv::ContractOp,
                       riscv::OuterContractOp>(operation))
          return;
        auto implementation = operation->getAttrOfType<
            riscv::ImplementationAttr>("implementation");
        if (!implementation || implementation.getFamily() != "widen-dot")
          return;
        mlir::Value lhs = operation->getOperand(0);
        mlir::Value rhs = operation->getOperand(1);
        if (!valueLMULBounds.count(lhs) || !valueLMULBounds.count(rhs) ||
            roles[lhs].laneAxis == 0 ||
            roles[lhs].laneAxis != roles[rhs].laneAxis)
          return;
        auto kernel = operation->getParentOfType<riscv::KernelOp>();
        if (!kernel)
          return;
        const int64_t axis = roles[lhs].laneAxis;
        const int64_t lhsExtent = riscv_internal::physicalExtent(lhs, axis);
        const int64_t rhsExtent = riscv_internal::physicalExtent(rhs, axis);
        if (lhsExtent <= 0 || lhsExtent != rhsExtent)
          return;
        auto laneCapacity = [&](mlir::Value value) {
          const int64_t sew = std::max<int64_t>(
              8, riscv_internal::logicalBitWidth(value.getType()));
          return std::min(
              lhsExtent,
              std::max<int64_t>(
                  1, kernel.getTarget().getVlenBits() *
                         valueLMULBounds.lookup(value) / (8 * sew)));
        };
        const int64_t desired = std::max(laneCapacity(lhs), laneCapacity(rhs));
        for (mlir::Value value : {lhs, rhs}) {
          const int64_t sew = std::max<int64_t>(
              8, riscv_internal::logicalBitWidth(value.getType()));
          const int64_t requested =
              std::max<int64_t>(1, (desired * sew * 8 +
                                    kernel.getTarget().getVlenBits() - 1) /
                                       kernel.getTarget().getVlenBits());
          const int64_t legal =
              roundLegalLMUL(kernel.getTarget(), requested, sew);
          if (legal && legal > valueLMULBounds.lookup(value)) {
            valueLMULBounds[value] = legal;
            widthChanged = true;
          }
        }
      });
    }

    // Each physical value receives the largest lane span legal for its own
    // type and the instantiated LMUL bound.  Structured-product operands and
    // a same-axis extract remain coupled where the target instruction requires
    // it; ordinary pointwise mismatches become explicit convert_layout edges
    // instead of conservatively shrinking the whole use-def chain.
    llvm::DenseMap<mlir::Value, int64_t> connectedLaneSpans;
    for (mlir::Value value : values) {
      int64_t axis = roles[value].laneAxis;
      if (!axis || roles[value].local || roles[value].ime)
        continue;
      auto kernel = value.getParentRegion()->getParentOfType<riscv::KernelOp>();
      if (!kernel)
        continue;
      int64_t sew = std::max<int64_t>(
          8, riscv_internal::logicalBitWidth(value.getType()));
      int64_t capacity = std::max<int64_t>(
          1, kernel.getTarget().getVlenBits() * valueLMULBounds.lookup(value) /
                 (8 * sew));
      // Indexed memory makes a storage relation executable; it does not erase
      // the number of logical elements represented by one encoded layer.
      // Preserve that per-use storage width on every target, then express any
      // wider VLEN as additional issue-time strips.
      if (auto storageLimit = encodedLaneLimit(value, axis))
        capacity = std::min(capacity, *storageLimit);
      int64_t extent = riscv_internal::physicalExtent(value, axis);
      connectedLaneSpans[value] =
          extent > 0 ? std::min(extent, capacity) : capacity;
    }
    bool laneSpanChanged = true;
    while (laneSpanChanged) {
      laneSpanChanged = false;
      getOperation().walk([&](mlir::Operation *operation) {
        if (!mlir::isa<riscv::DotOp, riscv::ContractOp,
                       riscv::OuterContractOp>(operation)) {
          if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(operation)) {
            mlir::Value input = extract.getInput();
            mlir::Value result = extract.getResult();
            if (!connectedLaneSpans.count(input) ||
                !connectedLaneSpans.count(result) ||
                roles[input].laneAxis == 0 ||
                roles[input].laneAxis != roles[result].laneAxis)
              return;
            int64_t span =
                std::min(connectedLaneSpans[input], connectedLaneSpans[result]);
            for (mlir::Value value : {input, result})
              if (connectedLaneSpans[value] != span) {
                connectedLaneSpans[value] = span;
                laneSpanChanged = true;
              }
            return;
          } else {
            return;
          }
        }
        auto implementation = operation->getAttrOfType<riscv::ImplementationAttr>(
            "implementation");
        if (!implementation || implementation.getFamily() != "widen-dot")
          return;
        mlir::Value lhs = operation->getOperand(0);
        mlir::Value rhs = operation->getOperand(1);
        if (!connectedLaneSpans.count(lhs) || !connectedLaneSpans.count(rhs) ||
            roles[lhs].laneAxis == 0 ||
            roles[lhs].laneAxis != roles[rhs].laneAxis)
          return;
        int64_t span =
            std::min(connectedLaneSpans[lhs], connectedLaneSpans[rhs]);
        for (mlir::Value value : {lhs, rhs})
          if (connectedLaneSpans[value] != span) {
            connectedLaneSpans[value] = span;
            laneSpanChanged = true;
          }
      });
    }

    for (mlir::Value value : values) {
      int64_t connectedLaneSpan =
          std::max<int64_t>(1, connectedLaneSpans.lookup(value));
      riscv::LayoutAttr layout =
          buildLayout(builder, value, roles[value], connectedLaneSpan,
                      valueLMULBounds.lookup(value));
      if (!layout) {
        value.getParentRegion()->getParentOp()->emitError(
            "no legal LMUL can preserve the inferred logical lane mapping");
        failed = true;
        continue;
      }
      setValueLayout(value, layout);
    }
    // A register lookup uses the result's RVV register group as a table source.
    // The table may contain fewer active lanes than the result (for example a
    // 16-entry codebook feeding a 32-lane result on VLEN256), but vrgather.vv
    // still requires the same SEW/LMUL register type.  Propagate that physical
    // width through the explicit materialize edge; do not manufacture or merge
    // either logical axis.
    getOperation().walk([&](riscv::LookupOp lookup) {
      auto materialize = riscv_internal::stripRepresentationConversions(
                             lookup.getTable())
                             .getDefiningOp<riscv::MaterializeOp>();
      auto table = mlir::dyn_cast<riscv::ValueType>(lookup.getTable().getType());
      auto indices =
          mlir::dyn_cast<riscv::ValueType>(lookup.getIndices().getType());
      auto result = mlir::dyn_cast<riscv::ValueType>(lookup.getResult().getType());
      if (!materialize || materialize.getPlacement() != "shared" || !table ||
          !indices || !result || table.getLayout().getCarrier() != "rvv" ||
          indices.getLayout().getCarrier() != "rvv" ||
          result.getLayout().getCarrier() != "rvv")
        return;
      if (table.getLayout().getSew() != result.getLayout().getSew() ||
          indices.getLayout().getSew() != result.getLayout().getSew()) {
        lookup.emitError(
            "register lookup operands have no common RVV SEW representation");
        failed = true;
        return;
      }
      const int64_t requiredLMUL = result.getLayout().getLmulEighths();
      auto tableElements =
          riscv_internal::staticProduct(table.getShape().asArrayRef());
      const int64_t tableCapacity =
          lookup->getParentOfType<riscv::KernelOp>().getTarget().getVlenBits() *
          requiredLMUL / (8 * result.getLayout().getSew());
      // A table larger than the selected result register group remains an
      // addressable table.  PlanRISCVMemory will choose an indexed-memory leaf
      // for this physical instance instead of inventing a partial register
      // table or changing the logical lookup.
      if (!tableElements || *tableElements > tableCapacity)
        return;
      if (table.getLayout().getLmulEighths() == requiredLMUL)
        return;
      auto update = [&](mlir::Value value) {
        auto type = mlir::dyn_cast<riscv::ValueType>(value.getType());
        if (!type || type.getLayout().getCarrier() != "rvv" ||
            type.getLayout().getSew() != result.getLayout().getSew())
          return;
        setValueLayout(value,
                       withRegisterWidth(builder, type.getLayout(), requiredLMUL));
      };
      update(materialize.getInput());
      update(materialize.getResult());
    });
    if (mlir::failed(weft::planRISCVResidency(getOperation()))) {
      signalPassFailure();
      return;
    }
    getOperation().walk([&](riscv::NewOp state) {
      if (auto value = mlir::dyn_cast<riscv::ValueType>(state.getResult().getType()))
        state.setPlacementAttr(builder.getStringAttr(
            value.getLayout().getCarrier() == "local" ? "local" :
            value.getLayout().getCarrier() == "scalar" ? "scalar" : "register"));
    });
    if (failed) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<riscv::ConvertLayoutOp> redundantConversions;
    getOperation().walk([&](riscv::ConvertLayoutOp conversion) {
      if (conversion.getConversion().getEffect() == "pure" &&
          conversion.getInput().getType() == conversion.getResult().getType())
        redundantConversions.push_back(conversion);
    });
    for (riscv::ConvertLayoutOp conversion : redundantConversions) {
      conversion.getResult().replaceAllUsesWith(conversion.getInput());
      conversion.erase();
    }

    insertUseConversions(builder);
  }

private:
  static mlir::LogicalResult convertYieldOperand(mlir::OpBuilder &builder,
                                                 mlir::Operation *terminator,
                                                 unsigned index,
                                                 mlir::Type required) {
    mlir::Value value = terminator->getOperand(index);
    if (value.getType() == required)
      return mlir::success();
    auto source = mlir::dyn_cast<riscv::ValueType>(value.getType());
    auto target = mlir::dyn_cast<riscv::ValueType>(required);
    if (!source || !target || source.getElementType() != target.getElementType() ||
        source.getShape() != target.getShape() ||
        source.getAxisIds() != target.getAxisIds())
      return terminator->emitError(
          "control-flow handoff changes the logical value domain");
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(terminator);
    auto conversion = builder.create<riscv::ConvertLayoutOp>(
        terminator->getLoc(), target, value,
        riscv_internal::layoutConversion(builder, source.getLayout(),
                                         target.getLayout()),
        riscv::AccessAttr(),
        riscv_internal::unselectedLeaf(builder));
    terminator->setOperand(index, conversion.getResult());
    return mlir::success();
  }

  bool reconcileControlCarries(mlir::OpBuilder &builder) {
    bool failed = false;
    getOperation().walk([&](riscv::LoopOp loop) {
      mlir::Block &body = loop.getBody().front();
      auto yield = mlir::cast<riscv::YieldOp>(body.getTerminator());
      for (auto [index, carried] : llvm::enumerate(loop.getCarried())) {
        mlir::Type type = carried.getType();
        body.getArgument(index + 1).setType(type);
        loop.getResult(index).setType(type);
        failed |= mlir::failed(convertYieldOperand(builder, yield, index, type));
      }
    });
    getOperation().walk([&](mlir::scf::ForOp loop) {
      for (auto [index, initial] : llvm::enumerate(loop.getInitArgs())) {
        mlir::Type type = initial.getType();
        loop.getRegionIterArg(index).setType(type);
        loop.getResult(index).setType(type);
        failed |= mlir::failed(convertYieldOperand(
            builder, loop.getBody()->getTerminator(), index, type));
      }
    });
    getOperation().walk([&](mlir::scf::IfOp branch) {
      for (auto [index, result] : llvm::enumerate(branch.getResults())) {
        failed |= mlir::failed(convertYieldOperand(
            builder, branch.thenYield(), index, result.getType()));
        failed |= mlir::failed(convertYieldOperand(
            builder, branch.elseYield(), index, result.getType()));
      }
    });
    getOperation().walk([&](mlir::scf::WhileOp loop) {
      auto condition = mlir::cast<mlir::scf::ConditionOp>(
          loop.getBefore().front().getTerminator());
      auto yield = mlir::cast<mlir::scf::YieldOp>(
          loop.getAfter().front().getTerminator());
      for (auto [index, initial] : llvm::enumerate(loop.getInits())) {
        mlir::Type type = initial.getType();
        loop.getBefore().front().getArgument(index).setType(type);
        loop.getAfter().front().getArgument(index).setType(type);
        loop.getResult(index).setType(type);
        failed |= mlir::failed(
            convertYieldOperand(builder, condition, index + 1, type));
        failed |= mlir::failed(convertYieldOperand(builder, yield, index, type));
      }
    });
    return failed;
  }

  bool reconcileStateInitializers(mlir::OpBuilder &builder) {
    bool failed = false;
    llvm::SmallVector<riscv::NewOp> states;
    getOperation().walk([&](riscv::NewOp state) {
      if (state.getInitialized())
        states.push_back(state);
    });
    for (riscv::NewOp state : states) {
      mlir::Value initial = state.getInitial();
      auto source = mlir::dyn_cast<riscv::ValueType>(initial.getType());
      auto target = mlir::dyn_cast<riscv::ValueType>(state.getResult().getType());
      // Canonical `new` permits a scalar initializer to broadcast to a shaped
      // state.  That is a numerical rule and does not require a layout edge.
      if (!source || !target || initial.getType() == state.getResult().getType())
        continue;
      if (source.getElementType() != target.getElementType() ||
          source.getShape() != target.getShape() ||
          source.getAxisIds() != target.getAxisIds()) {
        state.emitError(
            "physical state initializer changes the canonical logical domain");
        failed = true;
        continue;
      }
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(state);
      auto conversion = builder.create<riscv::ConvertLayoutOp>(
          state.getLoc(), target, initial,
          riscv_internal::layoutConversion(builder, source.getLayout(),
                                           target.getLayout()),
          riscv::AccessAttr(),
          riscv_internal::unselectedLeaf(builder));
      state.getInitialMutable().assign(conversion.getResult());
    }
    return failed;
  }

  bool projectReductionResult(mlir::OpBuilder &builder,
                              riscv::ReduceOp reduce) {
    auto input = mlir::dyn_cast<riscv::ValueType>(reduce.getInput().getType());
    auto result = mlir::dyn_cast<riscv::ValueType>(reduce.getResult().getType());
    if (!input || reduce.getAxis() < 0 ||
        reduce.getAxis() >= static_cast<int64_t>(input.getAxisIds().size())) {
      reduce.emitError(
          "cannot project a physical reduction result from its input layout");
      return false;
    }
    const size_t eliminated = static_cast<size_t>(reduce.getAxis());
    if (!result) {
      if (input.getAxisIds().size() == 1)
        return true;
      reduce.emitError(
          "a shaped reduction result is required when free axes survive");
      return false;
    }
    llvm::SmallVector<int64_t> expectedShape;
    llvm::SmallVector<int64_t> expectedAxes;
    for (size_t position = 0; position < input.getAxisIds().size(); ++position) {
      if (position == eliminated)
        continue;
      expectedShape.push_back(input.getShape()[position]);
      expectedAxes.push_back(input.getAxisIds()[position]);
    }
    if (result.getShape().asArrayRef() !=
            llvm::ArrayRef<int64_t>(expectedShape) ||
        result.getAxisIds().asArrayRef() !=
            llvm::ArrayRef<int64_t>(expectedAxes)) {
      reduce.emitError(
          "physical reduction result does not equal its input domain with one axis removed");
      return false;
    }
    auto kernel = reduce->getParentOfType<riscv::KernelOp>();
    riscv::LayoutAttr projected = riscv_internal::projectLayout(
        builder, result, input.getLayout(), kernel.getTarget());
    if (!projected) {
      reduce.emitError(
          "no legal target layout preserves every reduction free axis");
      return false;
    }
    setValueLayout(reduce.getResult(), projected);
    return true;
  }

  void insertUseConversions(mlir::OpBuilder &builder) {
    // A contraction may produce one scalar register tuple over its surviving
    // free axes while a downstream reduction consumes one of those axes in
    // RVV lanes.  Keep both operation anchors intact and make their handoff a
    // first-class physical conversion instead of forcing one layout onto the
    // shared SSA value.
    llvm::SmallVector<riscv::ReduceOp> reductions;
    getOperation().walk(
        [&](riscv::ReduceOp reduce) { reductions.push_back(reduce); });
    for (riscv::ReduceOp reduce : reductions) {
      if (!projectReductionResult(builder, reduce)) {
        signalPassFailure();
        continue;
      }
      mlir::Value input = reduce.getInput();
      auto inputType = mlir::dyn_cast<riscv::ValueType>(input.getType());
      if (!inputType || reduce.getAxis() < 0 ||
          reduce.getAxis() >=
              static_cast<int64_t>(inputType.getAxisIds().size()))
        continue;
      const int64_t eliminated = inputType.getAxisIds()[reduce.getAxis()];
      if (input.hasOneUse() &&
          mlir::isa_and_nonnull<riscv::DotOp, riscv::ContractOp,
                                riscv::OuterContractOp>(
              input.getDefiningOp()))
        continue;
      const auto axes = inputType.getAxisIds().asArrayRef();
      const auto time = inputType.getLayout().getTimeFactors().asArrayRef();
      const auto lane = inputType.getLayout().getLaneFactors().asArrayRef();
      const auto replica =
          inputType.getLayout().getReplicaFactors().asArrayRef();
      auto position = llvm::find(axes, eliminated);
      if (position == axes.end())
        continue;
      const size_t ordinal = static_cast<size_t>(position - axes.begin());
      if (ordinal < lane.size() && lane[ordinal] > 1)
        continue;
      // A reduction may consume an issue-time coordinate while preserving an
      // independent SIMD coordinate carried by a surviving free axis.  This is
      // already a closed parent representation: the reduction removes only the
      // selected time factor and does not require a lane transpose.  Treating
      // every non-lane reduction as a layout conflict used to reject grouped
      // reductions whose target lowering explicitly accumulates their streams.
      if (inputType.getLayout().getCarrier() == "rvv" &&
          ordinal < time.size() && time[ordinal] > 1 &&
          ordinal < replica.size() && replica[ordinal] == 1 &&
          llvm::any_of(llvm::enumerate(lane), [&](auto entry) {
            return entry.index() != ordinal && entry.value() > 1;
          }))
        continue;
      if (inputType.getLayout().getCarrier() != "scalar" &&
          inputType.getLayout().getCarrier() != "rvv") {
        auto diagnostic = reduce.emitError(
            "reduction layout conflict has neither a lane nor a time-axis realization");
        if (mlir::Operation *definition = input.getDefiningOp())
          diagnostic << "; producer=" << definition->getName();
        diagnostic << "; layout=" << inputType.getLayout();
        signalPassFailure();
        continue;
      }
      Roles consumerRoles;
      consumerRoles.anchored = true;
      consumerRoles.fullLaneExtent = true;
      auto resultType =
          mlir::dyn_cast<riscv::ValueType>(reduce.getResult().getType());
      int64_t connectedLaneSpan =
          std::max<int64_t>(1,
                            riscv_internal::physicalExtent(input, eliminated));
      if (resultType) {
        for (auto [axis, resultTime, resultLane, resultReplica] : llvm::zip(
                 resultType.getAxisIds().asArrayRef(),
                 resultType.getLayout().getTimeFactors().asArrayRef(),
                 resultType.getLayout().getLaneFactors().asArrayRef(),
                 resultType.getLayout().getReplicaFactors().asArrayRef())) {
          if (resultLane > 1) {
            if (!consumerRoles.laneAxis)
              consumerRoles.laneAxis = axis;
            else
              consumerRoles.coalescedLaneAxes.insert(axis);
            if (connectedLaneSpan <=
                std::numeric_limits<int64_t>::max() / resultLane)
              connectedLaneSpan *= resultLane;
          }
          if (resultReplica > 1)
            consumerRoles.replicaAxes.insert(axis);
          if (resultTime > 1)
            consumerRoles.sequentialAxes.insert(axis);
        }
      }
      if (consumerRoles.laneAxis)
        consumerRoles.coalescedLaneAxes.insert(eliminated);
      else
        consumerRoles.laneAxis = eliminated;
      addSmallReplicas(input, consumerRoles, consumerRoles.laneAxis);
      consumerRoles.replicaAxes.erase(eliminated);
      riscv::LayoutAttr required =
          buildLayout(builder, input, consumerRoles, connectedLaneSpan,
                      std::max<int64_t>(
                          lmulEighths,
                          inputType.getLayout().getLmulEighths()));
      if (!required) {
        reduce.emitError(
            "no legal RVV layout can consume the register-axis reduction");
        signalPassFailure();
        continue;
      }
      mlir::Type requiredType =
          riscv_internal::withLayout(input.getType(), required);
      if (requiredType == input.getType())
        continue;
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(reduce);
      auto conversion = builder.create<riscv::ConvertLayoutOp>(
          reduce.getLoc(), requiredType, input,
          riscv_internal::layoutConversion(builder, inputType.getLayout(),
                                           required),
          riscv::AccessAttr(), riscv_internal::unselectedLeaf(builder));
      reduce.getInputMutable().assign(conversion.getResult());
      if (!projectReductionResult(builder, reduce))
        signalPassFailure();
    }
    if (reconcileControlCarries(builder) ||
        reconcileStateInitializers(builder)) {
      signalPassFailure();
      return;
    }
    llvm::SmallVector<mlir::Operation *> operations;
    getOperation().walk([&](mlir::Operation *operation) {
      // Pointwise operations require one shared element mapping.  Structured
      // products do not: free and reduction operands deliberately retain
      // different lane/register decompositions and are reconciled by their
      // typed RVV step or IME pack operations during composite lowering.
      if (auto materialize = mlir::dyn_cast<riscv::MaterializeOp>(operation);
          materialize && (materialize.getPlacement() == "reload" ||
                          materialize.getPlacement() == "local"))
        return;
      if (pointwiseLike(operation) || mlir::isa<riscv::LookupOp>(operation))
        operations.push_back(operation);
    });
    for (mlir::Operation *operation : operations) {
      if (operation->getNumResults() != 1 ||
          !mlir::isa<riscv::ValueType>(operation->getResult(0).getType()))
        continue;
      auto result = mlir::cast<riscv::ValueType>(operation->getResult(0).getType());
      for (mlir::OpOperand &operand : operation->getOpOperands()) {
        if (!mlir::isa<riscv::ValueType>(operand.get().getType()))
          continue;
        if (mlir::isa<riscv::LookupOp>(operation) &&
            operand.getOperandNumber() == 0)
          continue;
        auto kernel = operation->getParentOfType<riscv::KernelOp>();
        mlir::Value sourceValue = operand.get();
        auto operandType = mlir::cast<riscv::ValueType>(sourceValue.getType());
        riscv::LayoutAttr required =
            mlir::isa<riscv::MaterializeOp>(operation) &&
                    operandType.getElementType() == result.getElementType() &&
                    operandType.getShape() == result.getShape() &&
                    operandType.getAxisIds() == result.getAxisIds()
                ? result.getLayout()
                : riscv_internal::projectLayout(builder, operandType,
                                                result.getLayout(),
                                                kernel.getTarget());
        if (!required) {
          if (auto lookup = mlir::dyn_cast<riscv::LookupOp>(operation);
              lookup && operand.getOperandNumber() == 1 &&
              riscv_internal::analyzeIndexedEntryRelation(
                  sourceValue, result, {}, {}))
            continue;
          operation->emitError()
              << "no legal target layout projects pointwise result "
              << result << " to operand " << operandType;
          signalPassFailure();
          continue;
        }
        mlir::Type requiredType =
            riscv_internal::withLayout(sourceValue.getType(), required);
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(operation);
        if (sourceValue.getType() != requiredType) {
          auto conversion = builder.create<riscv::ConvertLayoutOp>(
              operation->getLoc(), requiredType, sourceValue,
              riscv_internal::layoutConversion(builder,
                                               operandType.getLayout(), required),
              riscv::AccessAttr(),
              riscv_internal::unselectedLeaf(builder));
          riscv_internal::copyOrigin(operation, conversion);
          sourceValue = conversion.getResult();
          operandType = mlir::cast<riscv::ValueType>(sourceValue.getType());
        }

        if (operandType.getShape() == result.getShape() &&
            operandType.getAxisIds() == result.getAxisIds()) {
          operand.set(sourceValue);
          continue;
        }
        bool properSubdomain = operandType.getAxisIds().size() <
                                   result.getAxisIds().size() &&
                               operandType.getElementType() ==
                                   result.getElementType() &&
                               operandType.getLayout().getCarrier() == "rvv" &&
                               result.getLayout().getCarrier() == "rvv" &&
                               operandType.getLayout().getSew() ==
                                   result.getLayout().getSew();
        for (auto [position, axis] :
             llvm::enumerate(operandType.getAxisIds().asArrayRef())) {
          auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
          properSubdomain &=
              found != result.getAxisIds().asArrayRef().end() &&
              operandType.getShape()[position] ==
                  result.getShape()[static_cast<size_t>(
                      found - result.getAxisIds().asArrayRef().begin())];
        }
        auto inputLanes = riscv_internal::staticProduct(
            operandType.getLayout().getLaneFactors().asArrayRef());
        auto resultLanes = riscv_internal::staticProduct(
            result.getLayout().getLaneFactors().asArrayRef());
        // Missing logical axes that remain scalar time/register coordinates are
        // handled by the ordinary pointwise projection.  A physical broadcast
        // is needed only when the result introduces additional SIMD lanes.
        if (inputLanes && resultLanes && *resultLanes <= *inputLanes) {
          operand.set(sourceValue);
          continue;
        }
        if (operandType.getLayout().getCarrier() == "scalar") {
          operand.set(sourceValue);
          continue;
        }
        if (!properSubdomain || !inputLanes || !resultLanes ||
            *inputLanes <= 0 || *resultLanes <= *inputLanes) {
          operation->emitError(
              "pointwise broadcast has no typed RVV sub-domain realization; operand=")
              << operandType << ", result=" << result
              << ", proper_subdomain=" << properSubdomain
              << ", input_lanes=" << inputLanes.value_or(-1)
              << ", result_lanes=" << resultLanes.value_or(-1);
          signalPassFailure();
          continue;
        }
        auto broadcast = builder.create<riscv::RVVAxisBroadcastOp>(
            operation->getLoc(), operation->getResult(0).getType(), sourceValue,
            riscv_internal::leaf(
                builder, "rvv", "axis-broadcast", "rvv.axis-broadcast",
                "rvv.axis-broadcast",
                operandType.getLayout().getRegisterGroups(),
                result.getLayout().getRegisterGroups(), 1, 0, "none", "exact",
                {*inputLanes, *resultLanes}));
        riscv_internal::copyOrigin(operation, broadcast);
        operand.set(broadcast.getResult());
      }
    }

    // Structured products do not require one common full operand layout.  Each
    // operand owns its storage-compatible DotOperand-style representation; the
    // selected widening operation requires only equal reduction slices.  The
    // later partial plan records the exact part and lane offset taken from each
    // operand.  Forcing the wider operand through the narrower layout (or vice
    // versa) duplicates loads and recreates a second layout owner here.

    llvm::SmallVector<riscv::ExtractOp> gathers;
    getOperation().walk([&](riscv::ExtractOp extract) {
      if (llvm::any_of(extract.getSelectors(), [](mlir::Attribute selector) {
            return mlir::cast<mlir::StringAttr>(selector).getValue() ==
                   "gather";
          }))
        gathers.push_back(extract);
    });
    for (riscv::ExtractOp extract : gathers) {
      auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
      if (!result)
        continue;
      auto input = mlir::dyn_cast<riscv::ValueType>(extract.getInput().getType());
      auto encodedField = riscv_internal::sourceField(extract.getInput());
      const bool hasGatherSelector =
          llvm::any_of(extract.getSelectors(), [](mlir::Attribute selector) {
            return mlir::cast<mlir::StringAttr>(selector).getValue() == "gather";
          });
      const riscv_internal::FieldFacts encodedFacts =
          encodedField ? riscv_internal::fieldFacts(encodedField)
                       : riscv_internal::FieldFacts();
      const bool typedRegularJoined =
          encodedField && hasGatherSelector && encodedFacts.mapping == "joined" &&
          encodedFacts.bitOffset % 8 == 0 &&
          llvm::any_of(extract.getIndices(), [](mlir::Value index) {
            auto relation =
                riscv_internal::analyzeRegularIndexRelation(index);
            return relation && relation->stride == 1 && relation->repeat > 1;
          });
      const bool typedEncodedEntry =
          encodedField && hasGatherSelector &&
          (encodedFacts.mapping == "natural" || typedRegularJoined) &&
          encodedFacts.bitOffset % 8 == 0 &&
          !extract.getIndices().empty() &&
          llvm::all_of(extract.getIndices(), [](mlir::Value index) {
            mlir::Type element = riscv_internal::logicalElement(index.getType());
            auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
            return element.isIndex() || (integer && integer.isUnsigned());
          });
      if (input && input.getLayout().getCarrier() == "local" &&
          result.getLayout().getCarrier() == "rvv" &&
          input.getElementType() == result.getElementType() &&
          !typedEncodedEntry) {
        auto kernel = extract->getParentOfType<riscv::KernelOp>();
        auto required = registerGatherSourceLayout(
            builder, input, result.getLayout(), kernel.getTarget());
        if (required) {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPoint(extract);
          auto conversion = builder.create<riscv::ConvertLayoutOp>(
              extract.getLoc(),
              riscv_internal::withLayout(input, required), extract.getInput(),
              riscv_internal::layoutConversion(builder, input.getLayout(),
                                               required),
              riscv::AccessAttr(),
              riscv_internal::unselectedLeaf(builder));
          extract.getInputMutable().assign(conversion.getResult());
        }
      }
      size_t indexCursor = 0;
      for (auto [dimension, selectorAttribute] :
           llvm::enumerate(extract.getSelectors())) {
        llvm::StringRef selector =
            mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
        if (selector == "all")
          continue;
        if (indexCursor >= extract.getIndices().size()) {
          extract.emitError("gather selector has no physical index operand");
          signalPassFailure();
          break;
        }
        mlir::OpOperand &operand =
            extract->getOpOperand(1 + indexCursor++);
        if (selector != "gather" ||
            !mlir::isa<riscv::ValueType>(operand.get().getType()))
          continue;
        auto kernel = extract->getParentOfType<riscv::KernelOp>();
        riscv::LayoutAttr required = riscv_internal::projectLayout(
            builder, mlir::cast<riscv::ValueType>(operand.get().getType()),
            result.getLayout(), kernel.getTarget());
        if (!required) {
          llvm::SmallVector<int64_t> retainedAxes;
          llvm::SmallVector<int64_t> retainedShape;
          if (input)
            for (size_t position = 0; position < input.getAxisIds().size();
                 ++position) {
              if (position == dimension)
                continue;
              retainedAxes.push_back(input.getAxisIds()[position]);
              retainedShape.push_back(input.getShape()[position]);
            }
          if (riscv_internal::analyzeIndexedEntryRelation(
                  operand.get(), result, retainedAxes, retainedShape))
            continue;
          if (typedRegularJoined) {
            auto relation =
                riscv_internal::analyzeRegularIndexRelation(operand.get());
            if (relation && relation->stride == 1 && relation->repeat > 1)
              continue;
          }
          extract.emitError(
              "no legal target layout projects the gather result mapping to its index");
          signalPassFailure();
          continue;
        }
        mlir::Type requiredType =
            riscv_internal::withLayout(operand.get().getType(), required);
        if (operand.get().getType() == requiredType)
          continue;
        auto source =
            mlir::cast<riscv::ValueType>(operand.get().getType()).getLayout();
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPoint(extract);
        auto conversion = builder.create<riscv::ConvertLayoutOp>(
            extract.getLoc(), requiredType, operand.get(),
            riscv_internal::layoutConversion(builder, source, required),
            riscv::AccessAttr(),
            riscv_internal::unselectedLeaf(builder));
        operand.set(conversion.getResult());
      }
    }
  }

private:
  int64_t lmulEighths;
};

} // namespace

std::unique_ptr<mlir::Pass>
weft::createPropagateRISCVLayoutsPass(int64_t lmulEighths) {
  return std::make_unique<PropagateRISCVLayoutsPass>(lmulEighths);
}
