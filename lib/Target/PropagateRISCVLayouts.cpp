#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

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

std::optional<Roles> storageRolesFor(mlir::Value value) {
  auto extract = value.getDefiningOp<riscv::ExtractOp>();
  auto result = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!extract || !result)
    return std::nullopt;
  llvm::SmallVector<mlir::Value> worklist{value};
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;
  bool feedsWideningContraction = false;
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
  if (facts.mapping == "natural") {
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
      roles.coalescedLaneAxes.insert(nextAxis);
      consumed.insert(nextAxis);
      if (!checkedMultiply(span, nextExtent, span))
        return std::nullopt;
    }
  }
  roles.anchored = true;
  roles.fullLaneExtent = true;
  roles.memoryAnchored = true;
  for (int64_t axis : riscv_internal::logicalAxes(value.getType())) {
    const int64_t extent = riscv_internal::physicalExtent(value, axis);
    if (axis != contiguousAxis && extent > 0 && extent <= 16)
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

llvm::SmallVector<int64_t>
downstreamReducedFreeAxes(mlir::Operation *contraction,
                          llvm::ArrayRef<int64_t> reductionAxes) {
  llvm::SmallVector<int64_t> reducedAxes;
  if (!contraction || contraction->getNumResults() != 1)
    return reducedAxes;
  llvm::SmallVector<mlir::Value> worklist{contraction->getResult(0)};
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;
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
        if (!llvm::is_contained(reductionAxes, axis) &&
            !llvm::is_contained(reducedAxes, axis))
          reducedAxes.push_back(axis);
        worklist.push_back(reduce.getResult());
        continue;
      }
      if (!layoutPreservingPointwise(user) || user->getNumResults() != 1)
        continue;
      auto result = mlir::dyn_cast<riscv::ValueType>(user->getResult(0).getType());
      if (!result || result.getShape() != source.getShape() ||
          result.getAxisIds() != source.getAxisIds())
        continue;
      worklist.push_back(user->getResult(0));
    }
  }
  return reducedAxes;
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

void constrainGroupedMac(riscv::MacGroupsOp operation, mlir::Value value,
                         Roles &roles) {
  roles.anchored = true;
  roles.fullLaneExtent = true;
  // A grouped MAC consumes one shared reduction coordinate.  If one operand
  // also carries an output/cohort axis, that free axis is the reusable vector
  // direction.  Falling back to the final reduction axis is only valid for a
  // scalar-output vec-dot.  This is an axis relation, not a source-shape
  // matcher: canonical and derived encodings use the same rule.
  int64_t lane =
      freeAxisSharedByOneOperand(operation.getLhs(), operation.getRhs());
  int64_t groupedAxis = 0;
  auto lhsAxes = riscv_internal::logicalAxes(operation.getLhs().getType());
  auto rhsAxes = riscv_internal::logicalAxes(operation.getRhs().getType());
  if (!lhsAxes.empty() && !rhsAxes.empty() && lhsAxes.back() == rhsAxes.back())
    groupedAxis = lhsAxes.back();
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
       implementation.getFamily() == "widen-dot")) {
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
  for (int64_t axis : reduction)
    if (containsAxis(value.getType(), axis)) {
      roles.laneAxis = axis;
      break;
    }
  // A free axis that is consumed by a downstream reduction is not an output
  // microtile that must remain as register replicas.  It is a segmented lane
  // coordinate of the same local contraction.  Coalesce as much of it as the
  // instantiated LMUL permits; any remainder stays as explicit issue time.
  // This is derived from the typed contraction/use relation and applies to
  // dense, bit-plane, and codebook inputs alike.
  if (roles.laneAxis)
    for (int64_t freeAxis :
         downstreamReducedFreeAxes(operation.getOperation(), reduction))
      if (freeAxis != roles.laneAxis &&
          containsAxis(value.getType(), freeAxis)) {
        roles.coalescedLaneAxes.insert(freeAxis);
        roles.replicaAxes.erase(freeAxis);
      }
  // A shaped contraction result no longer contains the eliminated reduction
  // axis.  Its surviving free axes form a register/time tuple; a downstream
  // pointwise scale or reduction must not re-anchor one of those free axes as
  // the contraction's SIMD lane merely because that consumer uses lanes.
  if (!roles.laneAxis) {
    roles.registerTuple = true;
    addSmallReplicas(value, roles);
    return;
  }
  for (int64_t axis : riscv_internal::logicalAxes(value.getType()))
    if (!llvm::is_contained(reduction, axis) && axis != roles.laneAxis &&
        riscv_internal::physicalExtent(value, axis) > 0 &&
        riscv_internal::physicalExtent(value, axis) <= 16)
      roles.replicaAxes.insert(axis);
}

Roles rolesFor(mlir::Value value) {
  Roles roles;
  auto element = riscv_internal::logicalElement(value.getType());
  if (mlir::isa<kernel::EncodingType>(element))
    return roles;

  if (auto storage = storageRolesFor(value))
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
      auto axes = riscv_internal::logicalAxes(lookup.getIndices().getType());
      if (!axes.empty() && containsAxis(value.getType(), axes.back()))
        roles.laneAxis = axes.back();
      addSmallReplicas(value, roles, roles.laneAxis);
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
  auto axes = riscv_internal::logicalAxes(value.getType());
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

bool isDenseLoadBacked(riscv::MaterializeOp materialize) {
  auto load = riscv_internal::sourceLoad(materialize.getInput());
  if (!load)
    return false;
  auto encoding =
      mlir::dyn_cast<kernel::EncodingType>(load.getRegion().getType().getEncoding());
  return encoding && encoding.getKind() == "dense";
}

bool requiresAddressableReload(riscv::MaterializeOp materialize,
                               riscv::TargetAttr target) {
  auto value = mlir::dyn_cast<riscv::ValueType>(materialize.getResult().getType());
  if (!value || value.getLayout().getCarrier() != "rvv" ||
      !isDenseLoadBacked(materialize))
    return false;

  bool hasExtract = false;
  bool allExtracts = true;
  bool registerProjectionUnsupported = false;
  for (mlir::Operation *user : materialize.getResult().getUsers()) {
    auto extract = mlir::dyn_cast<riscv::ExtractOp>(user);
    if (!extract || extract.getInput() != materialize.getResult()) {
      allExtracts = false;
      continue;
    }
    hasExtract = true;
    int64_t selectedAxes = 0;
    for (mlir::Attribute selector : extract.getSelectors())
      if (mlir::cast<mlir::StringAttr>(selector).getValue() != "all")
        ++selectedAxes;
    registerProjectionUnsupported |= selectedAxes != 1;
  }
  if (!hasExtract || !allExtracts)
    return false;

  auto timeParts = riscv_internal::staticProduct(
      value.getLayout().getTimeFactors().asArrayRef());
  if (!timeParts)
    return true;
  const int64_t residentGroups =
      *timeParts * value.getLayout().getRegisterGroups();
  return registerProjectionUnsupported ||
         residentGroups > target.getVectorRegisters();
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
      getOperation().walk([&](riscv::ReduceOp reduce) {
        mlir::Value input = reduce.getInput();
        mlir::Value result = reduce.getResult();
        if (!mlir::isa<riscv::ValueType>(input.getType()) ||
            !mlir::isa<riscv::ValueType>(result.getType()))
          return;
        changed |= mergeRoles(roles[input], result, roles[result]);
        changed |= mergeRoles(roles[result], input, roles[input]);
      });
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
        if (projectedResult.registerTuple) {
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
      auto axes = riscv_internal::logicalAxes(input.getType());
      if (!mlir::isa<riscv::ValueType>(input.getType()) ||
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
    propagateRoles();

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
          1, kernel.getTarget().getVlenBits() * lmulEighths / (8 * sew));
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
                      lmulEighths);
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
    // Canonical materialize fixes the value's birth, lifetime, and sharing
    // scope.  It does not require the complete logical panel to reside in
    // registers.  A dense load-backed panel whose direct consumers are
    // projections remains addressable when the selected RVV representation
    // cannot exist as one legal register-resident value.  Composite lowering
    // will turn those projections into loads from the staged view.
    getOperation().walk([&](riscv::MaterializeOp materialize) {
      if (materialize.getPlacement() != "shared")
        return;
      if (auto value =
              mlir::dyn_cast<riscv::ValueType>(materialize.getResult().getType());
          value && value.getLayout().getCarrier() == "local" &&
          !mlir::isa<kernel::EncodingType>(value.getElementType())) {
        materialize.setPlacementAttr(builder.getStringAttr("local"));
        return;
      }
      auto kernel = materialize->getParentOfType<riscv::KernelOp>();
      if (kernel && requiresAddressableReload(materialize, kernel.getTarget()))
        materialize.setPlacementAttr(builder.getStringAttr("reload"));
    });
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

  void insertUseConversions(mlir::OpBuilder &builder) {
    if (reconcileControlCarries(builder) ||
        reconcileStateInitializers(builder)) {
      signalPassFailure();
      return;
    }
    // A contraction may produce one scalar register tuple over its surviving
    // free axes while a downstream reduction consumes one of those axes in
    // RVV lanes.  Keep both operation anchors intact and make their handoff a
    // first-class physical conversion instead of forcing one layout onto the
    // shared SSA value.
    llvm::SmallVector<riscv::ReduceOp> reductions;
    getOperation().walk(
        [&](riscv::ReduceOp reduce) { reductions.push_back(reduce); });
    for (riscv::ReduceOp reduce : reductions) {
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
      const auto lane = inputType.getLayout().getLaneFactors().asArrayRef();
      auto position = llvm::find(axes, eliminated);
      if (position == axes.end())
        continue;
      const size_t ordinal = static_cast<size_t>(position - axes.begin());
      if (ordinal < lane.size() && lane[ordinal] > 1)
        continue;
      if (inputType.getLayout().getCarrier() != "scalar") {
        reduce.emitError(
            "reduction layout conflict requires an unsupported non-scalar lane remap");
        signalPassFailure();
        continue;
      }
      Roles consumerRoles;
      consumerRoles.anchored = true;
      consumerRoles.fullLaneExtent = true;
      consumerRoles.laneAxis = eliminated;
      addSmallReplicas(input, consumerRoles, eliminated);
      const int64_t extent =
          std::max<int64_t>(1,
                            riscv_internal::physicalExtent(input, eliminated));
      riscv::LayoutAttr required =
          buildLayout(builder, input, consumerRoles, extent, lmulEighths);
      if (!required) {
        reduce.emitError(
            "no legal RVV layout can consume the scalar register-axis reduction");
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
          operation->emitError(
              "no legal target layout projects the pointwise result mapping to its operand");
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

    // A selected widening contraction anchors one common operand layout.  A
    // storage-contiguous producer may deliberately keep a narrower lane window
    // and therefore reaches the contraction through an explicit typed
    // register-to-lane conversion.  This is the same conflict boundary used by
    // pointwise operations above; do not force the compute layout back through
    // the memory edge.
    llvm::SmallVector<mlir::Operation *> wideningContractions;
    getOperation().walk([&](mlir::Operation *operation) {
      if (!mlir::isa<riscv::DotOp, riscv::ContractOp,
                     riscv::OuterContractOp>(operation))
        return;
      auto implementation =
          operation->getAttrOfType<riscv::ImplementationAttr>("implementation");
      if (implementation && implementation.getFamily() == "widen-dot")
        wideningContractions.push_back(operation);
    });
    for (mlir::Operation *operation : wideningContractions) {
      auto lhs = mlir::dyn_cast<riscv::ValueType>(operation->getOperand(0).getType());
      auto rhs = mlir::dyn_cast<riscv::ValueType>(operation->getOperand(1).getType());
      if (!lhs || !rhs || lhs.getElementType() != rhs.getElementType() ||
          lhs.getShape() != rhs.getShape() || lhs.getAxisIds() != rhs.getAxisIds())
        continue;
      auto lhsLanes = riscv_internal::staticProduct(
          lhs.getLayout().getLaneFactors().asArrayRef());
      auto rhsLanes = riscv_internal::staticProduct(
          rhs.getLayout().getLaneFactors().asArrayRef());
      if (!lhsLanes || !rhsLanes || *lhsLanes <= 0 || *rhsLanes <= 0 ||
          *lhsLanes == *rhsLanes)
        continue;
      const unsigned sourceIndex = *lhsLanes < *rhsLanes ? 0 : 1;
      auto source = sourceIndex ? rhs : lhs;
      auto target = sourceIndex ? lhs : rhs;
      if (source.getLayout().getCarrier() != "rvv" ||
          target.getLayout().getCarrier() != "rvv" ||
          source.getLayout().getSew() != target.getLayout().getSew())
        continue;
      mlir::Value sourceValue = operation->getOperand(sourceIndex);
      mlir::Type requiredType =
          riscv_internal::withLayout(sourceValue.getType(), target.getLayout());
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(operation);
      auto conversion = builder.create<riscv::ConvertLayoutOp>(
          operation->getLoc(), requiredType, sourceValue,
          riscv_internal::layoutConversion(builder, source.getLayout(),
                                           target.getLayout()),
          riscv::AccessAttr(), riscv_internal::unselectedLeaf(builder));
      riscv_internal::copyOrigin(operation, conversion);
      operation->setOperand(sourceIndex, conversion.getResult());
    }

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
      if (input && input.getLayout().getCarrier() == "local" &&
          result.getLayout().getCarrier() == "rvv" &&
          input.getElementType() == result.getElementType()) {
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
      for (mlir::Attribute selectorAttribute : extract.getSelectors()) {
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
