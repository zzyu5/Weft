#include "Weft/Transforms/Passes.h"

#include "Weft/Dialect/Execution/IR/ExecutionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Layout/IR/LayoutAlgebra.h"
#include "Weft/Dialect/RVVExecution/IR/RVVExecutionDialect.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>

namespace weft::transforms {

#define GEN_PASS_DEF_SELECTRISCVEXECUTION
#include "Weft/Transforms/Passes.h.inc"

namespace {

constexpr llvm::StringLiteral kNodeAttrName("weft_execution.node");

struct ValueLifetime {
  mlir::Value value;
  int64_t sourceNode = -1;
  int64_t resultIndex = -1;
  int64_t defineAt = -1;
  int64_t lastUseAt = -1;
  int64_t sew = -1;
};

struct TransientRegisterUse {
  int64_t at = -1;
  int64_t sew = 0;
  int64_t scalableCount = 0;
  int64_t fixedCount = 0;
};

struct GroupPlan {
  int64_t id = -1;
  int64_t scopeNode = -1;
  layout::BlockedLayoutAttr layout;
  llvm::SmallVector<ValueLifetime, 16> values;
  // Values that are re-created only for one converted use edge.  They
  // participate in physical pressure selection, but are deliberately not
  // materialized as value_layout records: their native layout remains the
  // only persistent value ownership fact.
  llvm::SmallVector<ValueLifetime, 16> pressureOnlyValues;
  llvm::SmallVector<int64_t, 4> unaries;
  llvm::SmallVector<int64_t, 4> reductions;
  llvm::SmallVector<int64_t, 4> contracts;
  llvm::SmallVector<TransientRegisterUse, 8> transients;
  int64_t laneRatio = -1;
  int64_t registerBudget = -1;
};

struct MultiGroupContractPlan {
  int64_t sourceNode = -1;
  int64_t lhsGroup = -1;
  int64_t rhsGroup = -1;
  int64_t resultGroup = -1;
  bool ordered = false;
};

struct LayoutConversionPlan {
  int64_t consumerNode = -1;
  int64_t consumerOperand = -1;
  int64_t targetGroup = -1;
};

struct MetaBindingPlan {
  int64_t argument = -1;
  int64_t value = 0;
};

struct KernelPlan {
  kernel::KernelOp kernel;
  std::string symbol;
  std::string target;
  llvm::DenseMap<mlir::Operation *, int64_t> nodes;
  llvm::DenseMap<mlir::Value, int64_t> valueGroups;
  llvm::SmallVector<MetaBindingPlan, 4> metaBindings;
  llvm::SmallVector<GroupPlan, 4> groups;
  llvm::SmallVector<MultiGroupContractPlan, 2> multiGroupContracts;
  llvm::SmallVector<LayoutConversionPlan, 4> conversions;
};

std::optional<int64_t> getPhysicalSEW(mlir::Type type) {
  auto block = mlir::dyn_cast<kernel::BlockType>(type);
  if (!block)
    return std::nullopt;
  mlir::Type element = block.getElementType();
  if (mlir::isa<kernel::PtrType>(element) || element.isIndex())
    return 64;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element)) {
    if (integer.getWidth() == 1)
      return 0;
    if (integer.getWidth() == 8 || integer.getWidth() == 16 ||
        integer.getWidth() == 32 || integer.getWidth() == 64)
      return integer.getWidth();
    return std::nullopt;
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(element)) {
    if (floating.getWidth() == 16 || floating.getWidth() == 32 ||
        floating.getWidth() == 64)
      return floating.getWidth();
  }
  return std::nullopt;
}

bool hasLegalLMUL(int64_t sew, int64_t laneRatio) {
  if (sew == 0)
    return true;
  if ((sew * 8) % laneRatio != 0)
    return false;
  int64_t eighths = sew * 8 / laneRatio;
  return eighths == 4 || eighths == 8 || eighths == 16 ||
         eighths == 32 || eighths == 64;
}

int64_t registerFootprint(int64_t sew, int64_t laneRatio) {
  if (sew == 0)
    return 1;
  int64_t eighths = sew * 8 / laneRatio;
  return std::max<int64_t>(1, eighths / 8);
}

bool isBlocked(mlir::Value value) {
  return mlir::isa<kernel::BlockType>(value.getType());
}

bool variesAlongVectorAxis(mlir::Value value,
                           layout::BlockedLayoutAttr selectedLayout) {
  auto block = mlir::dyn_cast<kernel::BlockType>(value.getType());
  if (!block || selectedLayout.getVectorAxes().size() != 1 ||
      static_cast<int64_t>(block.getShape().size()) != selectedLayout.getRank())
    return false;
  return block.getShape()[selectedLayout.getVectorAxes().front()] != 1;
}

std::optional<int64_t> getIntegerConstant(mlir::Value value) {
  auto constant = value.getDefiningOp<kernel::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  return integer ? std::optional<int64_t>(integer.getInt()) : std::nullopt;
}

enum class LaneStrideKind { Invariant, Unit, Strided };

std::optional<LaneStrideKind> laneStrideKind(mlir::Value value,
                                             int64_t logicalAxis) {
  auto type = mlir::dyn_cast<kernel::BlockType>(value.getType());
  if (!type)
    return LaneStrideKind::Invariant;
  int64_t rank = type.getShape().size();
  if (logicalAxis < 0 || logicalAxis >= rank)
    return std::nullopt;
  if (type.getShape()[logicalAxis] == 1)
    return LaneStrideKind::Invariant;

  if (value.getDefiningOp<kernel::ArangeOp>())
    return rank == 1 && logicalAxis == 0
               ? std::optional<LaneStrideKind>(LaneStrideKind::Unit)
               : std::nullopt;
  if (auto expand = value.getDefiningOp<kernel::ExpandDimsOp>()) {
    int64_t insertedAxis = expand.getAxisAttr().getInt();
    if (logicalAxis == insertedAxis)
      return LaneStrideKind::Invariant;
    int64_t inputAxis = logicalAxis < insertedAxis ? logicalAxis
                                                   : logicalAxis - 1;
    return laneStrideKind(expand.getInput(), inputAxis);
  }
  if (auto cast = value.getDefiningOp<kernel::CastOp>())
    return laneStrideKind(cast.getInput(), logicalAxis);

  auto binary = value.getDefiningOp<kernel::BinaryOp>();
  if (!binary)
    return std::nullopt;
  auto lhs = laneStrideKind(binary.getLhs(), logicalAxis);
  auto rhs = laneStrideKind(binary.getRhs(), logicalAxis);
  if (!lhs || !rhs)
    return std::nullopt;
  if (binary.getKind() == "add") {
    if (*lhs == LaneStrideKind::Invariant)
      return *rhs;
    if (*rhs == LaneStrideKind::Invariant)
      return *lhs;
    return LaneStrideKind::Strided;
  }
  if (binary.getKind() == "sub") {
    if (*rhs == LaneStrideKind::Invariant)
      return *lhs;
    return std::nullopt;
  }
  if (binary.getKind() != "mul")
    return std::nullopt;
  if (*lhs == LaneStrideKind::Invariant &&
      *rhs == LaneStrideKind::Invariant)
    return LaneStrideKind::Invariant;
  if (*lhs != LaneStrideKind::Invariant &&
      *rhs == LaneStrideKind::Invariant) {
    auto multiplier = getIntegerConstant(binary.getRhs());
    if (multiplier && *multiplier < 0)
      return std::nullopt;
    if (multiplier && *multiplier == 0)
      return LaneStrideKind::Invariant;
    if (multiplier && *multiplier == 1)
      return *lhs;
    return LaneStrideKind::Strided;
  }
  if (*lhs == LaneStrideKind::Invariant &&
      *rhs != LaneStrideKind::Invariant) {
    auto multiplier = getIntegerConstant(binary.getLhs());
    if (multiplier && *multiplier < 0)
      return std::nullopt;
    if (multiplier && *multiplier == 0)
      return LaneStrideKind::Invariant;
    if (multiplier && *multiplier == 1)
      return *rhs;
    return LaneStrideKind::Strided;
  }
  return std::nullopt;
}

std::optional<LaneStrideKind>
pointerLaneStride(mlir::Value value, layout::BlockedLayoutAttr layout) {
  auto pointerAdd = value.getDefiningOp<kernel::PtrAddOp>();
  if (!pointerAdd || !isBlocked(pointerAdd.getResult()))
    return std::nullopt;
  if (layout.getVectorAxes().size() != 1)
    return std::nullopt;
  int64_t vectorAxis = layout.getVectorAxes().front();
  auto offsetStride = laneStrideKind(pointerAdd.getOffset(), vectorAxis);
  if (!offsetStride)
    return std::nullopt;
  LaneStrideKind baseStride = LaneStrideKind::Invariant;
  if (isBlocked(pointerAdd.getBase())) {
    auto nested = pointerLaneStride(pointerAdd.getBase(), layout);
    if (!nested)
      return std::nullopt;
    baseStride = *nested;
  }
  if (baseStride == LaneStrideKind::Invariant)
    return *offsetStride;
  if (*offsetStride == LaneStrideKind::Invariant)
    return baseStride;
  return LaneStrideKind::Strided;
}

bool isViewProvenanceForRank(mlir::Value value, int64_t targetRank,
                             llvm::DenseSet<mlir::Value> &visited) {
  auto type = mlir::dyn_cast<kernel::BlockType>(value.getType());
  if (!type || !kernel::isIndexCoordinateProvenance(value) ||
      !visited.insert(value).second)
    return false;
  if (static_cast<int64_t>(type.getShape().size()) == targetRank)
    return true;
  if (static_cast<int64_t>(type.getShape().size()) > targetRank ||
      value.use_empty())
    return false;
  bool reachedTarget = false;
  for (mlir::Operation *user : value.getUsers()) {
    mlir::Value next;
    if (auto expand = llvm::dyn_cast<kernel::ExpandDimsOp>(user)) {
      if (expand.getInput() != value)
        return false;
      next = expand.getResult();
    } else if (auto binary = llvm::dyn_cast<kernel::BinaryOp>(user)) {
      auto input = mlir::dyn_cast<kernel::BlockType>(value.getType());
      auto result = mlir::dyn_cast<kernel::BlockType>(binary.getResult().getType());
      if (!input || !result || !input.getElementType().isIndex() ||
          !result.getElementType().isIndex() ||
          input.getShape().size() != result.getShape().size() ||
          (binary.getLhs() != value && binary.getRhs() != value))
        return false;
      next = binary.getResult();
    } else {
      return false;
    }
    llvm::DenseSet<mlir::Value> branchVisited = visited;
    if (!isViewProvenanceForRank(next, targetRank, branchVisited))
      return false;
    reachedTarget = true;
  }
  return reachedTarget;
}

bool isViewProvenanceForRank(mlir::Value value, int64_t targetRank) {
  llvm::DenseSet<mlir::Value> visited;
  return isViewProvenanceForRank(value, targetRank, visited);
}

void collectExpandedAxes(mlir::Value value, int64_t logicalAxis,
                         int64_t targetRank,
                         const llvm::DenseSet<mlir::Value> *members,
                         llvm::DenseSet<mlir::Value> &visited,
                         llvm::DenseSet<int64_t> &axes) {
  if (!visited.insert(value).second)
    return;
  auto type = mlir::dyn_cast<kernel::BlockType>(value.getType());
  if (!type)
    return;
  if (static_cast<int64_t>(type.getShape().size()) > targetRank)
    return;
  if (static_cast<int64_t>(type.getShape().size()) == targetRank) {
    if (!members || members->contains(value))
      axes.insert(logicalAxis);
  }
  for (mlir::Operation *user : value.getUsers()) {
    if (auto expand = llvm::dyn_cast<kernel::ExpandDimsOp>(user)) {
      if (expand.getInput() != value)
        continue;
      int64_t insertedAxis = expand.getAxisAttr().getInt();
      int64_t expandedAxis =
          logicalAxis >= insertedAxis ? logicalAxis + 1 : logicalAxis;
      llvm::DenseSet<mlir::Value> branchVisited = visited;
      collectExpandedAxes(expand.getResult(), expandedAxis, targetRank, members,
                          branchVisited, axes);
      continue;
    }
    auto binary = llvm::dyn_cast<kernel::BinaryOp>(user);
    auto result = binary ? mlir::dyn_cast<kernel::BlockType>(
                               binary.getResult().getType())
                         : kernel::BlockType();
    if (!binary || !result || !type.getElementType().isIndex() ||
        !result.getElementType().isIndex() ||
        result.getShape().size() != type.getShape().size() ||
        (binary.getLhs() != value && binary.getRhs() != value) ||
        !kernel::isIndexCoordinateProvenance(binary.getResult()))
      continue;
    llvm::DenseSet<mlir::Value> branchVisited = visited;
    collectExpandedAxes(binary.getResult(), logicalAxis, targetRank, members,
                        branchVisited, axes);
  }
}

std::optional<int64_t> expandedArangeAxis(kernel::ArangeOp arange,
                                          int64_t targetRank) {
  llvm::DenseSet<mlir::Value> visited;
  llvm::DenseSet<int64_t> axes;
  collectExpandedAxes(arange.getResult(), 0, targetRank, nullptr, visited, axes);
  if (axes.size() != 1)
    return std::nullopt;
  return *axes.begin();
}

mlir::LogicalResult validateLogicalExtents(
    mlir::Operation *scope, int64_t selectedRank,
    llvm::ArrayRef<mlir::Value> groupValues) {
  for (int64_t axis = 0; axis < selectedRank; ++axis) {
    mlir::Value representative;
    for (mlir::Value value : groupValues) {
      auto block = mlir::cast<kernel::BlockType>(value.getType());
      if (block.getShape()[axis] == 1)
        continue;
      if (!kernel::deriveLogicalExtent(value, axis)) {
        value.getDefiningOp()->emitOpError()
            << "has no canonical logical extent for selected local axis "
            << axis;
        return mlir::failure();
      }
      if (representative &&
          !kernel::haveSameLogicalExtent(representative, axis, value, axis)) {
        value.getDefiningOp()->emitOpError()
            << "disagrees with another value in its selected layout group at "
               "local axis "
            << axis;
        return mlir::failure();
      }
      representative = value;
    }
    // A layout axis may be present only as a canonical singleton broadcast
    // dimension.  Its logical extent is then the type-level constant one; it
    // needs no arange-like dynamic representative.
    if (!representative)
      continue;
  }
  return mlir::success();
}

bool establishesLayoutEquivalence(mlir::Operation &operation) {
  if (llvm::isa<kernel::BinaryOp, kernel::CompareOp, kernel::CastOp,
                kernel::UnaryOp, kernel::PtrAddOp, kernel::LoadOp,
                kernel::StoreOp, kernel::SplatOp>(operation))
    return true;
  auto contract = llvm::dyn_cast<kernel::ContractOp>(operation);
  if (!contract)
    return false;
  auto lhs = mlir::dyn_cast<kernel::BlockType>(contract.getLhs().getType());
  auto rhs = mlir::dyn_cast<kernel::BlockType>(contract.getRhs().getType());
  return lhs && rhs && lhs.getShape().size() == 1 &&
         rhs.getShape().size() == 1 && contract.getLhsAxes().size() == 1 &&
         contract.getRhsAxes().size() == 1 &&
         contract.getLhsAxes().front() == 0 &&
         contract.getRhsAxes().front() == 0;
}

bool isSupportedRankTwoContract(kernel::ContractOp contract) {
  auto lhs = mlir::dyn_cast<kernel::BlockType>(contract.getLhs().getType());
  auto rhs = mlir::dyn_cast<kernel::BlockType>(contract.getRhs().getType());
  auto result = mlir::dyn_cast<kernel::BlockType>(contract.getResult().getType());
  auto init = mlir::dyn_cast<kernel::BlockType>(contract.getInit().getType());
  bool supportedInit = contract.getInit().getType().isF32() ||
                       (init && init.getElementType().isF32() && result &&
                        init.getShape() == result.getShape());
  return lhs && rhs && result && lhs.getShape().size() == 2 &&
         rhs.getShape().size() == 2 && result.getShape().size() == 2 &&
         lhs.getElementType().isF32() && rhs.getElementType().isF32() &&
         result.getElementType().isF32() && supportedInit &&
         contract.getLhsAxes().size() == 1 &&
         contract.getRhsAxes().size() == 1 &&
         contract.getLhsAxes().front() == 1 &&
         contract.getRhsAxes().front() == 0;
}

llvm::DenseSet<mlir::Value>
collectSerialProjectionValues(mlir::Block &block, int64_t selectedRank,
                              int64_t vectorAxis) {
  llvm::DenseSet<mlir::Value> serialExtents;
  for (mlir::Operation &operation : block) {
    auto arange = llvm::dyn_cast<kernel::ArangeOp>(operation);
    if (!arange)
      continue;
    auto axis = expandedArangeAxis(arange, selectedRank);
    if (axis && *axis != vectorAxis)
      serialExtents.insert(arange.getExtent());
  }

  llvm::DenseSet<mlir::Value> serialValues;
  for (mlir::Operation &operation : block) {
    if (auto reduction = llvm::dyn_cast<kernel::ReduceOp>(operation)) {
      auto input = mlir::dyn_cast<kernel::BlockType>(reduction.getInput().getType());
      auto result = mlir::dyn_cast<kernel::BlockType>(reduction.getResult().getType());
      if (input && result &&
          static_cast<int64_t>(input.getShape().size()) == selectedRank &&
          static_cast<int64_t>(result.getShape().size()) == selectedRank - 1 &&
          reduction.getAxisAttr().getInt() == vectorAxis)
        serialValues.insert(reduction.getResult());
      continue;
    }
    if (auto arange = llvm::dyn_cast<kernel::ArangeOp>(operation)) {
      auto result = mlir::cast<kernel::BlockType>(arange.getResult().getType());
      if (static_cast<int64_t>(result.getShape().size()) == selectedRank - 1 &&
          serialExtents.contains(arange.getExtent()))
        serialValues.insert(arange.getResult());
    }
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::Operation &operation : block) {
      if (!llvm::isa<kernel::BinaryOp, kernel::CompareOp, kernel::CastOp,
                     kernel::UnaryOp, kernel::PtrAddOp>(operation))
        continue;
      if (auto unary = llvm::dyn_cast<kernel::UnaryOp>(operation))
        if (unary.getKind() != "neg" && unary.getKind() != "rsqrt" &&
            unary.getKind() != "exp")
          continue;
      bool hasBlockOperand = false;
      bool allSerial = true;
      for (mlir::Value operand : operation.getOperands()) {
        if (!isBlocked(operand))
          continue;
        hasBlockOperand = true;
        allSerial &= serialValues.contains(operand);
      }
      if (!hasBlockOperand || !allSerial)
        continue;
      for (mlir::Value result : operation.getResults()) {
        auto blockType = mlir::dyn_cast<kernel::BlockType>(result.getType());
        if (!blockType || static_cast<int64_t>(blockType.getShape().size()) !=
                              selectedRank - 1)
          continue;
        changed |= serialValues.insert(result).second;
      }
    }
  }
  return serialValues;
}

mlir::LogicalResult
validateBlockedOperation(mlir::Operation &operation,
                         layout::BlockedLayoutAttr selectedLayout) {
  return llvm::TypeSwitch<mlir::Operation *, mlir::LogicalResult>(&operation)
      .Case<kernel::ArangeOp, kernel::PtrAddOp, kernel::ExpandDimsOp,
            kernel::SplatOp, kernel::ForOp, kernel::YieldOp>(
          [](auto) { return mlir::success(); })
      .Case<kernel::BinaryOp>([&](kernel::BinaryOp binary) {
        bool lhsVector = variesAlongVectorAxis(binary.getLhs(), selectedLayout);
        bool rhsVector = variesAlongVectorAxis(binary.getRhs(), selectedLayout);
        if (!lhsVector && !rhsVector)
          return mlir::success();

        llvm::StringRef kind = binary.getKind();
        bool commutative = kind == "add" || kind == "mul" || kind == "and" ||
                           kind == "or" || kind == "xor" || kind == "max" ||
                           kind == "min";
        if (!lhsVector && !commutative) {
          binary.emitOpError(
              "first RVV execution slice does not support scalar-left "
              "non-commutative vector binary operations");
          return mlir::failure();
        }

        auto result = mlir::cast<kernel::BlockType>(binary.getResult().getType());
        mlir::Type element = result.getElementType();
        bool supported = false;
        if (element.isF32())
          supported = kind == "add" || kind == "sub" || kind == "mul" ||
                      kind == "div" || kind == "max" || kind == "min";
        else if (element.isIndex())
          supported = kind == "add" || kind == "sub" || kind == "mul" ||
                      kind == "and" || kind == "or" || kind == "xor" ||
                      kind == "max" || kind == "min";
        else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element))
          supported = integer.getWidth() == 1
                          ? kind == "and" || kind == "or" || kind == "xor"
                          : kind == "add" || kind == "sub" || kind == "mul" ||
                                kind == "and" || kind == "or" ||
                                kind == "xor" || kind == "max" ||
                                kind == "min";
        if (supported)
          return mlir::success();
        binary.emitOpError(
            "has no supported RVV vector binary instruction in the first "
            "execution slice");
        return mlir::failure();
      })
      .Case<kernel::CompareOp>([&](kernel::CompareOp compare) {
        bool lhsVector = variesAlongVectorAxis(compare.getLhs(), selectedLayout);
        bool rhsVector = variesAlongVectorAxis(compare.getRhs(), selectedLayout);
        if (!lhsVector && !rhsVector)
          return mlir::success();
        mlir::Value vectorOperand = lhsVector ? compare.getLhs() : compare.getRhs();
        auto block = mlir::cast<kernel::BlockType>(vectorOperand.getType());
        mlir::Type element = block.getElementType();
        if (element.isF32() || element.isIndex())
          return mlir::success();
        if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element))
          if (integer.getWidth() != 1)
            return mlir::success();
        compare.emitOpError(
            "has no supported RVV vector comparison type in the first "
            "execution slice");
        return mlir::failure();
      })
      .Case<kernel::CastOp>([](kernel::CastOp cast) {
        cast.emitOpError(
            "blocked cast lowering is not yet supported by the RVV execution "
            "owner");
        return mlir::failure();
      })
      .Case<kernel::UnaryOp>([&](kernel::UnaryOp unary) {
        auto result = mlir::cast<kernel::BlockType>(unary.getResult().getType());
        bool supported = unary.getKind() == "neg" ||
                         (unary.getKind() == "exp" &&
                          variesAlongVectorAxis(unary.getResult(),
                                               selectedLayout));
        if (supported && result.getElementType().isF32())
          return mlir::success();
        unary.emitOpError(
            "first RVV execution slice supports blocked f32 negation and f32 "
            "exponential varying along the selected vector axis");
        return mlir::failure();
      })
      .Case<kernel::LoadOp>([&](kernel::LoadOp load) {
        auto result = mlir::cast<kernel::BlockType>(load.getResult().getType());
        if (!result.getElementType().isF32()) {
          load.emitOpError(
              "first RVV execution slice supports only f32 blocked loads");
          return mlir::failure();
        }
        if (selectedLayout.getVectorAxes().empty() ||
            !isBlocked(load.getPointer()) ||
            pointerLaneStride(load.getPointer(), selectedLayout))
          return mlir::success();
        load.emitOpError(
            "first RVV execution slice requires a structurally affine "
            "blocked pointer along the selected vector axis");
        return mlir::failure();
      })
      .Case<kernel::StoreOp>([&](kernel::StoreOp store) {
        auto value = mlir::dyn_cast<kernel::BlockType>(store.getValue().getType());
        if (!value || !value.getElementType().isF32()) {
          store.emitOpError(
              "first RVV execution slice supports only f32 blocked stores");
          return mlir::failure();
        }
        auto stride = pointerLaneStride(store.getPointer(), selectedLayout);
        if (!isBlocked(store.getPointer()) ||
            (stride && *stride == LaneStrideKind::Unit))
          return mlir::success();
        store.emitOpError(
            "first RVV execution slice requires a unit-stride blocked store "
            "pointer along the selected vector axis");
        return mlir::failure();
      })
      .Case<kernel::ReduceOp>([&](kernel::ReduceOp reduction) {
        auto input =
            mlir::cast<kernel::BlockType>(reduction.getInput().getType());
        bool reducesSelectedVectorAxis =
            selectedLayout.getVectorAxes().size() == 1 &&
            reduction.getAxisAttr().getInt() ==
                selectedLayout.getVectorAxes().front();
        bool supportedKind = reduction.getKind() == "sum" ||
                             reduction.getKind() == "max" ||
                             reduction.getKind() == "min";
        if (input.getElementType().isF32() && supportedKind &&
            reducesSelectedVectorAxis)
          return mlir::success();
        reduction.emitOpError(
            "first RVV execution slice supports f32 sum/max/min reduction "
            "only over the selected vector axis");
        return mlir::failure();
      })
      .Case<kernel::ContractOp>([&](kernel::ContractOp contract) {
        if (isSupportedRankTwoContract(contract))
          return mlir::success();
        auto lhs = mlir::dyn_cast<kernel::BlockType>(contract.getLhs().getType());
        auto rhs = mlir::dyn_cast<kernel::BlockType>(contract.getRhs().getType());
        bool fullRankOneDot =
            selectedLayout.getRank() == 1 && lhs && rhs &&
            lhs.getShape().size() == 1 && rhs.getShape().size() == 1 &&
            lhs.getElementType().isF32() && rhs.getElementType().isF32() &&
            contract.getInit().getType().isF32() &&
            contract.getResult().getType().isF32() &&
            contract.getLhsAxes().size() == 1 &&
            contract.getRhsAxes().size() == 1 &&
            contract.getLhsAxes().front() == 0 &&
            contract.getRhsAxes().front() == 0 &&
            selectedLayout.getVectorAxes().size() == 1 &&
            selectedLayout.getVectorAxes().front() == 0;
        if (fullRankOneDot)
          return mlir::success();
        contract.emitOpError(
            "first RVV contract slice supports only full rank-one f32 dot");
        return mlir::failure();
      })
      .Default([&](mlir::Operation *) {
        operation.emitOpError(
            "has blocked values but no structural RVV execution rule");
        return mlir::failure();
      });
}

mlir::LogicalResult validateRematerializedProducerClosure(
    mlir::Value value, mlir::Block &block,
    layout::BlockedLayoutAttr targetLayout,
    mlir::Operation *consumer, llvm::DenseSet<mlir::Operation *> &visited,
    llvm::DenseSet<mlir::Value> &blockedValues) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition) {
    if (isBlocked(value)) {
      consumer->emitOpError(
          "cannot rematerialize a blocked scope argument without an explicit "
          "owner-local conversion rule");
      return mlir::failure();
    }
    return mlir::success();
  }
  if (definition->getBlock() != &block) {
    definition->emitOpError(
        "rematerialized layout conversion cannot capture a blocked producer "
        "from another canonical scope");
    return mlir::failure();
  }
  if (isBlocked(value))
    blockedValues.insert(value);
  if (!visited.insert(definition).second)
    return mlir::success();
  for (mlir::Value operand : definition->getOperands())
    if (mlir::failed(validateRematerializedProducerClosure(
            operand, block, targetLayout, consumer, visited, blockedValues)))
      return mlir::failure();
  bool hasBlockedValue = llvm::any_of(definition->getOperands(), isBlocked) ||
                         llvm::any_of(definition->getResults(), isBlocked);
  return hasBlockedValue ? validateBlockedOperation(*definition, targetLayout)
                         : mlir::success();
}

bool containsCanonicalStore(mlir::Operation &operation) {
  bool found = false;
  operation.walk([&](kernel::StoreOp) { found = true; });
  return found;
}

bool hasInterveningStore(mlir::Block &block, mlir::Operation *producer,
                         mlir::Operation *consumer) {
  bool afterProducer = false;
  for (mlir::Operation &operation : block) {
    if (&operation == producer) {
      afterProducer = true;
      continue;
    }
    if (&operation == consumer)
      return false;
    if (afterProducer && containsCanonicalStore(operation))
      return true;
  }
  return true;
}

mlir::Operation *directAncestorInBlock(mlir::Operation *operation,
                                       mlir::Block &block) {
  while (operation && operation->getBlock() != &block)
    operation = operation->getParentOp();
  return operation;
}

std::string makePlanSymbol(llvm::StringRef kernelName,
                           llvm::StringRef target) {
  std::string symbol = (kernelName + "__" + target).str();
  for (char &character : symbol) {
    unsigned char byte = static_cast<unsigned char>(character);
    if (!std::isalnum(byte) && character != '_')
      character = '_';
  }
  return symbol;
}

class SelectRISCvExecutionPass final
    : public impl::SelectRISCvExecutionBase<SelectRISCvExecutionPass> {
public:
  using impl::SelectRISCvExecutionBase<
      SelectRISCvExecutionPass>::SelectRISCvExecutionBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    if (target.empty() || !llvm::StringRef(target).starts_with("rv64") ||
        !llvm::StringRef(target).contains('v')) {
      module.emitError(
          "weft-select-riscv-execution requires an RV64 target with the V "
          "extension");
      signalPassFailure();
      return;
    }
    if (registerBudget <= 0 || registerBudget > 32) {
      module.emitError("register-budget must be in [1, 32]");
      signalPassFailure();
      return;
    }
    if (blockElements < 0) {
      module.emitError("block-elements must be zero (unset) or positive");
      signalPassFailure();
      return;
    }
    if (blockElements > 0 && !metaBindings.empty()) {
      module.emitError(
          "block-elements and meta-bindings are mutually exclusive; choose "
          "either an explicit uniform value or per-parameter bindings");
      signalPassFailure();
      return;
    }

    llvm::StringMap<int64_t> namedMetaBindings;
    if (mlir::failed(
            parseNamedMetaBindings(module, namedMetaBindings))) {
      signalPassFailure();
      return;
    }

    llvm::SmallVector<KernelPlan, 4> plans;
    llvm::StringSet<> usedNamedMetaBindings;
    int64_t nextGroup = 0;
    for (kernel::KernelOp source : module.getOps<kernel::KernelOp>()) {
      auto plan = analyzeKernel(module, source, nextGroup, namedMetaBindings,
                                usedNamedMetaBindings);
      if (mlir::failed(plan)) {
        signalPassFailure();
        return;
      }
      plans.push_back(std::move(*plan));
    }
    if (plans.empty()) {
      module.emitError(
          "weft-select-riscv-execution requires at least one canonical "
          "weft_kernel.kernel");
      signalPassFailure();
      return;
    }
    for (const auto &binding : namedMetaBindings) {
      if (usedNamedMetaBindings.contains(binding.getKey()))
        continue;
      module.emitError() << "meta binding '" << binding.getKey()
                         << "' matches no canonical constexpr parameter";
      signalPassFailure();
      return;
    }

    for (KernelPlan &plan : plans)
      materializePlan(module, plan);
  }

private:
  mlir::LogicalResult
  parseNamedMetaBindings(mlir::ModuleOp module,
                         llvm::StringMap<int64_t> &bindings) {
    for (const std::string &spelling : metaBindings) {
      llvm::StringRef specification(spelling);
      size_t separator = specification.find('=');
      if (separator == llvm::StringRef::npos || separator == 0 ||
          separator + 1 == specification.size() ||
          specification.find('=', separator + 1) != llvm::StringRef::npos) {
        module.emitError()
            << "meta binding '" << specification
            << "' must have exactly the form NAME=POSITIVE_INTEGER";
        return mlir::failure();
      }
      llvm::StringRef name = specification.take_front(separator);
      llvm::StringRef valueSpelling = specification.drop_front(separator + 1);
      int64_t value = 0;
      if (valueSpelling.getAsInteger(10, value) || value <= 0) {
        module.emitError()
            << "meta binding '" << specification
            << "' requires a positive base-10 integer value";
        return mlir::failure();
      }
      if (!bindings.try_emplace(name, value).second) {
        module.emitError() << "duplicates meta binding for canonical parameter '"
                           << name << "'";
        return mlir::failure();
      }
    }
    return mlir::success();
  }

  mlir::FailureOr<KernelPlan> analyzeKernel(mlir::ModuleOp module,
                                            kernel::KernelOp source,
                                            int64_t &nextGroup,
                                            const llvm::StringMap<int64_t>
                                                &namedMetaBindings,
                                            llvm::StringSet<>
                                                &usedNamedMetaBindings) {
    for (execution::PlanOp existing : module.getOps<execution::PlanOp>())
      if (existing.getKernelAttr().getValue() == source.getSymName()) {
        source.emitOpError("already has a selected execution plan");
        return mlir::failure();
      }

    bool hasStaleNode = false;
    source->walk([&](mlir::Operation *operation) {
      hasStaleNode |= operation->hasAttr(kNodeAttrName);
    });
    if (hasStaleNode) {
      source.emitOpError(
          "carries weft_execution.node indices without a selected plan");
      return mlir::failure();
    }

    KernelPlan plan;
    plan.kernel = source;
    plan.target = target;
    plan.symbol = makePlanSymbol(source.getSymName(), target);
    if (mlir::SymbolTable::lookupSymbolIn(module, plan.symbol)) {
      source.emitOpError() << "selected plan symbol @" << plan.symbol
                           << " already exists";
      return mlir::failure();
    }

    int64_t nextNode = 0;
    source->walk([&](mlir::Operation *operation) {
      plan.nodes.try_emplace(operation, nextNode++);
    });

    for (auto [index, argument] :
         llvm::enumerate(source.getBody().front().getArguments())) {
      auto meta = mlir::dyn_cast<kernel::ConstexprType>(argument.getType());
      if (!meta)
        continue;
      if (!meta.getValueType().isIndex()) {
        source.emitOpError(
            "first execution slice supports only index constexpr bindings");
        return mlir::failure();
      }
      bool hasMaterializedUse = false;
      for (mlir::Operation *user : argument.getUsers()) {
        auto materialized = llvm::dyn_cast<kernel::MetaValueOp>(user);
        if (!materialized) {
          user->emitOpError("constexpr arguments must first pass through meta_value");
          return mlir::failure();
        }
        hasMaterializedUse |= !materialized.getResult().use_empty();
      }
      if (!hasMaterializedUse) {
        source.emitOpError("constexpr argument has no canonical scalar use");
        return mlir::failure();
      }

      auto name = mlir::cast<mlir::StringAttr>(source.getArgNames()[index])
                      .getValue();
      int64_t selectedValue = blockElements;
      if (selectedValue == 0) {
        auto selected = namedMetaBindings.find(name);
        if (selected == namedMetaBindings.end()) {
          source.emitOpError()
              << "has no selected value for canonical constexpr parameter '"
              << name
              << "'; pass a per-parameter meta binding or an explicit "
                 "uniform block-elements value";
          return mlir::failure();
        }
        selectedValue = selected->second;
        usedNamedMetaBindings.insert(name);
      }
      plan.metaBindings.push_back(
          MetaBindingPlan{static_cast<int64_t>(index), selectedValue});
    }

    if (mlir::failed(analyzeBlock(source.getBody().front(), source, plan,
                                  nextGroup)))
      return mlir::failure();
    if (mlir::failed(finalizeLoopCarriedStates(plan)))
      return mlir::failure();
    return plan;
  }

  mlir::LogicalResult analyzeBlock(mlir::Block &block,
                                   mlir::Operation *scope,
                                   KernelPlan &plan, int64_t &nextGroup) {
    llvm::DenseMap<mlir::Operation *, int64_t> operationIndex;
    int64_t nextOperation = 0;
    for (mlir::Operation &operation : block)
      operationIndex.try_emplace(&operation, nextOperation++);

    llvm::MapVector<int64_t, llvm::SmallVector<mlir::Value, 16>> valuesByRank;
    llvm::DenseSet<mlir::Value> seenValues;
    for (mlir::Operation &operation : block) {
      for (mlir::Value result : operation.getResults()) {
        auto blockType = mlir::dyn_cast<kernel::BlockType>(result.getType());
        if (!blockType || !seenValues.insert(result).second)
          continue;
        int64_t rank = blockType.getShape().size();
        valuesByRank[rank].push_back(result);
      }
    }

    int64_t selectedRank = 0;
    for (const auto &[rank, values] : valuesByRank) {
      (void)values;
      selectedRank = std::max(selectedRank, rank);
    }
    llvm::SmallVector<llvm::SmallVector<mlir::Value, 16>, 4>
        valueLayoutComponents;
    if (selectedRank > 0) {
      auto selectedValues = valuesByRank.lookup(selectedRank);
      llvm::DenseMap<mlir::Value, unsigned> valueIndices;
      llvm::SmallVector<unsigned, 16> parents;
      for (auto [index, value] : llvm::enumerate(selectedValues)) {
        valueIndices.try_emplace(value, index);
        parents.push_back(index);
      }
      auto findRoot = [&](unsigned index) {
        while (parents[index] != index) {
          parents[index] = parents[parents[index]];
          index = parents[index];
        }
        return index;
      };
      auto unite = [&](unsigned lhs, unsigned rhs) {
        lhs = findRoot(lhs);
        rhs = findRoot(rhs);
        if (lhs != rhs)
          parents[rhs] = lhs;
      };
      for (mlir::Operation &operation : block) {
        if (auto loop = llvm::dyn_cast<kernel::ForOp>(operation)) {
          for (auto [init, result] :
               llvm::zip(loop.getInitArgs(), loop.getResults())) {
            auto initFound = valueIndices.find(init);
            auto resultFound = valueIndices.find(result);
            if (initFound != valueIndices.end() &&
                resultFound != valueIndices.end())
              unite(initFound->second, resultFound->second);
          }
          continue;
        }
        if (!establishesLayoutEquivalence(operation))
          continue;
        std::optional<unsigned> first;
        auto connect = [&](mlir::Value value) {
          auto found = valueIndices.find(value);
          if (found == valueIndices.end())
            return;
          if (first)
            unite(*first, found->second);
          else
            first = found->second;
        };
        for (mlir::Value operand : operation.getOperands())
          connect(operand);
        for (mlir::Value result : operation.getResults())
          connect(result);
      }
      llvm::MapVector<unsigned, llvm::SmallVector<mlir::Value, 16>> components;
      for (auto [index, value] : llvm::enumerate(selectedValues))
        components[findRoot(index)].push_back(value);
      for (auto &entry : components)
        valueLayoutComponents.push_back(std::move(entry.second));
    }
    llvm::DenseSet<mlir::Value> serialProjectionValues;
    if (selectedRank > 2) {
      for (mlir::Operation &operation : block) {
        if (!llvm::isa<kernel::ReduceOp, kernel::ContractOp>(operation))
          continue;
        operation.emitOpError(
            "rank-N RVV layout currently supports pointwise/memory structure "
            "only when N is greater than two; higher-rank reduction and "
            "contraction mappings are not yet selected");
        return mlir::failure();
      }
    }
    if (selectedRank == 2) {
      llvm::SmallDenseSet<int64_t, 2> reductionAxes;
      for (mlir::Operation &operation : block) {
        auto reduction = llvm::dyn_cast<kernel::ReduceOp>(operation);
        if (!reduction)
          continue;
        auto input =
            mlir::dyn_cast<kernel::BlockType>(reduction.getInput().getType());
        auto result =
            mlir::dyn_cast<kernel::BlockType>(reduction.getResult().getType());
        if (input && result &&
            static_cast<int64_t>(input.getShape().size()) == selectedRank &&
            static_cast<int64_t>(result.getShape().size()) == selectedRank - 1)
          reductionAxes.insert(reduction.getAxisAttr().getInt());
      }
      for (int64_t axis : reductionAxes) {
        llvm::DenseSet<mlir::Value> projected =
            collectSerialProjectionValues(block, selectedRank, axis);
        serialProjectionValues.insert(projected.begin(), projected.end());
      }
      for (mlir::Operation &operation : block) {
        auto store = llvm::dyn_cast<kernel::StoreOp>(operation);
        if (!store)
          continue;
        auto stored = mlir::dyn_cast<kernel::BlockType>(store.getValue().getType());
        if (!stored || static_cast<int64_t>(stored.getShape().size()) != 1)
          continue;
        bool serialStore = true;
        for (mlir::Value operand : store->getOperands())
          if (isBlocked(operand))
            serialStore &= serialProjectionValues.contains(operand);
        if (!serialStore || !stored.getElementType().isF32()) {
          store.emitOpError(
              "rank-2 reduction tail requires a fully derived rank-1 f32 "
              "serial pointer/value/mask store");
          return mlir::failure();
        }
      }
    }
    if (selectedRank > 1) {
      for (const auto &[rank, values] : valuesByRank) {
        if (rank == selectedRank)
          continue;
        for (mlir::Value value : values) {
          if (isViewProvenanceForRank(value, selectedRank) ||
              serialProjectionValues.contains(value))
            continue;
          value.getDefiningOp()->emitOpError(
              "mixed-rank physical block values require an explicit selected "
              "layout conversion; only canonical index-coordinate provenance "
              "reaching the selected rank through pointwise offset/expand_dims "
              "may remain unmaterialized");
          return mlir::failure();
        }
      }
    }

    llvm::DenseMap<mlir::Value, int64_t> valueGroup;
    llvm::DenseMap<mlir::Value, int64_t> serialValueGroups;
    for (llvm::SmallVector<mlir::Value, 16> &values :
         valueLayoutComponents) {
      if (mlir::failed(validateLogicalExtents(scope, selectedRank, values)))
        return mlir::failure();

      llvm::DenseSet<mlir::Value> groupMembers(values.begin(), values.end());
      bool serialContractLhs = false;
      bool vectorContractRole = false;
      llvm::SmallDenseSet<int64_t, 2> groupReductionAxes;
      for (mlir::Operation &operation : block)
        if (auto contract = llvm::dyn_cast<kernel::ContractOp>(operation)) {
          serialContractLhs |= isSupportedRankTwoContract(contract) &&
                               groupMembers.contains(contract.getLhs());
          vectorContractRole |=
              isSupportedRankTwoContract(contract) &&
              (groupMembers.contains(contract.getRhs()) ||
               groupMembers.contains(contract.getResult()));
        } else if (auto reduction =
                       llvm::dyn_cast<kernel::ReduceOp>(operation)) {
          if (groupMembers.contains(reduction.getInput()))
            groupReductionAxes.insert(reduction.getAxisAttr().getInt());
        }
      llvm::SmallVector<int64_t, 1> vectorAxes;
      if (!serialContractLhs) {
        int64_t selectedVectorAxis = -1;
        if (vectorContractRole) {
          selectedVectorAxis = 1;
        } else if (groupReductionAxes.size() == 1) {
          selectedVectorAxis = *groupReductionAxes.begin();
        } else {
          int64_t bestScore = std::numeric_limits<int64_t>::min();
          // Traverse high-to-low so an exact score tie preserves conventional
          // row-major vectorization without making it an entry classification.
          for (int64_t candidateAxis = selectedRank; candidateAxis > 0;
               --candidateAxis) {
            int64_t axis = candidateAxis - 1;
            bool hasVaryingMember = llvm::any_of(values, [&](mlir::Value value) {
              auto type = mlir::cast<kernel::BlockType>(value.getType());
              return type.getShape()[axis] != 1;
            });
            if (!hasVaryingMember)
              continue;
            auto candidateLayout = layout::getVectorFastestBlockedLayout(
                &getContext(), selectedRank, llvm::ArrayRef<int64_t>(axis));
            int64_t score = groupReductionAxes.contains(axis) ? 100 : 0;
            bool legal = true;
            for (mlir::Operation &operation : block) {
              auto isMember = [&](mlir::Value value) {
                return groupMembers.contains(value);
              };
              if (!llvm::any_of(operation.getOperands(), isMember) &&
                  !llvm::any_of(operation.getResults(), isMember))
                continue;
              if (auto store = llvm::dyn_cast<kernel::StoreOp>(operation)) {
                if (!isBlocked(store.getPointer()))
                  continue;
                auto stride =
                    pointerLaneStride(store.getPointer(), candidateLayout);
                legal &= stride && *stride == LaneStrideKind::Unit;
                score += legal ? 40 : 0;
                continue;
              }
              if (auto load = llvm::dyn_cast<kernel::LoadOp>(operation)) {
                if (!isBlocked(load.getPointer()))
                  continue;
                auto stride = pointerLaneStride(load.getPointer(), candidateLayout);
                if (!stride) {
                  legal = false;
                  continue;
                }
                score += *stride == LaneStrideKind::Unit ? 20 : 4;
                continue;
              }
              if (auto unary = llvm::dyn_cast<kernel::UnaryOp>(operation))
                if (unary.getKind() == "exp" &&
                    !variesAlongVectorAxis(unary.getResult(), candidateLayout))
                  legal = false;
            }
            if (legal && score > bestScore) {
              bestScore = score;
              selectedVectorAxis = axis;
            }
          }
        }
        if (selectedVectorAxis < 0) {
          scope->emitError(
              "has no logical axis satisfying its RVV memory/reduction layout "
              "constraints");
          return mlir::failure();
        }
        vectorAxes.push_back(selectedVectorAxis);
      }
      layout::BlockedLayoutAttr selected =
          layout::getVectorFastestBlockedLayout(&getContext(), selectedRank,
                                                vectorAxes);
      for (mlir::Value value : values) {
        auto valueType = mlir::cast<kernel::BlockType>(value.getType());
        auto valueLayout =
            layout::getVectorFastestBlockedLayout(
                &getContext(), valueType.getShape().size(), vectorAxes);
        auto unified = layout::unifyBlockedLayouts(selected, valueLayout);
        if (mlir::failed(unified)) {
          scope->emitError("cannot unify blocked SSA layouts in one scope");
          return mlir::failure();
        }
        selected = *unified;
      }

      llvm::SmallVector<kernel::ReduceOp, 2> convertedReductions;
      for (mlir::Operation &operation : block) {
        auto isMember = [&](mlir::Value value) {
          return groupMembers.contains(value);
        };
        if (!llvm::any_of(operation.getOperands(), isMember) &&
            !llvm::any_of(operation.getResults(), isMember))
          continue;
        if (auto reduction = llvm::dyn_cast<kernel::ReduceOp>(operation)) {
          if (groupMembers.contains(reduction.getInput()) &&
              selected.getVectorAxes().size() == 1 &&
              reduction.getAxisAttr().getInt() !=
                  selected.getVectorAxes().front()) {
            auto splat =
                reduction.getInput().getDefiningOp<kernel::SplatOp>();
            auto load = reduction.getInput().getDefiningOp<kernel::LoadOp>();
            if (!splat && !load) {
              reduction.emitOpError(
                  "a non-native vector-axis reduction currently requires a "
                  "canonical splat or masked load that the RVV owner can "
                  "rematerialize for this use");
              return mlir::failure();
            }
            if (load) {
              if (hasInterveningStore(block, load.getOperation(),
                                      reduction.getOperation())) {
                reduction.emitOpError(
                    "cannot rematerialize its canonical load across an "
                    "intervening store without alias proof");
                return mlir::failure();
              }
            }
            llvm::DenseSet<mlir::Operation *> visited;
            llvm::DenseSet<mlir::Value> blockedValues;
            auto targetLayout = layout::getVectorFastestBlockedLayout(
                &getContext(), selectedRank,
                llvm::ArrayRef<int64_t>(reduction.getAxisAttr().getInt()));
            if (mlir::failed(validateRematerializedProducerClosure(
                    reduction.getInput(), block, targetLayout,
                    reduction.getOperation(), visited, blockedValues)))
              return mlir::failure();
            convertedReductions.push_back(reduction);
            continue;
          }
        }
        if (mlir::failed(validateBlockedOperation(operation, selected)))
          return mlir::failure();
      }

      GroupPlan group;
      group.id = nextGroup++;
      group.scopeNode = plan.nodes.lookup(scope);
      group.layout = selected;
      group.registerBudget = registerBudget;

      for (mlir::Value value : values) {
        mlir::Operation *definition = value.getDefiningOp();
        if (!definition || definition->getBlock() != &block) {
          scope->emitError(
              "blocked values must be defined directly in their selected "
              "layout scope");
          return mlir::failure();
        }
        auto sew = getPhysicalSEW(value.getType());
        if (!sew) {
          definition->emitOpError(
              "has a block element type unsupported by the RVV owner");
          return mlir::failure();
        }
        int64_t resultIndex =
            mlir::cast<mlir::OpResult>(value).getResultNumber();
        ValueLifetime lifetime{value,
                               plan.nodes.lookup(definition),
                               resultIndex,
                               operationIndex.lookup(definition),
                               operationIndex.lookup(definition),
                               *sew};
        for (mlir::OpOperand &use : value.getUses()) {
          mlir::Operation *ancestor =
              directAncestorInBlock(use.getOwner(), block);
          if (!ancestor) {
            definition->emitOpError(
                "blocked value escapes its selected layout scope");
            return mlir::failure();
          }
          lifetime.lastUseAt =
              std::max(lifetime.lastUseAt, operationIndex.lookup(ancestor));
        }
        valueGroup.try_emplace(value, group.id);
        plan.valueGroups.try_emplace(value, group.id);
        group.values.push_back(lifetime);
      }

      for (mlir::Operation &operation : block) {
        if (auto unary = llvm::dyn_cast<kernel::UnaryOp>(operation)) {
          auto selectedGroup = valueGroup.find(unary.getResult());
          if (selectedGroup != valueGroup.end() &&
              selectedGroup->second == group.id && unary.getKind() == "exp") {
            group.unaries.push_back(plan.nodes.lookup(&operation));
            // exp_poly_v1 keeps the canonical input/result in group.values and
            // needs an additional owner-local peak working set.  Eight
            // scalable groups conservatively cover the finite-input
            // sanitization and polynomial/range stages; two fixed mask groups
            // cover the largest simultaneously-live predicate set.  This is a
            // physical pressure summary, not a lifetime for one hidden SSA
            // value and not a second copy of the algorithm.
            group.transients.push_back(
                TransientRegisterUse{operationIndex.lookup(&operation), 32, 8,
                                     2});
          }
          continue;
        }
        if (auto load = llvm::dyn_cast<kernel::LoadOp>(operation)) {
          auto selectedGroup = valueGroup.find(load.getResult());
          if (!selected.getVectorAxes().empty() &&
              selectedGroup != valueGroup.end() &&
              selectedGroup->second == group.id) {
            // The tumu load needs a passthrough vector in addition to the
            // persistent load result tracked in group.values.  A scalar mask
            // is broadcast to one physical predicate at this use; blocked
            // masks are already represented by their group value lifetime.
            group.transients.push_back(
                TransientRegisterUse{operationIndex.lookup(&operation), 32, 1,
                                     isBlocked(load.getMask()) ? 0 : 1});
          }
          continue;
        }
        if (auto store = llvm::dyn_cast<kernel::StoreOp>(operation)) {
          auto pointerGroup = valueGroup.find(store.getPointer());
          auto storedValueGroup = valueGroup.find(store.getValue());
          bool belongsToGroup =
              (pointerGroup != valueGroup.end() &&
               pointerGroup->second == group.id) ||
              (storedValueGroup != valueGroup.end() &&
               storedValueGroup->second == group.id);
          if (!selected.getVectorAxes().empty() && belongsToGroup &&
              !isBlocked(store.getMask()))
            group.transients.push_back(TransientRegisterUse{
                operationIndex.lookup(&operation), 0, 0, 1});
          continue;
        }
        if (auto reduction = llvm::dyn_cast<kernel::ReduceOp>(operation)) {
          auto selectedGroup = valueGroup.find(reduction.getInput());
          if (selectedGroup != valueGroup.end() &&
              selectedGroup->second == group.id &&
              selected.getVectorAxes().size() == 1 &&
              reduction.getAxisAttr().getInt() ==
                  selected.getVectorAxes().front()) {
            group.reductions.push_back(plan.nodes.lookup(&operation));
            serialValueGroups.try_emplace(reduction.getResult(), group.id);
            group.transients.push_back(
                TransientRegisterUse{operationIndex.lookup(&operation), 0, 0,
                                     2});
          }
          continue;
        }
        if (auto contract = llvm::dyn_cast<kernel::ContractOp>(operation)) {
          auto lhsGroup = valueGroup.find(contract.getLhs());
          auto rhsGroup = valueGroup.find(contract.getRhs());
          if (lhsGroup != valueGroup.end() && rhsGroup != valueGroup.end() &&
              lhsGroup->second == group.id && rhsGroup->second == group.id) {
            group.contracts.push_back(plan.nodes.lookup(&operation));
            group.transients.push_back(
                TransientRegisterUse{operationIndex.lookup(&operation), 32, 1,
                                     2});
          }
        }
      }

      if (mlir::failed(selectLaneRatio(group, scope)))
        return mlir::failure();
      plan.groups.push_back(std::move(group));

      llvm::SmallVector<GroupPlan, 2> conversionGroups;
      llvm::DenseMap<int64_t, unsigned> conversionGroupByAxis;
      for (kernel::ReduceOp reduction : convertedReductions) {
        int64_t axis = reduction.getAxisAttr().getInt();
        auto found = conversionGroupByAxis.find(axis);
        if (found == conversionGroupByAxis.end()) {
          GroupPlan converted;
          converted.id = nextGroup++;
          converted.scopeNode = plan.nodes.lookup(scope);
          converted.layout = layout::getVectorFastestBlockedLayout(
              &getContext(), selectedRank, llvm::ArrayRef<int64_t>(axis));
          converted.registerBudget = registerBudget;
          conversionGroups.push_back(std::move(converted));
          found = conversionGroupByAxis
                      .try_emplace(axis, conversionGroups.size() - 1)
                      .first;
        }
        GroupPlan &converted = conversionGroups[found->second];
        if (mlir::failed(
                validateBlockedOperation(*reduction, converted.layout)))
          return mlir::failure();
        int64_t point = operationIndex.lookup(reduction.getOperation());
        converted.reductions.push_back(plan.nodes.lookup(reduction.getOperation()));
        serialValueGroups.try_emplace(reduction.getResult(), converted.id);
        converted.transients.push_back(
            TransientRegisterUse{point, 0, 0, 2});

        llvm::DenseSet<mlir::Operation *> closureOperations;
        llvm::DenseSet<mlir::Value> closureValues;
        if (mlir::failed(validateRematerializedProducerClosure(
                reduction.getInput(), block, converted.layout,
                reduction.getOperation(), closureOperations, closureValues)))
          return mlir::failure();
        for (mlir::Value value : closureValues) {
          mlir::Operation *definition = value.getDefiningOp();
          auto sew = getPhysicalSEW(value.getType());
          if (!definition || !sew) {
            reduction.emitOpError(
                "cannot account rematerialized producer register pressure");
            return mlir::failure();
          }
          int64_t definitionPoint = operationIndex.lookup(definition);
          ValueLifetime lifetime{value,
                                 plan.nodes.lookup(definition),
                                 mlir::cast<mlir::OpResult>(value)
                                     .getResultNumber(),
                                 definitionPoint,
                                 definitionPoint,
                                 *sew};
          for (mlir::OpOperand &use : value.getUses()) {
            mlir::Operation *ancestor =
                directAncestorInBlock(use.getOwner(), block);
            if (ancestor != reduction.getOperation() &&
                !closureOperations.contains(ancestor))
              continue;
            lifetime.lastUseAt =
                std::max(lifetime.lastUseAt, operationIndex.lookup(ancestor));
          }
          auto existing = llvm::find_if(
              converted.pressureOnlyValues,
              [&](const ValueLifetime &candidate) {
                return candidate.value == value;
              });
          if (existing == converted.pressureOnlyValues.end())
            converted.pressureOnlyValues.push_back(lifetime);
          else
            existing->lastUseAt =
                std::max(existing->lastUseAt, lifetime.lastUseAt);
        }
        if (auto load = reduction.getInput().getDefiningOp<kernel::LoadOp>()) {
          // Mirror native masked-load pressure: the result is in
          // pressureOnlyValues, while this is its tumu passthrough vector.
          converted.transients.push_back(
              TransientRegisterUse{operationIndex.lookup(load.getOperation()),
                                   32, 1,
                                   isBlocked(load.getMask()) ? 0 : 1});
        }
        plan.conversions.push_back(LayoutConversionPlan{
            plan.nodes.lookup(reduction.getOperation()), 0, converted.id});
      }
      for (GroupPlan &converted : conversionGroups) {
        if (mlir::failed(selectLaneRatio(converted, scope)))
          return mlir::failure();
        plan.groups.push_back(std::move(converted));
      }
    }

    bool propagatedSerialGroups = true;
    while (propagatedSerialGroups) {
      propagatedSerialGroups = false;
      for (mlir::Operation &operation : block) {
        llvm::SmallDenseSet<int64_t, 2> operandGroups;
        for (mlir::Value operand : operation.getOperands()) {
          auto type = mlir::dyn_cast<kernel::BlockType>(operand.getType());
          if (!type || type.getShape().size() != 1)
            continue;
          auto selected = serialValueGroups.find(operand);
          if (selected != serialValueGroups.end())
            operandGroups.insert(selected->second);
        }
        if (operandGroups.size() > 1) {
          operation.emitOpError(
              "combines serial projections selected in different physical "
              "layout groups without an explicit use conversion");
          return mlir::failure();
        }
        if (operandGroups.empty())
          continue;
        int64_t selectedGroup = *operandGroups.begin();
        // A serial projection may be constructed independently from the
        // reduction value that eventually shares its Store (for example the
        // output index, pointer, and mask).  Layout-equivalent operations are
        // bidirectional constraints, so propagate the one known group back to
        // still-unassigned serial operands as well as forward to results.
        for (mlir::Value operand : operation.getOperands()) {
          auto type = mlir::dyn_cast<kernel::BlockType>(operand.getType());
          if (!type || type.getShape().size() != 1 ||
              !serialProjectionValues.contains(operand))
            continue;
          auto [found, inserted] =
              serialValueGroups.try_emplace(operand, selectedGroup);
          if (!inserted && found->second != selectedGroup) {
            operation.emitOpError(
                "consumes a serial projection with conflicting physical "
                "layout groups");
            return mlir::failure();
          }
          propagatedSerialGroups |= inserted;
        }
        for (mlir::Value result : operation.getResults()) {
          auto type = mlir::dyn_cast<kernel::BlockType>(result.getType());
          if (!type || type.getShape().size() != 1)
            continue;
          auto [found, inserted] =
              serialValueGroups.try_emplace(result, selectedGroup);
          if (!inserted && found->second != selectedGroup) {
            operation.emitOpError(
                "produces a serial projection with conflicting physical "
                "layout groups");
            return mlir::failure();
          }
          propagatedSerialGroups |= inserted;
        }
      }
    }

    for (mlir::Operation &operation : block) {
      auto store = llvm::dyn_cast<kernel::StoreOp>(operation);
      if (!store)
        continue;
      llvm::SmallDenseSet<int64_t, 2> operandGroups;
      bool hasBlockedOperand = false;
      for (mlir::Value operand : store->getOperands()) {
        if (!isBlocked(operand))
          continue;
        hasBlockedOperand = true;
        auto native = valueGroup.find(operand);
        if (native != valueGroup.end()) {
          operandGroups.insert(native->second);
          continue;
        }
        auto serial = serialValueGroups.find(operand);
        if (serial == serialValueGroups.end()) {
          store.emitOpError(
              "has a blocked operand without a selected physical layout "
              "group");
          return mlir::failure();
        }
        operandGroups.insert(serial->second);
      }
      if (!hasBlockedOperand)
        continue;
      if (operandGroups.size() != 1) {
        store.emitOpError(
            "requires its blocked pointer, value, and mask to use exactly one "
            "selected physical layout group");
        return mlir::failure();
      }
    }

    llvm::SmallDenseSet<int64_t, 4> contractGroups;
    int64_t rankTwoContractCount = 0;
    for (mlir::Operation &operation : block) {
      auto contract = llvm::dyn_cast<kernel::ContractOp>(operation);
      if (!contract || !isSupportedRankTwoContract(contract))
        continue;
      auto lhs = valueGroup.find(contract.getLhs());
      auto rhs = valueGroup.find(contract.getRhs());
      auto result = valueGroup.find(contract.getResult());
      if (lhs == valueGroup.end() || rhs == valueGroup.end() ||
          result == valueGroup.end()) {
        contract.emitOpError(
            "rank-two RVV contraction requires selected lhs, rhs, and result "
            "value-layout groups");
        return mlir::failure();
      }
      if (lhs->second == rhs->second || lhs->second == result->second ||
          rhs->second == result->second) {
        contract.emitOpError(
            "first rank-two RVV contraction requires distinct lhs, rhs, and "
            "result value-layout groups");
        return mlir::failure();
      }
      GroupPlan *rhsPlan = nullptr;
      GroupPlan *resultPlan = nullptr;
      for (GroupPlan &group : plan.groups) {
        if (group.id == rhs->second)
          rhsPlan = &group;
        if (group.id == result->second)
          resultPlan = &group;
      }
      if (!rhsPlan || !resultPlan ||
          mlir::failed(selectCommonLaneRatio(
              *rhsPlan, *resultPlan, contract,
              plan.nodes.lookup(contract.getOperation()))))
        return mlir::failure();
      plan.multiGroupContracts.push_back(MultiGroupContractPlan{
          plan.nodes.lookup(&operation), lhs->second, rhs->second,
          result->second, contract.getOrdered()});
      contractGroups.insert(lhs->second);
      contractGroups.insert(rhs->second);
      contractGroups.insert(result->second);
      ++rankTwoContractCount;
    }

    if (rankTwoContractCount > 0 &&
        (valueLayoutComponents.size() != 3 || rankTwoContractCount != 1 ||
         contractGroups.size() != 3)) {
      scope->emitError(
          "a rank-two RVV contraction currently requires exactly its three "
          "site-local lhs, rhs, and result groups in the source scope; "
          "independent non-contract groups are otherwise allowed");
      return mlir::failure();
    }

    for (mlir::Operation &operation : block)
      if (auto loop = llvm::dyn_cast<kernel::ForOp>(operation))
        if (mlir::failed(analyzeBlock(loop.getBody().front(), loop, plan,
                                      nextGroup)))
          return mlir::failure();
    return mlir::success();
  }

  GroupPlan *findPlannedGroup(KernelPlan &plan, int64_t id) {
    auto found = llvm::find_if(
        plan.groups, [&](const GroupPlan &group) { return group.id == id; });
    return found == plan.groups.end() ? nullptr : &*found;
  }

  mlir::LogicalResult finalizeLoopCarriedStates(KernelPlan &plan) {
    llvm::SmallVector<kernel::ForOp, 4> loops;
    plan.kernel->walk([&](kernel::ForOp loop) { loops.push_back(loop); });
    for (kernel::ForOp loop : loops) {
      mlir::Block &body = loop.getBody().front();
      auto yield = llvm::cast<kernel::YieldOp>(body.getTerminator());
      unsigned blockedCarried = 0;
      for (auto [index, values] : llvm::enumerate(llvm::zip(
               loop.getInitArgs(), loop.getResults(),
               body.getArguments().drop_front(), yield.getValues()))) {
        auto [init, result, argument, yielded] = values;
        bool blocked = isBlocked(init) || isBlocked(result) ||
                       isBlocked(argument) || isBlocked(yielded);
        if (!blocked)
          continue;
        ++blockedCarried;
        if (!isBlocked(init) || !isBlocked(result) || !isBlocked(argument) ||
            !isBlocked(yielded)) {
          loop.emitOpError(
              "a carried state cannot cross the scalar/block boundary");
          return mlir::failure();
        }
        auto contract = yielded.getDefiningOp<kernel::ContractOp>();
        if (!contract || contract->getBlock() != &body ||
            contract.getInit() != argument || !isSupportedRankTwoContract(contract)) {
          loop.emitOpError(
              "first block-carried RVV slice requires the yielded state to be "
              "one rank-two f32 contraction of its carried body argument");
          return mlir::failure();
        }
        auto initGroup = plan.valueGroups.find(init);
        auto resultGroup = plan.valueGroups.find(result);
        auto yieldedGroup = plan.valueGroups.find(yielded);
        auto rhsGroup = plan.valueGroups.find(contract.getRhs());
        if (initGroup == plan.valueGroups.end() ||
            resultGroup == plan.valueGroups.end() ||
            yieldedGroup == plan.valueGroups.end() ||
            rhsGroup == plan.valueGroups.end()) {
          loop.emitOpError(
              "cannot resolve every physical group of its carried contraction");
          return mlir::failure();
        }
        if (initGroup->second != resultGroup->second) {
          loop.emitOpError(
              "loop init and result must share one outer physical layout group");
          return mlir::failure();
        }
        GroupPlan *outer = findPlannedGroup(plan, initGroup->second);
        GroupPlan *inner = findPlannedGroup(plan, yieldedGroup->second);
        GroupPlan *rhs = findPlannedGroup(plan, rhsGroup->second);
        if (!outer || !inner || !rhs || outer->layout != inner->layout ||
            outer->layout.getRank() != 2 ||
            outer->layout.getVectorAxes().size() != 1 ||
            outer->layout.getVectorAxes().front() != 1) {
          loop.emitOpError(
              "carried rank-two accumulator requires identical outer/body "
              "N-vector layouts");
          return mlir::failure();
        }

        int64_t contractPoint = -1;
        int64_t loopPoint = -1;
        int64_t contractNode = plan.nodes.lookup(contract.getOperation());
        int64_t loopNode = plan.nodes.lookup(loop.getOperation());
        for (const ValueLifetime &value : inner->values)
          if (value.sourceNode == contractNode) {
            contractPoint = value.defineAt;
            break;
          }
        for (const ValueLifetime &value : outer->values)
          if (value.sourceNode == loopNode) {
            loopPoint = value.defineAt;
            break;
          }
        bool selectedCommonRatio = false;
        int64_t budget = std::min(
            outer->registerBudget,
            std::min(rhs->registerBudget, inner->registerBudget));
        for (int64_t laneRatio : {8, 16, 32, 64}) {
          if (contractPoint < 0 || loopPoint < 0 ||
              !laneRatioIsLegal(*outer, laneRatio) ||
              !laneRatioIsLegal(*rhs, laneRatio) ||
              !laneRatioIsLegal(*inner, laneRatio) ||
              !hasLegalLMUL(32, laneRatio))
            continue;
          int64_t joint = registerUseAt(*outer, laneRatio, loopPoint) +
                          registerUseAt(*rhs, laneRatio, contractPoint) +
                          registerUseAt(*inner, laneRatio, contractPoint) +
                          registerFootprint(32, laneRatio);
          if (joint > budget)
            continue;
          outer->laneRatio = laneRatio;
          rhs->laneRatio = laneRatio;
          inner->laneRatio = laneRatio;
          selectedCommonRatio = true;
          break;
        }
        if (!selectedCommonRatio) {
          loop.emitOpError()
              << "has no common carried/RHS/result RVV lane ratio within "
                 "register budget "
              << budget;
          return mlir::failure();
        }

        plan.conversions.push_back(LayoutConversionPlan{
            plan.nodes.lookup(loop.getOperation()),
            static_cast<int64_t>(3 + index), yieldedGroup->second});
        plan.conversions.push_back(LayoutConversionPlan{
            plan.nodes.lookup(yield.getOperation()),
            static_cast<int64_t>(index), resultGroup->second});
      }
      if (blockedCarried != 0 &&
          (blockedCarried != 1 || loop.getInitArgs().size() != 1)) {
        loop.emitOpError(
            "first block-carried RVV slice supports exactly one carried block "
            "and no additional carried state");
        return mlir::failure();
      }
    }
    return mlir::success();
  }

  int64_t registerUseAt(const GroupPlan &group, int64_t laneRatio,
                        int64_t point) const {
    int64_t liveRegisters = 0;
    auto countLiveValues = [&](llvm::ArrayRef<ValueLifetime> values) {
      for (const ValueLifetime &value : values)
        if (value.defineAt <= point && point <= value.lastUseAt)
          liveRegisters += registerFootprint(value.sew, laneRatio);
    };
    countLiveValues(group.values);
    countLiveValues(group.pressureOnlyValues);
    for (const TransientRegisterUse &use : group.transients)
      if (use.at == point)
        liveRegisters +=
            use.fixedCount +
            use.scalableCount * registerFootprint(use.sew, laneRatio);
    return liveRegisters;
  }

  bool laneRatioIsLegal(const GroupPlan &group, int64_t laneRatio) const {
    auto hasIllegalValue = [&](llvm::ArrayRef<ValueLifetime> values) {
      return llvm::any_of(values, [&](const ValueLifetime &value) {
        return !hasLegalLMUL(value.sew, laneRatio);
      });
    };
    if (hasIllegalValue(group.values) ||
        hasIllegalValue(group.pressureOnlyValues) ||
        llvm::any_of(group.transients, [&](const TransientRegisterUse &use) {
          return use.scalableCount > 0 && !hasLegalLMUL(use.sew, laneRatio);
        }))
      return false;
    int64_t finalPoint = 0;
    for (const ValueLifetime &value : group.values)
      finalPoint = std::max(finalPoint, value.lastUseAt);
    for (const ValueLifetime &value : group.pressureOnlyValues)
      finalPoint = std::max(finalPoint, value.lastUseAt);
    for (const TransientRegisterUse &use : group.transients)
      finalPoint = std::max(finalPoint, use.at);
    for (int64_t point = 0; point <= finalPoint; ++point)
      if (registerUseAt(group, laneRatio, point) > group.registerBudget)
        return false;
    return true;
  }

  mlir::LogicalResult selectCommonLaneRatio(GroupPlan &rhs, GroupPlan &result,
                                            kernel::ContractOp contract,
                                            int64_t sourceNode) {
    auto rhsAxes = rhs.layout.getVectorAxes();
    auto resultAxes = result.layout.getVectorAxes();
    if (rhs.id == result.id || rhs.layout.getRank() != 2 ||
        result.layout.getRank() != 2 || rhsAxes.size() != 1 ||
        resultAxes.size() != 1 || rhsAxes.front() != 1 ||
        resultAxes.front() != 1) {
      contract.emitOpError(
          "common lane-ratio selection requires distinct N-vector rhs and "
          "result groups");
      return mlir::failure();
    }
    int64_t contractPoint = -1;
    for (const ValueLifetime &value : result.values)
      if (value.sourceNode == sourceNode) {
        contractPoint = value.defineAt;
        break;
      }
    if (contractPoint < 0) {
      contract.emitOpError(
          "cannot locate rank-two contraction in result-group liveness");
      return mlir::failure();
    }
    int64_t budget = std::min(rhs.registerBudget, result.registerBudget);
    for (int64_t laneRatio : {8, 16, 32, 64}) {
      if (!laneRatioIsLegal(rhs, laneRatio) ||
          !laneRatioIsLegal(result, laneRatio) ||
          !hasLegalLMUL(32, laneRatio))
        continue;
      int64_t joint = registerUseAt(rhs, laneRatio, contractPoint) +
                      registerUseAt(result, laneRatio, contractPoint) +
                      registerFootprint(32, laneRatio);
      if (joint > budget)
        continue;
      rhs.laneRatio = laneRatio;
      result.laneRatio = laneRatio;
      return mlir::success();
    }
    contract.emitOpError()
        << "has no common rhs/result RVV lane ratio within joint register "
           "budget "
        << budget;
    return mlir::failure();
  }

  mlir::LogicalResult selectLaneRatio(GroupPlan &group,
                                      mlir::Operation *scope) {
    if (group.layout.getVectorAxes().empty()) {
      group.laneRatio = -1;
      return mlir::success();
    }
    for (int64_t laneRatio : {8, 16, 32, 64}) {
      if (laneRatioIsLegal(group, laneRatio)) {
        group.laneRatio = laneRatio;
        return mlir::success();
      }
    }
    scope->emitError()
        << "has no legal common RVV LMUL/lane ratio within vector-register "
           "budget "
        << group.registerBudget;
    return mlir::failure();
  }

  static mlir::Operation *createRecord(
      mlir::OpBuilder &builder, mlir::Location location, llvm::StringRef name,
      llvm::ArrayRef<mlir::NamedAttribute> attributes) {
    mlir::OperationState state(location, name);
    state.addAttributes(attributes);
    return builder.create(state);
  }

  void materializePlan(mlir::ModuleOp module, KernelPlan &plan) {
    mlir::OpBuilder builder(module.getContext());
    for (auto [operation, node] : plan.nodes)
      operation->setAttr(kNodeAttrName, builder.getI64IntegerAttr(node));

    builder.setInsertionPointToEnd(module.getBody());
    mlir::OperationState state(plan.kernel.getLoc(),
                               execution::PlanOp::getOperationName());
    state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr(plan.symbol));
    state.addAttribute("kernel", mlir::FlatSymbolRefAttr::get(
                                     module.getContext(),
                                     plan.kernel.getSymName()));
    state.addAttribute("target", builder.getStringAttr(plan.target));
    state.addRegion();
    auto selected = llvm::cast<execution::PlanOp>(builder.create(state));
    selected.getBody().emplaceBlock();

    builder.setInsertionPointToEnd(&selected.getBody().front());
    int64_t gridRank = plan.kernel.getGridRankAttr().getInt();
    for (int64_t axis = 0; axis < gridRank; ++axis)
      createRecord(builder, plan.kernel.getLoc(),
                   execution::TaskBindingOp::getOperationName(),
                   {builder.getNamedAttr("axis", builder.getI64IntegerAttr(axis)),
                    builder.getNamedAttr("mapping",
                                         builder.getStringAttr("abi"))});
    for (const MetaBindingPlan &binding : plan.metaBindings) {
      createRecord(
          builder, plan.kernel.getLoc(),
          execution::MetaBindingOp::getOperationName(),
          {builder.getNamedAttr(
               "argument", builder.getI64IntegerAttr(binding.argument)),
           builder.getNamedAttr(
               "value", builder.getI64IntegerAttr(binding.value))});
    }

    for (GroupPlan &group : plan.groups) {
      std::string ownerSymbol = "rvv_g" + std::to_string(group.id);
      createRecord(
          builder, plan.kernel.getLoc(), execution::GroupOp::getOperationName(),
          {builder.getNamedAttr("id", builder.getI64IntegerAttr(group.id)),
           builder.getNamedAttr("scope_node",
                                builder.getI64IntegerAttr(group.scopeNode)),
           builder.getNamedAttr("layout", group.layout),
           builder.getNamedAttr(
               "owner", mlir::FlatSymbolRefAttr::get(module.getContext(),
                                                      ownerSymbol))});

      llvm::sort(group.values, [](const ValueLifetime &lhs,
                                  const ValueLifetime &rhs) {
        return std::tie(lhs.sourceNode, lhs.resultIndex) <
               std::tie(rhs.sourceNode, rhs.resultIndex);
      });
      for (const ValueLifetime &value : group.values)
        createRecord(
            builder, plan.kernel.getLoc(),
            execution::ValueLayoutOp::getOperationName(),
            {builder.getNamedAttr(
                 "source_node", builder.getI64IntegerAttr(value.sourceNode)),
             builder.getNamedAttr(
                 "source_result", builder.getI64IntegerAttr(value.resultIndex)),
             builder.getNamedAttr("group",
                                  builder.getI64IntegerAttr(group.id))});

      llvm::SmallVector<mlir::NamedAttribute, 4> groupConfigAttrs{
          builder.getNamedAttr(mlir::SymbolTable::getSymbolAttrName(),
                               builder.getStringAttr(ownerSymbol)),
          builder.getNamedAttr("group", builder.getI64IntegerAttr(group.id)),
          builder.getNamedAttr("register_budget",
                               builder.getI64IntegerAttr(group.registerBudget))};
      if (group.laneRatio > 0)
        groupConfigAttrs.push_back(builder.getNamedAttr(
            "lane_ratio", builder.getI64IntegerAttr(group.laneRatio)));
      createRecord(builder, plan.kernel.getLoc(),
                   rvv_execution::GroupConfigOp::getOperationName(),
                   groupConfigAttrs);

      for (int64_t unary : group.unaries) {
        auto source = llvm::cast<kernel::UnaryOp>(llvm::find_if(
            plan.nodes,
            [&](const auto &entry) { return entry.second == unary; })->first);
        createRecord(
            builder, source.getLoc(),
            rvv_execution::UnaryConfigOp::getOperationName(),
            {builder.getNamedAttr("source_node",
                                  builder.getI64IntegerAttr(unary)),
             builder.getNamedAttr("group",
                                  builder.getI64IntegerAttr(group.id)),
             builder.getNamedAttr("strategy",
                                  builder.getStringAttr("exp_poly_v1"))});
      }

      for (int64_t reduction : group.reductions) {
        auto source = llvm::cast<kernel::ReduceOp>(llvm::find_if(
            plan.nodes, [&](const auto &entry) {
              return entry.second == reduction;
            })->first);
        createRecord(
            builder, source.getLoc(),
            rvv_execution::ReductionConfigOp::getOperationName(),
            {builder.getNamedAttr("source_node",
                                  builder.getI64IntegerAttr(reduction)),
             builder.getNamedAttr("group",
                                  builder.getI64IntegerAttr(group.id)),
             builder.getNamedAttr(
                 "strategy",
                 builder.getStringAttr(source.getOrdered() ? "ordered"
                                                           : "tree"))});
      }
      for (int64_t contract : group.contracts) {
        auto source = llvm::cast<kernel::ContractOp>(llvm::find_if(
            plan.nodes, [&](const auto &entry) {
              return entry.second == contract;
            })->first);
        createRecord(
            builder, source.getLoc(),
            rvv_execution::ContractConfigOp::getOperationName(),
            {builder.getNamedAttr("source_node",
                                  builder.getI64IntegerAttr(contract)),
             builder.getNamedAttr(
                 "groups", builder.getDenseI64ArrayAttr({group.id, group.id})),
             builder.getNamedAttr(
                 "strategy",
                 builder.getStringAttr(source.getOrdered() ? "ordered"
                                                           : "tree"))});
      }
    }

    for (const LayoutConversionPlan &conversion : plan.conversions)
      createRecord(
          builder, plan.kernel.getLoc(),
          execution::LayoutConversionOp::getOperationName(),
          {builder.getNamedAttr(
               "consumer_node",
               builder.getI64IntegerAttr(conversion.consumerNode)),
           builder.getNamedAttr(
               "consumer_operand",
               builder.getI64IntegerAttr(conversion.consumerOperand)),
           builder.getNamedAttr(
               "group", builder.getI64IntegerAttr(conversion.targetGroup))});

    for (const MultiGroupContractPlan &contract :
         plan.multiGroupContracts) {
      mlir::Operation *source = llvm::find_if(plan.nodes, [&](const auto &entry) {
                                  return entry.second == contract.sourceNode;
                                })->first;
      createRecord(
          builder, source->getLoc(),
          rvv_execution::ContractConfigOp::getOperationName(),
          {builder.getNamedAttr(
               "source_node", builder.getI64IntegerAttr(contract.sourceNode)),
           builder.getNamedAttr(
               "groups",
               builder.getDenseI64ArrayAttr(
                   {contract.lhsGroup, contract.rhsGroup,
                    contract.resultGroup})),
           builder.getNamedAttr(
               "strategy",
               builder.getStringAttr("sequential"))});
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createSelectRISCvExecutionPass() {
  return std::make_unique<SelectRISCvExecutionPass>();
}

std::unique_ptr<::mlir::Pass>
createSelectRISCvExecutionPass(SelectRISCvExecutionOptions options) {
  return std::make_unique<SelectRISCvExecutionPass>(std::move(options));
}

} // namespace weft::transforms
