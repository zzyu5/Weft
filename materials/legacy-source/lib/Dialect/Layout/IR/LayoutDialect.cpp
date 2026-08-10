#include "Weft/Dialect/Layout/IR/LayoutAlgebra.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace weft::layout;

#include "Weft/Dialect/Layout/IR/LayoutOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Weft/Dialect/Layout/IR/LayoutAttrs.cpp.inc"

namespace {

bool isAxisSetValid(int64_t rank, llvm::ArrayRef<int64_t> axes) {
  llvm::SmallDenseSet<int64_t, 8> seen;
  return llvm::all_of(axes, [&](int64_t axis) {
    return axis >= 0 && axis < rank && seen.insert(axis).second;
  });
}

} // namespace

mlir::LogicalResult BlockedLayoutAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, int64_t rank,
    llvm::ArrayRef<int64_t> order,
    llvm::ArrayRef<int64_t> vectorAxes) {
  if (rank <= 0)
    return emitError() << "blocked layout rank must be positive";
  if (order.size() != static_cast<size_t>(rank) ||
      !isAxisSetValid(rank, order))
    return emitError()
           << "blocked layout order must be a permutation of [0, rank)";
  if (!isAxisSetValid(rank, vectorAxes))
    return emitError()
           << "blocked layout vector_axes must be unique axes in [0, rank)";

  llvm::SmallDenseSet<int64_t, 8> vectorSet(vectorAxes.begin(),
                                            vectorAxes.end());
  int64_t previousPosition = -1;
  for (int64_t axis : vectorAxes) {
    auto position = llvm::find(order, axis) - order.begin();
    if (position <= previousPosition)
      return emitError()
             << "blocked layout vector_axes must follow traversal order";
    previousPosition = position;
  }
  return mlir::success();
}

BlockedLayoutAttr weft::layout::getBlockedLayout(
    mlir::MLIRContext *context, int64_t rank, llvm::ArrayRef<int64_t> order,
    llvm::ArrayRef<int64_t> vectorAxes) {
  return BlockedLayoutAttr::get(context, rank, order, vectorAxes);
}

BlockedLayoutAttr weft::layout::getIdentityBlockedLayout(
    mlir::MLIRContext *context, int64_t rank,
    llvm::ArrayRef<int64_t> vectorAxes) {
  llvm::SmallVector<int64_t, 4> order;
  order.reserve(rank);
  for (int64_t axis = rank; axis > 0; --axis)
    order.push_back(axis - 1);
  return getBlockedLayout(context, rank, order, vectorAxes);
}

BlockedLayoutAttr weft::layout::getVectorFastestBlockedLayout(
    mlir::MLIRContext *context, int64_t rank,
    llvm::ArrayRef<int64_t> vectorAxes) {
  llvm::SmallVector<int64_t, 4> order(vectorAxes.begin(), vectorAxes.end());
  llvm::SmallDenseSet<int64_t, 4> vectorSet(vectorAxes.begin(),
                                            vectorAxes.end());
  order.reserve(rank);
  for (int64_t axis = rank; axis > 0; --axis)
    if (!vectorSet.contains(axis - 1))
      order.push_back(axis - 1);
  return getBlockedLayout(context, rank, order, vectorAxes);
}

mlir::FailureOr<BlockedLayoutAttr> weft::layout::unifyBlockedLayouts(
    BlockedLayoutAttr lhs, BlockedLayoutAttr rhs) {
  if (lhs == rhs)
    return lhs;
  return mlir::failure();
}

mlir::FailureOr<BlockedLayoutAttr> weft::layout::projectBlockedLayout(
    BlockedLayoutAttr layout, llvm::ArrayRef<int64_t> removedAxes) {
  if (!isAxisSetValid(layout.getRank(), removedAxes))
    return mlir::failure();

  llvm::SmallDenseSet<int64_t, 8> removed(removedAxes.begin(),
                                          removedAxes.end());
  if (removed.size() == static_cast<size_t>(layout.getRank()))
    return mlir::failure();

  auto remapAxis = [&](int64_t axis) {
    int64_t shift = 0;
    for (int64_t removedAxis : removedAxes)
      if (removedAxis < axis)
        ++shift;
    return axis - shift;
  };

  llvm::SmallVector<int64_t, 4> projectedOrder;
  for (int64_t axis : layout.getOrder())
    if (!removed.contains(axis))
      projectedOrder.push_back(remapAxis(axis));

  llvm::SmallVector<int64_t, 4> projectedVectorAxes;
  for (int64_t axis : layout.getVectorAxes())
    if (!removed.contains(axis))
      projectedVectorAxes.push_back(remapAxis(axis));

  return getBlockedLayout(layout.getContext(),
                          layout.getRank() - removed.size(), projectedOrder,
                          projectedVectorAxes);
}

mlir::FailureOr<BlockedLayoutAttr> weft::layout::insertBlockedAxis(
    BlockedLayoutAttr layout, int64_t logicalAxis,
    int64_t traversalPosition, bool vectorized) {
  int64_t oldRank = layout.getRank();
  if (logicalAxis < 0 || logicalAxis > oldRank || traversalPosition < 0 ||
      traversalPosition > oldRank)
    return mlir::failure();

  auto shiftAxis = [&](int64_t axis) {
    return axis >= logicalAxis ? axis + 1 : axis;
  };
  llvm::SmallVector<int64_t, 4> order;
  order.reserve(oldRank + 1);
  for (auto [position, axis] : llvm::enumerate(layout.getOrder())) {
    if (static_cast<int64_t>(position) == traversalPosition)
      order.push_back(logicalAxis);
    order.push_back(shiftAxis(axis));
  }
  if (traversalPosition == oldRank)
    order.push_back(logicalAxis);

  llvm::SmallVector<int64_t, 4> vectorAxes;
  for (int64_t axis : layout.getVectorAxes())
    vectorAxes.push_back(shiftAxis(axis));
  if (vectorized) {
    auto insertion = llvm::find(order, logicalAxis) - order.begin();
    auto position = llvm::find_if(vectorAxes, [&](int64_t axis) {
      return (llvm::find(order, axis) - order.begin()) > insertion;
    });
    vectorAxes.insert(position, logicalAxis);
  }
  return getBlockedLayout(layout.getContext(), oldRank + 1, order, vectorAxes);
}

void WEFTLayoutDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Weft/Dialect/Layout/IR/LayoutAttrs.cpp.inc"
      >();
}
