#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
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
                         true};
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
                         true};
  }
  if (auto extract = value.getDefiningOp<riscv::ExtractOp>()) {
    auto field = extract.getInput().getDefiningOp<riscv::FieldOp>();
    auto type = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
    auto pattern = extract->getAttrOfType<mlir::DenseI64ArrayAttr>("index_pattern");
    auto position = type ? axisPosition(type, axis) : std::optional<size_t>();
    if (!field || !type || !pattern || pattern.size() != 3 || !position)
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
                  riscv::RVVWidenMultiplyOp,
                  riscv::RVVWidenScalarMultiplyOp,
                  riscv::RegisterMaterializeOp>(definition))
      for (mlir::Value operand : definition->getOperands())
        if (mlir::isa<riscv::ValueType>(operand.getType()))
          dependent |= dependsOnAny(operand, replacements, cache);
  cache[value] = dependent;
  return dependent;
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

riscv::ValueType flattenedLanePartialSlotType(mlir::Builder &builder,
                                              riscv::ValueType operand,
                                              int64_t reductionAxis) {
  auto inputElement =
      mlir::dyn_cast<mlir::IntegerType>(operand.getElementType());
  auto lanes = riscv_internal::staticProduct(
      operand.getLayout().getLaneFactors().asArrayRef());
  if (!axisPosition(operand, reductionAxis) || !inputElement ||
      inputElement.isSignless() || inputElement.getWidth() > 16 || !lanes ||
      *lanes <= 0)
    return {};
  const int64_t partialWidth =
      2 * std::max<int64_t>(8, inputElement.getWidth());
  const int64_t partialLMUL = operand.getLayout().getLmulEighths() * 2;
  auto axisIds = riscv_internal::integers(builder, {reductionAxis});
  auto one = riscv_internal::integers(builder, {1});
  auto lane = riscv_internal::integers(builder, {*lanes});
  auto layout = riscv::LayoutAttr::get(
      builder.getContext(), "rvv", axisIds, one, lane, one, one, one,
      partialWidth, partialLMUL, *lanes,
      std::max<int64_t>(1, (partialLMUL + 7) / 8),
      operand.getLayout().getValidity());
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
      1, "exact");
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
  auto slot = partialSlotType(builder, lhs, reductionAxis);
  auto rhsSlot = partialSlotType(builder, rhs, reductionAxis);
  auto result = mlir::dyn_cast<mlir::IntegerType>(leaf.dot.getResult().getType());
  if (!lhsStreams || !rhsStreams || *lhsStreams != 1 || *rhsStreams != 1 ||
      !slot || rhsSlot != slot || !result || !result.isSigned() ||
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
  for (int64_t extent : source.getShape().asArrayRef()) {
    if (extent <= 0)
      return {};
    replicas.push_back(extent);
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
        dot.getStreamReduction() != "per_stream" ||
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
  if (dot.getOver().size() != 1 ||
      (dot.getStreamReduction() != "per_stream" &&
       dot.getStreamReduction() != "fused"))
    return std::nullopt;
  return ReplicaScaledDotReduction{std::move(reductions),
                                   std::move(reductionConversions),
                                   std::move(reducedAxes), multiply, dot, dotSide,
                                   scale};
}

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

    llvm::SmallVector<riscv::ExtractOp> extracts;
    getOperation().walk(
        [&](riscv::ExtractOp extract) { extracts.push_back(extract); });
    for (riscv::ExtractOp extract : extracts) {
      auto field = extract.getInput().getDefiningOp<riscv::FieldOp>();
      auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
      if (!field || !result || extract.getAccess().getMapping() !=
                                   "grouped_layered" ||
          extract.getIndices().size() != 1 ||
          product(result.getLayout().getTimeFactors()) != 1)
        continue;
      auto point = extract.getIndices().front().getDefiningOp<riscv::PhysicalPointOp>();
      if (!point)
        continue;
      if (point.getResult().getType().getDomain().getTail() != "exact")
        continue;
      const int64_t axis = point.getResult().getType().getDomain().getAxisId();
      auto position = axisPosition(result, axis);
      if (!position || result.getShape()[*position] !=
                           result.getLayout().getLaneFactors()[*position])
        continue;
      rewriter.setInsertionPoint(extract);
      mlir::Value zero = rewriter.create<mlir::arith::ConstantIndexOp>(
          extract.getLoc(), 0);
      llvm::StringRef tail = result.getLayout().getValidity() == "tail"
                                 ? "agnostic"
                                 : "exact";
      auto window = rewriter.create<riscv::RVVStorageWindowOp>(
          extract.getLoc(), result, field.getResult(), point.getResult(), zero,
          axis, 0, 1, 1, result.getShape()[*position],
          result.getLayout().getLaneFactors()[*position], extract.getAccess(),
          riscv_internal::leaf(rewriter, "rvv", "storage-window",
                               "rvv.storage-window", "rvv.storage-window",
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
              result.getLayout().getReplicaFactors())
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
      auto lhsElement = mlir::dyn_cast<mlir::IntegerType>(lhs.getElementType());
      auto rhsElement = mlir::dyn_cast<mlir::IntegerType>(rhs.getElementType());
      if (!lhsElement || !rhsElement || lhsElement.isSignless() ||
          rhsElement.isSignless() ||
          (!lhsElement.isSigned() && !rhsElement.isSigned()))
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

    llvm::SmallVector<riscv::ReduceOp> replicaReductions;
    llvm::SmallVector<mlir::Value> replicaCleanupRoots;
    getOperation().walk([&](riscv::ReduceOp reduce) {
      if (mlir::isa<mlir::IntegerType>(reduce.getResult().getType()))
        replicaReductions.push_back(reduce);
    });
    for (riscv::ReduceOp reduce : replicaReductions) {
      auto match = matchReplicaScaledDotReduction(reduce);
      if (!match)
        continue;
      riscv::RVVWidenDotOp dot = match->dot;
      riscv::ValueType lhs = dot.getLhs().getType();
      riscv::ValueType rhs = dot.getRhs().getType();
      auto dotResult = mlir::dyn_cast<riscv::ValueType>(dot.getResult().getType());
      auto scaledType =
          mlir::dyn_cast<riscv::ValueType>(match->scaleMultiply.getResult().getType());
      auto resultElement = mlir::dyn_cast<mlir::IntegerType>(reduce.getResult().getType());
      const int64_t reductionAxis = dot.getOver()[0];
      auto lhsReduction = axisPosition(lhs, reductionAxis);
      auto rhsReduction = axisPosition(rhs, reductionAxis);
      auto slots = dotResult
                       ? riscv_internal::staticProduct(
                             dotResult.getLayout().getReplicaFactors().asArrayRef())
                       : std::optional<int64_t>();
      auto slotType = partialSlotType(rewriter, lhs, reductionAxis);
      bool complete = dotResult && scaledType && resultElement &&
                      resultElement.isSigned() && resultElement.getWidth() == 32 &&
                      lhsReduction && rhsReduction && slotType && slots &&
                      *slots >= 4 && ((*slots & (*slots - 1)) == 0) &&
                      dotResult.getLayout().getCarrier() == "scalar" &&
                      dotResult.getShape() == scaledType.getShape() &&
                      dotResult.getAxisIds() == scaledType.getAxisIds() &&
                      match->reducedAxes.size() ==
                          dotResult.getAxisIds().size() &&
                      llvm::all_of(dotResult.getAxisIds().asArrayRef(),
                                   [&](int64_t axis) {
                                     return llvm::is_contained(
                                         match->reducedAxes, axis);
                                   }) &&
                      lhs.getLayout().getTimeFactors()[*lhsReduction] == 1 &&
                      rhs.getLayout().getTimeFactors()[*rhsReduction] == 1;
      if (complete) {
        for (size_t position = 0; position < dotResult.getShape().size(); ++position)
          complete &= dotResult.getLayout().getTimeFactors()[position] == 1 &&
                      dotResult.getLayout().getLaneFactors()[position] == 1 &&
                      dotResult.getLayout().getReplicaFactors()[position] ==
                          dotResult.getShape()[position];
      }
      auto multiplyInstruction =
          complete ? partialMultiplyInstruction(lhs, rhs) : std::nullopt;
      if (!complete || !multiplyInstruction)
        continue;

      auto lhsStreams = riscv_internal::staticProduct(
          lhs.getLayout().getTimeFactors().asArrayRef());
      auto rhsStreams = riscv_internal::staticProduct(
          rhs.getLayout().getTimeFactors().asArrayRef());
      auto lhsReplicas = riscv_internal::staticProduct(
          lhs.getLayout().getReplicaFactors().asArrayRef());
      auto rhsReplicas = riscv_internal::staticProduct(
          rhs.getLayout().getReplicaFactors().asArrayRef());
      auto lhsLanes = riscv_internal::staticProduct(
          lhs.getLayout().getLaneFactors().asArrayRef());
      auto rhsLanes = riscv_internal::staticProduct(
          rhs.getLayout().getLaneFactors().asArrayRef());
      const int64_t reductionLanes =
          lhs.getLayout().getLaneFactors()[*lhsReduction];
      auto flattenedSlot = flattenedLanePartialSlotType(
          rewriter, lhs, reductionAxis);
      const int64_t laneSplit =
          lhsLanes && reductionLanes > 0 && *lhsLanes % reductionLanes == 0
              ? *lhsLanes / reductionLanes
              : 1;
      auto splitSlot = laneSplit > 1 && flattenedSlot
                           ? splitPartialSlotType(rewriter, flattenedSlot,
                                                  reductionAxis, laneSplit)
                           : riscv::ValueType();
      const bool groupedLanePartials =
          laneSplit > 1 && lhsStreams && rhsStreams && lhsReplicas &&
          rhsReplicas && lhsLanes && rhsLanes &&
          *lhsStreams == *rhsStreams && *lhsStreams > 0 &&
          *lhsReplicas == 1 && *rhsReplicas == 1 && *lhsLanes == *rhsLanes &&
          *lhsStreams * laneSplit == *slots && flattenedSlot && splitSlot &&
          splitSlot.getElementType() == slotType.getElementType();
      bool plannedReplicaLanePartials =
          laneSplit > 1 && lhsStreams && rhsStreams && lhsReplicas &&
          rhsReplicas && lhsLanes && rhsLanes &&
          dot.getReductionStreams() == 1 && *lhsStreams == *rhsStreams &&
          *lhsLanes == *rhsLanes && flattenedSlot && splitSlot &&
          splitSlot.getElementType() == slotType.getElementType();
      int64_t plannedSourceSlots = 0;
      if (plannedReplicaLanePartials) {
        plannedSourceSlots = *lhsStreams * *lhsReplicas;
        plannedReplicaLanePartials =
            plannedSourceSlots > 0 &&
            plannedSourceSlots == *rhsStreams * *rhsReplicas &&
            plannedSourceSlots * laneSplit == *slots &&
            dot.getLhsLaneOffsets().size() == static_cast<size_t>(*slots) &&
            dot.getRhsLaneOffsets().size() == static_cast<size_t>(*slots) &&
            dot.getLhsParts().size() == static_cast<size_t>(*slots) &&
            dot.getRhsParts().size() == static_cast<size_t>(*slots);
      }
      if (plannedReplicaLanePartials) {
        for (int64_t sourceSlot = 0; sourceSlot < plannedSourceSlots;
             ++sourceSlot) {
          const int64_t first = sourceSlot * laneSplit;
          const int64_t lhsPart = dot.getLhsParts()[first];
          const int64_t rhsPart = dot.getRhsParts()[first];
          for (int64_t chunk = 0; chunk < laneSplit; ++chunk) {
            const int64_t outputSlot = first + chunk;
            plannedReplicaLanePartials &=
                dot.getLhsParts()[outputSlot] == lhsPart &&
                dot.getRhsParts()[outputSlot] == rhsPart &&
                dot.getLhsLaneOffsets()[outputSlot] ==
                    chunk * reductionLanes &&
                dot.getRhsLaneOffsets()[outputSlot] ==
                    chunk * reductionLanes;
          }
        }
      }
      const bool splitLanePartials =
          groupedLanePartials || plannedReplicaLanePartials;
      if (dot.getStreamReduction() == "fused" && !splitLanePartials)
        continue;
      if (splitLanePartials)
        slotType = splitSlot;
      const int64_t sourceSlots =
          plannedReplicaLanePartials
              ? plannedSourceSlots
              : (groupedLanePartials ? *lhsStreams : *slots);
      llvm::SmallVector<int64_t> operandSlots(sourceSlots, 0);
      llvm::SmallVector<int64_t> lhsParts;
      llvm::SmallVector<int64_t> rhsParts;
      if (plannedReplicaLanePartials) {
        for (int64_t sourceSlot = 0; sourceSlot < sourceSlots; ++sourceSlot) {
          const int64_t first = sourceSlot * laneSplit;
          lhsParts.push_back(dot.getLhsParts()[first]);
          rhsParts.push_back(dot.getRhsParts()[first]);
        }
      } else if (groupedLanePartials) {
        for (int64_t stream = 0; stream < sourceSlots; ++stream) {
          lhsParts.push_back(stream);
          rhsParts.push_back(stream);
        }
      } else {
        for (int64_t slot = 0; slot < *slots; ++slot) {
          auto lhsPart = projectReplica(lhs, dotResult, slot);
          auto rhsPart = projectReplica(rhs, dotResult, slot);
          if (!lhsPart || !rhsPart) {
            complete = false;
            break;
          }
          lhsParts.push_back(*lhsPart);
          rhsParts.push_back(*rhsPart);
        }
      }
      if (!complete)
        continue;

      auto kernel = dot->getParentOfType<riscv::KernelOp>();
      const int64_t setGroups =
          sourceSlots *
          (splitLanePartials
               ? flattenedSlot.getLayout().getRegisterGroups()
               : slotType.getLayout().getRegisterGroups());
      const int64_t operandGroups = lhs.getLayout().getRegisterGroups() +
                                    rhs.getLayout().getRegisterGroups();
      if (!kernel || setGroups <= 0 || operandGroups < 0 ||
          setGroups > kernel.getTarget().getVectorRegisters() - operandGroups)
        continue;

      rewriter.setInsertionPoint(reduce);
      llvm::DenseMap<mlir::Value, mlir::Value> rematerialized;
      auto scalarScale =
          rematerializeScalarReplicas(match->scale, rewriter, rematerialized);
      auto scalarScaleType =
          mlir::succeeded(scalarScale)
              ? mlir::dyn_cast<riscv::ValueType>((*scalarScale).getType())
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
          !scaleParts || *scaleParts != *slots || !scaleElement ||
          !scaleElement.isSigned() || scaleElement.getWidth() != 32)
        continue;

      auto reducedSlot =
          reducedPartialSlotType(rewriter, slotType, reductionAxis);
      if (!reducedSlot)
        continue;
      riscv::ValueType setSlotType =
          splitLanePartials ? flattenedSlot : slotType;
      auto setType = riscv::PartialSetType::get(
          rewriter.getContext(), setSlotType, reductionAxis, sourceSlots,
          setSlotType.getShape()[0], setGroups);
      auto repackedType = splitLanePartials
                              ? riscv::PartialSetType::get(
                                    rewriter.getContext(), slotType,
                                    reductionAxis, *slots,
                                    slotType.getShape()[0], setGroups)
                              : setType;
      auto reducedType = riscv::PartialSetType::get(
          rewriter.getContext(), reducedSlot, reductionAxis, *slots,
          repackedType.getTermsPerSlot(),
          *slots * reducedSlot.getLayout().getRegisterGroups());
      const int64_t chains = *slots >= 4 && *slots % 2 == 0 ? 2 : 1;
      const int64_t fanout = *slots / chains;
      auto combinedType = riscv::PartialSetType::get(
          rewriter.getContext(), reducedSlot, reductionAxis, chains,
          reducedType.getTermsPerSlot() * fanout,
          chains * reducedSlot.getLayout().getRegisterGroups());
      llvm::SmallVector<int64_t> slotOrder;
      for (int64_t chain = 0; chain < chains; ++chain)
        for (int64_t slot = chain; slot < *slots; slot += chains)
          slotOrder.push_back(slot);
      llvm::SmallVector<mlir::Value> scales(*slots, *scalarScale);
      llvm::SmallVector<int64_t> scaleReplicas;
      for (int64_t slot = 0; slot < *slots; ++slot)
        scaleReplicas.push_back(slot);

      auto set = rewriter.create<riscv::RVVPartialSetOp>(
          reduce.getLoc(), setType, mlir::ValueRange{dot.getLhs()},
          mlir::ValueRange{dot.getRhs()},
          rewriter.getDenseI64ArrayAttr(operandSlots),
          rewriter.getDenseI64ArrayAttr(operandSlots),
          rewriter.getDenseI64ArrayAttr(lhsParts),
          rewriter.getDenseI64ArrayAttr(rhsParts), reductionAxis,
          *multiplyInstruction,
          riscv_internal::leaf(rewriter, "rvv", "partial-set",
                               "rvv.partial-set", "rvv.partial-set", 0,
                               setType.getResourceGroups(), 0, 0, "none",
                               "exact", {reductionAxis, sourceSlots}));
      riscv_internal::copyOrigin(dot, set);
      mlir::Value partials = set.getResult();
      if (splitLanePartials) {
        auto repacked = rewriter.create<riscv::RVVPartialRepackOp>(
            reduce.getLoc(), repackedType, partials, laneSplit,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-repack",
                "rvv.partial-repack.split", "rvv.partial-repack.split",
                setType.getResourceGroups(), repackedType.getResourceGroups(),
                0, 0, "none", "exact", {reductionAxis, laneSplit}));
        riscv_internal::copyOrigin(dot, repacked);
        partials = repacked.getResult();
      }
      auto reduced = rewriter.create<riscv::RVVPartialReduceOp>(
          reduce.getLoc(), reducedType, partials,
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
              repackedType.getResourceGroups(),
              reducedType.getResourceGroups(), 1, 0,
              "none", "exact", {reductionAxis, *slots}));
      riscv_internal::copyOrigin(dot, reduced);
      auto combined = rewriter.create<riscv::RVVPartialScaleCombineOp>(
          reduce.getLoc(), combinedType, reduced.getResult(), scales,
          rewriter.getDenseI64ArrayAttr(scaleReplicas),
          rewriter.getDenseI64ArrayAttr(slotOrder),
          riscv_internal::leaf(
              rewriter, "rvv", "partial-scale-combine",
              "rvv.partial-scale-combine", "rvv.partial-scale-combine",
              reducedType.getResourceGroups(), combinedType.getResourceGroups(), 1,
              0, "none", "exact",
              {reductionAxis, *slots, chains, fanout}));
      riscv_internal::copyOrigin(dot, combined);
      auto finalized = rewriter.create<riscv::RVVPartialFinalizeOp>(
          reduce.getLoc(), reduce.getResult().getType(), combined.getResult(),
          reductionAxis,
          riscv_internal::leaf(rewriter, "rvv", "partial-finalize",
                               *partialFinalizeInstruction(combinedType,
                                                           reductionAxis),
                               *partialFinalizeInstruction(combinedType,
                                                           reductionAxis),
                               combinedType.getResourceGroups(), 0, 0, 0,
                               "none", "exact", {reductionAxis, chains}));
      riscv_internal::copyOrigin(dot, finalized);
      reduce.getResult().replaceAllUsesWith(finalized.getResult());

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
          if (!operation || !llvm::all_of(operation->getResults(),
                                          [](mlir::Value result) {
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
        finalized.emitError(
            "partial-accumulator rewrite left a live reduction-chain operation");
        failed = true;
        break;
      }
      // Keep the source graph intact while the remaining reductions are still
      // being matched.  Deleting a producer here invalidates the operation
      // handles collected above and makes the rewrite allocator-dependent.
      replicaCleanupRoots.push_back(oldScale);
      replicaCleanupRoots.push_back(oldDotSide);
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
      if (loop.getInitArgs().size() != 1 || loop.getNumResults() != 1)
        continue;
      auto yield =
          mlir::dyn_cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
      if (!yield || yield.getNumOperands() != 1)
        continue;

      mlir::Value cursor = yield.getOperand(0);
      auto first = matchScaledPartialContribution(cursor);
      if (!first)
        continue;
      const int64_t slots = first->dot.getPartialUnroll();
      if (slots <= 1)
        continue;
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
        continue;
      std::reverse(contributions.begin(), contributions.end());

      const int64_t reductionAxis = contributions.front().dot.getOver()[0];
      auto carryType =
          mlir::dyn_cast<riscv::ValueType>(loop.getRegionIterArg(0).getType());
      auto carryParts =
          carryType
              ? riscv_internal::staticProduct(
                    carryType.getLayout().getReplicaFactors().asArrayRef())
              : std::optional<int64_t>();
      auto slotType = partialSlotType(
          rewriter, contributions.front().dot.getLhs().getType(), reductionAxis);
      bool complete = carryType && carryType.getLayout().getCarrier() == "scalar" &&
                      carryParts && *carryParts > 0 && slotType;
      for (ScaledPartialContribution &contribution : contributions) {
        riscv::ValueType lhs = contribution.dot.getLhs().getType();
        riscv::ValueType rhs = contribution.dot.getRhs().getType();
        auto lhsPosition = axisPosition(lhs, reductionAxis);
        auto rhsPosition = axisPosition(rhs, reductionAxis);
        complete &= contribution.dot.getPartialUnroll() == slots &&
                    contribution.dot.getOver().size() == 1 &&
                    contribution.dot.getOver()[0] == reductionAxis &&
                    contribution.dot.getResult().getType() == carryType &&
                    contribution.scaleMultiply.getResult().getType() == carryType &&
                    contribution.carryAdd.getResult().getType() == carryType &&
                    lhsPosition && rhsPosition &&
                    lhs.getLayout().getTimeFactors()[*lhsPosition] == 1 &&
                    rhs.getLayout().getTimeFactors()[*rhsPosition] == 1 &&
                    partialSlotType(rewriter, lhs, reductionAxis) == slotType;
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

      auto reducedSlot =
          reducedPartialSlotType(rewriter, slotType, reductionAxis);
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
      const int64_t chains = slots >= 4 && slots % 2 == 0 ? 2 : 1;
      const int64_t fanout = slots / chains;
      auto combinedType = riscv::PartialSetType::get(
          rewriter.getContext(), reducedSlot, reductionAxis, chains,
          reducedType.getTermsPerSlot() * fanout,
          chains * reducedSlot.getLayout().getRegisterGroups());
      llvm::SmallVector<int64_t> slotOrder;
      for (int64_t chain = 0; chain < chains; ++chain)
        for (int64_t slot = chain; slot < slots; slot += chains)
          slotOrder.push_back(slot);

      rewriter.setInsertionPoint(yield);
      llvm::SmallVector<mlir::Value> scalarParts;
      for (OutputPartialOperands &operands : outputOperands) {
        llvm::SmallVector<int64_t> operandSlots;
        for (int64_t slot = 0; slot < slots; ++slot)
          operandSlots.push_back(slot);
        auto set = rewriter.create<riscv::RVVPartialSetOp>(
            yield.getLoc(), setType, operands.lhs, operands.rhs,
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(operandSlots),
            rewriter.getDenseI64ArrayAttr(operands.lhsReplicas),
            rewriter.getDenseI64ArrayAttr(operands.rhsReplicas), reductionAxis,
            multiplyInstruction,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-set", "rvv.partial-set",
                "rvv.partial-set", 0, setType.getResourceGroups(), 0, 0,
                "none", "exact", {reductionAxis, slots}));
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
            rewriter.getDenseI64ArrayAttr(slotOrder),
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

    llvm::SmallVector<riscv::RVVWidenDotOp> independentDots;
    getOperation().walk([&](riscv::RVVWidenDotOp dot) {
      independentDots.push_back(dot);
    });
    for (riscv::RVVWidenDotOp dot : independentDots) {
      const bool fusedLogicalAxes =
          dot.getStreamReduction() == "fused" && dot.getOver().size() > 1;
      // A single-axis fused stream may still own a layered storage-window
      // topology handled below.  A multi-axis fused dot has already declared
      // every issue-time coordinate as a logical reduction axis, so its time
      // parts are independent typed partials rather than one storage stream.
      if (dot.getOver().empty() ||
          (dot.getStreamReduction() == "fused" && !fusedLogicalAxes))
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
      if ((fusedLogicalAxes && slots < 2) ||
          (!fusedLogicalAxes &&
           (slots < 4 || (slots & (slots - 1)) != 0)))
        continue;
      auto slotType = partialSlotType(rewriter, lhs, reductionAxis);
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
          vectorResult
              ? vectorPartialSlotType(rewriter, lhs, vectorResult, reductionAxis)
              : riscv::ValueType();
      auto rhsVectorSlotType =
          vectorResult
              ? vectorPartialSlotType(rewriter, rhs, vectorResult, reductionAxis)
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
          vectorConsumer && vectorResult && vectorSlotType &&
          rhsVectorSlotType == vectorSlotType && lhsStreams && rhsStreams &&
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
            rewriter.getDenseI64ArrayAttr(operandParts), reductionAxis,
            *multiplyInstruction,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-set", "rvv.partial-set",
                "rvv.partial-set", operandGroups, setGroups, 0, 0, "none",
                "exact", {reductionAxis, slots}));
        riscv_internal::copyOrigin(dot, set);
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
            rewriter.getDenseI64ArrayAttr(rhsParts), reductionAxis,
            *multiplyInstruction,
            riscv_internal::leaf(
                rewriter, "rvv", "partial-set", "rvv.partial-set",
                "rvv.partial-set", operandGroups, setGroups, 0, 0, "none",
                "exact", {reductionAxis, slots}));
        riscv_internal::copyOrigin(dot, set);
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
              sourceType.getReductionAxis(), *multiplyInstruction,
              riscv_internal::leaf(
                  rewriter, "rvv", "partial-set", "rvv.partial-set",
                  "rvv.partial-set", 0, sourceType.getResourceGroups(), 0, 0,
                  "none", "exact", {sourceType.getReductionAxis(), 1}));
          riscv_internal::copyOrigin(leaf.dot, set);
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
      if (dot.getStreamReduction() != "fused" || dot.getOver().size() != 1)
        continue;
      const int64_t reductionAxis = dot.getOver()[0];
      auto lhsType = dot.getLhs().getType();
      auto rhsType = dot.getRhs().getType();
      auto lhsPosition = axisPosition(lhsType, reductionAxis);
      auto rhsPosition = axisPosition(rhsType, reductionAxis);
      if (!lhsPosition || !rhsPosition)
        continue;
      const int64_t streams = lhsType.getLayout().getTimeFactors()[*lhsPosition];
      if (streams <= 1 || streams !=
                              rhsType.getLayout().getTimeFactors()[*rhsPosition])
        continue;

      llvm::DenseSet<mlir::Operation *> lhsVisited;
      llvm::DenseSet<mlir::Operation *> rhsVisited;
      llvm::SmallVector<ProjectedRoot> lhsRoots;
      llvm::SmallVector<ProjectedRoot> rhsRoots;
      collectProjectedRoots(dot.getLhs(), reductionAxis, lhsVisited, lhsRoots);
      collectProjectedRoots(dot.getRhs(), reductionAxis, rhsVisited, rhsRoots);
      llvm::SmallVector<ProjectedRoot> roots(lhsRoots.begin(), lhsRoots.end());
      roots.append(rhsRoots.begin(), rhsRoots.end());
      auto layeredIt = llvm::find_if(
          roots, [](const ProjectedRoot &root) { return root.layered; });
      if (layeredIt == roots.end() ||
          llvm::count_if(roots, [](const ProjectedRoot &root) {
            return root.layered;
          }) != 1)
        continue;
      ProjectedRoot layered = *layeredIt;
      if (!layered.field || !layered.origin)
        continue;

      riscv::AccessAttr access = layered.access;
      const int64_t group = access.getGroupSize();
      const int64_t layerExtent = access.getLayerSize();
      const int64_t layers = layerExtent > 0 ? group / layerExtent : 0;
      auto layeredInteger = mlir::dyn_cast<mlir::IntegerType>(
          mlir::cast<riscv::ValueType>(layered.field.getResult().getType())
              .getElementType());
      auto layeredPosition = axisPosition(layered.type, reductionAxis);
      if (!layeredPosition)
        continue;
      const int64_t lanes =
          layered.type.getLayout().getLaneFactors()[*layeredPosition];
      if (!layeredInteger || layeredInteger.isSigned() || group <= 0 ||
          layerExtent <= 0 || group % layerExtent || layers <= 1 ||
          layeredInteger.getWidth() * layers != 8 || access.getBitOffset() % 8 ||
          (access.getOrder() != "lo_first" && access.getOrder() != "hi_first") ||
          layered.origin.getResult().getType().getDomain().getTail() != "exact" ||
          lanes <= 1 || layerExtent % lanes || streams % layers ||
          streams <= layers)
        continue;
      const int64_t windowsPerLayer = layerExtent / lanes;
      const int64_t windowCount = streams / layers;
      auto layeredWindowType = projectOneWindow(rewriter, layered.type,
                                                 reductionAxis);
      llvm::DenseMap<mlir::Value, riscv::ValueType> windowTypes;
      bool completeRoots = static_cast<bool>(layeredWindowType);
      for (ProjectedRoot &root : roots) {
        auto position = axisPosition(root.type, reductionAxis);
        auto fieldType = root.field
                             ? mlir::dyn_cast<riscv::ValueType>(
                                   root.field.getResult().getType())
                             : riscv::ValueType();
        auto fieldPosition = fieldType
                                 ? axisPosition(fieldType, reductionAxis)
                                 : std::optional<size_t>();
        auto integer = fieldType
                           ? mlir::dyn_cast<mlir::IntegerType>(
                                 fieldType.getElementType())
                           : mlir::IntegerType();
        auto window = projectOneWindow(rewriter, root.type, reductionAxis);
        const bool bounded =
            position && fieldPosition && root.base >= 0 && root.stride > 0 &&
            root.repeat > 0 && root.extent == root.type.getShape()[*position] &&
            root.extent > 0 && fieldType.getShape()[*fieldPosition] > 0 &&
            root.base <= fieldType.getShape()[*fieldPosition] - 1 &&
            (root.extent - 1) / root.repeat <=
                (fieldType.getShape()[*fieldPosition] - 1 - root.base) /
                    root.stride;
        completeRoots &= root.field && root.origin && integer && window &&
                         bounded &&
                         root.origin.getResult().getType().getDomain().getTail() ==
                             "exact" &&
                         root.type.getLayout().getTimeFactors()[*position] == streams &&
                         root.type.getLayout().getLaneFactors()[*position] == lanes &&
                         root.extent == layered.extent &&
                         (root.repeat % lanes == 0 || lanes % root.repeat == 0) &&
                         (root.layered ||
                          (root.access.getMapping() == "natural" &&
                           root.access.getBitOffset() % 8 == 0 &&
                           integer.getWidth() >= 8 && integer.getWidth() <= 32 &&
                           integer.getWidth() % 8 == 0));
        if (window)
          windowTypes[root.value] = window;
      }
      auto accumulatorType =
          partialType(rewriter, lhsType, dot.getResult().getType(), reductionAxis);
      auto partialInteger =
          accumulatorType
              ? mlir::dyn_cast<mlir::IntegerType>(accumulatorType.getElementType())
              : mlir::IntegerType();
      if (!completeRoots || !accumulatorType || !partialInteger ||
          (partialInteger.getWidth() != 16 && partialInteger.getWidth() != 32))
        continue;

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
      auto storageType = riscv::LayeredWindowType::get(
          rewriter.getContext(),
          mlir::cast<riscv::ValueType>(layered.field.getResult().getType()),
          layeredWindowType, reductionAxis, layers, windowsPerLayer, rawGroups);
      auto storage = rewriter.create<riscv::RVVLayeredStorageLoadOp>(
          dot.getLoc(), storageType, layered.field.getResult(),
          layered.origin.getResult(), windowIndex, reductionAxis, layered.base,
          layered.extent, access,
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
        auto decoded = rewriter.create<riscv::RVVLayeredStorageDecodeOp>(
            dot.getLoc(), layeredWindowType, storage.getResult(), layer,
            riscv_internal::leaf(
                rewriter, "rvv", "layered-storage-decode",
                "rvv.layered-storage-decode", "rvv.layered-storage-decode",
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
          auto window = rewriter.create<riscv::RVVStorageWindowOp>(
              dot.getLoc(), windowType, root.field.getResult(),
              root.origin.getResult(), logicalOffset, reductionAxis, root.base,
              root.stride, root.repeat, root.extent, lanes,
              root.field.getAccess(),
              riscv_internal::leaf(
                  rewriter, "rvv", "storage-window", "rvv.storage-window",
                  "rvv.storage-window",
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

    if (failed)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
weft::createMaterializeRISCVPartialAccumulatorsPass() {
  return std::make_unique<MaterializeRISCVPartialAccumulatorsPass>();
}
