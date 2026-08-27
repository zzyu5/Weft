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

riscv::PhysicalPointOp findOriginPoint(mlir::Value value, int64_t axis) {
  llvm::SmallVector<mlir::Value> worklist{value};
  llvm::DenseSet<mlir::Value> visited;
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
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
    if (auto field = mlir::dyn_cast<riscv::FieldOp>(definition)) {
      worklist.push_back(field.getOwner());
      continue;
    }
    if (auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(definition)) {
      worklist.push_back(conversion.getInput());
      continue;
    }
    if (auto materialize =
            mlir::dyn_cast<riscv::RegisterMaterializeOp>(definition))
      worklist.push_back(materialize.getInput());
  }
  return {};
}

void collectLayeredRoots(mlir::Value value,
                         llvm::DenseSet<mlir::Operation *> &visited,
                         llvm::SmallVectorImpl<riscv::RVVLayeredStreamOp> &roots) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return;
  if (auto stream = mlir::dyn_cast<riscv::RVVLayeredStreamOp>(definition)) {
    roots.push_back(stream);
    return;
  }
  if (!mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                 riscv::NarrowOp, riscv::WidenOp, riscv::ConvertLayoutOp,
                 riscv::RegisterMaterializeOp>(definition))
    return;
  for (mlir::Value operand : definition->getOperands())
    if (mlir::isa<riscv::ValueType>(operand.getType()))
      collectLayeredRoots(operand, visited, roots);
}

void collectFieldRoots(mlir::Value value,
                       llvm::DenseSet<mlir::Operation *> &visited,
                       llvm::SmallVectorImpl<riscv::FieldOp> &roots) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return;
  if (auto field = mlir::dyn_cast<riscv::FieldOp>(definition)) {
    roots.push_back(field);
    return;
  }
  if (!mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                 riscv::NarrowOp, riscv::WidenOp, riscv::ConvertLayoutOp,
                 riscv::RegisterMaterializeOp>(definition))
    return;
  for (mlir::Value operand : definition->getOperands())
    if (mlir::isa<riscv::ValueType>(operand.getType()))
      collectFieldRoots(operand, visited, roots);
}

bool dependsOn(mlir::Value value, mlir::Value root,
               llvm::DenseMap<mlir::Value, bool> &cache) {
  if (value == root)
    return true;
  if (auto found = cache.find(value); found != cache.end())
    return found->second;
  bool dependent = false;
  if (mlir::Operation *definition = value.getDefiningOp())
    if (mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                  riscv::NarrowOp, riscv::WidenOp, riscv::ConvertLayoutOp,
                  riscv::RegisterMaterializeOp>(definition))
      for (mlir::Value operand : definition->getOperands())
        if (mlir::isa<riscv::ValueType>(operand.getType()))
          dependent |= dependsOn(operand, root, cache);
  cache[value] = dependent;
  return dependent;
}

mlir::FailureOr<mlir::Value> cloneWindowSlice(
    mlir::Value value, mlir::Value root, mlir::Value replacement, int64_t axis,
    mlir::IRRewriter &rewriter, llvm::DenseMap<mlir::Value, mlir::Value> &clones,
    llvm::DenseMap<mlir::Value, bool> &dependence) {
  if (value == root)
    return replacement;
  if (!dependsOn(value, root, dependence))
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
    return cloneWindowSlice(operand, root, replacement, axis, rewriter, clones,
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
      riscv_internal::integers(builder, one), 16, partialLMUL,
      lhs.getLayout().getVl(), groups, validity);
  auto element = mlir::IntegerType::get(builder.getContext(), 16,
                                        mlir::IntegerType::Signed);
  return riscv::ValueType::get(builder.getContext(), element,
                               riscv_internal::integers(builder, shape),
                               riscv_internal::integers(builder, axes), layout);
}

void eraseDeadChain(mlir::Value value, mlir::Value stop,
                    llvm::DenseSet<mlir::Operation *> &visited,
                    mlir::IRRewriter &rewriter) {
  if (value == stop)
    return;
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition || !visited.insert(definition).second)
    return;
  llvm::SmallVector<mlir::Value> operands(definition->getOperands());
  if (llvm::all_of(definition->getResults(), [](mlir::Value result) {
        return result.use_empty();
      }) &&
      mlir::isa<riscv::RVVLayeredStreamOp, riscv::UnaryOp, riscv::BinaryOp,
                riscv::CastOp, riscv::NarrowOp, riscv::WidenOp,
                riscv::ConvertLayoutOp, riscv::RegisterMaterializeOp>(definition)) {
    rewriter.eraseOp(definition);
    for (mlir::Value operand : operands)
      eraseDeadChain(operand, stop, visited, rewriter);
  }
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
          axis, extract.getAccess(),
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
      llvm::SmallVector<riscv::RVVLayeredStreamOp> lhsLayered;
      llvm::SmallVector<riscv::RVVLayeredStreamOp> rhsLayered;
      collectLayeredRoots(dot.getLhs(), lhsVisited, lhsLayered);
      collectLayeredRoots(dot.getRhs(), rhsVisited, rhsLayered);
      const bool lhsOwnsLayered = lhsLayered.size() == 1 && rhsLayered.empty();
      const bool rhsOwnsLayered = rhsLayered.size() == 1 && lhsLayered.empty();
      if (!lhsOwnsLayered && !rhsOwnsLayered)
        continue;
      riscv::RVVLayeredStreamOp layered =
          lhsOwnsLayered ? lhsLayered.front() : rhsLayered.front();
      mlir::Value layeredOperand = lhsOwnsLayered ? dot.getLhs() : dot.getRhs();
      mlir::Value otherOperand = lhsOwnsLayered ? dot.getRhs() : dot.getLhs();

      llvm::DenseSet<mlir::Operation *> fieldVisited;
      llvm::SmallVector<riscv::FieldOp> otherFields;
      collectFieldRoots(otherOperand, fieldVisited, otherFields);
      if (otherFields.size() != 1)
        continue;
      riscv::FieldOp otherField = otherFields.front();
      riscv::FieldOp layeredField = layered.getField().getDefiningOp<riscv::FieldOp>();
      if (!layeredField)
        continue;
      riscv::PhysicalPointOp layeredOrigin =
          findOriginPoint(layeredField.getOwner(), reductionAxis);
      riscv::PhysicalPointOp otherOrigin =
          findOriginPoint(otherField.getOwner(), reductionAxis);
      if (!layeredOrigin || !otherOrigin)
        continue;

      riscv::AccessAttr access = layered.getAccess();
      const int64_t group = access.getGroupSize();
      const int64_t layerExtent = access.getLayerSize();
      const int64_t layers = layerExtent > 0 ? group / layerExtent : 0;
      auto layeredInteger = mlir::dyn_cast<mlir::IntegerType>(
          layered.getField().getType().getElementType());
      auto layeredPosition =
          axisPosition(layered.getResult().getType(), reductionAxis);
      if (!layeredPosition)
        continue;
      const int64_t lanes =
          layered.getResult().getType().getLayout().getLaneFactors()[
              *layeredPosition];
      if (!layeredInteger || layeredInteger.isSigned() || group <= 0 ||
          layerExtent <= 0 || group % layerExtent || layers <= 1 ||
          layeredInteger.getWidth() * layers != 8 || access.getBitOffset() % 8 ||
          (access.getOrder() != "lo_first" && access.getOrder() != "hi_first") ||
          layeredOrigin.getResult().getType().getDomain().getTail() != "exact" ||
          otherOrigin.getResult().getType().getDomain().getTail() != "exact" ||
          lanes <= 1 || layerExtent % lanes || streams % layers ||
          streams <= layers)
        continue;
      const int64_t windowsPerLayer = layerExtent / lanes;
      const int64_t windowCount = streams / layers;
      auto layeredWindowType =
          projectOneWindow(rewriter, layered.getResult().getType(), reductionAxis);
      auto otherWindowType =
          projectOneWindow(rewriter,
                           mlir::cast<riscv::ValueType>(
                               otherField.getResult().getType()),
                           reductionAxis);
      auto accumulatorType =
          partialType(rewriter, lhsType, dot.getResult().getType(), reductionAxis);
      if (!layeredWindowType || !otherWindowType || !accumulatorType ||
          mlir::cast<mlir::IntegerType>(accumulatorType.getElementType()).getWidth() !=
              16)
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
          mlir::cast<riscv::ValueType>(layeredField.getResult().getType()),
          layeredWindowType, reductionAxis, layers, windowsPerLayer, rawGroups);
      auto storage = rewriter.create<riscv::RVVLayeredStorageLoadOp>(
          dot.getLoc(), storageType, layeredField.getResult(),
          layeredOrigin.getResult(), windowIndex, reductionAxis, access,
          riscv_internal::leaf(
                rewriter, "rvv", "layered-storage-load",
                "rvv.layered-storage-load", "rvv.layered-storage-load",
                mlir::cast<riscv::ValueType>(layeredField.getResult().getType())
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
        llvm::DenseMap<mlir::Value, mlir::Value> layeredClones;
        llvm::DenseMap<mlir::Value, bool> layeredDependence;
        auto layeredSlice = cloneWindowSlice(
            layeredOperand, layered.getResult(), decoded.getResult(),
            reductionAxis, rewriter, layeredClones, layeredDependence);
        if (mlir::failed(layeredSlice)) {
          failed = true;
          break;
        }

        mlir::Value layerValue = rewriter.create<mlir::arith::ConstantIndexOp>(
            dot.getLoc(), layer * layerExtent);
        mlir::Value layerBase = rewriter.create<mlir::arith::AddIOp>(
            dot.getLoc(), groupBase, layerValue);
        mlir::Value logicalOffset = rewriter.create<mlir::arith::AddIOp>(
            dot.getLoc(), layerBase, withinBase);
        llvm::StringRef otherTail =
            otherWindowType.getLayout().getValidity() == "tail" ? "agnostic"
                                                                 : "exact";
        auto otherWindow = rewriter.create<riscv::RVVStorageWindowOp>(
            dot.getLoc(), otherWindowType, otherField.getResult(),
            otherOrigin.getResult(), logicalOffset, reductionAxis,
            otherField.getAccess(),
            riscv_internal::leaf(
                rewriter, "rvv", "storage-window", "rvv.storage-window",
                "rvv.storage-window", 0,
                otherWindowType.getLayout().getRegisterGroups(), 1, 0, "none",
                otherTail));
        llvm::DenseMap<mlir::Value, mlir::Value> otherClones;
        llvm::DenseMap<mlir::Value, bool> otherDependence;
        auto otherSlice = cloneWindowSlice(
            otherOperand, otherField.getResult(), otherWindow.getResult(),
            reductionAxis, rewriter, otherClones, otherDependence);
        if (mlir::failed(otherSlice)) {
          failed = true;
          break;
        }

        mlir::Value stepLhs = lhsOwnsLayered ? *layeredSlice : *otherSlice;
        mlir::Value stepRhs = lhsOwnsLayered ? *otherSlice : *layeredSlice;
        auto accumulate = rewriter.create<riscv::RVVWidenAccumulateOp>(
            dot.getLoc(), accumulatorType, stepLhs, stepRhs, carried,
            reductionAxis,
            riscv_internal::leaf(
                rewriter, "rvv", "widen-accumulate", "rvv.vwmacc.partial",
                "rvv.vwmacc.partial",
                mlir::cast<riscv::ValueType>(stepLhs.getType())
                        .getLayout()
                        .getRegisterGroups() +
                    mlir::cast<riscv::ValueType>(stepRhs.getType())
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
                               "rvv.vwredsum.partial",
                               "rvv.vwredsum.partial",
                               accumulatorType.getLayout().getRegisterGroups(),
                               0, 2));
      riscv_internal::copyOrigin(dot, finalized);
      mlir::Value oldLhs = dot.getLhs();
      mlir::Value oldRhs = dot.getRhs();
      dot.getResult().replaceAllUsesWith(finalized.getResult());
      rewriter.eraseOp(dot);
      llvm::DenseSet<mlir::Operation *> erased;
      eraseDeadChain(oldLhs, layeredField.getResult(), erased, rewriter);
      eraseDeadChain(oldRhs, otherField.getResult(), erased, rewriter);
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
