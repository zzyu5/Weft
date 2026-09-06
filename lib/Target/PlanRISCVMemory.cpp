#include "Weft/Target/RISCVPasses.h"

#include "RISCVPhysicalSupport.h"

#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>

using namespace weft;

namespace {

using RegularIndex = riscv_internal::RegularIndexRelation;

struct RegularGatherCandidate {
  riscv::ExtractOp extract;
  riscv::ConvertLayoutOp resultConversion;
  llvm::SmallVector<riscv::ConvertLayoutOp> inputConversions;
  riscv::FieldOp field;
  riscv::ValueType resultType;
  mlir::Value index;
  int64_t sourceAxis = 0;
  int64_t reductionAxis = 0;
  int64_t base = 0;
  int64_t stride = 1;
  int64_t repeat = 1;
  int64_t lanes = 0;
  mlir::Value dynamicBase;
  llvm::SmallVector<int64_t> partBases;
};

struct ReplicaAffineIndex {
  int64_t constant = 0;
  int64_t dynamicMultiple = 0;
  llvm::DenseMap<int64_t, int64_t> axisCoefficients;
};

std::optional<ReplicaAffineIndex>
analyzeReplicaAffineIndex(mlir::Value value, unsigned depth = 0);

struct ScalarReplicaGatherPlan {
  int64_t arity = 0;
  llvm::SmallVector<int64_t> candidates;
  llvm::SmallVector<int64_t> keys;
  llvm::SmallVector<int64_t> localBases;
};

struct VectorReplicaGatherPlan {
  int64_t laneCount = 0;
  llvm::SmallVector<int64_t> sourceParts;
  llvm::SmallVector<int64_t> indexParts;
};

struct UnitEntryWindowPlan {
  mlir::Value base;
  llvm::SmallVector<int64_t> entryAxes;
  llvm::SmallVector<int64_t> entryExtents;
};

struct AffineWindowPlan {
  mlir::Value base;
  llvm::SmallVector<int64_t> axes;
  llvm::SmallVector<int64_t> extents;
};

using IndexedEntryLoadPlan = riscv_internal::IndexedEntryRelation;

std::optional<int64_t> positiveProduct(llvm::ArrayRef<int64_t> values) {
  int64_t product = 1;
  for (int64_t value : values) {
    if (value <= 0 || product > std::numeric_limits<int64_t>::max() / value)
      return std::nullopt;
    product *= value;
  }
  return product;
}

riscv::ValueType repeatIndexType(mlir::Builder &builder,
                                 riscv::ValueType value) {
  auto element = mlir::dyn_cast<mlir::IntegerType>(value.getElementType());
  if (!element || element.isSignless())
    return {};
  auto indexElement = mlir::IntegerType::get(
      builder.getContext(), value.getLayout().getSew(),
      mlir::IntegerType::Unsigned);
  return riscv::ValueType::get(builder.getContext(), indexElement,
                               value.getShape(), value.getAxisIds(),
                               value.getLayout());
}

riscv::ValueType repeatedScalarSupplyType(mlir::Builder &builder,
                                          riscv::ValueType value) {
  auto layout = value.getLayout();
  if (!layout || layout.getCarrier() != "rvv" ||
      llvm::any_of(layout.getFragmentFactors().asArrayRef(),
                   [](int64_t factor) { return factor != 1; }) ||
      llvm::any_of(layout.getLocalFactors().asArrayRef(),
                   [](int64_t factor) { return factor != 1; }))
    return {};
  unsigned width = 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(value.getElementType()))
    width = integer.getWidth();
  else if (auto floating =
               mlir::dyn_cast<mlir::FloatType>(value.getElementType()))
    width = floating.getWidth();
  if (!width)
    return {};
  auto scalarLayout = riscv::LayoutAttr::get(
      builder.getContext(), "scalar", value.getAxisIds(),
      layout.getTimeFactors(), layout.getLaneFactors(),
      layout.getReplicaFactors(), layout.getFragmentFactors(),
      layout.getLocalFactors(), std::max<unsigned>(8, width), 0,
      layout.getVl(), 0, layout.getValidity());
  return riscv::ValueType::get(builder.getContext(), value.getElementType(),
                               value.getShape(), value.getAxisIds(),
                               scalarLayout);
}

bool scalarRepeatOperation(mlir::Operation *operation) {
  return mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CompareOp,
                   riscv::CastOp, riscv::NarrowOp, riscv::WidenOp>(operation);
}

struct ScalarRepeatUsePlan {
  llvm::SmallVector<mlir::Operation *> scalarOperations;
  llvm::SmallVector<mlir::Operation *> vectorBoundaries;
};

std::optional<ScalarRepeatUsePlan>
planScalarRepeatUses(mlir::Value root) {
  ScalarRepeatUsePlan plan;
  llvm::DenseSet<mlir::Value> scalarValues;
  llvm::DenseSet<mlir::Operation *> scalarOperations;
  llvm::DenseSet<mlir::Operation *> vectorBoundaries;
  llvm::SmallVector<mlir::Value> worklist{root};
  scalarValues.insert(root);
  for (size_t cursor = 0; cursor < worklist.size(); ++cursor) {
    mlir::Value value = worklist[cursor];
    for (mlir::Operation *user : value.getUsers()) {
      if (!scalarRepeatOperation(user) || user->getNumResults() != 1)
        return std::nullopt;
      auto result = mlir::dyn_cast<riscv::ValueType>(user->getResult(0).getType());
      if (!result)
        return std::nullopt;
      bool hasForeignVector = false;
      for (mlir::Value operand : user->getOperands()) {
        auto operandType = mlir::dyn_cast<riscv::ValueType>(operand.getType());
        if (!operandType || scalarValues.contains(operand) ||
            operandType.getLayout().getCarrier() == "scalar")
          continue;
        hasForeignVector = true;
      }
      if (hasForeignVector) {
        if (!mlir::isa<riscv::BinaryOp, riscv::CompareOp>(user) ||
            result.getLayout().getCarrier() != "rvv")
          return std::nullopt;
        if (vectorBoundaries.insert(user).second)
          plan.vectorBoundaries.push_back(user);
        continue;
      }
      if (scalarOperations.insert(user).second) {
        plan.scalarOperations.push_back(user);
        scalarValues.insert(user->getResult(0));
        worklist.push_back(user->getResult(0));
      }
    }
  }
  if (plan.vectorBoundaries.empty())
    return std::nullopt;
  return plan;
}

bool materializeScalarRepeatUse(const ScalarRepeatUsePlan &plan,
                                mlir::IRRewriter &rewriter) {
  llvm::SmallVector<riscv::ValueType> scalarTypes;
  scalarTypes.reserve(plan.scalarOperations.size());
  for (mlir::Operation *operation : plan.scalarOperations) {
    auto result = mlir::cast<riscv::ValueType>(operation->getResult(0).getType());
    auto scalarType = repeatedScalarSupplyType(rewriter, result);
    if (!scalarType)
      return false;
    scalarTypes.push_back(scalarType);
  }
  for (auto [operation, scalarType] :
       llvm::zip(plan.scalarOperations, scalarTypes)) {
    operation->getResult(0).setType(scalarType);
    operation->setAttr("leaf", riscv_internal::unselectedLeaf(rewriter));
  }
  for (mlir::Operation *operation : plan.vectorBoundaries)
    operation->setAttr("leaf", riscv_internal::unselectedLeaf(rewriter));
  return true;
}

bool canMoveReadBefore(mlir::Operation *operation) {
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

bool checkedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) ||
      (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs))
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedSubtract(int64_t lhs, int64_t rhs, int64_t &result) {
  if ((rhs > 0 && lhs < std::numeric_limits<int64_t>::min() + rhs) ||
      (rhs < 0 && lhs > std::numeric_limits<int64_t>::max() + rhs))
    return false;
  result = lhs - rhs;
  return true;
}

bool checkedScale(int64_t value, int64_t factor, int64_t &result) {
  if (factor <= 0 || value > std::numeric_limits<int64_t>::max() / factor ||
      value < std::numeric_limits<int64_t>::min() / factor)
    return false;
  result = value * factor;
  return true;
}

std::optional<int64_t> constantInteger(mlir::Value value) {
  auto constant = value.getDefiningOp<riscv::ConstantOp>();
  if (constant) {
    auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
    return integer ? std::optional<int64_t>(integer.getInt()) : std::nullopt;
  }
  auto arithmetic = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!arithmetic)
    return std::nullopt;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(arithmetic.getValue());
  return integer ? std::optional<int64_t>(integer.getInt()) : std::nullopt;
}

mlir::FailureOr<mlir::Value>
materializeEntryIndices(const IndexedEntryLoadPlan &plan,
                        mlir::Operation *origin, riscv::ValueType result,
                        mlir::IRRewriter &rewriter) {
  auto type = mlir::dyn_cast<riscv::ValueType>(plan.entryIndices.getType());
  mlir::Value entryIndices = plan.entryIndices;
  mlir::Type element =
      type ? type.getElementType() : plan.entryIndices.getType();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
  if (!element.isIndex() && (!integer || integer.isSigned()))
    return origin->emitError(
               "indexed entry relation has no unsigned scalar or shaped index type"),
           mlir::failure();
  if (plan.indexDivisor != 1) {
    mlir::TypedAttr divisor;
    if (integer)
      divisor = rewriter.getIntegerAttr(integer, plan.indexDivisor);
    else if (element.isIndex())
      divisor = rewriter.getIndexAttr(plan.indexDivisor);
    if (!divisor)
      return origin->emitError(
                 "indexed entry relation has no integer divisor type"),
             mlir::failure();
    auto constant = rewriter.create<riscv::ConstantOp>(
        origin->getLoc(), element, divisor);
    auto divided = rewriter.create<riscv::BinaryOp>(
        origin->getLoc(), entryIndices.getType(), entryIndices,
        constant.getResult(), "div",
        riscv_internal::unselectedLeaf(rewriter));
    riscv_internal::copyOrigin(origin, divided);
    entryIndices = divided.getResult();
  }

  if (!type)
    return entryIndices;

  auto kernel = origin->getParentOfType<riscv::KernelOp>();
  auto required = riscv_internal::projectLayout(
      rewriter, type, result.getLayout(), kernel.getTarget());
  if (!required)
    return origin->emitError(
               "indexed entry relation has no legal entry-axis projection"),
           mlir::failure();
  mlir::Type requiredType = riscv_internal::withLayout(type, required);
  if (entryIndices.getType() == requiredType)
    return entryIndices;
  auto conversion = rewriter.create<riscv::ConvertLayoutOp>(
      origin->getLoc(), requiredType, entryIndices,
      riscv_internal::layoutConversion(rewriter, type.getLayout(), required),
      riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
  riscv_internal::copyOrigin(origin, conversion);
  return conversion.getResult();
}

mlir::FailureOr<mlir::Value>
materializeEntryByteOffsets(mlir::Value entryIndices, int64_t entryStride,
                            mlir::Operation *origin,
                            mlir::IRRewriter &rewriter) {
  auto type = mlir::dyn_cast<riscv::ValueType>(entryIndices.getType());
  mlir::Type element = type ? type.getElementType() : entryIndices.getType();
  auto integer = mlir::dyn_cast<mlir::IntegerType>(element);
  if (entryStride <= 0 ||
      (!element.isIndex() && (!integer || integer.isSigned())))
    return origin->emitError(
               "indexed entry relation has no positive typed byte stride"),
           mlir::failure();
  if (entryStride == 1)
    return entryIndices;
  const bool powerOfTwo = (entryStride & (entryStride - 1)) == 0;
  int64_t immediate = entryStride;
  if (powerOfTwo) {
    immediate = 0;
    for (int64_t value = entryStride; value > 1; value >>= 1)
      ++immediate;
  }
  mlir::TypedAttr stride;
  if (integer) {
    if (integer.getWidth() < 63) {
      const int64_t maximum = (int64_t{1} << integer.getWidth()) - 1;
      if (entryStride > maximum ||
          (powerOfTwo && immediate >= integer.getWidth()))
        return origin->emitError(
                   "indexed entry byte stride is not representable by its selected unsigned carrier"),
               mlir::failure();
    }
    stride = rewriter.getIntegerAttr(integer, immediate);
  } else if (element.isIndex())
    stride = rewriter.getIndexAttr(immediate);
  if (!stride)
    return origin->emitError(
               "indexed entry relation has no integer byte-offset type"),
           mlir::failure();
  auto constant = rewriter.create<riscv::ConstantOp>(
      origin->getLoc(), element, stride);
  auto offsets = rewriter.create<riscv::BinaryOp>(
      origin->getLoc(), entryIndices.getType(), entryIndices,
      constant.getResult(),
      powerOfTwo ? "shl" : "mul",
      riscv_internal::unselectedLeaf(rewriter));
  riscv_internal::copyOrigin(origin, offsets);
  return offsets.getResult();
}

std::optional<int64_t> entryByteStride(const IndexedEntryLoadPlan &plan,
                                       riscv::ValueType result) {
  const unsigned bits =
      riscv_internal::logicalBitWidth(result.getElementType());
  int64_t bytes = 0;
  if (!bits || bits % 8 ||
      !checkedScale(plan.entryStride, static_cast<int64_t>(bits / 8), bytes))
    return std::nullopt;
  return bytes;
}

bool supportsIndexedEntryLoad(const IndexedEntryLoadPlan &plan,
                              mlir::Type entryIndices,
                              riscv::ValueType result, int64_t alignment,
                              int64_t bitOffset) {
  if (result.getLayout().getCarrier() != "rvv")
    return false;
  auto entryType = mlir::dyn_cast<riscv::ValueType>(entryIndices);
  mlir::Type indexElement =
      entryType ? entryType.getElementType() : entryIndices;
  auto index = mlir::dyn_cast<mlir::IntegerType>(indexElement);
  if (!indexElement.isIndex() && (!index || index.isSigned()))
    return false;
  const unsigned payloadBits =
      riscv_internal::logicalBitWidth(result.getElementType());
  // Indexed-entry loads address complete storage elements.  A logical
  // sub-byte payload has no exact byte stride here and must stay with the
  // grouped/layered or joined storage owner instead of entering this path.
  if (!payloadBits || payloadBits % 8)
    return false;
  if (bitOffset % 8)
    return false;
  const uint64_t packedBits =
      payloadBits * static_cast<uint64_t>(plan.payloadExtent);
  auto resultAxes = result.getAxisIds().asArrayRef();
  auto payload = llvm::find(resultAxes, plan.payloadAxis);
  if (payload == resultAxes.end())
    return false;
  const size_t payloadPosition =
      static_cast<size_t>(payload - resultAxes.begin());
  auto layout = result.getLayout();
  const bool completePayloadLane =
      plan.payloadExtent > 0 &&
      layout.getTimeFactors()[payloadPosition] == 1 &&
      layout.getLaneFactors()[payloadPosition] == plan.payloadExtent &&
      layout.getReplicaFactors()[payloadPosition] == 1 &&
      layout.getFragmentFactors()[payloadPosition] == 1 &&
      layout.getLocalFactors()[payloadPosition] == 1;
  if (!completePayloadLane)
    return false;
  if (!entryType || entryType.getLayout().getCarrier() == "scalar")
    return true;
  if (entryType.getLayout().getCarrier() != "rvv")
    return false;
  return index && !index.isSigned() &&
         (index.getWidth() == 16 || index.getWidth() == 32 ||
          index.getWidth() == 64) &&
         payloadBits > 0 && plan.payloadExtent > 0 &&
         (packedBits == 32 || packedBits == 64) &&
         plan.entryStride == plan.payloadExtent &&
         bitOffset % packedBits == 0 &&
         alignment >= static_cast<int64_t>(packedBits / 8);
}

bool isSingleScalarCoordinate(mlir::Value value) {
  if (mlir::isa<mlir::IntegerType, mlir::IndexType>(value.getType()))
    return true;
  auto physical = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!physical)
    return false;
  auto layout = physical.getLayout();
  auto time = riscv_internal::staticProduct(layout.getTimeFactors().asArrayRef());
  auto lanes = riscv_internal::staticProduct(layout.getLaneFactors().asArrayRef());
  auto replicas =
      riscv_internal::staticProduct(layout.getReplicaFactors().asArrayRef());
  auto fragments =
      riscv_internal::staticProduct(layout.getFragmentFactors().asArrayRef());
  auto local = riscv_internal::staticProduct(layout.getLocalFactors().asArrayRef());
  return layout.getCarrier() == "scalar" && time && *time == 1 && lanes &&
         *lanes == 1 && replicas && *replicas == 1 && fragments &&
         *fragments == 1 && local && *local == 1;
}

mlir::Value stripEntryIndexConversions(mlir::Value value) {
  while (true) {
    if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>()) {
      value = conversion.getInput();
      continue;
    }
    if (auto cast = value.getDefiningOp<riscv::CastOp>()) {
      value = cast.getInput();
      continue;
    }
    if (auto narrow = value.getDefiningOp<riscv::NarrowOp>()) {
      value = narrow.getInput();
      continue;
    }
    if (auto widen = value.getDefiningOp<riscv::WidenOp>()) {
      value = widen.getInput();
      continue;
    }
    if (auto broadcast = value.getDefiningOp<riscv::RVVAxisBroadcastOp>()) {
      value = broadcast.getInput();
      continue;
    }
    return value;
  }
}

mlir::FailureOr<mlir::Value>
materializeScalarAffineComponent(mlir::Value value, mlir::Operation *origin,
                                 mlir::IRRewriter &rewriter,
                                 unsigned depth = 0) {
  if (depth > 24)
    return mlir::failure();
  if (isSingleScalarCoordinate(value))
    return value;

  auto physical = mlir::dyn_cast<riscv::ValueType>(value.getType());
  if (!physical)
    return mlir::failure();
  mlir::Type scalarType = physical.getElementType();
  auto makeConstant = [&](int64_t integer) -> mlir::FailureOr<mlir::Value> {
    mlir::TypedAttr attribute;
    if (auto type = mlir::dyn_cast<mlir::IntegerType>(scalarType))
      attribute = rewriter.getIntegerAttr(type, integer);
    else if (scalarType.isIndex())
      attribute = rewriter.getIndexAttr(integer);
    if (!attribute)
      return mlir::failure();
    auto constant = rewriter.create<riscv::ConstantOp>(origin->getLoc(),
                                                        scalarType, attribute);
    riscv_internal::copyOrigin(origin, constant);
    return constant.getResult();
  };
  auto recurse = [&](mlir::Value operand) {
    return materializeScalarAffineComponent(operand, origin, rewriter,
                                            depth + 1);
  };

  if (auto constant = constantInteger(value))
    return makeConstant(*constant);
  if (auto iota = value.getDefiningOp<riscv::IotaOp>())
    return makeConstant(iota.getStart());
  if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>())
    return recurse(conversion.getInput());
  if (auto broadcast = value.getDefiningOp<riscv::RVVAxisBroadcastOp>())
    return recurse(broadcast.getInput());
  if (auto materialize = value.getDefiningOp<riscv::RegisterMaterializeOp>())
    return recurse(materialize.getInput());

  if (auto cast = value.getDefiningOp<riscv::CastOp>()) {
    auto input = recurse(cast.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    if ((*input).getType() == scalarType)
      return *input;
    auto scalar = rewriter.create<riscv::CastOp>(
        origin->getLoc(), scalarType, *input,
        riscv_internal::unselectedLeaf(rewriter));
    riscv_internal::copyOrigin(origin, scalar);
    return scalar.getResult();
  }
  if (auto widen = value.getDefiningOp<riscv::WidenOp>()) {
    auto input = recurse(widen.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    if ((*input).getType() == scalarType)
      return *input;
    auto scalar = rewriter.create<riscv::WidenOp>(
        origin->getLoc(), scalarType, *input,
        riscv_internal::unselectedLeaf(rewriter));
    riscv_internal::copyOrigin(origin, scalar);
    return scalar.getResult();
  }
  if (auto narrow = value.getDefiningOp<riscv::NarrowOp>()) {
    auto input = recurse(narrow.getInput());
    if (mlir::failed(input))
      return mlir::failure();
    if ((*input).getType() == scalarType)
      return *input;
    auto scalar = rewriter.create<riscv::NarrowOp>(
        origin->getLoc(), scalarType, *input, narrow.getRounding(),
        narrow.getSaturate(), riscv_internal::unselectedLeaf(rewriter));
    riscv_internal::copyOrigin(origin, scalar);
    return scalar.getResult();
  }
  if (auto binary = value.getDefiningOp<riscv::BinaryOp>()) {
    if (binary.getKind() != "add" && binary.getKind() != "sub" &&
        binary.getKind() != "mul")
      return mlir::failure();
    auto lhs = recurse(binary.getLhs());
    auto rhs = recurse(binary.getRhs());
    if (mlir::failed(lhs) || mlir::failed(rhs))
      return mlir::failure();
    auto scalar = rewriter.create<riscv::BinaryOp>(
        origin->getLoc(), scalarType, *lhs, *rhs, binary.getKind(),
        riscv_internal::unselectedLeaf(rewriter));
    riscv_internal::copyOrigin(origin, scalar);
    return scalar.getResult();
  }
  return mlir::failure();
}

std::optional<UnitEntryWindowPlan>
materializeUnitEntryWindowBase(mlir::Value entryIndices,
                               riscv::ValueType result, int64_t payloadAxis,
                               int64_t payloadExtent, int64_t entryStride,
                               mlir::Operation *origin,
                               mlir::IRRewriter &rewriter) {
  auto entryType = mlir::dyn_cast<riscv::ValueType>(entryIndices.getType());
  auto resultAxes = result.getAxisIds().asArrayRef();
  auto resultShape = result.getShape().asArrayRef();
  if (!entryType || entryType.getAxisIds().empty() ||
      entryType.getAxisIds().size() != entryType.getShape().size() ||
      llvm::any_of(entryType.getShape().asArrayRef(),
                   [](int64_t extent) { return extent <= 0; }) ||
      entryStride != payloadExtent ||
      resultAxes.size() < entryType.getAxisIds().size() + 1 ||
      resultAxes.back() != payloadAxis || resultShape.back() != payloadExtent)
    return std::nullopt;
  const size_t entryBegin =
      resultAxes.size() - entryType.getAxisIds().size() - 1;
  if (!llvm::equal(resultAxes.slice(entryBegin, entryType.getAxisIds().size()),
                   entryType.getAxisIds().asArrayRef()) ||
      !llvm::equal(resultShape.slice(entryBegin, entryType.getShape().size()),
                   entryType.getShape().asArrayRef()))
    return std::nullopt;

  mlir::Value expression =
      riscv_internal::stripRepresentationConversions(entryIndices);
  int64_t divisor = 1;
  if (auto division = expression.getDefiningOp<riscv::BinaryOp>();
      division && division.getKind() == "div") {
    auto constant = constantInteger(division.getRhs());
    if (!constant || *constant <= 0)
      return std::nullopt;
    divisor = *constant;
    expression = division.getLhs();
  }
  auto affine = analyzeReplicaAffineIndex(expression);
  auto scalarBase =
      materializeScalarAffineComponent(expression, origin, rewriter);
  if (!affine || mlir::failed(scalarBase) ||
      affine->axisCoefficients.size() != entryType.getAxisIds().size() ||
      !riscv_internal::knownMultipleOf(*scalarBase, divisor))
    return std::nullopt;
  int64_t expectedCoefficient = divisor;
  for (int64_t position =
           static_cast<int64_t>(entryType.getAxisIds().size()) - 1;
       position >= 0; --position) {
    const size_t ordinal = static_cast<size_t>(position);
    if (affine->axisCoefficients.lookup(entryType.getAxisIds()[ordinal]) !=
        expectedCoefficient)
      return std::nullopt;
    if (!checkedScale(expectedCoefficient, entryType.getShape()[ordinal],
                      expectedCoefficient))
      return std::nullopt;
  }
  if (divisor != 1) {
    mlir::Type type = (*scalarBase).getType();
    mlir::TypedAttr divisorValue;
    if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
      divisorValue = rewriter.getIntegerAttr(integer, divisor);
    else if (type.isIndex())
      divisorValue = rewriter.getIndexAttr(divisor);
    if (!divisorValue)
      return std::nullopt;
    auto constant = rewriter.create<riscv::ConstantOp>(
        origin->getLoc(), type, divisorValue);
    auto divided = rewriter.create<riscv::BinaryOp>(
        origin->getLoc(), type, *scalarBase, constant.getResult(), "div",
        riscv_internal::unselectedLeaf(rewriter));
    riscv_internal::copyOrigin(origin, divided);
    scalarBase = divided.getResult();
  }
  return UnitEntryWindowPlan{
      *scalarBase,
      llvm::SmallVector<int64_t>(entryType.getAxisIds().asArrayRef().begin(),
                                 entryType.getAxisIds().asArrayRef().end()),
      llvm::SmallVector<int64_t>(entryType.getShape().asArrayRef().begin(),
                                 entryType.getShape().asArrayRef().end())};
}

bool supportsUnitEntryWindowLayout(riscv::ValueType result,
                                   size_t retainedAxes,
                                   llvm::ArrayRef<int64_t> entryAxes,
                                   llvm::ArrayRef<int64_t> entryExtents,
                                   int64_t payloadAxis,
                                   int64_t payloadExtent) {
  auto axes = result.getAxisIds().asArrayRef();
  auto shape = result.getShape().asArrayRef();
  const unsigned elementBits =
      riscv_internal::logicalBitWidth(result.getElementType());
  if (!elementBits || elementBits % 8 ||
      result.getLayout().getCarrier() != "rvv" ||
      entryAxes.empty() || entryAxes.size() != entryExtents.size() ||
      axes.size() != retainedAxes + entryAxes.size() + 1 ||
      shape.size() != axes.size() ||
      !llvm::equal(axes.slice(retainedAxes, entryAxes.size()), entryAxes) ||
      !llvm::equal(shape.slice(retainedAxes, entryExtents.size()),
                   entryExtents) ||
      axes.back() != payloadAxis || shape.back() != payloadExtent)
    return false;
  auto layout = result.getLayout();
  for (size_t position = 0; position < retainedAxes; ++position)
    if (layout.getTimeFactors()[position] != 1 ||
        layout.getLaneFactors()[position] != 1 ||
        layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1)
      return false;
  llvm::SmallVector<int64_t> extents(entryExtents.begin(), entryExtents.end());
  extents.push_back(payloadExtent);
  bool striped = false;
  int64_t precedingLaneSpan = 1;
  for (size_t offset = 0; offset < extents.size(); ++offset) {
    const size_t position = retainedAxes + offset;
    const int64_t time = layout.getTimeFactors()[position];
    const int64_t lane = layout.getLaneFactors()[position];
    if (time <= 0 || lane <= 0 || layout.getReplicaFactors()[position] != 1 ||
        layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1)
      return false;
    if (time == 1) {
      if (lane != extents[offset])
        return false;
    } else {
      // One outer entry coordinate may be striped through issue time.  It must
      // be the primary lane coordinate; every inner entry/payload coordinate
      // remains fully contiguous inside the same unit load.
      if (striped || precedingLaneSpan != 1 || lane <= 1 ||
          time > std::numeric_limits<int64_t>::max() / lane ||
          time * lane != extents[offset])
        return false;
      striped = true;
    }
    if (precedingLaneSpan > std::numeric_limits<int64_t>::max() / lane)
      return false;
    precedingLaneSpan *= lane;
  }
  return true;
}

bool supportsBitmaskWindowLayout(riscv::FieldOp field,
                                 riscv::ValueType result,
                                 size_t retainedAxes,
                                 llvm::ArrayRef<int64_t> windowAxes,
                                 llvm::ArrayRef<int64_t> windowExtents) {
  auto element = mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
  auto facts = field ? riscv_internal::fieldFacts(field)
                     : riscv_internal::FieldFacts();
  auto axes = result.getAxisIds().asArrayRef();
  auto shape = result.getShape().asArrayRef();
  auto elements = positiveProduct(windowExtents);
  if (!field || !element || element.isSigned() || element.getWidth() != 1 ||
      result.getLayout().getCarrier() != "rvv" || windowAxes.empty() ||
      windowAxes.size() != windowExtents.size() || !elements ||
      facts.mapping != "grouped_layered" || facts.group <= 0 ||
      facts.layer <= 0 || facts.group != facts.layer * 8 ||
      facts.order != "lo_first" || facts.bitOffset % 8 ||
      *elements % facts.group ||
      axes.size() != retainedAxes + windowAxes.size() ||
      shape.size() != axes.size() ||
      !llvm::equal(axes.drop_front(retainedAxes), windowAxes) ||
      !llvm::equal(shape.drop_front(retainedAxes), windowExtents))
    return false;
  auto layout = result.getLayout();
  for (size_t position = 0; position < retainedAxes; ++position)
    if (layout.getTimeFactors()[position] != 1 ||
        layout.getLaneFactors()[position] != 1 ||
        layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1)
      return false;
  for (size_t offset = 0; offset < windowExtents.size(); ++offset) {
    const size_t position = retainedAxes + offset;
    const int64_t time = layout.getTimeFactors()[position];
    const int64_t lane = layout.getLaneFactors()[position];
    const int64_t replica = layout.getReplicaFactors()[position];
    if (time <= 0 || lane <= 0 || replica <= 0 ||
        time > std::numeric_limits<int64_t>::max() / lane ||
        time * lane > std::numeric_limits<int64_t>::max() / replica ||
        time * lane * replica != windowExtents[offset] ||
        layout.getFragmentFactors()[position] != 1 ||
        layout.getLocalFactors()[position] != 1)
      return false;
  }
  return true;
}

std::optional<riscv::ValueType>
byteAlignedBitmaskLoadType(riscv::ValueType result,
                           llvm::ArrayRef<int64_t> windowAxes,
                           llvm::ArrayRef<int64_t> windowExtents,
                           const riscv_internal::FieldFacts &facts,
                           riscv::TargetAttr target, mlir::Builder &builder) {
  if (!target || windowAxes.empty() ||
      windowAxes.size() != windowExtents.size() || facts.group <= 0)
    return std::nullopt;

  auto byteAligned = [&](riscv::ValueType candidate) {
    auto offsets = riscv::bitmaskWindowPartOffsets(
        candidate, windowAxes, windowExtents);
    return offsets && llvm::all_of(*offsets, [](int64_t offset) {
             return offset >= 0 && offset % 8 == 0;
           });
  };
  if (byteAligned(result))
    return result;

  llvm::SmallVector<size_t> positions;
  for (auto [axis, extent] : llvm::zip(windowAxes, windowExtents)) {
    auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
    if (extent <= 0 || found == result.getAxisIds().asArrayRef().end())
      return std::nullopt;
    const size_t position = static_cast<size_t>(
        found - result.getAxisIds().asArrayRef().begin());
    if (result.getShape()[position] != extent)
      return std::nullopt;
    positions.push_back(position);
  }

  llvm::SmallVector<int64_t> time(
      result.getLayout().getTimeFactors().asArrayRef());
  llvm::SmallVector<int64_t> lane(
      result.getLayout().getLaneFactors().asArrayRef());
  auto replicas = positiveProduct(
      result.getLayout().getReplicaFactors().asArrayRef());
  const int64_t sew = result.getLayout().getSew();
  if (!replicas || sew <= 0 || target.getVlenBits() <= 0)
    return std::nullopt;

  auto build = [&]() -> std::optional<riscv::ValueType> {
    auto lanes = positiveProduct(lane);
    int64_t laneBits = 0;
    int64_t lmulNumerator = 0;
    if (!lanes || !checkedScale(*lanes, sew, laneBits) ||
        !checkedScale(laneBits, 8, lmulNumerator) ||
        lmulNumerator % target.getVlenBits())
      return std::nullopt;
    const int64_t lmul = lmulNumerator / target.getVlenBits();
    if (!llvm::is_contained(target.getLegalLMULEighths().asArrayRef(), lmul))
      return std::nullopt;
    int64_t groups = 0;
    if (!checkedScale(std::max<int64_t>(1, (lmul + 7) / 8), *replicas,
                      groups))
      return std::nullopt;
    auto layout = riscv::LayoutAttr::get(
        builder.getContext(), "rvv", result.getAxisIds(),
        builder.getDenseI64ArrayAttr(time), builder.getDenseI64ArrayAttr(lane),
        result.getLayout().getReplicaFactors(),
        result.getLayout().getFragmentFactors(),
        result.getLayout().getLocalFactors(), sew, lmul, *lanes, groups,
        result.getLayout().getValidity());
    if (!riscv::supportsRVVLayout(target, layout))
      return std::nullopt;
    return riscv::ValueType::get(builder.getContext(), result.getElementType(),
                                result.getShape(), result.getAxisIds(), layout);
  };

  // A unit mask load requires its lane coordinates to be one contiguous
  // interval in logical storage order. Move the consumer's lane width onto the
  // innermost window suffix, trying the original width before narrower divisors.
  // This exchanges lane and issue-time ownership without increasing
  // simultaneous register pressure. Only if no non-widening relation exists do
  // the candidates below absorb additional issue-time coordinates.
  auto tryContiguousLaneWidth = [&](int64_t requestedLanes)
      -> std::optional<riscv::ValueType> {
    if (requestedLanes <= 0)
      return std::nullopt;
    llvm::SmallVector<int64_t> contiguousTime(time);
    llvm::SmallVector<int64_t> contiguousLane(lane);
    int64_t remainingLanes = requestedLanes;
    bool representable = true;
    for (int64_t ordinal = static_cast<int64_t>(positions.size()) - 1;
         ordinal >= 0; --ordinal) {
      const size_t position = positions[static_cast<size_t>(ordinal)];
      const int64_t extent = windowExtents[static_cast<size_t>(ordinal)];
      const int64_t replica = result.getLayout().getReplicaFactors()[position];
      if (replica <= 0 || extent <= 0 || extent % replica) {
        representable = false;
        break;
      }
      const int64_t available = extent / replica;
      int64_t selectedLane = 1;
      if (remainingLanes > 1) {
        if (remainingLanes >= available) {
          if (remainingLanes % available) {
            representable = false;
            break;
          }
          selectedLane = available;
          remainingLanes /= available;
        } else {
          if (available % remainingLanes) {
            representable = false;
            break;
          }
          selectedLane = remainingLanes;
          remainingLanes = 1;
        }
      }
      contiguousLane[position] = selectedLane;
      contiguousTime[position] = available / selectedLane;
    }
    representable &= remainingLanes == 1;
    if (!representable)
      return std::nullopt;
    std::swap(time, contiguousTime);
    std::swap(lane, contiguousLane);
    auto candidate = build();
    std::swap(time, contiguousTime);
    std::swap(lane, contiguousLane);
    return candidate && byteAligned(*candidate) ? candidate : std::nullopt;
  };
  auto originalLanes = positiveProduct(lane);
  if (originalLanes) {
    // Prefer the widest contiguous divisor that does not exceed the consumer's
    // lane carrier. Narrowing creates more issue parts but never increases the
    // live RVV group width; widening is considered only by the fallback below.
    for (int64_t candidateLanes = *originalLanes; candidateLanes >= 1;
         --candidateLanes) {
      if (*originalLanes % candidateLanes)
        continue;
      if (auto candidate = tryContiguousLaneWidth(candidateLanes))
        return candidate;
    }
  }

  // A bitmask instruction starts at a byte address.  Absorb only as much of
  // the innermost issue-time decomposition into lanes as is necessary to make
  // every physical part begin on a byte boundary.  This is a storage-geometry
  // legality rule; the consumer layout remains connected by an explicit
  // conversion below.
  for (size_t position : llvm::reverse(positions)) {
    const int64_t originalTime = time[position];
    if (originalTime <= 1 || lane[position] <= 0)
      continue;
    for (int64_t factor = 2; factor <= originalTime; ++factor) {
      if (originalTime % factor)
        continue;
      llvm::SmallVector<int64_t> candidateTime(time);
      llvm::SmallVector<int64_t> candidateLane(lane);
      if (!checkedScale(candidateLane[position], factor,
                        candidateLane[position]))
        continue;
      candidateTime[position] /= factor;
      std::swap(time, candidateTime);
      std::swap(lane, candidateLane);
      auto candidate = build();
      std::swap(time, candidateTime);
      std::swap(lane, candidateLane);
      if (candidate && byteAligned(*candidate))
        return candidate;
    }
    if (!checkedScale(lane[position], originalTime, lane[position]))
      return std::nullopt;
    time[position] = 1;
  }
  auto candidate = build();
  return candidate && byteAligned(*candidate) ? candidate : std::nullopt;
}

std::optional<AffineWindowPlan>
materializeAffineWindowBase(mlir::Value index, riscv::ValueType result,
                            size_t retainedAxes, int64_t baseDivisor,
                            int64_t baseMultiplier, mlir::Operation *origin,
                            mlir::IRRewriter &rewriter) {
  auto indexType = mlir::dyn_cast<riscv::ValueType>(index.getType());
  auto resultAxes = result.getAxisIds().asArrayRef();
  auto resultShape = result.getShape().asArrayRef();
  if (!indexType || indexType.getAxisIds().empty() || baseDivisor <= 0 ||
      baseMultiplier <= 0 ||
      indexType.getAxisIds().size() != indexType.getShape().size() ||
      resultAxes.size() != retainedAxes + indexType.getAxisIds().size() ||
      !llvm::equal(resultAxes.drop_front(retainedAxes),
                   indexType.getAxisIds().asArrayRef()) ||
      !llvm::equal(resultShape.drop_front(retainedAxes),
                   indexType.getShape().asArrayRef()))
    return std::nullopt;

  mlir::Value expression = stripEntryIndexConversions(index);
  auto affine = analyzeReplicaAffineIndex(expression);
  if (!affine ||
      affine->axisCoefficients.size() != indexType.getAxisIds().size())
    return std::nullopt;
  int64_t coefficient = 1;
  for (int64_t position =
           static_cast<int64_t>(indexType.getAxisIds().size()) - 1;
       position >= 0; --position) {
    const size_t ordinal = static_cast<size_t>(position);
    if (affine->axisCoefficients.lookup(indexType.getAxisIds()[ordinal]) !=
        coefficient)
      return std::nullopt;
    if (!checkedScale(coefficient, indexType.getShape()[ordinal], coefficient))
      return std::nullopt;
  }

  auto scalarBase =
      materializeScalarAffineComponent(expression, origin, rewriter);
  if (mlir::failed(scalarBase) ||
      !riscv_internal::knownMultipleOf(*scalarBase, baseDivisor))
    return std::nullopt;
  if (baseDivisor != 1) {
    mlir::TypedAttr divisor;
    if (auto integer =
            mlir::dyn_cast<mlir::IntegerType>((*scalarBase).getType()))
      divisor = rewriter.getIntegerAttr(integer, baseDivisor);
    else if ((*scalarBase).getType().isIndex())
      divisor = rewriter.getIndexAttr(baseDivisor);
    if (!divisor)
      return std::nullopt;
    auto constant = rewriter.create<riscv::ConstantOp>(
        origin->getLoc(), (*scalarBase).getType(), divisor);
    auto divided = rewriter.create<riscv::BinaryOp>(
        origin->getLoc(), (*scalarBase).getType(), *scalarBase,
        constant.getResult(), "div", riscv_internal::unselectedLeaf(rewriter));
    riscv_internal::copyOrigin(origin, divided);
    scalarBase = divided.getResult();
  }
  if (baseMultiplier != 1) {
    mlir::TypedAttr multiplier;
    if (auto integer =
            mlir::dyn_cast<mlir::IntegerType>((*scalarBase).getType()))
      multiplier = rewriter.getIntegerAttr(integer, baseMultiplier);
    else if ((*scalarBase).getType().isIndex())
      multiplier = rewriter.getIndexAttr(baseMultiplier);
    if (!multiplier)
      return std::nullopt;
    auto constant = rewriter.create<riscv::ConstantOp>(
        origin->getLoc(), (*scalarBase).getType(), multiplier);
    auto multiplied = rewriter.create<riscv::BinaryOp>(
        origin->getLoc(), (*scalarBase).getType(), *scalarBase,
        constant.getResult(), "mul", riscv_internal::unselectedLeaf(rewriter));
    riscv_internal::copyOrigin(origin, multiplied);
    scalarBase = multiplied.getResult();
  }
  auto indexAxes = indexType.getAxisIds().asArrayRef();
  auto indexShape = indexType.getShape().asArrayRef();
  return AffineWindowPlan{
      *scalarBase,
      llvm::SmallVector<int64_t>(indexAxes.begin(), indexAxes.end()),
      llvm::SmallVector<int64_t>(indexShape.begin(), indexShape.end())};
}

std::optional<UnitEntryWindowPlan>
materializeAffineUnitWindowBase(mlir::Value index, riscv::ValueType result,
                                size_t retainedAxes, mlir::Operation *origin,
                                mlir::IRRewriter &rewriter) {
  auto indexType = mlir::dyn_cast<riscv::ValueType>(index.getType());
  if (!indexType || indexType.getShape().size() < 2)
    return std::nullopt;
  const int64_t payloadExtent = indexType.getShape().asArrayRef().back();
  auto window = materializeAffineWindowBase(
      index, result, retainedAxes, payloadExtent, 1, origin, rewriter);
  if (!window)
    return std::nullopt;
  llvm::ArrayRef<int64_t> axes(window->axes);
  llvm::ArrayRef<int64_t> extents(window->extents);
  return UnitEntryWindowPlan{
      window->base,
      llvm::SmallVector<int64_t>(axes.drop_back().begin(),
                                 axes.drop_back().end()),
      llvm::SmallVector<int64_t>(extents.drop_back().begin(),
                                 extents.drop_back().end())};
}

bool combineReplicaAffine(ReplicaAffineIndex &result,
                          const ReplicaAffineIndex &other, bool subtract) {
  if (subtract ? !checkedSubtract(result.constant, other.constant,
                                  result.constant)
               : !checkedAdd(result.constant, other.constant, result.constant))
    return false;
  for (const auto &[axis, coefficient] : other.axisCoefficients) {
    int64_t updated = result.axisCoefficients.lookup(axis);
    if (subtract ? !checkedSubtract(updated, coefficient, updated)
                 : !checkedAdd(updated, coefficient, updated))
      return false;
    if (updated)
      result.axisCoefficients[axis] = updated;
    else
      result.axisCoefficients.erase(axis);
  }
  if (other.dynamicMultiple) {
    if (!result.dynamicMultiple)
      result.dynamicMultiple = other.dynamicMultiple;
    else
      result.dynamicMultiple =
          std::gcd(result.dynamicMultiple, other.dynamicMultiple);
  }
  return true;
}

std::optional<ReplicaAffineIndex>
analyzeReplicaAffineIndex(mlir::Value value, unsigned depth) {
  if (depth > 24)
    return std::nullopt;
  if (auto conversion = value.getDefiningOp<riscv::ConvertLayoutOp>())
    return analyzeReplicaAffineIndex(conversion.getInput(), depth + 1);
  if (auto cast = value.getDefiningOp<riscv::CastOp>())
    return analyzeReplicaAffineIndex(cast.getInput(), depth + 1);
  if (auto widen = value.getDefiningOp<riscv::WidenOp>())
    return analyzeReplicaAffineIndex(widen.getInput(), depth + 1);
  if (auto broadcast = value.getDefiningOp<riscv::RVVAxisBroadcastOp>())
    return analyzeReplicaAffineIndex(broadcast.getInput(), depth + 1);
  if (auto constant = constantInteger(value)) {
    ReplicaAffineIndex result;
    result.constant = *constant;
    return result;
  }
  if (auto iota = value.getDefiningOp<riscv::IotaOp>()) {
    auto type = mlir::dyn_cast<riscv::ValueType>(iota.getResult().getType());
    if (!type || type.getAxisIds().size() != 1 ||
        iota.getStart() >
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
      return std::nullopt;
    ReplicaAffineIndex result;
    result.constant = static_cast<int64_t>(iota.getStart());
    result.axisCoefficients[type.getAxisIds()[0]] = 1;
    return result;
  }
  if (auto binary = value.getDefiningOp<riscv::BinaryOp>()) {
    if (binary.getKind() == "add" || binary.getKind() == "sub") {
      auto lhs = analyzeReplicaAffineIndex(binary.getLhs(), depth + 1);
      auto rhs = analyzeReplicaAffineIndex(binary.getRhs(), depth + 1);
      if (!lhs || !rhs ||
          !combineReplicaAffine(*lhs, *rhs, binary.getKind() == "sub"))
        return std::nullopt;
      return lhs;
    }
    if (binary.getKind() == "mul") {
      mlir::Value source;
      std::optional<int64_t> factor;
      if ((factor = constantInteger(binary.getRhs())))
        source = binary.getLhs();
      else if ((factor = constantInteger(binary.getLhs())))
        source = binary.getRhs();
      if (!source || !factor || *factor <= 0)
        return std::nullopt;
      auto result = analyzeReplicaAffineIndex(source, depth + 1);
      if (!result || !checkedScale(result->constant, *factor,
                                   result->constant))
        return std::nullopt;
      for (auto &[axis, coefficient] : result->axisCoefficients)
        if (!checkedScale(coefficient, *factor, coefficient))
          return std::nullopt;
      if (result->dynamicMultiple &&
          !checkedScale(result->dynamicMultiple, *factor,
                        result->dynamicMultiple))
        return std::nullopt;
      return result;
    }
    return std::nullopt;
  }
  if (!isSingleScalarCoordinate(value))
    return std::nullopt;
  ReplicaAffineIndex result;
  result.dynamicMultiple = 1;
  return result;
}

llvm::SmallVector<std::pair<int64_t, int64_t>>
replicaAxes(riscv::ValueType value) {
  llvm::SmallVector<std::pair<int64_t, int64_t>> result;
  for (auto [axis, factor] :
       llvm::zip(value.getAxisIds().asArrayRef(),
                 value.getLayout().getReplicaFactors().asArrayRef()))
    if (factor > 1)
      result.emplace_back(axis, factor);
  return result;
}

std::optional<llvm::DenseMap<int64_t, int64_t>>
replicaCoordinates(riscv::ValueType value, int64_t part) {
  auto axes = replicaAxes(value);
  llvm::DenseMap<int64_t, int64_t> result;
  int64_t remaining = part;
  for (int64_t position = static_cast<int64_t>(axes.size()) - 1;
       position >= 0; --position) {
    const auto &[axis, extent] = axes[static_cast<size_t>(position)];
    if (extent <= 0)
      return std::nullopt;
    result[axis] = remaining % extent;
    remaining /= extent;
  }
  if (remaining)
    return std::nullopt;
  return result;
}

std::optional<int64_t>
replicaPartFor(riscv::ValueType value,
               const llvm::DenseMap<int64_t, int64_t> &coordinates) {
  int64_t part = 0;
  for (auto [axis, factor] :
       llvm::zip(value.getAxisIds().asArrayRef(),
                 value.getLayout().getReplicaFactors().asArrayRef())) {
    if (factor <= 0 || part > std::numeric_limits<int64_t>::max() / factor)
      return std::nullopt;
    int64_t coordinate = 0;
    if (factor > 1) {
      auto found = coordinates.find(axis);
      if (found == coordinates.end() || found->second < 0 ||
          found->second >= factor)
        return std::nullopt;
      coordinate = found->second;
    }
    part = part * factor + coordinate;
  }
  return part;
}

std::optional<VectorReplicaGatherPlan>
planVectorReplicaGather(riscv::ExtractOp operation, mlir::Value index) {
  auto input = operation.getInput().getType();
  auto result = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  auto indexType = mlir::dyn_cast<riscv::ValueType>(index.getType());
  if (!result || !indexType || input.getLayout().getCarrier() != "rvv" ||
      result.getLayout().getCarrier() != "scalar" ||
      indexType.getLayout().getCarrier() != "scalar")
    return std::nullopt;

  size_t gatherDimension = operation.getSelectors().size();
  size_t indexCursor = 0;
  unsigned gatherCount = 0;
  for (auto [dimension, selectorAttribute] :
       llvm::enumerate(operation.getSelectors())) {
    llvm::StringRef selector =
        mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
    if (selector == "all")
      continue;
    if (indexCursor >= operation.getIndices().size())
      return std::nullopt;
    if (selector == "gather") {
      ++gatherCount;
      gatherDimension = dimension;
      if (operation.getIndices()[indexCursor] != index)
        return std::nullopt;
    }
    ++indexCursor;
  }
  if (gatherCount != 1 || indexCursor != operation.getIndices().size() ||
      gatherDimension >= input.getAxisIds().size())
    return std::nullopt;

  auto sourceTime = riscv_internal::staticProduct(
      input.getLayout().getTimeFactors().asArrayRef());
  auto sourceLanes = riscv_internal::staticProduct(
      input.getLayout().getLaneFactors().asArrayRef());
  auto sourceParts = riscv_internal::staticProduct(
      input.getLayout().getReplicaFactors().asArrayRef());
  auto resultTime = riscv_internal::staticProduct(
      result.getLayout().getTimeFactors().asArrayRef());
  auto resultLanes = riscv_internal::staticProduct(
      result.getLayout().getLaneFactors().asArrayRef());
  auto resultParts = riscv_internal::staticProduct(
      result.getLayout().getReplicaFactors().asArrayRef());
  auto indexTime = riscv_internal::staticProduct(
      indexType.getLayout().getTimeFactors().asArrayRef());
  auto indexLanes = riscv_internal::staticProduct(
      indexType.getLayout().getLaneFactors().asArrayRef());
  auto indexParts = riscv_internal::staticProduct(
      indexType.getLayout().getReplicaFactors().asArrayRef());
  if (!sourceTime || *sourceTime != 1 || !sourceLanes || *sourceLanes <= 1 ||
      !sourceParts || *sourceParts <= 0 || !resultTime || *resultTime != 1 ||
      !resultLanes || *resultLanes != 1 || !resultParts || *resultParts <= 0 ||
      !indexTime || *indexTime != 1 || !indexLanes || *indexLanes != 1 ||
      !indexParts || *indexParts <= 0)
    return std::nullopt;

  for (size_t dimension = 0; dimension < input.getAxisIds().size(); ++dimension) {
    const int64_t laneFactor = input.getLayout().getLaneFactors()[dimension];
    if (dimension == gatherDimension) {
      if (laneFactor != *sourceLanes || input.getShape()[dimension] != laneFactor)
        return std::nullopt;
    } else if (laneFactor != 1) {
      return std::nullopt;
    }
  }

  auto matchingResultReplica = [&](int64_t axis,
                                   int64_t factor) -> bool {
    auto found = llvm::find(result.getAxisIds().asArrayRef(), axis);
    if (found == result.getAxisIds().asArrayRef().end())
      return factor == 1;
    const size_t position = static_cast<size_t>(
        found - result.getAxisIds().asArrayRef().begin());
    return result.getLayout().getReplicaFactors()[position] == factor;
  };
  for (auto [axis, factor] :
       llvm::zip(indexType.getAxisIds().asArrayRef(),
                 indexType.getLayout().getReplicaFactors().asArrayRef()))
    if (!matchingResultReplica(axis, factor))
      return std::nullopt;
  for (auto [axis, factor] :
       llvm::zip(input.getAxisIds().asArrayRef(),
                 input.getLayout().getReplicaFactors().asArrayRef()))
    if (axis != input.getAxisIds()[gatherDimension] &&
        !matchingResultReplica(axis, factor))
      return std::nullopt;

  VectorReplicaGatherPlan plan;
  plan.laneCount = *sourceLanes;
  plan.sourceParts.reserve(*resultParts);
  plan.indexParts.reserve(*resultParts);
  for (int64_t part = 0; part < *resultParts; ++part) {
    auto coordinates = replicaCoordinates(result, part);
    if (!coordinates)
      return std::nullopt;
    auto sourcePart = replicaPartFor(input, *coordinates);
    auto indexPart = replicaPartFor(indexType, *coordinates);
    if (!sourcePart || !indexPart || *sourcePart < 0 ||
        *sourcePart >= *sourceParts || *indexPart < 0 ||
        *indexPart >= *indexParts)
      return std::nullopt;
    plan.sourceParts.push_back(*sourcePart);
    plan.indexParts.push_back(*indexPart);
  }
  return plan;
}

std::optional<ScalarReplicaGatherPlan>
planScalarReplicaGather(riscv::ExtractOp operation, mlir::Value index) {
  auto reject = [&](llvm::StringRef reason)
      -> std::optional<ScalarReplicaGatherPlan> {
    (void)reason;
    return std::nullopt;
  };
  auto input = operation.getInput().getType();
  auto result = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  auto indexType = mlir::dyn_cast<riscv::ValueType>(index.getType());
  if (!result || !indexType || input.getLayout().getCarrier() != "scalar" ||
      result.getLayout().getCarrier() != "scalar" ||
      indexType.getLayout().getCarrier() != "scalar")
    return reject("carrier");
  auto inputTime = riscv_internal::staticProduct(
      input.getLayout().getTimeFactors().asArrayRef());
  auto resultTime = riscv_internal::staticProduct(
      result.getLayout().getTimeFactors().asArrayRef());
  auto indexTime = riscv_internal::staticProduct(
      indexType.getLayout().getTimeFactors().asArrayRef());
  auto inputParts = riscv_internal::staticProduct(
      input.getLayout().getReplicaFactors().asArrayRef());
  auto resultParts = riscv_internal::staticProduct(
      result.getLayout().getReplicaFactors().asArrayRef());
  auto indexParts = riscv_internal::staticProduct(
      indexType.getLayout().getReplicaFactors().asArrayRef());
  if (!inputTime || *inputTime != 1 || !resultTime || *resultTime != 1 ||
      !indexTime || *indexTime != 1 || !inputParts || !resultParts ||
      !indexParts || *resultParts <= 1 || *indexParts <= 1)
    return reject("part counts");

  size_t gatherDimension = operation.getSelectors().size();
  size_t indexCursor = 0;
  unsigned gatherCount = 0;
  for (auto [dimension, selectorAttribute] :
       llvm::enumerate(operation.getSelectors())) {
    llvm::StringRef selector =
        mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
    if (selector == "all")
      continue;
    if (indexCursor >= operation.getIndices().size())
      return reject("selector index");
    if (selector == "gather") {
      ++gatherCount;
      gatherDimension = dimension;
      if (operation.getIndices()[indexCursor] != index)
        return reject("gather index identity");
    }
    ++indexCursor;
  }
  if (gatherCount != 1 || indexCursor != operation.getIndices().size() ||
      gatherDimension >= input.getAxisIds().size())
    return reject("gather selector");

  auto resultAxes = replicaAxes(result);
  auto indexAxes = replicaAxes(indexType);
  auto factorFor = [](llvm::ArrayRef<std::pair<int64_t, int64_t>> axes,
                      int64_t axis) -> std::optional<int64_t> {
    auto found = llvm::find_if(
        axes, [&](const auto &entry) { return entry.first == axis; });
    return found == axes.end() ? std::nullopt
                               : std::optional<int64_t>(found->second);
  };
  for (const auto &[axis, factor] : indexAxes)
    if (factorFor(resultAxes, axis) != factor)
      return reject("index/result replica axes");
  const int64_t gatherAxis = input.getAxisIds()[gatherDimension];
  const int64_t gatherFactor =
      input.getLayout().getReplicaFactors()[gatherDimension];
  if (gatherFactor <= 1 || gatherFactor % *indexParts)
    return reject("gather source factor");
  auto inputAxes = replicaAxes(input);
  for (const auto &[axis, factor] : inputAxes)
    if (axis != gatherAxis && factorFor(resultAxes, axis) != factor)
      return reject("retained source axes");
  auto affine = analyzeReplicaAffineIndex(index);
  if (!affine ||
      (affine->dynamicMultiple &&
       affine->dynamicMultiple % *indexParts))
    return reject("affine dynamic base");
  llvm::DenseSet<int64_t> replicaAxisSet;
  for (const auto &[axis, factor] : indexAxes)
    replicaAxisSet.insert(axis);
  for (const auto &[axis, coefficient] : affine->axisCoefficients)
    if (!replicaAxisSet.contains(axis))
      return reject("affine axis ownership");

  auto evaluateIndex = [&](const llvm::DenseMap<int64_t, int64_t> &coordinates)
      -> std::optional<int64_t> {
    int64_t relative = affine->constant;
    for (const auto &[axis, coefficient] : affine->axisCoefficients) {
      int64_t term = 0;
      if (!checkedScale(coordinates.lookup(axis), coefficient, term) ||
          !checkedAdd(relative, term, relative))
        return std::nullopt;
    }
    return relative;
  };
  auto evaluateResidue = [&](const llvm::DenseMap<int64_t, int64_t> &coordinates)
      -> std::optional<int64_t> {
    auto relative = evaluateIndex(coordinates);
    if (!relative)
      return std::nullopt;
    int64_t residue = *relative % *indexParts;
    if (residue < 0)
      residue += *indexParts;
    return residue;
  };

  llvm::DenseSet<int64_t> seen;
  for (int64_t part = 0; part < *indexParts; ++part) {
    auto coordinates = replicaCoordinates(indexType, part);
    if (!coordinates)
      return reject("replica coordinate");
    auto selected = affine->dynamicMultiple ? evaluateResidue(*coordinates)
                                            : evaluateIndex(*coordinates);
    if (!selected || (!affine->dynamicMultiple &&
                      (*selected < 0 || *selected >= gatherFactor)))
      return reject("affine evaluation");
    if (!seen.insert(*selected).second)
      return reject("residue permutation");
  }
  if (seen.size() != static_cast<size_t>(*indexParts))
    return reject("residue coverage");

  ScalarReplicaGatherPlan plan;
  plan.arity = affine->dynamicMultiple ? gatherFactor / *indexParts : 1;
  plan.candidates.reserve(*resultParts * plan.arity);
  plan.keys.reserve(*resultParts * plan.arity);
  plan.localBases.reserve(*resultParts);
  for (int64_t part = 0; part < *resultParts; ++part) {
    auto coordinates = replicaCoordinates(result, part);
    if (!coordinates)
      return reject("result replica coordinate");
    auto selected = affine->dynamicMultiple ? evaluateResidue(*coordinates)
                                            : evaluateIndex(*coordinates);
    if (!selected)
      return reject("result affine evaluation");
    std::optional<int64_t> localBase;
    for (int64_t cohort = 0; cohort < plan.arity; ++cohort) {
      const int64_t key =
          *selected + (affine->dynamicMultiple ? cohort * *indexParts : 0);
      int64_t sourcePart = 0;
      for (auto [axis, factor] :
           llvm::zip(input.getAxisIds().asArrayRef(),
                     input.getLayout().getReplicaFactors().asArrayRef())) {
        int64_t coordinate =
            axis == gatherAxis ? key : coordinates->lookup(axis);
        if (factor <= 0 || coordinate < 0 || coordinate >= factor)
          return reject("source replica coordinate");
        sourcePart = sourcePart * factor + coordinate;
      }
      if (sourcePart < 0 || sourcePart >= *inputParts)
        return reject("source replica part");
      const int64_t candidateBase = sourcePart - key;
      if (localBase && *localBase != candidateBase)
        return reject("local gather base");
      localBase = candidateBase;
      plan.candidates.push_back(sourcePart);
      plan.keys.push_back(key);
    }
    if (!localBase)
      return reject("local gather base missing");
    plan.localBases.push_back(*localBase);
  }
  return plan;
}

void eraseDeadRegularIndexChain(mlir::Value value,
                                llvm::DenseSet<mlir::Operation *> &visited,
                                mlir::IRRewriter &rewriter) {
  mlir::Operation *definition = value.getDefiningOp();
  const bool hasUses =
      definition && llvm::any_of(definition->getResults(), [](mlir::Value result) {
        return !result.use_empty();
      });
  if (!definition || !visited.insert(definition).second || hasUses ||
      !mlir::isa<riscv::IotaOp, riscv::BinaryOp, riscv::ConvertLayoutOp>(
          definition))
    return;
  llvm::SmallVector<mlir::Value> operands(definition->getOperands());
  rewriter.eraseOp(definition);
  for (mlir::Value operand : operands)
    eraseDeadRegularIndexChain(operand, visited, rewriter);
}

riscv::AccessAttr makeAccess(mlir::Builder &builder, llvm::StringRef form,
                             llvm::StringRef mapping, int64_t alignment,
                             int64_t group = 0, int64_t layer = 0,
                             int64_t joinFields = 0, int64_t joinLowBits = 0,
                             int64_t joinRole = 0,
                             int64_t bitOffset = 0, int64_t storageBits = 0,
                             llvm::StringRef order = "none") {
  return riscv::AccessAttr::get(
      builder.getContext(), form, mapping, std::max<int64_t>(1, alignment),
      form == "indexed" ? 32 : 0, form == "segment" ? 2 : 0, group,
      layer, joinFields, joinLowBits, joinRole, bitOffset, storageBits, order);
}

riscv::LeafAttr transferLeaf(mlir::Builder &builder, llvm::StringRef family,
                             llvm::StringRef instruction) {
  return riscv_internal::leaf(builder, "transfer", family, instruction,
                              instruction, 0, 0);
}

struct IndexedEntryPhysicalChoice {
  riscv::AccessAttr access;
  riscv::LeafAttr leaf;
};

IndexedEntryPhysicalChoice
selectIndexedEntryPhysicalChoice(mlir::Builder &builder,
                                 riscv::ValueType result,
                                 bool vectorEntries,
                                 int64_t sourceAlignment,
                                 int64_t entryByteStride) {
  const int64_t entryAlignment =
      std::gcd(std::max<int64_t>(1, sourceAlignment), entryByteStride);
  const unsigned payloadBits =
      riscv_internal::logicalBitWidth(result.getElementType());
  const int64_t payloadBytes =
      payloadBits && payloadBits % 8 == 0
          ? static_cast<int64_t>(payloadBits / 8)
          : 0;
  const bool byteLoad =
      !vectorEntries && payloadBytes > 1 && entryAlignment < payloadBytes;
  const llvm::StringRef family =
      vectorEntries ? "indexed-entry-gather" : "indexed-entry-load";
  const llvm::StringRef instruction =
      vectorEntries ? "rvv.indexed-entry-gather"
                    : byteLoad ? "rvv.indexed-entry-byte-load"
                               : "rvv.indexed-entry-load";
  return {
      makeAccess(builder, vectorEntries ? "indexed" : "unit", "natural",
                 vectorEntries ? 1 : entryAlignment),
      riscv_internal::leaf(builder, "rvv", family, instruction, instruction,
                           0, 0)};
}

mlir::LogicalResult verifyAccessCapability(mlir::Operation *operation,
                                           riscv::AccessAttr access) {
  auto kernel = operation->getParentOfType<riscv::KernelOp>();
  if (!kernel)
    return operation->emitError("physical memory edge is outside a RISC-V kernel");
  auto target = kernel.getTarget();
  if (access.getForm() == "indexed" && !target.getHasIndexedMemory())
    return operation->emitError(
        "selected indexed memory form is unsupported by the target profile");
  if (access.getForm() == "segment" && !target.getHasSegmentMemory())
    return operation->emitError(
        "selected segment memory form is unsupported by the target profile");
  return mlir::success();
}

bool laneTraversesRecords(riscv::FieldOp operation, mlir::Type physicalType) {
  auto field = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  auto physical = mlir::dyn_cast<riscv::ValueType>(physicalType);
  riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(operation);
  if (!field || !physical || physical.getLayout().getCarrier() != "rvv" ||
      facts.logicalRank < 0 ||
      facts.logicalRank > static_cast<int64_t>(field.getShape().size()))
    return false;
  int64_t laneAxis = 0;
  for (auto [axis, factor] :
       llvm::zip(physical.getAxisIds().asArrayRef(),
                 physical.getLayout().getLaneFactors().asArrayRef()))
    if (factor > 1) {
      if (laneAxis)
        return false;
      laneAxis = axis;
    }
  if (!laneAxis)
    return false;
  const int64_t outerRank =
      static_cast<int64_t>(field.getShape().size()) - facts.logicalRank;
  return llvm::is_contained(
      field.getAxisIds().asArrayRef().take_front(outerRank), laneAxis);
}

bool isNaturalByteIdentity(const riscv_internal::FieldFacts &facts,
                           mlir::Type elementType) {
  return facts.mapping == "grouped_layered" && facts.group > 0 &&
         facts.group == facts.layer &&
         riscv_internal::logicalBitWidth(elementType) == 8;
}

riscv::AccessAttr fieldAccess(mlir::Builder &builder,
                              riscv::FieldOp operation,
                              mlir::Type physicalType) {
  riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(operation);
  llvm::StringRef mapping = facts.mapping;
  // A byte-sized grouped field with one layer per group has the same affine
  // byte map as natural storage: floor(i / group) * group + i % group == i.
  // Record that identity in the selected physical edge so later memory ops do
  // not have to reinterpret a nominal grouped mapping as a byte base.
  auto result = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
  if (result && isNaturalByteIdentity(facts, result.getElementType()))
    mapping = "natural";
  llvm::StringRef form = mapping == "natural" ? "unit" : "indexed";
  if ((mapping == "natural" && facts.scalarPerRecord) ||
      laneTraversesRecords(operation, physicalType))
    form = "strided";
  return makeAccess(builder, form, mapping, facts.alignment, facts.group,
                    facts.layer, facts.joinFields, facts.joinLowBits,
                    facts.joinRole, facts.bitOffset, facts.storageBits,
                    facts.order);
}

bool collectEncodedScalarRoots(
    mlir::Value value, mlir::Block *block,
    llvm::DenseSet<mlir::Operation *> &visited,
    llvm::SmallVectorImpl<riscv::ExtractOp> &roots) {
  mlir::Operation *definition = value.getDefiningOp();
  if (!definition)
    return true;
  if (definition->getBlock() != block)
    return false;
  if (!visited.insert(definition).second)
    return true;
  if (auto extract = mlir::dyn_cast<riscv::ExtractOp>(definition)) {
    auto field = riscv_internal::sourceField(extract.getInput());
    if (!field || extract.getAccess().getMapping() != "natural" ||
        extract.getAccess().getBitOffset() % 8)
      return false;
    roots.push_back(extract);
    return true;
  }
  if (mlir::isa<riscv::ConstantOp, riscv::IotaOp>(definition))
    return true;
  if (definition->getNumResults() != 1 ||
      !mlir::isMemoryEffectFree(definition) ||
      mlir::isa<riscv::RegisterMaterializeOp>(definition) ||
      !mlir::isa<riscv::UnaryOp, riscv::BinaryOp, riscv::CastOp,
                 riscv::NarrowOp, riscv::WidenOp,
                 riscv::ConvertLayoutOp>(definition))
    return false;
  for (mlir::Value operand : definition->getOperands()) {
    auto type = mlir::dyn_cast<riscv::ValueType>(operand.getType());
    if (type && type.getLayout().getCarrier() != "scalar")
      return false;
    if (!collectEncodedScalarRoots(operand, block, visited, roots))
      return false;
  }
  return true;
}

bool needsScalarEncodedShare(mlir::Value value) {
  auto result = mlir::dyn_cast<riscv::ValueType>(value.getType());
  const bool scalar =
      value.getType().isIndex() ||
      mlir::isa<mlir::IntegerType, mlir::FloatType>(value.getType()) ||
      (result && result.getLayout().getCarrier() == "scalar");
  llvm::DenseSet<mlir::Operation *> visited;
  llvm::SmallVector<riscv::ExtractOp> roots;
  if (!scalar ||
      !collectEncodedScalarRoots(value, value.getParentBlock(), visited, roots) ||
      roots.empty())
    return false;
  auto rootField = riscv_internal::sourceField(roots.front().getInput());
  if (!rootField)
    return false;
  for (riscv::ExtractOp root : llvm::drop_begin(roots)) {
    auto field = riscv_internal::sourceField(root.getInput());
    if (!field || field.getOwner() != rootField.getOwner() ||
        field.getName() != rootField.getName())
      return false;
  }

  auto element = mlir::dyn_cast<mlir::IntegerType>(
      riscv_internal::logicalElement(value.getType()));
  if (!element || (element.getWidth() != 8 && element.getWidth() != 16 &&
                   element.getWidth() != 32))
    return false;

  unsigned uses = 0;
  for (mlir::OpOperand &use : value.getUses()) {
    if (use.getOwner()->getBlock() != value.getParentBlock() ||
        mlir::isa<riscv::RegisterMaterializeOp>(use.getOwner()))
      return false;
    ++uses;
  }
  return uses > 1;
}

struct CrossLevelEncodedSupply {
  mlir::scf::ForOp loop;
  riscv::FieldOp innerField;
  riscv::ExtractOp extract;
  riscv::BinaryOp decode;
  riscv::PhysicalPointOp point;
  riscv::PhysicalPointOp parent;
  riscv::FieldOp outerField;
  mlir::Value localIndex;
  mlir::Value mask;
  int64_t maskValue = 0;
  int64_t partition = 0;
};

bool isOnlySelector(riscv::ExtractOp extract, llvm::StringRef expected) {
  return extract.getSelectors().size() == 1 &&
         mlir::cast<mlir::StringAttr>(extract.getSelectors()[0])
                 .getValue() == expected &&
         extract.getIndices().size() == 1;
}

std::optional<mlir::Value>
subLevelOrdinal(riscv::PhysicalPointOp point,
                riscv::PhysicalPointOp parent, int64_t partition) {
  auto addition = point.getBase().getDefiningOp<mlir::arith::AddIOp>();
  if (!addition || partition <= 0)
    return std::nullopt;
  mlir::Value offset;
  if (addition.getLhs() == parent.getBase())
    offset = addition.getRhs();
  else if (addition.getRhs() == parent.getBase())
    offset = addition.getLhs();
  else
    return std::nullopt;
  auto multiply = offset.getDefiningOp<mlir::arith::MulIOp>();
  if (!multiply)
    return std::nullopt;
  auto lhs = constantInteger(multiply.getLhs());
  auto rhs = constantInteger(multiply.getRhs());
  if (lhs && *lhs == partition)
    return multiply.getRhs();
  if (rhs && *rhs == partition)
    return multiply.getLhs();
  return std::nullopt;
}

std::optional<mlir::Value>
materializeDomainBitmaskByteBase(riscv::FieldOp field,
                                 riscv::PhysicalPointOp point,
                                 const riscv_internal::FieldFacts &facts,
                                 mlir::Operation *origin,
                                 mlir::IRRewriter &rewriter) {
  auto load = field.getOwner().getDefiningOp<riscv::LoadOp>();
  auto slice = load ? load.getRegion().getDefiningOp<riscv::SliceOp>()
                    : riscv::SliceOp();
  riscv::PhysicalPointOp recordPoint;
  if (slice)
    for (mlir::Value index : slice.getIndices())
      if (auto candidate = index.getDefiningOp<riscv::PhysicalPointOp>()) {
        recordPoint = candidate;
        break;
      }
  if (!recordPoint || facts.group <= 0 || facts.layer <= 0)
    return std::nullopt;

  riscv::PhysicalPointOp cursor = point;
  while (cursor && cursor != recordPoint) {
    auto partition = constantInteger(cursor.getPartition());
    if (!partition || *partition <= 0 || *partition % facts.group)
      return std::nullopt;
    cursor = cursor.getParent().getDefiningOp<riscv::PhysicalPointOp>();
  }
  if (cursor != recordPoint)
    return std::nullopt;

  auto relative = rewriter.create<riscv::BinaryOp>(
      origin->getLoc(), rewriter.getIndexType(), point.getBase(),
      recordPoint.getBase(), "sub", riscv_internal::unselectedLeaf(rewriter));
  riscv_internal::copyOrigin(origin, relative);
  mlir::Value byteBase = relative.getResult();
  if (facts.group != 1) {
    auto group = rewriter.create<riscv::ConstantOp>(
        origin->getLoc(), rewriter.getIndexType(),
        rewriter.getIndexAttr(facts.group));
    auto divided = rewriter.create<riscv::BinaryOp>(
        origin->getLoc(), rewriter.getIndexType(), byteBase,
        group.getResult(), "div", riscv_internal::unselectedLeaf(rewriter));
    riscv_internal::copyOrigin(origin, group);
    riscv_internal::copyOrigin(origin, divided);
    byteBase = divided.getResult();
  }
  if (facts.layer == 1)
    return byteBase;
  auto layer = rewriter.create<riscv::ConstantOp>(
      origin->getLoc(), rewriter.getIndexType(),
      rewriter.getIndexAttr(facts.layer));
  auto multiplied = rewriter.create<riscv::BinaryOp>(
      origin->getLoc(), rewriter.getIndexType(), byteBase, layer.getResult(),
      "mul", riscv_internal::unselectedLeaf(rewriter));
  riscv_internal::copyOrigin(origin, layer);
  riscv_internal::copyOrigin(origin, multiplied);
  return multiplied.getResult();
}

riscv::FieldOp outerRVVFieldFor(riscv::FieldOp inner,
                                mlir::scf::ForOp loop) {
  auto innerType = mlir::dyn_cast<riscv::ValueType>(inner.getResult().getType());
  if (!innerType || innerType.getLayout().getCarrier() != "local")
    return {};
  riscv::FieldOp selected;
  for (mlir::Operation &operation : *loop->getBlock()) {
    auto candidate = mlir::dyn_cast<riscv::FieldOp>(operation);
    auto type = candidate
                    ? mlir::dyn_cast<riscv::ValueType>(
                          candidate.getResult().getType())
                    : riscv::ValueType();
    if (!candidate || candidate.getOwner() != inner.getOwner() ||
        candidate.getName() != inner.getName() || !type ||
        type.getLayout().getCarrier() != "rvv" ||
        type.getElementType() != innerType.getElementType() ||
        type.getShape() != innerType.getShape() ||
        type.getAxisIds() != innerType.getAxisIds())
      continue;
    const bool hasOuterUse = llvm::any_of(
        candidate.getResult().getUsers(),
        [&](mlir::Operation *user) { return !loop->isAncestor(user); });
    if (!hasOuterUse)
      continue;
    if (selected)
      return {};
    selected = candidate;
  }
  return selected;
}

std::optional<CrossLevelEncodedSupply>
matchCrossLevelEncodedSupply(riscv::ExtractOp extract) {
  CrossLevelEncodedSupply result;
  result.extract = extract;
  result.loop = extract->getParentOfType<mlir::scf::ForOp>();
  result.innerField = extract.getInput().getDefiningOp<riscv::FieldOp>();
  result.point = extract.getIndices().empty()
                     ? riscv::PhysicalPointOp()
                     : extract.getIndices().front().getDefiningOp<
                           riscv::PhysicalPointOp>();
  result.parent = result.point
                      ? result.point.getParent().getDefiningOp<
                            riscv::PhysicalPointOp>()
                      : riscv::PhysicalPointOp();
  auto partition = result.point
                       ? result.point.getPartition()
                             .getDefiningOp<mlir::arith::ConstantIndexOp>()
                       : mlir::arith::ConstantIndexOp();
  if (!result.loop || !result.innerField || !result.point || !result.parent ||
      !partition || !isOnlySelector(extract, "group_index") ||
      extract.getResult().getUses().empty() ||
      !extract.getResult().hasOneUse())
    return std::nullopt;
  result.partition = partition.value();
  auto ordinal = subLevelOrdinal(result.point, result.parent, result.partition);
  if (!ordinal)
    return std::nullopt;
  result.localIndex = *ordinal;

  result.decode = mlir::dyn_cast<riscv::BinaryOp>(
      extract.getResult().use_begin()->getOwner());
  if (!result.decode || result.decode.getKind() != "and" ||
      !result.decode.getResult().hasOneUse())
    return std::nullopt;
  if (result.decode.getLhs() == extract.getResult())
    result.mask = result.decode.getRhs();
  else if (result.decode.getRhs() == extract.getResult())
    result.mask = result.decode.getLhs();
  else
    return std::nullopt;
  auto mask = constantInteger(result.mask);
  if (!mask || *mask <= 0)
    return std::nullopt;
  result.maskValue = *mask;
  result.outerField = outerRVVFieldFor(result.innerField, result.loop);
  if (!result.outerField)
    return std::nullopt;
  return result;
}

bool sameCrossLevelSupply(const CrossLevelEncodedSupply &lhs,
                          const CrossLevelEncodedSupply &rhs) {
  return lhs.loop == rhs.loop && lhs.outerField == rhs.outerField &&
         lhs.parent == rhs.parent && lhs.maskValue == rhs.maskValue &&
         lhs.partition == rhs.partition;
}

std::optional<riscv::ValueType>
localValueForRVVSupply(mlir::Builder &builder, riscv::ValueType input) {
  auto layout = input.getLayout();
  if (layout.getCarrier() != "rvv" || input.getShape().empty())
    return std::nullopt;
  llvm::SmallVector<int64_t> ones(input.getShape().size(), 1);
  llvm::SmallVector<int64_t> local;
  local.reserve(input.getShape().size());
  int64_t laneDimensions = 0;
  for (size_t dimension = 0; dimension < input.getShape().size(); ++dimension) {
    const int64_t lane = layout.getLaneFactors()[dimension];
    const int64_t replica = layout.getReplicaFactors()[dimension];
    if (layout.getTimeFactors()[dimension] != 1 ||
        layout.getFragmentFactors()[dimension] != 1 ||
        layout.getLocalFactors()[dimension] != 1 || lane <= 0 || replica <= 0)
      return std::nullopt;
    if (lane > 1) {
      ++laneDimensions;
      if (dimension + 1 != input.getShape().size() || replica != 1)
        return std::nullopt;
      local.push_back(lane);
    } else {
      local.push_back(replica);
    }
    if (input.getShape()[dimension] != lane * replica)
      return std::nullopt;
  }
  if (laneDimensions != 1)
    return std::nullopt;
  auto localLayout = riscv::LayoutAttr::get(
      builder.getContext(), "local", input.getAxisIds(),
      riscv_internal::integers(builder, ones),
      riscv_internal::integers(builder, ones),
      riscv_internal::integers(builder, ones),
      riscv_internal::integers(builder, ones),
      riscv_internal::integers(builder, local), layout.getSew(), 0, 1, 0,
      "full");
  return riscv::ValueType::get(builder.getContext(), input.getElementType(),
                              input.getShape(), input.getAxisIds(), localLayout);
}

void materializeCrossLevelEncodedSupplies(mlir::ModuleOp module,
                                          mlir::IRRewriter &rewriter) {
  llvm::SmallVector<CrossLevelEncodedSupply> candidates;
  module.walk([&](riscv::ExtractOp extract) {
    if (auto candidate = matchCrossLevelEncodedSupply(extract))
      candidates.push_back(*candidate);
  });
  llvm::DenseSet<mlir::Operation *> consumed;
  int64_t nextAlias = 0;
  int64_t nextBirth = 0;
  module.walk([&](riscv::LocalAllocOp allocation) {
    auto type = allocation.getResult().getType();
    nextAlias = std::max(nextAlias, type.getAliasSet() + 1);
    nextBirth = std::max(nextBirth, type.getBirthId() + 1);
  });

  for (CrossLevelEncodedSupply &leader : candidates) {
    if (consumed.contains(leader.extract))
      continue;
    llvm::SmallVector<CrossLevelEncodedSupply *> group;
    for (CrossLevelEncodedSupply &candidate : candidates)
      if (!consumed.contains(candidate.extract) &&
          sameCrossLevelSupply(leader, candidate))
        group.push_back(&candidate);
    // One physical supply must eliminate at least two dynamic decode edges.
    if (group.size() < 2)
      continue;

    auto input = mlir::dyn_cast<riscv::ValueType>(
        leader.outerField.getResult().getType());
    auto localValue = input ? localValueForRVVSupply(rewriter, input)
                            : std::optional<riscv::ValueType>();
    auto element = input
                       ? mlir::dyn_cast<mlir::IntegerType>(input.getElementType())
                       : mlir::IntegerType();
    auto elements = input ? positiveProduct(input.getShape().asArrayRef())
                          : std::optional<int64_t>();
    if (!localValue || !element || element.isSigned() ||
        element.getWidth() % 8 || !elements || *elements <= 0)
      continue;
    if (*elements >
        std::numeric_limits<int64_t>::max() / (element.getWidth() / 8))
      continue;
    const int64_t bytes = *elements * (element.getWidth() / 8);
    const int64_t owner = leader.parent.getResult().getType().getDomain().getDomainId();
    const int64_t lifetime =
        leader.point.getResult().getType().getDomain().getDomainId();

    leader.outerField->moveBefore(leader.loop);
    rewriter.setInsertionPoint(leader.loop);
    auto decoded = rewriter.create<riscv::BinaryOp>(
        leader.outerField.getLoc(), input, leader.outerField.getResult(),
        leader.mask, "and", riscv_internal::unselectedLeaf(rewriter));
    riscv_internal::copyOrigin(leader.decode, decoded);
    auto allocationBytes = rewriter.create<mlir::arith::ConstantIndexOp>(
        leader.outerField.getLoc(), bytes);
    auto storageType = riscv::LocalType::get(
        rewriter.getContext(), input.getElementType(), input.getShape(),
        input.getAxisIds(), bytes, std::max<int64_t>(8, element.getWidth() / 8),
        nextAlias++, "pack", owner, nextBirth++, lifetime, "physical-share");
    auto allocation = rewriter.create<riscv::LocalAllocOp>(
        leader.outerField.getLoc(), storageType, allocationBytes);
    auto logicalElements = rewriter.create<mlir::arith::ConstantIndexOp>(
        leader.outerField.getLoc(), *elements);
    auto binding = rewriter.create<riscv::LocalBindOp>(
        leader.outerField.getLoc(), *localValue, allocation.getResult(),
        logicalElements);
    auto stored = rewriter.create<riscv::RVVLocalMaterializeOp>(
        leader.outerField.getLoc(), decoded.getResult(), binding.getResult(),
        riscv_internal::leaf(
            rewriter, "transfer", "local-materialize",
            "rvv.local-materialize", "rvv.local-materialize",
            input.getLayout().getRegisterGroups(), 0, 0, 0, "none", "exact",
            {}, bytes));
    riscv_internal::copyOrigin(leader.outerField, allocation);
    riscv_internal::copyOrigin(leader.outerField, binding);
    riscv_internal::copyOrigin(leader.outerField, stored);

    for (CrossLevelEncodedSupply *candidate : group) {
      rewriter.setInsertionPoint(candidate->decode);
      auto loaded = rewriter.create<riscv::LocalLoadOp>(
          candidate->decode.getLoc(), candidate->decode.getResult().getType(),
          binding.getResult(), candidate->localIndex,
          riscv_internal::leaf(rewriter, "transfer", "local-load",
                               "local.load.element", "local.load.element", 0,
                               0));
      riscv_internal::copyOrigin(candidate->decode, loaded);
      candidate->decode.getResult().replaceAllUsesWith(loaded.getResult());
      consumed.insert(candidate->extract);
      rewriter.eraseOp(candidate->decode);
      rewriter.eraseOp(candidate->extract);
      if (candidate->innerField.getResult().use_empty())
        rewriter.eraseOp(candidate->innerField);
    }
  }
}

class PlanRISCVMemoryPass
    : public mlir::PassWrapper<PlanRISCVMemoryPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  explicit PlanRISCVMemoryPass(bool nestedDomainOnly = false)
      : nestedDomainOnly(nestedDomainOnly) {}

  llvm::StringRef getArgument() const override {
    return nestedDomainOnly ? "weft-riscv-plan-nested-memory"
                            : "weft-riscv-plan-memory";
  }
  llvm::StringRef getDescription() const override {
    return "Select and materialize memory forms from descriptors and value layouts";
  }

  void runOnOperation() override {
    mlir::Builder builder(&getContext());
    mlir::IRRewriter rewriter(&getContext());
    bool failed = false;
    llvm::SmallVector<riscv::LoadOp> deadTableLoads;
    llvm::SmallVector<riscv::MaterializeOp> deadTableMaterializations;
    llvm::DenseMap<mlir::Value, mlir::Value> tableSnapshots;
    if (!nestedDomainOnly) {
    getOperation().walk([&](riscv::LoadOp operation) {
      auto memory = operation.getRegion().getType();
      auto encoding = mlir::cast<kernel::EncodingType>(memory.getEncoding());
      llvm::StringRef mapping = encoding.getKind() == "dense" ? "dense" : "opaque";
      auto access = encoding.getKind() == "dense"
                        ? riscv_internal::denseAccess(
                              builder, memory, operation.getResult().getType())
                        : makeAccess(builder, "unit", mapping,
                                     memory.getAlignment());
      if (mlir::failed(verifyAccessCapability(operation, access))) {
        failed = true;
        return;
      }
      llvm::StringRef form = access.getForm();
      operation.setAccessAttr(access);
      operation.setLeafAttr(
          transferLeaf(builder, "load", ("rvv.load." + form).str()));
    });
    getOperation().walk([&](riscv::StoreOp operation) {
      auto memory = operation.getRegion().getType();
      auto encoding = mlir::cast<kernel::EncodingType>(memory.getEncoding());
      llvm::StringRef mapping = encoding.getKind() == "dense" ? "dense" : "opaque";
      auto access = encoding.getKind() == "dense"
                        ? riscv_internal::denseAccess(
                              builder, memory, operation.getValue().getType())
                        : makeAccess(builder, "unit", mapping,
                                     memory.getAlignment());
      if (mlir::failed(verifyAccessCapability(operation, access))) {
        failed = true;
        return;
      }
      llvm::StringRef form = access.getForm();
      operation.setAccessAttr(access);
      auto value =
          mlir::dyn_cast<riscv::ValueType>(operation.getValue().getType());
      const bool scalar = !value || value.getLayout().getCarrier() == "scalar";
      operation.setLeafAttr(transferLeaf(
          builder, "store",
          scalar ? "scalar.store" : ("rvv.store." + form).str()));
    });
    getOperation().walk([&](riscv::FieldOp operation) {
      riscv_internal::FieldFacts facts = riscv_internal::fieldFacts(operation);
      if (facts.mapping == "opaque") {
        operation.emitError("encoded field has no complete storage mapping");
        failed = true;
        return;
      }
      auto access =
          fieldAccess(builder, operation, operation.getResult().getType());
      auto target = operation->getParentOfType<riscv::KernelOp>().getTarget();
      if (access.getForm() == "indexed" && !target.getHasIndexedMemory()) {
        operation.emitError(
            "encoded field requires indexed memory unsupported by the target profile");
        failed = true;
        return;
      }
      operation.setAccessAttr(access);
      operation.setLeafAttr(transferLeaf(
          builder, "encoded-field", ("rvv.encoded." + access.getMapping()).str()));
    });
    getOperation().walk([&](riscv::ExtractOp operation) {
      riscv::AccessAttr access;
      auto field = riscv_internal::sourceField(operation.getInput());
      auto inputLayout = riscv_internal::layoutOf(operation.getInput().getType());
      const bool gather = llvm::any_of(
          operation.getSelectors(), [](mlir::Attribute selector) {
            llvm::StringRef value =
                mlir::cast<mlir::StringAttr>(selector).getValue();
            return value == "gather" || value == "regular";
          });
      mlir::Value gatherIndex;
      size_t gatherCursor = 0;
      for (mlir::Attribute selectorAttribute : operation.getSelectors()) {
        llvm::StringRef selector =
            mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
        if (selector == "all")
          continue;
        if (gatherCursor >= operation.getIndices().size())
          break;
        if (selector == "gather")
          gatherIndex = operation.getIndices()[gatherCursor];
        ++gatherCursor;
      }
      auto inputValue =
          mlir::dyn_cast<riscv::ValueType>(operation.getInput().getType());
      auto resultValue =
          mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
      auto indexValue =
          gatherIndex ? mlir::dyn_cast<riscv::ValueType>(gatherIndex.getType())
                      : riscv::ValueType();
      auto inputTime = inputLayout
                           ? riscv_internal::staticProduct(
                                 inputLayout.getTimeFactors().asArrayRef())
                           : std::optional<int64_t>();
      auto inputReplicas =
          inputLayout
              ? riscv_internal::staticProduct(
                    inputLayout.getReplicaFactors().asArrayRef())
              : std::optional<int64_t>();
      const bool completeRegisterSource =
          inputValue && resultValue && indexValue && inputLayout &&
          inputLayout.getCarrier() == "rvv" &&
          resultValue.getLayout().getCarrier() == "rvv" &&
          indexValue.getLayout().getCarrier() == "rvv" &&
          inputTime && *inputTime == 1 && inputReplicas && *inputReplicas == 1;
      const bool compatibleRegisterGather =
          completeRegisterSource &&
          inputLayout.getSew() == resultValue.getLayout().getSew() &&
          inputLayout.getLmulEighths() ==
              resultValue.getLayout().getLmulEighths() &&
          indexValue.getLayout().getSew() == resultValue.getLayout().getSew() &&
          indexValue.getLayout().getLmulEighths() ==
              resultValue.getLayout().getLmulEighths();
      if (gatherIndex) {
        auto scalarPlan = planScalarReplicaGather(operation, gatherIndex);
        if (scalarPlan) {
          operation.setAccessAttr(
              makeAccess(builder, "register", "natural", 1));
          operation.setLeafAttr(riscv_internal::leaf(
              builder, "scalar", "extract", "scalar.replica-gather",
              "scalar.replica-gather", 0, 0));
          operation->setAttr("replica_gather_arity",
                             builder.getI64IntegerAttr(scalarPlan->arity));
          operation->setAttr(
              "replica_gather_candidates",
              builder.getDenseI64ArrayAttr(scalarPlan->candidates));
          operation->setAttr(
              "replica_gather_keys",
              builder.getDenseI64ArrayAttr(scalarPlan->keys));
          return;
        }
        auto vectorPlan = planVectorReplicaGather(operation, gatherIndex);
        if (vectorPlan) {
          operation.setAccessAttr(
              makeAccess(builder, "register", "natural", 1));
          operation.setLeafAttr(riscv_internal::leaf(
              builder, "rvv", "extract", "rvv.extract.lane-to-replica",
              "rvv.extract.lane-to-replica", 1, 0));
          operation->setAttr(
              "lane_gather_count",
              builder.getI64IntegerAttr(vectorPlan->laneCount));
          operation->setAttr(
              "lane_gather_source_parts",
              builder.getDenseI64ArrayAttr(vectorPlan->sourceParts));
          operation->setAttr(
              "lane_gather_index_parts",
              builder.getDenseI64ArrayAttr(vectorPlan->indexParts));
          return;
        }
      }
      if (gather && compatibleRegisterGather) {
        operation.setAccessAttr(makeAccess(builder, "register", "natural", 1));
        operation.setLeafAttr(riscv_internal::leaf(
            builder, "rvv", "extract", "rvv.extract.vrgather",
            "rvv.extract.vrgather", 0, 0));
        return;
      }
      // Encoded fields use the local carrier because their bytes are not yet a
      // numerical register value.  They nevertheless retain an encoded memory
      // edge selected on FieldOp; do not misclassify that representation as a
      // compiler-created local array.
      if (!field && inputLayout && inputLayout.getCarrier() == "local") {
        if (gather) {
          operation.setAccessAttr(makeAccess(builder, "indexed", "dense", 1));
          auto target =
              operation->getParentOfType<riscv::KernelOp>().getTarget();
          if (!target.getHasIndexedMemory()) {
            operation.emitError(
                "local vector gather is unsupported by the target profile");
            failed = true;
            return;
          }
          operation.setLeafAttr(riscv_internal::leaf(
              builder, "rvv", "local-gather", "rvv.local-gather",
              "rvv.local-gather", 0, 0));
        } else {
          operation.setAccessAttr(makeAccess(builder, "local", "dense", 1));
          operation.setLeafAttr(transferLeaf(builder, "local-extract",
                                             "local.extract"));
        }
        return;
      }
      if (field)
        access = field.getAccess();
      else
        access = makeAccess(builder, "unit", "dense", 1);
      llvm::StringRef form = access.getForm();
      // A shaped gather selector is a real indexed access irrespective of the
      // producer's storage mapping.
      if (llvm::any_of(operation.getSelectors(), [](mlir::Attribute selector) {
            llvm::StringRef value =
                mlir::cast<mlir::StringAttr>(selector).getValue();
            return value == "gather" || value == "regular";
          }))
        form = "indexed";
      if (field && laneTraversesRecords(field, operation.getResult().getType()))
        form = "strided";
      access = makeAccess(builder, form, access.getMapping(),
                          access.getAlignment(), access.getGroupSize(),
                          access.getLayerSize(), access.getJoinFields(),
                          access.getJoinLowBits(), access.getJoinRole(),
                          access.getBitOffset(), access.getStorageBits(),
                          access.getOrder());
      if (mlir::failed(verifyAccessCapability(operation, access))) {
        failed = true;
        return;
      }
      operation.setAccessAttr(access);
      operation.setLeafAttr(transferLeaf(
          builder, "extract", ("rvv.extract." + form).str()));
    });

    // A direct logical-u1 field may already carry the complete logical axis;
    // unlike the gather form below, there is no separate projected index from
    // which to recover a window.  The grouped/layered encoding nevertheless
    // defines one exact byte-mask window.  Materialize that physical access
    // before the generic layout conversion so emission never has to recover
    // packed storage geometry from a vector value.
    llvm::SmallVector<riscv::ConvertLayoutOp> directBitmaskConversions;
    getOperation().walk([&](riscv::ConvertLayoutOp conversion) {
      directBitmaskConversions.push_back(conversion);
    });
    for (riscv::ConvertLayoutOp conversion : directBitmaskConversions) {
      if (!conversion || !conversion->getBlock() ||
          conversion.getConversion().getEffect() != "pure")
        continue;
      auto field = conversion.getInput().getDefiningOp<riscv::FieldOp>();
      auto source = field ? mlir::dyn_cast<riscv::ValueType>(field.getResult().getType())
                          : riscv::ValueType();
      auto result = mlir::dyn_cast<riscv::ValueType>(conversion.getResult().getType());
      auto sourceElement = source
                               ? mlir::dyn_cast<mlir::IntegerType>(
                                     source.getElementType())
                               : mlir::IntegerType();
      auto resultElement = result
                               ? mlir::dyn_cast<mlir::IntegerType>(
                                     result.getElementType())
                               : mlir::IntegerType();
      const auto facts = field ? riscv_internal::fieldFacts(field)
                               : riscv_internal::FieldFacts();
      if (!field || !source || !result || !sourceElement ||
          sourceElement.isSigned() || sourceElement.getWidth() != 1 ||
          !resultElement || resultElement.isSigned() ||
          resultElement.getWidth() != 1 ||
          (source.getLayout().getCarrier() != "local" &&
           source.getLayout().getCarrier() != "scalar") ||
          result.getLayout().getCarrier() != "rvv" ||
          source.getShape() != result.getShape() ||
          source.getAxisIds() != result.getAxisIds() ||
          facts.mapping != "grouped_layered" || facts.group <= 0 ||
          facts.layer <= 0 || facts.group != facts.layer * 8 ||
          facts.order != "lo_first" || facts.bitOffset % 8)
        continue;

      std::optional<size_t> windowPosition;
      for (size_t position = 0; position < result.getShape().size(); ++position) {
        const int64_t extent = result.getShape()[position];
        const int64_t time = result.getLayout().getTimeFactors()[position];
        const int64_t lane = result.getLayout().getLaneFactors()[position];
        const int64_t replica = result.getLayout().getReplicaFactors()[position];
        const bool window = extent > 1 && time > 0 && lane > 0 && replica > 0 &&
                            time * lane * replica == extent &&
                            result.getLayout().getFragmentFactors()[position] == 1 &&
                            result.getLayout().getLocalFactors()[position] == 1;
        if (window) {
          if (windowPosition) {
            windowPosition.reset();
            break;
          }
          windowPosition = position;
          continue;
        }
        if (result.getLayout().getTimeFactors()[position] != 1 ||
            result.getLayout().getLaneFactors()[position] != 1 ||
            result.getLayout().getFragmentFactors()[position] != 1 ||
            result.getLayout().getLocalFactors()[position] != 1) {
          windowPosition.reset();
          break;
        }
      }
      if (!windowPosition ||
          result.getShape()[*windowPosition] % facts.group)
        continue;

      // `vlm` starts at a byte address.  If the consumer layout slices one
      // packed group into sub-byte issue windows (for example four lanes of a
      // grouped(8) field), select a byte-aligned load carrier here and leave a
      // typed layout conversion to the consumer representation.  Advancing the
      // pointer by `partOffset / 8` for a four-bit slice would otherwise reread
      // the low half of the same byte.
      const size_t position = *windowPosition;
      auto kernel = conversion->getParentOfType<riscv::KernelOp>();
      const int64_t windowAxis = source.getAxisIds()[position];
      const int64_t windowExtent = source.getShape()[position];
      auto selectedLoadResult = kernel ? byteAlignedBitmaskLoadType(
                                            result, {windowAxis}, {windowExtent},
                                            facts,
                                            kernel.getTarget(), builder)
                                       : std::nullopt;
      if (!selectedLoadResult)
        continue;
      riscv::ValueType loadResult = *selectedLoadResult;

      rewriter.setInsertionPoint(conversion);
      auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(
          conversion.getLoc(), 0);
      const bool tail = result.getLayout().getValidity() == "tail";
      const int64_t temporaryGroups = std::max<int64_t>(
          1, (loadResult.getLayout().getLmulEighths() + 7) / 8);
      auto partBitOffsets = riscv::bitmaskWindowPartOffsets(
          loadResult, {windowAxis}, {windowExtent});
      if (!partBitOffsets)
        continue;
      auto window = rewriter.create<riscv::RVVBitmaskWindowLoadOp>(
          conversion.getLoc(), loadResult, field.getResult(), zero,
          source.getAxisIds()[*windowPosition],
          rewriter.getDenseI64ArrayAttr(
              {source.getAxisIds()[*windowPosition]}),
          rewriter.getDenseI64ArrayAttr(
              {source.getShape()[*windowPosition]}),
          rewriter.getDenseI64ArrayAttr(*partBitOffsets),
          makeAccess(builder, "unit", facts.mapping, facts.alignment,
                     facts.group, facts.layer, facts.joinFields,
                     facts.joinLowBits, facts.joinRole, facts.bitOffset,
                     facts.storageBits, facts.order),
          riscv_internal::leaf(
              builder, "rvv", "bitmask-window-load",
              "rvv.bitmask-window-load", "rvv.bitmask-window-load", 0,
              loadResult.getLayout().getRegisterGroups(), temporaryGroups, 0,
              "none", tail ? "agnostic" : "exact"));
      riscv_internal::copyOrigin(conversion, window);
      mlir::Value replacement = window.getResult();
      if (loadResult != result) {
        auto bridge = rewriter.create<riscv::ConvertLayoutOp>(
            conversion.getLoc(), result, replacement,
            riscv_internal::layoutConversion(rewriter,
                                             loadResult.getLayout(),
                                             result.getLayout()),
            riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
        riscv_internal::copyOrigin(conversion, bridge);
        replacement = bridge.getResult();
      }
      conversion.getResult().replaceAllUsesWith(replacement);
      rewriter.eraseOp(conversion);
    }
    }

    llvm::SmallVector<riscv::ExtractOp> entryExtracts;
    getOperation().walk(
        [&](riscv::ExtractOp extract) { entryExtracts.push_back(extract); });
    for (riscv::ExtractOp extract : entryExtracts) {
      auto source = mlir::dyn_cast<riscv::ValueType>(extract.getInput().getType());
      auto result = mlir::dyn_cast<riscv::ValueType>(extract.getResult().getType());
      if (!source || !result ||
          (source.getLayout().getCarrier() != "local" &&
           source.getLayout().getCarrier() != "scalar") ||
          extract.getSelectors().size() != source.getAxisIds().size())
        continue;

      auto sourceField = riscv_internal::sourceField(extract.getInput());
      const auto sourceFacts =
          sourceField ? riscv_internal::fieldFacts(sourceField)
                      : riscv_internal::FieldFacts();
      int64_t sourceAlignment = sourceFacts.alignment;
      int64_t sourceBitOffset = sourceFacts.bitOffset;
      if (!sourceField)
        if (auto load = riscv_internal::sourceLoad(extract.getInput())) {
          sourceAlignment = load.getRegion().getType().getAlignment();
          sourceBitOffset = 0;
        }
      riscv::ConvertLayoutOp domainResultConversion;
      riscv::ValueType domainResult = result;
      if (result.getLayout().getCarrier() == "scalar" &&
          extract.getResult().hasOneUse()) {
        auto user = *extract.getResult().getUsers().begin();
        auto conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(user);
        auto converted = conversion
                             ? mlir::dyn_cast<riscv::ValueType>(
                                   conversion.getResult().getType())
                             : riscv::ValueType();
        if (conversion && converted &&
            conversion.getConversion().getEffect() == "pure" &&
            conversion.getConversion().getKind() == "time_to_lane" &&
            converted.getLayout().getCarrier() == "rvv") {
          domainResultConversion = conversion;
          domainResult = converted;
        }
      }
      auto resultElement =
          mlir::dyn_cast<mlir::IntegerType>(domainResult.getElementType());
      size_t domainCursor = 0;
      size_t domainDimension = source.getAxisIds().size();
      riscv::PhysicalPointOp domainPoint;
      bool onlyRetainOrDomain = true;
      for (auto [dimension, selectorAttribute] :
           llvm::enumerate(extract.getSelectors())) {
        llvm::StringRef selector =
            mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
        if (selector == "all")
          continue;
        if (selector != "domain" || domainPoint ||
            domainCursor >= extract.getIndices().size()) {
          onlyRetainOrDomain = false;
          break;
        }
        domainDimension = dimension;
        domainPoint = extract.getIndices()[domainCursor++]
                          .getDefiningOp<riscv::PhysicalPointOp>();
      }
      if (onlyRetainOrDomain && domainPoint &&
          domainCursor == extract.getIndices().size() && sourceField &&
          resultElement && !resultElement.isSigned() &&
          resultElement.getWidth() == 1 &&
          sourceFacts.mapping == "grouped_layered" &&
          sourceFacts.group > 0 && sourceFacts.layer > 0 &&
          sourceFacts.group == sourceFacts.layer * 8 &&
          sourceFacts.order == "lo_first" && sourceFacts.bitOffset % 8 == 0) {
        llvm::SmallVector<int64_t> retainedAxes;
        llvm::SmallVector<int64_t> retainedShape;
        for (size_t position = 0; position < source.getAxisIds().size();
             ++position) {
          if (position == domainDimension)
            continue;
          retainedAxes.push_back(source.getAxisIds()[position]);
          retainedShape.push_back(source.getShape()[position]);
        }
        auto resultAxes = domainResult.getAxisIds().asArrayRef();
        auto resultShape = domainResult.getShape().asArrayRef();
        if (resultAxes.size() >= retainedAxes.size() &&
            resultShape.size() >= retainedShape.size() &&
            llvm::equal(retainedAxes,
                        resultAxes.take_front(retainedAxes.size())) &&
            llvm::equal(retainedShape,
                        resultShape.take_front(retainedShape.size()))) {
          llvm::SmallVector<int64_t> windowAxes(
              resultAxes.drop_front(retainedAxes.size()).begin(),
              resultAxes.drop_front(retainedAxes.size()).end());
          llvm::SmallVector<int64_t> windowExtents(
              resultShape.drop_front(retainedShape.size()).begin(),
              resultShape.drop_front(retainedShape.size()).end());
          rewriter.setInsertionPoint(extract);
          auto kernel = extract->getParentOfType<riscv::KernelOp>();
          auto selectedLoadResult =
              kernel
                  ? byteAlignedBitmaskLoadType(
                        domainResult, windowAxes, windowExtents,
                        sourceFacts, kernel.getTarget(), builder)
                  : std::nullopt;
          auto partBitOffsets =
              selectedLoadResult
                  ? riscv::bitmaskWindowPartOffsets(
                        *selectedLoadResult, windowAxes, windowExtents)
                  : std::nullopt;
          if (selectedLoadResult && partBitOffsets &&
              supportsBitmaskWindowLayout(
                  sourceField, *selectedLoadResult, retainedAxes.size(),
                  windowAxes, windowExtents)) {
            auto byteBase = materializeDomainBitmaskByteBase(
                sourceField, domainPoint, sourceFacts, extract, rewriter);
            if (!byteBase)
              continue;
            riscv::ValueType loadResult = *selectedLoadResult;
            const bool tail = loadResult.getLayout().getValidity() == "tail";
            const int64_t temporaryGroups = std::max<int64_t>(
                1, (loadResult.getLayout().getLmulEighths() + 7) / 8);
            auto window = rewriter.create<riscv::RVVBitmaskWindowLoadOp>(
                extract.getLoc(), loadResult, extract.getInput(), *byteBase,
                source.getAxisIds()[domainDimension],
                rewriter.getDenseI64ArrayAttr(windowAxes),
                rewriter.getDenseI64ArrayAttr(windowExtents),
                rewriter.getDenseI64ArrayAttr(*partBitOffsets),
                makeAccess(builder, "unit", sourceFacts.mapping,
                           sourceFacts.alignment, sourceFacts.group,
                           sourceFacts.layer, sourceFacts.joinFields,
                           sourceFacts.joinLowBits, sourceFacts.joinRole,
                           sourceFacts.bitOffset, sourceFacts.storageBits,
                           sourceFacts.order),
                riscv_internal::leaf(
                    builder, "rvv", "bitmask-window-load",
                    "rvv.bitmask-window-load", "rvv.bitmask-window-load", 0,
                    loadResult.getLayout().getRegisterGroups(), temporaryGroups,
                    0,
                    "none", tail ? "agnostic" : "exact"));
            riscv_internal::copyOrigin(extract, window);
            mlir::Value replacement = window.getResult();
            if (loadResult != domainResult) {
              auto bridge = rewriter.create<riscv::ConvertLayoutOp>(
                  extract.getLoc(), domainResult, replacement,
                  riscv_internal::layoutConversion(
                      rewriter, loadResult.getLayout(),
                      domainResult.getLayout()),
                  riscv::AccessAttr(),
                  riscv_internal::unselectedLeaf(rewriter));
              riscv_internal::copyOrigin(extract, bridge);
              replacement = bridge.getResult();
            }
            if (domainResultConversion) {
              domainResultConversion.getResult().replaceAllUsesWith(replacement);
              rewriter.eraseOp(domainResultConversion);
            } else {
              extract.getResult().replaceAllUsesWith(replacement);
            }
            rewriter.eraseOp(extract);
            continue;
          }
        }
      }

      if (nestedDomainOnly)
        continue;

      size_t indexCursor = 0;
      size_t gatherDimension = source.getAxisIds().size();
      mlir::Value gatherIndex;
      bool onlyRetainOrGather = true;
      for (auto [dimension, selectorAttribute] :
           llvm::enumerate(extract.getSelectors())) {
        llvm::StringRef selector =
            mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
        if (selector == "all")
          continue;
        if (selector != "gather" || gatherIndex ||
            indexCursor >= extract.getIndices().size()) {
          onlyRetainOrGather = false;
          break;
        }
        gatherDimension = dimension;
        gatherIndex = extract.getIndices()[indexCursor++];
      }
      if (!onlyRetainOrGather || !gatherIndex ||
          indexCursor != extract.getIndices().size())
        continue;
      llvm::SmallVector<int64_t> retainedAxes;
      llvm::SmallVector<int64_t> retainedShape;
      for (size_t position = 0; position < source.getAxisIds().size(); ++position) {
        if (position == gatherDimension)
          continue;
        retainedAxes.push_back(source.getAxisIds()[position]);
        retainedShape.push_back(source.getShape()[position]);
      }
      rewriter.setInsertionPoint(extract);
      auto sourceElement =
          mlir::dyn_cast<mlir::IntegerType>(result.getElementType());
      if (sourceField && sourceElement && !sourceElement.isSigned() &&
          sourceElement.getWidth() == 1 &&
          sourceFacts.mapping == "grouped_layered" &&
          sourceFacts.group > 0 && sourceFacts.layer > 0 &&
          sourceFacts.group == sourceFacts.layer * 8) {
        auto bitmaskWindow = materializeAffineWindowBase(
            gatherIndex, result, retainedAxes.size(), sourceFacts.group,
            sourceFacts.layer, extract, rewriter);
        auto kernel = extract->getParentOfType<riscv::KernelOp>();
        auto selectedLoadResult =
            bitmaskWindow && kernel
                ? byteAlignedBitmaskLoadType(
                      result, bitmaskWindow->axes, bitmaskWindow->extents,
                      sourceFacts, kernel.getTarget(), builder)
                : std::nullopt;
        if (bitmaskWindow && selectedLoadResult &&
            supportsBitmaskWindowLayout(
                sourceField, *selectedLoadResult, retainedAxes.size(),
                bitmaskWindow->axes, bitmaskWindow->extents)) {
          riscv::ValueType loadResult = *selectedLoadResult;
          auto partBitOffsets = riscv::bitmaskWindowPartOffsets(
              loadResult, bitmaskWindow->axes, bitmaskWindow->extents);
          if (!partBitOffsets)
            continue;
          const bool tail = loadResult.getLayout().getValidity() == "tail";
          const int64_t temporaryGroups = std::max<int64_t>(
              1, (loadResult.getLayout().getLmulEighths() + 7) / 8);
          auto window = rewriter.create<riscv::RVVBitmaskWindowLoadOp>(
              extract.getLoc(), loadResult, extract.getInput(),
              bitmaskWindow->base, source.getAxisIds()[gatherDimension],
              rewriter.getDenseI64ArrayAttr(bitmaskWindow->axes),
              rewriter.getDenseI64ArrayAttr(bitmaskWindow->extents),
              rewriter.getDenseI64ArrayAttr(*partBitOffsets),
              makeAccess(builder, "unit", sourceFacts.mapping,
                         sourceFacts.alignment, sourceFacts.group,
                         sourceFacts.layer, sourceFacts.joinFields,
                         sourceFacts.joinLowBits, sourceFacts.joinRole,
                         sourceFacts.bitOffset, sourceFacts.storageBits,
                         sourceFacts.order),
              riscv_internal::leaf(
                  builder, "rvv", "bitmask-window-load",
                  "rvv.bitmask-window-load", "rvv.bitmask-window-load", 0,
                  loadResult.getLayout().getRegisterGroups(), temporaryGroups, 0,
                  "none", tail ? "agnostic" : "exact"));
          riscv_internal::copyOrigin(extract, window);
          mlir::Value replacement = window.getResult();
          if (loadResult != result) {
            auto bridge = rewriter.create<riscv::ConvertLayoutOp>(
                extract.getLoc(), result, replacement,
                riscv_internal::layoutConversion(
                    rewriter, loadResult.getLayout(), result.getLayout()),
                riscv::AccessAttr(), riscv_internal::unselectedLeaf(rewriter));
            riscv_internal::copyOrigin(extract, bridge);
            replacement = bridge.getResult();
          }
          mlir::Value fullIndices = gatherIndex;
          extract.getResult().replaceAllUsesWith(replacement);
          rewriter.eraseOp(extract);
          llvm::DenseSet<mlir::Operation *> visited;
          eraseDeadRegularIndexChain(fullIndices, visited, rewriter);
          continue;
        }
      }
      auto directWindow = materializeAffineUnitWindowBase(
          gatherIndex, result, retainedAxes.size(), extract, rewriter);
      const auto sourceAxes = source.getAxisIds().asArrayRef();
      const int64_t sourceAxis = source.getAxisIds()[gatherDimension];
      auto unitEntryAxesMatchSource = [&](llvm::ArrayRef<int64_t> entryAxes,
                                          int64_t payloadAxis) {
        // Entry coordinates are new logical axes.  The payload coordinate may
        // either be new as well, or rebind the one source axis being projected:
        // [entry, payload] -> entry * payload_extent + payload.  In the latter
        // case it still denotes a unit contiguous suffix, not an indexed load.
        return llvm::none_of(entryAxes, [&](int64_t axis) {
                 return llvm::is_contained(sourceAxes, axis);
               }) &&
               (!llvm::is_contained(sourceAxes, payloadAxis) ||
                payloadAxis == sourceAxis);
      };
      if (directWindow && sourceField &&
          unitEntryAxesMatchSource(
              directWindow->entryAxes, result.getAxisIds().asArrayRef().back()) &&
          supportsUnitEntryWindowLayout(
              result, retainedAxes.size(), directWindow->entryAxes,
              directWindow->entryExtents,
              result.getAxisIds().asArrayRef().back(),
              result.getShape().asArrayRef().back())) {
        auto window = rewriter.create<riscv::RVVUnitEntryWindowLoadOp>(
            extract.getLoc(), result, extract.getInput(), directWindow->base,
            source.getAxisIds()[gatherDimension],
            rewriter.getDenseI64ArrayAttr(directWindow->entryAxes),
            rewriter.getDenseI64ArrayAttr(directWindow->entryExtents),
            result.getAxisIds().asArrayRef().back(),
            result.getShape().asArrayRef().back(),
            result.getShape().asArrayRef().back(),
            makeAccess(builder, "unit", "natural", 1),
            riscv_internal::leaf(
                builder, "rvv", "unit-entry-window-load",
                "rvv.unit-entry-window-load", "rvv.unit-entry-window-load", 0,
                result.getLayout().getRegisterGroups()));
        riscv_internal::copyOrigin(extract, window);
        mlir::Value fullIndices = gatherIndex;
        extract.getResult().replaceAllUsesWith(window.getResult());
        rewriter.eraseOp(extract);
        llvm::DenseSet<mlir::Operation *> visited;
        eraseDeadRegularIndexChain(fullIndices, visited, rewriter);
        continue;
      }
      auto resultReplicas = riscv_internal::staticProduct(
          result.getLayout().getReplicaFactors().asArrayRef());
      // An affine byte-identity field window with register replicas is a
      // field-to-register relation, not an arbitrary indexed entry lookup.
      // Keep the extract until the replica-storage owner can materialize one
      // typed field-to-register relation for all replicas instead of issuing
      // an unrelated indexed entry load for each consumer part.
      if (directWindow && sourceField &&
          isNaturalByteIdentity(sourceFacts, source.getElementType()) &&
          result.getLayout().getCarrier() == "rvv" && resultReplicas &&
          *resultReplicas > 1)
        continue;
      auto plan = riscv_internal::analyzeIndexedEntryRelation(
          gatherIndex, result, retainedAxes, retainedShape);
      if (!plan)
        continue;
      auto entryIndices =
          materializeEntryIndices(*plan, extract, result, rewriter);
      if (mlir::failed(entryIndices)) {
        failed = true;
        continue;
      }
      auto unitWindow = materializeUnitEntryWindowBase(
          *entryIndices, result, plan->payloadAxis, plan->payloadExtent,
          plan->entryStride, extract, rewriter);
      const size_t retainedAxisCount = source.getAxisIds().size() - 1;
      if (unitWindow && sourceField &&
          unitEntryAxesMatchSource(unitWindow->entryAxes, plan->payloadAxis) &&
          supportsUnitEntryWindowLayout(
              result, retainedAxisCount, unitWindow->entryAxes,
              unitWindow->entryExtents, plan->payloadAxis,
              plan->payloadExtent)) {
        auto window = rewriter.create<riscv::RVVUnitEntryWindowLoadOp>(
            extract.getLoc(), result, extract.getInput(), unitWindow->base,
            source.getAxisIds()[gatherDimension],
            rewriter.getDenseI64ArrayAttr(unitWindow->entryAxes),
            rewriter.getDenseI64ArrayAttr(unitWindow->entryExtents),
            plan->payloadAxis, plan->payloadExtent,
            plan->entryStride, makeAccess(builder, "unit", "natural", 1),
            riscv_internal::leaf(
                builder, "rvv", "unit-entry-window-load",
                "rvv.unit-entry-window-load", "rvv.unit-entry-window-load", 0,
                result.getLayout().getRegisterGroups()));
        riscv_internal::copyOrigin(extract, window);
        mlir::Value fullIndices = gatherIndex;
        extract.getResult().replaceAllUsesWith(window.getResult());
        rewriter.eraseOp(extract);
        llvm::DenseSet<mlir::Operation *> visited;
        eraseDeadRegularIndexChain(fullIndices, visited, rewriter);
        continue;
      }
      mlir::Type entryType = (*entryIndices).getType();
      if (!supportsIndexedEntryLoad(*plan, entryType, result,
                                    sourceAlignment, sourceBitOffset)) {
        llvm::DenseSet<mlir::Operation *> visited;
        eraseDeadRegularIndexChain(*entryIndices, visited, rewriter);
        continue;
      }
      auto byteStride = entryByteStride(*plan, result);
      if (!byteStride) {
        extract.emitError(
            "indexed entry relation has no exact byte-address stride");
        failed = true;
        continue;
      }
      auto entryOffsets = materializeEntryByteOffsets(
          *entryIndices, *byteStride, extract, rewriter);
      if (mlir::failed(entryOffsets)) {
        failed = true;
        continue;
      }
      auto physicalEntryType = mlir::dyn_cast<riscv::ValueType>(entryType);
      const bool vectorEntries = physicalEntryType &&
                                 physicalEntryType.getLayout().getCarrier() ==
                                     "rvv";
      const IndexedEntryPhysicalChoice choice =
          selectIndexedEntryPhysicalChoice(builder, result, vectorEntries,
                                           sourceAlignment, *byteStride);
      auto entryLoad = rewriter.create<riscv::RVVIndexedEntryLoadOp>(
          extract.getLoc(), result, extract.getInput(), *entryOffsets,
          source.getAxisIds()[gatherDimension], plan->payloadAxis,
          plan->payloadExtent, *byteStride,
          choice.access, choice.leaf);
      riscv_internal::copyOrigin(extract, entryLoad);
      mlir::Value fullIndices = gatherIndex;
      extract.getResult().replaceAllUsesWith(entryLoad.getResult());
      rewriter.eraseOp(extract);
      llvm::DenseSet<mlir::Operation *> visited;
      eraseDeadRegularIndexChain(fullIndices, visited, rewriter);
    }

    if (nestedDomainOnly) {
      if (failed)
        signalPassFailure();
      return;
    }

    llvm::SmallVector<RegularGatherCandidate> regularCandidates;
    llvm::SmallVector<riscv::ExtractOp> extracts;
    getOperation().walk(
        [&](riscv::ExtractOp extract) { extracts.push_back(extract); });
    for (riscv::ExtractOp extract : extracts) {
      RegularGatherCandidate candidate;
      candidate.extract = extract;
      mlir::Value source = riscv_internal::stripRepresentationConversions(
          extract.getInput(), &candidate.inputConversions);
      candidate.field = source.getDefiningOp<riscv::FieldOp>();
      if (extract.getResult().hasOneUse())
        candidate.resultConversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(
            *extract.getResult().getUsers().begin());
      mlir::Type selectedResultType = candidate.resultConversion
                                          ? candidate.resultConversion.getResult().getType()
                                          : extract.getResult().getType();
      candidate.resultType =
          mlir::dyn_cast<riscv::ValueType>(selectedResultType);
      const llvm::StringRef fieldMapping =
          candidate.field ? candidate.field.getAccess().getMapping()
                          : llvm::StringRef();
      if (!candidate.field || !candidate.resultType ||
          extract.getAccess().getForm() != "indexed" ||
          (fieldMapping != "natural" && fieldMapping != "joined"))
        continue;

      size_t indexCursor = 0;
      size_t gatherDimension = extract.getSelectors().size();
      unsigned gatherCount = 0;
      bool onlyFreeAxes = true;
      bool alreadyRegular = false;
      llvm::SmallVector<mlir::Attribute> regularSelectors;
      for (auto [dimension, selectorAttribute] :
           llvm::enumerate(extract.getSelectors())) {
        llvm::StringRef selector =
            mlir::cast<mlir::StringAttr>(selectorAttribute).getValue();
        if (selector == "all") {
          regularSelectors.push_back(selectorAttribute);
          continue;
        }
        if (selector == "regular") {
          ++gatherCount;
          gatherDimension = dimension;
          alreadyRegular = true;
          regularSelectors.push_back(selectorAttribute);
          continue;
        }
        if (indexCursor >= extract.getIndices().size()) {
          onlyFreeAxes = false;
          break;
        }
        mlir::Value index = extract.getIndices()[indexCursor++];
        if (selector == "gather") {
          ++gatherCount;
          gatherDimension = dimension;
          candidate.index = index;
          regularSelectors.push_back(builder.getStringAttr("regular"));
        } else {
          onlyFreeAxes = false;
        }
      }
      std::optional<RegularIndex> pattern;
      if (onlyFreeAxes && gatherCount == 1 &&
          indexCursor == extract.getIndices().size()) {
        if (alreadyRegular) {
          auto attribute = extract->getAttrOfType<mlir::DenseI64ArrayAttr>(
              "index_pattern");
          if (attribute && attribute.size() == 3)
            pattern = RegularIndex{{}, attribute[0], attribute[1], attribute[2]};
        } else {
          pattern = riscv_internal::analyzeRegularIndexRelation(candidate.index);
        }
      }
      if (!pattern || pattern->base < 0 || pattern->stride != 1 ||
          pattern->repeat <= 0 ||
          gatherDimension >= candidate.resultType.getShape().size() ||
          gatherDimension >=
              candidate.resultType.getLayout().getLaneFactors().size())
        continue;

      auto input = mlir::dyn_cast<riscv::ValueType>(extract.getInput().getType());
      if (!input || gatherDimension >= input.getShape().size())
        continue;
      auto fieldType =
          mlir::dyn_cast<riscv::ValueType>(candidate.field.getResult().getType());
      if (!fieldType || gatherDimension >= fieldType.getAxisIds().size())
        continue;
      candidate.sourceAxis = fieldType.getAxisIds()[gatherDimension];
      // Replacing a shaped gather with a regular extract removes the explicit
      // index value, so the field itself must already carry the result axes.
      // A gather may legitimately introduce a new logical axis (for example a
      // typed table-entry coordinate).  That relation cannot become a bare
      // regular selector, but it can still become a typed regular-repeat
      // load/gather carrying distinct source and result axes.
      const bool preservesSourceAxes =
          fieldType.getShape().size() == candidate.resultType.getShape().size() &&
          fieldType.getAxisIds() == candidate.resultType.getAxisIds();
      bool preservesFreeAxes = true;
      if (preservesSourceAxes)
        for (size_t dimension = 0; dimension < fieldType.getShape().size();
             ++dimension)
          if (dimension != gatherDimension &&
              fieldType.getShape()[dimension] !=
                candidate.resultType.getShape()[dimension]) {
            preservesFreeAxes = false;
            break;
          }
      if (!preservesFreeAxes)
        continue;
      candidate.lanes =
          candidate.resultType.getLayout().getLaneFactors()[gatherDimension];
      candidate.reductionAxis = candidate.resultType.getAxisIds()[gatherDimension];
      const int64_t time =
          candidate.resultType.getLayout().getTimeFactors()[gatherDimension];
      auto totalTime = positiveProduct(
          candidate.resultType.getLayout().getTimeFactors().asArrayRef());
      if (candidate.lanes <= 1 || time <= 0 || !totalTime || *totalTime != time ||
          candidate.resultType.getShape()[gatherDimension] !=
              candidate.lanes * time ||
          (pattern->repeat % candidate.lanes &&
           candidate.lanes % pattern->repeat))
        continue;
      candidate.base = pattern->base;
      candidate.stride = pattern->stride;
      candidate.repeat = pattern->repeat;
      candidate.dynamicBase = pattern->dynamicBase;
      bool bounded = true;
      for (int64_t stream = 0; stream < time; ++stream) {
        const int64_t logicalOffset = stream * candidate.lanes;
        int64_t base = candidate.base +
                       (logicalOffset / candidate.repeat) * candidate.stride;
        int64_t last = base +
                       ((candidate.lanes - 1) / candidate.repeat) *
                           candidate.stride;
        bounded &= base >= 0 && last >= base &&
                   last < input.getShape()[gatherDimension];
        candidate.partBases.push_back(base);
      }
      if (!bounded)
        continue;
      if (!alreadyRegular && preservesSourceAxes && fieldMapping == "natural") {
        rewriter.setInsertionPoint(extract);
        auto replacement = rewriter.create<riscv::ExtractOp>(
            extract.getLoc(), selectedResultType, candidate.field.getResult(),
            mlir::ValueRange{}, rewriter.getArrayAttr(regularSelectors),
            extract.getAccess(), extract.getLeaf());
        replacement->setAttr(
            "index_pattern",
            rewriter.getDenseI64ArrayAttr(
                {pattern->base, pattern->stride, pattern->repeat}));
        riscv_internal::copyOrigin(extract, replacement);
        if (auto canonical = extract->getAttr("canonical_op"))
          replacement->setAttr("canonical_op", canonical);
        if (candidate.resultConversion) {
          candidate.resultConversion.getResult().replaceAllUsesWith(
              replacement.getResult());
          rewriter.eraseOp(candidate.resultConversion);
        } else {
          extract.getResult().replaceAllUsesWith(replacement.getResult());
        }
        rewriter.eraseOp(extract);
        for (riscv::ConvertLayoutOp inputConversion :
             llvm::reverse(candidate.inputConversions))
          if (inputConversion.getResult().use_empty())
            rewriter.eraseOp(inputConversion);
        llvm::DenseSet<mlir::Operation *> visited;
        eraseDeadRegularIndexChain(candidate.index, visited, rewriter);
        continue;
      }
      // A unit-repeat natural index is already a complete direct load form.
      // A joined field still needs a typed physical load because one logical
      // element is reconstructed from the joined byte geometry.
      if (candidate.repeat == 1 && fieldMapping == "natural")
        continue;
      candidate.resultConversion = {};
      candidate.inputConversions.clear();
      candidate.index = {};
      regularCandidates.push_back(std::move(candidate));
    }

    llvm::DenseSet<mlir::Operation *> consumedRegular;
    for (size_t candidateIndex = 0; candidateIndex < regularCandidates.size();
         ++candidateIndex) {
      RegularGatherCandidate &first = regularCandidates[candidateIndex];
      if (consumedRegular.contains(first.extract.getOperation()))
        continue;
      llvm::SmallVector<RegularGatherCandidate *> group{&first};
      consumedRegular.insert(first.extract.getOperation());
      for (size_t otherIndex = candidateIndex + 1;
           otherIndex < regularCandidates.size(); ++otherIndex) {
        RegularGatherCandidate &other = regularCandidates[otherIndex];
        if (consumedRegular.contains(other.extract.getOperation()) ||
            other.extract->getBlock() != first.extract->getBlock() ||
            other.field != first.field ||
            other.sourceAxis != first.sourceAxis ||
            other.repeat != first.repeat ||
            other.stride != first.stride || other.lanes != first.lanes ||
            other.dynamicBase != first.dynamicBase ||
            other.reductionAxis != first.reductionAxis ||
            other.resultType.getElementType() != first.resultType.getElementType() ||
            other.resultType.getLayout() != first.resultType.getLayout())
          continue;
        bool movable = true;
        for (mlir::Operation *cursor = first.extract->getNextNode(); cursor &&
             cursor != other.extract.getOperation(); cursor = cursor->getNextNode())
          movable &= canMoveReadBefore(cursor);
        if (!movable)
          continue;
        group.push_back(&other);
        consumedRegular.insert(other.extract.getOperation());
      }

      int64_t sourceBase = std::numeric_limits<int64_t>::max();
      int64_t sourceEnd = -1;
      for (RegularGatherCandidate *candidate : group)
        for (int64_t base : candidate->partBases) {
          sourceBase = std::min(sourceBase, base);
          sourceEnd = std::max(
              sourceEnd,
              base + (candidate->lanes - 1) / candidate->repeat);
        }
      const int64_t sourceCount = sourceEnd - sourceBase + 1;
      if (sourceBase < 0 || sourceCount <= 0 || sourceCount > first.lanes) {
        for (RegularGatherCandidate *candidate : group)
          consumedRegular.erase(candidate->extract.getOperation());
        continue;
      }

      llvm::SmallVector<mlir::Type> resultTypes;
      llvm::SmallVector<mlir::Attribute> partBases;
      for (RegularGatherCandidate *candidate : group) {
        resultTypes.push_back(candidate->resultType);
        llvm::SmallVector<int64_t> relative(candidate->partBases);
        for (int64_t &base : relative)
          base -= sourceBase;
        partBases.push_back(rewriter.getDenseI64ArrayAttr(relative));
      }
      const bool joined = first.field.getAccess().getMapping() == "joined";
      const int64_t temporaryGroups =
          (joined ? 3 : 2) *
          first.resultType.getLayout().getRegisterGroups();
      const bool powerOfTwo =
          first.repeat > 0 && (first.repeat & (first.repeat - 1)) == 0;
      const int64_t joinedGroup = first.field.getAccess().getGroupSize();
      const bool joinedGroupPowerOfTwo =
          joined && joinedGroup > 0 &&
          (joinedGroup & (joinedGroup - 1)) == 0;
      const llvm::StringRef instruction =
          joined && first.lanes > first.repeat
              ? joinedGroupPowerOfTwo
                    ? "rvv.regular-repeat-joined-gather.pow2"
                    : "rvv.regular-repeat-joined-gather.div"
          : joined ? "rvv.regular-repeat-joined-broadcast"
              : first.lanes <= first.repeat
                    ? "rvv.regular-repeat-broadcast"
                    : powerOfTwo ? "rvv.regular-repeat-gather.pow2"
                                 : "rvv.regular-repeat-gather.div";
      rewriter.setInsertionPoint(first.extract);
      mlir::Value sourceBaseValue = first.dynamicBase;
      if (!sourceBaseValue)
        sourceBaseValue = rewriter.create<mlir::arith::ConstantIndexOp>(
            first.extract.getLoc(), sourceBase);
      else if (sourceBase != 0) {
        mlir::Type element = sourceBaseValue.getType();
        mlir::TypedAttr constant;
        if (element.isIndex())
          constant = rewriter.getIndexAttr(sourceBase);
        else if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element))
          constant = rewriter.getIntegerAttr(integer, sourceBase);
        if (!constant) {
          for (RegularGatherCandidate *candidate : group)
            consumedRegular.erase(candidate->extract.getOperation());
          continue;
        }
        auto offset = rewriter.create<riscv::ConstantOp>(
            first.extract.getLoc(), element, constant);
        sourceBaseValue = rewriter.create<riscv::BinaryOp>(
            first.extract.getLoc(), element, sourceBaseValue,
            offset.getResult(), "add", riscv_internal::unselectedLeaf(rewriter));
      }

      llvm::SmallVector<mlir::Value> indices;
      if (first.lanes > first.repeat) {
        llvm::SmallVector<mlir::Type> indexTypes;
        int64_t indexGroups = 0;
        for (mlir::Type resultType : resultTypes) {
          auto indexType = repeatIndexType(
              rewriter, mlir::cast<riscv::ValueType>(resultType));
          if (!indexType) {
            for (RegularGatherCandidate *candidate : group)
              consumedRegular.erase(candidate->extract.getOperation());
            indexTypes.clear();
            break;
          }
          indexTypes.push_back(indexType);
          indexGroups += indexType.getLayout().getRegisterGroups();
        }
        if (indexTypes.empty())
          continue;
        if (auto reductionLoop =
                first.extract->getParentOfType<mlir::scf::ForOp>())
          rewriter.setInsertionPoint(reductionLoop);
        else
          rewriter.setInsertionPoint(first.extract);
        llvm::StringRef indexInstruction =
            powerOfTwo ? "rvv.regular-repeat-index.pow2"
                       : "rvv.regular-repeat-index.div";
        auto indexOp = rewriter.create<riscv::RVVRegularRepeatIndexOp>(
            first.extract.getLoc(), indexTypes, first.reductionAxis,
            first.repeat, rewriter.getArrayAttr(partBases),
            riscv_internal::leaf(rewriter, "rvv", "regular-repeat-index",
                                 indexInstruction, indexInstruction, 0,
                                 indexGroups, 1, 0, "none", "exact",
                                 {first.reductionAxis, first.repeat}));
        riscv_internal::copyOrigin(first.extract, indexOp);
        indices.append(indexOp.getResults().begin(), indexOp.getResults().end());
      }
      rewriter.setInsertionPoint(first.extract);
      auto gather = rewriter.create<riscv::RVVRegularRepeatGatherOp>(
          first.extract.getLoc(), resultTypes, first.field.getResult(),
          indices, first.sourceAxis, first.reductionAxis, sourceBaseValue,
          sourceCount, first.repeat,
          rewriter.getArrayAttr(partBases), first.field.getAccess(),
          riscv_internal::leaf(
              rewriter, "rvv", "regular-repeat-gather",
              instruction, instruction,
              mlir::cast<riscv::ValueType>(first.field.getResult().getType())
                  .getLayout()
                  .getRegisterGroups(),
              0, temporaryGroups, 0, "none", "exact",
              {first.sourceAxis, first.reductionAxis, sourceCount,
               first.repeat}));
      riscv_internal::copyOrigin(first.extract, gather);

      for (auto [index, candidate] : llvm::enumerate(group)) {
        mlir::Value replacement = gather.getResults()[index];
        if (candidate->resultConversion) {
          candidate->resultConversion.getResult().replaceAllUsesWith(replacement);
          rewriter.eraseOp(candidate->resultConversion);
        } else {
          candidate->extract.getResult().replaceAllUsesWith(replacement);
        }
        rewriter.eraseOp(candidate->extract);
        for (riscv::ConvertLayoutOp inputConversion :
             llvm::reverse(candidate->inputConversions))
          if (inputConversion.getResult().use_empty())
            rewriter.eraseOp(inputConversion);
        if (candidate->index) {
          llvm::DenseSet<mlir::Operation *> visited;
          eraseDeadRegularIndexChain(candidate->index, visited, rewriter);
        }
      }
    }

    llvm::SmallVector<riscv::RVVRegularRepeatGatherOp> repeatedGathers;
    getOperation().walk([&](riscv::RVVRegularRepeatGatherOp gather) {
      repeatedGathers.push_back(gather);
    });
    for (riscv::RVVRegularRepeatGatherOp gather : repeatedGathers) {
      if (gather.getResults().size() != 1 || !gather.getIndices().empty())
        continue;
      mlir::Value root = gather.getResults().front();
      riscv::ConvertLayoutOp conversion;
      if (root.hasOneUse()) {
        conversion = mlir::dyn_cast<riscv::ConvertLayoutOp>(
            *root.getUsers().begin());
        if (conversion && conversion.getConversion().getEffect() == "pure")
          root = conversion.getResult();
        else
          conversion = {};
      }
      auto targetType = mlir::dyn_cast<riscv::ValueType>(root.getType());
      if (!targetType || targetType.getLayout().getCarrier() != "rvv")
        continue;
      auto axis = llvm::find(targetType.getAxisIds().asArrayRef(),
                             gather.getReductionAxis());
      if (axis == targetType.getAxisIds().asArrayRef().end())
        continue;
      const size_t axisPosition = static_cast<size_t>(
          axis - targetType.getAxisIds().asArrayRef().begin());
      const int64_t lanes =
          targetType.getLayout().getLaneFactors()[axisPosition];
      const int64_t axisTime =
          targetType.getLayout().getTimeFactors()[axisPosition];
      auto timeParts = positiveProduct(
          targetType.getLayout().getTimeFactors().asArrayRef());
      auto replicas = positiveProduct(
          targetType.getLayout().getReplicaFactors().asArrayRef());
      if (lanes <= 0 || lanes > gather.getRepeat() ||
          gather.getRepeat() % lanes || !timeParts || !replicas ||
          *timeParts != axisTime)
        continue;
      auto usePlan = planScalarRepeatUses(root);
      auto scalarType = repeatedScalarSupplyType(rewriter, targetType);
      if (!usePlan || !scalarType)
        continue;
      llvm::SmallVector<int64_t> partBases;
      partBases.reserve(*replicas * *timeParts);
      bool inBounds = true;
      for (int64_t replica = 0; replica < *replicas; ++replica)
        for (int64_t stream = 0; stream < *timeParts; ++stream) {
          const int64_t base = stream * lanes / gather.getRepeat();
          inBounds &= base >= 0 && base < gather.getSourceCount();
          partBases.push_back(base);
        }
      if (!inBounds || !materializeScalarRepeatUse(*usePlan, rewriter))
        continue;
      rewriter.setInsertionPoint(gather);
      auto scalarLoad = rewriter.create<riscv::RVVRegularRepeatScalarLoadOp>(
          gather.getLoc(), scalarType, gather.getField(), gather.getSourceAxis(),
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
      root.replaceAllUsesWith(scalarLoad.getResult());
      if (conversion)
        rewriter.eraseOp(conversion);
      rewriter.eraseOp(gather);
    }

    getOperation().walk([&](riscv::ConvertLayoutOp operation) {
      auto result = mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
      if (!result || result.getLayout().getCarrier() != "rvv")
        return;

      riscv::AccessAttr access;
      mlir::Value source = riscv_internal::stripRepresentationConversions(
          operation.getInput());
      if (auto extract = source.getDefiningOp<riscv::ExtractOp>()) {
        if (!riscv_internal::sourceField(extract.getInput()))
          return;
        access = extract.getAccess();
      } else if (auto field = source.getDefiningOp<riscv::FieldOp>()) {
        access = fieldAccess(builder, field, operation.getResult().getType());
      } else {
        return;
      }
      if (access.getForm() != "unit" && access.getForm() != "strided" &&
          access.getForm() != "indexed" && access.getForm() != "segment")
        return;
      if (mlir::failed(verifyAccessCapability(operation, access))) {
        failed = true;
        return;
      }
      operation->setAttr("source_access", access);
    });
    getOperation().walk([&](riscv::Fold2Op operation) {
      auto field = riscv_internal::sourceField(operation.getInput());
      if (!field || field.getAccess().getMapping() != "natural" ||
          field.getAccess().getBitOffset() % 8 ||
          riscv_internal::logicalBitWidth(operation.getInput().getType()) != 16) {
        operation.emitError(
            "pair fold requires one byte-aligned natural i16 encoded field edge");
        failed = true;
        return;
      }
      operation.setAccessAttr(field.getAccess());
    });
    getOperation().walk([&](riscv::UpdateOp operation) {
      auto inputLayout = riscv_internal::layoutOf(operation.getInput().getType());
      if (!inputLayout || inputLayout.getCarrier() != "local")
        return;
      operation.setLeafAttr(
          transferLeaf(builder, "local-update", "local.update"));
    });
    llvm::SmallVector<riscv::LookupOp> entryLookups;
    getOperation().walk(
        [&](riscv::LookupOp operation) { entryLookups.push_back(operation); });
    for (riscv::LookupOp operation : entryLookups) {
      auto table = mlir::dyn_cast<riscv::MemDescType>(
          operation.getTable().getType());
      riscv::LoadOp tableLoad;
      if (!table) {
        tableLoad = operation.getTable().getDefiningOp<riscv::LoadOp>();
        if (tableLoad)
          table = mlir::dyn_cast<riscv::MemDescType>(
              tableLoad.getRegion().getType());
      }
      if (tableLoad && riscv_internal::readCrossesWrite(tableLoad, operation))
        continue;
      auto resultType =
          mlir::dyn_cast<riscv::ValueType>(operation.getResult().getType());
      auto plan = table && resultType
                      ? riscv_internal::analyzeIndexedEntryRelation(
                            operation.getIndices(), resultType, {}, {})
                      : std::optional<IndexedEntryLoadPlan>();
      if (!plan)
        continue;
      int64_t sourceAxis = 0;
      if (table.getAxisIds().size() == 1) {
        sourceAxis = table.getAxisIds()[0];
      } else if (table.getAxisIds().size() == 2 &&
                 table.getShape()[1] == plan->payloadExtent &&
                 table.getStrides()[0] == plan->entryStride &&
                 table.getStrides()[1] == 1) {
        sourceAxis = table.getAxisIds()[0];
      } else {
        continue;
      }
      int64_t alignment = table.getAlignment();
      int64_t bitOffset = 0;
      if (auto field = tableLoad
                           ? tableLoad.getRegion().getDefiningOp<riscv::FieldOp>()
                           : operation.getTable().getDefiningOp<riscv::FieldOp>()) {
        auto facts = riscv_internal::fieldFacts(field);
        alignment = facts.alignment;
        bitOffset = facts.bitOffset;
      }
      mlir::Value fullIndices = operation.getIndices();
      rewriter.setInsertionPoint(operation);
      auto entryIndices =
          materializeEntryIndices(*plan, operation, resultType, rewriter);
      if (mlir::failed(entryIndices)) {
        failed = true;
        continue;
      }
      mlir::Type entryType = (*entryIndices).getType();
      if (!supportsIndexedEntryLoad(*plan, entryType, resultType, alignment,
                                    bitOffset)) {
        llvm::DenseSet<mlir::Operation *> visited;
        eraseDeadRegularIndexChain(*entryIndices, visited, rewriter);
        continue;
      }
      auto byteStride = entryByteStride(*plan, resultType);
      if (!byteStride) {
        operation.emitError(
            "indexed entry relation has no exact byte-address stride");
        failed = true;
        continue;
      }
      auto entryOffsets = materializeEntryByteOffsets(
          *entryIndices, *byteStride, operation, rewriter);
      if (mlir::failed(entryOffsets)) {
        failed = true;
        continue;
      }
      auto physicalEntryType = mlir::dyn_cast<riscv::ValueType>(entryType);
      const bool vectorEntries = physicalEntryType &&
                                 physicalEntryType.getLayout().getCarrier() ==
                                     "rvv";
      const IndexedEntryPhysicalChoice choice =
          selectIndexedEntryPhysicalChoice(builder, resultType, vectorEntries,
                                           alignment, *byteStride);
      auto entryLoad = rewriter.create<riscv::RVVIndexedEntryLoadOp>(
          operation.getLoc(), resultType,
          tableLoad ? tableLoad.getRegion() : operation.getTable(), *entryOffsets,
          sourceAxis, plan->payloadAxis,
          plan->payloadExtent,
          *byteStride, choice.access, choice.leaf);
      riscv_internal::copyOrigin(operation, entryLoad);
      operation.getResult().replaceAllUsesWith(entryLoad.getResult());
      rewriter.eraseOp(operation);
      if (tableLoad && !llvm::is_contained(deadTableLoads, tableLoad))
        deadTableLoads.push_back(tableLoad);
      llvm::DenseSet<mlir::Operation *> visited;
      eraseDeadRegularIndexChain(fullIndices, visited, rewriter);
    }

    getOperation().walk([&](riscv::LookupOp operation) {
      // An unmaterialized admitted table remains an addressable memory edge.
      // Explicit materialize carries cross-use lifetime, so its selected RVV
      // value is consumed by a register gather instead of being reconstructed
      // as an indexed memory access at every use.
      auto tableLoad = operation.getTable().getDefiningOp<riscv::LoadOp>();
      auto tableMaterialize =
          operation.getTable().getDefiningOp<riscv::MaterializeOp>();
      if (!tableLoad && tableMaterialize)
        tableLoad =
            tableMaterialize.getInput().getDefiningOp<riscv::LoadOp>();
      auto indexLayout = riscv_internal::layoutOf(operation.getIndices().getType());
      bool indexed = indexLayout && indexLayout.getCarrier() == "rvv";
      auto tableLayout = riscv_internal::layoutOf(operation.getTable().getType());
      auto resultLayout = riscv_internal::layoutOf(operation.getResult().getType());
      auto tableParts = tableLayout
                            ? riscv_internal::staticProduct(
                                  tableLayout.getTimeFactors().asArrayRef())
                            : std::optional<int64_t>();
      auto tableReplicas = tableLayout
                               ? riscv_internal::staticProduct(
                                     tableLayout.getReplicaFactors().asArrayRef())
                               : std::optional<int64_t>();
      bool registerTable =
          indexed && tableLayout && resultLayout &&
          tableLayout.getCarrier() == "rvv" &&
          tableLayout.getSew() == resultLayout.getSew() &&
          tableLayout.getLmulEighths() == resultLayout.getLmulEighths() &&
          indexLayout.getSew() == resultLayout.getSew() &&
          indexLayout.getLmulEighths() == resultLayout.getLmulEighths() &&
          tableParts && *tableParts == 1 && tableReplicas &&
          *tableReplicas == 1;
      if (!registerTable && tableLoad) {
        if (riscv_internal::readCrossesWrite(tableLoad, operation)) {
          mlir::Value table = operation.getTable();
          mlir::Value snapshot = tableSnapshots.lookup(table);
          if (!snapshot) {
            auto memory = tableLoad.getRegion().getType();
            auto encoding = mlir::cast<kernel::EncodingType>(memory.getEncoding());
            auto kernel = operation->getParentOfType<riscv::KernelOp>();
            const int64_t elementBytes = memory.getStorageBits() / 8;
            if (encoding.getKind() != "dense" || memory.getShape().size() != 1 ||
                memory.getShape()[0] <= 0 || memory.getStrides()[0] != 1 ||
                memory.getStorageBits() % 8 || elementBytes <= 0 ||
                memory.getElements() != 1 || memory.getInterleaveRows() != 0 ||
                memory.getShape()[0] >
                    kernel.getTarget().getMaxPrivateStackBytes() / elementBytes) {
              operation.emitError(
                  "interfering table read requires a target-bounded static contiguous dense snapshot");
              failed = true;
              return;
            }
            int64_t identity = 0;
            getOperation().walk([&](riscv::KernelOp owner) {
              for (int64_t alias : owner.getArgAliasSets())
                identity = std::max(identity, alias + 1);
            });
            getOperation().walk([&](riscv::LocalAllocOp allocation) {
              auto type = allocation.getResult().getType();
              identity = std::max({identity, type.getAliasSet() + 1,
                                   type.getBirthId() + 1});
            });
            int64_t owner =
                tableMaterialize ? tableMaterialize.getOwnerDomainId() : 0;
            if (!tableMaterialize)
              for (mlir::Operation *parent = tableLoad->getParentOp(); parent;
                   parent = parent->getParentOp()) {
                if (auto loop = mlir::dyn_cast<riscv::LoopOp>(parent)) {
                  owner = loop.getDomain().getType().getDomainId();
                  break;
                }
                if (auto level = parent->getAttrOfType<riscv::LevelAttr>(
                        "weft.riscv.level")) {
                  owner = level.getDomainId();
                  break;
                }
              }
            const int64_t birth =
                tableMaterialize ? tableMaterialize.getBirthId() : identity;
            const int64_t lifetime = tableMaterialize
                ? tableMaterialize.getLifetimeEndDomainId() : owner;
            const int64_t bytes = memory.getShape()[0] * elementBytes;
            auto empty = rewriter.getDenseI64ArrayAttr({});
            auto storageType = riscv::LocalType::get(
                &getContext(), rewriter.getIntegerType(8), empty, empty, bytes,
                memory.getAlignment(), identity, "pack", owner, birth,
                lifetime, "read-snapshot");
            auto descriptor = riscv::MemDescType::get(
                &getContext(), memory.getEncoding(), memory.getShape(),
                memory.getAxisIds(), memory.getStrides(),
                rewriter.getDenseI64ArrayAttr({0}), memory.getAlignment(),
                "local", "read", identity, memory.getLayoutIdentity(),
                memory.getStorageBits(), 1, 0);
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPoint(tableLoad);
            auto byteCount = rewriter.create<mlir::arith::ConstantIndexOp>(
                tableLoad.getLoc(), bytes);
            auto allocation = rewriter.create<riscv::LocalAllocOp>(
                tableLoad.getLoc(), storageType, byteCount);
            auto copied = rewriter.create<riscv::DenseSnapshotOp>(
                tableLoad.getLoc(), descriptor, tableLoad.getRegion(),
                allocation.getResult(),
                riscv_internal::readSnapshotLeaf(rewriter, kernel.getTarget(), bytes));
            riscv_internal::copyOrigin(tableLoad, allocation);
            riscv_internal::copyOrigin(tableLoad, copied);
            snapshot = copied.getResult();
            tableSnapshots[table] = snapshot;
          }
          operation->setOperand(0, snapshot);
        } else {
          operation->setOperand(0, tableLoad.getRegion());
        }
        if (!llvm::is_contained(deadTableLoads, tableLoad))
          deadTableLoads.push_back(tableLoad);
        if (tableMaterialize &&
            !llvm::is_contained(deadTableMaterializations,
                                tableMaterialize))
          deadTableMaterializations.push_back(tableMaterialize);
      }
      if (registerTable) {
        operation.setAccessAttr(makeAccess(builder, "register", "natural", 1));
        operation.setLeafAttr(riscv_internal::leaf(
            builder, "rvv", "lookup", "rvv.vrgather", "rvv.vrgather", 0,
            0));
        return;
      }
      if (mlir::isa<riscv::ValueType>(operation.getTable().getType())) {
        operation.emitError(
            "materialized lookup table has no complete register or memory realization");
        failed = true;
        return;
      }
      llvm::StringRef form = indexed ? "indexed" : "unit";
      if (indexed &&
          !operation->getParentOfType<riscv::KernelOp>()
               .getTarget()
               .getHasIndexedMemory()) {
        operation.emitError(
            "vector lookup requires indexed memory unsupported by the target profile");
        failed = true;
        return;
      }
      operation.setAccessAttr(makeAccess(builder, form, "natural", 1));
      operation.setLeafAttr(riscv_internal::leaf(
          builder, indexed ? "rvv" : "scalar", "lookup",
          indexed ? "rvv.vluxei" : "scalar.lookup",
          indexed ? "rvv.vluxei" : "scalar.lookup", 0, 0));
    });
    for (riscv::MaterializeOp materialize : deadTableMaterializations)
      if (materialize && materialize.getResult().use_empty())
        materialize.erase();
    for (riscv::LoadOp load : deadTableLoads)
      if (load && load.getResult().use_empty())
        load.erase();

    // A complete encoded field can feed an outer vector consumer while a
    // nested Level projects the same bytes to scalar consumers.  Place one
    // typed local representation at their common dominating Level and make
    // each child projection a real local_load.  The relation is proved from
    // field identity, logical domain, point geometry, and one repeated decode;
    // terminal emission does not reconstruct it.
    materializeCrossLevelEncodedSupplies(getOperation(), rewriter);

    // A byte-aligned encoded scalar with several consumers is one physical
    // supply, not one reload per consumer.  Materialize it at its defining
    // point so all representation conversions and pointwise users share the
    // same SSA value.  This is deliberately limited to one block and one Level;
    // cross-block placement requires a dominance/LCA lifetime decision.
    llvm::SmallVector<mlir::Value> scalarShareCandidates;
    getOperation().walk([&](mlir::Operation *operation) {
      for (mlir::Value result : operation->getResults())
        if (needsScalarEncodedShare(result))
          scalarShareCandidates.push_back(result);
    });
    for (mlir::Value value : scalarShareCandidates) {
      mlir::Operation *definition = value.getDefiningOp();
      if (!definition || !definition->getBlock() ||
          !needsScalarEncodedShare(value))
        continue;
      rewriter.setInsertionPointAfter(definition);
      auto shared = rewriter.create<riscv::RegisterMaterializeOp>(
          definition->getLoc(), value.getType(), value, -1, -1, -1,
          "physical-share");
      riscv_internal::copyOrigin(definition, shared);
      value.replaceAllUsesExcept(shared.getResult(), shared.getOperation());
    }

    // Memory materialization can replace the source of an existing pure
    // representation conversion with exactly the representation it requested.
    // Close that edge in the same pass so the verifier never observes an
    // identity conversion between the memory decision and the following
    // layout canonicalization pass.
    llvm::SmallVector<riscv::ConvertLayoutOp> identityConversions;
    getOperation().walk([&](riscv::ConvertLayoutOp conversion) {
      if (conversion.getConversion().getEffect() == "pure" &&
          conversion.getInput().getType() == conversion.getResult().getType())
        identityConversions.push_back(conversion);
    });
    for (riscv::ConvertLayoutOp conversion : identityConversions) {
      conversion.getResult().replaceAllUsesWith(conversion.getInput());
      rewriter.eraseOp(conversion);
    }
    getOperation().walk([&](mlir::Operation *operation) {
      if (!mlir::isa<riscv::DotOp, riscv::ContractOp,
                     riscv::OuterContractOp>(operation))
        return;
      auto result =
          mlir::dyn_cast<riscv::ValueType>(operation->getResult(0).getType());
      int64_t laneAxis = 0;
      if (result) {
        for (auto [axis, factor] :
             llvm::zip(result.getAxisIds().asArrayRef(),
                       result.getLayout().getLaneFactors().asArrayRef()))
          if (factor > 1) {
            laneAxis = axis;
            break;
          }
      }
      if (!laneAxis)
        if (auto over =
                operation->getAttrOfType<mlir::DenseI64ArrayAttr>("over");
            over && !over.empty())
          laneAxis = over.asArrayRef().front();
      if (!laneAxis) {
        operation->emitError(
            "scalar contraction has no reduction axis for lane-memory planning");
        failed = true;
        return;
      }
      auto hasAxis = [&](mlir::Value value, int64_t axis) {
        return axis && llvm::is_contained(
                           riscv_internal::logicalAxes(value.getType()), axis);
      };
      unsigned operandIndex =
          hasAxis(operation->getOperand(1), laneAxis) ? 1 : 0;
      if (!laneAxis) {
        auto lhsLayout =
            riscv_internal::layoutOf(operation->getOperand(0).getType());
        auto rhsLayout =
            riscv_internal::layoutOf(operation->getOperand(1).getType());
        if (rhsLayout && rhsLayout.getCarrier() == "rvv" &&
            (!lhsLayout || lhsLayout.getCarrier() != "rvv"))
          operandIndex = 1;
      }
      operation->setAttr("lane_operand",
                         builder.getStringAttr(operandIndex ? "rhs" : "lhs"));
      auto accessFor = [&](mlir::Value value) {
        mlir::Operation *producer = value.getDefiningOp();
        while (auto conversion =
                   mlir::dyn_cast_or_null<riscv::ConvertLayoutOp>(producer))
          producer = conversion.getInput().getDefiningOp();
        return producer ? producer->getAttrOfType<riscv::AccessAttr>("access")
                        : riscv::AccessAttr();
      };
      riscv::AccessAttr access = accessFor(operation->getOperand(operandIndex));
      if (!access) {
        auto layout = riscv_internal::layoutOf(
            operation->getOperand(operandIndex).getType());
        if (layout && layout.getCarrier() == "rvv") {
          operation->setAttr("lane_memory_form",
                             builder.getStringAttr("register"));
          return;
        }
        operation->emitError(
            "selected contraction lane operand has neither memory access nor register representation");
        failed = true;
        return;
      }
      if (mlir::failed(verifyAccessCapability(operation, access))) {
        failed = true;
        return;
      }
      operation->setAttr("lane_memory_form",
                         builder.getStringAttr(access.getForm()));
    });
    if (failed)
      signalPassFailure();
  }

private:
  bool nestedDomainOnly;
};

} // namespace

std::unique_ptr<mlir::Pass> weft::createPlanRISCVMemoryPass() {
  return std::make_unique<PlanRISCVMemoryPass>();
}

std::unique_ptr<mlir::Pass> weft::createPlanRISCVNestedMemoryPass() {
  return std::make_unique<PlanRISCVMemoryPass>(true);
}
