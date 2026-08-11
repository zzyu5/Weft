#include "Weft/Compiler/Selection.h"

#include "Weft/Dialect/Execution/IR/ExecutionDialect.h"
#include "Weft/Dialect/Extension/IR/ExtensionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <limits>
#include <vector>

namespace {

using namespace weft;
using namespace weft::execution;
using namespace weft::kernel;

struct Candidate {
  llvm::StringRef provider;
  llvm::StringRef realization;
  llvm::StringRef strategy;
  int64_t sew = 0;
  llvm::StringRef lmul = "none";
  int64_t unroll = 1;
};

llvm::StringRef selectedRecordName(mlir::Operation *operation);

mlir::Type bareType(mlir::Type type) {
  if (auto masked = mlir::dyn_cast<MaskedType>(type))
    return masked.getValueType();
  return type;
}

mlir::Type elementType(mlir::Type type) {
  type = bareType(type);
  if (auto region = mlir::dyn_cast<RegionType>(type))
    return region.getElementType();
  if (auto block = mlir::dyn_cast<BlockType>(type))
    return block.getElementType();
  return type;
}

bool isTrueScalarPredicate(mlir::Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto integer = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                          : mlir::IntegerAttr{};
  return value.getType().isInteger(1) && integer && !integer.getValue().isZero();
}

bool isF32RegionDataValue(mlir::Type type) {
  type = bareType(type);
  auto region = mlir::dyn_cast<RegionType>(type);
  return region && region.getElementType().isF32();
}

bool isF32RegionPointerValue(mlir::Type type) {
  type = bareType(type);
  auto region = mlir::dyn_cast<RegionType>(type);
  auto pointer = region ? mlir::dyn_cast<PtrType>(region.getElementType())
                        : PtrType{};
  return pointer && pointer.getElementType().isF32();
}

bool isUnitStrideRegionPointer(mlir::Value value, mlir::BlockArgument coordinate) {
  auto pointer = value.getDefiningOp<PtrAddOp>();
  if (!pointer || pointer.getOffset() != coordinate)
    return false;
  mlir::Type base = bareType(pointer.getBase().getType());
  return mlir::isa<PtrType>(base);
}

bool isRVVElementwiseVLA(VLAOp vla, const RISCVTargetProfile &target) {
  if (!target.hasRVV)
    return false;
  mlir::Block &body = vla.getBody().front();
  auto coordinate = body.getArgument(0);
  bool hasMemory = false;
  for (mlir::Operation &operation : body.without_terminator()) {
    if (mlir::isa<ConstantOp, InvalidOp>(operation))
      continue;
    if (auto pointer = mlir::dyn_cast<PtrAddOp>(operation)) {
      mlir::Type result = bareType(pointer.getResult().getType());
      if (mlir::isa<RegionType>(result) &&
          !isUnitStrideRegionPointer(pointer.getResult(), coordinate))
        return false;
      if (mlir::isa<RegionType>(result) && !isF32RegionPointerValue(result))
        return false;
      continue;
    }
    if (auto binary = mlir::dyn_cast<BinaryOp>(operation)) {
      if (!llvm::is_contained({"add", "sub", "mul", "div"}, binary.getKind()))
        return false;
      if (mlir::isa<RegionType>(bareType(binary.getResult().getType())) &&
          !isF32RegionDataValue(binary.getResult().getType()))
        return false;
      continue;
    }
    if (auto load = mlir::dyn_cast<LoadOp>(operation)) {
      if (!isUnitStrideRegionPointer(load.getPointer(), coordinate) ||
          !isF32RegionDataValue(load.getResult().getType()) ||
          !isTrueScalarPredicate(load.getWhere()) ||
          !mlir::isa<mlir::NoneType>(load.getOther().getType()))
        return false;
      hasMemory = true;
      continue;
    }
    if (auto store = mlir::dyn_cast<StoreOp>(operation)) {
      if (!isUnitStrideRegionPointer(store.getPointer(), coordinate) ||
          !isF32RegionDataValue(store.getValue().getType()) ||
          !isTrueScalarPredicate(store.getWhere()))
        return false;
      hasMemory = true;
      continue;
    }
    if (auto reduce = mlir::dyn_cast<ReduceOp>(operation)) {
      if (reduce.getAxis() != -1 || reduce.getKind() != "add" ||
          reduce.getOrder() != "relaxed" ||
          !isF32RegionDataValue(reduce.getInput().getType()) ||
          !reduce.getResult().getType().isF32() ||
          !reduce.getIdentity().getType().isF32() ||
          !isTrueScalarPredicate(reduce.getWhere()))
        return false;
      if (!llvm::all_of(reduce.getResult().getUsers(),
                        [](mlir::Operation *user) { return mlir::isa<YieldOp>(user); }))
        return false;
      continue;
    }
    return false;
  }
  auto yield = mlir::cast<YieldOp>(body.getTerminator());
  if (yield.getNumOperands() != vla.getNumResults())
    return false;
  for (mlir::Value yielded : yield.getOperands())
    if (!yielded.getDefiningOp<ReduceOp>())
      return false;
  return hasMemory;
}

llvm::DenseSet<mlir::Operation *>
findRVVVLAs(KernelOp kernel, const RISCVTargetProfile &target) {
  llvm::DenseSet<mlir::Operation *> result;
  kernel.walk([&](VLAOp vla) {
    if (isRVVElementwiseVLA(vla, target))
      result.insert(vla.getOperation());
  });
  return result;
}

bool belongsToRVVVLA(mlir::Operation *operation,
                     const llvm::DenseSet<mlir::Operation *> &rvvVLAs) {
  if (auto vla = operation->getParentOfType<VLAOp>())
    return rvvVLAs.contains(vla.getOperation());
  return false;
}

bool isRVVF16F32Contract(ContractOp contract,
                         const RISCVTargetProfile &target) {
  if (!target.hasRVV || contract.getOrder() != "relaxed" ||
      contract.getMath() != "native" || !contract.getAccDtype().isF32() ||
      !contract.getOutDtype().isF32() || !contract.getOutputOrder().empty() ||
      contract.getLhsAxes().size() != 1 || contract.getLhsAxes().front() != 1 ||
      contract.getRhsAxes().size() != 1 || contract.getRhsAxes().front() != 0 ||
      !isTrueScalarPredicate(contract.getWhereLhs()) ||
      !isTrueScalarPredicate(contract.getWhereRhs()))
    return false;
  auto lhs = mlir::dyn_cast<BlockType>(bareType(contract.getLhs().getType()));
  auto rhs = mlir::dyn_cast<BlockType>(bareType(contract.getRhs().getType()));
  auto init = mlir::dyn_cast<BlockType>(bareType(contract.getInit().getType()));
  auto result = mlir::dyn_cast<BlockType>(bareType(contract.getResult().getType()));
  return lhs && rhs && init && result && lhs.getShape().size() == 2 &&
         rhs.getShape().size() == 2 && init.getShape().size() == 2 &&
         result.getShape().size() == 2 && lhs.getElementType().isF16() &&
         rhs.getElementType().isF16() && init.getElementType().isF32() &&
         result.getElementType().isF32();
}

struct RVVBlockSelection {
  llvm::DenseSet<mlir::Operation *> axes;
  llvm::DenseSet<mlir::Operation *> unitStrideLoads;
  llvm::DenseSet<mlir::Operation *> indexedLoads;
  llvm::DenseSet<mlir::Operation *> reductions;
};

bool isInteger(mlir::Type type, unsigned width, bool isSigned) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
  return integer && integer.getWidth() == width &&
         (isSigned ? integer.isSigned() : integer.isUnsigned());
}

bool isRankOneBlock(mlir::Type type, int64_t extent,
                    llvm::function_ref<bool(mlir::Type)> elementPredicate) {
  auto block = mlir::dyn_cast<BlockType>(bareType(type));
  return block && block.getShape().size() == 1 &&
         block.getShape().front() == extent &&
         elementPredicate(block.getElementType());
}

std::optional<int64_t> constantIndexValue(mlir::Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto integer = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                          : mlir::IntegerAttr{};
  if (!value.getType().isIndex() || !integer)
    return std::nullopt;
  return integer.getInt();
}

bool isZeroInteger(mlir::Value value, unsigned width, bool isSigned) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto integer = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                          : mlir::IntegerAttr{};
  return isInteger(value.getType(), width, isSigned) && integer &&
         integer.getValue().isZero();
}

enum class RVVBlockKind {
  Index,
  U8,
  U16,
  I8,
  I16,
  I32,
  Predicate,
  PointerU8,
  Tuple,
};

struct IntegerRange {
  int64_t minimum;
  int64_t maximum;
};

struct RVVBlockValue {
  RVVBlockKind kind;
  BlockAxisOp axis;
  int64_t extent = 0;
  bool unitStride = false;
  std::optional<IntegerRange> range;
  std::vector<RVVBlockValue> fields;
};

bool containsBlockPayload(mlir::Type type) {
  type = bareType(type);
  if (mlir::isa<BlockType>(type))
    return true;
  auto tuple = mlir::dyn_cast<TupleType>(type);
  return tuple && llvm::any_of(tuple.getTypes(), containsBlockPayload);
}

std::optional<RVVBlockKind> scalarKind(mlir::Type type) {
  if (type.isIndex())
    return RVVBlockKind::Index;
  if (type.isInteger(1))
    return RVVBlockKind::Predicate;
  if (isInteger(type, 8, false))
    return RVVBlockKind::U8;
  if (isInteger(type, 16, false))
    return RVVBlockKind::U16;
  if (isInteger(type, 8, true))
    return RVVBlockKind::I8;
  if (isInteger(type, 16, true))
    return RVVBlockKind::I16;
  if (isInteger(type, 32, true))
    return RVVBlockKind::I32;
  if (auto pointer = mlir::dyn_cast<PtrType>(type);
      pointer && isInteger(pointer.getElementType(), 8, false))
    return RVVBlockKind::PointerU8;
  return std::nullopt;
}

std::optional<RVVBlockKind> blockKind(mlir::Type type) {
  auto block = mlir::dyn_cast<BlockType>(bareType(type));
  return block ? scalarKind(block.getElementType()) : std::nullopt;
}

std::optional<IntegerRange> fullRange(RVVBlockKind kind) {
  switch (kind) {
  case RVVBlockKind::U8:
    return IntegerRange{0, 255};
  case RVVBlockKind::U16:
    return IntegerRange{0, 65535};
  case RVVBlockKind::I8:
    return IntegerRange{-128, 127};
  case RVVBlockKind::I16:
    return IntegerRange{-32768, 32767};
  default:
    return std::nullopt;
  }
}

std::optional<IntegerRange> scalarRange(mlir::Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto integer = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                          : mlir::IntegerAttr{};
  if (!integer)
    return std::nullopt;
  int64_t scalar = integer.getInt();
  return IntegerRange{scalar, scalar};
}

std::optional<IntegerRange> combineRange(llvm::StringRef kind,
                                         IntegerRange lhs,
                                         IntegerRange rhs) {
  __int128 candidates[4];
  if (kind == "add") {
    candidates[0] = static_cast<__int128>(lhs.minimum) + rhs.minimum;
    candidates[1] = static_cast<__int128>(lhs.minimum) + rhs.maximum;
    candidates[2] = static_cast<__int128>(lhs.maximum) + rhs.minimum;
    candidates[3] = static_cast<__int128>(lhs.maximum) + rhs.maximum;
  } else if (kind == "sub") {
    candidates[0] = static_cast<__int128>(lhs.minimum) - rhs.minimum;
    candidates[1] = static_cast<__int128>(lhs.minimum) - rhs.maximum;
    candidates[2] = static_cast<__int128>(lhs.maximum) - rhs.minimum;
    candidates[3] = static_cast<__int128>(lhs.maximum) - rhs.maximum;
  } else if (kind == "mul") {
    candidates[0] = static_cast<__int128>(lhs.minimum) * rhs.minimum;
    candidates[1] = static_cast<__int128>(lhs.minimum) * rhs.maximum;
    candidates[2] = static_cast<__int128>(lhs.maximum) * rhs.minimum;
    candidates[3] = static_cast<__int128>(lhs.maximum) * rhs.maximum;
  } else {
    return std::nullopt;
  }
  auto [minimum, maximum] = std::minmax_element(std::begin(candidates),
                                                std::end(candidates));
  if (*minimum < std::numeric_limits<int64_t>::min() ||
      *maximum > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return IntegerRange{static_cast<int64_t>(*minimum),
                      static_cast<int64_t>(*maximum)};
}

bool fitsI32(IntegerRange range) {
  return range.minimum >= std::numeric_limits<int32_t>::min() &&
         range.maximum <= std::numeric_limits<int32_t>::max();
}

class RVVBlockAnalyzer {
public:
  bool analyze(mlir::Value value, RVVBlockValue &result) {
    if (auto found = values.find(value); found != values.end()) {
      result = found->second;
      return true;
    }
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || !visiting.insert(value).second)
      return false;
    bool matched = analyzeDefinition(value, definition, result);
    visiting.erase(value);
    if (!matched || !claimOwner(result))
      return false;
    members.insert(definition);
    values[value] = result;
    return true;
  }

  BlockAxisOp axis;
  llvm::DenseSet<mlir::Operation *> members;
  llvm::DenseSet<mlir::Operation *> unitStrideLoads;
  llvm::DenseSet<mlir::Operation *> indexedLoads;

private:
  llvm::DenseMap<mlir::Value, RVVBlockValue> values;
  llvm::DenseSet<mlir::Value> visiting;

  bool claimOwner(const RVVBlockValue &value) {
    if (value.kind == RVVBlockKind::Tuple) {
      for (const RVVBlockValue &field : value.fields)
        if (!claimOwner(field))
          return false;
      return true;
    }
    if (!value.axis || value.extent <= 0)
      return false;
    if (axis && axis != value.axis)
      return false;
    axis = value.axis;
    return true;
  }

  bool sameOwner(const RVVBlockValue &lhs, const RVVBlockValue &rhs) {
    return lhs.axis == rhs.axis && lhs.extent == rhs.extent;
  }

  bool scalarOperand(mlir::Value value, RVVBlockKind expected) {
    auto kind = scalarKind(bareType(value.getType()));
    return kind && *kind == expected;
  }

  bool analyzeDefinition(mlir::Value value, mlir::Operation *definition,
                         RVVBlockValue &result) {
    if (auto axisOp = mlir::dyn_cast<BlockAxisOp>(definition)) {
      auto extent = constantIndexValue(axisOp.getExtent());
      if (!extent || *extent <= 0 ||
          !isRankOneBlock(value.getType(), *extent, [](mlir::Type type) {
            return type.isIndex();
          }))
        return false;
      result = RVVBlockValue{RVVBlockKind::Index, axisOp, *extent};
      return true;
    }

    if (auto pointer = mlir::dyn_cast<PtrAddOp>(definition)) {
      auto resultKind = blockKind(pointer.getResult().getType());
      auto resultBlock = mlir::dyn_cast<BlockType>(bareType(
          pointer.getResult().getType()));
      if (!resultKind || *resultKind != RVVBlockKind::PointerU8 ||
          !resultBlock || resultBlock.getShape().size() != 1)
        return false;
      if (mlir::isa<BlockType>(bareType(pointer.getOffset().getType()))) {
        RVVBlockValue offset;
        if (!analyze(pointer.getOffset(), offset) ||
            offset.kind != RVVBlockKind::Index ||
            !scalarOperand(pointer.getBase(), RVVBlockKind::PointerU8))
          return false;
        result = RVVBlockValue{RVVBlockKind::PointerU8, offset.axis,
                               offset.extent,
                               pointer.getOffset().getDefiningOp<BlockAxisOp>() !=
                                   nullptr};
        return true;
      }
      RVVBlockValue base;
      if (!analyze(pointer.getBase(), base) ||
          base.kind != RVVBlockKind::PointerU8 ||
          !scalarOperand(pointer.getOffset(), RVVBlockKind::Index))
        return false;
      result = base;
      return true;
    }

    if (auto load = mlir::dyn_cast<LoadOp>(definition)) {
      RVVBlockValue pointer;
      auto resultKind = blockKind(load.getResult().getType());
      if (!resultKind || *resultKind != RVVBlockKind::U8 ||
          !analyze(load.getPointer(), pointer) ||
          pointer.kind != RVVBlockKind::PointerU8 ||
          !isZeroInteger(load.getOther(), 8, false))
        return false;
      if (isTrueScalarPredicate(load.getWhere())) {
      } else {
        RVVBlockValue predicate;
        if (!analyze(load.getWhere(), predicate) ||
            predicate.kind != RVVBlockKind::Predicate ||
            !sameOwner(pointer, predicate))
          return false;
      }
      result = RVVBlockValue{RVVBlockKind::U8, pointer.axis, pointer.extent,
                             false, fullRange(RVVBlockKind::U8)};
      (pointer.unitStride ? unitStrideLoads : indexedLoads)
          .insert(load.getOperation());
      return true;
    }

    if (auto bitcast = mlir::dyn_cast<BitcastOp>(definition)) {
      RVVBlockValue input;
      auto source = blockKind(bitcast.getInput().getType());
      auto target = blockKind(bitcast.getResult().getType());
      if (!source || !target || !analyze(bitcast.getInput(), input))
        return false;
      if (*source == RVVBlockKind::U8 && *target == RVVBlockKind::I8)
        result = RVVBlockValue{*target, input.axis, input.extent, false,
                               fullRange(*target)};
      else if (*source == RVVBlockKind::U16 && *target == RVVBlockKind::I16)
        result = RVVBlockValue{*target, input.axis, input.extent, false,
                               fullRange(*target)};
      else
        return false;
      return true;
    }

    if (auto cast = mlir::dyn_cast<CastOp>(definition)) {
      RVVBlockValue input;
      auto source = blockKind(cast.getInput().getType());
      auto target = blockKind(cast.getResult().getType());
      if (!source || !target || !analyze(cast.getInput(), input))
        return false;
      bool supported =
          (*source == RVVBlockKind::Index && *target == RVVBlockKind::U8) ||
          (*source == RVVBlockKind::U8 && *target == RVVBlockKind::U16) ||
          (*source == RVVBlockKind::U8 && *target == RVVBlockKind::I32) ||
          (*source == RVVBlockKind::I8 && *target == RVVBlockKind::I32) ||
          (*source == RVVBlockKind::I16 && *target == RVVBlockKind::I32);
      if (!supported)
        return false;
      std::optional<IntegerRange> range = fullRange(*target);
      if (*target == RVVBlockKind::I32)
        range = input.range ? input.range : fullRange(*source);
      result = RVVBlockValue{*target, input.axis, input.extent, false, range};
      return true;
    }

    if (auto binary = mlir::dyn_cast<BinaryOp>(definition)) {
      auto resultKind = blockKind(binary.getResult().getType());
      if (!resultKind)
        return false;
      RVVBlockValue lhs;
      RVVBlockValue rhs;
      bool lhsBlock = containsBlockPayload(binary.getLhs().getType());
      bool rhsBlock = containsBlockPayload(binary.getRhs().getType());
      if (!lhsBlock ||
          (lhsBlock && !analyze(binary.getLhs(), lhs)) ||
          (rhsBlock && !analyze(binary.getRhs(), rhs)) ||
          (lhsBlock && rhsBlock && !sameOwner(lhs, rhs)) ||
          (!lhsBlock && !scalarOperand(binary.getLhs(), *resultKind)) ||
          (!rhsBlock && !scalarOperand(binary.getRhs(), *resultKind)))
        return false;
      llvm::StringRef kind = binary.getKind();
      bool supported = false;
      if (*resultKind == RVVBlockKind::Index)
        supported = llvm::is_contained({"add", "sub", "mul", "div", "mod"},
                                       kind);
      else if (*resultKind == RVVBlockKind::U8)
        supported = llvm::is_contained({"add", "and", "or", "shl", "shr"},
                                       kind);
      else if (*resultKind == RVVBlockKind::U16)
        supported = llvm::is_contained({"or", "shl"}, kind);
      else if (*resultKind == RVVBlockKind::I32)
        supported = llvm::is_contained({"add", "sub", "mul"}, kind);
      if (!supported)
        return false;
      const RVVBlockValue &owner = lhsBlock ? lhs : rhs;
      std::optional<IntegerRange> range = fullRange(*resultKind);
      if (*resultKind == RVVBlockKind::I32) {
        auto lhsRange = lhsBlock ? lhs.range : scalarRange(binary.getLhs());
        auto rhsRange = rhsBlock ? rhs.range : scalarRange(binary.getRhs());
        if (!lhsRange || !rhsRange)
          return false;
        range = combineRange(kind, *lhsRange, *rhsRange);
        if (!range || !fitsI32(*range))
          return false;
      }
      result = RVVBlockValue{*resultKind, owner.axis, owner.extent, false,
                             range};
      return true;
    }

    if (auto compare = mlir::dyn_cast<CompareOp>(definition)) {
      RVVBlockValue lhs;
      RVVBlockValue rhs;
      bool lhsBlock = containsBlockPayload(compare.getLhs().getType());
      bool rhsBlock = containsBlockPayload(compare.getRhs().getType());
      if (!lhsBlock ||
          (lhsBlock && !analyze(compare.getLhs(), lhs)) ||
          (rhsBlock && !analyze(compare.getRhs(), rhs)) ||
          (lhsBlock && rhsBlock && !sameOwner(lhs, rhs)) ||
          (lhsBlock && lhs.kind != RVVBlockKind::Index) ||
          (rhsBlock && rhs.kind != RVVBlockKind::Index) ||
          (!lhsBlock && !scalarOperand(compare.getLhs(), RVVBlockKind::Index)) ||
          (!rhsBlock && !scalarOperand(compare.getRhs(), RVVBlockKind::Index)) ||
          !llvm::is_contained({"lt", "ge"}, compare.getPredicate()))
        return false;
      const RVVBlockValue &owner = lhsBlock ? lhs : rhs;
      result = RVVBlockValue{RVVBlockKind::Predicate, owner.axis,
                             owner.extent};
      return true;
    }

    if (auto select = mlir::dyn_cast<SelectOp>(definition)) {
      RVVBlockValue predicate;
      RVVBlockValue trueValue;
      RVVBlockValue falseValue;
      if (!analyze(select.getPredicate(), predicate) ||
          !analyze(select.getTrueValue(), trueValue) ||
          !analyze(select.getFalseValue(), falseValue) ||
          predicate.kind != RVVBlockKind::Predicate ||
          trueValue.kind != RVVBlockKind::U8 ||
          falseValue.kind != RVVBlockKind::U8 ||
          !sameOwner(predicate, trueValue) ||
          !sameOwner(trueValue, falseValue))
        return false;
      result = trueValue;
      return true;
    }

    if (auto tuple = mlir::dyn_cast<TupleOp>(definition)) {
      result = RVVBlockValue{RVVBlockKind::Tuple};
      for (mlir::Value field : tuple.getOperands()) {
        RVVBlockValue fieldValue;
        if (!analyze(field, fieldValue))
          return false;
        result.fields.push_back(std::move(fieldValue));
      }
      for (mlir::Operation *user : tuple.getResult().getUsers())
        if (auto get = mlir::dyn_cast<TupleGetOp>(user);
            get && get.getResult().use_empty())
          members.insert(user);
      return !result.fields.empty();
    }

    if (auto get = mlir::dyn_cast<TupleGetOp>(definition)) {
      RVVBlockValue tuple;
      if (!analyze(get.getInput(), tuple) ||
          tuple.kind != RVVBlockKind::Tuple || get.getIndex() < 0 ||
          static_cast<size_t>(get.getIndex()) >= tuple.fields.size())
        return false;
      result = tuple.fields[get.getIndex()];
      return true;
    }
    return false;
  }
};

bool matchRVVBlockReduction(ReduceOp reduce, RVVBlockAnalyzer &analyzer) {
  if (reduce.getAxis() != 0 || reduce.getKind() != "add" ||
      reduce.getOrder() != "relaxed" ||
      !isInteger(reduce.getAccDtype(), 32, true) ||
      !isInteger(reduce.getResult().getType(), 32, true) ||
      !isZeroInteger(reduce.getIdentity(), 32, true) ||
      !isTrueScalarPredicate(reduce.getWhere()))
    return false;
  RVVBlockValue input;
  if (!analyzer.analyze(reduce.getInput(), input) ||
      input.kind != RVVBlockKind::I32 || !analyzer.axis || !input.range)
    return false;
  auto extent = constantIndexValue(analyzer.axis.getExtent());
  if (!extent || *extent != input.extent || *extent <= 0)
    return false;
  __int128 minimum = static_cast<__int128>(input.range->minimum) * *extent;
  __int128 maximum = static_cast<__int128>(input.range->maximum) * *extent;
  if (minimum < std::numeric_limits<int32_t>::min() ||
      maximum > std::numeric_limits<int32_t>::max())
    return false;

  analyzer.members.insert(reduce.getOperation());
  for (mlir::Operation *member : analyzer.members) {
    if (member->getBlock() != reduce->getBlock())
      return false;
    for (mlir::Value result : member->getResults()) {
      if (!containsBlockPayload(result.getType()))
        continue;
      for (mlir::Operation *user : result.getUsers())
        if (!analyzer.members.contains(user))
          return false;
    }
  }
  return true;
}

RVVBlockSelection findRVVBlocks(KernelOp kernel,
                                const RISCVTargetProfile &target) {
  RVVBlockSelection selected;
  if (!target.hasRVV)
    return selected;
  kernel.walk([&](ReduceOp reduce) {
    RVVBlockAnalyzer analyzer;
    if (!matchRVVBlockReduction(reduce, analyzer))
      return;
    selected.axes.insert(analyzer.axis.getOperation());
    selected.unitStrideLoads.insert(analyzer.unitStrideLoads.begin(),
                                     analyzer.unitStrideLoads.end());
    selected.indexedLoads.insert(analyzer.indexedLoads.begin(),
                                 analyzer.indexedLoads.end());
    selected.reductions.insert(reduce.getOperation());
  });
  return selected;
}

llvm::SmallVector<Candidate>
buildCandidates(mlir::Operation *canonical,
                const llvm::DenseSet<mlir::Operation *> &rvvVLAs,
                const RVVBlockSelection &rvvBlocks,
                const RISCVTargetProfile &target) {
  llvm::SmallVector<Candidate> candidates;
  llvm::StringRef name = selectedRecordName(canonical);
  if (name == "weft_execution.axis_plan")
    candidates.push_back({"scalar", "scalar", {}, 0, "none", 1});
  else if (name == "weft_execution.memory_plan")
    candidates.push_back({"scalar", {}, "scalar_direct"});
  else if (name == "weft_execution.reduce_plan")
    candidates.push_back({"scalar", {}, "scalar_linear"});
  else if (name == "weft_execution.scan_plan")
    candidates.push_back({"scalar", {}, "scalar_linear"});
  else if (name == "weft_execution.summary_plan")
    candidates.push_back({"scalar", {}, "scalar_linear"});
  else if (name == "weft_execution.contract_plan" &&
           mlir::isa<ContractOp>(canonical))
    candidates.push_back({"scalar", {}, "scalar_nested"});
  else if (name == "weft_execution.math_plan")
    candidates.push_back({"scalar", {}, "scalar_libm"});
  else if (name == "weft_execution.primitive_plan")
    candidates.push_back({"scalar", {}, "scalar_direct"});

  if (auto vla = mlir::dyn_cast<VLAOp>(canonical);
      vla && rvvVLAs.contains(vla.getOperation()))
    candidates.push_back({"rvv", "rvv", {}, 32, "m1", 1});
  if (mlir::isa<BlockAxisOp>(canonical) && rvvBlocks.axes.contains(canonical))
    candidates.push_back({"rvv", "rvv", {}, 32, "m1", 1});
  if (mlir::isa<LoadOp, StoreOp>(canonical) &&
      belongsToRVVVLA(canonical, rvvVLAs))
    candidates.push_back({"rvv", {}, "rvv_unit_stride"});
  if (mlir::isa<LoadOp>(canonical) &&
      rvvBlocks.unitStrideLoads.contains(canonical))
    candidates.push_back({"rvv", {}, "rvv_block_unit_stride"});
  if (mlir::isa<LoadOp>(canonical) &&
      rvvBlocks.indexedLoads.contains(canonical))
    candidates.push_back({"rvv", {}, "rvv_block_indexed"});
  if (auto reduce = mlir::dyn_cast<ReduceOp>(canonical);
      reduce && belongsToRVVVLA(canonical, rvvVLAs) &&
      reduce.getOrder() == "relaxed")
    candidates.push_back({"rvv", {}, "rvv_tree"});
  if (mlir::isa<ReduceOp>(canonical) &&
      rvvBlocks.reductions.contains(canonical))
    candidates.push_back({"rvv", {}, "rvv_block_tree"});
  if (auto contract = mlir::dyn_cast<ContractOp>(canonical);
      contract && isRVVF16F32Contract(contract, target))
    candidates.push_back({"rvv", {}, "rvv_f16_f32_contract"});
  return candidates;
}

mlir::FailureOr<Candidate>
chooseCandidate(mlir::Operation *canonical,
                const llvm::DenseSet<mlir::Operation *> &rvvVLAs,
                const RVVBlockSelection &rvvBlocks,
                const RISCVTargetProfile &target) {
  llvm::SmallVector<Candidate> candidates =
      buildCandidates(canonical, rvvVLAs, rvvBlocks, target);
  if (candidates.empty()) {
    canonical->emitError("has no legal realization provider for the selected target");
    return mlir::failure();
  }
  return candidates.back();
}

llvm::StringRef selectedRecordName(mlir::Operation *operation) {
  if (mlir::isa<MetaValueOp>(operation))
    return "weft_execution.meta_binding";
  if (mlir::isa<VLAOp, BlockAxisOp>(operation))
    return "weft_execution.axis_plan";
  if (mlir::isa<LoadOp, StoreOp, PrefetchOp, AtomicAddOp, FenceOp>(operation))
    return "weft_execution.memory_plan";
  if (mlir::isa<ReduceOp>(operation))
    return "weft_execution.reduce_plan";
  if (mlir::isa<ScanOp>(operation))
    return "weft_execution.scan_plan";
  if (mlir::isa<SummaryFoldOp>(operation))
    return "weft_execution.summary_plan";
  if (mlir::isa<ContractOp, weft::extension::BlockScaledContractOp>(operation))
    return "weft_execution.contract_plan";
  if (auto unary = mlir::dyn_cast<UnaryOp>(operation);
      unary && unary.getKind() != "neg")
    return "weft_execution.math_plan";
  if (mlir::isa<PermuteOp, LookupOp, DecodeOp, WidenOp, NarrowOp>(operation))
    return "weft_execution.primitive_plan";
  return {};
}

bool isMechanicallySupported(mlir::Operation *operation) {
  return mlir::isa<
      KernelOp, ReturnOp, YieldOp, ConditionOp, ConstantOp, IfOp, ForOp,
      WhileOp, FullOp, ExpandDimsOp, BroadcastToOp, ReshapeOp, TransposeOp,
      PtrAddOp, UnaryOp, BinaryOp, CompareOp, CastOp, BitcastOp, SelectOp,
      TupleOp, TupleGetOp, SpecialValueOp, InvalidOp, ValidOp, FillOp>(operation);
}

mlir::Operation *createOperation(mlir::OpBuilder &builder,
                                 llvm::StringRef name,
                                 mlir::Location location,
                                 mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::OperationState state(location, name);
  state.addAttributes(attrs);
  return builder.create(state);
}

mlir::NamedAttribute attr(mlir::OpBuilder &builder, llvm::StringRef name,
                          mlir::Attribute value) {
  return builder.getNamedAttr(name, value);
}

mlir::Operation *createSelectedRecord(mlir::OpBuilder &builder,
                                      mlir::Operation *canonical,
                                      int64_t anchor,
                                      const SelectionOptions &options,
                                      const llvm::DenseSet<mlir::Operation *> &rvvVLAs,
                                      const RVVBlockSelection &rvvBlocks,
                                      llvm::StringSet<> &consumedMeta) {
  llvm::StringRef name = selectedRecordName(canonical);
  if (name.empty())
    return nullptr;
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(attr(builder, "anchor", builder.getI64IntegerAttr(anchor)));

  if (auto meta = mlir::dyn_cast<MetaValueOp>(canonical)) {
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(meta.getInput());
    auto kernel = canonical->getParentOfType<KernelOp>();
    if (!argument || !kernel)
      return nullptr;
    auto nameAttr = mlir::dyn_cast<mlir::StringAttr>(
        kernel.getArgNames()[argument.getArgNumber()]);
    if (!nameAttr)
      return nullptr;
    auto binding = options.metaBindings.find(nameAttr.getValue());
    if (binding == options.metaBindings.end()) {
      canonical->emitError() << "missing --meta binding for "
                             << nameAttr.getValue();
      return nullptr;
    }
    consumedMeta.insert(nameAttr.getValue());
    attrs.push_back(attr(builder, "value",
                         builder.getI64IntegerAttr(binding->second)));
  } else {
    mlir::FailureOr<Candidate> selected =
        chooseCandidate(canonical, rvvVLAs, rvvBlocks, options.target);
    if (mlir::failed(selected))
      return nullptr;
    attrs.push_back(attr(builder, "provider", builder.getStringAttr(selected->provider)));
    if (name == "weft_execution.axis_plan") {
      attrs.push_back(attr(builder, "realization",
                           builder.getStringAttr(selected->realization)));
      attrs.push_back(attr(builder, "sew", builder.getI64IntegerAttr(selected->sew)));
      attrs.push_back(attr(builder, "lmul", builder.getStringAttr(selected->lmul)));
      attrs.push_back(attr(builder, "unroll", builder.getI64IntegerAttr(selected->unroll)));
    } else {
      attrs.push_back(attr(builder, "strategy", builder.getStringAttr(selected->strategy)));
    }
  }
  if (name == "weft_execution.contract_plan") {
    attrs.push_back(attr(builder, "micro_m", builder.getI64IntegerAttr(1)));
    attrs.push_back(attr(builder, "micro_n", builder.getI64IntegerAttr(1)));
    attrs.push_back(attr(builder, "micro_k", builder.getI64IntegerAttr(1)));
  }
  return createOperation(builder, name, canonical->getLoc(), attrs);
}

mlir::FailureOr<PlanOp> createPlan(mlir::ModuleOp module, KernelOp kernel,
                                   const SelectionOptions &options) {
  mlir::OpBuilder builder(module.getContext());
  std::string symbol = (kernel.getSymName() + "__selected").str();
  if (module.lookupSymbol(symbol)) {
    kernel.emitError("selected plan symbol already exists");
    return mlir::failure();
  }
  builder.setInsertionPointAfter(kernel);
  mlir::OperationState state(kernel.getLoc(), PlanOp::getOperationName());
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(symbol));
  state.addAttribute("kernel_ref", mlir::FlatSymbolRefAttr::get(
                                       module.getContext(), kernel.getSymName()));
  state.addAttribute("target_triple",
                     builder.getStringAttr(options.target.triple));
  state.addAttribute("march", builder.getStringAttr(options.target.march));
  state.addAttribute("abi", builder.getStringAttr(options.target.abi));
  state.addAttribute("xlen", builder.getI64IntegerAttr(options.target.xlen));
  state.addAttribute("little_endian",
                     builder.getBoolAttr(options.target.littleEndian));
  state.addAttribute("has_rvv", builder.getBoolAttr(options.target.hasRVV));
  state.addAttribute("vlen_bits",
                     builder.getI64IntegerAttr(options.target.vlenBits));
  state.addAttribute(
      "vector_registers",
      builder.getI64IntegerAttr(options.target.vectorRegisters));
  state.addRegion();
  auto plan = mlir::cast<PlanOp>(builder.create(state));
  plan.getBody().emplaceBlock();

  llvm::SmallVector<mlir::Operation *> planned;
  bool unsupported = false;
  kernel.walk([&](mlir::Operation *operation) {
    if (!selectedRecordName(operation).empty()) {
      planned.push_back(operation);
      return;
    }
    if (isMechanicallySupported(operation))
      return;
    operation->emitError("has no registered realization provider or mechanical lowering");
    unsupported = true;
  });
  if (unsupported) {
    plan.erase();
    return mlir::failure();
  }
  builder.setInsertionPointToStart(&plan.getBody().front());
  llvm::StringSet<> consumedMeta;
  llvm::DenseSet<mlir::Operation *> rvvVLAs =
      findRVVVLAs(kernel, options.target);
  RVVBlockSelection rvvBlocks = findRVVBlocks(kernel, options.target);
  for (auto [anchor, operation] : llvm::enumerate(planned)) {
    operation->setAttr(kCanonicalAnchorAttr,
                       builder.getI64IntegerAttr(anchor));
    if (!createSelectedRecord(builder, operation, anchor, options, rvvVLAs,
                              rvvBlocks, consumedMeta)) {
      plan.erase();
      return mlir::failure();
    }
  }
  for (const auto &binding : options.metaBindings) {
    if (!consumedMeta.contains(binding.getKey())) {
      kernel.emitError() << "--meta names no constexpr argument: "
                         << binding.getKey();
      plan.erase();
      return mlir::failure();
    }
  }
  createOperation(builder, EndOp::getOperationName(), kernel.getLoc(), {});
  return plan;
}

} // namespace

mlir::LogicalResult
weft::selectExecution(mlir::ModuleOp module,
                      const SelectionOptions &options) {
  module.getContext()->getOrLoadDialect<WEFTExecutionDialect>();
  llvm::SmallVector<KernelOp> kernels;
  for (KernelOp kernel : module.getOps<KernelOp>())
    kernels.push_back(kernel);
  if (kernels.empty())
    return module.emitError("module contains no canonical Weft kernel");
  for (KernelOp kernel : kernels)
    if (mlir::failed(createPlan(module, kernel, options)))
      return mlir::failure();
  return mlir::success();
}
