#include "Weft/Target/RISCVLowering.h"

#include "Weft/Dialect/Extension/IR/ExtensionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <functional>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace weft;
using namespace weft::extension;
using namespace weft::kernel;

enum class CValueKind {
  Scalar,
  Pointer,
  Coordinate,
  F16Vector,
  F32Vector,
  I8Vector,
  F32BlockStorage,
  Mask,
  Tuple,
};

enum class BlockValueKind {
  Scalar,
  Pointer,
  Index,
  U8,
  I8,
  U16,
  I16,
  I32,
  F32,
  Mask,
  Tuple,
};

enum class RVVBlockVectorShape {
  None,
  E8MF4,
  E8M1,
  E16M2,
  E32M1,
  E32M4,
};

struct BlockValue {
  mlir::Type type;
  BlockValueKind kind = BlockValueKind::Scalar;
  std::string spelling;
  std::string pointerBase;
  std::string pointerIndex;
  std::string contiguousIndex;
  std::vector<BlockValue> fields;
  BlockValueKind narrowKind = BlockValueKind::Scalar;
  std::string narrowSpelling;
  std::optional<uint64_t> unsignedMaximum;
  mlir::Value packedSource;
  std::string packedTransform;
  std::string packedTransformOperand;
  std::string widenedSpelling;
  RVVBlockVectorShape vectorShape = RVVBlockVectorShape::None;
};

enum class BlockDecodeRealization {
  RVVI8TableGather,
};

struct BlockDecodeDecision {
  BlockDecodeRealization realization =
      BlockDecodeRealization::RVVI8TableGather;
  int64_t tableExtent = 16;
  RVVBlockVectorShape codeShape = RVVBlockVectorShape::E8M1;
  RVVBlockVectorShape resultShape = RVVBlockVectorShape::E8M1;
};

enum class SymmetricI4I8Realization {
  SpacemiTIME1N16K32,
};

struct SymmetricI4I8Decision {
  SymmetricI4I8Realization realization =
      SymmetricI4I8Realization::SpacemiTIME1N16K32;
  mlir::Value activationBlockBase;
  mlir::Value packedBlockBase;
  mlir::Value activationScale;
  mlir::Value init;
  int64_t packedBlockBytes = 288;
};

struct CValue {
  mlir::Type type;
  CValueKind kind = CValueKind::Scalar;
  std::string spelling;
  bool lanePointer = false;
  std::vector<CValue> fields;
  std::string laneStride;
};

enum class LaneRelation {
  Independent,
  UnitStride,
  Strided,
  NonAffine,
};

enum class VLAMemoryMode {
  UnitStride,
  Strided,
};

enum class VLAActivityMode {
  AllActive,
  PredicateMask,
  ScalarPredicate,
};

enum class VLAStoreValueMode {
  ScalarBroadcast,
  Vector,
};

enum class VLAStateRealization {
  RVVAddReduction,
  RVVMaxReduction,
  RVVInclusiveAddScan,
  RVVArgMaxSummary,
};

struct VLAPredicateDecision {
  mlir::Operation *operation = nullptr;
  mlir::Value coordinate;
  mlir::Value scalar;
  VLAMemoryMode coordinateMode = VLAMemoryMode::UnitStride;
  std::string predicate;
};

struct VLAAccessDecision {
  mlir::Operation *operation = nullptr;
  mlir::Type elementType;
  VLAMemoryMode memoryMode = VLAMemoryMode::UnitStride;
  VLAActivityMode activityMode = VLAActivityMode::AllActive;
  VLAStoreValueMode storeValueMode = VLAStoreValueMode::Vector;
  mlir::Value predicate;
};

struct VLAStateDecision {
  mlir::Operation *operation = nullptr;
  VLAStateRealization realization = VLAStateRealization::RVVAddReduction;
  mlir::Type elementType;
  mlir::Value identity;
  VLAMemoryMode coordinateMode = VLAMemoryMode::UnitStride;
};

struct VLANarrowDecision {
  mlir::Operation *operation = nullptr;
  unsigned sourceLMUL = 4;
  unsigned intermediateLMUL = 2;
  unsigned resultLMUL = 1;
};

enum class VLAContractRealization {
  RVVF32FreeAxisMicrotile,
};

struct VLAContractDecision {
  mlir::Operation *operation = nullptr;
  VLAContractRealization realization =
      VLAContractRealization::RVVF32FreeAxisMicrotile;
  mlir::Operation *consumer = nullptr;
  mlir::Operation *lhsLoad = nullptr;
  mlir::Operation *rhsLoad = nullptr;
  mlir::Value rowAxis;
  mlir::Value reductionAxis;
  mlir::Value reductionExtent;
  unsigned rowTile = 1;
  unsigned lmul = 2;
  VLAMemoryMode rhsMemoryMode = VLAMemoryMode::UnitStride;
  VLAMemoryMode outputMemoryMode = VLAMemoryMode::UnitStride;
  bool lhsPredicateVariesByReduction = false;
  llvm::SmallVector<mlir::Operation *> absorbed;
};

struct VLARegionDecision {
  mlir::Operation *operation = nullptr;
  mlir::Value coordinate;
  unsigned dataSEW = 32;
  unsigned dataLMUL = 2;
  unsigned indexSEW = 64;
  unsigned indexLMUL = 4;
  unsigned maskRatio = 16;
  std::vector<VLAPredicateDecision> predicates;
  std::vector<VLAAccessDecision> accesses;
  std::vector<VLAStateDecision> states;
  std::vector<VLANarrowDecision> narrows;
  std::vector<VLAContractDecision> contracts;
};

enum class ContractRealization {
  RVVF32RowMicrotile,
};

struct ContractDecision {
  mlir::Operation *operation = nullptr;
  ContractRealization realization = ContractRealization::RVVF32RowMicrotile;
  mlir::Operation *consumer = nullptr;
  mlir::Operation *lhsLoad = nullptr;
  mlir::Operation *rhsLoad = nullptr;
  mlir::Value rowAxis;
  mlir::Value reductionAxis;
  mlir::Value reductionExtent;
  unsigned rowTile = 1;
  unsigned lmul = 4;
};

std::string sanitize(llvm::StringRef input) {
  std::string result;
  result.reserve(input.size() + 1);
  for (char character : input) {
    unsigned char byte = static_cast<unsigned char>(character);
    result.push_back(std::isalnum(byte) || character == '_' ? character : '_');
  }
  if (result.empty() || std::isdigit(static_cast<unsigned char>(result.front())))
    result.insert(result.begin(), '_');
  return result;
}

mlir::Type elementType(mlir::Type type) {
  if (auto masked = mlir::dyn_cast<MaskedType>(type))
    type = masked.getValueType();
  if (auto region = mlir::dyn_cast<RegionType>(type))
    return region.getElementType();
  if (auto block = mlir::dyn_cast<BlockType>(type))
    return block.getElementType();
  return type;
}

bool containsBlockType(mlir::Type type) {
  if (auto masked = mlir::dyn_cast<MaskedType>(type))
    return containsBlockType(masked.getValueType());
  if (mlir::isa<BlockType>(type))
    return true;
  if (auto tuple = mlir::dyn_cast<TupleType>(type))
    return llvm::any_of(tuple.getTypes(), containsBlockType);
  return false;
}

bool hasBlockPayload(mlir::Operation *operation) {
  return llvm::any_of(operation->getOperandTypes(), containsBlockType) ||
         llvm::any_of(operation->getResultTypes(), containsBlockType);
}

bool isF16(mlir::Type type) {
  auto floating = mlir::dyn_cast<mlir::FloatType>(type);
  return floating && floating.getWidth() == 16;
}

std::string scalarCType(mlir::Type type) {
  if (type.isIndex())
    return "size_t";
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type)) {
    if (integer.getWidth() == 1)
      return "bool";
    std::string prefix = integer.isUnsigned() ? "uint" : "int";
    return prefix + std::to_string(integer.getWidth()) + "_t";
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type)) {
    if (floating.getWidth() == 16)
      return "_Float16";
    if (floating.getWidth() == 32)
      return "float";
    if (floating.getWidth() == 64)
      return "double";
  }
  return {};
}

std::string pointerCType(PtrType pointer) {
  std::string element = scalarCType(pointer.getElementType());
  if (element.empty())
    return {};
  return (pointer.getAccess() == "read" ? "const " : "") + element + " *";
}

std::string floatLiteral(mlir::FloatAttr attribute) {
  std::ostringstream stream;
  stream << std::setprecision(17) << attribute.getValueAsDouble();
  std::string result = stream.str();
  if (result.find_first_of(".eE") == std::string::npos)
    result += ".0";
  if (mlir::cast<mlir::FloatType>(attribute.getType()).getWidth() <= 32)
    result += "f";
  return result;
}

std::string constantLiteral(mlir::Attribute attribute) {
  if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attribute)) {
    if (integer.getType().isInteger(1))
      return integer.getValue().isZero() ? "false" : "true";
    return std::to_string(integer.getInt());
  }
  if (auto floating = mlir::dyn_cast<mlir::FloatAttr>(attribute))
    return floatLiteral(floating);
  return {};
}

bool isTrue(mlir::Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto integer = constant ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                          : mlir::IntegerAttr{};
  return value.getType().isInteger(1) && integer && !integer.getValue().isZero();
}

bool isRegionValue(mlir::Type type) {
  if (auto masked = mlir::dyn_cast<MaskedType>(type))
    type = masked.getValueType();
  return mlir::isa<RegionType>(type);
}

std::optional<int64_t> integerConstantValue(mlir::Value value) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto integer = constant
                     ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                     : mlir::IntegerAttr{};
  if (!integer)
    return std::nullopt;
  return integer.getInt();
}

LaneRelation combineLaneRelations(LaneRelation lhs, LaneRelation rhs) {
  if (lhs == LaneRelation::NonAffine || rhs == LaneRelation::NonAffine)
    return LaneRelation::NonAffine;
  if (lhs == LaneRelation::Independent)
    return rhs;
  if (rhs == LaneRelation::Independent)
    return lhs;
  return LaneRelation::Strided;
}

LaneRelation classifyLaneRelation(mlir::Value value, mlir::Value coordinate) {
  if (value == coordinate)
    return LaneRelation::UnitStride;
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
      if (lhs != LaneRelation::Independent &&
          rhs != LaneRelation::Independent)
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
      return dependent == LaneRelation::NonAffine ? LaneRelation::NonAffine
                                                   : LaneRelation::Strided;
    }
  }
  if (auto expand = value.getDefiningOp<ExpandDimsOp>())
    return classifyLaneRelation(expand.getInput(), coordinate);
  if (auto broadcast = value.getDefiningOp<BroadcastToOp>())
    return classifyLaneRelation(broadcast.getInput(), coordinate);
  if (auto reshape = value.getDefiningOp<ReshapeOp>())
    return classifyLaneRelation(reshape.getInput(), coordinate);
  if (auto transpose = value.getDefiningOp<TransposeOp>())
    return classifyLaneRelation(transpose.getInput(), coordinate);
  return isRegionValue(value.getType()) ? LaneRelation::NonAffine
                                        : LaneRelation::Independent;
}

std::string reversePredicate(llvm::StringRef predicate) {
  return llvm::StringSwitch<std::string>(predicate)
      .Case("eq", "eq")
      .Case("ne", "ne")
      .Case("lt", "gt")
      .Case("le", "ge")
      .Case("gt", "lt")
      .Case("ge", "le")
      .Default("");
}

bool isFloatConstant(mlir::Value value, double expected) {
  auto constant = value.getDefiningOp<ConstantOp>();
  auto floating = constant
                      ? mlir::dyn_cast<mlir::FloatAttr>(constant.getValue())
                      : mlir::FloatAttr{};
  return floating && floating.getValueAsDouble() == expected;
}

bool isTupleField(mlir::Value value, mlir::BlockArgument tuple,
                  int64_t index) {
  auto get = value.getDefiningOp<TupleGetOp>();
  return get && get.getInput() == tuple && get.getIndex() == index;
}

bool matchesScaledSummaryTerm(mlir::Value value, mlir::BlockArgument state,
                              mlir::Value maximum) {
  auto multiply = value.getDefiningOp<BinaryOp>();
  if (!multiply || multiply.getKind() != "mul")
    return false;
  mlir::Value stateSum;
  mlir::Value exponential;
  if (isTupleField(multiply.getLhs(), state, 1)) {
    stateSum = multiply.getLhs();
    exponential = multiply.getRhs();
  } else if (isTupleField(multiply.getRhs(), state, 1)) {
    stateSum = multiply.getRhs();
    exponential = multiply.getLhs();
  } else {
    return false;
  }
  (void)stateSum;
  auto exp = exponential.getDefiningOp<UnaryOp>();
  if (!exp || exp.getKind() != "exp")
    return false;
  auto subtract = exp.getInput().getDefiningOp<BinaryOp>();
  return subtract && subtract.getKind() == "sub" &&
         isTupleField(subtract.getLhs(), state, 0) &&
         subtract.getRhs() == maximum;
}

bool isOnlineSoftmaxSummary(SummaryFoldOp op) {
  if (op.getOrder() != "preserve" || !isTrue(op.getWhere()) ||
      !mlir::isa<mlir::NoneType>(op.getCoordinate().getType()))
    return false;
  auto resultType = mlir::dyn_cast<TupleType>(op.getResult().getType());
  if (!resultType || resultType.getTypes().size() != 2 ||
      !resultType.getTypes()[0].isF32() || !resultType.getTypes()[1].isF32())
    return false;

  auto identity = op.getIdentity().getDefiningOp<TupleOp>();
  if (!identity || identity.getNumOperands() != 2 ||
      !identity.getOperand(0).getDefiningOp<SpecialValueOp>() ||
      identity.getOperand(0).getDefiningOp<SpecialValueOp>().getKind() !=
          "neg_inf" ||
      !isFloatConstant(identity.getOperand(1), 0.0))
    return false;

  mlir::Block &lift = op.getLift().front();
  auto liftYield = mlir::cast<YieldOp>(lift.getTerminator());
  auto lifted = liftYield.getOperand(0).getDefiningOp<TupleOp>();
  if (!lifted || lifted.getNumOperands() != 2 ||
      lifted.getOperand(0) != lift.getArgument(0) ||
      !isFloatConstant(lifted.getOperand(1), 1.0))
    return false;

  mlir::Block &merge = op.getMerge().front();
  auto mergeYield = mlir::cast<YieldOp>(merge.getTerminator());
  auto merged = mergeYield.getOperand(0).getDefiningOp<TupleOp>();
  if (!merged || merged.getNumOperands() != 2)
    return false;
  auto maximum = merged.getOperand(0).getDefiningOp<BinaryOp>();
  if (!maximum || maximum.getKind() != "max")
    return false;
  bool maxOperandsMatch =
      (isTupleField(maximum.getLhs(), merge.getArgument(0), 0) &&
       isTupleField(maximum.getRhs(), merge.getArgument(1), 0)) ||
      (isTupleField(maximum.getLhs(), merge.getArgument(1), 0) &&
       isTupleField(maximum.getRhs(), merge.getArgument(0), 0));
  if (!maxOperandsMatch)
    return false;
  auto sum = merged.getOperand(1).getDefiningOp<BinaryOp>();
  if (!sum || sum.getKind() != "add")
    return false;
  return (matchesScaledSummaryTerm(sum.getLhs(), merge.getArgument(0),
                                   maximum.getResult()) &&
          matchesScaledSummaryTerm(sum.getRhs(), merge.getArgument(1),
                                   maximum.getResult())) ||
         (matchesScaledSummaryTerm(sum.getLhs(), merge.getArgument(1),
                                   maximum.getResult()) &&
          matchesScaledSummaryTerm(sum.getRhs(), merge.getArgument(0),
                                   maximum.getResult()));
}

bool matchesTupleFieldCompare(mlir::Value value, llvm::StringRef predicate,
                              mlir::BlockArgument lhsState, int64_t lhsIndex,
                              mlir::BlockArgument rhsState,
                              int64_t rhsIndex) {
  auto compare = value.getDefiningOp<CompareOp>();
  return compare && compare.getPredicate() == predicate &&
         isTupleField(compare.getLhs(), lhsState, lhsIndex) &&
         isTupleField(compare.getRhs(), rhsState, rhsIndex);
}

bool matchesArgMaxTie(mlir::Value value, mlir::BlockArgument lhsState,
                      mlir::BlockArgument rhsState) {
  auto conjunction = value.getDefiningOp<BinaryOp>();
  if (!conjunction || conjunction.getKind() != "and")
    return false;
  auto matchesEquality = [&](mlir::Value candidate) {
    return matchesTupleFieldCompare(candidate, "eq", rhsState, 0, lhsState,
                                    0);
  };
  auto matchesLowerIndex = [&](mlir::Value candidate) {
    return matchesTupleFieldCompare(candidate, "lt", rhsState, 1, lhsState,
                                    1);
  };
  return (matchesEquality(conjunction.getLhs()) &&
          matchesLowerIndex(conjunction.getRhs())) ||
         (matchesEquality(conjunction.getRhs()) &&
          matchesLowerIndex(conjunction.getLhs()));
}

bool matchesArgMaxTakeRight(mlir::Value value, mlir::BlockArgument lhsState,
                            mlir::BlockArgument rhsState) {
  auto disjunction = value.getDefiningOp<BinaryOp>();
  if (!disjunction || disjunction.getKind() != "or")
    return false;
  auto matchesGreater = [&](mlir::Value candidate) {
    return matchesTupleFieldCompare(candidate, "gt", rhsState, 0, lhsState,
                                    0);
  };
  return (matchesGreater(disjunction.getLhs()) &&
          matchesArgMaxTie(disjunction.getRhs(), lhsState, rhsState)) ||
         (matchesGreater(disjunction.getRhs()) &&
          matchesArgMaxTie(disjunction.getLhs(), lhsState, rhsState));
}

bool isArgMaxSummary(SummaryFoldOp op) {
  if (op.getOrder() != "relaxed" || !isTrue(op.getWhere()) ||
      !elementType(op.getInput().getType()).isF32() ||
      !elementType(op.getCoordinate().getType()).isIndex())
    return false;
  auto resultType = mlir::dyn_cast<TupleType>(op.getResult().getType());
  if (!resultType || resultType.getTypes().size() != 2 ||
      !resultType.getTypes()[0].isF32() ||
      !resultType.getTypes()[1].isIndex())
    return false;

  auto identity = op.getIdentity().getDefiningOp<TupleOp>();
  if (!identity || identity.getNumOperands() != 2 ||
      !identity.getOperand(0).getDefiningOp<SpecialValueOp>() ||
      identity.getOperand(0).getDefiningOp<SpecialValueOp>().getKind() !=
          "neg_inf" ||
      integerConstantValue(identity.getOperand(1)) != 0)
    return false;

  mlir::Block &lift = op.getLift().front();
  auto liftYield = mlir::cast<YieldOp>(lift.getTerminator());
  auto lifted = liftYield.getOperand(0).getDefiningOp<TupleOp>();
  if (lift.getNumArguments() != 2 || !lifted ||
      lifted.getNumOperands() != 2 ||
      lifted.getOperand(0) != lift.getArgument(0) ||
      lifted.getOperand(1) != lift.getArgument(1))
    return false;

  mlir::Block &merge = op.getMerge().front();
  auto mergeYield = mlir::cast<YieldOp>(merge.getTerminator());
  auto merged = mergeYield.getOperand(0).getDefiningOp<TupleOp>();
  if (!merged || merged.getNumOperands() != 2)
    return false;
  auto valueSelect = merged.getOperand(0).getDefiningOp<SelectOp>();
  auto indexSelect = merged.getOperand(1).getDefiningOp<SelectOp>();
  if (!valueSelect || !indexSelect ||
      valueSelect.getPredicate() != indexSelect.getPredicate() ||
      !matchesArgMaxTakeRight(valueSelect.getPredicate(), merge.getArgument(0),
                              merge.getArgument(1)) ||
      !isTupleField(valueSelect.getTrueValue(), merge.getArgument(1), 0) ||
      !isTupleField(valueSelect.getFalseValue(), merge.getArgument(0), 0) ||
      !isTupleField(indexSelect.getTrueValue(), merge.getArgument(1), 1) ||
      !isTupleField(indexSelect.getFalseValue(), merge.getArgument(0), 1))
    return false;

  mlir::Block &finalize = op.getFinalize().front();
  auto finalizeYield = mlir::cast<YieldOp>(finalize.getTerminator());
  return finalize.getNumArguments() == 1 &&
         finalizeYield.getNumOperands() == 1 &&
         finalizeYield.getOperand(0) == finalize.getArgument(0);
}

class KernelEmitter {
public:
  KernelEmitter(KernelOp kernel, const RISCVLoweringOptions &options,
                llvm::raw_ostream &output)
      : kernel(kernel), options(options), output(output) {}

  mlir::LogicalResult emit() {
    mlir::Block &body = kernel.getBody().front();
    llvm::SmallVector<std::string> parameters;
    auto names = kernel.getArgNames();
    auto kinds = kernel.getArgKinds();
    for (auto [index, argument] : llvm::enumerate(body.getArguments())) {
      llvm::StringRef kind =
          mlir::cast<mlir::StringAttr>(kinds[index]).getValue();
      if (kind == "constexpr")
        continue;
      std::string name = sanitize(
          mlir::cast<mlir::StringAttr>(names[index]).getValue());
      mlir::Type type = argument.getType();
      std::string cType;
      CValueKind valueKind = CValueKind::Scalar;
      if (auto pointer = mlir::dyn_cast<PtrType>(type)) {
        cType = pointerCType(pointer);
        valueKind = CValueKind::Pointer;
        if (pointer.getNoAlias())
          cType += " restrict";
      } else {
        cType = scalarCType(type);
      }
      if (cType.empty())
        return kernel.emitError(
            "RISC-V intrinsic C ABI has an unsupported argument type");
      parameters.push_back(cType + " " + name);
      values[argument] = CValue{type, valueKind, name};
    }

    mlir::Type returnType = kernel.getReturnType();
    std::string returnTypeSpelling = mlir::isa<mlir::NoneType>(returnType)
                                         ? "void"
                                         : scalarCType(returnType);
    if (returnTypeSpelling.empty())
      return kernel.emitError(
          "RISC-V intrinsic C ABI has an unsupported return type");
    line(returnTypeSpelling + " " + sanitize(kernel.getSymName()) + "(" +
         llvm::join(parameters, ", ") + ") {");
    ++indent;
    for (mlir::Operation &operation : body) {
      if (auto returnOp = mlir::dyn_cast<ReturnOp>(operation)) {
        if (returnOp.getNumOperands() == 0)
          line("return;");
        else
          line("return " + require(returnOp.getOperand(0)).spelling + ";");
        continue;
      }
      if (mlir::failed(emitOperation(&operation)))
        return mlir::failure();
    }
    for (mlir::Operation *operation : deferredBlockOps)
      if (!loweredBlockOps.contains(operation) &&
          !llvm::all_of(operation->getResults(),
                        [](mlir::Value value) { return value.use_empty(); }))
        return operation->emitError(
            "logical block is not owned by a supported target lowering");
    --indent;
    line("}");
    line("");
    return mlir::success();
  }

private:
  KernelOp kernel;
  const RISCVLoweringOptions &options;
  llvm::raw_ostream &output;
  llvm::DenseMap<mlir::Value, CValue> values;
  llvm::DenseSet<mlir::Operation *> consumed;
  llvm::DenseSet<mlir::Operation *> deferredBlockOps;
  llvm::DenseSet<mlir::Operation *> loweredBlockOps;
  unsigned indent = 0;
  unsigned nextValue = 0;
  unsigned nextLoop = 0;
  bool inVLA = false;
  bool microBlockVectors = false;
  std::string activeVL;
  const VLARegionDecision *activeVLADecision = nullptr;

  void line(llvm::Twine text) {
    output.indent(indent * 2) << text << '\n';
  }

  std::string fresh(llvm::StringRef prefix) {
    return "__weft_" + prefix.str() + std::to_string(nextValue++);
  }

  CValue scalarExpression(mlir::Value result, std::string expression,
                          llvm::StringRef prefix, bool force = false) {
    if (force || (!result.use_empty() && !result.hasOneUse())) {
      std::string name = fresh(prefix);
      line("const " + scalarCType(elementType(result.getType())) + " " + name +
           " = " + expression + ";");
      expression = std::move(name);
    }
    return CValue{result.getType(), CValueKind::Scalar, std::move(expression)};
  }

  CValue require(mlir::Value value) const {
    auto found = values.find(value);
    return found == values.end() ? CValue{} : found->second;
  }

  std::string expression(mlir::Value value) const {
    if (auto constant = value.getDefiningOp<ConstantOp>())
      return constantLiteral(constant.getValue());
    if (auto special = value.getDefiningOp<SpecialValueOp>()) {
      if (special.getKind() == "neg_inf")
        return "-INFINITY";
      if (special.getKind() == "pos_inf")
        return "INFINITY";
      if (special.getKind() == "nan")
        return "NAN";
    }
    return require(value).spelling;
  }

  const VLAPredicateDecision *
  findPredicateDecision(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return nullptr;
    auto found = llvm::find_if(
        activeVLADecision->predicates,
        [&](const VLAPredicateDecision &decision) {
          return decision.operation == operation;
        });
    return found == activeVLADecision->predicates.end() ? nullptr : &*found;
  }

  const VLAAccessDecision *
  findAccessDecision(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return nullptr;
    auto found = llvm::find_if(
        activeVLADecision->accesses, [&](const VLAAccessDecision &decision) {
          return decision.operation == operation;
        });
    return found == activeVLADecision->accesses.end() ? nullptr : &*found;
  }

  const VLAStateDecision *
  findStateDecision(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return nullptr;
    auto found = llvm::find_if(
        activeVLADecision->states, [&](const VLAStateDecision &decision) {
          return decision.operation == operation;
        });
    return found == activeVLADecision->states.end() ? nullptr : &*found;
  }

  const VLANarrowDecision *
  findNarrowDecision(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return nullptr;
    auto found = llvm::find_if(
        activeVLADecision->narrows, [&](const VLANarrowDecision &decision) {
          return decision.operation == operation;
        });
    return found == activeVLADecision->narrows.end() ? nullptr : &*found;
  }

  const VLAContractDecision *
  findVLAContractDecision(mlir::Operation *consumer) const {
    if (!activeVLADecision)
      return nullptr;
    auto found = llvm::find_if(
        activeVLADecision->contracts,
        [&](const VLAContractDecision &decision) {
          return decision.consumer == consumer;
        });
    return found == activeVLADecision->contracts.end() ? nullptr : &*found;
  }

  bool isVLAContractAbsorbed(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return false;
    return llvm::any_of(
        activeVLADecision->contracts,
        [&](const VLAContractDecision &decision) {
          return llvm::is_contained(decision.absorbed, operation);
        });
  }

  void collectLocalDefinitions(
      mlir::Value value, mlir::Block &block,
      llvm::DenseSet<mlir::Operation *> &definitions,
      llvm::SmallVectorImpl<BlockAxisOp> &axes) const {
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || definition->getBlock() != &block ||
        !definitions.insert(definition).second)
      return;
    if (auto axis = mlir::dyn_cast<BlockAxisOp>(definition))
      axes.push_back(axis);
    for (mlir::Value operand : definition->getOperands())
      collectLocalDefinitions(operand, block, definitions, axes);
  }

  mlir::FailureOr<VLAContractDecision>
  decideVLAF32FreeAxisMicrotile(VLAOp vla, StoreOp store,
                                 ContractOp contract) {
    auto unwrapBlock = [](mlir::Type type) -> BlockType {
      if (auto masked = mlir::dyn_cast<MaskedType>(type))
        type = masked.getValueType();
      return mlir::dyn_cast<BlockType>(type);
    };
    auto unwrapRegion = [](mlir::Type type) -> RegionType {
      if (auto masked = mlir::dyn_cast<MaskedType>(type))
        type = masked.getValueType();
      return mlir::dyn_cast<RegionType>(type);
    };

    BlockType lhsType = unwrapBlock(contract.getLhs().getType());
    RegionType rhsType = unwrapRegion(contract.getRhs().getType());
    RegionType resultType = unwrapRegion(contract.getResult().getType());
    if (store.getValue() != contract.getResult() ||
        !contract.getResult().hasOneUse() || !lhsType || !rhsType ||
        !resultType || lhsType.getShape().size() != 2 ||
        rhsType.getShape().size() != 2 || resultType.getShape().size() != 2 ||
        rhsType.getShape()[0] != -1 || resultType.getShape()[0] != -1 ||
        lhsType.getShape()[0] != resultType.getShape()[1] ||
        lhsType.getShape()[1] != rhsType.getShape()[1] ||
        lhsType.getShape()[0] <= 0 || lhsType.getShape()[0] > 6 ||
        !lhsType.getElementType().isF32() ||
        !rhsType.getElementType().isF32() ||
        !resultType.getElementType().isF32() ||
        !isFloatConstant(contract.getInit(), 0.0) ||
        contract.getLhsAxes().size() != 1 ||
        contract.getRhsAxes().size() != 1 ||
        contract.getLhsAxes().front() != 1 ||
        contract.getRhsAxes().front() != 1 ||
        contract.getOrder() != "relaxed" || contract.getMath() != "native" ||
        !contract.getAccDtype().isF32() || !contract.getOutDtype().isF32() ||
        !isTrue(contract.getWhereLhs()) || !isTrue(contract.getWhereRhs())) {
      contract.emitError(
          "RVV VLA contract requires a [BM,K] x [VLA,K] local f32 primitive");
      return mlir::failure();
    }

    LoadOp lhsLoad = contract.getLhs().getDefiningOp<LoadOp>();
    LoadOp rhsLoad = contract.getRhs().getDefiningOp<LoadOp>();
    if (!lhsLoad || !rhsLoad || !isTrue(rhsLoad.getWhere()) ||
        (!isTrue(lhsLoad.getWhere()) &&
         !isFloatConstant(lhsLoad.getOther(), 0.0))) {
      contract.emitError("RVV VLA contract load facts are unavailable");
      return mlir::failure();
    }

    mlir::Block &block = *store->getBlock();
    llvm::DenseSet<mlir::Operation *> lhsDefinitions;
    llvm::DenseSet<mlir::Operation *> rhsDefinitions;
    llvm::DenseSet<mlir::Operation *> outputDefinitions;
    llvm::SmallVector<BlockAxisOp> lhsAxes;
    llvm::SmallVector<BlockAxisOp> rhsAxes;
    llvm::SmallVector<BlockAxisOp> outputAxes;
    collectLocalDefinitions(contract.getLhs(), block, lhsDefinitions, lhsAxes);
    collectLocalDefinitions(contract.getRhs(), block, rhsDefinitions, rhsAxes);
    collectLocalDefinitions(store.getPointer(), block, outputDefinitions,
                            outputAxes);
    collectLocalDefinitions(store.getWhere(), block, outputDefinitions,
                            outputAxes);
    if (rhsAxes.size() != 1 || lhsAxes.size() != 2 || outputAxes.size() != 1 ||
        !llvm::is_contained(lhsAxes, rhsAxes.front())) {
      contract.emitError(
          "RVV VLA contract requires explicit row and shared reduction axes");
      return mlir::failure();
    }
    BlockAxisOp reductionAxis = rhsAxes.front();
    BlockAxisOp rowAxis = lhsAxes.front() == reductionAxis
                              ? lhsAxes.back()
                              : lhsAxes.front();
    if (outputAxes.front() != rowAxis ||
        integerConstantValue(rowAxis.getExtent()) != lhsType.getShape()[0]) {
      contract.emitError(
          "RVV VLA contract result axes do not preserve the local row block");
      return mlir::failure();
    }

    mlir::Value coordinate = vla.getBody().front().getArgument(0);
    mlir::Value lhsRoot = pointerRoot(lhsLoad.getPointer());
    mlir::Value rhsRoot = pointerRoot(rhsLoad.getPointer());
    mlir::Value outputRoot = pointerRoot(store.getPointer());
    auto lhsPointer =
        lhsRoot ? mlir::dyn_cast<PtrType>(lhsRoot.getType()) : PtrType{};
    auto rhsPointer =
        rhsRoot ? mlir::dyn_cast<PtrType>(rhsRoot.getType()) : PtrType{};
    auto outputPointer =
        outputRoot ? mlir::dyn_cast<PtrType>(outputRoot.getType()) : PtrType{};
    LaneRelation rhsRelation =
        classifyLaneRelation(rhsLoad.getPointer(), coordinate);
    LaneRelation outputRelation =
        classifyLaneRelation(store.getPointer(), coordinate);
    if (!lhsPointer || !rhsPointer || !outputPointer ||
        !lhsPointer.getElementType().isF32() ||
        !rhsPointer.getElementType().isF32() ||
        !outputPointer.getElementType().isF32() ||
        (rhsRelation != LaneRelation::UnitStride &&
         rhsRelation != LaneRelation::Strided) ||
        (outputRelation != LaneRelation::UnitStride &&
         outputRelation != LaneRelation::Strided) ||
        !dependsOn(lhsLoad.getPointer(), rowAxis.getResult()) ||
        !dependsOn(lhsLoad.getPointer(), reductionAxis.getResult()) ||
        dependsOn(lhsLoad.getPointer(), coordinate) ||
        !dependsOn(rhsLoad.getPointer(), coordinate) ||
        !dependsOn(rhsLoad.getPointer(), reductionAxis.getResult()) ||
        dependsOn(rhsLoad.getPointer(), rowAxis.getResult()) ||
        !dependsOn(store.getPointer(), coordinate) ||
        !dependsOn(store.getPointer(), rowAxis.getResult()) ||
        dependsOn(store.getPointer(), reductionAxis.getResult()) ||
        dependsOn(store.getWhere(), reductionAxis.getResult())) {
      contract.emitError(
          "RVV VLA contract memory relations are unavailable");
      return mlir::failure();
    }

    llvm::DenseSet<mlir::Operation *> absorbed;
    llvm::SmallVector<BlockAxisOp> absorbedAxes;
    for (mlir::Value value : {store.getPointer(), store.getValue(),
                              store.getWhere()})
      collectLocalDefinitions(value, block, absorbed, absorbedAxes);

    VLAContractDecision decision;
    decision.operation = contract.getOperation();
    decision.realization =
        VLAContractRealization::RVVF32FreeAxisMicrotile;
    decision.consumer = store.getOperation();
    decision.lhsLoad = lhsLoad.getOperation();
    decision.rhsLoad = rhsLoad.getOperation();
    decision.rowAxis = rowAxis.getResult();
    decision.reductionAxis = reductionAxis.getResult();
    decision.reductionExtent = reductionAxis.getExtent();
    decision.rowTile = static_cast<unsigned>(lhsType.getShape()[0]);
    decision.lmul = 4;
    decision.rhsMemoryMode = rhsRelation == LaneRelation::UnitStride
                                 ? VLAMemoryMode::UnitStride
                                 : VLAMemoryMode::Strided;
    decision.outputMemoryMode = outputRelation == LaneRelation::UnitStride
                                    ? VLAMemoryMode::UnitStride
                                    : VLAMemoryMode::Strided;
    decision.lhsPredicateVariesByReduction =
        dependsOn(lhsLoad.getWhere(), reductionAxis.getResult());
    decision.absorbed.append(absorbed.begin(), absorbed.end());
    return decision;
  }

  mlir::FailureOr<VLARegionDecision> decideVLARegion(VLAOp op) {
    mlir::Block &body = op.getBody().front();
    VLARegionDecision decision;
    decision.operation = op.getOperation();
    decision.coordinate = body.getArgument(0);
    if (options.target.xlen == 32) {
      decision.indexSEW = 32;
      decision.indexLMUL = 2;
    } else if (options.target.xlen != 64) {
      op.emitError("VLA index realization requires a 32-bit or 64-bit target");
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Operation *> physicalOperations;
    std::function<void(mlir::Block &)> collectPhysicalOperations =
        [&](mlir::Block &block) {
          for (mlir::Operation &nested : block.without_terminator()) {
            physicalOperations.push_back(&nested);
            if (auto loop = mlir::dyn_cast<ForOp>(nested))
              collectPhysicalOperations(loop.getBody().front());
          }
        };
    collectPhysicalOperations(body);

    for (mlir::Operation *nested : physicalOperations) {
      auto store = mlir::dyn_cast<StoreOp>(nested);
      auto contract =
          store ? store.getValue().getDefiningOp<ContractOp>() : ContractOp{};
      if (!contract || !isRegionValue(contract.getResult().getType()))
        continue;
      mlir::FailureOr<VLAContractDecision> selected =
          decideVLAF32FreeAxisMicrotile(op, store, contract);
      if (mlir::failed(selected))
        return mlir::failure();
      decision.contracts.push_back(std::move(*selected));
    }
    if (!decision.contracts.empty()) {
      decision.dataLMUL = 4;
      decision.maskRatio = 8;
      decision.indexLMUL = options.target.xlen == 64 ? 8 : 4;
    }

    auto isContractOwned = [&](mlir::Operation *operation) {
      return llvm::any_of(
          decision.contracts, [&](const VLAContractDecision &contract) {
            return contract.consumer == operation ||
                   llvm::is_contained(contract.absorbed, operation);
          });
    };

    for (mlir::Operation *nested : physicalOperations) {
      if (isContractOwned(nested))
        continue;
      auto compare = mlir::dyn_cast<CompareOp>(nested);
      if (!compare || !isRegionValue(compare.getResult().getType()))
        continue;
      LaneRelation lhs =
          classifyLaneRelation(compare.getLhs(), decision.coordinate);
      LaneRelation rhs =
          classifyLaneRelation(compare.getRhs(), decision.coordinate);
      bool lhsCoordinate = lhs != LaneRelation::Independent &&
                           lhs != LaneRelation::NonAffine;
      bool rhsCoordinate = rhs != LaneRelation::Independent &&
                           rhs != LaneRelation::NonAffine;
      if (lhs == LaneRelation::NonAffine || rhs == LaneRelation::NonAffine ||
          lhsCoordinate == rhsCoordinate ||
          !elementType(lhsCoordinate ? compare.getLhs().getType()
                                     : compare.getRhs().getType())
               .isIndex()) {
        compare.emitError(
            "VLA predicate requires one affine index coordinate and one scalar");
        return mlir::failure();
      }
      std::string predicate =
          lhsCoordinate ? compare.getPredicate().str()
                        : reversePredicate(compare.getPredicate());
      if (predicate.empty()) {
        compare.emitError("VLA predicate has an unsupported comparison");
        return mlir::failure();
      }
      decision.predicates.push_back(VLAPredicateDecision{
          compare.getOperation(),
          lhsCoordinate ? compare.getLhs() : compare.getRhs(),
          lhsCoordinate ? compare.getRhs() : compare.getLhs(),
          (lhsCoordinate ? lhs : rhs) == LaneRelation::UnitStride
              ? VLAMemoryMode::UnitStride
              : VLAMemoryMode::Strided,
          std::move(predicate)});
    }

    auto addAccess = [&](mlir::Operation *operation, mlir::Value pointer,
                         mlir::Value predicate, mlir::Type accessedElement,
                         VLAStoreValueMode storeValueMode)
        -> mlir::LogicalResult {
      LaneRelation relation =
          classifyLaneRelation(pointer, decision.coordinate);
      if (relation == LaneRelation::Independent)
        return mlir::success();
      if (relation == LaneRelation::NonAffine)
        return operation->emitError(
            "VLA memory access is neither affine-strided nor explicitly indexed");
      VLAActivityMode activityMode = VLAActivityMode::AllActive;
      if (!isTrue(predicate)) {
        bool hasPredicateDecision = llvm::any_of(
            decision.predicates, [&](const VLAPredicateDecision &candidate) {
              return candidate.operation == predicate.getDefiningOp();
            });
        if (hasPredicateDecision) {
          activityMode = VLAActivityMode::PredicateMask;
        } else if (classifyLaneRelation(predicate, decision.coordinate) ==
                       LaneRelation::Independent &&
                   elementType(predicate.getType()).isInteger(1) &&
                   !isRegionValue(predicate.getType())) {
          activityMode = VLAActivityMode::ScalarPredicate;
        } else {
          return operation->emitError(
              "VLA memory predicate has no physical mask decision");
        }
      }
      decision.accesses.push_back(VLAAccessDecision{
          operation,
          accessedElement,
          relation == LaneRelation::UnitStride ? VLAMemoryMode::UnitStride
                                               : VLAMemoryMode::Strided,
          activityMode,
          storeValueMode,
          predicate});
      return mlir::success();
    };

    for (mlir::Operation *nested : physicalOperations) {
      if (isContractOwned(nested))
        continue;
      if (auto load = mlir::dyn_cast<LoadOp>(nested)) {
        if (mlir::failed(addAccess(
                load.getOperation(), load.getPointer(), load.getWhere(),
                elementType(load.getResult().getType()),
                VLAStoreValueMode::Vector)))
          return mlir::failure();
      } else if (auto store = mlir::dyn_cast<StoreOp>(nested)) {
        VLAStoreValueMode valueMode =
            isRegionValue(store.getValue().getType())
                ? VLAStoreValueMode::Vector
                : VLAStoreValueMode::ScalarBroadcast;
        if (mlir::failed(addAccess(
                store.getOperation(), store.getPointer(), store.getWhere(),
                elementType(store.getValue().getType()), valueMode)))
          return mlir::failure();
      }
    }

    for (mlir::Operation *nested : physicalOperations) {
      if (isContractOwned(nested))
        continue;
      auto narrow = mlir::dyn_cast<NarrowOp>(nested);
      if (!narrow)
        continue;
      if (!elementType(narrow.getInput().getType()).isF32() ||
          !elementType(narrow.getResult().getType()).isSignedInteger(8) ||
          narrow.getRounding() != "rne" || !narrow.getSaturation()) {
        narrow.emitError(
            "VLA narrow has no selected saturating f32-to-i8 realization");
        return mlir::failure();
      }
      decision.dataLMUL = 4;
      decision.maskRatio = 8;
      decision.indexLMUL = options.target.xlen == 64 ? 8 : 4;
      decision.narrows.push_back(VLANarrowDecision{narrow.getOperation()});
    }

    for (mlir::Operation &nested : body.without_terminator()) {
      if (isContractOwned(&nested))
        continue;
      if (auto reduce = mlir::dyn_cast<ReduceOp>(nested)) {
        if (reduce.getAxis() != -1 ||
            !elementType(reduce.getInput().getType()).isF32() ||
            !reduce.getResult().getType().isF32() || !isTrue(reduce.getWhere())) {
          reduce.emitError(
              "VLA reduction has no selected f32 all-active realization");
          return mlir::failure();
        }
        VLAStateRealization realization;
        if (reduce.getKind() == "add")
          realization = VLAStateRealization::RVVAddReduction;
        else if (reduce.getKind() == "max")
          realization = VLAStateRealization::RVVMaxReduction;
        else {
          reduce.emitError("VLA reduction kind has no physical realization");
          return mlir::failure();
        }
        decision.states.push_back(VLAStateDecision{
            reduce.getOperation(), realization, reduce.getResult().getType(),
            reduce.getIdentity()});
      } else if (auto scan = mlir::dyn_cast<ScanOp>(nested)) {
        if (!elementType(scan.getInput().getType()).isF32() ||
            !elementType(scan.getResult().getType()).isF32() ||
            !scan.getIdentity().getType().isF32() || !isTrue(scan.getWhere()) ||
            !mlir::isa<mlir::NoneType>(scan.getSegmentStart().getType()) ||
            scan.getKind() != "add" || !scan.getInclusive() ||
            scan.getOrder() != "ordered") {
          scan.emitError(
              "VLA scan has no selected all-active ordered f32 realization");
          return mlir::failure();
        }
        decision.states.push_back(VLAStateDecision{
            scan.getOperation(), VLAStateRealization::RVVInclusiveAddScan,
            elementType(scan.getResult().getType()), scan.getIdentity()});
      } else if (auto summary = mlir::dyn_cast<SummaryFoldOp>(nested)) {
        if (isArgMaxSummary(summary)) {
          LaneRelation coordinate =
              classifyLaneRelation(summary.getCoordinate(), decision.coordinate);
          if (coordinate != LaneRelation::UnitStride &&
              coordinate != LaneRelation::Strided) {
            summary.emitError("argmax coordinate is not affine in the VLA axis");
            return mlir::failure();
          }
          VLAStateDecision state{
              summary.getOperation(), VLAStateRealization::RVVArgMaxSummary,
              summary.getResult().getType(), summary.getIdentity()};
          state.coordinateMode = coordinate == LaneRelation::UnitStride
                                     ? VLAMemoryMode::UnitStride
                                     : VLAMemoryMode::Strided;
          decision.states.push_back(state);
        }
      }
    }
    if (!decision.narrows.empty() && !decision.states.empty()) {
      op.emitError(
          "one VLA region cannot currently share a narrow and aggregate realization");
      return mlir::failure();
    }
    return decision;
  }

  mlir::LogicalResult emitOperation(mlir::Operation *operation) {
    if (consumed.contains(operation))
      return mlir::success();
    if (const VLAContractDecision *decision =
            findVLAContractDecision(operation))
      return emitVLAContract(*decision);
    if (isVLAContractAbsorbed(operation))
      return mlir::success();
    if (auto op = mlir::dyn_cast<ReduceOp>(operation))
      return emitReduce(op);
    if (auto op = mlir::dyn_cast<ScanOp>(operation))
      return op.emitError("scan must be lowered by its enclosing VLA region");
    if (auto op = mlir::dyn_cast<GroupedAffineI4I8DotOp>(operation))
      return emitGroupedAffineI4I8Dot(op);
    if (auto op = mlir::dyn_cast<SymmetricI4I8ContractOp>(operation))
      return emitSymmetricI4I8Contract(op);
    if (auto op = mlir::dyn_cast<FullOp>(operation)) {
      auto block = mlir::dyn_cast<BlockType>(op.getResult().getType());
      if (block && block.getShape() == llvm::ArrayRef<int64_t>({16}) &&
          block.getElementType().isF32())
        return emitMaterializedBlockFull(op);
    }
    if (auto op = mlir::dyn_cast<ForOp>(operation))
      return emitFor(op);
    if (auto op = mlir::dyn_cast<WhileOp>(operation))
      return emitWhile(op);
    if (auto op = mlir::dyn_cast<StoreOp>(operation))
      if (auto lowered = tryEmitMaterializedF32BlockStore(op))
        return *lowered;
    if (hasBlockPayload(operation)) {
      if (auto op = mlir::dyn_cast<StoreOp>(operation))
        return emitBlockStores(op);
      deferredBlockOps.insert(operation);
      return mlir::success();
    }
    if (auto op = mlir::dyn_cast<ConstantOp>(operation))
      return emitConstant(op);
    if (auto op = mlir::dyn_cast<MetaValueOp>(operation))
      return emitMeta(op);
    if (auto op = mlir::dyn_cast<IfOp>(operation))
      return emitIf(op);
    if (auto op = mlir::dyn_cast<VLAOp>(operation))
      return emitVLA(op);
    if (auto op = mlir::dyn_cast<PtrAddOp>(operation))
      return emitPtrAdd(op);
    if (auto op = mlir::dyn_cast<UnaryOp>(operation))
      return emitUnary(op);
    if (auto op = mlir::dyn_cast<BinaryOp>(operation))
      return emitBinary(op);
    if (auto op = mlir::dyn_cast<CompareOp>(operation))
      return emitCompare(op);
    if (auto op = mlir::dyn_cast<CastOp>(operation))
      return emitCast(op);
    if (auto op = mlir::dyn_cast<NarrowOp>(operation))
      return emitNarrow(op);
    if (auto op = mlir::dyn_cast<BitcastOp>(operation))
      return emitBitcast(op);
    if (auto op = mlir::dyn_cast<SelectOp>(operation))
      return emitSelect(op);
    if (auto op = mlir::dyn_cast<TupleOp>(operation))
      return emitTuple(op);
    if (auto op = mlir::dyn_cast<TupleGetOp>(operation))
      return emitTupleGet(op);
    if (auto op = mlir::dyn_cast<SpecialValueOp>(operation))
      return emitSpecial(op);
    if (auto op = mlir::dyn_cast<InvalidOp>(operation)) {
      values[op.getResult()] =
          CValue{op.getResult().getType(), CValueKind::Scalar, {}};
      return mlir::success();
    }
    if (auto op = mlir::dyn_cast<LoadOp>(operation))
      return emitLoad(op);
    if (auto op = mlir::dyn_cast<StoreOp>(operation)) {
      return emitStore(op);
    }
    if (auto op = mlir::dyn_cast<SummaryFoldOp>(operation))
      return op.emitError(
          "summary_fold must be lowered by its enclosing VLA region");
    if (mlir::isa<YieldOp, ReturnOp>(operation))
      return mlir::success();
    return operation->emitError()
           << "RISC-V target lowering does not implement Kernel IR primitive "
           << operation->getName();
  }

  mlir::LogicalResult emitConstant(ConstantOp op) {
    std::string literal = constantLiteral(op.getValue());
    if (literal.empty())
      return op.emitError("RISC-V target cannot spell constant");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, literal};
    return mlir::success();
  }

  mlir::LogicalResult emitMeta(MetaValueOp op) {
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(op.getInput());
    if (!argument)
      return op.emitError("meta value must reference a kernel argument");
    auto names = kernel.getArgNames();
    llvm::StringRef name = mlir::cast<mlir::StringAttr>(
                               names[argument.getArgNumber()])
                               .getValue();
    auto binding = options.metaBindings.find(name);
    if (binding == options.metaBindings.end())
      return op.emitError() << "missing --meta binding for " << name;
    std::string value = std::to_string(binding->second);
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, value};
    return mlir::success();
  }

  mlir::LogicalResult emitMaterializedBlockFull(FullOp op) {
    auto block = mlir::dyn_cast<BlockType>(op.getResult().getType());
    CValue fill = require(op.getValue());
    if (!block || block.getShape() != llvm::ArrayRef<int64_t>({16}) ||
        !block.getElementType().isF32() || fill.kind != CValueKind::Scalar ||
        fill.spelling.empty())
      return op.emitError(
          "materialized block full requires a scalar-filled f32 block<16>");
    std::string storage = fresh("block_storage");
    std::string lane = fresh("block_init");
    line("float " + storage + "[16];");
    line("for (size_t " + lane + " = 0; " + lane + " < 16; ++" + lane + ")");
    ++indent;
    line(storage + "[" + lane + "] = " + fill.spelling + ";");
    --indent;
    values[op.getResult()] = CValue{op.getResult().getType(),
                                    CValueKind::F32BlockStorage, storage};
    loweredBlockOps.insert(op.getOperation());
    return mlir::success();
  }

  mlir::LogicalResult emitIf(IfOp op) {
    CValue condition = require(op.getCondition());
    if (condition.kind != CValueKind::Scalar || condition.spelling.empty())
      return op.emitError("RISC-V conditional requires a scalar predicate");
    auto thenYield =
        mlir::cast<YieldOp>(op.getThenRegion().front().getTerminator());
    auto elseYield =
        mlir::cast<YieldOp>(op.getElseRegion().front().getTerminator());
    if (thenYield.getNumOperands() != op.getNumResults() ||
        elseYield.getNumOperands() != op.getNumResults())
      return op.emitError("RISC-V conditional result arity is inconsistent");

    llvm::SmallVector<CValue> results;
    for (mlir::Value result : op.getResults()) {
      if (!mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
              result.getType()))
        return op.emitError(
            "RISC-V conditional currently yields only scalar values");
      CValue value{result.getType(), CValueKind::Scalar, fresh("if_result")};
      line(scalarCType(result.getType()) + " " + value.spelling + ";");
      results.push_back(value);
    }

    auto emitBranch = [&](mlir::Region &region) {
      mlir::Block &body = region.front();
      for (mlir::Operation &nested : body.without_terminator())
        if (mlir::failed(emitOperation(&nested)))
          return mlir::failure();
      auto yield = mlir::cast<YieldOp>(body.getTerminator());
      for (auto [destination, yielded] :
           llvm::zip(results, yield.getOperands())) {
        CValue value = require(yielded);
        if (value.kind != CValueKind::Scalar || value.spelling.empty()) {
          yield.emitError(
              "RISC-V conditional yielded an unavailable scalar value");
          return mlir::failure();
        }
        line(destination.spelling + " = " + value.spelling + ";");
      }
      return mlir::success();
    };

    line("if (" + condition.spelling + ") {");
    ++indent;
    if (mlir::failed(emitBranch(op.getThenRegion())))
      return mlir::failure();
    --indent;
    line("} else {");
    ++indent;
    if (mlir::failed(emitBranch(op.getElseRegion())))
      return mlir::failure();
    --indent;
    line("}");

    for (auto [result, value] : llvm::zip(op.getResults(), results))
      values[result] = value;
    return mlir::success();
  }

  mlir::LogicalResult emitFor(ForOp op) {
    if (auto lowered = tryEmitAffineI4I8NTiles(op))
      return *lowered;
    if (auto lowered = tryEmitF16GemmNTiles(op))
      return *lowered;
    mlir::Block &body = op.getBody().front();
    if (op.getInitArgs().size() != op.getNumResults() ||
        body.getNumArguments() != op.getNumResults() + 1)
      return op.emitError("ordered range carried-value arity is inconsistent");
    llvm::SmallVector<CValue> carried;
    for (auto [result, initial] :
         llvm::zip(op.getResults(), op.getOperands().drop_front(3))) {
      CValue init = require(initial);
      if (init.spelling.empty())
        return op.emitError("ordered range has an unavailable carried value");
      CValue value;
      if (init.kind == CValueKind::Scalar) {
        value = CValue{result.getType(), CValueKind::Scalar, fresh("carry")};
        line(scalarCType(elementType(result.getType())) + " " + value.spelling + " = " +
             init.spelling + ";");
      } else if (inVLA && init.kind == CValueKind::F32Vector) {
        value = CValue{result.getType(), CValueKind::F32Vector, fresh("carry")};
        line("vfloat32m" + std::to_string(activeVLADecision->dataLMUL) +
             "_t " + value.spelling + " = " + init.spelling + ";");
      } else if (inVLA && init.kind == CValueKind::Mask) {
        value = CValue{result.getType(), CValueKind::Mask, fresh("carry")};
        line("vbool" + std::to_string(activeVLADecision->maskRatio) + "_t " +
             value.spelling + " = " + init.spelling + ";");
      } else if (init.kind == CValueKind::F32BlockStorage) {
        auto block = mlir::dyn_cast<BlockType>(result.getType());
        if (!block || block.getShape() != llvm::ArrayRef<int64_t>({16}) ||
            !block.getElementType().isF32())
          return op.emitError(
              "ordered range block carry must be f32 block<16>");
        value = init;
        value.type = result.getType();
      } else {
        return op.emitError(
            "ordered range carried value has no physical realization");
      }
      values[result] = value;
      carried.push_back(value);
    }
    CValue lower = require(op.getLower());
    CValue upper = require(op.getUpper());
    CValue step = require(op.getStep());
    if (lower.spelling.empty() || upper.spelling.empty() ||
        step.spelling.empty())
      return op.emitError("ordered range has unavailable bounds");
    auto emitCarriedUpdates = [&](YieldOp yield,
                                  llvm::StringRef unavailableMessage) {
      if (yield.getNumOperands() != carried.size()) {
        yield.emitError("ordered range carried-value arity is inconsistent");
        return mlir::failure();
      }
      llvm::SmallVector<std::pair<CValue, std::string>> updates;
      for (auto [destination, yielded] :
           llvm::zip(carried, yield.getOperands())) {
        CValue value = require(yielded);
        if (value.spelling.empty()) {
          yield.emitError() << unavailableMessage;
          return mlir::failure();
        }
        if (destination.kind == CValueKind::F32BlockStorage) {
          if (value.kind != CValueKind::F32BlockStorage ||
              value.spelling != destination.spelling) {
            yield.emitError(
                "ordered range block carry must update its selected storage");
            return mlir::failure();
          }
          continue;
        }
        if (value.kind != destination.kind ||
            (value.kind != CValueKind::Scalar &&
             value.kind != CValueKind::F32Vector &&
             value.kind != CValueKind::Mask)) {
          yield.emitError() << unavailableMessage;
          return mlir::failure();
        }
        std::string next = fresh("next");
        std::string type;
        if (destination.kind == CValueKind::Scalar)
          type = scalarCType(elementType(destination.type));
        else if (destination.kind == CValueKind::F32Vector)
          type = "vfloat32m" + std::to_string(activeVLADecision->dataLMUL) + "_t";
        else
          type = "vbool" + std::to_string(activeVLADecision->maskRatio) + "_t";
        line(type + " " + next + " = " + value.spelling + ";");
        updates.emplace_back(destination, std::move(next));
      }
      for (const auto &[destination, next] : updates)
        line(destination.spelling + " = " + next + ";");
      return mlir::success();
    };
    std::optional<int64_t> constantLower = integerConstant(op.getLower());
    std::optional<int64_t> constantUpper = integerConstant(op.getUpper());
    std::optional<int64_t> constantStep = integerConstant(op.getStep());
    if (constantLower && constantUpper && constantStep && *constantStep > 0 &&
        *constantUpper >= *constantLower &&
        (*constantUpper - *constantLower + *constantStep - 1) / *constantStep <=
            8) {
      llvm::DenseSet<mlir::Operation *> consumedBefore = consumed;
      for (int64_t induction = *constantLower; induction < *constantUpper;
           induction += *constantStep) {
        consumed = consumedBefore;
        values[body.getArgument(0)] =
            CValue{body.getArgument(0).getType(), CValueKind::Scalar,
                   std::to_string(induction)};
        for (auto [argument, value] :
             llvm::zip(body.getArguments().drop_front(), carried))
          values[argument] = value;
        for (mlir::Operation &nested : body.without_terminator())
          if (mlir::failed(emitOperation(&nested)))
            return mlir::failure();
        auto yield = mlir::cast<YieldOp>(body.getTerminator());
        if (mlir::failed(emitCarriedUpdates(
                yield, "unrolled ordered range yielded an unavailable value")))
          return mlir::failure();
      }
      consumed = consumedBefore;
      return mlir::success();
    }
    std::string induction = "__weft_i" + std::to_string(nextLoop++);
    line("for (size_t " + induction + " = " + lower.spelling + "; " +
         induction + " < " + upper.spelling + "; " + induction + " += " +
         step.spelling + ") {");
    ++indent;
    values[body.getArgument(0)] =
        CValue{body.getArgument(0).getType(), CValueKind::Scalar, induction};
    for (auto [argument, value] :
         llvm::zip(body.getArguments().drop_front(), carried))
      values[argument] = value;
    for (mlir::Operation &nested : body.without_terminator())
      if (mlir::failed(emitOperation(&nested)))
        return mlir::failure();
    auto yield = mlir::cast<YieldOp>(body.getTerminator());
    if (mlir::failed(emitCarriedUpdates(
            yield, "ordered range yielded an unavailable value")))
      return mlir::failure();
    --indent;
    line("}");
    return mlir::success();
  }

  mlir::LogicalResult emitWhile(WhileOp op) {
    mlir::Block &condition = op.getConditionRegion().front();
    mlir::Block &body = op.getBodyRegion().front();
    if (op.getInitArgs().size() != op.getNumResults() ||
        condition.getNumArguments() != op.getNumResults() ||
        body.getNumArguments() != op.getNumResults())
      return op.emitError("ordered while carried-value arity is inconsistent");

    llvm::SmallVector<CValue> carried;
    for (auto [result, initial] : llvm::zip(op.getResults(), op.getInitArgs())) {
      CValue init = require(initial);
      if (init.kind != CValueKind::Scalar || init.spelling.empty() ||
          !mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
              result.getType()))
        return op.emitError(
            "ordered while currently carries only available scalar values");
      CValue value{result.getType(), CValueKind::Scalar, fresh("while_carry")};
      line(scalarCType(elementType(result.getType())) + " " + value.spelling +
           " = " + init.spelling + ";");
      values[result] = value;
      carried.push_back(value);
    }

    auto stageScalars = [&](mlir::ValueRange source, llvm::StringRef prefix,
                            llvm::SmallVectorImpl<CValue> &staged) {
      if (source.size() != carried.size())
        return mlir::failure();
      for (auto [expected, value] : llvm::zip(carried, source)) {
        CValue emitted = require(value);
        if (emitted.kind != CValueKind::Scalar || emitted.spelling.empty() ||
            emitted.type != expected.type)
          return mlir::failure();
        CValue next{emitted.type, CValueKind::Scalar, fresh(prefix)};
        line(scalarCType(elementType(emitted.type)) + " " + next.spelling +
             " = " + emitted.spelling + ";");
        staged.push_back(next);
      }
      return mlir::success();
    };

    line("while (1) {");
    ++indent;
    for (auto [argument, value] : llvm::zip(condition.getArguments(), carried))
      values[argument] = value;
    for (mlir::Operation &nested : condition.without_terminator())
      if (mlir::failed(emitOperation(&nested)))
        return mlir::failure();
    auto conditionTerminator =
        mlir::cast<ConditionOp>(condition.getTerminator());
    CValue predicate = require(conditionTerminator.getCondition());
    if (predicate.kind != CValueKind::Scalar || predicate.spelling.empty())
      return conditionTerminator.emitError(
          "ordered while condition has no scalar realization");
    llvm::SmallVector<CValue> forwarded;
    if (mlir::failed(stageScalars(conditionTerminator.getValues(),
                                  "while_forward", forwarded)))
      return conditionTerminator.emitError(
          "ordered while condition forwarded an unavailable scalar value");

    line("if (!(" + predicate.spelling + ")) {");
    ++indent;
    for (auto [destination, value] : llvm::zip(carried, forwarded))
      line(destination.spelling + " = " + value.spelling + ";");
    line("break;");
    --indent;
    line("}");

    for (auto [argument, value] : llvm::zip(body.getArguments(), forwarded))
      values[argument] = value;
    for (mlir::Operation &nested : body.without_terminator())
      if (mlir::failed(emitOperation(&nested)))
        return mlir::failure();
    auto yield = mlir::cast<YieldOp>(body.getTerminator());
    llvm::SmallVector<CValue> updated;
    if (mlir::failed(stageScalars(yield.getValues(), "while_next", updated)))
      return yield.emitError(
          "ordered while body yielded an unavailable scalar value");
    for (auto [destination, value] : llvm::zip(carried, updated))
      line(destination.spelling + " = " + value.spelling + ";");
    --indent;
    line("}");
    return mlir::success();
  }

  mlir::LogicalResult tryEmitF32ToF16VLA(VLAOp op) {
    if (op.getNumResults() != 0)
      return mlir::failure();
    mlir::Block &body = op.getBody().front();
    LoadOp load;
    CastOp cast;
    StoreOp store;
    for (mlir::Operation &nested : body.without_terminator()) {
      if (auto candidate = mlir::dyn_cast<LoadOp>(nested)) {
        if (load)
          return mlir::failure();
        load = candidate;
      } else if (auto candidate = mlir::dyn_cast<CastOp>(nested)) {
        if (cast)
          return mlir::failure();
        cast = candidate;
      } else if (auto candidate = mlir::dyn_cast<StoreOp>(nested)) {
        if (store)
          return mlir::failure();
        store = candidate;
      } else if (!mlir::isa<PtrAddOp, ConstantOp, InvalidOp>(nested))
        return mlir::failure();
    }
    if (!load || !cast || !store || cast.getInput() != load.getResult() ||
        store.getValue() != cast.getResult() || !isTrue(load.getWhere()) ||
        !isTrue(store.getWhere()) ||
        !elementType(load.getResult().getType()).isF32() ||
        !isF16(elementType(cast.getResult().getType())))
      return mlir::failure();

    mlir::Value coordinate = body.getArgument(0);
    if (!valueDependsOn(load.getPointer(), coordinate) ||
        !valueDependsOn(store.getPointer(), coordinate))
      return mlir::failure();
    std::optional<std::string> source =
        pointerBase(load.getPointer(), coordinate);
    std::optional<std::string> destination =
        pointerBase(store.getPointer(), coordinate);
    CValue begin = require(op.getBegin());
    CValue end = require(op.getEnd());
    if (!source || !destination || begin.spelling.empty() || end.spelling.empty())
      return mlir::failure();

    std::string strip = "__weft_vla" + std::to_string(nextLoop++);
    std::string vl = fresh("vl");
    std::string wide = fresh("load_f32");
    std::string narrow = fresh("narrow_f16");
    line("for (size_t " + strip + " = " + begin.spelling + "; " + strip +
         " < " + end.spelling + ";) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e32m8(" + end.spelling +
         " - " + strip + ");");
    line("vfloat32m8_t " + wide + " = __riscv_vle32_v_f32m8(" + *source +
         " + " + strip + ", " + vl + ");");
    line("vfloat16m4_t " + narrow +
         " = __riscv_vfncvt_f_f_w_f16m4(" + wide + ", " + vl + ");");
    line("__riscv_vse16_v_f16m4(" + *destination + " + " + strip + ", " +
         narrow + ", " + vl + ");");
    line(strip + " += " + vl + ";");
    --indent;
    line("}");
    return mlir::success();
  }

  mlir::LogicalResult tryEmitF16FillVLA(VLAOp op) {
    if (op.getNumResults() != 0)
      return mlir::failure();
    mlir::Block &body = op.getBody().front();
    StoreOp store;
    for (mlir::Operation &nested : body.without_terminator()) {
      if (mlir::isa<LoadOp>(nested))
        return mlir::failure();
      if (auto candidate = mlir::dyn_cast<StoreOp>(nested)) {
        if (store)
          return mlir::failure();
        store = candidate;
      } else if (!mlir::isa<PtrAddOp, ConstantOp>(nested))
        return mlir::failure();
    }
    if (!store || !isTrue(store.getWhere()) ||
        !isF16(elementType(store.getValue().getType())))
      return mlir::failure();
    std::string fill = expression(store.getValue());
    mlir::Value coordinate = body.getArgument(0);
    if (!valueDependsOn(store.getPointer(), coordinate))
      return mlir::failure();
    std::optional<std::string> destination =
        pointerBase(store.getPointer(), coordinate);
    CValue begin = require(op.getBegin());
    CValue end = require(op.getEnd());
    if (fill.empty() || !destination || begin.spelling.empty() ||
        end.spelling.empty())
      return mlir::failure();

    std::string strip = "__weft_vla" + std::to_string(nextLoop++);
    std::string vl = fresh("vl");
    std::string vector = fresh("fill_f16");
    line("for (size_t " + strip + " = " + begin.spelling + "; " + strip +
         " < " + end.spelling + ";) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e16m8(" + end.spelling +
         " - " + strip + ");");
    line("vfloat16m8_t " + vector + " = __riscv_vfmv_v_f_f16m8(" + fill +
         ", " + vl + ");");
    line("__riscv_vse16_v_f16m8(" + *destination + " + " + strip + ", " +
         vector + ", " + vl + ");");
    line(strip + " += " + vl + ";");
    --indent;
    line("}");
    return mlir::success();
  }

  mlir::LogicalResult tryEmitWideningF16DotVLA(VLAOp op) {
    if (op.getNumResults() != 1 || !op.getResult(0).getType().isF32())
      return mlir::failure();
    mlir::Block &body = op.getBody().front();
    unsigned loadCount = 0;
    unsigned castCount = 0;
    unsigned binaryCount = 0;
    unsigned reduceCount = 0;
    for (mlir::Operation &nested : body.without_terminator()) {
      if (mlir::isa<PtrAddOp, ConstantOp, InvalidOp>(nested))
        continue;
      if (mlir::isa<LoadOp>(nested))
        ++loadCount;
      else if (mlir::isa<CastOp>(nested))
        ++castCount;
      else if (mlir::isa<BinaryOp>(nested))
        ++binaryCount;
      else if (mlir::isa<ReduceOp>(nested))
        ++reduceCount;
      else
        return mlir::failure();
    }
    if (loadCount != 2 || castCount != 2 || binaryCount != 1 ||
        reduceCount != 1)
      return mlir::failure();
    auto yield = mlir::cast<YieldOp>(body.getTerminator());
    if (yield.getNumOperands() != 1)
      return mlir::failure();
    auto reduce = yield.getOperand(0).getDefiningOp<ReduceOp>();
    if (!reduce || reduce.getKind() != "add" || reduce.getAxis() != -1 ||
        !isTrue(reduce.getWhere()))
      return mlir::failure();
    auto multiply = reduce.getInput().getDefiningOp<BinaryOp>();
    if (!multiply || multiply.getKind() != "mul")
      return mlir::failure();
    auto lhsCast = multiply.getLhs().getDefiningOp<CastOp>();
    auto rhsCast = multiply.getRhs().getDefiningOp<CastOp>();
    if (!lhsCast || !rhsCast ||
        !elementType(lhsCast.getResult().getType()).isF32() ||
        !elementType(rhsCast.getResult().getType()).isF32())
      return mlir::failure();
    auto lhsLoad = lhsCast.getInput().getDefiningOp<LoadOp>();
    auto rhsLoad = rhsCast.getInput().getDefiningOp<LoadOp>();
    if (!lhsLoad || !rhsLoad || !isTrue(lhsLoad.getWhere()) ||
        !isTrue(rhsLoad.getWhere()) ||
        !isF16(elementType(lhsLoad.getResult().getType())) ||
        !isF16(elementType(rhsLoad.getResult().getType())))
      return mlir::failure();

    mlir::Value coordinate = body.getArgument(0);
    if (!valueDependsOn(lhsLoad.getPointer(), coordinate) ||
        !valueDependsOn(rhsLoad.getPointer(), coordinate))
      return mlir::failure();
    std::optional<std::string> lhs =
        pointerBase(lhsLoad.getPointer(), coordinate);
    std::optional<std::string> rhs =
        pointerBase(rhsLoad.getPointer(), coordinate);
    CValue begin = require(op.getBegin());
    CValue end = require(op.getEnd());
    std::string identity = expression(reduce.getIdentity());
    if (!lhs || !rhs || begin.spelling.empty() || end.spelling.empty() ||
        identity.empty())
      return mlir::failure();

    std::string fullVL = fresh("dot_vl");
    std::string accumulator = fresh("dot_acc");
    std::string strip = "__weft_vla" + std::to_string(nextLoop++);
    std::string vl = fresh("vl");
    std::string lhsVector = fresh("dot_lhs");
    std::string rhsVector = fresh("dot_rhs");
    std::string seed = fresh("dot_seed");
    std::string reduced = fresh("dot_reduced");
    std::string result = fresh("dot");
    line("const size_t " + fullVL + " = __riscv_vsetvlmax_e16m1();");
    line("vfloat32m2_t " + accumulator +
         " = __riscv_vfmv_v_f_f32m2(0.0f, " + fullVL + ");");
    line("size_t " + strip + " = " + begin.spelling + ";");
    line("for (; " + strip + " + " + fullVL + " <= " + end.spelling + "; " +
         strip + " += " + fullVL + ") {");
    ++indent;
    line("vfloat16m1_t " + lhsVector + " = __riscv_vle16_v_f16m1(" + *lhs +
         " + " + strip + ", " + fullVL + ");");
    line("vfloat16m1_t " + rhsVector + " = __riscv_vle16_v_f16m1(" + *rhs +
         " + " + strip + ", " + fullVL + ");");
    line(accumulator + " = __riscv_vfwmacc_vv_f32m2(" + accumulator + ", " +
         lhsVector + ", " + rhsVector + ", " + fullVL + ");");
    --indent;
    line("}");
    line("if (" + strip + " < " + end.spelling + ") {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e16m1(" + end.spelling +
         " - " + strip + ");");
    line("vfloat16m1_t " + lhsVector + " = __riscv_vle16_v_f16m1(" + *lhs +
         " + " + strip + ", " + vl + ");");
    line("vfloat16m1_t " + rhsVector + " = __riscv_vle16_v_f16m1(" + *rhs +
         " + " + strip + ", " + vl + ");");
    line(accumulator + " = __riscv_vfwmacc_vv_f32m2_tu(" + accumulator +
         ", " + lhsVector + ", " + rhsVector + ", " + vl + ");");
    --indent;
    line("}");
    line("vfloat32m1_t " + seed + " = __riscv_vfmv_v_f_f32m1(" + identity +
         ", 1);");
    line("vfloat32m1_t " + reduced +
         " = __riscv_vfredusum_vs_f32m2_f32m1(" + accumulator + ", " + seed +
         ", " + fullVL + ");");
    line("const float " + result + " = __riscv_vfmv_f_s_f32m1_f32(" +
         reduced + ");");
    CValue materialized{op.getResult(0).getType(), CValueKind::Scalar, result};
    values[reduce.getResult()] = materialized;
    values[op.getResult(0)] = materialized;
    return mlir::success();
  }

  mlir::LogicalResult tryEmitF16WeightedUpdateVLA(VLAOp op) {
    if (op.getNumResults() != 0)
      return mlir::failure();
    mlir::Block &body = op.getBody().front();
    StoreOp store;
    unsigned loadCount = 0;
    unsigned binaryCount = 0;
    for (mlir::Operation &nested : body.without_terminator()) {
      if (auto candidate = mlir::dyn_cast<StoreOp>(nested)) {
        if (store)
          return mlir::failure();
        store = candidate;
      } else if (mlir::isa<LoadOp>(nested))
        ++loadCount;
      else if (mlir::isa<BinaryOp>(nested))
        ++binaryCount;
      else if (!mlir::isa<PtrAddOp, ConstantOp, InvalidOp>(nested))
        return mlir::failure();
    }
    if (loadCount != 2 || binaryCount != 2)
      return mlir::failure();
    if (!store || !isTrue(store.getWhere()) ||
        !isF16(elementType(store.getValue().getType())))
      return mlir::failure();
    auto add = store.getValue().getDefiningOp<BinaryOp>();
    if (!add || add.getKind() != "add")
      return mlir::failure();

    BinaryOp multiply = add.getLhs().getDefiningOp<BinaryOp>();
    mlir::Value baseValue = add.getRhs();
    if (!multiply || multiply.getKind() != "mul") {
      multiply = add.getRhs().getDefiningOp<BinaryOp>();
      baseValue = add.getLhs();
    }
    if (!multiply || multiply.getKind() != "mul")
      return mlir::failure();
    auto baseLoad = baseValue.getDefiningOp<LoadOp>();
    if (!baseLoad || !isTrue(baseLoad.getWhere()) ||
        !isF16(elementType(baseLoad.getResult().getType())))
      return mlir::failure();

    LoadOp productLoad = multiply.getLhs().getDefiningOp<LoadOp>();
    mlir::Value weight = multiply.getRhs();
    if (!productLoad) {
      productLoad = multiply.getRhs().getDefiningOp<LoadOp>();
      weight = multiply.getLhs();
    }
    if (!productLoad || !isTrue(productLoad.getWhere()) ||
        !isF16(elementType(productLoad.getResult().getType())) ||
        !isF16(elementType(weight.getType())))
      return mlir::failure();

    mlir::Value destinationRoot = pointerRoot(store.getPointer());
    mlir::Value baseRoot = pointerRoot(baseLoad.getPointer());
    mlir::Value productRoot = pointerRoot(productLoad.getPointer());
    if (!destinationRoot ||
        (destinationRoot != baseRoot && destinationRoot != productRoot))
      return mlir::failure();

    mlir::Value coordinate = body.getArgument(0);
    if (!valueDependsOn(baseLoad.getPointer(), coordinate) ||
        !valueDependsOn(productLoad.getPointer(), coordinate) ||
        !valueDependsOn(store.getPointer(), coordinate))
      return mlir::failure();
    std::optional<std::string> base =
        pointerBase(baseLoad.getPointer(), coordinate);
    std::optional<std::string> product =
        pointerBase(productLoad.getPointer(), coordinate);
    std::optional<std::string> destination =
        pointerBase(store.getPointer(), coordinate);
    std::string weightExpression = expression(weight);
    CValue begin = require(op.getBegin());
    CValue end = require(op.getEnd());
    if (!base || !product || !destination || weightExpression.empty() ||
        begin.spelling.empty() || end.spelling.empty())
      return mlir::failure();

    std::string strip = "__weft_vla" + std::to_string(nextLoop++);
    std::string vl = fresh("vl");
    std::string baseVector = fresh("update_base");
    std::string productVector = fresh("update_value");
    std::string result = fresh("update_f16");
    line("for (size_t " + strip + " = " + begin.spelling + "; " + strip +
         " < " + end.spelling + ";) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e16m8(" + end.spelling +
         " - " + strip + ");");
    line("vfloat16m8_t " + baseVector + " = __riscv_vle16_v_f16m8(" + *base +
         " + " + strip + ", " + vl + ");");
    line("vfloat16m8_t " + productVector +
         " = __riscv_vle16_v_f16m8(" + *product + " + " + strip + ", " + vl +
         ");");
    line("vfloat16m8_t " + result + " = __riscv_vfmacc_vf_f16m8(" +
         baseVector + ", " + weightExpression + ", " + productVector + ", " +
         vl + ");");
    line("__riscv_vse16_v_f16m8(" + *destination + " + " + strip + ", " +
         result + ", " + vl + ");");
    line(strip + " += " + vl + ";");
    --indent;
    line("}");
    return mlir::success();
  }

  mlir::LogicalResult tryEmitF16ToF32NormalizeVLA(VLAOp op) {
    if (op.getNumResults() != 0)
      return mlir::failure();
    mlir::Block &body = op.getBody().front();
    StoreOp store;
    unsigned loadCount = 0;
    unsigned castCount = 0;
    unsigned binaryCount = 0;
    for (mlir::Operation &nested : body.without_terminator()) {
      if (auto candidate = mlir::dyn_cast<StoreOp>(nested)) {
        if (store)
          return mlir::failure();
        store = candidate;
      } else if (mlir::isa<LoadOp>(nested))
        ++loadCount;
      else if (mlir::isa<CastOp>(nested))
        ++castCount;
      else if (mlir::isa<BinaryOp>(nested))
        ++binaryCount;
      else if (!mlir::isa<PtrAddOp, ConstantOp, InvalidOp>(nested))
        return mlir::failure();
    }
    if (loadCount != 1 || castCount != 1 || binaryCount != 1)
      return mlir::failure();
    if (!store || !isTrue(store.getWhere()) ||
        !elementType(store.getValue().getType()).isF32())
      return mlir::failure();
    auto divide = store.getValue().getDefiningOp<BinaryOp>();
    if (!divide || divide.getKind() != "div")
      return mlir::failure();
    auto cast = divide.getLhs().getDefiningOp<CastOp>();
    if (!cast || !elementType(cast.getResult().getType()).isF32())
      return mlir::failure();
    auto load = cast.getInput().getDefiningOp<LoadOp>();
    if (!load || !isTrue(load.getWhere()) ||
        !isF16(elementType(load.getResult().getType())))
      return mlir::failure();

    mlir::Value coordinate = body.getArgument(0);
    if (!valueDependsOn(load.getPointer(), coordinate) ||
        !valueDependsOn(store.getPointer(), coordinate))
      return mlir::failure();
    std::optional<std::string> source =
        pointerBase(load.getPointer(), coordinate);
    std::optional<std::string> destination =
        pointerBase(store.getPointer(), coordinate);
    std::string divisor = expression(divide.getRhs());
    CValue begin = require(op.getBegin());
    CValue end = require(op.getEnd());
    if (!source || !destination || divisor.empty() || begin.spelling.empty() ||
        end.spelling.empty())
      return mlir::failure();

    std::string strip = "__weft_vla" + std::to_string(nextLoop++);
    std::string vl = fresh("vl");
    std::string half = fresh("load_f16");
    std::string wide = fresh("widen_f32");
    std::string normalized = fresh("normalize_f32");
    line("for (size_t " + strip + " = " + begin.spelling + "; " + strip +
         " < " + end.spelling + ";) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e32m8(" + end.spelling +
         " - " + strip + ");");
    line("vfloat16m4_t " + half + " = __riscv_vle16_v_f16m4(" + *source +
         " + " + strip + ", " + vl + ");");
    line("vfloat32m8_t " + wide + " = __riscv_vfwcvt_f_f_v_f32m8(" + half +
         ", " + vl + ");");
    line("vfloat32m8_t " + normalized + " = __riscv_vfdiv_vf_f32m8(" + wide +
         ", " + divisor + ", " + vl + ");");
    line("__riscv_vse32_v_f32m8(" + *destination + " + " + strip + ", " +
         normalized + ", " + vl + ");");
    line(strip + " += " + vl + ";");
    --indent;
    line("}");
    return mlir::success();
  }

  mlir::LogicalResult emitVLA(VLAOp op) {
    if (inVLA)
      return op.emitError("nested VLA regions are not supported");
    if (!options.target.hasRVV)
      return op.emitError(
          "VLA requires RVV on the selected target; no scalar fallback exists");
    if (mlir::succeeded(tryEmitF32ToF16VLA(op)))
      return mlir::success();
    if (mlir::succeeded(tryEmitF16FillVLA(op)))
      return mlir::success();
    if (mlir::succeeded(tryEmitWideningF16DotVLA(op)))
      return mlir::success();
    if (mlir::succeeded(tryEmitF16WeightedUpdateVLA(op)))
      return mlir::success();
    if (mlir::succeeded(tryEmitF16ToF32NormalizeVLA(op)))
      return mlir::success();
    if (mlir::succeeded(tryEmitSoftmaxEnvelope(op)))
      return mlir::success();
    mlir::FailureOr<VLARegionDecision> selected = decideVLARegion(op);
    if (mlir::failed(selected))
      return mlir::failure();
    VLARegionDecision decision = std::move(*selected);
    mlir::Block &body = op.getBody().front();
    llvm::DenseMap<mlir::Operation *, CValue> aggregates;
    for (const VLAStateDecision &state : decision.states) {
      if (state.realization == VLAStateRealization::RVVArgMaxSummary) {
        auto summary = mlir::cast<SummaryFoldOp>(state.operation);
        auto identity = summary.getIdentity().getDefiningOp<TupleOp>();
        CValue maximum{mlir::Float32Type::get(kernel.getContext()),
                       CValueKind::Scalar, fresh("argmax_value")};
        CValue index{mlir::IndexType::get(kernel.getContext()),
                     CValueKind::Scalar, fresh("argmax_index")};
        line("float " + maximum.spelling + " = " +
             expression(identity.getOperand(0)) + ";");
        line("size_t " + index.spelling + " = " +
             expression(identity.getOperand(1)) + ";");
        CValue aggregate{summary.getResult().getType(), CValueKind::Tuple, {}};
        aggregate.fields = {maximum, index};
        aggregates[state.operation] = aggregate;
        values[summary.getResult()] = aggregate;
        continue;
      }
      std::string identity = expression(state.identity);
      if (identity.empty())
        return state.operation->emitError("VLA state identity is unavailable");
      llvm::StringRef prefix =
          state.realization == VLAStateRealization::RVVInclusiveAddScan
              ? "scan_carry"
              : "reduce";
      CValue aggregate{state.elementType, CValueKind::Scalar, fresh(prefix)};
      std::string type = scalarCType(state.elementType);
      if (type.empty())
        return state.operation->emitError("VLA state type is unavailable");
      line(type + " " + aggregate.spelling + " = " + identity + ";");
      aggregates[state.operation] = aggregate;
      if (auto reduce = mlir::dyn_cast<ReduceOp>(state.operation))
        values[reduce.getResult()] = aggregate;
    }
    for (mlir::Operation &nested : body.without_terminator()) {
      if (auto summary = mlir::dyn_cast<SummaryFoldOp>(nested)) {
        if (aggregates.contains(summary.getOperation()))
          continue;
        if (!isOnlineSoftmaxSummary(summary))
          return summary.emitError(
              "RISC-V target does not implement this summary algebra");
        auto identity = summary.getIdentity().getDefiningOp<TupleOp>();
        CValue maximum{mlir::Float32Type::get(kernel.getContext()),
                       CValueKind::Scalar, fresh("summary_max")};
        CValue sum{mlir::Float32Type::get(kernel.getContext()),
                   CValueKind::Scalar, fresh("summary_sum")};
        line("float " + maximum.spelling + " = " +
             expression(identity.getOperand(0)) + ";");
        line("float " + sum.spelling + " = " +
             expression(identity.getOperand(1)) + ";");
        CValue aggregate{summary.getResult().getType(), CValueKind::Tuple, {}};
        aggregate.fields = {maximum, sum};
        aggregates[summary.getOperation()] = aggregate;
        values[summary.getResult()] = aggregate;
        continue;
      }
    }

    CValue begin = require(op.getBegin());
    CValue end = require(op.getEnd());
    if (begin.spelling.empty() || end.spelling.empty())
      return op.emitError("VLA bounds are unavailable");
    std::string strip = "__weft_vla" + std::to_string(nextLoop++);
    std::string vl = fresh("vl");
    line("for (size_t " + strip + " = " + begin.spelling + "; " + strip +
         " < " + end.spelling + ";) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e" +
         std::to_string(decision.dataSEW) + "m" +
         std::to_string(decision.dataLMUL) + "(" + end.spelling + " - " +
         strip + ");");

    bool previousInVLA = inVLA;
    std::string previousVL = activeVL;
    const VLARegionDecision *previousDecision = activeVLADecision;
    inVLA = true;
    activeVL = vl;
    activeVLADecision = &decision;
    CValue coordinate{body.getArgument(0).getType(), CValueKind::Coordinate,
                      strip};
    coordinate.laneStride = "1";
    values[body.getArgument(0)] = std::move(coordinate);
    for (mlir::Operation &nested : body.without_terminator()) {
      if (auto reduce = mlir::dyn_cast<ReduceOp>(nested)) {
        const VLAStateDecision *state =
            findStateDecision(reduce.getOperation());
        if (!state || mlir::failed(emitVectorReduce(
                          reduce, *state, aggregates[reduce.getOperation()])))
          return mlir::failure();
        continue;
      }
      if (auto scan = mlir::dyn_cast<ScanOp>(nested)) {
        const VLAStateDecision *state = findStateDecision(scan.getOperation());
        if (!state || mlir::failed(emitVectorScan(
                          scan, *state, aggregates[scan.getOperation()])))
          return mlir::failure();
        continue;
      }
      if (auto summary = mlir::dyn_cast<SummaryFoldOp>(nested)) {
        const VLAStateDecision *state =
            findStateDecision(summary.getOperation());
        mlir::LogicalResult lowered =
            state && state->realization == VLAStateRealization::RVVArgMaxSummary
                ? emitVectorArgMaxSummary(
                      summary, *state, aggregates[summary.getOperation()])
                : emitOnlineSoftmaxSummary(
                      summary, aggregates[summary.getOperation()]);
        if (mlir::failed(lowered))
          return mlir::failure();
        continue;
      }
      if (mlir::failed(emitOperation(&nested)))
        return mlir::failure();
    }
    line(strip + " += " + vl + ";");
    inVLA = previousInVLA;
    activeVL = previousVL;
    activeVLADecision = previousDecision;
    --indent;
    line("}");

    auto yield = mlir::cast<YieldOp>(body.getTerminator());
    for (auto [result, yielded] :
         llvm::zip(op.getResults(), yield.getOperands())) {
      CValue value = require(yielded);
      if ((value.spelling.empty() && value.kind != CValueKind::Tuple) ||
          (value.kind != CValueKind::Scalar &&
           value.kind != CValueKind::Tuple))
        return op.emitError("VLA result is not a materialized scalar aggregate");
      value.type = result.getType();
      values[result] = value;
    }
    return mlir::success();
  }

  mlir::LogicalResult emitPtrAdd(PtrAddOp op) {
    CValue base = require(op.getBase());
    CValue offset = require(op.getOffset());
    if (base.spelling.empty() || offset.spelling.empty())
      return op.emitError("pointer addition has an unavailable operand");
    if (base.kind != CValueKind::Pointer)
      return op.emitError("pointer addition base is not a pointer");
    CValue result{op.getResult().getType(), CValueKind::Pointer,
                  "(" + base.spelling + " + " + offset.spelling + ")"};
    if (offset.kind == CValueKind::Coordinate) {
      result.lanePointer = true;
      result.laneStride = offset.laneStride;
    } else if (inVLA && base.lanePointer) {
      result.lanePointer = true;
      result.laneStride = base.laneStride;
    }
    values[op.getResult()] = std::move(result);
    return mlir::success();
  }

  std::optional<std::string> pointerBase(mlir::Value value,
                                         mlir::Value coordinate) {
    if (value == coordinate)
      return std::nullopt;
    if (auto pointer = value.getDefiningOp<PtrAddOp>()) {
      if (pointer.getOffset() == coordinate)
        return pointerBase(pointer.getBase(), coordinate);
      std::optional<std::string> base =
          pointerBase(pointer.getBase(), coordinate);
      std::string offset = expression(pointer.getOffset());
      if (!base || offset.empty())
        return std::nullopt;
      return "(" + *base + " + " + offset + ")";
    }
    CValue materialized = require(value);
    if (materialized.kind != CValueKind::Pointer ||
        materialized.spelling.empty())
      return std::nullopt;
    return materialized.spelling;
  }

  bool valueDependsOn(mlir::Value value, mlir::Value target,
                      llvm::DenseSet<mlir::Value> &visited) const {
    if (value == target)
      return true;
    if (!value || !visited.insert(value).second)
      return false;
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition)
      return false;
    return llvm::any_of(definition->getOperands(), [&](mlir::Value operand) {
      return valueDependsOn(operand, target, visited);
    });
  }

  bool valueDependsOn(mlir::Value value, mlir::Value target) const {
    llvm::DenseSet<mlir::Value> visited;
    return valueDependsOn(value, target, visited);
  }

  bool sameBound(mlir::Value lhs, mlir::Value rhs) {
    if (lhs == rhs)
      return true;
    std::string lhsExpression = expression(lhs);
    std::string rhsExpression = expression(rhs);
    return !lhsExpression.empty() && lhsExpression == rhsExpression;
  }

  mlir::LogicalResult tryEmitSoftmaxEnvelope(VLAOp producer) {
    SummaryFoldOp summary;
    for (mlir::Operation &operation :
         producer.getBody().front().without_terminator()) {
      if (auto candidate = mlir::dyn_cast<SummaryFoldOp>(operation)) {
        if (summary)
          return mlir::failure();
        summary = candidate;
      }
    }
    if (!summary || !isOnlineSoftmaxSummary(summary) ||
        producer.getNumResults() != 1)
      return mlir::failure();

    TupleGetOp maximumGet;
    TupleGetOp sumGet;
    for (mlir::Operation *user : producer.getResult(0).getUsers()) {
      auto get = mlir::dyn_cast<TupleGetOp>(user);
      if (!get)
        return mlir::failure();
      if (get.getIndex() == 0 && !maximumGet)
        maximumGet = get;
      else if (get.getIndex() == 1 && !sumGet)
        sumGet = get;
      else
        return mlir::failure();
    }
    if (!maximumGet || !sumGet || maximumGet.getResult().use_empty() ||
        sumGet.getResult().use_empty())
      return mlir::failure();

    VLAOp consumer;
    for (mlir::Operation *user : maximumGet.getResult().getUsers()) {
      auto parent = user->getParentOfType<VLAOp>();
      if (!parent || (consumer && consumer != parent))
        return mlir::failure();
      consumer = parent;
    }
    for (mlir::Operation *user : sumGet.getResult().getUsers())
      if (user->getParentOfType<VLAOp>() != consumer)
        return mlir::failure();
    if (!consumer || consumer->getBlock() != producer->getBlock() ||
        !sameBound(producer.getBegin(), consumer.getBegin()) ||
        !sameBound(producer.getEnd(), consumer.getEnd()))
      return mlir::failure();

    LoadOp consumerLoad;
    StoreOp consumerStore;
    for (mlir::Operation &operation :
         consumer.getBody().front().without_terminator()) {
      if (auto load = mlir::dyn_cast<LoadOp>(operation)) {
        if (consumerLoad)
          return mlir::failure();
        consumerLoad = load;
      }
      if (auto store = mlir::dyn_cast<StoreOp>(operation)) {
        if (consumerStore)
          return mlir::failure();
        consumerStore = store;
      }
    }
    if (!consumerLoad || !consumerStore || !isTrue(consumerLoad.getWhere()) ||
        !isTrue(consumerStore.getWhere()))
      return mlir::failure();
    auto divide = consumerStore.getValue().getDefiningOp<BinaryOp>();
    if (!divide || divide.getKind() != "div" ||
        divide.getRhs() != sumGet.getResult())
      return mlir::failure();
    auto exponential = divide.getLhs().getDefiningOp<UnaryOp>();
    auto subtract = exponential
                        ? exponential.getInput().getDefiningOp<BinaryOp>()
                        : BinaryOp{};
    if (!exponential || exponential.getKind() != "exp" || !subtract ||
        subtract.getKind() != "sub" ||
        subtract.getLhs() != consumerLoad.getResult() ||
        subtract.getRhs() != maximumGet.getResult())
      return mlir::failure();
    auto producerLoad = summary.getInput().getDefiningOp<LoadOp>();
    if (!producerLoad || !isTrue(producerLoad.getWhere()))
      return mlir::failure();

    mlir::Value producerCoordinate =
        producer.getBody().front().getArgument(0);
    mlir::Value consumerCoordinate =
        consumer.getBody().front().getArgument(0);
    std::optional<std::string> input =
        pointerBase(producerLoad.getPointer(), producerCoordinate);
    std::optional<std::string> consumerInput =
        pointerBase(consumerLoad.getPointer(), consumerCoordinate);
    std::optional<std::string> outputPointer =
        pointerBase(consumerStore.getPointer(), consumerCoordinate);
    if (!input || !consumerInput || !outputPointer || *input != *consumerInput)
      return mlir::failure();

    std::string begin = expression(producer.getBegin());
    std::string end = expression(producer.getEnd());
    if (begin.empty() || end.empty())
      return mlir::failure();
    std::string maximum = fresh("softmax_max");
    std::string strip = fresh("softmax_i");
    std::string vl = fresh("vl");
    std::string inputVector = fresh("softmax_x");
    std::string seed = fresh("softmax_seed");
    std::string partial = fresh("softmax_partial");
    line("float " + maximum + " = -INFINITY;");
    line("for (size_t " + strip + " = " + begin + "; " + strip + " < " +
         end + ";) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e32m2(" + end + " - " +
         strip + ");");
    line("vfloat32m2_t " + inputVector + " = __riscv_vle32_v_f32m2(" +
         *input + " + " + strip + ", " + vl + ");");
    line("vfloat32m1_t " + seed + " = __riscv_vfmv_v_f_f32m1(" + maximum +
         ", 1);");
    line("vfloat32m1_t " + partial +
         " = __riscv_vfredmax_vs_f32m2_f32m1(" + inputVector + ", " + seed +
         ", " + vl + ");");
    line(maximum + " = __riscv_vfmv_f_s_f32m1_f32(" + partial + ");");
    line(strip + " += " + vl + ";");
    --indent;
    line("}");

    std::string sum = fresh("softmax_sum");
    std::string shifted = fresh("softmax_shifted");
    std::string exponentials = fresh("softmax_exp");
    std::string sumSeed = fresh("softmax_sum_seed");
    std::string sumPartial = fresh("softmax_sum_partial");
    line("float " + sum + " = 0.0f;");
    line("for (size_t " + strip + " = " + begin + "; " + strip + " < " +
         end + ";) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e32m2(" + end + " - " +
         strip + ");");
    line("vfloat32m2_t " + inputVector + " = __riscv_vle32_v_f32m2(" +
         *input + " + " + strip + ", " + vl + ");");
    line("vfloat32m2_t " + shifted + " = __riscv_vfsub_vf_f32m2(" +
         inputVector + ", " + maximum + ", " + vl + ");");
    line("vfloat32m2_t " + exponentials + " = __weft_exp_f32m2(" + shifted +
         ", " + vl + ");");
    line("__riscv_vse32_v_f32m2(" + *outputPointer + " + " + strip + ", " +
         exponentials + ", " + vl + ");");
    line("vfloat32m1_t " + sumSeed + " = __riscv_vfmv_v_f_f32m1(" + sum +
         ", 1);");
    line("vfloat32m1_t " + sumPartial +
         " = __riscv_vfredusum_vs_f32m2_f32m1(" + exponentials + ", " +
         sumSeed + ", " + vl + ");");
    line(sum + " = __riscv_vfmv_f_s_f32m1_f32(" + sumPartial + ");");
    line(strip + " += " + vl + ";");
    --indent;
    line("}");

    std::string inverse = fresh("softmax_inverse");
    std::string outputVector = fresh("softmax_y");
    line("const float " + inverse + " = 1.0f / " + sum + ";");
    line("for (size_t " + strip + " = " + begin + "; " + strip + " < " +
         end + ";) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e32m2(" + end + " - " +
         strip + ");");
    line("vfloat32m2_t " + outputVector + " = __riscv_vle32_v_f32m2(" +
         *outputPointer + " + " + strip + ", " + vl + ");");
    line(outputVector + " = __riscv_vfmul_vf_f32m2(" + outputVector + ", " +
         inverse + ", " + vl + ");");
    line("__riscv_vse32_v_f32m2(" + *outputPointer + " + " + strip + ", " +
         outputVector + ", " + vl + ");");
    line(strip + " += " + vl + ";");
    --indent;
    line("}");

    CValue maxValue{mlir::Float32Type::get(kernel.getContext()),
                    CValueKind::Scalar, maximum};
    CValue sumValue{mlir::Float32Type::get(kernel.getContext()),
                    CValueKind::Scalar, sum};
    CValue tuple{producer.getResult(0).getType(), CValueKind::Tuple, {}};
    tuple.fields = {maxValue, sumValue};
    values[summary.getResult()] = tuple;
    values[producer.getResult(0)] = tuple;
    consumed.insert(maximumGet.getOperation());
    consumed.insert(sumGet.getOperation());
    consumed.insert(consumer.getOperation());
    return mlir::success();
  }

  std::string scalarBinary(llvm::StringRef kind, llvm::StringRef lhs,
                           llvm::StringRef rhs) {
    llvm::StringRef spelling = llvm::StringSwitch<llvm::StringRef>(kind)
                                   .Case("add", "+")
                                   .Case("sub", "-")
                                   .Case("mul", "*")
                                   .Case("div", "/")
                                   .Case("mod", "%")
                                   .Case("and", "&")
                                   .Case("or", "|")
                                   .Case("xor", "^")
                                   .Case("shl", "<<")
                                   .Case("shr", ">>")
                                   .Default("");
    if (!spelling.empty())
      return "(" + lhs.str() + " " + spelling.str() + " " + rhs.str() + ")";
    if (kind == "max")
      return "fmaxf(" + lhs.str() + ", " + rhs.str() + ")";
    if (kind == "min")
      return "fminf(" + lhs.str() + ", " + rhs.str() + ")";
    return {};
  }

  mlir::LogicalResult emitBinary(BinaryOp op) {
    CValue lhs = require(op.getLhs());
    CValue rhs = require(op.getRhs());
    bool lhsCoordinate = lhs.kind == CValueKind::Coordinate;
    bool rhsCoordinate = rhs.kind == CValueKind::Coordinate;
    if (lhsCoordinate || rhsCoordinate) {
      if (lhsCoordinate && rhsCoordinate && op.getKind() == "mul")
        return op.emitError(
            "VLA coordinate multiplication is not an affine lane address");
      if (op.getKind() != "add" && op.getKind() != "sub" &&
          op.getKind() != "mul")
        return op.emitError(
            "VLA coordinate arithmetic must preserve an affine lane address");

      std::string firstLane =
          scalarBinary(op.getKind(), lhs.spelling, rhs.spelling);
      std::string laneStride;
      if (op.getKind() == "add") {
        laneStride = lhsCoordinate && rhsCoordinate
                         ? "(" + lhs.laneStride + " + " + rhs.laneStride + ")"
                         : (lhsCoordinate ? lhs.laneStride : rhs.laneStride);
      } else if (op.getKind() == "sub") {
        laneStride = lhsCoordinate && rhsCoordinate
                         ? "(" + lhs.laneStride + " - " + rhs.laneStride + ")"
                         : (lhsCoordinate ? lhs.laneStride
                                          : "(-" + rhs.laneStride + ")");
      } else if (lhsCoordinate)
        laneStride = "(" + lhs.laneStride + " * " + rhs.spelling + ")";
      else
        laneStride = "(" + lhs.spelling + " * " + rhs.laneStride + ")";

      CValue result{op.getResult().getType(), CValueKind::Coordinate,
                    firstLane};
      result.laneStride = std::move(laneStride);
      values[op.getResult()] = std::move(result);
      return mlir::success();
    }
    if (lhs.kind == CValueKind::Mask && rhs.kind == CValueKind::Mask) {
      std::string intrinsic =
          llvm::StringSwitch<std::string>(op.getKind())
              .Case("and", "__riscv_vmand_mm_b")
              .Case("or", "__riscv_vmor_mm_b")
              .Case("xor", "__riscv_vmxor_mm_b")
              .Default("");
      if (intrinsic.empty())
        return op.emitError("RVV mask lowering does not implement binary kind");
      std::string name = fresh("mask");
      std::string ratio = std::to_string(activeVLADecision->maskRatio);
      line("vbool" + ratio + "_t " + name + " = " + intrinsic + ratio + "(" +
           lhs.spelling + ", " + rhs.spelling + ", " + activeVL + ");");
      values[op.getResult()] =
          CValue{op.getResult().getType(), CValueKind::Mask, name};
      return mlir::success();
    }
    bool lhsVector = lhs.kind == CValueKind::F32Vector;
    bool rhsVector = rhs.kind == CValueKind::F32Vector;
    if (!lhsVector && !rhsVector) {
      std::string expression =
          scalarBinary(op.getKind(), lhs.spelling, rhs.spelling);
      if (expression.empty())
        return op.emitError("RISC-V scalar lowering does not implement binary kind");
      values[op.getResult()] = scalarExpression(
          op.getResult(), std::move(expression), "scalar");
      return mlir::success();
    }
    if (!inVLA || (!elementType(op.getResult().getType()).isF32()))
      return op.emitError(
          "RVV pointwise lowering currently requires VLA f32 values");
    std::string intrinsic;
    std::string first = lhs.spelling;
    std::string second = rhs.spelling;
    if (lhsVector && rhsVector) {
      intrinsic = llvm::StringSwitch<std::string>(op.getKind())
                      .Case("add", "__riscv_vfadd_vv_")
                      .Case("sub", "__riscv_vfsub_vv_")
                      .Case("mul", "__riscv_vfmul_vv_")
                      .Case("div", "__riscv_vfdiv_vv_")
                      .Case("max", "__riscv_vfmax_vv_")
                      .Case("min", "__riscv_vfmin_vv_")
                      .Default("");
    } else if (lhsVector) {
      intrinsic = llvm::StringSwitch<std::string>(op.getKind())
                      .Case("add", "__riscv_vfadd_vf_")
                      .Case("sub", "__riscv_vfsub_vf_")
                      .Case("mul", "__riscv_vfmul_vf_")
                      .Case("div", "__riscv_vfdiv_vf_")
                      .Case("max", "__riscv_vfmax_vf_")
                      .Case("min", "__riscv_vfmin_vf_")
                      .Default("");
    } else {
      intrinsic = llvm::StringSwitch<std::string>(op.getKind())
                      .Case("add", "__riscv_vfadd_vf_")
                      .Case("sub", "__riscv_vfrsub_vf_")
                      .Case("mul", "__riscv_vfmul_vf_")
                      .Case("div", "__riscv_vfrdiv_vf_")
                      .Case("max", "__riscv_vfmax_vf_")
                      .Case("min", "__riscv_vfmin_vf_")
                      .Default("");
      std::swap(first, second);
    }
    if (intrinsic.empty())
      return op.emitError("RVV pointwise lowering does not implement binary kind");
    std::string suffix =
        "f32m" + std::to_string(activeVLADecision->dataLMUL);
    intrinsic += suffix;
    std::string name = fresh("v");
    line("vfloat32m" + std::to_string(activeVLADecision->dataLMUL) + "_t " +
         name + " = " + intrinsic + "(" + first + ", " + second + ", " +
         activeVL + ");");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::F32Vector, name};
    return mlir::success();
  }

  mlir::LogicalResult emitUnary(UnaryOp op) {
    CValue input = require(op.getInput());
    if (input.kind == CValueKind::F32Vector) {
      std::string name = fresh("v");
      unsigned lmul = activeVLADecision->dataLMUL;
      std::string suffix = "f32m" + std::to_string(lmul);
      std::string vectorType = "vfloat32m" + std::to_string(lmul) + "_t";
      if (op.getKind() == "neg")
        line(vectorType + " " + name + " = __riscv_vfneg_v_" + suffix + "(" +
             input.spelling + ", " + activeVL + ");");
      else if (op.getKind() == "abs")
        line(vectorType + " " + name + " = __riscv_vfabs_v_" + suffix + "(" +
             input.spelling + ", " + activeVL + ");");
      else if (op.getKind() == "exp" && lmul == 2)
        line("vfloat32m2_t " + name + " = __weft_exp_f32m2(" +
             input.spelling + ", " + activeVL + ");");
      else
        return op.emitError(
            "RVV pointwise lowering does not implement unary kind");
      values[op.getResult()] =
          CValue{op.getResult().getType(), CValueKind::F32Vector, name};
      return mlir::success();
    }
    std::string expression;
    if (op.getKind() == "neg")
      expression = "(-" + input.spelling + ")";
    else if (op.getKind() == "exp")
      expression = "expf(" + input.spelling + ")";
    else if (op.getKind() == "log")
      expression = "logf(" + input.spelling + ")";
    else if (op.getKind() == "sqrt")
      expression = "sqrtf(" + input.spelling + ")";
    else if (op.getKind() == "rsqrt")
      expression = "(1.0f / sqrtf(" + input.spelling + "))";
    else if (op.getKind() == "abs")
      expression = "fabsf(" + input.spelling + ")";
    else if (op.getKind() == "sin")
      expression = "sinf(" + input.spelling + ")";
    else if (op.getKind() == "cos")
      expression = "cosf(" + input.spelling + ")";
    else if (op.getKind() == "floor")
      expression = "floorf(" + input.spelling + ")";
    else
      return op.emitError(
          "RISC-V scalar lowering does not implement unary kind");
    values[op.getResult()] = scalarExpression(
        op.getResult(), std::move(expression), "unary", op.getKind() == "exp");
    return mlir::success();
  }

  mlir::LogicalResult emitCompare(CompareOp op) {
    CValue lhs = require(op.getLhs());
    CValue rhs = require(op.getRhs());
    if (const VLAPredicateDecision *decision =
            findPredicateDecision(op.getOperation())) {
      CValue coordinate = require(decision->coordinate);
      CValue scalar = require(decision->scalar);
      if (coordinate.kind != CValueKind::Coordinate ||
          scalar.kind != CValueKind::Scalar || coordinate.spelling.empty() ||
          scalar.spelling.empty() || coordinate.laneStride.empty())
        return op.emitError("VLA predicate projection is unavailable");

      std::string vectorSuffix = "u" +
                                 std::to_string(activeVLADecision->indexSEW) +
                                 "m" +
                                 std::to_string(activeVLADecision->indexLMUL);
      std::string maskSuffix =
          vectorSuffix + "_b" +
          std::to_string(activeVLADecision->maskRatio);
      std::string vectorType = "vuint" +
                               std::to_string(activeVLADecision->indexSEW) +
                               "m" +
                               std::to_string(activeVLADecision->indexLMUL) +
                               "_t";
      std::string scalarType =
          "uint" + std::to_string(activeVLADecision->indexSEW) + "_t";
      std::string lane = fresh("lane_index");
      line(vectorType + " " + lane + " = __riscv_vid_v_" + vectorSuffix +
           "(" + activeVL + ");");
      if (decision->coordinateMode == VLAMemoryMode::Strided)
        line(lane + " = __riscv_vmul_vx_" + vectorSuffix + "(" + lane +
             ", (" + scalarType + ")(" + coordinate.laneStride + "), " +
             activeVL + ");");
      line(lane + " = __riscv_vadd_vx_" + vectorSuffix + "(" + lane +
           ", (" + scalarType + ")(" + coordinate.spelling + "), " + activeVL +
           ");");
      std::string intrinsic =
          llvm::StringSwitch<std::string>(decision->predicate)
              .Case("eq", "__riscv_vmseq_vx_")
              .Case("ne", "__riscv_vmsne_vx_")
              .Case("lt", "__riscv_vmsltu_vx_")
              .Case("le", "__riscv_vmsleu_vx_")
              .Case("gt", "__riscv_vmsgtu_vx_")
              .Case("ge", "__riscv_vmsgeu_vx_")
              .Default("");
      if (intrinsic.empty())
        return op.emitError("VLA predicate realization is unavailable");
      std::string mask = fresh("mask");
      line("vbool" + std::to_string(activeVLADecision->maskRatio) + "_t " +
           mask + " = " + intrinsic + maskSuffix + "(" + lane + ", (" +
           scalarType + ")(" + scalar.spelling + "), " + activeVL + ");");
      values[op.getResult()] =
          CValue{op.getResult().getType(), CValueKind::Mask, mask};
      return mlir::success();
    }
    if (lhs.kind != CValueKind::Scalar || rhs.kind != CValueKind::Scalar)
      return op.emitError("VLA predicate has no physical decision");
    llvm::StringRef spelling =
        llvm::StringSwitch<llvm::StringRef>(op.getPredicate())
            .Case("eq", "==")
            .Case("ne", "!=")
            .Case("lt", "<")
            .Case("le", "<=")
            .Case("gt", ">")
            .Case("ge", ">=")
            .Default("");
    if (spelling.empty())
      return op.emitError("comparison predicate is unsupported");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar,
               "(" + lhs.spelling + " " + spelling.str() + " " +
                   rhs.spelling + ")"};
    return mlir::success();
  }

  mlir::LogicalResult emitCast(CastOp op) {
    CValue input = require(op.getInput());
    if (input.kind == CValueKind::F16Vector &&
        elementType(op.getResult().getType()).isF32()) {
      std::string name = fresh("widen_f16");
      unsigned lmul = activeVLADecision->dataLMUL;
      line("vfloat32m" + std::to_string(lmul) + "_t " + name +
           " = __riscv_vfwcvt_f_f_v_f32m" + std::to_string(lmul) + "(" +
           input.spelling + ", " + activeVL + ");");
      values[op.getResult()] =
          CValue{op.getResult().getType(), CValueKind::F32Vector, name};
      return mlir::success();
    }
    if (input.kind != CValueKind::Scalar)
      return op.emitError("RVV cast lowering is not implemented yet");
    std::string target = scalarCType(elementType(op.getResult().getType()));
    if (target.empty())
      return op.emitError("cast target type is unsupported");
    values[op.getResult()] = scalarExpression(
        op.getResult(), "((" + target + ")(" + input.spelling + "))",
        "cast");
    return mlir::success();
  }

  mlir::LogicalResult emitNarrow(NarrowOp op) {
    const VLANarrowDecision *decision =
        findNarrowDecision(op.getOperation());
    CValue input = require(op.getInput());
    if (!decision || input.kind != CValueKind::F32Vector ||
        decision->sourceLMUL != activeVLADecision->dataLMUL)
      return op.emitError("VLA narrow projection is unavailable");
    std::string intermediate = fresh("narrow_i16");
    std::string result = fresh("narrow_i8");
    line("vint16m" + std::to_string(decision->intermediateLMUL) + "_t " +
         intermediate + " = __riscv_vfncvt_x_f_w_i16m" +
         std::to_string(decision->intermediateLMUL) + "(" + input.spelling +
         ", " + activeVL + ");");
    line("vint8m" + std::to_string(decision->resultLMUL) + "_t " + result +
         " = __riscv_vnclip_wx_i8m" +
         std::to_string(decision->resultLMUL) + "(" + intermediate +
         ", 0, __RISCV_VXRM_RNE, " + activeVL + ");");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::I8Vector, result};
    return mlir::success();
  }

  mlir::LogicalResult emitBitcast(BitcastOp op) {
    CValue input = require(op.getInput());
    if (input.kind != CValueKind::Scalar || input.spelling.empty())
      return op.emitError("RVV bitcast lowering requires a scalar input");
    mlir::Type source = elementType(op.getInput().getType());
    mlir::Type target = elementType(op.getResult().getType());
    std::string helper;
    if (source.isUnsignedInteger(8) && target.isSignedInteger(8))
      helper = "__weft_bitcast_u8_i8";
    else if (source.isUnsignedInteger(16) && target.isSignedInteger(16))
      helper = "__weft_bitcast_u16_i16";
    else if (source.isUnsignedInteger(16) && isF16(target))
      helper = "__weft_bitcast_u16_f16";
    else if (isF16(source) && target.isUnsignedInteger(16))
      helper = "__weft_bitcast_f16_u16";
    else if (source.isUnsignedInteger(32) && target.isF32())
      helper = "__weft_bitcast_u32_f32";
    else
      return op.emitError("RISC-V scalar bitcast pair is unsupported");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar,
               helper + "(" + input.spelling + ")"};
    return mlir::success();
  }

  mlir::LogicalResult emitSelect(SelectOp op) {
    CValue predicate = require(op.getPredicate());
    CValue trueValue = require(op.getTrueValue());
    CValue falseValue = require(op.getFalseValue());
    if (predicate.kind == CValueKind::Mask && inVLA &&
        elementType(op.getResult().getType()).isF32()) {
      unsigned lmul = activeVLADecision->dataLMUL;
      std::string suffix = "f32m" + std::to_string(lmul);
      auto materialize = [&](CValue value, llvm::StringRef prefix)
          -> std::optional<std::string> {
        if (value.kind == CValueKind::F32Vector)
          return value.spelling;
        if (value.kind != CValueKind::Scalar || value.spelling.empty())
          return std::nullopt;
        std::string name = fresh(prefix);
        line("vfloat32m" + std::to_string(lmul) + "_t " + name +
             " = __riscv_vfmv_v_f_" + suffix + "(" + value.spelling + ", " +
             activeVL + ");");
        return name;
      };
      std::optional<std::string> trueVector =
          materialize(trueValue, "select_true");
      std::optional<std::string> falseVector =
          materialize(falseValue, "select_false");
      if (!trueVector || !falseVector)
        return op.emitError("RVV select operands have no f32 realization");
      std::string name = fresh("select");
      line("vfloat32m" + std::to_string(lmul) + "_t " + name +
           " = __riscv_vmerge_vvm_" + suffix + "(" + *falseVector + ", " +
           *trueVector + ", " + predicate.spelling + ", " + activeVL + ");");
      values[op.getResult()] =
          CValue{op.getResult().getType(), CValueKind::F32Vector, name};
      return mlir::success();
    }
    if (predicate.kind != CValueKind::Scalar ||
        trueValue.kind != CValueKind::Scalar ||
        falseValue.kind != CValueKind::Scalar)
      return op.emitError("select operands have no selected realization");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar,
               "(" + predicate.spelling + " ? " + trueValue.spelling + " : " +
                   falseValue.spelling + ")"};
    return mlir::success();
  }

  mlir::LogicalResult emitTuple(TupleOp op) {
    CValue tuple{op.getResult().getType(), CValueKind::Tuple, {}};
    for (mlir::Value operand : op.getOperands()) {
      CValue field = require(operand);
      if (field.spelling.empty())
        return op.emitError("tuple field is unavailable");
      tuple.fields.push_back(std::move(field));
    }
    values[op.getResult()] = std::move(tuple);
    return mlir::success();
  }

  mlir::LogicalResult emitTupleGet(TupleGetOp op) {
    CValue tuple = require(op.getInput());
    if (tuple.kind != CValueKind::Tuple || op.getIndex() < 0 ||
        static_cast<size_t>(op.getIndex()) >= tuple.fields.size())
      return op.emitError("tuple field is unavailable");
    CValue field = tuple.fields[op.getIndex()];
    field.type = op.getResult().getType();
    values[op.getResult()] = field;
    return mlir::success();
  }

  mlir::LogicalResult emitSpecial(SpecialValueOp op) {
    std::string value;
    if (op.getKind() == "neg_inf")
      value = "-INFINITY";
    else if (op.getKind() == "pos_inf")
      value = "INFINITY";
    else if (op.getKind() == "nan")
      value = "NAN";
    else
      return op.emitError("special value kind is unsupported");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, value};
    return mlir::success();
  }

  std::optional<std::string> projectBlockScalar(
      mlir::Value value,
      const llvm::DenseMap<mlir::Value, std::string> &axisValues) {
    auto replacement = axisValues.find(value);
    if (replacement != axisValues.end())
      return replacement->second;
    if (!containsBlockType(value.getType())) {
      CValue materialized = require(value);
      if (!materialized.spelling.empty())
        return materialized.spelling;
    }
    if (auto pointer = value.getDefiningOp<PtrAddOp>()) {
      std::optional<std::string> base =
          projectBlockScalar(pointer.getBase(), axisValues);
      std::optional<std::string> offset =
          projectBlockScalar(pointer.getOffset(), axisValues);
      if (!base || !offset)
        return std::nullopt;
      return "(" + *base + " + " + *offset + ")";
    }
    if (auto binary = value.getDefiningOp<BinaryOp>()) {
      std::optional<std::string> lhs =
          projectBlockScalar(binary.getLhs(), axisValues);
      std::optional<std::string> rhs =
          projectBlockScalar(binary.getRhs(), axisValues);
      if (!lhs || !rhs)
        return std::nullopt;
      std::string projected = scalarBinary(binary.getKind(), *lhs, *rhs);
      return projected.empty() ? std::nullopt
                               : std::optional<std::string>(projected);
    }
    if (auto compare = value.getDefiningOp<CompareOp>()) {
      std::optional<std::string> lhs =
          projectBlockScalar(compare.getLhs(), axisValues);
      std::optional<std::string> rhs =
          projectBlockScalar(compare.getRhs(), axisValues);
      llvm::StringRef predicate =
          llvm::StringSwitch<llvm::StringRef>(compare.getPredicate())
              .Case("eq", "==")
              .Case("ne", "!=")
              .Case("lt", "<")
              .Case("le", "<=")
              .Case("gt", ">")
              .Case("ge", ">=")
              .Default("");
      if (!lhs || !rhs || predicate.empty())
        return std::nullopt;
      return "(" + *lhs + " " + predicate.str() + " " + *rhs + ")";
    }
    if (auto cast = value.getDefiningOp<CastOp>()) {
      std::optional<std::string> input =
          projectBlockScalar(cast.getInput(), axisValues);
      std::string type = scalarCType(elementType(cast.getResult().getType()));
      if (!input || type.empty())
        return std::nullopt;
      return "((" + type + ")(" + *input + "))";
    }
    if (auto expand = value.getDefiningOp<ExpandDimsOp>())
      return projectBlockScalar(expand.getInput(), axisValues);
    if (auto broadcast = value.getDefiningOp<BroadcastToOp>())
      return projectBlockScalar(broadcast.getInput(), axisValues);
    if (auto reshape = value.getDefiningOp<ReshapeOp>())
      return projectBlockScalar(reshape.getInput(), axisValues);
    if (auto transpose = value.getDefiningOp<TransposeOp>())
      return projectBlockScalar(transpose.getInput(), axisValues);
    std::string projected = expression(value);
    return projected.empty() ? std::nullopt
                             : std::optional<std::string>(projected);
  }

  std::optional<std::string> projectVLALaneStride(
      mlir::Value value, mlir::Value coordinate,
      const llvm::DenseMap<mlir::Value, std::string> &axisValues) {
    if (value == coordinate)
      return "1";
    if (axisValues.contains(value) || !dependsOn(value, coordinate))
      return "0";
    if (auto pointer = value.getDefiningOp<PtrAddOp>()) {
      std::optional<std::string> base =
          projectVLALaneStride(pointer.getBase(), coordinate, axisValues);
      std::optional<std::string> offset =
          projectVLALaneStride(pointer.getOffset(), coordinate, axisValues);
      if (!base || !offset)
        return std::nullopt;
      if (*base == "0")
        return offset;
      if (*offset == "0")
        return base;
      return "(" + *base + " + " + *offset + ")";
    }
    if (auto binary = value.getDefiningOp<BinaryOp>()) {
      bool lhsDependent = dependsOn(binary.getLhs(), coordinate);
      bool rhsDependent = dependsOn(binary.getRhs(), coordinate);
      if (binary.getKind() == "add" || binary.getKind() == "sub") {
        std::optional<std::string> lhs = projectVLALaneStride(
            binary.getLhs(), coordinate, axisValues);
        std::optional<std::string> rhs = projectVLALaneStride(
            binary.getRhs(), coordinate, axisValues);
        if (!lhs || !rhs)
          return std::nullopt;
        if (*rhs == "0")
          return lhs;
        if (*lhs == "0")
          return binary.getKind() == "add"
                     ? rhs
                     : std::optional<std::string>("(-" + *rhs + ")");
        return "(" + *lhs + (binary.getKind() == "add" ? " + " : " - ") +
               *rhs + ")";
      }
      if (binary.getKind() == "mul" && lhsDependent != rhsDependent) {
        mlir::Value dependent =
            lhsDependent ? binary.getLhs() : binary.getRhs();
        mlir::Value factor = lhsDependent ? binary.getRhs() : binary.getLhs();
        std::optional<std::string> stride =
            projectVLALaneStride(dependent, coordinate, axisValues);
        std::optional<std::string> scalar =
            projectBlockScalar(factor, axisValues);
        if (!stride || !scalar)
          return std::nullopt;
        if (*stride == "1")
          return scalar;
        return "(" + *stride + " * " + *scalar + ")";
      }
      return std::nullopt;
    }
    if (auto cast = value.getDefiningOp<CastOp>())
      return projectVLALaneStride(cast.getInput(), coordinate, axisValues);
    if (auto expand = value.getDefiningOp<ExpandDimsOp>())
      return projectVLALaneStride(expand.getInput(), coordinate, axisValues);
    if (auto broadcast = value.getDefiningOp<BroadcastToOp>())
      return projectVLALaneStride(broadcast.getInput(), coordinate, axisValues);
    if (auto reshape = value.getDefiningOp<ReshapeOp>())
      return projectVLALaneStride(reshape.getInput(), coordinate, axisValues);
    if (auto transpose = value.getDefiningOp<TransposeOp>())
      return projectVLALaneStride(transpose.getInput(), coordinate, axisValues);
    return std::nullopt;
  }

  mlir::LogicalResult emitVLAContract(const VLAContractDecision &decision) {
    auto contract = mlir::cast<ContractOp>(decision.operation);
    auto store = mlir::cast<StoreOp>(decision.consumer);
    auto lhsLoad = mlir::cast<LoadOp>(decision.lhsLoad);
    auto rhsLoad = mlir::cast<LoadOp>(decision.rhsLoad);
    std::string extent = expression(decision.reductionExtent);
    if (extent.empty() || activeVL.empty())
      return contract.emitError(
          "RVV VLA contract physical bounds are unavailable");

    std::string vectorSuffix = "f32m" + std::to_string(decision.lmul);
    std::string vectorType =
        "vfloat32m" + std::to_string(decision.lmul) + "_t";
    llvm::SmallVector<std::string> lhsActive;
    llvm::SmallVector<std::string> storeActive;
    llvm::SmallVector<std::string> outputPointers;
    for (unsigned lane = 0; lane < decision.rowTile; ++lane) {
      llvm::DenseMap<mlir::Value, std::string> axes;
      axes[decision.rowAxis] = std::to_string(lane);
      axes[decision.reductionAxis] = "0";
      std::optional<std::string> lhsPredicate =
          projectBlockScalar(lhsLoad.getWhere(), axes);
      std::optional<std::string> outputPredicate =
          projectBlockScalar(store.getWhere(), axes);
      std::optional<std::string> outputPointer =
          projectBlockScalar(store.getPointer(), axes);
      if ((!decision.lhsPredicateVariesByReduction && !lhsPredicate) ||
          !outputPredicate || !outputPointer)
        return contract.emitError(
            "RVV VLA contract row projection is unavailable");
      std::string outputCondition = fresh("vla_contract_store_active");
      if (!decision.lhsPredicateVariesByReduction) {
        std::string lhsCondition = fresh("vla_contract_lhs_active");
        line("const bool " + lhsCondition + " = " + *lhsPredicate + ";");
        lhsActive.push_back(std::move(lhsCondition));
      }
      line("const bool " + outputCondition + " = " + *outputPredicate + ";");
      storeActive.push_back(std::move(outputCondition));
      outputPointers.push_back(std::move(*outputPointer));
    }

    auto emitCompute = [&](bool guarded) -> mlir::LogicalResult {
      llvm::SmallVector<std::string> accumulators;
      for (unsigned lane = 0; lane < decision.rowTile; ++lane) {
        std::string accumulator = fresh("vla_contract_acc");
        line(vectorType + " " + accumulator + " = __riscv_vfmv_v_f_" +
             vectorSuffix + "(0.0f, " + activeVL + ");");
        accumulators.push_back(std::move(accumulator));
      }

      std::string reduction = fresh("vla_contract_k");
      llvm::DenseMap<mlir::Value, std::string> rhsAxes;
      rhsAxes[decision.reductionAxis] = reduction;
      std::optional<std::string> rhsPointer =
          projectBlockScalar(rhsLoad.getPointer(), rhsAxes);
      std::optional<std::string> rhsLaneStride = projectVLALaneStride(
          rhsLoad.getPointer(), activeVLADecision->coordinate, rhsAxes);
      if (!rhsPointer || !rhsLaneStride)
        return contract.emitError(
            "RVV VLA contract RHS projection is unavailable");
      line("for (size_t " + reduction + " = 0; " + reduction + " < " +
           extent + "; ++" + reduction + ") {");
      ++indent;
      std::string rhsVector = fresh("vla_contract_rhs");
      if (decision.rhsMemoryMode == VLAMemoryMode::UnitStride) {
        line(vectorType + " " + rhsVector + " = __riscv_vle32_v_" +
             vectorSuffix + "(" + *rhsPointer + ", " + activeVL + ");");
      } else {
        line(vectorType + " " + rhsVector + " = __riscv_vlse32_v_" +
             vectorSuffix + "(" + *rhsPointer +
             ", (ptrdiff_t)(sizeof(float) * (" + *rhsLaneStride + ")), " +
             activeVL + ");");
      }
      for (unsigned lane = 0; lane < decision.rowTile; ++lane) {
        llvm::DenseMap<mlir::Value, std::string> axes;
        axes[decision.rowAxis] = std::to_string(lane);
        axes[decision.reductionAxis] = reduction;
        std::optional<std::string> lhsPointer =
            projectBlockScalar(lhsLoad.getPointer(), axes);
        std::optional<std::string> lhsPredicate;
        if (decision.lhsPredicateVariesByReduction)
          lhsPredicate = projectBlockScalar(lhsLoad.getWhere(), axes);
        if (!lhsPointer ||
            (decision.lhsPredicateVariesByReduction && !lhsPredicate))
          return contract.emitError(
              "RVV VLA contract LHS projection is unavailable");
        bool guardedLoad = guarded || decision.lhsPredicateVariesByReduction;
        if (guardedLoad) {
          llvm::StringRef predicate = decision.lhsPredicateVariesByReduction
                                          ? *lhsPredicate
                                          : lhsActive[lane];
          line("if (" + predicate.str() + ") {");
          ++indent;
        }
        line(accumulators[lane] + " = __riscv_vfmacc_vf_" + vectorSuffix +
             "(" + accumulators[lane] + ", *" + *lhsPointer + ", " +
             rhsVector + ", " + activeVL + ");");
        if (guardedLoad) {
          --indent;
          line("}");
        }
      }
      --indent;
      line("}");

      for (unsigned lane = 0; lane < decision.rowTile; ++lane) {
        llvm::DenseMap<mlir::Value, std::string> axes;
        axes[decision.rowAxis] = std::to_string(lane);
        axes[decision.reductionAxis] = "0";
        std::optional<std::string> outputLaneStride = projectVLALaneStride(
            store.getPointer(), activeVLADecision->coordinate, axes);
        if (!outputLaneStride)
          return contract.emitError(
              "RVV VLA contract output stride is unavailable");
        if (guarded) {
          line("if (" + storeActive[lane] + ") {");
          ++indent;
        }
        if (decision.outputMemoryMode == VLAMemoryMode::UnitStride) {
          line("__riscv_vse32_v_" + vectorSuffix + "(" +
               outputPointers[lane] + ", " + accumulators[lane] + ", " +
               activeVL + ");");
        } else {
          line("__riscv_vsse32_v_" + vectorSuffix + "(" +
               outputPointers[lane] +
               ", (ptrdiff_t)(sizeof(float) * (" + *outputLaneStride + ")), " +
               accumulators[lane] + ", " + activeVL + ");");
        }
        if (guarded) {
          --indent;
          line("}");
        }
      }
      return mlir::success();
    };

    std::string fullTile = fresh("vla_contract_full_tile");
    llvm::SmallVector<llvm::StringRef> fullConditions;
    if (!decision.lhsPredicateVariesByReduction)
      fullConditions.append(lhsActive.begin(), lhsActive.end());
    fullConditions.append(storeActive.begin(), storeActive.end());
    line("const bool " + fullTile + " = " +
         llvm::join(fullConditions, " && ") + ";");
    line("if (" + fullTile + ") {");
    ++indent;
    if (mlir::failed(emitCompute(false)))
      return mlir::failure();
    --indent;
    line("} else {");
    ++indent;
    if (mlir::failed(emitCompute(true)))
      return mlir::failure();
    --indent;
    line("}");
    return mlir::success();
  }

  mlir::FailureOr<ContractDecision>
  decideLocalF32RowMicrotile(StoreOp store, ContractOp contract) {
    auto unwrapBlock = [](mlir::Type type) -> BlockType {
      if (auto masked = mlir::dyn_cast<MaskedType>(type))
        type = masked.getValueType();
      return mlir::dyn_cast<BlockType>(type);
    };
    BlockType lhsType = unwrapBlock(contract.getLhs().getType());
    BlockType rhsType = unwrapBlock(contract.getRhs().getType());
    BlockType resultType = unwrapBlock(contract.getResult().getType());
    auto init = contract.getInit().getDefiningOp<FullOp>();
    if (store.getValue() != contract.getResult() || !lhsType || !rhsType ||
        !resultType || lhsType.getShape().size() != 2 ||
        rhsType.getShape().size() != 1 || resultType.getShape().size() != 1 ||
        lhsType.getShape()[0] != resultType.getShape()[0] ||
        lhsType.getShape()[1] != rhsType.getShape()[0] ||
        resultType.getShape()[0] <= 0 || resultType.getShape()[0] > 6 ||
        !lhsType.getElementType().isF32() ||
        !rhsType.getElementType().isF32() ||
        !resultType.getElementType().isF32() || !init ||
        !isFloatConstant(init.getValue(), 0.0) ||
        contract.getLhsAxes().size() != 1 ||
        contract.getRhsAxes().size() != 1 ||
        contract.getLhsAxes().front() != 1 ||
        contract.getRhsAxes().front() != 0 ||
        contract.getOrder() != "relaxed" || contract.getMath() != "native" ||
        !contract.getAccDtype().isF32() || !contract.getOutDtype().isF32() ||
        !isTrue(contract.getWhereLhs()) || !isTrue(contract.getWhereRhs())) {
      contract.emitError(
          "RVV row microtile requires a [BM,K] x [K] local f32 contract");
      return mlir::failure();
    }

    LoadOp lhsLoad = contract.getLhs().getDefiningOp<LoadOp>();
    LoadOp rhsLoad = contract.getRhs().getDefiningOp<LoadOp>();
    if (!lhsLoad || !rhsLoad || !isTrue(rhsLoad.getWhere())) {
      contract.emitError("RVV row microtile load producers are unavailable");
      return mlir::failure();
    }

    llvm::DenseSet<mlir::Operation *> lhsClosure;
    llvm::DenseSet<mlir::Operation *> rhsClosure;
    llvm::SmallVector<BlockAxisOp> lhsAxes;
    llvm::SmallVector<BlockAxisOp> rhsAxes;
    if (mlir::failed(collectBlockClosure(contract.getLhs(), lhsClosure, lhsAxes,
                                         store.getOperation())) ||
        mlir::failed(collectBlockClosure(contract.getRhs(), rhsClosure, rhsAxes,
                                         store.getOperation())))
      return mlir::failure();
    if (rhsAxes.size() != 1 || lhsAxes.size() != 2 ||
        !llvm::is_contained(lhsAxes, rhsAxes.front())) {
      contract.emitError(
          "RVV row microtile requires explicit row and shared K block axes");
      return mlir::failure();
    }
    BlockAxisOp reductionAxis = rhsAxes.front();
    BlockAxisOp rowAxis = lhsAxes.front() == reductionAxis
                              ? lhsAxes.back()
                              : lhsAxes.front();
    mlir::Value lhsRoot = pointerRoot(lhsLoad.getPointer());
    mlir::Value rhsRoot = pointerRoot(rhsLoad.getPointer());
    mlir::Value outputRoot = pointerRoot(store.getPointer());
    auto lhsPointer = lhsRoot ? mlir::dyn_cast<PtrType>(lhsRoot.getType())
                              : PtrType{};
    auto rhsPointer = rhsRoot ? mlir::dyn_cast<PtrType>(rhsRoot.getType())
                              : PtrType{};
    auto outputPointer =
        outputRoot ? mlir::dyn_cast<PtrType>(outputRoot.getType()) : PtrType{};
    if (!lhsPointer || !rhsPointer || !outputPointer ||
        !lhsPointer.getElementType().isF32() ||
        !rhsPointer.getElementType().isF32() ||
        !outputPointer.getElementType().isF32() ||
        dependsOn(lhsLoad.getWhere(), reductionAxis.getResult()) ||
        dependsOn(store.getWhere(), reductionAxis.getResult()) ||
        (!isTrue(lhsLoad.getWhere()) &&
         !isFloatConstant(lhsLoad.getOther(), 0.0))) {
      contract.emitError("RVV row microtile pointer facts are unavailable");
      return mlir::failure();
    }

    ContractDecision decision;
    decision.operation = contract.getOperation();
    decision.realization = ContractRealization::RVVF32RowMicrotile;
    decision.consumer = store.getOperation();
    decision.lhsLoad = lhsLoad.getOperation();
    decision.rhsLoad = rhsLoad.getOperation();
    decision.rowAxis = rowAxis.getResult();
    decision.reductionAxis = reductionAxis.getResult();
    decision.reductionExtent = reductionAxis.getExtent();
    decision.rowTile = static_cast<unsigned>(resultType.getShape()[0]);
    decision.lmul = 4;
    return decision;
  }

  std::optional<mlir::LogicalResult>
  tryEmitLocalF32ContractBlockStore(StoreOp store) {
    auto contract = store.getValue().getDefiningOp<ContractOp>();
    if (!contract)
      return std::nullopt;
    mlir::FailureOr<ContractDecision> selected =
        decideLocalF32RowMicrotile(store, contract);
    if (mlir::failed(selected))
      return mlir::failure();
    ContractDecision decision = *selected;
    auto lhsLoad = mlir::cast<LoadOp>(decision.lhsLoad);
    auto rhsLoad = mlir::cast<LoadOp>(decision.rhsLoad);
    std::string extent = expression(decision.reductionExtent);
    if (extent.empty())
      return contract.emitError(
          "RVV row microtile access projection is unavailable");

    std::string fullVL = fresh("contract_vlmax");
    std::string vectorSuffix = "f32m" + std::to_string(decision.lmul);
    std::string vectorType = "vfloat32m" + std::to_string(decision.lmul) + "_t";
    line("const size_t " + fullVL + " = __riscv_vsetvlmax_e32m" +
         std::to_string(decision.lmul) + "();");

    llvm::SmallVector<std::string> lhsActive;
    llvm::SmallVector<std::string> storeActive;
    llvm::SmallVector<std::string> outputPointers;
    for (unsigned lane = 0; lane < decision.rowTile; ++lane) {
      llvm::DenseMap<mlir::Value, std::string> axes;
      axes[decision.rowAxis] = std::to_string(lane);
      axes[decision.reductionAxis] = "0";
      std::optional<std::string> lhsPredicate =
          projectBlockScalar(lhsLoad.getWhere(), axes);
      std::optional<std::string> outputPredicate =
          projectBlockScalar(store.getWhere(), axes);
      std::optional<std::string> outputPointer =
          projectBlockScalar(store.getPointer(), axes);
      if (!lhsPredicate || !outputPredicate || !outputPointer)
        return contract.emitError(
            "RVV row microtile row projection is unavailable");
      std::string lhsCondition = fresh("contract_lhs_active");
      std::string outputCondition = fresh("contract_store_active");
      line("const bool " + lhsCondition + " = " + *lhsPredicate + ";");
      line("const bool " + outputCondition + " = " + *outputPredicate + ";");
      lhsActive.push_back(std::move(lhsCondition));
      storeActive.push_back(std::move(outputCondition));
      outputPointers.push_back(std::move(*outputPointer));
    }

    auto emitCompute = [&](bool guarded) -> mlir::LogicalResult {
      llvm::SmallVector<std::string> accumulators;
      for (unsigned lane = 0; lane < decision.rowTile; ++lane) {
        std::string accumulator = fresh("contract_acc");
        line(vectorType + " " + accumulator + " = __riscv_vfmv_v_f_" +
             vectorSuffix + "(0.0f, " + fullVL + ");");
        accumulators.push_back(std::move(accumulator));
      }
      std::string strip = fresh("contract_k");
      std::string vl = fresh("contract_vl");
      llvm::DenseMap<mlir::Value, std::string> rhsAxes;
      rhsAxes[decision.reductionAxis] = strip;
      std::optional<std::string> rhs =
          projectBlockScalar(rhsLoad.getPointer(), rhsAxes);
      if (!rhs)
        return contract.emitError(
            "RVV row microtile RHS projection is unavailable");
      line("for (size_t " + strip + " = 0; " + strip + " < " + extent +
           ";) {");
      ++indent;
      line("const size_t " + vl + " = __riscv_vsetvl_e32m" +
           std::to_string(decision.lmul) + "(" + extent + " - " + strip +
           ");");
      std::string rhsVector = fresh("contract_rhs");
      line(vectorType + " " + rhsVector + " = __riscv_vle32_v_" +
           vectorSuffix + "(" + *rhs + ", " + vl + ");");
      for (unsigned lane = 0; lane < decision.rowTile; ++lane) {
        llvm::DenseMap<mlir::Value, std::string> axes;
        axes[decision.rowAxis] = std::to_string(lane);
        axes[decision.reductionAxis] = strip;
        std::optional<std::string> lhs =
            projectBlockScalar(lhsLoad.getPointer(), axes);
        if (!lhs)
          return contract.emitError(
              "RVV row microtile LHS projection is unavailable");
        if (guarded) {
          line("if (" + lhsActive[lane] + ") {");
          ++indent;
        }
        std::string lhsVector = fresh("contract_lhs");
        line(vectorType + " " + lhsVector + " = __riscv_vle32_v_" +
             vectorSuffix + "(" + *lhs + ", " + vl + ");");
        line(accumulators[lane] + " = __riscv_vfmacc_vv_" + vectorSuffix +
             "_tu(" + accumulators[lane] + ", " + lhsVector + ", " +
             rhsVector + ", " + vl + ");");
        if (guarded) {
          --indent;
          line("}");
        }
      }
      line(strip + " += " + vl + ";");
      --indent;
      line("}");
      for (unsigned lane = 0; lane < decision.rowTile; ++lane) {
        if (guarded) {
          line("if (" + storeActive[lane] + ") {");
          ++indent;
        }
        std::string seed = fresh("contract_seed");
        std::string reduced = fresh("contract_reduced");
        line("vfloat32m1_t " + seed +
             " = __riscv_vfmv_v_f_f32m1(0.0f, 1);");
        line("vfloat32m1_t " + reduced +
             " = __riscv_vfredusum_vs_" + vectorSuffix + "_f32m1(" +
             accumulators[lane] + ", " + seed + ", " + fullVL + ");");
        line("*" + outputPointers[lane] +
             " = __riscv_vfmv_f_s_f32m1_f32(" + reduced + ");");
        if (guarded) {
          --indent;
          line("}");
        }
      }
      return mlir::success();
    };

    std::string fullTile = fresh("contract_full_tile");
    line("const bool " + fullTile + " = " +
         llvm::join(lhsActive, " && ") + " && " +
         llvm::join(storeActive, " && ") + ";");
    line("if (" + fullTile + ") {");
    ++indent;
    if (mlir::failed(emitCompute(false)))
      return mlir::failure();
    --indent;
    line("} else {");
    ++indent;
    if (mlir::failed(emitCompute(true)))
      return mlir::failure();
    --indent;
    line("}");

    llvm::DenseSet<mlir::Operation *> closure;
    llvm::SmallVector<BlockAxisOp> axes;
    for (mlir::Value value : {store.getPointer(), store.getValue(),
                              store.getWhere()})
      if (containsBlockType(value.getType()) &&
          mlir::failed(collectBlockClosure(value, closure, axes,
                                           store.getOperation())))
        return mlir::failure();
    for (mlir::Operation *operation : closure)
      loweredBlockOps.insert(operation);
    return mlir::success();
  }

  mlir::LogicalResult emitLoad(LoadOp op) {
    CValue pointer = require(op.getPointer());
    if (pointer.kind != CValueKind::Pointer || pointer.spelling.empty())
      return op.emitError("load pointer is unavailable");
    if (inVLA && pointer.lanePointer) {
      const VLAAccessDecision *decision =
          findAccessDecision(op.getOperation());
      if (!decision)
        return op.emitError("VLA load has no physical memory decision");
      mlir::Type loadedElement = decision->elementType;
      if ((!loadedElement.isF32() && !isF16(loadedElement)) ||
          pointer.laneStride.empty())
        return op.emitError("VLA load element realization is unavailable");
      if (decision->activityMode == VLAActivityMode::PredicateMask)
        return op.emitError("masked VLA load has no selected realization");
      std::string name = fresh("load");
      bool f32 = loadedElement.isF32();
      unsigned lmul =
          f32 ? activeVLADecision->dataLMUL
              : activeVLADecision->dataLMUL / 2;
      if (lmul == 0)
        return op.emitError("VLA load has no compatible LMUL");
      std::string element = f32 ? "f32m" : "f16m";
      std::string vectorType =
          std::string(f32 ? "vfloat32m" : "vfloat16m") +
          std::to_string(lmul) + "_t";
      std::string suffix = element + std::to_string(lmul);
      auto emitRead = [&]() {
        if (decision->memoryMode == VLAMemoryMode::UnitStride) {
          line(name + " = __riscv_vle" + std::string(f32 ? "32" : "16") +
               "_v_" + suffix + "(" + pointer.spelling + ", " + activeVL +
               ");");
        } else {
          line(name + " = __riscv_vlse" + std::string(f32 ? "32" : "16") +
               "_v_" + suffix + "(" + pointer.spelling +
               ", (ptrdiff_t)(sizeof(" + std::string(f32 ? "float" : "_Float16") +
               ") * (" + pointer.laneStride + ")), " + activeVL + ");");
        }
      };
      line(vectorType + " " + name + ";");
      if (decision->activityMode == VLAActivityMode::ScalarPredicate) {
        CValue predicate = require(decision->predicate);
        CValue other = require(op.getOther());
        if (predicate.kind != CValueKind::Scalar || predicate.spelling.empty() ||
            other.kind != CValueKind::Scalar || other.spelling.empty())
          return op.emitError(
              "scalar-predicated VLA load requires a scalar predicate and other");
        line("if (" + predicate.spelling + ") {");
        ++indent;
        emitRead();
        --indent;
        line("} else {");
        ++indent;
        line(name + " = __riscv_vfmv_v_f_" + suffix + "(" + other.spelling +
             ", " + activeVL + ");");
        --indent;
        line("}");
      } else {
        emitRead();
      }
      if (loadedElement.isF32()) {
        values[op.getResult()] =
            CValue{op.getResult().getType(), CValueKind::F32Vector, name};
      } else {
        values[op.getResult()] =
            CValue{op.getResult().getType(), CValueKind::F16Vector, name};
      }
      return mlir::success();
    }
    CValue where = require(op.getWhere());
    CValue other = require(op.getOther());
    std::string read = "(*" + pointer.spelling + ")";
    std::string expression;
    if (isTrue(op.getWhere()))
      expression = read;
    else if (!other.spelling.empty())
      expression = "(" + where.spelling + " ? " + read + " : " +
                   other.spelling + ")";
    else
      return op.emitError("masked scalar load requires an explicit other value");
    values[op.getResult()] = scalarExpression(
        op.getResult(), std::move(expression), "load", true);
    return mlir::success();
  }

  mlir::LogicalResult emitStore(StoreOp op) {
    CValue pointer = require(op.getPointer());
    CValue value = require(op.getValue());
    if (pointer.kind != CValueKind::Pointer || pointer.spelling.empty())
      return op.emitError("store pointer is unavailable");
    if (inVLA && pointer.lanePointer) {
      const VLAAccessDecision *decision =
          findAccessDecision(op.getOperation());
      if (!decision)
        return op.emitError("VLA store has no physical memory decision");
      bool f32 = decision->elementType.isF32();
      bool f16 = isF16(decision->elementType);
      bool i8 = decision->elementType.isSignedInteger(8);
      if ((!f32 && !f16 && !i8) ||
          pointer.laneStride.empty())
        return op.emitError("VLA store element realization is unavailable");

      std::string vector = value.spelling;
      CValueKind expectedKind = f32   ? CValueKind::F32Vector
                                : f16 ? CValueKind::F16Vector
                                      : CValueKind::I8Vector;
      if (decision->storeValueMode == VLAStoreValueMode::ScalarBroadcast) {
        if (i8 || value.kind != CValueKind::Scalar || value.spelling.empty())
          return op.emitError("VLA store scalar broadcast is unavailable");
        vector = fresh("store_value");
        unsigned lmul =
            f32 ? activeVLADecision->dataLMUL
                : activeVLADecision->dataLMUL / 2;
        std::string suffix =
            std::string(f32 ? "f32m" : "f16m") + std::to_string(lmul);
        line(std::string(f32 ? "vfloat32m" : "vfloat16m") +
             std::to_string(lmul) + "_t " + vector +
             " = __riscv_vfmv_v_f_" + suffix + "(" + value.spelling + ", " +
             activeVL + ");");
      } else if (value.kind != expectedKind || value.spelling.empty()) {
        return op.emitError("VLA store vector projection is unavailable");
      }

      CValue predicate;
      if (decision->activityMode == VLAActivityMode::PredicateMask) {
        predicate = require(decision->predicate);
        if (predicate.kind != CValueKind::Mask || predicate.spelling.empty())
          return op.emitError("VLA store mask projection is unavailable");
      }
      bool scalarPredicated =
          decision->activityMode == VLAActivityMode::ScalarPredicate;
      if (scalarPredicated) {
        predicate = require(decision->predicate);
        if (predicate.kind != CValueKind::Scalar || predicate.spelling.empty())
          return op.emitError("VLA store scalar predicate is unavailable");
        line("if (" + predicate.spelling + ") {");
        ++indent;
      }
      unsigned lmul = f32   ? activeVLADecision->dataLMUL
                      : f16 ? activeVLADecision->dataLMUL / 2
                            : activeVLADecision->dataLMUL / 4;
      std::string sew = f32 ? "32" : f16 ? "16" : "8";
      std::string suffix =
          std::string(f32 ? "f32m" : f16 ? "f16m" : "i8m") +
          std::to_string(lmul);
      std::string elementCType = f32 ? "float" : f16 ? "_Float16" : "int8_t";
      if (decision->memoryMode == VLAMemoryMode::UnitStride) {
        if (decision->activityMode != VLAActivityMode::PredicateMask)
          line("__riscv_vse" + sew + "_v_" + suffix + "(" + pointer.spelling +
               ", " + vector + ", " + activeVL + ");");
        else
          line("__riscv_vse" + sew + "_v_" + suffix + "_m(" +
               predicate.spelling + ", " + pointer.spelling + ", " + vector +
               ", " + activeVL + ");");
      } else if (decision->activityMode != VLAActivityMode::PredicateMask) {
        line("__riscv_vsse" + sew + "_v_" + suffix + "(" + pointer.spelling +
             ", (ptrdiff_t)(sizeof(" + elementCType + ") * (" +
             pointer.laneStride + ")), " + vector + ", " + activeVL + ");");
      } else {
        line("__riscv_vsse" + sew + "_v_" + suffix + "_m(" +
             predicate.spelling + ", " + pointer.spelling +
             ", (ptrdiff_t)(sizeof(" + elementCType + ") * (" +
             pointer.laneStride + ")), " + vector + ", " + activeVL + ");");
      }
      if (scalarPredicated) {
        --indent;
        line("}");
      }
      return mlir::success();
    }
    CValue where = require(op.getWhere());
    if (value.kind != CValueKind::Scalar)
      return op.emitError("scalar store requires a scalar value");
    if (isTrue(op.getWhere()))
      line("*" + pointer.spelling + " = " + value.spelling + ";");
    else
      line("if (" + where.spelling + ") *" + pointer.spelling + " = " +
           value.spelling + ";");
    return mlir::success();
  }

  mlir::Value pointerRoot(mlir::Value value) const {
    if (mlir::isa<PtrType>(value.getType()))
      return value;
    auto pointer = value.getDefiningOp<PtrAddOp>();
    if (!pointer)
      return {};
    return pointerRoot(pointer.getBase());
  }

  mlir::Value blockStride(mlir::Value value,
                          llvm::DenseSet<mlir::Value> &visited) const {
    if (!value || !visited.insert(value).second)
      return {};
    if (auto binary = value.getDefiningOp<BinaryOp>()) {
      bool lhsBlock = containsBlockType(binary.getLhs().getType());
      bool rhsBlock = containsBlockType(binary.getRhs().getType());
      if (binary.getKind() == "mul" && lhsBlock != rhsBlock)
        return lhsBlock ? binary.getRhs() : binary.getLhs();
    }
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition)
      return {};
    for (mlir::Value operand : definition->getOperands())
      if (containsBlockType(operand.getType()))
        if (mlir::Value stride = blockStride(operand, visited))
          return stride;
    return {};
  }

  mlir::Value blockStride(mlir::Value value) const {
    llvm::DenseSet<mlir::Value> visited;
    return blockStride(value, visited);
  }

  bool dependsOn(mlir::Value value, mlir::Value target,
                 llvm::DenseSet<mlir::Value> &visited) const {
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

  bool dependsOn(mlir::Value value, mlir::Value target) const {
    llvm::DenseSet<mlir::Value> visited;
    return dependsOn(value, target, visited);
  }

  bool matchBinaryOperands(mlir::Value value, llvm::StringRef kind,
                           mlir::Value lhs, mlir::Value rhs,
                           bool commutative = false) const {
    auto binary = value.getDefiningOp<BinaryOp>();
    if (!binary || binary.getKind() != kind)
      return false;
    return (binary.getLhs() == lhs && binary.getRhs() == rhs) ||
           (commutative && binary.getLhs() == rhs && binary.getRhs() == lhs);
  }

  bool matchBinaryConstant(mlir::Value value, llvm::StringRef kind,
                           int64_t constant, mlir::Value &other,
                           bool commutative = true) const {
    auto binary = value.getDefiningOp<BinaryOp>();
    if (!binary || binary.getKind() != kind)
      return false;
    if (integerConstant(binary.getRhs()) == constant) {
      other = binary.getLhs();
      return true;
    }
    if (commutative && integerConstant(binary.getLhs()) == constant) {
      other = binary.getRhs();
      return true;
    }
    return false;
  }

  bool matchExpandedAxis(mlir::Value value, mlir::Value axis,
                         int64_t dimension) const {
    auto expand = value.getDefiningOp<ExpandDimsOp>();
    return expand && expand.getInput() == axis && expand.getAxis() == dimension;
  }

  bool matchBlockAxis(mlir::Value value, int64_t extent) const {
    auto axis = value.getDefiningOp<BlockAxisOp>();
    return axis && integerConstant(axis.getExtent()) == extent &&
           integerConstant(axis.getOffset()) == 0;
  }

  mlir::Value scalarPointerBase(mlir::Value value) const {
    while (value && !mlir::isa<PtrType>(value.getType())) {
      auto pointer = value.getDefiningOp<PtrAddOp>();
      if (!pointer)
        return {};
      value = pointer.getBase();
    }
    return value;
  }

  bool matchF16LELoad(mlir::Value value, LoadOp &low, LoadOp &high) const {
    auto widen = value.getDefiningOp<CastOp>();
    auto bits = widen ? widen.getInput().getDefiningOp<BitcastOp>() : BitcastOp{};
    auto combine = bits ? bits.getInput().getDefiningOp<BinaryOp>() : BinaryOp{};
    if (!widen || !bits || !combine || combine.getKind() != "or" ||
        !isF16(elementType(bits.getResult().getType())) ||
        !elementType(widen.getResult().getType()).isF32())
      return false;

    auto lowCast = combine.getLhs().getDefiningOp<CastOp>();
    auto shift = combine.getRhs().getDefiningOp<BinaryOp>();
    if (!lowCast || !shift || shift.getKind() != "shl" ||
        integerConstant(shift.getRhs()) != 8)
      return false;
    auto highCast = shift.getLhs().getDefiningOp<CastOp>();
    low = lowCast.getInput().getDefiningOp<LoadOp>();
    high = highCast ? highCast.getInput().getDefiningOp<LoadOp>() : LoadOp{};
    if (!low || !high || !isTrue(low.getWhere()) || !isTrue(high.getWhere()) ||
        !elementType(low.getResult().getType()).isUnsignedInteger(8) ||
        !elementType(high.getResult().getType()).isUnsignedInteger(8))
      return false;
    auto highPointer = high.getPointer().getDefiningOp<PtrAddOp>();
    return highPointer && highPointer.getBase() == low.getPointer() &&
           integerConstant(highPointer.getOffset()) == 1;
  }

  mlir::LogicalResult decideSymmetricI4I8Contract(
      SymmetricI4I8ContractOp op, SymmetricI4I8Decision &decision) {
    if (options.target.matrixExtension != "spacemit-ime1")
      return op.emitError(
          "symmetric i4/i8 contraction requires --matrix-extension=spacemit-ime1");
    if (options.target.vlenBits != 256)
      return op.emitError(
          "SpacemiT IME1 contraction requires an explicit 256-bit VLEN target fact");
    if (!options.target.littleEndian)
      return op.emitError(
          "symmetric i4/i8 contraction requires little-endian packed fields");

    LoadOp activationLoad = op.getActivation().getDefiningOp<LoadOp>();
    LoadOp packedCodeLoad = op.getPackedWeight().getDefiningOp<LoadOp>();
    LoadOp scaleLow;
    LoadOp scaleHigh;
    if (!activationLoad || !packedCodeLoad ||
        !matchF16LELoad(op.getWeightScale(), scaleLow, scaleHigh) ||
        !isTrue(activationLoad.getWhere()) ||
        !isTrue(packedCodeLoad.getWhere()))
      return op.emitError(
          "IME1 symmetric contraction requires explicit activation, scale, and packed-code loads");

    auto activationLane = activationLoad.getPointer().getDefiningOp<PtrAddOp>();
    if (!activationLane || !matchBlockAxis(activationLane.getOffset(), 32))
      return op.emitError(
          "IME1 symmetric contraction requires a contiguous K32 activation block");

    auto scaleLane = scaleLow.getPointer().getDefiningOp<PtrAddOp>();
    mlir::Value scaleColumn;
    if (!scaleLane ||
        !matchBinaryConstant(scaleLane.getOffset(), "mul", 2, scaleColumn) ||
        !matchBlockAxis(scaleColumn, 16) ||
        scalarPointerBase(scaleHigh.getPointer()) != scaleLane.getBase())
      return op.emitError(
          "IME1 symmetric contraction requires sixteen little-endian fp16 scales");
    mlir::Value packedBase = scaleLane.getBase();

    auto withinAdd = packedCodeLoad.getPointer().getDefiningOp<PtrAddOp>();
    auto within = withinAdd ? withinAdd.getOffset().getDefiningOp<BinaryOp>()
                            : BinaryOp{};
    auto columnAdd = withinAdd ? withinAdd.getBase().getDefiningOp<PtrAddOp>()
                               : PtrAddOp{};
    mlir::Value expandedColumn;
    auto halfAdd = columnAdd ? columnAdd.getBase().getDefiningOp<PtrAddOp>()
                             : PtrAddOp{};
    mlir::Value halfIndex;
    auto payloadBase = halfAdd ? halfAdd.getBase().getDefiningOp<PtrAddOp>()
                               : PtrAddOp{};
    if (!withinAdd || !within || within.getKind() != "mod" ||
        integerConstant(within.getRhs()) != 8 || !columnAdd ||
        !matchBinaryConstant(columnAdd.getOffset(), "mul", 8, expandedColumn) ||
        !matchExpandedAxis(expandedColumn, scaleColumn, 1) || !halfAdd ||
        !matchBinaryConstant(halfAdd.getOffset(), "mul", 128, halfIndex) ||
        !payloadBase || payloadBase.getBase() != packedBase ||
        integerConstant(payloadBase.getOffset()) != 32)
      return op.emitError(
          "IME1 symmetric contraction requires the explicit two-half packed-i4 payload");
    auto half = halfIndex.getDefiningOp<BinaryOp>();
    auto expandedByte = within.getLhs().getDefiningOp<ExpandDimsOp>();
    auto expandedHalfByte =
        half ? half.getLhs().getDefiningOp<ExpandDimsOp>() : ExpandDimsOp{};
    if (!half || half.getKind() != "div" ||
        integerConstant(half.getRhs()) != 8 || !expandedByte ||
        expandedByte.getAxis() != 0 || !expandedHalfByte ||
        expandedHalfByte.getAxis() != 0 ||
        expandedHalfByte.getInput() != expandedByte.getInput() ||
        !matchBlockAxis(expandedByte.getInput(), 16))
      return op.emitError(
          "IME1 symmetric contraction requires a local sixteen-byte packed axis");

    decision = SymmetricI4I8Decision{};
    decision.activationBlockBase = activationLane.getBase();
    decision.packedBlockBase = packedBase;
    decision.activationScale = op.getActivationScale();
    decision.init = op.getInit();
    return mlir::success();
  }

  mlir::LogicalResult emitSymmetricI4I8Contract(
      SymmetricI4I8ContractOp op) {
    SymmetricI4I8Decision decision;
    if (mlir::failed(decideSymmetricI4I8Contract(op, decision)))
      return mlir::failure();
    CValue activation = require(decision.activationBlockBase);
    CValue packed = require(decision.packedBlockBase);
    CValue scale = require(decision.activationScale);
    CValue accumulator = require(decision.init);
    if (decision.realization !=
            SymmetricI4I8Realization::SpacemiTIME1N16K32 ||
        decision.packedBlockBytes != 288 ||
        activation.kind != CValueKind::Pointer ||
        packed.kind != CValueKind::Pointer || scale.kind != CValueKind::Scalar ||
        accumulator.kind != CValueKind::F32BlockStorage ||
        activation.spelling.empty() || packed.spelling.empty() ||
        scale.spelling.empty() || accumulator.spelling.empty())
      return op.emitError(
          "selected IME1 symmetric contraction operands are unavailable");
    line("__weft_ime1_symmetric_i4_i8_n16_k32(" + scale.spelling + ", " +
         activation.spelling + ", " + packed.spelling + ", " +
         accumulator.spelling + ");");
    for (mlir::Value operand :
         {op.getActivation(), op.getPackedWeight(), op.getWeightScale(),
          op.getInit()})
      markDiscardedBlockTree(operand);
    loweredBlockOps.insert(op.getOperation());
    accumulator.type = op.getResult().getType();
    values[op.getResult()] = std::move(accumulator);
    return mlir::success();
  }

  std::optional<mlir::LogicalResult>
  tryEmitMaterializedF32BlockStore(StoreOp op) {
    auto stored = values.find(op.getValue());
    if (stored == values.end() ||
        stored->second.kind != CValueKind::F32BlockStorage)
      return std::nullopt;
    if (options.target.vlenBits != 256 || !isTrue(op.getWhere()))
      return std::optional<mlir::LogicalResult>(op.emitError(
          "materialized f32 block store requires an all-active VLEN256 target"));
    auto lanePointer = op.getPointer().getDefiningOp<PtrAddOp>();
    if (!lanePointer || !matchBlockAxis(lanePointer.getOffset(), 16))
      return std::optional<mlir::LogicalResult>(op.emitError(
          "materialized f32 block store requires a contiguous block<16> address"));
    CValue destination = require(lanePointer.getBase());
    if (destination.kind != CValueKind::Pointer ||
        destination.spelling.empty())
      return std::optional<mlir::LogicalResult>(op.emitError(
          "materialized f32 block store has no scalar pointer base"));
    std::string vl = fresh("ime_store_vl");
    std::string value = fresh("ime_store_value");
    line("const size_t " + vl + " = __riscv_vsetvl_e32m2(16);");
    line("vfloat32m2_t " + value + " = __riscv_vle32_v_f32m2(" +
         stored->second.spelling + ", " + vl + ");");
    line("__riscv_vse32_v_f32m2(" + destination.spelling + ", " + value +
         ", " + vl + ");");
    markDiscardedBlockTree(op.getPointer());
    loweredBlockOps.insert(op.getOperation());
    consumed.insert(op.getOperation());
    return std::optional<mlir::LogicalResult>(mlir::success());
  }

  mlir::LogicalResult emitGroupedAffineI4I8Dot(GroupedAffineI4I8DotOp op) {
    if (options.target.vlenBits != 128)
      return op.emitError(
          "grouped affine i4/i8 dot currently requires an explicit VLEN128 target fact");
    if (!options.target.littleEndian)
      return op.emitError(
          "grouped affine i4/i8 dot requires the source-declared little-endian fields");
    if (!op.getPackedWeight().hasOneUse() || !op.getScaleMin().hasOneUse() ||
        !op.getActivation().hasOneUse() ||
        !op.getActivationSumBytes().hasOneUse())
      return op.emitError(
          "grouped affine i4/i8 dot requires local single-use block operands");

    LoadOp packedLoad = op.getPackedWeight().getDefiningOp<LoadOp>();
    LoadOp scaleLoad = op.getScaleMin().getDefiningOp<LoadOp>();
    auto activationCast = op.getActivation().getDefiningOp<BitcastOp>();
    LoadOp activationLoad =
        activationCast ? activationCast.getInput().getDefiningOp<LoadOp>()
                       : LoadOp{};
    LoadOp sumLoad = op.getActivationSumBytes().getDefiningOp<LoadOp>();
    if (!packedLoad || !scaleLoad || !activationCast || !activationLoad ||
        !sumLoad || !isTrue(packedLoad.getWhere()) ||
        !isTrue(scaleLoad.getWhere()) || !isTrue(activationLoad.getWhere()) ||
        !isTrue(sumLoad.getWhere()))
      return op.emitError(
          "grouped affine i4/i8 dot requires explicit all-active block loads");

    auto blockBase = [&](LoadOp load, int64_t extent) -> mlir::Value {
      auto lane = load.getPointer().getDefiningOp<PtrAddOp>();
      if (!lane || !matchBlockAxis(lane.getOffset(), extent))
        return {};
      return lane.getBase();
    };
    mlir::Value packedBase = blockBase(packedLoad, 128);
    mlir::Value scaleBase = blockBase(scaleLoad, 12);
    mlir::Value activationBase = blockBase(activationLoad, 256);
    mlir::Value sumBase = blockBase(sumLoad, 32);
    if (!packedBase || !scaleBase || !activationBase || !sumBase)
      return op.emitError(
          "grouped affine i4/i8 dot block axes do not match its typed operands");

    CValue packed = require(packedBase);
    CValue scales = require(scaleBase);
    CValue activation = require(activationBase);
    CValue sums = require(sumBase);
    CValue dotScale = require(op.getDotScale());
    CValue minimumScale = require(op.getMinimumScale());
    CValue init = require(op.getInit());
    if (packed.kind != CValueKind::Pointer || scales.kind != CValueKind::Pointer ||
        activation.kind != CValueKind::Pointer || sums.kind != CValueKind::Pointer ||
        dotScale.kind != CValueKind::Scalar ||
        minimumScale.kind != CValueKind::Scalar || init.kind != CValueKind::Scalar ||
        packed.spelling.empty() || scales.spelling.empty() ||
        activation.spelling.empty() || sums.spelling.empty() ||
        dotScale.spelling.empty() || minimumScale.spelling.empty() ||
        init.spelling.empty())
      return op.emitError(
          "grouped affine i4/i8 dot operands are not materialized locally");

    std::string result = fresh("grouped_dot");
    line("const float " + result + " = __weft_grouped_affine_i4_i8_vl128(" +
         packed.spelling + ", " + scales.spelling + ", " +
         activation.spelling + ", " + sums.spelling + ", " +
         dotScale.spelling + ", " + minimumScale.spelling + ", " +
         init.spelling + ");");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, result};
    markDiscardedBlockTree(op.getPackedWeight());
    markDiscardedBlockTree(op.getScaleMin());
    markDiscardedBlockTree(op.getActivation());
    markDiscardedBlockTree(op.getActivationSumBytes());
    return mlir::success();
  }

  std::optional<mlir::LogicalResult> tryEmitAffineI4I8NTiles(ForOp nLoop) {
    mlir::Block &nBody = nLoop.getBody().front();
    ForOp kLoop;
    StoreOp store;
    for (mlir::Operation &operation : nBody.without_terminator()) {
      if (auto candidate = mlir::dyn_cast<ForOp>(operation)) {
        if (kLoop)
          return std::nullopt;
        kLoop = candidate;
      } else if (auto candidate = mlir::dyn_cast<StoreOp>(operation)) {
        if (store)
          return std::nullopt;
        store = candidate;
      }
    }
    if (!kLoop)
      return std::nullopt;

    mlir::Block &kBody = kLoop.getBody().front();
    AffineI4I8ContractOp contract;
    for (mlir::Operation &operation : kBody.without_terminator())
      if (auto candidate = mlir::dyn_cast<AffineI4I8ContractOp>(operation)) {
        if (contract)
          return std::nullopt;
        contract = candidate;
      }
    if (!contract)
      return std::nullopt;

    auto reject = [&](llvm::StringRef message)
        -> std::optional<mlir::LogicalResult> {
      contract.emitError(message);
      return std::optional<mlir::LogicalResult>(mlir::failure());
    };
    for (mlir::Operation &operation : nBody.without_terminator()) {
      if (&operation == kLoop.getOperation() ||
          (store && &operation == store.getOperation()) ||
          mlir::isa<ConstantOp, FullOp, BlockAxisOp, PtrAddOp>(operation))
        continue;
      return reject("IME1 N-tile lowering does not absorb additional source operations");
    }
    for (mlir::Operation &operation : kBody.without_terminator()) {
      if (&operation == contract.getOperation() ||
          mlir::isa<ConstantOp, BlockAxisOp, BinaryOp, PtrAddOp, LoadOp, CastOp,
                    BitcastOp, ExpandDimsOp>(operation))
        continue;
      return reject("IME1 contraction lowering does not absorb additional source operations");
    }
    if (options.target.matrixExtension != "spacemit-ime1")
      return reject("affine i4/i8 contraction requires --matrix-extension=spacemit-ime1");
    if (options.target.vlenBits != 256)
      return reject("SpacemiT IME1 contraction requires an explicit 256-bit VLEN target fact");
    if (nLoop.getNumResults() != 0 || nBody.getNumArguments() != 1 || !store ||
        !isTrue(store.getWhere()) || kLoop.getNumResults() != 1 ||
        kLoop.getInitArgs().size() != 1 || kBody.getNumArguments() != 2 ||
        store.getValue() != kLoop.getResult(0))
      return reject("IME1 lowering requires one explicit N loop, K recurrence, and output store");
    auto nYield = mlir::cast<YieldOp>(nBody.getTerminator());
    auto kYield = mlir::cast<YieldOp>(kBody.getTerminator());
    if (nYield.getNumOperands() != 0 || kYield.getNumOperands() != 1 ||
        kYield.getOperand(0) != contract.getResult() ||
        contract.getInit() != kBody.getArgument(1))
      return reject("IME1 lowering requires the extension result to be the sole K-loop carry");
    auto init = kLoop.getInitArgs().front().getDefiningOp<FullOp>();
    if (!init || !isFloatConstant(init.getValue(), 0.0))
      return reject("IME1 lowering requires the explicit contraction accumulator to start at zero");
    if (integerConstant(nLoop.getLower()) != 0 ||
        integerConstant(nLoop.getStep()) != 16 ||
        integerConstant(kLoop.getLower()) != 0 ||
        integerConstant(kLoop.getStep()) != 1)
      return reject("IME1 lowering requires the source-declared N16 and K-block traversal");

    LoadOp activationLoad = contract.getActivation().getDefiningOp<LoadOp>();
    LoadOp activationScaleLoad =
        contract.getActivationScale().getDefiningOp<LoadOp>();
    LoadOp zeroPointLoad =
        contract.getWeightZeroPoint().getDefiningOp<LoadOp>();
    LoadOp packedCodeLoad = contract.getPackedWeight().getDefiningOp<LoadOp>();
    LoadOp scaleLow;
    LoadOp scaleHigh;
    if (!activationLoad || !activationScaleLoad || !zeroPointLoad ||
        !packedCodeLoad || !matchF16LELoad(contract.getWeightScale(), scaleLow,
                                          scaleHigh) ||
        !isTrue(activationLoad.getWhere()) ||
        !isTrue(activationScaleLoad.getWhere()) ||
        !isTrue(zeroPointLoad.getWhere()) || !isTrue(packedCodeLoad.getWhere()))
      return reject("IME1 lowering requires explicit activation, scale, zero-point, and packed-code loads");

    mlir::Value nCoordinate = nBody.getArgument(0);
    mlir::Value kCoordinate = kBody.getArgument(0);
    ForOp rowLoop = nLoop->getParentOfType<ForOp>();
    if (!rowLoop || nLoop->getBlock() != &rowLoop.getBody().front())
      return reject("IME1 lowering requires the source-declared row traversal to remain outside N/K");
    mlir::Value rowCoordinate = rowLoop.getBody().front().getArgument(0);

    auto activationLane = activationLoad.getPointer().getDefiningOp<PtrAddOp>();
    auto activationBlock =
        activationLane ? activationLane.getBase().getDefiningOp<PtrAddOp>()
                       : PtrAddOp{};
    mlir::Value activationBlockIndex;
    if (!activationLane || !activationBlock ||
        !matchBlockAxis(activationLane.getOffset(), 32) ||
        !matchBinaryConstant(activationBlock.getOffset(), "mul", 32,
                             activationBlockIndex) ||
        activationBlockIndex != kCoordinate)
      return reject("IME1 lowering requires explicit contiguous K32 activation-code blocks");
    mlir::Value codeRow = activationBlock.getBase();

    auto activationScalePointer =
        activationScaleLoad.getPointer().getDefiningOp<PtrAddOp>();
    if (!activationScalePointer ||
        activationScalePointer.getOffset() != kCoordinate)
      return reject("IME1 lowering requires one explicit f32 activation scale per K32 block");
    mlir::Value scaleRow = activationScalePointer.getBase();

    auto outputLane = store.getPointer().getDefiningOp<PtrAddOp>();
    auto outputTile = outputLane ? outputLane.getBase().getDefiningOp<PtrAddOp>()
                                 : PtrAddOp{};
    if (!outputLane || !outputTile || !matchBlockAxis(outputLane.getOffset(), 16) ||
        outputTile.getOffset() != nCoordinate)
      return reject("IME1 lowering requires an explicit contiguous N16 output tile");
    mlir::Value outputRow = outputTile.getBase();

    auto zeroPointLane = zeroPointLoad.getPointer().getDefiningOp<PtrAddOp>();
    auto zeroPointBase =
        zeroPointLane ? zeroPointLane.getBase().getDefiningOp<PtrAddOp>()
                      : PtrAddOp{};
    if (!zeroPointLane || !zeroPointBase ||
        !matchBlockAxis(zeroPointLane.getOffset(), 16) ||
        integerConstant(zeroPointBase.getOffset()) != 32)
      return reject("IME1 lowering requires the explicit sixteen-byte zero-point field");
    mlir::Value columnAxis = zeroPointLane.getOffset();
    mlir::Value packedBase = zeroPointBase.getBase();

    auto scaleLane = scaleLow.getPointer().getDefiningOp<PtrAddOp>();
    mlir::Value scaleColumn;
    if (!scaleLane || scaleLane.getBase() != packedBase ||
        !matchBinaryConstant(scaleLane.getOffset(), "mul", 2, scaleColumn) ||
        scaleColumn != columnAxis || scalarPointerBase(scaleHigh.getPointer()) !=
                                          packedBase)
      return reject("IME1 lowering requires the explicit little-endian fp16 scale field");

    auto withinAdd = packedCodeLoad.getPointer().getDefiningOp<PtrAddOp>();
    auto within = withinAdd ? withinAdd.getOffset().getDefiningOp<BinaryOp>()
                            : BinaryOp{};
    auto columnAdd = withinAdd ? withinAdd.getBase().getDefiningOp<PtrAddOp>()
                               : PtrAddOp{};
    mlir::Value expandedColumn;
    auto halfAdd = columnAdd ? columnAdd.getBase().getDefiningOp<PtrAddOp>()
                             : PtrAddOp{};
    mlir::Value halfIndex;
    auto payloadBase = halfAdd ? halfAdd.getBase().getDefiningOp<PtrAddOp>()
                               : PtrAddOp{};
    if (!withinAdd || !within || within.getKind() != "mod" ||
        integerConstant(within.getRhs()) != 8 || !columnAdd ||
        !matchBinaryConstant(columnAdd.getOffset(), "mul", 8, expandedColumn) ||
        !matchExpandedAxis(expandedColumn, columnAxis, 1) || !halfAdd ||
        !matchBinaryConstant(halfAdd.getOffset(), "mul", 128, halfIndex) ||
        !payloadBase || payloadBase.getBase() != packedBase ||
        integerConstant(payloadBase.getOffset()) != 48)
      return reject("IME1 lowering requires the explicit two-half packed-i4 payload layout");
    auto half = halfIndex.getDefiningOp<BinaryOp>();
    auto expandedByte = within.getLhs().getDefiningOp<ExpandDimsOp>();
    auto expandedHalfByte =
        half ? half.getLhs().getDefiningOp<ExpandDimsOp>() : ExpandDimsOp{};
    if (!half || half.getKind() != "div" || integerConstant(half.getRhs()) != 8 ||
        !expandedByte || expandedByte.getAxis() != 0 ||
        !expandedHalfByte || expandedHalfByte.getAxis() != 0 ||
        expandedHalfByte.getInput() != expandedByte.getInput() ||
        !matchBlockAxis(expandedByte.getInput(), 16))
      return reject("IME1 lowering requires the source-declared packed-byte axis");

    auto packedPointer = packedBase.getDefiningOp<PtrAddOp>();
    mlir::Value linearBlock;
    if (!packedPointer ||
        !matchBinaryConstant(packedPointer.getOffset(), "mul", 304,
                             linearBlock))
      return reject("IME1 lowering requires the explicit 304-byte persistent packed block");
    auto blockAdd = linearBlock.getDefiningOp<BinaryOp>();
    mlir::Value groupStride;
    if (!blockAdd || blockAdd.getKind() != "add")
      return reject("IME1 lowering requires the explicit packed-block index relation");
    if (blockAdd.getLhs() == kCoordinate)
      groupStride = blockAdd.getRhs();
    else if (blockAdd.getRhs() == kCoordinate)
      groupStride = blockAdd.getLhs();
    else
      return reject("IME1 packed-block index must include the explicit K block");
    auto groupMultiply = groupStride.getDefiningOp<BinaryOp>();
    mlir::Value group;
    if (!groupMultiply || groupMultiply.getKind() != "mul")
      return reject("IME1 packed-block index must include the explicit N-group stride");
    if (groupMultiply.getLhs() == kLoop.getUpper())
      group = groupMultiply.getRhs();
    else if (groupMultiply.getRhs() == kLoop.getUpper())
      group = groupMultiply.getLhs();
    else
      return reject("IME1 packed-block N stride must use the explicit K-block count");
    auto groupDivide = group.getDefiningOp<BinaryOp>();
    if (!groupDivide || groupDivide.getKind() != "div" ||
        groupDivide.getLhs() != nCoordinate ||
        integerConstant(groupDivide.getRhs()) != 16)
      return reject("IME1 packed-block index must use the explicit N16 group");

    mlir::Value packedRoot = packedPointer.getBase();
    if (!dependsOn(codeRow, rowCoordinate) || !dependsOn(scaleRow, rowCoordinate) ||
        !dependsOn(outputRow, rowCoordinate) ||
        dependsOn(packedRoot, rowCoordinate) ||
        dependsOn(codeRow, nCoordinate) || dependsOn(scaleRow, nCoordinate) ||
        dependsOn(outputRow, kCoordinate))
      return reject("IME1 lowering cannot change the source-declared row/N/K ownership");

    CValue code = require(codeRow);
    CValue scales = require(scaleRow);
    CValue weights = require(packedRoot);
    CValue outputValue = require(outputRow);
    std::string columns = expression(nLoop.getUpper());
    std::string blocks = expression(kLoop.getUpper());
    if (code.kind != CValueKind::Pointer || scales.kind != CValueKind::Pointer ||
        weights.kind != CValueKind::Pointer ||
        outputValue.kind != CValueKind::Pointer || code.spelling.empty() ||
        scales.spelling.empty() || weights.spelling.empty() ||
        outputValue.spelling.empty() || columns.empty() || blocks.empty())
      return reject("IME1 lowering could not materialize the explicit source operands");

    std::string nTile = fresh("ime_n");
    std::string nCount = fresh("ime_n_count");
    line("for (size_t " + nTile + " = 0; " + nTile + " < " + columns +
         "; " + nTile + " += 16) {");
    ++indent;
    line("const size_t " + nCount + " = (" + columns + " - " + nTile +
         ") < 16 ? (" + columns + " - " + nTile + ") : 16;");
    line("__weft_ime1_affine_i4_i8_n16(" + scales.spelling + ", " +
         code.spelling + ", " + weights.spelling + " + (" + nTile +
         " / 16) * " + blocks + " * 304, " + outputValue.spelling + " + " +
         nTile + ", " + nCount + ", " + blocks + ");");
    --indent;
    line("}");
    return std::optional<mlir::LogicalResult>(mlir::success());
  }

  std::optional<mlir::LogicalResult> tryEmitF16GemmNTiles(ForOp nLoop) {
    if (nLoop.getNumResults() != 0)
      return std::nullopt;
    ForOp mLoop = nLoop->getParentOfType<ForOp>();
    if (!mLoop || mLoop.getBody().empty() || nLoop->getBlock() != &mLoop.getBody().front())
      return std::nullopt;

    mlir::Block &nBody = nLoop.getBody().front();
    ForOp kLoop;
    StoreOp store;
    for (mlir::Operation &operation : nBody.without_terminator()) {
      if (auto candidate = mlir::dyn_cast<ForOp>(operation)) {
        if (kLoop)
          return std::nullopt;
        kLoop = candidate;
      }
      if (auto candidate = mlir::dyn_cast<StoreOp>(operation)) {
        if (store)
          return std::nullopt;
        store = candidate;
      }
    }
    if (!kLoop || !store || kLoop.getNumResults() != 1 ||
        store.getValue() != kLoop.getResult(0) ||
        kLoop.getOperands().size() != 4)
      return std::nullopt;

    mlir::Block &kBody = kLoop.getBody().front();
    ContractOp contract;
    for (mlir::Operation &operation : kBody.without_terminator())
      if (auto candidate = mlir::dyn_cast<ContractOp>(operation)) {
        if (contract)
          return std::nullopt;
        contract = candidate;
      }
    if (!contract || contract.getResult() !=
                         mlir::cast<YieldOp>(kBody.getTerminator()).getOperand(0) ||
        contract.getInit() != kBody.getArgument(1) ||
        contract.getLhsAxes().size() != 1 ||
        contract.getRhsAxes().size() != 1 ||
        contract.getLhsAxes().front() != 1 ||
        contract.getRhsAxes().front() != 0 || contract.getOrder() != "relaxed" ||
        contract.getMath() != "native" || !contract.getAccDtype().isF32() ||
        !contract.getOutDtype().isF32() || !isTrue(contract.getWhereLhs()) ||
        !isTrue(contract.getWhereRhs()))
      return std::nullopt;

    LoadOp lhsLoad = contract.getLhs().getDefiningOp<LoadOp>();
    LoadOp rhsLoad = contract.getRhs().getDefiningOp<LoadOp>();
    auto init = kLoop.getOperand(3).getDefiningOp<FullOp>();
    if (!lhsLoad || !rhsLoad || !init ||
        !elementType(lhsLoad.getResult().getType()).isF16() ||
        !elementType(rhsLoad.getResult().getType()).isF16() ||
        !isFloatConstant(init.getValue(), 0.0))
      return std::nullopt;

    mlir::Value lhsRoot = pointerRoot(lhsLoad.getPointer());
    mlir::Value rhsRoot = pointerRoot(rhsLoad.getPointer());
    mlir::Value outputRoot = pointerRoot(store.getPointer());
    if (!lhsRoot || !rhsRoot || !outputRoot)
      return std::nullopt;
    auto lhsPointer = mlir::dyn_cast<PtrType>(lhsRoot.getType());
    auto rhsPointer = mlir::dyn_cast<PtrType>(rhsRoot.getType());
    auto outputPointer = mlir::dyn_cast<PtrType>(outputRoot.getType());
    mlir::Value lhsStride = blockStride(lhsLoad.getPointer());
    mlir::Value rhsStride = blockStride(rhsLoad.getPointer());
    mlir::Value outputStride = blockStride(store.getPointer());
    if (!lhsPointer || !rhsPointer || !outputPointer ||
        !lhsPointer.getElementType().isF16() ||
        !rhsPointer.getElementType().isF16() ||
        !outputPointer.getElementType().isF32() || !lhsStride || !rhsStride ||
        !outputStride)
      return std::nullopt;

    mlir::Value mCoordinate = mLoop.getBody().front().getArgument(0);
    mlir::Value nCoordinate = nBody.getArgument(0);
    mlir::Value kCoordinate = kBody.getArgument(0);
    if (!dependsOn(lhsLoad.getPointer(), mCoordinate) ||
        !dependsOn(lhsLoad.getPointer(), kCoordinate) ||
        dependsOn(lhsLoad.getPointer(), nCoordinate) ||
        !dependsOn(rhsLoad.getPointer(), kCoordinate) ||
        !dependsOn(rhsLoad.getPointer(), nCoordinate) ||
        dependsOn(rhsLoad.getPointer(), mCoordinate) ||
        !dependsOn(store.getPointer(), mCoordinate) ||
        !dependsOn(store.getPointer(), nCoordinate) ||
        dependsOn(store.getPointer(), kCoordinate))
      return std::nullopt;

    std::string mStep = expression(mLoop.getStep());
    std::string nStep = expression(nLoop.getStep());
    std::string nLower = expression(nLoop.getLower());
    std::string nUpper = expression(nLoop.getUpper());
    std::string kLower = expression(kLoop.getLower());
    std::string kUpper = expression(kLoop.getUpper());
    CValue mValue = require(mCoordinate);
    CValue mUpper = require(mLoop.getUpper());
    CValue lhs = require(lhsRoot);
    CValue rhs = require(rhsRoot);
    CValue outputValue = require(outputRoot);
    CValue lhsStrideValue = require(lhsStride);
    CValue rhsStrideValue = require(rhsStride);
    CValue outputStrideValue = require(outputStride);
    if (mStep != "4" || nStep != "8" || nLower.empty() || nUpper.empty() ||
        kLower.empty() || kUpper.empty() || mValue.spelling.empty() ||
        mUpper.spelling.empty() || lhs.spelling.empty() || rhs.spelling.empty() ||
        outputValue.spelling.empty() || lhsStrideValue.spelling.empty() ||
        rhsStrideValue.spelling.empty() || outputStrideValue.spelling.empty())
      return std::nullopt;

    std::string nTile = fresh("gemm_n");
    std::string vl = fresh("gemm_vl");
    std::string rows = fresh("gemm_rows");
    line("const size_t " + vl + " = __riscv_vsetvl_e16m1(8);");
    line("const size_t " + rows + " = (" + mUpper.spelling + " - " +
         mValue.spelling + ") < 4 ? (" + mUpper.spelling + " - " +
         mValue.spelling + ") : 4;");
    line("for (size_t " + nTile + " = " + nLower + "; " + nTile + " < " +
         nUpper + "; ++" + nTile + ") {");
    ++indent;

    auto emitRows = [&](unsigned rowCount, bool first) {
      line(std::string(first ? "if" : "else if") + " (" + rows + " == " +
           std::to_string(rowCount) + ") {");
      ++indent;
      llvm::SmallVector<std::string> accumulators;
      for (unsigned row = 0; row < rowCount; ++row) {
        std::string accumulator = fresh("gemm_acc");
        line("vfloat32m2_t " + accumulator +
             " = __riscv_vfmv_v_f_f32m2(0.0f, " + vl + ");");
        accumulators.push_back(std::move(accumulator));
      }
      std::string inner = fresh("gemm_k");
      std::string fullEnd = fresh("gemm_k_end");
      line("const size_t " + fullEnd + " = " + kUpper + " - ((" + kUpper +
           " - " + kLower + ") % " + vl + ");");
      line("for (size_t " + inner + " = " + kLower + "; " + inner + " < " +
           fullEnd + "; " + inner + " += " + vl + ") {");
      ++indent;
      std::string rhsVector = fresh("gemm_b");
      line("vfloat16m1_t " + rhsVector + " = __riscv_vle16_v_f16m1(" +
           rhs.spelling + " + " + nTile + " * " + rhsStrideValue.spelling +
           " + " + inner + ", " + vl + ");");
      for (unsigned row = 0; row < rowCount; ++row) {
        std::string lhsVector = fresh("gemm_a");
        line("vfloat16m1_t " + lhsVector + " = __riscv_vle16_v_f16m1(" +
             lhs.spelling + " + (" +
             mValue.spelling + " + " + std::to_string(row) + ") * " +
             lhsStrideValue.spelling + " + " + inner + ", " + vl + ");");
        line(accumulators[row] + " = __riscv_vfwmacc_vv_f32m2(" +
             accumulators[row] + ", " + lhsVector + ", " + rhsVector + ", " +
             vl + ");");
      }
      --indent;
      line("}");
      for (unsigned row = 0; row < rowCount; ++row) {
        std::string seed = fresh("gemm_seed");
        std::string partial = fresh("gemm_partial");
        std::string scalar = fresh("gemm_sum");
        line("vfloat32m1_t " + seed +
             " = __riscv_vfmv_v_f_f32m1(0.0f, 1);");
        line("vfloat32m1_t " + partial +
             " = __riscv_vfredusum_vs_f32m2_f32m1(" + accumulators[row] +
             ", " + seed + ", " + vl + ");");
        line("float " + scalar + " = __riscv_vfmv_f_s_f32m1_f32(" + partial +
             ");");
        std::string tail = fresh("gemm_tail");
        line("for (size_t " + tail + " = " + fullEnd + "; " + tail + " < " +
             kUpper + "; ++" + tail + ")");
        ++indent;
        line(scalar + " += (float)*(" + lhs.spelling + " + (" +
             mValue.spelling + " + " + std::to_string(row) + ") * " +
             lhsStrideValue.spelling + " + " + tail + ") * (float)*(" +
             rhs.spelling + " + " + nTile + " * " +
             rhsStrideValue.spelling + " + " + tail + ");");
        --indent;
        line("*(" + outputValue.spelling + " + (" + mValue.spelling + " + " +
             std::to_string(row) + ") * " + outputStrideValue.spelling + " + " +
             nTile + ") = " + scalar + ";");
      }
      --indent;
      line("}");
    };
    emitRows(4, true);
    emitRows(3, false);
    emitRows(2, false);
    emitRows(1, false);
    --indent;
    line("}");
    return mlir::success();
  }

  BlockValue lookupBlockValue(
      mlir::Value value,
      const llvm::DenseMap<mlir::Value, BlockValue> &blockValues) const {
    auto found = blockValues.find(value);
    if (found != blockValues.end())
      return found->second;
    CValue scalar = require(value);
    BlockValue result;
    result.type = value.getType();
    result.spelling = scalar.spelling;
    if (scalar.kind == CValueKind::Pointer) {
      result.kind = BlockValueKind::Pointer;
      result.pointerBase = scalar.spelling;
    }
    return result;
  }

  std::optional<int64_t> integerConstant(mlir::Value value) const {
    auto constant = value.getDefiningOp<ConstantOp>();
    auto integer = constant
                       ? mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue())
                       : mlir::IntegerAttr{};
    if (!integer)
      return std::nullopt;
    return integer.getInt();
  }

  void markDiscardedBlockTree(mlir::Value value) {
    if (!containsBlockType(value.getType()))
      return;
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || !loweredBlockOps.insert(definition).second)
      return;
    for (mlir::Value operand : definition->getOperands())
      markDiscardedBlockTree(operand);
  }

  mlir::LogicalResult collectBlockClosure(
      mlir::Value value, llvm::DenseSet<mlir::Operation *> &closure,
      llvm::SmallVectorImpl<BlockAxisOp> &axes, mlir::Operation *owner) {
    if (!containsBlockType(value.getType()))
      return mlir::success();
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition)
      return owner->emitError(
          "block operation cannot capture a block argument from another region");
    if (!closure.insert(definition).second)
      return mlir::success();
    if (definition->getBlock() != owner->getBlock())
      return owner->emitError(
          "block producer closure crosses a region boundary");
    if (auto get = mlir::dyn_cast<TupleGetOp>(definition)) {
      auto tuple = get.getInput().getDefiningOp<TupleOp>();
      if (!tuple)
        return get.emitError(
            "block tuple projection requires a local tuple producer");
      closure.insert(tuple.getOperation());
      for (auto [index, operand] : llvm::enumerate(tuple.getOperands())) {
        if (static_cast<int64_t>(index) == get.getIndex()) {
          if (mlir::failed(collectBlockClosure(operand, closure, axes, owner)))
            return mlir::failure();
          continue;
        }
        bool fieldIsLive = llvm::any_of(
            tuple.getResult().getUsers(), [&](mlir::Operation *user) {
              auto otherGet = mlir::dyn_cast<TupleGetOp>(user);
              return otherGet && otherGet.getIndex() ==
                                     static_cast<int64_t>(index) &&
                     !otherGet.getResult().use_empty();
            });
        if (!fieldIsLive)
          markDiscardedBlockTree(operand);
      }
      return mlir::success();
    }
    if (auto axis = mlir::dyn_cast<BlockAxisOp>(definition))
      axes.push_back(axis);
    for (mlir::Value operand : definition->getOperands())
      if (containsBlockType(operand.getType()) &&
          mlir::failed(collectBlockClosure(operand, closure, axes, owner)))
        return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult emitBlockPtrAdd(
      PtrAddOp op, llvm::DenseMap<mlir::Value, BlockValue> &blockValues,
      llvm::StringRef vl) {
    BlockValue base = lookupBlockValue(op.getBase(), blockValues);
    BlockValue offset = lookupBlockValue(op.getOffset(), blockValues);
    if (base.kind != BlockValueKind::Pointer || base.pointerBase.empty())
      return op.emitError("block pointer addition has no scalar pointer base");
    BlockValue result;
    result.type = op.getResult().getType();
    result.kind = BlockValueKind::Pointer;
    result.pointerBase = base.pointerBase;
    result.pointerIndex = base.pointerIndex;
    result.contiguousIndex = base.contiguousIndex;
    if (offset.kind == BlockValueKind::Scalar) {
      if (offset.spelling.empty())
        return op.emitError("block pointer has an unavailable scalar offset");
      result.pointerBase = "(" + result.pointerBase + " + " + offset.spelling + ")";
    } else if (offset.kind == BlockValueKind::Index) {
      if (!result.pointerIndex.empty())
        return op.emitError("nested vector pointer offsets are not implemented");
      result.pointerIndex = offset.spelling;
      result.contiguousIndex = offset.contiguousIndex;
    } else {
      return op.emitError("block pointer offset must be scalar or logical index");
    }
    (void)vl;
    blockValues[op.getResult()] = std::move(result);
    return mlir::success();
  }

  mlir::LogicalResult emitBlockIndexBinary(
      BinaryOp op, BlockValue lhs, BlockValue rhs,
      llvm::DenseMap<mlir::Value, BlockValue> &blockValues,
      llvm::StringRef vl) {
    bool lhsVector = lhs.kind == BlockValueKind::Index;
    bool rhsVector = rhs.kind == BlockValueKind::Index;
    if (!lhsVector && rhsVector &&
        (op.getKind() == "add" || op.getKind() == "mul" ||
         op.getKind() == "and" || op.getKind() == "or")) {
      std::swap(lhs, rhs);
      std::swap(lhsVector, rhsVector);
    }
    if (!lhsVector)
      return op.emitError("logical-index binary requires a vector left operand");
    std::string intrinsic;
    std::string rhsSpelling = rhs.spelling;
    if (rhsVector) {
      intrinsic = llvm::StringSwitch<std::string>(op.getKind())
                      .Case("add", "__riscv_vadd_vv_u16m2")
                      .Case("sub", "__riscv_vsub_vv_u16m2")
                      .Case("mul", "__riscv_vmul_vv_u16m2")
                      .Case("and", "__riscv_vand_vv_u16m2")
                      .Case("or", "__riscv_vor_vv_u16m2")
                      .Default("");
    } else {
      if (rhs.spelling.empty())
        return op.emitError("logical-index binary has an unavailable scalar");
      std::optional<int64_t> constant = integerConstant(op.getRhs());
      if ((op.getKind() == "div" || op.getKind() == "mod") && constant &&
          *constant > 0 && ((*constant & (*constant - 1)) == 0)) {
        unsigned shift = 0;
        for (uint64_t value = static_cast<uint64_t>(*constant); value > 1;
             value >>= 1)
          ++shift;
        if (op.getKind() == "div") {
          intrinsic = "__riscv_vsrl_vx_u16m2";
          rhsSpelling = std::to_string(shift);
        } else {
          intrinsic = "__riscv_vand_vx_u16m2";
          rhsSpelling = std::to_string(*constant - 1);
        }
      } else {
        intrinsic = llvm::StringSwitch<std::string>(op.getKind())
                        .Case("add", "__riscv_vadd_vx_u16m2")
                        .Case("sub", "__riscv_vsub_vx_u16m2")
                        .Case("mul", "__riscv_vmul_vx_u16m2")
                        .Case("div", "__riscv_vdivu_vx_u16m2")
                        .Case("mod", "__riscv_vremu_vx_u16m2")
                        .Case("and", "__riscv_vand_vx_u16m2")
                        .Case("or", "__riscv_vor_vx_u16m2")
                        .Case("shl", "__riscv_vsll_vx_u16m2")
                        .Case("shr", "__riscv_vsrl_vx_u16m2")
                        .Default("");
      }
    }
    if (intrinsic.empty())
      return op.emitError("RVV logical-index binary kind is unsupported");
    std::string name = fresh("block_index");
    line("vuint16m2_t " + name + " = " + intrinsic + "(" + lhs.spelling +
         ", " + rhsSpelling + ", " + vl.str() + ");");
    BlockValue result{op.getResult().getType(), BlockValueKind::Index, name};
    if (!rhsVector && op.getKind() == "add" &&
        !lhs.contiguousIndex.empty())
      result.contiguousIndex =
          "(" + lhs.contiguousIndex + " + " + rhsSpelling + ")";
    blockValues[op.getResult()] = std::move(result);
    return mlir::success();
  }

  mlir::LogicalResult materializeI32(BlockValue &value,
                                     mlir::Operation *owner,
                                     llvm::StringRef vl) {
    if (value.kind != BlockValueKind::I32 || !value.spelling.empty())
      return mlir::success();
    std::string name = fresh("block_i32");
    if (value.narrowKind == BlockValueKind::U8) {
      std::string extended = fresh("block_u32");
      line("vuint32m4_t " + extended + " = __riscv_vzext_vf4_u32m4(" +
           value.narrowSpelling + ", " + vl.str() + ");");
      line("vint32m4_t " + name +
           " = __riscv_vreinterpret_v_u32m4_i32m4(" + extended + ");");
    } else if (value.narrowKind == BlockValueKind::I8) {
      line("vint32m4_t " + name + " = __riscv_vsext_vf4_i32m4(" +
           value.narrowSpelling + ", " + vl.str() + ");");
    } else {
      return owner->emitError(
          "deferred i32 block value has no legal widening source");
    }
    value.spelling = name;
    return mlir::success();
  }

  mlir::LogicalResult emitBlockIntegerBinary(
      BinaryOp op, BlockValue lhs, BlockValue rhs, BlockValueKind resultKind,
      llvm::DenseMap<mlir::Value, BlockValue> &blockValues,
      llvm::StringRef vl) {
    bool feedsOnlyF32Casts = !op.getResult().use_empty() && llvm::all_of(
        op.getResult().getUsers(), [](mlir::Operation *user) {
          auto cast = mlir::dyn_cast<CastOp>(user);
          return cast && elementType(cast.getResult().getType()).isF32();
        });
    if (microBlockVectors && resultKind == BlockValueKind::U8 &&
        lhs.kind == BlockValueKind::U8 && !lhs.spelling.empty() &&
        rhs.kind == BlockValueKind::Scalar && !rhs.spelling.empty() &&
        (op.getKind() == "and" || op.getKind() == "shr") &&
        feedsOnlyF32Casts) {
      BlockValue result;
      result.type = op.getResult().getType();
      result.kind = BlockValueKind::U8;
      result.packedSource = op.getLhs();
      result.packedTransform = op.getKind().str();
      result.packedTransformOperand = rhs.spelling;
      blockValues[op.getResult()] = std::move(result);
      return mlir::success();
    }
    bool lhsVector = lhs.kind == resultKind;
    bool rhsVector = rhs.kind == resultKind;
    if (!lhsVector && rhsVector &&
        (op.getKind() == "add" || op.getKind() == "mul" ||
         op.getKind() == "and" || op.getKind() == "or" ||
         op.getKind() == "xor")) {
      std::swap(lhs, rhs);
      std::swap(lhsVector, rhsVector);
    }
    if (!lhsVector)
      return op.emitError("integer block binary requires a vector operand");

    auto isNarrowI8 = [](const BlockValue &value) {
      return value.kind == BlockValueKind::I32 &&
             value.narrowKind == BlockValueKind::I8 &&
             !value.narrowSpelling.empty();
    };
    auto isSmallU8 = [](const BlockValue &value) {
      return value.kind == BlockValueKind::I32 &&
             value.narrowKind == BlockValueKind::U8 &&
             !value.narrowSpelling.empty() && value.unsignedMaximum &&
             *value.unsignedMaximum <= 15;
    };
    if (resultKind == BlockValueKind::I32 && rhsVector &&
        op.getKind() == "mul" &&
        ((isSmallU8(lhs) && isNarrowI8(rhs)) ||
         (isNarrowI8(lhs) && isSmallU8(rhs)))) {
      BlockValue small = isSmallU8(lhs) ? lhs : rhs;
      BlockValue signedValue = isNarrowI8(lhs) ? lhs : rhs;
      std::string signedSmall = fresh("block_i8");
      std::string name = fresh("block_i16_product");
      line("vint8m1_t " + signedSmall +
           " = __riscv_vreinterpret_v_u8m1_i8m1(" + small.narrowSpelling +
           ");");
      line("vint16m2_t " + name + " = __riscv_vwmul_vv_i16m2(" +
           signedSmall + ", " + signedValue.narrowSpelling + ", " + vl.str() +
           ");");
      blockValues[op.getResult()] =
          BlockValue{op.getResult().getType(), BlockValueKind::I16, name};
      return mlir::success();
    }
    if (resultKind == BlockValueKind::I32) {
      if (mlir::failed(materializeI32(lhs, op.getOperation(), vl)) ||
          mlir::failed(materializeI32(rhs, op.getOperation(), vl)))
        return mlir::failure();
    }

    std::string suffix;
    std::string cType;
    if (resultKind == BlockValueKind::U8) {
      suffix = microBlockVectors ? "u8mf4" : "u8m1";
      cType = microBlockVectors ? "vuint8mf4_t" : "vuint8m1_t";
    } else if (resultKind == BlockValueKind::U16) {
      suffix = "u16m2";
      cType = "vuint16m2_t";
    } else if (resultKind == BlockValueKind::I32) {
      suffix = "i32m4";
      cType = "vint32m4_t";
    } else {
      return op.emitError("integer block type has no physical RVV mapping");
    }
    llvm::StringRef form = rhsVector ? "vv" : "vx";
    std::string stem = llvm::StringSwitch<std::string>(op.getKind())
                           .Case("add", "vadd")
                           .Case("sub", "vsub")
                           .Case("mul", "vmul")
                           .Case("and", "vand")
                           .Case("or", "vor")
                           .Case("xor", "vxor")
                           .Case("shl", "vsll")
                           .Case("shr", resultKind == BlockValueKind::I32
                                            ? "vsra"
                                            : "vsrl")
                           .Default("");
    if (stem.empty() || (!rhsVector && rhs.spelling.empty()))
      return op.emitError("RVV integer block binary kind is unsupported");
    std::string name = fresh("block_v");
    line(cType + " " + name + " = __riscv_" + stem + "_" + form.str() +
         "_" + suffix + "(" + lhs.spelling + ", " + rhs.spelling + ", " +
         vl.str() + ");");
    BlockValue result{op.getResult().getType(), resultKind, name};
    if (resultKind == BlockValueKind::U8)
      result.vectorShape = microBlockVectors ? RVVBlockVectorShape::E8MF4
                                             : RVVBlockVectorShape::E8M1;
    else if (resultKind == BlockValueKind::U16)
      result.vectorShape = RVVBlockVectorShape::E16M2;
    else if (resultKind == BlockValueKind::I32)
      result.vectorShape = RVVBlockVectorShape::E32M4;
    if (resultKind == BlockValueKind::U8 && op.getKind() == "and") {
      std::optional<int64_t> mask = integerConstant(op.getRhs());
      if (mask && *mask >= 0)
        result.unsignedMaximum = static_cast<uint64_t>(*mask);
    } else if (resultKind == BlockValueKind::U8 && op.getKind() == "shr") {
      std::optional<int64_t> shift = integerConstant(op.getRhs());
      if (shift && *shift >= 0 && *shift < 8)
        result.unsignedMaximum =
            lhs.unsignedMaximum.value_or(255) >> static_cast<unsigned>(*shift);
    }
    blockValues[op.getResult()] = std::move(result);
    return mlir::success();
  }

  mlir::LogicalResult emitBlockFloatBinary(
      BinaryOp op, BlockValue lhs, BlockValue rhs,
      llvm::DenseMap<mlir::Value, BlockValue> &blockValues,
      llvm::StringRef vl) {
    bool lhsVector = lhs.kind == BlockValueKind::F32;
    bool rhsVector = rhs.kind == BlockValueKind::F32;
    std::string first = lhs.spelling;
    std::string second = rhs.spelling;
    std::string stem;
    std::string form;
    if (lhsVector && rhsVector) {
      form = "vv";
      stem = llvm::StringSwitch<std::string>(op.getKind())
                 .Case("add", "vfadd")
                 .Case("sub", "vfsub")
                 .Case("mul", "vfmul")
                 .Case("div", "vfdiv")
                 .Case("max", "vfmax")
                 .Case("min", "vfmin")
                 .Default("");
    } else if (lhsVector) {
      form = "vf";
      stem = llvm::StringSwitch<std::string>(op.getKind())
                 .Case("add", "vfadd")
                 .Case("sub", "vfsub")
                 .Case("mul", "vfmul")
                 .Case("div", "vfdiv")
                 .Case("max", "vfmax")
                 .Case("min", "vfmin")
                 .Default("");
    } else if (rhsVector) {
      form = "vf";
      stem = llvm::StringSwitch<std::string>(op.getKind())
                 .Case("add", "vfadd")
                 .Case("sub", "vfrsub")
                 .Case("mul", "vfmul")
                 .Case("div", "vfrdiv")
                 .Case("max", "vfmax")
                 .Case("min", "vfmin")
                 .Default("");
      std::swap(first, second);
    } else {
      return op.emitError("f32 block binary requires a vector operand");
    }
    if (stem.empty() || first.empty() || second.empty())
      return op.emitError("RVV f32 block binary kind is unsupported");
    RVVBlockVectorShape shape = lhsVector ? lhs.vectorShape : rhs.vectorShape;
    if (lhsVector && rhsVector && lhs.vectorShape != rhs.vectorShape)
      return op.emitError("f32 block binary has incompatible physical vectors");
    std::string suffix;
    std::string cType;
    if (shape == RVVBlockVectorShape::E32M1) {
      suffix = "f32m1";
      cType = "vfloat32m1_t";
    } else if (shape == RVVBlockVectorShape::E32M4) {
      suffix = "f32m4";
      cType = "vfloat32m4_t";
    } else {
      return op.emitError("f32 block binary has no physical RVV shape");
    }
    std::string intrinsic =
        "__riscv_" + stem + "_" + form + "_" + suffix;
    std::string name = fresh("block_f32");
    line(cType + " " + name + " = " + intrinsic + "(" + first + ", " + second +
         ", " + vl.str() + ");");
    BlockValue result{op.getResult().getType(), BlockValueKind::F32, name};
    result.vectorShape = shape;
    blockValues[op.getResult()] = std::move(result);
    return mlir::success();
  }

  mlir::LogicalResult emitBlockBinary(
      BinaryOp op, llvm::DenseMap<mlir::Value, BlockValue> &blockValues,
      llvm::StringRef vl) {
    BlockValue lhs = lookupBlockValue(op.getLhs(), blockValues);
    BlockValue rhs = lookupBlockValue(op.getRhs(), blockValues);
    mlir::Type resultElement = elementType(op.getResult().getType());
    if (resultElement.isIndex())
      return emitBlockIndexBinary(op, lhs, rhs, blockValues, vl);
    if (resultElement.isUnsignedInteger(8))
      return emitBlockIntegerBinary(op, lhs, rhs, BlockValueKind::U8,
                                    blockValues, vl);
    if (resultElement.isUnsignedInteger(16))
      return emitBlockIntegerBinary(op, lhs, rhs, BlockValueKind::U16,
                                    blockValues, vl);
    if (resultElement.isSignedInteger(32))
      return emitBlockIntegerBinary(op, lhs, rhs, BlockValueKind::I32,
                                    blockValues, vl);
    if (resultElement.isF32())
      return emitBlockFloatBinary(op, lhs, rhs, blockValues, vl);
    return op.emitError("RVV block binary element type is unsupported");
  }

  mlir::LogicalResult emitBlockCompare(
      CompareOp op, llvm::DenseMap<mlir::Value, BlockValue> &blockValues,
      llvm::StringRef vl) {
    BlockValue lhs = lookupBlockValue(op.getLhs(), blockValues);
    BlockValue rhs = lookupBlockValue(op.getRhs(), blockValues);
    if (lhs.kind != BlockValueKind::Index ||
        rhs.kind != BlockValueKind::Scalar || rhs.spelling.empty())
      return op.emitError(
          "RVV block comparison currently requires index-vector vs scalar");
    std::string intrinsic =
        llvm::StringSwitch<std::string>(op.getPredicate())
            .Case("lt", "__riscv_vmsltu_vx_u16m2_b8")
            .Case("le", "__riscv_vmsleu_vx_u16m2_b8")
            .Case("gt", "__riscv_vmsgtu_vx_u16m2_b8")
            .Case("ge", "__riscv_vmsgeu_vx_u16m2_b8")
            .Case("eq", "__riscv_vmseq_vx_u16m2_b8")
            .Case("ne", "__riscv_vmsne_vx_u16m2_b8")
            .Default("");
    if (intrinsic.empty())
      return op.emitError("RVV block comparison predicate is unsupported");
    std::string name = fresh("block_mask");
    line("vbool8_t " + name + " = " + intrinsic + "(" + lhs.spelling + ", " +
         rhs.spelling + ", " + vl.str() + ");");
    blockValues[op.getResult()] =
        BlockValue{op.getResult().getType(), BlockValueKind::Mask, name};
    return mlir::success();
  }

  mlir::LogicalResult emitBlockCast(
      CastOp op, llvm::DenseMap<mlir::Value, BlockValue> &blockValues,
      llvm::StringRef vl) {
    BlockValue input = lookupBlockValue(op.getInput(), blockValues);
    mlir::Type target = elementType(op.getResult().getType());
    std::string name = fresh("block_cast");
    BlockValueKind kind;
    RVVBlockVectorShape shape = RVVBlockVectorShape::None;
    if (input.kind == BlockValueKind::Index && target.isUnsignedInteger(8)) {
      kind = BlockValueKind::U8;
      shape = RVVBlockVectorShape::E8M1;
      line("vuint8m1_t " + name + " = __riscv_vncvt_x_x_w_u8m1(" +
           input.spelling + ", " + vl.str() + ");");
    } else if (input.kind == BlockValueKind::U8 &&
               target.isUnsignedInteger(16)) {
      kind = BlockValueKind::U16;
      shape = RVVBlockVectorShape::E16M2;
      line("vuint16m2_t " + name + " = __riscv_vzext_vf2_u16m2(" +
           input.spelling + ", " + vl.str() + ");");
    } else if (input.kind == BlockValueKind::U8 &&
               target.isSignedInteger(32)) {
      kind = BlockValueKind::I32;
      name.clear();
    } else if (input.kind == BlockValueKind::I8 &&
               target.isSignedInteger(32)) {
      kind = BlockValueKind::I32;
      name.clear();
    } else if (input.kind == BlockValueKind::U16 &&
               target.isSignedInteger(32)) {
      kind = BlockValueKind::I32;
      shape = RVVBlockVectorShape::E32M4;
      std::string extended = fresh("block_u32");
      line("vuint32m4_t " + extended + " = __riscv_vzext_vf2_u32m4(" +
           input.spelling + ", " + vl.str() + ");");
      line("vint32m4_t " + name +
           " = __riscv_vreinterpret_v_u32m4_i32m4(" + extended + ");");
    } else if (input.kind == BlockValueKind::I16 &&
               target.isSignedInteger(32)) {
      kind = BlockValueKind::I32;
      shape = RVVBlockVectorShape::E32M4;
      line("vint32m4_t " + name + " = __riscv_vsext_vf2_i32m4(" +
           input.spelling + ", " + vl.str() + ");");
    } else if (input.kind == BlockValueKind::U8 && target.isF32()) {
      kind = BlockValueKind::F32;
      std::string extended;
      if (microBlockVectors) {
        shape = RVVBlockVectorShape::E32M1;
        std::string sourceSpelling = input.spelling;
        auto source = blockValues.find(input.packedSource);
        if (input.packedSource && !input.packedTransform.empty()) {
          if (source == blockValues.end() || source->second.spelling.empty())
            return op.emitError("packed decode cast has no u8 source vector");
          sourceSpelling = source->second.spelling;
        }
        std::string widened;
        if (source != blockValues.end() &&
            !source->second.widenedSpelling.empty()) {
          widened = source->second.widenedSpelling;
        } else {
          widened = fresh("block_packed_u32");
          line("vuint32m1_t " + widened +
               " = __riscv_vzext_vf4_u32m1(" + sourceSpelling + ", " +
               vl.str() + ");");
          if (source != blockValues.end())
            source->second.widenedSpelling = widened;
        }
        extended = widened;
        if (!input.packedTransform.empty()) {
          extended = fresh("block_code_u32");
          std::string stem =
              input.packedTransform == "and" ? "vand" : "vsrl";
          line("vuint32m1_t " + extended + " = __riscv_" + stem +
               "_vx_u32m1(" + widened + ", " +
               input.packedTransformOperand + ", " + vl.str() + ");");
        }
        line("vfloat32m1_t " + name + " = __riscv_vfcvt_f_xu_v_f32m1(" +
             extended + ", " + vl.str() + ");");
      } else {
        shape = RVVBlockVectorShape::E32M4;
        std::string u16Suffix = "u16m2";
        std::string u16Type = "vuint16m2_t";
        extended = fresh("block_u16");
        line(u16Type + " " + extended + " = __riscv_vzext_vf2_" +
             u16Suffix + "(" + input.spelling + ", " + vl.str() + ");");
        line("vfloat32m4_t " + name +
             " = __riscv_vfwcvt_f_xu_v_f32m4(" + extended + ", " + vl.str() +
             ");");
      }
    } else if (input.kind == BlockValueKind::I8 && target.isF32() &&
               input.vectorShape == RVVBlockVectorShape::E8M1) {
      kind = BlockValueKind::F32;
      shape = RVVBlockVectorShape::E32M4;
      std::string extended = fresh("block_i16");
      line("vint16m2_t " + extended + " = __riscv_vsext_vf2_i16m2(" +
           input.spelling + ", " + vl.str() + ");");
      line("vfloat32m4_t " + name +
           " = __riscv_vfwcvt_f_x_v_f32m4(" + extended + ", " + vl.str() +
           ");");
    } else {
      return op.emitError("RVV block cast pair is unsupported");
    }
    BlockValue result{op.getResult().getType(), kind, name};
    result.vectorShape = shape;
    if (kind == BlockValueKind::I32 && name.empty()) {
      result.narrowKind = input.kind;
      result.narrowSpelling = input.spelling;
      result.unsignedMaximum = input.unsignedMaximum;
    }
    blockValues[op.getResult()] = std::move(result);
    return mlir::success();
  }

  mlir::LogicalResult emitBlockBitcast(
      BitcastOp op, llvm::DenseMap<mlir::Value, BlockValue> &blockValues) {
    BlockValue input = lookupBlockValue(op.getInput(), blockValues);
    mlir::Type target = elementType(op.getResult().getType());
    std::string name = fresh("block_bits");
    BlockValueKind kind;
    RVVBlockVectorShape shape = RVVBlockVectorShape::None;
    if (input.kind == BlockValueKind::U8 && target.isSignedInteger(8)) {
      kind = BlockValueKind::I8;
      shape = input.vectorShape;
      if (shape == RVVBlockVectorShape::E8MF4)
        line("vint8mf4_t " + name +
             " = __riscv_vreinterpret_v_u8mf4_i8mf4(" + input.spelling +
             ");");
      else if (shape == RVVBlockVectorShape::E8M1)
        line("vint8m1_t " + name +
             " = __riscv_vreinterpret_v_u8m1_i8m1(" + input.spelling +
             ");");
      else
        return op.emitError("u8 block bitcast has no physical RVV shape");
    } else if (input.kind == BlockValueKind::U16 &&
               target.isSignedInteger(16)) {
      kind = BlockValueKind::I16;
      shape = RVVBlockVectorShape::E16M2;
      line("vint16m2_t " + name +
           " = __riscv_vreinterpret_v_u16m2_i16m2(" + input.spelling + ");");
    } else {
      return op.emitError("RVV block bitcast pair is unsupported");
    }
    BlockValue result{op.getResult().getType(), kind, name};
    result.vectorShape = shape;
    blockValues[op.getResult()] = std::move(result);
    return mlir::success();
  }

  mlir::LogicalResult emitBlockLoad(
      LoadOp op, llvm::DenseMap<mlir::Value, BlockValue> &blockValues,
      llvm::StringRef vl) {
    BlockValue pointer = lookupBlockValue(op.getPointer(), blockValues);
    BlockValue where = lookupBlockValue(op.getWhere(), blockValues);
    BlockValue other = lookupBlockValue(op.getOther(), blockValues);
    if (pointer.kind != BlockValueKind::Pointer || pointer.pointerBase.empty() ||
        (pointer.pointerIndex.empty() && pointer.contiguousIndex.empty()))
      return op.emitError("RVV block load has no vector address");
    if (!elementType(op.getResult().getType()).isUnsignedInteger(8))
      return op.emitError("RVV block load currently supports u8 elements");
    std::string name = fresh("block_load");
    std::string suffix = microBlockVectors ? "u8mf4" : "u8m1";
    std::string cType = microBlockVectors ? "vuint8mf4_t" : "vuint8m1_t";
    bool allActive = isTrue(op.getWhere());
    if (allActive && !pointer.contiguousIndex.empty()) {
      line(cType + " " + name + " = __riscv_vle8_v_" + suffix + "(" +
           pointer.pointerBase + " + " + pointer.contiguousIndex + ", " +
           vl.str() + ");");
    } else if (allActive) {
      line(cType + " " + name + " = __riscv_vluxei16_v_" + suffix + "(" +
           pointer.pointerBase + ", " + pointer.pointerIndex + ", " + vl.str() +
           ");");
    } else if (where.kind == BlockValueKind::Mask &&
               !pointer.pointerIndex.empty() && !other.spelling.empty()) {
      std::string maskedOff = fresh("block_other");
      line(cType + " " + maskedOff + " = __riscv_vmv_v_x_" + suffix + "(" +
           other.spelling + ", " + vl.str() + ");");
      line(cType + " " + name + " = __riscv_vluxei16_v_" + suffix + "_mu(" +
           where.spelling + ", " + maskedOff + ", " + pointer.pointerBase +
           ", " + pointer.pointerIndex + ", " + vl.str() + ");");
    } else {
      return op.emitError("RVV masked block load address is unsupported");
    }
    BlockValue result{op.getResult().getType(), BlockValueKind::U8, name};
    result.unsignedMaximum = 255;
    result.vectorShape = microBlockVectors ? RVVBlockVectorShape::E8MF4
                                           : RVVBlockVectorShape::E8M1;
    blockValues[op.getResult()] = std::move(result);
    return mlir::success();
  }

  mlir::LogicalResult decideBlockDecode(DecodeOp op,
                                        BlockDecodeDecision &decision) {
    auto codes = mlir::dyn_cast<BlockType>(op.getCodes().getType());
    auto table = mlir::dyn_cast<BlockType>(op.getTable().getType());
    auto result = mlir::dyn_cast<BlockType>(op.getResult().getType());
    if (!codes || !table || !result || codes.getShape().size() != 1 ||
        table.getShape().size() != 1 || result.getShape() != codes.getShape())
      return op.emitError(
          "RVV block decode requires rank-one codes, table, and result");
    if (codes.getShape().front() != 16 || table.getShape().front() != 16 ||
        !codes.getElementType().isUnsignedInteger(8) ||
        !table.getElementType().isSignedInteger(8) ||
        !result.getElementType().isSignedInteger(8) || !isTrue(op.getWhere()))
      return op.emitError(
          "RVV block decode requires 16 all-active u8 codes and an i8 table");
    decision = BlockDecodeDecision{};
    return mlir::success();
  }

  mlir::LogicalResult emitBlockDecode(
      DecodeOp op, const BlockDecodeDecision &decision,
      llvm::DenseMap<mlir::Value, BlockValue> &blockValues,
      llvm::StringRef vl) {
    BlockValue codes = lookupBlockValue(op.getCodes(), blockValues);
    BlockValue table = lookupBlockValue(op.getTable(), blockValues);
    if (decision.realization != BlockDecodeRealization::RVVI8TableGather ||
        decision.tableExtent != 16 ||
        decision.codeShape != RVVBlockVectorShape::E8M1 ||
        decision.resultShape != RVVBlockVectorShape::E8M1 ||
        codes.kind != BlockValueKind::U8 ||
        codes.vectorShape != decision.codeShape ||
        table.kind != BlockValueKind::I8 ||
        table.vectorShape != RVVBlockVectorShape::E8M1 ||
        codes.spelling.empty() || table.spelling.empty())
      return op.emitError(
          "RVV block decode operands do not match the selected realization");
    std::string name = fresh("block_decode");
    line("vint8m1_t " + name + " = __riscv_vrgather_vv_i8m1(" +
         table.spelling + ", " + codes.spelling + ", " + vl.str() + ");");
    BlockValue result{op.getResult().getType(), BlockValueKind::I8, name};
    result.vectorShape = decision.resultShape;
    blockValues[op.getResult()] = std::move(result);
    return mlir::success();
  }

  mlir::LogicalResult emitBlockStore(
      StoreOp op, llvm::DenseMap<mlir::Value, BlockValue> &blockValues,
      llvm::StringRef vl) {
    BlockValue pointer = lookupBlockValue(op.getPointer(), blockValues);
    BlockValue value = lookupBlockValue(op.getValue(), blockValues);
    auto pointerType =
        mlir::dyn_cast<PtrType>(elementType(op.getPointer().getType()));
    if (!pointerType || !pointerType.getElementType().isF32() ||
        pointer.kind != BlockValueKind::Pointer || pointer.pointerBase.empty() ||
        pointer.contiguousIndex.empty())
      return op.emitError(
          "RVV block store requires a contiguous f32 vector address");
    if (value.kind != BlockValueKind::F32 || value.spelling.empty() ||
        !isTrue(op.getWhere()))
      return op.emitError(
          "RVV block store requires an all-active f32 vector value");
    std::string suffix;
    if (value.vectorShape == RVVBlockVectorShape::E32M1)
      suffix = "f32m1";
    else if (value.vectorShape == RVVBlockVectorShape::E32M4)
      suffix = "f32m4";
    else
      return op.emitError("f32 block store has no physical RVV shape");
    line("__riscv_vse32_v_" + suffix + "(" + pointer.pointerBase + " + " +
         pointer.contiguousIndex + ", " + value.spelling + ", " + vl.str() +
         ");");
    return mlir::success();
  }

  mlir::LogicalResult emitBlockSelect(
      SelectOp op, llvm::DenseMap<mlir::Value, BlockValue> &blockValues,
      llvm::StringRef vl) {
    BlockValue predicate = lookupBlockValue(op.getPredicate(), blockValues);
    BlockValue trueValue = lookupBlockValue(op.getTrueValue(), blockValues);
    BlockValue falseValue = lookupBlockValue(op.getFalseValue(), blockValues);
    if (predicate.kind != BlockValueKind::Mask ||
        trueValue.kind != BlockValueKind::U8 ||
        falseValue.kind != BlockValueKind::U8)
      return op.emitError("RVV block select currently requires mask and u8 values");
    std::string name = fresh("block_select");
    line("vuint8m1_t " + name + " = __riscv_vmerge_vvm_u8m1(" +
         falseValue.spelling + ", " + trueValue.spelling + ", " +
         predicate.spelling + ", " + vl.str() + ");");
    blockValues[op.getResult()] =
        BlockValue{op.getResult().getType(), BlockValueKind::U8, name};
    return mlir::success();
  }

  mlir::LogicalResult emitBlockTuple(
      TupleOp op, llvm::DenseMap<mlir::Value, BlockValue> &blockValues) {
    BlockValue tuple;
    tuple.type = op.getResult().getType();
    tuple.kind = BlockValueKind::Tuple;
    for (mlir::Value operand : op.getOperands()) {
      BlockValue field = lookupBlockValue(operand, blockValues);
      tuple.fields.push_back(std::move(field));
    }
    blockValues[op.getResult()] = std::move(tuple);
    return mlir::success();
  }

  mlir::LogicalResult emitBlockTupleGet(
      TupleGetOp op, llvm::DenseMap<mlir::Value, BlockValue> &blockValues) {
    BlockValue tuple = lookupBlockValue(op.getInput(), blockValues);
    if (tuple.kind != BlockValueKind::Tuple || op.getIndex() < 0 ||
        static_cast<size_t>(op.getIndex()) >= tuple.fields.size())
      return op.emitError("RVV block tuple field is unavailable");
    BlockValue field = tuple.fields[op.getIndex()];
    if (field.spelling.empty() && field.kind != BlockValueKind::Tuple)
      return op.emitError("RVV block tuple field was not materialized");
    field.type = op.getResult().getType();
    blockValues[op.getResult()] = std::move(field);
    return mlir::success();
  }

  mlir::LogicalResult emitBlockOperation(
      mlir::Operation *operation,
      llvm::DenseMap<mlir::Value, BlockValue> &blockValues, llvm::StringRef vl) {
    if (mlir::isa<BlockAxisOp>(operation))
      return mlir::success();
    if (auto op = mlir::dyn_cast<PtrAddOp>(operation))
      return emitBlockPtrAdd(op, blockValues, vl);
    if (auto op = mlir::dyn_cast<BinaryOp>(operation))
      return emitBlockBinary(op, blockValues, vl);
    if (auto op = mlir::dyn_cast<CompareOp>(operation))
      return emitBlockCompare(op, blockValues, vl);
    if (auto op = mlir::dyn_cast<CastOp>(operation))
      return emitBlockCast(op, blockValues, vl);
    if (auto op = mlir::dyn_cast<BitcastOp>(operation))
      return emitBlockBitcast(op, blockValues);
    if (auto op = mlir::dyn_cast<LoadOp>(operation))
      return emitBlockLoad(op, blockValues, vl);
    if (auto op = mlir::dyn_cast<DecodeOp>(operation)) {
      BlockDecodeDecision decision;
      if (mlir::failed(decideBlockDecode(op, decision)))
        return mlir::failure();
      return emitBlockDecode(op, decision, blockValues, vl);
    }
    if (auto op = mlir::dyn_cast<StoreOp>(operation))
      return emitBlockStore(op, blockValues, vl);
    if (auto op = mlir::dyn_cast<SelectOp>(operation))
      return emitBlockSelect(op, blockValues, vl);
    if (auto op = mlir::dyn_cast<TupleOp>(operation))
      return emitBlockTuple(op, blockValues);
    if (auto op = mlir::dyn_cast<TupleGetOp>(operation))
      return emitBlockTupleGet(op, blockValues);
    return operation->emitError(
        "RISC-V target does not implement this block producer primitive");
  }

  mlir::LogicalResult emitBlockStores(StoreOp op) {
    if (auto lowered = tryEmitLocalF32ContractBlockStore(op))
      return *lowered;
    auto blockExtent = [](mlir::Type type) -> std::optional<int64_t> {
      if (auto masked = mlir::dyn_cast<MaskedType>(type))
        type = masked.getValueType();
      auto block = mlir::dyn_cast<BlockType>(type);
      if (!block || block.getShape().size() != 1 ||
          !block.getElementType().isF32())
        return std::nullopt;
      return block.getShape().front();
    };

    std::optional<int64_t> extent = blockExtent(op.getValue().getType());
    if (!extent || *extent <= 0 || *extent > 65535 ||
        !isTrue(op.getWhere()))
      return op.emitError(
          "RVV block store requires an all-active rank-one f32 value");

    llvm::DenseSet<mlir::Operation *> closure;
    llvm::SmallVector<BlockAxisOp> axes;
    if (mlir::failed(collectBlockClosure(op.getPointer(), closure, axes,
                                         op.getOperation())) ||
        mlir::failed(collectBlockClosure(op.getValue(), closure, axes,
                                         op.getOperation())))
      return mlir::failure();
    if (axes.size() != 1)
      return op.emitError(
          "RVV block store requires exactly one local logical block axis");
    BlockAxisOp axis = axes.front();
    auto axisType = axis.getResult().getType();
    if (axisType.getShape().size() != 1 ||
        axisType.getShape().front() != *extent)
      return op.emitError("block store extent does not match its logical axis");

    llvm::SmallVector<StoreOp> stores{op};
    for (mlir::Operation *candidate = op->getNextNode(); candidate;
         candidate = candidate->getNextNode()) {
      if (auto store = mlir::dyn_cast<StoreOp>(candidate)) {
        std::optional<int64_t> candidateExtent =
            blockExtent(store.getValue().getType());
        if (consumed.contains(candidate) || candidateExtent != extent ||
            !isTrue(store.getWhere()))
          break;
        llvm::DenseSet<mlir::Operation *> candidateClosure;
        llvm::SmallVector<BlockAxisOp> candidateAxes;
        if (mlir::failed(collectBlockClosure(store.getPointer(),
                                             candidateClosure, candidateAxes,
                                             store.getOperation())) ||
            mlir::failed(collectBlockClosure(store.getValue(), candidateClosure,
                                             candidateAxes,
                                             store.getOperation())) ||
            candidateAxes.size() != 1 || candidateAxes.front() != axis)
          break;
        stores.push_back(store);
        for (mlir::Operation *member : candidateClosure)
          closure.insert(member);
        continue;
      }
      if (hasBlockPayload(candidate))
        continue;
      if (mlir::isa<ConstantOp, InvalidOp, PtrAddOp, BinaryOp, CompareOp, CastOp,
                    BitcastOp, SelectOp, TupleOp, TupleGetOp, SpecialValueOp>(
              candidate)) {
        if (mlir::failed(emitOperation(candidate)))
          return mlir::failure();
        consumed.insert(candidate);
        continue;
      }
      break;
    }

    llvm::DenseSet<mlir::Operation *> storeOps;
    for (StoreOp store : stores)
      storeOps.insert(store.getOperation());
    for (mlir::Operation *operation : closure)
      for (mlir::Value result : operation->getResults())
        if (containsBlockType(result.getType()))
          for (mlir::Operation *user : result.getUsers())
            if (!storeOps.contains(user) && !closure.contains(user) &&
                !loweredBlockOps.contains(user) &&
                !llvm::all_of(user->getResults(),
                              [](mlir::Value value) { return value.use_empty(); }))
              return op.emitError(
                  "block producer escapes its operation-local store closure");

    std::string offset = expression(axis.getOffset());
    if (offset.empty())
      return op.emitError("block store axis offset is unavailable");
    std::string strip = fresh("block_i");
    std::string vl = fresh("block_vl");
    std::string lane = fresh("block_lane");
    bool needsLaneVector = llvm::any_of(
        axis.getResult().getUsers(), [&](mlir::Operation *user) {
          if (!closure.contains(user))
            return false;
          auto pointer = mlir::dyn_cast<PtrAddOp>(user);
          return !pointer || pointer.getOffset() != axis.getResult();
        });
    bool microClosure = llvm::all_of(closure, [](mlir::Operation *operation) {
      return mlir::isa<BlockAxisOp, PtrAddOp, LoadOp, BinaryOp, CastOp>(
          operation);
    });
    bool useMicroVectors =
        !needsLaneVector && *extent == 32 && microClosure;

    auto emitStrip = [&](llvm::StringRef stripOffset,
                         llvm::StringRef activeVL) -> mlir::LogicalResult {
      llvm::DenseMap<mlir::Value, BlockValue> blockValues;
      BlockValue coordinate{axis.getResult().getType(), BlockValueKind::Index,
                            needsLaneVector ? lane : ""};
      coordinate.contiguousIndex =
          "(" + stripOffset.str() + " + " + offset + ")";
      blockValues[axis.getResult()] = std::move(coordinate);
      bool previousMicroBlockVectors = microBlockVectors;
      microBlockVectors = useMicroVectors;
      for (mlir::Operation &candidate : *op->getBlock()) {
        if (mlir::isa<BlockAxisOp>(candidate) ||
            (!closure.contains(&candidate) && !storeOps.contains(&candidate)))
          continue;
        if (mlir::failed(emitBlockOperation(&candidate, blockValues, activeVL))) {
          microBlockVectors = previousMicroBlockVectors;
          return mlir::failure();
        }
      }
      microBlockVectors = previousMicroBlockVectors;
      return mlir::success();
    };

    if (useMicroVectors) {
      line("const size_t " + vl + " = __riscv_vsetvl_e8mf4(4);");
      llvm::SmallVector<llvm::DenseMap<mlir::Value, BlockValue>, 8>
          stripValues;
      for (int64_t stripOffset : {int64_t{0}, int64_t{4}, int64_t{8},
                                  int64_t{12}, int64_t{16}, int64_t{20},
                                  int64_t{24}, int64_t{28}}) {
        llvm::DenseMap<mlir::Value, BlockValue> blockValues;
        BlockValue coordinate{axis.getResult().getType(),
                              BlockValueKind::Index, ""};
        coordinate.contiguousIndex =
            "(" + std::to_string(stripOffset) + " + " + offset + ")";
        blockValues[axis.getResult()] = std::move(coordinate);
        stripValues.push_back(std::move(blockValues));
      }
      bool previousMicroBlockVectors = microBlockVectors;
      microBlockVectors = true;
      for (mlir::Operation &candidate : *op->getBlock()) {
        if (mlir::isa<BlockAxisOp>(candidate) ||
            (!closure.contains(&candidate) && !storeOps.contains(&candidate)))
          continue;
        for (auto &blockValues : stripValues)
          if (mlir::failed(emitBlockOperation(&candidate, blockValues, vl))) {
            microBlockVectors = previousMicroBlockVectors;
            return mlir::failure();
          }
      }
      microBlockVectors = previousMicroBlockVectors;
    } else if (!needsLaneVector && *extent == 32) {
      line("const size_t " + vl + " = __riscv_vsetvl_e8m1(16);");
      if (mlir::failed(emitStrip("0", vl)) ||
          mlir::failed(emitStrip("16", vl)))
        return mlir::failure();
    } else {
      line("for (size_t " + strip + " = 0; " + strip + " < " +
           std::to_string(*extent) + ";) {");
      ++indent;
      line("const size_t " + vl + " = __riscv_vsetvl_e8m1((" +
           std::to_string(*extent) + " - " + strip + ") < 16 ? (" +
           std::to_string(*extent) + " - " + strip + ") : 16);");
      if (needsLaneVector) {
        line("vuint16m2_t " + lane + " = __riscv_vid_v_u16m2(" + vl +
             ");");
        line(lane + " = __riscv_vadd_vx_u16m2(" + lane + ", " + strip +
             " + " + offset + ", " + vl + ");");
      }
      if (mlir::failed(emitStrip(strip, vl)))
        return mlir::failure();
      line(strip + " += " + vl + ";");
      --indent;
      line("}");
    }

    for (mlir::Operation *operation : closure)
      loweredBlockOps.insert(operation);
    for (StoreOp store : stores)
      consumed.insert(store.getOperation());
    return mlir::success();
  }

  mlir::LogicalResult emitBlockReduce(ReduceOp op) {
    auto isSupportedReduction = [](ReduceOp reduction) {
      auto input = mlir::dyn_cast<BlockType>(reduction.getInput().getType());
      return input && input.getShape().size() == 1 &&
             reduction.getAxis() == 0 && reduction.getKind() == "add" &&
             reduction.getResult().getType().isSignedInteger(32) &&
             input.getElementType().isSignedInteger(32) &&
             isTrue(reduction.getWhere());
    };
    if (!isSupportedReduction(op))
      return op.emitError(
          "RVV block reduction currently requires all-active rank-one i32 add");
    auto inputType = mlir::cast<BlockType>(op.getInput().getType());
    int64_t extent = inputType.getShape().front();
    if (extent <= 0 || extent > 65535)
      return op.emitError(
          "RVV block reduction requires a static extent in [1, 65535]");

    llvm::DenseSet<mlir::Operation *> closure;
    llvm::SmallVector<BlockAxisOp> firstAxes;
    if (mlir::failed(
            collectBlockClosure(op.getInput(), closure, firstAxes,
                                op.getOperation())))
      return mlir::failure();
    if (firstAxes.size() != 1)
      return op.emitError(
          "RVV block reduction requires exactly one local logical block axis");
    BlockAxisOp axis = firstAxes.front();
    auto axisType = axis.getResult().getType();
    if (axisType.getShape().size() != 1 || axisType.getShape().front() != extent)
      return op.emitError("block reduction extent does not match its logical axis");
    llvm::SmallVector<ReduceOp> reductions{op};
    for (mlir::Operation *candidate = op->getNextNode(); candidate;
         candidate = candidate->getNextNode()) {
      if (auto reduction = mlir::dyn_cast<ReduceOp>(candidate)) {
        if (consumed.contains(candidate) || !isSupportedReduction(reduction))
          break;
        auto candidateType = mlir::cast<BlockType>(reduction.getInput().getType());
        llvm::DenseSet<mlir::Operation *> candidateClosure;
        llvm::SmallVector<BlockAxisOp> candidateAxes;
        if (candidateType.getShape().front() != extent ||
            mlir::failed(collectBlockClosure(reduction.getInput(),
                                             candidateClosure, candidateAxes,
                                             reduction.getOperation())) ||
            candidateAxes.size() != 1 || candidateAxes.front() != axis)
          break;
        reductions.push_back(reduction);
        for (mlir::Operation *member : candidateClosure)
          closure.insert(member);
        continue;
      }
      if (hasBlockPayload(candidate) || mlir::isa<ConstantOp, InvalidOp>(candidate))
        continue;
      break;
    }

    llvm::DenseSet<mlir::Operation *> reductionOps;
    for (ReduceOp reduction : reductions)
      reductionOps.insert(reduction.getOperation());
    for (mlir::Operation *operation : closure)
      for (mlir::Value result : operation->getResults())
        if (containsBlockType(result.getType()))
          for (mlir::Operation *user : result.getUsers())
            if (!reductionOps.contains(user) && !closure.contains(user) &&
                !loweredBlockOps.contains(user) &&
                !llvm::all_of(user->getResults(),
                              [](mlir::Value value) { return value.use_empty(); }))
              return op.emitError(
                  "block producer escapes its operation-local reduction closure");

    std::string offset = expression(axis.getOffset());
    if (offset.empty())
      return op.emitError("block reduction axis offset is unavailable");
    llvm::SmallVector<std::string> accumulators;
    for (ReduceOp reduction : reductions) {
      std::string identity = expression(reduction.getIdentity());
      if (identity.empty())
        return reduction.emitError("block reduction identity is unavailable");
      std::string accumulator = fresh("block_reduce");
      line("int32_t " + accumulator + " = " + identity + ";");
      accumulators.push_back(std::move(accumulator));
    }
    std::string strip = fresh("block_i");
    std::string vl = fresh("block_vl");
    std::string lane = fresh("block_lane");
    bool needsLaneVector = llvm::any_of(
        axis.getResult().getUsers(), [&](mlir::Operation *user) {
          if (!closure.contains(user))
            return false;
          auto pointer = mlir::dyn_cast<PtrAddOp>(user);
          return !pointer || pointer.getOffset() != axis.getResult();
        });

    if (!needsLaneVector && extent == 32) {
      line("const size_t " + vl + " = __riscv_vsetvl_e8m1(16);");
      llvm::SmallVector<llvm::SmallVector<BlockValue, 2>, 2> stripInputs(
          reductions.size());
      llvm::DenseMap<mlir::Operation *, size_t> reductionIndices;
      for (auto [index, reduction] : llvm::enumerate(reductions))
        reductionIndices[reduction.getOperation()] = index;
      for (int64_t stripOffset : {int64_t{0}, int64_t{16}}) {
        llvm::DenseMap<mlir::Value, BlockValue> blockValues;
        BlockValue coordinate{axis.getResult().getType(), BlockValueKind::Index,
                              ""};
        coordinate.contiguousIndex =
            "(" + std::to_string(stripOffset) + " + " + offset + ")";
        blockValues[axis.getResult()] = std::move(coordinate);
        for (mlir::Operation &candidate : *op->getBlock()) {
          auto reduction = reductionIndices.find(&candidate);
          if (reduction != reductionIndices.end()) {
            ReduceOp current = reductions[reduction->second];
            BlockValue input = lookupBlockValue(current.getInput(), blockValues);
            if (input.kind == BlockValueKind::I32 &&
                mlir::failed(
                    materializeI32(input, current.getOperation(), vl)))
              return mlir::failure();
            if ((input.kind != BlockValueKind::I32 &&
                 input.kind != BlockValueKind::I16) ||
                input.spelling.empty())
              return current.emitError(
                  "fixed-strip reduction input is not an integer vector");
            stripInputs[reduction->second].push_back(std::move(input));
            if (&candidate == reductions.back().getOperation())
              break;
            continue;
          }
          if (!closure.contains(&candidate) ||
              mlir::isa<BlockAxisOp>(candidate))
            continue;
          if (mlir::failed(emitBlockOperation(&candidate, blockValues, vl)))
            return mlir::failure();
        }
      }
      for (auto [index, reduction] : llvm::enumerate(reductions)) {
        if (stripInputs[index].size() != 2 ||
            stripInputs[index][0].kind != stripInputs[index][1].kind)
          return reduction.emitError(
              "fixed-strip reduction produced incompatible physical values");
        BlockValueKind kind = stripInputs[index][0].kind;
        std::string combined = fresh("block_combined");
        std::string seed = fresh("block_seed");
        std::string partial = fresh("block_partial");
        if (kind == BlockValueKind::I16) {
          line("vint32m4_t " + combined + " = __riscv_vwadd_vv_i32m4(" +
               stripInputs[index][0].spelling + ", " +
               stripInputs[index][1].spelling + ", " + vl + ");");
          line("vint32m1_t " + seed + " = __riscv_vmv_v_x_i32m1(0, 1);");
          line("vint32m1_t " + partial +
               " = __riscv_vredsum_vs_i32m4_i32m1(" + combined + ", " + seed +
               ", " + vl + ");");
        } else if (kind == BlockValueKind::I32) {
          line("vint32m4_t " + combined + " = __riscv_vadd_vv_i32m4(" +
               stripInputs[index][0].spelling + ", " +
               stripInputs[index][1].spelling + ", " + vl + ");");
          line("vint32m1_t " + seed + " = __riscv_vmv_v_x_i32m1(0, 1);");
          line("vint32m1_t " + partial +
               " = __riscv_vredsum_vs_i32m4_i32m1(" + combined + ", " + seed +
               ", " + vl + ");");
        } else {
          return reduction.emitError(
              "fixed-strip reduction physical type is unsupported");
        }
        line(accumulators[index] + " += __riscv_vmv_x_s_i32m1_i32(" +
             partial + ");");
        values[reduction.getResult()] =
            CValue{reduction.getResult().getType(), CValueKind::Scalar,
                   accumulators[index]};
        if (reduction != op)
          consumed.insert(reduction.getOperation());
      }
      for (mlir::Operation *operation : closure)
        loweredBlockOps.insert(operation);
      return mlir::success();
    }

    line("for (size_t " + strip + " = 0; " + strip + " < " +
         std::to_string(extent) + ";) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e8m1((" +
         std::to_string(extent) + " - " + strip + ") < 16 ? (" +
         std::to_string(extent) + " - " + strip + ") : 16);");

    if (needsLaneVector) {
      line("vuint16m2_t " + lane + " = __riscv_vid_v_u16m2(" + vl + ");");
      line(lane + " = __riscv_vadd_vx_u16m2(" + lane + ", " + strip + " + " +
           offset + ", " + vl + ");");
    }

    llvm::DenseMap<mlir::Value, BlockValue> blockValues;
    BlockValue coordinate{axis.getResult().getType(), BlockValueKind::Index,
                          needsLaneVector ? lane : ""};
    coordinate.contiguousIndex = "(" + strip + " + " + offset + ")";
    blockValues[axis.getResult()] = std::move(coordinate);
    llvm::DenseMap<mlir::Operation *, size_t> reductionIndices;
    for (auto [index, reduction] : llvm::enumerate(reductions))
      reductionIndices[reduction.getOperation()] = index;
    for (mlir::Operation &candidate : *op->getBlock()) {
      auto reduction = reductionIndices.find(&candidate);
      if (reduction != reductionIndices.end()) {
        ReduceOp current = reductions[reduction->second];
        BlockValue input = lookupBlockValue(current.getInput(), blockValues);
        if (input.kind == BlockValueKind::I32 &&
            mlir::failed(materializeI32(input, current.getOperation(), vl)))
          return mlir::failure();
        if ((input.kind != BlockValueKind::I32 &&
             input.kind != BlockValueKind::I16) ||
            input.spelling.empty())
          return current.emitError(
              "RVV block reduction input is not an integer vector");
        std::string seed = fresh("block_seed");
        std::string partial = fresh("block_partial");
        const std::string &accumulator = accumulators[reduction->second];
        if (input.kind == BlockValueKind::I16) {
          line("vint16m1_t " + seed + " = __riscv_vmv_v_x_i16m1(0, 1);");
          line("vint16m1_t " + partial +
               " = __riscv_vredsum_vs_i16m2_i16m1(" + input.spelling + ", " +
               seed + ", " + vl + ");");
          line(accumulator +
               " += (int32_t)__riscv_vmv_x_s_i16m1_i16(" + partial + ");");
        } else {
          line("vint32m1_t " + seed + " = __riscv_vmv_v_x_i32m1(0, 1);");
          line("vint32m1_t " + partial +
               " = __riscv_vredsum_vs_i32m4_i32m1(" + input.spelling + ", " +
               seed + ", " + vl + ");");
          line(accumulator + " += __riscv_vmv_x_s_i32m1_i32(" + partial +
               ");");
        }
        if (&candidate == reductions.back().getOperation())
          break;
        continue;
      }
      if (&candidate == reductions.back().getOperation())
        break;
      if (!closure.contains(&candidate) ||
          mlir::isa<BlockAxisOp>(candidate))
        continue;
      if (mlir::failed(emitBlockOperation(&candidate, blockValues, vl)))
        return mlir::failure();
    }
    line(strip + " += " + vl + ";");
    --indent;
    line("}");
    for (auto [reduction, accumulator] : llvm::zip(reductions, accumulators)) {
      values[reduction.getResult()] = CValue{reduction.getResult().getType(),
                                             CValueKind::Scalar, accumulator};
      if (reduction != op)
        consumed.insert(reduction.getOperation());
    }
    for (mlir::Operation *operation : closure)
      loweredBlockOps.insert(operation);
    return mlir::success();
  }

  mlir::LogicalResult emitReduce(ReduceOp op) {
    if (inVLA)
      return op.emitError("VLA reduction must be owned by its VLA lowering");
    return emitBlockReduce(op);
  }

  mlir::LogicalResult emitVectorReduce(ReduceOp op,
                                       const VLAStateDecision &decision,
                                       const CValue &aggregate) {
    CValue input = require(op.getInput());
    if (input.kind != CValueKind::F32Vector ||
        aggregate.kind != CValueKind::Scalar || aggregate.spelling.empty())
      return op.emitError("RVV reduction projection is unavailable");
    std::string seed = fresh("seed");
    std::string partial = fresh("partial");
    line("vfloat32m1_t " + seed + " = __riscv_vfmv_v_f_f32m1(" +
         aggregate.spelling + ", 1);");
    if (decision.realization == VLAStateRealization::RVVAddReduction)
      line("vfloat32m1_t " + partial +
           " = __riscv_vfredusum_vs_f32m2_f32m1(" + input.spelling + ", " +
           seed + ", " + activeVL + ");");
    else if (decision.realization == VLAStateRealization::RVVMaxReduction)
      line("vfloat32m1_t " + partial +
           " = __riscv_vfredmax_vs_f32m2_f32m1(" + input.spelling + ", " +
           seed + ", " + activeVL + ");");
    else
      return op.emitError("selected VLA state is not a reduction");
    line(aggregate.spelling + " = __riscv_vfmv_f_s_f32m1_f32(" + partial +
         ");");
    values[op.getResult()] = aggregate;
    return mlir::success();
  }

  mlir::LogicalResult emitVectorScan(ScanOp op,
                                     const VLAStateDecision &decision,
                                     const CValue &carry) {
    CValue input = require(op.getInput());
    if (decision.realization != VLAStateRealization::RVVInclusiveAddScan ||
        input.kind != CValueKind::F32Vector ||
        carry.kind != CValueKind::Scalar || carry.spelling.empty())
      return op.emitError("RVV scan projection is unavailable");

    std::string indices = fresh("scan_indices");
    std::string prefix = fresh("scan_prefix");
    std::string offset = fresh("scan_offset");
    line("vuint32m2_t " + indices + " = __riscv_vid_v_u32m2(" + activeVL +
         ");");
    line("vfloat32m2_t " + prefix + " = " + input.spelling + ";");
    line("for (size_t " + offset + " = 1; " + offset + " < " + activeVL +
         "; " + offset + " <<= 1) {");
    ++indent;
    std::string shifted = fresh("scan_shifted");
    std::string active = fresh("scan_active");
    line("vfloat32m2_t " + shifted +
         " = __riscv_vslideup_vx_f32m2(__riscv_vundefined_f32m2(), " +
         prefix + ", " + offset + ", " + activeVL + ");");
    line("vbool16_t " + active + " = __riscv_vmsgeu_vx_u32m2_b16(" +
         indices + ", (uint32_t)" + offset + ", " + activeVL + ");");
    line(prefix + " = __riscv_vfadd_vv_f32m2_m(" + active + ", " + prefix +
         ", " + shifted + ", " + activeVL + ");");
    --indent;
    line("}");
    line(prefix + " = __riscv_vfadd_vf_f32m2(" + prefix + ", " +
         carry.spelling + ", " + activeVL + ");");
    std::string last = fresh("scan_last");
    line("vfloat32m2_t " + last + " = __riscv_vslidedown_vx_f32m2(" + prefix +
         ", " + activeVL + " - 1, " + activeVL + ");");
    line(carry.spelling + " = __riscv_vfmv_f_s_f32m2_f32(" + last + ");");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::F32Vector, prefix};
    return mlir::success();
  }

  mlir::LogicalResult emitVectorArgMaxSummary(
      SummaryFoldOp op, const VLAStateDecision &decision,
      const CValue &aggregate) {
    CValue input = require(op.getInput());
    CValue coordinate = require(op.getCoordinate());
    if (decision.realization != VLAStateRealization::RVVArgMaxSummary ||
        input.kind != CValueKind::F32Vector ||
        coordinate.kind != CValueKind::Coordinate ||
        coordinate.spelling.empty() || coordinate.laneStride.empty() ||
        aggregate.kind != CValueKind::Tuple || aggregate.fields.size() != 2)
      return op.emitError("RVV argmax summary projection is unavailable");

    const CValue &maximum = aggregate.fields[0];
    const CValue &index = aggregate.fields[1];
    std::string seed = fresh("argmax_seed");
    std::string reduced = fresh("argmax_reduced");
    std::string stripMaximum = fresh("argmax_strip_value");
    std::string equal = fresh("argmax_equal");
    std::string first = fresh("argmax_first");
    std::string stripIndex = fresh("argmax_strip_index");
    line("vfloat32m1_t " + seed +
         " = __riscv_vfmv_v_f_f32m1(-INFINITY, 1);");
    line("vfloat32m1_t " + reduced +
         " = __riscv_vfredmax_vs_f32m2_f32m1(" + input.spelling + ", " +
         seed + ", " + activeVL + ");");
    line("const float " + stripMaximum +
         " = __riscv_vfmv_f_s_f32m1_f32(" + reduced + ");");
    line("vbool16_t " + equal + " = __riscv_vmfeq_vf_f32m2_b16(" +
         input.spelling + ", " + stripMaximum + ", " + activeVL + ");");
    line("const long " + first + " = __riscv_vfirst_m_b16(" + equal + ", " +
         activeVL + ");");
    std::string laneOffset = first;
    if (decision.coordinateMode == VLAMemoryMode::Strided)
      laneOffset = "(" + first + " * (" + coordinate.laneStride + "))";
    line("const size_t " + stripIndex + " = (size_t)(" + coordinate.spelling +
         " + " + laneOffset + ");");
    line("if (" + stripMaximum + " > " + maximum.spelling + " || (" +
         stripMaximum + " == " + maximum.spelling + " && " + stripIndex +
         " < " + index.spelling + ")) {");
    ++indent;
    line(maximum.spelling + " = " + stripMaximum + ";");
    line(index.spelling + " = " + stripIndex + ";");
    --indent;
    line("}");
    values[op.getResult()] = aggregate;
    return mlir::success();
  }

  mlir::LogicalResult emitOnlineSoftmaxSummary(
      SummaryFoldOp op, const CValue &aggregate) {
    CValue input = require(op.getInput());
    if (!isOnlineSoftmaxSummary(op) ||
        input.kind != CValueKind::F32Vector ||
        aggregate.kind != CValueKind::Tuple || aggregate.fields.size() != 2)
      return op.emitError(
          "online summary requires an all-active f32 VLA input");
    const CValue &maximum = aggregate.fields[0];
    const CValue &sum = aggregate.fields[1];
    std::string maxSeed = fresh("summary_max_seed");
    std::string maxVector = fresh("summary_max_vector");
    std::string stripMaximum = fresh("summary_strip_max");
    std::string shifted = fresh("summary_shifted");
    std::string exponentials = fresh("summary_exp");
    std::string sumSeed = fresh("summary_sum_seed");
    std::string sumVector = fresh("summary_sum_vector");
    std::string stripSum = fresh("summary_strip_sum");
    std::string mergedMaximum = fresh("summary_merged_max");
    line("vfloat32m1_t " + maxSeed +
         " = __riscv_vfmv_v_f_f32m1(-INFINITY, 1);");
    line("vfloat32m1_t " + maxVector +
         " = __riscv_vfredmax_vs_f32m2_f32m1(" + input.spelling + ", " +
         maxSeed + ", " + activeVL + ");");
    line("const float " + stripMaximum +
         " = __riscv_vfmv_f_s_f32m1_f32(" + maxVector + ");");
    line("vfloat32m2_t " + shifted + " = __riscv_vfsub_vf_f32m2(" +
         input.spelling + ", " + stripMaximum + ", " + activeVL + ");");
    line("vfloat32m2_t " + exponentials + " = __weft_exp_f32m2(" +
         shifted + ", " + activeVL + ");");
    line("vfloat32m1_t " + sumSeed +
         " = __riscv_vfmv_v_f_f32m1(0.0f, 1);");
    line("vfloat32m1_t " + sumVector +
         " = __riscv_vfredusum_vs_f32m2_f32m1(" + exponentials + ", " +
         sumSeed + ", " + activeVL + ");");
    line("const float " + stripSum +
         " = __riscv_vfmv_f_s_f32m1_f32(" + sumVector + ");");
    line("const float " + mergedMaximum + " = fmaxf(" + maximum.spelling +
         ", " + stripMaximum + ");");
    line(sum.spelling + " = " + sum.spelling + " * expf(" +
         maximum.spelling + " - " + mergedMaximum + ") + " + stripSum +
         " * expf(" + stripMaximum + " - " + mergedMaximum + ");");
    line(maximum.spelling + " = " + mergedMaximum + ";");
    values[op.getResult()] = aggregate;
    return mlir::success();
  }
};

void emitPrelude(llvm::raw_ostream &output, bool usesExp, bool usesIME1,
                 bool usesGroupedI4I8) {
  output << R"c(#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

static inline __attribute__((unused)) int8_t __weft_bitcast_u8_i8(uint8_t bits) {
  union { uint8_t u; int8_t i; } value = { .u = bits };
  return value.i;
}

static inline __attribute__((unused)) int16_t __weft_bitcast_u16_i16(uint16_t bits) {
  union { uint16_t u; int16_t i; } value = { .u = bits };
  return value.i;
}

static inline __attribute__((unused)) _Float16 __weft_bitcast_u16_f16(uint16_t bits) {
  union { uint16_t u; _Float16 f; } value = { .u = bits };
  return value.f;
}

static inline __attribute__((unused)) uint16_t __weft_bitcast_f16_u16(_Float16 source) {
  union { uint16_t u; _Float16 f; } value = { .f = source };
  return value.u;
}

static inline __attribute__((unused)) float __weft_bitcast_u32_f32(uint32_t bits) {
  union { uint32_t u; float f; } value = { .u = bits };
  return value.f;
}

)c";
  if (usesGroupedI4I8) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_grouped_affine_i4_i8_vl128(
    const uint8_t *packed_weight, const uint8_t *scale_min,
    const uint8_t *activation_bytes, const uint8_t *activation_sum_bytes,
    float dot_scale, float minimum_scale, float init) {
  uint8_t scale[8];
  uint8_t minimum[8];
  for (size_t group = 0; group < 4; ++group) {
    scale[group] = scale_min[group] & UINT8_C(63);
    minimum[group] = scale_min[group + 4] & UINT8_C(63);
    scale[group + 4] = (scale_min[group + 8] & UINT8_C(15)) |
                       ((scale_min[group] >> 6) << 4);
    minimum[group + 4] = (scale_min[group + 8] >> 4) |
                         ((scale_min[group + 4] >> 6) << 4);
  }

  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  float sum = init;
  float temporary;
  float second;
  const uint8_t *q40;
  const uint8_t *q41;
  const uint8_t *q42;
  const uint8_t *q43;
  const int8_t *q80;
  const int8_t *q81;
  const int8_t *q82;
  const int8_t *q83;
  int s0;
  int s1;
  int s2;
  int s3;

  __asm__ volatile(
      "vsetivli zero, 4, e32, m1, ta, ma\n\t"
      "vmv.v.x v16, zero\n\t"
      "vsetivli zero, 8, e16, m1, ta, ma\n\t"
      "vle32.v v2, (%[bsums])\n\t"
      "vnsrl.wi v0, v2, 0\n\t"
      "vnsrl.wi v1, v2, 16\n\t"
      "vadd.vv v2, v0, v1\n\t"
      "vle8.v v3, (%[mins])\n\t"
      "vzext.vf2 v4, v3\n\t"
      "vwmul.vv v6, v4, v2\n\t"
      "vsetivli zero, 4, e32, m1, ta, ma\n\t"
      "vredsum.vs v0, v6, v16\n\t"
      "vredsum.vs v0, v7, v0\n\t"
      "vfcvt.f.x.v v0, v0\n\t"
      "vfmv.f.s %[temporary], v0\n\t"
      "vsetivli zero, 16, e8, m1, ta, ma\n\t"
      "vle8.v v0, (%[packed])\n\t"
      "fnmsub.s %[sum], %[minimum_scale], %[temporary], %[sum]\n\t"
      "addi %[q40], %[packed], 64\n\t"
      "addi %[q41], %[packed], 16\n\t"
      "addi %[q42], %[packed], 32\n\t"
      "addi %[q43], %[packed], 48\n\t"
      "addi %[q80], %[activation], 64\n\t"
      "vle8.v v1, (%[q41])\n\t"
      "vle8.v v2, (%[q42])\n\t"
      "addi %[q81], %[activation], 16\n\t"
      "addi %[q41], %[q41], 64\n\t"
      "addi %[q82], %[activation], 32\n\t"
      "vle8.v v3, (%[q43])\n\t"
      "vle8.v v8, (%[activation])\n\t"
      "addi %[q42], %[q42], 64\n\t"
      "addi %[q83], %[activation], 48\n\t"
      "addi %[q43], %[q43], 64\n\t"
      "vsrl.vi v4, v0, 4\n\t"
      "vle8.v v9, (%[q81])\n\t"
      "vle8.v v10, (%[q82])\n\t"
      "vand.vi v0, v0, 0xF\n\t"
      "addi %[q81], %[q81], 64\n\t"
      "vsrl.vi v5, v1, 4\n\t"
      "addi %[q82], %[q82], 64\n\t"
      "vle8.v v11, (%[q83])\n\t"
      "vle8.v v12, (%[q80])\n\t"
      "vand.vi v1, v1, 0xF\n\t"
      "addi %[q83], %[q83], 64\n\t"
      "vsrl.vi v6, v2, 4\n\t"
      "addi %[q80], %[q80], 64\n\t"
      "vle8.v v13, (%[q81])\n\t"
      "vle8.v v14, (%[q82])\n\t"
      "vand.vi v2, v2, 0xF\n\t"
      "addi %[q81], %[q81], 64\n\t"
      "vsrl.vi v7, v3, 4\n\t"
      "addi %[q82], %[q82], 64\n\t"
      "vwmul.vv v16, v0, v8\n\t"
      "vle8.v v15, (%[q83])\n\t"
      "vle8.v v0, (%[q40])\n\t"
      "vand.vi v3, v3, 0xF\n\t"
      "addi %[q83], %[q83], 64\n\t"
      "vwmul.vv v24, v2, v12\n\t"
      "vwmul.vv v20, v4, v10\n\t"
      "vwmul.vv v28, v6, v14\n\t"
      "vwmacc.vv v16, v1, v9\n\t"
      "vle8.v v1, (%[q41])\n\t"
      "vle8.v v2, (%[q42])\n\t"
      "vwmacc.vv v24, v3, v13\n\t"
      "vwmacc.vv v20, v5, v11\n\t"
      "vwmacc.vv v28, v7, v15\n\t"
      "addi %[q40], %[q80], 64\n\t"
      "addi %[q41], %[q81], 64\n\t"
      "vle8.v v3, (%[q43])\n\t"
      "vle8.v v8, (%[q80])\n\t"
      "addi %[q42], %[q82], 64\n\t"
      "addi %[q43], %[q83], 64\n\t"
      "vsrl.vi v4, v0, 4\n\t"
      "vle8.v v9, (%[q81])\n\t"
      "vle8.v v10, (%[q82])\n\t"
      "vand.vi v0, v0, 0xF\n\t"
      "vsrl.vi v5, v1, 4\n\t"
      "vsrl.vi v7, v3, 4\n\t"
      "vand.vi v3, v3, 0xF\n\t"
      "vle8.v v11, (%[q83])\n\t"
      "vle8.v v12, (%[q40])\n\t"
      "vand.vi v1, v1, 0xF\n\t"
      "vsrl.vi v6, v2, 4\n\t"
      "vand.vi v2, v2, 0xF\n\t"
      "vwmul.vv v18, v0, v8\n\t"
      "vle8.v v13, (%[q41])\n\t"
      "vle8.v v14, (%[q42])\n\t"
      "vwmul.vv v26, v2, v12\n\t"
      "vwmul.vv v22, v4, v10\n\t"
      "vwmul.vv v30, v6, v14\n\t"
      "vwmacc.vv v18, v1, v9\n\t"
      "vle8.v v15, (%[q43])\n\t"
      "vwmacc.vv v26, v3, v13\n\t"
      "vwmacc.vv v22, v5, v11\n\t"
      "vwmacc.vv v30, v7, v15\n\t"
      "vmv.v.x v0, zero\n\t"
      "vsetivli zero, 16, e16, m2, ta, ma\n\t"
      "vwredsum.vs v4, v16, v0\n\t"
      "lbu %[s0], 0(%[scale])\n\t"
      "vwredsum.vs v5, v20, v0\n\t"
      "lbu %[s1], 1(%[scale])\n\t"
      "vwredsum.vs v6, v24, v0\n\t"
      "lbu %[s2], 2(%[scale])\n\t"
      "vwredsum.vs v7, v28, v0\n\t"
      "lbu %[s3], 3(%[scale])\n\t"
      "vwredsum.vs v8, v18, v0\n\t"
      "lbu %[q40], 4(%[scale])\n\t"
      "vwredsum.vs v9, v22, v0\n\t"
      "lbu %[q41], 5(%[scale])\n\t"
      "vwredsum.vs v10, v26, v0\n\t"
      "lbu %[q42], 6(%[scale])\n\t"
      "vwredsum.vs v11, v30, v0\n\t"
      "lbu %[q43], 7(%[scale])\n\t"
      "vsetivli zero, 4, e32, m1, ta, ma\n\t"
      "vmul.vx v0, v4, %[s0]\n\t"
      "vmul.vx v1, v8, %[q40]\n\t"
      "vmacc.vx v0, %[s1], v5\n\t"
      "vmacc.vx v1, %[q41], v9\n\t"
      "vmacc.vx v0, %[s2], v6\n\t"
      "vmacc.vx v1, %[q42], v10\n\t"
      "vmacc.vx v0, %[s3], v7\n\t"
      "vmacc.vx v1, %[q43], v11\n\t"
      "vfcvt.f.x.v v0, v0\n\t"
      "vfcvt.f.x.v v1, v1\n\t"
      "vfmv.f.s %[second], v0\n\t"
      "vfmv.f.s %[temporary], v1\n\t"
      "fadd.s %[second], %[second], %[temporary]\n\t"
      "fmadd.s %[sum], %[dot_scale], %[second], %[sum]"
      : [temporary] "=&f"(temporary), [sum] "+&f"(sum),
        [second] "=&f"(second), [s0] "=&r"(s0), [s1] "=&r"(s1),
        [s2] "=&r"(s2), [s3] "=&r"(s3), [q40] "=&r"(q40),
        [q41] "=&r"(q41), [q42] "=&r"(q42), [q43] "=&r"(q43),
        [q80] "=&r"(q80), [q81] "=&r"(q81), [q82] "=&r"(q82),
        [q83] "=&r"(q83)
      : [dot_scale] "f"(dot_scale), [activation] "r"(activation),
        [packed] "r"(packed_weight), [scale] "r"(scale),
        [bsums] "r"(activation_sum_bytes), [mins] "r"(minimum),
        [minimum_scale] "f"(minimum_scale)
      : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
        "v15", "v16", "v17", "v18", "v19", "v20", "v21",
        "v22", "v23", "v24", "v25", "v26", "v27", "v28",
        "v29", "v30", "v31");
  return sum;
}

)c";
  }
  if (usesIME1) {
    output << R"ime(#define __WEFT_IME1_COMP_I4_I8_M1                                      \
  "vmadot       v16, v14, v0            \n\t"                         \
  "vmadot       v18, v14, v1            \n\t"                         \
  "vmadot       v20, v14, v2            \n\t"                         \
  "vmadot       v22, v14, v3            \n\t"                         \
  "vmadot       v16, v15, v4            \n\t"                         \
  "vmadot       v18, v15, v5            \n\t"                         \
  "vmadot       v20, v15, v6            \n\t"                         \
  "vmadot       v22, v15, v7            \n\t"

#define __WEFT_IME1_ACC_I4_I8_M1                                       \
  "vfcvt.f.x.v  v16, v16                \n\t"                         \
  "vfcvt.f.x.v  v18, v18                \n\t"                         \
  "vfcvt.f.x.v  v20, v20                \n\t"                         \
  "vfcvt.f.x.v  v22, v22                \n\t"                         \
  "addi         s2, s1, 8                \n\t"                         \
  "addi         s3, s1, 16               \n\t"                         \
  "addi         s4, s1, 24               \n\t"                         \
  "addi         s6, s5, 8                \n\t"                         \
  "vfmacc.vv    v28, v16, v24            \n\t"                         \
  "vfmacc.vv    v29, v18, v25            \n\t"                         \
  "vfmacc.vv    v30, v20, v26            \n\t"                         \
  "vfmacc.vv    v31, v22, v27            \n\t"

#define __WEFT_IME1_LOAD_I4_I8_M1                                      \
  "vle8.v       v4, (s1)                 \n\t"                         \
  "addi         s1, s1, 128              \n\t"                         \
  "vle8.v       v5, (s2)                 \n\t"                         \
  "addi         s2, s2, 128              \n\t"                         \
  "vle8.v       v6, (s3)                 \n\t"                         \
  "addi         s3, s3, 128              \n\t"                         \
  "vle8.v       v7, (s4)                 \n\t"                         \
  "addi         s4, s4, 128              \n\t"                         \
  "vsetvli      t0, zero, e8, mf4        \n\t"                         \
  "vle8.v       v14, (s5)                \n\t"                         \
  "addi         s5, s5, 16               \n\t"                         \
  "vle8.v       v15, (s6)                \n\t"                         \
  "addi         s6, s6, 16               \n\t"                         \
  "addi         t5, t5, -1               \n\t"                         \
  "vsetvli      t0, zero, e8, m1         \n\t"                         \
  "vand.vi      v0, v4, 15               \n\t"                         \
  "vand.vi      v1, v5, 15               \n\t"                         \
  "vand.vi      v2, v6, 15               \n\t"                         \
  "vand.vi      v3, v7, 15               \n\t"                         \
  "vsrl.vi      v4, v4, 4                \n\t"                         \
  "vsrl.vi      v5, v5, 4                \n\t"                         \
  "vsrl.vi      v6, v6, 4                \n\t"                         \
  "vsrl.vi      v7, v7, 4                \n\t"

#define __WEFT_IME1_LOAD_ZP_I4_I8_M1                                   \
  "vsetvli      t0, zero, e8, mf2       \n\t"                          \
  "vle8.v       v1, (s7)                 \n\t"                          \
  "vsetvli      t0, zero, e8, m1        \n\t"                          \
  "vrgather.vv  v8, v1, v13              \n\t"                          \
  "vadd.vi      v13, v13, 4              \n\t"                          \
  "vrgather.vv  v9, v1, v13              \n\t"                          \
  "vadd.vi      v13, v13, 4              \n\t"                          \
  "vrgather.vv  v10, v1, v13             \n\t"                          \
  "vadd.vi      v13, v13, 4              \n\t"                          \
  "vrgather.vv  v11, v1, v13             \n\t"                          \
  "vadd.vi      v13, v13, -12            \n\t"

static inline __attribute__((unused)) void
__weft_ime1_symmetric_i4_i8_n16_k32(
    float activation_scale, const int8_t *activation_code,
    const uint8_t *packed_weight, float *accumulator) {
  __asm__ volatile(
        "vsetvli      t0, zero, e32, mf2      \n\t"
        "addi         s1, %[C], 16           \n\t"
        "addi         s2, %[C], 32           \n\t"
        "addi         s3, %[C], 48           \n\t"
        "vle32.v      v28, (%[C])            \n\t"
        "vle32.v      v29, (s1)              \n\t"
        "vle32.v      v30, (s2)              \n\t"
        "vle32.v      v31, (s3)              \n\t"
        "addi         s1, %[B], 0            \n\t"
        "addi         s2, %[B], 8            \n\t"
        "addi         s3, %[B], 16           \n\t"
        "addi         s4, %[B], 24           \n\t"
        "vsetvli      t0, zero, e16, mf4     \n\t"
        "vle16.v      v4, (s1)               \n\t"
        "vle16.v      v5, (s2)               \n\t"
        "vle16.v      v6, (s3)               \n\t"
        "vle16.v      v7, (s4)               \n\t"
        "vfwcvt.f.f.v v8, v4                 \n\t"
        "vfwcvt.f.f.v v9, v5                 \n\t"
        "vfwcvt.f.f.v v10, v6                \n\t"
        "vfwcvt.f.f.v v11, v7                \n\t"
        "vsetvli      t0, zero, e32, mf2     \n\t"
        "vxor.vv      v16, v16, v16          \n\t"
        "vxor.vv      v18, v18, v18          \n\t"
        "vxor.vv      v20, v20, v20          \n\t"
        "vxor.vv      v22, v22, v22          \n\t"
        "vfmul.vf     v24, v8, %[AS]         \n\t"
        "vfmul.vf     v25, v9, %[AS]         \n\t"
        "vfmul.vf     v26, v10, %[AS]        \n\t"
        "vfmul.vf     v27, v11, %[AS]        \n\t"
        "addi         s1, %[B], 32           \n\t"
        "addi         s2, %[B], 64           \n\t"
        "addi         s3, %[B], 96           \n\t"
        "addi         s4, %[B], 128          \n\t"
        "addi         s5, %[A], 0            \n\t"
        "addi         s6, %[A], 8            \n\t"
        "li           t5, 2                  \n\t"
        "vsetvli      t0, zero, e8, m1       \n\t"
        "LOOP_INNER%=:                       \n\t"
        __WEFT_IME1_LOAD_I4_I8_M1
        "vadd.vi      v0, v0, -8             \n\t"
        "vadd.vi      v1, v1, -8             \n\t"
        "vadd.vi      v2, v2, -8             \n\t"
        "vadd.vi      v3, v3, -8             \n\t"
        "vadd.vi      v4, v4, -8             \n\t"
        "vadd.vi      v5, v5, -8             \n\t"
        "vadd.vi      v6, v6, -8             \n\t"
        "vadd.vi      v7, v7, -8             \n\t"
        __WEFT_IME1_COMP_I4_I8_M1
        "bnez         t5, LOOP_INNER%=       \n\t"
        "vsetvli      t0, zero, e32, mf2     \n\t"
        __WEFT_IME1_ACC_I4_I8_M1
        "addi         s1, %[C], 16           \n\t"
        "addi         s2, %[C], 32           \n\t"
        "addi         s3, %[C], 48           \n\t"
        "vse32.v      v28, (%[C])            \n\t"
        "vse32.v      v29, (s1)              \n\t"
        "vse32.v      v30, (s2)              \n\t"
        "vse32.v      v31, (s3)              \n\t"
        :
        : [AS] "f"(activation_scale), [A] "r"(activation_code),
          [B] "r"(packed_weight), [C] "r"(accumulator)
        : "cc", "memory", "t0", "t5", "s1", "s2", "s3", "s4", "s5",
          "s6");
}

static inline __attribute__((unused)) void __weft_ime1_affine_i4_i8_n16(
    const float *activation_scale, const int8_t *activation_code,
    const uint8_t *packed_weight, float *output, size_t nblks,
    size_t block_count) {
  const size_t inner = 2;
  const uint8_t *weight = packed_weight;
  float *destination = output;
  const float *scale = activation_scale;
  size_t count = block_count;

  __asm__ volatile(
        "vsetvli      t0, zero, e32, m4       \n\t"
        "vxor.vv      v28, v28, v28           \n\t"
        "vsetvli      t0, zero, e8, m1        \n\t"
        "vmv.v.i      v13, 3                  \n\t"
        "li           s1, 24                  \n\t"
        "vsetvli      t0, s1, e8, m1          \n\t"
        "vmv.v.i      v13, 2                  \n\t"
        "vsetvli      t0, zero, e8, mf2       \n\t"
        "vmv.v.i      v13, 1                  \n\t"
        "vsetvli      t0, zero, e8, mf4       \n\t"
        "vmv.v.i      v13, 0                  \n\t"
        "addi         s1, %[B], 0             \n\t"
        "addi         s2, %[B], 8             \n\t"
        "addi         s3, %[B], 16            \n\t"
        "addi         s4, %[B], 24            \n\t"
        "addi         s7, %[B], 32            \n\t"
        "addi         s5, %[A], 0             \n\t"
        "addi         s6, %[A], 8             \n\t"
        "LOOP_K%=:                            \n\t"
        "vsetvli      t0, zero, e16, mf4      \n\t"
        "vle16.v      v4, (s1)                \n\t"
        "addi         s1, s1, 48              \n\t"
        "vle16.v      v5, (s2)                \n\t"
        "addi         s2, s2, 72              \n\t"
        "vle16.v      v6, (s3)                \n\t"
        "addi         s3, s3, 96              \n\t"
        "vle16.v      v7, (s4)                \n\t"
        "addi         s4, s4, 120             \n\t"
        "flw          f1, 0(%[AS])            \n\t"
        "addi         %[AS], %[AS], 4         \n\t"
        "vfwcvt.f.f.v v8, v4                  \n\t"
        "vfwcvt.f.f.v v9, v5                  \n\t"
        "vfwcvt.f.f.v v10, v6                 \n\t"
        "vfwcvt.f.f.v v11, v7                 \n\t"
        "vsetvli      t0, zero, e32, mf2      \n\t"
        "addi         t5, %[INNER], 0         \n\t"
        "vxor.vv      v16, v16, v16           \n\t"
        "vxor.vv      v18, v18, v18           \n\t"
        "vxor.vv      v20, v20, v20           \n\t"
        "vxor.vv      v22, v22, v22           \n\t"
        "vfmul.vf     v24, v8, f1             \n\t"
        "vfmul.vf     v25, v9, f1             \n\t"
        "vfmul.vf     v26, v10, f1            \n\t"
        "vfmul.vf     v27, v11, f1            \n\t"
        "addi         %[CNT], %[CNT], -1      \n\t"
        __WEFT_IME1_LOAD_ZP_I4_I8_M1
        "LOOP_INNER%=:                        \n\t"
        __WEFT_IME1_LOAD_I4_I8_M1
        "vsub.vv      v0, v0, v8              \n\t"
        "vsub.vv      v4, v4, v8              \n\t"
        "vsub.vv      v1, v1, v9              \n\t"
        "vsub.vv      v5, v5, v9              \n\t"
        "vsub.vv      v2, v2, v10             \n\t"
        "vsub.vv      v6, v6, v10             \n\t"
        "vsub.vv      v3, v3, v11             \n\t"
        "vsub.vv      v7, v7, v11             \n\t"
        __WEFT_IME1_COMP_I4_I8_M1
        "bnez         t5, LOOP_INNER%=        \n\t"
        "vsetvli      t0, zero, e32, mf2      \n\t"
        __WEFT_IME1_ACC_I4_I8_M1
        "addi         s7, s1, 32              \n\t"
        "bnez         %[CNT], LOOP_K%=        \n\t"
        "addi         t3, zero, 16            \n\t"
        "addi         s1, %[C], 16            \n\t"
        "addi         s2, %[C], 32            \n\t"
        "addi         s3, %[C], 48            \n\t"
        "blt          %[NBLKS], t3, ST_TAIL%= \n\t"
        "vse32.v      v28, (%[C])             \n\t"
        "vse32.v      v29, (s1)               \n\t"
        "vse32.v      v30, (s2)               \n\t"
        "vse32.v      v31, (s3)               \n\t"
        "jal          x0, END%=               \n\t"
        "ST_TAIL%=:                           \n\t"
        "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
        "sub          %[NBLKS], %[NBLKS], t0  \n\t"
        "vse32.v      v28, (%[C])             \n\t"
        "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
        "sub          %[NBLKS], %[NBLKS], t0  \n\t"
        "vse32.v      v29, (s1)               \n\t"
        "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
        "sub          %[NBLKS], %[NBLKS], t0  \n\t"
        "vse32.v      v30, (s2)               \n\t"
        "vsetvli      t0, %[NBLKS], e32, mf2  \n\t"
        "sub          %[NBLKS], %[NBLKS], t0  \n\t"
        "vse32.v      v31, (s3)               \n\t"
        "END%=:                               \n\t"
        : [CNT] "+r"(count), [NBLKS] "+r"(nblks), [AS] "+r"(scale)
        : [INNER] "r"(inner), [A] "r"(activation_code), [B] "r"(weight),
          [C] "r"(destination)
      : "cc", "memory", "t0", "t5", "t3", "f1", "s1", "s2", "s3",
        "s4", "s5", "s6", "s7");
}

#undef __WEFT_IME1_COMP_I4_I8_M1
#undef __WEFT_IME1_ACC_I4_I8_M1
#undef __WEFT_IME1_LOAD_I4_I8_M1
#undef __WEFT_IME1_LOAD_ZP_I4_I8_M1

)ime";
  }
  if (!usesExp)
    return;
  output << R"c(static inline vfloat32m2_t __weft_exp_f32m2(
    vfloat32m2_t x, size_t vl) {
  const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
  const vfloat32m2_t z =
      __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
  const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
  const vfloat32m2_t b0 =
      __riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl);
  const vfloat32m2_t b =
      __riscv_vfnmsac_vf_f32m2(b0, 0x1.7f7d1cp-20f, n, vl);
  const vuint32m2_t z_bits = __riscv_vreinterpret_v_f32m2_u32m2(z);
  const vuint32m2_t e = __riscv_vsll_vx_u32m2(z_bits, 23, vl);
  const vuint32m2_t k_bits =
      __riscv_vadd_vx_u32m2(e, UINT32_C(0x3f800000), vl);
  const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(k_bits);
  const vfloat32m2_t abs_n = __riscv_vfabs_v_f32m2(n, vl);
  const vbool16_t extreme =
      __riscv_vmfgt_vf_f32m2_b16(abs_n, 126.0f, vl);
  const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
  const vfloat32m2_t j0 =
      __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl);
  const vfloat32m2_t j1 = __riscv_vfmacc_vf_f32m2(
      __riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl),
      0x1.555e66p-3f, b, vl);
  const vfloat32m2_t j2 = __riscv_vfmacc_vf_f32m2(
      __riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl),
      0x1.0e4020p-7f, b, vl);
  const vfloat32m2_t j3 = __riscv_vfmacc_vv_f32m2(j1, j2, u, vl);
  const vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(j0, j3, u, vl);
  const vfloat32m2_t fast = __riscv_vfmacc_vv_f32m2(k, k, j, vl);
  if (__riscv_vcpop_m_b16(extreme, vl) == 0)
    return fast;

  const vbool16_t negative =
      __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
  const vuint32m2_t d = __riscv_vmerge_vxm_u32m2(
      __riscv_vmv_v_x_u32m2(0, vl), UINT32_C(0x82000000), negative, vl);
  const vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(
      __riscv_vadd_vx_u32m2(d, UINT32_C(0x7f000000), vl));
  const vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(
      __riscv_vsub_vv_u32m2(e, d, vl));
  const vfloat32m2_t corrected = __riscv_vfmul_vv_f32m2(
      __riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl);
  const vfloat32m2_t bounded =
      __riscv_vmerge_vvm_f32m2(fast, corrected, extreme, vl);
  const vbool16_t overflow = __riscv_vmfgt_vf_f32m2_b16(
      __riscv_vfabs_v_f32m2(n, vl), 192.0f, vl);
  return __riscv_vmerge_vvm_f32m2(
      bounded, __riscv_vfmul_vv_f32m2(s1, s1, vl), overflow, vl);
}

)c";
}

} // namespace

mlir::LogicalResult weft::lowerToRISCVIntrinsicC(
    mlir::ModuleOp module, const RISCVLoweringOptions &options,
    llvm::raw_ostream &output) {
  if (!options.target.hasRVV)
    return module.emitError(
        "the intrinsic C target requires RVV; no fallback backend is installed");
  bool usesExp = false;
  bool usesIME1 = false;
  bool usesGroupedI4I8 = false;
  module.walk([&](UnaryOp op) { usesExp |= op.getKind() == "exp"; });
  module.walk([&](AffineI4I8ContractOp) { usesIME1 = true; });
  module.walk([&](SymmetricI4I8ContractOp) { usesIME1 = true; });
  module.walk([&](GroupedAffineI4I8DotOp) { usesGroupedI4I8 = true; });
  emitPrelude(output, usesExp, usesIME1, usesGroupedI4I8);

  llvm::SmallVector<KernelOp> kernels;
  for (KernelOp kernel : module.getOps<KernelOp>())
    kernels.push_back(kernel);
  if (kernels.empty())
    return module.emitError("RISC-V lowering requires a Weft kernel");
  for (KernelOp kernel : kernels) {
    KernelEmitter emitter(kernel, options, output);
    if (mlir::failed(emitter.emit()))
      return mlir::failure();
  }
  return mlir::success();
}
