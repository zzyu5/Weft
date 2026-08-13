#include "Weft/Target/RISCVLowering.h"

#include "Weft/Dialect/Extension/IR/ExtensionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
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

#include "RISCVQuantCodebooks.inc"

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
  U8Vector,
  U32Vector,
  IndexVector,
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

enum class BlockStoreRealization {
  RVVE8MF4MicroStrips,
  RVVE8M1FixedStrips,
  RVVE8M1DynamicStrips,
};

struct BlockStorePhysicalDecision {
  BlockStoreRealization realization =
      BlockStoreRealization::RVVE8M1DynamicStrips;
  RVVBlockVectorShape byteShape = RVVBlockVectorShape::E8M1;
  unsigned stripVL = 16;
  bool needsLaneVector = false;
};

enum class BlockReduceRealization {
  RVVE8M1FixedStrips,
  RVVE8M1DynamicStrips,
};

struct BlockReducePhysicalDecision {
  BlockReduceRealization realization =
      BlockReduceRealization::RVVE8M1DynamicStrips;
  unsigned stripVL = 16;
  bool needsLaneVector = false;
};

struct BlockStoreGroupDecision {
  mlir::Operation *operation = nullptr;
  mlir::Operation *axis = nullptr;
  int64_t extent = 0;
  llvm::SmallVector<mlir::Operation *> stores;
  llvm::SmallVector<mlir::Operation *> closure;
  llvm::SmallVector<mlir::Operation *> interstitial;
  BlockStorePhysicalDecision physical;
};

struct BlockReduceGroupDecision {
  mlir::Operation *operation = nullptr;
  mlir::Operation *axis = nullptr;
  int64_t extent = 0;
  llvm::SmallVector<mlir::Operation *> reductions;
  llvm::SmallVector<mlir::Operation *> closure;
  BlockReducePhysicalDecision physical;
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

struct LocalBlockMemoryFact {
  mlir::Value semanticValue;
  mlir::Value base;
  BlockType semanticType;
  BlockType storageType;
  int64_t semanticExtent = 0;
  int64_t storageExtent = 0;
};

enum class SignBitI8Realization {
  RVVWideningSignSum,
};

struct SignBitI8Decision {
  SignBitI8Realization realization =
      SignBitI8Realization::RVVWideningSignSum;
  LocalBlockMemoryFact signBits;
  LocalBlockMemoryFact activation;
  mlir::Value activationScale;
  mlir::Value signScale;
  mlir::Value init;
};

enum class E2M1E8M0I8Realization {
  RVVVLEN128TableDot,
};

struct E2M1E8M0I8Decision {
  E2M1E8M0I8Realization realization =
      E2M1E8M0I8Realization::RVVVLEN128TableDot;
  LocalBlockMemoryFact packedCodes;
  LocalBlockMemoryFact activation;
  mlir::Value exponent;
  mlir::Value activationScale;
  mlir::Value init;
};

enum class GroupedAffineI4I8Realization {
  RVVVLEN128GroupedDot,
};

struct GroupedAffineI4I8Decision {
  GroupedAffineI4I8Realization realization =
      GroupedAffineI4I8Realization::RVVVLEN128GroupedDot;
  LocalBlockMemoryFact packedWeight;
  LocalBlockMemoryFact scaleMin;
  LocalBlockMemoryFact activation;
  LocalBlockMemoryFact activationSum;
  mlir::Value dotScale;
  mlir::Value minimumScale;
  mlir::Value init;
};

enum class QuantCodebookI8Realization {
  RVVVLEN128LocalBlockDot,
};

struct QuantCodebookI8Decision {
  QuantCodebookI8Realization realization =
      QuantCodebookI8Realization::RVVVLEN128LocalBlockDot;
  llvm::SmallVector<LocalBlockMemoryFact> blocks;
  llvm::SmallVector<mlir::Value> scalars;
};

enum class SortIndicesRealization {
  StableF32Radix,
};

struct SortIndicesDecision {
  SortIndicesRealization realization =
      SortIndicesRealization::StableF32Radix;
  bool descending = false;
  unsigned radixBits = 8;
  unsigned passes = 4;
};

struct CValue {
  mlir::Type type;
  CValueKind kind = CValueKind::Scalar;
  std::string spelling;
  bool lanePointer = false;
  std::vector<CValue> fields;
  std::string laneStride;
  bool indexedPointer = false;
  std::string indexSpelling;
  unsigned vectorSEW = 0;
  unsigned vectorLMUL = 0;
};

enum class LaneRelation {
  Independent,
  UnitStride,
  Strided,
  Indexed,
  NonAffine,
};

enum class VLAMemoryMode {
  UnitStride,
  Strided,
  Indexed,
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
  RVVWideningF16DotReduction,
  RVVInclusiveAddScan,
  RVVSegmentedInclusiveAddScan,
  RVVArgMaxSummary,
  RVVOnlineSoftmaxSummary,
};

enum class VLAStripSchedule {
  Dynamic,
  FullThenTail,
};

struct VLAPhysicalConfig {
  unsigned dataSEW = 32;
  unsigned dataLMUL = 2;
  unsigned indexSEW = 64;
  unsigned indexLMUL = 4;
  unsigned maskRatio = 16;
  VLAStripSchedule stripSchedule = VLAStripSchedule::Dynamic;
};

struct VLAPredicateDecision {
  mlir::Operation *operation = nullptr;
  mlir::Value coordinate;
  mlir::Value scalar;
  VLAMemoryMode coordinateMode = VLAMemoryMode::UnitStride;
  std::string predicate;
  enum class Realization {
    RVVAffineIndexScalar,
    RVVVectorScalar,
  } realization = Realization::RVVAffineIndexScalar;
  unsigned vectorSEW = 0;
  unsigned vectorLMUL = 0;
};

struct VLAAccessDecision {
  mlir::Operation *operation = nullptr;
  mlir::Type elementType;
  VLAMemoryMode memoryMode = VLAMemoryMode::UnitStride;
  VLAActivityMode activityMode = VLAActivityMode::AllActive;
  VLAStoreValueMode storeValueMode = VLAStoreValueMode::Vector;
  mlir::Value predicate;
  unsigned elementSEW = 0;
  unsigned elementLMUL = 0;
  mlir::Value indexedOffset;
  unsigned indexSEW = 0;
  unsigned indexLMUL = 0;
  unsigned indexedCompanionF32Vectors = 0;
};

enum class VLABinaryRealization {
  RVVF16FusedMultiplyAdd,
};

struct VLABinaryDecision {
  mlir::Operation *operation = nullptr;
  VLABinaryRealization realization =
      VLABinaryRealization::RVVF16FusedMultiplyAdd;
  mlir::Operation *absorbedProducer = nullptr;
  mlir::Value accumulator;
  mlir::Value vector;
  mlir::Value scalar;
  unsigned lmul = 8;
};

enum class VLACastRealization {
  RVVWidenF16ToF32,
  RVVNarrowF32ToF16,
  RVVZeroExtendU32ToIndex,
};

enum class RVVTailPolicy {
  Agnostic,
  Undisturbed,
};

struct RVVVectorConfig {
  unsigned sew = 32;
  unsigned lmul = 2;
};

struct VLACastDecision {
  mlir::Operation *operation = nullptr;
  VLACastRealization realization = VLACastRealization::RVVWidenF16ToF32;
  unsigned sourceSEW = 16;
  unsigned sourceLMUL = 1;
  unsigned resultSEW = 32;
  unsigned resultLMUL = 2;
};

enum class VLAStatePlacement {
  ScalarCarry,
  VectorCarry,
  StackScratch,
};

struct VLAStateDecision {
  mlir::Operation *operation = nullptr;
  VLAStateRealization realization = VLAStateRealization::RVVAddReduction;
  mlir::Type elementType;
  mlir::Value identity;
  VLAMemoryMode coordinateMode = VLAMemoryMode::UnitStride;
  unsigned dataLMUL = 2;
  unsigned laneIndexLMUL = 2;
  unsigned maskRatio = 16;
  mlir::Operation *lhsLoad = nullptr;
  mlir::Operation *rhsLoad = nullptr;
  RVVVectorConfig inputShape{16, 1};
  RVVVectorConfig computeShape{32, 2};
  RVVVectorConfig reductionShape{32, 1};
  RVVTailPolicy tail = RVVTailPolicy::Undisturbed;
  llvm::SmallVector<mlir::Operation *> absorbed;
  VLAStatePlacement placement = VLAStatePlacement::ScalarCarry;
  mlir::Value segmentStart;
  mlir::Value segmentVector;
  unsigned segmentSEW = 0;
  unsigned segmentLMUL = 0;
};

struct VLANarrowDecision {
  mlir::Operation *operation = nullptr;
  unsigned sourceLMUL = 8;
  unsigned intermediateLMUL = 4;
  unsigned resultLMUL = 2;
};

enum class VLAContractRealization {
  RVVF32FreeAxisMicrotile,
  RVVF32FreeAxisVectorDot,
};

enum class VLAContractInitRealization {
  ZeroVector,
  ScalarBroadcast,
  MaterializedRegion,
};

struct VLAContractDecision {
  mlir::Operation *operation = nullptr;
  VLAContractRealization realization =
      VLAContractRealization::RVVF32FreeAxisMicrotile;
  mlir::Operation *consumer = nullptr;
  mlir::Operation *lhsLoad = nullptr;
  mlir::Operation *rhsLoad = nullptr;
  mlir::Operation *freeLoad = nullptr;
  mlir::Value blockedOperand;
  mlir::Value init;
  mlir::Value rowAxis;
  mlir::Value reductionAxis;
  mlir::Value reductionExtent;
  unsigned rowTile = 1;
  unsigned lmul = 2;
  unsigned kUnroll = 1;
  VLAMemoryMode rhsMemoryMode = VLAMemoryMode::UnitStride;
  VLAMemoryMode freeMemoryMode = VLAMemoryMode::UnitStride;
  VLAMemoryMode outputMemoryMode = VLAMemoryMode::UnitStride;
  bool lhsPredicateVariesByReduction = false;
  VLAContractInitRealization initRealization =
      VLAContractInitRealization::ZeroVector;
  llvm::SmallVector<mlir::Operation *> absorbed;
};

struct VLARegionDecision {
  mlir::Operation *operation = nullptr;
  mlir::Value coordinate;
  VLAPhysicalConfig physical;
  std::vector<VLAPredicateDecision> predicates;
  std::vector<VLAAccessDecision> accesses;
  std::vector<VLABinaryDecision> binaries;
  std::vector<VLACastDecision> casts;
  std::vector<VLAStateDecision> states;
  std::vector<VLANarrowDecision> narrows;
  std::vector<VLAContractDecision> contracts;
};

struct VLACandidateFacts {
  unsigned f32Loads = 0;
  unsigned f32Stores = 0;
  unsigned stridedAccesses = 0;
  unsigned indexedAccesses = 0;
  unsigned maxIndexedElementSEW = 0;
  unsigned maxIndexedCompanionF32Vectors = 0;
  unsigned maxEntityF32Vectors = 0;
  bool hasReductionState = false;
  bool hasOrderedScan = false;
  bool hasCoordinateSummary = false;
  bool hasOnlineSummary = false;
  bool requiresF32M2Math = false;
};

std::string rvvFloatSuffix(const RVVVectorConfig &config) {
  return "f" + std::to_string(config.sew) + "m" +
         std::to_string(config.lmul);
}

std::string rvvFloatType(const RVVVectorConfig &config) {
  return "vfloat" + std::to_string(config.sew) + "m" +
         std::to_string(config.lmul) + "_t";
}

struct AffineI4I8NTileDecision {
  mlir::Operation *operation = nullptr;
  mlir::Value activationCodes;
  mlir::Value activationScales;
  mlir::Value packedWeight;
  mlir::Value output;
  mlir::Value columns;
  mlir::Value blocks;
  unsigned nTile = 16;
  unsigned packedBlockBytes = 304;
};

struct F16GemmNTileDecision {
  mlir::Operation *operation = nullptr;
  mlir::Value row;
  mlir::Value rowUpper;
  mlir::Value lhs;
  mlir::Value rhs;
  mlir::Value output;
  mlir::Value lhsStride;
  mlir::Value rhsStride;
  mlir::Value outputStride;
  mlir::Value nLower;
  mlir::Value nUpper;
  unsigned sourceNTile = 8;
  mlir::Value kLower;
  mlir::Value kUpper;
  unsigned sourceKTile = 64;
  unsigned rowTile = 4;
  unsigned rowMicrotile = 4;
  unsigned inputLMUL = 1;
  unsigned computeLMUL = 2;
  VLAStripSchedule kStripSchedule = VLAStripSchedule::FullThenTail;
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
  unsigned kUnroll = 1;
};

struct F32ContractPhysicalConfig {
  unsigned lmul = 1;
  unsigned kUnroll = 1;
};

enum class F32ContractResourceModel {
  VLAFreeAxis,
  LocalRow,
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
  if (lhs == LaneRelation::Indexed || rhs == LaneRelation::Indexed)
    return lhs == rhs ? LaneRelation::Indexed : LaneRelation::NonAffine;
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
    if (mlir::failed(preparePhysicalDecisions()))
      return mlir::failure();
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
  mlir::LogicalResult preparePhysicalDecisions() {
    bool decisionFailure = false;
    kernel.walk([&](SortIndicesOp op) {
      if (decisionFailure)
        return;
      SortIndicesDecision decision;
      if (!options.target.littleEndian || options.target.xlen != 64 ||
          !options.target.hasRVV) {
        op.emitError(
            "RISC-V sort_indices requires a little-endian RV64 vector target");
        decisionFailure = true;
        return;
      }
      decision.descending = op.getOrder() == "descending";
      if (options.backend.sortRadixBits != 0 &&
          options.backend.sortRadixBits != 8 &&
          options.backend.sortRadixBits != 11) {
        op.emitError("sort_indices radix width must be 8 or 11 bits");
        decisionFailure = true;
        return;
      }
      decision.radixBits = options.backend.sortRadixBits == 0
                               ? 8
                               : static_cast<unsigned>(
                                     options.backend.sortRadixBits);
      decision.passes = (32 + decision.radixBits - 1) / decision.radixBits;
      if (!sortIndicesDecisions
               .try_emplace(op.getOperation(), std::move(decision))
               .second) {
        op.emitError(
            "one sort_indices primitive cannot own multiple physical decisions");
        decisionFailure = true;
      }
    });
    kernel.walk([&](VLAOp vla) {
      if (decisionFailure)
        return;
      mlir::FailureOr<VLARegionDecision> decision = decideVLARegion(vla);
      if (mlir::failed(decision)) {
        decisionFailure = true;
        return;
      }
      if (!vlaDecisions.try_emplace(vla.getOperation(), std::move(*decision))
               .second) {
        vla.emitError("one VLA region cannot own multiple physical decisions");
        decisionFailure = true;
      }
    });
    kernel.walk([&](AffineI4I8ContractOp contract) {
      if (decisionFailure)
        return;
      ForOp kLoop = contract->getParentOfType<ForOp>();
      ForOp nLoop = kLoop ? kLoop->getParentOfType<ForOp>() : ForOp{};
      if (!nLoop)
        return;
      if (affineI4I8Decisions.contains(nLoop.getOperation())) {
        contract.emitError("one source N loop cannot own multiple affine IME decisions");
        decisionFailure = true;
        return;
      }
      std::optional<AffineI4I8NTileDecision> decision =
          decideAffineI4I8NTiles(contract);
      if (decision)
        affineI4I8Decisions.try_emplace(nLoop.getOperation(),
                                       std::move(*decision));
    });
    kernel.walk([&](MatmulOp matmul) {
      if (decisionFailure)
        return;
      ForOp kLoop = matmul->getParentOfType<ForOp>();
      ForOp nLoop = kLoop ? kLoop->getParentOfType<ForOp>() : ForOp{};
      if (!nLoop)
        return;
      if (f16ContractDecisions.contains(nLoop.getOperation())) {
        matmul.emitError("one source N loop cannot own multiple F16 matmul decisions");
        decisionFailure = true;
        return;
      }
      std::optional<F16GemmNTileDecision> decision =
          decideF16GemmNTiles(matmul);
      if (decision)
        f16ContractDecisions.try_emplace(nLoop.getOperation(),
                                         std::move(*decision));
    });
    kernel.walk([&](StoreOp store) {
      if (decisionFailure || isRegionValue(store.getValue().getType()))
        return;
      auto dot = store.getValue().getDefiningOp<DotOp>();
      if (!dot || !dot.getResult().hasOneUse())
        return;
      mlir::FailureOr<ContractDecision> decision =
          decideLocalF32RowMicrotile(store, dot);
      if (mlir::failed(decision)) {
        decisionFailure = true;
        return;
      }
      if (!localF32ContractDecisions
               .try_emplace(store.getOperation(), std::move(*decision))
               .second) {
        store.emitError("one store cannot own multiple local dot decisions");
        decisionFailure = true;
      }
    });
    kernel.walk([&](SymmetricI4I8ContractOp op) {
      if (decisionFailure)
        return;
      SymmetricI4I8Decision decision;
      if (mlir::failed(decideSymmetricI4I8Contract(op, decision))) {
        decisionFailure = true;
        return;
      }
      symmetricI4I8Decisions.try_emplace(op.getOperation(),
                                         std::move(decision));
    });
    kernel.walk([&](SignBitI8DotOp op) {
      if (decisionFailure)
        return;
      SignBitI8Decision decision;
      if (mlir::failed(decideSignBitI8Dot(op, decision))) {
        decisionFailure = true;
        return;
      }
      signBitI8Decisions.try_emplace(op.getOperation(), std::move(decision));
    });
    kernel.walk([&](E2M1E8M0I8DotOp op) {
      if (decisionFailure)
        return;
      E2M1E8M0I8Decision decision;
      if (mlir::failed(decideE2M1E8M0I8Dot(op, decision))) {
        decisionFailure = true;
        return;
      }
      e2m1E8M0I8Decisions.try_emplace(op.getOperation(),
                                      std::move(decision));
    });
    kernel.walk([&](GroupedAffineI4I8DotOp op) {
      if (decisionFailure)
        return;
      GroupedAffineI4I8Decision decision;
      if (mlir::failed(decideGroupedAffineI4I8Dot(op, decision))) {
        decisionFailure = true;
        return;
      }
      groupedAffineI4I8Decisions.try_emplace(op.getOperation(),
                                             std::move(decision));
    });
    auto prepareQuantCodebookDot = [&](auto op,
                                       llvm::ArrayRef<mlir::Value> blocks,
                                       llvm::ArrayRef<mlir::Value> scalars) {
      if (decisionFailure)
        return;
      QuantCodebookI8Decision decision;
      if (mlir::failed(
              decideQuantCodebookI8Dot(op, blocks, scalars, decision))) {
        decisionFailure = true;
        return;
      }
      quantCodebookI8Decisions.try_emplace(op.getOperation(),
                                           std::move(decision));
    };
    kernel.walk([&](IQ2SI8DotOp op) {
      prepareQuantCodebookDot(
          op,
          {op.getCodes(), op.getHighBits(), op.getSignBits(), op.getScales(),
           op.getActivation()},
          {op.getWeightScale(), op.getActivationScale(), op.getInit()});
    });
    kernel.walk([&](IQ3SI8DotOp op) {
      prepareQuantCodebookDot(
          op,
          {op.getCodes(), op.getHighBits(), op.getSignBits(), op.getScales(),
           op.getActivation()},
          {op.getWeightScale(), op.getActivationScale(), op.getInit()});
    });
    kernel.walk([&](IQ1MI8DotOp op) {
      prepareQuantCodebookDot(
          op,
          {op.getCodes(), op.getHighDeltaBits(), op.getScales(),
           op.getActivation()},
          {op.getActivationScale(), op.getInit()});
    });
    kernel.walk([&](Q6KI8DotOp op) {
      prepareQuantCodebookDot(
          op,
          {op.getLowBits(), op.getHighBits(), op.getGroupScales(),
           op.getActivation()},
          {op.getWeightScale(), op.getActivationScale(), op.getInit()});
    });
    kernel.walk([&](DecodeOp op) {
      if (decisionFailure)
        return;
      BlockDecodeDecision decision;
      if (mlir::failed(decideBlockDecode(op, decision))) {
        decisionFailure = true;
        return;
      }
      if (!blockDecodeDecisions
               .try_emplace(op.getOperation(), std::move(decision))
               .second) {
        op.emitError("one decode primitive cannot own multiple physical decisions");
        decisionFailure = true;
      }
    });
    if (!decisionFailure && mlir::failed(prepareBlockGroupDecisions()))
      decisionFailure = true;
    if (decisionFailure)
      return mlir::failure();
    return mlir::success();
  }
  KernelOp kernel;
  const RISCVLoweringOptions &options;
  llvm::raw_ostream &output;
  llvm::DenseMap<mlir::Value, CValue> values;
  llvm::DenseMap<mlir::Operation *, VLARegionDecision> vlaDecisions;
  llvm::DenseMap<mlir::Operation *, AffineI4I8NTileDecision>
      affineI4I8Decisions;
  llvm::DenseMap<mlir::Operation *, F16GemmNTileDecision>
      f16ContractDecisions;
  llvm::DenseMap<mlir::Operation *, ContractDecision>
      localF32ContractDecisions;
  llvm::DenseMap<mlir::Operation *, SymmetricI4I8Decision>
      symmetricI4I8Decisions;
  llvm::DenseMap<mlir::Operation *, SignBitI8Decision> signBitI8Decisions;
  llvm::DenseMap<mlir::Operation *, E2M1E8M0I8Decision>
      e2m1E8M0I8Decisions;
  llvm::DenseMap<mlir::Operation *, GroupedAffineI4I8Decision>
      groupedAffineI4I8Decisions;
  llvm::DenseMap<mlir::Operation *, QuantCodebookI8Decision>
      quantCodebookI8Decisions;
  llvm::DenseMap<mlir::Operation *, BlockDecodeDecision> blockDecodeDecisions;
  llvm::DenseMap<mlir::Operation *, SortIndicesDecision> sortIndicesDecisions;
  llvm::DenseMap<mlir::Operation *, BlockStoreGroupDecision>
      blockStoreGroupDecisions;
  llvm::DenseMap<mlir::Operation *, BlockReduceGroupDecision>
      blockReduceGroupDecisions;
  llvm::DenseSet<mlir::Operation *> consumed;
  llvm::DenseSet<mlir::Operation *> deferredBlockOps;
  llvm::DenseSet<mlir::Operation *> loweredBlockOps;
  unsigned indent = 0;
  unsigned nextValue = 0;
  unsigned nextLoop = 0;
  bool inVLA = false;
  bool activeVLAFullStrip = false;
  RVVBlockVectorShape activeBlockByteShape = RVVBlockVectorShape::E8M1;
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

  const VLABinaryDecision *
  findBinaryDecision(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return nullptr;
    auto found = llvm::find_if(
        activeVLADecision->binaries, [&](const VLABinaryDecision &decision) {
          return decision.operation == operation;
        });
    return found == activeVLADecision->binaries.end() ? nullptr : &*found;
  }

  bool isAbsorbedBinaryProducer(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return false;
    return llvm::any_of(
        activeVLADecision->binaries, [&](const VLABinaryDecision &decision) {
          return decision.absorbedProducer == operation;
        });
  }

  bool isVLAStateAbsorbed(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return false;
    return llvm::any_of(
        activeVLADecision->states, [&](const VLAStateDecision &decision) {
          return llvm::is_contained(decision.absorbed, operation);
        });
  }

  const VLACastDecision *
  findCastDecision(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return nullptr;
    auto found = llvm::find_if(
        activeVLADecision->casts, [&](const VLACastDecision &decision) {
          return decision.operation == operation;
        });
    return found == activeVLADecision->casts.end() ? nullptr : &*found;
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

  std::optional<F32ContractPhysicalConfig>
  selectF32ContractPhysicalConfig(F32ContractResourceModel model,
                                  unsigned rowTile,
                                  unsigned defaultUnroll) const {
    constexpr unsigned vlaCandidates[] = {4, 2, 1};
    constexpr unsigned narrowRowCandidates[] = {4, 2, 1};
    constexpr unsigned mediumRowCandidates[] = {2, 4, 1};
    constexpr unsigned wideRowCandidates[] = {1, 2, 4};
    llvm::ArrayRef<unsigned> candidates = vlaCandidates;
    if (model == F32ContractResourceModel::LocalRow)
      candidates = rowTile <= 4   ? llvm::ArrayRef<unsigned>(narrowRowCandidates)
                   : rowTile <= 6 ? llvm::ArrayRef<unsigned>(mediumRowCandidates)
                                  : llvm::ArrayRef<unsigned>(wideRowCandidates);
    llvm::SmallVector<unsigned> unrollCandidates;
    if (options.backend.contractKUnroll != 0)
      unrollCandidates.push_back(
          static_cast<unsigned>(options.backend.contractKUnroll));
    else {
      unrollCandidates.push_back(defaultUnroll);
      for (unsigned candidate : {1u, 2u, 4u})
        if (!llvm::is_contained(unrollCandidates, candidate))
          unrollCandidates.push_back(candidate);
    }
    llvm::SmallVector<unsigned> requestedLMUL;
    if (options.backend.contractLMUL != 0) {
      requestedLMUL.push_back(
          static_cast<unsigned>(options.backend.contractLMUL));
      candidates = requestedLMUL;
    }
    auto legalUnroll = [](unsigned value) {
      return value == 1 || value == 2 || value == 4;
    };
    auto legalLMUL = [](unsigned value) {
      return value == 1 || value == 2 || value == 4 || value == 8;
    };
    for (unsigned unroll : unrollCandidates) {
      if (!legalUnroll(unroll))
        continue;
      for (unsigned lmul : candidates) {
        if (!legalLMUL(lmul))
          continue;
        unsigned accumulatorGroups = rowTile * lmul;
        unsigned streamedOperandGroups =
            (model == F32ContractResourceModel::LocalRow ? 2 : 1) * unroll *
            lmul;
        unsigned total = accumulatorGroups + streamedOperandGroups;
        bool hasAllocatorHeadroom =
            model == F32ContractResourceModel::VLAFreeAxis
                ? total <= options.target.vectorRegisters
                : total + 1 < options.target.vectorRegisters;
        if (hasAllocatorHeadroom)
          return F32ContractPhysicalConfig{lmul, unroll};
      }
    }
    return std::nullopt;
  }

  mlir::FailureOr<VLAContractDecision>
  decideVLAF32FreeAxisMicrotile(VLAOp vla, StoreOp store,
                                 DotOp dot) {
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

    BlockType lhsType = unwrapBlock(dot.getLhs().getType());
    RegionType rhsType = unwrapRegion(dot.getRhs().getType());
    RegionType resultType = unwrapRegion(dot.getResult().getType());
    if (store.getValue() != dot.getResult() ||
        !dot.getResult().hasOneUse() || !lhsType || !rhsType ||
        !resultType || lhsType.getShape().size() != 2 ||
        rhsType.getShape().size() != 2 || resultType.getShape().size() != 2 ||
        rhsType.getShape()[0] != -1 || resultType.getShape()[0] != -1 ||
        lhsType.getShape()[0] != resultType.getShape()[1] ||
        lhsType.getShape()[1] != rhsType.getShape()[1] ||
        lhsType.getShape()[0] <= 0 || lhsType.getShape()[0] > 6 ||
        !lhsType.getElementType().isF32() ||
        !rhsType.getElementType().isF32() ||
        !resultType.getElementType().isF32() ||
        !isFloatConstant(dot.getInit(), 0.0) ||
        dot.getOrder() != "relaxed" || dot.getMath() != "native" ||
        !dot.getAccDtype().isF32()) {
      dot.emitError(
          "RVV VLA dot requires a [BM,K] x [VLA,K] local f32 primitive");
      return mlir::failure();
    }

    LoadOp lhsLoad = dot.getLhs().getDefiningOp<LoadOp>();
    LoadOp rhsLoad = dot.getRhs().getDefiningOp<LoadOp>();
    if (!lhsLoad || !rhsLoad || !isTrue(rhsLoad.getWhere()) ||
        (!isTrue(lhsLoad.getWhere()) &&
         !isFloatConstant(lhsLoad.getOther(), 0.0))) {
      dot.emitError("RVV VLA dot load facts are unavailable");
      return mlir::failure();
    }

    mlir::Block &block = *store->getBlock();
    llvm::DenseSet<mlir::Operation *> lhsDefinitions;
    llvm::DenseSet<mlir::Operation *> rhsDefinitions;
    llvm::DenseSet<mlir::Operation *> outputDefinitions;
    llvm::SmallVector<BlockAxisOp> lhsAxes;
    llvm::SmallVector<BlockAxisOp> rhsAxes;
    llvm::SmallVector<BlockAxisOp> outputAxes;
    collectLocalDefinitions(dot.getLhs(), block, lhsDefinitions, lhsAxes);
    collectLocalDefinitions(dot.getRhs(), block, rhsDefinitions, rhsAxes);
    collectLocalDefinitions(store.getPointer(), block, outputDefinitions,
                            outputAxes);
    collectLocalDefinitions(store.getWhere(), block, outputDefinitions,
                            outputAxes);
    if (rhsAxes.size() != 1 || lhsAxes.size() != 2 || outputAxes.size() != 1 ||
        !llvm::is_contained(lhsAxes, rhsAxes.front())) {
      dot.emitError(
          "RVV VLA dot requires explicit row and shared reduction axes");
      return mlir::failure();
    }
    BlockAxisOp reductionAxis = rhsAxes.front();
    BlockAxisOp rowAxis = lhsAxes.front() == reductionAxis
                              ? lhsAxes.back()
                              : lhsAxes.front();
    if (outputAxes.front() != rowAxis ||
        integerConstantValue(rowAxis.getExtent()) != lhsType.getShape()[0]) {
      dot.emitError(
          "RVV VLA dot result axes do not preserve the local row block");
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
      dot.emitError(
          "RVV VLA dot memory relations are unavailable");
      return mlir::failure();
    }

    llvm::DenseSet<mlir::Operation *> absorbed;
    llvm::SmallVector<BlockAxisOp> absorbedAxes;
    for (mlir::Value value : {store.getPointer(), store.getValue(),
                              store.getWhere()})
      collectLocalDefinitions(value, block, absorbed, absorbedAxes);

    VLAContractDecision decision;
    decision.operation = dot.getOperation();
    decision.realization =
        VLAContractRealization::RVVF32FreeAxisMicrotile;
    decision.consumer = store.getOperation();
    decision.lhsLoad = lhsLoad.getOperation();
    decision.rhsLoad = rhsLoad.getOperation();
    decision.init = dot.getInit();
    decision.rowAxis = rowAxis.getResult();
    decision.reductionAxis = reductionAxis.getResult();
    decision.reductionExtent = reductionAxis.getExtent();
    decision.rowTile = static_cast<unsigned>(lhsType.getShape()[0]);
    std::optional<F32ContractPhysicalConfig> physical =
        selectF32ContractPhysicalConfig(F32ContractResourceModel::VLAFreeAxis,
                                        decision.rowTile, 1);
    if (!physical) {
      dot.emitError(
          "RVV VLA dot has no legal register microtile candidate");
      return mlir::failure();
    }
    decision.lmul = physical->lmul;
    decision.kUnroll = physical->kUnroll;
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

  mlir::FailureOr<VLAContractDecision>
  decideVLAF32FreeAxisVectorDot(VLAOp vla, StoreOp store,
                                DotOp dot) {
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

    RegionType lhsType = unwrapRegion(dot.getLhs().getType());
    BlockType rhsType = unwrapBlock(dot.getRhs().getType());
    RegionType initType = unwrapRegion(dot.getInit().getType());
    RegionType resultType = unwrapRegion(dot.getResult().getType());
    if (store.getValue() != dot.getResult() ||
        !dot.getResult().hasOneUse() || !lhsType || !rhsType ||
        !initType || !resultType || lhsType.getShape().size() != 2 ||
        rhsType.getShape().size() != 1 || initType.getShape().size() != 1 ||
        resultType.getShape().size() != 1 || lhsType.getShape()[0] != -1 ||
        initType.getShape()[0] != -1 || resultType.getShape()[0] != -1 ||
        lhsType.getShape()[1] != rhsType.getShape()[0] ||
        !lhsType.getElementType().isF32() ||
        !rhsType.getElementType().isF32() ||
        !initType.getElementType().isF32() ||
        !resultType.getElementType().isF32() ||
        dot.getOrder() != "relaxed" || dot.getMath() != "native" ||
        !dot.getAccDtype().isF32()) {
      dot.emitError(
          "RVV VLA vector dot requires a [VLA,K] x [K] local f32 primitive");
      return mlir::failure();
    }

    LoadOp freeLoad = dot.getLhs().getDefiningOp<LoadOp>();
    if (!freeLoad || !isTrue(freeLoad.getWhere())) {
      dot.emitError("RVV VLA vector-dot load facts are unavailable");
      return mlir::failure();
    }

    mlir::Block &block = *store->getBlock();
    llvm::DenseSet<mlir::Operation *> freeDefinitions;
    llvm::DenseSet<mlir::Operation *> blockedDefinitions;
    llvm::DenseSet<mlir::Operation *> outputDefinitions;
    llvm::SmallVector<BlockAxisOp> freeAxes;
    llvm::SmallVector<BlockAxisOp> blockedAxes;
    llvm::SmallVector<BlockAxisOp> outputAxes;
    collectLocalDefinitions(dot.getLhs(), block, freeDefinitions, freeAxes);
    collectLocalDefinitions(dot.getRhs(), block, blockedDefinitions,
                            blockedAxes);
    collectLocalDefinitions(store.getPointer(), block, outputDefinitions,
                            outputAxes);
    collectLocalDefinitions(store.getWhere(), block, outputDefinitions,
                            outputAxes);
    if (freeAxes.size() != 1 || blockedAxes.size() != 1 ||
        freeAxes.front() != blockedAxes.front() || !outputAxes.empty()) {
      dot.emitError(
          "RVV VLA vector dot requires one shared explicit reduction axis");
      return mlir::failure();
    }
    BlockAxisOp reductionAxis = freeAxes.front();

    mlir::Value coordinate = vla.getBody().front().getArgument(0);
    mlir::Value freeRoot = pointerRoot(freeLoad.getPointer());
    mlir::Value outputRoot = pointerRoot(store.getPointer());
    auto freePointer =
        freeRoot ? mlir::dyn_cast<PtrType>(freeRoot.getType()) : PtrType{};
    auto outputPointer =
        outputRoot ? mlir::dyn_cast<PtrType>(outputRoot.getType()) : PtrType{};
    LaneRelation freeRelation =
        classifyLaneRelation(freeLoad.getPointer(), coordinate);
    LaneRelation outputRelation =
        classifyLaneRelation(store.getPointer(), coordinate);
    if (!freePointer || !outputPointer ||
        !freePointer.getElementType().isF32() ||
        !outputPointer.getElementType().isF32() ||
        (freeRelation != LaneRelation::UnitStride &&
         freeRelation != LaneRelation::Strided) ||
        (outputRelation != LaneRelation::UnitStride &&
         outputRelation != LaneRelation::Strided) ||
        !dependsOn(freeLoad.getPointer(), coordinate) ||
        !dependsOn(freeLoad.getPointer(), reductionAxis.getResult()) ||
        !dependsOn(dot.getInit(), coordinate) ||
        dependsOn(dot.getInit(), reductionAxis.getResult()) ||
        !dependsOn(store.getPointer(), coordinate) ||
        dependsOn(store.getPointer(), reductionAxis.getResult()) ||
        !isTrue(store.getWhere()) ||
        dependsOn(dot.getRhs(), coordinate) ||
        !dependsOn(dot.getRhs(), reductionAxis.getResult())) {
      dot.emitError("RVV VLA vector-dot memory relations are unavailable");
      return mlir::failure();
    }

    llvm::DenseSet<mlir::Operation *> absorbed;
    llvm::SmallVector<BlockAxisOp> absorbedAxes;
    for (mlir::Value value : {store.getPointer(), store.getValue(),
                              store.getWhere()})
      collectLocalDefinitions(value, block, absorbed, absorbedAxes);
    llvm::DenseSet<mlir::Operation *> initDefinitions;
    llvm::SmallVector<BlockAxisOp> initAxes;
    collectLocalDefinitions(dot.getInit(), block, initDefinitions, initAxes);
    for (mlir::Operation *operation : initDefinitions)
      absorbed.erase(operation);

    VLAContractDecision decision;
    decision.operation = dot.getOperation();
    decision.realization = VLAContractRealization::RVVF32FreeAxisVectorDot;
    decision.consumer = store.getOperation();
    decision.freeLoad = freeLoad.getOperation();
    decision.blockedOperand = dot.getRhs();
    decision.init = dot.getInit();
    decision.reductionAxis = reductionAxis.getResult();
    decision.reductionExtent = reductionAxis.getExtent();
    decision.rowTile = 1;
    std::optional<F32ContractPhysicalConfig> physical =
        selectF32ContractPhysicalConfig(F32ContractResourceModel::VLAFreeAxis,
                                        decision.rowTile, 1);
    if (!physical) {
      dot.emitError(
          "RVV VLA vector dot has no legal register candidate");
      return mlir::failure();
    }
    decision.lmul = physical->lmul;
    decision.kUnroll = physical->kUnroll;
    decision.freeMemoryMode = freeRelation == LaneRelation::UnitStride
                                  ? VLAMemoryMode::UnitStride
                                  : VLAMemoryMode::Strided;
    decision.outputMemoryMode = outputRelation == LaneRelation::UnitStride
                                    ? VLAMemoryMode::UnitStride
                                    : VLAMemoryMode::Strided;
    decision.initRealization =
        VLAContractInitRealization::MaterializedRegion;
    decision.absorbed.append(absorbed.begin(), absorbed.end());
    return decision;
  }

  mlir::FailureOr<VLARegionDecision> decideVLARegion(VLAOp op) {
    mlir::Block &body = op.getBody().front();
    VLARegionDecision decision;
    decision.operation = op.getOperation();
    decision.coordinate = body.getArgument(0);
    if (options.target.xlen == 32) {
      decision.physical.indexSEW = 32;
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
      auto dot =
          store ? store.getValue().getDefiningOp<DotOp>() : DotOp{};
      if (!dot || !isRegionValue(dot.getResult().getType()))
        continue;
      mlir::FailureOr<VLAContractDecision> selected =
          isRegionValue(dot.getLhs().getType()) &&
                  containsBlockType(dot.getRhs().getType())
              ? decideVLAF32FreeAxisVectorDot(op, store, dot)
              : decideVLAF32FreeAxisMicrotile(op, store, dot);
      if (mlir::failed(selected))
        return mlir::failure();
      decision.contracts.push_back(std::move(*selected));
    }
    if (!decision.contracts.empty()) {
      decision.physical.dataLMUL = decision.contracts.front().lmul;
      if (!llvm::all_of(decision.contracts,
                        [&](const VLAContractDecision &dot) {
                          return dot.lmul == decision.physical.dataLMUL;
                        })) {
        op.emitError("one VLA region requires one shared dot LMUL");
        return mlir::failure();
      }
    }

    auto isContractOwned = [&](mlir::Operation *operation) {
      return llvm::any_of(
          decision.contracts, [&](const VLAContractDecision &dot) {
            return dot.consumer == operation ||
                   llvm::is_contained(dot.absorbed, operation);
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
      bool lhsVector = isRegionValue(compare.getLhs().getType());
      bool rhsVector = isRegionValue(compare.getRhs().getType());
      bool affineIndex = lhs != LaneRelation::NonAffine &&
                         rhs != LaneRelation::NonAffine &&
                         lhsCoordinate != rhsCoordinate &&
                         elementType(lhsCoordinate ? compare.getLhs().getType()
                                                   : compare.getRhs().getType())
                             .isIndex();
      bool vectorScalar = lhsVector != rhsVector &&
                          elementType(lhsVector ? compare.getLhs().getType()
                                                : compare.getRhs().getType()) ==
                              elementType(lhsVector ? compare.getRhs().getType()
                                                    : compare.getLhs().getType()) &&
                          !isRegionValue(lhsVector ? compare.getRhs().getType()
                                                   : compare.getLhs().getType());
      if (!affineIndex && !vectorScalar) {
        compare.emitError(
            "VLA predicate requires an affine index or typed vector/scalar relation");
        return mlir::failure();
      }
      bool lhsSelected = affineIndex ? lhsCoordinate : lhsVector;
      std::string predicate = lhsSelected
                                  ? compare.getPredicate().str()
                                  : reversePredicate(compare.getPredicate());
      if (predicate.empty()) {
        compare.emitError("VLA predicate has an unsupported comparison");
        return mlir::failure();
      }
      VLAPredicateDecision predicateDecision;
      predicateDecision.operation = compare.getOperation();
      predicateDecision.coordinate =
          lhsSelected ? compare.getLhs() : compare.getRhs();
      predicateDecision.scalar =
          lhsSelected ? compare.getRhs() : compare.getLhs();
      predicateDecision.predicate = std::move(predicate);
      if (affineIndex) {
        predicateDecision.coordinateMode =
            (lhsCoordinate ? lhs : rhs) == LaneRelation::UnitStride
                ? VLAMemoryMode::UnitStride
                : VLAMemoryMode::Strided;
      } else {
        predicateDecision.realization =
            VLAPredicateDecision::Realization::RVVVectorScalar;
        mlir::Type vectorElement =
            elementType(predicateDecision.coordinate.getType());
        if (vectorElement.isF32())
          predicateDecision.vectorSEW = 32;
        else if (vectorElement.isUnsignedInteger(8))
          predicateDecision.vectorSEW = 8;
        else {
          compare.emitError(
              "typed VLA vector predicate has no RVV element realization");
          return mlir::failure();
        }
      }
      decision.predicates.push_back(std::move(predicateDecision));
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
        llvm::DenseSet<mlir::Value> visitedPredicates;
        std::function<bool(mlir::Value)> hasPredicateDecision =
            [&](mlir::Value value) {
              if (!value || !visitedPredicates.insert(value).second)
                return false;
              if (llvm::any_of(
                      decision.predicates,
                      [&](const VLAPredicateDecision &candidate) {
                        return candidate.operation == value.getDefiningOp();
                      }))
                return true;
              auto binary = value.getDefiningOp<BinaryOp>();
              return binary &&
                     (binary.getKind() == "and" || binary.getKind() == "or" ||
                      binary.getKind() == "xor") &&
                     hasPredicateDecision(binary.getLhs()) &&
                     hasPredicateDecision(binary.getRhs());
            };
        if (hasPredicateDecision(predicate)) {
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
      VLAAccessDecision access;
      access.operation = operation;
      access.elementType = accessedElement;
      access.memoryMode = relation == LaneRelation::UnitStride
                              ? VLAMemoryMode::UnitStride
                              : relation == LaneRelation::Strided
                                    ? VLAMemoryMode::Strided
                                    : VLAMemoryMode::Indexed;
      access.activityMode = activityMode;
      access.storeValueMode = storeValueMode;
      access.predicate = predicate;
      if (relation == LaneRelation::Indexed) {
        auto pointerAdd = pointer.getDefiningOp<PtrAddOp>();
        if (!pointerAdd ||
            classifyLaneRelation(pointerAdd.getBase(), decision.coordinate) !=
                LaneRelation::Independent ||
            classifyLaneRelation(pointerAdd.getOffset(), decision.coordinate) !=
                LaneRelation::Indexed) {
          return operation->emitError(
              "indexed VLA memory requires one scalar base and one typed index vector");
        }
        access.indexedOffset = pointerAdd.getOffset();
        auto indexCast = access.indexedOffset.getDefiningOp<CastOp>();
        if (!indexCast ||
            !elementType(indexCast.getInput().getType()).isUnsignedInteger(32)) {
          return operation->emitError(
              "indexed VLA memory requires an explicit u32 index vector");
        }
        access.indexSEW = 32;
        if (auto load = mlir::dyn_cast<LoadOp>(operation)) {
          mlir::Operation *indexDefinition = indexCast.getInput().getDefiningOp();
          for (mlir::Operation *user : load.getResult().getUsers()) {
            if (user == indexDefinition)
              continue;
            unsigned companions = llvm::count_if(
                user->getOperands(), [&](mlir::Value operand) {
                  return operand != load.getResult() &&
                         isRegionValue(operand.getType()) &&
                         elementType(operand.getType()).isF32();
                });
            bool feedsReduction = llvm::any_of(
                user->getResults(), [](mlir::Value result) {
                  return llvm::any_of(result.getUsers(), [](mlir::Operation *next) {
                    return mlir::isa<ReduceOp>(next);
                  });
                });
            access.indexedCompanionF32Vectors = std::max(
                access.indexedCompanionF32Vectors,
                companions + static_cast<unsigned>(feedsReduction));
          }
        }
      }
      decision.accesses.push_back(std::move(access));
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
        LaneRelation storeRelation =
            classifyLaneRelation(store.getPointer(), decision.coordinate);
        mlir::Type storedElement = elementType(store.getValue().getType());
        if (storeRelation == LaneRelation::Indexed) {
          store.emitError(
              "indexed VLA store has no selected target realization");
          return mlir::failure();
        }
        if (storedElement.isUnsignedInteger(32)) {
          store.emitError(
              "unsigned VLA store has no selected target realization");
          return mlir::failure();
        }
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
    for (const VLAAccessDecision &access : decision.accesses) {
      if (mlir::isa<LoadOp>(access.operation) &&
          access.activityMode == VLAActivityMode::PredicateMask) {
        access.operation->emitError(
            "masked VLA load has no selected target realization");
        return mlir::failure();
      }
    }

    unsigned maxEntityF32Vectors = 0;
    for (mlir::Operation *operation : physicalOperations) {
      unsigned entityVectors = llvm::count_if(
          operation->getOperandTypes(), [](mlir::Type type) {
            return isRegionValue(type) && elementType(type).isF32();
          });
      entityVectors += llvm::count_if(
          operation->getResultTypes(), [](mlir::Type type) {
            return isRegionValue(type) && elementType(type).isF32();
          });
      if (auto loop = mlir::dyn_cast<ForOp>(operation))
        entityVectors += llvm::count_if(
            loop.getBody().front().getArgumentTypes(), [](mlir::Type type) {
              return isRegionValue(type) && elementType(type).isF32();
            });
      maxEntityF32Vectors = std::max(maxEntityF32Vectors, entityVectors);
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
      llvm::SmallVector<unsigned> candidates;
      if (options.backend.narrowLMUL != 0)
        candidates.push_back(static_cast<unsigned>(options.backend.narrowLMUL));
      else
        candidates = {8, 4};
      std::optional<unsigned> selected;
      for (unsigned candidate : candidates) {
        if (candidate != 4 && candidate != 8)
          continue;
        unsigned narrowGroups = candidate + candidate / 2 + candidate / 4;
        unsigned regionGroups = maxEntityF32Vectors * candidate;
        if (std::max(narrowGroups, regionGroups) + 1 <
            static_cast<unsigned>(options.target.vectorRegisters)) {
          selected = candidate;
          break;
        }
      }
      if (!selected) {
        narrow.emitError("VLA narrow requested an illegal f32 LMUL candidate");
        return mlir::failure();
      }
      unsigned sourceLMUL = *selected;
      decision.physical.dataLMUL = sourceLMUL;
      decision.narrows.push_back(VLANarrowDecision{
          narrow.getOperation(), sourceLMUL, sourceLMUL / 2,
          sourceLMUL / 4});
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
        VLAStateDecision state;
        state.operation = reduce.getOperation();
        state.elementType = reduce.getResult().getType();
        state.identity = reduce.getIdentity();
        auto multiply = reduce.getInput().getDefiningOp<BinaryOp>();
        auto lhsCast = multiply && multiply.getKind() == "mul"
                           ? multiply.getLhs().getDefiningOp<CastOp>()
                           : CastOp{};
        auto rhsCast = multiply && multiply.getKind() == "mul"
                           ? multiply.getRhs().getDefiningOp<CastOp>()
                           : CastOp{};
        auto lhsLoad = lhsCast ? lhsCast.getInput().getDefiningOp<LoadOp>()
                               : LoadOp{};
        auto rhsLoad = rhsCast ? rhsCast.getInput().getDefiningOp<LoadOp>()
                               : LoadOp{};
        bool wideningF16Dot =
            reduce.getKind() == "add" && multiply && lhsCast && rhsCast &&
            lhsLoad && rhsLoad &&
            elementType(lhsCast.getResult().getType()).isF32() &&
            elementType(rhsCast.getResult().getType()).isF32() &&
            isF16(elementType(lhsLoad.getResult().getType())) &&
            isF16(elementType(rhsLoad.getResult().getType())) &&
            isTrue(lhsLoad.getWhere()) && isTrue(rhsLoad.getWhere()) &&
            lhsLoad.getResult().hasOneUse() &&
            rhsLoad.getResult().hasOneUse() &&
            lhsCast.getResult().hasOneUse() &&
            rhsCast.getResult().hasOneUse() && multiply.getResult().hasOneUse() &&
            classifyLaneRelation(lhsLoad.getPointer(), decision.coordinate) ==
                LaneRelation::UnitStride &&
            classifyLaneRelation(rhsLoad.getPointer(), decision.coordinate) ==
                LaneRelation::UnitStride;
        if (wideningF16Dot) {
          state.realization =
              VLAStateRealization::RVVWideningF16DotReduction;
          state.lhsLoad = lhsLoad.getOperation();
          state.rhsLoad = rhsLoad.getOperation();
          state.absorbed = {lhsLoad.getOperation(), rhsLoad.getOperation(),
                            lhsCast.getOperation(), rhsCast.getOperation(),
                            multiply.getOperation()};
          state.placement = VLAStatePlacement::VectorCarry;
        } else if (reduce.getKind() == "add") {
          state.realization = VLAStateRealization::RVVAddReduction;
          state.placement = reduce.getOrder() == "relaxed" &&
                                    !op.getEnd().getDefiningOp<BinaryOp>() &&
                                    !integerConstantValue(op.getEnd()) &&
                                    !op->getParentOfType<IfOp>()
                                ? VLAStatePlacement::VectorCarry
                                : VLAStatePlacement::ScalarCarry;
        } else if (reduce.getKind() == "max") {
          state.realization = VLAStateRealization::RVVMaxReduction;
          state.placement = reduce.getOrder() == "relaxed" &&
                                    !op.getEnd().getDefiningOp<BinaryOp>() &&
                                    !integerConstantValue(op.getEnd()) &&
                                    !op->getParentOfType<IfOp>()
                                ? VLAStatePlacement::VectorCarry
                                : VLAStatePlacement::ScalarCarry;
        } else {
          reduce.emitError("VLA reduction kind has no physical realization");
          return mlir::failure();
        }
        decision.states.push_back(std::move(state));
      } else if (auto scan = mlir::dyn_cast<ScanOp>(nested)) {
        bool unsegmented =
            mlir::isa<mlir::NoneType>(scan.getSegmentStart().getType());
        bool segmented = !unsegmented &&
                         isRegionValue(scan.getSegmentStart().getType()) &&
                         elementType(scan.getSegmentStart().getType())
                             .isInteger(1) &&
                         llvm::any_of(
                             decision.predicates,
                             [&](const VLAPredicateDecision &predicate) {
                               return predicate.operation ==
                                      scan.getSegmentStart().getDefiningOp();
                             });
        if (!elementType(scan.getInput().getType()).isF32() ||
            !elementType(scan.getResult().getType()).isF32() ||
            !scan.getIdentity().getType().isF32() || !isTrue(scan.getWhere()) ||
            (!unsegmented && !segmented) ||
            scan.getKind() != "add" || !scan.getInclusive() ||
            scan.getOrder() != "ordered") {
          scan.emitError(
              "VLA scan has no selected all-active ordered f32 realization");
          return mlir::failure();
        }
        VLAStateDecision state{
            scan.getOperation(),
            segmented ? VLAStateRealization::RVVSegmentedInclusiveAddScan
                      : VLAStateRealization::RVVInclusiveAddScan,
            elementType(scan.getResult().getType()), scan.getIdentity()};
        state.segmentStart = scan.getSegmentStart();
        auto segmentCompare = scan.getSegmentStart().getDefiningOp<CompareOp>();
        if (segmentCompare)
          state.segmentVector =
              isRegionValue(segmentCompare.getLhs().getType())
                  ? segmentCompare.getLhs()
                  : segmentCompare.getRhs();
        decision.states.push_back(std::move(state));
      } else if (auto summary = mlir::dyn_cast<ArgMaxOp>(nested)) {
          LaneRelation coordinate =
              classifyLaneRelation(summary.getCoordinate(), decision.coordinate);
          if (coordinate != LaneRelation::UnitStride &&
              coordinate != LaneRelation::Strided) {
            summary.emitError("argmax coordinate is not affine in the VLA axis");
            return mlir::failure();
          }
          VLAStateDecision state{
              summary.getOperation(), VLAStateRealization::RVVArgMaxSummary,
              summary.getResult().getType()};
          state.coordinateMode = coordinate == LaneRelation::UnitStride
                                     ? VLAMemoryMode::UnitStride
                                     : VLAMemoryMode::Strided;
          decision.states.push_back(state);
      } else if (auto summary =
                     mlir::dyn_cast<OnlineSoftmaxSummaryOp>(nested)) {
          VLAStateDecision state{
              summary.getOperation(),
              VLAStateRealization::RVVOnlineSoftmaxSummary,
              summary.getResult().getType()};
          decision.states.push_back(std::move(state));
      } else if (auto summary = mlir::dyn_cast<SummaryFoldOp>(nested)) {
        summary.emitError(
            "RISC-V target does not implement this generic summary algebra");
        return mlir::failure();
      }
    }
    if (!decision.narrows.empty() && !decision.states.empty()) {
      op.emitError(
          "one VLA region cannot currently share a narrow and aggregate realization");
      return mlir::failure();
    }

    bool hasF32RegionValue = llvm::any_of(
        physicalOperations, [](mlir::Operation *operation) {
          return llvm::any_of(operation->getResultTypes(), [](mlir::Type type) {
            return isRegionValue(type) && elementType(type).isF32();
          });
        });
    bool hasWideningF16Dot = llvm::any_of(
        decision.states, [](const VLAStateDecision &state) {
          return state.realization ==
                 VLAStateRealization::RVVWideningF16DotReduction;
      });
    bool onlyF16Accesses = !decision.accesses.empty() &&
                           llvm::all_of(decision.accesses,
                                        [](const VLAAccessDecision &access) {
                                          return isF16(access.elementType);
                                        });
    bool hasFloatCast = llvm::any_of(
        physicalOperations, [](mlir::Operation *operation) {
          auto cast = mlir::dyn_cast<CastOp>(operation);
          if (!cast || !isRegionValue(cast.getResult().getType()))
            return false;
          mlir::Type source = elementType(cast.getInput().getType());
          mlir::Type result = elementType(cast.getResult().getType());
          return (isF16(source) && result.isF32()) ||
                 (source.isF32() && isF16(result));
        });
    bool requiresF32M2Math = llvm::any_of(
        physicalOperations, [](mlir::Operation *operation) {
          auto unary = mlir::dyn_cast<UnaryOp>(operation);
          return unary && unary.getKind() == "exp" &&
                 isRegionValue(unary.getResult().getType());
        });
    VLACandidateFacts candidateFacts;
    candidateFacts.maxEntityF32Vectors = maxEntityF32Vectors;
    candidateFacts.requiresF32M2Math = requiresF32M2Math;
    for (const VLAAccessDecision &access : decision.accesses) {
      candidateFacts.stridedAccesses +=
          access.memoryMode == VLAMemoryMode::Strided;
      candidateFacts.indexedAccesses +=
          access.memoryMode == VLAMemoryMode::Indexed;
      if (access.memoryMode == VLAMemoryMode::Indexed) {
        unsigned elementSEW = access.elementType.isF32() ? 32
                              : isF16(access.elementType) ? 16
                              : access.elementType.isUnsignedInteger(8) ? 8
                              : access.elementType.isUnsignedInteger(32) ? 32
                                                                        : 0;
        if (elementSEW == 0) {
          access.operation->emitError(
              "indexed VLA memory element has no typed resource model");
          return mlir::failure();
        }
        candidateFacts.maxIndexedElementSEW =
            std::max(candidateFacts.maxIndexedElementSEW, elementSEW);
        candidateFacts.maxIndexedCompanionF32Vectors = std::max(
            candidateFacts.maxIndexedCompanionF32Vectors,
            access.indexedCompanionF32Vectors);
      }
      if (access.elementType.isF32()) {
        candidateFacts.f32Loads += mlir::isa<LoadOp>(access.operation);
        candidateFacts.f32Stores += mlir::isa<StoreOp>(access.operation);
      }
    }
    for (const VLAStateDecision &state : decision.states) {
      candidateFacts.hasReductionState |=
          state.realization == VLAStateRealization::RVVAddReduction ||
          state.realization == VLAStateRealization::RVVMaxReduction;
      candidateFacts.hasOrderedScan |=
          state.realization == VLAStateRealization::RVVInclusiveAddScan ||
          state.realization ==
              VLAStateRealization::RVVSegmentedInclusiveAddScan;
      candidateFacts.hasCoordinateSummary |=
          state.realization == VLAStateRealization::RVVArgMaxSummary;
      candidateFacts.hasOnlineSummary |=
          state.realization == VLAStateRealization::RVVOnlineSoftmaxSummary;
    }
    if (decision.contracts.empty() && decision.narrows.empty() &&
        decision.states.empty() && decision.predicates.empty() &&
        !hasF32RegionValue && onlyF16Accesses) {
      decision.physical.dataSEW = 16;
      decision.physical.dataLMUL = 8;
    } else if (hasWideningF16Dot) {
      decision.physical.dataSEW = 32;
      decision.physical.dataLMUL = 2;
      decision.physical.stripSchedule = VLAStripSchedule::FullThenTail;
    } else if (decision.contracts.empty() && decision.narrows.empty() &&
               decision.states.empty() && decision.predicates.empty() &&
               hasFloatCast && !requiresF32M2Math) {
      decision.physical.dataSEW = 32;
      decision.physical.dataLMUL = 8;
    } else if (decision.contracts.empty() && decision.narrows.empty() &&
               !hasWideningF16Dot && maxEntityF32Vectors > 0) {
      llvm::SmallVector<unsigned> candidates;
      if (options.backend.vlaLMUL != 0) {
        unsigned requested = static_cast<unsigned>(options.backend.vlaLMUL);
        if (requested != 1 && requested != 2 && requested != 4 &&
            requested != 8) {
          op.emitError("VLA requested an illegal LMUL candidate");
          return mlir::failure();
        }
        candidates.push_back(requested);
      } else {
        if (candidateFacts.requiresF32M2Math ||
            candidateFacts.hasOnlineSummary)
          candidates = {2};
        else if (candidateFacts.hasOrderedScan)
          candidates = {1, 2, 4, 8};
        else if (candidateFacts.hasCoordinateSummary)
          candidates = {4, 2, 1, 8};
        else if (candidateFacts.hasReductionState)
          candidates = {8, 4, 2, 1};
        else if (candidateFacts.stridedAccesses > 0)
          candidates = {2, 4, 1, 8};
        else
          candidates = {4, 2, 8, 1};
      }
      std::optional<unsigned> selected;
      for (unsigned candidate : candidates) {
        if (requiresF32M2Math && candidate != 2)
          continue;
        bool hasAffinePredicate = llvm::any_of(
            decision.predicates, [](const VLAPredicateDecision &predicate) {
              return predicate.realization ==
                     VLAPredicateDecision::Realization::RVVAffineIndexScalar;
            });
        if (hasAffinePredicate && options.target.xlen == 64 &&
            candidate * 2 > 8)
          continue;
        bool elementShapesLegal = llvm::all_of(
            decision.accesses, [&](const VLAAccessDecision &access) {
              unsigned sew = access.elementType.isF32() ? 32
                             : isF16(access.elementType) ? 16
                             : access.elementType.isUnsignedInteger(8) ? 8
                             : access.elementType.isUnsignedInteger(32) ? 32
                             : access.elementType.isSignedInteger(8) ? 8
                                                                    : 0;
              return sew != 0 &&
                     (candidate * sew) % decision.physical.dataSEW == 0 &&
                     candidate * sew / decision.physical.dataSEW <= 8;
            });
        if (!elementShapesLegal)
          continue;
        unsigned primitiveGroups = 0;
        for (const VLAStateDecision &state : decision.states) {
          if (state.realization == VLAStateRealization::RVVInclusiveAddScan)
            primitiveGroups = std::max(primitiveGroups, 3 * candidate + 1);
          else if (state.realization ==
                   VLAStateRealization::RVVSegmentedInclusiveAddScan)
            primitiveGroups = std::max(primitiveGroups, 4 * candidate + 2);
          else if (state.realization == VLAStateRealization::RVVArgMaxSummary ||
                   state.realization == VLAStateRealization::RVVAddReduction ||
                   state.realization == VLAStateRealization::RVVMaxReduction)
            primitiveGroups = std::max(primitiveGroups, candidate + 2);
          else if (state.realization ==
                   VLAStateRealization::RVVOnlineSoftmaxSummary)
            primitiveGroups = std::max(primitiveGroups, 3 * candidate + 2);
        }
        unsigned indexGroups = 0;
        if (candidateFacts.indexedAccesses != 0) {
          unsigned indexLMUL = candidate * 32 / decision.physical.dataSEW;
          unsigned elementLMUL = candidate * candidateFacts.maxIndexedElementSEW /
                                 decision.physical.dataSEW;
          indexGroups = std::max(2 * indexLMUL + elementLMUL,
                                 2 * candidate) +
                        candidateFacts.maxIndexedCompanionF32Vectors * candidate;
        }
        unsigned predicateGroups = 0;
        for (const VLAPredicateDecision &predicate : decision.predicates) {
          if (predicate.realization ==
              VLAPredicateDecision::Realization::RVVVectorScalar) {
            unsigned scaled = candidate * predicate.vectorSEW;
            if (scaled % decision.physical.dataSEW != 0) {
              predicateGroups = options.target.vectorRegisters;
              break;
            }
            predicateGroups += scaled / decision.physical.dataSEW;
          } else {
            predicateGroups += options.target.xlen == 64 ? candidate * 2
                                                         : candidate;
          }
        }
        unsigned regionGroups = maxEntityF32Vectors * candidate;
        unsigned requiredGroups =
            std::max(regionGroups,
                     std::max(primitiveGroups, indexGroups + predicateGroups));
        if (requiredGroups + 1 <
            static_cast<unsigned>(options.target.vectorRegisters)) {
          selected = candidate;
          break;
        }
      }
      if (!selected) {
        op.emitError("VLA has no legal LMUL candidate for its entity resources");
        return mlir::failure();
      }
      decision.physical.dataLMUL = *selected;
    }
    if (options.backend.vlaLMUL != 0 && decision.contracts.empty() &&
        !hasWideningF16Dot && decision.narrows.empty() &&
        maxEntityF32Vectors == 0) {
      unsigned requested = static_cast<unsigned>(options.backend.vlaLMUL);
      if (requested != 1 && requested != 2 && requested != 4 &&
          requested != 8) {
        op.emitError("VLA requested an illegal LMUL candidate");
        return mlir::failure();
      }
      unsigned registerGroups = decision.accesses.size() * requested;
      if ((!decision.predicates.empty() && options.target.xlen == 64 &&
           requested * 2 > 8) ||
          registerGroups + 1 >=
              static_cast<unsigned>(options.target.vectorRegisters)) {
        op.emitError("VLA requested LMUL exceeds entity resources");
        return mlir::failure();
      }
      decision.physical.dataLMUL = requested;
    }
    if (options.backend.vlaLMUL != 0 &&
        decision.physical.dataSEW == 16 &&
        decision.contracts.empty() && decision.narrows.empty() &&
        !hasWideningF16Dot) {
      unsigned requested = static_cast<unsigned>(options.backend.vlaLMUL);
      if (requested != 1 && requested != 2 && requested != 4 &&
          requested != 8) {
        op.emitError("VLA requested an illegal LMUL candidate");
        return mlir::failure();
      }
      unsigned registerGroups = decision.accesses.size() * requested;
      if (registerGroups + 1 >=
          static_cast<unsigned>(options.target.vectorRegisters)) {
        op.emitError("VLA requested LMUL exceeds entity resources");
        return mlir::failure();
      }
      decision.physical.dataLMUL = requested;
    }

    for (mlir::Operation *operation : physicalOperations) {
      auto cast = mlir::dyn_cast<CastOp>(operation);
      if (!cast || !isRegionValue(cast.getResult().getType()))
        continue;
      mlir::Type source = elementType(cast.getInput().getType());
      mlir::Type result = elementType(cast.getResult().getType());
      VLACastDecision selected;
      selected.operation = cast.getOperation();
      if (isF16(source) && result.isF32()) {
        selected.realization = VLACastRealization::RVVWidenF16ToF32;
        selected.sourceSEW = 16;
        selected.resultSEW = 32;
      } else if (source.isF32() && isF16(result)) {
        selected.realization = VLACastRealization::RVVNarrowF32ToF16;
        selected.sourceSEW = 32;
        selected.resultSEW = 16;
      } else if (source.isUnsignedInteger(32) && result.isIndex()) {
        selected.realization =
            VLACastRealization::RVVZeroExtendU32ToIndex;
        selected.sourceSEW = 32;
        selected.resultSEW = 32;
      } else {
        cast.emitError("VLA cast has no selected RVV conversion");
        return mlir::failure();
      }
      selected.sourceLMUL =
          decision.physical.dataLMUL * selected.sourceSEW /
          decision.physical.dataSEW;
      selected.resultLMUL =
          decision.physical.dataLMUL * selected.resultSEW /
          decision.physical.dataSEW;
      if (selected.sourceLMUL == 0 || selected.sourceLMUL > 8 ||
          selected.resultLMUL == 0 || selected.resultLMUL > 8) {
        cast.emitError("VLA cast exceeds the legal RVV LMUL range");
        return mlir::failure();
      }
      decision.casts.push_back(selected);
    }

    for (mlir::Operation *operation : physicalOperations) {
      auto add = mlir::dyn_cast<BinaryOp>(operation);
      if (!add || add.getKind() != "add" ||
          !isF16(elementType(add.getResult().getType())))
        continue;
      BinaryOp multiply = add.getLhs().getDefiningOp<BinaryOp>();
      mlir::Value accumulator = add.getRhs();
      if (!multiply || multiply.getKind() != "mul" ||
          !multiply.getResult().hasOneUse()) {
        multiply = add.getRhs().getDefiningOp<BinaryOp>();
        accumulator = add.getLhs();
      }
      if (!multiply || multiply.getKind() != "mul" ||
          !multiply.getResult().hasOneUse() ||
          !isF16(elementType(accumulator.getType())))
        continue;
      mlir::Value vector = multiply.getLhs();
      mlir::Value scalar = multiply.getRhs();
      if (!isRegionValue(vector.getType()))
        std::swap(vector, scalar);
      if (!isRegionValue(vector.getType()) || isRegionValue(scalar.getType()) ||
          !isF16(elementType(vector.getType())) ||
          !isF16(elementType(scalar.getType())))
        continue;
      decision.binaries.push_back(VLABinaryDecision{
          add.getOperation(), VLABinaryRealization::RVVF16FusedMultiplyAdd,
          multiply.getOperation(), accumulator, vector, scalar,
          decision.physical.dataLMUL});
    }

    decision.physical.maskRatio =
        decision.physical.dataSEW / decision.physical.dataLMUL;
    decision.physical.indexLMUL = 0;
    bool hasAffinePredicate = llvm::any_of(
        decision.predicates, [](const VLAPredicateDecision &predicate) {
          return predicate.realization ==
                 VLAPredicateDecision::Realization::RVVAffineIndexScalar;
        });
    if (hasAffinePredicate) {
      decision.physical.indexLMUL = options.target.xlen == 64
                                        ? decision.physical.dataLMUL * 2
                                        : decision.physical.dataLMUL;
      if (decision.physical.indexLMUL > 8) {
        op.emitError("selected VLA data LMUL has no legal index-vector LMUL");
        return mlir::failure();
      }
    }

    for (VLAAccessDecision &access : decision.accesses) {
      if (access.elementType.isF32()) {
        access.elementSEW = 32;
      } else if (isF16(access.elementType)) {
        access.elementSEW = 16;
      } else if (access.elementType.isSignedInteger(8) ||
                 access.elementType.isUnsignedInteger(8)) {
        access.elementSEW = 8;
      } else if (access.elementType.isUnsignedInteger(32)) {
        access.elementSEW = 32;
      } else {
        access.operation->emitError(
            "VLA memory element has no selected RVV vector shape");
        return mlir::failure();
      }
      unsigned scaledLMUL =
          decision.physical.dataLMUL * access.elementSEW;
      if (scaledLMUL % decision.physical.dataSEW != 0) {
        access.operation->emitError(
            "VLA memory element has no compatible selected LMUL");
        return mlir::failure();
      }
      access.elementLMUL = scaledLMUL / decision.physical.dataSEW;
      if (access.elementLMUL == 0 || access.elementLMUL > 8) {
        access.operation->emitError(
            "VLA memory element exceeds the legal RVV LMUL range");
        return mlir::failure();
      }
      if (access.memoryMode == VLAMemoryMode::Indexed) {
        mlir::Type indexElement = elementType(access.indexedOffset.getType());
        if (!indexElement.isIndex()) {
          access.operation->emitError(
              "indexed VLA memory offset is not an index region");
          return mlir::failure();
        }
        auto cast = access.indexedOffset.getDefiningOp<CastOp>();
        if (!cast ||
            !elementType(cast.getInput().getType()).isUnsignedInteger(32)) {
          access.operation->emitError(
              "indexed VLA memory currently requires an explicit u32-to-index cast");
          return mlir::failure();
        }
        access.indexSEW = 32;
        auto selectedCast = llvm::find_if(
            decision.casts, [&](const VLACastDecision &candidate) {
              return candidate.operation == cast.getOperation();
            });
        if (selectedCast == decision.casts.end() ||
            selectedCast->realization !=
                VLACastRealization::RVVZeroExtendU32ToIndex) {
          access.operation->emitError(
              "indexed VLA memory has no matching index cast decision");
          return mlir::failure();
        }
        access.indexLMUL = selectedCast->sourceLMUL;
      }
    }

    for (VLAPredicateDecision &predicate : decision.predicates) {
      if (predicate.realization !=
          VLAPredicateDecision::Realization::RVVVectorScalar)
        continue;
      unsigned scaledLMUL =
          decision.physical.dataLMUL * predicate.vectorSEW;
      if (scaledLMUL % decision.physical.dataSEW != 0) {
        predicate.operation->emitError(
            "typed VLA predicate has no compatible selected LMUL");
        return mlir::failure();
      }
      predicate.vectorLMUL = scaledLMUL / decision.physical.dataSEW;
    }

    for (VLAStateDecision &state : decision.states) {
      if (state.realization ==
          VLAStateRealization::RVVWideningF16DotReduction)
        continue;
      state.dataLMUL = decision.physical.dataLMUL;
      state.laneIndexLMUL = decision.physical.dataLMUL;
      state.maskRatio = decision.physical.maskRatio;
      if (state.realization ==
          VLAStateRealization::RVVSegmentedInclusiveAddScan) {
        const VLAPredicateDecision *segmentPredicate = nullptr;
        auto found = llvm::find_if(
            decision.predicates, [&](const VLAPredicateDecision &predicate) {
              return predicate.operation == state.segmentStart.getDefiningOp();
            });
        if (found != decision.predicates.end())
          segmentPredicate = &*found;
        if (!segmentPredicate ||
            segmentPredicate->realization !=
                VLAPredicateDecision::Realization::RVVVectorScalar) {
          state.operation->emitError(
              "segmented scan has no typed segment predicate decision");
          return mlir::failure();
        }
        state.segmentSEW = segmentPredicate->vectorSEW;
        state.segmentLMUL = segmentPredicate->vectorLMUL;
      }
      if (state.realization ==
              VLAStateRealization::RVVOnlineSoftmaxSummary &&
          state.dataLMUL != 2) {
        state.operation->emitError(
            "online softmax summary requires the selected f32m2 exp realization");
        return mlir::failure();
      }
    }
    return decision;
  }

  mlir::LogicalResult emitOperation(mlir::Operation *operation) {
    if (consumed.contains(operation))
      return mlir::success();
    if (isVLAStateAbsorbed(operation))
      return mlir::success();
    if (const VLAContractDecision *decision =
            findVLAContractDecision(operation))
      return emitVLAContract(*decision);
    if (isVLAContractAbsorbed(operation))
      return mlir::success();
    if (auto op = mlir::dyn_cast<ReduceOp>(operation))
      return emitReduce(op);
    if (auto op = mlir::dyn_cast<SortIndicesOp>(operation))
      return emitSortIndices(op);
    if (auto op = mlir::dyn_cast<ScanOp>(operation))
      return op.emitError("scan must be lowered by its enclosing VLA region");
    if (auto op = mlir::dyn_cast<GroupedAffineI4I8DotOp>(operation))
      return emitGroupedAffineI4I8Dot(op);
    if (auto op = mlir::dyn_cast<SignBitI8DotOp>(operation))
      return emitSignBitI8Dot(op);
    if (auto op = mlir::dyn_cast<E2M1E8M0I8DotOp>(operation))
      return emitE2M1E8M0I8Dot(op);
    if (auto op = mlir::dyn_cast<IQ2SI8DotOp>(operation))
      return emitQuantCodebookI8Dot(op, "__weft_iq2_s_i8_vl128");
    if (auto op = mlir::dyn_cast<IQ3SI8DotOp>(operation))
      return emitQuantCodebookI8Dot(op, "__weft_iq3_s_i8_vl128");
    if (auto op = mlir::dyn_cast<IQ1MI8DotOp>(operation))
      return emitQuantCodebookI8Dot(op, "__weft_iq1_m_i8_vl128");
    if (auto op = mlir::dyn_cast<Q6KI8DotOp>(operation))
      return emitQuantCodebookI8Dot(op, "__weft_q6_k_i8_vl128");
    if (auto op = mlir::dyn_cast<SymmetricI4I8ContractOp>(operation))
      return emitSymmetricI4I8Contract(op);
    if (mlir::isa<MatmulOp>(operation))
      return operation->emitError(
          "matmul must be consumed by its prepared local physical decision");
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
    if (auto op = mlir::dyn_cast<ArgMaxOp>(operation))
      return op.emitError("argmax must be lowered by its enclosing VLA region");
    if (auto op = mlir::dyn_cast<OnlineSoftmaxSummaryOp>(operation))
      return op.emitError(
          "online_softmax_summary must be lowered by its enclosing VLA region");
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

  mlir::LogicalResult emitSortIndices(SortIndicesOp op) {
    auto found = sortIndicesDecisions.find(op.getOperation());
    if (found == sortIndicesDecisions.end())
      return op.emitError("sort_indices has no physical decision");
    const SortIndicesDecision &decision = found->second;
    if (decision.realization != SortIndicesRealization::StableF32Radix ||
        (decision.radixBits != 8 && decision.radixBits != 11) ||
        decision.passes !=
            (32 + decision.radixBits - 1) / decision.radixBits)
      return op.emitError("sort_indices decision has no intrinsic-C spelling");
    CValue input = require(op.getInput());
    CValue outputValue = require(op.getOutput());
    CValue extent = require(op.getExtent());
    if (input.kind != CValueKind::Pointer ||
        outputValue.kind != CValueKind::Pointer ||
        input.spelling.empty() || outputValue.spelling.empty() ||
        extent.kind != CValueKind::Scalar || extent.spelling.empty())
      return op.emitError("sort_indices operands are unavailable");

    std::string scratch = fresh("sort_scratch");
    std::string source = fresh("sort_source");
    std::string destination = fresh("sort_destination");
    std::string position = fresh("sort_position");
    std::string pass = fresh("sort_pass");
    std::string histogram = fresh("sort_histogram");
    std::string offsets = fresh("sort_offsets");
    std::string index = fresh("sort_index");
    std::string bits = fresh("sort_bits");
    std::string normalized = fresh("sort_normalized");
    std::string ordered = fresh("sort_ordered");
    std::string key = fresh("sort_key");
    std::string digit = fresh("sort_digit");
    std::string bucket = fresh("sort_bucket");
    std::string running = fresh("sort_running");
    std::string swap = fresh("sort_swap");
    std::string descendingTransform = decision.descending ? "~" : "";
    unsigned bucketCount = 1u << decision.radixBits;
    std::string digitMask = std::to_string(bucketCount - 1) + "U";

    line("uint32_t " + scratch + "[" + extent.spelling + "]; ");
    line("for (size_t " + position + " = 0; " + position + " < " +
         extent.spelling + "; ++" + position + ")");
    ++indent;
    line(outputValue.spelling + "[" + position + "] = (uint32_t)" +
         position + ";");
    --indent;
    line("uint32_t *" + source + " = " + outputValue.spelling + ";");
    line("uint32_t *" + destination + " = " + scratch + ";");
    line("for (unsigned " + pass + " = 0; " + pass + " < " +
         std::to_string(decision.passes) + "; ++" + pass + ") {");
    ++indent;
    line("size_t " + histogram + "[" + std::to_string(bucketCount) +
         "] = {0};");
    line("for (size_t " + position + " = 0; " + position + " < " +
         extent.spelling + "; ++" + position + ") {");
    ++indent;
    line("const uint32_t " + index + " = " + source + "[" + position + "]; ");
    line("const uint32_t " + bits + " = __weft_bitcast_f32_u32(" +
         input.spelling + "[" + index + "]); ");
    line("const uint32_t " + normalized + " = ((" + bits +
         " & UINT32_C(0x7fffffff)) == 0) ? 0 : " + bits + ";");
    line("const uint32_t " + ordered + " = " + normalized + " ^ ((" +
         normalized +
         " & UINT32_C(0x80000000)) ? UINT32_C(0xffffffff) : "
         "UINT32_C(0x80000000));");
    line("const uint32_t " + key + " = (((" + bits +
         " & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) ? "
         "UINT32_C(0xffffffff) : " + descendingTransform + ordered + ");");
    line("const unsigned " + digit + " = (unsigned)((" + key + " >> (" +
         pass + " * " + std::to_string(decision.radixBits) + "U)) & " +
         digitMask + ");");
    line("++" + histogram + "[" + digit + "]; ");
    --indent;
    line("}");
    line("size_t " + offsets + "[" + std::to_string(bucketCount) + "]; ");
    line("size_t " + running + " = 0;");
    line("for (unsigned " + bucket + " = 0; " + bucket + " < " +
         std::to_string(bucketCount) + "; ++" + bucket + ") {");
    ++indent;
    line(offsets + "[" + bucket + "] = " + running + ";");
    line(running + " += " + histogram + "[" + bucket + "]; ");
    --indent;
    line("}");
    line("for (size_t " + position + " = 0; " + position + " < " +
         extent.spelling + "; ++" + position + ") {");
    ++indent;
    line("const uint32_t " + index + " = " + source + "[" + position + "]; ");
    line("const uint32_t " + bits + " = __weft_bitcast_f32_u32(" +
         input.spelling + "[" + index + "]); ");
    line("const uint32_t " + normalized + " = ((" + bits +
         " & UINT32_C(0x7fffffff)) == 0) ? 0 : " + bits + ";");
    line("const uint32_t " + ordered + " = " + normalized + " ^ ((" +
         normalized +
         " & UINT32_C(0x80000000)) ? UINT32_C(0xffffffff) : "
         "UINT32_C(0x80000000));");
    line("const uint32_t " + key + " = (((" + bits +
         " & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) ? "
         "UINT32_C(0xffffffff) : " + descendingTransform + ordered + ");");
    line("const unsigned " + digit + " = (unsigned)((" + key + " >> (" +
         pass + " * " + std::to_string(decision.radixBits) + "U)) & " +
         digitMask + ");");
    line(destination + "[" + offsets + "[" + digit + "]++] = " + index +
         ";");
    --indent;
    line("}");
    line("uint32_t *" + swap + " = " + source + ";");
    line(source + " = " + destination + ";");
    line(destination + " = " + swap + ";");
    --indent;
    line("}");
    if (decision.passes % 2 != 0) {
      line("for (size_t " + position + " = 0; " + position + " < " +
           extent.spelling + "; ++" + position + ")");
      ++indent;
      line(outputValue.spelling + "[" + position + "] = " + source + "[" +
           position + "]; ");
      --indent;
    }
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
    auto affineI4I8Decision = affineI4I8Decisions.find(op.getOperation());
    if (affineI4I8Decision != affineI4I8Decisions.end())
      return emitAffineI4I8NTiles(affineI4I8Decision->second);
    auto f16Decision = f16ContractDecisions.find(op.getOperation());
    if (f16Decision != f16ContractDecisions.end())
      return emitF16GemmNTiles(f16Decision->second);
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
        line("vfloat32m" + std::to_string(activeVLADecision->physical.dataLMUL) +
             "_t " + value.spelling + " = " + init.spelling + ";");
      } else if (inVLA && init.kind == CValueKind::Mask) {
        value = CValue{result.getType(), CValueKind::Mask, fresh("carry")};
        line("vbool" + std::to_string(activeVLADecision->physical.maskRatio) + "_t " +
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
          type = "vfloat32m" + std::to_string(activeVLADecision->physical.dataLMUL) + "_t";
        else
          type = "vbool" + std::to_string(activeVLADecision->physical.maskRatio) + "_t";
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

  mlir::LogicalResult emitVLA(VLAOp op) {
    if (inVLA)
      return op.emitError("nested VLA regions are not supported");
    if (!options.target.hasRVV)
      return op.emitError(
          "VLA requires RVV on the selected target; no scalar fallback exists");
    auto selected = vlaDecisions.find(op.getOperation());
    if (selected == vlaDecisions.end())
      return op.emitError("VLA physical decision was not prepared");
    const VLARegionDecision &decision = selected->second;
    mlir::Block &body = op.getBody().front();
    llvm::DenseMap<mlir::Operation *, CValue> aggregates;
    for (const VLAStateDecision &state : decision.states) {
      if (state.realization ==
          VLAStateRealization::RVVWideningF16DotReduction) {
        std::string fullVL = fresh("dot_vl");
        std::string accumulator = fresh("dot_acc");
        line("const size_t " + fullVL + " = __riscv_vsetvlmax_e" +
             std::to_string(state.inputShape.sew) + "m" +
             std::to_string(state.inputShape.lmul) + "();");
        line(rvvFloatType(state.computeShape) + " " + accumulator +
             " = __riscv_vfmv_v_f_" + rvvFloatSuffix(state.computeShape) +
             "(0.0f, " + fullVL + ");");
        CValue aggregate{state.elementType, CValueKind::F32BlockStorage,
                         accumulator};
        aggregate.fields.push_back(
            CValue{mlir::IndexType::get(kernel.getContext()),
                   CValueKind::Scalar, fullVL});
        aggregates[state.operation] = aggregate;
        continue;
      }
      if ((state.realization == VLAStateRealization::RVVAddReduction ||
           state.realization == VLAStateRealization::RVVMaxReduction) &&
          state.placement == VLAStatePlacement::VectorCarry) {
        std::string accumulator = fresh("reduce_acc");
        std::string suffix = "f32m" + std::to_string(state.dataLMUL);
        line("vfloat32m" + std::to_string(state.dataLMUL) + "_t " +
             accumulator + " = __riscv_vfmv_v_f_" + suffix + "(" +
             expression(state.identity) + ", __riscv_vsetvlmax_e32m" +
             std::to_string(state.dataLMUL) + "());");
        CValue aggregate{state.elementType, CValueKind::F32BlockStorage,
                         accumulator};
        aggregates[state.operation] = aggregate;
        continue;
      }
      if (state.realization == VLAStateRealization::RVVArgMaxSummary) {
        auto summary = mlir::cast<ArgMaxOp>(state.operation);
        CValue maximum{mlir::Float32Type::get(kernel.getContext()),
                       CValueKind::Scalar, fresh("argmax_value")};
        CValue index{mlir::IndexType::get(kernel.getContext()),
                     CValueKind::Scalar, fresh("argmax_index")};
        line("float " + maximum.spelling + " = -INFINITY;");
        line("size_t " + index.spelling + " = 0;");
        CValue aggregate{summary.getResult().getType(), CValueKind::Tuple, {}};
        aggregate.fields = {maximum, index};
        aggregates[state.operation] = aggregate;
        values[summary.getResult()] = aggregate;
        continue;
      }
      if (state.realization ==
          VLAStateRealization::RVVOnlineSoftmaxSummary)
        continue;
      std::string identity = expression(state.identity);
      if (identity.empty())
        return state.operation->emitError("VLA state identity is unavailable");
      llvm::StringRef prefix =
          state.realization == VLAStateRealization::RVVInclusiveAddScan ||
                  state.realization ==
                      VLAStateRealization::RVVSegmentedInclusiveAddScan
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
    for (const VLAStateDecision &state : decision.states) {
      if (state.realization ==
          VLAStateRealization::RVVOnlineSoftmaxSummary) {
        auto summary = mlir::cast<OnlineSoftmaxSummaryOp>(state.operation);
        CValue maximum{mlir::Float32Type::get(kernel.getContext()),
                       CValueKind::Scalar, fresh("summary_max")};
        CValue sum{mlir::Float32Type::get(kernel.getContext()),
                   CValueKind::Scalar, fresh("summary_sum")};
        line("float " + maximum.spelling + " = -INFINITY;");
        line("float " + sum.spelling + " = 0.0f;");
        CValue aggregate{summary.getResult().getType(), CValueKind::Tuple, {}};
        aggregate.fields = {maximum, sum};
        aggregates[summary.getOperation()] = aggregate;
        values[summary.getResult()] = aggregate;
      }
    }

    CValue begin = require(op.getBegin());
    CValue end = require(op.getEnd());
    if (begin.spelling.empty() || end.spelling.empty())
      return op.emitError("VLA bounds are unavailable");
    std::string strip = "__weft_vla" + std::to_string(nextLoop++);
    std::string vl = fresh("vl");
    bool previousInVLA = inVLA;
    std::string previousVL = activeVL;
    const VLARegionDecision *previousDecision = activeVLADecision;
    auto emitStripBody = [&](bool fullStrip) -> mlir::LogicalResult {
      bool previousFullStrip = activeVLAFullStrip;
      inVLA = true;
      activeVLAFullStrip = fullStrip;
      activeVL = vl;
      activeVLADecision = &decision;
      auto restoreVLAState = llvm::make_scope_exit([&] {
        inVLA = previousInVLA;
        activeVLAFullStrip = previousFullStrip;
        activeVL = previousVL;
        activeVLADecision = previousDecision;
      });
      CValue coordinate{body.getArgument(0).getType(), CValueKind::Coordinate,
                        strip};
      coordinate.laneStride = "1";
      values[body.getArgument(0)] = std::move(coordinate);
      for (mlir::Operation &nested : body.without_terminator()) {
        if (auto reduce = mlir::dyn_cast<ReduceOp>(nested)) {
          const VLAStateDecision *state =
              findStateDecision(reduce.getOperation());
          if (!state)
            return reduce.emitError(
                "VLA reduction has no physical state decision");
          mlir::LogicalResult lowered =
              state->realization ==
                      VLAStateRealization::RVVWideningF16DotReduction
                  ? emitWideningF16DotStrip(
                        reduce, *state, aggregates[reduce.getOperation()])
                  : emitVectorReduce(reduce, *state,
                                     aggregates[reduce.getOperation()]);
          if (mlir::failed(lowered))
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
        if (mlir::isa<ArgMaxOp, OnlineSoftmaxSummaryOp>(nested)) {
          const VLAStateDecision *state =
              findStateDecision(&nested);
          if (!state)
            return nested.emitError(
                "VLA summary has no physical state decision");
          mlir::LogicalResult lowered =
              state->realization == VLAStateRealization::RVVArgMaxSummary
                  ? emitVectorArgMaxSummary(
                        mlir::cast<ArgMaxOp>(nested), *state,
                        aggregates[&nested])
                  : emitOnlineSoftmaxSummary(
                        mlir::cast<OnlineSoftmaxSummaryOp>(nested), *state,
                        aggregates[&nested]);
          if (mlir::failed(lowered))
            return mlir::failure();
          continue;
        }
        if (mlir::failed(emitOperation(&nested)))
          return mlir::failure();
      }
      return mlir::success();
    };

    if (decision.physical.stripSchedule == VLAStripSchedule::FullThenTail) {
      auto wideningState = llvm::find_if(
          decision.states, [](const VLAStateDecision &state) {
            return state.realization ==
                   VLAStateRealization::RVVWideningF16DotReduction;
          });
      if (wideningState == decision.states.end())
        return op.emitError("full-then-tail VLA has no owning state decision");
      auto aggregate = aggregates.find(wideningState->operation);
      if (aggregate == aggregates.end() || aggregate->second.fields.size() != 1 ||
          aggregate->second.fields.front().spelling.empty())
        return op.emitError("full-then-tail VLA has no selected full VL");
      std::string fullVL = aggregate->second.fields.front().spelling;
      line("size_t " + strip + " = " + begin.spelling + ";");
      line("for (; " + strip + " + " + fullVL + " <= " + end.spelling + "; " +
           strip + " += " + fullVL + ") {");
      ++indent;
      line("const size_t " + vl + " = " + fullVL + ";");
      if (mlir::failed(emitStripBody(true)))
        return mlir::failure();
      --indent;
      line("}");
      line("if (" + strip + " < " + end.spelling + ") {");
      ++indent;
      line("const size_t " + vl + " = __riscv_vsetvl_e" +
           std::to_string(decision.physical.dataSEW) + "m" +
           std::to_string(decision.physical.dataLMUL) + "(" + end.spelling +
           " - " + strip + ");");
      if (mlir::failed(emitStripBody(false)))
        return mlir::failure();
      --indent;
      line("}");
    } else {
      line("for (size_t " + strip + " = " + begin.spelling + "; " + strip +
           " < " + end.spelling + ";) {");
      ++indent;
      line("const size_t " + vl + " = __riscv_vsetvl_e" +
           std::to_string(decision.physical.dataSEW) + "m" +
           std::to_string(decision.physical.dataLMUL) + "(" + end.spelling +
           " - " + strip + ");");
      if (mlir::failed(emitStripBody(false)))
        return mlir::failure();
      line(strip + " += " + vl + ";");
      --indent;
      line("}");
    }

    for (const VLAStateDecision &state : decision.states) {
      if ((state.realization != VLAStateRealization::RVVAddReduction &&
           state.realization != VLAStateRealization::RVVMaxReduction) ||
          state.placement != VLAStatePlacement::VectorCarry)
        continue;
      auto reduce = mlir::cast<ReduceOp>(state.operation);
      CValue aggregate = aggregates[state.operation];
      std::string seed = fresh("reduce_seed");
      std::string reduced = fresh("reduce_final");
      std::string result = fresh("reduce_result");
      std::string suffix = "f32m" + std::to_string(state.dataLMUL);
      line("vfloat32m1_t " + seed +
           " = __riscv_vfmv_v_f_f32m1(" + expression(state.identity) +
           ", 1);");
      std::string intrinsic =
          state.realization == VLAStateRealization::RVVAddReduction
              ? "__riscv_vfredusum_vs_"
              : "__riscv_vfredmax_vs_";
      line("vfloat32m1_t " + reduced + " = " + intrinsic + suffix +
           "_f32m1(" + aggregate.spelling + ", " + seed +
           ", __riscv_vsetvlmax_e32m" + std::to_string(state.dataLMUL) +
           "());");
      line("const float " + result +
           " = __riscv_vfmv_f_s_f32m1_f32(" + reduced + ");");
      values[reduce.getResult()] =
          CValue{reduce.getResult().getType(), CValueKind::Scalar, result};
    }

    for (const VLAStateDecision &state : decision.states) {
      if (state.realization !=
          VLAStateRealization::RVVWideningF16DotReduction)
        continue;
      auto reduce = mlir::cast<ReduceOp>(state.operation);
      CValue aggregate = aggregates[state.operation];
      if (aggregate.fields.size() != 1 ||
          aggregate.fields.front().spelling.empty())
        return reduce.emitError("widening dot full VL is unavailable");
      const std::string &fullVL = aggregate.fields.front().spelling;
      std::string seed = fresh("dot_seed");
      std::string reduced = fresh("dot_reduced");
      std::string result = fresh("dot");
      line(rvvFloatType(state.reductionShape) + " " + seed +
           " = __riscv_vfmv_v_f_" + rvvFloatSuffix(state.reductionShape) +
           "(" + expression(state.identity) + ", 1);");
      line(rvvFloatType(state.reductionShape) + " " + reduced +
           " = __riscv_vfredusum_vs_" + rvvFloatSuffix(state.computeShape) +
           "_" + rvvFloatSuffix(state.reductionShape) + "(" +
           aggregate.spelling + ", " + seed + ", " + fullVL + ");");
      line("const float " + result + " = __riscv_vfmv_f_s_" +
           rvvFloatSuffix(state.reductionShape) + "_f32(" + reduced + ");");
      CValue materialized{reduce.getResult().getType(), CValueKind::Scalar,
                          result};
      values[reduce.getResult()] = materialized;
    }

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
    if (base.indexedPointer)
      return op.emitError(
          "indexed VLA pointer cannot receive a second offset; combine the explicit index expression before pointer addition");
    CValue result{op.getResult().getType(), CValueKind::Pointer,
                  "(" + base.spelling + " + " + offset.spelling + ")"};
    if (offset.kind == CValueKind::Coordinate) {
      result.lanePointer = true;
      result.laneStride = offset.laneStride;
    } else if (offset.kind == CValueKind::IndexVector) {
      result.spelling = base.spelling;
      result.indexedPointer = true;
      result.indexSpelling = offset.spelling;
      result.vectorSEW = offset.vectorSEW;
      result.vectorLMUL = offset.vectorLMUL;
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
    if (isAbsorbedBinaryProducer(op.getOperation()))
      return mlir::success();
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
      std::string ratio = std::to_string(activeVLADecision->physical.maskRatio);
      line("vbool" + ratio + "_t " + name + " = " + intrinsic + ratio + "(" +
           lhs.spelling + ", " + rhs.spelling + ", " + activeVL + ");");
      values[op.getResult()] =
          CValue{op.getResult().getType(), CValueKind::Mask, name};
      return mlir::success();
    }
    bool f32 = elementType(op.getResult().getType()).isF32();
    bool f16 = isF16(elementType(op.getResult().getType()));
    CValueKind vectorKind =
        f32 ? CValueKind::F32Vector : CValueKind::F16Vector;
    if (const VLABinaryDecision *decision =
            findBinaryDecision(op.getOperation())) {
      CValue accumulator = require(decision->accumulator);
      CValue vector = require(decision->vector);
      CValue scalar = require(decision->scalar);
      if (decision->realization !=
              VLABinaryRealization::RVVF16FusedMultiplyAdd ||
          accumulator.kind != CValueKind::F16Vector ||
          vector.kind != CValueKind::F16Vector ||
          scalar.kind != CValueKind::Scalar || accumulator.spelling.empty() ||
          vector.spelling.empty() || scalar.spelling.empty())
        return op.emitError("selected f16 fused multiply-add is unavailable");
      std::string suffix = "f16m" + std::to_string(decision->lmul);
      std::string name = fresh("fma");
      line("vfloat16m" + std::to_string(decision->lmul) + "_t " + name +
           " = __riscv_vfmacc_vf_" + suffix + "(" + accumulator.spelling +
           ", " + scalar.spelling + ", " + vector.spelling + ", " + activeVL +
           ");");
      values[op.getResult()] =
          CValue{op.getResult().getType(), CValueKind::F16Vector, name};
      return mlir::success();
    }
    bool lhsVector = lhs.kind == vectorKind;
    bool rhsVector = rhs.kind == vectorKind;
    if (!lhsVector && !rhsVector) {
      std::string expression =
          scalarBinary(op.getKind(), lhs.spelling, rhs.spelling);
      if (expression.empty())
        return op.emitError("RISC-V scalar lowering does not implement binary kind");
      values[op.getResult()] = scalarExpression(
          op.getResult(), std::move(expression), "scalar");
      return mlir::success();
    }
    if (!inVLA || (!f32 && !f16))
      return op.emitError(
          "RVV pointwise lowering requires a floating VLA result");
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
    unsigned lmul = activeVLADecision->physical.dataLMUL;
    std::string element = f32 ? "f32m" : "f16m";
    std::string vectorType = f32 ? "vfloat32m" : "vfloat16m";
    std::string suffix = element + std::to_string(lmul);
    intrinsic += suffix;
    std::string name = fresh("v");
    line(vectorType + std::to_string(lmul) + "_t " + name + " = " +
         intrinsic + "(" + first + ", " + second + ", " + activeVL + ");");
    values[op.getResult()] =
        CValue{op.getResult().getType(), vectorKind, name};
    return mlir::success();
  }

  mlir::LogicalResult emitUnary(UnaryOp op) {
    CValue input = require(op.getInput());
    if (input.kind == CValueKind::F32Vector) {
      std::string name = fresh("v");
      unsigned lmul = activeVLADecision->physical.dataLMUL;
      std::string suffix = "f32m" + std::to_string(lmul);
      std::string vectorType = "vfloat32m" + std::to_string(lmul) + "_t";
      if (op.getKind() == "neg")
        line(vectorType + " " + name + " = __riscv_vfneg_v_" + suffix + "(" +
             input.spelling + ", " + activeVL + ");");
      else if (op.getKind() == "abs")
        line(vectorType + " " + name + " = __riscv_vfabs_v_" + suffix + "(" +
             input.spelling + ", " + activeVL + ");");
      else if (op.getKind() == "sqrt")
        line(vectorType + " " + name + " = __riscv_vfsqrt_v_" + suffix +
             "(" + input.spelling + ", " + activeVL + ");");
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
      if (decision->realization ==
          VLAPredicateDecision::Realization::RVVVectorScalar) {
        CValueKind expected = decision->vectorSEW == 8
                                  ? CValueKind::U8Vector
                                  : CValueKind::F32Vector;
        if (coordinate.kind != expected || scalar.kind != CValueKind::Scalar ||
            coordinate.spelling.empty() || scalar.spelling.empty() ||
            decision->vectorLMUL == 0)
          return op.emitError("typed VLA predicate projection is unavailable");
        std::string vectorSuffix;
        std::string maskSuffix;
        std::string intrinsic;
        if (decision->vectorSEW == 8) {
          vectorSuffix = "u8m" + std::to_string(decision->vectorLMUL);
          maskSuffix = vectorSuffix + "_b" +
                       std::to_string(activeVLADecision->physical.maskRatio);
          intrinsic = llvm::StringSwitch<std::string>(decision->predicate)
                          .Case("eq", "__riscv_vmseq_vx_")
                          .Case("ne", "__riscv_vmsne_vx_")
                          .Case("lt", "__riscv_vmsltu_vx_")
                          .Case("le", "__riscv_vmsleu_vx_")
                          .Case("gt", "__riscv_vmsgtu_vx_")
                          .Case("ge", "__riscv_vmsgeu_vx_")
                          .Default("");
        } else {
          vectorSuffix = "f32m" + std::to_string(decision->vectorLMUL);
          maskSuffix = vectorSuffix + "_b" +
                       std::to_string(activeVLADecision->physical.maskRatio);
          intrinsic = llvm::StringSwitch<std::string>(decision->predicate)
                          .Case("eq", "__riscv_vmfeq_vf_")
                          .Case("ne", "__riscv_vmfne_vf_")
                          .Case("lt", "__riscv_vmflt_vf_")
                          .Case("le", "__riscv_vmfle_vf_")
                          .Case("gt", "__riscv_vmfgt_vf_")
                          .Case("ge", "__riscv_vmfge_vf_")
                          .Default("");
        }
        if (intrinsic.empty())
          return op.emitError("typed VLA predicate realization is unavailable");
        std::string mask = fresh("mask");
        line("vbool" +
             std::to_string(activeVLADecision->physical.maskRatio) + "_t " +
             mask + " = " + intrinsic + maskSuffix + "(" +
             coordinate.spelling + ", " + scalar.spelling + ", " + activeVL +
             ");");
        values[op.getResult()] =
            CValue{op.getResult().getType(), CValueKind::Mask, mask};
        return mlir::success();
      }
      if (coordinate.kind != CValueKind::Coordinate ||
          scalar.kind != CValueKind::Scalar || coordinate.spelling.empty() ||
          scalar.spelling.empty() || coordinate.laneStride.empty())
        return op.emitError("VLA predicate projection is unavailable");

      std::string vectorSuffix = "u" +
                                 std::to_string(activeVLADecision->physical.indexSEW) +
                                 "m" +
                                 std::to_string(activeVLADecision->physical.indexLMUL);
      std::string maskSuffix =
          vectorSuffix + "_b" +
          std::to_string(activeVLADecision->physical.maskRatio);
      std::string vectorType = "vuint" +
                               std::to_string(activeVLADecision->physical.indexSEW) +
                               "m" +
                               std::to_string(activeVLADecision->physical.indexLMUL) +
                               "_t";
      std::string scalarType =
          "uint" + std::to_string(activeVLADecision->physical.indexSEW) + "_t";
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
      line("vbool" + std::to_string(activeVLADecision->physical.maskRatio) + "_t " +
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
    if (const VLACastDecision *decision =
            findCastDecision(op.getOperation())) {
      if (decision->realization ==
          VLACastRealization::RVVZeroExtendU32ToIndex) {
        if (input.kind != CValueKind::U32Vector || input.spelling.empty())
          return op.emitError(
              "selected u32-to-index VLA cast operand is unavailable");
        CValue result{op.getResult().getType(), CValueKind::IndexVector,
                      input.spelling};
        result.vectorSEW = 32;
        result.vectorLMUL = decision->sourceLMUL;
        values[op.getResult()] = std::move(result);
        return mlir::success();
      }
      CValueKind sourceKind =
          decision->sourceSEW == 16 ? CValueKind::F16Vector
                                    : CValueKind::F32Vector;
      CValueKind resultKind =
          decision->resultSEW == 16 ? CValueKind::F16Vector
                                    : CValueKind::F32Vector;
      if (input.kind != sourceKind || input.spelling.empty())
        return op.emitError("selected VLA cast operand is unavailable");
      std::string name = fresh(decision->resultSEW == 32 ? "widen_f16"
                                                        : "narrow_f32");
      std::string resultSuffix = "f" + std::to_string(decision->resultSEW) +
                                 "m" + std::to_string(decision->resultLMUL);
      std::string intrinsic =
          decision->realization == VLACastRealization::RVVWidenF16ToF32
              ? "__riscv_vfwcvt_f_f_v_"
              : "__riscv_vfncvt_f_f_w_";
      line("vfloat" + std::to_string(decision->resultSEW) + "m" +
           std::to_string(decision->resultLMUL) + "_t " + name + " = " +
           intrinsic + resultSuffix + "(" + input.spelling + ", " + activeVL +
           ");");
      values[op.getResult()] = CValue{op.getResult().getType(), resultKind, name};
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
        decision->sourceLMUL != activeVLADecision->physical.dataLMUL)
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
    else if (source.isF32() && target.isUnsignedInteger(32))
      helper = "__weft_bitcast_f32_u32";
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
      unsigned lmul = activeVLADecision->physical.dataLMUL;
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
    if (auto load = value.getDefiningOp<LoadOp>()) {
      std::optional<std::string> pointer =
          projectBlockScalar(load.getPointer(), axisValues);
      if (!pointer)
        return std::nullopt;
      std::string read = "(*" + *pointer + ")";
      if (isTrue(load.getWhere()))
        return read;
      std::optional<std::string> predicate =
          projectBlockScalar(load.getWhere(), axisValues);
      std::optional<std::string> other =
          projectBlockScalar(load.getOther(), axisValues);
      if (!predicate || !other)
        return std::nullopt;
      return "(" + *predicate + " ? " + read + " : " + *other + ")";
    }
    if (auto unary = value.getDefiningOp<UnaryOp>()) {
      std::optional<std::string> input =
          projectBlockScalar(unary.getInput(), axisValues);
      if (!input)
        return std::nullopt;
      if (unary.getKind() == "neg")
        return "(-" + *input + ")";
      if (unary.getKind() == "abs")
        return "fabsf(" + *input + ")";
      return std::nullopt;
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
    if (decision.realization ==
        VLAContractRealization::RVVF32FreeAxisVectorDot) {
      auto dot = mlir::cast<DotOp>(decision.operation);
      auto store = mlir::cast<StoreOp>(decision.consumer);
      auto freeLoad = mlir::cast<LoadOp>(decision.freeLoad);
      std::string extent = expression(decision.reductionExtent);
      CValue init = require(decision.init);
      if (extent.empty() || activeVL.empty() ||
          decision.initRealization !=
              VLAContractInitRealization::MaterializedRegion ||
          init.kind != CValueKind::F32Vector || init.spelling.empty())
        return dot.emitError(
            "RVV VLA vector-dot physical operands are unavailable");

      std::string vectorSuffix = "f32m" + std::to_string(decision.lmul);
      std::string vectorType =
          "vfloat32m" + std::to_string(decision.lmul) + "_t";
      llvm::DenseMap<mlir::Value, std::string> baseAxes;
      baseAxes[decision.reductionAxis] = "0";
      std::optional<std::string> outputPointer =
          projectBlockScalar(store.getPointer(), baseAxes);
      std::optional<std::string> outputLaneStride = projectVLALaneStride(
          store.getPointer(), activeVLADecision->coordinate, baseAxes);
      if (!outputPointer || !outputLaneStride)
        return dot.emitError(
            "RVV VLA vector-dot output projection is unavailable");

      std::string accumulator = fresh("vla_contract_acc");
      line(vectorType + " " + accumulator + " = " + init.spelling + ";");
      std::string reduction = fresh("vla_contract_k");
      line("for (size_t " + reduction + " = 0; " + reduction + " < " +
           extent + "; " + reduction + " += " +
           std::to_string(decision.kUnroll) + ") {");
      ++indent;
      for (unsigned unroll = 0; unroll < decision.kUnroll; ++unroll) {
        std::string coordinate =
            unroll == 0 ? reduction
                        : "(" + reduction + " + " + std::to_string(unroll) + ")";
        line("if (" + coordinate + " < " + extent + ") {");
        ++indent;
        llvm::DenseMap<mlir::Value, std::string> axes;
        axes[decision.reductionAxis] = coordinate;
        std::optional<std::string> freePointer =
            projectBlockScalar(freeLoad.getPointer(), axes);
        std::optional<std::string> freeLaneStride = projectVLALaneStride(
            freeLoad.getPointer(), activeVLADecision->coordinate, axes);
        std::optional<std::string> blocked =
            projectBlockScalar(decision.blockedOperand, axes);
        if (!freePointer || !freeLaneStride || !blocked)
          return dot.emitError(
              "RVV VLA vector-dot operand projection is unavailable");
        std::string vector = fresh("vla_contract_free");
        if (decision.freeMemoryMode == VLAMemoryMode::UnitStride) {
          line(vectorType + " " + vector + " = __riscv_vle32_v_" +
               vectorSuffix + "(" + *freePointer + ", " + activeVL + ");");
        } else {
          line(vectorType + " " + vector + " = __riscv_vlse32_v_" +
               vectorSuffix + "(" + *freePointer +
               ", (ptrdiff_t)(sizeof(float) * (" + *freeLaneStride + ")), " +
               activeVL + ");");
        }
        line(accumulator + " = __riscv_vfmacc_vf_" + vectorSuffix + "(" +
             accumulator + ", " + *blocked + ", " + vector + ", " + activeVL +
             ");");
        --indent;
        line("}");
      }
      --indent;
      line("}");
      if (decision.outputMemoryMode == VLAMemoryMode::UnitStride) {
        line("__riscv_vse32_v_" + vectorSuffix + "(" + *outputPointer + ", " +
             accumulator + ", " + activeVL + ");");
      } else {
        line("__riscv_vsse32_v_" + vectorSuffix + "(" + *outputPointer +
             ", (ptrdiff_t)(sizeof(float) * (" + *outputLaneStride + ")), " +
             accumulator + ", " + activeVL + ");");
      }
      return mlir::success();
    }

    auto dot = mlir::cast<DotOp>(decision.operation);
    auto store = mlir::cast<StoreOp>(decision.consumer);
    auto lhsLoad = mlir::cast<LoadOp>(decision.lhsLoad);
    auto rhsLoad = mlir::cast<LoadOp>(decision.rhsLoad);
    std::string extent = expression(decision.reductionExtent);
    if (extent.empty() || activeVL.empty())
      return dot.emitError(
          "RVV VLA dot physical bounds are unavailable");

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
        return dot.emitError(
            "RVV VLA dot row projection is unavailable");
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
      line("for (size_t " + reduction + " = 0; " + reduction + " < " +
           extent + "; " + reduction + " += " +
           std::to_string(decision.kUnroll) + ") {");
      ++indent;
      for (unsigned unroll = 0; unroll < decision.kUnroll; ++unroll) {
        std::string coordinate =
            unroll == 0 ? reduction
                        : "(" + reduction + " + " + std::to_string(unroll) + ")";
        line("if (" + coordinate + " < " + extent + ") {");
        ++indent;
        llvm::DenseMap<mlir::Value, std::string> rhsAxes;
        rhsAxes[decision.reductionAxis] = coordinate;
        std::optional<std::string> rhsPointer =
            projectBlockScalar(rhsLoad.getPointer(), rhsAxes);
        std::optional<std::string> rhsLaneStride = projectVLALaneStride(
            rhsLoad.getPointer(), activeVLADecision->coordinate, rhsAxes);
        if (!rhsPointer || !rhsLaneStride)
          return dot.emitError(
              "RVV VLA dot RHS projection is unavailable");
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
          axes[decision.reductionAxis] = coordinate;
          std::optional<std::string> lhsPointer =
              projectBlockScalar(lhsLoad.getPointer(), axes);
          std::optional<std::string> lhsPredicate;
          if (decision.lhsPredicateVariesByReduction)
            lhsPredicate = projectBlockScalar(lhsLoad.getWhere(), axes);
          if (!lhsPointer ||
              (decision.lhsPredicateVariesByReduction && !lhsPredicate))
            return dot.emitError(
                "RVV VLA dot LHS projection is unavailable");
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
          return dot.emitError(
              "RVV VLA dot output stride is unavailable");
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
  decideLocalF32RowMicrotile(StoreOp store, DotOp dot) {
    auto unwrapBlock = [](mlir::Type type) -> BlockType {
      if (auto masked = mlir::dyn_cast<MaskedType>(type))
        type = masked.getValueType();
      return mlir::dyn_cast<BlockType>(type);
    };
    BlockType lhsType = unwrapBlock(dot.getLhs().getType());
    BlockType rhsType = unwrapBlock(dot.getRhs().getType());
    BlockType resultType = unwrapBlock(dot.getResult().getType());
    auto init = dot.getInit().getDefiningOp<FullOp>();
    if (store.getValue() != dot.getResult() || !lhsType || !rhsType ||
        !resultType || lhsType.getShape().size() != 2 ||
        rhsType.getShape().size() != 1 || resultType.getShape().size() != 1 ||
        lhsType.getShape()[0] != resultType.getShape()[0] ||
        lhsType.getShape()[1] != rhsType.getShape()[0] ||
        resultType.getShape()[0] <= 0 || resultType.getShape()[0] > 8 ||
        !lhsType.getElementType().isF32() ||
        !rhsType.getElementType().isF32() ||
        !resultType.getElementType().isF32() || !init ||
        !isFloatConstant(init.getValue(), 0.0) ||
        dot.getOrder() != "relaxed" || dot.getMath() != "native" ||
        !dot.getAccDtype().isF32()) {
      dot.emitError(
          "RVV row microtile requires a [BM,K] x [K] local f32 dot");
      return mlir::failure();
    }

    LoadOp lhsLoad = dot.getLhs().getDefiningOp<LoadOp>();
    LoadOp rhsLoad = dot.getRhs().getDefiningOp<LoadOp>();
    if (!lhsLoad || !rhsLoad || !isTrue(rhsLoad.getWhere())) {
      dot.emitError("RVV row microtile load producers are unavailable");
      return mlir::failure();
    }

    llvm::DenseSet<mlir::Operation *> lhsClosure;
    llvm::DenseSet<mlir::Operation *> rhsClosure;
    llvm::SmallVector<BlockAxisOp> lhsAxes;
    llvm::SmallVector<BlockAxisOp> rhsAxes;
    if (mlir::failed(collectBlockClosure(dot.getLhs(), lhsClosure, lhsAxes,
                                         store.getOperation())) ||
        mlir::failed(collectBlockClosure(dot.getRhs(), rhsClosure, rhsAxes,
                                         store.getOperation())))
      return mlir::failure();
    if (rhsAxes.size() != 1 || lhsAxes.size() != 2 ||
        !llvm::is_contained(lhsAxes, rhsAxes.front())) {
      dot.emitError(
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
      dot.emitError("RVV row microtile pointer facts are unavailable");
      return mlir::failure();
    }

    ContractDecision decision;
    decision.operation = dot.getOperation();
    decision.realization = ContractRealization::RVVF32RowMicrotile;
    decision.consumer = store.getOperation();
    decision.lhsLoad = lhsLoad.getOperation();
    decision.rhsLoad = rhsLoad.getOperation();
    decision.rowAxis = rowAxis.getResult();
    decision.reductionAxis = reductionAxis.getResult();
    decision.reductionExtent = reductionAxis.getExtent();
    decision.rowTile = static_cast<unsigned>(resultType.getShape()[0]);
    std::optional<F32ContractPhysicalConfig> physical =
        selectF32ContractPhysicalConfig(F32ContractResourceModel::LocalRow,
                                        decision.rowTile, 1);
    if (!physical) {
      dot.emitError(
          "RVV row microtile has no legal register-resource candidate");
      return mlir::failure();
    }
    decision.lmul = physical->lmul;
    decision.kUnroll = physical->kUnroll;
    return decision;
  }

  std::optional<mlir::LogicalResult>
  tryEmitLocalF32ContractBlockStore(StoreOp store) {
    auto selected = localF32ContractDecisions.find(store.getOperation());
    if (selected == localF32ContractDecisions.end())
      return std::nullopt;
    const ContractDecision &decision = selected->second;
    auto dot = mlir::cast<DotOp>(decision.operation);
    auto lhsLoad = mlir::cast<LoadOp>(decision.lhsLoad);
    auto rhsLoad = mlir::cast<LoadOp>(decision.rhsLoad);
    std::string extent = expression(decision.reductionExtent);
    if (extent.empty())
      return dot.emitError(
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
        return dot.emitError(
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
      auto emitChunk = [&](llvm::StringRef coordinate) -> mlir::LogicalResult {
        llvm::DenseMap<mlir::Value, std::string> rhsAxes;
        rhsAxes[decision.reductionAxis] = coordinate.str();
        std::optional<std::string> rhs =
            projectBlockScalar(rhsLoad.getPointer(), rhsAxes);
        if (!rhs)
          return dot.emitError(
              "RVV row microtile RHS projection is unavailable");
        std::string rhsVector = fresh("contract_rhs");
        line(vectorType + " " + rhsVector + " = __riscv_vle32_v_" +
             vectorSuffix + "(" + *rhs + ", " + vl + ");");
        for (unsigned lane = 0; lane < decision.rowTile; ++lane) {
          llvm::DenseMap<mlir::Value, std::string> axes;
          axes[decision.rowAxis] = std::to_string(lane);
          axes[decision.reductionAxis] = coordinate.str();
          std::optional<std::string> lhs =
              projectBlockScalar(lhsLoad.getPointer(), axes);
          if (!lhs)
            return dot.emitError(
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
        return mlir::success();
      };
      line("for (size_t " + strip + " = 0; " + strip + " < " + extent +
           ";) {");
      ++indent;
      if (decision.kUnroll == 1) {
        line("const size_t " + vl + " = __riscv_vsetvl_e32m" +
             std::to_string(decision.lmul) + "(" + extent + " - " + strip +
             ");");
        if (mlir::failed(emitChunk(strip)))
          return mlir::failure();
        line(strip + " += " + vl + ";");
      } else {
        std::string remaining = fresh("contract_remaining");
        line("const size_t " + remaining + " = " + extent + " - " + strip +
             ";");
        line("if (" + remaining + " >= " +
             std::to_string(decision.kUnroll) + ") {");
        ++indent;
        line("const size_t " + vl + " = __riscv_vsetvl_e32m" +
             std::to_string(decision.lmul) + "(" + remaining + " / " +
             std::to_string(decision.kUnroll) + ");");
        for (unsigned unroll = 0; unroll < decision.kUnroll; ++unroll) {
          std::string coordinate =
              unroll == 0
                  ? strip
                  : "(" + strip + " + " + std::to_string(unroll) + " * " + vl +
                        ")";
          if (mlir::failed(emitChunk(coordinate)))
            return mlir::failure();
        }
        line(strip + " += " + std::to_string(decision.kUnroll) + " * " + vl +
             ";");
        --indent;
        line("} else {");
        ++indent;
        line("const size_t " + vl + " = __riscv_vsetvl_e32m" +
             std::to_string(decision.lmul) + "(" + remaining + ");");
        if (mlir::failed(emitChunk(strip)))
          return mlir::failure();
        line(strip + " += " + vl + ";");
        --indent;
        line("}");
      }
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
    if (inVLA && (pointer.lanePointer || pointer.indexedPointer)) {
      const VLAAccessDecision *decision =
          findAccessDecision(op.getOperation());
      if (!decision)
        return op.emitError("VLA load has no physical memory decision");
      mlir::Type loadedElement = decision->elementType;
      bool f32 = loadedElement.isF32();
      bool f16 = isF16(loadedElement);
      bool u8 = loadedElement.isUnsignedInteger(8);
      bool u32 = loadedElement.isUnsignedInteger(32);
      if ((!f32 && !f16 && !u8 && !u32) ||
          (!pointer.lanePointer && !pointer.indexedPointer) ||
          (pointer.lanePointer && pointer.laneStride.empty()))
        return op.emitError("VLA load element realization is unavailable");
      if (decision->activityMode == VLAActivityMode::PredicateMask)
        return op.emitError("masked VLA load has no selected realization");
      std::string name = fresh("load");
      unsigned lmul = decision->elementLMUL;
      if (lmul == 0 || decision->elementSEW == 0)
        return op.emitError("VLA load has no selected vector shape");
      std::string element = f32   ? "f32m"
                            : f16 ? "f16m"
                            : u8  ? "u8m"
                                  : "u32m";
      std::string vectorType =
          std::string(f32   ? "vfloat32m"
                      : f16 ? "vfloat16m"
                      : u8  ? "vuint8m"
                            : "vuint32m") +
          std::to_string(lmul) + "_t";
      std::string suffix = element + std::to_string(lmul);
      auto emitRead = [&]() {
        if (decision->memoryMode == VLAMemoryMode::UnitStride) {
          line(name + " = __riscv_vle" +
               std::to_string(decision->elementSEW) +
               "_v_" + suffix + "(" + pointer.spelling + ", " + activeVL +
               ");");
        } else if (decision->memoryMode == VLAMemoryMode::Strided) {
          std::string cType = f32   ? "float"
                              : f16 ? "_Float16"
                              : u8  ? "uint8_t"
                                    : "uint32_t";
          line(name + " = __riscv_vlse" +
               std::to_string(decision->elementSEW) +
               "_v_" + suffix + "(" + pointer.spelling +
               ", (ptrdiff_t)(sizeof(" + cType +
               ") * (" + pointer.laneStride + ")), " + activeVL + ");");
        } else {
          if (decision->indexSEW != 32)
            return false;
          std::string indexWidth = std::to_string(decision->indexSEW);
          std::string offsets = fresh("byte_offsets");
          std::string indexSuffix =
              "u" + indexWidth + "m" + std::to_string(decision->indexLMUL);
          line("vuint" + indexWidth + "m" +
               std::to_string(decision->indexLMUL) + "_t " +
               offsets + " = __riscv_vmul_vx_" + indexSuffix + "(" +
               pointer.indexSpelling + ", (uint" + indexWidth + "_t)sizeof(" +
               std::string(f32 ? "float" : f16 ? "_Float16" : u8 ? "uint8_t"
                                                                        : "uint32_t") +
               "), " + activeVL + ");");
          line(name + " = __riscv_vluxei" + indexWidth + "_v_" + suffix + "(" +
               pointer.spelling + ", " + offsets + ", " + activeVL + ");");
        }
        return true;
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
        if (!emitRead())
          return op.emitError(
              "indexed VLA load decision has no intrinsic-C spelling");
        --indent;
        line("} else {");
        ++indent;
        std::string broadcast = f32 || f16 ? "__riscv_vfmv_v_f_"
                                          : "__riscv_vmv_v_x_";
        line(name + " = " + broadcast + suffix + "(" + other.spelling + ", " +
             activeVL + ");");
        --indent;
        line("}");
      } else {
        if (!emitRead())
          return op.emitError(
              "indexed VLA load decision has no intrinsic-C spelling");
      }
      if (f32) {
        values[op.getResult()] =
            CValue{op.getResult().getType(), CValueKind::F32Vector, name};
      } else if (f16) {
        values[op.getResult()] =
            CValue{op.getResult().getType(), CValueKind::F16Vector, name};
      } else if (u8) {
        values[op.getResult()] =
            CValue{op.getResult().getType(), CValueKind::U8Vector, name};
      } else {
        values[op.getResult()] =
            CValue{op.getResult().getType(), CValueKind::U32Vector, name};
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
      bool u8 = decision->elementType.isUnsignedInteger(8);
      if ((!f32 && !f16 && !i8 && !u8) ||
          pointer.laneStride.empty())
        return op.emitError("VLA store element realization is unavailable");

      std::string vector = value.spelling;
      CValueKind expectedKind = f32   ? CValueKind::F32Vector
                                : f16 ? CValueKind::F16Vector
                                : i8  ? CValueKind::I8Vector
                                      : CValueKind::U8Vector;
      if (decision->storeValueMode == VLAStoreValueMode::ScalarBroadcast) {
        if (value.kind != CValueKind::Scalar || value.spelling.empty())
          return op.emitError("VLA store scalar broadcast is unavailable");
        vector = fresh("store_value");
        unsigned lmul = decision->elementLMUL;
        std::string suffix =
            std::string(f32   ? "f32m"
                        : f16 ? "f16m"
                        : i8  ? "i8m"
                              : "u8m") +
            std::to_string(lmul);
        std::string vectorType =
            std::string(f32   ? "vfloat32m"
                        : f16 ? "vfloat16m"
                        : i8  ? "vint8m"
                              : "vuint8m") +
            std::to_string(lmul) + "_t";
        std::string broadcast = f32 || f16 ? "__riscv_vfmv_v_f_"
                                          : "__riscv_vmv_v_x_";
        line(vectorType + " " + vector + " = " + broadcast + suffix + "(" +
             value.spelling + ", " + activeVL + ");");
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
      unsigned lmul = decision->elementLMUL;
      if (lmul == 0 || decision->elementSEW == 0)
        return op.emitError("VLA store has no selected vector shape");
      std::string sew = std::to_string(decision->elementSEW);
      std::string suffix =
          std::string(f32   ? "f32m"
                      : f16 ? "f16m"
                      : i8  ? "i8m"
                            : "u8m") +
          std::to_string(lmul);
      std::string elementCType =
          f32 ? "float" : f16 ? "_Float16" : i8 ? "int8_t" : "uint8_t";
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

  std::optional<LocalBlockMemoryFact>
  resolveLocalBlockMemoryFact(mlir::Value semanticValue) const {
    auto blockType = [](mlir::Type type) -> BlockType {
      if (auto masked = mlir::dyn_cast<MaskedType>(type))
        type = masked.getValueType();
      return mlir::dyn_cast<BlockType>(type);
    };

    BlockType semanticType = blockType(semanticValue.getType());
    if (!semanticType || semanticType.getShape().size() != 1 ||
        semanticType.getShape().front() <= 0)
      return std::nullopt;

    mlir::Value storageValue = semanticValue;
    while (auto bitcast = storageValue.getDefiningOp<BitcastOp>())
      storageValue = bitcast.getInput();
    LoadOp load = storageValue.getDefiningOp<LoadOp>();
    BlockType storageType = blockType(storageValue.getType());
    if (!load || !storageType || storageType.getShape().size() != 1 ||
        storageType.getShape().front() <= 0 || !isTrue(load.getWhere()))
      return std::nullopt;

    auto lane = load.getPointer().getDefiningOp<PtrAddOp>();
    int64_t storageExtent = storageType.getShape().front();
    if (!lane || !matchBlockAxis(lane.getOffset(), storageExtent))
      return std::nullopt;

    return LocalBlockMemoryFact{
        semanticValue,
        lane.getBase(),
        semanticType,
        storageType,
        semanticType.getShape().front(),
        storageExtent,
    };
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
    auto packedPointer = packedBase.getDefiningOp<PtrAddOp>();
    mlir::Value packedBlockIndex;
    int64_t packedBlockBytes = 288;
    if (!packedPointer ||
        !matchBinaryConstant(packedPointer.getOffset(), "mul",
                             packedBlockBytes, packedBlockIndex))
      return op.emitError(
          "IME1 symmetric contraction requires the explicit 288-byte persistent packed block");
    (void)packedBlockIndex;

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
    decision.packedBlockBytes = packedBlockBytes;
    return mlir::success();
  }

  mlir::LogicalResult emitSymmetricI4I8Contract(
      SymmetricI4I8ContractOp op) {
    auto selected = symmetricI4I8Decisions.find(op.getOperation());
    if (selected == symmetricI4I8Decisions.end())
      return op.emitError("symmetric i4/i8 physical decision was not prepared");
    const SymmetricI4I8Decision &decision = selected->second;
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

  mlir::LogicalResult decideSignBitI8Dot(SignBitI8DotOp op,
                                         SignBitI8Decision &decision) {
    if (options.target.vlenBits < 128)
      return op.emitError(
          "sign-bit/i8 dot requires an explicit VLEN of at least 128 bits");
    if (!options.target.littleEndian)
      return op.emitError(
          "sign-bit/i8 dot requires little-endian source bit order");
    auto signBits = resolveLocalBlockMemoryFact(op.getSignBits());
    auto activation = resolveLocalBlockMemoryFact(op.getActivation());
    if (!signBits || !activation)
      return op.emitError(
          "sign-bit/i8 dot requires contiguous all-active local block memory facts");

    decision = SignBitI8Decision{};
    decision.signBits = *signBits;
    decision.activation = *activation;
    decision.activationScale = op.getActivationScale();
    decision.signScale = op.getSignScale();
    decision.init = op.getInit();
    return mlir::success();
  }

  mlir::LogicalResult emitSignBitI8Dot(SignBitI8DotOp op) {
    auto prepared = signBitI8Decisions.find(op.getOperation());
    if (prepared == signBitI8Decisions.end())
      return op.emitError("sign-bit/i8 physical decision was not prepared");
    const SignBitI8Decision &decision = prepared->second;
    CValue signBits = require(decision.signBits.base);
    CValue activation = require(decision.activation.base);
    CValue activationScale = require(decision.activationScale);
    CValue signScale = require(decision.signScale);
    CValue init = require(decision.init);
    if (decision.realization !=
            SignBitI8Realization::RVVWideningSignSum ||
        signBits.kind != CValueKind::Pointer ||
        activation.kind != CValueKind::Pointer ||
        activationScale.kind != CValueKind::Scalar ||
        signScale.kind != CValueKind::Scalar || init.kind != CValueKind::Scalar ||
        signBits.spelling.empty() || activation.spelling.empty() ||
        activationScale.spelling.empty() || signScale.spelling.empty() ||
        init.spelling.empty())
      return op.emitError(
          "selected sign-bit/i8 dot operands are unavailable");

    std::string vl = fresh("sign_dot_vl");
    std::string codes = fresh("sign_dot_codes");
    std::string wide = fresh("sign_dot_wide");
    std::string mask = fresh("sign_dot_mask");
    std::string negative = fresh("sign_dot_negative");
    std::string selected = fresh("sign_dot_selected");
    std::string seed = fresh("sign_dot_seed");
    std::string reduced = fresh("sign_dot_reduced");
    std::string integerSum = fresh("sign_dot_sum");
    std::string result = fresh("sign_dot");
    line("const size_t " + vl + " = __riscv_vsetvl_e8m2(32);");
    line("vint8m2_t " + codes + " = __riscv_vle8_v_i8m2(" +
         "(const int8_t *)(const void *)" + activation.spelling + ", " + vl +
         ");");
    line("vint16m4_t " + wide + " = __riscv_vsext_vf2_i16m4(" + codes +
         ", " + vl + ");");
    line("vbool4_t " + mask + " = __riscv_vlm_v_b4(" + signBits.spelling +
         ", " + vl + ");");
    line("vint16m4_t " + negative + " = __riscv_vneg_v_i16m4(" + wide +
         ", " + vl + ");");
    line("vint16m4_t " + selected + " = __riscv_vmerge_vvm_i16m4(" +
         negative + ", " + wide + ", " + mask + ", " + vl + ");");
    line("vint32m1_t " + seed + " = __riscv_vmv_v_x_i32m1(0, 1);");
    line("vint32m1_t " + reduced +
         " = __riscv_vwredsum_vs_i16m4_i32m1(" + selected + ", " + seed +
         ", " + vl + ");");
    line("const int32_t " + integerSum +
         " = __riscv_vmv_x_s_i32m1_i32(" + reduced + ");");
    line("const float " + result + " = " + init.spelling + " + (float)" +
         integerSum + " * " + activationScale.spelling + " * " +
         signScale.spelling + ";");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, result};
    markDiscardedBlockTree(decision.signBits.semanticValue);
    markDiscardedBlockTree(decision.activation.semanticValue);
    return mlir::success();
  }

  mlir::LogicalResult decideE2M1E8M0I8Dot(
      E2M1E8M0I8DotOp op, E2M1E8M0I8Decision &decision) {
    if (options.target.vlenBits != 128)
      return op.emitError(
          "E2M1/E8M0 i8 dot requires an explicit VLEN128 target fact");
    if (!options.target.littleEndian)
      return op.emitError(
          "E2M1/E8M0 i8 dot requires little-endian packed nibbles");
    auto packedCodes = resolveLocalBlockMemoryFact(op.getPackedCodes());
    auto activation = resolveLocalBlockMemoryFact(op.getActivation());
    if (!packedCodes || !activation)
      return op.emitError(
          "E2M1/E8M0 i8 dot requires contiguous all-active local block memory facts");

    decision = E2M1E8M0I8Decision{};
    decision.packedCodes = *packedCodes;
    decision.activation = *activation;
    decision.exponent = op.getExponent();
    decision.activationScale = op.getActivationScale();
    decision.init = op.getInit();
    return mlir::success();
  }

  mlir::LogicalResult emitE2M1E8M0I8Dot(E2M1E8M0I8DotOp op) {
    auto selected = e2m1E8M0I8Decisions.find(op.getOperation());
    if (selected == e2m1E8M0I8Decisions.end())
      return op.emitError("E2M1/E8M0 physical decision was not prepared");
    const E2M1E8M0I8Decision &decision = selected->second;
    CValue packed = require(decision.packedCodes.base);
    CValue activation = require(decision.activation.base);
    CValue exponent = require(decision.exponent);
    CValue activationScale = require(decision.activationScale);
    CValue init = require(decision.init);
    if (decision.realization != E2M1E8M0I8Realization::RVVVLEN128TableDot ||
        packed.kind != CValueKind::Pointer ||
        activation.kind != CValueKind::Pointer ||
        exponent.kind != CValueKind::Scalar ||
        activationScale.kind != CValueKind::Scalar ||
        init.kind != CValueKind::Scalar || packed.spelling.empty() ||
        activation.spelling.empty() || exponent.spelling.empty() ||
        activationScale.spelling.empty() || init.spelling.empty())
      return op.emitError(
          "selected E2M1/E8M0 i8 dot operands are unavailable");

    std::string vl16 = fresh("e2m1_vl16");
    std::string vl32 = fresh("e2m1_vl32");
    std::string table = fresh("e2m1_table");
    std::string packedVector = fresh("e2m1_packed");
    std::string low = fresh("e2m1_low");
    std::string high = fresh("e2m1_high");
    std::string indices = fresh("e2m1_indices");
    std::string activationVector = fresh("e2m1_activation");
    std::string decoded = fresh("e2m1_decoded");
    std::string products = fresh("e2m1_products");
    std::string seed = fresh("e2m1_seed");
    std::string reduced = fresh("e2m1_reduced");
    std::string integerSum = fresh("e2m1_sum");
    std::string result = fresh("e2m1_dot");
    line("const size_t " + vl16 + " = __riscv_vsetvl_e8m1(16);");
    line("vint8m2_t " + table +
         " = __riscv_vle8_v_i8m2(__weft_e2m1_doubled, " + vl16 + ");");
    line("vuint8m1_t " + packedVector + " = __riscv_vle8_v_u8m1(" +
         packed.spelling + ", " + vl16 + ");");
    line("vuint8m1_t " + low + " = __riscv_vand_vx_u8m1(" + packedVector +
         ", UINT8_C(15), " + vl16 + ");");
    line("vuint8m1_t " + high + " = __riscv_vsrl_vx_u8m1(" + packedVector +
         ", 4, " + vl16 + ");");
    line("vuint8m2_t " + indices +
         " = __riscv_vcreate_v_u8m1_u8m2(" + low + ", " + high + ");");
    line("const size_t " + vl32 + " = __riscv_vsetvl_e8m2(32);");
    line("vint8m2_t " + activationVector + " = __riscv_vle8_v_i8m2(" +
         "(const int8_t *)(const void *)" + activation.spelling + ", " + vl32 +
         ");");
    line("vint8m2_t " + decoded + " = __riscv_vrgather_vv_i8m2(" + table +
         ", " + indices + ", " + vl32 + ");");
    line("vint16m4_t " + products + " = __riscv_vwmul_vv_i16m4(" +
         activationVector + ", " + decoded + ", " + vl32 + ");");
    line("vint32m1_t " + seed + " = __riscv_vmv_v_x_i32m1(0, 1);");
    line("vint32m1_t " + reduced +
         " = __riscv_vwredsum_vs_i16m4_i32m1(" + products + ", " + seed +
         ", " + vl32 + ");");
    line("const int32_t " + integerSum +
         " = __riscv_vmv_x_s_i32m1_i32(" + reduced + ");");
    line("const float " + result + " = " + init.spelling + " + (float)" +
         integerSum + " * __weft_e8m0_half(" + exponent.spelling + ") * " +
         activationScale.spelling + ";");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, result};
    markDiscardedBlockTree(decision.packedCodes.semanticValue);
    markDiscardedBlockTree(decision.activation.semanticValue);
    return mlir::success();
  }

  mlir::LogicalResult decideGroupedAffineI4I8Dot(
      GroupedAffineI4I8DotOp op, GroupedAffineI4I8Decision &decision) {
    if (options.target.vlenBits != 128)
      return op.emitError(
          "grouped affine i4/i8 dot currently requires an explicit VLEN128 target fact");
    if (!options.target.littleEndian)
      return op.emitError(
          "grouped affine i4/i8 dot requires the source-declared little-endian fields");
    auto packedWeight = resolveLocalBlockMemoryFact(op.getPackedWeight());
    auto scaleMin = resolveLocalBlockMemoryFact(op.getScaleMin());
    auto activation = resolveLocalBlockMemoryFact(op.getActivation());
    auto activationSum =
        resolveLocalBlockMemoryFact(op.getActivationSumBytes());
    if (!packedWeight || !scaleMin || !activation || !activationSum)
      return op.emitError(
          "grouped affine i4/i8 dot requires contiguous all-active local block memory facts");

    decision = GroupedAffineI4I8Decision{};
    decision.packedWeight = *packedWeight;
    decision.scaleMin = *scaleMin;
    decision.activation = *activation;
    decision.activationSum = *activationSum;
    decision.dotScale = op.getDotScale();
    decision.minimumScale = op.getMinimumScale();
    decision.init = op.getInit();
    return mlir::success();
  }

  mlir::LogicalResult emitGroupedAffineI4I8Dot(GroupedAffineI4I8DotOp op) {
    auto selected = groupedAffineI4I8Decisions.find(op.getOperation());
    if (selected == groupedAffineI4I8Decisions.end())
      return op.emitError("grouped affine i4/i8 physical decision was not prepared");
    const GroupedAffineI4I8Decision &decision = selected->second;

    CValue packed = require(decision.packedWeight.base);
    CValue scales = require(decision.scaleMin.base);
    CValue activation = require(decision.activation.base);
    CValue sums = require(decision.activationSum.base);
    CValue dotScale = require(decision.dotScale);
    CValue minimumScale = require(decision.minimumScale);
    CValue init = require(decision.init);
    if (decision.realization !=
            GroupedAffineI4I8Realization::RVVVLEN128GroupedDot ||
        packed.kind != CValueKind::Pointer || scales.kind != CValueKind::Pointer ||
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
    markDiscardedBlockTree(decision.packedWeight.semanticValue);
    markDiscardedBlockTree(decision.scaleMin.semanticValue);
    markDiscardedBlockTree(decision.activation.semanticValue);
    markDiscardedBlockTree(decision.activationSum.semanticValue);
    return mlir::success();
  }

  template <typename OpTy>
  mlir::LogicalResult decideQuantCodebookI8Dot(
      OpTy op, llvm::ArrayRef<mlir::Value> blocks,
      llvm::ArrayRef<mlir::Value> scalars,
      QuantCodebookI8Decision &decision) {
    if (options.target.vlenBits != 128)
      return op.emitError(
          "quant codebook/i8 dot requires an explicit VLEN128 target fact");
    if (!options.target.littleEndian)
      return op.emitError(
          "quant codebook/i8 dot requires little-endian packed fields");
    decision = QuantCodebookI8Decision{};
    for (mlir::Value block : blocks) {
      auto fact = resolveLocalBlockMemoryFact(block);
      if (!fact)
        return op.emitError(
            "quant codebook/i8 dot requires contiguous all-active local block memory facts");
      decision.blocks.push_back(*fact);
    }
    decision.scalars.append(scalars.begin(), scalars.end());
    return mlir::success();
  }

  template <typename OpTy>
  mlir::LogicalResult emitQuantCodebookI8Dot(OpTy op,
                                              llvm::StringRef helper) {
    auto prepared = quantCodebookI8Decisions.find(op.getOperation());
    if (prepared == quantCodebookI8Decisions.end())
      return op.emitError(
          "quant codebook/i8 physical decision was not prepared");
    const QuantCodebookI8Decision &decision = prepared->second;
    if (decision.realization !=
        QuantCodebookI8Realization::RVVVLEN128LocalBlockDot)
      return op.emitError("quant codebook/i8 realization is unavailable");
    llvm::SmallVector<CValue> operands;
    for (const LocalBlockMemoryFact &block : decision.blocks) {
      CValue value = require(block.base);
      if (value.kind != CValueKind::Pointer || value.spelling.empty())
        return op.emitError(
            "quant codebook/i8 block base was not materialized");
      operands.push_back(std::move(value));
    }
    for (mlir::Value scalar : decision.scalars) {
      CValue value = require(scalar);
      if (value.kind != CValueKind::Scalar || value.spelling.empty())
        return op.emitError(
            "quant codebook/i8 scalar operand was not materialized");
      operands.push_back(std::move(value));
    }
    std::string result = fresh("quant_dot");
    llvm::SmallVector<std::string> spellings;
    for (const CValue &operand : operands)
      spellings.push_back(operand.spelling);
    line("const float " + result + " = " + helper.str() + "(" +
         llvm::join(spellings, ", ") + ");");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, result};
    for (const LocalBlockMemoryFact &block : decision.blocks)
      markDiscardedBlockTree(block.semanticValue);
    return mlir::success();
  }

  std::optional<AffineI4I8NTileDecision>
  decideAffineI4I8NTiles(AffineI4I8ContractOp contract) {
    ForOp kLoop = contract->getParentOfType<ForOp>();
    if (!kLoop || contract->getBlock() != &kLoop.getBody().front())
      return std::nullopt;
    ForOp nLoop = kLoop->getParentOfType<ForOp>();
    if (!nLoop || kLoop->getBlock() != &nLoop.getBody().front())
      return std::nullopt;
    mlir::Block &nBody = nLoop.getBody().front();
    ForOp ownedKLoop;
    StoreOp store;
    for (mlir::Operation &operation : nBody.without_terminator()) {
      if (auto candidate = mlir::dyn_cast<ForOp>(operation)) {
        if (ownedKLoop)
          return std::nullopt;
        ownedKLoop = candidate;
      } else if (auto candidate = mlir::dyn_cast<StoreOp>(operation)) {
        if (store)
          return std::nullopt;
        store = candidate;
      }
    }
    if (!ownedKLoop || ownedKLoop != kLoop)
      return std::nullopt;

    mlir::Block &kBody = kLoop.getBody().front();
    if (llvm::any_of(kBody.without_terminator(), [&](mlir::Operation &operation) {
          auto candidate = mlir::dyn_cast<AffineI4I8ContractOp>(operation);
          return candidate && candidate != contract;
        }))
      return std::nullopt;

    auto reject = [&](llvm::StringRef message)
        -> std::optional<AffineI4I8NTileDecision> {
      contract.emitError(message);
      return std::nullopt;
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
    std::optional<int64_t> selectedNTile = integerConstant(nLoop.getStep());
    int64_t kBlock = 32;
    int64_t packedBlockBytes = 304;
    if (integerConstant(nLoop.getLower()) != 0 || selectedNTile != 16 ||
        integerConstant(kLoop.getLower()) != 0 ||
        integerConstant(kLoop.getStep()) != 1)
      return reject("IME1 lowering requires the source-declared N16 and K-block traversal");
    int64_t nTile = *selectedNTile;

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
        !matchBlockAxis(activationLane.getOffset(), kBlock) ||
        !matchBinaryConstant(activationBlock.getOffset(), "mul", kBlock,
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
    if (!outputLane || !outputTile ||
        !matchBlockAxis(outputLane.getOffset(), nTile) ||
        outputTile.getOffset() != nCoordinate)
      return reject("IME1 lowering requires an explicit contiguous N16 output tile");
    mlir::Value outputRow = outputTile.getBase();

    auto zeroPointLane = zeroPointLoad.getPointer().getDefiningOp<PtrAddOp>();
    auto zeroPointBase =
        zeroPointLane ? zeroPointLane.getBase().getDefiningOp<PtrAddOp>()
                      : PtrAddOp{};
    if (!zeroPointLane || !zeroPointBase ||
        !matchBlockAxis(zeroPointLane.getOffset(), nTile) ||
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
        !matchBlockAxis(expandedByte.getInput(), nTile))
      return reject("IME1 lowering requires the source-declared packed-byte axis");

    auto packedPointer = packedBase.getDefiningOp<PtrAddOp>();
    mlir::Value linearBlock;
    if (!packedPointer ||
        !matchBinaryConstant(packedPointer.getOffset(), "mul", packedBlockBytes,
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
        integerConstant(groupDivide.getRhs()) != nTile)
      return reject("IME1 packed-block index must use the explicit N16 group");

    mlir::Value packedRoot = packedPointer.getBase();
    if (!dependsOn(codeRow, rowCoordinate) || !dependsOn(scaleRow, rowCoordinate) ||
        !dependsOn(outputRow, rowCoordinate) ||
        dependsOn(packedRoot, rowCoordinate) ||
        dependsOn(codeRow, nCoordinate) || dependsOn(scaleRow, nCoordinate) ||
        dependsOn(outputRow, kCoordinate))
      return reject("IME1 lowering cannot change the source-declared row/N/K ownership");

    AffineI4I8NTileDecision decision;
    decision.operation = contract.getOperation();
    decision.activationCodes = codeRow;
    decision.activationScales = scaleRow;
    decision.packedWeight = packedRoot;
    decision.output = outputRow;
    decision.columns = nLoop.getUpper();
    decision.blocks = kLoop.getUpper();
    decision.nTile = static_cast<unsigned>(nTile);
    decision.packedBlockBytes = static_cast<unsigned>(packedBlockBytes);
    return decision;
  }

  mlir::LogicalResult emitAffineI4I8NTiles(
      const AffineI4I8NTileDecision &decision) {
    CValue code = require(decision.activationCodes);
    CValue scales = require(decision.activationScales);
    CValue weights = require(decision.packedWeight);
    CValue outputValue = require(decision.output);
    std::string columns = expression(decision.columns);
    std::string blocks = expression(decision.blocks);
    if (code.kind != CValueKind::Pointer || scales.kind != CValueKind::Pointer ||
        weights.kind != CValueKind::Pointer ||
        outputValue.kind != CValueKind::Pointer || code.spelling.empty() ||
        scales.spelling.empty() || weights.spelling.empty() ||
        outputValue.spelling.empty() || columns.empty() || blocks.empty())
      return decision.operation->emitError(
          "IME1 emission could not project the selected source operands");
    std::string nTile = fresh("ime_n");
    std::string nCount = fresh("ime_n_count");
    line("for (size_t " + nTile + " = 0; " + nTile + " < " +
         columns + "; " + nTile + " += " +
         std::to_string(decision.nTile) + ") {");
    ++indent;
    line("const size_t " + nCount + " = (" + columns + " - " +
         nTile + ") < " + std::to_string(decision.nTile) + " ? (" +
         columns + " - " + nTile + ") : " +
         std::to_string(decision.nTile) + ";");
    line("__weft_ime1_affine_i4_i8_n16(" + scales.spelling + ", " +
         code.spelling + ", " + weights.spelling + " + (" +
         nTile + " / " + std::to_string(decision.nTile) + ") * " +
         blocks + " * " + std::to_string(decision.packedBlockBytes) +
         ", " + outputValue.spelling + " + " + nTile + ", " + nCount + ", " +
         blocks + ");");
    --indent;
    line("}");
    return mlir::success();
  }

  std::optional<F16GemmNTileDecision> decideF16GemmNTiles(MatmulOp matmul) {
    ForOp kLoop = matmul->getParentOfType<ForOp>();
    if (!kLoop || matmul->getBlock() != &kLoop.getBody().front())
      return std::nullopt;
    ForOp nLoop = kLoop->getParentOfType<ForOp>();
    if (!nLoop || kLoop->getBlock() != &nLoop.getBody().front())
      return std::nullopt;
    if (nLoop.getNumResults() != 0)
      return std::nullopt;
    ForOp mLoop = nLoop->getParentOfType<ForOp>();
    if (!mLoop || mLoop.getBody().empty() || nLoop->getBlock() != &mLoop.getBody().front())
      return std::nullopt;

    mlir::Block &nBody = nLoop.getBody().front();
    ForOp ownedKLoop;
    StoreOp store;
    for (mlir::Operation &operation : nBody.without_terminator()) {
      if (auto candidate = mlir::dyn_cast<ForOp>(operation)) {
        if (ownedKLoop)
          return std::nullopt;
        ownedKLoop = candidate;
      }
      if (auto candidate = mlir::dyn_cast<StoreOp>(operation)) {
        if (store)
          return std::nullopt;
        store = candidate;
      }
    }
    if (!ownedKLoop || ownedKLoop != kLoop || !store ||
        kLoop.getNumResults() != 1 ||
        store.getValue() != kLoop.getResult(0) ||
        kLoop.getOperands().size() != 4)
      return std::nullopt;

    mlir::Block &kBody = kLoop.getBody().front();
    if (llvm::any_of(kBody.without_terminator(), [&](mlir::Operation &operation) {
          auto candidate = mlir::dyn_cast<MatmulOp>(operation);
          return candidate && candidate != matmul;
        }) ||
        matmul.getResult() !=
                         mlir::cast<YieldOp>(kBody.getTerminator()).getOperand(0) ||
        matmul.getInit() != kBody.getArgument(1) ||
        matmul.getOrder() != "relaxed" || matmul.getMath() != "native" ||
        !matmul.getAccDtype().isF32())
      return std::nullopt;

    LoadOp lhsLoad = matmul.getLhs().getDefiningOp<LoadOp>();
    LoadOp rhsLoad = matmul.getRhs().getDefiningOp<LoadOp>();
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

    auto physicalExtent = [&](mlir::Value value) -> std::optional<int64_t> {
      if (std::optional<int64_t> constant = integerConstantValue(value))
        return constant;
      auto meta = value.getDefiningOp<MetaValueOp>();
      auto argument = meta
                          ? mlir::dyn_cast<mlir::BlockArgument>(meta.getInput())
                          : mlir::BlockArgument{};
      if (!argument)
        return std::nullopt;
      llvm::StringRef name = mlir::cast<mlir::StringAttr>(
                                 kernel.getArgNames()[argument.getArgNumber()])
                                 .getValue();
      auto binding = options.metaBindings.find(name);
      return binding == options.metaBindings.end()
                 ? std::nullopt
                 : std::optional<int64_t>(binding->second);
    };
    std::optional<int64_t> rowTile = physicalExtent(mLoop.getStep());
    std::optional<int64_t> columnTile = physicalExtent(nLoop.getStep());
    std::optional<int64_t> reductionTile = physicalExtent(kLoop.getStep());
    if (!rowTile || *rowTile <= 0 || *rowTile > 8 || !columnTile ||
        *columnTile <= 0 || !reductionTile || *reductionTile <= 0 ||
        *columnTile <= 0 || !reductionTile || *reductionTile <= 0)
      return std::nullopt;

    F16GemmNTileDecision decision;
    decision.operation = matmul.getOperation();
    decision.row = mCoordinate;
    decision.rowUpper = mLoop.getUpper();
    decision.lhs = lhsRoot;
    decision.rhs = rhsRoot;
    decision.output = outputRoot;
    decision.lhsStride = lhsStride;
    decision.rhsStride = rhsStride;
    decision.outputStride = outputStride;
    decision.nLower = nLoop.getLower();
    decision.nUpper = nLoop.getUpper();
    decision.sourceNTile = static_cast<unsigned>(*columnTile);
    decision.kLower = kLoop.getLower();
    decision.kUpper = kLoop.getUpper();
    decision.sourceKTile = static_cast<unsigned>(*reductionTile);
    decision.rowTile = static_cast<unsigned>(*rowTile);
    llvm::SmallVector<unsigned> rowCandidates;
    if (options.backend.f16RowMicrotile != 0)
      rowCandidates.push_back(
          static_cast<unsigned>(options.backend.f16RowMicrotile));
    else
      for (unsigned rows = decision.rowTile; rows > 0; --rows)
        if (decision.rowTile % rows == 0)
          rowCandidates.push_back(rows);
    llvm::SmallVector<unsigned> lmulCandidates;
    if (options.backend.f16InputLMUL != 0)
      lmulCandidates.push_back(
          static_cast<unsigned>(options.backend.f16InputLMUL));
    else
      lmulCandidates = {1, 2, 4};
    std::optional<std::pair<unsigned, unsigned>> selected;
    for (unsigned inputLMUL : lmulCandidates) {
      if (inputLMUL != 1 && inputLMUL != 2 && inputLMUL != 4)
        continue;
      for (unsigned rows : rowCandidates) {
        if (rows == 0 || rows > decision.rowTile ||
            decision.rowTile % rows != 0)
          continue;
        unsigned registerGroups = (3 * rows + 1) * inputLMUL;
        if (2 * inputLMUL <= 8 &&
            registerGroups + 1 <
                static_cast<unsigned>(options.target.vectorRegisters)) {
          selected = std::make_pair(rows, inputLMUL);
          break;
        }
      }
      if (selected)
        break;
    }
    if (!selected)
      return std::nullopt;
    decision.rowMicrotile = selected->first;
    decision.inputLMUL = selected->second;
    decision.computeLMUL = 2 * selected->second;
    decision.kStripSchedule = VLAStripSchedule::FullThenTail;
    return decision;
  }

  mlir::LogicalResult emitF16GemmNTiles(
      const F16GemmNTileDecision &decision) {
    if (decision.kStripSchedule != VLAStripSchedule::FullThenTail)
      return decision.operation->emitError(
          "F16 contraction has no selected K strip schedule");
    CValue rowValue = require(decision.row);
    CValue lhsValue = require(decision.lhs);
    CValue rhsValue = require(decision.rhs);
    CValue outputValue = require(decision.output);
    CValue lhsStrideValue = require(decision.lhsStride);
    CValue rhsStrideValue = require(decision.rhsStride);
    CValue outputStrideValue = require(decision.outputStride);
    std::string rowUpper = expression(decision.rowUpper);
    std::string nLower = expression(decision.nLower);
    std::string nUpper = expression(decision.nUpper);
    std::string kLower = expression(decision.kLower);
    std::string kUpper = expression(decision.kUpper);
    if (rowValue.spelling.empty() || rowUpper.empty() ||
        lhsValue.kind != CValueKind::Pointer ||
        rhsValue.kind != CValueKind::Pointer ||
        outputValue.kind != CValueKind::Pointer || lhsValue.spelling.empty() ||
        rhsValue.spelling.empty() || outputValue.spelling.empty() ||
        lhsStrideValue.spelling.empty() || rhsStrideValue.spelling.empty() ||
        outputStrideValue.spelling.empty() || nLower.empty() || nUpper.empty() ||
        kLower.empty() || kUpper.empty())
      return decision.operation->emitError(
          "F16 emission could not project the selected source operands");
    const std::string &row = rowValue.spelling;
    const std::string &lhs = lhsValue.spelling;
    const std::string &rhs = rhsValue.spelling;
    const std::string &outputPointer = outputValue.spelling;
    const std::string &lhsStride = lhsStrideValue.spelling;
    const std::string &rhsStride = rhsStrideValue.spelling;
    const std::string &outputStride = outputStrideValue.spelling;
    std::string nTile = fresh("gemm_n");
    std::string rows = fresh("gemm_rows");
    line("const size_t " + rows + " = (" + rowUpper + " - " + row +
         ") < " + std::to_string(decision.rowTile) + " ? (" + rowUpper +
         " - " + row + ") : " +
         std::to_string(decision.rowTile) + ";");
    line("for (size_t " + nTile + " = " + nLower + "; " + nTile + " < " +
         nUpper + "; " + nTile + " += " +
         std::to_string(decision.sourceNTile) + ") {");
    ++indent;
    std::string nTileEnd = fresh("gemm_n_tile_end");
    line("const size_t " + nTileEnd + " = (" + nUpper + " - " +
         nTile + ") < " + std::to_string(decision.sourceNTile) + " ? " +
         nUpper + " : " + nTile + " + " +
         std::to_string(decision.sourceNTile) + ";");
    std::string nColumn = fresh("gemm_n_column");
    line("for (size_t " + nColumn + " = " + nTile + "; " + nColumn + " < " +
         nTileEnd + "; ++" + nColumn + ") {");
    ++indent;

    auto emitRows = [&](unsigned rowCount, llvm::StringRef rowBase,
                        llvm::StringRef condition, bool first) {
      line(std::string(first ? "if" : "else if") + " (" +
           condition.str() + ") {");
      ++indent;
      llvm::SmallVector<std::string> accumulators;
      std::string fullVL = fresh("gemm_full_vl");
      line("const size_t " + fullVL + " = __riscv_vsetvlmax_e16m" +
           std::to_string(decision.inputLMUL) + "();");
      for (unsigned row = 0; row < rowCount; ++row) {
        std::string accumulator = fresh("gemm_acc");
        line("vfloat32m" + std::to_string(decision.computeLMUL) + "_t " +
             accumulator + " = __riscv_vfmv_v_f_f32m" +
             std::to_string(decision.computeLMUL) + "(0.0f, " +
             fullVL + ");");
        accumulators.push_back(std::move(accumulator));
      }
      std::string sourceK = fresh("gemm_source_k");
      std::string sourceKEnd = fresh("gemm_source_k_end");
      line("for (size_t " + sourceK + " = " + kLower + "; " + sourceK +
           " < " + kUpper + "; " + sourceK + " += " +
           std::to_string(decision.sourceKTile) + ") {");
      ++indent;
      line("const size_t " + sourceKEnd + " = (" + kUpper + " - " +
           sourceK + ") < " + std::to_string(decision.sourceKTile) + " ? " +
           kUpper + " : " + sourceK + " + " +
           std::to_string(decision.sourceKTile) + ";");
      std::string inner = fresh("gemm_k");
      std::string fullEnd = fresh("gemm_k_full_end");
      line("const size_t " + fullEnd + " = " + sourceKEnd + " - ((" +
           sourceKEnd + " - " + sourceK + ") % " + fullVL + ");");
      line("for (size_t " + inner + " = " + sourceK + "; " + inner +
           " < " + fullEnd + "; " + inner + " += " + fullVL + ") {");
      ++indent;
      std::string rhsVector = fresh("gemm_b");
      line("vfloat16m" + std::to_string(decision.inputLMUL) + "_t " +
           rhsVector + " = __riscv_vle16_v_f16m" +
           std::to_string(decision.inputLMUL) + "(" +
           rhs + " + " + nColumn + " * " + rhsStride +
           " + " + inner + ", " + fullVL + ");");
      for (unsigned row = 0; row < rowCount; ++row) {
        std::string lhsVector = fresh("gemm_a");
        line("vfloat16m" + std::to_string(decision.inputLMUL) + "_t " +
             lhsVector + " = __riscv_vle16_v_f16m" +
             std::to_string(decision.inputLMUL) + "(" +
             lhs + " + (" + rowBase.str() + " + " +
             std::to_string(row) + ") * " + lhsStride + " + " +
             inner + ", " + fullVL + ");");
        line(accumulators[row] + " = __riscv_vfwmacc_vv_f32m" +
             std::to_string(decision.computeLMUL) + "(" +
             accumulators[row] + ", " + lhsVector + ", " + rhsVector + ", " +
             fullVL + ");");
      }
      --indent;
      line("}");
      line("if (" + fullEnd + " < " + sourceKEnd + ") {");
      ++indent;
      std::string tailVL = fresh("gemm_tail_vl");
      line("const size_t " + tailVL + " = __riscv_vsetvl_e16m" +
           std::to_string(decision.inputLMUL) + "(" + sourceKEnd + " - " +
           fullEnd + ");");
      std::string tailRhs = fresh("gemm_tail_b");
      line("vfloat16m" + std::to_string(decision.inputLMUL) + "_t " + tailRhs +
           " = __riscv_vle16_v_f16m" +
           std::to_string(decision.inputLMUL) + "(" + rhs + " + " +
           nColumn + " * " + rhsStride + " + " + fullEnd + ", " +
           tailVL + ");");
      for (unsigned row = 0; row < rowCount; ++row) {
        std::string tailLhs = fresh("gemm_tail_a");
        line("vfloat16m" + std::to_string(decision.inputLMUL) + "_t " +
             tailLhs + " = __riscv_vle16_v_f16m" +
             std::to_string(decision.inputLMUL) + "(" + lhs + " + (" +
             rowBase.str() + " + " + std::to_string(row) + ") * " +
             lhsStride + " + " + fullEnd + ", " + tailVL + ");");
        line(accumulators[row] + " = __riscv_vfwmacc_vv_f32m" +
             std::to_string(decision.computeLMUL) + "_tu(" +
             accumulators[row] + ", " + tailLhs + ", " + tailRhs + ", " +
             tailVL + ");");
      }
      --indent;
      line("}");
      --indent;
      line("}");
      for (unsigned row = 0; row < rowCount; ++row) {
        std::string seed = fresh("gemm_seed");
        std::string partial = fresh("gemm_partial");
        std::string scalar = fresh("gemm_sum");
        line("vfloat32m1_t " + seed +
             " = __riscv_vfmv_v_f_f32m1(0.0f, 1);");
        line("vfloat32m1_t " + partial +
             " = __riscv_vfredusum_vs_f32m" +
             std::to_string(decision.computeLMUL) + "_f32m1(" +
             accumulators[row] + ", " + seed + ", " + fullVL + ");");
        line("float " + scalar + " = __riscv_vfmv_f_s_f32m1_f32(" + partial +
             ");");
        line("*(" + outputPointer + " + (" + rowBase.str() + " + " +
             std::to_string(row) + ") * " + outputStride + " + " +
             nColumn + ") = " + scalar + ";");
      }
      --indent;
      line("}");
    };
    if (decision.rowMicrotile == decision.rowTile) {
      for (unsigned rowCount = decision.rowMicrotile; rowCount > 0; --rowCount)
        emitRows(rowCount, row,
                 rows + " == " + std::to_string(rowCount),
                 rowCount == decision.rowMicrotile);
    } else {
      std::string rowBase = fresh("gemm_row");
      line("for (size_t " + rowBase + " = 0; " + rowBase + " < " + rows +
           "; " + rowBase + " += " +
           std::to_string(decision.rowMicrotile) + ") {");
      ++indent;
      std::string localRows = fresh("gemm_local_rows");
      line("const size_t " + localRows + " = " + rows + " - " + rowBase +
           ";");
      for (unsigned rowCount = decision.rowMicrotile; rowCount > 0; --rowCount)
        emitRows(rowCount, "(" + row + " + " + rowBase + ")",
                 rowCount == decision.rowMicrotile
                     ? "(" + localRows + " >= " +
                           std::to_string(decision.rowMicrotile) + ")"
                     : "(" + localRows + " == " + std::to_string(rowCount) +
                           ")",
                 rowCount == decision.rowMicrotile);
      --indent;
      line("}");
    }
    --indent;
    line("}");
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
      llvm::SmallVectorImpl<BlockAxisOp> &axes, mlir::Operation *owner,
      bool markDiscarded = true) {
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
          if (mlir::failed(collectBlockClosure(operand, closure, axes, owner,
                                               markDiscarded)))
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
        if (!fieldIsLive && markDiscarded)
          markDiscardedBlockTree(operand);
      }
      return mlir::success();
    }
    if (auto axis = mlir::dyn_cast<BlockAxisOp>(definition))
      axes.push_back(axis);
    for (mlir::Value operand : definition->getOperands())
      if (containsBlockType(operand.getType()) &&
          mlir::failed(collectBlockClosure(operand, closure, axes, owner,
                                           markDiscarded)))
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
    if (activeBlockByteShape == RVVBlockVectorShape::E8MF4 &&
        resultKind == BlockValueKind::U8 &&
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
      suffix = activeBlockByteShape == RVVBlockVectorShape::E8MF4 ? "u8mf4"
                                                                  : "u8m1";
      cType = activeBlockByteShape == RVVBlockVectorShape::E8MF4
                  ? "vuint8mf4_t"
                  : "vuint8m1_t";
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
      result.vectorShape = activeBlockByteShape;
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
      if (activeBlockByteShape == RVVBlockVectorShape::E8MF4) {
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
    std::string suffix = activeBlockByteShape == RVVBlockVectorShape::E8MF4
                             ? "u8mf4"
                             : "u8m1";
    std::string cType = activeBlockByteShape == RVVBlockVectorShape::E8MF4
                            ? "vuint8mf4_t"
                            : "vuint8m1_t";
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
    result.vectorShape = activeBlockByteShape;
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
      auto found = blockDecodeDecisions.find(op.getOperation());
      if (found == blockDecodeDecisions.end())
        return op.emitError("block decode has no selected physical decision");
      return emitBlockDecode(op, found->second, blockValues, vl);
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

  bool blockClosureNeedsLaneVector(
      BlockAxisOp axis,
      const llvm::DenseSet<mlir::Operation *> &closure) const {
    return llvm::any_of(axis.getResult().getUsers(),
                        [&](mlir::Operation *user) {
                          if (!closure.contains(user))
                            return false;
                          auto pointer = mlir::dyn_cast<PtrAddOp>(user);
                          return !pointer ||
                                 pointer.getOffset() != axis.getResult();
                        });
  }

  BlockStorePhysicalDecision decideBlockStorePhysical(
      int64_t extent, BlockAxisOp axis,
      const llvm::DenseSet<mlir::Operation *> &closure) const {
    bool needsLaneVector = blockClosureNeedsLaneVector(axis, closure);
    bool microClosure = llvm::all_of(closure, [](mlir::Operation *operation) {
      return mlir::isa<BlockAxisOp, PtrAddOp, LoadOp, BinaryOp, CastOp>(
          operation);
    });
    llvm::SmallVector<BlockStorePhysicalDecision> candidates;
    candidates.push_back(BlockStorePhysicalDecision{
        BlockStoreRealization::RVVE8MF4MicroStrips,
        RVVBlockVectorShape::E8MF4, 4, false});
    candidates.push_back(BlockStorePhysicalDecision{
        BlockStoreRealization::RVVE8M1FixedStrips,
        RVVBlockVectorShape::E8M1, 16, false});
    candidates.push_back(BlockStorePhysicalDecision{
        BlockStoreRealization::RVVE8M1DynamicStrips,
        RVVBlockVectorShape::E8M1, 16, needsLaneVector});
    for (const BlockStorePhysicalDecision &candidate : candidates) {
      if (candidate.realization ==
              BlockStoreRealization::RVVE8MF4MicroStrips &&
          (needsLaneVector || extent != 32 || !microClosure))
        continue;
      if (candidate.realization == BlockStoreRealization::RVVE8M1FixedStrips &&
          (needsLaneVector || extent != 32))
        continue;
      unsigned dataGroups =
          candidate.byteShape == RVVBlockVectorShape::E8MF4 ? 4 : 8;
      unsigned laneGroups = candidate.needsLaneVector ? 2 : 0;
      if (dataGroups + laneGroups + 1 >=
          static_cast<unsigned>(options.target.vectorRegisters))
        continue;
      return candidate;
    }
    return BlockStorePhysicalDecision{};
  }

  BlockReducePhysicalDecision decideBlockReducePhysical(
      int64_t extent, BlockAxisOp axis,
      const llvm::DenseSet<mlir::Operation *> &closure) const {
    bool needsLaneVector = blockClosureNeedsLaneVector(axis, closure);
    llvm::SmallVector<BlockReducePhysicalDecision> candidates{
        {BlockReduceRealization::RVVE8M1FixedStrips, 16, false},
        {BlockReduceRealization::RVVE8M1DynamicStrips, 16,
         needsLaneVector}};
    for (const BlockReducePhysicalDecision &candidate : candidates) {
      if (candidate.realization == BlockReduceRealization::RVVE8M1FixedStrips &&
          (needsLaneVector || extent != 32))
        continue;
      unsigned liveGroups = 8 + (candidate.needsLaneVector ? 2 : 0) + 1;
      if (liveGroups + 1 >=
          static_cast<unsigned>(options.target.vectorRegisters))
        continue;
      return candidate;
    }
    return BlockReducePhysicalDecision{};
  }

  std::optional<int64_t> f32BlockExtent(mlir::Type type) const {
    if (auto masked = mlir::dyn_cast<MaskedType>(type))
      type = masked.getValueType();
    auto block = mlir::dyn_cast<BlockType>(type);
    if (!block || block.getShape().size() != 1 ||
        !block.getElementType().isF32())
      return std::nullopt;
    return block.getShape().front();
  }

  bool isSupportedBlockReduction(ReduceOp reduction) const {
    auto input = mlir::dyn_cast<BlockType>(reduction.getInput().getType());
    return input && input.getShape().size() == 1 &&
           reduction.getAxis() == 0 && reduction.getKind() == "add" &&
           reduction.getResult().getType().isSignedInteger(32) &&
           input.getElementType().isSignedInteger(32) &&
           isTrue(reduction.getWhere());
  }

  bool isOwnedByPreparedRegion(mlir::Operation *operation) const {
    if (operation->getParentOfType<VLAOp>())
      return true;
    for (mlir::Operation *parent = operation->getParentOp(); parent;
         parent = parent->getParentOp())
      if (affineI4I8Decisions.contains(parent) ||
          f16ContractDecisions.contains(parent))
        return true;
    return false;
  }

  bool isMaterializedF32BlockStore(StoreOp store) const {
    mlir::Operation *definition = store.getValue().getDefiningOp();
    return mlir::isa_and_nonnull<FullOp, SymmetricI4I8ContractOp,
                                 AffineI4I8ContractOp>(definition);
  }

  bool blockClosureEscapes(
      const llvm::DenseSet<mlir::Operation *> &closure,
      const llvm::DenseSet<mlir::Operation *> &owners) const {
    for (mlir::Operation *operation : closure)
      for (mlir::Value result : operation->getResults()) {
        if (!containsBlockType(result.getType()))
          continue;
        for (mlir::Operation *user : result.getUsers())
          if (!owners.contains(user) && !closure.contains(user) &&
              !llvm::all_of(user->getResults(),
                            [](mlir::Value value) { return value.use_empty(); }))
            return true;
      }
    return false;
  }

  mlir::LogicalResult prepareBlockStoreGroup(
      StoreOp op, llvm::DenseSet<mlir::Operation *> &plannedStores) {
    std::optional<int64_t> extent = f32BlockExtent(op.getValue().getType());
    if (!extent || *extent <= 0 || *extent > 65535 || !isTrue(op.getWhere()))
      return op.emitError(
          "RVV block store requires an all-active rank-one f32 value");
    llvm::DenseSet<mlir::Operation *> closure;
    llvm::SmallVector<BlockAxisOp> axes;
    if (mlir::failed(collectBlockClosure(op.getPointer(), closure, axes,
                                         op.getOperation(), false)) ||
        mlir::failed(collectBlockClosure(op.getValue(), closure, axes,
                                         op.getOperation(), false)))
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
    llvm::SmallVector<mlir::Operation *> interstitial;
    for (mlir::Operation *candidate = op->getNextNode(); candidate;
         candidate = candidate->getNextNode()) {
      if (auto store = mlir::dyn_cast<StoreOp>(candidate)) {
        std::optional<int64_t> candidateExtent =
            f32BlockExtent(store.getValue().getType());
        if (plannedStores.contains(candidate) || candidateExtent != extent ||
            !isTrue(store.getWhere()) ||
            localF32ContractDecisions.contains(candidate) ||
            isOwnedByPreparedRegion(candidate) ||
            isMaterializedF32BlockStore(store))
          break;
        llvm::DenseSet<mlir::Operation *> candidateClosure;
        llvm::SmallVector<BlockAxisOp> candidateAxes;
        if (mlir::failed(collectBlockClosure(store.getPointer(), candidateClosure,
                                             candidateAxes,
                                             store.getOperation(), false)) ||
            mlir::failed(collectBlockClosure(store.getValue(), candidateClosure,
                                             candidateAxes,
                                             store.getOperation(), false)) ||
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
        interstitial.push_back(candidate);
        continue;
      }
      break;
    }

    llvm::DenseSet<mlir::Operation *> owners;
    for (StoreOp store : stores)
      owners.insert(store.getOperation());
    if (blockClosureEscapes(closure, owners))
      return op.emitError(
          "block producer escapes its operation-local store closure");

    BlockStoreGroupDecision decision;
    decision.operation = op.getOperation();
    decision.axis = axis.getOperation();
    decision.extent = *extent;
    decision.physical = decideBlockStorePhysical(*extent, axis, closure);
    for (StoreOp store : stores) {
      decision.stores.push_back(store.getOperation());
      plannedStores.insert(store.getOperation());
    }
    decision.closure.append(closure.begin(), closure.end());
    decision.interstitial = std::move(interstitial);
    blockStoreGroupDecisions.try_emplace(op.getOperation(), std::move(decision));
    return mlir::success();
  }

  mlir::LogicalResult prepareBlockReduceGroup(
      ReduceOp op, llvm::DenseSet<mlir::Operation *> &plannedReductions) {
    auto inputType = mlir::cast<BlockType>(op.getInput().getType());
    int64_t extent = inputType.getShape().front();
    if (extent <= 0 || extent > 65535)
      return op.emitError(
          "RVV block reduction requires a static extent in [1, 65535]");
    llvm::DenseSet<mlir::Operation *> closure;
    llvm::SmallVector<BlockAxisOp> axes;
    if (mlir::failed(collectBlockClosure(op.getInput(), closure, axes,
                                         op.getOperation(), false)))
      return mlir::failure();
    if (axes.size() != 1)
      return op.emitError(
          "RVV block reduction requires exactly one local logical block axis");
    BlockAxisOp axis = axes.front();
    auto axisType = axis.getResult().getType();
    if (axisType.getShape().size() != 1 ||
        axisType.getShape().front() != extent)
      return op.emitError("block reduction extent does not match its logical axis");

    llvm::SmallVector<ReduceOp> reductions{op};
    for (mlir::Operation *candidate = op->getNextNode(); candidate;
         candidate = candidate->getNextNode()) {
      if (auto reduction = mlir::dyn_cast<ReduceOp>(candidate)) {
        if (plannedReductions.contains(candidate) ||
            !isSupportedBlockReduction(reduction))
          break;
        auto candidateType = mlir::cast<BlockType>(reduction.getInput().getType());
        llvm::DenseSet<mlir::Operation *> candidateClosure;
        llvm::SmallVector<BlockAxisOp> candidateAxes;
        if (candidateType.getShape().front() != extent ||
            mlir::failed(collectBlockClosure(reduction.getInput(),
                                             candidateClosure, candidateAxes,
                                             reduction.getOperation(), false)) ||
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
    llvm::DenseSet<mlir::Operation *> owners;
    for (ReduceOp reduction : reductions)
      owners.insert(reduction.getOperation());
    if (blockClosureEscapes(closure, owners))
      return op.emitError(
          "block producer escapes its operation-local reduction closure");

    BlockReduceGroupDecision decision;
    decision.operation = op.getOperation();
    decision.axis = axis.getOperation();
    decision.extent = extent;
    decision.physical = decideBlockReducePhysical(extent, axis, closure);
    for (ReduceOp reduction : reductions) {
      decision.reductions.push_back(reduction.getOperation());
      plannedReductions.insert(reduction.getOperation());
    }
    decision.closure.append(closure.begin(), closure.end());
    blockReduceGroupDecisions.try_emplace(op.getOperation(), std::move(decision));
    return mlir::success();
  }

  mlir::LogicalResult prepareBlockGroupDecisions() {
    bool failed = false;
    llvm::DenseSet<mlir::Operation *> plannedStores;
    llvm::DenseSet<mlir::Operation *> plannedReductions;
    kernel.walk([&](StoreOp store) {
      if (failed || plannedStores.contains(store.getOperation()) ||
          !hasBlockPayload(store.getOperation()) ||
          localF32ContractDecisions.contains(store.getOperation()) ||
          isOwnedByPreparedRegion(store.getOperation()) ||
          isMaterializedF32BlockStore(store))
        return;
      if (mlir::failed(prepareBlockStoreGroup(store, plannedStores)))
        failed = true;
    });
    kernel.walk([&](ReduceOp reduction) {
      if (failed || plannedReductions.contains(reduction.getOperation()) ||
          reduction->getParentOfType<VLAOp>())
        return;
      if (!isSupportedBlockReduction(reduction)) {
        reduction.emitError(
            "RVV block reduction currently requires all-active rank-one i32 add");
        failed = true;
        return;
      }
      if (mlir::failed(prepareBlockReduceGroup(reduction, plannedReductions)))
        failed = true;
    });
    return failed ? mlir::failure() : mlir::success();
  }

  mlir::LogicalResult emitBlockStores(StoreOp op) {
    if (auto lowered = tryEmitLocalF32ContractBlockStore(op))
      return *lowered;
    auto selected = blockStoreGroupDecisions.find(op.getOperation());
    if (selected == blockStoreGroupDecisions.end())
      return op.emitError("block store has no selected physical group decision");
    const BlockStoreGroupDecision &decision = selected->second;
    int64_t extent = decision.extent;
    BlockAxisOp axis = mlir::cast<BlockAxisOp>(decision.axis);
    llvm::DenseSet<mlir::Operation *> closure;
    closure.insert(decision.closure.begin(), decision.closure.end());
    llvm::DenseSet<mlir::Operation *> storeOps;
    storeOps.insert(decision.stores.begin(), decision.stores.end());
    for (mlir::Operation *interstitial : decision.interstitial) {
      if (mlir::failed(emitOperation(interstitial)))
        return mlir::failure();
      consumed.insert(interstitial);
    }

    std::string offset = expression(axis.getOffset());
    if (offset.empty())
      return op.emitError("block store axis offset is unavailable");
    std::string strip = fresh("block_i");
    std::string vl = fresh("block_vl");
    std::string lane = fresh("block_lane");
    const BlockStorePhysicalDecision &physical = decision.physical;

    auto emitStrip = [&](llvm::StringRef stripOffset,
                         llvm::StringRef activeVL) -> mlir::LogicalResult {
      llvm::DenseMap<mlir::Value, BlockValue> blockValues;
      BlockValue coordinate{axis.getResult().getType(), BlockValueKind::Index,
                            physical.needsLaneVector ? lane : ""};
      coordinate.contiguousIndex =
          "(" + stripOffset.str() + " + " + offset + ")";
      blockValues[axis.getResult()] = std::move(coordinate);
      RVVBlockVectorShape previousBlockByteShape = activeBlockByteShape;
      activeBlockByteShape = physical.byteShape;
      for (mlir::Operation &candidate : *op->getBlock()) {
        if (mlir::isa<BlockAxisOp>(candidate) ||
            (!closure.contains(&candidate) && !storeOps.contains(&candidate)))
          continue;
        if (mlir::failed(emitBlockOperation(&candidate, blockValues, activeVL))) {
          activeBlockByteShape = previousBlockByteShape;
          return mlir::failure();
        }
      }
      activeBlockByteShape = previousBlockByteShape;
      return mlir::success();
    };

    if (physical.realization ==
        BlockStoreRealization::RVVE8MF4MicroStrips) {
      line("const size_t " + vl + " = __riscv_vsetvl_e8mf4(" +
           std::to_string(physical.stripVL) + ");");
      llvm::SmallVector<llvm::DenseMap<mlir::Value, BlockValue>, 8>
          stripValues;
      for (int64_t stripOffset = 0; stripOffset < extent;
           stripOffset += physical.stripVL) {
        llvm::DenseMap<mlir::Value, BlockValue> blockValues;
        BlockValue coordinate{axis.getResult().getType(),
                              BlockValueKind::Index, ""};
        coordinate.contiguousIndex =
            "(" + std::to_string(stripOffset) + " + " + offset + ")";
        blockValues[axis.getResult()] = std::move(coordinate);
        stripValues.push_back(std::move(blockValues));
      }
      RVVBlockVectorShape previousBlockByteShape = activeBlockByteShape;
      activeBlockByteShape = physical.byteShape;
      for (mlir::Operation &candidate : *op->getBlock()) {
        if (mlir::isa<BlockAxisOp>(candidate) ||
            (!closure.contains(&candidate) && !storeOps.contains(&candidate)))
          continue;
        for (auto &blockValues : stripValues)
          if (mlir::failed(emitBlockOperation(&candidate, blockValues, vl))) {
            activeBlockByteShape = previousBlockByteShape;
            return mlir::failure();
          }
      }
      activeBlockByteShape = previousBlockByteShape;
    } else if (physical.realization ==
               BlockStoreRealization::RVVE8M1FixedStrips) {
      line("const size_t " + vl + " = __riscv_vsetvl_e8m1(" +
           std::to_string(physical.stripVL) + ");");
      if (mlir::failed(emitStrip("0", vl)) ||
          mlir::failed(
              emitStrip(std::to_string(physical.stripVL), vl)))
        return mlir::failure();
    } else {
      line("for (size_t " + strip + " = 0; " + strip + " < " +
           std::to_string(extent) + ";) {");
      ++indent;
      line("const size_t " + vl + " = __riscv_vsetvl_e8m1((" +
           std::to_string(extent) + " - " + strip + ") < " +
           std::to_string(physical.stripVL) + " ? (" +
           std::to_string(extent) + " - " + strip + ") : " +
           std::to_string(physical.stripVL) + ");");
      if (physical.needsLaneVector) {
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
    consumed.insert(decision.stores.begin(), decision.stores.end());
    return mlir::success();
  }

  mlir::LogicalResult emitBlockReduce(ReduceOp op) {
    auto selected = blockReduceGroupDecisions.find(op.getOperation());
    if (selected == blockReduceGroupDecisions.end())
      return op.emitError(
          "block reduction has no selected physical group decision");
    const BlockReduceGroupDecision &decision = selected->second;
    int64_t extent = decision.extent;
    BlockAxisOp axis = mlir::cast<BlockAxisOp>(decision.axis);
    llvm::DenseSet<mlir::Operation *> closure;
    closure.insert(decision.closure.begin(), decision.closure.end());
    llvm::SmallVector<ReduceOp> reductions;
    for (mlir::Operation *reduction : decision.reductions)
      reductions.push_back(mlir::cast<ReduceOp>(reduction));

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
    const BlockReducePhysicalDecision &physical = decision.physical;

    if (physical.realization ==
        BlockReduceRealization::RVVE8M1FixedStrips) {
      line("const size_t " + vl + " = __riscv_vsetvl_e8m1(" +
           std::to_string(physical.stripVL) + ");");
      llvm::SmallVector<llvm::SmallVector<BlockValue, 2>, 2> stripInputs(
          reductions.size());
      llvm::DenseMap<mlir::Operation *, size_t> reductionIndices;
      for (auto [index, reduction] : llvm::enumerate(reductions))
        reductionIndices[reduction.getOperation()] = index;
      for (int64_t stripOffset = 0; stripOffset < extent;
           stripOffset += physical.stripVL) {
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
         std::to_string(extent) + " - " + strip + ") < " +
         std::to_string(physical.stripVL) + " ? (" +
         std::to_string(extent) + " - " + strip + ") : " +
         std::to_string(physical.stripVL) + ");");

    if (physical.needsLaneVector) {
      line("vuint16m2_t " + lane + " = __riscv_vid_v_u16m2(" + vl + ");");
      line(lane + " = __riscv_vadd_vx_u16m2(" + lane + ", " + strip + " + " +
           offset + ", " + vl + ");");
    }

    llvm::DenseMap<mlir::Value, BlockValue> blockValues;
    BlockValue coordinate{axis.getResult().getType(), BlockValueKind::Index,
                          physical.needsLaneVector ? lane : ""};
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
    if (input.kind != CValueKind::F32Vector || aggregate.spelling.empty())
      return op.emitError("RVV reduction projection is unavailable");
    if (decision.placement == VLAStatePlacement::VectorCarry) {
      if (aggregate.kind != CValueKind::F32BlockStorage)
        return op.emitError("RVV vector reduction carry is unavailable");
      std::string suffix = "f32m" + std::to_string(decision.dataLMUL);
      if (decision.realization == VLAStateRealization::RVVAddReduction)
        line(aggregate.spelling + " = __riscv_vfadd_vv_" + suffix + "_tu(" +
             aggregate.spelling + ", " + aggregate.spelling + ", " +
             input.spelling + ", " + activeVL + ");");
      else if (decision.realization == VLAStateRealization::RVVMaxReduction)
        line(aggregate.spelling + " = __riscv_vfmax_vv_" + suffix + "_tu(" +
             aggregate.spelling + ", " + aggregate.spelling + ", " +
             input.spelling + ", " + activeVL + ");");
      else
        return op.emitError("selected VLA state is not a reduction");
      return mlir::success();
    }
    if (aggregate.kind != CValueKind::Scalar)
      return op.emitError("RVV scalar reduction carry is unavailable");
    std::string seed = fresh("seed");
    std::string partial = fresh("partial");
    std::string inputSuffix = "f32m" + std::to_string(decision.dataLMUL);
    line("vfloat32m1_t " + seed + " = __riscv_vfmv_v_f_f32m1(" +
         aggregate.spelling + ", 1);");
    if (decision.realization == VLAStateRealization::RVVAddReduction)
      line("vfloat32m1_t " + partial +
           " = __riscv_vfredusum_vs_" + inputSuffix + "_f32m1(" +
           input.spelling + ", " + seed + ", " + activeVL + ");");
    else if (decision.realization == VLAStateRealization::RVVMaxReduction)
      line("vfloat32m1_t " + partial +
           " = __riscv_vfredmax_vs_" + inputSuffix + "_f32m1(" +
           input.spelling + ", " + seed + ", " + activeVL + ");");
    else
      return op.emitError("selected VLA state is not a reduction");
    line(aggregate.spelling + " = __riscv_vfmv_f_s_f32m1_f32(" + partial +
         ");");
    values[op.getResult()] = aggregate;
    return mlir::success();
  }

  mlir::LogicalResult emitWideningF16DotStrip(
      ReduceOp op, const VLAStateDecision &decision, CValue &aggregate) {
    if (aggregate.kind != CValueKind::F32BlockStorage ||
        aggregate.spelling.empty())
      return op.emitError("widening dot accumulator is unavailable");
    auto lhsLoad = mlir::dyn_cast_or_null<LoadOp>(decision.lhsLoad);
    auto rhsLoad = mlir::dyn_cast_or_null<LoadOp>(decision.rhsLoad);
    if (!lhsLoad || !rhsLoad)
      return op.emitError("widening dot load facts are unavailable");
    mlir::Value coordinate = activeVLADecision->coordinate;
    std::optional<std::string> lhs = pointerBase(lhsLoad.getPointer(), coordinate);
    std::optional<std::string> rhs = pointerBase(rhsLoad.getPointer(), coordinate);
    if (!lhs || !rhs)
      return op.emitError("widening dot pointer projection is unavailable");
    std::string lhsVector = fresh("dot_lhs");
    std::string rhsVector = fresh("dot_rhs");
    line(rvvFloatType(decision.inputShape) + " " + lhsVector +
         " = __riscv_vle" + std::to_string(decision.inputShape.sew) + "_v_" +
         rvvFloatSuffix(decision.inputShape) + "(" + *lhs + " + " +
         require(coordinate).spelling + ", " + activeVL + ");");
    line(rvvFloatType(decision.inputShape) + " " + rhsVector +
         " = __riscv_vle" + std::to_string(decision.inputShape.sew) + "_v_" +
         rvvFloatSuffix(decision.inputShape) + "(" + *rhs + " + " +
         require(coordinate).spelling + ", " + activeVL + ");");
    std::string intrinsic = "__riscv_vfwmacc_vv_" +
                            rvvFloatSuffix(decision.computeShape);
    if (decision.tail == RVVTailPolicy::Undisturbed && !activeVLAFullStrip) {
      line(aggregate.spelling + " = " + intrinsic + "_tu(" +
           aggregate.spelling + ", " + lhsVector + ", " + rhsVector + ", " +
           activeVL + ");");
    } else {
      line(aggregate.spelling + " = " + intrinsic + "(" + aggregate.spelling +
           ", " + lhsVector + ", " + rhsVector + ", " + activeVL + ");");
    }
    return mlir::success();
  }

  mlir::LogicalResult emitVectorScan(ScanOp op,
                                     const VLAStateDecision &decision,
                                     const CValue &carry) {
    CValue input = require(op.getInput());
    bool segmented =
        decision.realization ==
        VLAStateRealization::RVVSegmentedInclusiveAddScan;
    if ((decision.realization != VLAStateRealization::RVVInclusiveAddScan &&
         !segmented) ||
        input.kind != CValueKind::F32Vector ||
        carry.kind != CValueKind::Scalar || carry.spelling.empty())
      return op.emitError("RVV scan projection is unavailable");

    CValue segmentMask;
    CValue segmentValues;
    if (segmented) {
      segmentMask = require(decision.segmentStart);
      segmentValues = require(decision.segmentVector);
      if (segmentMask.kind != CValueKind::Mask ||
          segmentValues.kind != CValueKind::U8Vector ||
          segmentMask.spelling.empty() || segmentValues.spelling.empty() ||
          decision.segmentSEW != 8 || decision.segmentLMUL == 0)
        return op.emitError("segmented scan start projection is unavailable");
    }

    std::string indices = fresh("scan_indices");
    std::string prefix = fresh("scan_prefix");
    std::string offset = fresh("scan_offset");
    std::string dataSuffix = "f32m" + std::to_string(decision.dataLMUL);
    std::string indexSuffix =
        "u32m" + std::to_string(decision.laneIndexLMUL);
    std::string maskSuffix =
        indexSuffix + "_b" + std::to_string(decision.maskRatio);
    line("vuint32m" + std::to_string(decision.laneIndexLMUL) + "_t " +
         indices + " = __riscv_vid_v_" + indexSuffix + "(" + activeVL +
         ");");
    line("vfloat32m" + std::to_string(decision.dataLMUL) + "_t " + prefix +
         " = " + input.spelling + ";");
    std::string propagatedSegments;
    if (segmented) {
      propagatedSegments = fresh("scan_segments");
      line("vuint8m" + std::to_string(decision.segmentLMUL) + "_t " +
           propagatedSegments + " = " + segmentValues.spelling + ";");
    }
    line("for (size_t " + offset + " = 1; " + offset + " < " + activeVL +
         "; " + offset + " <<= 1) {");
    ++indent;
    std::string shifted = fresh("scan_shifted");
    std::string active = fresh("scan_active");
    line("vfloat32m" + std::to_string(decision.dataLMUL) + "_t " + shifted +
         " = __riscv_vslideup_vx_" + dataSuffix +
         "(__riscv_vundefined_" + dataSuffix + "(), " + prefix + ", " +
         offset + ", " + activeVL + ");");
    line("vbool" + std::to_string(decision.maskRatio) + "_t " + active +
         " = __riscv_vmsgeu_vx_" + maskSuffix + "(" + indices +
         ", (uint32_t)" + offset + ", " + activeVL + ");");
    if (segmented) {
      std::string noSegment = fresh("scan_no_segment");
      std::string segmentSuffix =
          "u8m" + std::to_string(decision.segmentLMUL);
      std::string segmentMaskSuffix =
          segmentSuffix + "_b" + std::to_string(decision.maskRatio);
      line("vbool" + std::to_string(decision.maskRatio) + "_t " + noSegment +
           " = __riscv_vmseq_vx_" + segmentMaskSuffix + "(" +
           propagatedSegments + ", 0, " + activeVL + ");");
      line(active + " = __riscv_vmand_mm_b" +
           std::to_string(decision.maskRatio) + "(" + active + ", " +
           noSegment + ", " + activeVL + ");");
    }
    line(prefix + " = __riscv_vfadd_vv_" + dataSuffix + "_m(" + active +
         ", " + prefix + ", " + shifted + ", " + activeVL + ");");
    if (segmented) {
      std::string shiftedSegments = fresh("scan_shifted_segments");
      std::string segmentSuffix =
          "u8m" + std::to_string(decision.segmentLMUL);
      line("vuint8m" + std::to_string(decision.segmentLMUL) + "_t " +
           shiftedSegments + " = __riscv_vslideup_vx_" + segmentSuffix +
           "(" + propagatedSegments + ", " + propagatedSegments + ", " +
           offset + ", " + activeVL + ");");
      line(propagatedSegments + " = __riscv_vor_vv_" + segmentSuffix + "(" +
           propagatedSegments + ", " + shiftedSegments + ", " + activeVL +
           ");");
    }
    --indent;
    line("}");
    if (segmented) {
      std::string firstSegment = fresh("scan_first_segment");
      std::string continuationLength = fresh("scan_continuation_length");
      std::string continuation = fresh("scan_continuation");
      line("const long " + firstSegment + " = __riscv_vfirst_m_b" +
           std::to_string(decision.maskRatio) + "(" + segmentMask.spelling +
           ", " + activeVL + ");");
      line("const size_t " + continuationLength + " = " + firstSegment +
           " < 0 ? " + activeVL + " : (size_t)" + firstSegment + ";");
      line("vbool" + std::to_string(decision.maskRatio) + "_t " + continuation +
           " = __riscv_vmsltu_vx_" + maskSuffix + "(" + indices +
           ", (uint32_t)" + continuationLength + ", " + activeVL + ");");
      line(prefix + " = __riscv_vfadd_vf_" + dataSuffix + "_m(" +
           continuation + ", " + prefix + ", " + carry.spelling + ", " +
           activeVL + ");");
    } else {
      line(prefix + " = __riscv_vfadd_vf_" + dataSuffix + "(" + prefix +
           ", " + carry.spelling + ", " + activeVL + ");");
    }
    std::string last = fresh("scan_last");
    line("vfloat32m" + std::to_string(decision.dataLMUL) + "_t " + last +
         " = __riscv_vslidedown_vx_" + dataSuffix + "(" + prefix + ", " +
         activeVL + " - 1, " + activeVL + ");");
    line(carry.spelling + " = __riscv_vfmv_f_s_" + dataSuffix + "_f32(" +
         last + ");");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::F32Vector, prefix};
    return mlir::success();
  }

  mlir::LogicalResult emitVectorArgMaxSummary(
      ArgMaxOp op, const VLAStateDecision &decision,
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
    std::string dataSuffix = "f32m" + std::to_string(decision.dataLMUL);
    line("vfloat32m1_t " + seed +
         " = __riscv_vfmv_v_f_f32m1(-INFINITY, 1);");
    line("vfloat32m1_t " + reduced +
         " = __riscv_vfredmax_vs_" + dataSuffix + "_f32m1(" +
         input.spelling + ", " + seed + ", " + activeVL + ");");
    line("const float " + stripMaximum +
         " = __riscv_vfmv_f_s_f32m1_f32(" + reduced + ");");
    line("vbool" + std::to_string(decision.maskRatio) + "_t " + equal +
         " = __riscv_vmfeq_vf_" + dataSuffix + "_b" +
         std::to_string(decision.maskRatio) + "(" + input.spelling + ", " +
         stripMaximum + ", " + activeVL + ");");
    line("const long " + first + " = __riscv_vfirst_m_b" +
         std::to_string(decision.maskRatio) + "(" + equal + ", " + activeVL +
         ");");
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
      OnlineSoftmaxSummaryOp op, const VLAStateDecision &decision,
      const CValue &aggregate) {
    CValue input = require(op.getInput());
    if (decision.realization !=
            VLAStateRealization::RVVOnlineSoftmaxSummary ||
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
    std::string dataSuffix = "f32m" + std::to_string(decision.dataLMUL);
    line("vfloat32m1_t " + maxSeed +
         " = __riscv_vfmv_v_f_f32m1(-INFINITY, 1);");
    line("vfloat32m1_t " + maxVector +
         " = __riscv_vfredmax_vs_" + dataSuffix + "_f32m1(" +
         input.spelling + ", " + maxSeed + ", " + activeVL + ");");
    line("const float " + stripMaximum +
         " = __riscv_vfmv_f_s_f32m1_f32(" + maxVector + ");");
    line("vfloat32m" + std::to_string(decision.dataLMUL) + "_t " + shifted +
         " = __riscv_vfsub_vf_" + dataSuffix + "(" + input.spelling + ", " +
         stripMaximum + ", " + activeVL + ");");
    line("vfloat32m" + std::to_string(decision.dataLMUL) + "_t " +
         exponentials + " = __weft_exp_f32m2(" + shifted + ", " + activeVL +
         ");");
    line("vfloat32m1_t " + sumSeed +
         " = __riscv_vfmv_v_f_f32m1(0.0f, 1);");
    line("vfloat32m1_t " + sumVector +
         " = __riscv_vfredusum_vs_" + dataSuffix + "_f32m1(" +
         exponentials + ", " + sumSeed + ", " + activeVL + ");");
    line("const float " + stripSum +
         " = __riscv_vfmv_f_s_f32m1_f32(" + sumVector + ");");
    line("__weft_online_summary_merge_f32(&" + maximum.spelling + ", &" +
         sum.spelling + ", " + stripMaximum + ", " + stripSum + ");");
    values[op.getResult()] = aggregate;
    return mlir::success();
  }
};

void emitPrelude(llvm::raw_ostream &output, bool usesExp, bool usesIME1,
                 bool usesGroupedI4I8, bool usesE2M1E8M0I8,
                 bool usesQuantCodebookI8) {
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

static inline __attribute__((unused)) uint32_t __weft_bitcast_f32_u32(float source) {
  union { uint32_t u; float f; } value = { .f = source };
  return value.u;
}

static inline __attribute__((unused)) float __weft_bitcast_u32_f32(uint32_t bits) {
  union { uint32_t u; float f; } value = { .u = bits };
  return value.f;
}

)c";
  if (usesE2M1E8M0I8) {
    output << R"c(static const int8_t __attribute__((unused))
__weft_e2m1_doubled[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

static inline __attribute__((always_inline, unused)) float
__weft_e8m0_half(uint8_t exponent) {
  const uint32_t bits = exponent < UINT8_C(2)
                            ? (UINT32_C(0x00200000) << exponent)
                            : ((uint32_t)(exponent - UINT8_C(1)) << 23);
  return __weft_bitcast_u32_f32(bits);
}

)c";
  }
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
  if (usesQuantCodebookI8) {
    auto emitI64Table = [&](llvm::StringRef name, const int64_t *values,
                            size_t count) {
      output << "static const int64_t " << name << "[" << count << "] = {\n";
      for (size_t index = 0; index < count; ++index) {
        if (index % 4 == 0)
          output << "  ";
        output << values[index];
        if (index + 1 != count)
          output << ", ";
        output << (index % 4 == 3 ? "\n" : "");
      }
      if (count % 4 != 0)
        output << "\n";
      output << "};\n\n";
    };
    auto emitI32Table = [&](llvm::StringRef name, const int32_t *values,
                            size_t count) {
      output << "static const int32_t " << name << "[" << count << "] = {\n";
      for (size_t index = 0; index < count; ++index) {
        if (index % 8 == 0)
          output << "  ";
        output << values[index];
        if (index + 1 != count)
          output << ", ";
        output << (index % 8 == 7 ? "\n" : "");
      }
      if (count % 8 != 0)
        output << "\n";
      output << "};\n\n";
    };
    emitI64Table("__weft_iq1_m_grid", __weft_iq1_m_grid, 2048);
    emitI64Table("__weft_iq2_s_grid", __weft_iq2_s_grid, 1024);
    emitI32Table("__weft_iq3_s_grid", __weft_iq3_s_grid, 512);
    output << R"c(static inline __attribute__((always_inline, unused)) int32_t
__weft_i8_dot(const int8_t *lhs, const int8_t *rhs, size_t count) {
  int32_t result = 0;
  size_t offset = 0;
  while (offset < count) {
    const size_t vl = __riscv_vsetvl_e8m2(count - offset);
    const vint8m2_t left = __riscv_vle8_v_i8m2(lhs + offset, vl);
    const vint8m2_t right = __riscv_vle8_v_i8m2(rhs + offset, vl);
    const vint16m4_t products = __riscv_vwmul_vv_i16m4(left, right, vl);
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    result += __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m4_i32m1(products, zero, vl));
    offset += vl;
  }
  return result;
}

static inline __attribute__((always_inline, unused)) vuint16m1_t
__weft_get_u16m2_u16m1(vuint16m2_t value, size_t segment) {
  return segment == 0 ? __riscv_vget_v_u16m2_u16m1(value, 0)
                      : __riscv_vget_v_u16m2_u16m1(value, 1);
}

static inline __attribute__((always_inline, unused)) vint8m2_t
__weft_get_i8m4_i8m2(vint8m4_t value, size_t segment) {
  return segment == 0 ? __riscv_vget_v_i8m4_i8m2(value, 0)
                      : __riscv_vget_v_i8m4_i8m2(value, 1);
}

static inline __attribute__((always_inline, unused)) vint8m2_t
__weft_get_i8m8_i8m2(vint8m8_t value, size_t segment) {
  if (segment == 0)
    return __riscv_vget_v_i8m8_i8m2(value, 0);
  if (segment == 1)
    return __riscv_vget_v_i8m8_i8m2(value, 1);
  if (segment == 2)
    return __riscv_vget_v_i8m8_i8m2(value, 2);
  return __riscv_vget_v_i8m8_i8m2(value, 3);
}

static inline __attribute__((always_inline, unused)) float
__weft_iq2_s_i8_vl128(
    const uint8_t *codes, const uint8_t *high_bits,
    const uint8_t *sign_bits, const uint8_t *scales,
    const uint8_t *activation_bytes, float weight_scale,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  const size_t vl32 = __riscv_vsetvl_e8m2(32);
  const vuint8m2_t lane = __riscv_vid_v_u8m2(vl32);
  const vuint8m2_t sign_source_index = __riscv_vsrl_vx_u8m2(lane, 3, vl32);
  const vuint8m2_t sign_bit = __riscv_vsll_vv_u8m2(
      __riscv_vmv_v_x_u8m2(1, vl32),
      __riscv_vand_vx_u8m2(lane, 7, vl32), vl32);
  int32_t integer_sum = 0;
  for (size_t group = 0; group < 8; ++group) {
    uint16_t byte_offsets[4];
    for (size_t vector = 0; vector < 4; ++vector)
      byte_offsets[vector] = (uint16_t)((codes[group * 4 + vector] |
          (((uint16_t)high_bits[group] << (8 - 2 * vector)) & 0x300)) * 8);
    const vuint16mf2_t offsets =
        __riscv_vle16_v_u16mf2(byte_offsets, 4);
    const vuint64m2_t packed = __riscv_vluxei16_v_u64m2(
        (const uint64_t *)(const void *)__weft_iq2_s_grid, offsets, 4);
    const vint8m2_t grid = __riscv_vreinterpret_v_u8m2_i8m2(
        __riscv_vreinterpret_v_u64m2_u8m2(packed));
    const vuint8mf4_t packed_signs =
        __riscv_vle8_v_u8mf4(sign_bits + group * 4, 4);
    const vuint8m2_t expanded_signs = __riscv_vrgather_vv_u8m2(
        __riscv_vlmul_ext_v_u8mf4_u8m2(packed_signs), sign_source_index,
        vl32);
    const vbool4_t negative = __riscv_vmsne_vx_u8m2_b4(
        __riscv_vand_vv_u8m2(expanded_signs, sign_bit, vl32), 0, vl32);
    const vint8m2_t q8 = __riscv_vle8_v_i8m2(
        activation + group * 32, vl32);
    const vint8m2_t signed_q8 =
        __riscv_vrsub_vx_i8m2_mu(negative, q8, q8, 0, vl32);
    const vint16m4_t products =
        __riscv_vwmul_vv_i16m4(grid, signed_q8, vl32);
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    const int32_t first = __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m2_i32m1(
            __riscv_vget_v_i16m4_i16m2(products, 0), zero, 16));
    const int32_t second = __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m2_i32m1(
            __riscv_vget_v_i16m4_i16m2(products, 1), zero, 16));
    integer_sum += first * (1 + 2 * (scales[group] & 15));
    integer_sum += second * (1 + 2 * (scales[group] >> 4));
  }
  return init + 0.125f * (float)integer_sum * weight_scale * activation_scale;
}

static inline __attribute__((always_inline, unused)) float
__weft_iq3_s_i8_vl128(
    const uint8_t *codes, const uint8_t *high_bits,
    const uint8_t *sign_bits, const uint8_t *scales,
    const uint8_t *activation_bytes, float weight_scale,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  int8_t decoded[32];
  int32_t integer_sum = 0;
  for (size_t group = 0; group < 8; ++group) {
    for (size_t vector = 0; vector < 8; ++vector) {
      const uint16_t index = (uint16_t)codes[group * 8 + vector] |
          (uint16_t)(((uint16_t)high_bits[group] << (8 - vector)) & 0x100);
      const int8_t *grid = (const int8_t *)(const void *)&__weft_iq3_s_grid[index];
      const uint8_t signs = sign_bits[group * 4 + vector / 2];
      const size_t sign_base = (vector & 1) * 4;
      for (size_t lane = 0; lane < 4; ++lane)
        decoded[vector * 4 + lane] =
            (signs & (UINT8_C(1) << (sign_base + lane))) ? -grid[lane]
                                                         : grid[lane];
    }
    const int32_t dot = __weft_i8_dot(decoded, activation + group * 32, 32);
    const uint8_t packed_scale = scales[group / 2];
    const int32_t scale =
        1 + 2 * ((group & 1) ? (packed_scale >> 4) : (packed_scale & 15));
    integer_sum += dot * scale;
  }
  return init + (float)integer_sum * weight_scale * activation_scale;
}

static inline __attribute__((always_inline, unused)) float
__weft_iq1_m_i8_vl128(
    const uint8_t *codes, const uint8_t *high_delta_bits,
    const uint8_t *scales, const uint8_t *activation_bytes,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  const uint16_t *packed_scales = (const uint16_t *)(const void *)scales;
  const uint16_t scale_bits =
      (packed_scales[0] >> 12) | ((packed_scales[1] >> 8) & 0x00f0) |
      ((packed_scales[2] >> 4) & 0x0f00) | (packed_scales[3] & 0xf000);
  const float block_scale = (float)__weft_bitcast_u16_f16(scale_bits);
  vint32m4_t grid_accumulator = __riscv_vmv_v_x_i32m4(0, 16);
  vint32m4_t delta_accumulator = __riscv_vmv_v_x_i32m4(0, 16);
  const uint16_t index_shift_values[16] = {
      8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4};
  const vuint16m2_t index_shifts =
      __riscv_vle16_v_u16m2(index_shift_values, 16);
  const uint16_t delta_mask_values[16] = {
      0x08, 0x80, 0x08, 0x80, 0x08, 0x80, 0x08, 0x80,
      0x08, 0x80, 0x08, 0x80, 0x08, 0x80, 0x08, 0x80};
  const vuint16m2_t delta_masks =
      __riscv_vle16_v_u16m2(delta_mask_values, 16);
  for (size_t half = 0; half < 2; ++half) {
    const vuint8mf2_t high8 =
        __riscv_vle8_v_u8mf2(high_delta_bits + half * 8, 8);
    const vuint16m1_t high_low = __riscv_vzext_vf2_u16m1(high8, 8);
    const vuint16m1_t high_high =
        __riscv_vsll_vx_u16m1(high_low, 8, 8);
    const vuint16m2_t high = __riscv_vzext_vf2_u16m2(
        __riscv_vreinterpret_v_u16m1_u8m1(
            __riscv_vor_vv_u16m1(high_low, high_high, 8)), 16);
    const vuint16m2_t low = __riscv_vzext_vf2_u16m2(
        __riscv_vle8_v_u8m1(codes + half * 16, 16), 16);
    vuint16m2_t indices = __riscv_vor_vv_u16m2(
        low,
        __riscv_vand_vx_u16m2(
            __riscv_vsll_vv_u16m2(high, index_shifts, 16), 0x700, 16),
        16);
    indices = __riscv_vsll_vx_u16m2(indices, 3, 16);
    const vbool8_t negative = __riscv_vmsgtu_vx_u16m2_b8(
        __riscv_vand_vv_u16m2(high, delta_masks, 16), 0, 16);
    const vint64m8_t positive =
        __riscv_vmv_v_x_i64m8(INT64_C(0x0101010101010101), 16);
    const vint8m8_t delta = __riscv_vreinterpret_v_i64m8_i8m8(
        __riscv_vmerge_vxm_i64m8(positive, INT64_C(-1), negative, 16));
    for (size_t quarter = 0; quarter < 2; ++quarter) {
      const vint8m4_t grid = __riscv_vreinterpret_v_i64m4_i8m4(
          __riscv_vreinterpret_v_u64m4_i64m4(
              __riscv_vluxei16_v_u64m4(
                  (const uint64_t *)(const void *)__weft_iq1_m_grid,
                  __weft_get_u16m2_u16m1(indices, quarter), 8)));
      for (size_t pair = 0; pair < 2; ++pair) {
        const size_t segment = quarter * 2 + pair;
        const vint8m2_t q8 = __riscv_vle8_v_i8m2(
            activation + half * 128 + segment * 32, 32);
        const vint16m4_t grid_product = __riscv_vwmul_vv_i16m4(
            __weft_get_i8m4_i8m2(grid, pair), q8, 32);
        const vint16m4_t delta_product = __riscv_vwmul_vv_i16m4(
            __weft_get_i8m8_i8m2(delta, segment), q8, 32);
        const uint16_t scale_word = packed_scales[half * 2 + segment / 2];
        const unsigned shift = 6 * (segment % 2);
        const int16_t first_scale = 1 + 2 * ((scale_word >> shift) & 7);
        const int16_t second_scale =
            1 + 2 * ((scale_word >> (shift + 3)) & 7);
        grid_accumulator = __riscv_vwmacc_vx_i32m4(
            grid_accumulator, first_scale,
            __riscv_vget_v_i16m4_i16m2(grid_product, 0), 16);
        grid_accumulator = __riscv_vwmacc_vx_i32m4(
            grid_accumulator, second_scale,
            __riscv_vget_v_i16m4_i16m2(grid_product, 1), 16);
        delta_accumulator = __riscv_vwmacc_vx_i32m4(
            delta_accumulator, first_scale,
            __riscv_vget_v_i16m4_i16m2(delta_product, 0), 16);
        delta_accumulator = __riscv_vwmacc_vx_i32m4(
            delta_accumulator, second_scale,
            __riscv_vget_v_i16m4_i16m2(delta_product, 1), 16);
      }
    }
  }
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t grid_sum = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vredsum_vs_i32m4_i32m1(grid_accumulator, zero, 16));
  const int32_t delta_sum = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vredsum_vs_i32m4_i32m1(delta_accumulator, zero, 16));
  return init + block_scale * activation_scale *
                    ((float)grid_sum + 0.125f * (float)delta_sum);
}

static inline __attribute__((always_inline, unused)) float
__weft_q6_k_i8_vl128(
    const uint8_t *low_bits, const uint8_t *high_bits,
    const uint8_t *group_scale_bytes, const uint8_t *activation_bytes,
    float weight_scale, float activation_scale, float init) {
  const int8_t *group_scales =
      (const int8_t *)(const void *)group_scale_bytes;
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  const uint8_t *low = low_bits;
  const uint8_t *high = high_bits;
  const int8_t *scale = group_scales;
  const int8_t *q8 = activation;
  int low_high;
  float temporary;
  float result = init;
  const float combined_scale = weight_scale * activation_scale;
  for (size_t half = 0; half < 2; ++half) {
    __asm__ volatile(
        "addi %[low_high], %[low], 32\n\t"
        "ld t0, 0(%[scale])\n\t"
        "addi %[scale], %[scale], 8\n\t"
        "slli t6, t0, 1 * 8\n\t"
        "lb zero, 0(%[low])\n\t"
        "slli t5, t0, 2 * 8\n\t"
        "slli t4, t0, 3 * 8\n\t"
        "lb zero, 0(%[low_high])\n\t"
        "slli t3, t0, 4 * 8\n\t"
        "slli t2, t0, 5 * 8\n\t"
        "lb zero, 0(%[high])\n\t"
        "lb zero, 31(%[low_high])\n\t"
        "slli t1, t0, 6 * 8\n\t"
        "srai a7, t0, 56\n\t"
        "vsetvli zero, %[vl32], e8, m2\n\t"
        "vle8.v v8, (%[low])\n\t"
        "srai t6, t6, 56\n\t"
        "srai t5, t5, 56\n\t"
        "srai t4, t4, 56\n\t"
        "srai t3, t3, 56\n\t"
        "vle8.v v10, (%[low_high])\n\t"
        "addi %[low], %[low], 64\n\t"
        "slli t0, t0, 7 * 8\n\t"
        "srai t2, t2, 56\n\t"
        "srai t1, t1, 56\n\t"
        "srai t0, t0, 56\n\t"
        "vle8.v v4, (%[high])\n\t"
        "vsrl.vi v12, v8, 4\n\t"
        "vsrl.vi v14, v10, 4\n\t"
        "lb zero, 0(%[q8])\n\t"
        "vand.vi v8, v8, 0xF\n\t"
        "vand.vi v10, v10, 0xF\n\t"
        "lb zero, 32(%[q8])\n\t"
        "vsll.vi v0, v4, 4\n\t"
        "vsll.vi v2, v4, 2\n\t"
        "lb zero, 64(%[q8])\n\t"
        "vsrl.vi v6, v4, 2\n\t"
        "lb zero, 96(%[q8])\n\t"
        "vand.vx v0, v0, %[mask]\n\t"
        "vand.vx v2, v2, %[mask]\n\t"
        "vand.vx v4, v4, %[mask]\n\t"
        "vand.vx v6, v6, %[mask]\n\t"
        "vor.vv v8, v8, v0\n\t"
        "vor.vv v10, v10, v2\n\t"
        "vor.vv v12, v12, v4\n\t"
        "vor.vv v14, v14, v6\n\t"
        "lb zero, 127(%[q8])\n\t"
        "vsetvli zero, %[vl128], e8, m8\n\t"
        "vle8.v v0, (%[q8])\n\t"
        "vsub.vx v8, v8, %[vl32]\n\t"
        "vsetvli zero, %[vl64], e8, m4\n\t"
        "vwmul.vv v16, v0, v8\n\t"
        "vwmul.vv v24, v4, v12\n\t"
        "vsetivli zero, 16, e16, m2\n\t"
        "vmv.v.x v0, zero\n\t"
        "vwredsum.vs v10, v16, v0\n\t"
        "vwredsum.vs v9, v18, v0\n\t"
        "vwredsum.vs v8, v20, v0\n\t"
        "vwredsum.vs v7, v22, v0\n\t"
        "vwredsum.vs v11, v24, v0\n\t"
        "vwredsum.vs v12, v26, v0\n\t"
        "vwredsum.vs v13, v28, v0\n\t"
        "vwredsum.vs v14, v30, v0\n\t"
        "vsetivli zero, 4, e32, m1\n\t"
        "vmul.vx v0, v10, t0\n\t"
        "vmul.vx v1, v9, t1\n\t"
        "vmacc.vx v0, t2, v8\n\t"
        "vmacc.vx v1, t3, v7\n\t"
        "vmacc.vx v0, t4, v11\n\t"
        "vmacc.vx v1, t5, v12\n\t"
        "vmacc.vx v0, t6, v13\n\t"
        "vmacc.vx v1, a7, v14\n\t"
        "vadd.vv v0, v0, v1\n\t"
        "vfcvt.f.x.v v0, v0\n\t"
        "vfmv.f.s %[temporary], v0\n\t"
        "fmadd.s %[result], %[combined_scale], %[temporary], %[result]"
        : [low] "+&r"(low), [low_high] "=&r"(low_high),
          [scale] "+&r"(scale), [result] "+&f"(result),
          [temporary] "=&f"(temporary)
        : [high] "r"(high), [q8] "r"(q8), [vl32] "r"(32),
          [vl64] "r"(64), [vl128] "r"(128), [mask] "r"(0x30),
          [combined_scale] "f"(combined_scale)
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
          "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
          "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
          "v31", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a7");
    high += 32;
    q8 += 128;
  }
  return result;
}

)c";
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

static inline __attribute__((always_inline, unused)) void
__weft_online_summary_merge_f32(
    float *maximum, float *sum, float strip_maximum, float strip_sum) {
  if (*maximum == -INFINITY) {
    *maximum = strip_maximum;
    *sum = strip_sum;
    return;
  }
  if (strip_maximum <= *maximum) {
    *sum += strip_sum * expf(strip_maximum - *maximum);
    return;
  }
  *sum = *sum * expf(*maximum - strip_maximum) + strip_sum;
  *maximum = strip_maximum;
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
  bool usesE2M1E8M0I8 = false;
  bool usesQuantCodebookI8 = false;
  module.walk([&](UnaryOp op) { usesExp |= op.getKind() == "exp"; });
  module.walk([&](AffineI4I8ContractOp) { usesIME1 = true; });
  module.walk([&](SymmetricI4I8ContractOp) { usesIME1 = true; });
  module.walk([&](GroupedAffineI4I8DotOp) { usesGroupedI4I8 = true; });
  module.walk([&](E2M1E8M0I8DotOp) { usesE2M1E8M0I8 = true; });
  module.walk([&](IQ2SI8DotOp) { usesQuantCodebookI8 = true; });
  module.walk([&](IQ3SI8DotOp) { usesQuantCodebookI8 = true; });
  module.walk([&](IQ1MI8DotOp) { usesQuantCodebookI8 = true; });
  module.walk([&](Q6KI8DotOp) { usesQuantCodebookI8 = true; });
  emitPrelude(output, usesExp, usesIME1, usesGroupedI4I8,
              usesE2M1E8M0I8, usesQuantCodebookI8);

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
