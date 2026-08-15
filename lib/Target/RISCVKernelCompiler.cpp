#include "Weft/Target/RISCVLowering.h"

#include "RISCVIntrinsicC.h"
#include "RISCVKernelCompiler.h"
#include "RISCVKernelFacts.h"
#include "RISCVPhysicalPlanning.h"

#include "Weft/Dialect/Extension/IR/ExtensionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using namespace weft;
using namespace weft::extension;
using namespace weft::kernel;
using namespace weft::riscv_internal;

enum class CValueKind {
  Scalar,
  Pointer,
  Coordinate,
  F16Vector,
  F32Vector,
  F32VectorBundle,
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
  RVVVectorShape vectorShape;
};

enum class BlockDecodeRealization {
  RVVI8TableGather,
};

struct BlockDecodeDecision {
  BlockDecodeRealization realization =
      BlockDecodeRealization::RVVI8TableGather;
  int64_t tableExtent = 16;
};

enum class LoadF16LERealization {
  ScalarBytes,
};

struct LoadF16LEDecision {
  LoadF16LERealization realization = LoadF16LERealization::ScalarBytes;
};

enum class BlockValueRealization {
  Direct,
  DeferredPackedTransform,
  DeferredI32Widen,
  WideningI8Product,
};

struct BlockOperationDecision {
  mlir::Operation *operation = nullptr;
  BlockValueRealization realization = BlockValueRealization::Direct;
  mlir::Value source;
  std::string transform;
  std::optional<uint64_t> unsignedMaximum;
  unsigned maskRatio = 0;
};

enum class BlockStoreRealization {
  Unsupported,
  RVVE8MF4MicroStrips,
  RVVE8M1FixedStrips,
  RVVE8M1DynamicStrips,
};

struct BlockStorePhysicalDecision {
  BlockStoreRealization realization = BlockStoreRealization::Unsupported;
  RVVVectorShape byteShape;
  unsigned stripVL = 0;
  bool needsLaneVector = false;
  RVVVectorShape laneShape;
};

struct SelectedBlockStorePhysical {
  BlockStorePhysicalDecision decision;
  PhysicalResourceBudget resources;
};

enum class BlockReduceRealization {
  Unsupported,
  RVVE8M1FixedStrips,
  RVVE8M1DynamicStrips,
};

struct BlockReducePhysicalDecision {
  BlockReduceRealization realization = BlockReduceRealization::Unsupported;
  unsigned stripVL = 0;
  bool needsLaneVector = false;
  RVVVectorShape laneShape;
};

struct SelectedBlockReducePhysical {
  BlockReducePhysicalDecision decision;
  PhysicalResourceBudget resources;
};

struct BlockStoreGroupDecision {
  mlir::Operation *operation = nullptr;
  mlir::Operation *axis = nullptr;
  int64_t extent = 0;
  llvm::SmallVector<mlir::Operation *> scalarPreamble;
  llvm::SmallVector<mlir::Operation *> stores;
  llvm::SmallVector<mlir::Operation *> closure;
  llvm::SmallVector<BlockOperationDecision> operations;
};

struct BlockReductionValueDecision {
  mlir::Operation *operation = nullptr;
  RVVVectorShape inputShape;
  RVVVectorShape combinedShape;
  RVVVectorShape seedShape;
  bool wideningCombine = false;
};

struct BlockReduceGroupDecision {
  mlir::Operation *operation = nullptr;
  mlir::Operation *axis = nullptr;
  int64_t extent = 0;
  llvm::SmallVector<mlir::Operation *> reductions;
  llvm::SmallVector<mlir::Operation *> closure;
  llvm::SmallVector<BlockOperationDecision> operations;
  llvm::SmallVector<BlockReductionValueDecision> values;
};

enum class MaterializedBlockStoreRealization {
  ScalarRankOne,
  ScalarRankTwo,
  RVVContiguousRankOne,
};

struct MaterializedBlockStoreDecision {
  mlir::Operation *operation = nullptr;
  MaterializedBlockStoreRealization realization =
      MaterializedBlockStoreRealization::ScalarRankTwo;
  mlir::Value rowAxis;
  mlir::Value columnAxis;
  int64_t rows = 0;
  int64_t columns = 0;
};

enum class MaterializedF32PointwiseRealization {
  Binary,
  Unary,
};

struct MaterializedF32PointwiseDecision {
  mlir::Operation *operation = nullptr;
  MaterializedF32PointwiseRealization realization =
      MaterializedF32PointwiseRealization::Binary;
  int64_t elements = 0;
};

struct LocalBlockMemoryFact {
  mlir::Value semanticValue;
  mlir::Value pointer;
  mlir::Value storageAxis;
  BlockType semanticType;
  BlockType storageType;
  int64_t semanticExtent = 0;
  int64_t storageExtent = 0;
};

struct SymmetricI4I8Decision {
  I4I8FragmentRealization realization = I4I8FragmentRealization::RVVN16K32;
  IntrinsicCLeaf leaf = IntrinsicCLeaf::None;
  LocalBlockMemoryFact activationBlock;
  mlir::Value packedBlockBase;
  mlir::Value activation;
  mlir::Value activationScale;
  mlir::Value init;
  int64_t packedBlockBytes = 288;
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

struct E2M1E8M0I8Decision {
  E2M1E8M0I8Realization realization =
      E2M1E8M0I8Realization::RVVVLEN128TableDot;
  IntrinsicCLeaf leaf = IntrinsicCLeaf::None;
  RVVVectorShape packedShape;
  RVVVectorShape activationShape;
  PhysicalResourceBudget resources;
  LocalBlockMemoryFact packedCodes;
  LocalBlockMemoryFact activation;
  mlir::Value exponent;
  mlir::Value activationScale;
  mlir::Value init;
};

struct GroupedAffineI4I8Decision {
  GroupedAffineI4I8Realization realization =
      GroupedAffineI4I8Realization::RVVVLEN128GroupedDot;
  IntrinsicCLeaf leaf = IntrinsicCLeaf::None;
  RVVVectorShape packedShape;
  RVVVectorShape scaleShape;
  RVVVectorShape activationShape;
  RVVVectorShape activationSumShape;
  PhysicalResourceBudget resources;
  LocalBlockMemoryFact packedWeight;
  LocalBlockMemoryFact scaleMin;
  LocalBlockMemoryFact activation;
  LocalBlockMemoryFact activationSum;
  mlir::Value dotScale;
  mlir::Value minimumScale;
  mlir::Value init;
};

struct QuantCodebookI8Decision {
  QuantI8DotSemantic semantic = QuantI8DotSemantic::IQ2S;
  QuantI8DotRealization realization =
      QuantI8DotRealization::RVVFixedLaneLocalBlockDot;
  IntrinsicCLeaf leaf = IntrinsicCLeaf::None;
  llvm::SmallVector<LocalBlockMemoryFact> blocks;
  llvm::SmallVector<mlir::Value> scalars;
  unsigned semanticLanes = 0;
  RVVVectorShape byteShape;
  unsigned reductionSegments = 0;
  PhysicalResourceBudget resources;
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
  std::string logicalValidity;
};

enum class VLAMemoryMode {
  UnitStride,
  Strided,
  Indexed,
  Segment2,
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

enum class VLAInactiveLaneRealization {
  ExplicitPassthrough,
  ZeroCarrierWithLogicalValidity,
};

enum class VLAStateValidityRealization {
  AllActive,
  ReductionIdentity,
  NegativeInfinity,
  OnlineSoftmax,
};

enum class VLAStateRealization {
  RVVAddReduction,
  RVVMaxReduction,
  RVVI8AddReductionI32,
  RVVInclusiveAddScan,
  RVVSegmentedInclusiveAddScan,
  RVVArgMaxSummary,
  RVVOnlineSoftmaxSummary,
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
};

struct VLAAccessDecision {
  mlir::Operation *operation = nullptr;
  mlir::Type elementType;
  VLAMemoryMode memoryMode = VLAMemoryMode::UnitStride;
  VLAActivityMode activityMode = VLAActivityMode::AllActive;
  VLAStoreValueMode storeValueMode = VLAStoreValueMode::Vector;
  mlir::Value predicate;
  mlir::Value indexedOffset;
  unsigned indexedSEW = 0;
  mlir::Value bundleAxis;
  unsigned bundleVectors = 0;
  VLAInactiveLaneRealization inactiveLane =
      VLAInactiveLaneRealization::ExplicitPassthrough;
};

enum class VLASegment2AccessKind {
  Load,
  Store,
};

struct VLASegment2Decision {
  VLASegment2AccessKind kind = VLASegment2AccessKind::Load;
  mlir::Operation *field0 = nullptr;
  mlir::Operation *field1 = nullptr;
  mlir::Operation *emission = nullptr;
  mlir::Value base;
};

enum class VLALookupRealization {
  RVVF32Table16GatherE16,
};

struct VLALookupDecision {
  mlir::Operation *operation = nullptr;
  VLALookupRealization realization =
      VLALookupRealization::RVVF32Table16GatherE16;
  LocalBlockMemoryFact table;
  mlir::Value indices;
  int64_t tableExtent = 16;
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
};

enum class VLACastRealization {
  RVVIdentity,
  RVVWidenF16ToF32,
  RVVNarrowF32ToF16,
  RVVZeroExtendU32ToIndex,
  RVVIndexToF32,
};

struct VLACastDecision {
  mlir::Operation *operation = nullptr;
  VLACastRealization realization = VLACastRealization::RVVWidenF16ToF32;
};

enum class VLAIndexBinaryRealization {
  RVVUnsignedVectorScalar,
  RVVUnsignedVectorVector,
};

struct VLAIndexBinaryDecision {
  mlir::Operation *operation = nullptr;
  VLAIndexBinaryRealization realization =
      VLAIndexBinaryRealization::RVVUnsignedVectorScalar;
  bool scalarOnLHS = false;
};

struct VLAIndexSelectDecision {
  mlir::Operation *operation = nullptr;
  bool trueScalar = false;
  bool falseScalar = false;
};

enum class VLAUnaryRealization {
  RVVF32M2ExpPolynomial,
  RVVF32M2TanhViaExp,
  RVVF32M2ScalarLibmSin,
  RVVF32M2ScalarLibmCos,
};

struct VLAUnaryDecision {
  mlir::Operation *operation = nullptr;
  VLAUnaryRealization realization =
      VLAUnaryRealization::RVVF32M2ExpPolynomial;
};

struct VLAStateDecision {
  mlir::Operation *operation = nullptr;
  VLAStateRealization realization = VLAStateRealization::RVVAddReduction;
  mlir::Type elementType;
  mlir::Value identity;
  VLAMemoryMode coordinateMode = VLAMemoryMode::UnitStride;
  VLAStatePlacement placement = VLAStatePlacement::ScalarCarry;
  mlir::Value segmentStart;
  mlir::Value segmentVector;
  VLAStateValidityRealization validity =
      VLAStateValidityRealization::AllActive;
};

struct VLANarrowDecision {
  mlir::Operation *operation = nullptr;
};

enum class VLADotRealization {
  RVVF32FreeAxisMicrotile,
  RVVF32FreeAxisVectorDot,
};

enum class VLADotInitRealization {
  ZeroVector,
  ScalarBroadcast,
  MaterializedRegion,
};

struct VLADotDecision {
  mlir::Operation *operation = nullptr;
  VLADotRealization realization =
      VLADotRealization::RVVF32FreeAxisMicrotile;
  mlir::Operation *lhsLoad = nullptr;
  mlir::Operation *rhsLoad = nullptr;
  mlir::Operation *freeLoad = nullptr;
  mlir::Value blockedOperand;
  mlir::Value init;
  mlir::Value rowAxis;
  mlir::Value reductionAxis;
  mlir::Value reductionExtent;
  unsigned rowTile = 1;
  VLAMemoryMode rhsMemoryMode = VLAMemoryMode::UnitStride;
  VLAMemoryMode freeMemoryMode = VLAMemoryMode::UnitStride;
  VLAMemoryMode outputMemoryMode = VLAMemoryMode::UnitStride;
  bool lhsPredicateVariesByReduction = false;
  F32DotPhysicalConfig physical;
  VLADotInitRealization initRealization =
      VLADotInitRealization::ZeroVector;
  llvm::SmallVector<mlir::Operation *> absorbed;
};

enum class PhysicalHandoff {
  Share,
  Convert,
  Rematerialize,
  Reload,
  Shuffle,
  LocalPack,
};

struct PhysicalValueDecision {
  mlir::Value value;
  RVVVectorShape shape;
};

struct PhysicalHandoffDecision {
  mlir::Operation *consumer = nullptr;
  mlir::Value value;
  PhysicalHandoff kind = PhysicalHandoff::Share;
  RVVVectorShape sourceShape;
  RVVVectorShape resultShape;
};

struct PhysicalTemporaryDecision {
  mlir::Operation *owner = nullptr;
  RVVVectorShape shape;
};

struct PhysicalStorageDecision {
  mlir::Value value;
  mlir::Value reusedStorage;
  int64_t elements = 0;
  int64_t alignment = 0;
};

struct PhysicalEntityPlan {
  llvm::SmallVector<PhysicalValueDecision> values;
  llvm::SmallVector<PhysicalHandoffDecision> handoffs;
  llvm::SmallVector<PhysicalTemporaryDecision> temporaries;
  llvm::SmallVector<PhysicalStorageDecision> storages;
  PhysicalResourceBudget resources;
  RVVVectorShape vlaDataShape;
  RVVVectorShape vlaIndexShape;
  unsigned vlaMaskRatio = 0;
  std::optional<BlockStorePhysicalDecision> blockStore;
  std::optional<BlockReducePhysicalDecision> blockReduce;
  bool hasValueShapeConflict = false;
};

struct VLACandidateFacts {
  unsigned f32Loads = 0;
  unsigned f32Stores = 0;
  unsigned stridedAccesses = 0;
  unsigned indexedAccesses = 0;
  unsigned maxIndexedElementSEW = 0;
  unsigned maxIndexedOffsetSEW = 0;
  unsigned maxEntityF32Vectors = 0;
  unsigned segmentLoadPairs = 0;
  unsigned segmentStorePairs = 0;
  bool hasF32TableLookup = false;
  bool hasReductionState = false;
  bool hasOrderedScan = false;
  bool hasCoordinateSummary = false;
  bool hasOnlineSummary = false;
  bool requiresF32M2Math = false;
  bool hasF32Division = false;
  bool hasIndexVector = false;
};

struct VLAValueLifetimeSnapshot {
  unsigned f32 = 0;
  unsigned f16 = 0;
  unsigned index = 0;
  unsigned byte = 0;
  unsigned u32 = 0;
  unsigned mask = 0;
};

struct VLAPlanCandidate {
  RVVVectorShape dataShape;
  RVVVectorShape indexShape;
  unsigned maskRatio = 0;
  PhysicalResourceBudget resources;
};

struct VLANarrowPhysicalConfig {
  RVVVectorShape sourceShape;
  RVVVectorShape intermediateShape;
  RVVVectorShape resultShape;
};

void recordPhysicalValue(PhysicalEntityPlan &plan, mlir::Value value,
                         RVVVectorShape shape) {
  if (!value || !shape)
    return;
  auto found = llvm::find_if(plan.values, [&](const PhysicalValueDecision &item) {
    return item.value == value;
  });
  if (found == plan.values.end())
    plan.values.push_back(PhysicalValueDecision{value, shape});
  else if (found->shape != shape)
    plan.hasValueShapeConflict = true;
}

void recordPhysicalHandoff(PhysicalEntityPlan &plan,
                           mlir::Operation *consumer, mlir::Value value,
                           PhysicalHandoff kind, RVVVectorShape source,
                           RVVVectorShape result) {
  plan.handoffs.push_back(
      PhysicalHandoffDecision{consumer, value, kind, source, result});
}

void recordPhysicalTemporary(PhysicalEntityPlan &plan, mlir::Operation *owner,
                             RVVVectorShape shape) {
  if (owner && shape)
    plan.temporaries.push_back(PhysicalTemporaryDecision{owner, shape});
}

struct VLARegionDecision {
  mlir::Operation *operation = nullptr;
  mlir::Value coordinate;
  std::vector<VLAPredicateDecision> predicates;
  std::vector<VLAAccessDecision> accesses;
  std::vector<VLASegment2Decision> segment2;
  std::vector<VLALookupDecision> lookups;
  std::vector<VLABinaryDecision> binaries;
  std::vector<VLAIndexBinaryDecision> indexBinaries;
  std::vector<VLAIndexSelectDecision> indexSelects;
  std::vector<VLAUnaryDecision> unaries;
  std::vector<VLACastDecision> casts;
  std::vector<VLAStateDecision> states;
  std::vector<VLANarrowDecision> narrows;
  std::vector<VLADotDecision> dots;
};

struct AffineI4I8Decision {
  mlir::Operation *operation = nullptr;
  I4I8FragmentRealization realization = I4I8FragmentRealization::RVVN16K32;
  IntrinsicCLeaf leaf = IntrinsicCLeaf::None;
  LocalBlockMemoryFact activationBlock;
  mlir::Value packedBlockBase;
  mlir::Value activation;
  mlir::Value activationScale;
  mlir::Value init;
  unsigned packedBlockBytes = 304;
};

struct F16GemmNTileDecision {
  mlir::Operation *operation = nullptr;
  mlir::Operation *lhsLoad = nullptr;
  mlir::Operation *rhsLoad = nullptr;
  mlir::Value lhsRowAxis;
  mlir::Value lhsReductionAxis;
  mlir::Value rhsReductionAxis;
  mlir::Value rhsColumnAxis;
  unsigned rowTile = 4;
  unsigned columnTile = 8;
  unsigned reductionTile = 64;
  F16MatmulPhysicalConfig physical;
};

enum class DotRealization {
  RVVF32RowMicrotile,
};

struct DotDecision {
  mlir::Operation *operation = nullptr;
  DotRealization realization = DotRealization::RVVF32RowMicrotile;
  mlir::Operation *lhsLoad = nullptr;
  mlir::Operation *rhsLoad = nullptr;
  mlir::Value rowAxis;
  mlir::Value reductionAxis;
  mlir::Value reductionExtent;
  unsigned rowTile = 1;
  F32DotPhysicalConfig physical;
};

template <typename Decision>
struct PlannedPhysicalDecision {
  Decision realization;
  PhysicalEntityPlan entity;

  PlannedPhysicalDecision() = default;
  PlannedPhysicalDecision(Decision value) : realization(std::move(value)) {}
};

struct RISCVPhysicalPlan {
  llvm::DenseMap<mlir::Operation *, PlannedPhysicalDecision<VLARegionDecision>>
      vlaRegions;
  llvm::DenseMap<mlir::Operation *, PlannedPhysicalDecision<AffineI4I8Decision>>
      affineI4I8;
  llvm::DenseMap<mlir::Operation *, PlannedPhysicalDecision<F16GemmNTileDecision>>
      f16Matmuls;
  llvm::DenseMap<mlir::Operation *, PlannedPhysicalDecision<DotDecision>> dots;
  llvm::DenseMap<mlir::Operation *, PlannedPhysicalDecision<SymmetricI4I8Decision>>
      symmetricI4I8;
  llvm::DenseMap<mlir::Operation *, PlannedPhysicalDecision<SignBitI8Decision>>
      signBitI8;
  llvm::DenseMap<mlir::Operation *, PlannedPhysicalDecision<E2M1E8M0I8Decision>>
      e2m1E8M0I8;
  llvm::DenseMap<mlir::Operation *,
                 PlannedPhysicalDecision<GroupedAffineI4I8Decision>>
      groupedAffineI4I8;
  llvm::DenseMap<mlir::Operation *,
                 PlannedPhysicalDecision<QuantCodebookI8Decision>>
      quantCodebookI8;
  llvm::DenseMap<mlir::Operation *, PlannedPhysicalDecision<BlockDecodeDecision>>
      blockDecodes;
  llvm::DenseMap<mlir::Operation *, PlannedPhysicalDecision<LoadF16LEDecision>>
      f16LELoads;
  llvm::DenseMap<mlir::Operation *, PlannedPhysicalDecision<SortIndicesDecision>>
      sortIndices;
  llvm::DenseMap<mlir::Operation *,
                 PlannedPhysicalDecision<BlockStoreGroupDecision>>
      blockStoreGroups;
  llvm::DenseMap<mlir::Operation *,
                 PlannedPhysicalDecision<BlockReduceGroupDecision>>
      blockReduceGroups;
  llvm::DenseMap<mlir::Operation *,
                 PlannedPhysicalDecision<MaterializedBlockStoreDecision>>
      materializedBlockStores;
  llvm::DenseMap<mlir::Operation *,
                 PlannedPhysicalDecision<MaterializedF32PointwiseDecision>>
      materializedF32Pointwise;
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
  return logicalElementType(type);
}

bool containsBlockType(mlir::Type type) {
  type = unwrapLogicalValidity(type);
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
    if (integer.getWidth() != 8 && integer.getWidth() != 16 &&
        integer.getWidth() != 32 && integer.getWidth() != 64)
      return {};
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
  return logicalShapeKind(type) == LogicalShapeKind::Region;
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

class KernelCompiler {
public:
  KernelCompiler(KernelOp kernel, const RISCVLoweringOptions &options,
                llvm::raw_ostream &output,
                SelectedIntrinsicCLeaves &selectedLeaves)
      : kernel(kernel), options(options), output(output),
        selectedLeaves(selectedLeaves) {}

  mlir::LogicalResult compile() {
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
        if (pointer.getNoAlias() || pointer.getRestrictLike())
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
    collectSelectedIntrinsicCLeaves();
    line(returnTypeSpelling + " " + sanitize(kernel.getSymName()) + "(" +
         llvm::join(parameters, ", ") + ") {");
    ++indent;
    for (mlir::Operation &operation : body) {
      llvm::DenseSet<mlir::Operation *> visiting;
      if (isStorageShapeOnly(&operation, visiting))
        continue;
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
  bool isStorageShapeOnly(
      mlir::Operation *operation,
      llvm::DenseSet<mlir::Operation *> &visiting) const {
    if (!mlir::isa<ConstantOp, MetaValueOp, BinaryOp, CastOp>(operation) ||
        operation->getNumResults() == 0 || !visiting.insert(operation).second)
      return false;
    bool hasUse = false;
    for (mlir::Value result : operation->getResults()) {
      for (mlir::OpOperand &use : result.getUses()) {
        hasUse = true;
        mlir::Operation *owner = use.getOwner();
        if (mlir::isa<StorageOp>(owner))
          continue;
        if (!isStorageShapeOnly(owner, visiting)) {
          visiting.erase(operation);
          return false;
        }
      }
    }
    visiting.erase(operation);
    return hasUse;
  }

  mlir::LogicalResult preparePhysicalDecisions() {
    bool decisionFailure = false;
    if (mlir::failed(weft::riscv_internal::analyzeKernelPhysicalFacts(
            kernel, kernelFacts)))
      return mlir::failure();
    if (mlir::failed(prepareMaterializedBlockValues()))
      return mlir::failure();
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
      PlannedPhysicalDecision<SortIndicesDecision> planned;
      planned.realization = std::move(decision);
      planned.entity.resources.architecturalGroups =
          options.target.vectorRegisters;
      int64_t bucketCount = int64_t{1} << planned.realization.radixBits;
      int64_t privateBytes =
          2 * bucketCount * static_cast<int64_t>(sizeof(size_t));
      if (privateBytes > options.target.maxPrivateStackBytes) {
        op.emitError(
            "sort_indices histogram exceeds target private-storage budget");
        decisionFailure = true;
        return;
      }
      planned.entity.storages.push_back(
          PhysicalStorageDecision{{}, {}, 2 * bucketCount,
                                  static_cast<int64_t>(alignof(size_t))});
      if (!physicalPlan.sortIndices
               .try_emplace(op.getOperation(), std::move(planned))
               .second) {
        op.emitError(
            "one sort_indices primitive cannot own multiple physical decisions");
        decisionFailure = true;
      }
    });
    kernel.walk([&](LoadF16LEOp op) {
      if (decisionFailure)
        return;
      if (!options.target.littleEndian) {
        op.emitError("load_f16_le requires a little-endian target");
        decisionFailure = true;
        return;
      }
      PlannedPhysicalDecision<LoadF16LEDecision> planned;
      initializeEntityPlan(planned.entity);
      planned.realization.realization = LoadF16LERealization::ScalarBytes;
      if (!physicalPlan.f16LELoads
               .try_emplace(op.getOperation(), std::move(planned))
               .second) {
        op.emitError("one load_f16_le cannot own multiple physical decisions");
        decisionFailure = true;
      }
    });
    kernel.walk([&](VLAOp vla) {
      if (decisionFailure)
        return;
      mlir::FailureOr<PlannedPhysicalDecision<VLARegionDecision>> decision =
          decideVLARegion(vla);
      if (mlir::failed(decision)) {
        decisionFailure = true;
        return;
      }
      if (!physicalPlan.vlaRegions.try_emplace(vla.getOperation(), std::move(*decision))
               .second) {
        vla.emitError("one VLA region cannot own multiple physical decisions");
        decisionFailure = true;
      }
    });
    kernel.walk([&](AffineI4I8DotOp dot) {
      if (decisionFailure)
        return;
      PlannedPhysicalDecision<AffineI4I8Decision> planned;
      if (mlir::failed(
              decideAffineI4I8Dot(dot, planned.realization))) {
        decisionFailure = true;
        return;
      }
      finalizeAffineI4I8Plan(dot, planned);
      physicalPlan.affineI4I8.try_emplace(dot.getOperation(),
                                         std::move(planned));
    });
    kernel.walk([&](MatmulOp matmul) {
      if (decisionFailure)
        return;
      std::optional<PlannedPhysicalDecision<F16GemmNTileDecision>> decision =
          decideF16GemmNTiles(matmul);
      if (decision)
        physicalPlan.f16Matmuls.try_emplace(matmul.getOperation(),
                                         std::move(*decision));
    });
    kernel.walk([&](DotOp dot) {
      if (decisionFailure || isRegionValue(dot.getResult().getType()))
        return;
      mlir::FailureOr<PlannedPhysicalDecision<DotDecision>> decision =
          decideLocalF32RowMicrotile(dot);
      if (mlir::failed(decision)) {
        decisionFailure = true;
        return;
      }
      if (!physicalPlan.dots
               .try_emplace(dot.getOperation(), std::move(*decision))
               .second) {
        dot.emitError("one dot cannot own multiple physical decisions");
        decisionFailure = true;
        return;
      }
    });
    kernel.walk([&](SymmetricI4I8DotOp op) {
      if (decisionFailure)
        return;
      PlannedPhysicalDecision<SymmetricI4I8Decision> planned;
      if (mlir::failed(
              decideSymmetricI4I8Dot(op, planned.realization))) {
        decisionFailure = true;
        return;
      }
      finalizeSymmetricI4I8Plan(op, planned);
      physicalPlan.symmetricI4I8.try_emplace(op.getOperation(),
                                             std::move(planned));
    });
    kernel.walk([&](SignBitI8DotOp op) {
      if (decisionFailure)
        return;
      PlannedPhysicalDecision<SignBitI8Decision> planned;
      if (mlir::failed(decideSignBitI8Dot(op, planned.realization))) {
        decisionFailure = true;
        return;
      }
      finalizeSignBitI8Plan(op, planned);
      physicalPlan.signBitI8.try_emplace(op.getOperation(), std::move(planned));
    });
    kernel.walk([&](E2M1E8M0I8DotOp op) {
      if (decisionFailure)
        return;
      PlannedPhysicalDecision<E2M1E8M0I8Decision> planned;
      if (mlir::failed(decideE2M1E8M0I8Dot(op, planned.realization))) {
        decisionFailure = true;
        return;
      }
      finalizeE2M1E8M0I8Plan(op, planned);
      physicalPlan.e2m1E8M0I8.try_emplace(op.getOperation(),
                                          std::move(planned));
    });
    kernel.walk([&](GroupedAffineI4I8DotOp op) {
      if (decisionFailure)
        return;
      PlannedPhysicalDecision<GroupedAffineI4I8Decision> planned;
      if (mlir::failed(decideGroupedAffineI4I8Dot(op, planned.realization))) {
        decisionFailure = true;
        return;
      }
      finalizeGroupedAffineI4I8Plan(op, planned);
      physicalPlan.groupedAffineI4I8.try_emplace(op.getOperation(),
                                                 std::move(planned));
    });
    auto prepareQuantCodebookDot = [&](auto op,
                                       QuantI8DotSemantic semantic,
                                       llvm::ArrayRef<mlir::Value> blocks,
                                       llvm::ArrayRef<mlir::Value> scalars) {
      if (decisionFailure)
        return;
      PlannedPhysicalDecision<QuantCodebookI8Decision> planned;
      if (mlir::failed(
              decideQuantCodebookI8Dot(op, semantic, blocks, scalars,
                                       planned.realization))) {
        decisionFailure = true;
        return;
      }
      finalizeQuantCodebookI8Plan(op, planned);
      physicalPlan.quantCodebookI8.try_emplace(op.getOperation(),
                                               std::move(planned));
    };
    kernel.walk([&](IQ2SI8DotOp op) {
      prepareQuantCodebookDot(
          op, QuantI8DotSemantic::IQ2S,
          {op.getCodes(), op.getHighBits(), op.getSignBits(), op.getScales(),
           op.getActivation()},
          {op.getWeightScale(), op.getActivationScale(), op.getInit()});
    });
    kernel.walk([&](IQ3SI8DotOp op) {
      prepareQuantCodebookDot(
          op, QuantI8DotSemantic::IQ3S,
          {op.getCodes(), op.getHighBits(), op.getSignBits(), op.getScales(),
           op.getActivation()},
          {op.getWeightScale(), op.getActivationScale(), op.getInit()});
    });
    kernel.walk([&](IQ1MI8DotOp op) {
      prepareQuantCodebookDot(
          op, QuantI8DotSemantic::IQ1M,
          {op.getCodes(), op.getHighDeltaBits(), op.getScales(),
           op.getActivation()},
          {op.getActivationScale(), op.getInit()});
    });
    kernel.walk([&](Q6KI8DotOp op) {
      prepareQuantCodebookDot(
          op, QuantI8DotSemantic::Q6K,
          {op.getLowBits(), op.getHighBits(), op.getGroupScales(),
           op.getActivation()},
          {op.getWeightScale(), op.getActivationScale(), op.getInit()});
    });
    kernel.walk([&](DecodeOp op) {
      if (decisionFailure)
        return;
      PlannedPhysicalDecision<BlockDecodeDecision> planned;
      if (mlir::failed(decideBlockDecode(op, planned.realization))) {
        decisionFailure = true;
        return;
      }
      finalizeBlockDecodePlan(op, planned);
      if (!physicalPlan.blockDecodes
               .try_emplace(op.getOperation(), std::move(planned))
               .second) {
        op.emitError("one decode primitive cannot own multiple physical decisions");
        decisionFailure = true;
      }
    });
    if (!decisionFailure && mlir::failed(prepareMaterializedF32Pointwise()))
      decisionFailure = true;
    if (!decisionFailure && mlir::failed(prepareBlockGroupDecisions()))
      decisionFailure = true;
    kernel.walk([&](StoreOp store) {
      if (decisionFailure || !isMaterializedF32BlockStore(store))
        return;
      if (mlir::failed(
              decideMaterializedBlockStore(store, store.getValue().getType())))
        decisionFailure = true;
    });
    if (decisionFailure)
      return mlir::failure();
    return mlir::success();
  }

  void collectSelectedIntrinsicCLeaves() {
    for (const auto &entry : physicalPlan.vlaRegions) {
      const VLARegionDecision &decision = entry.second.realization;
      if (!decision.unaries.empty())
        selectedLeaves.add(IntrinsicCLeaf::RVVF32M2Math);
      for (const VLAStateDecision &state : decision.states)
        if (state.realization == VLAStateRealization::RVVOnlineSoftmaxSummary)
          selectedLeaves.add(IntrinsicCLeaf::RVVF32M2Math);
    }
    for (const auto &entry : physicalPlan.affineI4I8)
      selectedLeaves.add(entry.second.realization.leaf);
    for (const auto &entry : physicalPlan.symmetricI4I8)
      selectedLeaves.add(entry.second.realization.leaf);
    for (const auto &entry : physicalPlan.e2m1E8M0I8)
      selectedLeaves.add(entry.second.realization.leaf);
    for (const auto &entry : physicalPlan.groupedAffineI4I8)
      selectedLeaves.add(entry.second.realization.leaf);
    for (const auto &entry : physicalPlan.quantCodebookI8)
      selectedLeaves.add(entry.second.realization.leaf);
  }

  KernelOp kernel;
  const RISCVLoweringOptions &options;
  llvm::raw_ostream &output;
  SelectedIntrinsicCLeaves &selectedLeaves;
  llvm::DenseMap<mlir::Value, CValue> values;
  KernelPhysicalFacts kernelFacts;
  RISCVPhysicalPlan physicalPlan;
  const llvm::SmallVector<BlockOperationDecision> *activeBlockOperations =
      nullptr;
  const PhysicalEntityPlan *activeBlockEntity = nullptr;
  llvm::DenseSet<mlir::Operation *> consumed;
  llvm::DenseSet<mlir::Operation *> deferredBlockOps;
  llvm::DenseSet<mlir::Operation *> loweredBlockOps;
  llvm::DenseMap<mlir::Operation *, int64_t> materializedBlockElements;
  llvm::DenseMap<mlir::Value, int64_t> controlBlockElements;
  unsigned indent = 0;
  unsigned nextValue = 0;
  unsigned nextLoop = 0;
  bool inVLA = false;
  std::string activeVL;
  const VLARegionDecision *activeVLADecision = nullptr;
  const PhysicalEntityPlan *activePhysicalEntity = nullptr;

  void line(llvm::Twine text) {
    output.indent(indent * 2) << text << '\n';
  }

  std::string fresh(llvm::StringRef prefix) {
    return "__weft_" + prefix.str() + std::to_string(nextValue++);
  }

  CValue scalarExpression(mlir::Value result, std::string expression,
                          llvm::StringRef prefix, bool force = false) {
    if (force || !result.use_empty()) {
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

  mlir::FailureOr<std::string>
  combineLogicalValidity(mlir::Operation *owner,
                         llvm::ArrayRef<CValue> operands) {
    llvm::SmallVector<std::string> predicates;
    for (const CValue &operand : operands) {
      if (operand.logicalValidity.empty() ||
          llvm::is_contained(predicates, operand.logicalValidity))
        continue;
      predicates.push_back(operand.logicalValidity);
    }
    if (predicates.empty())
      return std::string{};
    if (!activePhysicalEntity || activePhysicalEntity->vlaMaskRatio == 0 ||
        activeVL.empty()) {
      owner->emitError(
          "logical validity has no selected physical mask realization");
      return mlir::failure();
    }
    std::string merged = predicates.front();
    std::string ratio =
        std::to_string(activePhysicalEntity->vlaMaskRatio);
    for (llvm::StringRef predicate : llvm::drop_begin(predicates)) {
      std::string next = fresh("valid");
      line("vbool" + ratio + "_t " + next + " = __riscv_vmand_mm_b" +
           ratio + "(" + merged + ", " + predicate.str() + ", " + activeVL +
           ");");
      merged = std::move(next);
    }
    return merged;
  }

  mlir::LogicalResult
  attachLogicalValidity(mlir::Operation *owner, mlir::Type resultType,
                        CValue &result, llvm::ArrayRef<CValue> operands) {
    mlir::FailureOr<std::string> validity =
        combineLogicalValidity(owner, operands);
    if (mlir::failed(validity))
      return mlir::failure();
    bool resultIsMasked = mlir::isa<MaskedType>(resultType);
    if (resultIsMasked != !validity->empty())
      return owner->emitError(
          "physical logical-validity projection disagrees with the Kernel IR type");
    result.logicalValidity = std::move(*validity);
    return mlir::success();
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

  const VLASegment2Decision *
  findSegment2AccessDecision(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return nullptr;
    auto found = llvm::find_if(
        activeVLADecision->segment2,
        [&](const VLASegment2Decision &decision) {
          return decision.field0 == operation || decision.field1 == operation;
        });
    return found == activeVLADecision->segment2.end() ? nullptr : &*found;
  }

  const VLALookupDecision *
  findLookupDecision(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return nullptr;
    auto found = llvm::find_if(
        activeVLADecision->lookups,
        [&](const VLALookupDecision &decision) {
          return decision.operation == operation;
        });
    return found == activeVLADecision->lookups.end() ? nullptr : &*found;
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

  const VLAIndexBinaryDecision *
  findIndexBinaryDecision(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return nullptr;
    auto found = llvm::find_if(
        activeVLADecision->indexBinaries,
        [&](const VLAIndexBinaryDecision &decision) {
          return decision.operation == operation;
        });
    return found == activeVLADecision->indexBinaries.end() ? nullptr : &*found;
  }

  const VLAIndexSelectDecision *
  findIndexSelectDecision(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return nullptr;
    auto found = llvm::find_if(
        activeVLADecision->indexSelects,
        [&](const VLAIndexSelectDecision &decision) {
          return decision.operation == operation;
        });
    return found == activeVLADecision->indexSelects.end() ? nullptr : &*found;
  }

  const VLAUnaryDecision *
  findUnaryDecision(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return nullptr;
    auto found = llvm::find_if(
        activeVLADecision->unaries, [&](const VLAUnaryDecision &decision) {
          return decision.operation == operation;
        });
    return found == activeVLADecision->unaries.end() ? nullptr : &*found;
  }

  bool isAbsorbedBinaryProducer(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return false;
    return llvm::any_of(
        activeVLADecision->binaries, [&](const VLABinaryDecision &decision) {
          return decision.absorbedProducer == operation;
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

  const VLADotDecision *
  findVLADotDecision(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return nullptr;
    auto found = llvm::find_if(
        activeVLADecision->dots,
        [&](const VLADotDecision &decision) {
          return decision.operation == operation;
        });
    return found == activeVLADecision->dots.end() ? nullptr : &*found;
  }

  const BlockOperationDecision *
  findBlockOperationDecision(mlir::Operation *operation) const {
    if (!activeBlockOperations)
      return nullptr;
    auto found = llvm::find_if(
        *activeBlockOperations, [&](const BlockOperationDecision &decision) {
          return decision.operation == operation;
        });
    return found == activeBlockOperations->end() ? nullptr : &*found;
  }

  const PhysicalValueDecision *
  findBlockValueDecision(mlir::Value value) const {
    if (!activeBlockEntity)
      return nullptr;
    auto found = llvm::find_if(
        activeBlockEntity->values,
        [&](const PhysicalValueDecision &decision) {
          return decision.value == value;
        });
    return found == activeBlockEntity->values.end() ? nullptr : &*found;
  }

  const PhysicalTemporaryDecision *
  findBlockTemporaryDecision(mlir::Operation *owner) const {
    if (!activeBlockEntity)
      return nullptr;
    auto found = llvm::find_if(
        activeBlockEntity->temporaries,
        [&](const PhysicalTemporaryDecision &decision) {
          return decision.owner == owner;
        });
    return found == activeBlockEntity->temporaries.end() ? nullptr : &*found;
  }

  const PhysicalHandoffDecision *
  findVLAHandoff(mlir::Operation *consumer, mlir::Value value) const {
    if (!activePhysicalEntity)
      return nullptr;
    auto found = llvm::find_if(
        activePhysicalEntity->handoffs,
        [&](const PhysicalHandoffDecision &handoff) {
          return handoff.consumer == consumer && handoff.value == value;
        });
    return found == activePhysicalEntity->handoffs.end() ? nullptr : &*found;
  }

  const PhysicalValueDecision *
  findVLAValueDecision(mlir::Value value) const {
    if (!activePhysicalEntity)
      return nullptr;
    auto found = llvm::find_if(
        activePhysicalEntity->values,
        [&](const PhysicalValueDecision &decision) {
          return decision.value == value;
        });
    return found == activePhysicalEntity->values.end() ? nullptr : &*found;
  }

  const PhysicalTemporaryDecision *
  findVLATemporaryDecision(mlir::Operation *owner) const {
    if (!activePhysicalEntity)
      return nullptr;
    auto found = llvm::find_if(
        activePhysicalEntity->temporaries,
        [&](const PhysicalTemporaryDecision &decision) {
          return decision.owner == owner;
        });
    return found == activePhysicalEntity->temporaries.end() ? nullptr : &*found;
  }

  std::optional<unsigned> activeVLADataLMUL() const {
    return activePhysicalEntity
               ? rvvIntegerLMUL(activePhysicalEntity->vlaDataShape)
               : std::nullopt;
  }

  std::optional<unsigned> activeVLAIndexLMUL() const {
    return activePhysicalEntity
               ? rvvIntegerLMUL(activePhysicalEntity->vlaIndexShape)
               : std::nullopt;
  }

  std::optional<unsigned> physicalValueLMUL(mlir::Value value) const {
    const PhysicalValueDecision *decision = findVLAValueDecision(value);
    return decision ? rvvIntegerLMUL(decision->shape) : std::nullopt;
  }

  template <typename Decision>
  mlir::LogicalResult requireEntityPlan(
      mlir::Operation *owner,
      const PlannedPhysicalDecision<Decision> &planned) const {
    const PhysicalResourceBudget &resources = planned.entity.resources;
    if (resources.architecturalGroups != options.target.vectorRegisters ||
        planned.entity.hasValueShapeConflict ||
        resources.peakGroups > resources.architecturalGroups)
      return owner->emitError("physical entity resource plan is invalid");
    return mlir::success();
  }

  const PhysicalHandoffDecision *
  findPhysicalHandoff(const PhysicalEntityPlan &entity,
                      mlir::Operation *consumer, mlir::Value value) const {
    auto found = llvm::find_if(
        entity.handoffs, [&](const PhysicalHandoffDecision &handoff) {
          return handoff.consumer == consumer && handoff.value == value;
        });
    return found == entity.handoffs.end() ? nullptr : &*found;
  }

  mlir::LogicalResult requirePhysicalHandoff(
      mlir::Operation *owner, const PhysicalEntityPlan &entity,
      mlir::Value value, PhysicalHandoff kind) const {
    const PhysicalHandoffDecision *handoff =
        findPhysicalHandoff(entity, owner, value);
    if (!handoff || handoff->kind != kind || !handoff->sourceShape ||
        !handoff->resultShape)
      return owner->emitError("local primitive physical handoff is incomplete");
    return mlir::success();
  }

  bool isVLADotAbsorbed(mlir::Operation *operation) const {
    if (!activeVLADecision)
      return false;
    return llvm::any_of(
        activeVLADecision->dots,
        [&](const VLADotDecision &decision) {
          return llvm::is_contained(decision.absorbed, operation);
        });
  }

  void collectLocalDefinitions(
      mlir::Value value, mlir::Block &block,
      llvm::DenseSet<mlir::Operation *> &definitions,
      llvm::SmallVectorImpl<BlockIndexOp> &axes) const {
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition || definition->getBlock() != &block ||
        !definitions.insert(definition).second)
      return;
    if (auto axis = mlir::dyn_cast<BlockIndexOp>(definition))
      axes.push_back(axis);
    for (mlir::Value operand : definition->getOperands())
      collectLocalDefinitions(operand, block, definitions, axes);
  }

  void retainOnlyPrivateDefinitions(
      llvm::DenseSet<mlir::Operation *> &definitions,
      mlir::Operation *consumer) const {
    llvm::SmallVector<mlir::Operation *> required;
    for (mlir::Operation *definition : definitions) {
      if (mlir::isa<BlockIndexOp>(definition))
        continue;
      if (llvm::any_of(definition->getUsers(), [&](mlir::Operation *user) {
            return user != consumer && !definitions.contains(user);
          }))
        required.push_back(definition);
    }
    while (!required.empty()) {
      mlir::Operation *operation = required.pop_back_val();
      if (!definitions.erase(operation))
        continue;
      for (mlir::Value operand : operation->getOperands())
        if (mlir::Operation *producer = operand.getDefiningOp();
            producer && definitions.contains(producer))
          required.push_back(producer);
    }
  }

  mlir::FailureOr<VLADotDecision>
  decideVLAF32FreeAxisMicrotile(VLAOp vla, DotOp dot) {
    BlockType lhsType =
        mlir::dyn_cast<BlockType>(unwrapLogicalValidity(dot.getLhs().getType()));
    RegionType rhsType =
        mlir::dyn_cast<RegionType>(unwrapLogicalValidity(dot.getRhs().getType()));
    RegionType resultType = mlir::dyn_cast<RegionType>(
        unwrapLogicalValidity(dot.getResult().getType()));
    if (!lhsType || !rhsType || !resultType ||
        lhsType.getShape().size() != 2 ||
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

    LoadOp lhsLoad = reloadableLoad(dot.getLhs());
    LoadOp rhsLoad = reloadableLoad(dot.getRhs());
    if (!lhsLoad || !rhsLoad || !isTrue(rhsLoad.getWhere()) ||
        (!isTrue(lhsLoad.getWhere()) &&
         !isFloatConstant(lhsLoad.getOther(), 0.0))) {
      dot.emitError("RVV VLA dot load facts are unavailable");
      return mlir::failure();
    }

    mlir::Block &block = *dot->getBlock();
    std::optional<StructuredProductFacts> product =
        structuredProductFacts(dot.getLhs(), dot.getRhs(), dot.getResult());
    mlir::Value coordinate = vla.getBody().front().getArgument(0);
    if (!product || product->lhsFreeAxes.size() != 1 ||
        product->rhsFreeAxes.size() != 1 ||
        product->rhsFreeAxes.front() != coordinate ||
        product->reductionAxes.size() != 1 ||
        product->resultAxes.size() != 2) {
      dot.emitError(
          "RVV VLA dot requires explicit row and shared reduction axes");
      return mlir::failure();
    }
    BlockIndexOp reductionAxis =
        product->reductionAxes.front().getDefiningOp<BlockIndexOp>();
    BlockIndexOp rowAxis =
        product->lhsFreeAxes.front().getDefiningOp<BlockIndexOp>();
    if (!reductionAxis || !rowAxis) {
      dot.emitError("RVV VLA dot block-axis facts are unavailable");
      return mlir::failure();
    }
    if (integerConstantValue(rowAxis.getExtent()) != lhsType.getShape()[0]) {
      dot.emitError(
          "RVV VLA dot result axes do not preserve the local row block");
      return mlir::failure();
    }

    const MemoryAccessFact *lhsAccess = memoryFact(lhsLoad.getOperation());
    const MemoryAccessFact *rhsAccess = memoryFact(rhsLoad.getOperation());
    if (!lhsAccess || !rhsAccess) {
      dot.emitError("RVV VLA dot typed memory facts are unavailable");
      return mlir::failure();
    }
    mlir::Value lhsRoot = lhsAccess->root;
    mlir::Value rhsRoot = rhsAccess->root;
    auto lhsPointer =
        lhsRoot ? mlir::dyn_cast<PtrType>(lhsRoot.getType()) : PtrType{};
    auto rhsPointer =
        rhsRoot ? mlir::dyn_cast<PtrType>(rhsRoot.getType()) : PtrType{};
    LaneRelation rhsRelation = memoryRelation(*rhsAccess, coordinate);
    if (!lhsPointer || !rhsPointer ||
        !lhsPointer.getElementType().isF32() ||
        !rhsPointer.getElementType().isF32() ||
        (rhsRelation != LaneRelation::UnitStride &&
         rhsRelation != LaneRelation::Strided) ||
        memoryRelation(*lhsAccess, rowAxis.getResult()) ==
            LaneRelation::Independent ||
        memoryRelation(*lhsAccess, reductionAxis.getResult()) ==
            LaneRelation::Independent ||
        memoryRelation(*lhsAccess, coordinate) != LaneRelation::Independent ||
        memoryRelation(*rhsAccess, coordinate) == LaneRelation::Independent ||
        memoryRelation(*rhsAccess, reductionAxis.getResult()) ==
            LaneRelation::Independent ||
        memoryRelation(*rhsAccess, rowAxis.getResult()) !=
            LaneRelation::Independent) {
      dot.emitError(
          "RVV VLA dot memory relations are unavailable");
      return mlir::failure();
    }

    llvm::DenseSet<mlir::Operation *> absorbed;
    llvm::SmallVector<BlockIndexOp> absorbedAxes;
    for (mlir::Value value : {dot.getLhs(), dot.getRhs()})
      collectLocalDefinitions(value, block, absorbed, absorbedAxes);

    VLADotDecision decision;
    decision.operation = dot.getOperation();
    decision.realization =
        VLADotRealization::RVVF32FreeAxisMicrotile;
    decision.lhsLoad = lhsLoad.getOperation();
    decision.rhsLoad = rhsLoad.getOperation();
    decision.init = dot.getInit();
    decision.rowAxis = rowAxis.getResult();
    decision.reductionAxis = reductionAxis.getResult();
    decision.reductionExtent = reductionAxis.getExtent();
    decision.rowTile = static_cast<unsigned>(lhsType.getShape()[0]);
    F32DotCandidateFacts candidateFacts;
    candidateFacts.model = F32DotResourceModel::VLAFreeAxis;
    candidateFacts.rowTile = decision.rowTile;
    if (std::optional<int64_t> extent =
            integerConstantValue(decision.reductionExtent);
        extent && *extent >= 0)
      candidateFacts.reductionExtent = static_cast<uint64_t>(*extent);
    candidateFacts.unitStrideOperands =
        1 + (rhsRelation == LaneRelation::UnitStride ? 1 : 0);
    candidateFacts.stridedOperands =
        rhsRelation == LaneRelation::Strided ? 1 : 0;
    candidateFacts.reductionPredicate =
        valueDependsOnAxis(lhsLoad.getWhere(), reductionAxis.getResult());
    candidateFacts.predicateGroups =
        candidateFacts.reductionPredicate ? 1 : 0;
    std::optional<F32DotPhysicalConfig> physical =
        weft::riscv_internal::selectF32DotPhysicalConfig(
            candidateFacts, options.target, options.backend);
    if (!physical) {
      dot.emitError(
          "RVV VLA dot has no legal register microtile candidate");
      return mlir::failure();
    }
    decision.rhsMemoryMode = rhsRelation == LaneRelation::UnitStride
                                 ? VLAMemoryMode::UnitStride
                                 : VLAMemoryMode::Strided;
    decision.lhsPredicateVariesByReduction =
        valueDependsOnAxis(lhsLoad.getWhere(), reductionAxis.getResult());
    decision.physical = *physical;
    retainOnlyPrivateDefinitions(absorbed, dot.getOperation());
    decision.absorbed.append(absorbed.begin(), absorbed.end());
    return decision;
  }

  mlir::FailureOr<VLADotDecision>
  decideVLAF32FreeAxisVectorDot(VLAOp vla, DotOp dot) {
    RegionType lhsType = mlir::dyn_cast<RegionType>(
        unwrapLogicalValidity(dot.getLhs().getType()));
    BlockType rhsType =
        mlir::dyn_cast<BlockType>(unwrapLogicalValidity(dot.getRhs().getType()));
    RegionType initType = mlir::dyn_cast<RegionType>(
        unwrapLogicalValidity(dot.getInit().getType()));
    RegionType resultType = mlir::dyn_cast<RegionType>(
        unwrapLogicalValidity(dot.getResult().getType()));
    if (!lhsType || !rhsType || !initType || !resultType ||
        lhsType.getShape().size() != 2 ||
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

    LoadOp freeLoad = reloadableLoad(dot.getLhs());
    if (!freeLoad || !isTrue(freeLoad.getWhere())) {
      dot.emitError("RVV VLA vector-dot load facts are unavailable");
      return mlir::failure();
    }

    mlir::Block &block = *dot->getBlock();
    std::optional<StructuredProductFacts> product =
        structuredProductFacts(dot.getLhs(), dot.getRhs(), dot.getResult());
    mlir::Value coordinate = vla.getBody().front().getArgument(0);
    if (!product || product->lhsFreeAxes.size() != 1 ||
        product->lhsFreeAxes.front() != coordinate ||
        !product->rhsFreeAxes.empty() ||
        product->reductionAxes.size() != 1 ||
        product->resultAxes.size() != 1) {
      dot.emitError(
          "RVV VLA vector dot requires one shared explicit reduction axis");
      return mlir::failure();
    }
    BlockIndexOp reductionAxis =
        product->reductionAxes.front().getDefiningOp<BlockIndexOp>();
    if (!reductionAxis) {
      dot.emitError("RVV VLA vector dot block-axis fact is unavailable");
      return mlir::failure();
    }

    const MemoryAccessFact *freeAccess = memoryFact(freeLoad.getOperation());
    if (!freeAccess) {
      dot.emitError("RVV VLA vector-dot typed memory fact is unavailable");
      return mlir::failure();
    }
    mlir::Value freeRoot = freeAccess->root;
    auto freePointer =
        freeRoot ? mlir::dyn_cast<PtrType>(freeRoot.getType()) : PtrType{};
    LaneRelation freeRelation = memoryRelation(*freeAccess, coordinate);
    if (!freePointer || !freePointer.getElementType().isF32() ||
        (freeRelation != LaneRelation::UnitStride &&
         freeRelation != LaneRelation::Strided) ||
        memoryRelation(*freeAccess, coordinate) == LaneRelation::Independent ||
        memoryRelation(*freeAccess, reductionAxis.getResult()) ==
            LaneRelation::Independent ||
        !valueDependsOnAxis(dot.getInit(), coordinate) ||
        valueDependsOnAxis(dot.getInit(), reductionAxis.getResult()) ||
        valueDependsOnAxis(dot.getRhs(), coordinate) ||
        !valueDependsOnAxis(dot.getRhs(), reductionAxis.getResult())) {
      dot.emitError("RVV VLA vector-dot memory relations are unavailable");
      return mlir::failure();
    }

    llvm::DenseSet<mlir::Operation *> absorbed;
    llvm::SmallVector<BlockIndexOp> absorbedAxes;
    for (mlir::Value value : {dot.getLhs(), dot.getRhs()})
      collectLocalDefinitions(value, block, absorbed, absorbedAxes);
    llvm::DenseSet<mlir::Operation *> initDefinitions;
    llvm::SmallVector<BlockIndexOp> initAxes;
    collectLocalDefinitions(dot.getInit(), block, initDefinitions, initAxes);
    for (mlir::Operation *operation : initDefinitions)
      absorbed.erase(operation);

    VLADotDecision decision;
    decision.operation = dot.getOperation();
    decision.realization = VLADotRealization::RVVF32FreeAxisVectorDot;
    decision.freeLoad = freeLoad.getOperation();
    decision.blockedOperand = dot.getRhs();
    decision.init = dot.getInit();
    decision.reductionAxis = reductionAxis.getResult();
    decision.reductionExtent = reductionAxis.getExtent();
    decision.rowTile = 1;
    F32DotCandidateFacts candidateFacts;
    candidateFacts.model = F32DotResourceModel::VLAFreeAxis;
    candidateFacts.rowTile = decision.rowTile;
    if (std::optional<int64_t> extent =
            integerConstantValue(decision.reductionExtent);
        extent && *extent >= 0)
      candidateFacts.reductionExtent = static_cast<uint64_t>(*extent);
    candidateFacts.unitStrideOperands =
        freeRelation == LaneRelation::UnitStride ? 1 : 0;
    candidateFacts.stridedOperands =
        freeRelation == LaneRelation::Strided ? 1 : 0;
    candidateFacts.handoffGroups = 1;
    candidateFacts.materializedInit = true;
    std::optional<F32DotPhysicalConfig> physical =
        weft::riscv_internal::selectF32DotPhysicalConfig(
            candidateFacts, options.target, options.backend);
    if (!physical) {
      dot.emitError(
          "RVV VLA vector dot has no legal register candidate");
      return mlir::failure();
    }
    decision.freeMemoryMode = freeRelation == LaneRelation::UnitStride
                                  ? VLAMemoryMode::UnitStride
                                  : VLAMemoryMode::Strided;
    decision.initRealization =
        VLADotInitRealization::MaterializedRegion;
    decision.physical = *physical;
    retainOnlyPrivateDefinitions(absorbed, dot.getOperation());
    decision.absorbed.append(absorbed.begin(), absorbed.end());
    return decision;
  }

  mlir::FailureOr<PlannedPhysicalDecision<VLARegionDecision>>
  decideVLARegion(VLAOp op) {
    mlir::Block &body = op.getBody().front();
    PlannedPhysicalDecision<VLARegionDecision> planned;
    VLARegionDecision &decision = planned.realization;
    PhysicalEntityPlan &entity = planned.entity;
    decision.operation = op.getOperation();
    decision.coordinate = body.getArgument(0);
    if (options.target.xlen != 32 && options.target.xlen != 64) {
      op.emitError("VLA index realization requires a 32-bit or 64-bit target");
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Operation *> physicalOperations;
    std::function<void(mlir::Block &)> collectPhysicalOperations =
        [&](mlir::Block &block) {
          for (mlir::Operation &nested : block.without_terminator()) {
            physicalOperations.push_back(&nested);
            if (!mlir::isa<ForOp, IfOp, WhileOp>(nested))
              continue;
            for (mlir::Region &region : nested.getRegions())
              for (mlir::Block &nestedBlock : region)
                collectPhysicalOperations(nestedBlock);
          }
        };
    collectPhysicalOperations(body);

    llvm::DenseMap<mlir::Operation *, VLANarrowPhysicalConfig> narrowPhysical;
    for (mlir::Operation *nested : physicalOperations) {
      auto dot = mlir::dyn_cast<DotOp>(nested);
      if (!dot || !isRegionValue(dot.getResult().getType()))
        continue;
      mlir::FailureOr<VLADotDecision> selected =
          isRegionValue(dot.getLhs().getType()) &&
                  containsBlockType(dot.getRhs().getType())
              ? decideVLAF32FreeAxisVectorDot(op, dot)
              : decideVLAF32FreeAxisMicrotile(op, dot);
      if (mlir::failed(selected))
        return mlir::failure();
      decision.dots.push_back(std::move(*selected));
    }
    auto isDotOwned = [&](mlir::Operation *operation) {
      return llvm::any_of(
          decision.dots, [&](const VLADotDecision &dot) {
            return llvm::is_contained(dot.absorbed, operation);
          });
    };

    for (mlir::Operation *nested : physicalOperations) {
      if (isDotOwned(nested))
        continue;
      if (auto unary = mlir::dyn_cast<UnaryOp>(nested)) {
        if (isRegionValue(unary.getResult().getType()) &&
            elementType(unary.getResult().getType()).isF32()) {
          if (unary.getMath() == "strict" &&
              (unary.getKind() == "exp" || unary.getKind() == "tanh")) {
            unary.emitError(
                "strict VLA exp/tanh has no exact RISC-V realization");
            return mlir::failure();
          }
          std::optional<VLAUnaryRealization> realization =
              llvm::StringSwitch<std::optional<VLAUnaryRealization>>(
                  unary.getKind())
                  .Case("exp", VLAUnaryRealization::RVVF32M2ExpPolynomial)
                  .Case("tanh", VLAUnaryRealization::RVVF32M2TanhViaExp)
                  .Case("sin", VLAUnaryRealization::RVVF32M2ScalarLibmSin)
                  .Case("cos", VLAUnaryRealization::RVVF32M2ScalarLibmCos)
                  .Default(std::nullopt);
          if (realization) {
            decision.unaries.push_back(
                VLAUnaryDecision{unary.getOperation(), *realization});
            continue;
          }
        }
      }
      if (auto binary = mlir::dyn_cast<BinaryOp>(nested)) {
        if (!isRegionValue(binary.getResult().getType()) ||
            !elementType(binary.getResult().getType()).isIndex() ||
            classifyLaneRelation(binary.getResult(), decision.coordinate) !=
                LaneRelation::Indexed)
          continue;
        bool lhsRegion = isRegionValue(binary.getLhs().getType());
        bool rhsRegion = isRegionValue(binary.getRhs().getType());
        if ((!lhsRegion && !rhsRegion) ||
            !elementType(binary.getLhs().getType()).isIndex() ||
            !elementType(binary.getRhs().getType()).isIndex() ||
            !llvm::StringSwitch<bool>(binary.getKind())
                 .Cases("add", "sub", "mul", "div", "mod", true)
                 .Default(false)) {
          binary.emitError(
              "VLA index binary has no unsigned RVV realization");
          return mlir::failure();
        }
        if (!lhsRegion &&
            (binary.getKind() == "div" || binary.getKind() == "mod")) {
          binary.emitError(
              "scalar-left VLA index division/remainder has no RVV realization");
          return mlir::failure();
        }
        decision.indexBinaries.push_back(VLAIndexBinaryDecision{
            binary.getOperation(),
            lhsRegion && rhsRegion
                ? VLAIndexBinaryRealization::RVVUnsignedVectorVector
                : VLAIndexBinaryRealization::RVVUnsignedVectorScalar,
            !lhsRegion});
        continue;
      }
      if (auto select = mlir::dyn_cast<SelectOp>(nested)) {
        if (!isRegionValue(select.getResult().getType()) ||
            !elementType(select.getResult().getType()).isIndex())
          continue;
        bool trueRegion = isRegionValue(select.getTrueValue().getType());
        bool falseRegion = isRegionValue(select.getFalseValue().getType());
        if ((!trueRegion && !falseRegion) ||
            !isRegionValue(select.getPredicate().getType())) {
          select.emitError(
              "VLA index select requires a vector predicate and an index vector operand");
          return mlir::failure();
        }
        decision.indexSelects.push_back(
            VLAIndexSelectDecision{select.getOperation(), !trueRegion,
                                   !falseRegion});
      }
    }

    for (mlir::Operation *nested : physicalOperations) {
      if (isDotOwned(nested))
        continue;
      auto compare = mlir::dyn_cast<CompareOp>(nested);
      if (!compare || !isRegionValue(compare.getResult().getType()))
        continue;
      LaneRelation lhs =
          classifyLaneRelation(compare.getLhs(), decision.coordinate);
      LaneRelation rhs =
          classifyLaneRelation(compare.getRhs(), decision.coordinate);
      bool lhsCoordinate = lhs == LaneRelation::UnitStride ||
                           lhs == LaneRelation::Strided;
      bool rhsCoordinate = rhs == LaneRelation::UnitStride ||
                           rhs == LaneRelation::Strided;
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
        else if (vectorElement.isIndex())
          predicateDecision.vectorSEW = options.target.xlen;
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
        std::function<bool(mlir::Value, llvm::DenseSet<mlir::Value>)>
            hasPredicateDecision =
                [&](mlir::Value value,
                    llvm::DenseSet<mlir::Value> visitedPredicates) {
              if (!value || !visitedPredicates.insert(value).second)
                return false;
              if (llvm::any_of(
                      decision.predicates,
                      [&](const VLAPredicateDecision &candidate) {
                        return candidate.operation == value.getDefiningOp();
                      }))
                return true;
              if (auto binary = value.getDefiningOp<BinaryOp>())
                return (binary.getKind() == "and" ||
                        binary.getKind() == "or" ||
                        binary.getKind() == "xor") &&
                       hasPredicateDecision(binary.getLhs(), visitedPredicates) &&
                       hasPredicateDecision(binary.getRhs(), visitedPredicates);
              if (auto conditional = value.getDefiningOp<IfOp>()) {
                auto result = mlir::dyn_cast<mlir::OpResult>(value);
                if (!result)
                  return false;
                unsigned number = result.getResultNumber();
                auto thenYield = mlir::cast<YieldOp>(
                    conditional.getThenRegion().front().getTerminator());
                auto elseYield = mlir::cast<YieldOp>(
                    conditional.getElseRegion().front().getTerminator());
                return number < thenYield.getNumOperands() &&
                       number < elseYield.getNumOperands() &&
                       hasPredicateDecision(thenYield.getOperand(number),
                                            visitedPredicates) &&
                       hasPredicateDecision(elseYield.getOperand(number),
                                            visitedPredicates);
              }
              if (auto loop = value.getDefiningOp<ForOp>()) {
                auto result = mlir::dyn_cast<mlir::OpResult>(value);
                auto yield =
                    mlir::cast<YieldOp>(loop.getBody().front().getTerminator());
                unsigned number = result ? result.getResultNumber() : 0;
                return result && number < loop.getInitArgs().size() &&
                       number < yield.getNumOperands() &&
                       hasPredicateDecision(loop.getInitArgs()[number],
                                            visitedPredicates) &&
                       hasPredicateDecision(yield.getOperand(number),
                                            visitedPredicates);
              }
              if (auto loop = value.getDefiningOp<WhileOp>()) {
                auto result = mlir::dyn_cast<mlir::OpResult>(value);
                auto condition = mlir::cast<ConditionOp>(
                    loop.getConditionRegion().front().getTerminator());
                auto yield = mlir::cast<YieldOp>(
                    loop.getBodyRegion().front().getTerminator());
                unsigned number = result ? result.getResultNumber() : 0;
                return result && number < loop.getInitArgs().size() &&
                       number < condition.getValues().size() &&
                       number < yield.getNumOperands() &&
                       hasPredicateDecision(loop.getInitArgs()[number],
                                            visitedPredicates) &&
                       hasPredicateDecision(condition.getValues()[number],
                                            visitedPredicates) &&
                       hasPredicateDecision(yield.getOperand(number),
                                            visitedPredicates);
              }
              if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
                mlir::Operation *parent = argument.getOwner()->getParentOp();
                if (auto loop = mlir::dyn_cast_or_null<ForOp>(parent)) {
                  if (argument.getArgNumber() == 0)
                    return false;
                  unsigned number = argument.getArgNumber() - 1;
                  return number < loop.getInitArgs().size() &&
                         hasPredicateDecision(loop.getInitArgs()[number],
                                              visitedPredicates);
                }
                if (auto loop = mlir::dyn_cast_or_null<WhileOp>(parent)) {
                  unsigned number = argument.getArgNumber();
                  return number < loop.getInitArgs().size() &&
                         hasPredicateDecision(loop.getInitArgs()[number],
                                              visitedPredicates);
                }
              }
              return false;
            };
        if (hasPredicateDecision(predicate, {})) {
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
      if (auto load = mlir::dyn_cast<LoadOp>(operation)) {
        bool carriesValidity =
            mlir::isa<MaskedType>(load.getResult().getType());
        if (carriesValidity) {
          if (activityMode == VLAActivityMode::AllActive) {
            load.emitError(
                "masked VLA load requires a nontrivial logical predicate");
            return mlir::failure();
          }
          access.inactiveLane =
              VLAInactiveLaneRealization::ZeroCarrierWithLogicalValidity;
        }
      }
      if (relation == LaneRelation::Indexed) {
        access.indexedOffset =
            findIndexedOffset(pointer, decision.coordinate);
        if (!access.indexedOffset) {
          return operation->emitError(
              "indexed VLA memory requires one scalar base and one typed index vector");
        }
        auto indexCast = access.indexedOffset.getDefiningOp<CastOp>();
        access.indexedSEW =
            indexCast &&
                    elementType(indexCast.getInput().getType()).isUnsignedInteger(32)
                ? 32
                : static_cast<unsigned>(options.target.xlen);
      }
      decision.accesses.push_back(std::move(access));
      return mlir::success();
    };

    for (mlir::Operation *nested : physicalOperations) {
      if (isDotOwned(nested))
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
        auto storedRegion = mlir::dyn_cast<RegionType>(
            unwrapLogicalValidity(store.getValue().getType()));
        if (storedRegion && storedRegion.getShape().size() == 2) {
          int64_t vectors = storedRegion.getShape()[1];
          llvm::SmallVector<BlockIndexOp> axes =
              collectBlockAxes(store.getPointer());
          BlockIndexOp bundleAxis;
          for (BlockIndexOp axis : axes) {
            if (!haveSameLogicalExtent(store.getValue(), 1, axis.getResult(), 0))
              continue;
            if (bundleAxis && bundleAxis != axis) {
              bundleAxis = {};
              break;
            }
            bundleAxis = axis;
          }
          if (vectors <= 0 || !bundleAxis)
            return store.emitError(
                "VLA block-valued store requires one explicit trailing block axis");
          decision.accesses.back().bundleAxis = bundleAxis.getResult();
          decision.accesses.back().bundleVectors =
              static_cast<unsigned>(vectors);
        }
      }
    }

    struct SegmentFieldCandidate {
      VLAAccessDecision *access = nullptr;
      InterleavedFieldAddress address;
    };
    llvm::SmallVector<SegmentFieldCandidate> segmentFields;
    for (VLAAccessDecision &access : decision.accesses) {
      mlir::Value pointer;
      if (auto load = mlir::dyn_cast<LoadOp>(access.operation))
        pointer = load.getPointer();
      else if (auto store = mlir::dyn_cast<StoreOp>(access.operation))
        pointer = store.getPointer();
      if (!pointer || !access.elementType.isF32() ||
          access.activityMode != VLAActivityMode::AllActive ||
          access.memoryMode != VLAMemoryMode::Strided)
        continue;
      std::optional<InterleavedFieldAddress> address =
          matchInterleavedFieldAddress(pointer, decision.coordinate);
      if (address)
        segmentFields.push_back(SegmentFieldCandidate{&access, *address});
    }
    llvm::DenseSet<mlir::Operation *> pairedSegmentAccesses;
    auto hasInterveningEffect = [](mlir::Operation *lhs,
                                   mlir::Operation *rhs) {
      mlir::Operation *first = lhs->isBeforeInBlock(rhs) ? lhs : rhs;
      mlir::Operation *second = first == lhs ? rhs : lhs;
      for (mlir::Operation *operation = first->getNextNode();
           operation && operation != second;
           operation = operation->getNextNode())
        if (!mlir::isMemoryEffectFree(operation))
          return true;
      return false;
    };
    for (SegmentFieldCandidate &first : segmentFields) {
      if (pairedSegmentAccesses.contains(first.access->operation))
        continue;
      for (SegmentFieldCandidate &second : segmentFields) {
        if (&first == &second ||
            pairedSegmentAccesses.contains(second.access->operation) ||
            first.access->operation->getBlock() !=
                second.access->operation->getBlock() ||
            first.address.root != second.address.root ||
            !haveSameInvariantAddress(first.address, second.address) ||
            first.address.field == second.address.field ||
            mlir::isa<LoadOp>(first.access->operation) !=
                mlir::isa<LoadOp>(second.access->operation) ||
            hasInterveningEffect(first.access->operation,
                                 second.access->operation))
          continue;
        VLAAccessDecision *field0 =
            first.address.field == 0 ? first.access : second.access;
        VLAAccessDecision *field1 =
            first.address.field == 1 ? first.access : second.access;
        bool loadPair = mlir::isa<LoadOp>(field0->operation);
        mlir::Operation *emission = loadPair
                                        ? (field0->operation->isBeforeInBlock(
                                               field1->operation)
                                               ? field0->operation
                                               : field1->operation)
                                        : (field0->operation->isBeforeInBlock(
                                               field1->operation)
                                               ? field1->operation
                                               : field0->operation);
        VLASegment2Decision segment;
        segment.kind = loadPair ? VLASegment2AccessKind::Load
                                : VLASegment2AccessKind::Store;
        segment.field0 = field0->operation;
        segment.field1 = field1->operation;
        segment.emission = emission;
        segment.base = first.address.field == 0 ? first.address.pointer
                                                : second.address.pointer;
        decision.segment2.push_back(std::move(segment));
        field0->memoryMode = VLAMemoryMode::Segment2;
        field1->memoryMode = VLAMemoryMode::Segment2;
        pairedSegmentAccesses.insert(field0->operation);
        pairedSegmentAccesses.insert(field1->operation);
        break;
      }
    }

    for (mlir::Operation *nested : physicalOperations) {
      auto lookup = mlir::dyn_cast<LookupOp>(nested);
      if (!lookup)
        continue;
      auto table = resolveLocalBlockMemoryFact(lookup.getTable());
      auto result = mlir::dyn_cast<RegionType>(
          unwrapLogicalValidity(lookup.getResult().getType()));
      if (!table || table->semanticType.getShape() != llvm::ArrayRef<int64_t>({16}) ||
          !table->semanticType.getElementType().isF32() || !result ||
          !result.getElementType().isF32() ||
          !isRegionValue(lookup.getIndices().getType()) ||
          !elementType(lookup.getIndices().getType()).isUnsignedInteger(8) ||
          !isTrue(lookup.getWhere())) {
        lookup.emitError(
            "RVV VLA lookup requires an explicit all-active f32 table<16> and u8 indices");
        return mlir::failure();
      }
      if (options.target.vlenBits < 128) {
        lookup.emitError(
            "RVV VLA f32 table<16> lookup requires target VLEN of at least 128 bits");
        return mlir::failure();
      }
      VLALookupDecision lookupDecision;
      lookupDecision.operation = lookup.getOperation();
      lookupDecision.table = *table;
      lookupDecision.indices = lookup.getIndices();
      decision.lookups.push_back(std::move(lookupDecision));
      if (mlir::failed(markRematerializedBlockTrees(
              {lookup.getTable()}, lookup.getOperation())))
        return mlir::failure();
    }

    llvm::SmallVector<VLAValueLifetimeSnapshot> lifetimeSnapshots =
        vlaValueLifetimeSnapshots(op);
    unsigned maxEntityF32Vectors = 0;
    for (const VLAValueLifetimeSnapshot &snapshot : lifetimeSnapshots)
      maxEntityF32Vectors = std::max(maxEntityF32Vectors, snapshot.f32);

    for (mlir::Operation *nested : physicalOperations) {
      if (isDotOwned(nested))
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
      else {
        candidates = {8, 4, 2, 1};
        std::optional<int64_t> lower = integerConstantValue(op.getBegin());
        std::optional<int64_t> upper = integerConstantValue(op.getEnd());
        if (lower && upper && *upper > *lower) {
          const uint64_t lanes = static_cast<uint64_t>(*upper - *lower);
          llvm::sort(candidates, [&](unsigned lhs, unsigned rhs) {
            auto covers = [&](unsigned candidate) {
              return static_cast<uint64_t>(options.target.vlenBits) * candidate >=
                     lanes * 32;
            };
            bool lhsCovers = covers(lhs);
            bool rhsCovers = covers(rhs);
            if (lhsCovers != rhsCovers)
              return lhsCovers;
            return lhsCovers ? lhs < rhs : lhs > rhs;
          });
        }
      }
      std::optional<unsigned> selected;
      for (unsigned candidate : candidates) {
        if (candidate != 1 && candidate != 2 && candidate != 4 &&
            candidate != 8)
          continue;
        RVVVectorShape sourceShape = rvvShape(32, candidate);
        std::optional<RVVVectorShape> intermediateShape =
            rvvShapeForSameLanes(sourceShape, 16, options.target);
        std::optional<RVVVectorShape> resultShape =
            rvvShapeForSameLanes(sourceShape, 8, options.target);
        if (!options.target.supportsVectorShape(sourceShape.sew,
                                                sourceShape.lmulEighths) ||
            !intermediateShape || !resultShape)
          continue;
        unsigned narrowGroups = rvvRegisterGroups(sourceShape) +
                                rvvRegisterGroups(*intermediateShape) +
                                rvvRegisterGroups(*resultShape);
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
      RVVVectorShape sourceShape = rvvShape(32, sourceLMUL);
      decision.narrows.push_back(VLANarrowDecision{narrow.getOperation()});
      narrowPhysical.try_emplace(
          narrow.getOperation(),
          VLANarrowPhysicalConfig{
              sourceShape,
              *rvvShapeForSameLanes(sourceShape, 16, options.target),
              *rvvShapeForSameLanes(sourceShape, 8, options.target)});
    }

    for (mlir::Operation &nested : body.without_terminator()) {
      if (isDotOwned(&nested))
        continue;
      if (auto reduce = mlir::dyn_cast<ReduceOp>(nested)) {
        bool f32Reduction =
            elementType(reduce.getInput().getType()).isF32() &&
            reduce.getResult().getType().isF32();
        bool i8ToI32Reduction =
            elementType(reduce.getInput().getType()).isSignedInteger(8) &&
            reduce.getResult().getType().isSignedInteger(32);
        if (reduce.getAxis() != -1 ||
            (!f32Reduction && !i8ToI32Reduction) ||
            !isTrue(reduce.getWhere())) {
          reduce.emitError(
              "VLA reduction has no selected all-active f32 or i8-to-i32 realization");
          return mlir::failure();
        }
        VLAStateDecision state;
        state.operation = reduce.getOperation();
        state.elementType = reduce.getResult().getType();
        state.identity = reduce.getIdentity();
        bool maskedInput =
            mlir::isa<MaskedType>(reduce.getInput().getType());
        if (maskedInput)
          state.validity = VLAStateValidityRealization::ReductionIdentity;
        if (i8ToI32Reduction && reduce.getKind() == "add") {
          state.realization = VLAStateRealization::RVVI8AddReductionI32;
        } else if (reduce.getKind() == "add") {
          state.realization = VLAStateRealization::RVVAddReduction;
        } else if (reduce.getKind() == "max") {
          state.realization = VLAStateRealization::RVVMaxReduction;
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
        if (mlir::isa<MaskedType>(scan.getInput().getType()) ||
            !elementType(scan.getInput().getType()).isF32() ||
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
          if (mlir::isa<MaskedType>(summary.getInput().getType()))
            state.validity = VLAStateValidityRealization::NegativeInfinity;
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
          if (mlir::isa<MaskedType>(summary.getInput().getType()))
            state.validity = VLAStateValidityRealization::OnlineSoftmax;
          decision.states.push_back(std::move(state));
      }
    }
    unsigned reductionStateCount = llvm::count_if(
        decision.states, [](const VLAStateDecision &state) {
          return state.realization == VLAStateRealization::RVVAddReduction ||
                 state.realization == VLAStateRealization::RVVMaxReduction;
        });
    for (VLAStateDecision &state : decision.states) {
      if (state.realization != VLAStateRealization::RVVAddReduction &&
          state.realization != VLAStateRealization::RVVMaxReduction)
        continue;
      auto reduce = mlir::cast<ReduceOp>(state.operation);
      std::optional<VLAStatePlacement> placement =
          weft::riscv_internal::selectReductionStatePlacement(
              ReductionStatePlacementFacts{reduce.getOrder() == "relaxed",
                                           reductionStateCount},
              options.target, options.backend);
      if (!placement) {
        if (options.backend.reductionStatePlacement < 0 ||
            options.backend.reductionStatePlacement > 2)
          reduce.emitError(
              "VLA reduction state placement config must be 0, 1, or 2");
        else
          reduce.emitError(
              "ordered VLA reduction cannot use vector state placement");
        return mlir::failure();
      }
      state.placement = *placement;
    }
    bool hasF32RegionValue = llvm::any_of(
        physicalOperations, [](mlir::Operation *operation) {
          return llvm::any_of(operation->getResultTypes(), [](mlir::Type type) {
            return isRegionValue(type) && elementType(type).isF32();
          });
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
                 (source.isF32() && isF16(result)) ||
                 (source.isIndex() && result.isF32());
        });
    bool requiresF32M2Math = llvm::any_of(
        physicalOperations, [&](mlir::Operation *operation) {
          auto unary = mlir::dyn_cast<UnaryOp>(operation);
          return unary &&
                 llvm::any_of(decision.unaries,
                              [&](const VLAUnaryDecision &selected) {
                                return selected.operation == unary.getOperation();
                              });
        });
    VLACandidateFacts candidateFacts;
    candidateFacts.maxEntityF32Vectors = maxEntityF32Vectors;
    candidateFacts.requiresF32M2Math = requiresF32M2Math;
    candidateFacts.hasF32Division = llvm::any_of(
        physicalOperations, [](mlir::Operation *operation) {
          auto binary = mlir::dyn_cast<BinaryOp>(operation);
          return binary && binary.getKind() == "div" &&
                 isRegionValue(binary.getResult().getType()) &&
                 elementType(binary.getResult().getType()).isF32();
        });
    candidateFacts.hasIndexVector =
        !decision.indexBinaries.empty() || !decision.indexSelects.empty() ||
        llvm::any_of(lifetimeSnapshots,
                     [](const VLAValueLifetimeSnapshot &snapshot) {
                       return snapshot.index != 0;
                     }) ||
        llvm::any_of(decision.predicates,
                     [&](const VLAPredicateDecision &predicate) {
                       return predicate.realization ==
                                  VLAPredicateDecision::Realization::RVVVectorScalar &&
                              predicate.vectorSEW == options.target.xlen;
                     }) ||
        llvm::any_of(physicalOperations, [&](mlir::Operation *operation) {
          auto cast = mlir::dyn_cast<CastOp>(operation);
          return cast && isRegionValue(cast.getResult().getType()) &&
                 elementType(cast.getInput().getType()).isIndex() &&
                 elementType(cast.getResult().getType()).isF32();
        });
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
        candidateFacts.maxIndexedOffsetSEW = std::max(
            candidateFacts.maxIndexedOffsetSEW, access.indexedSEW);
      }
      if (access.elementType.isF32()) {
        candidateFacts.f32Loads += mlir::isa<LoadOp>(access.operation);
        candidateFacts.f32Stores += mlir::isa<StoreOp>(access.operation);
      }
    }
    candidateFacts.segmentLoadPairs = llvm::count_if(
        decision.segment2, [](const VLASegment2Decision &segment) {
          return segment.kind == VLASegment2AccessKind::Load;
        });
    candidateFacts.segmentStorePairs = llvm::count_if(
        decision.segment2, [](const VLASegment2Decision &segment) {
          return segment.kind == VLASegment2AccessKind::Store;
        });
    candidateFacts.hasF32TableLookup = !decision.lookups.empty();
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
    const unsigned dataSEW =
        !hasF32RegionValue && onlyF16Accesses && !hasFloatCast ? 16 : 32;
    auto legalLMUL = [&](unsigned value) {
      return options.target.supportsVectorShape(
          dataSEW, static_cast<int>(value * 8));
    };
    std::optional<unsigned> requiredLMUL;
    auto requireLMUL = [&](unsigned value, llvm::StringRef source) {
      if (!legalLMUL(value)) {
        op.emitError() << source << " requires an invalid LMUL " << value;
        return mlir::failure();
      }
      if (requiredLMUL && *requiredLMUL != value) {
        op.emitError() << "VLA entity LMUL requirements conflict: "
                       << *requiredLMUL << " versus " << value << " from "
                       << source;
        return mlir::failure();
      }
      requiredLMUL = value;
      return mlir::success();
    };
    for (const VLADotDecision &dot : decision.dots)
      if (mlir::failed(requireLMUL(dot.physical.lmul, "dot")))
        return mlir::failure();
    for (const VLANarrowDecision &narrow : decision.narrows)
      if (mlir::failed(requireLMUL(
              rvvIntegerLMUL(narrowPhysical.lookup(narrow.operation).sourceShape)
                  .value_or(0),
              "narrow")))
        return mlir::failure();
    if ((candidateFacts.requiresF32M2Math || candidateFacts.hasOnlineSummary) &&
        mlir::failed(requireLMUL(2, "f32m2 math")))
      return mlir::failure();
    if (options.backend.vlaLMUL != 0 &&
        mlir::failed(requireLMUL(
            static_cast<unsigned>(options.backend.vlaLMUL),
            "explicit backend config")))
      return mlir::failure();

    auto preferenceRank = [](unsigned candidate,
                             llvm::ArrayRef<unsigned> order) {
      auto found = llvm::find(order, candidate);
      return static_cast<unsigned>(std::distance(order.begin(), found));
    };
    llvm::SmallVector<unsigned> candidates =
        integerLMULCandidates(options.target, dataSEW);
    if (requiredLMUL) {
      candidates = {*requiredLMUL};
    } else {
      constexpr unsigned base128[] = {4, 2, 8, 1};
      constexpr unsigned base256[] = {4, 2, 1, 8};
      constexpr unsigned f16OrCast[] = {8, 4, 2, 1};
      constexpr unsigned scan[] = {1, 2, 4, 8};
      constexpr unsigned coordinateSummary[] = {4, 2, 1, 8};
      constexpr unsigned reduction[] = {8, 4, 2, 1};
      constexpr unsigned division[] = {2, 1, 4, 8};
      constexpr unsigned strided[] = {2, 4, 1, 8};
      llvm::sort(candidates, [&](unsigned lhs, unsigned rhs) {
        auto score = [&](unsigned candidate) {
          unsigned value = preferenceRank(
              candidate, options.target.vlenBits >= 256
                             ? llvm::ArrayRef<unsigned>(base256)
                             : llvm::ArrayRef<unsigned>(base128));
          if (dataSEW == 16 || hasFloatCast)
            value += 8 * preferenceRank(candidate, f16OrCast);
          if (candidateFacts.hasOrderedScan)
            value += 8 * preferenceRank(candidate, scan);
          if (candidateFacts.hasCoordinateSummary)
            value += 8 * preferenceRank(candidate, coordinateSummary);
          if (candidateFacts.hasReductionState)
            value += 8 * preferenceRank(candidate, reduction);
          if (candidateFacts.hasF32Division)
            value += 8 * preferenceRank(candidate, division);
          for (unsigned access = 0; access < candidateFacts.stridedAccesses;
               ++access)
            value += 8 * preferenceRank(candidate, strided);
          return value;
        };
        unsigned lhsScore = score(lhs);
        unsigned rhsScore = score(rhs);
        return lhsScore != rhsScore ? lhsScore < rhsScore : lhs > rhs;
      });
    }

    std::optional<VLAPlanCandidate> selected;
    for (unsigned candidate : candidates) {
      RVVVectorShape candidateDataShape = rvvShape(dataSEW, candidate);
      bool hasAffinePredicate = llvm::any_of(
          decision.predicates, [](const VLAPredicateDecision &predicate) {
            return predicate.realization ==
                   VLAPredicateDecision::Realization::RVVAffineIndexScalar;
          });
      if ((hasAffinePredicate || candidateFacts.hasIndexVector) &&
          (candidate * 8 * static_cast<unsigned>(options.target.xlen)) %
                  dataSEW !=
              0)
        continue;
      if ((hasAffinePredicate || candidateFacts.hasIndexVector) &&
          candidate * static_cast<unsigned>(options.target.xlen) / dataSEW > 8)
        continue;
      if ((hasAffinePredicate || candidateFacts.hasIndexVector) &&
          !options.target.supportsVectorShape(
              static_cast<unsigned>(options.target.xlen),
              static_cast<int>(candidate *
                               static_cast<unsigned>(options.target.xlen) /
                               dataSEW * 8)))
        continue;
      if (candidateFacts.indexedAccesses != 0 &&
          ((candidate * candidateFacts.maxIndexedOffsetSEW) % dataSEW != 0 ||
           candidate * candidateFacts.maxIndexedOffsetSEW / dataSEW > 8))
        continue;
      bool elementShapesLegal = llvm::all_of(
          decision.accesses, [&](const VLAAccessDecision &access) {
            unsigned sew = access.elementType.isF32() ? 32
                           : isF16(access.elementType) ? 16
                           : access.elementType.isUnsignedInteger(8) ? 8
                           : access.elementType.isUnsignedInteger(32) ? 32
                           : access.elementType.isSignedInteger(8) ? 8
                                                                  : 0;
            if (sew == 0 || (candidate * 8 * sew) % dataSEW != 0)
              return false;
            int lmulEighths =
                static_cast<int>(candidate * 8 * sew / dataSEW);
            return options.target.supportsVectorShape(sew, lmulEighths);
          });
      if (!elementShapesLegal)
        continue;
      if ((candidateFacts.segmentLoadPairs != 0 ||
           candidateFacts.segmentStorePairs != 0) &&
          !options.target.supportsSegmentVectorMemory(
              2, 32, static_cast<int>(candidate * 8)))
        continue;
      unsigned carriedStateGroups = 0;
      unsigned transientStateGroups = 0;
      for (const VLAStateDecision &state : decision.states) {
        if (state.realization == VLAStateRealization::RVVInclusiveAddScan)
          transientStateGroups =
              std::max(transientStateGroups, 3 * candidate + 1);
        else if (state.realization ==
                 VLAStateRealization::RVVSegmentedInclusiveAddScan)
          transientStateGroups =
              std::max(transientStateGroups, 4 * candidate + 2);
        else if (state.realization == VLAStateRealization::RVVAddReduction ||
                 state.realization == VLAStateRealization::RVVMaxReduction) {
          if (state.placement == VLAStatePlacement::VectorCarry) {
            carriedStateGroups += candidate;
            transientStateGroups =
                std::max(transientStateGroups, std::max(candidate, 2u));
          } else {
            transientStateGroups =
                std::max(transientStateGroups, candidate + 2);
          }
        } else if (state.realization ==
                   VLAStateRealization::RVVI8AddReductionI32)
          transientStateGroups =
              std::max(transientStateGroups,
                       std::max(1u, candidate / 4) + 1);
        else if (state.realization ==
                   VLAStateRealization::RVVArgMaxSummary)
          transientStateGroups =
              std::max(transientStateGroups, candidate + 2);
        else if (state.realization ==
                 VLAStateRealization::RVVOnlineSoftmaxSummary)
          transientStateGroups =
              std::max(transientStateGroups, 3 * candidate + 2);
      }
      unsigned stateGroups = carriedStateGroups + transientStateGroups;
      unsigned primitiveGroups = stateGroups;
      for (const VLANarrowDecision &narrow : decision.narrows) {
        const VLANarrowPhysicalConfig &physical =
            narrowPhysical.lookup(narrow.operation);
        primitiveGroups = std::max(
            primitiveGroups,
            rvvRegisterGroups(physical.sourceShape) +
                rvvRegisterGroups(physical.intermediateShape) +
                rvvRegisterGroups(physical.resultShape));
      }
      unsigned indexGroups = 0;
      if (candidateFacts.indexedAccesses != 0) {
        unsigned indexLMUL =
            candidate * candidateFacts.maxIndexedOffsetSEW / dataSEW;
        indexGroups = indexLMUL;
      }
      auto groupsFor = [&](unsigned sew, unsigned count) {
        if (count == 0)
          return 0u;
        unsigned scaled = candidate * 8 * sew;
        if (scaled % dataSEW != 0)
          return static_cast<unsigned>(options.target.vectorRegisters);
        int lmulEighths = static_cast<int>(scaled / dataSEW);
        if (!options.target.supportsVectorShape(sew, lmulEighths))
          return static_cast<unsigned>(options.target.vectorRegisters);
        return count * rvvRegisterGroups(RVVVectorShape{sew, lmulEighths});
      };
      unsigned valueGroups = 0;
      unsigned predicateGroups = 0;
      for (const VLAValueLifetimeSnapshot &snapshot : lifetimeSnapshots) {
        unsigned snapshotGroups =
            groupsFor(32, snapshot.f32) + groupsFor(16, snapshot.f16) +
            groupsFor(8, snapshot.byte) + groupsFor(32, snapshot.u32) +
            groupsFor(static_cast<unsigned>(options.target.xlen),
                      snapshot.index) +
            snapshot.mask;
        valueGroups = std::max(valueGroups, snapshotGroups);
        predicateGroups = std::max(predicateGroups, snapshot.mask);
      }
      unsigned memoryGroups = indexGroups;
      memoryGroups +=
          2 * candidate *
          std::max(candidateFacts.segmentLoadPairs,
                   candidateFacts.segmentStorePairs);
      unsigned lookupGroups = 0;
      bool lookupShapesLegal = true;
      for (const VLALookupDecision &lookup : decision.lookups) {
        std::optional<RVVVectorShape> codeShape =
            rvvShapeForSameLanes(candidateDataShape, 8, options.target);
        std::optional<RVVVectorShape> index16Shape =
            rvvShapeForSameLanes(candidateDataShape, 16, options.target);
        std::optional<RVVVectorShape> index32Shape =
            rvvShapeForSameLanes(candidateDataShape, 32, options.target);
        if (!codeShape || !index16Shape || !index32Shape) {
          lookupShapesLegal = false;
          break;
        }
        unsigned conversionGroups =
            rvvRegisterGroups(*codeShape) +
            rvvRegisterGroups(*index16Shape) +
            rvvRegisterGroups(*index32Shape);
        unsigned gatherGroups = 3 * rvvRegisterGroups(*index32Shape);
        lookupGroups =
            std::max(lookupGroups, std::max(conversionGroups, gatherGroups));
      }
      if (!lookupShapesLegal)
        continue;
      unsigned sharedGroups = valueGroups + memoryGroups;
      unsigned requiredGroups =
          std::max(sharedGroups,
                   std::max(primitiveGroups, lookupGroups) + memoryGroups);
      if (requiredGroups + 1 <
          static_cast<unsigned>(options.target.vectorRegisters)) {
        VLAPlanCandidate plan;
        plan.dataShape = candidateDataShape;
        if (hasAffinePredicate || candidateFacts.hasIndexVector)
          plan.indexShape =
              rvvShapeForSameLanes(candidateDataShape, options.target.xlen,
                                   options.target)
                  .value_or(RVVVectorShape{});
        plan.maskRatio = rvvMaskRatio(plan.dataShape).value_or(0);
        if (plan.maskRatio == 0)
          continue;
        plan.resources.architecturalGroups = options.target.vectorRegisters;
        plan.resources.valueGroups = valueGroups;
        plan.resources.memoryGroups = memoryGroups;
        plan.resources.predicateGroups = predicateGroups;
        plan.resources.stateGroups = stateGroups;
        plan.resources.primitiveGroups = std::max(primitiveGroups, lookupGroups);
        plan.resources.peakGroups = requiredGroups + 1;
        selected = plan;
        break;
      }
    }
    if (!selected) {
      op.emitError() << "VLA has no legal LMUL candidate for "
                     << maxEntityF32Vectors
                     << " simultaneously live f32 values and its typed memory/state resources";
      return mlir::failure();
    }
    entity.vlaDataShape = selected->dataShape;
    entity.vlaIndexShape = selected->indexShape;
    entity.vlaMaskRatio = selected->maskRatio;
    entity.resources = selected->resources;

    auto regionShape = [&](mlir::Type type) -> RVVVectorShape {
      if (!isRegionValue(type))
        return {};
      mlir::Type element = elementType(type);
      unsigned sew = element.isF32() ? 32
                     : isF16(element) ? 16
                     : (element.isSignedInteger(8) ||
                        element.isUnsignedInteger(8))
                         ? 8
                     : element.isUnsignedInteger(32) ? 32
                                                     : 0;
      return sew == 0
                 ? RVVVectorShape{}
                 : rvvShapeForSameLanes(entity.vlaDataShape, sew,
                                        options.target)
                       .value_or(RVVVectorShape{});
    };
    for (mlir::Operation *operation : physicalOperations) {
      for (mlir::Value operand : operation->getOperands())
        recordPhysicalValue(entity, operand, regionShape(operand.getType()));
      for (mlir::Value result : operation->getResults())
        recordPhysicalValue(entity, result, regionShape(result.getType()));
      if (auto loop = mlir::dyn_cast<ForOp>(operation))
        for (mlir::BlockArgument argument :
             loop.getBody().front().getArguments().drop_front())
          recordPhysicalValue(entity, argument, regionShape(argument.getType()));
      if (auto loop = mlir::dyn_cast<WhileOp>(operation))
        for (mlir::Region *region :
             {&loop.getConditionRegion(), &loop.getBodyRegion()})
          for (mlir::BlockArgument argument : region->front().getArguments())
            recordPhysicalValue(entity, argument,
                                regionShape(argument.getType()));
    }

    for (mlir::Operation *operation : physicalOperations) {
      auto cast = mlir::dyn_cast<CastOp>(operation);
      if (!cast || !isRegionValue(cast.getResult().getType()))
        continue;
      mlir::Type source = elementType(cast.getInput().getType());
      mlir::Type result = elementType(cast.getResult().getType());
      VLACastDecision selectedCast;
      selectedCast.operation = cast.getOperation();
      if (cast.getInput().getType() == cast.getResult().getType()) {
        selectedCast.realization = VLACastRealization::RVVIdentity;
        RVVVectorShape shape = regionShape(cast.getInput().getType());
        if (!shape) {
          cast.emitError("VLA identity cast has no selected RVV shape");
          return mlir::failure();
        }
        recordPhysicalValue(entity, cast.getInput(), shape);
        recordPhysicalValue(entity, cast.getResult(), shape);
        recordPhysicalHandoff(entity, cast.getOperation(), cast.getInput(),
                              PhysicalHandoff::Share, shape, shape);
        decision.casts.push_back(selectedCast);
        continue;
      }
      unsigned sourceSEW = 0;
      unsigned resultSEW = 0;
      if (isF16(source) && result.isF32()) {
        selectedCast.realization = VLACastRealization::RVVWidenF16ToF32;
        sourceSEW = 16;
        resultSEW = 32;
      } else if (source.isF32() && isF16(result)) {
        selectedCast.realization = VLACastRealization::RVVNarrowF32ToF16;
        sourceSEW = 32;
        resultSEW = 16;
      } else if (source.isUnsignedInteger(32) && result.isIndex()) {
        selectedCast.realization =
            VLACastRealization::RVVZeroExtendU32ToIndex;
        sourceSEW = 32;
        resultSEW = 32;
      } else if (source.isIndex() && result.isF32()) {
        selectedCast.realization = VLACastRealization::RVVIndexToF32;
        sourceSEW = options.target.xlen;
        resultSEW = 32;
      } else {
        cast.emitError("VLA cast has no selected RVV conversion");
        return mlir::failure();
      }
      RVVVectorShape sourceShape =
          rvvShapeForSameLanes(entity.vlaDataShape, sourceSEW, options.target)
              .value_or(RVVVectorShape{});
      RVVVectorShape resultShape =
          rvvShapeForSameLanes(entity.vlaDataShape, resultSEW, options.target)
              .value_or(RVVVectorShape{});
      if (!sourceShape || !resultShape) {
        cast.emitError("VLA cast exceeds the legal RVV LMUL range");
        return mlir::failure();
      }
      recordPhysicalValue(entity, cast.getInput(), sourceShape);
      recordPhysicalValue(entity, cast.getResult(), resultShape);
      recordPhysicalHandoff(entity, cast.getOperation(), cast.getInput(),
                            PhysicalHandoff::Convert, sourceShape,
                            resultShape);
      decision.casts.push_back(selectedCast);
    }

    if ((!decision.indexBinaries.empty() || !decision.indexSelects.empty()) &&
        !entity.vlaIndexShape) {
      op.emitError("VLA index operations have no selected native index shape");
      return mlir::failure();
    }
    auto recordIndexOperand = [&](mlir::Operation *consumer,
                                  mlir::Value operand) {
      if (!isRegionValue(operand.getType()))
        return;
      recordPhysicalValue(entity, operand, entity.vlaIndexShape);
      LaneRelation relation =
          classifyLaneRelation(operand, decision.coordinate);
      recordPhysicalHandoff(
          entity, consumer, operand,
          relation == LaneRelation::UnitStride ||
                  relation == LaneRelation::Strided
              ? PhysicalHandoff::Rematerialize
              : PhysicalHandoff::Share,
          entity.vlaIndexShape, entity.vlaIndexShape);
    };
    for (const VLAIndexBinaryDecision &index : decision.indexBinaries) {
      auto binary = mlir::cast<BinaryOp>(index.operation);
      recordIndexOperand(index.operation, binary.getLhs());
      recordIndexOperand(index.operation, binary.getRhs());
      recordPhysicalValue(entity, binary.getResult(), entity.vlaIndexShape);
    }
    for (const VLAIndexSelectDecision &index : decision.indexSelects) {
      auto select = mlir::cast<SelectOp>(index.operation);
      recordIndexOperand(index.operation, select.getTrueValue());
      recordIndexOperand(index.operation, select.getFalseValue());
      recordPhysicalValue(entity, select.getResult(), entity.vlaIndexShape);
      if (index.trueScalar || index.falseScalar)
        recordPhysicalTemporary(entity, index.operation, entity.vlaIndexShape);
    }
    for (const VLAUnaryDecision &unary : decision.unaries) {
      auto operation = mlir::cast<UnaryOp>(unary.operation);
      RVVVectorShape shape =
          rvvShapeForSameLanes(entity.vlaDataShape, 32, options.target)
              .value_or(RVVVectorShape{});
      recordPhysicalValue(entity, operation.getInput(), shape);
      recordPhysicalValue(entity, operation.getResult(), shape);
      recordPhysicalHandoff(entity, unary.operation, operation.getInput(),
                            PhysicalHandoff::Share, shape, shape);
      if (unary.realization == VLAUnaryRealization::RVVF32M2ScalarLibmSin ||
          unary.realization == VLAUnaryRealization::RVVF32M2ScalarLibmCos)
        recordPhysicalTemporary(entity, unary.operation, shape);
    }

    for (mlir::Operation *operation : physicalOperations) {
      auto add = mlir::dyn_cast<BinaryOp>(operation);
      if (!add || add.getKind() != "add" ||
          !isF16(elementType(add.getResult().getType())))
        continue;
      BinaryOp multiply = add.getLhs().getDefiningOp<BinaryOp>();
      mlir::Value accumulator = add.getRhs();
      if (!multiply || multiply.getKind() != "mul") {
        multiply = add.getRhs().getDefiningOp<BinaryOp>();
        accumulator = add.getLhs();
      }
      if (!multiply || multiply.getKind() != "mul" ||
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
      llvm::DenseSet<mlir::Operation *> absorbed{multiply.getOperation()};
      retainOnlyPrivateDefinitions(absorbed, add.getOperation());
      mlir::Operation *absorbedProducer =
          absorbed.contains(multiply.getOperation()) ? multiply.getOperation()
                                                     : nullptr;
      decision.binaries.push_back(VLABinaryDecision{
          add.getOperation(), VLABinaryRealization::RVVF16FusedMultiplyAdd,
          absorbedProducer, accumulator, vector, scalar});
      RVVVectorShape fmaShape =
          rvvShapeForSameLanes(entity.vlaDataShape, 16, options.target)
              .value_or(RVVVectorShape{});
      recordPhysicalValue(entity, accumulator, fmaShape);
      recordPhysicalValue(entity, vector, fmaShape);
      recordPhysicalValue(entity, add.getResult(), fmaShape);
    }

    bool hasAffinePredicate = llvm::any_of(
        decision.predicates, [](const VLAPredicateDecision &predicate) {
          return predicate.realization ==
                 VLAPredicateDecision::Realization::RVVAffineIndexScalar;
        });
    if (hasAffinePredicate) {
      if (!entity.vlaIndexShape ||
          rvvIntegerLMUL(entity.vlaIndexShape).value_or(9) > 8) {
        op.emitError("selected VLA data LMUL has no legal index-vector LMUL");
        return mlir::failure();
      }
    }

    for (VLAAccessDecision &access : decision.accesses) {
      if (!access.elementType.isF32() && !isF16(access.elementType) &&
          !access.elementType.isSignedInteger(8) &&
          !access.elementType.isUnsignedInteger(8) &&
          !access.elementType.isUnsignedInteger(32)) {
        access.operation->emitError(
            "VLA memory element has no selected RVV vector shape");
        return mlir::failure();
      }
      if (isF16(access.elementType) && !options.target.hasVectorF16) {
        access.operation->emitError(
            "f16 VLA memory requires the target vector-f16 extension");
        return mlir::failure();
      }
      unsigned elementSEW = access.elementType.isF32() ? 32
                            : isF16(access.elementType) ? 16
                            : (access.elementType.isSignedInteger(8) ||
                               access.elementType.isUnsignedInteger(8))
                                ? 8
                                : 32;
      RVVVectorShape elementShape =
          rvvShapeForSameLanes(entity.vlaDataShape, elementSEW, options.target)
              .value_or(RVVVectorShape{});
      if (!elementShape) {
        access.operation->emitError(
            "VLA memory element exceeds the legal RVV LMUL range");
        return mlir::failure();
      }
      if (auto load = mlir::dyn_cast<LoadOp>(access.operation)) {
        recordPhysicalValue(entity, load.getResult(), elementShape);
        recordPhysicalHandoff(entity, access.operation, load.getResult(),
                              PhysicalHandoff::Share, elementShape,
                              elementShape);
      } else if (auto store = mlir::dyn_cast<StoreOp>(access.operation)) {
        recordPhysicalValue(entity, store.getValue(), elementShape);
        recordPhysicalHandoff(entity, access.operation, store.getValue(),
                              access.storeValueMode ==
                                      VLAStoreValueMode::ScalarBroadcast
                                  ? PhysicalHandoff::Rematerialize
                                  : PhysicalHandoff::Share,
                              elementShape, elementShape);
      }
      if (access.memoryMode == VLAMemoryMode::Indexed) {
        mlir::Type indexElement = elementType(access.indexedOffset.getType());
        if (!indexElement.isIndex()) {
          access.operation->emitError(
              "indexed VLA memory offset is not an index region");
          return mlir::failure();
        }
        auto indexValue = llvm::find_if(
            entity.values, [&](const PhysicalValueDecision &value) {
              return value.value == access.indexedOffset;
            });
        if (indexValue == entity.values.end()) {
          access.operation->emitError(
              "indexed VLA memory has no selected index value shape");
          return mlir::failure();
        }
        RVVVectorShape indexShape = indexValue->shape;
        if (!options.target.supportsIndexedVectorMemory(
                elementShape.sew, elementShape.lmulEighths,
                indexShape.sew, indexShape.lmulEighths)) {
          access.operation->emitError(
              "indexed VLA memory shapes are illegal for the target profile");
          return mlir::failure();
        }
        recordPhysicalHandoff(entity, access.operation,
                              access.indexedOffset, PhysicalHandoff::Share,
                              indexShape, indexShape);
      }
    }

    for (VLALookupDecision &lookup : decision.lookups) {
      auto operation = mlir::cast<LookupOp>(lookup.operation);
      RVVVectorShape codeShape =
          rvvShapeForSameLanes(entity.vlaDataShape, 8, options.target)
              .value_or(RVVVectorShape{});
      RVVVectorShape index16Shape =
          rvvShapeForSameLanes(entity.vlaDataShape, 16, options.target)
              .value_or(RVVVectorShape{});
      RVVVectorShape index32Shape =
          rvvShapeForSameLanes(entity.vlaDataShape, 32, options.target)
              .value_or(RVVVectorShape{});
      if (!codeShape || !index16Shape || !index32Shape) {
        lookup.operation->emitError(
            "RVV VLA lookup has no legal typed value shapes");
        return mlir::failure();
      }
      recordPhysicalValue(entity, operation.getIndices(), codeShape);
      recordPhysicalValue(entity, operation.getTable(), index32Shape);
      recordPhysicalValue(entity, operation.getResult(), index32Shape);
      recordPhysicalTemporary(entity, lookup.operation, index16Shape);
      recordPhysicalTemporary(entity, lookup.operation, index32Shape);
      recordPhysicalHandoff(entity, lookup.operation, operation.getIndices(),
                            PhysicalHandoff::Convert, codeShape, index32Shape);
      recordPhysicalHandoff(entity, lookup.operation, operation.getTable(),
                            PhysicalHandoff::Reload, index32Shape,
                            index32Shape);
    }

    for (VLAPredicateDecision &predicate : decision.predicates) {
      if (predicate.realization !=
          VLAPredicateDecision::Realization::RVVVectorScalar)
        continue;
      RVVVectorShape predicateShape =
          rvvShapeForSameLanes(entity.vlaDataShape, predicate.vectorSEW,
                               options.target)
              .value_or(RVVVectorShape{});
      if (!predicateShape) {
        predicate.operation->emitError(
            "typed VLA predicate has no legal selected vector shape");
        return mlir::failure();
      }
      recordPhysicalValue(entity, predicate.coordinate, predicateShape);
      recordPhysicalHandoff(entity, predicate.operation, predicate.coordinate,
                            PhysicalHandoff::Share, predicateShape,
                            predicateShape);
    }

    for (VLAStateDecision &state : decision.states) {
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
        if (!llvm::any_of(entity.values,
                          [&](const PhysicalValueDecision &value) {
                            return value.value == state.segmentVector;
                          })) {
          state.operation->emitError(
              "segmented scan has no physical segment value shape");
          return mlir::failure();
        }
      }
      if (state.realization ==
              VLAStateRealization::RVVOnlineSoftmaxSummary &&
          rvvIntegerLMUL(rvvShapeForSameLanes(entity.vlaDataShape, 32,
                                              options.target)
                             .value_or(RVVVectorShape{})) != 2) {
        state.operation->emitError(
            "online softmax summary requires the selected f32m2 exp realization");
        return mlir::failure();
      }
      unsigned stateInputSEW =
          state.realization == VLAStateRealization::RVVI8AddReductionI32 ? 8
                                                                         : 32;
      RVVVectorShape stateShape =
          rvvShapeForSameLanes(entity.vlaDataShape, stateInputSEW,
                               options.target)
              .value_or(RVVVectorShape{});
      recordPhysicalValue(entity, state.operation->getOperand(0), stateShape);
      recordPhysicalHandoff(
          entity, state.operation, state.identity,
          state.placement == VLAStatePlacement::VectorCarry
              ? PhysicalHandoff::Share
              : PhysicalHandoff::Rematerialize,
          stateShape, stateShape);
    }
    for (const VLANarrowDecision &narrow : decision.narrows) {
      const VLANarrowPhysicalConfig &physical =
          narrowPhysical.lookup(narrow.operation);
      auto narrowOp = mlir::cast<NarrowOp>(narrow.operation);
      recordPhysicalValue(entity, narrowOp.getInput(), physical.sourceShape);
      recordPhysicalValue(entity, narrowOp.getResult(), physical.resultShape);
      recordPhysicalTemporary(entity, narrow.operation,
                              physical.intermediateShape);
      recordPhysicalHandoff(entity, narrow.operation, narrowOp.getInput(),
                            PhysicalHandoff::Convert, physical.sourceShape,
                            physical.resultShape);
    }
    for (const VLADotDecision &dot : decision.dots) {
      const F32DotPhysicalConfig &physical = dot.physical;
      unsigned dotGroups =
          dot.rowTile * physical.lmul +
          (dot.realization ==
                   VLADotRealization::RVVF32FreeAxisMicrotile
               ? 2
               : 1) *
              physical.kUnroll * physical.lmul;
      entity.resources.primitiveGroups =
          std::max(entity.resources.primitiveGroups, dotGroups);
      unsigned handoffGroups = entity.resources.memoryGroups +
                               entity.resources.predicateGroups;
      entity.resources.peakGroups =
          std::max(entity.resources.peakGroups,
                   dotGroups + handoffGroups + 1);
      RVVVectorShape dotShape = rvvShape(32, physical.lmul);
      recordPhysicalValue(entity, dot.init, dotShape);
      if (auto operation = mlir::dyn_cast<DotOp>(dot.operation))
        recordPhysicalValue(entity, operation.getResult(), dotShape);
      recordPhysicalHandoff(entity, dot.operation, dot.init,
                            dot.initRealization ==
                                    VLADotInitRealization::MaterializedRegion
                                ? PhysicalHandoff::Share
                                : PhysicalHandoff::Rematerialize,
                            dotShape, dotShape);
    }
    if (entity.resources.peakGroups >=
        static_cast<unsigned>(options.target.vectorRegisters)) {
      op.emitError(
          "VLA has no joint physical plan within the target register budget");
      return mlir::failure();
    }
    return planned;
  }

  mlir::LogicalResult emitOperation(mlir::Operation *operation) {
    if (consumed.contains(operation))
      return mlir::success();
    if (const VLADotDecision *decision =
            findVLADotDecision(operation))
      return emitVLADot(*decision);
    if (isVLADotAbsorbed(operation))
      return mlir::success();
    if (auto op = mlir::dyn_cast<ReduceOp>(operation))
      return emitReduce(op);
    if (auto op = mlir::dyn_cast<SortIndicesOp>(operation))
      return emitSortIndices(op);
    if (auto op = mlir::dyn_cast<ScanOp>(operation))
      return op.emitError("scan must be lowered by its enclosing VLA region");
    if (auto op = mlir::dyn_cast<GroupedAffineI4I8DotOp>(operation))
      return emitGroupedAffineI4I8Dot(op);
    if (auto op = mlir::dyn_cast<AffineI4I8DotOp>(operation))
      return emitAffineI4I8Dot(op);
    if (auto op = mlir::dyn_cast<SignBitI8DotOp>(operation))
      return emitSignBitI8Dot(op);
    if (auto op = mlir::dyn_cast<E2M1E8M0I8DotOp>(operation))
      return emitE2M1E8M0I8Dot(op);
    if (auto op = mlir::dyn_cast<IQ2SI8DotOp>(operation))
      return emitQuantCodebookI8Dot(op);
    if (auto op = mlir::dyn_cast<IQ3SI8DotOp>(operation))
      return emitQuantCodebookI8Dot(op);
    if (auto op = mlir::dyn_cast<IQ1MI8DotOp>(operation))
      return emitQuantCodebookI8Dot(op);
    if (auto op = mlir::dyn_cast<Q6KI8DotOp>(operation))
      return emitQuantCodebookI8Dot(op);
    if (auto op = mlir::dyn_cast<SymmetricI4I8DotOp>(operation))
      return emitSymmetricI4I8Dot(op);
    if (auto op = mlir::dyn_cast<DotOp>(operation))
      if (!isRegionValue(op.getResult().getType()))
        return emitLocalF32Dot(op);
    if (auto op = mlir::dyn_cast<MatmulOp>(operation))
      return emitF16Matmul(op);
    if (auto op = mlir::dyn_cast<FullOp>(operation)) {
      if (auto block = mlir::dyn_cast<BlockType>(op.getResult().getType());
          block && block.getElementType().isF32() &&
          materializedBlockElements.contains(op.getOperation()))
        return emitMaterializedBlockFull(op);
    }
    if (auto op = mlir::dyn_cast<ForOp>(operation))
      return emitFor(op);
    if (auto op = mlir::dyn_cast<WhileOp>(operation))
      return emitWhile(op);
    if (auto op = mlir::dyn_cast<IfOp>(operation))
      return emitIf(op);
    if (auto op = mlir::dyn_cast<StoreOp>(operation))
      if (auto lowered = tryEmitMaterializedF32BlockStore(op))
        return *lowered;
    if (physicalPlan.materializedF32Pointwise.contains(operation))
      return emitMaterializedF32Pointwise(operation);
    if (auto op = mlir::dyn_cast<LookupOp>(operation))
      if (inVLA)
        return emitVLALookup(op);
    if (auto op = mlir::dyn_cast<ExpandDimsOp>(operation)) {
      if (inVLA && isRegionValue(op.getResult().getType())) {
        CValue input = require(op.getInput());
        if (input.spelling.empty())
          return op.emitError("VLA singleton-axis view input is unavailable");
        input.type = op.getResult().getType();
        values[op.getResult()] = std::move(input);
        return mlir::success();
      }
    }
    if (inVLA) {
      if (auto op = mlir::dyn_cast<LoadOp>(operation))
        return emitLoad(op);
      if (auto op = mlir::dyn_cast<StoreOp>(operation))
        return emitStore(op);
    }
    if (auto op = mlir::dyn_cast<LoadF16LEOp>(operation))
      return emitScalarLoadF16LE(op);
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
    if (mlir::isa<StorageOp>(operation))
      return mlir::success();
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
    auto found = physicalPlan.sortIndices.find(op.getOperation());
    if (found == physicalPlan.sortIndices.end())
      return op.emitError("sort_indices has no physical decision");
    if (mlir::failed(requireEntityPlan(op.getOperation(), found->second)))
      return mlir::failure();
    const SortIndicesDecision &decision = found->second.realization;
    if (decision.realization != SortIndicesRealization::StableF32Radix ||
        (decision.radixBits != 8 && decision.radixBits != 11) ||
        decision.passes !=
            (32 + decision.radixBits - 1) / decision.radixBits)
      return op.emitError("sort_indices decision has no intrinsic-C spelling");
    int64_t expectedPrivateElements =
        2 * (int64_t{1} << decision.radixBits);
    auto privateStorage = llvm::find_if(
        found->second.entity.storages,
        [&](const PhysicalStorageDecision &storage) {
          return !storage.value &&
                 storage.elements == expectedPrivateElements;
        });
    if (privateStorage == found->second.entity.storages.end())
      return op.emitError(
          "sort_indices private histogram storage decision is missing");
    CValue input = require(op.getInput());
    CValue outputValue = require(op.getOutput());
    CValue scratchValue = require(op.getScratch());
    CValue extent = require(op.getExtent());
    if (input.kind != CValueKind::Pointer ||
        outputValue.kind != CValueKind::Pointer ||
        scratchValue.kind != CValueKind::Pointer || input.spelling.empty() ||
        outputValue.spelling.empty() || scratchValue.spelling.empty() ||
        extent.kind != CValueKind::Scalar || extent.spelling.empty())
      return op.emitError("sort_indices operands are unavailable");

    std::string scratch = scratchValue.spelling;
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
    if (!block || block.getShape().empty() || block.getShape().size() > 2 ||
        !block.getElementType().isF32() || fill.kind != CValueKind::Scalar ||
        fill.spelling.empty() ||
        llvm::any_of(block.getShape(), [](int64_t extent) {
          return extent == 0 || extent < -1;
        }) ||
        op.getAxes().size() != block.getShape().size())
      return op.emitError(
          "materialized block full requires a rank-one or rank-two f32 block");
    auto selected = materializedBlockElements.find(op.getOperation());
    if (selected == materializedBlockElements.end() || selected->second <= 0)
      return op.emitError(
          "materialized block has no bounded physical storage decision");
    std::string elementCount = std::to_string(selected->second);
    std::string storage = fresh("block_storage");
    std::string lane = fresh("block_init");
    line("float " + storage + "[" + elementCount + "];");
    line("for (size_t " + lane + " = 0; " + lane + " < " + elementCount +
         "; ++" + lane + ")");
    ++indent;
    line(storage + "[" + lane + "] = " + fill.spelling + ";");
    --indent;
    values[op.getResult()] = CValue{op.getResult().getType(),
                                    CValueKind::F32BlockStorage, storage};
    loweredBlockOps.insert(op.getOperation());
    return mlir::success();
  }

  mlir::LogicalResult copyF32Block(mlir::Operation *owner,
                                   const CValue &destination,
                                   const CValue &source, int64_t elements,
                                   llvm::StringRef prefix) {
    if (destination.kind != CValueKind::F32BlockStorage ||
        source.kind != CValueKind::F32BlockStorage ||
        destination.spelling.empty() || source.spelling.empty() ||
        elements <= 0)
      return owner->emitError("block control handoff has no f32 storage");
    if (destination.spelling == source.spelling)
      return mlir::success();
    std::string lane = fresh(prefix);
    line("for (size_t " + lane + " = 0; " + lane + " < " +
         std::to_string(elements) + "; ++" + lane + ")");
    ++indent;
    line(destination.spelling + "[" + lane + "] = " + source.spelling +
         "[" + lane + "];");
    --indent;
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
      if (mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
              result.getType())) {
        CValue value{result.getType(), CValueKind::Scalar,
                     result.use_empty() ? std::string{} : fresh("if_result")};
        if (!value.spelling.empty())
          line(scalarCType(result.getType()) + " " + value.spelling + ";");
        results.push_back(std::move(value));
        continue;
      }
      if (auto pointer = mlir::dyn_cast<PtrType>(result.getType())) {
        CValue value{result.getType(), CValueKind::Pointer,
                     result.use_empty() ? std::string{} : fresh("if_pointer")};
        if (!value.spelling.empty())
          line(pointerCType(pointer) + " " + value.spelling + ";");
        results.push_back(std::move(value));
        continue;
      }
      if (inVLA && isRegionValue(result.getType()) &&
          elementType(result.getType()).isF32()) {
        std::optional<unsigned> lmul = physicalValueLMUL(result);
        if (!lmul)
          return op.emitError(
              "RISC-V conditional VLA result has no physical value shape");
        CValue value{result.getType(), CValueKind::F32Vector,
                     result.use_empty() ? std::string{} : fresh("if_vector")};
        value.vectorSEW = 32;
        value.vectorLMUL = *lmul;
        if (!value.spelling.empty())
          line("vfloat32m" + std::to_string(*lmul) + "_t " + value.spelling +
               ";");
        results.push_back(std::move(value));
        continue;
      }
      if (inVLA && isRegionValue(result.getType()) &&
          elementType(result.getType()).isInteger(1)) {
        if (!activePhysicalEntity || activePhysicalEntity->vlaMaskRatio == 0)
          return op.emitError(
              "RISC-V conditional VLA mask has no physical shape");
        CValue value{result.getType(), CValueKind::Mask,
                     result.use_empty() ? std::string{} : fresh("if_mask")};
        if (!value.spelling.empty())
          line("vbool" +
               std::to_string(activePhysicalEntity->vlaMaskRatio) + "_t " +
               value.spelling + ";");
        results.push_back(std::move(value));
        continue;
      }
      auto selected = controlBlockElements.find(result);
      if (selected == controlBlockElements.end())
        return op.emitError(
            "RISC-V conditional block result has no physical storage decision");
      CValue value{result.getType(), CValueKind::F32BlockStorage,
                   result.use_empty() ? std::string{} : fresh("if_block")};
      if (!value.spelling.empty())
        line("float " + value.spelling + "[" +
             std::to_string(selected->second) + "];");
      results.push_back(std::move(value));
    }

    auto emitBranch = [&](mlir::Region &region) {
      mlir::Block &body = region.front();
      for (mlir::Operation &nested : body.without_terminator())
        if (mlir::failed(emitOperation(&nested)))
          return mlir::failure();
      auto yield = mlir::cast<YieldOp>(body.getTerminator());
      for (auto [destination, yielded] :
           llvm::zip(results, yield.getOperands())) {
        if (destination.spelling.empty())
          continue;
        CValue value = require(yielded);
        if (destination.kind == CValueKind::Scalar ||
            destination.kind == CValueKind::Pointer) {
          if (value.kind != destination.kind || value.spelling.empty()) {
            yield.emitError(
                "RISC-V conditional yielded an unavailable scalar or pointer value");
            return mlir::failure();
          }
          line(destination.spelling + " = " + value.spelling + ";");
          continue;
        }
        if (destination.kind == CValueKind::F32Vector ||
            destination.kind == CValueKind::Mask) {
          if (value.kind != destination.kind || value.spelling.empty()) {
            yield.emitError(
                "RISC-V conditional yielded an unavailable VLA value");
            return mlir::failure();
          }
          if (destination.kind == CValueKind::F32Vector) {
            std::optional<unsigned> lmul = physicalValueLMUL(yielded);
            if (!lmul || *lmul != destination.vectorLMUL) {
              yield.emitError(
                  "RISC-V conditional yielded an incompatible VLA vector");
              return mlir::failure();
            }
          }
          line(destination.spelling + " = " + value.spelling + ";");
          continue;
        }
        std::optional<int64_t> elements =
            physicalBlockElementCount(destination.type);
        if (!elements ||
            mlir::failed(copyF32Block(yield.getOperation(), destination, value,
                                      *elements, "if_copy")))
          return mlir::failure();
      }
      return mlir::success();
    };

    std::string cCondition = condition.spelling;
    if (!llvm::StringRef(cCondition).starts_with("(") ||
        !llvm::StringRef(cCondition).ends_with(")"))
      cCondition = "(" + cCondition + ")";
    line("if " + cCondition + " {");
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
        std::optional<unsigned> lmul = physicalValueLMUL(initial);
        if (!lmul)
          return op.emitError("ordered VLA carry has no physical value shape");
        value = CValue{result.getType(), CValueKind::F32Vector, fresh("carry")};
        value.vectorSEW = 32;
        value.vectorLMUL = *lmul;
        line("vfloat32m" + std::to_string(*lmul) +
             "_t " + value.spelling + " = " + init.spelling + ";");
      } else if (inVLA && init.kind == CValueKind::Mask) {
        value = CValue{result.getType(), CValueKind::Mask, fresh("carry")};
        if (!activePhysicalEntity || activePhysicalEntity->vlaMaskRatio == 0)
          return op.emitError("ordered VLA mask carry has no physical shape");
        line("vbool" + std::to_string(activePhysicalEntity->vlaMaskRatio) + "_t " +
             value.spelling + " = " + init.spelling + ";");
      } else if (init.kind == CValueKind::F32BlockStorage) {
        auto block = mlir::dyn_cast<BlockType>(result.getType());
        auto selected = controlBlockElements.find(result);
        if (!block || block.getShape().empty() || block.getShape().size() > 2 ||
            !block.getElementType().isF32() ||
            selected == controlBlockElements.end())
          return op.emitError(
              "ordered range block carry must be rank-one or rank-two f32");
        value = CValue{result.getType(), CValueKind::F32BlockStorage,
                       fresh("for_block_carry")};
        line("float " + value.spelling + "[" +
             std::to_string(selected->second) + "];");
        if (mlir::failed(copyF32Block(op.getOperation(), value, init,
                                      selected->second, "for_init")))
          return mlir::failure();
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
          std::optional<int64_t> elements =
              physicalBlockElementCount(destination.type);
          if (!elements ||
              mlir::failed(copyF32Block(yield.getOperation(), destination,
                                        value, *elements, "for_next")))
            return mlir::failure();
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
        else if (destination.kind == CValueKind::F32Vector) {
          std::optional<unsigned> lmul = physicalValueLMUL(yielded);
          if (!lmul) {
            yield.emitError("ordered VLA update has no physical value shape");
            return mlir::failure();
          }
          type = "vfloat32m" + std::to_string(*lmul) + "_t";
        } else {
          if (!activePhysicalEntity || activePhysicalEntity->vlaMaskRatio == 0) {
            yield.emitError("ordered VLA mask update has no physical shape");
            return mlir::failure();
          }
          type = "vbool" +
                 std::to_string(activePhysicalEntity->vlaMaskRatio) + "_t";
        }
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
      if (init.spelling.empty())
        return op.emitError("ordered while has an unavailable carried value");
      CValue value;
      if (init.kind == CValueKind::Scalar &&
          mlir::isa<mlir::IndexType, mlir::IntegerType, mlir::FloatType>(
              result.getType())) {
        value = CValue{result.getType(), CValueKind::Scalar,
                       fresh("while_carry")};
        line(scalarCType(elementType(result.getType())) + " " + value.spelling +
             " = " + init.spelling + ";");
      } else if (inVLA && init.kind == CValueKind::F32Vector &&
                 isRegionValue(result.getType())) {
        std::optional<unsigned> lmul = physicalValueLMUL(initial);
        if (!lmul)
          return op.emitError(
              "ordered VLA while carry has no physical value shape");
        value = CValue{result.getType(), CValueKind::F32Vector,
                       fresh("while_vector_carry")};
        value.vectorSEW = 32;
        value.vectorLMUL = *lmul;
        line("vfloat32m" + std::to_string(*lmul) + "_t " + value.spelling +
             " = " + init.spelling + ";");
      } else if (inVLA && init.kind == CValueKind::Mask &&
                 isRegionValue(result.getType())) {
        if (!activePhysicalEntity || activePhysicalEntity->vlaMaskRatio == 0)
          return op.emitError(
              "ordered VLA while mask carry has no physical shape");
        value = CValue{result.getType(), CValueKind::Mask,
                       fresh("while_mask_carry")};
        line("vbool" + std::to_string(activePhysicalEntity->vlaMaskRatio) +
             "_t " + value.spelling + " = " + init.spelling + ";");
      } else if (init.kind == CValueKind::F32BlockStorage) {
        auto selected = controlBlockElements.find(result);
        if (selected == controlBlockElements.end())
          return op.emitError(
              "ordered while block carry has no physical storage decision");
        value = CValue{result.getType(), CValueKind::F32BlockStorage,
                       fresh("while_block_carry")};
        line("float " + value.spelling + "[" +
             std::to_string(selected->second) + "];");
        if (mlir::failed(copyF32Block(op.getOperation(), value, init,
                                      selected->second, "while_init")))
          return mlir::failure();
      } else {
        return op.emitError(
            "ordered while carried value has no physical realization");
      }
      values[result] = value;
      carried.push_back(value);
    }

    auto stageValues = [&](mlir::ValueRange source, llvm::StringRef prefix,
                           mlir::Operation *owner,
                           llvm::SmallVectorImpl<CValue> &staged) {
      if (source.size() != carried.size())
        return mlir::failure();
      for (auto [expected, value] : llvm::zip(carried, source)) {
        CValue emitted = require(value);
        if (emitted.kind != expected.kind || emitted.spelling.empty() ||
            emitted.type != expected.type)
          return mlir::failure();
        CValue next{emitted.type, emitted.kind, fresh(prefix)};
        if (emitted.kind == CValueKind::Scalar) {
          line(scalarCType(elementType(emitted.type)) + " " + next.spelling +
               " = " + emitted.spelling + ";");
        } else if (inVLA && emitted.kind == CValueKind::F32Vector) {
          std::optional<unsigned> lmul = physicalValueLMUL(value);
          if (!lmul || expected.vectorLMUL != *lmul)
            return mlir::failure();
          next.vectorSEW = 32;
          next.vectorLMUL = *lmul;
          line("vfloat32m" + std::to_string(*lmul) + "_t " + next.spelling +
               " = " + emitted.spelling + ";");
        } else if (inVLA && emitted.kind == CValueKind::Mask) {
          if (!activePhysicalEntity || activePhysicalEntity->vlaMaskRatio == 0)
            return mlir::failure();
          line("vbool" + std::to_string(activePhysicalEntity->vlaMaskRatio) +
               "_t " + next.spelling + " = " + emitted.spelling + ";");
        } else if (emitted.kind == CValueKind::F32BlockStorage) {
          std::optional<int64_t> elements =
              physicalBlockElementCount(emitted.type);
          if (!elements)
            return mlir::failure();
          line("float " + next.spelling + "[" + std::to_string(*elements) +
               "];");
          if (mlir::failed(copyF32Block(owner, next, emitted, *elements,
                                        "while_stage")))
            return mlir::failure();
        } else {
          return mlir::failure();
        }
        staged.push_back(next);
      }
      return mlir::success();
    };

    auto assignValues = [&](llvm::ArrayRef<CValue> destination,
                            llvm::ArrayRef<CValue> source,
                            mlir::Operation *owner) {
      for (auto [target, value] : llvm::zip(destination, source)) {
        if (target.kind == CValueKind::Scalar ||
            target.kind == CValueKind::F32Vector ||
            target.kind == CValueKind::Mask) {
          line(target.spelling + " = " + value.spelling + ";");
          continue;
        }
        std::optional<int64_t> elements =
            physicalBlockElementCount(target.type);
        if (!elements ||
            mlir::failed(copyF32Block(owner, target, value, *elements,
                                      "while_assign")))
          return mlir::failure();
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
    if (mlir::failed(stageValues(conditionTerminator.getValues(),
                                 "while_forward",
                                 conditionTerminator.getOperation(),
                                 forwarded)))
      return conditionTerminator.emitError(
          "ordered while condition forwarded an unavailable value");

    line("if (!(" + predicate.spelling + ")) {");
    ++indent;
    if (mlir::failed(assignValues(carried, forwarded,
                                  conditionTerminator.getOperation())))
      return mlir::failure();
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
    if (mlir::failed(stageValues(yield.getValues(), "while_next",
                                 yield.getOperation(), updated)))
      return yield.emitError(
          "ordered while body yielded an unavailable value");
    if (mlir::failed(assignValues(carried, updated, yield.getOperation())))
      return mlir::failure();
    --indent;
    line("}");
    return mlir::success();
  }

  mlir::LogicalResult emitVLA(VLAOp op) {
    if (inVLA)
      return op.emitError("nested VLA regions are not supported");
    auto selected = physicalPlan.vlaRegions.find(op.getOperation());
    if (selected == physicalPlan.vlaRegions.end())
      return op.emitError("VLA physical decision was not prepared");
    if (mlir::failed(requireEntityPlan(op.getOperation(), selected->second)))
      return mlir::failure();
    const VLARegionDecision &decision = selected->second.realization;
    const PhysicalEntityPlan &entity = selected->second.entity;
    std::optional<unsigned> dataLMUL = rvvIntegerLMUL(entity.vlaDataShape);
    if (!dataLMUL || entity.vlaMaskRatio == 0)
      return op.emitError("VLA entity has no selected data or mask shape");
    mlir::Block &body = op.getBody().front();
    llvm::DenseMap<mlir::Operation *, CValue> aggregates;
    for (const VLAStateDecision &state : decision.states) {
      if ((state.realization == VLAStateRealization::RVVAddReduction ||
           state.realization == VLAStateRealization::RVVMaxReduction) &&
          state.placement == VLAStatePlacement::VectorCarry) {
        std::string accumulator = fresh("reduce_acc");
        std::string suffix = "f32m" + std::to_string(*dataLMUL);
        line("vfloat32m" + std::to_string(*dataLMUL) + "_t " +
             accumulator + " = __riscv_vfmv_v_f_" + suffix + "(" +
             expression(state.identity) + ", __riscv_vsetvlmax_e32m" +
             std::to_string(*dataLMUL) + "());");
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
    const PhysicalEntityPlan *previousEntity = activePhysicalEntity;
    auto emitStripBody = [&]() -> mlir::LogicalResult {
      inVLA = true;
      activeVL = vl;
      activeVLADecision = &decision;
      activePhysicalEntity = &entity;
      auto restoreVLAState = llvm::make_scope_exit([&] {
        inVLA = previousInVLA;
        activeVL = previousVL;
        activeVLADecision = previousDecision;
        activePhysicalEntity = previousEntity;
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
          mlir::LogicalResult lowered = emitVectorReduce(
              reduce, *state, aggregates[reduce.getOperation()]);
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

    line("for (size_t " + strip + " = " + begin.spelling + "; " + strip +
         " < " + end.spelling + ";) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e" +
         std::to_string(entity.vlaDataShape.sew) + "m" +
         std::to_string(*dataLMUL) + "(" + end.spelling +
         " - " + strip + ");");
    if (mlir::failed(emitStripBody()))
      return mlir::failure();
    line(strip + " += " + vl + ";");
    --indent;
    line("}");

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
      std::string suffix = "f32m" + std::to_string(*dataLMUL);
      line("vfloat32m1_t " + seed +
           " = __riscv_vfmv_v_f_f32m1(" + expression(state.identity) +
           ", 1);");
      std::string intrinsic =
          state.realization == VLAStateRealization::RVVAddReduction
              ? "__riscv_vfredusum_vs_"
              : "__riscv_vfredmax_vs_";
      line("vfloat32m1_t " + reduced + " = " + intrinsic + suffix +
           "_f32m1(" + aggregate.spelling + ", " + seed +
           ", __riscv_vsetvlmax_e32m" + std::to_string(*dataLMUL) +
           "());");
      line("const float " + result +
           " = __riscv_vfmv_f_s_f32m1_f32(" + reduced + ");");
      values[reduce.getResult()] =
          CValue{reduce.getResult().getType(), CValueKind::Scalar, result};
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
    if (base.indexedPointer) {
      if (offset.kind != CValueKind::Scalar)
        return op.emitError(
            "indexed VLA pointer can receive only a lane-independent offset");
      CValue result{op.getResult().getType(), CValueKind::Pointer,
                    "(" + base.spelling + " + " + offset.spelling + ")"};
      result.indexedPointer = true;
      result.indexSpelling = base.indexSpelling;
      result.vectorSEW = base.vectorSEW;
      result.vectorLMUL = base.vectorLMUL;
      values[op.getResult()] = std::move(result);
      return mlir::success();
    }
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

  CValue materializeIndexVector(mlir::Operation *consumer,
                                mlir::Value semantic, CValue input) {
    const PhysicalValueDecision *physical = findVLAValueDecision(semantic);
    const PhysicalHandoffDecision *handoff =
        findVLAHandoff(consumer, semantic);
    std::optional<unsigned> lmul =
        physical ? rvvIntegerLMUL(physical->shape) : std::nullopt;
    if (!physical || !handoff || handoff->sourceShape != physical->shape ||
        !lmul ||
        (physical->shape.sew != 32 && physical->shape.sew != 64))
      return {};
    if (input.kind == CValueKind::IndexVector) {
      if ((handoff->kind != PhysicalHandoff::Share &&
           handoff->kind != PhysicalHandoff::Convert) ||
          input.spelling.empty() || input.vectorSEW != physical->shape.sew ||
          input.vectorLMUL != *lmul)
        return {};
      return input;
    }
    if ((handoff->kind != PhysicalHandoff::Rematerialize &&
         handoff->kind != PhysicalHandoff::Convert) ||
        input.kind != CValueKind::Coordinate || input.spelling.empty() ||
        input.laneStride.empty())
      return {};
    std::string suffix = "u" + std::to_string(physical->shape.sew) + "m" +
                         std::to_string(*lmul);
    std::string vectorType = "vuint" + std::to_string(physical->shape.sew) +
                             "m" + std::to_string(*lmul) + "_t";
    std::string scalarType =
        "uint" + std::to_string(physical->shape.sew) + "_t";
    std::string name = fresh("index");
    line(vectorType + " " + name + " = __riscv_vid_v_" + suffix + "(" +
         activeVL + ");");
    if (input.laneStride != "1")
      line(name + " = __riscv_vmul_vx_" + suffix + "(" + name + ", (" +
           scalarType + ")(" + input.laneStride + "), " + activeVL + ");");
    line(name + " = __riscv_vadd_vx_" + suffix + "(" + name + ", (" +
         scalarType + ")(" + input.spelling + "), " + activeVL + ");");
    CValue result{semantic.getType(), CValueKind::IndexVector, name};
    result.vectorSEW = physical->shape.sew;
    result.vectorLMUL = *lmul;
    return result;
  }

  mlir::LogicalResult emitBinary(BinaryOp op) {
    if (isAbsorbedBinaryProducer(op.getOperation()))
      return mlir::success();
    CValue lhs = require(op.getLhs());
    CValue rhs = require(op.getRhs());
    bool lhsBundle = lhs.kind == CValueKind::F32VectorBundle;
    bool rhsBundle = rhs.kind == CValueKind::F32VectorBundle;
    if (lhsBundle || rhsBundle) {
      if (!inVLA || (lhsBundle && rhsBundle && lhs.fields.size() != rhs.fields.size()))
        return op.emitError(
            "VLA structured pointwise operands have incompatible bundles");
      size_t fields = lhsBundle ? lhs.fields.size() : rhs.fields.size();
      std::optional<unsigned> lmul = physicalValueLMUL(op.getResult());
      if (!lmul || fields == 0)
        return op.emitError(
            "VLA structured pointwise result has no physical shape");
      std::string suffix = "f32m" + std::to_string(*lmul);
      CValue result{op.getResult().getType(), CValueKind::F32VectorBundle, {}};
      for (size_t index = 0; index < fields; ++index) {
        CValue left = lhsBundle ? lhs.fields[index] : lhs;
        CValue right = rhsBundle ? rhs.fields[index] : rhs;
        bool leftVector = left.kind == CValueKind::F32Vector;
        bool rightVector = right.kind == CValueKind::F32Vector;
        if ((!leftVector && left.kind != CValueKind::Scalar) ||
            (!rightVector && right.kind != CValueKind::Scalar) ||
            (!leftVector && !rightVector) || left.spelling.empty() ||
            right.spelling.empty())
          return op.emitError(
              "VLA structured pointwise field is unavailable");
        std::string intrinsic;
        std::string first = left.spelling;
        std::string second = right.spelling;
        if (leftVector && rightVector) {
          intrinsic = llvm::StringSwitch<std::string>(op.getKind())
                          .Case("add", "__riscv_vfadd_vv_")
                          .Case("sub", "__riscv_vfsub_vv_")
                          .Case("mul", "__riscv_vfmul_vv_")
                          .Case("div", "__riscv_vfdiv_vv_")
                          .Case("max", "__riscv_vfmax_vv_")
                          .Case("min", "__riscv_vfmin_vv_")
                          .Default("");
        } else if (leftVector) {
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
          return op.emitError(
              "VLA structured pointwise kind has no RVV realization");
        std::string name = fresh("bundle");
        line("vfloat32m" + std::to_string(*lmul) + "_t " + name + " = " +
             intrinsic + suffix + "(" + first + ", " + second + ", " +
             activeVL + ");");
        result.fields.push_back(
            CValue{op.getResult().getType(), CValueKind::F32Vector, name});
      }
      if (mlir::failed(attachLogicalValidity(
              op.getOperation(), op.getResult().getType(), result, {lhs, rhs})))
        return mlir::failure();
      values[op.getResult()] = std::move(result);
      return mlir::success();
    }
    if (const VLAIndexBinaryDecision *decision =
            findIndexBinaryDecision(op.getOperation())) {
      bool lhsRegion = isRegionValue(op.getLhs().getType());
      bool rhsRegion = isRegionValue(op.getRhs().getType());
      CValue left = lhsRegion
                        ? materializeIndexVector(op.getOperation(), op.getLhs(), lhs)
                        : lhs;
      CValue right =
          rhsRegion
              ? materializeIndexVector(op.getOperation(), op.getRhs(), rhs)
              : rhs;
      const PhysicalValueDecision *resultShape =
          findVLAValueDecision(op.getResult());
      std::optional<unsigned> resultLMUL =
          resultShape ? rvvIntegerLMUL(resultShape->shape) : std::nullopt;
      if (!resultShape || !resultLMUL ||
          (resultShape->shape.sew != 32 && resultShape->shape.sew != 64))
        return op.emitError("VLA index binary has no physical result shape");
      std::string suffix = "u" + std::to_string(resultShape->shape.sew) + "m" +
                           std::to_string(*resultLMUL);
      std::string intrinsic;
      std::string first;
      std::string second;
      if (decision->realization ==
          VLAIndexBinaryRealization::RVVUnsignedVectorVector) {
        if (left.kind != CValueKind::IndexVector ||
            right.kind != CValueKind::IndexVector || left.spelling.empty() ||
            right.spelling.empty())
          return op.emitError("VLA index vector/vector operands are unavailable");
        intrinsic = llvm::StringSwitch<std::string>(op.getKind())
                        .Case("add", "__riscv_vadd_vv_")
                        .Case("sub", "__riscv_vsub_vv_")
                        .Case("mul", "__riscv_vmul_vv_")
                        .Case("div", "__riscv_vdivu_vv_")
                        .Case("mod", "__riscv_vremu_vv_")
                        .Default("");
        first = left.spelling;
        second = right.spelling;
      } else {
        CValue vector = decision->scalarOnLHS ? right : left;
        CValue scalar = decision->scalarOnLHS ? left : right;
        if (vector.kind != CValueKind::IndexVector ||
            scalar.kind != CValueKind::Scalar || vector.spelling.empty() ||
            scalar.spelling.empty())
          return op.emitError("VLA index vector/scalar operands are unavailable");
        if (decision->scalarOnLHS && op.getKind() == "sub")
          intrinsic = "__riscv_vrsub_vx_";
        else
          intrinsic = llvm::StringSwitch<std::string>(op.getKind())
                          .Case("add", "__riscv_vadd_vx_")
                          .Case("sub", "__riscv_vsub_vx_")
                          .Case("mul", "__riscv_vmul_vx_")
                          .Case("div", "__riscv_vdivu_vx_")
                          .Case("mod", "__riscv_vremu_vx_")
                          .Default("");
        first = vector.spelling;
        second = "(uint" + std::to_string(resultShape->shape.sew) + "_t)(" +
                 scalar.spelling + ")";
      }
      if (intrinsic.empty())
        return op.emitError("VLA index binary realization is unavailable");
      std::string name = fresh("index");
      line("vuint" + std::to_string(resultShape->shape.sew) + "m" +
           std::to_string(*resultLMUL) + "_t " + name + " = " + intrinsic +
           suffix + "(" + first + ", " + second + ", " + activeVL + ");");
      CValue result{op.getResult().getType(), CValueKind::IndexVector, name};
      result.vectorSEW = resultShape->shape.sew;
      result.vectorLMUL = *resultLMUL;
      if (mlir::failed(attachLogicalValidity(
              op.getOperation(), op.getResult().getType(), result, {lhs, rhs})))
        return mlir::failure();
      values[op.getResult()] = std::move(result);
      return mlir::success();
    }
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
      if (!activePhysicalEntity || activePhysicalEntity->vlaMaskRatio == 0)
        return op.emitError("RVV mask operation has no physical mask shape");
      std::string ratio = std::to_string(activePhysicalEntity->vlaMaskRatio);
      line("vbool" + ratio + "_t " + name + " = " + intrinsic + ratio + "(" +
           lhs.spelling + ", " + rhs.spelling + ", " + activeVL + ");");
      CValue result{op.getResult().getType(), CValueKind::Mask, name};
      if (mlir::failed(attachLogicalValidity(
              op.getOperation(), op.getResult().getType(), result, {lhs, rhs})))
        return mlir::failure();
      values[op.getResult()] = std::move(result);
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
      std::optional<unsigned> lmul = physicalValueLMUL(op.getResult());
      if (!lmul)
        return op.emitError("f16 fused multiply-add has no physical shape");
      std::string suffix = "f16m" + std::to_string(*lmul);
      std::string name = fresh("fma");
      line("vfloat16m" + std::to_string(*lmul) + "_t " + name +
           " = __riscv_vfmacc_vf_" + suffix + "(" + accumulator.spelling +
           ", " + scalar.spelling + ", " + vector.spelling + ", " + activeVL +
           ");");
      CValue result{op.getResult().getType(), CValueKind::F16Vector, name};
      if (mlir::failed(attachLogicalValidity(
              op.getOperation(), op.getResult().getType(), result,
              {accumulator, vector, scalar})))
        return mlir::failure();
      values[op.getResult()] = std::move(result);
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
    std::optional<unsigned> selectedLMUL = physicalValueLMUL(op.getResult());
    if (!selectedLMUL)
      return op.emitError("RVV pointwise result has no physical value shape");
    unsigned lmul = *selectedLMUL;
    std::string element = f32 ? "f32m" : "f16m";
    std::string vectorType = f32 ? "vfloat32m" : "vfloat16m";
    std::string suffix = element + std::to_string(lmul);
    intrinsic += suffix;
    std::string name = fresh("v");
    line(vectorType + std::to_string(lmul) + "_t " + name + " = " +
         intrinsic + "(" + first + ", " + second + ", " + activeVL + ");");
    CValue result{op.getResult().getType(), vectorKind, name};
    if (mlir::failed(attachLogicalValidity(
            op.getOperation(), op.getResult().getType(), result, {lhs, rhs})))
      return mlir::failure();
    values[op.getResult()] = std::move(result);
    return mlir::success();
  }

  mlir::LogicalResult emitUnary(UnaryOp op) {
    CValue input = require(op.getInput());
    if (input.kind == CValueKind::F32VectorBundle) {
      std::optional<unsigned> lmul = physicalValueLMUL(op.getResult());
      if (!inVLA || !lmul || input.fields.empty())
        return op.emitError("VLA structured unary has no physical shape");
      std::string suffix = "f32m" + std::to_string(*lmul);
      CValue result{op.getResult().getType(), CValueKind::F32VectorBundle, {}};
      for (const CValue &field : input.fields) {
        if (field.kind != CValueKind::F32Vector || field.spelling.empty())
          return op.emitError("VLA structured unary field is unavailable");
        std::string name = fresh("bundle");
        if (op.getKind() == "neg")
          line("vfloat32m" + std::to_string(*lmul) + "_t " + name +
               " = __riscv_vfneg_v_" + suffix + "(" + field.spelling +
               ", " + activeVL + ");");
        else if (op.getKind() == "sqrt")
          line("vfloat32m" + std::to_string(*lmul) + "_t " + name +
               " = __riscv_vfsqrt_v_" + suffix + "(" + field.spelling +
               ", " + activeVL + ");");
        else
          return op.emitError(
              "VLA structured unary kind has no RVV realization");
        result.fields.push_back(
            CValue{op.getResult().getType(), CValueKind::F32Vector, name});
      }
      if (mlir::failed(attachLogicalValidity(
              op.getOperation(), op.getResult().getType(), result, {input})))
        return mlir::failure();
      values[op.getResult()] = std::move(result);
      return mlir::success();
    }
    if (input.kind == CValueKind::F32Vector) {
      std::string name = fresh("v");
      std::optional<unsigned> selectedLMUL = physicalValueLMUL(op.getResult());
      if (!selectedLMUL)
        return op.emitError("RVV unary result has no physical value shape");
      unsigned lmul = *selectedLMUL;
      std::string suffix = "f32m" + std::to_string(lmul);
      std::string vectorType = "vfloat32m" + std::to_string(lmul) + "_t";
      if (const VLAUnaryDecision *decision =
              findUnaryDecision(op.getOperation())) {
        const PhysicalHandoffDecision *handoff =
            findVLAHandoff(op.getOperation(), op.getInput());
        if (lmul != 2 || !handoff || handoff->kind != PhysicalHandoff::Share ||
            handoff->sourceShape != handoff->resultShape ||
            handoff->resultShape != rvvShape(32, 2))
          return op.emitError("VLA unary physical handoff is incomplete");
        std::string helper =
            decision->realization == VLAUnaryRealization::RVVF32M2ExpPolynomial
                ? "__weft_exp_f32m2"
            : decision->realization == VLAUnaryRealization::RVVF32M2TanhViaExp
                ? "__weft_tanh_f32m2"
            : decision->realization ==
                      VLAUnaryRealization::RVVF32M2ScalarLibmSin
                ? "__weft_sin_f32m2"
                : "__weft_cos_f32m2";
        if ((decision->realization ==
                 VLAUnaryRealization::RVVF32M2ScalarLibmSin ||
             decision->realization ==
                 VLAUnaryRealization::RVVF32M2ScalarLibmCos) &&
            !findVLATemporaryDecision(op.getOperation()))
          return op.emitError("VLA trig unary has no selected local temporary");
        line("vfloat32m2_t " + name + " = " + helper + "(" +
             input.spelling + ", " + activeVL + ");");
      } else if (op.getKind() == "neg")
        line(vectorType + " " + name + " = __riscv_vfneg_v_" + suffix + "(" +
             input.spelling + ", " + activeVL + ");");
      else if (op.getKind() == "abs")
        line(vectorType + " " + name + " = __riscv_vfabs_v_" + suffix + "(" +
             input.spelling + ", " + activeVL + ");");
      else if (op.getKind() == "sqrt")
        line(vectorType + " " + name + " = __riscv_vfsqrt_v_" + suffix +
             "(" + input.spelling + ", " + activeVL + ");");
      else
        return op.emitError(
            "RVV pointwise lowering does not implement unary kind");
      CValue result{op.getResult().getType(), CValueKind::F32Vector, name};
      if (mlir::failed(attachLogicalValidity(
              op.getOperation(), op.getResult().getType(), result, {input})))
        return mlir::failure();
      values[op.getResult()] = std::move(result);
      return mlir::success();
    }
    std::string expression;
    if (op.getKind() == "neg")
      expression = "(-" + input.spelling + ")";
    else if (op.getKind() == "exp")
      expression = "expf(" + input.spelling + ")";
    else if (op.getKind() == "tanh")
      expression = "tanhf(" + input.spelling + ")";
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
        const PhysicalValueDecision *vectorShape =
            findVLAValueDecision(decision->coordinate);
        const PhysicalHandoffDecision *handoff =
            findVLAHandoff(op.getOperation(), decision->coordinate);
        std::string shapeSuffix =
            vectorShape ? rvvShapeSuffix(vectorShape->shape) : std::string{};
        bool indexVector =
            elementType(decision->coordinate.getType()).isIndex();
        CValueKind expected = indexVector
                                  ? CValueKind::IndexVector
                                  : decision->vectorSEW == 8
                                        ? CValueKind::U8Vector
                                        : CValueKind::F32Vector;
        if (coordinate.kind != expected || scalar.kind != CValueKind::Scalar ||
            coordinate.spelling.empty() || scalar.spelling.empty() ||
            shapeSuffix.empty() || !handoff ||
            handoff->kind != PhysicalHandoff::Share ||
            handoff->sourceShape != vectorShape->shape ||
            handoff->resultShape != vectorShape->shape || !activePhysicalEntity ||
            activePhysicalEntity->vlaMaskRatio == 0)
          return op.emitError("typed VLA predicate projection is unavailable");
        std::string vectorSuffix;
        std::string maskSuffix;
        std::string intrinsic;
        if (decision->vectorSEW == 8 || indexVector) {
          vectorSuffix = "u" + shapeSuffix;
          maskSuffix = vectorSuffix + "_b" +
                       std::to_string(activePhysicalEntity->vlaMaskRatio);
          intrinsic = llvm::StringSwitch<std::string>(decision->predicate)
                          .Case("eq", "__riscv_vmseq_vx_")
                          .Case("ne", "__riscv_vmsne_vx_")
                          .Case("lt", "__riscv_vmsltu_vx_")
                          .Case("le", "__riscv_vmsleu_vx_")
                          .Case("gt", "__riscv_vmsgtu_vx_")
                          .Case("ge", "__riscv_vmsgeu_vx_")
                          .Default("");
        } else {
          vectorSuffix = "f" + shapeSuffix;
          maskSuffix = vectorSuffix + "_b" +
                       std::to_string(activePhysicalEntity->vlaMaskRatio);
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
             std::to_string(activePhysicalEntity->vlaMaskRatio) + "_t " +
             mask + " = " + intrinsic + maskSuffix + "(" +
             coordinate.spelling + ", " + scalar.spelling + ", " + activeVL +
             ");");
        CValue result{op.getResult().getType(), CValueKind::Mask, mask};
        if (mlir::failed(attachLogicalValidity(
                op.getOperation(), op.getResult().getType(), result,
                {lhs, rhs})))
          return mlir::failure();
        values[op.getResult()] = std::move(result);
        return mlir::success();
      }
      if (coordinate.kind != CValueKind::Coordinate ||
          scalar.kind != CValueKind::Scalar || coordinate.spelling.empty() ||
          scalar.spelling.empty() || coordinate.laneStride.empty())
        return op.emitError("VLA predicate projection is unavailable");

      if (!activePhysicalEntity || !activePhysicalEntity->vlaIndexShape ||
          activePhysicalEntity->vlaMaskRatio == 0)
        return op.emitError("VLA predicate has no physical index/mask shape");
      std::optional<unsigned> indexLMUL = activeVLAIndexLMUL();
      if (!indexLMUL)
        return op.emitError("VLA predicate index shape has no intrinsic spelling");
      unsigned indexSEW = activePhysicalEntity->vlaIndexShape.sew;

      std::string vectorSuffix = "u" +
                                 std::to_string(indexSEW) + "m" +
                                 std::to_string(*indexLMUL);
      std::string maskSuffix =
          vectorSuffix + "_b" +
          std::to_string(activePhysicalEntity->vlaMaskRatio);
      std::string vectorType = "vuint" +
                               std::to_string(indexSEW) + "m" +
                               std::to_string(*indexLMUL) + "_t";
      std::string scalarType =
          "uint" + std::to_string(indexSEW) + "_t";
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
      line("vbool" + std::to_string(activePhysicalEntity->vlaMaskRatio) + "_t " +
           mask + " = " + intrinsic + maskSuffix + "(" + lane + ", (" +
           scalarType + ")(" + scalar.spelling + "), " + activeVL + ");");
      CValue result{op.getResult().getType(), CValueKind::Mask, mask};
      if (mlir::failed(attachLogicalValidity(
              op.getOperation(), op.getResult().getType(), result, {lhs, rhs})))
        return mlir::failure();
      values[op.getResult()] = std::move(result);
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
      const PhysicalHandoffDecision *handoff =
          findVLAHandoff(op.getOperation(), op.getInput());
      const PhysicalValueDecision *resultShape =
          findVLAValueDecision(op.getResult());
      if (decision->realization == VLACastRealization::RVVIdentity) {
        if (!handoff || handoff->kind != PhysicalHandoff::Share ||
            !resultShape || handoff->sourceShape != handoff->resultShape ||
            handoff->resultShape != resultShape->shape)
          return op.emitError("VLA identity cast handoff is inconsistent");
        input.type = op.getResult().getType();
        values[op.getResult()] = std::move(input);
        return mlir::success();
      }
      if (!handoff || handoff->kind != PhysicalHandoff::Convert ||
          !resultShape || handoff->resultShape != resultShape->shape)
        return op.emitError("VLA cast physical handoff is inconsistent");
      std::optional<unsigned> sourceLMUL =
          rvvIntegerLMUL(handoff->sourceShape);
      std::optional<unsigned> resultLMUL =
          rvvIntegerLMUL(handoff->resultShape);
      if (!sourceLMUL || !resultLMUL)
        return op.emitError("VLA cast shape has no intrinsic-C spelling");
      if (decision->realization == VLACastRealization::RVVIndexToF32) {
        CValue index =
            materializeIndexVector(op.getOperation(), op.getInput(), input);
        if (index.kind != CValueKind::IndexVector || index.spelling.empty() ||
            handoff->resultShape.sew != 32)
          return op.emitError(
              "selected index-to-f32 VLA cast operand is unavailable");
        std::string name = fresh("index_f32");
        std::string intrinsic = handoff->sourceShape.sew == 64
                                    ? "__riscv_vfncvt_f_xu_w_"
                                    : "__riscv_vfcvt_f_xu_v_";
        std::string suffix = "f32m" + std::to_string(*resultLMUL);
        line("vfloat32m" + std::to_string(*resultLMUL) + "_t " + name +
             " = " + intrinsic + suffix + "(" + index.spelling + ", " +
             activeVL + ");");
        CValue result{op.getResult().getType(), CValueKind::F32Vector, name};
        if (mlir::failed(attachLogicalValidity(
                op.getOperation(), op.getResult().getType(), result, {input})))
          return mlir::failure();
        values[op.getResult()] = std::move(result);
        return mlir::success();
      }
      if (decision->realization ==
          VLACastRealization::RVVZeroExtendU32ToIndex) {
        if (input.kind != CValueKind::U32Vector || input.spelling.empty())
          return op.emitError(
              "selected u32-to-index VLA cast operand is unavailable");
        CValue result{op.getResult().getType(), CValueKind::IndexVector,
                      input.spelling};
        result.vectorSEW = handoff->sourceShape.sew;
        result.vectorLMUL = *sourceLMUL;
        if (mlir::failed(attachLogicalValidity(
                op.getOperation(), op.getResult().getType(), result, {input})))
          return mlir::failure();
        values[op.getResult()] = std::move(result);
        return mlir::success();
      }
      CValueKind sourceKind =
          handoff->sourceShape.sew == 16 ? CValueKind::F16Vector
                                         : CValueKind::F32Vector;
      CValueKind resultKind =
          handoff->resultShape.sew == 16 ? CValueKind::F16Vector
                                         : CValueKind::F32Vector;
      if (input.kind != sourceKind || input.spelling.empty())
        return op.emitError("selected VLA cast operand is unavailable");
      std::string name = fresh(handoff->resultShape.sew == 32 ? "widen_f16"
                                                              : "narrow_f32");
      std::string resultSuffix =
          "f" + std::to_string(handoff->resultShape.sew) + "m" +
          std::to_string(*resultLMUL);
      std::string intrinsic =
          decision->realization == VLACastRealization::RVVWidenF16ToF32
              ? "__riscv_vfwcvt_f_f_v_"
              : "__riscv_vfncvt_f_f_w_";
      line("vfloat" + std::to_string(handoff->resultShape.sew) + "m" +
           std::to_string(*resultLMUL) + "_t " + name + " = " +
           intrinsic + resultSuffix + "(" + input.spelling + ", " + activeVL +
           ");");
      CValue result{op.getResult().getType(), resultKind, name};
      if (mlir::failed(attachLogicalValidity(
              op.getOperation(), op.getResult().getType(), result, {input})))
        return mlir::failure();
      values[op.getResult()] = std::move(result);
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
    const PhysicalHandoffDecision *handoff =
        findVLAHandoff(op.getOperation(), op.getInput());
    const PhysicalTemporaryDecision *intermediateShape =
        findVLATemporaryDecision(op.getOperation());
    const PhysicalValueDecision *resultShape =
        findVLAValueDecision(op.getResult());
    std::string sourceSuffix =
        handoff ? rvvShapeSuffix(handoff->sourceShape) : std::string{};
    std::string intermediateSuffix =
        intermediateShape ? rvvShapeSuffix(intermediateShape->shape)
                          : std::string{};
    std::string resultSuffix =
        resultShape ? rvvShapeSuffix(resultShape->shape) : std::string{};
    if (!decision || input.kind != CValueKind::F32Vector || !handoff ||
        handoff->kind != PhysicalHandoff::Convert || sourceSuffix.empty() ||
        intermediateSuffix.empty() || resultSuffix.empty())
      return op.emitError("VLA narrow projection is unavailable");
    std::string intermediate = fresh("narrow_i16");
    std::string result = fresh("narrow_i8");
    line("vint" + intermediateSuffix + "_t " + intermediate +
         " = __riscv_vfncvt_x_f_w_i" + intermediateSuffix + "(" +
         input.spelling + ", " + activeVL + ");");
    line("vint" + resultSuffix + "_t " + result +
         " = __riscv_vnclip_wx_i" + resultSuffix + "(" + intermediate +
         ", 0, __RISCV_VXRM_RNE, " + activeVL + ");");
    CValue value{op.getResult().getType(), CValueKind::I8Vector, result};
    if (mlir::failed(attachLogicalValidity(
            op.getOperation(), op.getResult().getType(), value, {input})))
      return mlir::failure();
    values[op.getResult()] = std::move(value);
    return mlir::success();
  }

  mlir::LogicalResult emitBitcast(BitcastOp op) {
    CValue input = require(op.getInput());
    mlir::Type source = elementType(op.getInput().getType());
    mlir::Type target = elementType(op.getResult().getType());
    if (inVLA &&
        ((input.kind == CValueKind::I8Vector &&
          target.isUnsignedInteger(8)) ||
         (input.kind == CValueKind::U8Vector &&
          target.isSignedInteger(8)))) {
      const PhysicalValueDecision *sourceShape =
          findVLAValueDecision(op.getInput());
      const PhysicalValueDecision *resultShape =
          findVLAValueDecision(op.getResult());
      std::string shapeSuffix =
          resultShape ? rvvShapeSuffix(resultShape->shape) : std::string{};
      if (!sourceShape || !resultShape ||
          sourceShape->shape != resultShape->shape || shapeSuffix.empty() ||
          input.spelling.empty())
        return op.emitError("RVV byte bitcast has no shared physical shape");
      bool toUnsigned = target.isUnsignedInteger(8);
      std::string name = fresh("reinterpret_i8");
      std::string sourcePrefix = toUnsigned ? "i" : "u";
      std::string resultPrefix = toUnsigned ? "u" : "i";
      line("v" + std::string(toUnsigned ? "uint" : "int") + shapeSuffix +
           "_t " + name + " = __riscv_vreinterpret_v_" + sourcePrefix +
           shapeSuffix + "_" + resultPrefix + shapeSuffix + "(" +
           input.spelling + ");");
      CValue value{op.getResult().getType(),
                   toUnsigned ? CValueKind::U8Vector : CValueKind::I8Vector,
                   name};
      if (mlir::failed(attachLogicalValidity(
              op.getOperation(), op.getResult().getType(), value, {input})))
        return mlir::failure();
      values[op.getResult()] = std::move(value);
      return mlir::success();
    }
    if (input.kind != CValueKind::Scalar || input.spelling.empty())
      return op.emitError("RVV bitcast lowering requires a scalar input");
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
    else if (source.isSignedInteger(8) && target.isUnsignedInteger(8)) {
      values[op.getResult()] = scalarExpression(
          op.getResult(), "((uint8_t)(" + input.spelling + "))", "bitcast");
      return mlir::success();
    } else if (source.isSignedInteger(16) &&
               target.isUnsignedInteger(16)) {
      values[op.getResult()] = scalarExpression(
          op.getResult(), "((uint16_t)(" + input.spelling + "))", "bitcast");
      return mlir::success();
    } else
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
    if (const VLAIndexSelectDecision *decision =
            findIndexSelectDecision(op.getOperation())) {
      const PhysicalValueDecision *resultShape =
          findVLAValueDecision(op.getResult());
      const PhysicalTemporaryDecision *temporary =
          findVLATemporaryDecision(op.getOperation());
      std::optional<unsigned> lmul =
          resultShape ? rvvIntegerLMUL(resultShape->shape) : std::nullopt;
      if (predicate.kind != CValueKind::Mask || predicate.spelling.empty() ||
          !resultShape || !lmul ||
          (resultShape->shape.sew != 32 && resultShape->shape.sew != 64) ||
          ((decision->trueScalar || decision->falseScalar) &&
           (!temporary || temporary->shape != resultShape->shape)))
        return op.emitError("VLA index select physical decision is incomplete");
      std::string suffix = "u" + std::to_string(resultShape->shape.sew) + "m" +
                           std::to_string(*lmul);
      std::string vectorType =
          "vuint" + std::to_string(resultShape->shape.sew) + "m" +
          std::to_string(*lmul) + "_t";
      auto materialize = [&](mlir::Value semantic, CValue value,
                             bool scalar) -> std::optional<std::string> {
        if (!scalar) {
          CValue index =
              materializeIndexVector(op.getOperation(), semantic, value);
          if (index.kind != CValueKind::IndexVector || index.spelling.empty())
            return std::nullopt;
          return index.spelling;
        }
        if (value.kind != CValueKind::Scalar || value.spelling.empty())
          return std::nullopt;
        std::string name = fresh("select_index");
        line(vectorType + " " + name + " = __riscv_vmv_v_x_" + suffix +
             "((uint" + std::to_string(resultShape->shape.sew) + "_t)(" +
             value.spelling + "), " + activeVL + ");");
        return name;
      };
      std::optional<std::string> trueVector = materialize(
          op.getTrueValue(), trueValue, decision->trueScalar);
      std::optional<std::string> falseVector = materialize(
          op.getFalseValue(), falseValue, decision->falseScalar);
      if (!trueVector || !falseVector)
        return op.emitError("VLA index select operands are unavailable");
      std::string name = fresh("select_index");
      line(vectorType + " " + name + " = __riscv_vmerge_vvm_" + suffix +
           "(" + *falseVector + ", " + *trueVector + ", " +
           predicate.spelling + ", " + activeVL + ");");
      CValue result{op.getResult().getType(), CValueKind::IndexVector, name};
      result.vectorSEW = resultShape->shape.sew;
      result.vectorLMUL = *lmul;
      if (mlir::failed(attachLogicalValidity(
              op.getOperation(), op.getResult().getType(), result,
              {predicate, trueValue, falseValue})))
        return mlir::failure();
      values[op.getResult()] = std::move(result);
      return mlir::success();
    }
    if (predicate.kind == CValueKind::Mask && inVLA &&
        elementType(op.getResult().getType()).isF32()) {
      std::optional<unsigned> selectedLMUL = physicalValueLMUL(op.getResult());
      if (!selectedLMUL)
        return op.emitError("RVV select result has no physical shape");
      unsigned lmul = *selectedLMUL;
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
      CValue result{op.getResult().getType(), CValueKind::F32Vector, name};
      if (mlir::failed(attachLogicalValidity(
              op.getOperation(), op.getResult().getType(), result,
              {predicate, trueValue, falseValue})))
        return mlir::failure();
      values[op.getResult()] = std::move(result);
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
    bool dependsOnProjectedAxis =
        llvm::any_of(axisValues, [&](const auto &axis) {
          return dependsOn(value, axis.first);
        });
    if (!containsBlockType(value.getType()) && !dependsOnProjectedAxis) {
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
    std::string projected = expression(value);
    return projected.empty() ? std::nullopt
                             : std::optional<std::string>(projected);
  }

  std::optional<std::string>
  projectLocalBlockMemoryBase(const LocalBlockMemoryFact &fact) {
    if (!fact.pointer || !fact.storageAxis)
      return std::nullopt;
    llvm::DenseMap<mlir::Value, std::string> axes;
    axes[fact.storageAxis] = "0";
    return projectBlockScalar(fact.pointer, axes);
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
    return std::nullopt;
  }

  mlir::LogicalResult emitVLADot(const VLADotDecision &decision) {
    const PhysicalHandoffDecision *handoff =
        findVLAHandoff(decision.operation, decision.init);
    const PhysicalValueDecision *initShape =
        findVLAValueDecision(decision.init);
    if (!initShape)
      return decision.operation->emitError(
          "VLA dot physical value shape is unavailable");
    RVVVectorShape dotShape = initShape->shape;
    std::optional<unsigned> lmul = rvvIntegerLMUL(dotShape);
    if (!lmul || dotShape.sew != 32 || !activePhysicalEntity ||
        *lmul != decision.physical.lmul || decision.physical.kUnroll == 0)
      return decision.operation->emitError(
          "VLA dot physical entity is incomplete");
    PhysicalHandoff expected =
        decision.initRealization ==
                VLADotInitRealization::MaterializedRegion
            ? PhysicalHandoff::Share
            : PhysicalHandoff::Rematerialize;
    if (!handoff || handoff->kind != expected ||
        handoff->sourceShape != dotShape ||
        handoff->resultShape != dotShape)
      return decision.operation->emitError(
          "VLA dot physical handoff is inconsistent");
    if (decision.realization ==
        VLADotRealization::RVVF32FreeAxisVectorDot) {
      auto dot = mlir::cast<DotOp>(decision.operation);
      auto freeLoad = mlir::cast<LoadOp>(decision.freeLoad);
      std::string extent = expression(decision.reductionExtent);
      CValue init = require(decision.init);
      if (extent.empty() || activeVL.empty() ||
          decision.initRealization !=
              VLADotInitRealization::MaterializedRegion ||
          init.kind != CValueKind::F32Vector || init.spelling.empty())
        return dot.emitError(
            "RVV VLA vector-dot physical operands are unavailable");

      std::string vectorSuffix = "f32m" + std::to_string(*lmul);
      std::string vectorType =
          "vfloat32m" + std::to_string(*lmul) + "_t";

      std::string accumulator = fresh("vla_dot_acc");
      line(vectorType + " " + accumulator + " = " + init.spelling + ";");
      std::string reduction = fresh("vla_dot_k");
      line("for (size_t " + reduction + " = 0; " + reduction + " < " +
           extent + "; " + reduction + " += " +
           std::to_string(decision.physical.kUnroll) + ") {");
      ++indent;
      for (unsigned unroll = 0;
           unroll < decision.physical.kUnroll; ++unroll) {
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
        std::string vector = fresh("vla_dot_free");
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
      values[dot.getResult()] =
          CValue{dot.getResult().getType(), CValueKind::F32Vector,
                 accumulator};
      return mlir::success();
    }

    auto dot = mlir::cast<DotOp>(decision.operation);
    auto lhsLoad = mlir::cast<LoadOp>(decision.lhsLoad);
    auto rhsLoad = mlir::cast<LoadOp>(decision.rhsLoad);
    std::string extent = expression(decision.reductionExtent);
    if (extent.empty() || activeVL.empty())
      return dot.emitError(
          "RVV VLA dot physical bounds are unavailable");

    std::string vectorSuffix = "f32m" + std::to_string(*lmul);
    std::string vectorType =
        "vfloat32m" + std::to_string(*lmul) + "_t";
    llvm::SmallVector<std::string> lhsActive;
    for (unsigned lane = 0; lane < decision.rowTile; ++lane) {
      llvm::DenseMap<mlir::Value, std::string> axes;
      axes[decision.rowAxis] = std::to_string(lane);
      axes[decision.reductionAxis] = "0";
      std::optional<std::string> lhsPredicate =
          projectBlockScalar(lhsLoad.getWhere(), axes);
      if (!decision.lhsPredicateVariesByReduction && !lhsPredicate)
        return dot.emitError(
            "RVV VLA dot row projection is unavailable");
      if (!decision.lhsPredicateVariesByReduction) {
        std::string lhsCondition = fresh("vla_dot_lhs_active");
        line("const bool " + lhsCondition + " = " + *lhsPredicate + ";");
        lhsActive.push_back(std::move(lhsCondition));
      }
    }

    llvm::SmallVector<std::string> accumulators;
    auto emitCompute = [&](bool guarded) -> mlir::LogicalResult {
      for (unsigned lane = 0; lane < decision.rowTile; ++lane) {
        std::string accumulator = fresh("vla_dot_acc");
        line(vectorType + " " + accumulator + " = __riscv_vfmv_v_f_" +
             vectorSuffix + "(0.0f, " + activeVL + ");");
        accumulators.push_back(std::move(accumulator));
      }

      std::string reduction = fresh("vla_dot_k");
      line("for (size_t " + reduction + " = 0; " + reduction + " < " +
           extent + "; " + reduction + " += " +
           std::to_string(decision.physical.kUnroll) + ") {");
      ++indent;
      for (unsigned unroll = 0;
           unroll < decision.physical.kUnroll; ++unroll) {
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
        std::string rhsVector = fresh("vla_dot_rhs");
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
      return mlir::success();
    };

    if (mlir::failed(emitCompute(true)))
      return mlir::failure();
    CValue result{dot.getResult().getType(), CValueKind::F32VectorBundle, {}};
    for (const std::string &accumulator : accumulators)
      result.fields.push_back(
          CValue{dot.getResult().getType(), CValueKind::F32Vector,
                 accumulator});
    values[dot.getResult()] = std::move(result);
    return mlir::success();
  }

  mlir::FailureOr<PlannedPhysicalDecision<DotDecision>>
  decideLocalF32RowMicrotile(DotOp dot) {
    BlockType lhsType =
        mlir::dyn_cast<BlockType>(unwrapLogicalValidity(dot.getLhs().getType()));
    BlockType rhsType =
        mlir::dyn_cast<BlockType>(unwrapLogicalValidity(dot.getRhs().getType()));
    BlockType resultType = mlir::dyn_cast<BlockType>(
        unwrapLogicalValidity(dot.getResult().getType()));
    if (!lhsType || !rhsType || !resultType ||
        lhsType.getShape().size() != 2 ||
        rhsType.getShape().size() != 1 || resultType.getShape().size() != 1 ||
        lhsType.getShape()[0] != resultType.getShape()[0] ||
        lhsType.getShape()[1] != rhsType.getShape()[0] ||
        resultType.getShape()[0] <= 0 || resultType.getShape()[0] > 8 ||
        !lhsType.getElementType().isF32() ||
        !rhsType.getElementType().isF32() || !resultType.getElementType().isF32() ||
        !mlir::isa<BlockType>(dot.getInit().getType()) ||
        mlir::cast<BlockType>(dot.getInit().getType()).getShape() !=
            resultType.getShape() ||
        mlir::cast<BlockType>(dot.getInit().getType()).getAxisIds() !=
            resultType.getAxisIds() ||
        !mlir::cast<BlockType>(dot.getInit().getType())
             .getElementType()
             .isF32() ||
        dot.getOrder() != "relaxed" || dot.getMath() != "native" ||
        !dot.getAccDtype().isF32()) {
      dot.emitError(
          "RVV row microtile requires a [BM,K] x [K] local f32 dot");
      return mlir::failure();
    }

    LoadOp lhsLoad = reloadableLoad(dot.getLhs());
    LoadOp rhsLoad = reloadableLoad(dot.getRhs());
    if (!lhsLoad || !rhsLoad || !isTrue(rhsLoad.getWhere())) {
      dot.emitError("RVV row microtile load producers are unavailable");
      return mlir::failure();
    }

    std::optional<StructuredProductFacts> product =
        structuredProductFacts(dot.getLhs(), dot.getRhs(), dot.getResult());
    if (!product || product->lhsFreeAxes.size() != 1 ||
        !product->rhsFreeAxes.empty() || product->reductionAxes.size() != 1 ||
        product->resultAxes.size() != 1) {
      dot.emitError(
          "RVV row microtile requires explicit row and shared K block axes");
      return mlir::failure();
    }
    mlir::Value rowCoordinate = product->lhsFreeAxes.front();
    mlir::Value reductionCoordinate = product->reductionAxes.front();
    BlockIndexOp rowAxis = rowCoordinate.getDefiningOp<BlockIndexOp>();
    BlockIndexOp reductionAxis =
        reductionCoordinate.getDefiningOp<BlockIndexOp>();
    const MemoryAccessFact *lhsAccess = memoryFact(lhsLoad.getOperation());
    const MemoryAccessFact *rhsAccess = memoryFact(rhsLoad.getOperation());
    if (!rowAxis || !reductionAxis || !lhsAccess || !rhsAccess) {
      dot.emitError("RVV row microtile typed axis/access facts are unavailable");
      return mlir::failure();
    }
    mlir::Value lhsRoot = lhsAccess->root;
    mlir::Value rhsRoot = rhsAccess->root;
    auto lhsPointer = lhsRoot ? mlir::dyn_cast<PtrType>(lhsRoot.getType())
                              : PtrType{};
    auto rhsPointer = rhsRoot ? mlir::dyn_cast<PtrType>(rhsRoot.getType())
                              : PtrType{};
    if (!lhsPointer || !rhsPointer ||
        !lhsPointer.getElementType().isF32() ||
        !rhsPointer.getElementType().isF32() ||
        valueDependsOnAxis(lhsLoad.getWhere(), reductionCoordinate) ||
        (!isTrue(lhsLoad.getWhere()) &&
         !isFloatConstant(lhsLoad.getOther(), 0.0))) {
      dot.emitError("RVV row microtile pointer facts are unavailable");
      return mlir::failure();
    }

    PlannedPhysicalDecision<DotDecision> planned;
    DotDecision &decision = planned.realization;
    decision.operation = dot.getOperation();
    decision.realization = DotRealization::RVVF32RowMicrotile;
    decision.lhsLoad = lhsLoad.getOperation();
    decision.rhsLoad = rhsLoad.getOperation();
    decision.rowAxis = rowAxis.getResult();
    decision.reductionAxis = reductionAxis.getResult();
    decision.reductionExtent = reductionAxis.getExtent();
    decision.rowTile = static_cast<unsigned>(resultType.getShape()[0]);
    LaneRelation lhsRelation =
        memoryRelation(*lhsAccess, reductionCoordinate);
    LaneRelation rhsRelation =
        memoryRelation(*rhsAccess, reductionCoordinate);
    if ((lhsRelation != LaneRelation::UnitStride &&
         lhsRelation != LaneRelation::Strided) ||
        (rhsRelation != LaneRelation::UnitStride &&
         rhsRelation != LaneRelation::Strided) ||
        memoryRelation(*lhsAccess, rowCoordinate) ==
            LaneRelation::Independent ||
        memoryRelation(*rhsAccess, rowCoordinate) !=
            LaneRelation::Independent) {
      dot.emitError("RVV row microtile memory relations are unavailable");
      return mlir::failure();
    }
    F32DotCandidateFacts candidateFacts;
    candidateFacts.model = F32DotResourceModel::LocalRow;
    candidateFacts.rowTile = decision.rowTile;
    if (std::optional<int64_t> extent =
            integerConstantValue(decision.reductionExtent);
        extent && *extent >= 0)
      candidateFacts.reductionExtent = static_cast<uint64_t>(*extent);
    candidateFacts.unitStrideOperands =
        (lhsRelation == LaneRelation::UnitStride) +
        (rhsRelation == LaneRelation::UnitStride);
    candidateFacts.stridedOperands =
        (lhsRelation == LaneRelation::Strided) +
        (rhsRelation == LaneRelation::Strided);
    candidateFacts.reductionPredicate =
        valueDependsOnAxis(lhsLoad.getWhere(), reductionCoordinate);
    candidateFacts.predicateGroups =
        candidateFacts.reductionPredicate ? 1 : 0;
    std::optional<F32DotPhysicalConfig> physical =
        weft::riscv_internal::selectF32DotPhysicalConfig(
            candidateFacts, options.target, options.backend);
    if (!physical) {
      dot.emitError(
          "RVV row microtile has no legal register-resource candidate");
      return mlir::failure();
    }
    decision.physical = *physical;
    RVVVectorShape computeShape = rvvShape(32, physical->lmul);
    for (mlir::Value value :
         {dot.getLhs(), dot.getRhs(), dot.getInit(), dot.getResult()})
      recordPhysicalValue(planned.entity, value, computeShape);
    recordPhysicalHandoff(planned.entity, dot.getOperation(), dot.getLhs(),
                          PhysicalHandoff::Reload, computeShape, computeShape);
    recordPhysicalHandoff(planned.entity, dot.getOperation(), dot.getRhs(),
                          PhysicalHandoff::Reload, computeShape, computeShape);
    recordPhysicalHandoff(planned.entity, dot.getOperation(), dot.getInit(),
                          PhysicalHandoff::Share, computeShape, computeShape);
    recordPhysicalHandoff(planned.entity, dot.getOperation(), dot.getResult(),
                          PhysicalHandoff::LocalPack, computeShape,
                          computeShape);
    planned.entity.storages.push_back(PhysicalStorageDecision{
        dot.getResult(), {}, static_cast<int64_t>(decision.rowTile), 16});
    planned.entity.resources.architecturalGroups =
        options.target.vectorRegisters;
    planned.entity.resources.valueGroups = decision.rowTile * physical->lmul;
    planned.entity.resources.memoryGroups =
        2 * physical->kUnroll * physical->lmul;
    planned.entity.resources.primitiveGroups =
        planned.entity.resources.valueGroups +
        planned.entity.resources.memoryGroups;
    planned.entity.resources.peakGroups =
        planned.entity.resources.primitiveGroups + 1;
    return planned;
  }

  mlir::LogicalResult emitLocalF32Dot(DotOp dot) {
    auto selected = physicalPlan.dots.find(dot.getOperation());
    if (selected == physicalPlan.dots.end())
      return dot.emitError("local dot has no selected physical decision");
    if (mlir::failed(requireEntityPlan(dot.getOperation(), selected->second)))
      return mlir::failure();
    const DotDecision &decision = selected->second.realization;
    const PhysicalEntityPlan &entity = selected->second.entity;
    if (mlir::failed(requirePhysicalHandoff(
            dot.getOperation(), entity, dot.getInit(), PhysicalHandoff::Share)))
      return mlir::failure();
    CValue init = require(dot.getInit());
    if (init.kind != CValueKind::F32BlockStorage || init.spelling.empty())
      return dot.emitError("local dot init block storage is unavailable");
    auto lhsLoad = mlir::cast<LoadOp>(decision.lhsLoad);
    auto rhsLoad = mlir::cast<LoadOp>(decision.rhsLoad);
    std::string extent = expression(decision.reductionExtent);
    if (extent.empty())
      return dot.emitError(
          "RVV row microtile access projection is unavailable");

    auto resultShape = llvm::find_if(
        entity.values, [&](const PhysicalValueDecision &value) {
          return value.value == dot.getResult();
        });
    if (resultShape == entity.values.end() ||
        entity.resources.peakGroups > entity.resources.architecturalGroups)
      return dot.emitError("RVV row microtile physical entity is incomplete");
    auto resultStorage = llvm::find_if(
        entity.storages, [&](const PhysicalStorageDecision &storage) {
          return storage.value == dot.getResult();
        });
    if (resultStorage == entity.storages.end() ||
        resultStorage->elements != static_cast<int64_t>(decision.rowTile) ||
        resultStorage->reusedStorage)
      return dot.emitError("RVV row microtile result storage is incomplete");
    std::optional<unsigned> lmul = rvvIntegerLMUL(resultShape->shape);
    if (!lmul || resultShape->shape.sew != 32 ||
        *lmul != decision.physical.lmul || decision.physical.kUnroll == 0)
      return dot.emitError("RVV row microtile value shape is unavailable");
    std::string fullVL = fresh("dot_vlmax");
    std::string vectorSuffix = "f32m" + std::to_string(*lmul);
    std::string vectorType = "vfloat32m" + std::to_string(*lmul) + "_t";
    std::string storage = fresh("dot_result");
    line("float " + storage + "[" + std::to_string(decision.rowTile) + "];");
    line("const size_t " + fullVL + " = __riscv_vsetvlmax_e32m" +
         std::to_string(*lmul) + "();");

    llvm::SmallVector<std::string> lhsActive;
    for (unsigned lane = 0; lane < decision.rowTile; ++lane) {
      llvm::DenseMap<mlir::Value, std::string> axes;
      axes[decision.rowAxis] = std::to_string(lane);
      axes[decision.reductionAxis] = "0";
      std::optional<std::string> lhsPredicate =
          projectBlockScalar(lhsLoad.getWhere(), axes);
      if (!lhsPredicate)
        return dot.emitError(
            "RVV row microtile row projection is unavailable");
      std::string lhsCondition = fresh("dot_lhs_active");
      line("const bool " + lhsCondition + " = " + *lhsPredicate + ";");
      lhsActive.push_back(std::move(lhsCondition));
    }

    auto emitCompute = [&](bool guarded) -> mlir::LogicalResult {
      llvm::SmallVector<std::string> accumulators;
      for (unsigned lane = 0; lane < decision.rowTile; ++lane) {
        std::string accumulator = fresh("dot_acc");
        line(vectorType + " " + accumulator + " = __riscv_vfmv_v_f_" +
             vectorSuffix + "(0.0f, " + fullVL + ");");
        accumulators.push_back(std::move(accumulator));
      }
      std::string strip = fresh("dot_k");
      std::string vl = fresh("dot_vl");
      auto emitChunk = [&](llvm::StringRef coordinate) -> mlir::LogicalResult {
        llvm::DenseMap<mlir::Value, std::string> rhsAxes;
        rhsAxes[decision.reductionAxis] = coordinate.str();
        std::optional<std::string> rhs =
            projectBlockScalar(rhsLoad.getPointer(), rhsAxes);
        if (!rhs)
          return dot.emitError(
              "RVV row microtile RHS projection is unavailable");
        std::string rhsVector = fresh("dot_rhs");
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
          std::string lhsVector = fresh("dot_lhs");
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
      if (decision.physical.kUnroll == 1) {
        line("const size_t " + vl + " = __riscv_vsetvl_e32m" +
             std::to_string(*lmul) + "(" + extent + " - " + strip +
             ");");
        if (mlir::failed(emitChunk(strip)))
          return mlir::failure();
        line(strip + " += " + vl + ";");
      } else {
        std::string remaining = fresh("dot_remaining");
        line("const size_t " + remaining + " = " + extent + " - " + strip +
             ";");
        line("if (" + remaining + " >= " +
             std::to_string(decision.physical.kUnroll) + ") {");
        ++indent;
        line("const size_t " + vl + " = __riscv_vsetvl_e32m" +
             std::to_string(*lmul) + "(" + remaining + " / " +
             std::to_string(decision.physical.kUnroll) + ");");
        for (unsigned unroll = 0; unroll < decision.physical.kUnroll; ++unroll) {
          std::string coordinate =
              unroll == 0
                  ? strip
                  : "(" + strip + " + " + std::to_string(unroll) + " * " + vl +
                        ")";
          if (mlir::failed(emitChunk(coordinate)))
            return mlir::failure();
        }
        line(strip + " += " + std::to_string(decision.physical.kUnroll) + " * " + vl +
             ";");
        --indent;
        line("} else {");
        ++indent;
        line("const size_t " + vl + " = __riscv_vsetvl_e32m" +
             std::to_string(*lmul) + "(" + remaining + ");");
        if (mlir::failed(emitChunk(strip)))
          return mlir::failure();
        line(strip + " += " + vl + ";");
        --indent;
        line("}");
      }
      --indent;
      line("}");
      for (unsigned lane = 0; lane < decision.rowTile; ++lane) {
        std::string seed = fresh("dot_seed");
        std::string reduced = fresh("dot_reduced");
        line("vfloat32m1_t " + seed +
             " = __riscv_vfmv_v_f_f32m1(0.0f, 1);");
        line("vfloat32m1_t " + reduced +
             " = __riscv_vfredusum_vs_" + vectorSuffix + "_f32m1(" +
             accumulators[lane] + ", " + seed + ", " + fullVL + ");");
        line(storage + "[" + std::to_string(lane) + "] = " + init.spelling +
             "[" + std::to_string(lane) +
             "] + __riscv_vfmv_f_s_f32m1_f32(" + reduced + ");");
      }
      return mlir::success();
    };

    std::string fullTile = fresh("dot_full_tile");
    line("const bool " + fullTile + " = " +
         llvm::join(lhsActive, " && ") + ";");
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

    if (mlir::failed(markRematerializedBlockTrees(
            {dot.getLhs(), dot.getRhs(), dot.getInit()}, dot.getOperation())))
      return mlir::failure();
    loweredBlockOps.insert(dot.getOperation());
    values[dot.getResult()] = CValue{dot.getResult().getType(),
                                    CValueKind::F32BlockStorage, storage};
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
      std::string name = fresh("load");
      const PhysicalValueDecision *valueShape =
          findVLAValueDecision(op.getResult());
      std::string shapeSuffix =
          valueShape ? rvvShapeSuffix(valueShape->shape) : std::string{};
      if (!valueShape || shapeSuffix.empty())
        return op.emitError("VLA load has no selected vector shape");
      if (decision->memoryMode == VLAMemoryMode::Indexed) {
        const PhysicalValueDecision *indexShape =
            findVLAValueDecision(decision->indexedOffset);
        const PhysicalHandoffDecision *indexHandoff =
            findVLAHandoff(op.getOperation(), decision->indexedOffset);
        if (!indexShape || !indexHandoff ||
            indexHandoff->kind != PhysicalHandoff::Share ||
            indexHandoff->sourceShape != indexShape->shape ||
            indexHandoff->resultShape != indexShape->shape)
          return op.emitError(
              "indexed VLA load physical handoff is incomplete");
      }
      unsigned elementSEW = valueShape->shape.sew;
      if (decision->memoryMode == VLAMemoryMode::Segment2) {
        const VLASegment2Decision *segment =
            findSegment2AccessDecision(op.getOperation());
        if (!segment)
          return op.emitError("segment2 load has no owning physical decision");
        if (segment->emission != op.getOperation())
          return mlir::success();
        if (segment->kind != VLASegment2AccessKind::Load || !f32 ||
            elementSEW != 32)
          return op.emitError("segment2 load decision has no intrinsic-C spelling");
        auto field0 = mlir::cast<LoadOp>(segment->field0);
        auto field1 = mlir::cast<LoadOp>(segment->field1);
        llvm::DenseMap<mlir::Value, std::string> axes;
        axes[activeVLADecision->coordinate] = "0";
        std::optional<std::string> base =
            projectBlockScalar(segment->base, axes);
        CValue coordinate = require(activeVLADecision->coordinate);
        if (!base ||
            coordinate.kind != CValueKind::Coordinate || coordinate.spelling.empty())
          return op.emitError("segment2 load base is unavailable");
        std::string tuple = fresh("segment2");
        std::string first = fresh("segment2_field0");
        std::string second = fresh("segment2_field1");
        std::string suffix = "f" + shapeSuffix;
        line("vfloat" + shapeSuffix + "x2_t " + tuple +
             " = __riscv_vlseg2e32_v_" + suffix + "x2(" + *base +
             " + 2 * " + coordinate.spelling + ", " + activeVL + ");");
        line("vfloat" + shapeSuffix + "_t " + first +
             " = __riscv_vget_v_" + suffix + "x2_" + suffix + "(" + tuple +
             ", 0);");
        line("vfloat" + shapeSuffix + "_t " + second +
             " = __riscv_vget_v_" + suffix + "x2_" + suffix + "(" + tuple +
             ", 1);");
        values[field0.getResult()] =
            CValue{field0.getResult().getType(), CValueKind::F32Vector, first};
        values[field1.getResult()] =
            CValue{field1.getResult().getType(), CValueKind::F32Vector, second};
        consumed.insert(segment->field0);
        consumed.insert(segment->field1);
        return mlir::success();
      }
      std::string element = f32   ? "f"
                            : f16 ? "f"
                            : u8  ? "u"
                                  : "u";
      std::string vectorType =
          std::string(f32 || f16 ? "vfloat" : "vuint") + shapeSuffix + "_t";
      std::string suffix = element + shapeSuffix;
      auto emitRead = [&]() {
        if (decision->memoryMode == VLAMemoryMode::UnitStride) {
          line(name + " = __riscv_vle" +
               std::to_string(elementSEW) +
               "_v_" + suffix + "(" + pointer.spelling + ", " + activeVL +
               ");");
        } else if (decision->memoryMode == VLAMemoryMode::Strided) {
          std::string cType = f32   ? "float"
                              : f16 ? "_Float16"
                              : u8  ? "uint8_t"
                                    : "uint32_t";
          line(name + " = __riscv_vlse" +
               std::to_string(elementSEW) +
               "_v_" + suffix + "(" + pointer.spelling +
               ", (ptrdiff_t)(sizeof(" + cType +
               ") * (" + pointer.laneStride + ")), " + activeVL + ");");
        } else if (decision->memoryMode == VLAMemoryMode::Indexed) {
          const PhysicalValueDecision *indexShape =
              findVLAValueDecision(decision->indexedOffset);
          std::optional<unsigned> indexLMUL =
              indexShape ? rvvIntegerLMUL(indexShape->shape) : std::nullopt;
          if (!indexShape ||
              (indexShape->shape.sew != 32 && indexShape->shape.sew != 64) ||
              !indexLMUL)
            return false;
          std::string indexWidth = std::to_string(indexShape->shape.sew);
          std::string offsets = fresh("byte_offsets");
          std::string indexSuffix =
              "u" + indexWidth + "m" + std::to_string(*indexLMUL);
          line("vuint" + indexWidth + "m" +
               std::to_string(*indexLMUL) + "_t " +
               offsets + " = __riscv_vmul_vx_" + indexSuffix + "(" +
               pointer.indexSpelling + ", (uint" + indexWidth + "_t)sizeof(" +
               std::string(f32 ? "float" : f16 ? "_Float16" : u8 ? "uint8_t"
                                                                        : "uint32_t") +
               "), " + activeVL + ");");
          line(name + " = __riscv_vluxei" + indexWidth + "_v_" + suffix + "(" +
               pointer.spelling + ", " + offsets + ", " + activeVL + ");");
        } else {
          return false;
        }
        return true;
      };
      line(vectorType + " " + name + ";");
      std::string logicalValidity;
      bool carriesLogicalValidity =
          decision->inactiveLane ==
          VLAInactiveLaneRealization::ZeroCarrierWithLogicalValidity;
      if (decision->activityMode == VLAActivityMode::PredicateMask) {
        CValue predicate = require(decision->predicate);
        CValue other = require(op.getOther());
        if (predicate.kind != CValueKind::Mask || predicate.spelling.empty())
          return op.emitError(
              "masked VLA load requires a selected physical predicate mask");
        if (!carriesLogicalValidity &&
            (other.kind != CValueKind::Scalar || other.spelling.empty()))
          return op.emitError(
              "predicated VLA load requires an explicit scalar passthrough");
        std::string broadcast = f32 || f16 ? "__riscv_vfmv_v_f_"
                                          : "__riscv_vmv_v_x_";
        std::string passthrough = carriesLogicalValidity
                                      ? (f32 || f16 ? "0.0f" : "0")
                                      : other.spelling;
        line(name + " = " + broadcast + suffix + "(" + passthrough + ", " +
             activeVL + ");");
        std::string intrinsic;
        std::string arguments;
        if (decision->memoryMode == VLAMemoryMode::UnitStride) {
          intrinsic = "__riscv_vle" + std::to_string(elementSEW) + "_v_" +
                      suffix + "_tumu";
          arguments = predicate.spelling + ", " + name + ", " +
                      pointer.spelling + ", " + activeVL;
        } else if (decision->memoryMode == VLAMemoryMode::Strided) {
          std::string cType = f32   ? "float"
                              : f16 ? "_Float16"
                              : u8  ? "uint8_t"
                                    : "uint32_t";
          intrinsic = "__riscv_vlse" + std::to_string(elementSEW) + "_v_" +
                      suffix + "_tumu";
          arguments = predicate.spelling + ", " + name + ", " +
                      pointer.spelling + ", (ptrdiff_t)(sizeof(" + cType +
                      ") * (" + pointer.laneStride + ")), " + activeVL;
        } else {
          const PhysicalValueDecision *indexShape =
              findVLAValueDecision(decision->indexedOffset);
          std::optional<unsigned> indexLMUL =
              indexShape ? rvvIntegerLMUL(indexShape->shape) : std::nullopt;
          if (!indexShape ||
              (indexShape->shape.sew != 32 && indexShape->shape.sew != 64) ||
              !indexLMUL)
            return op.emitError("masked indexed load has no physical index shape");
          std::string indexWidth = std::to_string(indexShape->shape.sew);
          std::string offsets = fresh("byte_offsets");
          line("vuint" + indexWidth + "m" + std::to_string(*indexLMUL) +
               "_t " + offsets + " = __riscv_vmul_vx_u" + indexWidth + "m" +
               std::to_string(*indexLMUL) + "(" + pointer.indexSpelling +
               ", (uint" + indexWidth + "_t)sizeof(" +
               std::string(f32 ? "float" : f16 ? "_Float16" : u8 ? "uint8_t"
                                                                        : "uint32_t") +
               "), " + activeVL + ");");
          intrinsic = "__riscv_vluxei" + indexWidth + "_v_" + suffix +
                      "_tumu";
          arguments = predicate.spelling + ", " + name + ", " +
                      pointer.spelling + ", " + offsets + ", " + activeVL;
        }
        line(name + " = " + intrinsic + "(" + arguments + ");");
        if (carriesLogicalValidity)
          logicalValidity = predicate.spelling;
      } else if (decision->activityMode == VLAActivityMode::ScalarPredicate) {
        CValue predicate = require(decision->predicate);
        CValue other = require(op.getOther());
        if (predicate.kind != CValueKind::Scalar || predicate.spelling.empty())
          return op.emitError(
              "scalar-predicated VLA load requires a selected scalar predicate");
        if (!carriesLogicalValidity &&
            (other.kind != CValueKind::Scalar || other.spelling.empty()))
          return op.emitError(
              "scalar-predicated VLA load requires an explicit passthrough");
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
        std::string passthrough = carriesLogicalValidity
                                      ? (f32 || f16 ? "0.0f" : "0")
                                      : other.spelling;
        line(name + " = " + broadcast + suffix + "(" + passthrough + ", " +
             activeVL + ");");
        --indent;
        line("}");
        if (carriesLogicalValidity) {
          if (!activePhysicalEntity || activePhysicalEntity->vlaMaskRatio == 0)
            return op.emitError(
                "logical validity has no selected physical mask shape");
          std::string ratio =
              std::to_string(activePhysicalEntity->vlaMaskRatio);
          logicalValidity = fresh("valid");
          line("vbool" + ratio + "_t " + logicalValidity + " = " +
               predicate.spelling + " ? __riscv_vmset_m_b" + ratio + "(" +
               activeVL + ") : __riscv_vmclr_m_b" + ratio + "(" + activeVL +
               ");");
        }
      } else {
        if (!emitRead())
          return op.emitError(
              "indexed VLA load decision has no intrinsic-C spelling");
      }
      CValue result{op.getResult().getType(),
                    f32   ? CValueKind::F32Vector
                    : f16 ? CValueKind::F16Vector
                    : u8  ? CValueKind::U8Vector
                          : CValueKind::U32Vector,
                    name};
      result.logicalValidity = std::move(logicalValidity);
      if (mlir::isa<MaskedType>(op.getResult().getType()) !=
          !result.logicalValidity.empty())
        return op.emitError(
            "VLA load physical validity disagrees with its Kernel IR result");
      values[op.getResult()] = std::move(result);
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

  mlir::LogicalResult emitScalarLoadF16LE(LoadF16LEOp op) {
    auto selected = physicalPlan.f16LELoads.find(op.getOperation());
    if (selected == physicalPlan.f16LELoads.end() ||
        selected->second.realization.realization !=
            LoadF16LERealization::ScalarBytes)
      return op.emitError("scalar load_f16_le has no physical decision");
    if (mlir::failed(requireEntityPlan(op.getOperation(), selected->second)))
      return mlir::failure();
    CValue base = require(op.getBase());
    if (base.kind != CValueKind::Pointer || base.spelling.empty())
      return op.emitError("scalar load_f16_le base pointer is unavailable");
    std::string result = fresh("f16_le");
    line("const float " + result + " = __weft_load_f16_le(" +
         base.spelling + ");");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, result};
    return mlir::success();
  }

  mlir::LogicalResult emitVLALookup(LookupOp op) {
    const VLALookupDecision *decision =
        findLookupDecision(op.getOperation());
    if (!decision ||
        decision->realization !=
            VLALookupRealization::RVVF32Table16GatherE16 ||
        decision->tableExtent != 16)
      return op.emitError("VLA lookup has no complete physical decision");
    CValue indices = require(op.getIndices());
    std::optional<std::string> tableBase =
        projectLocalBlockMemoryBase(decision->table);
    if (indices.kind != CValueKind::U8Vector || indices.spelling.empty() ||
        !tableBase)
      return op.emitError("VLA lookup operands are unavailable");
    const PhysicalHandoffDecision *indexHandoff =
        findVLAHandoff(op.getOperation(), op.getIndices());
    const PhysicalHandoffDecision *tableHandoff =
        findVLAHandoff(op.getOperation(), op.getTable());
    const PhysicalValueDecision *codeValue =
        findVLAValueDecision(op.getIndices());
    const PhysicalValueDecision *tableValue =
        findVLAValueDecision(op.getTable());
    const PhysicalValueDecision *resultValue =
        findVLAValueDecision(op.getResult());
    auto findTemporary = [&](unsigned sew) -> const PhysicalTemporaryDecision * {
      if (!activePhysicalEntity)
        return nullptr;
      auto found = llvm::find_if(
          activePhysicalEntity->temporaries,
          [&](const PhysicalTemporaryDecision &temporary) {
            return temporary.owner == op.getOperation() &&
                   temporary.shape.sew == sew;
          });
      return found == activePhysicalEntity->temporaries.end() ? nullptr : &*found;
    };
    const PhysicalTemporaryDecision *index16 = findTemporary(16);
    const PhysicalTemporaryDecision *index32 = findTemporary(32);
    if (!indexHandoff || indexHandoff->kind != PhysicalHandoff::Convert ||
        !codeValue || codeValue->shape != indexHandoff->sourceShape ||
        indexHandoff->sourceShape != codeValue->shape ||
        !index32 || indexHandoff->resultShape != index32->shape ||
        !tableHandoff || tableHandoff->kind != PhysicalHandoff::Reload ||
        !tableValue || tableValue->shape != index32->shape ||
        tableHandoff->sourceShape != tableValue->shape ||
        tableHandoff->resultShape != tableValue->shape ||
        !resultValue || resultValue->shape != index32->shape || !index16)
      return op.emitError("VLA lookup handoff plan is incomplete");
    std::string index16Suffix = rvvShapeSuffix(index16->shape);
    std::string index32Suffix = rvvShapeSuffix(index32->shape);
    std::string tableSuffix = rvvShapeSuffix(tableValue->shape);
    std::string resultSuffix = rvvShapeSuffix(resultValue->shape);
    if (index16->shape.sew != 16 || index32->shape.sew != 32 ||
        tableValue->shape.sew != 32 || resultValue->shape.sew != 32 ||
        index16Suffix.empty() ||
        index32Suffix.empty() || tableSuffix.empty() || resultSuffix.empty())
      return op.emitError("VLA lookup shapes have no RVV spelling");
    std::string tableVL = fresh("lookup_table_vl");
    std::string table = fresh("lookup_table");
    std::string indices16 = fresh("lookup_u16");
    std::string indices32 = fresh("lookup_u32");
    std::string gathered = fresh("lookup");
    line("const size_t " + tableVL + " = __riscv_vsetvl_e" + tableSuffix +
         "(" + std::to_string(decision->tableExtent) + ");");
    line("vfloat" + tableSuffix + "_t " + table +
         " = __riscv_vle32_v_f" + tableSuffix + "(" + *tableBase +
         ", " + tableVL + ");");
    line("vuint" + index16Suffix + "_t " + indices16 +
         " = __riscv_vzext_vf2_u" + index16Suffix + "(" + indices.spelling +
         ", " + activeVL + ");");
    line("vuint" + index32Suffix + "_t " + indices32 +
         " = __riscv_vzext_vf2_u" + index32Suffix + "(" + indices16 + ", " +
         activeVL + ");");
    line("vfloat" + resultSuffix + "_t " + gathered +
         " = __riscv_vrgather_vv_f" + resultSuffix + "(" + table + ", " +
         indices32 + ", " + activeVL + ");");
    CValue result{op.getResult().getType(), CValueKind::F32Vector, gathered};
    result.vectorSEW = resultValue->shape.sew;
    result.vectorLMUL = rvvIntegerLMUL(resultValue->shape).value_or(0);
    values[op.getResult()] = std::move(result);
    return mlir::success();
  }

  mlir::LogicalResult emitStore(StoreOp op) {
    CValue value = require(op.getValue());
    if (inVLA && value.kind == CValueKind::F32VectorBundle) {
      const VLAAccessDecision *decision =
          findAccessDecision(op.getOperation());
      const PhysicalHandoffDecision *handoff =
          findVLAHandoff(op.getOperation(), op.getValue());
      std::optional<unsigned> selectedLMUL =
          handoff ? rvvIntegerLMUL(handoff->resultShape) : std::nullopt;
      if (!decision || !handoff || !selectedLMUL ||
          !decision->elementType.isF32() || !decision->bundleAxis ||
          decision->bundleVectors != value.fields.size() ||
          decision->activityMode == VLAActivityMode::PredicateMask ||
          (decision->memoryMode != VLAMemoryMode::UnitStride &&
           decision->memoryMode != VLAMemoryMode::Strided))
        return op.emitError(
            "VLA structured store has no selected bundle handoff");
      std::string suffix = "f32m" + std::to_string(*selectedLMUL);
      for (unsigned lane = 0; lane < decision->bundleVectors; ++lane) {
        const CValue &field = value.fields[lane];
        llvm::DenseMap<mlir::Value, std::string> axes;
        axes[decision->bundleAxis] = std::to_string(lane);
        std::optional<std::string> pointer =
            projectBlockScalar(op.getPointer(), axes);
        std::optional<std::string> predicate =
            projectBlockScalar(op.getWhere(), axes);
        std::optional<std::string> stride = projectVLALaneStride(
            op.getPointer(), activeVLADecision->coordinate, axes);
        if (field.kind != CValueKind::F32Vector || field.spelling.empty() ||
            !pointer || !predicate || !stride)
          return op.emitError(
              "VLA structured store projection is unavailable");
        line("if (" + *predicate + ") {");
        ++indent;
        if (decision->memoryMode == VLAMemoryMode::UnitStride)
          line("__riscv_vse32_v_" + suffix + "(" + *pointer + ", " +
               field.spelling + ", " + activeVL + ");");
        else
          line("__riscv_vsse32_v_" + suffix + "(" + *pointer +
               ", (ptrdiff_t)(sizeof(float) * (" + *stride + ")), " +
               field.spelling + ", " + activeVL + ");");
        --indent;
        line("}");
      }
      if (mlir::failed(markRematerializedBlockTrees(
              {op.getPointer(), op.getWhere()}, op.getOperation())))
        return mlir::failure();
      return mlir::success();
    }
    CValue pointer = require(op.getPointer());
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
      const PhysicalHandoffDecision *handoff =
          findVLAHandoff(op.getOperation(), op.getValue());
      std::string shapeSuffix =
          handoff ? rvvShapeSuffix(handoff->resultShape) : std::string{};
      if (!handoff || shapeSuffix.empty())
        return op.emitError("VLA store has no physical value handoff");
      unsigned elementSEW = handoff->resultShape.sew;

      std::string vector = value.spelling;
      CValueKind expectedKind = f32   ? CValueKind::F32Vector
                                : f16 ? CValueKind::F16Vector
                                : i8  ? CValueKind::I8Vector
                                      : CValueKind::U8Vector;
      if (decision->storeValueMode == VLAStoreValueMode::ScalarBroadcast) {
        if (value.kind != CValueKind::Scalar || value.spelling.empty())
          return op.emitError("VLA store scalar broadcast is unavailable");
        vector = fresh("store_value");
        std::string suffix =
            std::string(f32 || f16 ? "f" : i8 ? "i" : "u") +
            shapeSuffix;
        std::string vectorType =
            std::string(f32 || f16 ? "vfloat" : i8 ? "vint" : "vuint") +
            shapeSuffix + "_t";
        std::string broadcast = f32 || f16 ? "__riscv_vfmv_v_f_"
                                          : "__riscv_vmv_v_x_";
        line(vectorType + " " + vector + " = " + broadcast + suffix + "(" +
             value.spelling + ", " + activeVL + ");");
      } else if (value.kind != expectedKind || value.spelling.empty()) {
        if (decision->memoryMode != VLAMemoryMode::Segment2)
          return op.emitError("VLA store vector projection is unavailable");
      }

      if (decision->memoryMode == VLAMemoryMode::Segment2) {
        const VLASegment2Decision *segment =
            findSegment2AccessDecision(op.getOperation());
        if (!segment)
          return op.emitError("segment2 store has no owning physical decision");
        if (segment->emission != op.getOperation())
          return mlir::success();
        if (segment->kind != VLASegment2AccessKind::Store || !f32 ||
            elementSEW != 32)
          return op.emitError("segment2 store decision has no intrinsic-C spelling");
        auto field0 = mlir::cast<StoreOp>(segment->field0);
        auto field1 = mlir::cast<StoreOp>(segment->field1);
        CValue first = require(field0.getValue());
        CValue second = require(field1.getValue());
        llvm::DenseMap<mlir::Value, std::string> axes;
        axes[activeVLADecision->coordinate] = "0";
        std::optional<std::string> base =
            projectBlockScalar(segment->base, axes);
        CValue coordinate = require(activeVLADecision->coordinate);
        if (first.kind != CValueKind::F32Vector ||
            second.kind != CValueKind::F32Vector || first.spelling.empty() ||
            second.spelling.empty() || !base ||
            coordinate.kind != CValueKind::Coordinate ||
            coordinate.spelling.empty())
          return op.emitError() << "segment2 store operands are unavailable "
                                << static_cast<int>(first.kind) << "/"
                                << static_cast<int>(second.kind) << " names="
                                << first.spelling << "/" << second.spelling;
        std::string tuple = fresh("segment2_store");
        std::string suffix = "f" + shapeSuffix;
        line("vfloat" + shapeSuffix + "x2_t " + tuple +
             " = __riscv_vcreate_v_" + suffix + "x2(" + first.spelling + ", " +
             second.spelling + ");");
        line("__riscv_vsseg2e32_v_" + suffix + "x2(" + *base +
             " + 2 * " + coordinate.spelling + ", " + tuple + ", " + activeVL +
             ");");
        consumed.insert(segment->field0);
        consumed.insert(segment->field1);
        return mlir::success();
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
      std::string sew = std::to_string(elementSEW);
      std::string suffix =
          std::string(f32 || f16 ? "f" : i8 ? "i" : "u") + shapeSuffix;
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
      } else if (decision->memoryMode == VLAMemoryMode::Strided &&
                 decision->activityMode != VLAActivityMode::PredicateMask) {
        line("__riscv_vsse" + sew + "_v_" + suffix + "(" + pointer.spelling +
             ", (ptrdiff_t)(sizeof(" + elementCType + ") * (" +
             pointer.laneStride + ")), " + vector + ", " + activeVL + ");");
      } else if (decision->memoryMode == VLAMemoryMode::Strided) {
        line("__riscv_vsse" + sew + "_v_" + suffix + "_m(" +
             predicate.spelling + ", " + pointer.spelling +
             ", (ptrdiff_t)(sizeof(" + elementCType + ") * (" +
             pointer.laneStride + ")), " + vector + ", " + activeVL + ");");
      } else
        return op.emitError("VLA store memory decision has no intrinsic spelling");
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

  mlir::Value containingVLACoordinate(mlir::Value value) const {
    mlir::Operation *anchor = value.getDefiningOp();
    if (!anchor) {
      auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
      anchor = argument ? argument.getOwner()->getParentOp() : nullptr;
    }
    if (!anchor)
      return {};
    VLAOp vla = mlir::dyn_cast<VLAOp>(anchor);
    if (!vla)
      vla = anchor->getParentOfType<VLAOp>();
    return vla ? vla.getBody().front().getArgument(0) : mlir::Value{};
  }

  bool valueDependsOnAxis(mlir::Value value, mlir::Value axis) const {
    auto found = kernelFacts.values.find(value);
    return found != kernelFacts.values.end() &&
           llvm::is_contained(found->second.axisDependencies, axis);
  }

  LoadOp reloadableLoad(mlir::Value value) const {
    auto found = kernelFacts.values.find(value);
    return found == kernelFacts.values.end() || !found->second.reloadSource
               ? LoadOp{}
               : mlir::dyn_cast<LoadOp>(found->second.reloadSource);
  }

  const MemoryAccessFact *memoryFact(mlir::Operation *operation) const {
    auto found = kernelFacts.memory.find(operation);
    return found == kernelFacts.memory.end() ? nullptr : &found->second;
  }

  LaneRelation memoryRelation(const MemoryAccessFact &access,
                              mlir::Value axis) const {
    auto found = access.axisRelations.find(axis);
    return found == access.axisRelations.end() ? LaneRelation::Independent
                                               : found->second;
  }

  std::optional<StructuredProductFacts>
  structuredProductFacts(mlir::Value lhs, mlir::Value rhs,
                         mlir::Value result) const {
    auto lhsFact = kernelFacts.values.find(lhs);
    auto rhsFact = kernelFacts.values.find(rhs);
    auto resultFact = kernelFacts.values.find(result);
    if (lhsFact == kernelFacts.values.end() ||
        rhsFact == kernelFacts.values.end() ||
        resultFact == kernelFacts.values.end())
      return std::nullopt;
    StructuredProductFacts facts;
    facts.lhsAxes = lhsFact->second.logicalAxes;
    facts.rhsAxes = rhsFact->second.logicalAxes;
    facts.resultAxes = resultFact->second.logicalAxes;
    for (mlir::Value axis : facts.lhsAxes) {
      bool rhsHas = llvm::is_contained(facts.rhsAxes, axis);
      bool resultHas = llvm::is_contained(facts.resultAxes, axis);
      if (rhsHas && !resultHas)
        facts.reductionAxes.push_back(axis);
      else if (resultHas && !rhsHas)
        facts.lhsFreeAxes.push_back(axis);
    }
    for (mlir::Value axis : facts.rhsAxes) {
      bool lhsHas = llvm::is_contained(facts.lhsAxes, axis);
      bool resultHas = llvm::is_contained(facts.resultAxes, axis);
      if (resultHas && !lhsHas)
        facts.rhsFreeAxes.push_back(axis);
    }
    for (mlir::Value axis : facts.resultAxes) {
      if (!llvm::is_contained(facts.lhsAxes, axis))
        facts.lhsBroadcastAxes.push_back(axis);
      if (!llvm::is_contained(facts.rhsAxes, axis))
        facts.rhsBroadcastAxes.push_back(axis);
    }
    return facts;
  }

  llvm::SmallVector<VLAValueLifetimeSnapshot>
  vlaValueLifetimeSnapshots(VLAOp owner) const {
    mlir::Value coordinate = owner.getBody().front().getArgument(0);
    llvm::SmallVector<VLAValueLifetimeSnapshot> snapshots;
    std::function<void(mlir::Block &)> analyzeBlock =
        [&](mlir::Block &block) {
      llvm::SmallVector<mlir::Operation *> operations;
      for (mlir::Operation &operation : block.without_terminator())
        operations.push_back(&operation);
      for (mlir::Operation *operation : operations)
        for (mlir::Region &region : operation->getRegions())
          for (mlir::Block &nested : region)
            analyzeBlock(nested);
      if (operations.empty())
        return;

      llvm::DenseMap<mlir::Operation *, unsigned> positions;
      for (auto [position, operation] : llvm::enumerate(operations))
        positions.try_emplace(operation, static_cast<unsigned>(position));
      llvm::SmallVector<VLAValueLifetimeSnapshot> blockSnapshots(
          operations.size());
      for (const auto &entry : kernelFacts.values) {
        const ValueUseFact &fact = entry.second;
        if (!isRegionValue(fact.value.getType()) ||
            containingVLACoordinate(fact.value) != coordinate)
          continue;
        mlir::Type element = elementType(fact.value.getType());
        enum class Kind { F32, F16, Index, Byte, U32, Mask } kind;
        if (element.isF32())
          kind = Kind::F32;
        else if (isF16(element))
          kind = Kind::F16;
        else if (element.isIndex()) {
          if (classifyLaneRelation(fact.value, coordinate) !=
              LaneRelation::Indexed)
            continue;
          kind = Kind::Index;
        }
        else if (element.isSignedInteger(8) ||
                 element.isUnsignedInteger(8))
          kind = Kind::Byte;
        else if (element.isUnsignedInteger(32))
          kind = Kind::U32;
        else if (element.isInteger(1))
          kind = Kind::Mask;
        else
          continue;
        mlir::Operation *definition = fact.value.getDefiningOp();
        mlir::Block *definitionBlock = definition
                                           ? definition->getBlock()
                                           : mlir::cast<mlir::BlockArgument>(
                                                 fact.value)
                                                 .getOwner();
        if (definitionBlock != &block) {
          mlir::Operation *nestedDefinition =
              definition ? definition : definitionBlock->getParentOp();
          while (nestedDefinition && nestedDefinition->getBlock() != &block)
            nestedDefinition = nestedDefinition->getParentOp();
          if (nestedDefinition)
            continue;
        }
        unsigned begin = 0;
        bool participates = false;
        if (definitionBlock == &block) {
          if (definition) {
            auto found = positions.find(definition);
            if (found != positions.end()) {
              begin = found->second;
              participates = true;
            }
          } else {
            participates = true;
          }
        }
        unsigned end = begin;
        for (mlir::Operation *consumer : fact.consumers) {
          mlir::Operation *anchor = consumer;
          while (anchor && anchor->getBlock() != &block)
            anchor = anchor->getParentOp();
          if (!anchor)
            continue;
          participates = true;
          auto found = positions.find(anchor);
          end = found == positions.end()
                    ? static_cast<unsigned>(operations.size() - 1)
                    : std::max(end, found->second);
        }
        if (!participates)
          continue;
        for (unsigned position = begin; position <= end; ++position) {
          VLAValueLifetimeSnapshot &snapshot = blockSnapshots[position];
          switch (kind) {
          case Kind::F32:
            ++snapshot.f32;
            break;
          case Kind::F16:
            ++snapshot.f16;
            break;
          case Kind::Index:
            ++snapshot.index;
            break;
          case Kind::Byte:
            ++snapshot.byte;
            break;
          case Kind::U32:
            ++snapshot.u32;
            break;
          case Kind::Mask:
            ++snapshot.mask;
            break;
          }
        }
      }
      snapshots.append(blockSnapshots.begin(), blockSnapshots.end());
    };
    analyzeBlock(owner.getBody().front());
    return snapshots;
  }

  mlir::Value pointerRoot(mlir::Value value) const {
    if (mlir::isa<PtrType>(value.getType()))
      return value;
    auto pointer = value.getDefiningOp<PtrAddOp>();
    if (!pointer)
      return {};
    return pointerRoot(pointer.getBase());
  }

  std::optional<int64_t> physicalExtent(mlir::Value value) {
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
  }

  void collectBlockAxes(mlir::Value value,
                        llvm::DenseSet<mlir::Value> &visited,
                        llvm::SmallVectorImpl<BlockIndexOp> &axes) const {
    if (!value || !visited.insert(value).second)
      return;
    if (auto axis = value.getDefiningOp<BlockIndexOp>()) {
      if (!llvm::is_contained(axes, axis))
        axes.push_back(axis);
      return;
    }
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      auto loop = mlir::dyn_cast_or_null<ForOp>(
          argument.getOwner()->getParentOp());
      if (loop && argument.getArgNumber() > 0 &&
          argument.getArgNumber() - 1 < loop.getInitArgs().size())
        collectBlockAxes(loop.getInitArgs()[argument.getArgNumber() - 1],
                         visited, axes);
      return;
    }
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition)
      return;
    for (mlir::Value operand : definition->getOperands())
      collectBlockAxes(operand, visited, axes);
  }

  llvm::SmallVector<BlockIndexOp> collectBlockAxes(mlir::Value value) const {
    llvm::DenseSet<mlir::Value> visited;
    llvm::SmallVector<BlockIndexOp> axes;
    collectBlockAxes(value, visited, axes);
    return axes;
  }

  BlockIndexOp findBlockAxisDefinition(int64_t identity) {
    auto found = kernelFacts.blockAxes.find(identity);
    return found == kernelFacts.blockAxes.end()
               ? BlockIndexOp{}
               : found->second.getDefiningOp<BlockIndexOp>();
  }

  llvm::SmallVector<BlockIndexOp> blockAxesForType(mlir::Type type) {
    llvm::SmallVector<BlockIndexOp> axes;
    for (int64_t identity : logicalAxisIds(type)) {
      if (identity <= 0)
        continue;
      auto found = kernelFacts.blockAxes.find(identity);
      BlockIndexOp axis = found == kernelFacts.blockAxes.end()
                              ? BlockIndexOp{}
                              : found->second.getDefiningOp<BlockIndexOp>();
      if (axis && !llvm::is_contained(axes, axis))
        axes.push_back(axis);
    }
    return axes;
  }

  std::optional<int64_t> physicalBlockElementCount(mlir::Type type) {
    auto block = mlir::dyn_cast<BlockType>(unwrapLogicalValidity(type));
    if (!block || !block.getElementType().isF32() || block.getShape().empty() ||
        block.getShape().size() > 2)
      return std::nullopt;
    int64_t elements = 1;
    for (auto [dimension, identity] :
         llvm::zip(block.getShape(), block.getAxisIds())) {
      std::optional<int64_t> extent;
      if (dimension > 0)
        extent = dimension;
      else if (identity > 0) {
        BlockIndexOp axis = findBlockAxisDefinition(identity);
        if (axis)
          extent = physicalExtent(axis.getExtent());
      }
      if (!extent || *extent <= 0 ||
          elements > options.target.maxPrivateStackBytes /
                         static_cast<int64_t>(sizeof(float)) / *extent)
        return std::nullopt;
      elements *= *extent;
    }
    return elements;
  }

  mlir::LogicalResult prepareMaterializedBlockValues() {
    bool failed = false;
    kernel.walk([&](FullOp full) {
      if (failed)
        return;
      auto block = mlir::dyn_cast<BlockType>(full.getResult().getType());
      if (!block || !block.getElementType().isF32())
        return;
      std::optional<int64_t> elements =
          physicalBlockElementCount(full.getResult().getType());
      if (!elements) {
        full.emitError(
            "f32 block constructor has no bounded compiler-private storage realization");
        failed = true;
        return;
      }
      materializedBlockElements.try_emplace(full.getOperation(), *elements);
    });
    kernel.walk([&](mlir::Operation *operation) {
      if (failed || !mlir::isa<IfOp, ForOp, WhileOp>(operation))
        return;
      for (mlir::Value result : operation->getResults()) {
        auto block =
            mlir::dyn_cast<BlockType>(unwrapLogicalValidity(result.getType()));
        if (!block)
          continue;
        std::optional<int64_t> elements =
            physicalBlockElementCount(result.getType());
        if (!elements) {
          operation->emitError(
              "block control carry has no bounded f32 storage realization");
          failed = true;
          return;
        }
        controlBlockElements.try_emplace(result, *elements);
      }
    });
    return failed ? mlir::failure() : mlir::success();
  }

  mlir::LogicalResult selectMaterializedBlockStorage(mlir::Value value,
                                                     mlir::Operation *owner) {
    if (!physicalBlockElementCount(value.getType()))
      return owner->emitError(
          "selected local primitive has no bounded f32 accumulator domain");
    return mlir::success();
  }

  std::optional<int64_t>
  materializedF32ElementCount(mlir::Value value,
                              llvm::DenseSet<mlir::Value> &visited) const {
    if (!value || !visited.insert(value).second)
      return std::nullopt;
    if (auto selected = controlBlockElements.find(value);
        selected != controlBlockElements.end())
      return selected->second;
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
      mlir::Operation *parent = argument.getOwner()->getParentOp();
      if (auto loop = mlir::dyn_cast_or_null<ForOp>(parent);
          loop && argument.getArgNumber() > 0 &&
          argument.getArgNumber() - 1 < loop.getInitArgs().size())
        return materializedF32ElementCount(
            loop.getInitArgs()[argument.getArgNumber() - 1], visited);
      if (auto loop = mlir::dyn_cast_or_null<WhileOp>(parent);
          loop && argument.getArgNumber() < loop.getInitArgs().size())
        return materializedF32ElementCount(
            loop.getInitArgs()[argument.getArgNumber()], visited);
      return std::nullopt;
    }
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition)
      return std::nullopt;
    if (auto full = mlir::dyn_cast<FullOp>(definition)) {
      auto selected = materializedBlockElements.find(full.getOperation());
      return selected == materializedBlockElements.end()
                 ? std::nullopt
                 : std::optional<int64_t>(selected->second);
    }
    if (auto dot = mlir::dyn_cast<DotOp>(definition)) {
      auto selected = physicalPlan.dots.find(dot.getOperation());
      return selected == physicalPlan.dots.end()
                 ? std::nullopt
                 : std::optional<int64_t>(selected->second.realization.rowTile);
    }
    if (auto matmul = mlir::dyn_cast<MatmulOp>(definition)) {
      auto selected = physicalPlan.f16Matmuls.find(matmul.getOperation());
      return selected == physicalPlan.f16Matmuls.end()
                 ? std::nullopt
                 : std::optional<int64_t>(
                       selected->second.realization.rowTile *
                       selected->second.realization.columnTile);
    }
    auto pointwise = physicalPlan.materializedF32Pointwise.find(definition);
    if (pointwise != physicalPlan.materializedF32Pointwise.end())
      return pointwise->second.realization.elements;
    if (auto result = mlir::dyn_cast<mlir::OpResult>(value))
      if (mlir::isa<IfOp, ForOp, WhileOp>(result.getOwner()))
        if (auto selected = controlBlockElements.find(value);
            selected != controlBlockElements.end())
          return selected->second;
    auto block = mlir::dyn_cast<BlockType>(unwrapLogicalValidity(value.getType()));
    if (!block || !block.getElementType().isF32())
      return std::nullopt;
    int64_t elements = 1;
    for (int64_t dimension : block.getShape()) {
      if (dimension <= 0 ||
          elements > options.target.maxPrivateStackBytes /
                         static_cast<int64_t>(sizeof(float)) / dimension)
        return std::nullopt;
      elements *= dimension;
    }
    bool structuredPrimitive =
        physicalPlan.affineI4I8.contains(definition) ||
        physicalPlan.symmetricI4I8.contains(definition);
    return structuredPrimitive ? std::optional<int64_t>(elements)
                               : std::nullopt;
  }

  std::optional<int64_t> materializedF32ElementCount(mlir::Value value) const {
    llvm::DenseSet<mlir::Value> visited;
    return materializedF32ElementCount(value, visited);
  }

  bool isMaterializedF32Value(mlir::Value value) const {
    return materializedF32ElementCount(value).has_value();
  }

  mlir::LogicalResult prepareMaterializedF32Pointwise() {
    bool failed = false;
    kernel.walk([&](mlir::Operation *operation) {
      if (failed || operation->getNumResults() != 1 ||
          (!mlir::isa<BinaryOp>(operation) && !mlir::isa<UnaryOp>(operation)))
        return;
      mlir::Value result = operation->getResult(0);
      auto resultType =
          mlir::dyn_cast<BlockType>(unwrapLogicalValidity(result.getType()));
      if (!resultType || !resultType.getElementType().isF32())
        return;
      std::optional<int64_t> elements;
      for (mlir::Value operand : operation->getOperands()) {
        if (!containsBlockType(operand.getType()))
          continue;
        std::optional<int64_t> operandElements =
            materializedF32ElementCount(operand);
        if (!operandElements) {
          if (elements) {
            operation->emitError(
                "materialized pointwise operands require one physical storage domain");
            failed = true;
          }
          return;
        }
        if (elements && elements != operandElements) {
          operation->emitError(
              "materialized pointwise operands have incompatible storage extents");
          failed = true;
          return;
        }
        elements = operandElements;
      }
      if (!elements)
        return;
      if (*elements <= 0 ||
          *elements > options.target.maxPrivateStackBytes /
                          static_cast<int64_t>(sizeof(float))) {
        operation->emitError(
            "materialized pointwise result exceeds target private-storage budget");
        failed = true;
        return;
      }
      MaterializedF32PointwiseDecision decision;
      decision.operation = operation;
      decision.realization = mlir::isa<BinaryOp>(operation)
                                 ? MaterializedF32PointwiseRealization::Binary
                                 : MaterializedF32PointwiseRealization::Unary;
      decision.elements = *elements;
      PlannedPhysicalDecision<MaterializedF32PointwiseDecision> planned(
          std::move(decision));
      initializeEntityPlan(planned.entity);
      mlir::Value reusedStorage;
      for (mlir::Value operand : operation->getOperands()) {
        if (!containsBlockType(operand.getType()) ||
            materializedF32ElementCount(operand) != elements)
          continue;
        auto use = kernelFacts.values.find(operand);
        if (use == kernelFacts.values.end() || use->second.crossesRegion ||
            use->second.controlCarried || use->second.consumers.size() != 1 ||
            use->second.consumers.front() != operation)
          continue;
        reusedStorage = operand;
        break;
      }
      planned.entity.storages.push_back(PhysicalStorageDecision{
          result, reusedStorage, *elements, 16});
      if (!physicalPlan.materializedF32Pointwise
               .try_emplace(operation, std::move(planned))
               .second) {
        operation->emitError(
            "one materialized pointwise operation cannot own multiple decisions");
        failed = true;
      }
    });
    return failed ? mlir::failure() : mlir::success();
  }

  mlir::LogicalResult decideMaterializedBlockStore(StoreOp op,
                                                   mlir::Type valueType) {
    auto block = mlir::dyn_cast<BlockType>(valueType);
    if (!block || !block.getElementType().isF32())
      return op.emitError(
          "materialized f32 store requires a typed f32 block value");
    MaterializedBlockStoreDecision decision;
    decision.operation = op.getOperation();
    bool canUseContiguous16 =
        block.getShape() == llvm::ArrayRef<int64_t>({16}) &&
        options.target.supportsVLENAtLeast(128) && isTrue(op.getWhere());
    if (block.getShape().size() == 1 && !canUseContiguous16) {
      llvm::SmallVector<BlockIndexOp> axes = collectBlockAxes(op.getPointer());
      BlockIndexOp axis = findDimensionAxis(op.getPointer(), 0, axes);
      std::optional<int64_t> elements = materializedF32ElementCount(op.getValue());
      if (!axis || !elements || *elements <= 0 ||
          !isContiguousPrefixPredicate(op.getWhere(), axis.getResult()))
        return op.emitError(
            "materialized rank-one f32 store requires one typed local axis and prefix predicate");
      decision.realization = MaterializedBlockStoreRealization::ScalarRankOne;
      decision.rowAxis = axis.getResult();
      decision.columns = *elements;
    } else if (block.getShape().size() == 2) {
      llvm::SmallVector<BlockIndexOp> axes = collectBlockAxes(op.getPointer());
      BlockIndexOp rowAxis = findDimensionAxis(op.getPointer(), 0, axes);
      BlockIndexOp columnAxis = findDimensionAxis(op.getPointer(), 1, axes);
      std::optional<int64_t> rows =
          rowAxis ? physicalExtent(rowAxis.getExtent()) : std::nullopt;
      std::optional<int64_t> columns =
          columnAxis ? physicalExtent(columnAxis.getExtent()) : std::nullopt;
      if (!rowAxis || !columnAxis || !rows || !columns || *rows <= 0 ||
          *columns <= 0 ||
          !isContiguousPrefixPredicate(op.getWhere(), rowAxis.getResult()) ||
          !isContiguousPrefixPredicate(op.getWhere(), columnAxis.getResult()))
        return op.emitError(
            "materialized rank-two f32 store requires typed local axes and prefix predicates");
      decision.realization =
          MaterializedBlockStoreRealization::ScalarRankTwo;
      decision.rowAxis = rowAxis.getResult();
      decision.columnAxis = columnAxis.getResult();
      decision.rows = *rows;
      decision.columns = *columns;
    } else if (canUseContiguous16) {
      const MemoryAccessFact *access = memoryFact(op.getOperation());
      llvm::SmallVector<BlockIndexOp> axes = collectBlockAxes(op.getPointer());
      BlockIndexOp columnAxis = findDimensionAxis(op.getPointer(), 0, axes);
      if (!access || !columnAxis ||
          memoryRelation(*access, columnAxis.getResult()) !=
              LaneRelation::UnitStride)
        return op.emitError(
            "materialized f32 block store requires one unit-stride typed axis");
      decision.realization =
          MaterializedBlockStoreRealization::RVVContiguousRankOne;
      decision.columns = 16;
      decision.columnAxis = columnAxis.getResult();
    } else {
      return op.emitError(
          "materialized f32 block store has no selected target realization");
    }
    PlannedPhysicalDecision<MaterializedBlockStoreDecision> planned(
        std::move(decision));
    initializeEntityPlan(planned.entity);
    if (planned.realization.realization ==
        MaterializedBlockStoreRealization::RVVContiguousRankOne) {
      RVVVectorShape vectorShape = kRVVE32M2;
      recordPhysicalValue(planned.entity, op.getValue(),
                          vectorShape);
      recordPhysicalHandoff(planned.entity, op.getOperation(), op.getValue(),
                            PhysicalHandoff::Share, vectorShape, vectorShape);
      planned.entity.resources.valueGroups =
          rvvRegisterGroups(vectorShape);
      planned.entity.resources.memoryGroups =
          planned.entity.resources.valueGroups;
      planned.entity.resources.peakGroups =
          planned.entity.resources.valueGroups + 1;
    }
    if (!physicalPlan.materializedBlockStores
             .try_emplace(op.getOperation(), std::move(planned))
             .second)
      return op.emitError(
          "one materialized block store cannot own multiple decisions");
    return mlir::success();
  }

  BlockIndexOp findDimensionAxis(mlir::Value value, int64_t dimension,
                                llvm::ArrayRef<BlockIndexOp> axes) const {
    BlockIndexOp selected;
    for (BlockIndexOp axis : axes) {
      if (!haveSameLogicalExtent(value, dimension, axis.getResult(), 0))
        continue;
      if (selected && selected != axis)
        return {};
      selected = axis;
    }
    return selected;
  }

  bool isContiguousPrefixPredicate(mlir::Value predicate,
                                   mlir::Value coordinate) const {
    if (!dependsOn(predicate, coordinate))
      return true;
    if (auto binary = predicate.getDefiningOp<BinaryOp>())
      return binary.getKind() == "and" &&
             isContiguousPrefixPredicate(binary.getLhs(), coordinate) &&
             isContiguousPrefixPredicate(binary.getRhs(), coordinate);
    auto compare = predicate.getDefiningOp<CompareOp>();
    return compare &&
           (compare.getPredicate() == "lt" || compare.getPredicate() == "le") &&
           classifyLaneRelation(compare.getLhs(), coordinate) ==
               LaneRelation::UnitStride &&
           !dependsOn(compare.getRhs(), coordinate);
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

  struct InterleavedFieldAddress {
    mlir::Value root;
    mlir::Value pointer;
    llvm::DenseMap<mlir::Value, int64_t> invariantTerms;
    unsigned field = 0;
  };

  struct AffineAddressExpression {
    int64_t coordinateCoefficient = 0;
    int64_t constant = 0;
    llvm::DenseMap<mlir::Value, int64_t> invariantTerms;
  };

  bool collectAffineAddress(mlir::Value value, mlir::Value coordinate,
                            int64_t coefficient,
                            AffineAddressExpression &result) const {
    if (std::optional<int64_t> constant = integerConstantValue(value)) {
      result.constant += coefficient * *constant;
      return true;
    }
    if (value == coordinate) {
      result.coordinateCoefficient += coefficient;
      return true;
    }
    if (auto pointer = value.getDefiningOp<PtrAddOp>())
      return collectAffineAddress(pointer.getBase(), coordinate, coefficient,
                                  result) &&
             collectAffineAddress(pointer.getOffset(), coordinate, coefficient,
                                  result);
    if (auto binary = value.getDefiningOp<BinaryOp>()) {
      if (binary.getKind() == "add" || binary.getKind() == "sub")
        return collectAffineAddress(binary.getLhs(), coordinate, coefficient,
                                    result) &&
               collectAffineAddress(
                   binary.getRhs(), coordinate,
                   binary.getKind() == "add" ? coefficient : -coefficient,
                   result);
      if (binary.getKind() == "mul") {
        if (std::optional<int64_t> factor =
                integerConstantValue(binary.getLhs()))
          return collectAffineAddress(binary.getRhs(), coordinate,
                                      coefficient * *factor, result);
        if (std::optional<int64_t> factor =
                integerConstantValue(binary.getRhs()))
          return collectAffineAddress(binary.getLhs(), coordinate,
                                      coefficient * *factor, result);
      }
    }
    if (auto cast = value.getDefiningOp<CastOp>())
      return collectAffineAddress(cast.getInput(), coordinate, coefficient,
                                  result);
    if (auto expand = value.getDefiningOp<ExpandDimsOp>())
      return collectAffineAddress(expand.getInput(), coordinate, coefficient,
                                  result);
    if (dependsOn(value, coordinate))
      return false;
    int64_t &term = result.invariantTerms[value];
    term += coefficient;
    if (term == 0)
      result.invariantTerms.erase(value);
    return true;
  }

  bool haveSameInvariantAddress(
      const InterleavedFieldAddress &lhs,
      const InterleavedFieldAddress &rhs) const {
    if (lhs.invariantTerms.size() != rhs.invariantTerms.size())
      return false;
    return llvm::all_of(lhs.invariantTerms, [&](const auto &term) {
      auto found = rhs.invariantTerms.find(term.first);
      return found != rhs.invariantTerms.end() && found->second == term.second;
    });
  }

  std::optional<LocalBlockMemoryFact>
  resolveLocalBlockMemoryFact(mlir::Value semanticValue) const {
    auto blockType = [](mlir::Type type) -> BlockType {
      return mlir::dyn_cast<BlockType>(unwrapLogicalValidity(type));
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

    int64_t storageExtent = storageType.getShape().front();
    const MemoryAccessFact *access = memoryFact(load.getOperation());
    llvm::SmallVector<BlockIndexOp> axes = collectBlockAxes(load.getPointer());
    BlockIndexOp storageAxis = findDimensionAxis(load.getPointer(), 0, axes);
    if (!access || !storageAxis ||
        integerConstantValue(storageAxis.getExtent()) != storageExtent ||
        memoryRelation(*access, storageAxis.getResult()) !=
            LaneRelation::UnitStride)
      return std::nullopt;

    return LocalBlockMemoryFact{
        semanticValue,
        load.getPointer(),
        storageAxis.getResult(),
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

  std::optional<InterleavedFieldAddress>
  matchInterleavedFieldAddress(mlir::Value pointer,
                               mlir::Value coordinate) const {
    mlir::Value root = scalarPointerBase(pointer);
    AffineAddressExpression expression;
    if (!root || !mlir::isa<PtrType>(root.getType()) ||
        !collectAffineAddress(pointer, coordinate, 1, expression) ||
        expression.coordinateCoefficient != 2 || expression.constant < 0 ||
        expression.constant > 1)
      return std::nullopt;
    auto rootTerm = expression.invariantTerms.find(root);
    if (rootTerm == expression.invariantTerms.end() || rootTerm->second != 1)
      return std::nullopt;
    expression.invariantTerms.erase(rootTerm);
    return InterleavedFieldAddress{
        root, pointer, std::move(expression.invariantTerms),
        static_cast<unsigned>(expression.constant)};
  }

  void initializeEntityPlan(PhysicalEntityPlan &entity) const {
    entity.resources.architecturalGroups = options.target.vectorRegisters;
  }

  template <typename Decision>
  void recordLocalBlockReloads(PlannedPhysicalDecision<Decision> &planned,
                               mlir::Operation *consumer,
                               llvm::ArrayRef<LocalBlockMemoryFact> blocks,
                               RVVVectorShape shape) const {
    for (const LocalBlockMemoryFact &block : blocks) {
      recordPhysicalValue(planned.entity, block.semanticValue, shape);
      recordPhysicalHandoff(planned.entity, consumer, block.semanticValue,
                            PhysicalHandoff::Reload, shape, shape);
    }
  }

  void finalizeSymmetricI4I8Plan(
      SymmetricI4I8DotOp op,
      PlannedPhysicalDecision<SymmetricI4I8Decision> &planned) const {
    initializeEntityPlan(planned.entity);
    RVVVectorShape codeShape = planned.realization.realization ==
                                       I4I8FragmentRealization::SpacemiTIME1N16K32
                                   ? kRVVE8MF4
                                   : rvvShape(8, 2);
    RVVVectorShape accumulatorShape = rvvShape(32, 4);
    recordPhysicalValue(planned.entity, planned.realization.activation,
                        codeShape);
    recordPhysicalHandoff(planned.entity, op.getOperation(),
                          planned.realization.activation,
                          PhysicalHandoff::LocalPack, codeShape, codeShape);
    recordPhysicalValue(planned.entity, op.getInit(), accumulatorShape);
    recordPhysicalValue(planned.entity, op.getResult(), accumulatorShape);
    recordPhysicalHandoff(planned.entity, op.getOperation(), op.getInit(),
                          PhysicalHandoff::Share, accumulatorShape,
                          accumulatorShape);
    planned.entity.resources.valueGroups = 4;
    planned.entity.resources.memoryGroups = 2;
    planned.entity.resources.primitiveGroups =
        planned.realization.realization ==
                I4I8FragmentRealization::SpacemiTIME1N16K32
            ? 28
            : 12;
    planned.entity.resources.peakGroups =
        planned.entity.resources.primitiveGroups + 1;
  }

  void finalizeAffineI4I8Plan(
      AffineI4I8DotOp op,
      PlannedPhysicalDecision<AffineI4I8Decision> &planned) const {
    initializeEntityPlan(planned.entity);
    RVVVectorShape codeShape = planned.realization.realization ==
                                       I4I8FragmentRealization::SpacemiTIME1N16K32
                                   ? kRVVE8MF4
                                   : rvvShape(8, 2);
    RVVVectorShape accumulatorShape = rvvShape(32, 4);
    recordPhysicalValue(planned.entity, planned.realization.activation,
                        codeShape);
    recordPhysicalHandoff(planned.entity, op.getOperation(),
                          planned.realization.activation,
                          PhysicalHandoff::LocalPack, codeShape, codeShape);
    recordPhysicalValue(planned.entity, op.getInit(), accumulatorShape);
    recordPhysicalValue(planned.entity, op.getResult(), accumulatorShape);
    recordPhysicalHandoff(planned.entity, op.getOperation(), op.getInit(),
                          PhysicalHandoff::Share, accumulatorShape,
                          accumulatorShape);
    planned.entity.resources.valueGroups = 4;
    planned.entity.resources.memoryGroups = 2;
    planned.entity.resources.primitiveGroups =
        planned.realization.realization ==
                I4I8FragmentRealization::SpacemiTIME1N16K32
            ? 28
            : 13;
    planned.entity.resources.peakGroups =
        planned.entity.resources.primitiveGroups + 1;
  }

  void finalizeSignBitI8Plan(
      SignBitI8DotOp op,
      PlannedPhysicalDecision<SignBitI8Decision> &planned) const {
    initializeEntityPlan(planned.entity);
    recordLocalBlockReloads(planned, op.getOperation(),
                            {planned.realization.signBits,
                             planned.realization.activation},
                            rvvShape(8, 2));
    planned.entity.resources.valueGroups = 4;
    planned.entity.resources.memoryGroups = 4;
    planned.entity.resources.primitiveGroups = 9;
    planned.entity.resources.peakGroups = 10;
  }

  void finalizeE2M1E8M0I8Plan(
      E2M1E8M0I8DotOp op,
      PlannedPhysicalDecision<E2M1E8M0I8Decision> &planned) const {
    initializeEntityPlan(planned.entity);
    recordLocalBlockReloads(planned, op.getOperation(),
                            {planned.realization.packedCodes},
                            planned.realization.packedShape);
    recordLocalBlockReloads(planned, op.getOperation(),
                            {planned.realization.activation},
                            planned.realization.activationShape);
    planned.entity.resources = planned.realization.resources;
  }

  void finalizeGroupedAffineI4I8Plan(
      GroupedAffineI4I8DotOp op,
      PlannedPhysicalDecision<GroupedAffineI4I8Decision> &planned) const {
    initializeEntityPlan(planned.entity);
    recordLocalBlockReloads(planned, op.getOperation(),
                            {planned.realization.packedWeight},
                            planned.realization.packedShape);
    recordLocalBlockReloads(planned, op.getOperation(),
                            {planned.realization.scaleMin},
                            planned.realization.scaleShape);
    recordLocalBlockReloads(planned, op.getOperation(),
                            {planned.realization.activation},
                            planned.realization.activationShape);
    recordLocalBlockReloads(planned, op.getOperation(),
                            {planned.realization.activationSum},
                            planned.realization.activationSumShape);
    planned.entity.resources = planned.realization.resources;
  }

  template <typename OpTy>
  void finalizeQuantCodebookI8Plan(
      OpTy op, PlannedPhysicalDecision<QuantCodebookI8Decision> &planned) const {
    initializeEntityPlan(planned.entity);
    RVVVectorShape shape = planned.realization.byteShape;
    recordLocalBlockReloads(planned, op.getOperation(),
                            planned.realization.blocks, shape);
    planned.entity.resources = planned.realization.resources;
  }

  void finalizeBlockDecodePlan(
      DecodeOp op,
      PlannedPhysicalDecision<BlockDecodeDecision> &planned) const {
    initializeEntityPlan(planned.entity);
    recordPhysicalValue(planned.entity, op.getCodes(), kRVVE8M1);
    recordPhysicalValue(planned.entity, op.getTable(), kRVVE8M1);
    recordPhysicalValue(planned.entity, op.getResult(), kRVVE8M1);
    recordPhysicalHandoff(planned.entity, op.getOperation(), op.getCodes(),
                          PhysicalHandoff::Share, kRVVE8M1, kRVVE8M1);
    planned.entity.resources.valueGroups = 3;
    planned.entity.resources.primitiveGroups = 3;
    planned.entity.resources.peakGroups = 4;
  }

  mlir::LogicalResult decideSymmetricI4I8Dot(
      SymmetricI4I8DotOp op, SymmetricI4I8Decision &decision) {
    std::optional<I4I8FragmentRealization> fragment =
        weft::riscv_internal::selectI4I8FragmentRealization(options.target);
    if (!fragment)
      return op.emitError(
          "symmetric i4/i8 RVV dot requires legal e8m1/e16m2/e32m4 "
          "integer-widening and f16-to-f32 vector shapes");
    if (!options.target.littleEndian)
      return op.emitError(
          "symmetric i4/i8 dot requires little-endian packed fields");

    std::optional<LocalBlockMemoryFact> activation =
        resolveLocalBlockMemoryFact(op.getActivation());
    if (!activation)
      return op.emitError(
          "symmetric i4/i8 dot requires one typed K32 activation memory fact");
    if (activation->semanticExtent != 32 || activation->storageExtent != 32 ||
        !activation->semanticType.getElementType().isSignedInteger(8) ||
        !activation->storageType.getElementType().isSignedInteger(8))
      return op.emitError(
          "symmetric i4/i8 dot requires contiguous signed-i8 K32 activation memory");
    mlir::Value packedBase = op.getPackedBase();
    int64_t packedBlockBytes = 288;
    mlir::Value packedRoot = pointerRoot(packedBase);
    auto packedType =
        packedRoot ? mlir::dyn_cast<PtrType>(packedRoot.getType()) : PtrType{};
    if (!packedType || packedType.getStorageClass() != "persistent" ||
        packedType.getStorageFormat() != "q4_0_n16_k32_288b")
      return op.emitError(
          "symmetric i4/i8 dot requires persistent format q4_0_n16_k32_288b");
    decision = SymmetricI4I8Decision{};
    decision.realization = *fragment;
    bool useIME1 = *fragment == I4I8FragmentRealization::SpacemiTIME1N16K32;
    decision.leaf = useIME1 ? IntrinsicCLeaf::IME1SymmetricI4I8N16K32
                            : IntrinsicCLeaf::RVVSymmetricI4I8N16K32;
    decision.activationBlock = *activation;
    decision.packedBlockBase = packedBase;
    decision.activation = op.getActivation();
    decision.activationScale = op.getActivationScale();
    decision.init = op.getInit();
    decision.packedBlockBytes = packedBlockBytes;
    if (mlir::failed(selectMaterializedBlockStorage(op.getInit(),
                                                   op.getOperation())))
      return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult emitSymmetricI4I8Dot(
      SymmetricI4I8DotOp op) {
    auto selected = physicalPlan.symmetricI4I8.find(op.getOperation());
    if (selected == physicalPlan.symmetricI4I8.end())
      return op.emitError("symmetric i4/i8 physical decision was not prepared");
    if (mlir::failed(requireEntityPlan(op.getOperation(), selected->second)))
      return mlir::failure();
    const SymmetricI4I8Decision &decision = selected->second.realization;
    if (mlir::failed(requirePhysicalHandoff(
            op.getOperation(), selected->second.entity, decision.activation,
            PhysicalHandoff::LocalPack)))
      return mlir::failure();
    std::optional<std::string> activation =
        projectLocalBlockMemoryBase(decision.activationBlock);
    CValue packed = require(decision.packedBlockBase);
    CValue scale = require(decision.activationScale);
    CValue accumulator = require(decision.init);
    if (decision.packedBlockBytes != 288 ||
        !activation ||
        packed.kind != CValueKind::Pointer || scale.kind != CValueKind::Scalar ||
        accumulator.kind != CValueKind::F32BlockStorage ||
        packed.spelling.empty() ||
        scale.spelling.empty() || accumulator.spelling.empty())
      return op.emitError(
          "selected symmetric i4/i8 fragment operands are unavailable");
    llvm::StringRef helper = intrinsicCLeafName(decision.leaf);
    if (helper.empty())
      return op.emitError("symmetric i4/i8 leaf spelling was not selected");
    line(helper.str() + "(" + scale.spelling + ", " +
         *activation + ", " + packed.spelling + ", " +
         accumulator.spelling + ");");
    if (mlir::failed(markRematerializedBlockTrees(
            {decision.activation, decision.init}, op.getOperation())))
      return mlir::failure();
    loweredBlockOps.insert(op.getOperation());
    accumulator.type = op.getResult().getType();
    values[op.getResult()] = std::move(accumulator);
    return mlir::success();
  }

  mlir::LogicalResult emitMaterializedF32Pointwise(
      mlir::Operation *operation) {
    auto selected = physicalPlan.materializedF32Pointwise.find(operation);
    if (selected == physicalPlan.materializedF32Pointwise.end())
      return operation->emitError(
          "materialized pointwise operation has no physical decision");
    if (mlir::failed(requireEntityPlan(operation, selected->second)))
      return mlir::failure();
    const MaterializedF32PointwiseDecision &decision =
        selected->second.realization;
    auto storageDecision = llvm::find_if(
        selected->second.entity.storages,
        [&](const PhysicalStorageDecision &storage) {
          return storage.value == operation->getResult(0);
        });
    if (storageDecision == selected->second.entity.storages.end() ||
        storageDecision->elements != decision.elements ||
        decision.elements <= 0)
      return operation->emitError(
          "materialized pointwise storage decision is incomplete");

    std::string storage;
    if (storageDecision->reusedStorage) {
      CValue reused = require(storageDecision->reusedStorage);
      if (reused.kind != CValueKind::F32BlockStorage ||
          reused.spelling.empty())
        return operation->emitError(
            "materialized pointwise reused storage is unavailable");
      storage = reused.spelling;
    } else {
      storage = fresh("pointwise_result");
      line("float " + storage + "[" + std::to_string(decision.elements) +
           "];");
    }
    std::string lane = fresh("pointwise_lane");
    auto operandAt = [&](mlir::Value value) -> std::optional<std::string> {
      CValue operand = require(value);
      if (operand.kind == CValueKind::F32BlockStorage &&
          !operand.spelling.empty())
        return operand.spelling + "[" + lane + "]";
      if (operand.kind == CValueKind::Scalar && !operand.spelling.empty())
        return operand.spelling;
      return std::nullopt;
    };
    std::string rhsExpression;
    if (auto binary = mlir::dyn_cast<BinaryOp>(operation)) {
      std::optional<std::string> lhs = operandAt(binary.getLhs());
      std::optional<std::string> rhs = operandAt(binary.getRhs());
      if (!lhs || !rhs) {
        CValue lhsValue = require(binary.getLhs());
        CValue rhsValue = require(binary.getRhs());
        return binary.emitError()
               << "materialized binary operands are unavailable (lhs kind "
               << static_cast<int>(lhsValue.kind) << ", rhs kind "
               << static_cast<int>(rhsValue.kind) << ")";
      }
      rhsExpression = llvm::StringSwitch<std::string>(binary.getKind())
                          .Case("add", "(" + *lhs + " + " + *rhs + ")")
                          .Case("sub", "(" + *lhs + " - " + *rhs + ")")
                          .Case("mul", "(" + *lhs + " * " + *rhs + ")")
                          .Case("div", "(" + *lhs + " / " + *rhs + ")")
                          .Case("mod", "fmodf(" + *lhs + ", " + *rhs + ")")
                          .Case("max", "fmaxf(" + *lhs + ", " + *rhs + ")")
                          .Case("min", "fminf(" + *lhs + ", " + *rhs + ")")
                          .Default("");
    } else if (auto unary = mlir::dyn_cast<UnaryOp>(operation)) {
      std::optional<std::string> input = operandAt(unary.getInput());
      if (!input)
        return unary.emitError("materialized unary operand is unavailable");
      rhsExpression = llvm::StringSwitch<std::string>(unary.getKind())
                          .Case("neg", "(-" + *input + ")")
                          .Case("exp", "expf(" + *input + ")")
                          .Case("tanh", "tanhf(" + *input + ")")
                          .Case("log", "logf(" + *input + ")")
                          .Case("sqrt", "sqrtf(" + *input + ")")
                          .Case("rsqrt", "(1.0f / sqrtf(" + *input + "))")
                          .Case("sin", "sinf(" + *input + ")")
                          .Case("cos", "cosf(" + *input + ")")
                          .Case("floor", "floorf(" + *input + ")")
                          .Default("");
    }
    if (rhsExpression.empty())
      return operation->emitError(
          "materialized pointwise decision has no intrinsic-C spelling");
    line("for (size_t " + lane + " = 0; " + lane + " < " +
         std::to_string(decision.elements) + "; ++" + lane + ")");
    ++indent;
    line(storage + "[" + lane + "] = " + rhsExpression + ";");
    --indent;
    values[operation->getResult(0)] =
        CValue{operation->getResult(0).getType(),
               CValueKind::F32BlockStorage, storage};
    loweredBlockOps.insert(operation);
    return mlir::success();
  }

  std::optional<mlir::LogicalResult>
  tryEmitMaterializedF32BlockStore(StoreOp op) {
    auto selected = physicalPlan.materializedBlockStores.find(op.getOperation());
    if (selected == physicalPlan.materializedBlockStores.end())
      return std::nullopt;
    if (mlir::failed(requireEntityPlan(op.getOperation(), selected->second)))
      return std::optional<mlir::LogicalResult>(mlir::failure());
    const MaterializedBlockStoreDecision &decision = selected->second.realization;
    const PhysicalEntityPlan &entity = selected->second.entity;
    auto stored = values.find(op.getValue());
    if (stored == values.end() ||
        stored->second.kind != CValueKind::F32BlockStorage)
      return std::optional<mlir::LogicalResult>(
          op.emitError("materialized f32 block storage is unavailable"));
    if (decision.realization ==
        MaterializedBlockStoreRealization::ScalarRankOne) {
      std::string lane = fresh("block_store_lane");
      line("for (size_t " + lane + " = 0; " + lane + " < " +
           std::to_string(decision.columns) + "; ++" + lane + ") {");
      ++indent;
      llvm::DenseMap<mlir::Value, std::string> axisValues;
      axisValues[decision.rowAxis] = lane;
      std::optional<std::string> pointer =
          projectBlockScalar(op.getPointer(), axisValues);
      std::optional<std::string> predicate =
          projectBlockScalar(op.getWhere(), axisValues);
      if (!pointer || !predicate)
        return std::optional<mlir::LogicalResult>(op.emitError(
            "materialized rank-one f32 store projection is unavailable"));
      line("if (" + *predicate + ") *(" + *pointer + ") = " +
           stored->second.spelling + "[" + lane + "];");
      --indent;
      line("}");
      if (mlir::failed(markRematerializedBlockTrees(
              {op.getPointer(), op.getWhere()}, op.getOperation())))
        return mlir::failure();
      loweredBlockOps.insert(op.getOperation());
      consumed.insert(op.getOperation());
      return std::optional<mlir::LogicalResult>(mlir::success());
    }
    if (decision.realization ==
        MaterializedBlockStoreRealization::ScalarRankTwo) {
      std::string row = fresh("block_store_row");
      std::string column = fresh("block_store_column");
      line("for (size_t " + row + " = 0; " + row + " < " +
           std::to_string(decision.rows) + "; ++" + row + ") {");
      ++indent;
      line("for (size_t " + column + " = 0; " + column + " < " +
           std::to_string(decision.columns) + "; ++" + column + ") {");
      ++indent;
      llvm::DenseMap<mlir::Value, std::string> axisValues;
      axisValues[decision.rowAxis] = row;
      axisValues[decision.columnAxis] = column;
      std::optional<std::string> pointer =
          projectBlockScalar(op.getPointer(), axisValues);
      std::optional<std::string> predicate =
          projectBlockScalar(op.getWhere(), axisValues);
      if (!pointer || !predicate)
        return std::optional<mlir::LogicalResult>(op.emitError(
            "materialized rank-two f32 store projection is unavailable"));
      line("if (" + *predicate + ") *(" + *pointer + ") = " +
           stored->second.spelling + "[" + row + " * " +
           std::to_string(decision.columns) + " + " + column + "];");
      --indent;
      line("}");
      --indent;
      line("}");
      if (mlir::failed(markRematerializedBlockTrees(
              {op.getPointer(), op.getWhere()}, op.getOperation())))
        return mlir::failure();
      loweredBlockOps.insert(op.getOperation());
      consumed.insert(op.getOperation());
      return std::optional<mlir::LogicalResult>(mlir::success());
    }
    if (decision.realization !=
            MaterializedBlockStoreRealization::RVVContiguousRankOne ||
        decision.columns != 16)
      return std::optional<mlir::LogicalResult>(
          op.emitError("materialized block store decision has no spelling"));
    auto valuePlan = llvm::find_if(
        entity.values, [&](const PhysicalValueDecision &value) {
          return value.value == op.getValue();
        });
    if (valuePlan == entity.values.end() || valuePlan->shape != kRVVE32M2)
      return std::optional<mlir::LogicalResult>(
          op.emitError("materialized block store has no physical value shape"));
    llvm::DenseMap<mlir::Value, std::string> axes;
    axes[decision.columnAxis] = "0";
    std::optional<std::string> destination =
        projectBlockScalar(op.getPointer(), axes);
    if (!destination)
      return std::optional<mlir::LogicalResult>(op.emitError(
          "materialized f32 block store has no scalar pointer base"));
    std::string offset = fresh("block_store_offset");
    std::string vl = fresh("block_store_vl");
    std::string value = fresh("block_store_value");
    line("for (size_t " + offset + " = 0; " + offset + " < 16;) {");
    ++indent;
    line("const size_t " + vl + " = __riscv_vsetvl_e32m2(16 - " + offset +
         ");");
    line("vfloat32m2_t " + value + " = __riscv_vle32_v_f32m2(" +
         stored->second.spelling + " + " + offset + ", " + vl + ");");
    line("__riscv_vse32_v_f32m2(" + *destination + " + " + offset +
         ", " + value + ", " + vl + ");");
    line(offset + " += " + vl + ";");
    --indent;
    line("}");
    if (mlir::failed(markRematerializedBlockTrees(
            {op.getPointer()}, op.getOperation())))
      return mlir::failure();
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
    auto prepared = physicalPlan.signBitI8.find(op.getOperation());
    if (prepared == physicalPlan.signBitI8.end())
      return op.emitError("sign-bit/i8 physical decision was not prepared");
    if (mlir::failed(requireEntityPlan(op.getOperation(), prepared->second)))
      return mlir::failure();
    const SignBitI8Decision &decision = prepared->second.realization;
    for (const LocalBlockMemoryFact *block :
         {&decision.signBits, &decision.activation})
      if (mlir::failed(requirePhysicalHandoff(
              op.getOperation(), prepared->second.entity,
              block->semanticValue, PhysicalHandoff::Reload)))
        return mlir::failure();
    std::optional<std::string> signBits =
        projectLocalBlockMemoryBase(decision.signBits);
    std::optional<std::string> activation =
        projectLocalBlockMemoryBase(decision.activation);
    CValue activationScale = require(decision.activationScale);
    CValue signScale = require(decision.signScale);
    CValue init = require(decision.init);
    if (decision.realization !=
            SignBitI8Realization::RVVWideningSignSum ||
        !signBits || !activation ||
        activationScale.kind != CValueKind::Scalar ||
        signScale.kind != CValueKind::Scalar || init.kind != CValueKind::Scalar ||
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
         "(const int8_t *)(const void *)" + *activation + ", " + vl +
         ");");
    line("vint16m4_t " + wide + " = __riscv_vsext_vf2_i16m4(" + codes +
         ", " + vl + ");");
    line("vbool4_t " + mask + " = __riscv_vlm_v_b4(" + *signBits +
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
    if (mlir::failed(markRematerializedBlockTrees(
            {decision.signBits.semanticValue,
             decision.activation.semanticValue},
            op.getOperation())))
      return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult decideE2M1E8M0I8Dot(
      E2M1E8M0I8DotOp op, E2M1E8M0I8Decision &decision) {
    auto packedCodes = resolveLocalBlockMemoryFact(op.getPackedCodes());
    auto activation = resolveLocalBlockMemoryFact(op.getActivation());
    if (!packedCodes || !activation)
      return op.emitError(
          "E2M1/E8M0 i8 dot requires contiguous all-active local block memory facts");

    std::optional<SelectedE2M1E8M0I8Physical> selected =
        selectE2M1E8M0I8Physical(options.target);
    if (!selected)
      return op.emitError(
          "E2M1/E8M0 i8 dot has no resource-legal RVV realization for the target");

    decision = E2M1E8M0I8Decision{};
    decision.realization = selected->realization;
    decision.packedShape = selected->packedShape;
    decision.activationShape = selected->activationShape;
    decision.resources = selected->resources;
    switch (selected->realization) {
    case E2M1E8M0I8Realization::RVVVLEN128TableDot:
      decision.leaf = IntrinsicCLeaf::E2M1E8M0I8VLEN128;
      break;
    case E2M1E8M0I8Realization::RVVVLEN256TableDot:
      decision.leaf = IntrinsicCLeaf::E2M1E8M0I8VLEN256;
      break;
    case E2M1E8M0I8Realization::RVVScalableLocalBlockDot:
      decision.leaf = IntrinsicCLeaf::E2M1E8M0I8Scalable;
      break;
    }
    decision.packedCodes = *packedCodes;
    decision.activation = *activation;
    decision.exponent = op.getExponent();
    decision.activationScale = op.getActivationScale();
    decision.init = op.getInit();
    return mlir::success();
  }

  mlir::LogicalResult emitE2M1E8M0I8Dot(E2M1E8M0I8DotOp op) {
    auto selected = physicalPlan.e2m1E8M0I8.find(op.getOperation());
    if (selected == physicalPlan.e2m1E8M0I8.end())
      return op.emitError("E2M1/E8M0 physical decision was not prepared");
    if (mlir::failed(requireEntityPlan(op.getOperation(), selected->second)))
      return mlir::failure();
    const E2M1E8M0I8Decision &decision = selected->second.realization;
    for (const LocalBlockMemoryFact *block :
         {&decision.packedCodes, &decision.activation})
      if (mlir::failed(requirePhysicalHandoff(
              op.getOperation(), selected->second.entity,
              block->semanticValue, PhysicalHandoff::Reload)))
        return mlir::failure();
    std::optional<std::string> packed =
        projectLocalBlockMemoryBase(decision.packedCodes);
    std::optional<std::string> activation =
        projectLocalBlockMemoryBase(decision.activation);
    CValue exponent = require(decision.exponent);
    CValue activationScale = require(decision.activationScale);
    CValue init = require(decision.init);
    if ((decision.realization != E2M1E8M0I8Realization::RVVVLEN128TableDot &&
         decision.realization != E2M1E8M0I8Realization::RVVVLEN256TableDot &&
         decision.realization !=
             E2M1E8M0I8Realization::RVVScalableLocalBlockDot) ||
        !packed || !activation ||
        exponent.kind != CValueKind::Scalar ||
        activationScale.kind != CValueKind::Scalar ||
        init.kind != CValueKind::Scalar || exponent.spelling.empty() ||
        activationScale.spelling.empty() || init.spelling.empty())
      return op.emitError(
          "selected E2M1/E8M0 i8 dot operands are unavailable");

    llvm::StringRef helper = intrinsicCLeafName(decision.leaf);
    if (helper.empty())
      return op.emitError("E2M1/E8M0 leaf spelling was not selected");
    std::string result = fresh("e2m1_dot");
    line("const float " + result + " = " + helper.str() + "(" + *packed +
         ", " + exponent.spelling + ", " + *activation + ", " +
         activationScale.spelling + ", " + init.spelling + ");");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, result};
    if (mlir::failed(markRematerializedBlockTrees(
            {decision.packedCodes.semanticValue,
             decision.activation.semanticValue},
            op.getOperation())))
      return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult decideGroupedAffineI4I8Dot(
      GroupedAffineI4I8DotOp op, GroupedAffineI4I8Decision &decision) {
    auto packedWeight = resolveLocalBlockMemoryFact(op.getPackedWeight());
    auto scaleMin = resolveLocalBlockMemoryFact(op.getScaleMin());
    auto activation = resolveLocalBlockMemoryFact(op.getActivation());
    auto activationSum =
        resolveLocalBlockMemoryFact(op.getActivationSumBytes());
    if (!packedWeight || !scaleMin || !activation || !activationSum)
      return op.emitError(
          "grouped affine i4/i8 dot requires contiguous all-active local block memory facts");

    std::optional<SelectedGroupedAffineI4I8Physical> selected =
        selectGroupedAffineI4I8Physical(options.target);
    if (!selected)
      return op.emitError(
          "grouped affine i4/i8 dot has no resource-legal RVV realization for the target");

    decision = GroupedAffineI4I8Decision{};
    decision.realization = selected->realization;
    decision.packedShape = selected->packedShape;
    decision.scaleShape = selected->scaleShape;
    decision.activationShape = selected->activationShape;
    decision.activationSumShape = selected->activationSumShape;
    decision.resources = selected->resources;
    switch (selected->realization) {
    case GroupedAffineI4I8Realization::RVVVLEN128GroupedDot:
      decision.leaf = IntrinsicCLeaf::GroupedAffineI4I8VLEN128;
      break;
    case GroupedAffineI4I8Realization::RVVVLEN256GroupedDot:
      decision.leaf = IntrinsicCLeaf::GroupedAffineI4I8VLEN256;
      break;
    case GroupedAffineI4I8Realization::RVVScalableLocalBlockDot:
      decision.leaf = IntrinsicCLeaf::GroupedAffineI4I8Scalable;
      break;
    }
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
    auto selected = physicalPlan.groupedAffineI4I8.find(op.getOperation());
    if (selected == physicalPlan.groupedAffineI4I8.end())
      return op.emitError("grouped affine i4/i8 physical decision was not prepared");
    if (mlir::failed(requireEntityPlan(op.getOperation(), selected->second)))
      return mlir::failure();
    const GroupedAffineI4I8Decision &decision = selected->second.realization;
    for (const LocalBlockMemoryFact *block :
         {&decision.packedWeight, &decision.scaleMin, &decision.activation,
          &decision.activationSum})
      if (mlir::failed(requirePhysicalHandoff(
              op.getOperation(), selected->second.entity,
              block->semanticValue, PhysicalHandoff::Reload)))
        return mlir::failure();

    std::optional<std::string> packed =
        projectLocalBlockMemoryBase(decision.packedWeight);
    std::optional<std::string> scales =
        projectLocalBlockMemoryBase(decision.scaleMin);
    std::optional<std::string> activation =
        projectLocalBlockMemoryBase(decision.activation);
    std::optional<std::string> sums =
        projectLocalBlockMemoryBase(decision.activationSum);
    CValue dotScale = require(decision.dotScale);
    CValue minimumScale = require(decision.minimumScale);
    CValue init = require(decision.init);
    if ((decision.realization !=
             GroupedAffineI4I8Realization::RVVVLEN128GroupedDot &&
         decision.realization !=
             GroupedAffineI4I8Realization::RVVVLEN256GroupedDot &&
         decision.realization !=
             GroupedAffineI4I8Realization::RVVScalableLocalBlockDot) ||
        !packed || !scales || !activation || !sums ||
        dotScale.kind != CValueKind::Scalar ||
        minimumScale.kind != CValueKind::Scalar || init.kind != CValueKind::Scalar ||
        dotScale.spelling.empty() || minimumScale.spelling.empty() ||
        init.spelling.empty())
      return op.emitError(
          "grouped affine i4/i8 dot operands are not materialized locally");

    std::string result = fresh("grouped_dot");
    llvm::StringRef helper = intrinsicCLeafName(decision.leaf);
    if (helper.empty())
      return op.emitError("grouped affine i4/i8 leaf spelling was not selected");
    line("const float " + result + " = " + helper.str() + "(" +
         *packed + ", " + *scales + ", " +
         *activation + ", " + *sums + ", " +
         dotScale.spelling + ", " + minimumScale.spelling + ", " +
         init.spelling + ");");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, result};
    if (mlir::failed(markRematerializedBlockTrees(
            {decision.packedWeight.semanticValue,
             decision.scaleMin.semanticValue,
             decision.activation.semanticValue,
             decision.activationSum.semanticValue},
            op.getOperation())))
      return mlir::failure();
    return mlir::success();
  }

  template <typename OpTy>
  mlir::LogicalResult decideQuantCodebookI8Dot(
      OpTy op, QuantI8DotSemantic semantic,
      llvm::ArrayRef<mlir::Value> blocks,
      llvm::ArrayRef<mlir::Value> scalars,
      QuantCodebookI8Decision &decision) {
    std::optional<SelectedQuantI8DotPhysical> selected =
        weft::riscv_internal::selectQuantI8DotPhysical(
            QuantI8DotCandidateFacts{semantic}, options.target);
    if (!selected)
      return op.emitError(
          "quant codebook/i8 dot has no legal target realization");
    decision = QuantCodebookI8Decision{};
    decision.semantic = semantic;
    decision.realization = selected->realization;
    decision.semanticLanes = selected->semanticLanes;
    decision.byteShape = selected->byteShape;
    decision.reductionSegments = selected->reductionSegments;
    decision.resources = selected->resources;
    switch (semantic) {
    case QuantI8DotSemantic::IQ2S:
      if (decision.realization ==
          QuantI8DotRealization::RVVFixedLaneLocalBlockDot) {
        if (decision.semanticLanes == 32)
          decision.leaf = IntrinsicCLeaf::IQ2SI8FixedLanes32;
        else if (decision.semanticLanes == 64)
          decision.leaf = IntrinsicCLeaf::IQ2SI8FixedLanes64;
        else
          return op.emitError(
              "fixed-lane IQ2_S/i8 dot requires 32 or 64 semantic lanes");
      } else {
        decision.leaf = IntrinsicCLeaf::IQ2SI8Scalable;
      }
      break;
    case QuantI8DotSemantic::IQ3S:
      decision.leaf = IntrinsicCLeaf::IQ3SI8Scalable;
      break;
    case QuantI8DotSemantic::IQ1M:
      decision.leaf = decision.realization ==
                              QuantI8DotRealization::RVVFixedLaneLocalBlockDot
                          ? IntrinsicCLeaf::IQ1MI8Fixed
                          : IntrinsicCLeaf::IQ1MI8Scalable;
      break;
    case QuantI8DotSemantic::Q6K:
      if (decision.realization ==
          QuantI8DotRealization::RVVFixedLaneLocalBlockDot) {
        if (decision.semanticLanes == 32)
          decision.leaf = IntrinsicCLeaf::Q6KI8FixedLanes32;
        else if (decision.semanticLanes == 64)
          decision.leaf = IntrinsicCLeaf::Q6KI8FixedLanes64;
        else
          return op.emitError(
              "fixed-lane Q6_K/i8 dot requires 32 or 64 semantic lanes");
      } else {
        decision.leaf = IntrinsicCLeaf::Q6KI8Scalable;
      }
      break;
    }
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
  mlir::LogicalResult emitQuantCodebookI8Dot(OpTy op) {
    auto prepared = physicalPlan.quantCodebookI8.find(op.getOperation());
    if (prepared == physicalPlan.quantCodebookI8.end())
      return op.emitError(
          "quant codebook/i8 physical decision was not prepared");
    if (mlir::failed(requireEntityPlan(op.getOperation(), prepared->second)))
      return mlir::failure();
    const QuantCodebookI8Decision &decision = prepared->second.realization;
    for (const LocalBlockMemoryFact &block : decision.blocks)
      if (mlir::failed(requirePhysicalHandoff(
              op.getOperation(), prepared->second.entity, block.semanticValue,
              PhysicalHandoff::Reload)))
        return mlir::failure();
    if (decision.realization !=
            QuantI8DotRealization::RVVFixedLaneLocalBlockDot &&
        decision.realization !=
            QuantI8DotRealization::RVVScalableLocalBlockDot)
      return op.emitError("quant codebook/i8 realization is unavailable");
    llvm::SmallVector<std::string> operands;
    for (const LocalBlockMemoryFact &block : decision.blocks) {
      std::optional<std::string> value = projectLocalBlockMemoryBase(block);
      if (!value)
        return op.emitError(
            "quant codebook/i8 block base was not materialized");
      operands.push_back(std::move(*value));
    }
    for (mlir::Value scalar : decision.scalars) {
      CValue value = require(scalar);
      if (value.kind != CValueKind::Scalar || value.spelling.empty())
        return op.emitError(
            "quant codebook/i8 scalar operand was not materialized");
      operands.push_back(std::move(value.spelling));
    }
    std::string result = fresh("quant_dot");
    llvm::StringRef helper = intrinsicCLeafName(decision.leaf);
    if (helper.empty())
      return op.emitError("quant codebook/i8 leaf spelling was not selected");
    line("const float " + result + " = " + helper.str() + "(" +
         llvm::join(operands, ", ") + ");");
    values[op.getResult()] =
        CValue{op.getResult().getType(), CValueKind::Scalar, result};
    llvm::SmallVector<mlir::Value> blockValues;
    for (const LocalBlockMemoryFact &block : decision.blocks)
      blockValues.push_back(block.semanticValue);
    if (mlir::failed(
            markRematerializedBlockTrees(blockValues, op.getOperation())))
      return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult decideAffineI4I8Dot(
      AffineI4I8DotOp op, AffineI4I8Decision &decision) {
    std::optional<I4I8FragmentRealization> fragment =
        weft::riscv_internal::selectI4I8FragmentRealization(options.target);
    if (!fragment)
      return op.emitError(
          "affine i4/i8 RVV dot requires legal e8m1/e16m2/e32m4 "
          "integer-widening and f16-to-f32 vector shapes");
    if (!options.target.littleEndian)
      return op.emitError(
          "affine i4/i8 dot requires little-endian packed fields");

    std::optional<LocalBlockMemoryFact> activation =
        resolveLocalBlockMemoryFact(op.getActivation());
    if (!activation)
      return op.emitError(
          "affine i4/i8 dot requires one typed K32 activation memory fact");
    if (activation->semanticExtent != 32 || activation->storageExtent != 32 ||
        !activation->semanticType.getElementType().isSignedInteger(8) ||
        !activation->storageType.getElementType().isSignedInteger(8))
      return op.emitError(
          "affine i4/i8 dot requires contiguous signed-i8 K32 activation memory");
    mlir::Value packedBase = op.getPackedBase();
    int64_t packedBlockBytes = 304;
    mlir::Value packedRoot = pointerRoot(packedBase);
    auto packedType =
        packedRoot ? mlir::dyn_cast<PtrType>(packedRoot.getType()) : PtrType{};
    if (!packedType || packedType.getStorageClass() != "persistent" ||
        packedType.getStorageFormat() != "q4_k_n16_k32_304b")
      return op.emitError(
          "affine i4/i8 dot requires persistent format q4_k_n16_k32_304b");
    decision = AffineI4I8Decision{};
    decision.operation = op.getOperation();
    decision.realization = *fragment;
    bool useIME1 = *fragment == I4I8FragmentRealization::SpacemiTIME1N16K32;
    decision.leaf = useIME1 ? IntrinsicCLeaf::IME1AffineI4I8N16K32
                            : IntrinsicCLeaf::RVVAffineI4I8N16K32;
    decision.activationBlock = *activation;
    decision.packedBlockBase = packedBase;
    decision.activation = op.getActivation();
    decision.activationScale = op.getActivationScale();
    decision.init = op.getInit();
    decision.packedBlockBytes = packedBlockBytes;
    if (mlir::failed(selectMaterializedBlockStorage(op.getInit(),
                                                   op.getOperation())))
      return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult emitAffineI4I8Dot(AffineI4I8DotOp op) {
    auto selected = physicalPlan.affineI4I8.find(op.getOperation());
    if (selected == physicalPlan.affineI4I8.end())
      return op.emitError("affine i4/i8 physical decision was not prepared");
    if (mlir::failed(requireEntityPlan(op.getOperation(), selected->second)))
      return mlir::failure();
    const AffineI4I8Decision &decision = selected->second.realization;
    if (mlir::failed(requirePhysicalHandoff(
            op.getOperation(), selected->second.entity, decision.activation,
            PhysicalHandoff::LocalPack)))
      return mlir::failure();
    std::optional<std::string> activation =
        projectLocalBlockMemoryBase(decision.activationBlock);
    CValue packed = require(decision.packedBlockBase);
    CValue scale = require(decision.activationScale);
    CValue accumulator = require(decision.init);
    if (decision.packedBlockBytes != 304 ||
        !activation ||
        packed.kind != CValueKind::Pointer || scale.kind != CValueKind::Scalar ||
        accumulator.kind != CValueKind::F32BlockStorage ||
        packed.spelling.empty() ||
        scale.spelling.empty() || accumulator.spelling.empty())
      return op.emitError(
          "selected affine i4/i8 fragment operands are unavailable");
    llvm::StringRef helper = intrinsicCLeafName(decision.leaf);
    if (helper.empty())
      return op.emitError("affine i4/i8 leaf spelling was not selected");
    line(helper.str() + "(" + scale.spelling + ", " +
         *activation + ", " + packed.spelling + ", " +
         accumulator.spelling + ");");
    if (mlir::failed(markRematerializedBlockTrees(
            {decision.activation, decision.init}, op.getOperation())))
      return mlir::failure();
    loweredBlockOps.insert(op.getOperation());
    accumulator.type = op.getResult().getType();
    values[op.getResult()] = std::move(accumulator);
    return mlir::success();
  }

  std::optional<PlannedPhysicalDecision<F16GemmNTileDecision>>
  decideF16GemmNTiles(MatmulOp matmul) {
    if (!options.target.hasVectorF16 || !options.target.hasWideningFloat)
      return std::nullopt;
    LoadOp lhsLoad = reloadableLoad(matmul.getLhs());
    LoadOp rhsLoad = reloadableLoad(matmul.getRhs());
    auto blockType = [](mlir::Type type) -> BlockType {
      type = unwrapLogicalValidity(type);
      return mlir::dyn_cast<BlockType>(type);
    };
    BlockType lhsType = blockType(matmul.getLhs().getType());
    BlockType rhsType = blockType(matmul.getRhs().getType());
    BlockType initType = blockType(matmul.getInit().getType());
    BlockType resultType = blockType(matmul.getResult().getType());
    if (!lhsLoad || !rhsLoad || !lhsType || !rhsType || !initType ||
        !resultType || lhsType.getShape().size() != 2 ||
        rhsType.getShape().size() != 2 || initType.getShape().size() != 2 ||
        resultType.getShape().size() != 2 ||
        !lhsType.getElementType().isF16() ||
        !rhsType.getElementType().isF16() ||
        !initType.getElementType().isF32() ||
        !resultType.getElementType().isF32() ||
        matmul.getOrder() != "relaxed" || matmul.getMath() != "native" ||
        !matmul.getAccDtype().isF32())
      return std::nullopt;

    std::optional<StructuredProductFacts> product = structuredProductFacts(
        matmul.getLhs(), matmul.getRhs(), matmul.getResult());
    if (!product || product->lhsFreeAxes.size() != 1 ||
        product->rhsFreeAxes.size() != 1 ||
        product->reductionAxes.size() != 1 ||
        product->resultAxes.size() != 2)
      return std::nullopt;
    BlockIndexOp lhsRowAxis =
        product->lhsFreeAxes.front().getDefiningOp<BlockIndexOp>();
    BlockIndexOp reductionAxis =
        product->reductionAxes.front().getDefiningOp<BlockIndexOp>();
    BlockIndexOp rhsColumnAxis =
        product->rhsFreeAxes.front().getDefiningOp<BlockIndexOp>();
    if (!lhsRowAxis || !reductionAxis || !rhsColumnAxis)
      return std::nullopt;
    std::optional<int64_t> rowTile =
        physicalExtent(lhsRowAxis.getExtent());
    std::optional<int64_t> columnTile =
        physicalExtent(rhsColumnAxis.getExtent());
    std::optional<int64_t> reductionTile =
        physicalExtent(reductionAxis.getExtent());
    if (!rowTile || *rowTile <= 0 || *rowTile > 8 || !columnTile ||
        *columnTile <= 0 || !reductionTile || *reductionTile <= 0)
      return std::nullopt;
    const MemoryAccessFact *lhsAccess = memoryFact(lhsLoad.getOperation());
    const MemoryAccessFact *rhsAccess = memoryFact(rhsLoad.getOperation());
    if (!lhsAccess || !rhsAccess ||
        memoryRelation(*lhsAccess, reductionAxis.getResult()) !=
            LaneRelation::UnitStride ||
        memoryRelation(*rhsAccess, reductionAxis.getResult()) !=
            LaneRelation::UnitStride ||
        memoryRelation(*lhsAccess, lhsRowAxis.getResult()) ==
            LaneRelation::Independent ||
        memoryRelation(*lhsAccess, rhsColumnAxis.getResult()) !=
            LaneRelation::Independent ||
        memoryRelation(*rhsAccess, rhsColumnAxis.getResult()) ==
            LaneRelation::Independent ||
        memoryRelation(*rhsAccess, lhsRowAxis.getResult()) !=
            LaneRelation::Independent)
      return std::nullopt;
    if (!isContiguousPrefixPredicate(lhsLoad.getWhere(),
                                     lhsRowAxis.getResult()) ||
        !isContiguousPrefixPredicate(lhsLoad.getWhere(),
                                     reductionAxis.getResult()) ||
        !isContiguousPrefixPredicate(rhsLoad.getWhere(),
                                     reductionAxis.getResult()) ||
        !isContiguousPrefixPredicate(rhsLoad.getWhere(),
                                     rhsColumnAxis.getResult()))
      return std::nullopt;

    PlannedPhysicalDecision<F16GemmNTileDecision> planned;
    F16GemmNTileDecision &decision = planned.realization;
    decision.operation = matmul.getOperation();
    decision.lhsLoad = lhsLoad.getOperation();
    decision.rhsLoad = rhsLoad.getOperation();
    decision.lhsRowAxis = lhsRowAxis.getResult();
    decision.lhsReductionAxis = reductionAxis.getResult();
    decision.rhsReductionAxis = reductionAxis.getResult();
    decision.rhsColumnAxis = rhsColumnAxis.getResult();
    decision.rowTile = static_cast<unsigned>(*rowTile);
    decision.columnTile = static_cast<unsigned>(*columnTile);
    decision.reductionTile = static_cast<unsigned>(*reductionTile);
    F16MatmulCandidateFacts candidateFacts{decision.rowTile,
                                           decision.reductionTile};
    std::optional<SelectedF16MatmulPhysical> selected =
        weft::riscv_internal::selectF16MatmulPhysicalConfig(
            candidateFacts, options.target, options.backend);
    if (!selected)
      return std::nullopt;
    unsigned inputLMUL = selected->config.inputLMUL;
    unsigned computeLMUL = 2 * selected->config.inputLMUL;
    decision.physical = selected->config;
    if (mlir::failed(selectMaterializedBlockStorage(matmul.getInit(),
                                                   matmul.getOperation())))
      return std::nullopt;
    RVVVectorShape inputShape = rvvShape(16, inputLMUL);
    RVVVectorShape computeShape = rvvShape(32, computeLMUL);
    recordPhysicalValue(planned.entity, matmul.getLhs(), inputShape);
    recordPhysicalValue(planned.entity, matmul.getRhs(), inputShape);
    recordPhysicalValue(planned.entity, matmul.getInit(), computeShape);
    recordPhysicalValue(planned.entity, matmul.getResult(), computeShape);
    recordPhysicalHandoff(planned.entity, matmul.getOperation(), matmul.getLhs(),
                          PhysicalHandoff::Reload, inputShape, inputShape);
    recordPhysicalHandoff(planned.entity, matmul.getOperation(), matmul.getRhs(),
                          PhysicalHandoff::Reload, inputShape, inputShape);
    recordPhysicalHandoff(planned.entity, matmul.getOperation(), matmul.getInit(),
                          PhysicalHandoff::LocalPack, computeShape, computeShape);
    planned.entity.resources = selected->resources;
    return planned;
  }

  mlir::LogicalResult emitF16Matmul(MatmulOp op) {
    auto prepared = physicalPlan.f16Matmuls.find(op.getOperation());
    if (prepared == physicalPlan.f16Matmuls.end())
      return op.emitError("matmul has no selected local physical decision");
    if (mlir::failed(requireEntityPlan(op.getOperation(), prepared->second)))
      return mlir::failure();
    const F16GemmNTileDecision &decision = prepared->second.realization;
    const PhysicalEntityPlan &entity = prepared->second.entity;
    const F16MatmulPhysicalConfig &physical = decision.physical;
    bool scheduleMismatch =
        (physical.pipelineStages == 1 &&
         physical.loadSchedule !=
             F16MatmulLoadSchedule::StreamRHSThenRows) ||
        (physical.pipelineStages == 2 &&
         physical.loadSchedule !=
             F16MatmulLoadSchedule::BatchLoadsThenCompute);
    if (physical.rowMicrotile == 0 || physical.kUnroll == 0 ||
        physical.pipelineStages == 0 || physical.pipelineStages > 2 ||
        scheduleMismatch ||
        entity.resources.peakGroups > entity.resources.architecturalGroups)
      return op.emitError("F16 matmul has no selected K strip schedule");
    auto inputShape = llvm::find_if(
        entity.values, [&](const PhysicalValueDecision &value) {
          return value.value == op.getLhs();
        });
    auto computeShape = llvm::find_if(
        entity.values, [&](const PhysicalValueDecision &value) {
          return value.value == op.getResult();
        });
    if (inputShape == entity.values.end() || computeShape == entity.values.end())
      return op.emitError("F16 matmul has no physical value shape");
    std::optional<unsigned> inputLMUL = rvvIntegerLMUL(inputShape->shape);
    std::optional<unsigned> computeLMUL = rvvIntegerLMUL(computeShape->shape);
    if (!inputLMUL || !computeLMUL || inputShape->shape.sew != 16 ||
        computeShape->shape.sew != 32 || *inputLMUL != physical.inputLMUL ||
        *computeLMUL != 2 * physical.inputLMUL)
      return op.emitError("F16 matmul value shape has no intrinsic-C spelling");
    CValue accumulator = require(op.getInit());
    if (accumulator.kind != CValueKind::F32BlockStorage ||
        accumulator.spelling.empty())
      return op.emitError("F16 matmul accumulator storage is unavailable");
    LoadOp lhsLoad = mlir::cast<LoadOp>(decision.lhsLoad);
    LoadOp rhsLoad = mlir::cast<LoadOp>(decision.rhsLoad);

    auto project = [&](mlir::Value value, llvm::StringRef row,
                       llvm::StringRef reduction,
                       llvm::StringRef column) -> std::optional<std::string> {
      llvm::DenseMap<mlir::Value, std::string> axes;
      axes[decision.lhsRowAxis] = row.str();
      axes[decision.lhsReductionAxis] = reduction.str();
      axes[decision.rhsReductionAxis] = reduction.str();
      axes[decision.rhsColumnAxis] = column.str();
      return projectBlockScalar(value, axes);
    };

    std::string column = fresh("matmul_n");
    line("for (size_t " + column + " = 0; " + column + " < " +
         std::to_string(decision.columnTile) + "; ++" + column + ") {");
    ++indent;
    std::optional<std::string> finalPredicate =
        project(lhsLoad.getWhere(), "0", "(" +
                    std::to_string(decision.reductionTile) + " - 1)",
                column);
    std::optional<std::string> finalRhsPredicate =
        project(rhsLoad.getWhere(), "0", "(" +
                    std::to_string(decision.reductionTile) + " - 1)",
                column);
    if (!finalPredicate || !finalRhsPredicate)
      return op.emitError("F16 matmul predicate projection is unavailable");
    std::string activeK = fresh("matmul_active_k");
    line("size_t " + activeK + " = " +
         std::to_string(decision.reductionTile) + ";");
    std::optional<std::string> tailPredicate =
        project(lhsLoad.getWhere(), "0", "(" + activeK + " - 1)", column);
    std::optional<std::string> tailRhsPredicate =
        project(rhsLoad.getWhere(), "0", "(" + activeK + " - 1)", column);
    if (!tailPredicate || !tailRhsPredicate)
      return op.emitError("F16 matmul tail predicate projection is unavailable");
    line("while (" + activeK + " > 0 && !(" + *tailPredicate + " && " +
         *tailRhsPredicate + ")) --" + activeK + ";");

    for (unsigned rowBase = 0; rowBase < decision.rowTile;
         rowBase += physical.rowMicrotile) {
      unsigned rowCount =
          std::min(physical.rowMicrotile, decision.rowTile - rowBase);
      std::string fullVL = fresh("matmul_full_vl");
      line("const size_t " + fullVL + " = __riscv_vsetvlmax_e16m" +
           std::to_string(*inputLMUL) + "();");
      llvm::SmallVector<std::string> accumulators;
      llvm::SmallVector<std::string> rowPredicates;
      for (unsigned row = 0; row < rowCount; ++row) {
        std::string rowIndex = std::to_string(rowBase + row);
        std::optional<std::string> active =
            project(lhsLoad.getWhere(), rowIndex, "0", column);
        std::optional<std::string> rhsActive =
            project(rhsLoad.getWhere(), rowIndex, "0", column);
        if (!active || !rhsActive)
          return op.emitError("F16 matmul row predicate projection is unavailable");
        rowPredicates.push_back("(" + *active + " && " + *rhsActive + ")");
        std::string vector = fresh("matmul_acc");
        line("vfloat32m" + std::to_string(*computeLMUL) + "_t " +
             vector + " = __riscv_vfmv_v_f_f32m" +
             std::to_string(*computeLMUL) + "(0.0f, " + fullVL +
             ");");
        accumulators.push_back(std::move(vector));
      }
      std::string reduction = fresh("matmul_k");
      std::string vl = fresh("matmul_vl");
      auto coordinateAt = [&](llvm::StringRef base, unsigned unroll) {
        return unroll == 0
                   ? base.str()
                   : "(" + base.str() + " + " +
                         std::to_string(unroll) + " * " + vl + ")";
      };
      auto emitStreamedChunk = [&](llvm::StringRef coordinate)
          -> mlir::LogicalResult {
        std::optional<std::string> rhsPointer =
            project(rhsLoad.getPointer(), "0", coordinate, column);
        if (!rhsPointer)
          return op.emitError("F16 matmul RHS address projection is unavailable");
        std::string rhsVector = fresh("matmul_rhs");
        line("vfloat16m" + std::to_string(*inputLMUL) + "_t " +
             rhsVector + " = __riscv_vle16_v_f16m" +
             std::to_string(*inputLMUL) + "(" + *rhsPointer + ", " + vl +
             ");");
        for (unsigned row = 0; row < rowCount; ++row) {
          std::string rowIndex = std::to_string(rowBase + row);
          std::optional<std::string> lhsPointer =
              project(lhsLoad.getPointer(), rowIndex, coordinate, column);
          if (!lhsPointer)
            return op.emitError("F16 matmul LHS address projection is unavailable");
          line("if (" + rowPredicates[row] + ") {");
          ++indent;
          std::string lhsVector = fresh("matmul_lhs");
          line("vfloat16m" + std::to_string(*inputLMUL) + "_t " +
               lhsVector + " = __riscv_vle16_v_f16m" +
               std::to_string(*inputLMUL) + "(" + *lhsPointer + ", " + vl +
               ");");
          line(accumulators[row] + " = __riscv_vfwmacc_vv_f32m" +
               std::to_string(*computeLMUL) + "_tu(" + accumulators[row] +
               ", " + lhsVector + ", " + rhsVector + ", " + vl + ");");
          --indent;
          line("}");
        }
        return mlir::success();
      };
      auto emitBatchedChunks = [&](llvm::StringRef base, unsigned count)
          -> mlir::LogicalResult {
        llvm::SmallVector<std::string> rhsVectors;
        for (unsigned unroll = 0; unroll < count; ++unroll) {
          std::string coordinate = coordinateAt(base, unroll);
          std::optional<std::string> rhsPointer =
              project(rhsLoad.getPointer(), "0", coordinate, column);
          if (!rhsPointer)
            return op.emitError("F16 matmul RHS address projection is unavailable");
          std::string rhsVector = fresh("matmul_rhs_stage");
          line("vfloat16m" + std::to_string(*inputLMUL) + "_t " +
               rhsVector + " = __riscv_vle16_v_f16m" +
               std::to_string(*inputLMUL) + "(" + *rhsPointer + ", " + vl +
               ");");
          rhsVectors.push_back(std::move(rhsVector));
        }
        for (unsigned row = 0; row < rowCount; ++row) {
          line("if (" + rowPredicates[row] + ") {");
          ++indent;
          llvm::SmallVector<std::string> lhsVectors;
          for (unsigned unroll = 0; unroll < count; ++unroll) {
            std::string coordinate = coordinateAt(base, unroll);
            std::optional<std::string> lhsPointer =
                project(lhsLoad.getPointer(),
                        std::to_string(rowBase + row), coordinate, column);
            if (!lhsPointer)
              return op.emitError("F16 matmul LHS address projection is unavailable");
            std::string lhsVector = fresh("matmul_lhs_stage");
            line("vfloat16m" + std::to_string(*inputLMUL) + "_t " +
                 lhsVector + " = __riscv_vle16_v_f16m" +
                 std::to_string(*inputLMUL) + "(" + *lhsPointer + ", " + vl +
                 ");");
            lhsVectors.push_back(std::move(lhsVector));
          }
          for (unsigned unroll = 0; unroll < count; ++unroll)
            line(accumulators[row] + " = __riscv_vfwmacc_vv_f32m" +
                 std::to_string(*computeLMUL) + "_tu(" + accumulators[row] +
                 ", " + lhsVectors[unroll] + ", " + rhsVectors[unroll] +
                 ", " + vl + ");");
          --indent;
          line("}");
        }
        return mlir::success();
      };
      line("for (size_t " + reduction + " = 0; " + reduction + " < " +
           activeK + ";) {");
      ++indent;
      if (physical.kUnroll == 1) {
        line("const size_t " + vl + " = __riscv_vsetvl_e16m" +
             std::to_string(*inputLMUL) + "(" + activeK + " - " +
             reduction + ");");
        if (mlir::failed(emitStreamedChunk(reduction)))
          return mlir::failure();
        line(reduction + " += " + vl + ";");
      } else {
        std::string remaining = fresh("matmul_remaining");
        line("const size_t " + remaining + " = " + activeK + " - " +
             reduction + ";");
        line("if (" + remaining + " >= " +
             std::to_string(physical.kUnroll) + ") {");
        ++indent;
        line("const size_t " + vl + " = __riscv_vsetvl_e16m" +
             std::to_string(*inputLMUL) + "(" + remaining + " / " +
             std::to_string(physical.kUnroll) + ");");
        if (physical.loadSchedule ==
            F16MatmulLoadSchedule::BatchLoadsThenCompute) {
          if (mlir::failed(
                  emitBatchedChunks(reduction, physical.kUnroll)))
            return mlir::failure();
        } else {
          for (unsigned unroll = 0; unroll < physical.kUnroll; ++unroll)
            if (mlir::failed(
                    emitStreamedChunk(coordinateAt(reduction, unroll))))
              return mlir::failure();
        }
        line(reduction + " += " + std::to_string(physical.kUnroll) +
             " * " + vl + ";");
        --indent;
        line("} else {");
        ++indent;
        line("const size_t " + vl + " = __riscv_vsetvl_e16m" +
             std::to_string(*inputLMUL) + "(" + remaining + ");");
        if (mlir::failed(emitStreamedChunk(reduction)))
          return mlir::failure();
        line(reduction + " += " + vl + ";");
        --indent;
        line("}");
      }
      --indent;
      line("}");
      for (unsigned row = 0; row < rowCount; ++row) {
        std::string seed = fresh("matmul_seed");
        std::string partial = fresh("matmul_partial");
        std::string scalar = fresh("matmul_sum");
        line("if (" + rowPredicates[row] + ") {");
        ++indent;
        line("vfloat32m1_t " + seed +
             " = __riscv_vfmv_v_f_f32m1(0.0f, 1);");
        line("vfloat32m1_t " + partial +
             " = __riscv_vfredusum_vs_f32m" +
             std::to_string(*computeLMUL) + "_f32m1(" +
             accumulators[row] + ", " + seed + ", " + fullVL + ");");
        line("const float " + scalar +
             " = __riscv_vfmv_f_s_f32m1_f32(" + partial + ");");
        line(accumulator.spelling + "[" +
             std::to_string(rowBase + row) + " * " +
             std::to_string(decision.columnTile) + " + " + column + "] += " +
             scalar + ";");
        --indent;
        line("}");
      }
    }
    --indent;
    line("}");
    values[op.getResult()] = accumulator;
    values[op.getResult()].type = op.getResult().getType();
    if (mlir::failed(markRematerializedBlockTrees(
            {op.getLhs(), op.getRhs()}, op.getOperation())))
      return mlir::failure();
    loweredBlockOps.insert(op.getOperation());
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

  BlockOperationDecision decideBlockOperation(mlir::Operation *operation,
                                              RVVVectorShape byteShape,
                                              PhysicalEntityPlan &entity,
                                              const llvm::DenseMap<
                                                  mlir::Operation *,
                                                  BlockOperationDecision>
                                                  &prior) const {
    BlockOperationDecision decision;
    decision.operation = operation;
    auto setResultShape = [&](RVVVectorShape shape) {
      if (!shape)
        return;
      for (mlir::Value result : operation->getResults())
        recordPhysicalValue(entity, result, shape);
    };
    auto sameLanes = [&](unsigned sew) {
      return rvvShapeForSameLanes(byteShape, sew, options.target)
          .value_or(RVVVectorShape{});
    };
    auto valueShape = [&](mlir::Value value) {
      auto found = llvm::find_if(
          entity.values, [&](const PhysicalValueDecision &physical) {
            return physical.value == value;
          });
      return found == entity.values.end() ? RVVVectorShape{} : found->shape;
    };
    if (mlir::isa<BlockIndexOp>(operation)) {
      setResultShape(sameLanes(16));
      return decision;
    }
    if (auto load = mlir::dyn_cast<LoadOp>(operation)) {
      if (elementType(load.getResult().getType()).isUnsignedInteger(8))
        setResultShape(byteShape);
      return decision;
    }
    if (auto bitcast = mlir::dyn_cast<BitcastOp>(operation)) {
      mlir::Type target = elementType(bitcast.getResult().getType());
      if (target.isSignedInteger(8))
        setResultShape(byteShape);
      else if (target.isSignedInteger(16))
        setResultShape(sameLanes(16));
      return decision;
    }
    if (auto compare = mlir::dyn_cast<CompareOp>(operation)) {
      RVVVectorShape inputShape = valueShape(compare.getLhs());
      if (!inputShape)
        inputShape = valueShape(compare.getRhs());
      setResultShape(inputShape);
      decision.maskRatio = rvvMaskRatio(inputShape).value_or(0);
      return decision;
    }
    if (auto cast = mlir::dyn_cast<CastOp>(operation)) {
      mlir::Type source = elementType(cast.getInput().getType());
      mlir::Type target = elementType(cast.getResult().getType());
      RVVVectorShape sourceShape = valueShape(cast.getInput());
      if (!sourceShape &&
          (source.isUnsignedInteger(8) || source.isSignedInteger(8)))
        sourceShape = byteShape;
      if (!sourceShape && source.isIndex())
        sourceShape = sameLanes(16);
      unsigned targetSEW = target.isUnsignedInteger(8) ? 8
                           : target.isUnsignedInteger(16) ||
                                   target.isSignedInteger(16)
                               ? 16
                           : target.isSignedInteger(32) || target.isF32() ? 32
                                                                          : 0;
      RVVVectorShape resultShape =
          rvvShapeForSameLanes(sourceShape, targetSEW, options.target)
              .value_or(RVVVectorShape{});
      if (targetSEW != 0)
        setResultShape(resultShape);
      if (target.isUnsignedInteger(8)) {
        return decision;
      } else if (target.isUnsignedInteger(16)) {
        return decision;
      } else if (target.isSignedInteger(32) &&
               (source.isUnsignedInteger(8) || source.isSignedInteger(8))) {
        decision.realization = BlockValueRealization::DeferredI32Widen;
        recordPhysicalTemporary(entity, operation, resultShape);
      } else if (target.isSignedInteger(32) && source.isUnsignedInteger(16)) {
        recordPhysicalTemporary(entity, operation, resultShape);
      } else if (target.isF32()) {
        RVVVectorShape intermediate = resultShape == kRVVE32M1
                                          ? resultShape
                                          : rvvShapeForSameLanes(
                                                sourceShape, 16, options.target)
                                                .value_or(RVVVectorShape{});
        recordPhysicalTemporary(entity, operation, intermediate);
      }
      return decision;
    }
    if (auto binary = mlir::dyn_cast<BinaryOp>(operation)) {
      mlir::Type result = elementType(binary.getResult().getType());
      if (result.isIndex())
        setResultShape(sameLanes(16));
      else if (result.isUnsignedInteger(8)) {
        setResultShape(byteShape);
        if (byteShape == kRVVE8MF4 &&
            (binary.getKind() == "and" || binary.getKind() == "shr")) {
          decision.realization =
              BlockValueRealization::DeferredPackedTransform;
          decision.source = binary.getLhs();
          decision.transform = binary.getKind().str();
        }
        if (binary.getKind() == "and") {
          std::optional<int64_t> mask = integerConstant(binary.getRhs());
          if (mask && *mask >= 0)
            decision.unsignedMaximum = static_cast<uint64_t>(*mask);
        }
      } else if (result.isUnsignedInteger(16))
        setResultShape(sameLanes(16));
      else if (result.isSignedInteger(32)) {
        auto narrowSource = [](mlir::Value value) -> mlir::Value {
          auto cast = value.getDefiningOp<CastOp>();
          return cast && elementType(cast.getResult().getType()).isSignedInteger(32)
                     ? cast.getInput()
                     : mlir::Value{};
        };
        mlir::Value lhsNarrow = narrowSource(binary.getLhs());
        mlir::Value rhsNarrow = narrowSource(binary.getRhs());
        auto unsignedMaximum = [&](mlir::Value value)
            -> std::optional<uint64_t> {
          mlir::Operation *definition = value.getDefiningOp();
          auto found = definition ? prior.find(definition) : prior.end();
          return found == prior.end() ? std::nullopt
                                      : found->second.unsignedMaximum;
        };
        bool lhsSmallU8 = lhsNarrow &&
                          elementType(lhsNarrow.getType()).isUnsignedInteger(8) &&
                          unsignedMaximum(lhsNarrow).value_or(256) <= 15;
        bool rhsSmallU8 = rhsNarrow &&
                          elementType(rhsNarrow.getType()).isUnsignedInteger(8) &&
                          unsignedMaximum(rhsNarrow).value_or(256) <= 15;
        bool lhsI8 = lhsNarrow &&
                     elementType(lhsNarrow.getType()).isSignedInteger(8);
        bool rhsI8 = rhsNarrow &&
                     elementType(rhsNarrow.getType()).isSignedInteger(8);
        if (binary.getKind() == "mul" &&
            ((lhsSmallU8 && rhsI8) || (rhsSmallU8 && lhsI8))) {
          decision.realization = BlockValueRealization::WideningI8Product;
          setResultShape(sameLanes(16));
        } else {
          setResultShape(sameLanes(32));
        }
      } else if (result.isF32()) {
        for (mlir::Value operand : binary->getOperands()) {
          RVVVectorShape producerShape = valueShape(operand);
          if (producerShape && producerShape.sew == 32) {
            setResultShape(producerShape);
            break;
          }
        }
      }
      return decision;
    }
    if (auto select = mlir::dyn_cast<SelectOp>(operation)) {
      mlir::Type result = elementType(select.getResult().getType());
      if (result.isUnsignedInteger(8))
        setResultShape(byteShape);
      return decision;
    }
    return decision;
  }

  llvm::SmallVector<BlockOperationDecision> decideBlockOperations(
      const llvm::DenseSet<mlir::Operation *> &closure,
      RVVVectorShape byteShape, PhysicalEntityPlan &entity) const {
    llvm::SmallVector<mlir::Operation *> ordered;
    llvm::DenseSet<mlir::Operation *> visited;
    std::function<void(mlir::Operation *)> visit = [&](mlir::Operation *operation) {
      if (!operation || !closure.contains(operation) ||
          !visited.insert(operation).second)
        return;
      for (mlir::Value operand : operation->getOperands())
        visit(operand.getDefiningOp());
      ordered.push_back(operation);
    };
    for (mlir::Operation *operation : closure)
      visit(operation);
    llvm::SmallVector<BlockOperationDecision> decisions;
    llvm::DenseMap<mlir::Operation *, BlockOperationDecision> prior;
    for (mlir::Operation *operation : ordered) {
      BlockOperationDecision decision =
          decideBlockOperation(operation, byteShape, entity, prior);
      prior.try_emplace(operation, decision);
      decisions.push_back(std::move(decision));
    }
    return decisions;
  }

  mlir::LogicalResult collectBlockClosure(
      mlir::Value value, llvm::DenseSet<mlir::Operation *> &closure,
      llvm::SmallVectorImpl<BlockIndexOp> &axes, mlir::Operation *owner,
      bool allowCaptured = false) {
    if (!containsBlockType(value.getType()) && !isRegionValue(value.getType()))
      return mlir::success();
    mlir::Operation *definition = value.getDefiningOp();
    if (!definition && allowCaptured)
      return mlir::success();
    if (!definition)
      return owner->emitError(
          "block operation cannot capture a block argument from another region");
    if (definition->getBlock() != owner->getBlock() && !allowCaptured)
      return owner->emitError(
          "block producer closure crosses a region boundary");
    if (!closure.insert(definition).second)
      return mlir::success();
    if (auto get = mlir::dyn_cast<TupleGetOp>(definition)) {
      auto tuple = get.getInput().getDefiningOp<TupleOp>();
      if (!tuple)
        return get.emitError(
            "block tuple projection requires a local tuple producer");
      if (tuple->getBlock() != owner->getBlock() && !allowCaptured)
        return owner->emitError(
            "block tuple producer closure crosses a region boundary");
      closure.insert(tuple.getOperation());
      for (auto [index, operand] : llvm::enumerate(tuple.getOperands())) {
        if (static_cast<int64_t>(index) == get.getIndex()) {
          if (mlir::failed(collectBlockClosure(operand, closure, axes, owner,
                                               allowCaptured)))
            return mlir::failure();
        }
      }
      return mlir::success();
    }
    if (auto axis = mlir::dyn_cast<BlockIndexOp>(definition))
      axes.push_back(axis);
    for (mlir::Value operand : definition->getOperands())
      if ((containsBlockType(operand.getType()) ||
           isRegionValue(operand.getType())) &&
          mlir::failed(collectBlockClosure(operand, closure, axes, owner,
                                           allowCaptured)))
        return mlir::failure();
    return mlir::success();
  }

  mlir::LogicalResult markRematerializedBlockTrees(
      llvm::ArrayRef<mlir::Value> values, mlir::Operation *consumer) {
    llvm::DenseSet<mlir::Operation *> closure;
    llvm::SmallVector<BlockIndexOp> axes;
    for (mlir::Value value : values) {
      if (!containsBlockType(value.getType()) && !isRegionValue(value.getType()))
        continue;
      if (mlir::failed(
              collectBlockClosure(value, closure, axes, consumer, true)))
        return mlir::failure();
    }
    loweredBlockOps.insert(closure.begin(), closure.end());
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
    const BlockOperationDecision *physical =
        findBlockOperationDecision(op.getOperation());
    const PhysicalValueDecision *valuePlan =
        findBlockValueDecision(op.getResult());
    if (!physical || !valuePlan || valuePlan->shape.sew != 16)
      return op.emitError("logical-index binary has no selected u16 shape");
    std::string shape = rvvShapeSuffix(valuePlan->shape);
    if (shape.empty())
      return op.emitError("logical-index binary shape has no RVV spelling");
    std::string suffix = "u" + shape;
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
      std::string stem = llvm::StringSwitch<std::string>(op.getKind())
                             .Case("add", "vadd")
                             .Case("sub", "vsub")
                             .Case("mul", "vmul")
                             .Case("and", "vand")
                             .Case("or", "vor")
                             .Default("");
      if (!stem.empty())
        intrinsic = "__riscv_" + stem + "_vv_" + suffix;
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
          intrinsic = "__riscv_vsrl_vx_" + suffix;
          rhsSpelling = std::to_string(shift);
        } else {
          intrinsic = "__riscv_vand_vx_" + suffix;
          rhsSpelling = std::to_string(*constant - 1);
        }
      } else {
        std::string stem = llvm::StringSwitch<std::string>(op.getKind())
                               .Case("add", "vadd")
                               .Case("sub", "vsub")
                               .Case("mul", "vmul")
                               .Case("div", "vdivu")
                               .Case("mod", "vremu")
                               .Case("and", "vand")
                               .Case("or", "vor")
                               .Case("shl", "vsll")
                               .Case("shr", "vsrl")
                               .Default("");
        if (!stem.empty())
          intrinsic = "__riscv_" + stem + "_vx_" + suffix;
      }
    }
    if (intrinsic.empty())
      return op.emitError("RVV logical-index binary kind is unsupported");
    std::string name = fresh("block_index");
    line("vuint" + shape + "_t " + name + " = " + intrinsic + "(" + lhs.spelling +
         ", " + rhsSpelling + ", " + vl.str() + ");");
    BlockValue result{op.getResult().getType(), BlockValueKind::Index, name};
    result.vectorShape = valuePlan->shape;
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
    if (value.vectorShape.sew != 32)
      return owner->emitError("deferred i32 block value has no selected shape");
    std::string resultShape = rvvShapeSuffix(value.vectorShape);
    if (resultShape.empty())
      return owner->emitError("deferred i32 block shape has no RVV spelling");
    std::string name = fresh("block_i32");
    if (value.narrowKind == BlockValueKind::U8) {
      std::string extended = fresh("block_u32");
      line("vuint" + resultShape + "_t " + extended +
           " = __riscv_vzext_vf4_u" + resultShape + "(" +
           value.narrowSpelling + ", " + vl.str() + ");");
      line("vint" + resultShape + "_t " + name +
           " = __riscv_vreinterpret_v_u" + resultShape + "_i" +
           resultShape + "(" + extended + ");");
    } else if (value.narrowKind == BlockValueKind::I8) {
      line("vint" + resultShape + "_t " + name +
           " = __riscv_vsext_vf4_i" + resultShape + "(" +
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
    const BlockOperationDecision *physical =
        findBlockOperationDecision(op.getOperation());
    const PhysicalValueDecision *valuePlan =
        findBlockValueDecision(op.getResult());
    if (!physical || !valuePlan || !valuePlan->shape)
      return op.emitError("integer block binary has no physical value decision");
    RVVVectorShape selectedShape = valuePlan->shape;
    if (physical->realization ==
        BlockValueRealization::DeferredPackedTransform) {
      if (resultKind != BlockValueKind::U8 || lhs.kind != BlockValueKind::U8 ||
          lhs.spelling.empty() || rhs.kind != BlockValueKind::Scalar ||
          rhs.spelling.empty())
        return op.emitError(
            "deferred packed transform operands do not match its decision");
      BlockValue result;
      result.type = op.getResult().getType();
      result.kind = BlockValueKind::U8;
      std::string suffix = rvvShapeSuffix(selectedShape);
      std::string stem = physical->transform == "and" ? "vand" : "vsrl";
      std::string name = fresh("block_packed_u8");
      line("__attribute__((unused)) vuint" + suffix + "_t " + name +
           " = __riscv_" + stem +
           "_vx_u" + suffix + "(" + lhs.spelling + ", " + rhs.spelling +
           ", " + vl.str() + ");");
      result.spelling = name;
      result.packedSource = physical->source;
      result.packedTransform = physical->transform;
      result.packedTransformOperand = rhs.spelling;
      result.vectorShape = selectedShape;
      result.unsignedMaximum = physical->unsignedMaximum;
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
    if (physical->realization == BlockValueRealization::WideningI8Product) {
      if (resultKind != BlockValueKind::I32 || !rhsVector ||
          op.getKind() != "mul" ||
          !((isSmallU8(lhs) && isNarrowI8(rhs)) ||
            (isNarrowI8(lhs) && isSmallU8(rhs))))
        return op.emitError(
            "widening i8 product operands do not match its decision");
      BlockValue small = isSmallU8(lhs) ? lhs : rhs;
      BlockValue signedValue = isNarrowI8(lhs) ? lhs : rhs;
      RVVVectorShape productShape = selectedShape;
      RVVVectorShape sourceShape =
          rvvShapeForSameLanes(productShape, 8, options.target)
              .value_or(RVVVectorShape{});
      std::string productSuffix = rvvShapeSuffix(productShape);
      std::string sourceSuffix = rvvShapeSuffix(sourceShape);
      if (productShape.sew != 16 || sourceShape.sew != 8 ||
          productSuffix.empty() || sourceSuffix.empty())
        return op.emitError(
            "widening i8 product has no physical RVV spelling");
      std::string signedSmall = fresh("block_i8");
      std::string name = fresh("block_i16_product");
      line("vint" + sourceSuffix + "_t " + signedSmall +
           " = __riscv_vreinterpret_v_u" + sourceSuffix + "_i" +
           sourceSuffix + "(" + small.narrowSpelling + ");");
      line("vint" + productSuffix + "_t " + name +
           " = __riscv_vwmul_vv_i" + productSuffix + "(" +
           signedSmall + ", " + signedValue.narrowSpelling + ", " + vl.str() +
           ");");
      BlockValue result{op.getResult().getType(), BlockValueKind::I16, name};
      result.vectorShape = productShape;
      blockValues[op.getResult()] = std::move(result);
      return mlir::success();
    }
    if (resultKind == BlockValueKind::I32) {
      if (mlir::failed(materializeI32(lhs, op.getOperation(), vl)) ||
          mlir::failed(materializeI32(rhs, op.getOperation(), vl)))
        return mlir::failure();
    }

    std::string suffix;
    std::string cType;
    unsigned expectedSEW = resultKind == BlockValueKind::U8 ? 8
                           : resultKind == BlockValueKind::U16 ? 16
                           : resultKind == BlockValueKind::I32 ? 32
                                                               : 0;
    std::string shapeSuffix = rvvShapeSuffix(selectedShape);
    if (expectedSEW == 0 || selectedShape.sew != expectedSEW ||
        shapeSuffix.empty())
      return op.emitError("integer block type has no physical RVV mapping");
    bool signedResult = resultKind == BlockValueKind::I32;
    suffix = std::string(signedResult ? "i" : "u") + shapeSuffix;
    cType = std::string(signedResult ? "vint" : "vuint") + shapeSuffix + "_t";
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
    result.vectorShape = selectedShape;
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
    const BlockOperationDecision *physical =
        findBlockOperationDecision(op.getOperation());
    const PhysicalValueDecision *valuePlan =
        findBlockValueDecision(op.getResult());
    if (!physical || !valuePlan || !valuePlan->shape)
      return op.emitError("f32 block binary has no physical value decision");
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
    RVVVectorShape shape = valuePlan->shape;
    if (lhsVector && rhsVector && lhs.vectorShape != rhs.vectorShape)
      return op.emitError("f32 block binary has incompatible physical vectors");
    if ((lhsVector && lhs.vectorShape != shape) ||
        (rhsVector && rhs.vectorShape != shape))
      return op.emitError("f32 block binary operands do not match its decision");
    std::string shapeSuffix = rvvShapeSuffix(shape);
    if (shape.sew != 32 || shapeSuffix.empty())
      return op.emitError("f32 block binary has no physical RVV shape");
    std::string suffix = "f" + shapeSuffix;
    std::string cType = "vfloat" + shapeSuffix + "_t";
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
    const BlockOperationDecision *physical =
        findBlockOperationDecision(op.getOperation());
    const PhysicalValueDecision *valuePlan =
        findBlockValueDecision(op.getResult());
    if (!physical || !valuePlan || !valuePlan->shape)
      return op.emitError("block comparison has no physical mask decision");
    BlockValue lhs = lookupBlockValue(op.getLhs(), blockValues);
    BlockValue rhs = lookupBlockValue(op.getRhs(), blockValues);
    if (lhs.kind != BlockValueKind::Index ||
        rhs.kind != BlockValueKind::Scalar || rhs.spelling.empty())
      return op.emitError(
          "RVV block comparison currently requires index-vector vs scalar");
    RVVVectorShape inputShape = valuePlan->shape;
    std::string inputSuffix = rvvShapeSuffix(inputShape);
    std::string stem = llvm::StringSwitch<std::string>(op.getPredicate())
                           .Case("lt", "vmsltu")
                           .Case("le", "vmsleu")
                           .Case("gt", "vmsgtu")
                           .Case("ge", "vmsgeu")
                           .Case("eq", "vmseq")
                           .Case("ne", "vmsne")
                           .Default("");
    if (stem.empty() || inputShape.sew != 16 || inputSuffix.empty() ||
        physical->maskRatio == 0 ||
        (lhs.vectorShape && lhs.vectorShape != inputShape))
      return op.emitError("RVV block comparison predicate is unsupported");
    std::string intrinsic = "__riscv_" + stem + "_vx_u" + inputSuffix +
                            "_b" + std::to_string(physical->maskRatio);
    std::string name = fresh("block_mask");
    line("vbool" + std::to_string(physical->maskRatio) + "_t " + name +
         " = " + intrinsic + "(" + lhs.spelling + ", " + rhs.spelling +
         ", " + vl.str() + ");");
    BlockValue result{op.getResult().getType(), BlockValueKind::Mask, name};
    result.vectorShape = inputShape;
    blockValues[op.getResult()] = std::move(result);
    return mlir::success();
  }

  mlir::LogicalResult emitBlockCast(
      CastOp op, llvm::DenseMap<mlir::Value, BlockValue> &blockValues,
      llvm::StringRef vl) {
    const BlockOperationDecision *physical =
        findBlockOperationDecision(op.getOperation());
    const PhysicalValueDecision *valuePlan =
        findBlockValueDecision(op.getResult());
    if (!physical || !valuePlan || !valuePlan->shape)
      return op.emitError("block cast has no physical value decision");
    BlockValue input = lookupBlockValue(op.getInput(), blockValues);
    mlir::Type target = elementType(op.getResult().getType());
    std::string name = fresh("block_cast");
    BlockValueKind kind;
    RVVVectorShape shape = valuePlan->shape;
    std::string shapeSuffix = rvvShapeSuffix(shape);
    if (shapeSuffix.empty())
      return op.emitError("block cast result shape has no RVV spelling");
    const PhysicalTemporaryDecision *temporary =
        findBlockTemporaryDecision(op.getOperation());
    if (input.kind == BlockValueKind::Index && target.isUnsignedInteger(8)) {
      if (shape.sew != 8 || input.spelling.empty())
        return op.emitError("logical index to u8 cast has no selected vectors");
      kind = BlockValueKind::U8;
      line("vuint" + shapeSuffix + "_t " + name +
           " = __riscv_vncvt_x_x_w_u" + shapeSuffix + "(" + input.spelling +
           ", " + vl.str() + ");");
    } else if (input.kind == BlockValueKind::U8 &&
               target.isUnsignedInteger(16)) {
      if (shape.sew != 16 || input.spelling.empty())
        return op.emitError("u8 to u16 cast has no selected vectors");
      kind = BlockValueKind::U16;
      line("vuint" + shapeSuffix + "_t " + name +
           " = __riscv_vzext_vf2_u" + shapeSuffix + "(" + input.spelling +
           ", " + vl.str() + ");");
    } else if (input.kind == BlockValueKind::U8 &&
               target.isSignedInteger(32)) {
      if (physical->realization != BlockValueRealization::DeferredI32Widen ||
          shape.sew != 32 || !temporary || temporary->shape != shape)
        return op.emitError("deferred u8 to i32 cast decision is incomplete");
      kind = BlockValueKind::I32;
      name.clear();
    } else if (input.kind == BlockValueKind::I8 &&
               target.isSignedInteger(32)) {
      if (physical->realization != BlockValueRealization::DeferredI32Widen ||
          shape.sew != 32 || !temporary || temporary->shape != shape)
        return op.emitError("deferred i8 to i32 cast decision is incomplete");
      kind = BlockValueKind::I32;
      name.clear();
    } else if (input.kind == BlockValueKind::U16 &&
               target.isSignedInteger(32)) {
      if (shape.sew != 32 || input.spelling.empty() || !temporary ||
          temporary->shape != shape)
        return op.emitError("u16 to i32 cast decision is incomplete");
      kind = BlockValueKind::I32;
      std::string extended = fresh("block_u32");
      line("vuint" + shapeSuffix + "_t " + extended +
           " = __riscv_vzext_vf2_u" + shapeSuffix + "(" + input.spelling +
           ", " + vl.str() + ");");
      line("vint" + shapeSuffix + "_t " + name +
           " = __riscv_vreinterpret_v_u" + shapeSuffix + "_i" + shapeSuffix +
           "(" + extended + ");");
    } else if (input.kind == BlockValueKind::I16 &&
               target.isSignedInteger(32)) {
      if (shape.sew != 32 || input.spelling.empty())
        return op.emitError("i16 to i32 cast decision is incomplete");
      kind = BlockValueKind::I32;
      line("vint" + shapeSuffix + "_t " + name +
           " = __riscv_vsext_vf2_i" + shapeSuffix + "(" + input.spelling +
           ", " + vl.str() + ");");
    } else if (input.kind == BlockValueKind::U8 && target.isF32()) {
      if (shape.sew != 32 || input.spelling.empty() || !temporary)
        return op.emitError("u8 to f32 cast decision is incomplete");
      kind = BlockValueKind::F32;
      std::string extended;
      if (temporary->shape == shape) {
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
          line("vuint" + shapeSuffix + "_t " + widened +
               " = __riscv_vzext_vf4_u" + shapeSuffix + "(" +
               sourceSpelling + ", " + vl.str() + ");");
          if (source != blockValues.end())
            source->second.widenedSpelling = widened;
        }
        extended = widened;
        if (!input.packedTransform.empty()) {
          extended = fresh("block_code_u32");
          std::string stem =
              input.packedTransform == "and" ? "vand" : "vsrl";
          line("vuint" + shapeSuffix + "_t " + extended + " = __riscv_" +
               stem + "_vx_u" + shapeSuffix + "(" + widened + ", " +
               input.packedTransformOperand + ", " + vl.str() + ");");
        }
        line("vfloat" + shapeSuffix + "_t " + name +
             " = __riscv_vfcvt_f_xu_v_f" + shapeSuffix + "(" + extended +
             ", " + vl.str() + ");");
      } else {
        RVVVectorShape intermediateShape = temporary->shape;
        std::string intermediateSuffix = rvvShapeSuffix(intermediateShape);
        if (intermediateShape.sew != 16 || intermediateSuffix.empty())
          return op.emitError("u8 to f32 intermediate has no RVV spelling");
        extended = fresh("block_u16");
        line("vuint" + intermediateSuffix + "_t " + extended +
             " = __riscv_vzext_vf2_u" + intermediateSuffix + "(" +
             input.spelling + ", " + vl.str() + ");");
        line("vfloat" + shapeSuffix + "_t " + name +
             " = __riscv_vfwcvt_f_xu_v_f" + shapeSuffix + "(" + extended +
             ", " + vl.str() + ");");
      }
    } else if (input.kind == BlockValueKind::I8 && target.isF32()) {
      if (shape.sew != 32 || input.spelling.empty() || !temporary)
        return op.emitError("i8 to f32 cast decision is incomplete");
      RVVVectorShape intermediateShape = temporary->shape;
      std::string intermediateSuffix = rvvShapeSuffix(intermediateShape);
      if (intermediateShape.sew != 16 || intermediateSuffix.empty())
        return op.emitError("i8 to f32 intermediate has no RVV spelling");
      kind = BlockValueKind::F32;
      std::string extended = fresh("block_i16");
      line("vint" + intermediateSuffix + "_t " + extended +
           " = __riscv_vsext_vf2_i" + intermediateSuffix + "(" +
           input.spelling + ", " + vl.str() + ");");
      line("vfloat" + shapeSuffix + "_t " + name +
           " = __riscv_vfwcvt_f_x_v_f" + shapeSuffix + "(" + extended +
           ", " + vl.str() + ");");
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
    const BlockOperationDecision *physical =
        findBlockOperationDecision(op.getOperation());
    const PhysicalValueDecision *valuePlan =
        findBlockValueDecision(op.getResult());
    if (!physical || !valuePlan || !valuePlan->shape)
      return op.emitError("block bitcast has no physical value decision");
    BlockValue input = lookupBlockValue(op.getInput(), blockValues);
    mlir::Type target = elementType(op.getResult().getType());
    std::string name = fresh("block_bits");
    BlockValueKind kind;
    RVVVectorShape shape = valuePlan->shape;
    std::string shapeSuffix = rvvShapeSuffix(shape);
    if (shapeSuffix.empty() || input.vectorShape != shape)
      return op.emitError("block bitcast operands do not match its decision");
    if (input.kind == BlockValueKind::U8 && target.isSignedInteger(8)) {
      if (shape.sew != 8)
        return op.emitError("u8 block bitcast has no physical RVV shape");
      kind = BlockValueKind::I8;
      line("vint" + shapeSuffix + "_t " + name +
           " = __riscv_vreinterpret_v_u" + shapeSuffix + "_i" + shapeSuffix +
           "(" + input.spelling + ");");
    } else if (input.kind == BlockValueKind::U16 &&
               target.isSignedInteger(16)) {
      if (shape.sew != 16)
        return op.emitError("u16 block bitcast has no physical RVV shape");
      kind = BlockValueKind::I16;
      line("vint" + shapeSuffix + "_t " + name +
           " = __riscv_vreinterpret_v_u" + shapeSuffix + "_i" + shapeSuffix +
           "(" + input.spelling + ");");
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
    const BlockOperationDecision *physical =
        findBlockOperationDecision(op.getOperation());
    const PhysicalValueDecision *valuePlan =
        findBlockValueDecision(op.getResult());
    if (!physical || !valuePlan || !valuePlan->shape)
      return op.emitError("block load has no physical memory handoff decision");
    BlockValue pointer = lookupBlockValue(op.getPointer(), blockValues);
    BlockValue where = lookupBlockValue(op.getWhere(), blockValues);
    BlockValue other = lookupBlockValue(op.getOther(), blockValues);
    if (pointer.kind != BlockValueKind::Pointer || pointer.pointerBase.empty() ||
        (pointer.pointerIndex.empty() && pointer.contiguousIndex.empty()))
      return op.emitError("RVV block load has no vector address");
    if (!elementType(op.getResult().getType()).isUnsignedInteger(8))
      return op.emitError("RVV block load currently supports u8 elements");
    std::string name = fresh("block_load");
    RVVVectorShape selectedShape = valuePlan->shape;
    std::string shapeSuffix = rvvShapeSuffix(selectedShape);
    if (shapeSuffix.empty())
      return op.emitError("block load has no RVV shape spelling");
    std::string suffix = "u" + shapeSuffix;
    std::string cType = "vuint" + shapeSuffix + "_t";
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
    result.vectorShape = selectedShape;
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
    if (options.target.vlenBits < 128 ||
        !options.target.supportsVectorShape(kRVVE8M1.sew,
                                            kRVVE8M1.lmulEighths))
      return op.emitError(
          "RVV block decode requires a legal e8m1 vector with 16 lanes");
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
        decision.tableExtent != 16 || codes.kind != BlockValueKind::U8 ||
        table.kind != BlockValueKind::I8 ||
        codes.spelling.empty() || table.spelling.empty())
      return op.emitError(
          "RVV block decode operands do not match the selected realization");
    auto planned = physicalPlan.blockDecodes.find(op.getOperation());
    if (planned == physicalPlan.blockDecodes.end() ||
        mlir::failed(requireEntityPlan(op.getOperation(), planned->second)))
      return mlir::failure();
    const PhysicalHandoffDecision *handoff =
        findPhysicalHandoff(planned->second.entity, op.getOperation(),
                            op.getCodes());
    auto codeShape = llvm::find_if(
        planned->second.entity.values,
        [&](const PhysicalValueDecision &value) {
          return value.value == op.getCodes();
        });
    auto tableShape = llvm::find_if(
        planned->second.entity.values,
        [&](const PhysicalValueDecision &value) {
          return value.value == op.getTable();
        });
    auto resultShape = llvm::find_if(
        planned->second.entity.values,
        [&](const PhysicalValueDecision &value) {
          return value.value == op.getResult();
        });
    if (!handoff || handoff->kind != PhysicalHandoff::Share ||
        codeShape == planned->second.entity.values.end() ||
        tableShape == planned->second.entity.values.end() ||
        resultShape == planned->second.entity.values.end() ||
        codeShape->shape != handoff->sourceShape ||
        resultShape->shape != handoff->resultShape ||
        codes.vectorShape != codeShape->shape ||
        table.vectorShape != tableShape->shape)
      return op.emitError("block decode physical handoff is incomplete");
    std::string resultSuffix = rvvShapeSuffix(resultShape->shape);
    if (resultShape->shape.sew != 8 || resultSuffix.empty())
      return op.emitError("block decode result shape has no RVV spelling");
    std::string intrinsicSuffix = "i" + resultSuffix;
    std::string name = fresh("block_decode");
    line("vint" + resultSuffix + "_t " + name +
         " = __riscv_vrgather_vv_" + intrinsicSuffix + "(" + table.spelling +
         ", " + codes.spelling + ", " + vl.str() + ");");
    BlockValue result{op.getResult().getType(), BlockValueKind::I8, name};
    result.vectorShape = handoff->resultShape;
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
    std::string shapeSuffix = rvvShapeSuffix(value.vectorShape);
    if (value.vectorShape.sew != 32 || shapeSuffix.empty())
      return op.emitError("f32 block store has no physical RVV shape");
    line("__riscv_vse32_v_f" + shapeSuffix + "(" + pointer.pointerBase + " + " +
         pointer.contiguousIndex + ", " + value.spelling + ", " + vl.str() +
         ");");
    return mlir::success();
  }

  mlir::LogicalResult emitBlockSelect(
      SelectOp op, llvm::DenseMap<mlir::Value, BlockValue> &blockValues,
      llvm::StringRef vl) {
    const BlockOperationDecision *physical =
        findBlockOperationDecision(op.getOperation());
    const PhysicalValueDecision *valuePlan =
        findBlockValueDecision(op.getResult());
    if (!physical || !valuePlan || !valuePlan->shape)
      return op.emitError("block select has no physical value decision");
    RVVVectorShape selectedShape = valuePlan->shape;
    BlockValue predicate = lookupBlockValue(op.getPredicate(), blockValues);
    BlockValue trueValue = lookupBlockValue(op.getTrueValue(), blockValues);
    BlockValue falseValue = lookupBlockValue(op.getFalseValue(), blockValues);
    if (predicate.kind != BlockValueKind::Mask ||
        trueValue.kind != BlockValueKind::U8 ||
        falseValue.kind != BlockValueKind::U8 ||
        trueValue.vectorShape != selectedShape ||
        falseValue.vectorShape != selectedShape)
      return op.emitError("RVV block select currently requires mask and u8 values");
    std::string shapeSuffix = rvvShapeSuffix(selectedShape);
    if (shapeSuffix.empty())
      return op.emitError("block select has no RVV shape spelling");
    std::string suffix = "u" + shapeSuffix;
    std::string name = fresh("block_select");
    line("vuint" + shapeSuffix + "_t " + name + " = __riscv_vmerge_vvm_" +
         suffix + "(" + falseValue.spelling + ", " + trueValue.spelling +
         ", " + predicate.spelling + ", " + vl.str() + ");");
    BlockValue result{op.getResult().getType(), BlockValueKind::U8, name};
    result.vectorShape = selectedShape;
    blockValues[op.getResult()] = std::move(result);
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
    if (mlir::isa<BlockIndexOp>(operation))
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
      auto found = physicalPlan.blockDecodes.find(op.getOperation());
      if (found == physicalPlan.blockDecodes.end())
        return op.emitError("block decode has no selected physical decision");
      return emitBlockDecode(op, found->second.realization, blockValues, vl);
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
      BlockIndexOp axis,
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

  bool supportsBlockByteShape(mlir::Operation *operation,
                              RVVVectorShape byteShape) const {
    if (byteShape == kRVVE8MF4)
      return mlir::isa<BlockIndexOp, PtrAddOp, LoadOp, BinaryOp, CastOp>(
          operation);
    return mlir::isa<BlockIndexOp, PtrAddOp, LoadOp, BinaryOp, CompareOp,
                     CastOp, BitcastOp, DecodeOp, SelectOp, TupleOp,
                     TupleGetOp>(operation);
  }

  SelectedBlockStorePhysical decideBlockStorePhysical(
      int64_t extent, BlockIndexOp axis,
      const llvm::DenseSet<mlir::Operation *> &closure) const {
    bool needsLaneVector = blockClosureNeedsLaneVector(axis, closure);
    llvm::SmallVector<BlockStorePhysicalDecision> candidates;
    candidates.push_back(BlockStorePhysicalDecision{
        BlockStoreRealization::RVVE8MF4MicroStrips,
        kRVVE8MF4, 4, false, {}});
    candidates.push_back(BlockStorePhysicalDecision{
        BlockStoreRealization::RVVE8M1FixedStrips,
        kRVVE8M1, 16, false, {}});
    candidates.push_back(BlockStorePhysicalDecision{
        BlockStoreRealization::RVVE8M1DynamicStrips,
        kRVVE8M1, 16, needsLaneVector,
        needsLaneVector ? kRVVE16M2 : RVVVectorShape{}});
    for (const BlockStorePhysicalDecision &candidate : candidates) {
      bool operationsLegal = llvm::all_of(
          closure, [&](mlir::Operation *operation) {
            return supportsBlockByteShape(operation, candidate.byteShape);
          });
      if (!operationsLegal)
        continue;
      if (candidate.realization ==
              BlockStoreRealization::RVVE8MF4MicroStrips &&
          (needsLaneVector || extent != 32))
        continue;
      if (candidate.realization == BlockStoreRealization::RVVE8M1FixedStrips &&
          (needsLaneVector || extent != 32))
        continue;
      unsigned dataGroups =
          candidate.byteShape == kRVVE8MF4 ? 4 : 8;
      unsigned laneGroups = rvvRegisterGroups(candidate.laneShape);
      PhysicalResourceBudget resources;
      resources.architecturalGroups = options.target.vectorRegisters;
      resources.valueGroups = dataGroups;
      resources.memoryGroups = laneGroups;
      resources.primitiveGroups = dataGroups + laneGroups;
      resources.peakGroups = resources.primitiveGroups + 1;
      if (resources.peakGroups >=
          static_cast<unsigned>(options.target.vectorRegisters))
        continue;
      return SelectedBlockStorePhysical{candidate, resources};
    }
    return SelectedBlockStorePhysical{};
  }

  SelectedBlockReducePhysical decideBlockReducePhysical(
      int64_t extent, BlockIndexOp axis,
      const llvm::DenseSet<mlir::Operation *> &closure) const {
    bool needsLaneVector = blockClosureNeedsLaneVector(axis, closure);
    if (!llvm::all_of(closure, [&](mlir::Operation *operation) {
          return supportsBlockByteShape(operation, kRVVE8M1);
        }))
      return SelectedBlockReducePhysical{};
    llvm::SmallVector<BlockReducePhysicalDecision> candidates{
        {BlockReduceRealization::RVVE8M1FixedStrips, 16, false, {}},
        {BlockReduceRealization::RVVE8M1DynamicStrips, 16,
         needsLaneVector, needsLaneVector ? kRVVE16M2 : RVVVectorShape{}}};
    for (const BlockReducePhysicalDecision &candidate : candidates) {
      if (candidate.realization == BlockReduceRealization::RVVE8M1FixedStrips &&
          (needsLaneVector || extent != 32))
        continue;
      unsigned liveGroups = 8 + rvvRegisterGroups(candidate.laneShape) + 1;
      PhysicalResourceBudget resources;
      resources.architecturalGroups = options.target.vectorRegisters;
      resources.valueGroups = 8;
      resources.memoryGroups = rvvRegisterGroups(candidate.laneShape);
      resources.primitiveGroups = liveGroups;
      resources.peakGroups = liveGroups + 1;
      if (resources.peakGroups >=
          static_cast<unsigned>(options.target.vectorRegisters))
        continue;
      return SelectedBlockReducePhysical{candidate, resources};
    }
    return SelectedBlockReducePhysical{};
  }

  std::optional<int64_t> f32BlockExtent(mlir::Type type) const {
    type = unwrapLogicalValidity(type);
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
      if (physicalPlan.affineI4I8.contains(parent) ||
          physicalPlan.f16Matmuls.contains(parent))
        return true;
    return false;
  }

  bool isMaterializedF32BlockStore(StoreOp store) const {
    return isMaterializedF32Value(store.getValue());
  }

  mlir::LogicalResult prepareBlockStoreGroup(
      StoreOp op, llvm::DenseSet<mlir::Operation *> &plannedStores) {
    std::optional<int64_t> extent = f32BlockExtent(op.getValue().getType());
    if (!extent || *extent <= 0 || *extent > 65535 || !isTrue(op.getWhere()))
      return op.emitError(
          "RVV block store requires an all-active rank-one f32 value");
    llvm::DenseSet<mlir::Operation *> closure;
    llvm::SmallVector<BlockIndexOp> axes;
    if (mlir::failed(collectBlockClosure(op.getPointer(), closure, axes,
                                         op.getOperation(), true)) ||
        mlir::failed(collectBlockClosure(op.getValue(), closure, axes,
                                         op.getOperation(), true)))
      return mlir::failure();
    if (axes.size() != 1)
      return op.emitError(
          "RVV block store requires exactly one local logical block axis");
    BlockIndexOp axis = axes.front();
    auto axisType = axis.getResult().getType();
    if (axisType.getShape().size() != 1 ||
        axisType.getShape().front() != *extent)
      return op.emitError("block store extent does not match its logical axis");

    llvm::SmallVector<StoreOp> stores{op};
    llvm::SmallVector<mlir::Operation *> scalarPreamble;
    llvm::DenseSet<mlir::Operation *> scalarPreambleSet;
    for (mlir::Operation *candidateOperation = op->getNextNode();
         candidateOperation;
         candidateOperation = candidateOperation->getNextNode()) {
      auto candidate = mlir::dyn_cast<StoreOp>(candidateOperation);
      if (!candidate || plannedStores.contains(candidateOperation) ||
          !hasBlockPayload(candidateOperation) ||
          isMaterializedF32BlockStore(candidate) || !isTrue(candidate.getWhere()))
        continue;
      bool interveningEffect = false;
      for (mlir::Operation *between = op->getNextNode();
           between && between != candidateOperation;
           between = between->getNextNode())
        if (!llvm::any_of(stores, [&](StoreOp selected) {
              return selected.getOperation() == between;
            }) &&
            !mlir::isMemoryEffectFree(between)) {
          interveningEffect = true;
          break;
        }
      if (interveningEffect)
        continue;
      llvm::DenseSet<mlir::Operation *> candidateClosure;
      llvm::SmallVector<BlockIndexOp> candidateAxes;
      if (mlir::failed(collectBlockClosure(candidate.getPointer(),
                                           candidateClosure, candidateAxes,
                                           candidateOperation, true)) ||
          mlir::failed(collectBlockClosure(candidate.getValue(),
                                           candidateClosure, candidateAxes,
                                           candidateOperation, true)))
        return mlir::failure();
      if (f32BlockExtent(candidate.getValue().getType()) != extent ||
          candidateAxes.size() != 1 ||
          candidateAxes.front().getOperation() != axis.getOperation())
        continue;
      bool sharesProducer = llvm::any_of(
          candidateClosure, [&](mlir::Operation *producer) {
            return !mlir::isa<BlockIndexOp>(producer) && closure.contains(producer);
          });
      if (!sharesProducer)
        continue;
      llvm::SmallVector<mlir::Operation *> candidatePreamble;
      llvm::DenseSet<mlir::Operation *> candidatePreambleSet;
      bool scalarProjectionLegal = true;
      std::function<void(mlir::Value)> collectScalar = [&](mlir::Value value) {
        mlir::Operation *definition = value.getDefiningOp();
        if (!definition || definition->getBlock() != op->getBlock() ||
            definition == op.getOperation() ||
            definition->isBeforeInBlock(op.getOperation()))
          return;
        if (hasBlockPayload(definition) ||
            candidatePreambleSet.contains(definition))
          return;
        if (!mlir::isa<ConstantOp, MetaValueOp, SpecialValueOp, BinaryOp,
                       CastOp, PtrAddOp>(definition)) {
          scalarProjectionLegal = false;
          return;
        }
        for (mlir::Value operand : definition->getOperands())
          collectScalar(operand);
        if (scalarProjectionLegal && candidatePreambleSet.insert(definition).second)
          candidatePreamble.push_back(definition);
      };
      for (mlir::Operation *producer : candidateClosure)
        for (mlir::Value operand : producer->getOperands())
          if (!containsBlockType(operand.getType()) &&
              !isRegionValue(operand.getType()))
            collectScalar(operand);
      if (!scalarProjectionLegal)
        continue;
      closure.insert(candidateClosure.begin(), candidateClosure.end());
      for (mlir::Operation *preamble : candidatePreamble)
        if (scalarPreambleSet.insert(preamble).second)
          scalarPreamble.push_back(preamble);
      stores.push_back(candidate);
    }

    BlockStoreGroupDecision decision;
    decision.operation = op.getOperation();
    decision.axis = axis.getOperation();
    decision.extent = *extent;
    decision.scalarPreamble = scalarPreamble;
    SelectedBlockStorePhysical selected =
        decideBlockStorePhysical(*extent, axis, closure);
    if (selected.decision.realization == BlockStoreRealization::Unsupported)
      return op.emitError(
          "RVV block store has no legal physical resource candidate");
    for (StoreOp store : stores) {
      decision.stores.push_back(store.getOperation());
      plannedStores.insert(store.getOperation());
    }
    decision.closure.append(closure.begin(), closure.end());
    PlannedPhysicalDecision<BlockStoreGroupDecision> planned(
        std::move(decision));
    initializeEntityPlan(planned.entity);
    planned.realization.operations =
        decideBlockOperations(closure, selected.decision.byteShape,
                              planned.entity);
    for (StoreOp store : stores) {
      auto storedValue = llvm::find_if(
          planned.entity.values, [&](const PhysicalValueDecision &value) {
            return value.value == store.getValue();
          });
      if (storedValue != planned.entity.values.end())
        recordPhysicalHandoff(planned.entity, store.getOperation(),
                              store.getValue(), PhysicalHandoff::Rematerialize,
                              storedValue->shape, storedValue->shape);
    }
    planned.entity.blockStore = selected.decision;
    planned.entity.resources = selected.resources;
    physicalPlan.blockStoreGroups.try_emplace(op.getOperation(),
                                               std::move(planned));
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
    llvm::SmallVector<BlockIndexOp> axes;
    if (mlir::failed(collectBlockClosure(op.getInput(), closure, axes,
                                         op.getOperation(), false)))
      return mlir::failure();
    if (axes.size() != 1)
      return op.emitError(
          "RVV block reduction requires exactly one local logical block axis");
    BlockIndexOp axis = axes.front();
    auto axisType = axis.getResult().getType();
    if (axisType.getShape().size() != 1 ||
        axisType.getShape().front() != extent)
      return op.emitError("block reduction extent does not match its logical axis");

    llvm::SmallVector<ReduceOp> reductions{op};

    BlockReduceGroupDecision decision;
    decision.operation = op.getOperation();
    decision.axis = axis.getOperation();
    decision.extent = extent;
    SelectedBlockReducePhysical selected =
        decideBlockReducePhysical(extent, axis, closure);
    if (selected.decision.realization == BlockReduceRealization::Unsupported)
      return op.emitError(
          "RVV block reduction has no legal physical resource candidate");
    for (ReduceOp reduction : reductions) {
      decision.reductions.push_back(reduction.getOperation());
      plannedReductions.insert(reduction.getOperation());
    }
    decision.closure.append(closure.begin(), closure.end());
    PlannedPhysicalDecision<BlockReduceGroupDecision> planned(
        std::move(decision));
    initializeEntityPlan(planned.entity);
    planned.realization.operations =
        decideBlockOperations(closure, kRVVE8M1, planned.entity);
    auto reducedValue = llvm::find_if(
        planned.entity.values, [&](const PhysicalValueDecision &value) {
          return value.value == op.getInput();
        });
    if (reducedValue == planned.entity.values.end())
      return op.emitError("block reduction input has no physical value decision");
    BlockReductionValueDecision reductionValue;
    reductionValue.operation = op.getOperation();
    reductionValue.inputShape = reducedValue->shape;
    reductionValue.combinedShape = reducedValue->shape;
    if (selected.decision.realization ==
            BlockReduceRealization::RVVE8M1FixedStrips &&
        reducedValue->shape.sew == 16) {
      reductionValue.combinedShape =
          rvvShapeForSameLanes(reducedValue->shape, 32, options.target)
              .value_or(RVVVectorShape{});
      reductionValue.wideningCombine = true;
    }
    reductionValue.seedShape = RVVVectorShape{
        reductionValue.combinedShape.sew, 8};
    if (!reductionValue.inputShape || !reductionValue.combinedShape ||
        !options.target.supportsVectorShape(
            reductionValue.seedShape.sew,
            reductionValue.seedShape.lmulEighths))
      return op.emitError(
          "block reduction has no legal input/combine/seed vector shapes");
    recordPhysicalHandoff(
        planned.entity, op.getOperation(), op.getInput(),
        reductionValue.wideningCombine ? PhysicalHandoff::Convert
                                       : PhysicalHandoff::Rematerialize,
        reductionValue.inputShape, reductionValue.combinedShape);
    recordPhysicalTemporary(planned.entity, op.getOperation(),
                            reductionValue.seedShape);
    planned.realization.values.push_back(reductionValue);
    planned.entity.blockReduce = selected.decision;
    planned.entity.resources = selected.resources;
    physicalPlan.blockReduceGroups.try_emplace(op.getOperation(),
                                                std::move(planned));
    return mlir::success();
  }

  mlir::LogicalResult prepareBlockGroupDecisions() {
    bool failed = false;
    llvm::DenseSet<mlir::Operation *> plannedStores;
    llvm::DenseSet<mlir::Operation *> plannedReductions;
    kernel.walk([&](StoreOp store) {
      if (failed || plannedStores.contains(store.getOperation()) ||
          !hasBlockPayload(store.getOperation()) ||
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
    auto selected = physicalPlan.blockStoreGroups.find(op.getOperation());
    if (selected == physicalPlan.blockStoreGroups.end())
      return op.emitError("block store has no selected physical group decision");
    if (mlir::failed(requireEntityPlan(op.getOperation(), selected->second)))
      return mlir::failure();
    const BlockStoreGroupDecision &decision = selected->second.realization;
    const PhysicalEntityPlan &entity = selected->second.entity;
    if (!entity.blockStore)
      return op.emitError("block store physical entity is incomplete");
    for (mlir::Operation *storeOperation : decision.stores) {
      auto store = mlir::cast<StoreOp>(storeOperation);
      const PhysicalHandoffDecision *storeHandoff =
          findPhysicalHandoff(entity, storeOperation, store.getValue());
      if (!storeHandoff ||
          storeHandoff->kind != PhysicalHandoff::Rematerialize ||
          storeHandoff->sourceShape != storeHandoff->resultShape)
        return store.emitError("block store physical handoff is incomplete");
    }
    const BlockStorePhysicalDecision &physical = *entity.blockStore;
    int64_t extent = decision.extent;
    BlockIndexOp axis = mlir::cast<BlockIndexOp>(decision.axis);
    llvm::DenseSet<mlir::Operation *> closure;
    closure.insert(decision.closure.begin(), decision.closure.end());

    for (mlir::Operation *preamble : decision.scalarPreamble)
      if (mlir::failed(emitOperation(preamble)))
        return mlir::failure();

    std::string offset = expression(axis.getOffset());
    if (offset.empty())
      return op.emitError("block store axis offset is unavailable");
    std::string strip = fresh("block_i");
    std::string vl = fresh("block_vl");
    std::string lane = fresh("block_lane");
    auto emitStrip = [&](llvm::StringRef stripOffset,
                         llvm::StringRef activeVL) -> mlir::LogicalResult {
      llvm::DenseMap<mlir::Value, BlockValue> blockValues;
      BlockValue coordinate{axis.getResult().getType(), BlockValueKind::Index,
                            physical.needsLaneVector ? lane : ""};
      if (physical.needsLaneVector)
        coordinate.vectorShape = physical.laneShape;
      coordinate.contiguousIndex =
          "(" + stripOffset.str() + " + " + offset + ")";
      blockValues[axis.getResult()] = std::move(coordinate);
      const llvm::SmallVector<BlockOperationDecision> *previousOperations =
          activeBlockOperations;
      const PhysicalEntityPlan *previousEntity = activeBlockEntity;
      activeBlockOperations = &decision.operations;
      activeBlockEntity = &entity;
      for (const BlockOperationDecision &operation : decision.operations) {
        if (mlir::isa<BlockIndexOp>(operation.operation))
          continue;
        if (mlir::failed(
                emitBlockOperation(operation.operation, blockValues, activeVL))) {
          activeBlockOperations = previousOperations;
          activeBlockEntity = previousEntity;
          return mlir::failure();
        }
      }
      for (mlir::Operation *store : decision.stores)
        if (mlir::failed(emitBlockOperation(store, blockValues, activeVL))) {
          activeBlockOperations = previousOperations;
          activeBlockEntity = previousEntity;
          return mlir::failure();
        }
      activeBlockOperations = previousOperations;
      activeBlockEntity = previousEntity;
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
      const llvm::SmallVector<BlockOperationDecision> *previousOperations =
          activeBlockOperations;
      const PhysicalEntityPlan *previousEntity = activeBlockEntity;
      activeBlockOperations = &decision.operations;
      activeBlockEntity = &entity;
      for (const BlockOperationDecision &operation : decision.operations) {
        if (mlir::isa<BlockIndexOp>(operation.operation))
          continue;
        for (auto &blockValues : stripValues)
          if (mlir::failed(
                  emitBlockOperation(operation.operation, blockValues, vl))) {
            activeBlockOperations = previousOperations;
            activeBlockEntity = previousEntity;
            return mlir::failure();
          }
      }
      for (mlir::Operation *store : decision.stores)
        for (auto &blockValues : stripValues)
          if (mlir::failed(emitBlockOperation(store, blockValues, vl))) {
            activeBlockOperations = previousOperations;
            activeBlockEntity = previousEntity;
            return mlir::failure();
          }
      activeBlockOperations = previousOperations;
      activeBlockEntity = previousEntity;
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
        std::string laneShape = rvvShapeSuffix(physical.laneShape);
        if (physical.laneShape.sew != 16 || laneShape.empty())
          return op.emitError("block lane vector has no selected physical shape");
        std::string laneSuffix = "u" + laneShape;
        line("vuint" + laneShape + "_t " + lane + " = __riscv_vid_v_" +
             laneSuffix + "(" + vl + ");");
        line(lane + " = __riscv_vadd_vx_" + laneSuffix + "(" + lane + ", " + strip +
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
    consumed.insert(decision.scalarPreamble.begin(),
                    decision.scalarPreamble.end());
    consumed.insert(decision.stores.begin(), decision.stores.end());
    return mlir::success();
  }

  mlir::LogicalResult emitBlockReduce(ReduceOp op) {
    auto selected = physicalPlan.blockReduceGroups.find(op.getOperation());
    if (selected == physicalPlan.blockReduceGroups.end())
      return op.emitError(
          "block reduction has no selected physical group decision");
    if (mlir::failed(requireEntityPlan(op.getOperation(), selected->second)))
      return mlir::failure();
    const BlockReduceGroupDecision &decision = selected->second.realization;
    const PhysicalEntityPlan &entity = selected->second.entity;
    if (!entity.blockReduce)
      return op.emitError("block reduction physical entity is incomplete");
    const PhysicalHandoffDecision *reduceHandoff =
        findPhysicalHandoff(entity, op.getOperation(), op.getInput());
    auto selectedReduction = llvm::find_if(
        decision.values, [&](const BlockReductionValueDecision &value) {
          return value.operation == op.getOperation();
        });
    auto seedTemporary = llvm::find_if(
        entity.temporaries, [&](const PhysicalTemporaryDecision &temporary) {
          return temporary.owner == op.getOperation();
        });
    if (selectedReduction == decision.values.end() || !reduceHandoff ||
        seedTemporary == entity.temporaries.end() ||
        seedTemporary->shape != selectedReduction->seedShape ||
        reduceHandoff->kind !=
            (selectedReduction->wideningCombine ? PhysicalHandoff::Convert
                                                : PhysicalHandoff::Rematerialize) ||
        reduceHandoff->sourceShape != selectedReduction->inputShape ||
        reduceHandoff->resultShape != selectedReduction->combinedShape)
      return op.emitError("block reduction physical handoff is incomplete");
    const BlockReducePhysicalDecision &physical = *entity.blockReduce;
    int64_t extent = decision.extent;
    BlockIndexOp axis = mlir::cast<BlockIndexOp>(decision.axis);
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
    const llvm::SmallVector<BlockOperationDecision> *previousOperations =
        activeBlockOperations;
    const PhysicalEntityPlan *previousEntity = activeBlockEntity;
    activeBlockOperations = &decision.operations;
    activeBlockEntity = &entity;
    auto restoreOperations = llvm::make_scope_exit(
        [&]() {
          activeBlockOperations = previousOperations;
          activeBlockEntity = previousEntity;
        });

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
              mlir::isa<BlockIndexOp>(candidate))
            continue;
          if (mlir::failed(emitBlockOperation(&candidate, blockValues, vl)))
            return mlir::failure();
        }
      }
      for (auto [index, reduction] : llvm::enumerate(reductions)) {
        auto reductionValue = llvm::find_if(
            decision.values, [&](const BlockReductionValueDecision &value) {
              return value.operation == reduction.getOperation();
            });
        if (stripInputs[index].size() != 2 ||
            stripInputs[index][0].kind != stripInputs[index][1].kind ||
            reductionValue == decision.values.end() ||
            stripInputs[index][0].vectorShape != reductionValue->inputShape ||
            stripInputs[index][1].vectorShape != reductionValue->inputShape)
          return reduction.emitError(
              "fixed-strip reduction produced incompatible physical values");
        std::string inputSuffix = rvvShapeSuffix(reductionValue->inputShape);
        std::string combinedSuffix =
            rvvShapeSuffix(reductionValue->combinedShape);
        std::string seedSuffix = rvvShapeSuffix(reductionValue->seedShape);
        if (inputSuffix.empty() || combinedSuffix.empty() || seedSuffix.empty())
          return reduction.emitError(
              "fixed-strip reduction shapes have no RVV spelling");
        std::string combined = fresh("block_combined");
        std::string seed = fresh("block_seed");
        std::string partial = fresh("block_partial");
        BlockValueKind expectedKind = reductionValue->wideningCombine
                                          ? BlockValueKind::I16
                                          : BlockValueKind::I32;
        if (stripInputs[index][0].kind != expectedKind ||
            reductionValue->combinedShape.sew != 32 ||
            reductionValue->seedShape.sew != 32)
          return reduction.emitError(
              "fixed-strip reduction physical type is unsupported");
        std::string addStem =
            reductionValue->wideningCombine ? "vwadd" : "vadd";
        line("vint" + combinedSuffix + "_t " + combined + " = __riscv_" +
             addStem + "_vv_i" + combinedSuffix + "(" +
             stripInputs[index][0].spelling + ", " +
             stripInputs[index][1].spelling + ", " + vl + ");");
        line("vint" + seedSuffix + "_t " + seed + " = __riscv_vmv_v_x_i" +
             seedSuffix + "(0, 1);");
        line("vint" + seedSuffix + "_t " + partial +
             " = __riscv_vredsum_vs_i" + combinedSuffix + "_i" + seedSuffix +
             "(" + combined + ", " + seed + ", " + vl + ");");
        line(accumulators[index] + " += __riscv_vmv_x_s_i" + seedSuffix +
             "_i32(" + partial + ");");
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
      std::string laneShape = rvvShapeSuffix(physical.laneShape);
      if (physical.laneShape.sew != 16 || laneShape.empty())
        return op.emitError("block lane vector has no selected physical shape");
      std::string laneSuffix = "u" + laneShape;
      line("vuint" + laneShape + "_t " + lane + " = __riscv_vid_v_" +
           laneSuffix + "(" + vl + ");");
      line(lane + " = __riscv_vadd_vx_" + laneSuffix + "(" + lane + ", " +
           strip + " + " + offset + ", " + vl + ");");
    }

    llvm::DenseMap<mlir::Value, BlockValue> blockValues;
    BlockValue coordinate{axis.getResult().getType(), BlockValueKind::Index,
                          physical.needsLaneVector ? lane : ""};
    if (physical.needsLaneVector)
      coordinate.vectorShape = physical.laneShape;
    coordinate.contiguousIndex = "(" + strip + " + " + offset + ")";
    blockValues[axis.getResult()] = std::move(coordinate);
    llvm::DenseMap<mlir::Operation *, size_t> reductionIndices;
    for (auto [index, reduction] : llvm::enumerate(reductions))
      reductionIndices[reduction.getOperation()] = index;
    for (mlir::Operation &candidate : *op->getBlock()) {
      auto reduction = reductionIndices.find(&candidate);
      if (reduction != reductionIndices.end()) {
        ReduceOp current = reductions[reduction->second];
        auto reductionValue = llvm::find_if(
            decision.values, [&](const BlockReductionValueDecision &value) {
              return value.operation == current.getOperation();
            });
        BlockValue input = lookupBlockValue(current.getInput(), blockValues);
        if (input.kind == BlockValueKind::I32 &&
            mlir::failed(materializeI32(input, current.getOperation(), vl)))
          return mlir::failure();
        if (reductionValue == decision.values.end() ||
            (input.kind != BlockValueKind::I32 &&
             input.kind != BlockValueKind::I16) ||
            input.spelling.empty() ||
            input.vectorShape != reductionValue->inputShape ||
            reductionValue->combinedShape != reductionValue->inputShape)
          return current.emitError(
              "RVV block reduction input is not an integer vector");
        std::string inputSuffix = rvvShapeSuffix(reductionValue->inputShape);
        std::string seedSuffix = rvvShapeSuffix(reductionValue->seedShape);
        if (inputSuffix.empty() || seedSuffix.empty() ||
            reductionValue->seedShape.sew != reductionValue->inputShape.sew)
          return current.emitError(
              "RVV block reduction shapes have no compatible spelling");
        std::string seed = fresh("block_seed");
        std::string partial = fresh("block_partial");
        const std::string &accumulator = accumulators[reduction->second];
        line("vint" + seedSuffix + "_t " + seed + " = __riscv_vmv_v_x_i" +
             seedSuffix + "(0, 1);");
        line("vint" + seedSuffix + "_t " + partial +
             " = __riscv_vredsum_vs_i" + inputSuffix + "_i" + seedSuffix +
             "(" + input.spelling + ", " + seed + ", " + vl + ");");
        std::string extracted = "__riscv_vmv_x_s_i" + seedSuffix + "_i" +
                                std::to_string(reductionValue->seedShape.sew) +
                                "(" + partial + ")";
        line(accumulator + " += " +
             (reductionValue->seedShape.sew == 16
                  ? "(int32_t)" + extracted
                  : extracted) +
             ";");
        if (&candidate == reductions.back().getOperation())
          break;
        continue;
      }
      if (&candidate == reductions.back().getOperation())
        break;
      if (!closure.contains(&candidate) ||
          mlir::isa<BlockIndexOp>(candidate))
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

  mlir::FailureOr<std::string>
  materializeVLAStateInput(mlir::Operation *owner, mlir::Value semanticInput,
                           const CValue &input,
                           const VLAStateDecision &decision) {
    if (decision.validity == VLAStateValidityRealization::AllActive) {
      if (!input.logicalValidity.empty()) {
        owner->emitError(
            "masked VLA state input has no selected validity realization");
        return mlir::failure();
      }
      return input.spelling;
    }
    if (input.logicalValidity.empty()) {
      owner->emitError(
          "selected VLA state validity requires a logical predicate mask");
      return mlir::failure();
    }
    std::optional<unsigned> dataLMUL = physicalValueLMUL(semanticInput);
    if (!dataLMUL) {
      owner->emitError("VLA state validity has no physical input shape");
      return mlir::failure();
    }
    std::string fill;
    if (decision.validity == VLAStateValidityRealization::ReductionIdentity)
      fill = expression(decision.identity);
    else if (decision.validity ==
                 VLAStateValidityRealization::NegativeInfinity ||
             decision.validity == VLAStateValidityRealization::OnlineSoftmax)
      fill = "-INFINITY";
    if (fill.empty()) {
      owner->emitError("VLA state validity has no selected inactive value");
      return mlir::failure();
    }
    std::string suffix = "f32m" + std::to_string(*dataLMUL);
    std::string inactive = fresh("state_inactive");
    std::string projected = fresh("state_input");
    line("vfloat32m" + std::to_string(*dataLMUL) + "_t " + inactive +
         " = __riscv_vfmv_v_f_" + suffix + "(" + fill + ", " + activeVL +
         ");");
    line("vfloat32m" + std::to_string(*dataLMUL) + "_t " + projected +
         " = __riscv_vmerge_vvm_" + suffix + "(" + inactive + ", " +
         input.spelling + ", " + input.logicalValidity + ", " + activeVL +
         ");");
    return projected;
  }

  mlir::LogicalResult emitVectorReduce(ReduceOp op,
                                       const VLAStateDecision &decision,
                                       const CValue &aggregate) {
    CValue input = require(op.getInput());
    if (decision.realization ==
        VLAStateRealization::RVVI8AddReductionI32) {
      if (input.kind != CValueKind::I8Vector ||
          aggregate.kind != CValueKind::Scalar ||
          aggregate.type != op.getResult().getType() ||
          aggregate.spelling.empty() || !input.logicalValidity.empty())
        return op.emitError("RVV i8-to-i32 reduction projection is unavailable");
      const PhysicalValueDecision *inputShape =
          findVLAValueDecision(op.getInput());
      std::string shapeSuffix =
          inputShape ? rvvShapeSuffix(inputShape->shape) : std::string{};
      if (!inputShape || inputShape->shape.sew != 8 || shapeSuffix.empty())
        return op.emitError("RVV i8 reduction has no physical input shape");
      std::string seed = fresh("i8_reduce_seed");
      std::string partial = fresh("i8_reduce_partial");
      std::string suffix = "i" + shapeSuffix;
      line("vint16m1_t " + seed +
           " = __riscv_vmv_v_x_i16m1(0, 1);");
      line("vint16m1_t " + partial + " = __riscv_vwredsum_vs_" + suffix +
           "_i16m1(" + input.spelling + ", " + seed + ", " + activeVL +
           ");");
      line(aggregate.spelling + " += (int32_t)__riscv_vmv_x_s_i16m1_i16(" +
           partial + ");");
      values[op.getResult()] = aggregate;
      return mlir::success();
    }
    if (input.kind != CValueKind::F32Vector || aggregate.spelling.empty())
      return op.emitError("RVV reduction projection is unavailable");
    mlir::FailureOr<std::string> projected = materializeVLAStateInput(
        op.getOperation(), op.getInput(), input, decision);
    if (mlir::failed(projected))
      return mlir::failure();
    std::optional<unsigned> dataLMUL = physicalValueLMUL(op.getInput());
    if (!dataLMUL)
      return op.emitError("RVV reduction has no physical input shape");
    if (decision.placement == VLAStatePlacement::VectorCarry) {
      if (aggregate.kind != CValueKind::F32BlockStorage)
        return op.emitError("RVV vector reduction carry is unavailable");
      std::string suffix = "f32m" + std::to_string(*dataLMUL);
      if (decision.realization == VLAStateRealization::RVVAddReduction)
        line(aggregate.spelling + " = __riscv_vfadd_vv_" + suffix + "_tu(" +
             aggregate.spelling + ", " + aggregate.spelling + ", " +
             *projected + ", " + activeVL + ");");
      else if (decision.realization == VLAStateRealization::RVVMaxReduction)
        line(aggregate.spelling + " = __riscv_vfmax_vv_" + suffix + "_tu(" +
             aggregate.spelling + ", " + aggregate.spelling + ", " +
             *projected + ", " + activeVL + ");");
      else
        return op.emitError("selected VLA state is not a reduction");
      return mlir::success();
    }
    if (aggregate.kind != CValueKind::Scalar)
      return op.emitError("RVV scalar reduction carry is unavailable");
    std::string seed = fresh("seed");
    std::string partial = fresh("partial");
    std::string inputSuffix = "f32m" + std::to_string(*dataLMUL);
    line("vfloat32m1_t " + seed + " = __riscv_vfmv_v_f_f32m1(" +
         aggregate.spelling + ", 1);");
    if (decision.realization == VLAStateRealization::RVVAddReduction)
      line("vfloat32m1_t " + partial +
           " = __riscv_vfredusum_vs_" + inputSuffix + "_f32m1(" +
           *projected + ", " + seed + ", " + activeVL + ");");
    else if (decision.realization == VLAStateRealization::RVVMaxReduction)
      line("vfloat32m1_t " + partial +
           " = __riscv_vfredmax_vs_" + inputSuffix + "_f32m1(" +
           *projected + ", " + seed + ", " + activeVL + ");");
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
    bool segmented =
        decision.realization ==
        VLAStateRealization::RVVSegmentedInclusiveAddScan;
    if ((decision.realization != VLAStateRealization::RVVInclusiveAddScan &&
         !segmented) ||
        input.kind != CValueKind::F32Vector ||
        carry.kind != CValueKind::Scalar || carry.spelling.empty())
        return op.emitError("RVV scan projection is unavailable");
    std::optional<unsigned> dataLMUL = physicalValueLMUL(op.getInput());
    if (!dataLMUL || !activePhysicalEntity ||
        activePhysicalEntity->vlaMaskRatio == 0)
      return op.emitError("RVV scan has no physical data/mask shape");
    unsigned maskRatio = activePhysicalEntity->vlaMaskRatio;
    unsigned laneIndexLMUL = *dataLMUL;

    CValue segmentMask;
    CValue segmentValues;
    std::string segmentShapeSuffix;
    if (segmented) {
      segmentMask = require(decision.segmentStart);
      segmentValues = require(decision.segmentVector);
      const PhysicalValueDecision *segmentShape =
          findVLAValueDecision(decision.segmentVector);
      segmentShapeSuffix =
          segmentShape ? rvvShapeSuffix(segmentShape->shape) : std::string{};
      if (segmentMask.kind != CValueKind::Mask ||
          segmentValues.kind != CValueKind::U8Vector ||
          segmentMask.spelling.empty() || segmentValues.spelling.empty() ||
          !segmentShape || segmentShape->shape.sew != 8 ||
          segmentShapeSuffix.empty())
        return op.emitError("segmented scan start projection is unavailable");
    }

    std::string indices = fresh("scan_indices");
    std::string prefix = fresh("scan_prefix");
    std::string offset = fresh("scan_offset");
    std::string dataSuffix = "f32m" + std::to_string(*dataLMUL);
    std::string indexSuffix =
        "u32m" + std::to_string(laneIndexLMUL);
    std::string maskSuffix =
        indexSuffix + "_b" + std::to_string(maskRatio);
    line("vuint32m" + std::to_string(laneIndexLMUL) + "_t " +
         indices + " = __riscv_vid_v_" + indexSuffix + "(" + activeVL +
         ");");
    line("vfloat32m" + std::to_string(*dataLMUL) + "_t " + prefix +
         " = " + input.spelling + ";");
    std::string propagatedSegments;
    if (segmented) {
      propagatedSegments = fresh("scan_segments");
      line("vuint" + segmentShapeSuffix + "_t " +
           propagatedSegments + " = " + segmentValues.spelling + ";");
    }
    line("for (size_t " + offset + " = 1; " + offset + " < " + activeVL +
         "; " + offset + " <<= 1) {");
    ++indent;
    std::string shifted = fresh("scan_shifted");
    std::string active = fresh("scan_active");
    line("vfloat32m" + std::to_string(*dataLMUL) + "_t " + shifted +
         " = __riscv_vslideup_vx_" + dataSuffix +
         "(__riscv_vundefined_" + dataSuffix + "(), " + prefix + ", " +
         offset + ", " + activeVL + ");");
    line("vbool" + std::to_string(maskRatio) + "_t " + active +
         " = __riscv_vmsgeu_vx_" + maskSuffix + "(" + indices +
         ", (uint32_t)" + offset + ", " + activeVL + ");");
    if (segmented) {
      std::string noSegment = fresh("scan_no_segment");
      std::string segmentSuffix = "u" + segmentShapeSuffix;
      std::string segmentMaskSuffix =
          segmentSuffix + "_b" + std::to_string(maskRatio);
      line("vbool" + std::to_string(maskRatio) + "_t " + noSegment +
           " = __riscv_vmseq_vx_" + segmentMaskSuffix + "(" +
           propagatedSegments + ", 0, " + activeVL + ");");
      line(active + " = __riscv_vmand_mm_b" +
           std::to_string(maskRatio) + "(" + active + ", " +
           noSegment + ", " + activeVL + ");");
    }
    line(prefix + " = __riscv_vfadd_vv_" + dataSuffix + "_m(" + active +
         ", " + prefix + ", " + shifted + ", " + activeVL + ");");
    if (segmented) {
      std::string shiftedSegments = fresh("scan_shifted_segments");
      std::string segmentSuffix = "u" + segmentShapeSuffix;
      line("vuint" + segmentShapeSuffix + "_t " +
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
           std::to_string(maskRatio) + "(" + segmentMask.spelling +
           ", " + activeVL + ");");
      line("const size_t " + continuationLength + " = " + firstSegment +
           " < 0 ? " + activeVL + " : (size_t)" + firstSegment + ";");
      line("vbool" + std::to_string(maskRatio) + "_t " + continuation +
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
    line("vfloat32m" + std::to_string(*dataLMUL) + "_t " + last +
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
    std::optional<unsigned> dataLMUL = physicalValueLMUL(op.getInput());
    if (!dataLMUL || !activePhysicalEntity ||
        activePhysicalEntity->vlaMaskRatio == 0)
      return op.emitError("RVV argmax has no physical data/mask shape");
    unsigned maskRatio = activePhysicalEntity->vlaMaskRatio;
    mlir::FailureOr<std::string> projected = materializeVLAStateInput(
        op.getOperation(), op.getInput(), input, decision);
    if (mlir::failed(projected))
      return mlir::failure();

    const CValue &maximum = aggregate.fields[0];
    const CValue &index = aggregate.fields[1];
    std::string seed = fresh("argmax_seed");
    std::string reduced = fresh("argmax_reduced");
    std::string stripMaximum = fresh("argmax_strip_value");
    std::string equal = fresh("argmax_equal");
    std::string first = fresh("argmax_first");
    std::string stripIndex = fresh("argmax_strip_index");
    std::string dataSuffix = "f32m" + std::to_string(*dataLMUL);
    line("vfloat32m1_t " + seed +
         " = __riscv_vfmv_v_f_f32m1(-INFINITY, 1);");
    line("vfloat32m1_t " + reduced +
         " = __riscv_vfredmax_vs_" + dataSuffix + "_f32m1(" +
         *projected + ", " + seed + ", " + activeVL + ");");
    line("const float " + stripMaximum +
         " = __riscv_vfmv_f_s_f32m1_f32(" + reduced + ");");
    line("vbool" + std::to_string(maskRatio) + "_t " + equal +
         " = __riscv_vmfeq_vf_" + dataSuffix + "_b" +
         std::to_string(maskRatio) + "(" + *projected + ", " +
         stripMaximum + ", " + activeVL + ");");
    if (!input.logicalValidity.empty())
      line(equal + " = __riscv_vmand_mm_b" + std::to_string(maskRatio) +
           "(" + equal + ", " + input.logicalValidity + ", " + activeVL +
           ");");
    line("const long " + first + " = __riscv_vfirst_m_b" +
         std::to_string(maskRatio) + "(" + equal + ", " + activeVL +
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
    std::optional<unsigned> dataLMUL = physicalValueLMUL(op.getInput());
    if (!dataLMUL)
      return op.emitError("online summary has no physical input shape");
    mlir::FailureOr<std::string> projected = materializeVLAStateInput(
        op.getOperation(), op.getInput(), input, decision);
    if (mlir::failed(projected))
      return mlir::failure();
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
    std::string dataSuffix = "f32m" + std::to_string(*dataLMUL);
    line("vfloat32m1_t " + maxSeed +
         " = __riscv_vfmv_v_f_f32m1(-INFINITY, 1);");
    line("vfloat32m1_t " + maxVector +
         " = __riscv_vfredmax_vs_" + dataSuffix + "_f32m1(" +
         *projected + ", " + maxSeed + ", " + activeVL + ");");
    line("const float " + stripMaximum +
         " = __riscv_vfmv_f_s_f32m1_f32(" + maxVector + ");");
    line("vfloat32m" + std::to_string(*dataLMUL) + "_t " + shifted +
         " = __riscv_vfsub_vf_" + dataSuffix + "(" + *projected + ", " +
         stripMaximum + ", " + activeVL + ");");
    line("vfloat32m" + std::to_string(*dataLMUL) + "_t " +
         exponentials + " = __weft_exp_f32m2(" + shifted + ", " + activeVL +
         ");");
    if (!input.logicalValidity.empty()) {
      std::string zero = fresh("summary_zero");
      std::string validExponentials = fresh("summary_valid_exp");
      line("vfloat32m" + std::to_string(*dataLMUL) + "_t " + zero +
           " = __riscv_vfmv_v_f_" + dataSuffix + "(0.0f, " + activeVL +
           ");");
      line("vfloat32m" + std::to_string(*dataLMUL) + "_t " +
           validExponentials + " = __riscv_vmerge_vvm_" + dataSuffix + "(" +
           zero + ", " + exponentials + ", " + input.logicalValidity + ", " +
           activeVL + ");");
      exponentials = std::move(validExponentials);
    }
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


} // namespace

mlir::LogicalResult weft::riscv_internal::compileRISCVKernelsToIntrinsicC(
    mlir::ModuleOp module, const RISCVLoweringOptions &options,
    llvm::raw_ostream &output,
    SelectedIntrinsicCLeaves &selectedLeaves) {
  llvm::SmallVector<weft::kernel::KernelOp> kernels;
  for (weft::kernel::KernelOp kernel :
       module.getOps<weft::kernel::KernelOp>())
    kernels.push_back(kernel);
  if (kernels.empty())
    return module.emitError("RISC-V lowering requires a Weft kernel");
  for (weft::kernel::KernelOp kernel : kernels) {
    KernelCompiler compiler(kernel, options, output, selectedLeaves);
    if (mlir::failed(compiler.compile()))
      return mlir::failure();
  }
  return mlir::success();
}
