#ifndef WEFT_LIB_TARGET_RISCVPHYSICALPLANNING_H
#define WEFT_LIB_TARGET_RISCVPHYSICALPLANNING_H

#include "Weft/Target/RISCVLowering.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <tuple>

namespace weft::riscv_internal {

enum class LaneRelation;

struct RVVVectorShape {
  unsigned sew = 0;
  int lmulEighths = 0;

  bool operator==(const RVVVectorShape &other) const {
    return sew == other.sew && lmulEighths == other.lmulEighths;
  }
  bool operator!=(const RVVVectorShape &other) const {
    return !(*this == other);
  }
  bool operator<(const RVVVectorShape &other) const {
    return sew < other.sew ||
           (sew == other.sew && lmulEighths < other.lmulEighths);
  }
  explicit operator bool() const { return sew != 0 && lmulEighths != 0; }
};

inline constexpr RVVVectorShape kRVVE8MF4{8, 2};
inline constexpr RVVVectorShape kRVVE8M1{8, 8};
inline constexpr RVVVectorShape kRVVE16M2{16, 16};
inline constexpr RVVVectorShape kRVVE32M1{32, 8};
inline constexpr RVVVectorShape kRVVE32M2{32, 16};
inline constexpr RVVVectorShape kRVVE32M4{32, 32};

std::optional<unsigned> rvvIntegerLMUL(const RVVVectorShape &shape);
unsigned rvvRegisterGroups(const RVVVectorShape &shape);
std::optional<RVVVectorShape>
rvvShapeForSemanticLanes(unsigned sew, unsigned semanticLanes,
                         const RISCVTargetProfile &target);
std::optional<RVVVectorShape>
rvvShapeForSameLanes(const RVVVectorShape &source, unsigned resultSEW,
                     const RISCVTargetProfile &target);
std::optional<unsigned> rvvMaskRatio(const RVVVectorShape &dataShape);
std::optional<unsigned>
rvvLaneCapacity(const RVVVectorShape &shape,
                const RISCVTargetProfile &target);
RVVVectorShape rvvShape(unsigned sew, unsigned lmul);
llvm::SmallVector<unsigned>
integerLMULCandidates(const RISCVTargetProfile &target, unsigned sew,
                      unsigned maximum = 8);
bool fitsPrivateStorage(int64_t elements, unsigned elementBytes,
                        const RISCVTargetProfile &target);
std::optional<int64_t>
extendPrivateStorageElementCount(int64_t currentElements, int64_t extent,
                                 unsigned elementBytes,
                                 const RISCVTargetProfile &target);

struct PhysicalResourceBudget {
  unsigned architecturalGroups = 0;
  unsigned valueGroups = 0;
  unsigned memoryGroups = 0;
  unsigned indexGroups = 0;
  unsigned predicateGroups = 0;
  unsigned stateGroups = 0;
  unsigned primitiveGroups = 0;
  unsigned peakGroups = 0;
};

enum class LoadF16LERealization {
  ScalarBytes,
  ScalarAlignedHalf,
};

struct LoadF16LEDecision {
  LoadF16LERealization realization = LoadF16LERealization::ScalarBytes;
};

std::optional<LoadF16LEDecision>
selectLoadF16LEPhysical(bool knownAligned,
                        const RISCVTargetProfile &target);

enum class BlockDecodeRealization {
  RVVI8TableGather,
};

struct BlockDecodeCandidateFacts {
  unsigned codeExtent = 0;
  unsigned tableExtent = 0;
};

struct SelectedBlockDecodePhysical {
  BlockDecodeRealization realization =
      BlockDecodeRealization::RVVI8TableGather;
  RVVVectorShape codeShape;
  RVVVectorShape tableShape;
  RVVVectorShape resultShape;
  unsigned tableExtent = 0;
  PhysicalResourceBudget resources;
};

std::optional<SelectedBlockDecodePhysical>
selectBlockDecodePhysical(const BlockDecodeCandidateFacts &facts,
                          const RISCVTargetProfile &target);

enum class BlockValueRealization {
  Direct,
  DeferredPackedTransform,
  DeferredI32Widen,
  WideningI8Product,
};

enum class BlockPhysicalType {
  None,
  Index,
  U8,
  I8,
  U16,
  I16,
  I32,
  F32,
  Mask,
};

enum class BlockOperationSemantic {
  Axis,
  Load,
  Bitcast,
  Compare,
  Cast,
  Binary,
  Select,
  Other,
};

enum class BlockBinarySemantic {
  Other,
  And,
  ShiftRight,
  Multiply,
};

struct BlockOperationCandidateFacts {
  BlockOperationSemantic operation = BlockOperationSemantic::Other;
  BlockBinarySemantic binary = BlockBinarySemantic::Other;
  BlockPhysicalType sourceType = BlockPhysicalType::None;
  BlockPhysicalType targetType = BlockPhysicalType::None;
  BlockPhysicalType resultType = BlockPhysicalType::None;
  RVVVectorShape byteShape;
  RVVVectorShape sourceShape;
  RVVVectorShape lhsShape;
  RVVVectorShape rhsShape;
  RVVVectorShape trueShape;
  RVVVectorShape falseShape;
  bool lhsNarrowU8 = false;
  bool rhsNarrowU8 = false;
  bool lhsNarrowI8 = false;
  bool rhsNarrowI8 = false;
  std::optional<uint64_t> lhsUnsignedMaximum;
  std::optional<uint64_t> rhsUnsignedMaximum;
  std::optional<uint64_t> resultUnsignedMaximum;
  std::optional<unsigned> shiftAmount;
};

struct SelectedBlockOperationPhysical {
  BlockValueRealization realization = BlockValueRealization::Direct;
  RVVVectorShape resultShape;
  RVVVectorShape temporaryShape;
  std::optional<uint64_t> unsignedMaximum;
  unsigned maskRatio = 0;
};

std::optional<SelectedBlockOperationPhysical>
selectBlockOperationPhysical(const BlockOperationCandidateFacts &facts,
                             const RISCVTargetProfile &target);

enum class BlockStoreRealization {
  RVVMicroStrips,
  RVVFixedStrips,
  RVVDynamicStrips,
};

struct BlockStorePhysicalDecision {
  BlockStoreRealization realization = BlockStoreRealization::RVVFixedStrips;
  RVVVectorShape byteShape;
  unsigned stripVL = 0;
  bool needsLaneVector = false;
  RVVVectorShape laneShape;
};

struct BlockStoreCandidateFacts {
  int64_t extent = 0;
  bool needsLaneVector = false;
  bool supportsMicroStripOperations = false;
  bool supportsStandardOperations = false;
};

struct SelectedBlockStorePhysical {
  BlockStorePhysicalDecision decision;
  PhysicalResourceBudget resources;
};

std::optional<SelectedBlockStorePhysical>
selectBlockStorePhysical(const BlockStoreCandidateFacts &facts,
                         const RISCVTargetProfile &target);

enum class BlockReduceRealization {
  RVVFixedStrips,
  RVVDynamicStrips,
};

struct BlockReducePhysicalDecision {
  BlockReduceRealization realization = BlockReduceRealization::RVVFixedStrips;
  unsigned stripVL = 0;
  bool needsLaneVector = false;
  RVVVectorShape laneShape;
};

struct BlockReduceCandidateFacts {
  int64_t extent = 0;
  bool needsLaneVector = false;
  bool supportsStandardOperations = false;
  RVVVectorShape inputShape;
};

struct SelectedBlockReducePhysical {
  BlockReducePhysicalDecision decision;
  RVVVectorShape inputShape;
  RVVVectorShape combinedShape;
  RVVVectorShape seedShape;
  bool wideningCombine = false;
  PhysicalResourceBudget resources;
};

std::optional<SelectedBlockReducePhysical>
selectBlockReducePhysical(const BlockReduceCandidateFacts &facts,
                          const RISCVTargetProfile &target);

enum class MaterializedBlockStoreRealization {
  ScalarRankOne,
  ScalarRankTwo,
  RVVContiguousRankOne,
};

struct MaterializedBlockStoreCandidateFacts {
  unsigned rank = 0;
  int64_t rows = 0;
  int64_t columns = 0;
  bool allActive = false;
  bool prefixPredicated = false;
  bool unitStride = false;
};

struct SelectedMaterializedBlockStorePhysical {
  MaterializedBlockStoreRealization realization =
      MaterializedBlockStoreRealization::ScalarRankOne;
  int64_t rows = 0;
  int64_t columns = 0;
  RVVVectorShape vectorShape;
  PhysicalResourceBudget resources;
};

std::optional<SelectedMaterializedBlockStorePhysical>
selectMaterializedBlockStorePhysical(
    const MaterializedBlockStoreCandidateFacts &facts,
    const RISCVTargetProfile &target);

enum class SortIndicesStructure {
  StableF32Radix,
};

struct SortIndicesCandidateFacts {
  bool descending = false;
  int64_t configuredRadixBits = 0;
};

struct SelectedSortIndicesPhysical {
  SortIndicesStructure structure = SortIndicesStructure::StableF32Radix;
  bool descending = false;
  unsigned radixBits = 0;
  unsigned passes = 0;
  int64_t privateElements = 0;
  int64_t privateAlignment = 0;
  PhysicalResourceBudget resources;
};

std::optional<SelectedSortIndicesPhysical>
selectSortIndicesPhysical(const SortIndicesCandidateFacts &facts,
                          const RISCVTargetProfile &target);

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

enum class VLAInactiveLaneRealization {
  ExplicitPassthrough,
  ZeroCarrierWithLogicalValidity,
};

struct VLAAccessCandidateFacts {
  LaneRelation relation;
  bool write = false;
  bool predicateAllActive = false;
  bool predicateVector = false;
  bool predicateScalar = false;
  bool carriesLogicalValidity = false;
  bool indexedOffsetFromU32 = false;
};

struct SelectedVLAAccessPhysical {
  VLAMemoryMode memoryMode = VLAMemoryMode::UnitStride;
  VLAActivityMode activityMode = VLAActivityMode::AllActive;
  VLAInactiveLaneRealization inactiveLane =
      VLAInactiveLaneRealization::ExplicitPassthrough;
  unsigned indexedSEW = 0;
};

std::optional<SelectedVLAAccessPhysical>
selectVLAAccessPhysical(const VLAAccessCandidateFacts &facts,
                        const RISCVTargetProfile &target);

enum class VLASegment2AccessKind {
  Load,
  Store,
};

struct SelectedVLASegment2Physical {
  VLASegment2AccessKind kind = VLASegment2AccessKind::Load;
  bool emitAtEarlierAccess = true;
  int64_t coordinateScale = 2;
};

std::optional<SelectedVLASegment2Physical>
selectVLASegment2Physical(bool load,
                          const RISCVTargetProfile &target);

enum class VLAPredicateRealization {
  RVVAffineIndexScalar,
  RVVVectorScalar,
};

struct VLAPredicateCandidateFacts {
  bool affineIndex = false;
  bool vectorScalar = false;
  LaneRelation coordinateRelation;
  unsigned vectorSEW = 0;
};

struct SelectedVLAPredicatePhysical {
  VLAPredicateRealization realization =
      VLAPredicateRealization::RVVAffineIndexScalar;
  VLAMemoryMode coordinateMode = VLAMemoryMode::UnitStride;
  unsigned vectorSEW = 0;
};

std::optional<SelectedVLAPredicatePhysical>
selectVLAPredicatePhysical(const VLAPredicateCandidateFacts &facts,
                           const RISCVTargetProfile &target);

enum class VLAIndexBinaryRealization {
  RVVUnsignedVectorScalar,
  RVVUnsignedVectorVector,
};

struct VLAIndexBinaryCandidateFacts {
  bool lhsVector = false;
  bool rhsVector = false;
  bool scalarLeftDivision = false;
};

std::optional<VLAIndexBinaryRealization>
selectVLAIndexBinaryPhysical(const VLAIndexBinaryCandidateFacts &facts);

enum class VLACastSemantic {
  Identity,
  F16ToF32,
  F32ToF16,
  U32ToIndex,
  IndexToF32,
};

enum class VLACastRealization {
  RVVIdentity,
  RVVWidenF16ToF32,
  RVVNarrowF32ToF16,
  RVVZeroExtendU32ToIndex,
  RVVIndexToF32,
};

struct VLACastCandidateFacts {
  VLACastSemantic semantic = VLACastSemantic::Identity;
  RVVVectorShape dataShape;
  unsigned identitySEW = 0;
};

struct SelectedVLACastPhysical {
  VLACastRealization realization = VLACastRealization::RVVIdentity;
  RVVVectorShape sourceShape;
  RVVVectorShape resultShape;
};

std::optional<SelectedVLACastPhysical>
selectVLACastPhysical(const VLACastCandidateFacts &facts,
                      const RISCVTargetProfile &target);

enum class VLAUnarySemantic {
  Exp,
  Tanh,
  Sin,
  Cos,
};

enum class VLAUnaryRealization {
  RVVExpPolynomial,
  RVVTanhViaExp,
  RVVScalarLibmSin,
  RVVScalarLibmCos,
};

std::optional<VLAUnaryRealization>
selectVLAUnaryPhysical(VLAUnarySemantic semantic,
                       const RISCVTargetProfile &target);

enum class VLAStateSemantic {
  F32AddReduction,
  F32MaxReduction,
  I8AddReductionI32,
  InclusiveAddScan,
  SegmentedInclusiveAddScan,
  ArgMaxSummary,
  OnlineSoftmaxSummary,
};

enum class VLAStateCarryRepresentation {
  Scalar,
  Vector,
  ScalarTuple,
};

enum class VLAStateStripUpdate {
  AddReduction,
  MaxReduction,
  WideningAddReduction,
  InclusiveAddScan,
  SegmentedInclusiveAddScan,
  ArgMaxSummary,
  OnlineSoftmaxSummary,
};

enum class VLAStateFinalize {
  Direct,
  HorizontalAdd,
  HorizontalMax,
};

enum class VLALookupRealization {
  RVVTableGather,
};

struct VLALookupCandidateFacts {
  unsigned tableExtent = 0;
  bool tableF32 = false;
  bool resultF32 = false;
  bool indicesU8 = false;
  bool allActive = false;
};

struct SelectedVLALookupPhysical {
  VLALookupRealization realization = VLALookupRealization::RVVTableGather;
  unsigned tableExtent = 0;
};

std::optional<SelectedVLALookupPhysical>
selectVLALookupPhysical(const VLALookupCandidateFacts &facts,
                        const RISCVTargetProfile &target);

struct VLAValueLifetimeSnapshot {
  unsigned f32 = 0;
  unsigned f16 = 0;
  unsigned index = 0;
  unsigned byte = 0;
  unsigned u32 = 0;
  unsigned mask = 0;
};

enum class LocalPrimitiveKind {
  None,
  F32Math,
  SymmetricI4I8,
  AffineI4I8,
  GroupedAffineI4I8,
  E2M1E8M0I8,
  PackedI4I8,
  PackedI5I8,
  PackedI3GroupedI8,
  Base3TernaryI8,
  PackedI2TernaryI8,
  SignedCodebook8I8,
  SignedCodebook4I8,
  PackedU9U7CodebookI8,
  PackedU11GridDeltaI8,
  NibbleCodebookI8,
  IQ2SI8,
  IQ3SI8,
  IQ1MI8,
  Q6KI8,
};

enum class LocalImplementationStructure {
  None,
  RVVRegisterMicrokernel,
  RVVStripLoop,
  SpacemitIME1Fragment,
};

enum class LocalImplementationLeaf {
  None,
  RVVF32Math,
  RVVSymmetricI4I8N16,
  RVVSymmetricI4I8M4N16,
  IME1SymmetricI4I8N16,
  IME1SymmetricI4I8M4N16,
  RVVAffineI4I8N16,
  RVVAffineI4I8M4N16,
  IME1AffineI4I8N16,
  IME1AffineI4I8M4N16,
  RVVGroupedAffineI4I8Strip,
  RVVE2M1E8M0I8RegisterMF2,
  RVVE2M1E8M0I8RegisterM1M2,
  RVVE2M1E8M0I8Strip,
  RVVPackedI4I8Register,
  RVVPackedI5I8Register,
  RVVPackedI3GroupedI8Register,
  RVVBase3TernaryI8Register,
  RVVPackedI2TernaryI8Register,
  RVVSignedCodebook8I8Register,
  RVVSignedCodebook4I8Register,
  RVVPackedU9U7CodebookI8Register,
  RVVPackedU11GridDeltaI8Register,
  RVVNibbleCodebookI8Register,
  RVVIQ2SI8Register,
  RVVIQ2SI8Strip,
  RVVIQ3SI8Register,
  RVVIQ3SI8Strip,
  RVVIQ1MI8Register,
  RVVIQ1MI8Strip,
  RVVQ6KI8Register,
  RVVQ6KI8Strip,
};

struct LocalImplementationParameters {
  unsigned rowMicrotile = 1;
  unsigned semanticLanes = 0;
  unsigned entryWidth = 0;
  RVVVectorShape primaryShape;
  RVVVectorShape secondaryShape;

  bool operator==(const LocalImplementationParameters &other) const {
    return rowMicrotile == other.rowMicrotile &&
           semanticLanes == other.semanticLanes &&
           entryWidth == other.entryWidth &&
           primaryShape == other.primaryShape &&
           secondaryShape == other.secondaryShape;
  }
  bool operator<(const LocalImplementationParameters &other) const {
    return std::tie(rowMicrotile, semanticLanes, entryWidth, primaryShape,
                    secondaryShape) <
           std::tie(other.rowMicrotile, other.semanticLanes,
                    other.entryWidth, other.primaryShape,
                    other.secondaryShape);
  }
};

struct LocalImplementation {
  LocalPrimitiveKind primitive = LocalPrimitiveKind::None;
  LocalImplementationStructure structure =
      LocalImplementationStructure::None;
  LocalImplementationLeaf leaf = LocalImplementationLeaf::None;
  LocalImplementationParameters parameters;

  explicit operator bool() const {
    return primitive != LocalPrimitiveKind::None &&
           structure != LocalImplementationStructure::None &&
           leaf != LocalImplementationLeaf::None;
  }
  bool operator==(const LocalImplementation &other) const {
    return primitive == other.primitive && structure == other.structure &&
           leaf == other.leaf && parameters == other.parameters;
  }
  bool operator<(const LocalImplementation &other) const {
    return std::tie(primitive, structure, leaf, parameters) <
           std::tie(other.primitive, other.structure, other.leaf,
                    other.parameters);
  }
};

std::optional<LocalImplementation>
selectF32MathLocalImplementation(const RISCVTargetProfile &target);

struct I4I8FragmentCandidateFacts {
  unsigned rowTile = 1;
  bool affine = false;
};

struct SelectedI4I8FragmentPhysical {
  LocalImplementation implementation;
  RVVVectorShape codeShape;
  RVVVectorShape activationScaleShape;
  RVVVectorShape accumulatorShape;
  PhysicalResourceBudget resources;
};

std::optional<SelectedI4I8FragmentPhysical>
selectI4I8FragmentPhysical(const I4I8FragmentCandidateFacts &facts,
                           const RISCVTargetProfile &target);

struct VLAStateCandidateFacts {
  VLAStateSemantic semantic = VLAStateSemantic::F32AddReduction;
  bool relaxedOrder = false;
  unsigned reductionStateCount = 0;
};

struct SelectedVLAStatePhysical {
  VLAStateCarryRepresentation carry = VLAStateCarryRepresentation::Scalar;
  VLAStateStripUpdate stripUpdate = VLAStateStripUpdate::AddReduction;
  VLAStateFinalize finalize = VLAStateFinalize::Direct;
  bool wholeVLALifetime = true;
};

std::optional<SelectedVLAStatePhysical>
selectVLAStatePhysical(const VLAStateCandidateFacts &facts,
                       const RISCVTargetProfile &target,
                       const RISCVBackendConfig &config);

struct VLAStateResourceFact {
  VLAStateCarryRepresentation carry = VLAStateCarryRepresentation::Scalar;
  VLAStateStripUpdate stripUpdate = VLAStateStripUpdate::AddReduction;
  bool wholeVLALifetime = true;
};

struct VLAIndexedMemoryFact {
  unsigned elementSEW = 0;
  unsigned offsetSEW = 0;
};

struct VLANarrowPhysical {
  RVVVectorShape sourceShape;
  RVVVectorShape intermediateShape;
  RVVVectorShape resultShape;
  PhysicalResourceBudget resources;
};

struct VLAEntityCandidateFacts {
  unsigned dataSEW = 32;
  unsigned stridedAccesses = 0;
  unsigned indexedAccesses = 0;
  unsigned maxIndexedOffsetSEW = 0;
  unsigned segmentLoadPairs = 0;
  unsigned segmentStorePairs = 0;
  unsigned lookupCount = 0;
  bool hasF32Division = false;
  bool hasFloatCast = false;
  bool hasIndexVector = false;
  bool hasAffinePredicate = false;
  bool hasNarrow = false;
  RVVVectorShape requiredDataShape;
  llvm::SmallVector<unsigned> requiredLMULs;
  llvm::SmallVector<unsigned> accessElementSEWs;
  llvm::SmallVector<VLAIndexedMemoryFact> indexedMemory;
  llvm::SmallVector<VLAValueLifetimeSnapshot> lifetimes;
  llvm::SmallVector<VLAStateResourceFact> states;
  llvm::SmallVector<PhysicalResourceBudget> localPrimitiveResources;
};

struct SelectedVLAEntityPhysical {
  RVVVectorShape dataShape;
  RVVVectorShape indexShape;
  unsigned maskRatio = 0;
  std::optional<VLANarrowPhysical> narrow;
  PhysicalResourceBudget resources;
};

std::optional<SelectedVLAEntityPhysical>
selectVLAEntityPhysical(const VLAEntityCandidateFacts &facts,
                        const RISCVTargetProfile &target);

struct SelectedSignBitI8Physical {
  RVVVectorShape activationShape;
  RVVVectorShape widenedShape;
  RVVVectorShape reductionShape;
  unsigned maskRatio = 0;
  PhysicalResourceBudget resources;
};

std::optional<SelectedSignBitI8Physical>
selectSignBitI8Physical(const RISCVTargetProfile &target);

enum class TernaryI8DotSemantic {
  Base3Digits,
  PackedI2Fields,
};

struct TernaryI8DotCandidateFacts {
  TernaryI8DotSemantic semantic = TernaryI8DotSemantic::Base3Digits;
};

struct SelectedTernaryI8DotPhysical {
  LocalImplementation implementation;
  RVVVectorShape byteShape32;
  RVVVectorShape byteShape16;
  RVVVectorShape widenedShape32;
  RVVVectorShape widenedShape16;
  RVVVectorShape reductionShape;
  PhysicalResourceBudget resources;
};

std::optional<SelectedTernaryI8DotPhysical>
selectTernaryI8DotPhysical(const TernaryI8DotCandidateFacts &facts,
                           const RISCVTargetProfile &target);

struct CodebookGatherI8CandidateFacts {
  LocalPrimitiveKind primitive = LocalPrimitiveKind::None;
  unsigned codeByteExtent = 0;
  unsigned entryWidth = 0;
};

struct SelectedCodebookGatherI8Physical {
  LocalImplementation implementation;
  unsigned entryWidth = 0;
  RVVVectorShape codeShape;
  RVVVectorShape indexShape;
  RVVVectorShape tableShape;
  RVVVectorShape activationShape;
  RVVVectorShape productShape;
  RVVVectorShape reductionShape;
  PhysicalResourceBudget resources;
};

std::optional<SelectedCodebookGatherI8Physical>
selectCodebookGatherI8Physical(const CodebookGatherI8CandidateFacts &facts,
                               const RISCVTargetProfile &target);

struct SelectedNibbleCodebookI8Physical {
  LocalImplementation implementation;
  RVVVectorShape packedShape;
  RVVVectorShape tableShape;
  RVVVectorShape activationShape;
  RVVVectorShape productShape;
  RVVVectorShape reductionShape;
  PhysicalResourceBudget resources;
};

std::optional<SelectedNibbleCodebookI8Physical>
selectNibbleCodebookI8Physical(const RISCVTargetProfile &target);

enum class QuantI8DotSemantic {
  PackedI4,
  PackedI5,
  PackedI3Grouped,
  IQ2S,
  IQ3S,
  IQ1M,
  Q6K,
};

struct QuantI8DotCandidateFacts {
  QuantI8DotSemantic semantic = QuantI8DotSemantic::IQ2S;
  unsigned semanticExtent = 256;
};

struct SelectedQuantI8DotPhysical {
  LocalImplementation implementation;
  unsigned semanticLanes = 0;
  RVVVectorShape byteShape;
  unsigned reductionSegments = 0;
  PhysicalResourceBudget resources;
};

std::optional<SelectedQuantI8DotPhysical>
selectQuantI8DotPhysical(const QuantI8DotCandidateFacts &facts,
                         const RISCVTargetProfile &target);

struct SelectedE2M1E8M0I8Physical {
  LocalImplementation implementation;
  RVVVectorShape packedShape;
  RVVVectorShape activationShape;
  PhysicalResourceBudget resources;
};

std::optional<SelectedE2M1E8M0I8Physical>
selectE2M1E8M0I8Physical(const RISCVTargetProfile &target);

struct GroupedAffineI4I8CandidateFacts {
  unsigned packedExtent = 0;
  unsigned scaleExtent = 0;
  unsigned activationExtent = 0;
  unsigned activationSumExtent = 0;
};

struct SelectedGroupedAffineI4I8Physical {
  LocalImplementation implementation;
  RVVVectorShape packedShape;
  RVVVectorShape scaleShape;
  RVVVectorShape activationShape;
  RVVVectorShape activationSumShape;
  RVVVectorShape widenedShape;
  RVVVectorShape reductionShape;
  PhysicalResourceBudget resources;
};

std::optional<SelectedGroupedAffineI4I8Physical>
selectGroupedAffineI4I8Physical(
    const GroupedAffineI4I8CandidateFacts &facts,
    const RISCVTargetProfile &target);

enum class F32DotResourceModel {
  VLAFreeAxis,
  LocalRow,
};

enum class F32DotStructure {
  RVVVLAMicrotile,
  RVVVLAVectorDot,
  RVVLocalRowMicrokernel,
};

enum class DenseVectorOrganization {
  ReductionAxis,
  FreeMAxis,
  FreeNAxis,
};

enum class DenseLoadSchedule {
  Streamed,
  DoubleBuffered,
};

struct DenseMicrokernelResourceFacts {
  RVVVectorShape inputShape;
  RVVVectorShape accumulatorShape;
  unsigned accumulatorVectors = 0;
  unsigned lhsVectorsPerWindow = 0;
  unsigned rhsVectorsPerWindow = 0;
  unsigned loadWindow = 1;
  unsigned predicateGroups = 0;
  unsigned stateGroups = 0;
  unsigned handoffGroups = 0;
};

std::optional<PhysicalResourceBudget>
calculateDenseMicrokernelResources(
    const DenseMicrokernelResourceFacts &facts,
    const RISCVTargetProfile &target);

struct F32DotParameters {
  unsigned lmul = 1;
  unsigned kUnroll = 1;
};

struct F32DotCandidateFacts {
  F32DotResourceModel model = F32DotResourceModel::LocalRow;
  unsigned rowTile = 1;
  std::optional<uint64_t> reductionExtent;
  unsigned unitStrideOperands = 0;
  unsigned stridedOperands = 0;
  unsigned indexedOperands = 0;
  unsigned predicateGroups = 0;
  unsigned stateGroups = 0;
  unsigned handoffGroups = 0;
  bool reductionPredicate = false;
  bool materializedInit = false;
  bool vlaVectorFreeAxis = false;
};

struct SelectedF32DotPhysical {
  F32DotStructure structure = F32DotStructure::RVVLocalRowMicrokernel;
  DenseVectorOrganization vectorOrganization =
      DenseVectorOrganization::ReductionAxis;
  DenseLoadSchedule loadSchedule = DenseLoadSchedule::Streamed;
  F32DotParameters parameters;
  PhysicalResourceBudget resources;
};

std::optional<SelectedF32DotPhysical>
selectF32DotPhysicalConfig(const F32DotCandidateFacts &facts,
                           const RISCVTargetProfile &target,
                           const RISCVBackendConfig &config);

struct F16MatmulParameters {
  unsigned rowMicrotile = 1;
  unsigned columnMicrotile = 1;
  unsigned inputLMUL = 1;
  unsigned kUnroll = 1;
  unsigned pipelineDepth = 1;
  unsigned loadLookahead = 0;
};

struct F16MatmulCandidateFacts {
  unsigned rowTile = 1;
  unsigned columnTile = 1;
  unsigned reductionTile = 1;
};

struct SelectedF16MatmulPhysical {
  DenseVectorOrganization vectorOrganization =
      DenseVectorOrganization::ReductionAxis;
  DenseLoadSchedule loadSchedule = DenseLoadSchedule::Streamed;
  F16MatmulParameters parameters;
  PhysicalResourceBudget resources;
};

std::optional<SelectedF16MatmulPhysical>
selectF16MatmulPhysicalConfig(const F16MatmulCandidateFacts &facts,
                              const RISCVTargetProfile &target,
                              const RISCVBackendConfig &config);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVPHYSICALPLANNING_H
