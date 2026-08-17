#ifndef WEFT_LIB_TARGET_RISCVPHYSICALPLANNING_H
#define WEFT_LIB_TARGET_RISCVPHYSICALPLANNING_H

#include "RISCVAxisMapping.h"
#include "Weft/Target/RISCVLowering.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

namespace weft::riscv_internal {

enum class LaneRelation;

bool fitsPrivateStorage(int64_t elements, unsigned elementBytes,
                        const RISCVTargetProfile &target);
std::optional<int64_t>
extendPrivateStorageElementCount(int64_t currentElements, int64_t extent,
                                 unsigned elementBytes,
                                 const RISCVTargetProfile &target);

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

struct BlockDecodeCandidateFacts {
  unsigned codeExtent = 0;
  unsigned tableExtent = 0;
};

struct SelectedBlockDecodePhysical {
  CorePhysicalMapping mapping;
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

struct BlockStorePhysicalDecision {
  CorePhysicalMapping mapping;
  bool needsLaneVector = false;
  RVVVectorShape laneShape;
};

struct BlockStoreCandidateFacts {
  CoreMappingProblem mapping;
  bool needsLaneVector = false;
};

struct SelectedBlockStorePhysical {
  BlockStorePhysicalDecision decision;
  PhysicalResourceBudget resources;
};

std::optional<SelectedBlockStorePhysical>
selectBlockStorePhysical(const BlockStoreCandidateFacts &facts,
                         const RISCVTargetProfile &target);

struct BlockReducePhysicalDecision {
  CorePhysicalMapping mapping;
  bool needsLaneVector = false;
  RVVVectorShape laneShape;
};

struct BlockReduceCandidateFacts {
  CoreMappingProblem mapping;
  bool needsLaneVector = false;
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

struct MaterializedBlockStoreCandidateFacts {
  CoreMappingProblem mapping;
  bool allActive = false;
  bool prefixPredicated = false;
  bool unitStride = false;
};

struct SelectedMaterializedBlockStorePhysical {
  CorePhysicalMapping mapping;
  PhysicalResourceBudget resources;
};

std::optional<SelectedMaterializedBlockStorePhysical>
selectMaterializedBlockStorePhysical(
    const MaterializedBlockStoreCandidateFacts &facts,
    const RISCVTargetProfile &target);

struct SortIndicesCandidateFacts {
  CoreMappingProblem mapping;
  bool descending = false;
  int64_t configuredRadixBits = 0;
};

struct SelectedSortIndicesPhysical {
  CorePhysicalMapping mapping;
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

struct VLASegment2CandidateFacts {
  bool load = true;
  unsigned fields = 2;
  unsigned elementSEW = 0;
};

struct SelectedVLASegment2Physical {
  VLASegment2AccessKind kind = VLASegment2AccessKind::Load;
  bool emitAtEarlierAccess = true;
  int64_t coordinateScale = 2;
  unsigned fields = 2;
  unsigned elementSEW = 0;
};

std::optional<SelectedVLASegment2Physical>
selectVLASegment2Physical(const VLASegment2CandidateFacts &facts,
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

struct LocalImplementation {
  LocalPrimitiveKind primitive = LocalPrimitiveKind::None;
  CorePhysicalMapping mapping;
  llvm::SmallVector<RVVVectorShape, 4> valueShapes;
  unsigned entryWidth = 0;
  std::string helperSymbol;

  explicit operator bool() const {
    return primitive != LocalPrimitiveKind::None && mapping;
  }
  bool operator==(const LocalImplementation &other) const {
    return primitive == other.primitive && mapping == other.mapping &&
           valueShapes == other.valueShapes && entryWidth == other.entryWidth &&
           helperSymbol == other.helperSymbol;
  }
  bool operator<(const LocalImplementation &other) const {
    if (primitive != other.primitive)
      return primitive < other.primitive;
    if (!(mapping == other.mapping))
      return mapping < other.mapping;
    if (valueShapes != other.valueShapes)
      return std::lexicographical_compare(valueShapes.begin(), valueShapes.end(),
                                          other.valueShapes.begin(),
                                          other.valueShapes.end());
    if (entryWidth != other.entryWidth)
      return entryWidth < other.entryWidth;
    return helperSymbol < other.helperSymbol;
  }
};

std::optional<LocalImplementation>
selectF32MathLocalImplementation(const RISCVTargetProfile &target);

struct I4I8FragmentCandidateFacts {
  CoreMappingProblem mapping;
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
                           const RISCVTargetProfile &target,
                           const RISCVBackendConfig &config);

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

struct VLAIndexedMemoryFact {
  unsigned elementSEW = 0;
  unsigned offsetSEW = 0;
};

struct VLASegmentMemoryFact {
  bool write = false;
  unsigned fields = 0;
  unsigned elementSEW = 0;
};

struct VLANarrowPhysical {
  RVVVectorShape sourceShape;
  RVVVectorShape intermediateShape;
  RVVVectorShape resultShape;
  PhysicalResourceBudget resources;
};

struct VLAEntityCandidateFacts {
  CoreMappingProblem mapping;
  unsigned dataSEW = 32;
  unsigned lookupCount = 0;
  bool hasIndexVector = false;
  bool hasAffinePredicate = false;
  bool hasNarrow = false;
  RVVVectorShape requiredDataShape;
  llvm::SmallVector<unsigned> requiredLMULs;
  llvm::SmallVector<unsigned> accessElementSEWs;
  llvm::SmallVector<VLAIndexedMemoryFact> indexedMemory;
  llvm::SmallVector<VLASegmentMemoryFact> segmentMemory;
  llvm::SmallVector<VLAValueLifetimeSnapshot> lifetimes;
  llvm::SmallVector<SelectedVLAStatePhysical> states;
  llvm::SmallVector<PhysicalResourceBudget> localPrimitiveResources;
};

struct SelectedVLAEntityPhysical {
  CorePhysicalMapping mapping;
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

struct RVVShapeMultiplicity {
  RVVVectorShape shape;
  unsigned count = 0;
};

struct QuantDecodeResourceFacts {
  llvm::SmallVector<RVVShapeMultiplicity> loadedValues;
  llvm::SmallVector<RVVShapeMultiplicity> indexValues;
  llvm::SmallVector<RVVShapeMultiplicity> temporaryValues;
  unsigned predicateGroups = 0;
};

std::optional<PhysicalResourceBudget>
calculateQuantDecodeResources(const QuantDecodeResourceFacts &facts,
                              const RISCVTargetProfile &target);

enum class TernaryI8DotSemantic {
  Base3Digits,
  PackedI2Fields,
};

struct TernaryI8DotCandidateFacts {
  TernaryI8DotSemantic semantic = TernaryI8DotSemantic::Base3Digits;
};

enum class TernaryDecodeTopology {
  Base3Digits,
  PackedI2Fields,
};

struct SelectedTernaryI8DotPhysical {
  LocalImplementation implementation;
  TernaryDecodeTopology decode = TernaryDecodeTopology::Base3Digits;
  RVVVectorShape primarySourceShape;
  RVVVectorShape secondarySourceShape;
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
  RVVVectorShape codeShape;
  RVVVectorShape activationShape;
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
  RVVVectorShape operandShape;
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
  CoreMappingProblem mapping;
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

struct F32DotCandidateFacts {
  CoreMappingProblem mapping;
  std::optional<uint64_t> reductionExtent;
  unsigned unitStrideOperands = 0;
  unsigned stridedOperands = 0;
  unsigned indexedOperands = 0;
  unsigned predicateGroups = 0;
  unsigned stateGroups = 0;
  unsigned handoffGroups = 0;
  bool reductionPredicate = false;
  bool materializedInit = false;
};

struct SelectedF32DotPhysical {
  CorePhysicalMapping mapping;
  PhysicalResourceBudget resources;
};

std::optional<SelectedF32DotPhysical>
selectF32DotPhysicalConfig(const F32DotCandidateFacts &facts,
                           const RISCVTargetProfile &target,
                           const RISCVBackendConfig &config);

struct F16MatmulCandidateFacts {
  CoreMappingProblem mapping;
};

struct SelectedF16MatmulPhysical {
  CorePhysicalMapping mapping;
  PhysicalResourceBudget resources;
};

std::optional<SelectedF16MatmulPhysical>
selectF16MatmulPhysicalConfig(const F16MatmulCandidateFacts &facts,
                              const RISCVTargetProfile &target,
                              const RISCVBackendConfig &config);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVPHYSICALPLANNING_H
