#ifndef WEFT_LIB_TARGET_RISCVPHYSICALPLANNING_H
#define WEFT_LIB_TARGET_RISCVPHYSICALPLANNING_H

#include "RISCVAxisMapping.h"
#include "RISCVKernelFacts.h"
#include "Weft/Target/RISCVLowering.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

namespace weft::riscv_internal {

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

enum class PhysicalMemoryMode {
  UnitStride,
  Strided,
  Indexed,
  Segment2,
};

enum class PhysicalActivityMode {
  AllActive,
  PredicateMask,
  ScalarPredicate,
};

enum class PhysicalInactiveLaneRealization {
  ExplicitPassthrough,
  ZeroCarrierWithLogicalValidity,
};

struct BlockMemoryCandidateFacts {
  LaneRelation relation = LaneRelation::Independent;
  bool write = false;
  bool predicateAllActive = false;
  bool predicateVector = false;
  unsigned elementSEW = 0;
  unsigned indexSEW = 0;
};

struct SelectedBlockMemoryPhysical {
  PhysicalMemoryMode memoryMode = PhysicalMemoryMode::UnitStride;
  PhysicalActivityMode activityMode = PhysicalActivityMode::AllActive;
  PhysicalInactiveLaneRealization inactiveLane =
      PhysicalInactiveLaneRealization::ExplicitPassthrough;
  unsigned indexedSEW = 0;
};

std::optional<SelectedBlockMemoryPhysical>
selectBlockMemoryPhysical(const BlockMemoryCandidateFacts &facts,
                          const RISCVTargetProfile &target);

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
  std::optional<BlockMemoryCandidateFacts> memory;
};

struct SelectedBlockOperationPhysical {
  BlockValueRealization realization = BlockValueRealization::Direct;
  RVVVectorShape resultShape;
  RVVVectorShape temporaryShape;
  std::optional<uint64_t> unsignedMaximum;
  unsigned maskRatio = 0;
  std::optional<SelectedBlockMemoryPhysical> memory;
};

std::optional<SelectedBlockOperationPhysical>
selectBlockOperationPhysical(const BlockOperationCandidateFacts &facts,
                             const RISCVTargetProfile &target);

enum class BlockStoreRealization {
  RegisterRepetition,
  SequentialStrip,
  MultiStripValues,
};

struct BlockStorePhysicalDecision {
  CorePhysicalMapping mapping;
  bool needsLaneVector = false;
  RVVVectorShape laneShape;
  BlockStoreRealization realization = BlockStoreRealization::SequentialStrip;
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
  unsigned inputSEW = 0;
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
  PhysicalMemoryMode memoryMode = PhysicalMemoryMode::UnitStride;
  PhysicalActivityMode activityMode = PhysicalActivityMode::AllActive;
  PhysicalInactiveLaneRealization inactiveLane =
      PhysicalInactiveLaneRealization::ExplicitPassthrough;
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
  PhysicalMemoryMode coordinateMode = PhysicalMemoryMode::UnitStride;
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
  unsigned operationOrdinal = 0;
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

enum class LocalHardwareOperationKind {
  None,
  RVVIntrinsic,
  RVVInlineAsm,
  MatrixFragment,
};

enum class LocalOperationProjection {
  None,
  PackedDotLaneSlide,
  PackedDotLaneCreate,
  PackedDotRegisterChunks,
  NibbleCodebookCombined,
  NibbleCodebookSplit,
  SignSourceDirect,
  SignSourceExtend,
};

struct LocalOperandWindow {
  unsigned valueIndex = 0;
  PhysicalMemoryMode memoryMode = PhysicalMemoryMode::UnitStride;
  unsigned vectorsPerStep = 1;
  unsigned reuseCount = 1;
  bool advancesIteration = true;

  bool operator==(const LocalOperandWindow &other) const {
    return std::tie(valueIndex, memoryMode, vectorsPerStep, reuseCount,
                    advancesIteration) ==
           std::tie(other.valueIndex, other.memoryMode,
                    other.vectorsPerStep, other.reuseCount,
                    other.advancesIteration);
  }
  bool operator<(const LocalOperandWindow &other) const {
    return std::tie(valueIndex, memoryMode, vectorsPerStep, reuseCount,
                    advancesIteration) <
           std::tie(other.valueIndex, other.memoryMode,
                    other.vectorsPerStep, other.reuseCount,
                    other.advancesIteration);
  }
};

enum class LocalPipelineActionKind {
  Load,
  Compute,
};

struct LocalPipelineAction {
  LocalPipelineActionKind kind = LocalPipelineActionKind::Load;
  unsigned iteration = 0;
  unsigned buffer = 0;

  bool operator==(const LocalPipelineAction &other) const {
    return std::tie(kind, iteration, buffer) ==
           std::tie(other.kind, other.iteration, other.buffer);
  }
  bool operator<(const LocalPipelineAction &other) const {
    return std::tie(kind, iteration, buffer) <
           std::tie(other.kind, other.iteration, other.buffer);
  }
};

struct LocalPipelineSchedule {
  unsigned bufferCount = 1;
  unsigned prefetchDistance = 0;
  llvm::SmallVector<LocalPipelineAction, 8> actions;

  bool operator==(const LocalPipelineSchedule &other) const {
    return bufferCount == other.bufferCount &&
           prefetchDistance == other.prefetchDistance &&
           actions == other.actions;
  }
  bool operator<(const LocalPipelineSchedule &other) const {
    if (std::tie(bufferCount, prefetchDistance) !=
        std::tie(other.bufferCount, other.prefetchDistance))
      return std::tie(bufferCount, prefetchDistance) <
             std::tie(other.bufferCount, other.prefetchDistance);
    return std::lexicographical_compare(actions.begin(), actions.end(),
                                        other.actions.begin(),
                                        other.actions.end());
  }
};

struct LocalDecodeSchedule {
  unsigned chunksPerStep = 1;
  unsigned groupsPerChunk = 1;
  unsigned reductionSegments = 1;

  bool operator==(const LocalDecodeSchedule &other) const {
    return std::tie(chunksPerStep, groupsPerChunk, reductionSegments) ==
           std::tie(other.chunksPerStep, other.groupsPerChunk,
                    other.reductionSegments);
  }
  bool operator<(const LocalDecodeSchedule &other) const {
    return std::tie(chunksPerStep, groupsPerChunk, reductionSegments) <
           std::tie(other.chunksPerStep, other.groupsPerChunk,
                    other.reductionSegments);
  }
};

struct LocalMicrokernelSchedule {
  std::optional<unsigned> iterationAxis;
  unsigned sequentialIterations = 1;
  unsigned laneFactor = 1;
  unsigned iterationRegisterFactor = 1;
  unsigned iterationElements = 1;
  unsigned unrollFactor = 1;
  unsigned accumulatorCount = 1;
  LocalPipelineSchedule pipeline;
  llvm::SmallVector<LocalOperandWindow, 4> operands;
  LocalDecodeSchedule decode;

  explicit operator bool() const {
    return iterationAxis && sequentialIterations != 0 && laneFactor != 0 &&
           iterationRegisterFactor != 0 && iterationElements != 0 &&
           unrollFactor != 0 && accumulatorCount != 0 &&
           pipeline.bufferCount != 0 &&
           pipeline.prefetchDistance < pipeline.bufferCount &&
           !pipeline.actions.empty();
  }
  bool operator==(const LocalMicrokernelSchedule &other) const {
    return iterationAxis == other.iterationAxis &&
           sequentialIterations == other.sequentialIterations &&
           laneFactor == other.laneFactor &&
           iterationRegisterFactor == other.iterationRegisterFactor &&
           iterationElements == other.iterationElements &&
           unrollFactor == other.unrollFactor &&
           accumulatorCount == other.accumulatorCount &&
           pipeline == other.pipeline && operands == other.operands &&
           decode == other.decode;
  }
  bool operator<(const LocalMicrokernelSchedule &other) const {
    if (std::tie(iterationAxis, sequentialIterations, laneFactor,
                 iterationRegisterFactor, iterationElements, unrollFactor,
                 accumulatorCount, pipeline) !=
        std::tie(other.iterationAxis, other.sequentialIterations,
                 other.laneFactor, other.iterationRegisterFactor,
                 other.iterationElements, other.unrollFactor,
                 other.accumulatorCount, other.pipeline))
      return std::tie(iterationAxis, sequentialIterations, laneFactor,
                      iterationRegisterFactor, iterationElements,
                      unrollFactor, accumulatorCount, pipeline) <
             std::tie(other.iterationAxis, other.sequentialIterations,
                      other.laneFactor, other.iterationRegisterFactor,
                      other.iterationElements, other.unrollFactor,
                      other.accumulatorCount, other.pipeline);
    if (operands != other.operands)
      return std::lexicographical_compare(operands.begin(), operands.end(),
                                          other.operands.begin(),
                                          other.operands.end());
    return decode < other.decode;
  }
};

struct LocalHardwareOperation {
  LocalHardwareOperationKind kind = LocalHardwareOperationKind::None;
  unsigned rows = 1;
  LocalOperationProjection projection = LocalOperationProjection::None;

  explicit operator bool() const {
    return kind != LocalHardwareOperationKind::None;
  }
  bool operator==(const LocalHardwareOperation &other) const {
    return kind == other.kind && rows == other.rows &&
           projection == other.projection;
  }
  bool operator<(const LocalHardwareOperation &other) const {
    return std::tie(kind, rows, projection) <
           std::tie(other.kind, other.rows, other.projection);
  }
};

struct LocalImplementation {
  LocalPrimitiveKind primitive = LocalPrimitiveKind::None;
  LocalHardwareOperation operation;
  CorePhysicalMapping mapping;
  LocalMicrokernelSchedule schedule;
  llvm::SmallVector<RVVVectorShape, 4> valueShapes;
  unsigned entryWidth = 0;

  explicit operator bool() const {
    return primitive != LocalPrimitiveKind::None && mapping;
  }
  bool operator==(const LocalImplementation &other) const {
    return primitive == other.primitive && operation == other.operation &&
           mapping == other.mapping && schedule == other.schedule &&
           valueShapes == other.valueShapes && entryWidth == other.entryWidth;
  }
  bool operator<(const LocalImplementation &other) const {
    if (primitive != other.primitive)
      return primitive < other.primitive;
    if (!(operation == other.operation))
      return operation < other.operation;
    if (!(mapping == other.mapping))
      return mapping < other.mapping;
    if (!(schedule == other.schedule))
      return schedule < other.schedule;
    if (valueShapes != other.valueShapes)
      return std::lexicographical_compare(valueShapes.begin(), valueShapes.end(),
                                          other.valueShapes.begin(),
                                          other.valueShapes.end());
    if (entryWidth != other.entryWidth)
      return entryWidth < other.entryWidth;
    return false;
  }
};

std::string localImplementationSymbol(
    const LocalImplementation &implementation);

std::optional<LocalImplementation>
selectF32MathLocalImplementation(const CorePhysicalMapping &mapping,
                                 const RISCVTargetProfile &target);

struct I4I8FragmentCandidateFacts {
  CoreMappingProblem mapping;
  bool affine = false;
  unsigned rowsAxis = 0;
  unsigned columnsAxis = 0;
  unsigned reductionAxis = 0;
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
  unsigned inputSEW = 0;
  unsigned resultSEW = 0;
  bool floatingInput = false;
  bool floatingResult = false;
  bool signedInput = false;
  bool preservesLanePositions = false;
  bool maskedInput = false;
  bool requiresF32Math = false;
  LaneRelation coordinateRelation = LaneRelation::Independent;
  bool inputLaneMapped = false;
  bool resultLaneMapped = false;
  bool resultControlCarried = false;
  bool resultCrossesRegion = false;
};

struct SelectedVLAStatePhysical {
  VLAStateCarryRepresentation carry = VLAStateCarryRepresentation::Scalar;
  VLAStateStripUpdate stripUpdate = VLAStateStripUpdate::AddReduction;
  VLAStateFinalize finalize = VLAStateFinalize::Direct;
  bool wholeVLALifetime = true;
  RVVVectorShape inputShape;
  RVVVectorShape carryShape;
  RVVVectorShape seedShape;
  PhysicalMemoryMode coordinateMode = PhysicalMemoryMode::UnitStride;
  bool requiresF32Math = false;
};

llvm::SmallVector<SelectedVLAStatePhysical, 2>
enumerateVLAStatePhysical(const VLAStateCandidateFacts &facts,
                          const RISCVTargetProfile &target);

struct VLAIndexedMemoryFact {
  unsigned operationOrdinal = 0;
  unsigned elementSEW = 0;
  unsigned offsetSEW = 0;
};

struct VLASegmentMemoryFact {
  unsigned operationOrdinal = 0;
  bool write = false;
  unsigned fields = 0;
  unsigned elementSEW = 0;
};

struct VLAStateCandidateSet {
  unsigned operationOrdinal = 0;
  llvm::SmallVector<SelectedVLAStatePhysical, 2> candidates;
};

struct VLALocalPrimitiveResourceFact {
  unsigned operationOrdinal = 0;
  PhysicalResourceRequirements requirements;
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
  llvm::SmallVector<unsigned, 2> lookupOperationOrdinals;
  bool hasIndexVector = false;
  bool hasAffinePredicate = false;
  llvm::SmallVector<unsigned, 2> narrowOperationOrdinals;
  RVVVectorShape requiredDataShape;
  llvm::SmallVector<unsigned> requiredLMULs;
  llvm::SmallVector<unsigned> accessElementSEWs;
  llvm::SmallVector<VLAIndexedMemoryFact> indexedMemory;
  llvm::SmallVector<VLASegmentMemoryFact> segmentMemory;
  llvm::SmallVector<VLAValueLifetimeSnapshot> lifetimes;
  llvm::SmallVector<VLAStateCandidateSet, 4> stateCandidates;
  llvm::SmallVector<VLALocalPrimitiveResourceFact, 2>
      localPrimitiveRequirements;
};

struct VLAElementPhysicalShape {
  unsigned sew = 0;
  RVVVectorShape shape;
};

struct SelectedVLAEntityPhysical {
  CorePhysicalMapping mapping;
  llvm::SmallVector<VLAElementPhysicalShape, 4> elementShapes;
  llvm::SmallVector<SelectedVLAStatePhysical, 4> states;
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

enum class AxisMappedMultiplicity {
  Fixed,
  RegisterFactor,
  LaneCapacity,
  LogicalChunk,
};

struct AxisMappedLiveValue {
  PhysicalLiveClass liveClass = PhysicalLiveClass::Value;
  RVVVectorShape shape;
  AxisMappedMultiplicity multiplicity = AxisMappedMultiplicity::Fixed;
  std::optional<unsigned> axis;
  unsigned factor = 1;
  unsigned chunk = 1;
};

struct AxisMappedResourceFacts {
  const CorePhysicalMapping *mapping = nullptr;
  llvm::SmallVector<AxisMappedLiveValue> values;
  unsigned predicateGroups = 0;
};

std::optional<PhysicalResourceBudget>
calculateAxisMappedResources(const AxisMappedResourceFacts &facts,
                             const RISCVTargetProfile &target);

struct TernaryI8DotCandidateFacts {
  CoreMappingProblem mapping;
  LocalPrimitiveKind primitive = LocalPrimitiveKind::None;
  unsigned primaryExtent = 0;
  unsigned secondaryExtent = 0;
};

struct SelectedTernaryI8DotPhysical {
  LocalImplementation implementation;
  RVVVectorShape primarySourceShape;
  RVVVectorShape secondarySourceShape;
  PhysicalResourceBudget resources;
};

std::optional<SelectedTernaryI8DotPhysical>
selectTernaryI8DotPhysical(const TernaryI8DotCandidateFacts &facts,
                           const RISCVTargetProfile &target);

struct CodebookGatherI8CandidateFacts {
  CoreMappingProblem mapping;
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

struct NibbleCodebookI8CandidateFacts {
  CoreMappingProblem mapping;
  unsigned packedExtent = 0;
  unsigned tableExtent = 0;
};

std::optional<SelectedNibbleCodebookI8Physical>
selectNibbleCodebookI8Physical(const NibbleCodebookI8CandidateFacts &facts,
                               const RISCVTargetProfile &target);

struct QuantI8DotCandidateFacts {
  CoreMappingProblem mapping;
  LocalPrimitiveKind primitive = LocalPrimitiveKind::None;
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

struct E2M1E8M0I8CandidateFacts {
  CoreMappingProblem mapping;
  unsigned packedExtent = 0;
};

std::optional<SelectedE2M1E8M0I8Physical>
selectE2M1E8M0I8Physical(const E2M1E8M0I8CandidateFacts &facts,
                         const RISCVTargetProfile &target);

struct GroupedAffineI4I8CandidateFacts {
  CoreMappingProblem mapping;
  unsigned scaleMinExtent = 0;
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

struct LocalOperandDependenceFacts {
  PhysicalMemoryMode memoryMode = PhysicalMemoryMode::UnitStride;
  bool advancesIteration = false;
  bool feedsPrimitive = false;
  bool addressDependsOnAccumulator = false;
  bool predicateDependsOnAccumulator = false;
  unsigned consumerCount = 0;
  bool crossesControl = false;
};

struct LocalScheduleDependenceFacts {
  llvm::SmallVector<LocalOperandDependenceFacts, 4> operands;
  unsigned resultConsumerCount = 0;
  bool resultCrossesControl = false;
};

std::optional<PhysicalResourceBudget>
calculateLocalMicrokernelResources(const LocalMicrokernelSchedule &schedule,
                                   RVVVectorShape inputShape,
                                   RVVVectorShape accumulatorShape,
                                   unsigned predicateGroups,
                                   unsigned stateGroups,
                                   unsigned handoffGroups,
                                   const RISCVTargetProfile &target);

struct F32DotCandidateFacts {
  CoreMappingProblem mapping;
  llvm::SmallVector<unsigned, 3> lhsAxes;
  llvm::SmallVector<unsigned, 3> rhsAxes;
  unsigned reductionAxis = 0;
  std::optional<uint64_t> reductionExtent;
  unsigned unitStrideOperands = 0;
  unsigned stridedOperands = 0;
  unsigned indexedOperands = 0;
  unsigned predicateGroups = 0;
  unsigned stateGroups = 0;
  unsigned handoffGroups = 0;
  bool reductionPredicate = false;
  bool materializedInit = false;
  LocalScheduleDependenceFacts schedule;
};

struct SelectedF32DotPhysical {
  CorePhysicalMapping mapping;
  LocalMicrokernelSchedule schedule;
  PhysicalResourceRequirements requirements;
  PhysicalResourceBudget resources;
};

std::optional<SelectedF32DotPhysical>
selectF32DotPhysicalConfig(const F32DotCandidateFacts &facts,
                           const RISCVTargetProfile &target,
                           const RISCVBackendConfig &config);

struct F16MatmulCandidateFacts {
  CoreMappingProblem mapping;
  unsigned lhsFreeAxis = 0;
  unsigned rhsFreeAxis = 0;
  unsigned reductionAxis = 0;
  bool nLaneStrided = false;
  LocalScheduleDependenceFacts schedule;
};

struct SelectedF16MatmulPhysical {
  CorePhysicalMapping mapping;
  LocalMicrokernelSchedule schedule;
  RVVVectorShape accumulatorShape;
  PhysicalResourceBudget resources;
};

std::optional<SelectedF16MatmulPhysical>
selectF16MatmulPhysicalConfig(const F16MatmulCandidateFacts &facts,
                              const RISCVTargetProfile &target,
                              const RISCVBackendConfig &config);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVPHYSICALPLANNING_H
