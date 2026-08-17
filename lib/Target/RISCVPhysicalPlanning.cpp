#include "RISCVPhysicalPlanning.h"

#include "RISCVKernelFacts.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <tuple>

namespace weft::riscv_internal {

static unsigned localLaneFactor(const LocalImplementation &implementation) {
  for (const PhysicalAxisDecomposition &axis : implementation.mapping.axes)
    if (axis.laneFactor > 1)
      return axis.laneFactor * axis.registerFactor;
  return 0;
}

static unsigned localRegisterFactor(const LocalImplementation &implementation,
                                    unsigned axis) {
  const PhysicalAxisDecomposition *mapping =
      findAxisMapping(implementation.mapping, axis);
  return mapping ? mapping->registerFactor : 1;
}

static unsigned localFragmentFactor(const LocalImplementation &implementation,
                                    unsigned axis) {
  const PhysicalAxisDecomposition *mapping =
      findAxisMapping(implementation.mapping, axis);
  return mapping ? mapping->fragmentFactor : 1;
}

static unsigned localSequentialFactor(const LocalImplementation &implementation,
                                      unsigned axis) {
  const PhysicalAxisDecomposition *mapping =
      findAxisMapping(implementation.mapping, axis);
  return mapping ? mapping->sequentialFactor : 0;
}

static LocalImplementation makeRVVLocalImplementation(
    LocalPrimitiveKind primitive, CoreInstructionKind instruction,
    unsigned laneAxis, unsigned laneFactor, RVVVectorShape primaryShape,
    RVVVectorShape secondaryShape = {}, unsigned rowRegisterFactor = 1,
    unsigned reductionExtent = 0, unsigned entryWidth = 0,
    const RISCVTargetProfile *target = nullptr) {
  LocalImplementation implementation;
  implementation.primitive = primitive;
  implementation.entryWidth = entryWidth;
  implementation.valueShapes.push_back(primaryShape);
  if (secondaryShape)
    implementation.valueShapes.push_back(secondaryShape);
  if (!target)
    return implementation;
  CoreMappingProblem problem;
  llvm::SmallVector<unsigned, 4> laneRegisterFactors;
  for (unsigned factor : {1u, 2u, 4u, 8u})
    if (laneFactor % factor == 0)
      laneRegisterFactors.push_back(factor);
  problem.axes = {
      LogicalAxisConstraint{kCoreAxisM, LogicalAxisRole::Free,
                            rowRegisterFactor, false, false, false,
                            {rowRegisterFactor}},
      LogicalAxisConstraint{kCoreAxisN, LogicalAxisRole::Free,
                            laneAxis == kCoreAxisN
                                ? std::optional<uint64_t>(laneFactor)
                                : std::optional<uint64_t>(1),
                            false, laneAxis == kCoreAxisN,
                            laneAxis == kCoreAxisN,
                            laneAxis == kCoreAxisN ? laneRegisterFactors
                                                   : llvm::SmallVector<unsigned, 4>{1}},
      LogicalAxisConstraint{
          kCoreAxisK, LogicalAxisRole::Reduction,
          reductionExtent == 0 ? std::optional<uint64_t>(laneFactor)
                               : std::optional<uint64_t>(reductionExtent),
          false, laneAxis == kCoreAxisK, laneAxis == kCoreAxisK,
          laneAxis == kCoreAxisK ? laneRegisterFactors
                                 : llvm::SmallVector<unsigned, 4>{1}}};
  for (LogicalAxisConstraint &axis : problem.axes)
    if (axis.id == laneAxis)
      axis.laneFactorLimit = laneFactor;
  problem.laneSEW = primaryShape.sew;
  problem.laneInstruction = instruction;
  problem.laneShapeCandidates = {primaryShape};
  llvm::SmallVector<CorePhysicalMapping> mappings =
      enumerateCorePhysicalMappings(problem, *target);
  auto selected = llvm::find_if(mappings, [&](const CorePhysicalMapping &mapping) {
    const PhysicalAxisDecomposition *axis = findAxisMapping(mapping, laneAxis);
    return axis && axis->laneFactor * axis->registerFactor == laneFactor;
  });
  if (selected != mappings.end())
    implementation.mapping = std::move(*selected);
  return implementation;
}

static bool validateLocalInstructionMapping(
    const LocalImplementation &implementation) {
  if (!implementation || implementation.valueShapes.empty())
    return false;
  const RVVVectorShape primary = implementation.valueShapes.front();
  const RVVVectorShape secondary = implementation.valueShapes.size() > 1
                                       ? implementation.valueShapes[1]
                                       : RVVVectorShape{};
  const unsigned lanes = localLaneFactor(implementation);
  const unsigned rows =
      implementation.mapping.instruction == CoreInstructionKind::SpacemitIME1MMA
          ? localFragmentFactor(implementation, kCoreAxisM)
          : localRegisterFactor(implementation, kCoreAxisM);
  const bool rvvDot = implementation.mapping.instruction ==
                      CoreInstructionKind::RVVWideningIntegerDot;
  const bool ime = implementation.mapping.instruction ==
                   CoreInstructionKind::SpacemitIME1MMA;
  const bool strip = localSequentialFactor(implementation, kCoreAxisK) > 1;
  switch (implementation.primitive) {
  case LocalPrimitiveKind::None:
    return false;
  case LocalPrimitiveKind::F32Math:
    return implementation.mapping.instruction ==
               CoreInstructionKind::RVVElementwise &&
           primary == kRVVE32M2;
  case LocalPrimitiveKind::SymmetricI4I8:
  case LocalPrimitiveKind::AffineI4I8:
    return (rvvDot || ime) && rows != 0 && (rows == 1 || rows == 4) &&
           ((rvvDot && lanes == 16) ||
            (ime && localFragmentFactor(implementation, kCoreAxisN) == 16 &&
             localFragmentFactor(implementation, kCoreAxisK) == 32)) &&
           primary == kRVVE8M1 && secondary == kRVVE32M4;
  case LocalPrimitiveKind::GroupedAffineI4I8:
    return rvvDot && (lanes == 16 || lanes == 32) &&
           primary == kRVVE8M1 && secondary == kRVVE16M2;
  case LocalPrimitiveKind::E2M1E8M0I8:
    return rvvDot &&
           ((lanes == 32 && primary == RVVVectorShape{8, 4} &&
             secondary == RVVVectorShape{8, 4}) ||
            (lanes == 32 && primary == kRVVE8M1 &&
             secondary == RVVVectorShape{8, 16}) ||
            (strip && primary == kRVVE8M1 && !secondary));
  case LocalPrimitiveKind::PackedI4I8:
  case LocalPrimitiveKind::PackedI5I8:
  case LocalPrimitiveKind::Base3TernaryI8:
  case LocalPrimitiveKind::PackedI2TernaryI8:
  case LocalPrimitiveKind::NibbleCodebookI8:
    return rvvDot && lanes == 32 &&
           (primary == RVVVectorShape{8, 8} ||
            primary == RVVVectorShape{8, 16});
  case LocalPrimitiveKind::PackedI3GroupedI8:
    return rvvDot && (lanes == 32 || lanes == 64) &&
           primary == RVVVectorShape{8, 16};
  case LocalPrimitiveKind::SignedCodebook8I8:
  case LocalPrimitiveKind::PackedU9U7CodebookI8:
  case LocalPrimitiveKind::PackedU11GridDeltaI8:
    return implementation.mapping.instruction ==
               CoreInstructionKind::RVVIndexedGather &&
           lanes == 32 && implementation.entryWidth == 8 &&
           (primary == RVVVectorShape{8, 8} ||
            primary == RVVVectorShape{8, 16});
  case LocalPrimitiveKind::SignedCodebook4I8:
    return implementation.mapping.instruction ==
               CoreInstructionKind::RVVIndexedGather &&
           lanes == 32 && implementation.entryWidth == 4 &&
           (primary == RVVVectorShape{8, 8} ||
            primary == RVVVectorShape{8, 16});
  case LocalPrimitiveKind::IQ2SI8:
  case LocalPrimitiveKind::IQ1MI8:
  case LocalPrimitiveKind::Q6KI8:
    return rvvDot &&
           (((lanes == 32 || lanes == 64) &&
             primary == RVVVectorShape{8, 16}) ||
            (strip && lanes == 16 && primary == RVVVectorShape{8, 16}));
  case LocalPrimitiveKind::IQ3SI8:
    return rvvDot &&
           ((lanes == 64 && primary == RVVVectorShape{8, 16}) ||
            (strip && lanes == 16 && primary == RVVVectorShape{8, 16}));
  }
  return false;
}

bool fitsPrivateStorage(int64_t elements, unsigned elementBytes,
                        const RISCVTargetProfile &target) {
  return elements > 0 && elementBytes > 0 &&
         elements <= target.maxPrivateStackBytes /
                         static_cast<int64_t>(elementBytes);
}

std::optional<int64_t>
extendPrivateStorageElementCount(int64_t currentElements, int64_t extent,
                                 unsigned elementBytes,
                                 const RISCVTargetProfile &target) {
  if (currentElements <= 0 || extent <= 0 || elementBytes == 0 ||
      currentElements > target.maxPrivateStackBytes /
                            static_cast<int64_t>(elementBytes) / extent)
    return std::nullopt;
  return currentElements * extent;
}

std::optional<LoadF16LEDecision>
selectLoadF16LEPhysical(bool knownAligned,
                        const RISCVTargetProfile &target) {
  if (!target.littleEndian)
    return std::nullopt;
  return LoadF16LEDecision{
      knownAligned ? LoadF16LERealization::ScalarAlignedHalf
                   : LoadF16LERealization::ScalarBytes};
}

std::optional<SelectedBlockDecodePhysical>
selectBlockDecodePhysical(const BlockDecodeCandidateFacts &facts,
                          const RISCVTargetProfile &target) {
  if (!target.hasRVV || target.vlenBits < 128 || facts.codeExtent != 16 ||
      facts.tableExtent != 16 ||
      !target.supportsVectorShape(kRVVE8M1.sew,
                                  kRVVE8M1.lmulEighths))
    return std::nullopt;
  SelectedBlockDecodePhysical selected;
  selected.codeShape = kRVVE8M1;
  selected.tableShape = kRVVE8M1;
  selected.resultShape = kRVVE8M1;
  selected.tableExtent = facts.tableExtent;
  selected.resources.architecturalGroups = target.vectorRegisters;
  selected.resources.valueGroups = 3;
  selected.resources.primitiveGroups = 3;
  selected.resources.peakGroups = 4;
  if (selected.resources.peakGroups >=
      static_cast<unsigned>(target.vectorRegisters))
    return std::nullopt;
  return selected;
}

std::optional<SelectedBlockOperationPhysical>
selectBlockOperationPhysical(const BlockOperationCandidateFacts &facts,
                             const RISCVTargetProfile &target) {
  if (!target.hasRVV || !facts.byteShape ||
      !target.supportsVectorShape(facts.byteShape.sew,
                                  facts.byteShape.lmulEighths))
    return std::nullopt;
  auto sameLanes = [&](unsigned sew) {
    return rvvShapeForSameLanes(facts.byteShape, sew, target);
  };
  SelectedBlockOperationPhysical selected;
  switch (facts.operation) {
  case BlockOperationSemantic::Axis:
    selected.resultShape = sameLanes(16).value_or(RVVVectorShape{});
    break;
  case BlockOperationSemantic::Load:
    if (facts.resultType == BlockPhysicalType::U8)
      selected.resultShape = facts.byteShape;
    break;
  case BlockOperationSemantic::Bitcast:
    if (facts.targetType == BlockPhysicalType::I8)
      selected.resultShape = facts.byteShape;
    else if (facts.targetType == BlockPhysicalType::I16)
      selected.resultShape = sameLanes(16).value_or(RVVVectorShape{});
    break;
  case BlockOperationSemantic::Compare:
    selected.resultShape = facts.lhsShape ? facts.lhsShape : facts.rhsShape;
    selected.maskRatio = rvvMaskRatio(selected.resultShape).value_or(0);
    if (!selected.resultShape || selected.maskRatio == 0)
      return std::nullopt;
    break;
  case BlockOperationSemantic::Cast: {
    RVVVectorShape sourceShape = facts.sourceShape;
    if (!sourceShape && (facts.sourceType == BlockPhysicalType::U8 ||
                         facts.sourceType == BlockPhysicalType::I8))
      sourceShape = facts.byteShape;
    if (!sourceShape && facts.sourceType == BlockPhysicalType::Index)
      sourceShape = sameLanes(16).value_or(RVVVectorShape{});
    unsigned targetSEW =
        facts.targetType == BlockPhysicalType::U8
            ? 8
            : (facts.targetType == BlockPhysicalType::Index ||
               facts.targetType == BlockPhysicalType::U16 ||
               facts.targetType == BlockPhysicalType::I16)
                  ? 16
                  : (facts.targetType == BlockPhysicalType::I32 ||
                     facts.targetType == BlockPhysicalType::F32)
                        ? 32
                        : 0;
    if (targetSEW != 0)
      selected.resultShape =
          rvvShapeForSameLanes(sourceShape, targetSEW, target)
              .value_or(RVVVectorShape{});
    if (targetSEW != 0 && !selected.resultShape)
      return std::nullopt;
    if (facts.targetType == BlockPhysicalType::I32 &&
        (facts.sourceType == BlockPhysicalType::U8 ||
         facts.sourceType == BlockPhysicalType::I8)) {
      selected.realization = BlockValueRealization::DeferredI32Widen;
      selected.temporaryShape = selected.resultShape;
    } else if (facts.targetType == BlockPhysicalType::I32 &&
               facts.sourceType == BlockPhysicalType::U16) {
      selected.temporaryShape = selected.resultShape;
    } else if (facts.targetType == BlockPhysicalType::F32) {
      selected.temporaryShape =
          facts.sourceType == BlockPhysicalType::U16
              ? sourceShape
              : selected.resultShape == kRVVE32M1
                    ? selected.resultShape
                    : rvvShapeForSameLanes(sourceShape, 16, target)
                          .value_or(RVVVectorShape{});
      if (!selected.temporaryShape)
        return std::nullopt;
    }
    break;
  }
  case BlockOperationSemantic::Binary:
    selected.unsignedMaximum = facts.resultUnsignedMaximum;
    if (facts.resultType == BlockPhysicalType::U8 &&
        facts.binary == BlockBinarySemantic::ShiftRight && facts.shiftAmount &&
        *facts.shiftAmount < 8)
      selected.unsignedMaximum =
          facts.lhsUnsignedMaximum.value_or(255) >> *facts.shiftAmount;
    if (facts.resultType == BlockPhysicalType::Index)
      selected.resultShape = sameLanes(16).value_or(RVVVectorShape{});
    else if (facts.resultType == BlockPhysicalType::U8) {
      selected.resultShape = facts.byteShape;
      if (facts.byteShape == kRVVE8MF4 &&
          (facts.binary == BlockBinarySemantic::And ||
           facts.binary == BlockBinarySemantic::ShiftRight))
        selected.realization = BlockValueRealization::DeferredPackedTransform;
    } else if (facts.resultType == BlockPhysicalType::U16) {
      selected.resultShape = sameLanes(16).value_or(RVVVectorShape{});
    } else if (facts.resultType == BlockPhysicalType::I32) {
      bool lhsSmallU8 = facts.lhsNarrowU8 &&
                        facts.lhsUnsignedMaximum.value_or(256) <= 15;
      bool rhsSmallU8 = facts.rhsNarrowU8 &&
                        facts.rhsUnsignedMaximum.value_or(256) <= 15;
      bool wideningProduct =
          facts.binary == BlockBinarySemantic::Multiply &&
          ((lhsSmallU8 && facts.rhsNarrowI8) ||
           (rhsSmallU8 && facts.lhsNarrowI8));
      if (wideningProduct) {
        selected.realization = BlockValueRealization::WideningI8Product;
        selected.resultShape = sameLanes(16).value_or(RVVVectorShape{});
        selected.temporaryShape = sameLanes(8).value_or(RVVVectorShape{});
      } else {
        selected.resultShape = sameLanes(32).value_or(RVVVectorShape{});
      }
    } else if (facts.resultType == BlockPhysicalType::F32) {
      selected.resultShape =
          facts.lhsShape && facts.lhsShape.sew == 32 ? facts.lhsShape
          : facts.rhsShape && facts.rhsShape.sew == 32
              ? facts.rhsShape
              : RVVVectorShape{};
    }
    break;
  case BlockOperationSemantic::Select:
    if (facts.resultType == BlockPhysicalType::U8)
      selected.resultShape = facts.byteShape;
    else if (facts.resultType == BlockPhysicalType::Index)
      selected.resultShape = sameLanes(16).value_or(RVVVectorShape{});
    else if (facts.resultType == BlockPhysicalType::F32)
      selected.resultShape = facts.trueShape ? facts.trueShape : facts.falseShape;
    break;
  case BlockOperationSemantic::Other:
    break;
  }
  return selected;
}

std::optional<SelectedBlockStorePhysical>
selectBlockStorePhysical(const BlockStoreCandidateFacts &facts,
                         const RISCVTargetProfile &target) {
  llvm::SmallVector<BlockStorePhysicalDecision> candidates;
  if (facts.supportsMicroStripOperations &&
      target.supportsVectorShape(kRVVE8MF4.sew,
                                 kRVVE8MF4.lmulEighths))
    candidates.push_back(BlockStorePhysicalDecision{
        BlockStoreRealization::RVVMicroStrips, kRVVE8MF4, 4, false, {}});
  if (facts.supportsStandardOperations &&
      target.supportsVectorShape(kRVVE8M1.sew, kRVVE8M1.lmulEighths)) {
    candidates.push_back(BlockStorePhysicalDecision{
        BlockStoreRealization::RVVFixedStrips, kRVVE8M1, 16, false, {}});
    if (!facts.needsLaneVector ||
        target.supportsVectorShape(kRVVE16M2.sew,
                                   kRVVE16M2.lmulEighths))
      candidates.push_back(BlockStorePhysicalDecision{
          BlockStoreRealization::RVVDynamicStrips, kRVVE8M1, 16,
          facts.needsLaneVector,
          facts.needsLaneVector ? kRVVE16M2 : RVVVectorShape{}});
  }
  for (const BlockStorePhysicalDecision &candidate : candidates) {
    if (candidate.realization == BlockStoreRealization::RVVMicroStrips &&
        (facts.needsLaneVector || facts.extent != 32))
      continue;
    if (candidate.realization == BlockStoreRealization::RVVFixedStrips &&
        (facts.needsLaneVector || facts.extent != 32))
      continue;
    unsigned dataGroups = candidate.byteShape == kRVVE8MF4 ? 4 : 8;
    unsigned laneGroups = rvvRegisterGroups(candidate.laneShape);
    PhysicalResourceBudget resources;
    resources.architecturalGroups = target.vectorRegisters;
    resources.valueGroups = dataGroups;
    resources.memoryGroups = laneGroups;
    resources.primitiveGroups = dataGroups + laneGroups;
    resources.peakGroups = resources.primitiveGroups + 1;
    if (resources.peakGroups >=
        static_cast<unsigned>(target.vectorRegisters))
      continue;
    return SelectedBlockStorePhysical{candidate, resources};
  }
  return std::nullopt;
}

std::optional<SelectedBlockReducePhysical>
selectBlockReducePhysical(const BlockReduceCandidateFacts &facts,
                          const RISCVTargetProfile &target) {
  if (!facts.supportsStandardOperations || !facts.inputShape ||
      !target.supportsVectorShape(kRVVE8M1.sew, kRVVE8M1.lmulEighths))
    return std::nullopt;
  BlockReducePhysicalDecision fixed{
      BlockReduceRealization::RVVFixedStrips, 16, false, {}};
  BlockReducePhysicalDecision dynamic{
      BlockReduceRealization::RVVDynamicStrips, 16, facts.needsLaneVector,
      facts.needsLaneVector ? kRVVE16M2 : RVVVectorShape{}};
  llvm::SmallVector<BlockReducePhysicalDecision> candidates;
  if (facts.extent == 16 && facts.inputShape.sew == 16) {
    candidates.push_back(dynamic);
    candidates.push_back(fixed);
  } else {
    candidates.push_back(fixed);
    candidates.push_back(dynamic);
  }
  for (const BlockReducePhysicalDecision &candidate : candidates) {
    if (candidate.needsLaneVector &&
        !target.supportsVectorShape(candidate.laneShape.sew,
                                    candidate.laneShape.lmulEighths))
      continue;
    if (candidate.realization == BlockReduceRealization::RVVFixedStrips &&
        (facts.needsLaneVector || (facts.extent != 16 && facts.extent != 32)))
      continue;

    SelectedBlockReducePhysical selected;
    selected.decision = candidate;
    selected.inputShape = facts.inputShape;
    selected.combinedShape = facts.inputShape;
    if (candidate.realization == BlockReduceRealization::RVVFixedStrips &&
        facts.inputShape.sew == 16) {
      std::optional<RVVVectorShape> combined =
          rvvShapeForSameLanes(facts.inputShape, 32, target);
      if (!combined)
        continue;
      selected.combinedShape = *combined;
      selected.wideningCombine = true;
    }
    selected.seedShape = RVVVectorShape{selected.combinedShape.sew, 8};
    if (!target.supportsVectorShape(selected.seedShape.sew,
                                    selected.seedShape.lmulEighths))
      continue;

    unsigned liveGroups = 8 + rvvRegisterGroups(candidate.laneShape) + 1;
    selected.resources.architecturalGroups = target.vectorRegisters;
    selected.resources.valueGroups = 8;
    selected.resources.memoryGroups = rvvRegisterGroups(candidate.laneShape);
    selected.resources.primitiveGroups = liveGroups;
    selected.resources.peakGroups = liveGroups + 1;
    if (selected.resources.peakGroups >=
        static_cast<unsigned>(target.vectorRegisters))
      continue;
    return selected;
  }
  return std::nullopt;
}

std::optional<SelectedMaterializedBlockStorePhysical>
selectMaterializedBlockStorePhysical(
    const MaterializedBlockStoreCandidateFacts &facts,
    const RISCVTargetProfile &target) {
  SelectedMaterializedBlockStorePhysical selected;
  selected.resources.architecturalGroups = target.vectorRegisters;
  selected.rows = facts.rows;
  selected.columns = facts.columns;
  if (facts.rank == 1 && facts.columns == 16 && facts.allActive &&
      facts.unitStride) {
    std::optional<RVVVectorShape> shape =
        rvvShapeForSemanticLanes(32, 16, target);
    if (!shape)
      return std::nullopt;
    selected.realization =
        MaterializedBlockStoreRealization::RVVContiguousRankOne;
    selected.vectorShape = *shape;
    selected.resources.valueGroups = rvvRegisterGroups(*shape);
    selected.resources.memoryGroups = selected.resources.valueGroups;
    selected.resources.peakGroups = selected.resources.valueGroups + 1;
  } else if (facts.rank == 1 && facts.columns > 0 &&
             facts.prefixPredicated) {
    selected.realization = MaterializedBlockStoreRealization::ScalarRankOne;
  } else if (facts.rank == 2 && facts.rows > 0 && facts.columns > 0 &&
             facts.prefixPredicated) {
    selected.realization = MaterializedBlockStoreRealization::ScalarRankTwo;
  } else {
    return std::nullopt;
  }
  if (selected.resources.peakGroups >
      static_cast<unsigned>(target.vectorRegisters))
    return std::nullopt;
  return selected;
}

std::optional<SelectedSortIndicesPhysical>
selectSortIndicesPhysical(const SortIndicesCandidateFacts &facts,
                          const RISCVTargetProfile &target) {
  if (!target.littleEndian || target.xlen != 64 || !target.hasRVV ||
      (facts.configuredRadixBits != 0 && facts.configuredRadixBits != 8 &&
       facts.configuredRadixBits != 11))
    return std::nullopt;
  SelectedSortIndicesPhysical selected;
  selected.descending = facts.descending;
  selected.radixBits = facts.configuredRadixBits == 0
                           ? 8
                           : static_cast<unsigned>(facts.configuredRadixBits);
  selected.passes = (32 + selected.radixBits - 1) / selected.radixBits;
  selected.privateElements = 2 * (int64_t{1} << selected.radixBits);
  selected.privateAlignment = target.xlen / 8;
  int64_t privateBytes = selected.privateElements * target.xlen / 8;
  if (privateBytes > target.maxPrivateStackBytes)
    return std::nullopt;
  selected.resources.architecturalGroups = target.vectorRegisters;
  return selected;
}

std::optional<SelectedVLAAccessPhysical>
selectVLAAccessPhysical(const VLAAccessCandidateFacts &facts,
                        const RISCVTargetProfile &target) {
  if (!target.hasRVV || facts.relation == LaneRelation::Independent ||
      facts.relation == LaneRelation::NonAffine ||
      (facts.write && facts.relation == LaneRelation::Indexed))
    return std::nullopt;
  SelectedVLAAccessPhysical selected;
  switch (facts.relation) {
  case LaneRelation::UnitStride:
    selected.memoryMode = VLAMemoryMode::UnitStride;
    break;
  case LaneRelation::Strided:
    selected.memoryMode = VLAMemoryMode::Strided;
    break;
  case LaneRelation::Indexed:
    if (!target.hasIndexedMemory)
      return std::nullopt;
    selected.memoryMode = VLAMemoryMode::Indexed;
    selected.indexedSEW =
        facts.indexedOffsetFromU32 ? 32 : static_cast<unsigned>(target.xlen);
    if (selected.indexedSEW != 32 && selected.indexedSEW != 64)
      return std::nullopt;
    break;
  case LaneRelation::Independent:
  case LaneRelation::NonAffine:
    return std::nullopt;
  }
  if (facts.predicateAllActive)
    selected.activityMode = VLAActivityMode::AllActive;
  else if (facts.predicateVector)
    selected.activityMode = VLAActivityMode::PredicateMask;
  else if (facts.predicateScalar)
    selected.activityMode = VLAActivityMode::ScalarPredicate;
  else
    return std::nullopt;
  if (facts.carriesLogicalValidity) {
    if (selected.activityMode == VLAActivityMode::AllActive)
      return std::nullopt;
    selected.inactiveLane =
        VLAInactiveLaneRealization::ZeroCarrierWithLogicalValidity;
  }
  return selected;
}

std::optional<SelectedVLASegment2Physical>
selectVLASegment2Physical(const VLASegment2CandidateFacts &facts,
                          const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasSegmentMemory || facts.fields != 2 ||
      facts.elementSEW != 32)
    return std::nullopt;
  return SelectedVLASegment2Physical{
      facts.load ? VLASegment2AccessKind::Load
                 : VLASegment2AccessKind::Store,
      facts.load, 2, facts.fields, facts.elementSEW};
}

std::optional<SelectedVLAPredicatePhysical>
selectVLAPredicatePhysical(const VLAPredicateCandidateFacts &facts,
                           const RISCVTargetProfile &target) {
  if (!target.hasRVV || (!facts.affineIndex && !facts.vectorScalar))
    return std::nullopt;
  SelectedVLAPredicatePhysical selected;
  if (facts.affineIndex) {
    if (facts.coordinateRelation != LaneRelation::UnitStride &&
        facts.coordinateRelation != LaneRelation::Strided)
      return std::nullopt;
    selected.realization = VLAPredicateRealization::RVVAffineIndexScalar;
    selected.coordinateMode =
        facts.coordinateRelation == LaneRelation::UnitStride
            ? VLAMemoryMode::UnitStride
            : VLAMemoryMode::Strided;
    return selected;
  }
  if (facts.vectorSEW != 8 && facts.vectorSEW != 32 &&
      facts.vectorSEW != static_cast<unsigned>(target.xlen))
    return std::nullopt;
  selected.realization = VLAPredicateRealization::RVVVectorScalar;
  selected.vectorSEW = facts.vectorSEW;
  return selected;
}

std::optional<VLAIndexBinaryRealization>
selectVLAIndexBinaryPhysical(const VLAIndexBinaryCandidateFacts &facts) {
  if ((!facts.lhsVector && !facts.rhsVector) || facts.scalarLeftDivision)
    return std::nullopt;
  return facts.lhsVector && facts.rhsVector
             ? VLAIndexBinaryRealization::RVVUnsignedVectorVector
             : VLAIndexBinaryRealization::RVVUnsignedVectorScalar;
}

std::optional<SelectedVLACastPhysical>
selectVLACastPhysical(const VLACastCandidateFacts &facts,
                      const RISCVTargetProfile &target) {
  if (!target.hasRVV || !facts.dataShape)
    return std::nullopt;
  SelectedVLACastPhysical selected;
  unsigned sourceSEW = 0;
  unsigned resultSEW = 0;
  switch (facts.semantic) {
  case VLACastSemantic::Identity:
    selected.realization = VLACastRealization::RVVIdentity;
    sourceSEW = resultSEW = facts.identitySEW;
    break;
  case VLACastSemantic::F16ToF32:
    selected.realization = VLACastRealization::RVVWidenF16ToF32;
    sourceSEW = 16;
    resultSEW = 32;
    break;
  case VLACastSemantic::F32ToF16:
    selected.realization = VLACastRealization::RVVNarrowF32ToF16;
    sourceSEW = 32;
    resultSEW = 16;
    break;
  case VLACastSemantic::U32ToIndex:
    selected.realization = VLACastRealization::RVVZeroExtendU32ToIndex;
    sourceSEW = resultSEW = 32;
    break;
  case VLACastSemantic::IndexToF32:
    selected.realization = VLACastRealization::RVVIndexToF32;
    sourceSEW = target.xlen;
    resultSEW = 32;
    break;
  }
  selected.sourceShape =
      rvvShapeForSameLanes(facts.dataShape, sourceSEW, target)
          .value_or(RVVVectorShape{});
  selected.resultShape =
      rvvShapeForSameLanes(facts.dataShape, resultSEW, target)
          .value_or(RVVVectorShape{});
  if (!selected.sourceShape || !selected.resultShape)
    return std::nullopt;
  return selected;
}

std::optional<VLAUnaryRealization>
selectVLAUnaryPhysical(VLAUnarySemantic semantic,
                       const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasF ||
      !target.supportsVectorShape(kRVVE32M2.sew,
                                  kRVVE32M2.lmulEighths))
    return std::nullopt;
  switch (semantic) {
  case VLAUnarySemantic::Exp:
    return VLAUnaryRealization::RVVExpPolynomial;
  case VLAUnarySemantic::Tanh:
    return VLAUnaryRealization::RVVTanhViaExp;
  case VLAUnarySemantic::Sin:
    return VLAUnaryRealization::RVVScalarLibmSin;
  case VLAUnarySemantic::Cos:
    return VLAUnaryRealization::RVVScalarLibmCos;
  }
  return std::nullopt;
}

std::optional<SelectedVLAStatePhysical>
selectVLAStatePhysical(const VLAStateCandidateFacts &facts,
                       const RISCVTargetProfile &target,
                       const RISCVBackendConfig &config) {
  if (!target.hasRVV || config.structures.reductionStatePlacement < 0 ||
      config.structures.reductionStatePlacement > 2)
    return std::nullopt;
  SelectedVLAStatePhysical selected;
  switch (facts.semantic) {
  case VLAStateSemantic::F32AddReduction:
    if (!target.hasF)
      return std::nullopt;
    selected.stripUpdate = VLAStateStripUpdate::AddReduction;
    break;
  case VLAStateSemantic::F32MaxReduction:
    if (!target.hasF)
      return std::nullopt;
    selected.stripUpdate = VLAStateStripUpdate::MaxReduction;
    break;
  case VLAStateSemantic::I8AddReductionI32:
    if (!target.hasWideningInteger)
      return std::nullopt;
    selected.stripUpdate = VLAStateStripUpdate::WideningAddReduction;
    break;
  case VLAStateSemantic::InclusiveAddScan:
    if (!target.hasF || facts.relaxedOrder)
      return std::nullopt;
    selected.stripUpdate = VLAStateStripUpdate::InclusiveAddScan;
    break;
  case VLAStateSemantic::SegmentedInclusiveAddScan:
    if (!target.hasF || facts.relaxedOrder)
      return std::nullopt;
    selected.stripUpdate = VLAStateStripUpdate::SegmentedInclusiveAddScan;
    break;
  case VLAStateSemantic::ArgMaxSummary:
    if (!target.hasF)
      return std::nullopt;
    selected.carry = VLAStateCarryRepresentation::ScalarTuple;
    selected.stripUpdate = VLAStateStripUpdate::ArgMaxSummary;
    return selected;
  case VLAStateSemantic::OnlineSoftmaxSummary:
    if (!target.hasF || facts.relaxedOrder)
      return std::nullopt;
    selected.carry = VLAStateCarryRepresentation::ScalarTuple;
    selected.stripUpdate = VLAStateStripUpdate::OnlineSoftmaxSummary;
    return selected;
  }
  const bool f32Reduction =
      facts.semantic == VLAStateSemantic::F32AddReduction ||
      facts.semantic == VLAStateSemantic::F32MaxReduction;
  if (f32Reduction) {
    if (config.structures.reductionStatePlacement == 2 &&
        !facts.relaxedOrder)
      return std::nullopt;
    const bool vectorCarry =
        config.structures.reductionStatePlacement == 2 ||
        (config.structures.reductionStatePlacement == 0 &&
         facts.relaxedOrder &&
         (target.vlenBits < 256 || facts.reductionStateCount > 1));
    if (vectorCarry) {
      selected.carry = VLAStateCarryRepresentation::Vector;
      selected.finalize =
          facts.semantic == VLAStateSemantic::F32AddReduction
              ? VLAStateFinalize::HorizontalAdd
              : VLAStateFinalize::HorizontalMax;
    }
  }
  return selected;
}

std::optional<LocalImplementation>
selectF32MathLocalImplementation(const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasF ||
      !target.supportsVectorShape(kRVVE32M2.sew,
                                  kRVVE32M2.lmulEighths))
    return std::nullopt;
  std::optional<unsigned> lanes = rvvLaneCapacity(kRVVE32M2, target);
  if (!lanes)
    return std::nullopt;
  LocalImplementation implementation = makeRVVLocalImplementation(
      LocalPrimitiveKind::F32Math, CoreInstructionKind::RVVElementwise,
      kCoreAxisN, *lanes, kRVVE32M2, {}, 1, 0, 0, &target);
  if (!validateLocalInstructionMapping(implementation))
    return std::nullopt;
  return implementation;
}

std::optional<SelectedVLALookupPhysical>
selectVLALookupPhysical(const VLALookupCandidateFacts &facts,
                        const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasIndexedMemory || target.vlenBits < 128 ||
      facts.tableExtent != 16 || !facts.tableF32 || !facts.resultF32 ||
      !facts.indicesU8 || !facts.allActive)
    return std::nullopt;
  return SelectedVLALookupPhysical{VLALookupRealization::RVVTableGather,
                                   facts.tableExtent};
}

std::optional<SelectedI4I8FragmentPhysical>
selectI4I8FragmentPhysical(const I4I8FragmentCandidateFacts &facts,
                           const RISCVTargetProfile &target,
                           const RISCVBackendConfig &config) {
  if (!target.littleEndian || (facts.rowTile != 1 && facts.rowTile != 4) ||
      config.structures.i4I8FragmentImplementation < 0 ||
      config.structures.i4I8FragmentImplementation > 2)
    return std::nullopt;
  const bool supportsIME =
      facts.rowTile == 4
          ? target.supportsSpacemitIME1I4I8M4N16K32()
          : target.supportsSpacemitIME1I4I8N16K32();
  const bool supportsRVV =
      target.hasF && target.hasVectorF16 && target.hasWideningInteger &&
      target.hasWideningFloat && target.supportsVLENAtLeast(128) &&
      target.supportsVectorShape(8, 8) &&
      target.supportsVectorShape(16, 16) &&
      target.supportsVectorShape(32, 32);
  CoreMappingProblem problem;
  problem.axes = {
      LogicalAxisConstraint{kCoreAxisM, LogicalAxisRole::Free, facts.rowTile,
                            false, false, false, {facts.rowTile}},
      LogicalAxisConstraint{kCoreAxisN, LogicalAxisRole::Free, 16, false,
                            supportsRVV, false, {1}},
      LogicalAxisConstraint{kCoreAxisK, LogicalAxisRole::Reduction, 32, false,
                            false, false, {1}}};
  problem.laneSEW = 8;
  problem.laneInstruction = CoreInstructionKind::RVVWideningIntegerDot;
  problem.laneLMULCandidates = {1};
  if (supportsIME)
    problem.fragments.push_back(FragmentMappingConstraint{
        CoreInstructionKind::SpacemitIME1MMA,
        {{kCoreAxisM, facts.rowTile}, {kCoreAxisN, 16}, {kCoreAxisK, 32}},
        28});

  struct Candidate {
    SelectedI4I8FragmentPhysical physical;
    unsigned instructionCost = 0;
  };
  llvm::SmallVector<Candidate, 2> candidates;
  for (CorePhysicalMapping mapping :
       enumerateCorePhysicalMappings(problem, target)) {
    const bool ime =
        mapping.instruction == CoreInstructionKind::SpacemitIME1MMA;
    if ((ime && !supportsIME) || (!ime && !supportsRVV))
      continue;
    SelectedI4I8FragmentPhysical selected;
    selected.codeShape = kRVVE8M1;
    selected.activationScaleShape = kRVVE32M1;
    selected.accumulatorShape = kRVVE32M4;
    selected.implementation.primitive =
        facts.affine ? LocalPrimitiveKind::AffineI4I8
                     : LocalPrimitiveKind::SymmetricI4I8;
    selected.implementation.mapping = std::move(mapping);
    selected.implementation.valueShapes = {selected.codeShape,
                                           selected.accumulatorShape};
    if (!validateLocalInstructionMapping(selected.implementation))
      continue;
    const PhysicalAxisDecomposition *mappedRows =
        findAxisMapping(selected.implementation.mapping, kCoreAxisM);
    if (!mappedRows)
      continue;
    const unsigned privateGroups =
        ime ? selected.implementation.mapping.fixedFragmentGroups
            : mappedRows->registerFactor *
                      rvvRegisterGroups(selected.accumulatorShape) +
                  8 + static_cast<unsigned>(facts.affine);
    PhysicalResourceRequirements requirements;
    requirements.live.push_back(
        PhysicalLiveRange{ime ? PhysicalLiveClass::Fragment
                              : PhysicalLiveClass::Temporary,
                          {}, 1, privateGroups});
    std::optional<PhysicalResourceBudget> resources =
        calculatePhysicalResources(requirements, target);
    if (!resources)
      continue;
    selected.resources = *resources;
    candidates.push_back(Candidate{std::move(selected),
                                   ime ? 1u
                                       : facts.rowTile == 4 ? 16u : 4u});
  }
  const int64_t requested = config.structures.i4I8FragmentImplementation;
  if (requested != 0)
    llvm::erase_if(candidates, [&](const Candidate &candidate) {
      const bool ime = candidate.physical.implementation.mapping.instruction ==
                       CoreInstructionKind::SpacemitIME1MMA;
      return requested == 1 ? ime : !ime;
    });
  if (candidates.empty())
    return std::nullopt;
  llvm::sort(candidates, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.instructionCost, lhs.physical.resources.peakGroups) <
           std::tie(rhs.instructionCost, rhs.physical.resources.peakGroups);
  });
  return candidates.front().physical;
}

std::optional<SelectedVLAEntityPhysical>
selectVLAEntityPhysical(const VLAEntityCandidateFacts &facts,
                        const RISCVTargetProfile &target) {
  if ((facts.dataSEW != 16 && facts.dataSEW != 32) || !target.hasRVV ||
      target.vectorRegisters <= 0 || facts.mapping.axes.size() != 1 ||
      facts.mapping.axes.front().id != kCoreAxisVLA ||
      !facts.mapping.axes.front().requireLane)
    return std::nullopt;
  if (facts.requiredDataShape &&
      (facts.requiredDataShape.sew != facts.dataSEW ||
       !target.supportsVectorShape(facts.requiredDataShape.sew,
                                   facts.requiredDataShape.lmulEighths)))
    return std::nullopt;

  CoreMappingProblem problem = facts.mapping;
  problem.laneSEW = facts.dataSEW;
  problem.laneInstruction = CoreInstructionKind::RVVElementwise;
  std::optional<unsigned> requiredLMUL;
  for (unsigned lmul : facts.requiredLMULs) {
    if (lmul == 0 ||
        !target.supportsVectorShape(facts.dataSEW,
                                    static_cast<int>(lmul * 8)) ||
        (requiredLMUL && *requiredLMUL != lmul))
      return std::nullopt;
    requiredLMUL = lmul;
  }
  if (facts.requiredDataShape)
    problem.laneShapeCandidates = {facts.requiredDataShape};
  else if (requiredLMUL)
    problem.laneLMULCandidates = {*requiredLMUL};
  else if (problem.laneShapeCandidates.empty() &&
           problem.laneLMULCandidates.empty())
    problem.laneLMULCandidates =
        integerLMULCandidates(target, facts.dataSEW);

  struct Candidate {
    SelectedVLAEntityPhysical physical;
    unsigned lanePenalty = 0;
    unsigned peakGroups = 0;
    unsigned inverseLanes = 0;
  };
  llvm::SmallVector<Candidate, 8> legal;
  for (CorePhysicalMapping mapping :
       enumerateCorePhysicalMappings(problem, target)) {
    RVVVectorShape dataShape = mapping.laneShape;
    const PhysicalAxisDecomposition *vlaAxis =
        findAxisMapping(mapping, kCoreAxisVLA);
    if (!vlaAxis || vlaAxis->laneFactor == 0 ||
        (facts.requiredDataShape && dataShape != facts.requiredDataShape))
      continue;
    const unsigned mappedLaneFactor = vlaAxis->laneFactor;
    const bool needsIndexShape =
        facts.hasAffinePredicate || facts.hasIndexVector;
    std::optional<RVVVectorShape> indexShape =
        needsIndexShape
            ? rvvShapeForSameLanes(dataShape,
                                   static_cast<unsigned>(target.xlen), target)
            : std::optional<RVVVectorShape>{};
    if (needsIndexShape && !indexShape)
      continue;

    bool elementShapesLegal =
        llvm::all_of(facts.accessElementSEWs, [&](unsigned sew) {
          if (sew == 16 && !target.hasVectorF16)
            return false;
          return sew != 0 && rvvShapeForSameLanes(dataShape, sew, target);
        });
    if (!elementShapesLegal)
      continue;
    bool indexedShapesLegal =
        llvm::all_of(facts.indexedMemory,
                     [&](const VLAIndexedMemoryFact &memory) {
          std::optional<RVVVectorShape> elementShape =
              rvvShapeForSameLanes(dataShape, memory.elementSEW, target);
          std::optional<RVVVectorShape> indexShape =
              rvvShapeForSameLanes(dataShape, memory.offsetSEW, target);
          return elementShape && indexShape &&
                 target.supportsIndexedVectorMemory(
                     elementShape->sew, elementShape->lmulEighths,
                     indexShape->sew, indexShape->lmulEighths);
        });
    if (!indexedShapesLegal)
      continue;
    bool segmentShapesLegal = llvm::all_of(
        facts.segmentMemory, [&](const VLASegmentMemoryFact &memory) {
          std::optional<RVVVectorShape> elementShape =
              rvvShapeForSameLanes(dataShape, memory.elementSEW, target);
          return memory.fields != 0 && elementShape &&
                 target.supportsSegmentVectorMemory(
                     memory.fields, elementShape->sew,
                     elementShape->lmulEighths);
        });
    if (!segmentShapesLegal)
      continue;

    llvm::SmallVector<PhysicalLiveRange, 8> persistent;
    for (const SelectedVLAStatePhysical &state : facts.states)
      if (state.carry == VLAStateCarryRepresentation::Vector &&
          state.wholeVLALifetime)
        persistent.push_back(
            {PhysicalLiveClass::State, dataShape, 1, 0});

    PhysicalResourceBudget aggregate;
    aggregate.architecturalGroups = target.vectorRegisters;
    auto mergeBudget = [&](const PhysicalResourceBudget &budget) {
      aggregate.valueGroups =
          std::max(aggregate.valueGroups, budget.valueGroups);
      aggregate.memoryGroups =
          std::max(aggregate.memoryGroups, budget.memoryGroups);
      aggregate.indexGroups =
          std::max(aggregate.indexGroups, budget.indexGroups);
      aggregate.predicateGroups =
          std::max(aggregate.predicateGroups, budget.predicateGroups);
      aggregate.stateGroups =
          std::max(aggregate.stateGroups, budget.stateGroups);
      aggregate.primitiveGroups =
          std::max(aggregate.primitiveGroups, budget.primitiveGroups);
      aggregate.peakGroups =
          std::max(aggregate.peakGroups, budget.peakGroups);
    };
    auto appendSnapshot = [&](PhysicalResourceRequirements &requirements,
                              const VLAValueLifetimeSnapshot &snapshot) {
      auto append = [&](PhysicalLiveClass kind, unsigned sew, unsigned count) {
        if (count == 0)
          return true;
        std::optional<RVVVectorShape> shape =
            rvvShapeForSameLanes(dataShape, sew, target);
        if (!shape)
          return false;
        requirements.live.push_back({kind, *shape, count, 0});
        return true;
      };
      if (!append(PhysicalLiveClass::Value, 32, snapshot.f32) ||
          !append(PhysicalLiveClass::Value, 16, snapshot.f16) ||
          !append(PhysicalLiveClass::Value, 8, snapshot.byte) ||
          !append(PhysicalLiveClass::Value, 32, snapshot.u32) ||
          !append(PhysicalLiveClass::Index,
                  static_cast<unsigned>(target.xlen), snapshot.index))
        return false;
      if (snapshot.mask != 0)
        requirements.live.push_back(
            {PhysicalLiveClass::Predicate, {}, 1, snapshot.mask});
      return true;
    };
    auto applyPhase = [&](llvm::ArrayRef<PhysicalLiveRange> extra) {
      VLAValueLifetimeSnapshot empty;
      llvm::ArrayRef<VLAValueLifetimeSnapshot> snapshots = facts.lifetimes;
      if (snapshots.empty())
        snapshots = llvm::ArrayRef<VLAValueLifetimeSnapshot>(&empty, 1);
      for (const VLAValueLifetimeSnapshot &snapshot : snapshots) {
        PhysicalResourceRequirements requirements;
        requirements.live.append(persistent.begin(), persistent.end());
        requirements.live.append(extra.begin(), extra.end());
        if (!appendSnapshot(requirements, snapshot))
          return false;
        std::optional<PhysicalResourceBudget> budget =
            calculatePhysicalResources(requirements, target);
        if (!budget)
          return false;
        mergeBudget(*budget);
      }
      return true;
    };
    if (!applyPhase({}))
      continue;

    bool resourceLegal = true;
    for (const VLAIndexedMemoryFact &memory : facts.indexedMemory) {
      std::optional<RVVVectorShape> elementShape =
          rvvShapeForSameLanes(dataShape, memory.elementSEW, target);
      std::optional<RVVVectorShape> offsetShape =
          rvvShapeForSameLanes(dataShape, memory.offsetSEW, target);
      if (!elementShape || !offsetShape ||
          !applyPhase({PhysicalLiveRange{PhysicalLiveClass::Memory,
                                        *elementShape, 1, 0},
                      PhysicalLiveRange{PhysicalLiveClass::Index,
                                        *offsetShape, 1, 0}})) {
        resourceLegal = false;
        break;
      }
    }
    if (!resourceLegal)
      continue;
    for (const VLASegmentMemoryFact &memory : facts.segmentMemory) {
      std::optional<RVVVectorShape> elementShape =
          rvvShapeForSameLanes(dataShape, memory.elementSEW, target);
      if (!elementShape ||
          !applyPhase({PhysicalLiveRange{PhysicalLiveClass::Memory,
                                        *elementShape, memory.fields, 0}})) {
        resourceLegal = false;
        break;
      }
    }
    if (!resourceLegal)
      continue;

    std::optional<VLANarrowPhysical> narrow;
    if (facts.hasNarrow) {
      if (facts.dataSEW != 32)
        continue;
      std::optional<RVVVectorShape> intermediateShape =
          rvvShapeForSameLanes(dataShape, 16, target);
      std::optional<RVVVectorShape> resultShape =
          rvvShapeForSameLanes(dataShape, 8, target);
      if (!intermediateShape || !resultShape)
        continue;
      VLANarrowPhysical selected;
      selected.sourceShape = dataShape;
      selected.intermediateShape = *intermediateShape;
      selected.resultShape = *resultShape;
      PhysicalResourceRequirements narrowRequirements;
      narrowRequirements.live = {
          {PhysicalLiveClass::Temporary, dataShape, 1, 0},
          {PhysicalLiveClass::Temporary, *intermediateShape, 1, 0},
          {PhysicalLiveClass::Temporary, *resultShape, 1, 0}};
      std::optional<PhysicalResourceBudget> narrowResources =
          calculatePhysicalResources(narrowRequirements, target);
      if (!narrowResources || !applyPhase(narrowRequirements.live))
        continue;
      selected.resources = *narrowResources;
      narrow = selected;
    }

    for (const SelectedVLAStatePhysical &state : facts.states) {
      llvm::SmallVector<PhysicalLiveRange, 6> transient;
      switch (state.stripUpdate) {
      case VLAStateStripUpdate::InclusiveAddScan:
        transient = {{PhysicalLiveClass::Temporary, dataShape, 3, 0},
                     {PhysicalLiveClass::Predicate, {}, 1, 1}};
        break;
      case VLAStateStripUpdate::SegmentedInclusiveAddScan:
        transient = {{PhysicalLiveClass::Temporary, dataShape, 4, 0},
                     {PhysicalLiveClass::Predicate, {}, 1, 2}};
        break;
      case VLAStateStripUpdate::AddReduction:
      case VLAStateStripUpdate::MaxReduction:
        transient = {{PhysicalLiveClass::Temporary, dataShape, 1, 0}};
        if (state.carry != VLAStateCarryRepresentation::Vector ||
            rvvRegisterGroups(dataShape) < 2)
          transient.push_back({PhysicalLiveClass::Temporary, {}, 1, 2});
        break;
      case VLAStateStripUpdate::WideningAddReduction: {
        std::optional<RVVVectorShape> byteShape =
            rvvShapeForSameLanes(dataShape, 8, target);
        if (!byteShape) {
          resourceLegal = false;
          break;
        }
        transient = {{PhysicalLiveClass::Temporary, *byteShape, 1, 0},
                     {PhysicalLiveClass::Temporary, {}, 1, 1}};
        break;
      }
      case VLAStateStripUpdate::ArgMaxSummary:
        if (!indexShape) {
          indexShape = rvvShapeForSameLanes(
              dataShape, static_cast<unsigned>(target.xlen), target);
        }
        if (!indexShape) {
          resourceLegal = false;
          break;
        }
        transient = {{PhysicalLiveClass::Temporary, dataShape, 1, 0},
                     {PhysicalLiveClass::Index, *indexShape, 1, 0}};
        break;
      case VLAStateStripUpdate::OnlineSoftmaxSummary:
        transient = {{PhysicalLiveClass::Temporary, dataShape, 3, 0},
                     {PhysicalLiveClass::Temporary, {}, 1, 2}};
        break;
      }
      if (!resourceLegal || !applyPhase(transient)) {
        resourceLegal = false;
        break;
      }
    }
    if (!resourceLegal)
      continue;

    if (facts.lookupCount != 0) {
      std::optional<RVVVectorShape> codeShape =
          rvvShapeForSameLanes(dataShape, 8, target);
      std::optional<RVVVectorShape> index16Shape =
          rvvShapeForSameLanes(dataShape, 16, target);
      std::optional<RVVVectorShape> index32Shape =
          rvvShapeForSameLanes(dataShape, 32, target);
      if (!codeShape || !index16Shape || !index32Shape)
        continue;
      if (!applyPhase({
              {PhysicalLiveClass::Temporary, *codeShape, 1, 0},
              {PhysicalLiveClass::Temporary, *index16Shape, 1, 0},
              {PhysicalLiveClass::Temporary, *index32Shape, 1, 0}}) ||
          !applyPhase({
              {PhysicalLiveClass::Index, *index32Shape, 1, 0},
              {PhysicalLiveClass::Memory, *index32Shape, 1, 0},
              {PhysicalLiveClass::Temporary, *index32Shape, 1, 0}}))
        continue;
    }

    for (const PhysicalResourceBudget &resource :
         facts.localPrimitiveResources) {
      if (resource.peakGroups > static_cast<unsigned>(target.vectorRegisters)) {
        resourceLegal = false;
        break;
      }
      mergeBudget(resource);
      if (resource.primitiveGroups != 0 &&
          !applyPhase({PhysicalLiveRange{PhysicalLiveClass::Temporary, {}, 1,
                                        resource.primitiveGroups}})) {
        resourceLegal = false;
        break;
      }
    }
    if (!resourceLegal)
      continue;

    SelectedVLAEntityPhysical selected;
    selected.mapping = std::move(mapping);
    selected.dataShape = dataShape;
    if (needsIndexShape)
      selected.indexShape = *indexShape;
    selected.maskRatio = rvvMaskRatio(dataShape).value_or(0);
    if (selected.maskRatio == 0)
      continue;
    selected.narrow = narrow;
    selected.resources = aggregate;
    const unsigned preferredGroups =
        std::max(1u, std::min(4u, static_cast<unsigned>(target.vectorRegisters) /
                                      8u));
    const unsigned laneGroups = rvvRegisterGroups(dataShape);
    legal.push_back(Candidate{
        std::move(selected),
        static_cast<unsigned>(std::abs(static_cast<int>(laneGroups) -
                                       static_cast<int>(preferredGroups))),
        aggregate.peakGroups,
        std::numeric_limits<unsigned>::max() - mappedLaneFactor});
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.lanePenalty, lhs.peakGroups, lhs.inverseLanes) <
           std::tie(rhs.lanePenalty, rhs.peakGroups, rhs.inverseLanes);
  });
  return std::move(legal.front().physical);
}

std::optional<SelectedSignBitI8Physical>
selectSignBitI8Physical(const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.littleEndian || target.vlenBits < 128 ||
      !target.hasWideningInteger)
    return std::nullopt;
  std::optional<RVVVectorShape> activation =
      rvvShapeForSemanticLanes(8, 32, target);
  if (!activation)
    return std::nullopt;
  std::optional<RVVVectorShape> widened =
      rvvShapeForSameLanes(*activation, 16, target);
  std::optional<unsigned> maskRatio =
      widened ? rvvMaskRatio(*widened) : std::nullopt;
  if (!widened || !maskRatio ||
      !target.supportsVectorShape(kRVVE32M1.sew,
                                  kRVVE32M1.lmulEighths))
    return std::nullopt;

  SelectedSignBitI8Physical selected;
  selected.activationShape = *activation;
  selected.widenedShape = *widened;
  selected.reductionShape = kRVVE32M1;
  selected.maskRatio = *maskRatio;
  unsigned activationGroups = rvvRegisterGroups(*activation);
  unsigned widenedGroups = rvvRegisterGroups(*widened);
  selected.resources.architecturalGroups = target.vectorRegisters;
  selected.resources.valueGroups = widenedGroups;
  selected.resources.memoryGroups = activationGroups;
  selected.resources.predicateGroups = 1;
  selected.resources.primitiveGroups = 3 * widenedGroups + 1;
  selected.resources.peakGroups = selected.resources.primitiveGroups;
  if (selected.resources.peakGroups >=
      static_cast<unsigned>(target.vectorRegisters))
    return std::nullopt;
  return selected;
}

std::optional<PhysicalResourceBudget>
calculateQuantDecodeResources(const QuantDecodeResourceFacts &facts,
                              const RISCVTargetProfile &target) {
  PhysicalResourceRequirements requirements;
  requirements.reservedGroups = 0;
  for (const RVVShapeMultiplicity &value : facts.loadedValues)
    requirements.live.push_back(
        PhysicalLiveRange{PhysicalLiveClass::Value, value.shape, value.count});
  for (const RVVShapeMultiplicity &value : facts.indexValues)
    requirements.live.push_back(
        PhysicalLiveRange{PhysicalLiveClass::Index, value.shape, value.count});
  for (const RVVShapeMultiplicity &value : facts.temporaryValues)
    requirements.live.push_back(PhysicalLiveRange{
        PhysicalLiveClass::Temporary, value.shape, value.count});
  if (facts.predicateGroups != 0)
    requirements.live.push_back(PhysicalLiveRange{
        PhysicalLiveClass::Predicate, {}, 1, facts.predicateGroups});
  std::optional<PhysicalResourceBudget> resources =
      calculatePhysicalResources(requirements, target);
  if (!resources ||
      resources->peakGroups >= static_cast<unsigned>(target.vectorRegisters))
    return std::nullopt;
  resources->memoryGroups = resources->valueGroups;
  resources->primitiveGroups +=
      resources->valueGroups + resources->indexGroups;
  return resources;
}

std::optional<SelectedTernaryI8DotPhysical>
selectTernaryI8DotPhysical(const TernaryI8DotCandidateFacts &facts,
                           const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasWideningInteger || !target.littleEndian ||
      target.vlenBits < 128)
    return std::nullopt;

  std::optional<RVVVectorShape> byte32 =
      rvvShapeForSemanticLanes(8, 32, target);
  std::optional<RVVVectorShape> byte16 =
      rvvShapeForSemanticLanes(8, 16, target);
  std::optional<RVVVectorShape> widened32 =
      byte32 ? rvvShapeForSameLanes(*byte32, 16, target) : std::nullopt;
  std::optional<RVVVectorShape> widened16 =
      byte16 ? rvvShapeForSameLanes(*byte16, 16, target) : std::nullopt;
  if (!byte32 || !byte16 || !widened32 || !widened16 ||
      !target.supportsVectorShape(kRVVE32M1.sew,
                                  kRVVE32M1.lmulEighths))
    return std::nullopt;

  if (*byte32 != RVVVectorShape{8, 16} &&
      *byte32 != RVVVectorShape{8, 8})
    return std::nullopt;

  SelectedTernaryI8DotPhysical selected;
  LocalPrimitiveKind primitive =
      facts.semantic == TernaryI8DotSemantic::Base3Digits
          ? LocalPrimitiveKind::Base3TernaryI8
          : LocalPrimitiveKind::PackedI2TernaryI8;
  selected.implementation = makeRVVLocalImplementation(
      primitive, CoreInstructionKind::RVVWideningIntegerDot, kCoreAxisK,
      32, *byte32, *byte16, 1, 32, 0, &target);
  if (!validateLocalInstructionMapping(selected.implementation))
    return std::nullopt;
  selected.decode = facts.semantic == TernaryI8DotSemantic::Base3Digits
                        ? TernaryDecodeTopology::Base3Digits
                        : TernaryDecodeTopology::PackedI2Fields;
  selected.primarySourceShape = *byte32;
  selected.secondarySourceShape = *byte16;

  QuantDecodeResourceFacts resources;
  resources.loadedValues = {{*byte32, 1}, {*byte16, 1}};
  switch (facts.semantic) {
  case TernaryI8DotSemantic::Base3Digits:
    resources.temporaryValues = {{*widened32, 3}, {kRVVE32M1, 1}};
    break;
  case TernaryI8DotSemantic::PackedI2Fields:
    resources.temporaryValues =
        {{*byte32, 2}, {*widened32, 1}, {kRVVE32M1, 1}};
    break;
  }
  std::optional<PhysicalResourceBudget> budget =
      calculateQuantDecodeResources(resources, target);
  if (!budget)
    return std::nullopt;
  selected.resources = *budget;
  return selected;
}

std::optional<SelectedCodebookGatherI8Physical>
selectCodebookGatherI8Physical(const CodebookGatherI8CandidateFacts &facts,
                               const RISCVTargetProfile &target) {
  const bool supportedPrimitive =
      facts.primitive == LocalPrimitiveKind::SignedCodebook8I8 ||
      facts.primitive == LocalPrimitiveKind::SignedCodebook4I8 ||
      facts.primitive == LocalPrimitiveKind::PackedU9U7CodebookI8 ||
      facts.primitive == LocalPrimitiveKind::PackedU11GridDeltaI8;
  if (!target.hasRVV || !target.hasIndexedMemory ||
      !target.hasWideningInteger || !target.littleEndian ||
      target.vlenBits < 128 || !supportedPrimitive ||
      (facts.entryWidth != 4 && facts.entryWidth != 8))
    return std::nullopt;

  unsigned codeCount = 32 / facts.entryWidth;
  if (facts.codeByteExtent != codeCount &&
      facts.codeByteExtent != 2 * codeCount)
    return std::nullopt;
  unsigned tableSEW = facts.entryWidth * 8;
  std::optional<RVVVectorShape> codeShape =
      rvvShapeForSemanticLanes(8, facts.codeByteExtent, target);
  std::optional<RVVVectorShape> indexShape =
      rvvShapeForSemanticLanes(16, codeCount, target);
  std::optional<RVVVectorShape> tableShape =
      rvvShapeForSemanticLanes(tableSEW, codeCount, target);
  std::optional<RVVVectorShape> activationShape =
      rvvShapeForSemanticLanes(8, 32, target);
  std::optional<RVVVectorShape> productShape =
      activationShape
          ? rvvShapeForSameLanes(*activationShape, 16, target)
          : std::nullopt;
  if (!codeShape || !indexShape || !tableShape || !activationShape ||
      !productShape ||
      !target.supportsVectorShape(kRVVE32M1.sew,
                                  kRVVE32M1.lmulEighths) ||
      !target.supportsIndexedVectorMemory(
          tableShape->sew, tableShape->lmulEighths, indexShape->sew,
          indexShape->lmulEighths))
    return std::nullopt;

  if (*activationShape != RVVVectorShape{8, 16} &&
      *activationShape != RVVVectorShape{8, 8})
    return std::nullopt;

  SelectedCodebookGatherI8Physical selected;
  selected.implementation = makeRVVLocalImplementation(
      facts.primitive, CoreInstructionKind::RVVIndexedGather, kCoreAxisK, 32,
      *activationShape, *tableShape, 1, 32, facts.entryWidth, &target);
  if (!validateLocalInstructionMapping(selected.implementation))
    return std::nullopt;
  selected.codeShape = *codeShape;
  selected.activationShape = *activationShape;
  QuantDecodeResourceFacts resources;
  resources.loadedValues =
      {{*codeShape, 1}, {*tableShape, 1}, {*activationShape, 1}};
  resources.indexValues = {{*indexShape, 1}};
  resources.temporaryValues =
      {{*tableShape, 1}, {*activationShape, 1}, {*productShape, 1},
       {kRVVE32M1, 2}};
  resources.predicateGroups = 1;
  std::optional<PhysicalResourceBudget> budget =
      calculateQuantDecodeResources(resources, target);
  if (!budget)
    return std::nullopt;
  selected.resources = *budget;
  return selected;
}

std::optional<SelectedNibbleCodebookI8Physical>
selectNibbleCodebookI8Physical(const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasWideningInteger || !target.littleEndian ||
      target.vlenBits < 128)
    return std::nullopt;
  std::optional<RVVVectorShape> packedShape =
      rvvShapeForSemanticLanes(8, 16, target);
  std::optional<RVVVectorShape> activationShape =
      rvvShapeForSemanticLanes(8, 32, target);
  if (!activationShape ||
      (*activationShape != RVVVectorShape{8, 16} &&
       *activationShape != RVVVectorShape{8, 8}))
    return std::nullopt;
  const bool combined = *activationShape == RVVVectorShape{8, 16};
  std::optional<RVVVectorShape> productShape =
      rvvShapeForSemanticLanes(16, combined ? 32 : 16, target);
  if (!packedShape || !activationShape || !productShape ||
      !target.supportsVectorShape(kRVVE32M1.sew,
                                  kRVVE32M1.lmulEighths))
    return std::nullopt;

  SelectedNibbleCodebookI8Physical selected;
  selected.implementation = makeRVVLocalImplementation(
      LocalPrimitiveKind::NibbleCodebookI8,
      CoreInstructionKind::RVVWideningIntegerDot, kCoreAxisK, 32,
      *activationShape, *packedShape, 1, 32, 0, &target);
  if (!validateLocalInstructionMapping(selected.implementation))
    return std::nullopt;
  selected.packedShape = *packedShape;
  selected.tableShape = combined ? *activationShape : *packedShape;
  selected.activationShape = *activationShape;
  QuantDecodeResourceFacts resources;
  resources.loadedValues =
      {{*packedShape, 1}, {selected.tableShape, 1}, {*activationShape, 1}};
  resources.temporaryValues =
      {{*productShape, combined ? 1u : 2u}, {kRVVE32M1, 2}};
  std::optional<PhysicalResourceBudget> budget =
      calculateQuantDecodeResources(resources, target);
  if (!budget)
    return std::nullopt;
  selected.resources = *budget;
  return selected;
}

std::optional<SelectedQuantI8DotPhysical>
selectQuantI8DotPhysical(const QuantI8DotCandidateFacts &facts,
                         const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasWideningInteger || target.vlenBits < 128 ||
      !target.littleEndian ||
      !target.supportsVectorShape(kRVVE32M1.sew,
                                  kRVVE32M1.lmulEighths))
    return std::nullopt;
  if (facts.semanticExtent == 0)
    return std::nullopt;
  LocalPrimitiveKind primitive = LocalPrimitiveKind::None;
  switch (facts.semantic) {
  case QuantI8DotSemantic::PackedI4:
    primitive = LocalPrimitiveKind::PackedI4I8;
    break;
  case QuantI8DotSemantic::PackedI5:
    primitive = LocalPrimitiveKind::PackedI5I8;
    break;
  case QuantI8DotSemantic::PackedI3Grouped:
    primitive = LocalPrimitiveKind::PackedI3GroupedI8;
    break;
  case QuantI8DotSemantic::IQ2S:
    primitive = LocalPrimitiveKind::IQ2SI8;
    break;
  case QuantI8DotSemantic::IQ3S:
    primitive = LocalPrimitiveKind::IQ3SI8;
    break;
  case QuantI8DotSemantic::IQ1M:
    primitive = LocalPrimitiveKind::IQ1MI8;
    break;
  case QuantI8DotSemantic::Q6K:
    primitive = LocalPrimitiveKind::Q6KI8;
    break;
  }

  llvm::SmallVector<unsigned> laneCandidates;
  for (unsigned lanes : {64u, 32u})
    if (lanes <= facts.semanticExtent && facts.semanticExtent % lanes == 0)
      laneCandidates.push_back(lanes);
  if (laneCandidates.empty())
    return std::nullopt;

  for (unsigned lanes : laneCandidates) {
    std::optional<RVVVectorShape> byteShape =
        rvvShapeForSemanticLanes(8, lanes, target);
    std::optional<RVVVectorShape> productShape =
        byteShape ? rvvShapeForSameLanes(*byteShape, 16, target)
                  : std::nullopt;
    if (!byteShape || !productShape)
      continue;
    const unsigned segments = lanes / 16;
    QuantDecodeResourceFacts resources;
    resources.loadedValues = {{*byteShape, 2}};
    resources.temporaryValues =
        {{*byteShape, 2}, {kRVVE32M1, 4 * segments}};
    std::optional<PhysicalResourceBudget> budget =
        calculateQuantDecodeResources(resources, target);
    if (budget) {
      SelectedQuantI8DotPhysical selected;
      selected.implementation = makeRVVLocalImplementation(
          primitive, CoreInstructionKind::RVVWideningIntegerDot,
          kCoreAxisK, lanes, *byteShape, {}, 1, facts.semanticExtent, 0,
          &target);
      if (!validateLocalInstructionMapping(selected.implementation))
        continue;
      selected.operandShape = *byteShape;
      selected.resources = *budget;
      return selected;
    }
  }

  if (facts.semantic == QuantI8DotSemantic::PackedI4 ||
      facts.semantic == QuantI8DotSemantic::PackedI5 ||
      facts.semantic == QuantI8DotSemantic::PackedI3Grouped)
    return std::nullopt;
  const RVVVectorShape stripShape{8, 16};
  if (!target.supportsVectorShape(stripShape.sew, stripShape.lmulEighths) ||
      !target.supportsVectorShape(16, 32))
    return std::nullopt;
  SelectedQuantI8DotPhysical selected;
  selected.implementation = makeRVVLocalImplementation(
      primitive, CoreInstructionKind::RVVWideningIntegerDot, kCoreAxisK, 16,
      stripShape, {}, 1, facts.semanticExtent, 0, &target);
  if (!validateLocalInstructionMapping(selected.implementation))
    return std::nullopt;
  selected.operandShape = stripShape;
  QuantDecodeResourceFacts stripResources;
  stripResources.loadedValues = {{stripShape, 2}};
  stripResources.temporaryValues = {{kRVVE32M1, 2}};
  std::optional<PhysicalResourceBudget> stripBudget =
      calculateQuantDecodeResources(stripResources, target);
  if (!stripBudget)
    return std::nullopt;
  selected.resources = *stripBudget;
  return selected;
}

std::optional<SelectedE2M1E8M0I8Physical>
selectE2M1E8M0I8Physical(const RISCVTargetProfile &target) {
  if (!target.hasRVV || target.vlenBits < 128 || !target.littleEndian)
    return std::nullopt;

  struct Candidate {
    RVVVectorShape packed;
    RVVVectorShape activation;
  };
  llvm::SmallVector<Candidate> candidates = {
      {{8, 4}, {8, 4}},
      {{8, 8}, {8, 16}},
  };
  for (const Candidate &candidate : candidates) {
    const std::optional<unsigned> packedLanes =
        rvvLaneCapacity(candidate.packed, target);
    const std::optional<unsigned> activationLanes =
        rvvLaneCapacity(candidate.activation, target);
    if (!packedLanes || !activationLanes || *packedLanes < 16 ||
        *activationLanes < 16 ||
        !target.supportsVectorShape(16,
                                    candidate.activation.lmulEighths * 2))
      continue;
    RVVVectorShape productShape{16,
                                candidate.activation.lmulEighths * 2};
    QuantDecodeResourceFacts resources;
    resources.loadedValues =
        {{candidate.packed, 1}, {candidate.activation, 1}};
    resources.temporaryValues =
        {{candidate.packed, 2}, {candidate.activation, 2},
         {productShape, 1}, {kRVVE32M1, 2}};
    std::optional<PhysicalResourceBudget> budget =
        calculateQuantDecodeResources(resources, target);
    if (!budget)
      continue;
    SelectedE2M1E8M0I8Physical selected;
    selected.implementation = makeRVVLocalImplementation(
        LocalPrimitiveKind::E2M1E8M0I8,
        CoreInstructionKind::RVVWideningIntegerDot, kCoreAxisK, 32,
        candidate.packed, candidate.activation, 1, 32, 0, &target);
    if (!validateLocalInstructionMapping(selected.implementation))
      continue;
    selected.packedShape = candidate.packed;
    selected.activationShape = candidate.activation;
    selected.resources = *budget;
    return selected;
  }

  const RVVVectorShape stripShape{8, 8};
  if (!target.supportsVectorShape(stripShape.sew, stripShape.lmulEighths) ||
      !target.supportsVectorShape(16, 16))
    return std::nullopt;
  SelectedE2M1E8M0I8Physical selected;
  std::optional<unsigned> stripLanes = rvvLaneCapacity(stripShape, target);
  if (!stripLanes)
    return std::nullopt;
  selected.implementation = makeRVVLocalImplementation(
      LocalPrimitiveKind::E2M1E8M0I8,
      CoreInstructionKind::RVVWideningIntegerDot, kCoreAxisK, *stripLanes,
      stripShape, {}, 1, 32, 0, &target);
  if (!validateLocalInstructionMapping(selected.implementation))
    return std::nullopt;
  selected.packedShape = stripShape;
  selected.activationShape = stripShape;
  QuantDecodeResourceFacts stripResources;
  stripResources.loadedValues = {{stripShape, 2}};
  stripResources.temporaryValues = {{RVVVectorShape{16, 16}, 1},
                                    {kRVVE32M1, 2}};
  std::optional<PhysicalResourceBudget> stripBudget =
      calculateQuantDecodeResources(stripResources, target);
  if (!stripBudget)
    return std::nullopt;
  selected.resources = *stripBudget;
  return selected;
}

std::optional<SelectedGroupedAffineI4I8Physical>
selectGroupedAffineI4I8Physical(
    const GroupedAffineI4I8CandidateFacts &facts,
    const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasWideningInteger ||
      target.vlenBits < 128 || !target.littleEndian)
    return std::nullopt;
  if (facts.packedExtent != 128 || facts.scaleExtent != 12 ||
      facts.activationExtent != 256 || facts.activationSumExtent != 32)
    return std::nullopt;

  std::optional<RVVVectorShape> scaleShape =
      rvvShapeForSemanticLanes(8, 12, target);
  std::optional<RVVVectorShape> activationSumShape =
      rvvShapeForSemanticLanes(8, 32, target);
  if (!scaleShape || !activationSumShape)
    return std::nullopt;

  SelectedGroupedAffineI4I8Physical selected;
  selected.scaleShape = *scaleShape;
  selected.activationSumShape = *activationSumShape;
  if (!target.supportsVectorShape(8, 8) ||
      !target.supportsVectorShape(16, 16) ||
      !target.supportsVectorShape(32, 8))
    return std::nullopt;
  selected.packedShape = kRVVE8M1;
  selected.activationShape = kRVVE8M1;
  selected.widenedShape = kRVVE16M2;
  selected.reductionShape = kRVVE32M1;

  std::optional<unsigned> nativeLanes =
      rvvLaneCapacity(selected.packedShape, target);
  if (nativeLanes && (*nativeLanes == 16 || *nativeLanes == 32)) {
    selected.implementation = makeRVVLocalImplementation(
        LocalPrimitiveKind::GroupedAffineI4I8,
        CoreInstructionKind::RVVWideningIntegerDot, kCoreAxisK,
        *nativeLanes, selected.packedShape, selected.widenedShape, 1,
        *nativeLanes, 0, &target);
    if (!validateLocalInstructionMapping(selected.implementation))
      return std::nullopt;
    selected.resources.architecturalGroups = target.vectorRegisters;
    selected.resources.valueGroups =
        rvvRegisterGroups(selected.packedShape) +
        rvvRegisterGroups(selected.scaleShape) +
        rvvRegisterGroups(selected.activationShape) +
        rvvRegisterGroups(selected.activationSumShape);
    selected.resources.memoryGroups = selected.resources.valueGroups;
    selected.resources.primitiveGroups = *nativeLanes == 16 ? 32 : 8;
    selected.resources.peakGroups = selected.resources.primitiveGroups;
    if (selected.resources.peakGroups >
        static_cast<unsigned>(target.vectorRegisters))
      return std::nullopt;
    return selected;
  }

  std::optional<unsigned> stripLanes =
      rvvLaneCapacity(selected.packedShape, target);
  if (!stripLanes)
    return std::nullopt;
  selected.implementation = makeRVVLocalImplementation(
      LocalPrimitiveKind::GroupedAffineI4I8,
      CoreInstructionKind::RVVWideningIntegerDot, kCoreAxisK, *stripLanes,
      selected.packedShape, selected.widenedShape, 1, 32, 0, &target);
  if (!validateLocalInstructionMapping(selected.implementation))
    return std::nullopt;
  QuantDecodeResourceFacts resources;
  resources.loadedValues =
      {{selected.packedShape, 1}, {selected.scaleShape, 1},
       {selected.activationShape, 1}, {selected.activationSumShape, 1}};
  resources.temporaryValues =
      {{selected.widenedShape, 1}, {selected.reductionShape, 2}};
  std::optional<PhysicalResourceBudget> budget =
      calculateQuantDecodeResources(resources, target);
  if (!budget)
    return std::nullopt;
  selected.resources = *budget;
  return selected;
}

std::optional<PhysicalResourceBudget>
calculateDenseMicrokernelResources(
    const DenseMicrokernelResourceFacts &facts,
    const RISCVTargetProfile &target) {
  if (!facts.inputShape || !facts.accumulatorShape ||
      facts.accumulatorVectors == 0 || facts.loadWindow == 0 ||
      !target.supportsVectorShape(facts.inputShape.sew,
                                  facts.inputShape.lmulEighths) ||
      !target.supportsVectorShape(facts.accumulatorShape.sew,
                                  facts.accumulatorShape.lmulEighths))
    return std::nullopt;
  PhysicalResourceRequirements requirements;
  requirements.live = {
      PhysicalLiveRange{PhysicalLiveClass::Value, facts.accumulatorShape,
                        facts.accumulatorVectors},
      PhysicalLiveRange{PhysicalLiveClass::Memory, facts.inputShape,
                        (facts.lhsVectorsPerWindow +
                         facts.rhsVectorsPerWindow) *
                            facts.loadWindow}};
  if (facts.predicateGroups != 0)
    requirements.live.push_back(PhysicalLiveRange{
        PhysicalLiveClass::Predicate, {}, 1, facts.predicateGroups});
  if (facts.stateGroups != 0)
    requirements.live.push_back(PhysicalLiveRange{
        PhysicalLiveClass::State, {}, 1, facts.stateGroups});
  if (facts.handoffGroups != 0)
    requirements.live.push_back(PhysicalLiveRange{
        PhysicalLiveClass::Handoff, {}, 1, facts.handoffGroups});
  std::optional<PhysicalResourceBudget> resources =
      calculatePhysicalResources(requirements, target);
  if (!resources)
    return std::nullopt;
  resources->primitiveGroups = resources->valueGroups + resources->memoryGroups;
  return resources;
}

std::optional<SelectedF32DotPhysical>
selectF32DotPhysicalConfig(const F32DotCandidateFacts &facts,
                           const RISCVTargetProfile &target,
                           const RISCVBackendConfig &config) {
  CoreMappingProblem problem = facts.mapping;
  problem.laneSEW = 32;
  problem.laneInstruction = CoreInstructionKind::RVVFMA;
  if (config.parameters.dotLMUL != 0)
    problem.laneLMULCandidates = {
        static_cast<unsigned>(config.parameters.dotLMUL)};
  else
    problem.laneLMULCandidates = integerLMULCandidates(target, 32);
  problem.unrollCandidates =
      config.parameters.dotKUnroll != 0
          ? llvm::SmallVector<unsigned, 4>{
                static_cast<unsigned>(config.parameters.dotKUnroll)}
          : llvm::SmallVector<unsigned, 4>{1, 2, 4};
  problem.pipelineBufferCandidates = {1};

  const unsigned baseF32Lanes =
      rvvLaneCapacity(rvvShape(32, 1), target).value_or(0);
  if (baseF32Lanes == 0)
    return std::nullopt;
  unsigned preferredUnroll = 1;
  bool costlyHandoff = facts.materializedInit || facts.reductionPredicate ||
                       facts.indexedOperands != 0;
  if (!costlyHandoff && facts.reductionExtent)
    preferredUnroll = *facts.reductionExtent >= 64
                          ? 4
                          : *facts.reductionExtent >= 32 ? 2 : 1;

  struct Candidate {
    CorePhysicalMapping mapping;
    PhysicalResourceBudget resources;
    unsigned tailPenalty = 0;
    unsigned sequentialPenalty = 0;
    unsigned memoryPenalty = 0;
    unsigned lanePenalty = 0;
    unsigned unrollPenalty = 0;
    unsigned peakGroups = 0;
  };
  llvm::SmallVector<Candidate, 32> legal;
  for (CorePhysicalMapping mapping :
       enumerateCorePhysicalMappings(problem, target)) {
    const PhysicalAxisDecomposition *laneAxis = llvm::find_if(
        mapping.axes, [](const PhysicalAxisDecomposition &axis) {
          return axis.laneFactor > 1;
        });
    const PhysicalAxisDecomposition *reduction =
        findAxisMapping(mapping, kCoreAxisK);
    if (laneAxis == mapping.axes.end() || !reduction ||
        (reduction->unrollFactor != 1 && reduction->unrollFactor != 2 &&
         reduction->unrollFactor != 4))
      continue;
    unsigned accumulatorVectors = 1;
    for (const PhysicalAxisDecomposition &axis : mapping.axes)
      if (axis.role == LogicalAxisRole::Free)
        accumulatorVectors *= axis.registerFactor;
    unsigned lhsVectorsPerWindow = 0;
    unsigned rhsVectorsPerWindow = 0;
    if (laneAxis->id == kCoreAxisK) {
      lhsVectorsPerWindow = accumulatorVectors;
      rhsVectorsPerWindow = 1;
    } else if (laneAxis->id == kCoreAxisM) {
      lhsVectorsPerWindow = 1;
    } else if (laneAxis->id == kCoreAxisN) {
      rhsVectorsPerWindow = 1;
    }
    std::optional<PhysicalResourceBudget> resources =
        calculateDenseMicrokernelResources(
            {mapping.laneShape, mapping.laneShape, accumulatorVectors,
             lhsVectorsPerWindow, rhsVectorsPerWindow,
             reduction->unrollFactor,
             facts.predicateGroups, facts.stateGroups, facts.handoffGroups},
            target);
    if (!resources)
      continue;
    const bool freeLane = laneAxis->role == LogicalAxisRole::Free;
    unsigned sequentialPenalty = 1;
    for (const PhysicalAxisDecomposition &axis : mapping.axes)
      if (axis.role == LogicalAxisRole::Free)
        sequentialPenalty *= std::max(1u, axis.sequentialFactor);
    const unsigned desiredLanes =
        freeLane ? (accumulatorVectors == 1 ? 16 : 8)
                 : (accumulatorVectors >= 8
                        ? 4
                        : std::max(8u, 2 * baseF32Lanes));
    legal.push_back(Candidate{
        std::move(mapping), *resources,
        static_cast<unsigned>(
            facts.reductionExtent &&
            *facts.reductionExtent %
                    (reduction->unrollFactor *
                     (laneAxis->id == kCoreAxisK ? laneAxis->laneFactor : 1)) !=
                0),
        sequentialPenalty,
        facts.stridedOperands * reduction->unrollFactor +
            2 * facts.indexedOperands * reduction->unrollFactor +
            facts.predicateGroups + facts.handoffGroups,
        static_cast<unsigned>(std::abs(
            static_cast<int>(laneAxis->laneFactor) -
            static_cast<int>(desiredLanes))),
        static_cast<unsigned>(std::abs(
            static_cast<int>(reduction->unrollFactor) -
            static_cast<int>(preferredUnroll))),
        resources->peakGroups});
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.tailPenalty, lhs.sequentialPenalty, lhs.memoryPenalty,
                    lhs.lanePenalty, lhs.unrollPenalty, lhs.peakGroups) <
           std::tie(rhs.tailPenalty, rhs.sequentialPenalty, rhs.memoryPenalty,
                    rhs.lanePenalty, rhs.unrollPenalty, rhs.peakGroups);
  });
  return SelectedF32DotPhysical{std::move(legal.front().mapping),
                                legal.front().resources};
}

std::optional<SelectedF16MatmulPhysical>
selectF16MatmulPhysicalConfig(const F16MatmulCandidateFacts &facts,
                              const RISCVTargetProfile &target,
                              const RISCVBackendConfig &config) {
  if (!target.hasRVV || !target.hasF || !target.hasVectorF16 ||
      !target.hasWideningFloat)
    return std::nullopt;
  const unsigned baseInputLanes =
      rvvLaneCapacity(rvvShape(16, 1), target).value_or(0);
  if (baseInputLanes == 0)
    return std::nullopt;
  CoreMappingProblem problem = facts.mapping;
  problem.laneSEW = 16;
  problem.laneInstruction = CoreInstructionKind::RVVWideningFMA;
  if (config.parameters.f16InputLMUL != 0)
    problem.laneLMULCandidates = {
        static_cast<unsigned>(config.parameters.f16InputLMUL)};
  else
    problem.laneLMULCandidates = integerLMULCandidates(target, 16, 4);
  problem.unrollCandidates =
      config.parameters.f16KUnroll != 0
          ? llvm::SmallVector<unsigned, 4>{
                static_cast<unsigned>(config.parameters.f16KUnroll)}
          : llvm::SmallVector<unsigned, 4>{1, 2, 4};
  problem.pipelineBufferCandidates =
      config.parameters.f16LoadBufferCount != 0
          ? llvm::SmallVector<unsigned, 4>{
                static_cast<unsigned>(config.parameters.f16LoadBufferCount)}
          : llvm::SmallVector<unsigned, 4>{1, 2};
  LogicalAxisConstraint *mConstraint = nullptr;
  LogicalAxisConstraint *nConstraint = nullptr;
  for (LogicalAxisConstraint &axis : problem.axes) {
    if (axis.id == kCoreAxisM)
      mConstraint = &axis;
    if (axis.id == kCoreAxisN)
      nConstraint = &axis;
  }
  if (!mConstraint || !nConstraint || !mConstraint->extent ||
      !nConstraint->extent)
    return std::nullopt;
  const unsigned rowTile = static_cast<unsigned>(*mConstraint->extent);
  const unsigned columnTile = static_cast<unsigned>(*nConstraint->extent);
  if (config.parameters.f16RowMicrotile != 0)
    mConstraint->registerFactors = {
        static_cast<unsigned>(config.parameters.f16RowMicrotile)};
  if (config.parameters.f16ColumnMicrotile != 0)
    nConstraint->registerFactors = {
        static_cast<unsigned>(config.parameters.f16ColumnMicrotile)};
  const unsigned preferredRows =
      std::min(rowTile, std::max(1u, baseInputLanes / 4));
  const unsigned preferredLMUL = 1;
  std::optional<uint64_t> reductionExtent;
  for (const LogicalAxisConstraint &axis : problem.axes)
    if (axis.id == kCoreAxisK)
      reductionExtent = axis.extent;
  if (!reductionExtent)
    return std::nullopt;
  const unsigned preferredUnroll =
      *reductionExtent < 32 ? 1 : baseInputLanes >= 16 ? 4 : 2;
  const unsigned preferredBuffers = *reductionExtent >= 64 ? 2 : 1;
  const unsigned preferredColumns =
      baseInputLanes <= 8 && columnTile >= 2 && columnTile % 2 == 0
          ? 2
          : 1;
  struct Candidate {
    CorePhysicalMapping mapping;
    PhysicalResourceBudget resources;
    unsigned tailPenalty = 0;
    unsigned rowPenalty = 0;
    unsigned columnPenalty = 0;
    unsigned lmulPenalty = 0;
    unsigned unrollPenalty = 0;
    unsigned bufferPenalty = 0;
  };
  llvm::SmallVector<Candidate, 64> legal;
  for (CorePhysicalMapping mapping :
       enumerateCorePhysicalMappings(problem, target)) {
    const PhysicalAxisDecomposition *m =
        findAxisMapping(mapping, kCoreAxisM);
    const PhysicalAxisDecomposition *n =
        findAxisMapping(mapping, kCoreAxisN);
    const PhysicalAxisDecomposition *k =
        findAxisMapping(mapping, kCoreAxisK);
    std::optional<unsigned> inputLMUL = rvvIntegerLMUL(mapping.laneShape);
    std::optional<RVVVectorShape> accumulatorShape =
        rvvShapeForSameLanes(mapping.laneShape, 32, target);
    if (!m || !n || !k || !inputLMUL || !accumulatorShape ||
        (k->unrollFactor != 1 && k->unrollFactor != 2 &&
         k->unrollFactor != 4) ||
        (mapping.pipeline.bufferCount != 1 &&
         mapping.pipeline.bufferCount != 2) ||
        (mapping.pipeline.bufferCount == 2 && k->unrollFactor < 2))
      continue;
    const unsigned rows = m->registerFactor;
    const unsigned columns = n->registerFactor;
    const unsigned bufferCount = mapping.pipeline.bufferCount;
    std::optional<PhysicalResourceBudget> resources =
        calculateDenseMicrokernelResources(
            {mapping.laneShape, *accumulatorShape, rows * columns,
             bufferCount == 2 ? rows : 1, columns, bufferCount},
            target);
    if (!resources)
      continue;
    legal.push_back(Candidate{
        std::move(mapping), *resources,
        static_cast<unsigned>(*reductionExtent %
                                  (k->unrollFactor * k->laneFactor) !=
                              0),
        static_cast<unsigned>(std::abs(static_cast<int>(rows) -
                                       static_cast<int>(preferredRows))),
        static_cast<unsigned>(std::abs(static_cast<int>(columns) -
                                       static_cast<int>(preferredColumns))),
        static_cast<unsigned>(std::abs(static_cast<int>(*inputLMUL) -
                                       static_cast<int>(preferredLMUL))),
        static_cast<unsigned>(std::abs(static_cast<int>(k->unrollFactor) -
                                       static_cast<int>(preferredUnroll))),
        static_cast<unsigned>(std::abs(static_cast<int>(bufferCount) -
                                       static_cast<int>(preferredBuffers)))});
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.tailPenalty, lhs.rowPenalty, lhs.columnPenalty,
                    lhs.lmulPenalty,
                    lhs.unrollPenalty, lhs.bufferPenalty,
                    lhs.resources.peakGroups) <
           std::tie(rhs.tailPenalty, rhs.rowPenalty, rhs.columnPenalty,
                    rhs.lmulPenalty,
                    rhs.unrollPenalty, rhs.bufferPenalty,
                    rhs.resources.peakGroups);
  });
  return SelectedF16MatmulPhysical{std::move(legal.front().mapping),
                                   legal.front().resources};
}

} // namespace weft::riscv_internal
