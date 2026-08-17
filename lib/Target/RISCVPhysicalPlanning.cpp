#include "RISCVPhysicalPlanning.h"

#include "RISCVKernelFacts.h"
#include "RISCVRVVSpelling.h"

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

static llvm::SmallVector<CorePhysicalMapping, 16>
enumerateRVVLocalMappings(CoreInstructionKind instruction, unsigned laneAxis,
                          unsigned semanticExtent, unsigned laneSEW,
                          unsigned rowExtent,
                          llvm::ArrayRef<RVVVectorShape> laneShapes,
                          const RISCVTargetProfile &target) {
  if ((laneAxis != kCoreAxisN && laneAxis != kCoreAxisK) ||
      semanticExtent == 0 || rowExtent == 0)
    return {};
  llvm::SmallVector<unsigned, 4> repetitions =
      registerFactorCandidates(semanticExtent,
                               std::min(semanticExtent, 8u));
  CoreMappingProblem problem;
  problem.axes = {
      LogicalAxisConstraint{kCoreAxisM, LogicalAxisRole::Free, rowExtent,
                            false, false, false, {rowExtent}},
      LogicalAxisConstraint{kCoreAxisN, LogicalAxisRole::Free,
                            laneAxis == kCoreAxisN
                                ? std::optional<uint64_t>(semanticExtent)
                                : std::optional<uint64_t>(1),
                            false, laneAxis == kCoreAxisN,
                            laneAxis == kCoreAxisN,
                            laneAxis == kCoreAxisN ? repetitions
                                                   : llvm::SmallVector<unsigned, 4>{1}},
      LogicalAxisConstraint{kCoreAxisK, LogicalAxisRole::Reduction,
                            laneAxis == kCoreAxisK
                                ? std::optional<uint64_t>(semanticExtent)
                                : std::optional<uint64_t>(1),
                            false, laneAxis == kCoreAxisK,
                            laneAxis == kCoreAxisK,
                            laneAxis == kCoreAxisK ? repetitions
                                                   : llvm::SmallVector<unsigned, 4>{1}}};
  problem.laneSEW = laneSEW;
  problem.laneInstruction = instruction;
  problem.laneShapeCandidates.append(laneShapes.begin(), laneShapes.end());
  llvm::SmallVector<CorePhysicalMapping, 16> mappings;
  for (CorePhysicalMapping mapping :
       enumerateCorePhysicalMappings(problem, target)) {
    const PhysicalAxisDecomposition *axis =
        findAxisMapping(mapping, laneAxis);
    if (!axis || axis->laneFactor * axis->registerFactor > semanticExtent)
      continue;
    mappings.push_back(std::move(mapping));
  }
  return mappings;
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
  const unsigned hardwareLanes =
      mappedHardwareLaneFactor(implementation.mapping);
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
    return rvvDot && (hardwareLanes == 16 || hardwareLanes == 32) &&
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

static std::string
selectLocalImplementationSymbol(const LocalImplementation &implementation) {
  const RVVVectorShape primary = implementation.valueShapes.empty()
                                     ? RVVVectorShape{}
                                     : implementation.valueShapes.front();
  const unsigned lanes = localLaneFactor(implementation);
  const unsigned hardwareLanes =
      mappedHardwareLaneFactor(implementation.mapping);
  const unsigned rows =
      implementation.mapping.instruction == CoreInstructionKind::SpacemitIME1MMA
          ? localFragmentFactor(implementation, kCoreAxisM)
          : localRegisterFactor(implementation, kCoreAxisM);
  const bool sequential =
      localSequentialFactor(implementation, kCoreAxisK) > 1;
  const std::string shape = rvvShapeSuffix(primary);
  const std::string registerSuffix =
      "_register_l" + std::to_string(lanes) + "_e" + shape;
  switch (implementation.primitive) {
  case LocalPrimitiveKind::None:
  case LocalPrimitiveKind::F32Math:
    return {};
  case LocalPrimitiveKind::SymmetricI4I8:
  case LocalPrimitiveKind::AffineI4I8: {
    const bool affine =
        implementation.primitive == LocalPrimitiveKind::AffineI4I8;
    const bool ime = implementation.mapping.instruction ==
                     CoreInstructionKind::SpacemitIME1MMA;
    std::string name = "__weft_" + std::string(ime ? "ime1_" : "rvv_") +
                       (affine ? "affine" : "symmetric") + "_i4_i8_";
    if (rows == 4)
      name += "m4_";
    return name + "n16_k32";
  }
  case LocalPrimitiveKind::GroupedAffineI4I8:
    return sequential
               ? "__weft_grouped_affine_i4_i8_strip"
               : "__weft_grouped_affine_i4_i8_register_l" +
                     std::to_string(hardwareLanes) + "_e" + shape;
  case LocalPrimitiveKind::E2M1E8M0I8:
    if (sequential)
      return "__weft_e2m1_e8m0_i8_strip";
    return primary == RVVVectorShape{8, 4}
               ? "__weft_e2m1_e8m0_i8_register_e8mf2"
               : "__weft_e2m1_e8m0_i8_register_e8m1_e8m2";
  case LocalPrimitiveKind::PackedI4I8:
    return "__weft_packed_i4_i8" + registerSuffix;
  case LocalPrimitiveKind::PackedI5I8:
    return "__weft_packed_i5_i8" + registerSuffix;
  case LocalPrimitiveKind::PackedI3GroupedI8:
    return "__weft_packed_i3_grouped_i8" + registerSuffix;
  case LocalPrimitiveKind::Base3TernaryI8:
    return "__weft_base3_ternary_i8" + registerSuffix;
  case LocalPrimitiveKind::PackedI2TernaryI8:
    return "__weft_packed_i2_ternary_i8" + registerSuffix;
  case LocalPrimitiveKind::SignedCodebook8I8:
    return "__weft_signed_codebook8_i8" + registerSuffix;
  case LocalPrimitiveKind::SignedCodebook4I8:
    return "__weft_signed_codebook4_i8" + registerSuffix;
  case LocalPrimitiveKind::PackedU9U7CodebookI8:
    return "__weft_packed_u9_u7_codebook_i8" + registerSuffix;
  case LocalPrimitiveKind::PackedU11GridDeltaI8:
    return "__weft_packed_u11_grid_delta_i8" + registerSuffix;
  case LocalPrimitiveKind::NibbleCodebookI8:
    return "__weft_nibble_codebook_i8" + registerSuffix;
  case LocalPrimitiveKind::IQ2SI8:
    return sequential ? "__weft_iq2_s_i8_strip"
                      : "__weft_iq2_s_i8" + registerSuffix;
  case LocalPrimitiveKind::IQ3SI8:
    return sequential ? "__weft_iq3_s_i8_strip"
                      : "__weft_iq3_s_i8" + registerSuffix;
  case LocalPrimitiveKind::IQ1MI8:
    return sequential ? "__weft_iq1_m_i8_strip"
                      : "__weft_iq1_m_i8" + registerSuffix;
  case LocalPrimitiveKind::Q6KI8:
    return sequential ? "__weft_q6_k_i8_strip"
                      : "__weft_q6_k_i8" + registerSuffix;
  }
  return {};
}

static bool finalizeLocalInstructionMapping(LocalImplementation &implementation) {
  if (!validateLocalInstructionMapping(implementation))
    return false;
  implementation.helperSymbol = selectLocalImplementationSymbol(implementation);
  return implementation.primitive == LocalPrimitiveKind::F32Math ||
         !implementation.helperSymbol.empty();
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
      facts.tableExtent != 16)
    return std::nullopt;
  std::optional<RVVVectorShape> laneShape =
      rvvShapeForSemanticLanes(8, facts.codeExtent, target);
  if (!laneShape)
    return std::nullopt;
  CoreMappingProblem problem;
  problem.axes = {
      LogicalAxisConstraint{kCoreAxisPacked, LogicalAxisRole::Packed,
                            facts.codeExtent, false, true, true, {1}}};
  problem.laneSEW = 8;
  problem.laneInstruction = CoreInstructionKind::RVVIndexedGather;
  problem.laneShapeCandidates = {*laneShape};
  llvm::SmallVector<CorePhysicalMapping> mappings =
      enumerateCorePhysicalMappings(problem, target);
  if (mappings.size() != 1)
    return std::nullopt;
  SelectedBlockDecodePhysical selected;
  selected.mapping = std::move(mappings.front());
  selected.codeShape = *laneShape;
  selected.tableShape = *laneShape;
  selected.resultShape = *laneShape;
  selected.tableExtent = facts.tableExtent;
  PhysicalResourceRequirements requirements;
  requirements.live = {
      {PhysicalLiveClass::Memory, selected.codeShape, 1, 0},
      {PhysicalLiveClass::Memory, selected.tableShape, 1, 0},
      {PhysicalLiveClass::Temporary, selected.resultShape, 1, 0}};
  std::optional<PhysicalResourceBudget> resources =
      calculatePhysicalResources(requirements, target);
  if (!resources)
    return std::nullopt;
  selected.resources = *resources;
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
    PhysicalResourceRequirements requirements;
    requirements.live = {{PhysicalLiveClass::Value, {}, 1, dataGroups}};
    if (candidate.laneShape)
      requirements.live.push_back(
          {PhysicalLiveClass::Memory, candidate.laneShape, 1, 0});
    std::optional<PhysicalResourceBudget> resources =
        calculatePhysicalResources(requirements, target);
    if (!resources)
      continue;
    return SelectedBlockStorePhysical{candidate, *resources};
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

    PhysicalResourceRequirements requirements;
    requirements.live = {
        {PhysicalLiveClass::Value, {}, 1, 8},
        {PhysicalLiveClass::Temporary, {}, 1, 1}};
    if (candidate.laneShape)
      requirements.live.push_back(
          {PhysicalLiveClass::Memory, candidate.laneShape, 1, 0});
    std::optional<PhysicalResourceBudget> resources =
        calculatePhysicalResources(requirements, target);
    if (!resources)
      continue;
    selected.resources = *resources;
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
    PhysicalResourceRequirements requirements;
    requirements.live = {{PhysicalLiveClass::Value, *shape, 1, 0}};
    std::optional<PhysicalResourceBudget> resources =
        calculatePhysicalResources(requirements, target);
    if (!resources)
      return std::nullopt;
    selected.resources = *resources;
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
  llvm::SmallVector<RVVVectorShape, 1> shapes = {kRVVE32M2};
  for (CorePhysicalMapping mapping : enumerateRVVLocalMappings(
           CoreInstructionKind::RVVElementwise, kCoreAxisN, *lanes, 32, 1,
           shapes, target)) {
    LocalImplementation implementation;
    implementation.primitive = LocalPrimitiveKind::F32Math;
    implementation.mapping = std::move(mapping);
    implementation.valueShapes = {kRVVE32M2};
    if (finalizeLocalInstructionMapping(implementation))
      return implementation;
  }
  return std::nullopt;
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
  auto axis = [&](unsigned id) -> const LogicalAxisConstraint * {
    auto found = llvm::find_if(facts.mapping.axes,
                               [&](const LogicalAxisConstraint &candidate) {
                                 return candidate.id == id;
                               });
    return found == facts.mapping.axes.end() ? nullptr : &*found;
  };
  const LogicalAxisConstraint *rows = axis(kCoreAxisM);
  const LogicalAxisConstraint *columns = axis(kCoreAxisN);
  const LogicalAxisConstraint *reduction = axis(kCoreAxisK);
  if (!target.littleEndian || !rows || !columns || !reduction ||
      !rows->extent || (*rows->extent != 1 && *rows->extent != 4) ||
      rows->role != LogicalAxisRole::Free || !columns->extent ||
      *columns->extent != 16 || columns->role != LogicalAxisRole::Free ||
      !reduction->extent || *reduction->extent != 32 ||
      reduction->role != LogicalAxisRole::Reduction ||
      config.structures.i4I8FragmentImplementation < 0 ||
      config.structures.i4I8FragmentImplementation > 2)
    return std::nullopt;
  const unsigned rowExtent = static_cast<unsigned>(*rows->extent);
  const bool supportsIME =
      rowExtent == 4
          ? target.supportsSpacemitIME1I4I8M4N16K32()
          : target.supportsSpacemitIME1I4I8N16K32();
  const bool supportsRVV =
      target.hasF && target.hasVectorF16 && target.hasWideningInteger &&
      target.hasWideningFloat && target.supportsVLENAtLeast(128) &&
      target.supportsVectorShape(8, 8) &&
      target.supportsVectorShape(16, 16) &&
      target.supportsVectorShape(32, 32);
  CoreMappingProblem problem = facts.mapping;
  for (LogicalAxisConstraint &constraint : problem.axes) {
    constraint.allowLane = constraint.id == kCoreAxisN && supportsRVV;
    constraint.requireLane = constraint.allowLane;
    constraint.registerFactors =
        constraint.id == kCoreAxisM
            ? llvm::SmallVector<unsigned, 4>{rowExtent}
            : llvm::SmallVector<unsigned, 4>{1};
  }
  problem.laneSEW = 8;
  problem.laneInstruction = CoreInstructionKind::RVVWideningIntegerDot;
  problem.laneLMULCandidates = {1};
  if (supportsIME)
    problem.fragments.push_back(FragmentMappingConstraint{
        CoreInstructionKind::SpacemitIME1MMA,
        {{kCoreAxisM, rowExtent}, {kCoreAxisN, 16}, {kCoreAxisK, 32}},
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
    if (!finalizeLocalInstructionMapping(selected.implementation))
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
                                   ime ? 1u : rowExtent * 4u});
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
  PhysicalResourceRequirements requirements;
  requirements.live = {
      {PhysicalLiveClass::Memory, selected.activationShape, 1, 0},
      {PhysicalLiveClass::Temporary, selected.widenedShape, 3, 0},
      {PhysicalLiveClass::Temporary, selected.reductionShape, 1, 0},
      {PhysicalLiveClass::Predicate, {}, 1, 1}};
  std::optional<PhysicalResourceBudget> resources =
      calculatePhysicalResources(requirements, target);
  if (!resources)
    return std::nullopt;
  selected.resources = *resources;
  return selected;
}

std::optional<PhysicalResourceBudget>
calculateQuantDecodeResources(const QuantDecodeResourceFacts &facts,
                              const RISCVTargetProfile &target) {
  PhysicalResourceRequirements requirements;
  for (const RVVShapeMultiplicity &value : facts.loadedValues)
    requirements.live.push_back(
        PhysicalLiveRange{PhysicalLiveClass::Memory, value.shape, value.count});
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
  return resources;
}

std::optional<SelectedTernaryI8DotPhysical>
selectTernaryI8DotPhysical(const TernaryI8DotCandidateFacts &facts,
                           const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasWideningInteger || !target.littleEndian ||
      target.vlenBits < 128)
    return std::nullopt;

  std::optional<RVVVectorShape> byte16 =
      rvvShapeForSemanticLanes(8, 16, target);
  if (!byte16 ||
      !target.supportsVectorShape(kRVVE32M1.sew,
                                  kRVVE32M1.lmulEighths))
    return std::nullopt;
  LocalPrimitiveKind primitive =
      facts.semantic == TernaryI8DotSemantic::Base3Digits
          ? LocalPrimitiveKind::Base3TernaryI8
          : LocalPrimitiveKind::PackedI2TernaryI8;
  struct Candidate {
    SelectedTernaryI8DotPhysical physical;
    unsigned registerFactor = 0;
  };
  llvm::SmallVector<Candidate, 8> legal;
  llvm::SmallVector<RVVVectorShape, 8> laneShapes =
      rvvShapeCandidates(target, 8);
  for (CorePhysicalMapping mapping : enumerateRVVLocalMappings(
           CoreInstructionKind::RVVWideningIntegerDot, kCoreAxisK, 32, 8, 1,
           laneShapes, target)) {
    std::optional<RVVVectorShape> widened =
        rvvShapeForSameLanes(mapping.laneShape, 16, target);
    const PhysicalAxisDecomposition *reduction =
        findAxisMapping(mapping, kCoreAxisK);
    if (!widened || !reduction)
      continue;
    LocalImplementation implementation;
    implementation.primitive = primitive;
    implementation.mapping = mapping;
    implementation.valueShapes = {mapping.laneShape, *byte16};
    if (!finalizeLocalInstructionMapping(implementation))
      continue;
    QuantDecodeResourceFacts resources;
    resources.loadedValues = {
        {mapping.laneShape, reduction->registerFactor}, {*byte16, 1}};
    switch (facts.semantic) {
    case TernaryI8DotSemantic::Base3Digits:
      resources.temporaryValues = {
          {*widened, 3 * reduction->registerFactor}, {kRVVE32M1, 1}};
      break;
    case TernaryI8DotSemantic::PackedI2Fields:
      resources.temporaryValues = {
          {mapping.laneShape, 2 * reduction->registerFactor},
          {*widened, reduction->registerFactor},
          {kRVVE32M1, 1}};
      break;
    }
    std::optional<PhysicalResourceBudget> budget =
        calculateQuantDecodeResources(resources, target);
    if (!budget)
      continue;
    SelectedTernaryI8DotPhysical selected;
    selected.implementation = std::move(implementation);
    selected.decode = facts.semantic == TernaryI8DotSemantic::Base3Digits
                          ? TernaryDecodeTopology::Base3Digits
                          : TernaryDecodeTopology::PackedI2Fields;
    selected.primarySourceShape = mapping.laneShape;
    selected.secondarySourceShape = *byte16;
    selected.resources = *budget;
    legal.push_back(
        Candidate{std::move(selected), reduction->registerFactor});
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.physical.resources.peakGroups, lhs.registerFactor) <
           std::tie(rhs.physical.resources.peakGroups, rhs.registerFactor);
  });
  return std::move(legal.front().physical);
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
  if (!codeShape || !indexShape || !tableShape ||
      !target.supportsVectorShape(kRVVE32M1.sew,
                                  kRVVE32M1.lmulEighths) ||
      !target.supportsIndexedVectorMemory(
          tableShape->sew, tableShape->lmulEighths, indexShape->sew,
          indexShape->lmulEighths))
    return std::nullopt;

  struct Candidate {
    SelectedCodebookGatherI8Physical physical;
    unsigned registerFactor = 0;
  };
  llvm::SmallVector<Candidate, 8> legal;
  llvm::SmallVector<RVVVectorShape, 8> laneShapes =
      rvvShapeCandidates(target, 8);
  for (CorePhysicalMapping mapping : enumerateRVVLocalMappings(
           CoreInstructionKind::RVVIndexedGather, kCoreAxisK, 32, 8, 1,
           laneShapes, target)) {
    const PhysicalAxisDecomposition *reduction =
        findAxisMapping(mapping, kCoreAxisK);
    std::optional<RVVVectorShape> productShape =
        rvvShapeForSameLanes(mapping.laneShape, 16, target);
    if (!reduction || !productShape)
      continue;
    LocalImplementation implementation;
    implementation.primitive = facts.primitive;
    implementation.mapping = mapping;
    implementation.valueShapes = {mapping.laneShape, *tableShape};
    implementation.entryWidth = facts.entryWidth;
    if (!finalizeLocalInstructionMapping(implementation))
      continue;
    QuantDecodeResourceFacts resources;
    resources.loadedValues = {
        {*codeShape, 1}, {*tableShape, 1},
        {mapping.laneShape, reduction->registerFactor}};
    resources.indexValues = {{*indexShape, 1}};
    resources.temporaryValues = {
        {*tableShape, 1},
        {mapping.laneShape, reduction->registerFactor},
        {*productShape, reduction->registerFactor},
        {kRVVE32M1, 2}};
    resources.predicateGroups = 1;
    std::optional<PhysicalResourceBudget> budget =
        calculateQuantDecodeResources(resources, target);
    if (!budget)
      continue;
    SelectedCodebookGatherI8Physical selected;
    selected.implementation = std::move(implementation);
    selected.codeShape = *codeShape;
    selected.activationShape = mapping.laneShape;
    selected.resources = *budget;
    legal.push_back(
        Candidate{std::move(selected), reduction->registerFactor});
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.physical.resources.peakGroups, lhs.registerFactor) <
           std::tie(rhs.physical.resources.peakGroups, rhs.registerFactor);
  });
  return std::move(legal.front().physical);
}

std::optional<SelectedNibbleCodebookI8Physical>
selectNibbleCodebookI8Physical(const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasWideningInteger || !target.littleEndian ||
      target.vlenBits < 128)
    return std::nullopt;
  std::optional<RVVVectorShape> packedShape =
      rvvShapeForSemanticLanes(8, 16, target);
  if (!packedShape ||
      !target.supportsVectorShape(kRVVE32M1.sew,
                                  kRVVE32M1.lmulEighths))
    return std::nullopt;
  struct Candidate {
    SelectedNibbleCodebookI8Physical physical;
    unsigned registerFactor = 0;
  };
  llvm::SmallVector<Candidate, 8> legal;
  llvm::SmallVector<RVVVectorShape, 8> laneShapes =
      rvvShapeCandidates(target, 8);
  for (CorePhysicalMapping mapping : enumerateRVVLocalMappings(
           CoreInstructionKind::RVVWideningIntegerDot, kCoreAxisK, 32, 8, 1,
           laneShapes, target)) {
    const PhysicalAxisDecomposition *reduction =
        findAxisMapping(mapping, kCoreAxisK);
    if (!reduction)
      continue;
    const bool combined = mapping.laneShape == RVVVectorShape{8, 16};
    RVVVectorShape tableShape = combined ? mapping.laneShape : *packedShape;
    std::optional<RVVVectorShape> productShape =
        rvvShapeForSemanticLanes(16, combined ? 32 : 16, target);
    if (!productShape)
      continue;
    LocalImplementation implementation;
    implementation.primitive = LocalPrimitiveKind::NibbleCodebookI8;
    implementation.mapping = mapping;
    implementation.valueShapes = {mapping.laneShape, *packedShape};
    if (!finalizeLocalInstructionMapping(implementation))
      continue;
    QuantDecodeResourceFacts resources;
    resources.loadedValues = {
        {*packedShape, 1}, {tableShape, 1},
        {mapping.laneShape, reduction->registerFactor}};
    resources.temporaryValues = {
        {*productShape,
         (combined ? 1u : 2u) * reduction->registerFactor},
        {kRVVE32M1, 2}};
    std::optional<PhysicalResourceBudget> budget =
        calculateQuantDecodeResources(resources, target);
    if (!budget)
      continue;
    SelectedNibbleCodebookI8Physical selected;
    selected.implementation = std::move(implementation);
    selected.packedShape = *packedShape;
    selected.tableShape = tableShape;
    selected.activationShape = mapping.laneShape;
    selected.resources = *budget;
    legal.push_back(
        Candidate{std::move(selected), reduction->registerFactor});
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.physical.resources.peakGroups, lhs.registerFactor) <
           std::tie(rhs.physical.resources.peakGroups, rhs.registerFactor);
  });
  return std::move(legal.front().physical);
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

  struct Candidate {
    SelectedQuantI8DotPhysical physical;
    unsigned sequentialFactor = 0;
    unsigned registerFactor = 0;
  };
  llvm::SmallVector<Candidate, 16> legal;
  llvm::SmallVector<RVVVectorShape, 8> laneShapes =
      rvvShapeCandidates(target, 8);
  for (CorePhysicalMapping mapping : enumerateRVVLocalMappings(
           CoreInstructionKind::RVVWideningIntegerDot, kCoreAxisK,
           facts.semanticExtent, 8, 1, laneShapes, target)) {
    LocalImplementation implementation;
    implementation.primitive = primitive;
    implementation.mapping = mapping;
    implementation.valueShapes = {mapping.laneShape};
    if (!finalizeLocalInstructionMapping(implementation))
      continue;
    const PhysicalAxisDecomposition *reduction =
        findAxisMapping(mapping, kCoreAxisK);
    if (!reduction)
      continue;
    const unsigned logicalLanes =
        reduction->laneFactor * reduction->registerFactor;
    QuantDecodeResourceFacts resources;
    if (reduction->sequentialFactor > 1 && logicalLanes == 16) {
      resources.loadedValues = {{mapping.laneShape, 2}};
      resources.temporaryValues = {{kRVVE32M1, 2}};
    } else {
      resources.loadedValues = {
          {mapping.laneShape, 2 * reduction->registerFactor}};
      resources.temporaryValues = {
          {mapping.laneShape, 2 * reduction->registerFactor},
          {kRVVE32M1, 4 * std::max(1u, logicalLanes / 16)}};
    }
    std::optional<PhysicalResourceBudget> budget =
        calculateQuantDecodeResources(resources, target);
    if (!budget)
      continue;
    SelectedQuantI8DotPhysical selected;
    selected.implementation = std::move(implementation);
    selected.operandShape = mapping.laneShape;
    selected.resources = *budget;
    legal.push_back(Candidate{std::move(selected),
                              reduction->sequentialFactor,
                              reduction->registerFactor});
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.sequentialFactor, lhs.physical.resources.peakGroups,
                    lhs.registerFactor) <
           std::tie(rhs.sequentialFactor, rhs.physical.resources.peakGroups,
                    rhs.registerFactor);
  });
  return std::move(legal.front().physical);
}

std::optional<SelectedE2M1E8M0I8Physical>
selectE2M1E8M0I8Physical(const RISCVTargetProfile &target) {
  if (!target.hasRVV || target.vlenBits < 128 || !target.littleEndian)
    return std::nullopt;

  struct Candidate {
    SelectedE2M1E8M0I8Physical physical;
    unsigned sequentialFactor = 0;
    unsigned registerFactor = 0;
  };
  llvm::SmallVector<Candidate, 16> legal;
  llvm::SmallVector<RVVVectorShape, 8> laneShapes =
      rvvShapeCandidates(target, 8);
  for (CorePhysicalMapping mapping : enumerateRVVLocalMappings(
           CoreInstructionKind::RVVWideningIntegerDot, kCoreAxisK, 32, 8, 1,
           laneShapes, target)) {
    const PhysicalAxisDecomposition *reduction =
        findAxisMapping(mapping, kCoreAxisK);
    if (!reduction)
      continue;
    llvm::SmallVector<RVVVectorShape, 9> activationShapes(
        laneShapes.begin(), laneShapes.end());
    activationShapes.push_back({});
    for (RVVVectorShape activationShape : activationShapes) {
      LocalImplementation implementation;
      implementation.primitive = LocalPrimitiveKind::E2M1E8M0I8;
      implementation.mapping = mapping;
      implementation.valueShapes = {mapping.laneShape};
      if (activationShape)
        implementation.valueShapes.push_back(activationShape);
      if (!finalizeLocalInstructionMapping(implementation))
        continue;
      const unsigned logicalLanes =
          reduction->laneFactor * reduction->registerFactor;
      const unsigned packedRepetitions = reduction->registerFactor;
      unsigned activationRepetitions = packedRepetitions;
      RVVVectorShape selectedActivation =
          activationShape ? activationShape : mapping.laneShape;
      if (std::optional<unsigned> capacity =
              rvvLaneCapacity(selectedActivation, target))
        activationRepetitions =
            (logicalLanes + *capacity - 1) / *capacity;
      else
        continue;
      std::optional<RVVVectorShape> productShape =
          rvvShapeForSameLanes(selectedActivation, 16, target);
      if (!productShape)
        continue;
      QuantDecodeResourceFacts resources;
      resources.loadedValues = {
          {mapping.laneShape, packedRepetitions},
          {selectedActivation, activationRepetitions}};
      resources.temporaryValues = {
          {mapping.laneShape, 2 * packedRepetitions},
          {selectedActivation, 2 * activationRepetitions},
          {*productShape, activationRepetitions},
          {kRVVE32M1, 2}};
      std::optional<PhysicalResourceBudget> budget =
          calculateQuantDecodeResources(resources, target);
      if (!budget)
        continue;
      SelectedE2M1E8M0I8Physical selected;
      selected.implementation = std::move(implementation);
      selected.packedShape = mapping.laneShape;
      selected.activationShape = selectedActivation;
      selected.resources = *budget;
      legal.push_back(Candidate{std::move(selected),
                                reduction->sequentialFactor,
                                reduction->registerFactor});
    }
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.sequentialFactor, lhs.physical.resources.peakGroups,
                    lhs.registerFactor) <
           std::tie(rhs.sequentialFactor, rhs.physical.resources.peakGroups,
                    rhs.registerFactor);
  });
  return std::move(legal.front().physical);
}

std::optional<SelectedGroupedAffineI4I8Physical>
selectGroupedAffineI4I8Physical(
    const GroupedAffineI4I8CandidateFacts &facts,
    const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasWideningInteger ||
      target.vlenBits < 128 || !target.littleEndian)
    return std::nullopt;
  auto axis = [&](unsigned id) -> const LogicalAxisConstraint * {
    auto found = llvm::find_if(facts.mapping.axes,
                               [&](const LogicalAxisConstraint &candidate) {
                                 return candidate.id == id;
                               });
    return found == facts.mapping.axes.end() ? nullptr : &*found;
  };
  const LogicalAxisConstraint *rows = axis(kCoreAxisM);
  const LogicalAxisConstraint *reduction = axis(kCoreAxisK);
  const LogicalAxisConstraint *groups = axis(kCoreAxisGroup);
  const LogicalAxisConstraint *packed = axis(kCoreAxisPacked);
  if (!rows || !rows->extent || *rows->extent != 1 ||
      rows->role != LogicalAxisRole::Free || !reduction ||
      !reduction->extent || *reduction->extent != 32 ||
      reduction->role != LogicalAxisRole::Reduction || !groups ||
      !groups->extent || *groups->extent != 8 ||
      groups->role != LogicalAxisRole::Group || !packed ||
      !packed->extent || *packed->extent != 2 ||
      packed->role != LogicalAxisRole::Packed)
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

  CoreMappingProblem problem = facts.mapping;
  for (LogicalAxisConstraint &constraint : problem.axes) {
    constraint.allowLane = constraint.id == kCoreAxisK;
    constraint.requireLane = constraint.allowLane;
    constraint.registerFactors =
        constraint.id == kCoreAxisK
            ? llvm::SmallVector<unsigned, 4>{1, 2}
            : llvm::SmallVector<unsigned, 4>{1};
  }
  problem.laneSEW = 8;
  problem.laneInstruction = CoreInstructionKind::RVVWideningIntegerDot;
  problem.laneShapeCandidates = {selected.packedShape};
  struct Candidate {
    SelectedGroupedAffineI4I8Physical physical;
    unsigned sequentialFactor = 0;
  };
  llvm::SmallVector<Candidate, 4> legal;
  for (CorePhysicalMapping mapping :
       enumerateCorePhysicalMappings(problem, target)) {
    const PhysicalAxisDecomposition *reduction =
        findAxisMapping(mapping, kCoreAxisK);
    if (!reduction ||
        reduction->laneFactor * reduction->registerFactor > 32)
      continue;
    SelectedGroupedAffineI4I8Physical candidate = selected;
    candidate.implementation.primitive =
        LocalPrimitiveKind::GroupedAffineI4I8;
    candidate.implementation.mapping = mapping;
    candidate.implementation.valueShapes = {candidate.packedShape,
                                            candidate.widenedShape};
    if (!finalizeLocalInstructionMapping(candidate.implementation))
      continue;
    const bool registerAsm =
        reduction->sequentialFactor == 1 && reduction->laneFactor == 16;
    if (registerAsm) {
      PhysicalResourceRequirements resources;
      resources.reservedGroups = 0;
      resources.live = {
          {PhysicalLiveClass::Temporary, {}, 1,
           static_cast<unsigned>(target.vectorRegisters)}};
      std::optional<PhysicalResourceBudget> budget =
          calculatePhysicalResources(resources, target);
      if (!budget)
        continue;
      candidate.resources = *budget;
    } else {
      QuantDecodeResourceFacts resources;
      resources.loadedValues =
          {{candidate.packedShape, reduction->registerFactor},
           {candidate.scaleShape, 1},
           {candidate.activationShape, reduction->registerFactor},
           {candidate.activationSumShape, 1}};
      resources.temporaryValues =
          {{candidate.widenedShape, reduction->registerFactor},
           {candidate.reductionShape, 2}};
      std::optional<PhysicalResourceBudget> budget =
          calculateQuantDecodeResources(resources, target);
      if (!budget)
        continue;
      candidate.resources = *budget;
    }
    legal.push_back(
        Candidate{std::move(candidate), reduction->sequentialFactor});
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.sequentialFactor, lhs.physical.resources.peakGroups) <
           std::tie(rhs.sequentialFactor, rhs.physical.resources.peakGroups);
  });
  return std::move(legal.front().physical);
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
