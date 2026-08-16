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

std::optional<unsigned> rvvIntegerLMUL(const RVVVectorShape &shape) {
  if (!shape || shape.lmulEighths < 8 || shape.lmulEighths % 8 != 0)
    return std::nullopt;
  return static_cast<unsigned>(shape.lmulEighths / 8);
}

unsigned rvvRegisterGroups(const RVVVectorShape &shape) {
  return static_cast<unsigned>((std::max(shape.lmulEighths, 0) + 7) / 8);
}

std::optional<RVVVectorShape>
rvvShapeForSemanticLanes(unsigned sew, unsigned semanticLanes,
                         const RISCVTargetProfile &target) {
  if (!target.supportsSEW(sew) || semanticLanes == 0 || target.vlenBits <= 0)
    return std::nullopt;
  uint64_t requiredBits = static_cast<uint64_t>(sew) * semanticLanes;
  uint64_t requiredLMULEighths =
      (requiredBits * 8 + static_cast<uint64_t>(target.vlenBits) - 1) /
      static_cast<uint64_t>(target.vlenBits);
  auto legal = llvm::find_if(target.legalLMULEighths, [&](int value) {
    return static_cast<uint64_t>(value) >= requiredLMULEighths &&
           target.supportsVectorShape(sew, value);
  });
  if (legal == target.legalLMULEighths.end())
    return std::nullopt;
  return RVVVectorShape{sew, *legal};
}

std::optional<RVVVectorShape>
rvvShapeForSameLanes(const RVVVectorShape &source, unsigned resultSEW,
                     const RISCVTargetProfile &target) {
  if (!source || resultSEW == 0)
    return std::nullopt;
  int64_t scaled = static_cast<int64_t>(source.lmulEighths) * resultSEW;
  if (scaled % source.sew != 0)
    return std::nullopt;
  RVVVectorShape result{resultSEW, static_cast<int>(scaled / source.sew)};
  return target.supportsVectorShape(result.sew, result.lmulEighths)
             ? std::optional<RVVVectorShape>(result)
             : std::nullopt;
}

std::optional<unsigned> rvvMaskRatio(const RVVVectorShape &dataShape) {
  if (!dataShape || dataShape.lmulEighths <= 0)
    return std::nullopt;
  unsigned scaledSEW = dataShape.sew * 8;
  if (scaledSEW % static_cast<unsigned>(dataShape.lmulEighths) != 0)
    return std::nullopt;
  unsigned ratio = scaledSEW / static_cast<unsigned>(dataShape.lmulEighths);
  constexpr unsigned legalRatios[] = {1, 2, 4, 8, 16, 32, 64};
  return llvm::is_contained(legalRatios, ratio)
             ? std::optional<unsigned>(ratio)
             : std::nullopt;
}

std::optional<unsigned>
rvvLaneCapacity(const RVVVectorShape &shape,
                const RISCVTargetProfile &target) {
  if (!shape || target.vlenBits <= 0 ||
      !target.supportsVectorShape(shape.sew, shape.lmulEighths))
    return std::nullopt;
  const uint64_t bits = static_cast<uint64_t>(target.vlenBits) *
                        static_cast<uint64_t>(shape.lmulEighths) / 8;
  if (bits < shape.sew || bits % shape.sew != 0)
    return std::nullopt;
  return static_cast<unsigned>(bits / shape.sew);
}

static bool selectLocalImplementationLeaf(LocalImplementation &implementation) {
  const LocalImplementationParameters &parameters = implementation.parameters;
  const bool rvvRegister =
      implementation.structure ==
      LocalImplementationStructure::RVVRegisterMicrokernel;
  const bool ime = implementation.structure ==
                   LocalImplementationStructure::SpacemitIME1Fragment;
  const bool strip =
      implementation.structure == LocalImplementationStructure::RVVStripLoop;
  const bool register32E8M1OrM2 =
      rvvRegister && parameters.semanticLanes == 32 &&
      (parameters.primaryShape == RVVVectorShape{8, 8} ||
       parameters.primaryShape == RVVVectorShape{8, 16});
  const bool register32Or64E8M2 =
      rvvRegister &&
      (parameters.semanticLanes == 32 || parameters.semanticLanes == 64) &&
      parameters.primaryShape == RVVVectorShape{8, 16};
  const bool strip16E8M2 =
      strip && parameters.semanticLanes == 16 &&
      parameters.primaryShape == RVVVectorShape{8, 16};

  switch (implementation.primitive) {
  case LocalPrimitiveKind::None:
    return false;
  case LocalPrimitiveKind::F32Math:
    if (rvvRegister && parameters.primaryShape == kRVVE32M2)
      implementation.leaf = LocalImplementationLeaf::RVVF32Math;
    break;
  case LocalPrimitiveKind::SymmetricI4I8:
  case LocalPrimitiveKind::AffineI4I8:
    if (parameters.semanticLanes != 16 ||
        parameters.primaryShape != kRVVE8M1 ||
        parameters.secondaryShape != kRVVE32M4 ||
        (parameters.rowMicrotile != 1 && parameters.rowMicrotile != 4))
      break;
    if (implementation.primitive == LocalPrimitiveKind::SymmetricI4I8)
      implementation.leaf =
          ime ? (parameters.rowMicrotile == 4
                     ? LocalImplementationLeaf::IME1SymmetricI4I8M4N16
                     : LocalImplementationLeaf::IME1SymmetricI4I8N16)
              : rvvRegister
                    ? (parameters.rowMicrotile == 4
                           ? LocalImplementationLeaf::RVVSymmetricI4I8M4N16
                           : LocalImplementationLeaf::RVVSymmetricI4I8N16)
                    : LocalImplementationLeaf::None;
    else
      implementation.leaf =
          ime ? (parameters.rowMicrotile == 4
                     ? LocalImplementationLeaf::IME1AffineI4I8M4N16
                     : LocalImplementationLeaf::IME1AffineI4I8N16)
              : rvvRegister
                    ? (parameters.rowMicrotile == 4
                           ? LocalImplementationLeaf::RVVAffineI4I8M4N16
                           : LocalImplementationLeaf::RVVAffineI4I8N16)
                    : LocalImplementationLeaf::None;
    break;
  case LocalPrimitiveKind::GroupedAffineI4I8:
    if (strip && parameters.semanticLanes == 32 &&
        parameters.primaryShape == kRVVE8M1 &&
        parameters.secondaryShape == kRVVE16M2)
      implementation.leaf =
          LocalImplementationLeaf::RVVGroupedAffineI4I8Strip;
    break;
  case LocalPrimitiveKind::E2M1E8M0I8:
    if (strip && parameters.semanticLanes == 0 &&
        parameters.primaryShape == kRVVE8M1 && !parameters.secondaryShape)
      implementation.leaf = LocalImplementationLeaf::RVVE2M1E8M0I8Strip;
    else if (rvvRegister && parameters.semanticLanes == 32 &&
             parameters.primaryShape == RVVVectorShape{8, 4} &&
             parameters.secondaryShape == RVVVectorShape{8, 4})
      implementation.leaf =
          LocalImplementationLeaf::RVVE2M1E8M0I8RegisterMF2;
    else if (rvvRegister && parameters.semanticLanes == 32 &&
             parameters.primaryShape == kRVVE8M1 &&
             parameters.secondaryShape == RVVVectorShape{8, 16})
      implementation.leaf =
          LocalImplementationLeaf::RVVE2M1E8M0I8RegisterM1M2;
    break;
  case LocalPrimitiveKind::PackedI4I8:
    if (register32E8M1OrM2)
      implementation.leaf = LocalImplementationLeaf::RVVPackedI4I8Register;
    break;
  case LocalPrimitiveKind::PackedI5I8:
    if (register32E8M1OrM2)
      implementation.leaf = LocalImplementationLeaf::RVVPackedI5I8Register;
    break;
  case LocalPrimitiveKind::PackedI3GroupedI8:
    if (register32Or64E8M2)
      implementation.leaf =
          LocalImplementationLeaf::RVVPackedI3GroupedI8Register;
    break;
  case LocalPrimitiveKind::Base3TernaryI8:
    if (register32E8M1OrM2)
      implementation.leaf =
          LocalImplementationLeaf::RVVBase3TernaryI8Register;
    break;
  case LocalPrimitiveKind::PackedI2TernaryI8:
    if (register32E8M1OrM2)
      implementation.leaf =
          LocalImplementationLeaf::RVVPackedI2TernaryI8Register;
    break;
  case LocalPrimitiveKind::SignedCodebook8I8:
    if (register32E8M1OrM2 && parameters.entryWidth == 8)
      implementation.leaf =
          LocalImplementationLeaf::RVVSignedCodebook8I8Register;
    break;
  case LocalPrimitiveKind::SignedCodebook4I8:
    if (register32E8M1OrM2 && parameters.entryWidth == 4)
      implementation.leaf =
          LocalImplementationLeaf::RVVSignedCodebook4I8Register;
    break;
  case LocalPrimitiveKind::PackedU9U7CodebookI8:
    if (register32E8M1OrM2 && parameters.entryWidth == 8)
      implementation.leaf =
          LocalImplementationLeaf::RVVPackedU9U7CodebookI8Register;
    break;
  case LocalPrimitiveKind::PackedU11GridDeltaI8:
    if (register32E8M1OrM2 && parameters.entryWidth == 8)
      implementation.leaf =
          LocalImplementationLeaf::RVVPackedU11GridDeltaI8Register;
    break;
  case LocalPrimitiveKind::NibbleCodebookI8:
    if (register32E8M1OrM2)
      implementation.leaf =
          LocalImplementationLeaf::RVVNibbleCodebookI8Register;
    break;
  case LocalPrimitiveKind::IQ2SI8:
    if (strip16E8M2)
      implementation.leaf = LocalImplementationLeaf::RVVIQ2SI8Strip;
    else if (register32Or64E8M2)
      implementation.leaf = LocalImplementationLeaf::RVVIQ2SI8Register;
    break;
  case LocalPrimitiveKind::IQ3SI8:
    if (strip16E8M2)
      implementation.leaf = LocalImplementationLeaf::RVVIQ3SI8Strip;
    else if (rvvRegister && parameters.semanticLanes == 64 &&
             parameters.primaryShape == RVVVectorShape{8, 16})
      implementation.leaf = LocalImplementationLeaf::RVVIQ3SI8Register;
    break;
  case LocalPrimitiveKind::IQ1MI8:
    if (strip16E8M2)
      implementation.leaf = LocalImplementationLeaf::RVVIQ1MI8Strip;
    else if (register32Or64E8M2)
      implementation.leaf = LocalImplementationLeaf::RVVIQ1MI8Register;
    break;
  case LocalPrimitiveKind::Q6KI8:
    if (strip16E8M2)
      implementation.leaf = LocalImplementationLeaf::RVVQ6KI8Strip;
    else if (register32Or64E8M2)
      implementation.leaf = LocalImplementationLeaf::RVVQ6KI8Register;
    break;
  }
  return implementation.leaf != LocalImplementationLeaf::None;
}

RVVVectorShape rvvShape(unsigned sew, unsigned lmul) {
  return RVVVectorShape{sew, static_cast<int>(lmul * 8)};
}

llvm::SmallVector<unsigned>
integerLMULCandidates(const RISCVTargetProfile &target, unsigned sew,
                      unsigned maximum) {
  llvm::SmallVector<unsigned> candidates;
  for (int eighths : target.legalLMULEighths) {
    if (eighths < 8 || eighths % 8 != 0)
      continue;
    unsigned lmul = static_cast<unsigned>(eighths / 8);
    if (lmul <= maximum && target.supportsVectorShape(sew, eighths))
      candidates.push_back(lmul);
  }
  llvm::sort(candidates);
  return candidates;
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
selectVLASegment2Physical(bool load,
                          const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasSegmentMemory)
    return std::nullopt;
  return SelectedVLASegment2Physical{
      load ? VLASegment2AccessKind::Load : VLASegment2AccessKind::Store,
      load};
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
         facts.relaxedOrder && facts.reductionStateCount > 1);
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
  LocalImplementation implementation;
  implementation.primitive = LocalPrimitiveKind::F32Math;
  implementation.structure =
      LocalImplementationStructure::RVVRegisterMicrokernel;
  implementation.parameters.primaryShape = kRVVE32M2;
  if (!selectLocalImplementationLeaf(implementation))
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
  struct Candidate {
    SelectedI4I8FragmentPhysical physical;
    unsigned instructionCost = 0;
  };
  llvm::SmallVector<Candidate> candidates;
  auto appendCandidate = [&](LocalImplementationStructure structure,
                             unsigned instructionCost) {
    const bool ime =
        structure == LocalImplementationStructure::SpacemitIME1Fragment;
    if ((ime && !supportsIME) || (!ime && !supportsRVV))
      return;
    SelectedI4I8FragmentPhysical selected;
    selected.implementation.primitive =
        facts.affine ? LocalPrimitiveKind::AffineI4I8
                     : LocalPrimitiveKind::SymmetricI4I8;
    selected.implementation.structure = structure;
    selected.implementation.parameters.rowMicrotile = facts.rowTile;
    selected.implementation.parameters.semanticLanes = 16;
    selected.codeShape = kRVVE8M1;
    selected.activationScaleShape = kRVVE32M1;
    selected.accumulatorShape = kRVVE32M4;
    selected.implementation.parameters.primaryShape = selected.codeShape;
    selected.implementation.parameters.secondaryShape =
        selected.accumulatorShape;
    if (!selectLocalImplementationLeaf(selected.implementation))
      return;
    for (RVVVectorShape shape : {selected.codeShape,
                                 selected.activationScaleShape,
                                 selected.accumulatorShape})
      if (!target.supportsVectorShape(shape.sew, shape.lmulEighths))
        return;

    selected.resources.architecturalGroups = target.vectorRegisters;
    selected.resources.valueGroups = facts.rowTile == 4 ? 20 : 4;
    selected.resources.memoryGroups = facts.rowTile == 4 ? 5 : 2;
    selected.resources.primitiveGroups =
        ime ? 28 : facts.rowTile == 4 ? (facts.affine ? 25 : 24)
                                      : (facts.affine ? 13 : 12);
    selected.resources.peakGroups = selected.resources.primitiveGroups + 1;
    if (selected.resources.peakGroups >=
        static_cast<unsigned>(target.vectorRegisters))
      return;
    candidates.push_back(Candidate{std::move(selected), instructionCost});
  };
  appendCandidate(LocalImplementationStructure::RVVRegisterMicrokernel,
                  facts.rowTile == 4 ? 16 : 4);
  appendCandidate(LocalImplementationStructure::SpacemitIME1Fragment, 1);
  const int64_t requested = config.structures.i4I8FragmentImplementation;
  if (requested != 0)
    llvm::erase_if(candidates, [&](const Candidate &candidate) {
      const bool ime = candidate.physical.implementation.structure ==
                       LocalImplementationStructure::SpacemitIME1Fragment;
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
      target.vectorRegisters <= 0)
    return std::nullopt;
  if (facts.requiredDataShape &&
      (facts.requiredDataShape.sew != facts.dataSEW ||
       !target.supportsVectorShape(facts.requiredDataShape.sew,
                                   facts.requiredDataShape.lmulEighths)))
    return std::nullopt;

  std::optional<unsigned> requiredLMUL;
  for (unsigned lmul : facts.requiredLMULs) {
    if (lmul == 0 ||
        !target.supportsVectorShape(facts.dataSEW,
                                    static_cast<int>(lmul * 8)) ||
        (requiredLMUL && *requiredLMUL != lmul))
      return std::nullopt;
    requiredLMUL = lmul;
  }

  llvm::SmallVector<unsigned> candidates =
      integerLMULCandidates(target, facts.dataSEW);
  if (requiredLMUL) {
    candidates = {*requiredLMUL};
  } else {
    unsigned desiredLanes = 16;
    const bool hasReductionState = llvm::any_of(
        facts.states, [](const VLAStateResourceFact &state) {
          return state.stripUpdate == VLAStateStripUpdate::AddReduction ||
                 state.stripUpdate == VLAStateStripUpdate::MaxReduction ||
                 state.stripUpdate ==
                     VLAStateStripUpdate::WideningAddReduction;
        });
    const bool hasOrderedScan = llvm::any_of(
        facts.states, [](const VLAStateResourceFact &state) {
          return state.stripUpdate == VLAStateStripUpdate::InclusiveAddScan ||
                 state.stripUpdate ==
                     VLAStateStripUpdate::SegmentedInclusiveAddScan;
        });
    if (facts.dataSEW == 16 || facts.hasFloatCast || hasReductionState ||
        facts.hasNarrow)
      desiredLanes = 32;
    if (hasOrderedScan || facts.hasF32Division)
      desiredLanes = 8;
    llvm::sort(candidates, [&](unsigned lhs, unsigned rhs) {
      auto score = [&](unsigned candidate) {
        const std::optional<unsigned> lanes =
            rvvLaneCapacity(rvvShape(facts.dataSEW, candidate), target);
        if (!lanes)
          return std::numeric_limits<unsigned>::max();
        unsigned value = static_cast<unsigned>(std::abs(
            static_cast<int>(*lanes) - static_cast<int>(desiredLanes)));
        value += facts.stridedAccesses * candidate;
        value += facts.indexedAccesses * 2 * candidate;
        return value;
      };
      unsigned lhsScore = score(lhs);
      unsigned rhsScore = score(rhs);
      return lhsScore != rhsScore ? lhsScore < rhsScore : lhs > rhs;
    });
  }

  for (unsigned candidate : candidates) {
    RVVVectorShape dataShape = rvvShape(facts.dataSEW, candidate);
    if (facts.requiredDataShape && dataShape != facts.requiredDataShape)
      continue;
    const bool needsIndexShape =
        facts.hasAffinePredicate || facts.hasIndexVector;
    if (needsIndexShape &&
        (candidate * 8 * static_cast<unsigned>(target.xlen)) %
                facts.dataSEW !=
            0)
      continue;
    if (needsIndexShape &&
        candidate * static_cast<unsigned>(target.xlen) / facts.dataSEW > 8)
      continue;
    if (needsIndexShape &&
        !target.supportsVectorShape(
            static_cast<unsigned>(target.xlen),
            static_cast<int>(candidate * static_cast<unsigned>(target.xlen) /
                             facts.dataSEW * 8)))
      continue;
    if (facts.indexedAccesses != 0 &&
        ((candidate * facts.maxIndexedOffsetSEW) % facts.dataSEW != 0 ||
         candidate * facts.maxIndexedOffsetSEW / facts.dataSEW > 8))
      continue;

    bool elementShapesLegal =
        llvm::all_of(facts.accessElementSEWs, [&](unsigned sew) {
          if (sew == 16 && !target.hasVectorF16)
            return false;
          if (sew == 0 || (candidate * 8 * sew) % facts.dataSEW != 0)
            return false;
          return target.supportsVectorShape(
              sew, static_cast<int>(candidate * 8 * sew / facts.dataSEW));
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
    if ((facts.segmentLoadPairs != 0 || facts.segmentStorePairs != 0) &&
        !target.supportsSegmentVectorMemory(2, 32,
                                            static_cast<int>(candidate * 8)))
      continue;

    unsigned carriedStateGroups = 0;
    unsigned transientStateGroups = 0;
    for (const VLAStateResourceFact &state : facts.states) {
      switch (state.stripUpdate) {
      case VLAStateStripUpdate::InclusiveAddScan:
        transientStateGroups =
            std::max(transientStateGroups, 3 * candidate + 1);
        break;
      case VLAStateStripUpdate::SegmentedInclusiveAddScan:
        transientStateGroups =
            std::max(transientStateGroups, 4 * candidate + 2);
        break;
      case VLAStateStripUpdate::AddReduction:
      case VLAStateStripUpdate::MaxReduction:
        if (state.carry == VLAStateCarryRepresentation::Vector &&
            state.wholeVLALifetime) {
          carriedStateGroups += candidate;
          transientStateGroups =
              std::max(transientStateGroups, std::max(candidate, 2u));
        } else {
          transientStateGroups =
              std::max(transientStateGroups, candidate + 2);
        }
        break;
      case VLAStateStripUpdate::WideningAddReduction:
        transientStateGroups =
            std::max(transientStateGroups,
                     std::max(1u, candidate / 4) + 1);
        break;
      case VLAStateStripUpdate::ArgMaxSummary:
        transientStateGroups =
            std::max(transientStateGroups, candidate + 2);
        break;
      case VLAStateStripUpdate::OnlineSoftmaxSummary:
        transientStateGroups =
            std::max(transientStateGroups, 3 * candidate + 2);
        break;
      }
    }
    unsigned stateGroups = carriedStateGroups + transientStateGroups;
    unsigned primitiveGroups = stateGroups;

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
      selected.resources.architecturalGroups = target.vectorRegisters;
      selected.resources.valueGroups = rvvRegisterGroups(dataShape);
      selected.resources.primitiveGroups =
          rvvRegisterGroups(dataShape) +
          rvvRegisterGroups(*intermediateShape) +
          rvvRegisterGroups(*resultShape);
      selected.resources.peakGroups =
          selected.resources.primitiveGroups + 1;
      primitiveGroups =
          std::max(primitiveGroups, selected.resources.primitiveGroups);
      narrow = selected;
    }

    for (const PhysicalResourceBudget &resource :
         facts.localPrimitiveResources)
      primitiveGroups =
          std::max(primitiveGroups, resource.primitiveGroups);

    unsigned indexGroups = 0;
    if (facts.indexedAccesses != 0)
      indexGroups =
          candidate * facts.maxIndexedOffsetSEW / facts.dataSEW;

    auto groupsFor = [&](unsigned sew, unsigned count) {
      if (count == 0)
        return 0u;
      unsigned scaled = candidate * 8 * sew;
      if (scaled % facts.dataSEW != 0)
        return static_cast<unsigned>(target.vectorRegisters);
      int lmulEighths = static_cast<int>(scaled / facts.dataSEW);
      if (!target.supportsVectorShape(sew, lmulEighths))
        return static_cast<unsigned>(target.vectorRegisters);
      return count * rvvRegisterGroups(RVVVectorShape{sew, lmulEighths});
    };
    unsigned valueGroups = 0;
    unsigned predicateGroups = 0;
    for (const VLAValueLifetimeSnapshot &snapshot : facts.lifetimes) {
      unsigned snapshotGroups =
          groupsFor(32, snapshot.f32) + groupsFor(16, snapshot.f16) +
          groupsFor(8, snapshot.byte) + groupsFor(32, snapshot.u32) +
          groupsFor(static_cast<unsigned>(target.xlen), snapshot.index) +
          snapshot.mask;
      valueGroups = std::max(valueGroups, snapshotGroups);
      predicateGroups = std::max(predicateGroups, snapshot.mask);
    }

    unsigned memoryGroups = indexGroups;
    memoryGroups +=
        2 * candidate *
        std::max(facts.segmentLoadPairs, facts.segmentStorePairs);

    unsigned lookupGroups = 0;
    if (facts.lookupCount != 0) {
      std::optional<RVVVectorShape> codeShape =
          rvvShapeForSameLanes(dataShape, 8, target);
      std::optional<RVVVectorShape> index16Shape =
          rvvShapeForSameLanes(dataShape, 16, target);
      std::optional<RVVVectorShape> index32Shape =
          rvvShapeForSameLanes(dataShape, 32, target);
      if (!codeShape || !index16Shape || !index32Shape)
        continue;
      unsigned conversionGroups = rvvRegisterGroups(*codeShape) +
                                  rvvRegisterGroups(*index16Shape) +
                                  rvvRegisterGroups(*index32Shape);
      unsigned gatherGroups = 3 * rvvRegisterGroups(*index32Shape);
      lookupGroups = std::max(conversionGroups, gatherGroups);
      primitiveGroups = std::max(primitiveGroups, lookupGroups);
    }

    unsigned sharedGroups = valueGroups + memoryGroups;
    unsigned requiredGroups =
        std::max(sharedGroups, primitiveGroups + memoryGroups);
    unsigned peakGroups = requiredGroups + 1;
    for (const PhysicalResourceBudget &resource :
         facts.localPrimitiveResources) {
      unsigned handoffGroups = memoryGroups + predicateGroups;
      peakGroups =
          std::max(peakGroups,
                   std::max(resource.peakGroups,
                            resource.primitiveGroups + handoffGroups + 1));
    }
    if (peakGroups >= static_cast<unsigned>(target.vectorRegisters))
      continue;

    SelectedVLAEntityPhysical selected;
    selected.dataShape = dataShape;
    if (needsIndexShape)
      selected.indexShape =
          rvvShapeForSameLanes(dataShape, target.xlen, target)
              .value_or(RVVVectorShape{});
    selected.maskRatio = rvvMaskRatio(dataShape).value_or(0);
    if (selected.maskRatio == 0)
      continue;
    selected.narrow = narrow;
    selected.resources.architecturalGroups = target.vectorRegisters;
    selected.resources.valueGroups = valueGroups;
    selected.resources.memoryGroups = memoryGroups;
    selected.resources.indexGroups = indexGroups;
    selected.resources.predicateGroups = predicateGroups;
    selected.resources.stateGroups = stateGroups;
    selected.resources.primitiveGroups = primitiveGroups;
    selected.resources.peakGroups = peakGroups;
    return selected;
  }
  return std::nullopt;
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
  auto groups = [&](llvm::ArrayRef<RVVShapeMultiplicity> values)
      -> std::optional<unsigned> {
    unsigned total = 0;
    for (const RVVShapeMultiplicity &value : values) {
      if (!value.shape || value.count == 0 ||
          !target.supportsVectorShape(value.shape.sew,
                                      value.shape.lmulEighths))
        return std::nullopt;
      total += value.count * rvvRegisterGroups(value.shape);
    }
    return total;
  };
  std::optional<unsigned> loaded = groups(facts.loadedValues);
  std::optional<unsigned> carried = groups(facts.carriedValues);
  std::optional<unsigned> indices = groups(facts.indexValues);
  std::optional<unsigned> temporaries = groups(facts.temporaryValues);
  if (!loaded || !carried || !indices || !temporaries)
    return std::nullopt;

  PhysicalResourceBudget resources;
  resources.architecturalGroups = target.vectorRegisters;
  resources.valueGroups = *loaded + *carried;
  resources.memoryGroups = *loaded;
  resources.indexGroups = *indices;
  resources.predicateGroups = facts.predicateGroups;
  resources.primitiveGroups =
      resources.valueGroups + resources.indexGroups + *temporaries;
  resources.peakGroups = resources.primitiveGroups + resources.predicateGroups;
  if (resources.peakGroups >=
      static_cast<unsigned>(target.vectorRegisters))
    return std::nullopt;
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
  selected.implementation.primitive =
      facts.semantic == TernaryI8DotSemantic::Base3Digits
          ? LocalPrimitiveKind::Base3TernaryI8
          : LocalPrimitiveKind::PackedI2TernaryI8;
  selected.implementation.structure =
      LocalImplementationStructure::RVVRegisterMicrokernel;
  selected.implementation.parameters.semanticLanes = 32;
  selected.implementation.parameters.primaryShape = *byte32;
  selected.implementation.parameters.secondaryShape = *byte16;
  if (!selectLocalImplementationLeaf(selected.implementation))
    return std::nullopt;
  selected.decode = facts.semantic == TernaryI8DotSemantic::Base3Digits
                        ? TernaryDecodeTopology::Base3Digits
                        : TernaryDecodeTopology::PackedI2Fields;
  selected.primaryWidening =
      RVVWideningChain{*byte32, *widened32, kRVVE32M1, 32, 2};
  selected.secondaryWidening =
      RVVWideningChain{*byte16, *widened16, kRVVE32M1, 16, 1};

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
  selected.implementation.primitive = facts.primitive;
  selected.implementation.structure =
      LocalImplementationStructure::RVVRegisterMicrokernel;
  selected.implementation.parameters.semanticLanes = 32;
  selected.implementation.parameters.entryWidth = facts.entryWidth;
  selected.implementation.parameters.primaryShape = *activationShape;
  selected.implementation.parameters.secondaryShape = *tableShape;
  if (!selectLocalImplementationLeaf(selected.implementation))
    return std::nullopt;
  selected.gather.codeCount = codeCount;
  selected.gather.indexShift = facts.entryWidth == 8 ? 3 : 2;
  selected.gather.tableSEW = tableSEW;
  selected.gather.gatherByteStride = facts.entryWidth;
  selected.codeShape = *codeShape;
  selected.indexShape = *indexShape;
  selected.tableShape = *tableShape;
  selected.widening =
      RVVWideningChain{*activationShape, *productShape, kRVVE32M1, 32, 2};
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
  selected.implementation.primitive = LocalPrimitiveKind::NibbleCodebookI8;
  selected.implementation.structure =
      LocalImplementationStructure::RVVRegisterMicrokernel;
  selected.implementation.parameters.semanticLanes = 32;
  selected.implementation.parameters.primaryShape = *activationShape;
  selected.implementation.parameters.secondaryShape = *packedShape;
  if (!selectLocalImplementationLeaf(selected.implementation))
    return std::nullopt;
  selected.packedShape = *packedShape;
  selected.tableShape = combined ? *activationShape : *packedShape;
  selected.widening =
      RVVWideningChain{*activationShape, *productShape, kRVVE32M1, 32, 2};
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
  QuantDecodeTopology decode = QuantDecodeTopology::PackedNibble;
  switch (facts.semantic) {
  case QuantI8DotSemantic::PackedI4:
    primitive = LocalPrimitiveKind::PackedI4I8;
    decode = QuantDecodeTopology::PackedNibble;
    break;
  case QuantI8DotSemantic::PackedI5:
    primitive = LocalPrimitiveKind::PackedI5I8;
    decode = QuantDecodeTopology::PackedNibbleHighBit;
    break;
  case QuantI8DotSemantic::PackedI3Grouped:
    primitive = LocalPrimitiveKind::PackedI3GroupedI8;
    decode = QuantDecodeTopology::GroupedBitPlane;
    break;
  case QuantI8DotSemantic::IQ2S:
    primitive = LocalPrimitiveKind::IQ2SI8;
    decode = QuantDecodeTopology::GridSignLookup;
    break;
  case QuantI8DotSemantic::IQ3S:
    primitive = LocalPrimitiveKind::IQ3SI8;
    decode = QuantDecodeTopology::GridSignHighBitLookup;
    break;
  case QuantI8DotSemantic::IQ1M:
    primitive = LocalPrimitiveKind::IQ1MI8;
    decode = QuantDecodeTopology::GridDeltaLookup;
    break;
  case QuantI8DotSemantic::Q6K:
    primitive = LocalPrimitiveKind::Q6KI8;
    decode = QuantDecodeTopology::SplitBitPlane;
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
      selected.implementation.primitive = primitive;
      selected.implementation.structure =
          LocalImplementationStructure::RVVRegisterMicrokernel;
      selected.implementation.parameters.semanticLanes = lanes;
      selected.implementation.parameters.primaryShape = *byteShape;
      if (!selectLocalImplementationLeaf(selected.implementation))
        continue;
      selected.decode = decode;
      selected.widening = RVVWideningChain{*byteShape, *productShape,
                                           kRVVE32M1, lanes, segments};
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
  selected.implementation.primitive = primitive;
  selected.implementation.structure = LocalImplementationStructure::RVVStripLoop;
  selected.implementation.parameters.semanticLanes = 16;
  selected.implementation.parameters.primaryShape = stripShape;
  if (!selectLocalImplementationLeaf(selected.implementation))
    return std::nullopt;
  selected.decode = decode;
  selected.widening = RVVWideningChain{stripShape, RVVVectorShape{16, 32},
                                       kRVVE32M1, 16, 1};
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
    selected.implementation.primitive = LocalPrimitiveKind::E2M1E8M0I8;
    selected.implementation.structure =
        LocalImplementationStructure::RVVRegisterMicrokernel;
    selected.implementation.parameters.semanticLanes = 32;
    selected.implementation.parameters.primaryShape = candidate.packed;
    selected.implementation.parameters.secondaryShape = candidate.activation;
    if (!selectLocalImplementationLeaf(selected.implementation))
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
  selected.implementation.primitive = LocalPrimitiveKind::E2M1E8M0I8;
  selected.implementation.structure = LocalImplementationStructure::RVVStripLoop;
  selected.implementation.parameters.semanticLanes = 0;
  selected.implementation.parameters.primaryShape = stripShape;
  if (!selectLocalImplementationLeaf(selected.implementation))
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
  selected.implementation.primitive = LocalPrimitiveKind::GroupedAffineI4I8;
  selected.implementation.structure = LocalImplementationStructure::RVVStripLoop;
  selected.implementation.parameters.semanticLanes = 32;
  selected.implementation.parameters.primaryShape = kRVVE8M1;
  selected.implementation.parameters.secondaryShape = kRVVE16M2;
  if (!selectLocalImplementationLeaf(selected.implementation))
    return std::nullopt;
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
  PhysicalResourceBudget resources;
  resources.architecturalGroups = target.vectorRegisters;
  resources.valueGroups =
      facts.accumulatorVectors * rvvRegisterGroups(facts.accumulatorShape);
  resources.memoryGroups =
      (facts.lhsVectorsPerWindow + facts.rhsVectorsPerWindow) *
      facts.loadWindow * rvvRegisterGroups(facts.inputShape);
  resources.predicateGroups = facts.predicateGroups;
  resources.stateGroups = facts.stateGroups;
  resources.primitiveGroups = resources.valueGroups + resources.memoryGroups;
  resources.peakGroups = resources.primitiveGroups + resources.predicateGroups +
                         resources.stateGroups + facts.handoffGroups + 1;
  if (resources.peakGroups >
      static_cast<unsigned>(target.vectorRegisters))
    return std::nullopt;
  return resources;
}

std::optional<SelectedF32DotPhysical>
selectF32DotPhysicalConfig(const F32DotCandidateFacts &facts,
                           const RISCVTargetProfile &target,
                           const RISCVBackendConfig &config) {
  auto legalUnroll = [](unsigned value) {
    return value == 1 || value == 2 || value == 4;
  };
  llvm::SmallVector<unsigned> lmulCandidates =
      integerLMULCandidates(target, 32);
  llvm::SmallVector<unsigned> unrollCandidates = {1, 2, 4};
  if (config.parameters.dotLMUL != 0)
    lmulCandidates = {static_cast<unsigned>(config.parameters.dotLMUL)};
  if (config.parameters.dotKUnroll != 0)
    unrollCandidates = {
        static_cast<unsigned>(config.parameters.dotKUnroll)};

  const unsigned desiredLanes =
      facts.model == F32DotResourceModel::VLAFreeAxis
          ? (facts.rowTile == 1 ? 16 : 8)
          : (facts.rowTile >= 8 ? 4 : 8);
  unsigned preferredUnroll = 1;
  bool costlyHandoff = facts.materializedInit || facts.reductionPredicate ||
                       facts.indexedOperands != 0;
  if (!costlyHandoff && facts.reductionExtent)
    preferredUnroll = *facts.reductionExtent >= 64
                          ? 4
                          : *facts.reductionExtent >= 32 ? 2 : 1;

  struct Candidate {
    F32DotParameters parameters;
    PhysicalResourceBudget resources;
    unsigned tailPenalty = 0;
    unsigned memoryPenalty = 0;
    unsigned lanePenalty = 0;
    unsigned unrollPenalty = 0;
    unsigned peakGroups = 0;
  };
  llvm::SmallVector<Candidate> legal;
  for (unsigned lmul : lmulCandidates) {
    const RVVVectorShape vectorShape = rvvShape(32, lmul);
    const std::optional<unsigned> laneCapacity =
        rvvLaneCapacity(vectorShape, target);
    if (!laneCapacity)
      continue;
    for (unsigned unroll : unrollCandidates) {
      if (!legalUnroll(unroll))
        continue;
      unsigned streamedOperands =
          std::max(1u, facts.unitStrideOperands + facts.stridedOperands +
                           facts.indexedOperands);
      std::optional<PhysicalResourceBudget> resources =
          calculateDenseMicrokernelResources(
              {vectorShape, vectorShape, facts.rowTile, streamedOperands, 0,
               unroll, facts.predicateGroups, facts.stateGroups,
               facts.handoffGroups},
              target);
      if (!resources)
        continue;
      legal.push_back(Candidate{
          F32DotParameters{lmul, unroll}, *resources,
          static_cast<unsigned>(facts.reductionExtent &&
                                *facts.reductionExtent % unroll != 0),
          facts.stridedOperands * unroll +
              2 * facts.indexedOperands * unroll + facts.predicateGroups +
              facts.handoffGroups,
          static_cast<unsigned>(std::abs(static_cast<int>(*laneCapacity) -
                                         static_cast<int>(desiredLanes))),
          static_cast<unsigned>(std::abs(static_cast<int>(unroll) -
                                         static_cast<int>(preferredUnroll))),
          resources->peakGroups});
    }
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.tailPenalty, lhs.memoryPenalty, lhs.lanePenalty,
                    lhs.unrollPenalty, lhs.peakGroups) <
           std::tie(rhs.tailPenalty, rhs.memoryPenalty, rhs.lanePenalty,
                    rhs.unrollPenalty, rhs.peakGroups);
  });
  const F32DotStructure structure =
      facts.model == F32DotResourceModel::LocalRow
          ? F32DotStructure::RVVLocalRowMicrokernel
          : facts.vlaVectorFreeAxis ? F32DotStructure::RVVVLAVectorDot
                                    : F32DotStructure::RVVVLAMicrotile;
  const DenseVectorOrganization vectorOrganization =
      structure == F32DotStructure::RVVVLAVectorDot
          ? DenseVectorOrganization::FreeMAxis
          : structure == F32DotStructure::RVVVLAMicrotile
                ? DenseVectorOrganization::FreeNAxis
                : DenseVectorOrganization::ReductionAxis;
  return SelectedF32DotPhysical{
      structure, vectorOrganization, DenseLoadSchedule::Streamed,
      legal.front().parameters,
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
  const unsigned preferredRows =
      std::min(facts.rowTile, std::max(1u, baseInputLanes / 4));
  llvm::SmallVector<unsigned> rowCandidates;
  if (config.parameters.f16RowMicrotile != 0) {
    rowCandidates.push_back(
        static_cast<unsigned>(config.parameters.f16RowMicrotile));
  } else {
    for (unsigned rows = facts.rowTile; rows > 0; --rows)
      if (facts.rowTile % rows == 0)
        rowCandidates.push_back(rows);
    auto preferred = llvm::find(rowCandidates, preferredRows);
    if (preferred != rowCandidates.end())
      std::rotate(rowCandidates.begin(), preferred, std::next(preferred));
  }

  llvm::SmallVector<unsigned> lmulCandidates =
      integerLMULCandidates(target, 16, 4);
  if (config.parameters.f16InputLMUL != 0)
    lmulCandidates = {
        static_cast<unsigned>(config.parameters.f16InputLMUL)};
  llvm::SmallVector<unsigned> unrollCandidates = {1, 2, 4};
  if (config.parameters.f16KUnroll != 0)
    unrollCandidates = {
        static_cast<unsigned>(config.parameters.f16KUnroll)};
  llvm::SmallVector<unsigned> bufferCandidates = {1, 2};
  if (config.parameters.f16LoadBufferCount != 0)
    bufferCandidates = {
        static_cast<unsigned>(config.parameters.f16LoadBufferCount)};

  llvm::SmallVector<unsigned> columnCandidates;
  if (config.parameters.f16ColumnMicrotile != 0) {
    columnCandidates.push_back(
        static_cast<unsigned>(config.parameters.f16ColumnMicrotile));
  } else {
    columnCandidates.push_back(1);
    if (facts.columnTile >= 2 && facts.columnTile % 2 == 0)
      columnCandidates.insert(columnCandidates.begin(), 2);
  }

  const unsigned preferredLMUL = 1;
  const unsigned preferredUnroll =
      facts.reductionTile < 32 ? 1 : baseInputLanes >= 16 ? 4 : 2;
  const unsigned preferredBuffers = facts.reductionTile >= 64 ? 2 : 1;
  const unsigned preferredColumns =
      baseInputLanes <= 8 && facts.columnTile >= 2 &&
              facts.columnTile % 2 == 0
          ? 2
          : 1;
  struct Candidate {
    DenseLoadSchedule loadSchedule;
    F16MatmulParameters parameters;
    PhysicalResourceBudget resources;
    unsigned tailPenalty = 0;
    unsigned rowPenalty = 0;
    unsigned columnPenalty = 0;
    unsigned lmulPenalty = 0;
    unsigned unrollPenalty = 0;
    unsigned bufferPenalty = 0;
  };
  llvm::SmallVector<Candidate> legal;
  for (unsigned rows : rowCandidates) {
    if (rows == 0 || rows > facts.rowTile || facts.rowTile % rows != 0)
      continue;
    for (unsigned columns : columnCandidates) {
      if (columns == 0 || columns > facts.columnTile ||
          facts.columnTile % columns != 0)
        continue;
      for (unsigned inputLMUL : lmulCandidates) {
      if (!target.supportsVectorShape(16, static_cast<int>(inputLMUL * 8)) ||
          !target.supportsVectorShape(32,
                                      static_cast<int>(2 * inputLMUL * 8)))
        continue;
      unsigned computeLMUL = 2 * inputLMUL;
      const std::optional<unsigned> laneCapacity =
          rvvLaneCapacity(rvvShape(16, inputLMUL), target);
      if (!laneCapacity)
        continue;
      const unsigned lanes = *laneCapacity;
      for (unsigned unroll : unrollCandidates) {
        if (unroll != 1 && unroll != 2 && unroll != 4)
          continue;
        for (unsigned buffers : bufferCandidates) {
          if ((buffers != 1 && buffers != 2) ||
              (buffers == 2 && unroll < 2))
            continue;
          F16MatmulParameters parameters;
          parameters.rowMicrotile = rows;
          parameters.columnMicrotile = columns;
          parameters.inputLMUL = inputLMUL;
          parameters.kUnroll = unroll;
          parameters.loadBufferCount = buffers;
          DenseLoadSchedule loadSchedule =
              buffers == 2 ? DenseLoadSchedule::RegisterDoubleBuffered
                           : DenseLoadSchedule::Streamed;
          std::optional<PhysicalResourceBudget> resources =
              calculateDenseMicrokernelResources(
                  {rvvShape(16, inputLMUL), rvvShape(32, computeLMUL),
                   rows * columns,
                   loadSchedule ==
                           DenseLoadSchedule::RegisterDoubleBuffered
                       ? rows
                       : 1,
                   columns,
                   loadSchedule ==
                           DenseLoadSchedule::RegisterDoubleBuffered
                       ? 2u
                       : 1u},
                  target);
          if (!resources)
            continue;
          legal.push_back(Candidate{
              loadSchedule,
              parameters,
              *resources,
              static_cast<unsigned>(facts.reductionTile % (unroll * lanes) !=
                                    0),
              static_cast<unsigned>(std::abs(
                  static_cast<int>(rows) -
                  static_cast<int>(preferredRows))),
              static_cast<unsigned>(std::abs(
                  static_cast<int>(columns) -
                  static_cast<int>(preferredColumns))),
              static_cast<unsigned>(std::abs(static_cast<int>(inputLMUL) -
                                             static_cast<int>(preferredLMUL))),
              static_cast<unsigned>(std::abs(static_cast<int>(unroll) -
                                             static_cast<int>(preferredUnroll))),
              static_cast<unsigned>(std::abs(
                  static_cast<int>(buffers) -
                  static_cast<int>(preferredBuffers)))});
      }
    }
  }
    }
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
  return SelectedF16MatmulPhysical{
      DenseVectorOrganization::ReductionAxis, legal.front().loadSchedule,
      legal.front().parameters, legal.front().resources};
}

} // namespace weft::riscv_internal
