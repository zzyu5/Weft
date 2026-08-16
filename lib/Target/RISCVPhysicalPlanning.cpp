#include "RISCVPhysicalPlanning.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
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

std::optional<SelectedI4I8FragmentPhysical>
selectI4I8FragmentPhysical(const I4I8FragmentCandidateFacts &facts,
                           const RISCVTargetProfile &target) {
  if (facts.rowTile != 1 && facts.rowTile != 4)
    return std::nullopt;
  SelectedI4I8FragmentPhysical selected;
  if (facts.rowTile == 4 && target.supportsSpacemitIME1I4I8M4N16K32())
    selected.realization =
        I4I8FragmentRealization::SpacemiTIME1M4N16K32;
  else if (facts.rowTile == 1 &&
           target.supportsSpacemitIME1I4I8N16K32())
    selected.realization = I4I8FragmentRealization::SpacemiTIME1N16K32;
  else {
  bool supportsRVV = target.hasF && target.hasVectorF16 &&
                     target.hasWideningInteger && target.hasWideningFloat &&
                     target.supportsVLENAtLeast(128) &&
                     target.supportsVectorShape(8, 8) &&
                     target.supportsVectorShape(16, 16) &&
                     target.supportsVectorShape(32, 32);
  if (!supportsRVV)
    return std::nullopt;
    selected.realization = facts.rowTile == 4
                               ? I4I8FragmentRealization::RVVM4N16K32
                               : I4I8FragmentRealization::RVVN16K32;
  }

  selected.codeShape = facts.rowTile == 4
                           ? rvvShape(8, target.vlenBits >= 256 ? 4 : 8)
                           : selected.realization ==
                                     I4I8FragmentRealization::SpacemiTIME1N16K32
                                 ? kRVVE8MF4
                                 : rvvShape(8, 2);
  selected.activationScaleShape = kRVVE32M1;
  selected.accumulatorShape = kRVVE32M4;
  for (RVVVectorShape shape : {selected.codeShape,
                               selected.activationScaleShape,
                               selected.accumulatorShape})
    if (!target.supportsVectorShape(shape.sew, shape.lmulEighths))
      return std::nullopt;

  bool ime = selected.realization ==
                 I4I8FragmentRealization::SpacemiTIME1N16K32 ||
             selected.realization ==
                 I4I8FragmentRealization::SpacemiTIME1M4N16K32;
  selected.resources.architecturalGroups = target.vectorRegisters;
  selected.resources.valueGroups = facts.rowTile == 4 ? 20 : 4;
  selected.resources.memoryGroups = facts.rowTile == 4 ? 5 : 2;
  selected.resources.primitiveGroups =
      ime ? 28 : facts.rowTile == 4 ? (facts.affine ? 25 : 24)
                                    : (facts.affine ? 13 : 12);
  selected.resources.peakGroups = selected.resources.primitiveGroups + 1;
  if (selected.resources.peakGroups >=
      static_cast<unsigned>(target.vectorRegisters))
    return std::nullopt;
  return selected;
}

std::optional<VLAStatePlacement>
selectReductionStatePlacement(const ReductionStatePlacementFacts &facts,
                              const RISCVTargetProfile &target,
                              const RISCVBackendConfig &config) {
  if (config.reductionStatePlacement < 0 ||
      config.reductionStatePlacement > 2)
    return std::nullopt;
  if (config.reductionStatePlacement == 2)
    return facts.relaxedOrder
               ? std::optional<VLAStatePlacement>(
                     VLAStatePlacement::VectorCarry)
               : std::nullopt;
  if (config.reductionStatePlacement == 1)
    return VLAStatePlacement::ScalarCarry;
  bool vectorCarry = facts.relaxedOrder;
  if (target.vlenBits >= 256 && facts.reductionStateCount == 1)
    vectorCarry = false;
  return vectorCarry ? VLAStatePlacement::VectorCarry
                     : VLAStatePlacement::ScalarCarry;
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

std::optional<SelectedTernaryI8DotPhysical>
selectTernaryI8DotPhysical(const TernaryI8DotCandidateFacts &facts,
                           const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasWideningInteger || !target.littleEndian ||
      (target.vlenBits != 128 && target.vlenBits != 256))
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

  SelectedTernaryI8DotPhysical selected;
  selected.realization =
      target.vlenBits == 128
          ? TernaryI8DotRealization::RVVVLEN128LocalBlockDot
          : TernaryI8DotRealization::RVVVLEN256LocalBlockDot;
  selected.byteShape32 = *byte32;
  selected.byteShape16 = *byte16;
  selected.widenedShape32 = *widened32;
  selected.widenedShape16 = *widened16;
  selected.reductionShape = kRVVE32M1;

  unsigned byteGroups = rvvRegisterGroups(*byte32);
  unsigned widenedGroups = rvvRegisterGroups(*widened32);
  selected.resources.architecturalGroups = target.vectorRegisters;
  selected.resources.valueGroups = byteGroups + widenedGroups;
  selected.resources.memoryGroups = 2 * byteGroups;
  switch (facts.semantic) {
  case TernaryI8DotSemantic::Base3Digits:
    selected.resources.primitiveGroups = byteGroups + 3 * widenedGroups + 1;
    break;
  case TernaryI8DotSemantic::PackedI2Fields:
    selected.resources.primitiveGroups = 3 * byteGroups + widenedGroups + 1;
    break;
  }
  selected.resources.peakGroups = selected.resources.primitiveGroups;
  if (selected.resources.peakGroups >=
      static_cast<unsigned>(target.vectorRegisters))
    return std::nullopt;
  return selected;
}

std::optional<SelectedCodebookGatherI8Physical>
selectCodebookGatherI8Physical(const CodebookGatherI8CandidateFacts &facts,
                               const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasIndexedMemory ||
      !target.hasWideningInteger || !target.littleEndian ||
      (target.vlenBits != 128 && target.vlenBits != 256) ||
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

  SelectedCodebookGatherI8Physical selected;
  selected.realization =
      target.vlenBits == 128
          ? CodebookGatherI8Realization::RVVVLEN128GatherDot
          : CodebookGatherI8Realization::RVVVLEN256GatherDot;
  selected.entryWidth = facts.entryWidth;
  selected.codeShape = *codeShape;
  selected.indexShape = *indexShape;
  selected.tableShape = *tableShape;
  selected.activationShape = *activationShape;
  selected.productShape = *productShape;
  selected.reductionShape = kRVVE32M1;
  selected.resources.architecturalGroups = target.vectorRegisters;
  selected.resources.valueGroups =
      rvvRegisterGroups(*tableShape) + rvvRegisterGroups(*activationShape);
  selected.resources.memoryGroups =
      rvvRegisterGroups(*codeShape) + rvvRegisterGroups(*activationShape);
  selected.resources.indexGroups = rvvRegisterGroups(*indexShape);
  selected.resources.predicateGroups = 1;
  selected.resources.primitiveGroups =
      rvvRegisterGroups(*indexShape) + 2 * rvvRegisterGroups(*tableShape) +
      2 * rvvRegisterGroups(*activationShape) +
      rvvRegisterGroups(*productShape) + 2;
  selected.resources.peakGroups = selected.resources.primitiveGroups;
  if (selected.resources.peakGroups >=
      static_cast<unsigned>(target.vectorRegisters))
    return std::nullopt;
  return selected;
}

std::optional<SelectedQuantI8DotPhysical>
selectQuantI8DotPhysical(const QuantI8DotCandidateFacts &facts,
                         const RISCVTargetProfile &target) {
  if (!target.hasRVV || target.vlenBits < 128 || !target.littleEndian)
    return std::nullopt;
  if (facts.semanticExtent == 0)
    return std::nullopt;
  SelectedQuantI8DotPhysical selected;
  selected.semanticLanes =
      std::min(facts.semanticExtent, target.vlenBits >= 256 ? 64u : 32u);
  std::optional<RVVVectorShape> byteShape =
      rvvShapeForSemanticLanes(8, selected.semanticLanes, target);
  if (!byteShape)
    return std::nullopt;
  selected.byteShape = *byteShape;
  selected.reductionSegments = selected.semanticLanes / 16;
  selected.resources.architecturalGroups = target.vectorRegisters;
  selected.resources.valueGroups = rvvRegisterGroups(*byteShape) * 2;
  selected.resources.memoryGroups = selected.resources.valueGroups;

  bool fixedLeaf = facts.semantic == QuantI8DotSemantic::PackedI4 ||
                   facts.semantic == QuantI8DotSemantic::PackedI5 ||
                   facts.semantic == QuantI8DotSemantic::IQ2S ||
                   facts.semantic == QuantI8DotSemantic::Q6K ||
                   facts.semantic == QuantI8DotSemantic::IQ1M ||
                   (facts.semantic == QuantI8DotSemantic::IQ3S &&
                    selected.semanticLanes == 64);
  unsigned fixedPeak = 4 * selected.reductionSegments +
                       4 * rvvRegisterGroups(*byteShape);
  if (fixedLeaf && fixedPeak < static_cast<unsigned>(target.vectorRegisters)) {
    selected.realization =
        QuantI8DotRealization::RVVFixedLaneLocalBlockDot;
    selected.resources.primitiveGroups = fixedPeak;
  } else {
    selected.realization =
        QuantI8DotRealization::RVVScalableLocalBlockDot;
    selected.resources.primitiveGroups = 6;
  }
  selected.resources.peakGroups = selected.resources.primitiveGroups;
  if (selected.resources.peakGroups >
      static_cast<unsigned>(target.vectorRegisters))
    return std::nullopt;
  return selected;
}

std::optional<SelectedE2M1E8M0I8Physical>
selectE2M1E8M0I8Physical(const RISCVTargetProfile &target) {
  if (!target.hasRVV || target.vlenBits < 128 || !target.littleEndian)
    return std::nullopt;

  SelectedE2M1E8M0I8Physical selected;
  selected.resources.architecturalGroups = target.vectorRegisters;
  if (target.vlenBits == 128 && target.supportsVectorShape(8, 8) &&
      target.supportsVectorShape(8, 16) &&
      target.supportsVectorShape(16, 32)) {
    selected.realization = E2M1E8M0I8Realization::RVVVLEN128TableDot;
    selected.packedShape = RVVVectorShape{8, 8};
    selected.activationShape = RVVVectorShape{8, 16};
    selected.resources.primitiveGroups = 12;
  } else if (target.vlenBits == 256 &&
             target.supportsVectorShape(8, 4) &&
             target.supportsVectorShape(16, 8)) {
    selected.realization = E2M1E8M0I8Realization::RVVVLEN256TableDot;
    selected.packedShape = RVVVectorShape{8, 4};
    selected.activationShape = RVVVectorShape{8, 4};
    selected.resources.primitiveGroups = 8;
  } else if (target.vlenBits > 128 &&
             target.supportsVectorShape(8, 8) &&
             target.supportsVectorShape(16, 16)) {
    selected.realization =
        E2M1E8M0I8Realization::RVVScalableLocalBlockDot;
    selected.packedShape = RVVVectorShape{8, 8};
    selected.activationShape = RVVVectorShape{8, 8};
    selected.resources.primitiveGroups = 4;
  } else {
    return std::nullopt;
  }

  selected.resources.valueGroups =
      rvvRegisterGroups(selected.packedShape) +
      rvvRegisterGroups(selected.activationShape);
  selected.resources.memoryGroups = selected.resources.valueGroups;
  selected.resources.peakGroups = selected.resources.primitiveGroups + 1;
  if (selected.resources.peakGroups >
      static_cast<unsigned>(target.vectorRegisters))
    return std::nullopt;
  return selected;
}

std::optional<SelectedGroupedAffineI4I8Physical>
selectGroupedAffineI4I8Physical(const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasWideningInteger ||
      target.vlenBits < 128 || !target.littleEndian)
    return std::nullopt;

  std::optional<RVVVectorShape> scaleShape =
      rvvShapeForSemanticLanes(8, 12, target);
  std::optional<RVVVectorShape> activationSumShape =
      rvvShapeForSemanticLanes(8, 32, target);
  if (!scaleShape || !activationSumShape)
    return std::nullopt;

  SelectedGroupedAffineI4I8Physical selected;
  selected.resources.architecturalGroups = target.vectorRegisters;
  selected.scaleShape = *scaleShape;
  selected.activationSumShape = *activationSumShape;
  if (target.vlenBits == 128 && target.supportsVectorShape(8, 8) &&
      target.supportsVectorShape(16, 16) &&
      target.supportsVectorShape(32, 8)) {
    selected.realization =
        GroupedAffineI4I8Realization::RVVVLEN128GroupedDot;
    selected.packedShape = RVVVectorShape{8, 8};
    selected.activationShape = RVVVectorShape{8, 8};
    selected.resources.primitiveGroups = 32;
  } else if (target.vlenBits == 256 &&
             target.supportsVectorShape(8, 8) &&
             target.supportsVectorShape(16, 16) &&
             target.supportsVectorShape(32, 8)) {
    selected.realization =
        GroupedAffineI4I8Realization::RVVVLEN256GroupedDot;
    selected.packedShape = RVVVectorShape{8, 8};
    selected.activationShape = RVVVectorShape{8, 8};
    selected.resources.primitiveGroups = 8;
  } else if (target.vlenBits > 128 &&
             target.supportsVectorShape(8, 8) &&
             target.supportsVectorShape(16, 16) &&
             target.supportsVectorShape(32, 8)) {
    selected.realization =
        GroupedAffineI4I8Realization::RVVScalableLocalBlockDot;
    selected.packedShape = RVVVectorShape{8, 8};
    selected.activationShape = RVVVectorShape{8, 8};
    selected.resources.primitiveGroups = 6;
  } else {
    return std::nullopt;
  }

  selected.resources.valueGroups =
      rvvRegisterGroups(selected.packedShape) +
      rvvRegisterGroups(selected.scaleShape) +
      rvvRegisterGroups(selected.activationShape) +
      rvvRegisterGroups(selected.activationSumShape);
  selected.resources.memoryGroups = selected.resources.valueGroups;
  selected.resources.peakGroups = selected.resources.primitiveGroups;
  if (selected.resources.peakGroups >
      static_cast<unsigned>(target.vectorRegisters))
    return std::nullopt;
  return selected;
}

std::optional<F32DotPhysicalConfig>
selectF32DotPhysicalConfig(const F32DotCandidateFacts &facts,
                           const RISCVTargetProfile &target,
                           const RISCVBackendConfig &config) {
  auto legalUnroll = [](unsigned value) {
    return value == 1 || value == 2 || value == 4;
  };
  llvm::SmallVector<unsigned> lmulCandidates =
      integerLMULCandidates(target, 32);
  llvm::SmallVector<unsigned> unrollCandidates = {1, 2, 4};
  if (config.dotLMUL != 0)
    lmulCandidates = {static_cast<unsigned>(config.dotLMUL)};
  if (config.dotKUnroll != 0)
    unrollCandidates = {static_cast<unsigned>(config.dotKUnroll)};

  bool wideVector = target.vlenBits >= 256;
  unsigned preferredLMUL =
      facts.model == F32DotResourceModel::VLAFreeAxis
          ? (facts.rowTile == 1 ? 4 : 2)
          : (wideVector ? 2 : (facts.rowTile >= 8 ? 1 : 2));
  unsigned preferredUnroll = 1;
  bool costlyHandoff = facts.materializedInit || facts.reductionPredicate ||
                       facts.indexedOperands != 0;
  if (!costlyHandoff) {
    if (facts.model == F32DotResourceModel::VLAFreeAxis)
      preferredUnroll = facts.rowTile == 1 ? 1 : (wideVector ? 4 : 2);
    else if (wideVector)
      preferredUnroll =
          facts.rowTile >= 8 ? 2 : facts.rowTile <= 4 ? 4 : 1;
  }

  struct Candidate {
    F32DotPhysicalConfig physical;
    unsigned tailPenalty = 0;
    unsigned memoryPenalty = 0;
    unsigned lmulPenalty = 0;
    unsigned unrollPenalty = 0;
    unsigned peakGroups = 0;
  };
  llvm::SmallVector<Candidate> legal;
  for (unsigned lmul : lmulCandidates) {
    if (!target.supportsVectorShape(32, static_cast<int>(lmul * 8)))
      continue;
    for (unsigned unroll : unrollCandidates) {
      if (!legalUnroll(unroll))
        continue;
      unsigned accumulatorGroups = facts.rowTile * lmul;
      unsigned streamedOperands =
          std::max(1u, facts.unitStrideOperands + facts.stridedOperands +
                           facts.indexedOperands);
      unsigned memoryGroups = streamedOperands * unroll * lmul;
      unsigned primitiveGroups = accumulatorGroups + memoryGroups;
      unsigned peakGroups = primitiveGroups + facts.predicateGroups +
                            facts.stateGroups + facts.handoffGroups + 1;
      if (peakGroups > static_cast<unsigned>(target.vectorRegisters))
        continue;
      legal.push_back(Candidate{
          F32DotPhysicalConfig{lmul, unroll},
          static_cast<unsigned>(facts.reductionExtent &&
                                *facts.reductionExtent % unroll != 0),
          facts.stridedOperands * unroll +
              2 * facts.indexedOperands * unroll + facts.predicateGroups +
              facts.handoffGroups,
          static_cast<unsigned>(std::abs(static_cast<int>(lmul) -
                                         static_cast<int>(preferredLMUL))),
          static_cast<unsigned>(std::abs(static_cast<int>(unroll) -
                                         static_cast<int>(preferredUnroll))),
          peakGroups});
    }
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.tailPenalty, lhs.memoryPenalty, lhs.lmulPenalty,
                    lhs.unrollPenalty, lhs.peakGroups) <
           std::tie(rhs.tailPenalty, rhs.memoryPenalty, rhs.lmulPenalty,
                    rhs.unrollPenalty, rhs.peakGroups);
  });
  return legal.front().physical;
}

std::optional<SelectedF16MatmulPhysical>
selectF16MatmulPhysicalConfig(const F16MatmulCandidateFacts &facts,
                              const RISCVTargetProfile &target,
                              const RISCVBackendConfig &config) {
  llvm::SmallVector<unsigned> rowCandidates;
  if (config.f16RowMicrotile != 0) {
    rowCandidates.push_back(static_cast<unsigned>(config.f16RowMicrotile));
  } else {
    for (unsigned rows = facts.rowTile; rows > 0; --rows)
      if (facts.rowTile % rows == 0)
        rowCandidates.push_back(rows);
    unsigned preferredRows = target.vlenBits >= 256
                                 ? std::min(4u, facts.rowTile)
                                 : std::min(2u, facts.rowTile);
    auto preferred = llvm::find(rowCandidates, preferredRows);
    if (preferred != rowCandidates.end())
      std::rotate(rowCandidates.begin(), preferred, std::next(preferred));
  }

  llvm::SmallVector<unsigned> lmulCandidates =
      integerLMULCandidates(target, 16, 4);
  if (config.f16InputLMUL != 0)
    lmulCandidates = {static_cast<unsigned>(config.f16InputLMUL)};
  llvm::SmallVector<unsigned> unrollCandidates = {1, 2, 4};
  if (config.f16KUnroll != 0)
    unrollCandidates = {static_cast<unsigned>(config.f16KUnroll)};
  llvm::SmallVector<unsigned> stageCandidates = {1, 2};
  if (config.f16PipelineStages != 0)
    stageCandidates = {static_cast<unsigned>(config.f16PipelineStages)};

  const unsigned preferredLMUL = 1;
  const unsigned preferredUnroll =
      facts.reductionTile < 32 ? 1 : target.vlenBits >= 256 ? 4 : 2;
  const unsigned preferredStages = 1;
  struct Candidate {
    F16MatmulPhysicalConfig physical;
    PhysicalResourceBudget resources;
    unsigned tailPenalty = 0;
    unsigned rowPenalty = 0;
    unsigned lmulPenalty = 0;
    unsigned unrollPenalty = 0;
    unsigned stagePenalty = 0;
  };
  llvm::SmallVector<Candidate> legal;
  for (unsigned rows : rowCandidates) {
    if (rows == 0 || rows > facts.rowTile || facts.rowTile % rows != 0)
      continue;
    for (unsigned inputLMUL : lmulCandidates) {
      if (!target.supportsVectorShape(16, static_cast<int>(inputLMUL * 8)) ||
          !target.supportsVectorShape(32,
                                      static_cast<int>(2 * inputLMUL * 8)))
        continue;
      unsigned computeLMUL = 2 * inputLMUL;
      unsigned lanes =
          static_cast<unsigned>(target.vlenBits * inputLMUL / 16);
      if (lanes == 0)
        continue;
      for (unsigned unroll : unrollCandidates) {
        if (unroll != 1 && unroll != 2 && unroll != 4)
          continue;
        for (unsigned stages : stageCandidates) {
          if ((stages != 1 && stages != 2) || (stages == 2 && unroll < 2))
            continue;
          F16MatmulPhysicalConfig physical;
          physical.rowMicrotile = rows;
          physical.inputLMUL = inputLMUL;
          physical.kUnroll = unroll;
          physical.pipelineStages = stages;
          physical.loadSchedule =
              stages == 2 ? F16MatmulLoadSchedule::BatchLoadsThenCompute
                          : F16MatmulLoadSchedule::StreamRHSThenRows;
          PhysicalResourceBudget resources;
          resources.architecturalGroups = target.vectorRegisters;
          resources.valueGroups = rows * computeLMUL;
          resources.memoryGroups = 2 * inputLMUL;
          unsigned pipelineGroups =
              stages == 2 ? 2 * unroll * inputLMUL : 0;
          unsigned operandGroups =
              stages == 2 ? pipelineGroups : resources.memoryGroups;
          resources.primitiveGroups = resources.valueGroups + operandGroups;
          resources.peakGroups = resources.primitiveGroups + 1;
          if (resources.peakGroups >
              static_cast<unsigned>(target.vectorRegisters))
            continue;
          legal.push_back(Candidate{
              physical,
              resources,
              static_cast<unsigned>(facts.reductionTile % (unroll * lanes) !=
                                    0),
              static_cast<unsigned>(std::abs(
                  static_cast<int>(rows) -
                  static_cast<int>(target.vlenBits >= 256
                                       ? std::min(4u, facts.rowTile)
                                       : std::min(2u, facts.rowTile)))),
              static_cast<unsigned>(std::abs(static_cast<int>(inputLMUL) -
                                             static_cast<int>(preferredLMUL))),
              static_cast<unsigned>(std::abs(static_cast<int>(unroll) -
                                             static_cast<int>(preferredUnroll))),
              static_cast<unsigned>(std::abs(static_cast<int>(stages) -
                                             static_cast<int>(preferredStages)))});
        }
      }
    }
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.tailPenalty, lhs.rowPenalty, lhs.lmulPenalty,
                    lhs.unrollPenalty, lhs.stagePenalty,
                    lhs.resources.peakGroups) <
           std::tie(rhs.tailPenalty, rhs.rowPenalty, rhs.lmulPenalty,
                    rhs.unrollPenalty, rhs.stagePenalty,
                    rhs.resources.peakGroups);
  });
  return SelectedF16MatmulPhysical{legal.front().physical,
                                  legal.front().resources};
}

} // namespace weft::riscv_internal
