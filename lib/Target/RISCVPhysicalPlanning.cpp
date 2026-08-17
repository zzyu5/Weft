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

struct LocalAxisMappingCandidate {
  CorePhysicalMapping mapping;
  PhysicalAxisDecomposition axis;
};

static llvm::SmallVector<LocalAxisMappingCandidate, 16>
enumerateRVVLocalAxisMappings(CoreMappingProblem problem,
                              CoreInstructionKind instruction,
                              unsigned laneAxis, unsigned laneSEW,
                              const RISCVTargetProfile &target) {
  auto laneConstraint = llvm::find_if(
      problem.axes, [&](const LogicalAxisConstraint &axis) {
        return axis.id == laneAxis;
      });
  if (laneConstraint == problem.axes.end() || !laneConstraint->extent ||
      *laneConstraint->extent == 0)
    return {};
  for (LogicalAxisConstraint &axis : problem.axes) {
    axis.allowLane = axis.id == laneAxis;
    axis.requireLane = axis.allowLane;
  }
  problem.laneSEW = laneSEW;
  problem.laneInstruction = instruction;
  llvm::SmallVector<RVVVectorShape, 8> laneShapes =
      rvvShapeCandidates(target, laneSEW);
  problem.laneShapeCandidates.assign(laneShapes.begin(), laneShapes.end());
  llvm::SmallVector<LocalAxisMappingCandidate, 16> candidates;
  for (CorePhysicalMapping mapping :
       enumerateCorePhysicalMappings(problem, target)) {
    const PhysicalAxisDecomposition *axis =
        findAxisMapping(mapping, laneAxis);
    if (!axis)
      continue;
    candidates.push_back({std::move(mapping), *axis});
  }
  return candidates;
}

static const LogicalAxisConstraint *
findLogicalAxisConstraint(const CoreMappingProblem &problem, unsigned axis) {
  auto found = llvm::find_if(
      problem.axes,
      [&](const LogicalAxisConstraint &constraint) {
        return constraint.id == axis;
      });
  return found == problem.axes.end() ? nullptr : &*found;
}

static const LogicalAxisConstraint *
findUniqueLogicalAxisConstraint(const CoreMappingProblem &problem,
                                LogicalAxisRole role) {
  const LogicalAxisConstraint *result = nullptr;
  for (const LogicalAxisConstraint &axis : problem.axes) {
    if (axis.role != role)
      continue;
    if (result)
      return nullptr;
    result = &axis;
  }
  return result;
}

static std::optional<LocalMicrokernelSchedule> buildLocalMicrokernelSchedule(
    const CorePhysicalMapping &mapping, unsigned iterationAxis,
    unsigned accumulatorCount, llvm::ArrayRef<LocalOperandWindow> operands,
    LocalDecodeSchedule decode = {}, unsigned prefetchDistance = 0) {
  const PhysicalAxisDecomposition *iteration =
      findAxisMapping(mapping, iterationAxis);
  const PhysicalAxisDecomposition *lane =
      mapping.laneAxis
          ? findAxisMapping(mapping, *mapping.laneAxis)
          : nullptr;
  if (!iteration || iteration->unrollFactor == 0 || accumulatorCount == 0 ||
      mapping.pipeline.bufferCount == 0 ||
      prefetchDistance >= mapping.pipeline.bufferCount)
    return std::nullopt;
  LocalMicrokernelSchedule schedule;
  schedule.iterationAxis = iterationAxis;
  schedule.sequentialIterations = std::max(1u, iteration->sequentialFactor);
  schedule.laneFactor =
      lane ? lane->laneFactor
           : std::max(1u, iteration->fragmentFactor);
  schedule.iterationRegisterFactor = iteration->registerFactor;
  schedule.iterationElements =
      iteration->laneFactor * iteration->registerFactor *
      iteration->fragmentFactor;
  schedule.unrollFactor = iteration->unrollFactor;
  schedule.accumulatorCount = accumulatorCount;
  schedule.pipeline.bufferCount = mapping.pipeline.bufferCount;
  schedule.pipeline.prefetchDistance = prefetchDistance;
  for (unsigned iteration = 0; iteration < prefetchDistance; ++iteration)
    schedule.pipeline.actions.push_back(
        {LocalPipelineActionKind::Load, iteration,
         iteration % schedule.pipeline.bufferCount});
  for (unsigned iteration = 0; iteration < schedule.unrollFactor;
       ++iteration) {
    const unsigned next = iteration + prefetchDistance;
    if (next < schedule.unrollFactor)
      schedule.pipeline.actions.push_back(
          {LocalPipelineActionKind::Load, next,
           next % schedule.pipeline.bufferCount});
    schedule.pipeline.actions.push_back(
        {LocalPipelineActionKind::Compute, iteration,
         iteration % schedule.pipeline.bufferCount});
  }
  schedule.operands.append(operands.begin(), operands.end());
  schedule.decode = decode;
  if (!schedule)
    return std::nullopt;
  return schedule;
}

static bool selectLocalMicrokernelSchedule(
    LocalImplementation &implementation, unsigned iterationAxis,
    unsigned accumulatorCount, llvm::ArrayRef<LocalOperandWindow> operands,
    LocalDecodeSchedule decode = {}, unsigned prefetchDistance = 0) {
  std::optional<LocalMicrokernelSchedule> schedule =
      buildLocalMicrokernelSchedule(implementation.mapping, iterationAxis,
                                    accumulatorCount, operands, decode,
                                    prefetchDistance);
  if (!schedule)
    return false;
  implementation.schedule = std::move(*schedule);
  return true;
}

std::string localImplementationSymbol(
    const LocalImplementation &implementation) {
  const LocalHardwareOperation &operation = implementation.operation;
  const RVVVectorShape laneShape = implementation.mapping.laneShape;
  const unsigned lanes = implementation.schedule.iterationElements;
  const bool sequential = implementation.schedule.sequentialIterations > 1;
  const std::string shape = rvvShapeSuffix(laneShape);
  const std::string registerSuffix =
      lanes && !shape.empty()
          ? "_register_l" + std::to_string(lanes) + "_e" + shape
          : std::string{};
  switch (implementation.primitive) {
  case LocalPrimitiveKind::None:
  case LocalPrimitiveKind::F32Math:
    return {};
  case LocalPrimitiveKind::SymmetricI4I8:
  case LocalPrimitiveKind::AffineI4I8: {
    const bool affine = implementation.primitive == LocalPrimitiveKind::AffineI4I8;
    const bool fragment =
        operation.kind == LocalHardwareOperationKind::MatrixFragment;
    if ((!fragment && operation.kind !=
                          LocalHardwareOperationKind::RVVIntrinsic) ||
        (operation.rows != 1 && operation.rows != 4))
      return {};
    return std::string("__weft_") + (fragment ? "ime1_" : "rvv_") +
           (affine ? "affine" : "symmetric") + "_i4_i8_" +
           (operation.rows == 4 ? "m4_" : "") + "n16_k32";
  }
  case LocalPrimitiveKind::GroupedAffineI4I8:
    if (operation.kind == LocalHardwareOperationKind::RVVIntrinsic && sequential)
      return "__weft_grouped_affine_i4_i8_strip";
    return (operation.kind == LocalHardwareOperationKind::RVVIntrinsic ||
            operation.kind == LocalHardwareOperationKind::RVVInlineAsm) &&
                   !registerSuffix.empty()
               ? "__weft_grouped_affine_i4_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::E2M1E8M0I8:
    if (operation.kind == LocalHardwareOperationKind::RVVIntrinsic && sequential)
      return "__weft_e2m1_e8m0_i8_strip";
    if (operation.kind != LocalHardwareOperationKind::RVVIntrinsic)
      return {};
    if (laneShape == RVVVectorShape{8, 4})
      return "__weft_e2m1_e8m0_i8_register_e8mf2";
    if (laneShape == kRVVE8M1)
      return "__weft_e2m1_e8m0_i8_register_e8m1_e8m2";
    return {};
  case LocalPrimitiveKind::PackedI4I8:
    return operation.kind == LocalHardwareOperationKind::RVVIntrinsic &&
                   !registerSuffix.empty()
               ? "__weft_packed_i4_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::PackedI5I8:
    return operation.kind == LocalHardwareOperationKind::RVVIntrinsic &&
                   !registerSuffix.empty()
               ? "__weft_packed_i5_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::PackedI3GroupedI8:
    return operation.kind == LocalHardwareOperationKind::RVVIntrinsic &&
                   !registerSuffix.empty()
               ? "__weft_packed_i3_grouped_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::Base3TernaryI8:
    return operation.kind == LocalHardwareOperationKind::RVVIntrinsic &&
                   !registerSuffix.empty()
               ? "__weft_base3_ternary_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::PackedI2TernaryI8:
    return operation.kind == LocalHardwareOperationKind::RVVIntrinsic &&
                   !registerSuffix.empty()
               ? "__weft_packed_i2_ternary_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::SignedCodebook8I8:
    return operation.kind == LocalHardwareOperationKind::RVVIntrinsic &&
                   !registerSuffix.empty()
               ? "__weft_signed_codebook8_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::SignedCodebook4I8:
    return operation.kind == LocalHardwareOperationKind::RVVIntrinsic &&
                   !registerSuffix.empty()
               ? "__weft_signed_codebook4_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::PackedU9U7CodebookI8:
    return operation.kind == LocalHardwareOperationKind::RVVIntrinsic &&
                   !registerSuffix.empty()
               ? "__weft_packed_u9_u7_codebook_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::PackedU11GridDeltaI8:
    return operation.kind == LocalHardwareOperationKind::RVVIntrinsic &&
                   !registerSuffix.empty()
               ? "__weft_packed_u11_grid_delta_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::NibbleCodebookI8:
    return operation.kind == LocalHardwareOperationKind::RVVIntrinsic &&
                   !registerSuffix.empty()
               ? "__weft_nibble_codebook_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::IQ2SI8:
    return operation.kind == LocalHardwareOperationKind::RVVIntrinsic &&
                   !registerSuffix.empty()
               ? "__weft_iq2_s_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::IQ3SI8:
    return operation.kind == LocalHardwareOperationKind::RVVIntrinsic &&
                   !registerSuffix.empty()
               ? "__weft_iq3_s_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::IQ1MI8:
    return operation.kind == LocalHardwareOperationKind::RVVIntrinsic &&
                   !registerSuffix.empty()
               ? "__weft_iq1_m_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::Q6KI8:
    return (operation.kind == LocalHardwareOperationKind::RVVIntrinsic ||
            operation.kind == LocalHardwareOperationKind::RVVInlineAsm) &&
                   !registerSuffix.empty()
               ? "__weft_q6_k_i8" + registerSuffix
               : std::string{};
  }
  return {};
}

static void selectLocalHardwareOperation(
    LocalImplementation &implementation, LocalHardwareOperationKind kind,
    LocalOperationProjection projection = LocalOperationProjection::None,
    unsigned rows = 1) {
  implementation.operation = LocalHardwareOperation{kind, rows, projection};
}

static bool finalizeLocalHardwareOperation(
    const LocalImplementation &implementation) {
  if (!implementation || implementation.valueShapes.empty())
    return false;
  if (implementation.primitive == LocalPrimitiveKind::F32Math)
    return !implementation.operation;
  return implementation.operation && implementation.schedule &&
         !localImplementationSymbol(implementation).empty();
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
  if (!target.hasRVV || facts.codeExtent != 16 ||
      facts.tableExtent != 16)
    return std::nullopt;
  std::optional<RVVVectorShape> laneShape =
      rvvShapeForSemanticLanes(8, facts.codeExtent, target);
  if (!laneShape)
    return std::nullopt;
  CoreMappingProblem problem;
  problem.axes = {
      LogicalAxisConstraint{0, LogicalAxisRole::Packed,
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
    if (!facts.memory)
      return std::nullopt;
    selected.memory = selectBlockMemoryPhysical(*facts.memory, target);
    if (!selected.memory)
      return std::nullopt;
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
              : rvvRegisterGroups(selected.resultShape) == 1
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
      if (facts.byteShape.lmulEighths < 8 &&
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
    return std::nullopt;
  }
  if (!selected.resultShape)
    return std::nullopt;
  return selected;
}

std::optional<SelectedBlockMemoryPhysical>
selectBlockMemoryPhysical(const BlockMemoryCandidateFacts &facts,
                          const RISCVTargetProfile &target) {
  if (!target.hasRVV || (facts.elementSEW != 8 && facts.elementSEW != 32))
    return std::nullopt;

  SelectedBlockMemoryPhysical selected;
  if (facts.relation == LaneRelation::UnitStride) {
    selected.memoryMode = PhysicalMemoryMode::UnitStride;
  } else if (facts.relation == LaneRelation::Indexed &&
             target.hasIndexedMemory && facts.indexSEW == 16) {
    selected.memoryMode = PhysicalMemoryMode::Indexed;
    selected.indexedSEW = facts.indexSEW;
  } else {
    return std::nullopt;
  }

  if (facts.predicateAllActive) {
    selected.activityMode = PhysicalActivityMode::AllActive;
  } else if (!facts.write && facts.predicateVector &&
             selected.memoryMode == PhysicalMemoryMode::Indexed) {
    selected.activityMode = PhysicalActivityMode::PredicateMask;
  } else {
    return std::nullopt;
  }
  return selected;
}

std::optional<SelectedBlockStorePhysical>
selectBlockStorePhysical(const BlockStoreCandidateFacts &facts,
                         const RISCVTargetProfile &target) {
  if (facts.mapping.axes.size() != 1 ||
      facts.mapping.axes.front().role != LogicalAxisRole::Free ||
      !facts.mapping.axes.front().extent ||
      !facts.mapping.axes.front().requireLane)
    return std::nullopt;
  struct Candidate {
    SelectedBlockStorePhysical physical;
    unsigned sequentialFactor = 0;
    unsigned peakGroups = 0;
  };
  llvm::SmallVector<Candidate, 8> legal;
  for (CorePhysicalMapping mapping :
       enumerateCorePhysicalMappings(facts.mapping, target)) {
    const PhysicalAxisDecomposition *axis =
        findUniqueAxisMapping(mapping, LogicalAxisRole::Free);
    if (!axis || !mapping.laneShape || mapping.laneShape.sew != 8 ||
        axis->laneFactor == 0 || axis->registerFactor == 0)
      continue;
    const unsigned laneFactor = axis->laneFactor;
    const unsigned registerFactor = axis->registerFactor;
    const unsigned sequentialFactor = axis->sequentialFactor;
    const uint64_t realized =
        static_cast<uint64_t>(laneFactor) * registerFactor;
    if ((sequentialFactor == 1 && realized != *axis->extent) ||
        (sequentialFactor > 1 && registerFactor != 1) ||
        (facts.needsLaneVector && registerFactor != 1))
      continue;
    RVVVectorShape laneShape;
    if (facts.needsLaneVector) {
      std::optional<RVVVectorShape> selectedLane =
          rvvShapeForSameLanes(mapping.laneShape, 16, target);
      if (!selectedLane)
        continue;
      laneShape = *selectedLane;
    }
    PhysicalResourceRequirements requirements;
    requirements.live = {{PhysicalLiveClass::Value, mapping.laneShape,
                          registerFactor, 0}};
    if (laneShape)
      requirements.live.push_back(
          {PhysicalLiveClass::Index, laneShape, 1, 0});
    std::optional<PhysicalResourceBudget> resources =
        calculatePhysicalResources(requirements, target);
    if (!resources)
      continue;
    SelectedBlockStorePhysical selected;
    selected.decision.mapping = std::move(mapping);
    selected.decision.needsLaneVector = facts.needsLaneVector;
    selected.decision.laneShape = laneShape;
    if (selected.decision.mapping.laneShape.lmulEighths < 8 &&
        sequentialFactor == 1 && registerFactor > 1)
      selected.decision.realization = BlockStoreRealization::MultiStripValues;
    else if (sequentialFactor == 1)
      selected.decision.realization = BlockStoreRealization::RegisterRepetition;
    else
      selected.decision.realization = BlockStoreRealization::SequentialStrip;
    selected.resources = *resources;
    legal.push_back(
        Candidate{std::move(selected), sequentialFactor, resources->peakGroups});
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.sequentialFactor, lhs.peakGroups) <
           std::tie(rhs.sequentialFactor, rhs.peakGroups);
  });
  return std::move(legal.front().physical);
}

std::optional<SelectedBlockReducePhysical>
selectBlockReducePhysical(const BlockReduceCandidateFacts &facts,
                          const RISCVTargetProfile &target) {
  if (facts.inputSEW == 0 || facts.mapping.axes.size() != 1 ||
      facts.mapping.axes.front().role != LogicalAxisRole::Reduction ||
      !facts.mapping.axes.front().extent ||
      !facts.mapping.axes.front().requireLane)
    return std::nullopt;
  struct Candidate {
    SelectedBlockReducePhysical physical;
    unsigned sequentialFactor = 0;
    unsigned peakGroups = 0;
  };
  llvm::SmallVector<Candidate, 4> legal;
  for (CorePhysicalMapping mapping :
       enumerateCorePhysicalMappings(facts.mapping, target)) {
    const PhysicalAxisDecomposition *axis =
        findUniqueAxisMapping(mapping, LogicalAxisRole::Reduction);
    if (!axis || !mapping.laneShape || mapping.laneShape.sew != 8 ||
        axis->laneFactor == 0 ||
        axis->registerFactor == 0 || axis->registerFactor > 2)
      continue;
    const unsigned laneFactor = axis->laneFactor;
    const unsigned registerFactor = axis->registerFactor;
    const unsigned sequentialFactor = axis->sequentialFactor;
    const uint64_t realized =
        static_cast<uint64_t>(laneFactor) * registerFactor;
    if ((sequentialFactor == 1 && realized != *axis->extent) ||
        (sequentialFactor > 1 && registerFactor != 1) ||
        (facts.needsLaneVector && registerFactor != 1))
      continue;
    RVVVectorShape laneShape;
    if (facts.needsLaneVector) {
      std::optional<RVVVectorShape> selectedLane =
          rvvShapeForSameLanes(mapping.laneShape, 16, target);
      if (!selectedLane)
        continue;
      laneShape = *selectedLane;
    }
    std::optional<RVVVectorShape> inputShape =
        rvvShapeForSameLanes(mapping.laneShape, facts.inputSEW, target);
    if (!inputShape)
      continue;

    SelectedBlockReducePhysical selected;
    selected.decision.mapping = std::move(mapping);
    selected.decision.needsLaneVector = facts.needsLaneVector;
    selected.decision.laneShape = laneShape;
    selected.inputShape = *inputShape;
    selected.combinedShape = *inputShape;
    if (sequentialFactor == 1 && inputShape->sew == 16) {
      std::optional<RVVVectorShape> combined =
          rvvShapeForSameLanes(*inputShape, 32, target);
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
        {PhysicalLiveClass::Value, *inputShape, registerFactor, 0},
        {PhysicalLiveClass::Temporary, selected.seedShape, 1, 0}};
    if (laneShape)
      requirements.live.push_back(
          {PhysicalLiveClass::Index, laneShape, 1, 0});
    std::optional<PhysicalResourceBudget> resources =
        calculatePhysicalResources(requirements, target);
    if (!resources)
      continue;
    selected.resources = *resources;
    legal.push_back(Candidate{std::move(selected), sequentialFactor,
                              resources->peakGroups});
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.sequentialFactor, lhs.peakGroups) <
           std::tie(rhs.sequentialFactor, rhs.peakGroups);
  });
  return std::move(legal.front().physical);
}

std::optional<SelectedMaterializedBlockStorePhysical>
selectMaterializedBlockStorePhysical(
    const MaterializedBlockStoreCandidateFacts &facts,
    const RISCVTargetProfile &target) {
  const bool rankOne =
      facts.mapping.axes.size() == 1 &&
      facts.mapping.axes.front().role == LogicalAxisRole::Free &&
      facts.mapping.axes.front().extent;
  const bool rankTwo =
      facts.mapping.axes.size() == 2 &&
      facts.mapping.axes[0].role == LogicalAxisRole::Free &&
      facts.mapping.axes[0].extent &&
      facts.mapping.axes[1].role == LogicalAxisRole::Free &&
      facts.mapping.axes[1].extent;
  if ((!rankOne && !rankTwo) || !facts.prefixPredicated)
    return std::nullopt;

  CoreMappingProblem problem = facts.mapping;
  problem.allowSequentialOnly = true;
  problem.sequentialInstruction = CoreInstructionKind::Scalar;
  problem.laneSEW = 32;
  problem.laneInstruction = CoreInstructionKind::RVVElementwise;
  const bool vectorizable = facts.allActive && facts.unitStride;
  if (vectorizable) {
    LogicalAxisConstraint &laneAxis =
        rankOne ? problem.axes.front() : problem.axes[1];
    laneAxis.allowLane = true;
    problem.laneShapeCandidates = rvvShapeCandidates(target, 32);
  } else {
    for (LogicalAxisConstraint &axis : problem.axes) {
      axis.allowLane = false;
      axis.requireLane = false;
    }
    problem.laneShapeCandidates.clear();
  }

  struct Candidate {
    SelectedMaterializedBlockStorePhysical physical;
    unsigned instructionCost = 0;
    unsigned sequentialFactor = 0;
    unsigned peakGroups = 0;
  };
  llvm::SmallVector<Candidate, 8> legal;
  for (CorePhysicalMapping mapping :
       enumerateCorePhysicalMappings(problem, target)) {
    const bool vector =
        mapping.instruction == CoreInstructionKind::RVVElementwise;
    if (vector && (!vectorizable || !mapping.laneShape))
      continue;
    if (!vector && mapping.instruction != CoreInstructionKind::Scalar)
      continue;
    PhysicalResourceRequirements requirements;
    if (vector)
      requirements.live = {
          {PhysicalLiveClass::Value, mapping.laneShape, 1, 0}};
    std::optional<PhysicalResourceBudget> resources =
        calculatePhysicalResources(requirements, target);
    if (!resources)
      continue;
    unsigned sequentialFactor = 1;
    for (const PhysicalAxisDecomposition &axis : mapping.axes)
      sequentialFactor *= std::max(1u, axis.sequentialFactor);
    SelectedMaterializedBlockStorePhysical selected;
    selected.mapping = std::move(mapping);
    selected.resources = *resources;
    legal.push_back(Candidate{std::move(selected), vector ? 0u : 1u,
                              sequentialFactor, resources->peakGroups});
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.instructionCost, lhs.sequentialFactor,
                    lhs.peakGroups) <
           std::tie(rhs.instructionCost, rhs.sequentialFactor,
                    rhs.peakGroups);
  });
  return std::move(legal.front().physical);
}

std::optional<SelectedSortIndicesPhysical>
selectSortIndicesPhysical(const SortIndicesCandidateFacts &facts,
                          const RISCVTargetProfile &target) {
  if (!target.littleEndian || target.xlen != 64 || !target.hasRVV ||
      (facts.configuredRadixBits != 0 && facts.configuredRadixBits != 8 &&
       facts.configuredRadixBits != 11))
    return std::nullopt;
  if (facts.mapping.axes.size() != 1 ||
      !facts.mapping.axes.front().ordered ||
      facts.mapping.axes.front().allowLane ||
      facts.mapping.axes.front().requireLane)
    return std::nullopt;
  CoreMappingProblem problem = facts.mapping;
  problem.allowSequentialOnly = true;
  problem.sequentialInstruction = CoreInstructionKind::Scalar;
  llvm::SmallVector<CorePhysicalMapping> mappings =
      enumerateCorePhysicalMappings(problem, target);
  if (mappings.size() != 1 || mappings.front().instruction !=
                                  CoreInstructionKind::Scalar ||
      mappings.front().axes.size() != 1 ||
      !mappings.front().axes.front().ordered)
    return std::nullopt;
  SelectedSortIndicesPhysical selected;
  selected.mapping = std::move(mappings.front());
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
  PhysicalResourceRequirements requirements;
  std::optional<PhysicalResourceBudget> resources =
      calculatePhysicalResources(requirements, target);
  if (!resources)
    return std::nullopt;
  selected.resources = *resources;
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
    selected.memoryMode = PhysicalMemoryMode::UnitStride;
    break;
  case LaneRelation::Strided:
    selected.memoryMode = PhysicalMemoryMode::Strided;
    break;
  case LaneRelation::Indexed:
    if (!target.hasIndexedMemory)
      return std::nullopt;
    selected.memoryMode = PhysicalMemoryMode::Indexed;
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
    selected.activityMode = PhysicalActivityMode::AllActive;
  else if (facts.predicateVector)
    selected.activityMode = PhysicalActivityMode::PredicateMask;
  else if (facts.predicateScalar)
    selected.activityMode = PhysicalActivityMode::ScalarPredicate;
  else
    return std::nullopt;
  if (facts.carriesLogicalValidity) {
    if (selected.activityMode == PhysicalActivityMode::AllActive)
      return std::nullopt;
    selected.inactiveLane =
        PhysicalInactiveLaneRealization::ZeroCarrierWithLogicalValidity;
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
            ? PhysicalMemoryMode::UnitStride
            : PhysicalMemoryMode::Strided;
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
    sourceSEW = 32;
    resultSEW = target.xlen;
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
  if (!target.hasRVV || !target.hasF)
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

llvm::SmallVector<SelectedVLAStatePhysical, 2>
enumerateVLAStatePhysical(const VLAStateCandidateFacts &facts,
                          const RISCVTargetProfile &target) {
  llvm::SmallVector<SelectedVLAStatePhysical, 2> candidates;
  if (!target.hasRVV || !facts.inputLaneMapped)
    return candidates;
  if (facts.resultLaneMapped != facts.preservesLanePositions)
    return candidates;
  SelectedVLAStatePhysical scalar;
  scalar.wholeVLALifetime = facts.resultControlCarried ||
                            facts.resultCrossesRegion ||
                            !facts.resultLaneMapped;
  scalar.requiresF32Math = facts.requiresF32Math;
  const bool f32Input = facts.floatingInput && facts.inputSEW == 32;
  const bool f32Result = facts.floatingResult && facts.resultSEW == 32;
  switch (facts.semantic) {
  case VLAStateSemantic::F32AddReduction:
    if (!target.hasF || !f32Input || !f32Result)
      return candidates;
    scalar.stripUpdate = VLAStateStripUpdate::AddReduction;
    break;
  case VLAStateSemantic::F32MaxReduction:
    if (!target.hasF || !f32Input || !f32Result)
      return candidates;
    scalar.stripUpdate = VLAStateStripUpdate::MaxReduction;
    break;
  case VLAStateSemantic::I8AddReductionI32:
    if (!target.hasWideningInteger || !facts.signedInput ||
        facts.inputSEW != 8 || facts.floatingResult || facts.resultSEW != 32)
      return candidates;
    scalar.stripUpdate = VLAStateStripUpdate::WideningAddReduction;
    candidates.push_back(scalar);
    return candidates;
  case VLAStateSemantic::InclusiveAddScan:
    if (!target.hasF || facts.relaxedOrder || facts.maskedInput || !f32Input ||
        !f32Result)
      return candidates;
    scalar.stripUpdate = VLAStateStripUpdate::InclusiveAddScan;
    candidates.push_back(scalar);
    return candidates;
  case VLAStateSemantic::SegmentedInclusiveAddScan:
    if (!target.hasF || facts.relaxedOrder || facts.maskedInput || !f32Input ||
        !f32Result)
      return candidates;
    scalar.stripUpdate = VLAStateStripUpdate::SegmentedInclusiveAddScan;
    candidates.push_back(scalar);
    return candidates;
  case VLAStateSemantic::ArgMaxSummary:
    if (!target.hasF || !f32Input ||
        (facts.coordinateRelation != LaneRelation::UnitStride &&
         facts.coordinateRelation != LaneRelation::Strided))
      return candidates;
    scalar.carry = VLAStateCarryRepresentation::ScalarTuple;
    scalar.stripUpdate = VLAStateStripUpdate::ArgMaxSummary;
    scalar.coordinateMode =
        facts.coordinateRelation == LaneRelation::UnitStride
            ? PhysicalMemoryMode::UnitStride
            : PhysicalMemoryMode::Strided;
    candidates.push_back(scalar);
    return candidates;
  case VLAStateSemantic::OnlineSoftmaxSummary:
    if (!target.hasF || facts.relaxedOrder || !f32Input ||
        !facts.requiresF32Math)
      return candidates;
    scalar.carry = VLAStateCarryRepresentation::ScalarTuple;
    scalar.stripUpdate = VLAStateStripUpdate::OnlineSoftmaxSummary;
    candidates.push_back(scalar);
    return candidates;
  }
  candidates.push_back(scalar);
  if (facts.relaxedOrder) {
    SelectedVLAStatePhysical vector = scalar;
    vector.carry = VLAStateCarryRepresentation::Vector;
    vector.finalize = facts.semantic == VLAStateSemantic::F32AddReduction
                          ? VLAStateFinalize::HorizontalAdd
                          : VLAStateFinalize::HorizontalMax;
    candidates.push_back(vector);
  }
  return candidates;
}

static bool resolveVLAStateShapes(SelectedVLAStatePhysical &state,
                                  RVVVectorShape dataShape,
                                  const RISCVTargetProfile &target) {
  state.inputShape = dataShape;
  switch (state.stripUpdate) {
  case VLAStateStripUpdate::WideningAddReduction:
    state.inputShape =
        rvvShapeForSameLanes(dataShape, 8, target).value_or(RVVVectorShape{});
    state.seedShape = RVVVectorShape{16, 8};
    break;
  case VLAStateStripUpdate::AddReduction:
  case VLAStateStripUpdate::MaxReduction:
  case VLAStateStripUpdate::ArgMaxSummary:
  case VLAStateStripUpdate::OnlineSoftmaxSummary:
    state.seedShape = kRVVE32M1;
    break;
  case VLAStateStripUpdate::InclusiveAddScan:
  case VLAStateStripUpdate::SegmentedInclusiveAddScan:
    break;
  }
  if (state.carry == VLAStateCarryRepresentation::Vector)
    state.carryShape = dataShape;
  auto supported = [&](RVVVectorShape shape) {
    return !shape ||
           target.supportsVectorShape(shape.sew, shape.lmulEighths);
  };
  return state.inputShape && supported(state.inputShape) &&
         supported(state.carryShape) && supported(state.seedShape);
}

std::optional<LocalImplementation>
selectF32MathLocalImplementation(const CorePhysicalMapping &mapping,
                                 const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasF || !mapping || !mapping.laneAxis ||
      mapping.instruction != CoreInstructionKind::RVVElementwise ||
      mapping.laneShape.sew != 32 ||
      !target.supportsVectorShape(mapping.laneShape.sew,
                                  mapping.laneShape.lmulEighths) ||
      !rvvIntegerLMUL(mapping.laneShape))
    return std::nullopt;
  LocalImplementation implementation;
  implementation.primitive = LocalPrimitiveKind::F32Math;
  implementation.mapping = mapping;
  implementation.valueShapes = {mapping.laneShape};
  return finalizeLocalHardwareOperation(implementation)
             ? std::optional<LocalImplementation>(std::move(implementation))
             : std::nullopt;
}

std::optional<SelectedVLALookupPhysical>
selectVLALookupPhysical(const VLALookupCandidateFacts &facts,
                        const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasIndexedMemory ||
      facts.tableExtent != 16 || !facts.tableF32 || !facts.resultF32 ||
      !facts.indicesU8 || !facts.allActive)
    return std::nullopt;
  return SelectedVLALookupPhysical{VLALookupRealization::RVVTableGather,
                                   facts.tableExtent};
}

std::optional<SelectedI4I8FragmentPhysical>
selectI4I8FragmentPhysical(const I4I8FragmentCandidateFacts &facts,
                           const RISCVTargetProfile &target) {
  auto axis = [&](unsigned id) -> const LogicalAxisConstraint * {
    auto found = llvm::find_if(facts.mapping.axes,
                               [&](const LogicalAxisConstraint &candidate) {
                                 return candidate.id == id;
                               });
    return found == facts.mapping.axes.end() ? nullptr : &*found;
  };
  const LogicalAxisConstraint *rows = axis(facts.rowsAxis);
  const LogicalAxisConstraint *columns = axis(facts.columnsAxis);
  const LogicalAxisConstraint *reduction = axis(facts.reductionAxis);
  if (!target.littleEndian || !rows || !columns || !reduction ||
      !rows->extent || *rows->extent == 0 ||
      *rows->extent > std::numeric_limits<unsigned>::max() ||
      rows->role != LogicalAxisRole::Free || !columns->extent ||
      *columns->extent == 0 || columns->role != LogicalAxisRole::Free ||
      !reduction->extent || *reduction->extent == 0 ||
      reduction->role != LogicalAxisRole::Reduction)
    return std::nullopt;
  const unsigned rowExtent = static_cast<unsigned>(*rows->extent);
  const bool rvvLeafShape =
      (rowExtent == 1 || rowExtent == 4) && *columns->extent == 16 &&
      *reduction->extent == 32;
  const bool supportsRVV =
      target.hasF && target.hasVectorF16 && target.hasWideningInteger &&
      target.hasWideningFloat &&
      target.supportsVectorShape(8, 8) &&
      target.supportsVectorShape(16, 16) &&
      target.supportsVectorShape(32, 32) &&
      rvvLaneCapacity(kRVVE8M1, target).value_or(0) >= 16 && rvvLeafShape;
  CoreMappingProblem problem = facts.mapping;
  for (LogicalAxisConstraint &constraint : problem.axes) {
    constraint.allowLane = constraint.id == facts.columnsAxis && supportsRVV;
    constraint.requireLane = constraint.allowLane;
    constraint.registerFactors =
        constraint.id == facts.rowsAxis
            ? llvm::SmallVector<unsigned, 4>{rowExtent}
            : llvm::SmallVector<unsigned, 4>{1};
  }
  problem.laneSEW = 8;
  problem.laneInstruction = CoreInstructionKind::RVVWideningIntegerDot;
  problem.laneLMULCandidates = {1};
  for (const RISCVFragmentCapability &capability :
       target.fragmentCapabilities) {
    if (capability.instruction !=
            RISCVFragmentInstruction::SpacemitIME1I4I8MMA ||
        capability.lhsElementBits != 4 || capability.rhsElementBits != 8 ||
        capability.accumulatorElementBits != 32 ||
        capability.mFactor != rowExtent ||
        capability.nFactor != *columns->extent ||
        capability.kFactor != *reduction->extent)
      continue;
    problem.fragments.push_back(FragmentMappingConstraint{
        CoreInstructionKind::MatrixFragment,
        {{facts.rowsAxis, capability.mFactor},
         {facts.columnsAxis, capability.nFactor},
         {facts.reductionAxis, capability.kFactor}},
        capability.fixedResourceGroups});
  }

  struct Candidate {
    SelectedI4I8FragmentPhysical physical;
    unsigned instructionCost = 0;
  };
  llvm::SmallVector<Candidate, 2> candidates;
  for (CorePhysicalMapping mapping :
       enumerateCorePhysicalMappings(problem, target)) {
    const bool fragmentMapping =
        mapping.instruction == CoreInstructionKind::MatrixFragment;
    if (!fragmentMapping && !supportsRVV)
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
    const PhysicalAxisDecomposition *mappedRows =
        findAxisMapping(selected.implementation.mapping, facts.rowsAxis);
    if (!mappedRows)
      continue;
    selectLocalHardwareOperation(
        selected.implementation,
        fragmentMapping ? LocalHardwareOperationKind::MatrixFragment
                        : LocalHardwareOperationKind::RVVIntrinsic,
        LocalOperationProjection::None,
        fragmentMapping ? mappedRows->fragmentFactor
                        : mappedRows->registerFactor);
    if (!selectLocalMicrokernelSchedule(
            selected.implementation, facts.reductionAxis,
            fragmentMapping ? mappedRows->fragmentFactor
                            : mappedRows->registerFactor,
            {{0, PhysicalMemoryMode::UnitStride, 1,
              std::max(1u, rowExtent), true},
             {1, PhysicalMemoryMode::UnitStride,
              std::max(1u, rowExtent), 1, true}},
            {2, 16, 1}))
      continue;
    if (!finalizeLocalHardwareOperation(selected.implementation))
      continue;
    const unsigned privateGroups =
        fragmentMapping
            ? selected.implementation.mapping.fixedFragmentGroups
            : mappedRows->registerFactor *
                      rvvRegisterGroups(selected.accumulatorShape) +
                  8 + static_cast<unsigned>(facts.affine);
    PhysicalResourceRequirements requirements;
    requirements.live.push_back(
        PhysicalLiveRange{fragmentMapping ? PhysicalLiveClass::Fragment
                                          : PhysicalLiveClass::Temporary,
                          {}, 1, privateGroups});
    std::optional<PhysicalResourceBudget> resources =
        calculatePhysicalResources(requirements, target);
    if (!resources)
      continue;
    selected.resources = *resources;
    candidates.push_back(Candidate{
        std::move(selected), fragmentMapping ? 1u : rowExtent * 4u});
  }
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
    unsigned statePenalty = 0;
    unsigned peakGroups = 0;
    unsigned inverseLanes = 0;
  };
  using StateCombination = llvm::SmallVector<SelectedVLAStatePhysical, 4>;
  llvm::SmallVector<StateCombination, 8> stateCombinations(1);
  for (const VLAStateCandidateSet &stateSet : facts.stateCandidates) {
    if (stateSet.candidates.empty())
      return std::nullopt;
    llvm::SmallVector<StateCombination, 8> expanded;
    for (const StateCombination &combination : stateCombinations)
      for (const SelectedVLAStatePhysical &state : stateSet.candidates) {
        StateCombination candidate = combination;
        candidate.push_back(state);
        expanded.push_back(std::move(candidate));
      }
    stateCombinations = std::move(expanded);
  }
  llvm::SmallVector<Candidate, 8> legal;
  for (CorePhysicalMapping mapping :
       enumerateCorePhysicalMappings(problem, target)) {
    RVVVectorShape dataShape = mapping.laneShape;
    llvm::SmallVector<VLAElementPhysicalShape, 4> elementShapes;
    auto selectElementShape = [&](unsigned sew)
        -> std::optional<RVVVectorShape> {
      auto found = llvm::find_if(
          elementShapes, [&](const VLAElementPhysicalShape &selected) {
            return selected.sew == sew;
          });
      if (found != elementShapes.end())
        return found->shape;
      if (sew == 0 || (sew == 16 && !target.hasVectorF16))
        return std::nullopt;
      std::optional<RVVVectorShape> shape =
          rvvShapeForSameLanes(dataShape, sew, target);
      if (!shape)
        return std::nullopt;
      elementShapes.push_back(VLAElementPhysicalShape{sew, *shape});
      return shape;
    };
    if (!selectElementShape(facts.dataSEW))
      continue;
    const PhysicalAxisDecomposition *vlaAxis =
        mapping.laneAxis ? findAxisMapping(mapping, *mapping.laneAxis) : nullptr;
    if (!vlaAxis || vlaAxis->laneFactor == 0 ||
        (facts.requiredDataShape && dataShape != facts.requiredDataShape))
      continue;
    const unsigned mappedLaneFactor = vlaAxis->laneFactor;
    const bool needsIndexShape =
        facts.hasAffinePredicate || facts.hasIndexVector;
    std::optional<RVVVectorShape> indexShape =
        needsIndexShape
            ? selectElementShape(static_cast<unsigned>(target.xlen))
            : std::optional<RVVVectorShape>{};
    if (needsIndexShape && !indexShape)
      continue;

    bool elementShapesLegal =
        llvm::all_of(facts.accessElementSEWs, [&](unsigned sew) {
          return selectElementShape(sew).has_value();
        });
    if (!elementShapesLegal)
      continue;
    bool indexedShapesLegal =
        llvm::all_of(facts.indexedMemory,
                     [&](const VLAIndexedMemoryFact &memory) {
          std::optional<RVVVectorShape> elementShape =
              selectElementShape(memory.elementSEW);
          std::optional<RVVVectorShape> indexShape =
              selectElementShape(memory.offsetSEW);
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
              selectElementShape(memory.elementSEW);
          return memory.fields != 0 && elementShape &&
                 target.supportsSegmentVectorMemory(
                     memory.fields, elementShape->sew,
                     elementShape->lmulEighths);
        });
    if (!segmentShapesLegal)
      continue;

    for (const StateCombination &stateChoices : stateCombinations) {
    StateCombination states = stateChoices;
    if (!llvm::all_of(states, [&](SelectedVLAStatePhysical &state) {
          return resolveVLAStateShapes(state, dataShape, target);
        }))
      continue;

    llvm::SmallVector<PhysicalLiveRange, 8> persistent;
    for (const SelectedVLAStatePhysical &state : states)
      if (state.carry == VLAStateCarryRepresentation::Vector &&
          state.wholeVLALifetime)
        persistent.push_back(
            {PhysicalLiveClass::State, state.carryShape, 1, 0});

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
            selectElementShape(sew);
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
    auto applyPhase = [&](llvm::ArrayRef<PhysicalLiveRange> extra,
                          unsigned reservedGroups = 1) {
      VLAValueLifetimeSnapshot empty;
      llvm::ArrayRef<VLAValueLifetimeSnapshot> snapshots = facts.lifetimes;
      if (snapshots.empty())
        snapshots = llvm::ArrayRef<VLAValueLifetimeSnapshot>(&empty, 1);
      for (const VLAValueLifetimeSnapshot &snapshot : snapshots) {
        PhysicalResourceRequirements requirements;
        requirements.reservedGroups = reservedGroups;
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
    auto applyOperationPhase = [&](unsigned operationOrdinal,
                                   llvm::ArrayRef<PhysicalLiveRange> extra,
                                   unsigned reservedGroups = 1) {
      auto snapshot = llvm::find_if(
          facts.lifetimes, [&](const VLAValueLifetimeSnapshot &candidate) {
            return candidate.operationOrdinal == operationOrdinal;
          });
      if (snapshot == facts.lifetimes.end())
        return false;
      PhysicalResourceRequirements requirements;
      requirements.reservedGroups = reservedGroups;
      requirements.live.append(persistent.begin(), persistent.end());
      requirements.live.append(extra.begin(), extra.end());
      if (!appendSnapshot(requirements, *snapshot))
        return false;
      std::optional<PhysicalResourceBudget> budget =
          calculatePhysicalResources(requirements, target);
      if (!budget)
        return false;
      mergeBudget(*budget);
      return true;
    };
    if (!applyPhase({}))
      continue;

    bool resourceLegal = true;
    for (const VLAIndexedMemoryFact &memory : facts.indexedMemory) {
      std::optional<RVVVectorShape> elementShape =
          selectElementShape(memory.elementSEW);
      std::optional<RVVVectorShape> offsetShape =
          selectElementShape(memory.offsetSEW);
      if (!elementShape || !offsetShape ||
          !applyOperationPhase(
              memory.operationOrdinal,
              {PhysicalLiveRange{PhysicalLiveClass::Memory, *elementShape, 1, 0},
               PhysicalLiveRange{PhysicalLiveClass::Index, *offsetShape, 1, 0}})) {
        resourceLegal = false;
        break;
      }
    }
    if (!resourceLegal)
      continue;
    for (const VLASegmentMemoryFact &memory : facts.segmentMemory) {
      std::optional<RVVVectorShape> elementShape =
          selectElementShape(memory.elementSEW);
      if (!elementShape ||
          !applyOperationPhase(
              memory.operationOrdinal,
              {PhysicalLiveRange{PhysicalLiveClass::Memory, *elementShape,
                                 memory.fields, 0}})) {
        resourceLegal = false;
        break;
      }
    }
    if (!resourceLegal)
      continue;

    std::optional<VLANarrowPhysical> narrow;
    if (!facts.narrowOperationOrdinals.empty()) {
      if (facts.dataSEW != 32)
        continue;
      std::optional<RVVVectorShape> intermediateShape =
          selectElementShape(16);
      std::optional<RVVVectorShape> resultShape =
          selectElementShape(8);
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
      if (!narrowResources ||
          !llvm::all_of(facts.narrowOperationOrdinals, [&](unsigned ordinal) {
            return applyOperationPhase(ordinal, narrowRequirements.live);
          }))
        continue;
      selected.resources = *narrowResources;
      narrow = selected;
    }

    for (auto [stateIndex, state] : llvm::enumerate(states)) {
      llvm::SmallVector<PhysicalLiveRange, 6> transient;
      switch (state.stripUpdate) {
      case VLAStateStripUpdate::InclusiveAddScan:
        transient = {{PhysicalLiveClass::Temporary, state.inputShape, 3, 0},
                     {PhysicalLiveClass::Predicate, {}, 1, 1}};
        break;
      case VLAStateStripUpdate::SegmentedInclusiveAddScan:
        transient = {{PhysicalLiveClass::Temporary, state.inputShape, 4, 0},
                     {PhysicalLiveClass::Predicate, {}, 1, 2}};
        break;
      case VLAStateStripUpdate::AddReduction:
      case VLAStateStripUpdate::MaxReduction:
        transient = {{PhysicalLiveClass::Temporary, state.inputShape, 1, 0}};
        if (state.carry != VLAStateCarryRepresentation::Vector ||
            rvvRegisterGroups(state.carryShape) < 2)
          transient.push_back(
              {PhysicalLiveClass::Temporary, state.seedShape, 2, 0});
        break;
      case VLAStateStripUpdate::WideningAddReduction:
        transient = {{PhysicalLiveClass::Temporary, state.inputShape, 1, 0},
                     {PhysicalLiveClass::Temporary, state.seedShape, 1, 0}};
        break;
      case VLAStateStripUpdate::ArgMaxSummary:
        if (!indexShape) {
          indexShape = selectElementShape(static_cast<unsigned>(target.xlen));
        }
        if (!indexShape) {
          resourceLegal = false;
          break;
        }
        transient = {{PhysicalLiveClass::Temporary, state.inputShape, 1, 0},
                     {PhysicalLiveClass::Index, *indexShape, 1, 0},
                     {PhysicalLiveClass::Temporary, state.seedShape, 2, 0}};
        break;
      case VLAStateStripUpdate::OnlineSoftmaxSummary:
        transient = {{PhysicalLiveClass::Temporary, state.inputShape, 3, 0},
                     {PhysicalLiveClass::Temporary, state.seedShape, 4, 0}};
        break;
      }
      if (!resourceLegal ||
          !applyOperationPhase(
              facts.stateCandidates[stateIndex].operationOrdinal, transient)) {
        resourceLegal = false;
        break;
      }
    }
    if (!resourceLegal)
      continue;

    if (!facts.lookupOperationOrdinals.empty()) {
      std::optional<RVVVectorShape> codeShape =
          selectElementShape(8);
      std::optional<RVVVectorShape> index16Shape =
          selectElementShape(16);
      std::optional<RVVVectorShape> index32Shape =
          selectElementShape(32);
      if (!codeShape || !index16Shape || !index32Shape)
        continue;
      if (!llvm::all_of(facts.lookupOperationOrdinals, [&](unsigned ordinal) {
            return applyOperationPhase(
                       ordinal,
                       {{PhysicalLiveClass::Temporary, *codeShape, 1, 0},
                        {PhysicalLiveClass::Temporary, *index16Shape, 1, 0},
                        {PhysicalLiveClass::Temporary, *index32Shape, 1, 0}}) &&
                   applyOperationPhase(
                       ordinal,
                       {{PhysicalLiveClass::Index, *index32Shape, 1, 0},
                        {PhysicalLiveClass::Memory, *index32Shape, 1, 0},
                        {PhysicalLiveClass::Temporary, *index32Shape, 1, 0}});
          }))
        continue;
    }

    for (const VLALocalPrimitiveResourceFact &primitive :
         facts.localPrimitiveRequirements) {
      if (!applyOperationPhase(primitive.operationOrdinal,
                               primitive.requirements.live,
                               primitive.requirements.reservedGroups)) {
        resourceLegal = false;
        break;
      }
    }
    if (!resourceLegal)
      continue;

    SelectedVLAEntityPhysical selected;
    selected.mapping = mapping;
    selected.elementShapes = elementShapes;
    selected.states = states;
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
    const unsigned statePenalty = llvm::count_if(
        states, [](const SelectedVLAStatePhysical &state) {
          return state.carry == VLAStateCarryRepresentation::Scalar &&
                 (state.stripUpdate == VLAStateStripUpdate::AddReduction ||
                  state.stripUpdate == VLAStateStripUpdate::MaxReduction);
        });
    legal.push_back(Candidate{
        std::move(selected),
        static_cast<unsigned>(std::abs(static_cast<int>(laneGroups) -
                                       static_cast<int>(preferredGroups))),
        statePenalty,
        aggregate.peakGroups,
        std::numeric_limits<unsigned>::max() - mappedLaneFactor});
    }
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.lanePenalty, lhs.statePenalty, lhs.peakGroups,
                    lhs.inverseLanes) <
           std::tie(rhs.lanePenalty, rhs.statePenalty, rhs.peakGroups,
                    rhs.inverseLanes);
  });
  return std::move(legal.front().physical);
}

std::optional<SelectedSignBitI8Physical>
selectSignBitI8Physical(const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.littleEndian ||
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
calculateAxisMappedResources(const AxisMappedResourceFacts &facts,
                             const RISCVTargetProfile &target) {
  if (!facts.mapping)
    return std::nullopt;
  PhysicalResourceRequirements requirements;
  for (const AxisMappedLiveValue &value : facts.values) {
    const PhysicalAxisDecomposition *axis =
        value.axis ? findAxisMapping(*facts.mapping, *value.axis) : nullptr;
    unsigned count = value.factor;
    switch (value.multiplicity) {
    case AxisMappedMultiplicity::Fixed:
      break;
    case AxisMappedMultiplicity::RegisterFactor:
      if (!axis)
        return std::nullopt;
      count *= axis->registerFactor;
      break;
    case AxisMappedMultiplicity::LaneCapacity: {
      std::optional<unsigned> capacity = rvvLaneCapacity(value.shape, target);
      if (!axis || !capacity || *capacity == 0)
        return std::nullopt;
      const unsigned logicalLanes = axis->laneFactor * axis->registerFactor;
      count *= (logicalLanes + *capacity - 1) / *capacity;
      break;
    }
    case AxisMappedMultiplicity::LogicalChunk: {
      if (!axis || value.chunk == 0)
        return std::nullopt;
      const unsigned logicalLanes = axis->laneFactor * axis->registerFactor;
      count *= (logicalLanes + value.chunk - 1) / value.chunk;
      break;
    }
    }
    if (count == 0)
      return std::nullopt;
    requirements.live.push_back(
        PhysicalLiveRange{value.liveClass, value.shape, count});
  }
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
  if (!target.hasRVV || !target.hasWideningInteger || !target.littleEndian)
    return std::nullopt;

  const LogicalAxisConstraint *reduction =
      findUniqueLogicalAxisConstraint(facts.mapping, LogicalAxisRole::Reduction);
  if (!reduction || reduction->role != LogicalAxisRole::Reduction ||
      !reduction->extent)
    return std::nullopt;
  const bool base3 = facts.primitive == LocalPrimitiveKind::Base3TernaryI8;
  const bool packedI2 =
      facts.primitive == LocalPrimitiveKind::PackedI2TernaryI8;
  if (!base3 && !packedI2)
    return std::nullopt;
  if (base3) {
    if (facts.primaryExtent != 48 || facts.secondaryExtent != 4 ||
        *reduction->extent != 256)
      return std::nullopt;
  } else {
    if (facts.primaryExtent != 64 || facts.secondaryExtent != 0 ||
        *reduction->extent != 256)
      return std::nullopt;
  }

  std::optional<RVVVectorShape> byte16 =
      rvvShapeForSemanticLanes(8, 16, target);
  if (!byte16 ||
      !target.supportsVectorShape(kRVVE32M1.sew,
                                  kRVVE32M1.lmulEighths))
    return std::nullopt;
  struct Candidate {
    SelectedTernaryI8DotPhysical physical;
    unsigned registerFactor = 0;
  };
  llvm::SmallVector<Candidate, 8> legal;
  for (LocalAxisMappingCandidate candidate : enumerateRVVLocalAxisMappings(
           facts.mapping, CoreInstructionKind::RVVWideningIntegerDot,
           reduction->id, 8, target)) {
    const RVVVectorShape laneShape = candidate.mapping.laneShape;
    const unsigned logicalLanes =
        candidate.axis.laneFactor * candidate.axis.registerFactor;
    if (base3) {
      if (candidate.axis.laneFactor != 32 ||
          candidate.axis.registerFactor != 1 ||
          candidate.axis.sequentialFactor != 8)
        continue;
    } else if ((candidate.axis.laneFactor != 16 &&
                candidate.axis.laneFactor != 32) ||
               logicalLanes != 32 || candidate.axis.sequentialFactor != 8) {
      continue;
    }
    std::optional<RVVVectorShape> widened =
        rvvShapeForSameLanes(laneShape, 16, target);
    std::optional<RVVVectorShape> halfWidened =
        rvvShapeForSameLanes(*byte16, 16, target);
    RVVVectorShape highWord{32, byte16->lmulEighths};
    if (!widened || !halfWidened ||
        !target.supportsVectorShape(highWord.sew, highWord.lmulEighths))
      continue;
    LocalImplementation implementation;
    implementation.primitive = facts.primitive;
    implementation.mapping = std::move(candidate.mapping);
    implementation.valueShapes =
        packedI2
            ? llvm::SmallVector<RVVVectorShape, 4>{laneShape, *widened}
            : llvm::SmallVector<RVVVectorShape, 4>{
                  laneShape, *byte16, *widened, *halfWidened, highWord};
    selectLocalHardwareOperation(implementation,
                                 LocalHardwareOperationKind::RVVIntrinsic);
    if (!selectLocalMicrokernelSchedule(
            implementation, reduction->id, 1,
            {{0, PhysicalMemoryMode::UnitStride, 1, 1, true},
             {1, PhysicalMemoryMode::UnitStride,
              std::max(1u, candidate.axis.registerFactor), 1, true}},
            {base3 ? 5u : 4u,
             std::max(1u, candidate.axis.laneFactor / 16),
             std::max(1u, logicalLanes / 16)}))
      continue;
    if (!finalizeLocalHardwareOperation(implementation))
      continue;
    AxisMappedResourceFacts resources;
    resources.mapping = &implementation.mapping;
    resources.values = {
        {PhysicalLiveClass::Memory, laneShape,
         AxisMappedMultiplicity::RegisterFactor, candidate.axis.id, 1},
        {PhysicalLiveClass::Memory, *byte16}};
    if (base3) {
      resources.values.append(
          {{PhysicalLiveClass::Temporary, *widened,
            AxisMappedMultiplicity::RegisterFactor, candidate.axis.id, 3},
           {PhysicalLiveClass::Temporary, kRVVE32M1}});
    } else {
      resources.values.append(
          {{PhysicalLiveClass::Temporary, laneShape,
            AxisMappedMultiplicity::RegisterFactor, candidate.axis.id, 2},
           {PhysicalLiveClass::Temporary, *widened,
            AxisMappedMultiplicity::RegisterFactor, candidate.axis.id, 1},
           {PhysicalLiveClass::Temporary, kRVVE32M1}});
    }
    std::optional<PhysicalResourceBudget> budget =
        calculateAxisMappedResources(resources, target);
    if (!budget)
      continue;
    SelectedTernaryI8DotPhysical selected;
    selected.implementation = std::move(implementation);
    selected.primarySourceShape = laneShape;
    selected.secondarySourceShape = *byte16;
    selected.resources = *budget;
    legal.push_back(Candidate{std::move(selected),
                              candidate.axis.registerFactor});
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.physical.resources.peakGroups, lhs.registerFactor) <
           std::tie(rhs.physical.resources.peakGroups, rhs.registerFactor);
  });
  return std::move(legal.front().physical);
}

struct QuantPrimitivePhysicalRule {
  llvm::SmallVector<RVVVectorShape, 8> valueShapes;
  LocalHardwareOperationKind operation =
      LocalHardwareOperationKind::RVVIntrinsic;
  LocalOperationProjection projection = LocalOperationProjection::None;
  llvm::SmallVector<LocalOperandWindow, 4> operands;
  LocalDecodeSchedule decode;
  unsigned accumulatorCount = 1;
  unsigned entryWidth = 0;
  llvm::SmallVector<AxisMappedLiveValue, 10> liveValues;
  unsigned predicateGroups = 0;
  std::optional<PhysicalResourceRequirements> directResources;
};

struct LocalPrimitivePhysicalCandidate {
  LocalImplementation implementation;
  RVVVectorShape laneShape;
  PhysicalResourceBudget resources;
  unsigned sequentialFactor = 0;
  unsigned laneGroups = 0;
  unsigned registerFactor = 0;
};

template <typename RuleBuilder>
static llvm::SmallVector<LocalPrimitivePhysicalCandidate, 16>
enumerateLocalPrimitivePhysicalCandidates(
    const CoreMappingProblem &mapping, LocalPrimitiveKind primitive,
    CoreInstructionKind instruction, unsigned iterationAxis, unsigned laneSEW,
    const RISCVTargetProfile &target, RuleBuilder &&deriveRule) {
  llvm::SmallVector<LocalPrimitivePhysicalCandidate, 16> legal;
  for (LocalAxisMappingCandidate candidate : enumerateRVVLocalAxisMappings(
           mapping, instruction, iterationAxis, laneSEW, target)) {
    std::optional<QuantPrimitivePhysicalRule> rule = deriveRule(candidate);
    if (!rule)
      continue;
    const RVVVectorShape laneShape = candidate.mapping.laneShape;
    const unsigned sequentialFactor = candidate.axis.sequentialFactor;
    const unsigned registerFactor = candidate.axis.registerFactor;
    LocalImplementation implementation;
    implementation.primitive = primitive;
    implementation.mapping = std::move(candidate.mapping);
    implementation.valueShapes.append(rule->valueShapes.begin(),
                                      rule->valueShapes.end());
    implementation.entryWidth = rule->entryWidth;
    selectLocalHardwareOperation(implementation, rule->operation,
                                 rule->projection);
    if (!selectLocalMicrokernelSchedule(
            implementation, iterationAxis, rule->accumulatorCount,
            rule->operands, rule->decode) ||
        !finalizeLocalHardwareOperation(implementation))
      continue;
    std::optional<PhysicalResourceBudget> budget;
    if (rule->directResources) {
      budget = calculatePhysicalResources(*rule->directResources, target);
    } else {
      AxisMappedResourceFacts resources;
      resources.mapping = &implementation.mapping;
      resources.values.append(rule->liveValues.begin(), rule->liveValues.end());
      resources.predicateGroups = rule->predicateGroups;
      budget = calculateAxisMappedResources(resources, target);
    }
    if (!budget)
      continue;
    legal.push_back(LocalPrimitivePhysicalCandidate{
        std::move(implementation), laneShape, *budget, sequentialFactor,
        rvvRegisterGroups(laneShape), registerFactor});
  }
  return legal;
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
      !supportedPrimitive ||
      (facts.entryWidth != 4 && facts.entryWidth != 8) ||
      !target.supportsVectorShape(kRVVE8M1.sew, kRVVE8M1.lmulEighths))
    return std::nullopt;

  const LogicalAxisConstraint *reduction =
      findUniqueLogicalAxisConstraint(facts.mapping, LogicalAxisRole::Reduction);
  if (!reduction || reduction->role != LogicalAxisRole::Reduction ||
      reduction->extent != std::optional<uint64_t>(32))
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

  llvm::SmallVector<LocalPrimitivePhysicalCandidate, 16> legal =
      enumerateLocalPrimitivePhysicalCandidates(
          facts.mapping, facts.primitive,
          CoreInstructionKind::RVVIndexedGather, reduction->id, 8, target,
          [&](const LocalAxisMappingCandidate &candidate)
              -> std::optional<QuantPrimitivePhysicalRule> {
            const RVVVectorShape laneShape = candidate.mapping.laneShape;
            if (candidate.axis.laneFactor != 32 ||
                candidate.axis.registerFactor != 1 ||
                candidate.axis.sequentialFactor != 1)
              return std::nullopt;
            std::optional<RVVVectorShape> productShape =
                rvvShapeForSameLanes(laneShape, 16, target);
            if (!productShape)
              return std::nullopt;
            std::optional<RVVVectorShape> halfProductShape =
                rvvShapeForSemanticLanes(16, 16, target);
            RVVVectorShape signWordShape{32, kRVVE8M1.lmulEighths};
            if (!halfProductShape ||
                !target.supportsVectorShape(signWordShape.sew,
                                            signWordShape.lmulEighths))
              return std::nullopt;
            QuantPrimitivePhysicalRule rule;
            if (facts.primitive == LocalPrimitiveKind::SignedCodebook8I8 ||
                facts.primitive == LocalPrimitiveKind::SignedCodebook4I8)
              rule.valueShapes = {laneShape, *codeShape, *indexShape,
                                  *tableShape, *productShape, kRVVE8M1,
                                  signWordShape};
            else if (facts.primitive ==
                     LocalPrimitiveKind::PackedU9U7CodebookI8)
              rule.valueShapes = {laneShape, *codeShape, *indexShape,
                                  *tableShape, *productShape, kRVVE8M1,
                                  *halfProductShape, signWordShape};
            else if (facts.primitive ==
                     LocalPrimitiveKind::PackedU11GridDeltaI8)
              rule.valueShapes = {laneShape, *codeShape, *indexShape,
                                  *tableShape, *productShape};
            else
              return std::nullopt;
            rule.entryWidth = facts.entryWidth;
            if (facts.primitive == LocalPrimitiveKind::SignedCodebook8I8 ||
                facts.primitive == LocalPrimitiveKind::SignedCodebook4I8 ||
                facts.primitive == LocalPrimitiveKind::PackedU9U7CodebookI8)
              rule.projection =
                  rule.valueShapes[5] == laneShape
                      ? LocalOperationProjection::SignSourceDirect
                      : LocalOperationProjection::SignSourceExtend;
            rule.operands = {
                {0, PhysicalMemoryMode::UnitStride, 1, 1, true},
                {3, PhysicalMemoryMode::Indexed, 1, 1, false},
                {4, PhysicalMemoryMode::UnitStride,
                 std::max(1u, candidate.axis.registerFactor), 1, true}};
            rule.decode = {codeCount, 1, 2};
            rule.liveValues = {
                {PhysicalLiveClass::Memory, *codeShape},
                {PhysicalLiveClass::Memory, *tableShape},
                {PhysicalLiveClass::Memory, laneShape,
                 AxisMappedMultiplicity::RegisterFactor, candidate.axis.id},
                {PhysicalLiveClass::Index, *indexShape},
                {PhysicalLiveClass::Temporary, *tableShape},
                {PhysicalLiveClass::Temporary, laneShape,
                 AxisMappedMultiplicity::RegisterFactor, candidate.axis.id},
                {PhysicalLiveClass::Temporary, *productShape,
                 AxisMappedMultiplicity::RegisterFactor, candidate.axis.id},
                {PhysicalLiveClass::Temporary, kRVVE32M1,
                 AxisMappedMultiplicity::Fixed, std::nullopt, 2}};
            rule.predicateGroups = 1;
            return rule;
          });
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const LocalPrimitivePhysicalCandidate &lhs,
                       const LocalPrimitivePhysicalCandidate &rhs) {
    return std::tie(lhs.resources.peakGroups, lhs.registerFactor) <
           std::tie(rhs.resources.peakGroups, rhs.registerFactor);
  });
  LocalPrimitivePhysicalCandidate selected = std::move(legal.front());
  return SelectedCodebookGatherI8Physical{
      std::move(selected.implementation), *codeShape, selected.laneShape,
      selected.resources};
}

std::optional<SelectedNibbleCodebookI8Physical>
selectNibbleCodebookI8Physical(const NibbleCodebookI8CandidateFacts &facts,
                               const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasWideningInteger || !target.littleEndian ||
      facts.packedExtent != 16 ||
      facts.tableExtent != 16)
    return std::nullopt;
  const LogicalAxisConstraint *reduction =
      findUniqueLogicalAxisConstraint(facts.mapping, LogicalAxisRole::Reduction);
  if (!reduction || reduction->role != LogicalAxisRole::Reduction ||
      reduction->extent != std::optional<uint64_t>(32))
    return std::nullopt;
  std::optional<RVVVectorShape> packedShape =
      rvvShapeForSemanticLanes(8, 16, target);
  if (!packedShape ||
      !target.supportsVectorShape(kRVVE32M1.sew,
                                  kRVVE32M1.lmulEighths))
    return std::nullopt;
  llvm::SmallVector<LocalPrimitivePhysicalCandidate, 16> legal =
      enumerateLocalPrimitivePhysicalCandidates(
          facts.mapping, LocalPrimitiveKind::NibbleCodebookI8,
          CoreInstructionKind::RVVWideningIntegerDot, reduction->id, 8, target,
          [&](const LocalAxisMappingCandidate &candidate)
              -> std::optional<QuantPrimitivePhysicalRule> {
            const RVVVectorShape laneShape = candidate.mapping.laneShape;
            const unsigned logicalLanes =
                candidate.axis.laneFactor * candidate.axis.registerFactor;
            if ((candidate.axis.laneFactor != 16 &&
                 candidate.axis.laneFactor != 32) ||
                logicalLanes != 32 || candidate.axis.sequentialFactor != 1)
              return std::nullopt;
            const bool combined = candidate.axis.laneFactor == 32 &&
                                  packedShape->lmulEighths >= 8 &&
                                  packedShape->lmulEighths * 2 ==
                                      laneShape.lmulEighths;
            RVVVectorShape tableShape = combined ? laneShape : *packedShape;
            std::optional<RVVVectorShape> productShape =
                rvvShapeForSameLanes(combined ? laneShape : *packedShape, 16,
                                     target);
            if (!productShape)
              return std::nullopt;
            QuantPrimitivePhysicalRule rule;
            rule.valueShapes = {laneShape, *packedShape, tableShape,
                                *productShape};
            rule.projection =
                combined ? LocalOperationProjection::NibbleCodebookCombined
                         : LocalOperationProjection::NibbleCodebookSplit;
            rule.operands = {
                {1, PhysicalMemoryMode::UnitStride, 1, 1, true},
                {2, PhysicalMemoryMode::Indexed, 1, 1, false},
                {0, PhysicalMemoryMode::UnitStride,
                 std::max(1u, candidate.axis.registerFactor), 1, true}};
            rule.decode = {2,
                           std::max(1u, candidate.axis.laneFactor / 16),
                           std::max(1u, logicalLanes / 16)};
            rule.liveValues = {
                {PhysicalLiveClass::Memory, *packedShape},
                {PhysicalLiveClass::Memory, tableShape},
                {PhysicalLiveClass::Memory, laneShape,
                 AxisMappedMultiplicity::RegisterFactor, candidate.axis.id},
                {PhysicalLiveClass::Temporary, *productShape,
                 AxisMappedMultiplicity::RegisterFactor, candidate.axis.id,
                 combined ? 1u : 2u},
                {PhysicalLiveClass::Temporary, kRVVE32M1,
                 AxisMappedMultiplicity::Fixed, std::nullopt, 2}};
            return rule;
          });
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const LocalPrimitivePhysicalCandidate &lhs,
                       const LocalPrimitivePhysicalCandidate &rhs) {
    return std::tie(lhs.resources.peakGroups, lhs.registerFactor) <
           std::tie(rhs.resources.peakGroups, rhs.registerFactor);
  });
  LocalPrimitivePhysicalCandidate selected = std::move(legal.front());
  const RVVVectorShape selectedTableShape =
      selected.implementation.valueShapes[2];
  return SelectedNibbleCodebookI8Physical{
      std::move(selected.implementation), *packedShape,
      selectedTableShape, selected.laneShape, selected.resources};
}

static std::optional<QuantPrimitivePhysicalRule>
derivePackedQuantRule(LocalPrimitiveKind primitive,
                      const LocalAxisMappingCandidate &candidate,
                      const RISCVTargetProfile &target) {
  if (primitive != LocalPrimitiveKind::PackedI4I8 &&
      primitive != LocalPrimitiveKind::PackedI5I8)
    return std::nullopt;
  const RVVVectorShape laneShape = candidate.mapping.laneShape;
  const unsigned logicalLanes =
      candidate.axis.laneFactor * candidate.axis.registerFactor;
  if ((candidate.axis.laneFactor != 16 &&
       candidate.axis.laneFactor != 32) ||
      logicalLanes != 32 || candidate.axis.sequentialFactor != 1)
    return std::nullopt;
  RVVVectorShape decodeShape = laneShape;
  if (candidate.axis.laneFactor == 32) {
    std::optional<RVVVectorShape> halfShape =
        rvvShapeForSemanticLanes(8, 16, target);
    if (halfShape && halfShape->lmulEighths >= 8 &&
        halfShape->lmulEighths * 2 == laneShape.lmulEighths)
      decodeShape = *halfShape;
  }
  std::optional<RVVVectorShape> widenedShape =
      rvvShapeForSameLanes(laneShape, 16, target);
  if (!widenedShape)
    return std::nullopt;
  QuantPrimitivePhysicalRule rule;
  rule.valueShapes = {laneShape, decodeShape, *widenedShape};
  rule.projection =
      candidate.axis.laneFactor != 32
          ? LocalOperationProjection::PackedDotRegisterChunks
          : decodeShape == laneShape
                ? LocalOperationProjection::PackedDotLaneSlide
                : LocalOperationProjection::PackedDotLaneCreate;
  rule.operands = {
      {0, PhysicalMemoryMode::UnitStride,
       std::max(1u, candidate.axis.registerFactor), 1, true},
      {1, PhysicalMemoryMode::UnitStride,
       std::max(1u, candidate.axis.registerFactor), 1, true}};
  rule.decode = {2, std::max(1u, candidate.axis.laneFactor / 16),
                 std::max(1u, logicalLanes / 16)};
  rule.liveValues = {
      {PhysicalLiveClass::Memory, laneShape,
       AxisMappedMultiplicity::RegisterFactor, candidate.axis.id, 2},
      {PhysicalLiveClass::Temporary, *widenedShape,
       AxisMappedMultiplicity::RegisterFactor, candidate.axis.id, 2},
      {PhysicalLiveClass::Temporary, kRVVE32M1,
       AxisMappedMultiplicity::LogicalChunk, candidate.axis.id, 4, 16}};
  return rule;
}

static std::optional<QuantPrimitivePhysicalRule>
derivePackedI3QuantRule(const LocalAxisMappingCandidate &candidate,
                        const RISCVTargetProfile &target) {
  const RVVVectorShape laneShape = candidate.mapping.laneShape;
  const unsigned logicalLanes =
      candidate.axis.laneFactor * candidate.axis.registerFactor;
  if ((candidate.axis.laneFactor != 16 &&
       candidate.axis.laneFactor != 32) ||
      logicalLanes == 0 || (256 % logicalLanes) != 0 ||
      candidate.axis.sequentialFactor != 256 / logicalLanes)
    return std::nullopt;
  std::optional<RVVVectorShape> productShape =
      rvvShapeForSameLanes(laneShape, 16, target);
  std::optional<RVVVectorShape> segmentProductShape =
      rvvShapeForSemanticLanes(16, 16, target);
  if (!productShape || !segmentProductShape)
    return std::nullopt;
  QuantPrimitivePhysicalRule rule;
  rule.valueShapes = {laneShape, *productShape, *segmentProductShape};
  rule.operands = {
      {0, PhysicalMemoryMode::UnitStride, 3, 1, true},
      {1, PhysicalMemoryMode::UnitStride, 1, 1, true}};
  rule.decode = {3, std::max(1u, candidate.axis.laneFactor / 16),
                 std::max(1u, logicalLanes / 16)};
  rule.liveValues = {
      {PhysicalLiveClass::Memory, laneShape,
       AxisMappedMultiplicity::RegisterFactor, candidate.axis.id, 3},
      {PhysicalLiveClass::Temporary, laneShape,
       AxisMappedMultiplicity::RegisterFactor, candidate.axis.id},
      {PhysicalLiveClass::Temporary, *productShape,
       AxisMappedMultiplicity::RegisterFactor, candidate.axis.id},
      {PhysicalLiveClass::Temporary, *segmentProductShape,
       AxisMappedMultiplicity::RegisterFactor, candidate.axis.id,
       candidate.axis.laneFactor / 16},
      {PhysicalLiveClass::Temporary, kRVVE32M1}};
  rule.predicateGroups = 1;
  return rule;
}

static std::optional<QuantPrimitivePhysicalRule>
deriveIQ2QuantRule(const LocalAxisMappingCandidate &candidate,
                   const RISCVTargetProfile &target) {
  if ((candidate.axis.laneFactor != 32 &&
       candidate.axis.laneFactor != 64) ||
      candidate.axis.registerFactor != 1 ||
      candidate.axis.sequentialFactor != 256 / candidate.axis.laneFactor)
    return std::nullopt;
  const RVVVectorShape laneShape = candidate.mapping.laneShape;
  const unsigned groupsPerVector = candidate.axis.laneFactor / 32;
  std::optional<RVVVectorShape> indexShape =
      rvvShapeForSemanticLanes(16, 4 * groupsPerVector, target);
  RVVVectorShape tableShape{64, laneShape.lmulEighths};
  std::optional<RVVVectorShape> signShape =
      rvvShapeForSemanticLanes(8, 4 * groupsPerVector, target);
  std::optional<RVVVectorShape> productShape =
      rvvShapeForSameLanes(laneShape, 16, target);
  std::optional<RVVVectorShape> segmentProductShape =
      rvvShapeForSemanticLanes(16, 16, target);
  if (!target.hasIndexedMemory || !indexShape || !signShape || !productShape ||
      !segmentProductShape ||
      !target.supportsIndexedVectorMemory(
          tableShape.sew, tableShape.lmulEighths, indexShape->sew,
          indexShape->lmulEighths))
    return std::nullopt;
  QuantPrimitivePhysicalRule rule;
  rule.valueShapes = {laneShape, *indexShape, tableShape, *signShape,
                      *productShape, *segmentProductShape};
  rule.operands = {
      {0, PhysicalMemoryMode::UnitStride, 1, 1, true},
      {2, PhysicalMemoryMode::Indexed, 1, 1, false},
      {3, PhysicalMemoryMode::UnitStride, 1, 1, true}};
  rule.decode = {4 * groupsPerVector, groupsPerVector,
                 std::max(1u, candidate.axis.laneFactor / 16)};
  rule.liveValues = {
      {PhysicalLiveClass::Index, *indexShape},
      {PhysicalLiveClass::Memory, tableShape},
      {PhysicalLiveClass::Memory, *signShape},
      {PhysicalLiveClass::Memory, laneShape},
      {PhysicalLiveClass::Temporary, *productShape},
      {PhysicalLiveClass::Temporary, *segmentProductShape,
       AxisMappedMultiplicity::Fixed, std::nullopt,
       candidate.axis.laneFactor / 16},
      {PhysicalLiveClass::Temporary, kRVVE32M1}};
  rule.predicateGroups = 1;
  return rule;
}

static std::optional<QuantPrimitivePhysicalRule>
deriveIQ3QuantRule(const LocalAxisMappingCandidate &candidate,
                   const RISCVTargetProfile &target) {
  if (candidate.axis.laneFactor != 64 ||
      candidate.axis.registerFactor != 1 ||
      candidate.axis.sequentialFactor != 4)
    return std::nullopt;
  const RVVVectorShape laneShape = candidate.mapping.laneShape;
  std::optional<RVVVectorShape> codeShape =
      rvvShapeForSemanticLanes(8, 16, target);
  std::optional<RVVVectorShape> indexShape =
      rvvShapeForSemanticLanes(16, 16, target);
  RVVVectorShape tableShape{32, laneShape.lmulEighths};
  std::optional<RVVVectorShape> signShape =
      rvvShapeForSemanticLanes(8, 8, target);
  std::optional<RVVVectorShape> productShape =
      rvvShapeForSameLanes(laneShape, 16, target);
  std::optional<RVVVectorShape> halfProductShape =
      rvvShapeForSemanticLanes(16, 32, target);
  if (!target.hasIndexedMemory || !codeShape || !indexShape || !signShape ||
      !productShape || !halfProductShape ||
      !target.supportsIndexedVectorMemory(
          tableShape.sew, tableShape.lmulEighths, indexShape->sew,
          indexShape->lmulEighths))
    return std::nullopt;
  QuantPrimitivePhysicalRule rule;
  rule.valueShapes = {laneShape, *codeShape, *indexShape, tableShape,
                      *signShape, *productShape, *halfProductShape};
  rule.operands = {
      {1, PhysicalMemoryMode::UnitStride, 1, 1, true},
      {3, PhysicalMemoryMode::Indexed, 1, 1, false},
      {4, PhysicalMemoryMode::UnitStride, 1, 1, true}};
  rule.decode = {3, 2, 2};
  rule.liveValues = {
      {PhysicalLiveClass::Memory, *codeShape},
      {PhysicalLiveClass::Index, *indexShape},
      {PhysicalLiveClass::Memory, tableShape},
      {PhysicalLiveClass::Memory, *signShape},
      {PhysicalLiveClass::Memory, laneShape},
      {PhysicalLiveClass::Temporary, *productShape},
      {PhysicalLiveClass::Temporary, *halfProductShape,
       AxisMappedMultiplicity::Fixed, std::nullopt, 2},
      {PhysicalLiveClass::Temporary, kRVVE32M1}};
  rule.predicateGroups = 1;
  return rule;
}

static std::optional<QuantPrimitivePhysicalRule>
deriveIQ1QuantRule(const LocalAxisMappingCandidate &candidate,
                   const RISCVTargetProfile &target) {
  if ((candidate.axis.laneFactor != 32 &&
       candidate.axis.laneFactor != 64) ||
      candidate.axis.registerFactor != 1 ||
      candidate.axis.sequentialFactor != 256 / candidate.axis.laneFactor)
    return std::nullopt;
  const RVVVectorShape laneShape = candidate.mapping.laneShape;
  const unsigned groupsPerVector = candidate.axis.laneFactor / 32;
  std::optional<RVVVectorShape> indexShape =
      rvvShapeForSemanticLanes(16, 4 * groupsPerVector, target);
  RVVVectorShape tableShape{64, laneShape.lmulEighths};
  std::optional<RVVVectorShape> productShape =
      rvvShapeForSameLanes(laneShape, 16, target);
  std::optional<RVVVectorShape> segmentProductShape =
      rvvShapeForSemanticLanes(16, 16, target);
  if (!target.hasIndexedMemory || !indexShape || !productShape ||
      !segmentProductShape ||
      !target.supportsIndexedVectorMemory(
          tableShape.sew, tableShape.lmulEighths, indexShape->sew,
          indexShape->lmulEighths))
    return std::nullopt;
  QuantPrimitivePhysicalRule rule;
  rule.valueShapes = {laneShape, *indexShape, tableShape, tableShape,
                      *productShape, *segmentProductShape};
  rule.operands = {
      {0, PhysicalMemoryMode::UnitStride, 1, 1, true},
      {2, PhysicalMemoryMode::Indexed, 2, 1, false},
      {3, PhysicalMemoryMode::UnitStride, 1, 1, true}};
  rule.decode = {4 * groupsPerVector, groupsPerVector,
                 std::max(1u, candidate.axis.laneFactor / 16)};
  rule.liveValues = {
      {PhysicalLiveClass::Index, *indexShape},
      {PhysicalLiveClass::Memory, tableShape},
      {PhysicalLiveClass::Memory, tableShape},
      {PhysicalLiveClass::Memory, laneShape},
      {PhysicalLiveClass::Temporary, *productShape,
       AxisMappedMultiplicity::Fixed, std::nullopt, 2},
      {PhysicalLiveClass::Temporary, *segmentProductShape,
       AxisMappedMultiplicity::Fixed, std::nullopt,
       2 * (candidate.axis.laneFactor / 16)},
      {PhysicalLiveClass::Temporary, kRVVE32M1}};
  return rule;
}

static std::optional<QuantPrimitivePhysicalRule>
deriveQ6QuantRule(const LocalAxisMappingCandidate &candidate,
                  const RISCVTargetProfile &target) {
  if ((candidate.axis.laneFactor != 32 &&
       candidate.axis.laneFactor != 64) ||
      candidate.axis.registerFactor != 1 ||
      candidate.axis.sequentialFactor != 256 / candidate.axis.laneFactor)
    return std::nullopt;
  const RVVVectorShape laneShape = candidate.mapping.laneShape;
  std::optional<RVVVectorShape> chunkShape =
      rvvShapeForSemanticLanes(8, 32, target);
  std::optional<RVVVectorShape> productShape =
      rvvShapeForSameLanes(laneShape, 16, target);
  std::optional<RVVVectorShape> segmentProductShape =
      rvvShapeForSemanticLanes(16, 16, target);
  if (!chunkShape || !productShape || !segmentProductShape)
    return std::nullopt;
  QuantPrimitivePhysicalRule rule;
  rule.valueShapes = {laneShape, *chunkShape, *productShape,
                      *segmentProductShape};
  rule.operands = {
      {1, PhysicalMemoryMode::UnitStride,
       3 * std::max(1u, candidate.axis.laneFactor / 32), 1, true},
      {0, PhysicalMemoryMode::UnitStride, 1, 1, true}};
  rule.decode = {std::max(1u, candidate.axis.laneFactor / 32), 3,
                 std::max(1u, candidate.axis.laneFactor / 16)};
  const unsigned logicalLanes =
      candidate.axis.laneFactor * candidate.axis.registerFactor;
  if (logicalLanes == 32 && laneShape == RVVVectorShape{8, 16}) {
    rule.operation = LocalHardwareOperationKind::RVVInlineAsm;
    rule.decode = {2, 4, 8};
    PhysicalResourceRequirements resources;
    resources.reservedGroups = 0;
    resources.live = {{PhysicalLiveClass::Temporary, {}, 1, 32}};
    rule.directResources = std::move(resources);
    return rule;
  }
  rule.liveValues = {
      {PhysicalLiveClass::Memory, *chunkShape,
       AxisMappedMultiplicity::Fixed, std::nullopt,
       3 * (candidate.axis.laneFactor / 32)},
      {PhysicalLiveClass::Temporary, *chunkShape,
       AxisMappedMultiplicity::Fixed, std::nullopt,
       2 * (candidate.axis.laneFactor / 32)},
      {PhysicalLiveClass::Memory, laneShape},
      {PhysicalLiveClass::Temporary, *productShape},
      {PhysicalLiveClass::Temporary, *segmentProductShape,
       AxisMappedMultiplicity::Fixed, std::nullopt,
       candidate.axis.laneFactor / 16},
      {PhysicalLiveClass::Temporary, kRVVE32M1}};
  return rule;
}

static std::optional<QuantPrimitivePhysicalRule>
deriveQuantPrimitiveRule(LocalPrimitiveKind primitive,
                         const LocalAxisMappingCandidate &candidate,
                         const RISCVTargetProfile &target) {
  switch (primitive) {
  case LocalPrimitiveKind::PackedI4I8:
  case LocalPrimitiveKind::PackedI5I8:
    return derivePackedQuantRule(primitive, candidate, target);
  case LocalPrimitiveKind::PackedI3GroupedI8:
    return derivePackedI3QuantRule(candidate, target);
  case LocalPrimitiveKind::IQ2SI8:
    return deriveIQ2QuantRule(candidate, target);
  case LocalPrimitiveKind::IQ3SI8:
    return deriveIQ3QuantRule(candidate, target);
  case LocalPrimitiveKind::IQ1MI8:
    return deriveIQ1QuantRule(candidate, target);
  case LocalPrimitiveKind::Q6KI8:
    return deriveQ6QuantRule(candidate, target);
  default:
    return std::nullopt;
  }
}

std::optional<SelectedQuantI8DotPhysical>
selectQuantI8DotPhysical(const QuantI8DotCandidateFacts &facts,
                         const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.hasWideningInteger ||
      !target.littleEndian ||
      !target.supportsVectorShape(kRVVE32M1.sew,
                                  kRVVE32M1.lmulEighths))
    return std::nullopt;
  const LogicalAxisConstraint *reduction =
      findUniqueLogicalAxisConstraint(facts.mapping, LogicalAxisRole::Reduction);
  if (!reduction || reduction->role != LogicalAxisRole::Reduction ||
      !reduction->extent || *reduction->extent == 0)
    return std::nullopt;
  const LocalPrimitiveKind primitive = facts.primitive;
  llvm::SmallVector<LocalPrimitivePhysicalCandidate, 16> legal =
      enumerateLocalPrimitivePhysicalCandidates(
          facts.mapping, primitive,
          CoreInstructionKind::RVVWideningIntegerDot, reduction->id, 8, target,
          [&](const LocalAxisMappingCandidate &candidate) {
            return deriveQuantPrimitiveRule(primitive, candidate, target);
          });
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const LocalPrimitivePhysicalCandidate &lhs,
                       const LocalPrimitivePhysicalCandidate &rhs) {
    return std::tie(lhs.sequentialFactor, lhs.laneGroups,
                    lhs.resources.peakGroups, lhs.registerFactor) <
           std::tie(rhs.sequentialFactor, rhs.laneGroups,
                    rhs.resources.peakGroups, rhs.registerFactor);
  });
  LocalPrimitivePhysicalCandidate selected = std::move(legal.front());
  return SelectedQuantI8DotPhysical{std::move(selected.implementation),
                                    selected.laneShape, selected.resources};
}

std::optional<SelectedE2M1E8M0I8Physical>
selectE2M1E8M0I8Physical(const E2M1E8M0I8CandidateFacts &facts,
                         const RISCVTargetProfile &target) {
  if (!target.hasRVV || !target.littleEndian ||
      facts.packedExtent != 16)
    return std::nullopt;
  const LogicalAxisConstraint *reduction =
      findUniqueLogicalAxisConstraint(facts.mapping, LogicalAxisRole::Reduction);
  if (!reduction || reduction->role != LogicalAxisRole::Reduction ||
      reduction->extent != std::optional<uint64_t>(32))
    return std::nullopt;

  struct Candidate {
    SelectedE2M1E8M0I8Physical physical;
    unsigned sequentialFactor = 0;
    unsigned registerFactor = 0;
  };
  llvm::SmallVector<Candidate, 16> legal;
  llvm::SmallVector<RVVVectorShape, 8> laneShapes =
      rvvShapeCandidates(target, 8);
  for (LocalAxisMappingCandidate candidate : enumerateRVVLocalAxisMappings(
           facts.mapping, CoreInstructionKind::RVVWideningIntegerDot,
           reduction->id, 8, target)) {
    const RVVVectorShape laneShape = candidate.mapping.laneShape;
    llvm::SmallVector<RVVVectorShape, 9> activationShapes(
        laneShapes.begin(), laneShapes.end());
    activationShapes.push_back({});
    for (RVVVectorShape activationShape : activationShapes) {
      LocalImplementation implementation;
      implementation.primitive = LocalPrimitiveKind::E2M1E8M0I8;
      implementation.mapping = candidate.mapping;
      implementation.valueShapes = {laneShape};
      if (activationShape)
        implementation.valueShapes.push_back(activationShape);
      const unsigned logicalLanes =
          candidate.axis.laneFactor * candidate.axis.registerFactor;
      const bool sequential = candidate.axis.sequentialFactor > 1;
      const bool registerMF2 =
          logicalLanes == 32 && laneShape == RVVVectorShape{8, 4} &&
          activationShape == RVVVectorShape{8, 4};
      const bool registerM1M2 =
          logicalLanes == 32 && laneShape == kRVVE8M1 &&
          activationShape == RVVVectorShape{8, 16};
      const bool strip = sequential && laneShape == kRVVE8M1 &&
                         !activationShape;
      if (!registerMF2 && !registerM1M2 && !strip)
        continue;
      selectLocalHardwareOperation(implementation,
                                   LocalHardwareOperationKind::RVVIntrinsic);
      if (!selectLocalMicrokernelSchedule(
              implementation, reduction->id, 1,
              {{0, PhysicalMemoryMode::UnitStride, 1, 1, true},
               {1, PhysicalMemoryMode::UnitStride,
                activationShape ? 1u : 0u, 1, true}},
              {1, std::max(1u, logicalLanes / 16),
               std::max(1u, logicalLanes / 16)}))
        continue;
      if (!finalizeLocalHardwareOperation(implementation))
        continue;
      RVVVectorShape selectedActivation =
          activationShape ? activationShape : laneShape;
      std::optional<RVVVectorShape> productShape =
          rvvShapeForSameLanes(selectedActivation, 16, target);
      if (!productShape)
        continue;
      AxisMappedResourceFacts resources;
      resources.mapping = &implementation.mapping;
      resources.values = {
          {PhysicalLiveClass::Memory, laneShape,
           AxisMappedMultiplicity::RegisterFactor, candidate.axis.id},
          {PhysicalLiveClass::Memory, selectedActivation,
           AxisMappedMultiplicity::LaneCapacity, candidate.axis.id},
          {PhysicalLiveClass::Temporary, laneShape,
           AxisMappedMultiplicity::RegisterFactor, candidate.axis.id, 2},
          {PhysicalLiveClass::Temporary, selectedActivation,
           AxisMappedMultiplicity::LaneCapacity, candidate.axis.id, 2},
          {PhysicalLiveClass::Temporary, *productShape,
           AxisMappedMultiplicity::LaneCapacity, candidate.axis.id},
          {PhysicalLiveClass::Temporary, kRVVE32M1,
           AxisMappedMultiplicity::Fixed, std::nullopt, 2}};
      std::optional<PhysicalResourceBudget> budget =
          calculateAxisMappedResources(resources, target);
      if (!budget)
        continue;
      SelectedE2M1E8M0I8Physical selected;
      selected.implementation = std::move(implementation);
      selected.packedShape = laneShape;
      selected.activationShape = selectedActivation;
      selected.resources = *budget;
      legal.push_back(Candidate{std::move(selected),
                                candidate.axis.sequentialFactor,
                                candidate.axis.registerFactor});
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
  if (!target.hasRVV || !target.hasWideningInteger || !target.littleEndian)
    return std::nullopt;
  const LogicalAxisConstraint *rows = findUniqueLogicalAxisConstraint(
      facts.mapping, LogicalAxisRole::Free);
  const LogicalAxisConstraint *reduction = findUniqueLogicalAxisConstraint(
      facts.mapping, LogicalAxisRole::Reduction);
  const LogicalAxisConstraint *groups = findUniqueLogicalAxisConstraint(
      facts.mapping, LogicalAxisRole::Group);
  const LogicalAxisConstraint *packed = findUniqueLogicalAxisConstraint(
      facts.mapping, LogicalAxisRole::Packed);
  if (!rows || !rows->extent || *rows->extent == 0 ||
      rows->role != LogicalAxisRole::Free || !reduction ||
      !reduction->extent || *reduction->extent == 0 ||
      reduction->role != LogicalAxisRole::Reduction || !groups ||
      !groups->extent || *groups->extent == 0 ||
      groups->role != LogicalAxisRole::Group || !packed ||
      !packed->extent || *packed->extent == 0 ||
      packed->role != LogicalAxisRole::Packed || facts.scaleMinExtent == 0 ||
      facts.activationSumExtent == 0)
    return std::nullopt;

  std::optional<RVVVectorShape> scaleShape =
      rvvShapeForSemanticLanes(8, facts.scaleMinExtent, target);
  std::optional<RVVVectorShape> activationSumShape =
      rvvShapeForSemanticLanes(8, facts.activationSumExtent, target);
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
    constraint.allowLane = constraint.id == reduction->id;
    constraint.requireLane = constraint.allowLane;
    constraint.registerFactors =
        constraint.id == reduction->id
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
    const PhysicalAxisDecomposition *mappedReduction =
        findAxisMapping(mapping, reduction->id);
    if (!mappedReduction ||
        mappedReduction->laneFactor * mappedReduction->registerFactor > 32)
      continue;
    SelectedGroupedAffineI4I8Physical candidate = selected;
    candidate.implementation.primitive =
        LocalPrimitiveKind::GroupedAffineI4I8;
    candidate.implementation.mapping = mapping;
    candidate.implementation.valueShapes = {candidate.packedShape,
                                            candidate.widenedShape};
    const bool registerAsm =
        mappedReduction->sequentialFactor == 1 &&
        mappedReduction->laneFactor == 16;
    selectLocalHardwareOperation(
        candidate.implementation,
        registerAsm ? LocalHardwareOperationKind::RVVInlineAsm
                    : LocalHardwareOperationKind::RVVIntrinsic);
    if (!selectLocalMicrokernelSchedule(
            candidate.implementation, reduction->id, 1,
            {{0, PhysicalMemoryMode::UnitStride, 1, 1, true},
             {1, PhysicalMemoryMode::UnitStride, 1, 1, true},
             {2, PhysicalMemoryMode::UnitStride, 1, 1, false}},
            {2, std::max(1u, mappedReduction->laneFactor / 16),
             std::max(1u, mappedReduction->laneFactor / 16)}))
      continue;
    if (!finalizeLocalHardwareOperation(candidate.implementation))
      continue;
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
      AxisMappedResourceFacts resources;
      resources.mapping = &candidate.implementation.mapping;
      resources.values = {
          {PhysicalLiveClass::Memory, candidate.packedShape,
           AxisMappedMultiplicity::RegisterFactor, mappedReduction->id},
          {PhysicalLiveClass::Memory, candidate.scaleShape},
          {PhysicalLiveClass::Memory, candidate.activationShape,
           AxisMappedMultiplicity::RegisterFactor, mappedReduction->id},
          {PhysicalLiveClass::Memory, candidate.activationSumShape},
          {PhysicalLiveClass::Temporary, candidate.widenedShape,
           AxisMappedMultiplicity::RegisterFactor, mappedReduction->id},
          {PhysicalLiveClass::Temporary, candidate.reductionShape,
           AxisMappedMultiplicity::Fixed, std::nullopt, 2}};
      std::optional<PhysicalResourceBudget> budget =
          calculateAxisMappedResources(resources, target);
      if (!budget)
        continue;
      candidate.resources = *budget;
    }
    legal.push_back(
        Candidate{std::move(candidate), mappedReduction->sequentialFactor});
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.sequentialFactor, lhs.physical.resources.peakGroups) <
           std::tie(rhs.sequentialFactor, rhs.physical.resources.peakGroups);
  });
  return std::move(legal.front().physical);
}

static std::optional<PhysicalResourceRequirements>
localMicrokernelRequirements(const LocalMicrokernelSchedule &schedule,
                             RVVVectorShape inputShape,
                             RVVVectorShape accumulatorShape,
                             unsigned predicateGroups,
                             unsigned stateGroups,
                             unsigned handoffGroups,
                             const RISCVTargetProfile &target) {
  if (!schedule || !inputShape || !accumulatorShape ||
      !target.supportsVectorShape(inputShape.sew,
                                  inputShape.lmulEighths) ||
      !target.supportsVectorShape(accumulatorShape.sew,
                                  accumulatorShape.lmulEighths))
    return std::nullopt;
  PhysicalResourceRequirements requirements;
  requirements.live.push_back(
      PhysicalLiveRange{PhysicalLiveClass::Value, accumulatorShape,
                        schedule.accumulatorCount});
  for (const LocalOperandWindow &operand : schedule.operands) {
    if (operand.reuseCount == 0)
      return std::nullopt;
    if (operand.vectorsPerStep == 0)
      continue;
    const unsigned copies =
        operand.advancesIteration ? schedule.pipeline.bufferCount : 1;
    requirements.live.push_back(
        PhysicalLiveRange{PhysicalLiveClass::Memory, inputShape,
                          operand.vectorsPerStep * copies});
  }
  if (predicateGroups != 0)
    requirements.live.push_back(PhysicalLiveRange{
        PhysicalLiveClass::Predicate, {}, 1, predicateGroups});
  if (stateGroups != 0)
    requirements.live.push_back(PhysicalLiveRange{
        PhysicalLiveClass::State, {}, 1, stateGroups});
  if (handoffGroups != 0)
    requirements.live.push_back(PhysicalLiveRange{
        PhysicalLiveClass::Handoff, {}, 1, handoffGroups});
  return requirements;
}

std::optional<PhysicalResourceBudget>
calculateLocalMicrokernelResources(const LocalMicrokernelSchedule &schedule,
                                   RVVVectorShape inputShape,
                                   RVVVectorShape accumulatorShape,
                                   unsigned predicateGroups,
                                   unsigned stateGroups,
                                   unsigned handoffGroups,
                                   const RISCVTargetProfile &target) {
  std::optional<PhysicalResourceRequirements> requirements =
      localMicrokernelRequirements(schedule, inputShape, accumulatorShape,
                                   predicateGroups, stateGroups,
                                   handoffGroups, target);
  if (!requirements)
    return std::nullopt;
  return calculatePhysicalResources(*requirements, target);
}

static llvm::SmallVector<unsigned, 4> localPipelineBufferCandidates(
    const LocalScheduleDependenceFacts &facts, unsigned maximumDepth) {
  llvm::SmallVector<unsigned, 4> candidates{1};
  if (facts.operands.empty() || llvm::any_of(
                                    facts.operands,
                                    [](const LocalOperandDependenceFacts &operand) {
                                      return !operand.advancesIteration ||
                                             !operand.feedsPrimitive ||
                                             operand.addressDependsOnAccumulator ||
                                             operand.predicateDependsOnAccumulator ||
                                             operand.crossesControl;
                                    }))
    return candidates;
  for (unsigned depth = 2; depth <= maximumDepth; ++depth)
    candidates.push_back(depth);
  return candidates;
}

std::optional<SelectedF32DotPhysical>
selectF32DotPhysicalConfig(const F32DotCandidateFacts &facts,
                           const RISCVTargetProfile &target,
                           const RISCVBackendConfig &config) {
  if (config.parameters.dotLoadBufferCount < 0 ||
      config.parameters.dotLoadBufferCount > 4)
    return std::nullopt;
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
  problem.pipelineBufferCandidates =
      config.parameters.dotLoadBufferCount != 0
          ? llvm::SmallVector<unsigned, 4>{
                static_cast<unsigned>(config.parameters.dotLoadBufferCount)}
          : localPipelineBufferCandidates(facts.schedule, 4);
  const llvm::SmallVector<unsigned, 4> legalPipelineBuffers =
      localPipelineBufferCandidates(facts.schedule, 4);

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
    LocalMicrokernelSchedule schedule;
    PhysicalResourceRequirements requirements;
    PhysicalResourceBudget resources;
    unsigned tailPenalty = 0;
    unsigned sequentialPenalty = 0;
    unsigned memoryPenalty = 0;
    unsigned lanePenalty = 0;
    unsigned unrollPenalty = 0;
    unsigned pipelinePenalty = 0;
    unsigned peakGroups = 0;
  };
  llvm::SmallVector<Candidate, 32> legal;
  for (CorePhysicalMapping mapping :
       enumerateCorePhysicalMappings(problem, target)) {
    const PhysicalAxisDecomposition *laneAxis =
        mapping.laneAxis ? findAxisMapping(mapping, *mapping.laneAxis) : nullptr;
    const PhysicalAxisDecomposition *reduction =
        findAxisMapping(mapping, facts.reductionAxis);
    if (!laneAxis || !reduction ||
        (reduction->unrollFactor != 1 && reduction->unrollFactor != 2 &&
         reduction->unrollFactor != 4) ||
        mapping.pipeline.bufferCount == 0 ||
        mapping.pipeline.bufferCount > reduction->unrollFactor ||
        (mapping.pipeline.bufferCount > 1 &&
         !llvm::is_contained(legalPipelineBuffers,
                             mapping.pipeline.bufferCount)))
      continue;
    unsigned accumulatorVectors = 1;
    for (const PhysicalAxisDecomposition &axis : mapping.axes)
      if (axis.role == LogicalAxisRole::Free)
        accumulatorVectors *= axis.registerFactor;
    auto operandVectorsPerWindow = [&](llvm::ArrayRef<unsigned> operandAxes) {
      if (!llvm::is_contained(operandAxes, laneAxis->id))
        return 0u;
      unsigned vectors = 1;
      for (const PhysicalAxisDecomposition &axis : mapping.axes)
        if (axis.id != laneAxis->id &&
            axis.role == LogicalAxisRole::Free &&
            llvm::is_contained(operandAxes, axis.id))
          vectors *= axis.registerFactor;
      return vectors;
    };
    const unsigned lhsVectorsPerWindow =
        operandVectorsPerWindow(facts.lhsAxes);
    const unsigned rhsVectorsPerWindow =
        operandVectorsPerWindow(facts.rhsAxes);
    if (lhsVectorsPerWindow == 0 && rhsVectorsPerWindow == 0)
      continue;
    if (facts.schedule.operands.size() != 2)
      continue;
    llvm::SmallVector<LocalOperandWindow, 2> operandWindows = {
        {0, facts.schedule.operands[0].memoryMode, lhsVectorsPerWindow,
         std::max(1u, accumulatorVectors /
                          std::max(1u, lhsVectorsPerWindow)),
         facts.schedule.operands[0].advancesIteration},
        {1, facts.schedule.operands[1].memoryMode, rhsVectorsPerWindow,
         std::max(1u, accumulatorVectors /
                          std::max(1u, rhsVectorsPerWindow)),
         facts.schedule.operands[1].advancesIteration}};
    std::optional<LocalMicrokernelSchedule> schedule =
        buildLocalMicrokernelSchedule(
            mapping, facts.reductionAxis, accumulatorVectors, operandWindows,
            {}, mapping.pipeline.bufferCount - 1);
    if (!schedule)
      continue;
    std::optional<PhysicalResourceRequirements> requirements =
        localMicrokernelRequirements(
            *schedule, mapping.laneShape, mapping.laneShape,
            facts.predicateGroups, facts.stateGroups, facts.handoffGroups,
            target);
    std::optional<PhysicalResourceBudget> resources =
        requirements ? calculatePhysicalResources(*requirements, target)
                     : std::nullopt;
    if (!requirements || !resources)
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
    const unsigned preferredBuffers =
        reduction->unrollFactor >= 2 &&
                llvm::is_contained(legalPipelineBuffers, 2u)
            ? 2
            : 1;
    const unsigned pipelinePenalty = static_cast<unsigned>(std::abs(
        static_cast<int>(mapping.pipeline.bufferCount) -
        static_cast<int>(preferredBuffers)));
    legal.push_back(Candidate{
        std::move(mapping), std::move(*schedule),
        std::move(*requirements), *resources,
        static_cast<unsigned>(
            facts.reductionExtent &&
            *facts.reductionExtent %
                    (reduction->unrollFactor *
                     (laneAxis->id == facts.reductionAxis
                          ? laneAxis->laneFactor
                          : 1)) !=
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
        pipelinePenalty,
        resources->peakGroups});
  }
  if (legal.empty())
    return std::nullopt;
  llvm::sort(legal, [](const Candidate &lhs, const Candidate &rhs) {
    return std::tie(lhs.tailPenalty, lhs.sequentialPenalty, lhs.memoryPenalty,
                    lhs.lanePenalty, lhs.unrollPenalty, lhs.pipelinePenalty,
                    lhs.peakGroups) <
           std::tie(rhs.tailPenalty, rhs.sequentialPenalty, rhs.memoryPenalty,
                    rhs.lanePenalty, rhs.unrollPenalty, rhs.pipelinePenalty,
                    rhs.peakGroups);
  });
  return SelectedF32DotPhysical{
      std::move(legal.front().mapping), std::move(legal.front().schedule),
      std::move(legal.front().requirements), legal.front().resources};
}

std::optional<SelectedF16MatmulPhysical>
selectF16MatmulPhysicalConfig(const F16MatmulCandidateFacts &facts,
                              const RISCVTargetProfile &target,
                              const RISCVBackendConfig &config) {
  if (!target.hasRVV || !target.hasF || !target.hasVectorF16 ||
      !target.hasWideningFloat || config.parameters.f16LoadBufferCount < 0 ||
      config.parameters.f16LoadBufferCount > 4)
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
          : localPipelineBufferCandidates(facts.schedule, 4);
  const llvm::SmallVector<unsigned, 4> legalPipelineBuffers =
      localPipelineBufferCandidates(facts.schedule, 4);
  LogicalAxisConstraint *mConstraint = nullptr;
  LogicalAxisConstraint *nConstraint = nullptr;
  for (LogicalAxisConstraint &axis : problem.axes) {
    if (axis.id == facts.lhsFreeAxis)
      mConstraint = &axis;
    if (axis.id == facts.rhsFreeAxis)
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
    if (axis.id == facts.reductionAxis)
      reductionExtent = axis.extent;
  if (!reductionExtent)
    return std::nullopt;
  const unsigned preferredUnroll =
      *reductionExtent < 32 ? 1 : baseInputLanes >= 16 ? 4 : 2;
  const unsigned preferredBuffers =
      *reductionExtent >= 64 && llvm::is_contained(legalPipelineBuffers, 2u)
          ? 2
          : 1;
  const unsigned preferredColumns =
      baseInputLanes <= 8 && columnTile >= 2 && columnTile % 2 == 0
          ? 2
          : 1;
  struct Candidate {
    CorePhysicalMapping mapping;
    LocalMicrokernelSchedule schedule;
    RVVVectorShape accumulatorShape;
    PhysicalResourceBudget resources;
    unsigned tailPenalty = 0;
    unsigned lanePenalty = 0;
    unsigned rowPenalty = 0;
    unsigned columnPenalty = 0;
    unsigned memoryPenalty = 0;
    unsigned lmulPenalty = 0;
    unsigned unrollPenalty = 0;
    unsigned bufferPenalty = 0;
  };
  llvm::SmallVector<Candidate, 64> legal;
  for (CorePhysicalMapping mapping :
       enumerateCorePhysicalMappings(problem, target)) {
    const PhysicalAxisDecomposition *m =
        findAxisMapping(mapping, facts.lhsFreeAxis);
    const PhysicalAxisDecomposition *n =
        findAxisMapping(mapping, facts.rhsFreeAxis);
    const PhysicalAxisDecomposition *k =
        findAxisMapping(mapping, facts.reductionAxis);
    const PhysicalAxisDecomposition *lane =
        mapping.laneAxis ? findAxisMapping(mapping, *mapping.laneAxis) : nullptr;
    std::optional<unsigned> inputLMUL = rvvIntegerLMUL(mapping.laneShape);
    std::optional<RVVVectorShape> accumulatorShape =
        rvvShapeForSameLanes(mapping.laneShape, 32, target);
    if (!m || !n || !k || !lane ||
        (lane->id != facts.rhsFreeAxis &&
         lane->id != facts.reductionAxis) ||
        !inputLMUL ||
        !accumulatorShape ||
        (k->unrollFactor != 1 && k->unrollFactor != 2 &&
         k->unrollFactor != 4) ||
        mapping.pipeline.bufferCount == 0 ||
        mapping.pipeline.bufferCount > k->unrollFactor ||
        (mapping.pipeline.bufferCount > 1 &&
         !llvm::is_contained(legalPipelineBuffers,
                             mapping.pipeline.bufferCount)) ||
        (lane->id == facts.rhsFreeAxis &&
         (n->registerFactor != 1 || mapping.pipeline.bufferCount != 1)))
      continue;
    const unsigned rows = m->registerFactor;
    const unsigned columns = n->registerFactor;
    const unsigned bufferCount = mapping.pipeline.bufferCount;
    const bool columnLane = lane->id == facts.rhsFreeAxis;
    const unsigned accumulatorVectors =
        columnLane ? rows : rows * columns;
    const unsigned lhsVectors = columnLane ? 0 : (bufferCount > 1 ? rows : 1);
    const unsigned rhsVectors = columnLane ? 1 : columns;
    if (facts.schedule.operands.size() != 2)
      continue;
    llvm::SmallVector<LocalOperandWindow, 2> operandWindows = {
        {0, facts.schedule.operands[0].memoryMode, lhsVectors,
         std::max(1u, accumulatorVectors /
                          std::max(1u, lhsVectors)),
         facts.schedule.operands[0].advancesIteration},
        {1, facts.schedule.operands[1].memoryMode, rhsVectors,
         std::max(1u, accumulatorVectors /
                          std::max(1u, rhsVectors)),
         facts.schedule.operands[1].advancesIteration}};
    std::optional<LocalMicrokernelSchedule> schedule =
        buildLocalMicrokernelSchedule(
            mapping, facts.reductionAxis, accumulatorVectors, operandWindows,
            {}, bufferCount - 1);
    if (!schedule)
      continue;
    std::optional<PhysicalResourceBudget> resources =
        calculateLocalMicrokernelResources(
            *schedule, mapping.laneShape, *accumulatorShape, 0, 0, 0,
            target);
    if (!resources)
      continue;
    const unsigned tailPenalty =
        columnLane
            ? static_cast<unsigned>(*reductionExtent % k->unrollFactor != 0) +
                  static_cast<unsigned>(columnTile % lane->laneFactor != 0)
            : static_cast<unsigned>(*reductionExtent %
                                        (k->unrollFactor * k->laneFactor) !=
                                    0);
    const unsigned realizedColumns =
        columnLane ? lane->laneFactor : columns;
    const unsigned preferredRealizedColumns =
        columnLane ? std::min(columnTile, lane->laneFactor) : preferredColumns;
    legal.push_back(Candidate{
        std::move(mapping), std::move(*schedule), *accumulatorShape,
        *resources, tailPenalty,
        static_cast<unsigned>(!columnLane),
        static_cast<unsigned>(std::abs(static_cast<int>(rows) -
                                       static_cast<int>(preferredRows))),
        static_cast<unsigned>(std::abs(static_cast<int>(realizedColumns) -
                                       static_cast<int>(preferredRealizedColumns))),
        static_cast<unsigned>(columnLane && facts.nLaneStrided),
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
    return std::tie(lhs.tailPenalty, lhs.memoryPenalty, lhs.lanePenalty,
                    lhs.rowPenalty, lhs.columnPenalty, lhs.lmulPenalty,
                    lhs.unrollPenalty, lhs.bufferPenalty,
                    lhs.resources.peakGroups) <
           std::tie(rhs.tailPenalty, rhs.memoryPenalty, rhs.lanePenalty,
                    rhs.rowPenalty, rhs.columnPenalty, rhs.lmulPenalty,
                    rhs.unrollPenalty, rhs.bufferPenalty,
                    rhs.resources.peakGroups);
  });
  return SelectedF16MatmulPhysical{
      std::move(legal.front().mapping), std::move(legal.front().schedule),
      legal.front().accumulatorShape, legal.front().resources};
}

} // namespace weft::riscv_internal
