#include "RISCVAxisMapping.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <functional>
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

llvm::SmallVector<RVVVectorShape, 8>
rvvShapeCandidates(const RISCVTargetProfile &target, unsigned sew) {
  llvm::SmallVector<RVVVectorShape, 8> candidates;
  for (int eighths : target.legalLMULEighths)
    if (target.supportsVectorShape(sew, eighths))
      candidates.push_back(RVVVectorShape{sew, eighths});
  llvm::sort(candidates);
  return candidates;
}

llvm::SmallVector<unsigned, 4>
registerFactorCandidates(uint64_t extent, unsigned maximum) {
  llvm::SmallVector<unsigned, 4> candidates;
  for (unsigned factor = 1;
       factor <= maximum && static_cast<uint64_t>(factor) <= extent; ++factor)
    if (extent % factor == 0)
      candidates.push_back(factor);
  return candidates;
}

bool PhysicalAxisDecomposition::operator==(
    const PhysicalAxisDecomposition &other) const {
  return std::tie(id, role, extent, ordered, sequentialFactor, laneFactor,
                  registerFactor, unrollFactor, fragmentFactor) ==
         std::tie(other.id, other.role, other.extent, other.ordered,
                  other.sequentialFactor, other.laneFactor,
                  other.registerFactor, other.unrollFactor,
                  other.fragmentFactor);
}

bool PhysicalAxisDecomposition::operator<(
    const PhysicalAxisDecomposition &other) const {
  return std::tie(id, role, extent, ordered, sequentialFactor, laneFactor,
                  registerFactor, unrollFactor, fragmentFactor) <
         std::tie(other.id, other.role, other.extent, other.ordered,
                  other.sequentialFactor, other.laneFactor,
                  other.registerFactor, other.unrollFactor,
                  other.fragmentFactor);
}

bool LocalPipelineMapping::operator==(
    const LocalPipelineMapping &other) const {
  return bufferCount == other.bufferCount;
}

bool LocalPipelineMapping::operator<(
    const LocalPipelineMapping &other) const {
  return bufferCount < other.bufferCount;
}

bool CorePhysicalMapping::operator==(const CorePhysicalMapping &other) const {
  return instruction == other.instruction && laneAxis == other.laneAxis &&
         axes == other.axes &&
         laneShape == other.laneShape && pipeline == other.pipeline &&
         fixedFragmentGroups == other.fixedFragmentGroups;
}

bool CorePhysicalMapping::operator<(const CorePhysicalMapping &other) const {
  if (instruction != other.instruction)
    return instruction < other.instruction;
  if (laneAxis != other.laneAxis)
    return laneAxis < other.laneAxis;
  if (axes != other.axes)
    return std::lexicographical_compare(axes.begin(), axes.end(),
                                        other.axes.begin(), other.axes.end());
  return std::tie(laneShape, pipeline, fixedFragmentGroups) <
         std::tie(other.laneShape, other.pipeline,
                  other.fixedFragmentGroups);
}

static unsigned ceilDiv(uint64_t value, uint64_t divisor) {
  return static_cast<unsigned>((value + divisor - 1) / divisor);
}

static void finalizeSequentialFactors(CorePhysicalMapping &mapping) {
  for (PhysicalAxisDecomposition &axis : mapping.axes) {
    if (!axis.extent) {
      axis.sequentialFactor = 0;
      continue;
    }
    uint64_t realized = static_cast<uint64_t>(axis.laneFactor) *
                        axis.registerFactor * axis.unrollFactor *
                        axis.fragmentFactor;
    axis.sequentialFactor = ceilDiv(*axis.extent, std::max<uint64_t>(1, realized));
  }
}

llvm::SmallVector<CorePhysicalMapping>
enumerateCorePhysicalMappings(const CoreMappingProblem &problem,
                              const RISCVTargetProfile &target) {
  llvm::SmallVector<CorePhysicalMapping> mappings;
  llvm::SmallVector<unsigned> requiredLaneAxes;
  llvm::SmallVector<unsigned> possibleLaneAxes;
  for (const LogicalAxisConstraint &axis : problem.axes) {
    if (axis.requireLane)
      requiredLaneAxes.push_back(axis.id);
    if (axis.allowLane)
      possibleLaneAxes.push_back(axis.id);
  }
  if (requiredLaneAxes.size() > 1)
    return mappings;
  if (!requiredLaneAxes.empty())
    possibleLaneAxes = requiredLaneAxes;

  auto registerCombinations = [&](auto &&emit) {
    llvm::SmallVector<unsigned, 4> factors(problem.axes.size(), 1);
    std::function<void(size_t)> visit = [&](size_t index) {
      if (index == problem.axes.size()) {
        emit(factors);
        return;
      }
      const LogicalAxisConstraint &axis = problem.axes[index];
      llvm::ArrayRef<unsigned> choices = axis.registerFactors;
      if (choices.empty()) {
        factors[index] = 1;
        visit(index + 1);
        return;
      }
      for (unsigned factor : choices) {
        if (factor == 0 || (axis.extent && factor > *axis.extent))
          continue;
        factors[index] = factor;
        visit(index + 1);
      }
    };
    visit(0);
  };

  llvm::SmallVector<RVVVectorShape> laneShapes(
      problem.laneShapeCandidates.begin(), problem.laneShapeCandidates.end());
  if (laneShapes.empty()) {
    llvm::SmallVector<unsigned> lmulCandidates(
        problem.laneLMULCandidates.begin(), problem.laneLMULCandidates.end());
    if (lmulCandidates.empty() && problem.laneSEW != 0)
      lmulCandidates = integerLMULCandidates(target, problem.laneSEW);
    for (unsigned lmul : lmulCandidates)
      laneShapes.push_back(rvvShape(problem.laneSEW, lmul));
  }
  llvm::SmallVector<unsigned, 4> unrollCandidates = problem.unrollCandidates;
  if (unrollCandidates.empty())
    unrollCandidates = {1};
  llvm::SmallVector<unsigned, 4> bufferCandidates =
      problem.pipelineBufferCandidates;
  if (bufferCandidates.empty())
    bufferCandidates = {1};

  for (unsigned laneAxis : possibleLaneAxes) {
    for (RVVVectorShape laneShape : laneShapes) {
      std::optional<unsigned> laneCapacity = rvvLaneCapacity(laneShape, target);
      if (!laneCapacity)
        continue;
      auto laneConstraint = llvm::find_if(
          problem.axes, [&](const LogicalAxisConstraint &axis) {
            return axis.id == laneAxis;
          });
      if (laneConstraint == problem.axes.end())
        continue;
      uint64_t allowedLanes = *laneCapacity;
      if (laneConstraint->extent)
        allowedLanes = std::min<uint64_t>(allowedLanes,
                                          *laneConstraint->extent);
      if (laneConstraint->laneFactorLimit != 0)
        allowedLanes =
            std::min<uint64_t>(allowedLanes,
                               laneConstraint->laneFactorLimit);
      const unsigned laneFactor = static_cast<unsigned>(allowedLanes);
      registerCombinations([&](llvm::ArrayRef<unsigned> registers) {
        for (unsigned unroll : unrollCandidates) {
          if (unroll == 0)
            continue;
          for (unsigned buffers : bufferCandidates) {
            if (buffers == 0)
              continue;
            CorePhysicalMapping mapping;
            mapping.instruction = problem.laneInstruction;
            mapping.laneAxis = laneAxis;
            mapping.laneShape = laneShape;
            mapping.pipeline.bufferCount = buffers;
            for (size_t index = 0; index < problem.axes.size(); ++index) {
              const LogicalAxisConstraint &axis = problem.axes[index];
              PhysicalAxisDecomposition decomposition;
              decomposition.id = axis.id;
              decomposition.role = axis.role;
              decomposition.extent = axis.extent;
              decomposition.ordered = axis.ordered;
              decomposition.laneFactor = axis.id == laneAxis ? laneFactor : 1;
              decomposition.registerFactor = registers[index];
              decomposition.unrollFactor =
                  problem.unrollAxis && *problem.unrollAxis == axis.id
                      ? unroll
                      : 1;
              mapping.axes.push_back(decomposition);
            }
            finalizeSequentialFactors(mapping);
            mappings.push_back(std::move(mapping));
          }
        }
      });
    }
  }

  if (problem.allowSequentialOnly && requiredLaneAxes.empty()) {
    registerCombinations([&](llvm::ArrayRef<unsigned> registers) {
      for (unsigned unroll : unrollCandidates) {
        for (unsigned buffers : bufferCandidates) {
          CorePhysicalMapping mapping;
          mapping.instruction =
              problem.sequentialInstruction == CoreInstructionKind::None
                  ? problem.laneInstruction
                  : problem.sequentialInstruction;
          mapping.pipeline.bufferCount = buffers;
          for (size_t index = 0; index < problem.axes.size(); ++index) {
            const LogicalAxisConstraint &axis = problem.axes[index];
            mapping.axes.push_back(PhysicalAxisDecomposition{
                axis.id, axis.role, axis.extent, axis.ordered, 0, 1,
                registers[index],
                problem.unrollAxis && *problem.unrollAxis == axis.id ? unroll
                                                                    : 1,
                1});
          }
          finalizeSequentialFactors(mapping);
          mappings.push_back(std::move(mapping));
        }
      }
    });
  }

  for (const FragmentMappingConstraint &fragment : problem.fragments) {
    CorePhysicalMapping mapping;
    mapping.instruction = fragment.instruction;
    mapping.fixedFragmentGroups = fragment.fixedResourceGroups;
    for (const LogicalAxisConstraint &axis : problem.axes) {
      unsigned fragmentFactor = 1;
      auto found = llvm::find_if(fragment.factors,
                                 [&](const FragmentAxisFactor &factor) {
                                   return factor.axis == axis.id;
                                 });
      if (found != fragment.factors.end())
        fragmentFactor = found->factor;
      mapping.axes.push_back(PhysicalAxisDecomposition{
          axis.id, axis.role, axis.extent, axis.ordered, 0, 1, 1, 1,
          fragmentFactor});
    }
    finalizeSequentialFactors(mapping);
    mappings.push_back(std::move(mapping));
  }
  return mappings;
}

const PhysicalAxisDecomposition *
findAxisMapping(const CorePhysicalMapping &mapping, unsigned axis) {
  auto found = llvm::find_if(mapping.axes,
                             [&](const PhysicalAxisDecomposition &candidate) {
                               return candidate.id == axis;
                             });
  return found == mapping.axes.end() ? nullptr : &*found;
}

unsigned mappedLaneSpan(const CorePhysicalMapping &mapping) {
  const PhysicalAxisDecomposition *axis =
      findAxisMapping(mapping, mapping.laneAxis);
  return axis ? axis->laneFactor * axis->registerFactor : 0;
}

unsigned mappedHardwareLaneFactor(const CorePhysicalMapping &mapping) {
  const PhysicalAxisDecomposition *axis =
      findAxisMapping(mapping, mapping.laneAxis);
  return axis ? axis->laneFactor : 0;
}

unsigned mappedRegisterFactor(const CorePhysicalMapping &mapping,
                              unsigned axis) {
  const PhysicalAxisDecomposition *decomposition =
      findAxisMapping(mapping, axis);
  return decomposition ? decomposition->registerFactor : 1;
}

unsigned mappedFragmentFactor(const CorePhysicalMapping &mapping,
                              unsigned axis) {
  const PhysicalAxisDecomposition *decomposition =
      findAxisMapping(mapping, axis);
  return decomposition ? decomposition->fragmentFactor : 1;
}

bool mappedAxisHasSequentialIteration(const CorePhysicalMapping &mapping,
                                      unsigned axis) {
  const PhysicalAxisDecomposition *decomposition =
      findAxisMapping(mapping, axis);
  return decomposition && decomposition->sequentialFactor > 1;
}

std::optional<PhysicalResourceBudget>
calculatePhysicalResources(const PhysicalResourceRequirements &requirements,
                           const RISCVTargetProfile &target) {
  if (!target.supportsFixedRVV())
    return std::nullopt;
  PhysicalResourceBudget budget;
  budget.architecturalGroups = target.vectorRegisters;
  for (const PhysicalLiveRange &range : requirements.live) {
    if (range.count == 0)
      continue;
    unsigned groups = range.fixedGroups;
    if (range.shape) {
      if (!target.supportsVectorShape(range.shape.sew,
                                      range.shape.lmulEighths))
        return std::nullopt;
      groups += range.count * rvvRegisterGroups(range.shape);
    } else if (range.fixedGroups == 0) {
      return std::nullopt;
    }
    switch (range.kind) {
    case PhysicalLiveClass::Value:
      budget.valueGroups += groups;
      break;
    case PhysicalLiveClass::Memory:
      budget.memoryGroups += groups;
      break;
    case PhysicalLiveClass::Index:
      budget.indexGroups += groups;
      break;
    case PhysicalLiveClass::Predicate:
      budget.predicateGroups += groups;
      break;
    case PhysicalLiveClass::State:
      budget.stateGroups += groups;
      break;
    case PhysicalLiveClass::Temporary:
    case PhysicalLiveClass::Fragment:
    case PhysicalLiveClass::Handoff:
      budget.primitiveGroups += groups;
      break;
    }
  }
  budget.peakGroups = budget.valueGroups + budget.memoryGroups +
                      budget.indexGroups + budget.predicateGroups +
                      budget.stateGroups + budget.primitiveGroups +
                      requirements.reservedGroups;
  if (budget.peakGroups > static_cast<unsigned>(target.vectorRegisters))
    return std::nullopt;
  return budget;
}

} // namespace weft::riscv_internal
