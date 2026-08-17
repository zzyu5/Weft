#ifndef WEFT_LIB_TARGET_RISCVAXISMAPPING_H
#define WEFT_LIB_TARGET_RISCVAXISMAPPING_H

#include "Weft/Target/RISCVTargetProfile.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace weft::riscv_internal {

inline constexpr unsigned kCoreAxisM = 0;
inline constexpr unsigned kCoreAxisN = 1;
inline constexpr unsigned kCoreAxisK = 2;
inline constexpr unsigned kCoreAxisGroup = 3;
inline constexpr unsigned kCoreAxisPacked = 4;
inline constexpr unsigned kCoreAxisVLA = 5;
inline constexpr unsigned kCoreAxisBlock = 6;

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
llvm::SmallVector<RVVVectorShape, 8>
rvvShapeCandidates(const RISCVTargetProfile &target, unsigned sew);
llvm::SmallVector<unsigned, 4>
registerFactorCandidates(uint64_t extent, unsigned maximum);

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

enum class LogicalAxisRole {
  Free,
  Reduction,
  Broadcast,
  Packed,
  Group,
  State,
};

enum class CoreInstructionKind {
  None,
  Scalar,
  RVVElementwise,
  RVVFMA,
  RVVWideningFMA,
  RVVWideningIntegerDot,
  RVVIndexedGather,
  RVVReduction,
  RVVSegmentMemory,
  SpacemitIME1MMA,
};

struct LogicalAxisConstraint {
  unsigned id = 0;
  LogicalAxisRole role = LogicalAxisRole::Free;
  std::optional<uint64_t> extent;
  bool ordered = false;
  bool allowLane = false;
  bool requireLane = false;
  llvm::SmallVector<unsigned, 4> registerFactors{1};
  unsigned laneFactorLimit = 0;
};

struct FragmentAxisFactor {
  unsigned axis = 0;
  unsigned factor = 1;
};

struct FragmentMappingConstraint {
  CoreInstructionKind instruction = CoreInstructionKind::None;
  llvm::SmallVector<FragmentAxisFactor, 4> factors;
  unsigned fixedResourceGroups = 0;
};

struct CoreMappingProblem {
  llvm::SmallVector<LogicalAxisConstraint, 4> axes;
  unsigned laneSEW = 0;
  CoreInstructionKind laneInstruction = CoreInstructionKind::None;
  CoreInstructionKind sequentialInstruction = CoreInstructionKind::None;
  llvm::SmallVector<RVVVectorShape, 4> laneShapeCandidates;
  llvm::SmallVector<unsigned, 4> laneLMULCandidates;
  std::optional<unsigned> unrollAxis;
  llvm::SmallVector<unsigned, 4> unrollCandidates{1};
  llvm::SmallVector<unsigned, 4> pipelineBufferCandidates{1};
  llvm::SmallVector<FragmentMappingConstraint, 2> fragments;
  bool allowSequentialOnly = false;
};

struct PhysicalAxisDecomposition {
  unsigned id = 0;
  LogicalAxisRole role = LogicalAxisRole::Free;
  std::optional<uint64_t> extent;
  bool ordered = false;
  unsigned sequentialFactor = 0;
  unsigned laneFactor = 1;
  unsigned registerFactor = 1;
  unsigned unrollFactor = 1;
  unsigned fragmentFactor = 1;

  bool operator==(const PhysicalAxisDecomposition &other) const;
  bool operator<(const PhysicalAxisDecomposition &other) const;
};

struct LocalPipelineMapping {
  unsigned bufferCount = 1;

  bool operator==(const LocalPipelineMapping &other) const;
  bool operator<(const LocalPipelineMapping &other) const;
};

struct CorePhysicalMapping {
  CoreInstructionKind instruction = CoreInstructionKind::None;
  llvm::SmallVector<PhysicalAxisDecomposition, 4> axes;
  RVVVectorShape laneShape;
  LocalPipelineMapping pipeline;
  unsigned fixedFragmentGroups = 0;

  explicit operator bool() const {
    return instruction != CoreInstructionKind::None && !axes.empty();
  }
  bool operator==(const CorePhysicalMapping &other) const;
  bool operator<(const CorePhysicalMapping &other) const;
};

llvm::SmallVector<CorePhysicalMapping>
enumerateCorePhysicalMappings(const CoreMappingProblem &problem,
                              const RISCVTargetProfile &target);

const PhysicalAxisDecomposition *
findAxisMapping(const CorePhysicalMapping &mapping, unsigned axis);
unsigned mappedLaneSpan(const CorePhysicalMapping &mapping);
unsigned mappedHardwareLaneFactor(const CorePhysicalMapping &mapping);
unsigned mappedRegisterFactor(const CorePhysicalMapping &mapping,
                              unsigned axis);
unsigned mappedFragmentFactor(const CorePhysicalMapping &mapping,
                              unsigned axis);
bool mappedAxisHasSequentialIteration(const CorePhysicalMapping &mapping,
                                      unsigned axis);

enum class PhysicalLiveClass {
  Value,
  Memory,
  Index,
  Predicate,
  State,
  Temporary,
  Fragment,
  Handoff,
};

struct PhysicalLiveRange {
  PhysicalLiveClass kind = PhysicalLiveClass::Temporary;
  RVVVectorShape shape;
  unsigned count = 0;
  unsigned fixedGroups = 0;
};

struct PhysicalResourceRequirements {
  llvm::SmallVector<PhysicalLiveRange, 16> live;
  unsigned reservedGroups = 1;
};

std::optional<PhysicalResourceBudget>
calculatePhysicalResources(const PhysicalResourceRequirements &requirements,
                           const RISCVTargetProfile &target);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVAXISMAPPING_H
