#ifndef WEFT_LIB_TARGET_RISCVPHYSICALPLANNING_H
#define WEFT_LIB_TARGET_RISCVPHYSICALPLANNING_H

#include "Weft/Target/RISCVLowering.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace weft::riscv_internal {

struct RVVVectorShape {
  unsigned sew = 0;
  int lmulEighths = 0;

  bool operator==(const RVVVectorShape &other) const {
    return sew == other.sew && lmulEighths == other.lmulEighths;
  }
  bool operator!=(const RVVVectorShape &other) const {
    return !(*this == other);
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
RVVVectorShape rvvShape(unsigned sew, unsigned lmul);
llvm::SmallVector<unsigned>
integerLMULCandidates(const RISCVTargetProfile &target, unsigned sew,
                      unsigned maximum = 8);

struct PhysicalResourceBudget {
  unsigned architecturalGroups = 0;
  unsigned valueGroups = 0;
  unsigned memoryGroups = 0;
  unsigned predicateGroups = 0;
  unsigned stateGroups = 0;
  unsigned primitiveGroups = 0;
  unsigned pipelineGroups = 0;
  unsigned peakGroups = 0;
};

enum class I4I8FragmentRealization {
  RVVN16K32,
  SpacemiTIME1N16K32,
};

std::optional<I4I8FragmentRealization>
selectI4I8FragmentRealization(const RISCVTargetProfile &target);

enum class VLAStatePlacement {
  ScalarCarry,
  VectorCarry,
};

struct ReductionStatePlacementFacts {
  bool relaxedOrder = false;
  unsigned reductionStateCount = 0;
};

std::optional<VLAStatePlacement>
selectReductionStatePlacement(const ReductionStatePlacementFacts &facts,
                              const RISCVTargetProfile &target,
                              const RISCVBackendConfig &config);

enum class QuantI8DotSemantic {
  IQ2S,
  IQ3S,
  IQ1M,
  Q6K,
};

enum class QuantI8DotRealization {
  RVVFixedLaneLocalBlockDot,
  RVVScalableLocalBlockDot,
};

struct QuantI8DotCandidateFacts {
  QuantI8DotSemantic semantic = QuantI8DotSemantic::IQ2S;
};

struct SelectedQuantI8DotPhysical {
  QuantI8DotRealization realization =
      QuantI8DotRealization::RVVScalableLocalBlockDot;
  unsigned semanticLanes = 0;
  RVVVectorShape byteShape;
  unsigned reductionSegments = 0;
  PhysicalResourceBudget resources;
};

std::optional<SelectedQuantI8DotPhysical>
selectQuantI8DotPhysical(const QuantI8DotCandidateFacts &facts,
                         const RISCVTargetProfile &target);

enum class E2M1E8M0I8Realization {
  RVVVLEN128TableDot,
  RVVVLEN256TableDot,
  RVVScalableLocalBlockDot,
};

struct SelectedE2M1E8M0I8Physical {
  E2M1E8M0I8Realization realization =
      E2M1E8M0I8Realization::RVVScalableLocalBlockDot;
  RVVVectorShape packedShape;
  RVVVectorShape activationShape;
  PhysicalResourceBudget resources;
};

std::optional<SelectedE2M1E8M0I8Physical>
selectE2M1E8M0I8Physical(const RISCVTargetProfile &target);

enum class F32DotResourceModel {
  VLAFreeAxis,
  LocalRow,
};

struct F32DotPhysicalConfig {
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
};

std::optional<F32DotPhysicalConfig>
selectF32DotPhysicalConfig(const F32DotCandidateFacts &facts,
                           const RISCVTargetProfile &target,
                           const RISCVBackendConfig &config);

enum class F16MatmulLoadSchedule {
  StreamRHSThenRows,
  BatchLoadsThenCompute,
};

struct F16MatmulPhysicalConfig {
  unsigned rowMicrotile = 1;
  unsigned inputLMUL = 1;
  unsigned kUnroll = 1;
  unsigned pipelineStages = 1;
  F16MatmulLoadSchedule loadSchedule =
      F16MatmulLoadSchedule::StreamRHSThenRows;
};

struct F16MatmulCandidateFacts {
  unsigned rowTile = 1;
  unsigned reductionTile = 1;
};

struct SelectedF16MatmulPhysical {
  F16MatmulPhysicalConfig config;
  PhysicalResourceBudget resources;
};

std::optional<SelectedF16MatmulPhysical>
selectF16MatmulPhysicalConfig(const F16MatmulCandidateFacts &facts,
                              const RISCVTargetProfile &target,
                              const RISCVBackendConfig &config);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVPHYSICALPLANNING_H
