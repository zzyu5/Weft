#ifndef WEFT_TARGET_RISCVCOMPILER_H
#define WEFT_TARGET_RISCVCOMPILER_H

#include "Weft/Target/RISCVTargetProfile.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <string>

namespace weft {

enum class F16MatmulLaneAxis {
  Auto,
  Column,
  Reduction,
};

struct RISCVCandidateParameters {
  int64_t vlaLMUL = 0;
  int64_t dotLMUL = 0;
  int64_t dotKUnroll = 0;
  int64_t dotLoadBufferCount = 0;
  int64_t f16InputLMUL = 0;
  int64_t f16RowMicrotile = 0;
  int64_t f16ColumnMicrotile = 0;
  int64_t f16KUnroll = 0;
  int64_t f16LoadBufferCount = 0;
  F16MatmulLaneAxis f16LaneAxis = F16MatmulLaneAxis::Auto;
  int64_t narrowLMUL = 0;
  int64_t sortRadixBits = 0;
};

struct RISCVBackendConfig {
  RISCVCandidateParameters parameters;
};

struct RISCVCompilerOptions {
  RISCVTargetProfile target;
  llvm::StringMap<int64_t> metaBindings;
  RISCVBackendConfig backend;
};

struct RISCVArtifact {
  std::string intrinsicC;
  std::string header;
};

/// Run the unique MLIR RISC-V compilation pass and return both public
/// artifacts. Physical facts, candidates, and selected decisions remain
/// transient state owned by this invocation.
mlir::FailureOr<RISCVArtifact>
compileRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options);

} // namespace weft

#endif // WEFT_TARGET_RISCVCOMPILER_H
