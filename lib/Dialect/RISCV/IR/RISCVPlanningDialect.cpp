#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"

using namespace weft::riscv;

#include "Weft/Dialect/RISCV/IR/RISCVOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Weft/Dialect/RISCV/IR/RISCVOps.cpp.inc"

namespace {

mlir::LogicalResult requireDictionaryKeys(mlir::Operation *operation,
                                          mlir::DictionaryAttr dictionary,
                                          llvm::ArrayRef<llvm::StringRef> keys,
                                          llvm::StringRef name) {
  for (llvm::StringRef key : keys)
    if (!dictionary.get(key))
      return operation->emitOpError()
             << name << " is missing required key '" << key << "'";
  return mlir::success();
}

} // namespace

mlir::LogicalResult ProblemOp::verify() {
  if (getKernel().empty() || getCandidates().empty())
    return emitOpError("planning problem requires a kernel and candidates");
  if (getStage() != "facts" && getStage() != "representations" &&
      getStage() != "instructions" && getStage() != "resources")
    return emitOpError("unknown planning problem stage");
  return requireDictionaryKeys(*this, getTarget(),
                               {"march", "abi", "vlen_bits",
                                "vector_registers", "supported_sew",
                                "legal_lmul_eighths", "matrix_fragment_count",
                                "matrix_fragments",
                                "max_private_stack_bytes"},
                               "target");
}

mlir::LogicalResult AssignmentOp::verify() {
  if (getKernel().empty() || getValues().empty() || getOperations().empty())
    return emitOpError(
        "complete physical assignment requires a kernel, values and operations");
  if (getStatus() != "complete")
    return emitOpError("only complete physical assignments may be materialized");
  if (failed(requireDictionaryKeys(
          *this, getCandidate(), {"id", "specialization", "auto_bindings"},
          "candidate")))
    return mlir::failure();
  if (failed(requireDictionaryKeys(
          *this, getCDecisions(),
          {"sew", "lmul", "vl", "tail", "accumulator_grouping",
           "register_budget", "partial_layout", "horizontal_reduce",
           "vlen_specialization"},
          "C decisions")))
    return mlir::failure();
  if (failed(requireDictionaryKeys(
          *this, getDDecisions(),
          {"nibble_unpack", "mac_instruction", "scale_broadcast",
           "byte_interleave", "load_stride_alignment",
           "prefetch_distance", "pipeline_unroll"},
          "D decisions")))
    return mlir::failure();
  return requireDictionaryKeys(*this, getResources(),
                               {"vector_register_budget", "peak_vector_groups",
                                "spill", "stack_bytes"},
                               "resources");
}

void WEFTRISCVDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/RISCV/IR/RISCVOps.cpp.inc"
      >();
}
