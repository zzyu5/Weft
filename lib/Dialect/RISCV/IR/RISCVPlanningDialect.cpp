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
  if (getKernel().empty())
    return emitOpError("physicalization candidate requires a kernel");
  if (getStage() != "facts" && getStage() != "representations" &&
      getStage() != "conversions" && getStage() != "storage-mappings" &&
      getStage() != "operations" &&
      getStage() != "schedule" &&
      getStage() != "resources" && getStage() != "invalid")
    return emitOpError("unknown physicalization candidate stage");
  if (failed(requireDictionaryKeys(
          *this, getCandidate(), {"id", "specialization", "auto_bindings"},
          "candidate")))
    return mlir::failure();
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
  return requireDictionaryKeys(*this, getResources(),
                               {"vector_register_budget", "peak_vector_groups",
                                "spill", "stack_bytes", "cost"},
                               "resources");
}

void WEFTRISCVDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Weft/Dialect/RISCV/IR/RISCVOps.cpp.inc"
      >();
}
