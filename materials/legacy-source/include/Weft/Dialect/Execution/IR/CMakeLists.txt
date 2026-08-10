add_mlir_interface(SelectedOwnerOpInterface)

set(LLVM_TARGET_DEFINITIONS ExecutionOps.td)
mlir_tablegen(ExecutionOpsDialect.h.inc -gen-dialect-decls -dialect=weft_execution)
mlir_tablegen(ExecutionOpsDialect.cpp.inc -gen-dialect-defs -dialect=weft_execution)
mlir_tablegen(ExecutionOps.h.inc -gen-op-decls)
mlir_tablegen(ExecutionOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRExecutionOpsIncGen)
