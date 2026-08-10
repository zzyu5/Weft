set(LLVM_TARGET_DEFINITIONS RVVExecutionOps.td)
mlir_tablegen(RVVExecutionOpsDialect.h.inc -gen-dialect-decls -dialect=weft_rvv_execution)
mlir_tablegen(RVVExecutionOpsDialect.cpp.inc -gen-dialect-defs -dialect=weft_rvv_execution)
mlir_tablegen(RVVExecutionOps.h.inc -gen-op-decls)
mlir_tablegen(RVVExecutionOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRRVVExecutionOpsIncGen)
add_dependencies(MLIRRVVExecutionOpsIncGen
  MLIRSelectedOwnerOpInterfaceIncGen)
