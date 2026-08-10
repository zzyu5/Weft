set(LLVM_TARGET_DEFINITIONS IMEExecutionOps.td)
mlir_tablegen(IMEExecutionOpsDialect.h.inc -gen-dialect-decls -dialect=weft_ime_execution)
mlir_tablegen(IMEExecutionOpsDialect.cpp.inc -gen-dialect-defs -dialect=weft_ime_execution)
mlir_tablegen(IMEExecutionOps.h.inc -gen-op-decls)
mlir_tablegen(IMEExecutionOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRIMEExecutionOpsIncGen)
add_dependencies(MLIRIMEExecutionOpsIncGen
  MLIRSelectedOwnerOpInterfaceIncGen)
