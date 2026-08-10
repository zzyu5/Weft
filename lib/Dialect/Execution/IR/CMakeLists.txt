add_mlir_dialect_library(WeftExecutionDialect
  ExecutionDialect.cpp
  SelectedOwnerOpInterface.cpp

  DEPENDS
  MLIRExecutionOpsIncGen
  MLIRSelectedOwnerOpInterfaceIncGen

  LINK_LIBS PUBLIC
  WeftKernelDialect
  WeftLayoutDialect
  MLIRIR
)
