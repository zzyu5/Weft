add_mlir_dialect_library(WeftRVVExecutionDialect
  RVVExecutionDialect.cpp

  DEPENDS
  MLIRRVVExecutionOpsIncGen

  LINK_LIBS PUBLIC
  WeftExecutionDialect
  WeftKernelDialect
  WeftLayoutDialect
  MLIRIR
)
