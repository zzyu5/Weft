add_mlir_dialect_library(WeftExecutionDialect
  ExecutionDialect.cpp

  DEPENDS
  WeftExecutionIncGen

  LINK_LIBS PUBLIC
  WeftExtensionDialect
  WeftKernelDialect
  MLIRIR
)
