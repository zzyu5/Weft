add_mlir_dialect_library(WeftIMEExecutionDialect
  IMEExecutionDialect.cpp

  DEPENDS
  MLIRIMEExecutionOpsIncGen

  LINK_LIBS PUBLIC
  WeftExecutionDialect
  WeftKernelDialect
  WeftLayoutDialect
  MLIRIR
)

