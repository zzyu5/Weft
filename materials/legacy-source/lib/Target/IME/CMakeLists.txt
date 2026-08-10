add_mlir_library(WeftIMETarget
  SelectedExecutionIMESource.cpp

  LINK_LIBS PUBLIC
  WeftExecutionDialect
  WeftIMEExecutionDialect
  WeftKernelDialect
  WeftLayoutDialect
  MLIRIR
  MLIRSupport
)

