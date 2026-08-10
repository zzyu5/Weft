add_mlir_library(WeftRVVTarget
  RVVTargetProfileBinding.cpp
  RVVTargetSupportBundle.cpp
  SelectedExecutionRVVSource.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftExecDialect
  WeftExecutionDialect
  WeftKernelDialect
  WeftLayoutDialect
  WeftPlugin
  WeftRVVExecutionDialect
  WeftRVVConstruction
  WeftRVVDialect
  WeftSupport
  WeftTransforms
  WeftTarget
  MLIRIR
  MLIRPass
  MLIRSupport
)
