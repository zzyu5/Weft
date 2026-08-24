add_mlir_dialect_library(WeftKernelDialect
  KernelDialect.cpp

  DEPENDS
  WeftKernelIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRSideEffectInterfaces
)
