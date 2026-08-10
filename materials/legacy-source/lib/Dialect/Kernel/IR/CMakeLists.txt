add_mlir_dialect_library(WeftKernelDialect
  KernelDialect.cpp

  DEPENDS
  MLIRKernelOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
