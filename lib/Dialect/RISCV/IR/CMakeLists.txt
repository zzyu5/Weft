add_mlir_dialect_library(WeftRISCVDialect
  RISCVDialect.cpp

  DEPENDS
  WeftRISCVIncGen

  LINK_LIBS PUBLIC
  WeftKernelDialect
  MLIRIR
)
