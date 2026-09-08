add_mlir_dialect_library(WeftRISCVDialect
  RISCVDialect.cpp
  FragmentOps.cpp
  Verification.cpp

  DEPENDS
  WeftRISCVIncGen

  LINK_LIBS PUBLIC
  WeftRISCVContracts
  WeftKernelDialect
  MLIRIR
)
