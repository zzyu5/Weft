add_mlir_dialect_library(WeftRISCVDialect
  RISCVPlanningDialect.cpp

  DEPENDS
  WeftRISCVIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
