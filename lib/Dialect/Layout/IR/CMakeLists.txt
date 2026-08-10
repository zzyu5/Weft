add_mlir_dialect_library(WeftLayoutDialect
  LayoutDialect.cpp

  DEPENDS
  MLIRLayoutAttrsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
