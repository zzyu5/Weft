add_mlir_dialect_library(WeftExtensionDialect
  ExtensionDialect.cpp

  DEPENDS
  WeftExtensionIncGen
  WeftKernelIncGen

  LINK_LIBS PUBLIC
  WeftKernelDialect
  MLIRIR
)
