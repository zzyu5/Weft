add_mlir_library(WeftPlugin
  ExtensionBundle.cpp
  ExtensionPlugin.cpp
  LoweringBoundary.cpp

  LINK_LIBS PUBLIC
  WeftSupport
  MLIRIR
  MLIRSupport
)

add_subdirectory(Construction)
add_subdirectory(RVV)
add_subdirectory(IME)
add_subdirectory(Scalar)
add_subdirectory(Builtin)
