add_mlir_library(WeftBuiltinPlugins
  BuiltinExtensionPlugins.cpp

  LINK_LIBS PUBLIC
  WeftPlugin
  WeftIMEPlugin
  WeftRVVPlugin
  WeftScalarPlugin
  MLIRSupport
)
