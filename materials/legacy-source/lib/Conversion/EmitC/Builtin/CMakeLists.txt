add_mlir_library(WeftConversionEmitCBuiltin
  BuiltinBackendEmitters.cpp

  LINK_LIBS PUBLIC
  WeftConversionEmitC
  WeftConversionRVV
  WeftIMEBackendEmitter
  WeftScalarBackendEmitter
  MLIRIR
  MLIRSupport
)
