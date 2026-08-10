add_subdirectory(Dialect)
add_subdirectory(Support)
add_subdirectory(Conversion)
add_subdirectory(Plugin)
add_subdirectory(Target)
add_subdirectory(Transforms)

add_mlir_library(WeftInitAll
  InitWeftDialects.cpp

  LINK_LIBS PUBLIC
  WeftExecDialect
  WeftExecutionDialect
  WeftIMEExecutionDialect
  WeftKernelDialect
  WeftLayoutDialect
  WeftPlugin
  WeftRVVExecutionDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRVectorDialect
)
