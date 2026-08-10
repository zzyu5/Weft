add_library(WeftCompiler STATIC Selection.cpp)

target_link_libraries(WeftCompiler PUBLIC
  WeftExecutionDialect
  WeftExtensionDialect
  WeftKernelDialect
  WeftRISCVTarget
  MLIRIR
)
