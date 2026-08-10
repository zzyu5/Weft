add_mlir_tool(weft-compile weft-compile.cpp)

target_link_libraries(weft-compile PRIVATE
  WeftCompiler
  WeftExecutionDialect
  WeftExtensionDialect
  WeftKernelDialect
  WeftRISCVTarget
  MLIRIR
  MLIRParser
)
