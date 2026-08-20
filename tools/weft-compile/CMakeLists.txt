add_mlir_tool(weft-compile weft-compile.cpp)

target_link_libraries(weft-compile PRIVATE
  WeftKernelDialect
  WeftRISCVDialect
  WeftRISCVTarget
  MLIRIR
  MLIRParser
)
