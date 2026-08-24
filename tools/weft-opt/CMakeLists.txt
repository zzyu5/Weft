add_mlir_tool(weft-opt weft-opt.cpp)

target_link_libraries(weft-opt PRIVATE
  WeftKernelDialect
  MLIRArithDialect
  MLIROptLib
  MLIRIR
  MLIRSCFDialect
  MLIRSupport
)
