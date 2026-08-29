add_mlir_tool(weft-opt weft-opt.cpp)

target_link_libraries(weft-opt PRIVATE
  WeftKernelDialect
  WeftRISCVDialect
  MLIRArithDialect
  MLIROptLib
  MLIRIR
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
)
