add_mlir_tool(weft-opt weft-opt.cpp)

target_link_libraries(weft-opt PRIVATE
  WeftExtensionDialect
  WeftKernelDialect
  MLIROptLib
  MLIRIR
  MLIRSupport
)
