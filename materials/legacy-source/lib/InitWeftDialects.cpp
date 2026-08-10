#include "Weft/InitWeftDialects.h"

#include "Weft/Dialect/Exec/IR/ExecOps.h"
#include "Weft/Dialect/Execution/IR/ExecutionDialect.h"
#include "Weft/Dialect/IMEExecution/IR/IMEExecutionDialect.h"
#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/Layout/IR/LayoutDialect.h"
#include "Weft/Dialect/RVVExecution/IR/RVVExecutionDialect.h"
#include "Weft/Plugin/ExtensionPlugin.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"

namespace weft {

void registerAllDialects(mlir::DialectRegistry &registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::scf::SCFDialect,
                  mlir::memref::MemRefDialect, mlir::vector::VectorDialect,
                  weft::exec::WEFTExecDialect,
                  weft::execution::WEFTExecutionDialect,
                  weft::ime_execution::WEFTIMEExecutionDialect,
                  weft::kernel::WEFTKernelDialect,
                  weft::layout::WEFTLayoutDialect,
                  weft::rvv_execution::WEFTRVVExecutionDialect>();
}

void registerPluginDialects(const plugin::ExtensionPluginRegistry &plugins,
                            mlir::DialectRegistry &registry) {
  plugins.registerDialectsForEnabledPlugins(registry);
}

} // namespace weft
