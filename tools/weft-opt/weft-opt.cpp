#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<weft::kernel::WEFTKernelDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Weft canonical Kernel IR driver\n",
                        registry));
}
