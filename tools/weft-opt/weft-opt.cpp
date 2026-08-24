#include "Weft/Dialect/Kernel/IR/KernelDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<weft::kernel::WEFTKernelDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Weft canonical Kernel IR driver\n",
                        registry));
}
