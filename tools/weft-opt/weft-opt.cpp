#include "Weft/Dialect/Kernel/IR/KernelDialect.h"
#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Target/RISCVPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerTransformsPasses();
  mlir::registerPass(
      []() { return weft::createShareRISCVLayeredWindowsPass(); });
  mlir::registerPass(
      []() { return weft::createCanonicalizeRISCVLayoutsPass(); });
  mlir::registerPass(
      []() { return weft::createEliminateDeadRISCVLayoutsPass(); });
  mlir::registerPass([]() { return weft::createPlanRISCVMemoryPass(); });
  mlir::registerPass([]() { return weft::createSelectRISCVMemoryByteProjectionsPass(); });
  mlir::registerPass(
      []() { return weft::createMaterializeRISCVReplicaStorageLoadsPass(); });
  mlir::registerPass([]() { return weft::createHoistRISCVLoopInvariantsPass(); });
  mlir::registerPass([]() { return weft::createSelectRISCVOperationsPass(); });
  mlir::registerPass([]() { return weft::createFinalizeRISCVLeavesPass(); });
  mlir::registerPass(
      []() { return weft::createMaterializeRISCVReadSnapshotsPass(); });
  mlir::registerPass([]() { return weft::createMaterializeRISCVResourcesPass(); });
  mlir::registerPass([]() { return weft::createCloseRISCVLeafResourcesPass(); });
  mlir::registerPass([]() { return weft::createFuseRISCVPhysicalIssueLoopsPass(); });
  mlir::registerPass([]() { return weft::createVerifyFinalRISCVPass(); });
  mlir::DialectRegistry registry;
  registry.insert<weft::kernel::WEFTKernelDialect,
                  weft::riscv::WEFTRISCVDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv,
                        "Weft Canonical and RISC-V Physical IR driver\n",
                        registry));
}
