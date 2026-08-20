#include "Weft/Target/RISCVCompiler.h"

#include "Weft/Dialect/RISCV/IR/RISCVPlanningDialect.h"
#include "Weft/Target/RISCVPasses.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace {

void printDictionary(llvm::raw_ostream &output, mlir::DictionaryAttr dictionary,
                     llvm::StringRef indent) {
  for (mlir::NamedAttribute named : dictionary) {
    output << indent << named.getName() << " = ";
    named.getValue().print(output);
    output << '\n';
  }
}

void printAssignment(llvm::raw_ostream &output,
                     weft::riscv::AssignmentOp assignment) {
  output << "kernel = @" << assignment.getKernel() << '\n';
  output << "status = " << assignment.getStatus() << '\n';
  output << "\ntarget\n";
  printDictionary(output, assignment.getTarget(), "  ");
  output << "\ncandidate\n";
  printDictionary(output, assignment.getCandidate(), "  ");
  output << "\nC: vector representation decisions\n";
  printDictionary(output, assignment.getCDecisions(), "  ");
  output << "\nD: instruction and memory decisions\n";
  printDictionary(output, assignment.getDDecisions(), "  ");
  output << "\nresources\n";
  printDictionary(output, assignment.getResources(), "  ");
  output << "\nvalues\n";
  for (mlir::Attribute attribute : assignment.getValues()) {
    auto value = mlir::cast<mlir::DictionaryAttr>(attribute);
    output << "  " << mlir::cast<mlir::StringAttr>(value.get("id")).getValue()
           << "  "
           << mlir::cast<mlir::StringAttr>(value.get("logical_type")).getValue()
           << '\n';
    for (llvm::StringRef key :
         {"physical_kind", "physical_encoding_kind", "physical_sew",
          "lane_axis", "lmul", "vl", "register_groups", "storage",
          "encoding_family", "layout_identity"})
      if (mlir::Attribute field = value.get(key)) {
        output << "    " << key << " = ";
        field.print(output);
        output << '\n';
      }
  }
  output << "\noperations\n";
  for (mlir::Attribute attribute : assignment.getOperations()) {
    auto operation = mlir::cast<mlir::DictionaryAttr>(attribute);
    output << "  "
           << mlir::cast<mlir::StringAttr>(operation.get("id")).getValue()
           << "  "
           << mlir::cast<mlir::StringAttr>(operation.get("source_op")).getValue()
           << " -> "
           << mlir::cast<mlir::StringAttr>(operation.get("realization")).getValue()
           << '\n';
    for (llvm::StringRef key : {"validity", "level_mapping"})
      if (mlir::Attribute field = operation.get(key)) {
        output << "    " << key << " = ";
        field.print(output);
        output << '\n';
      }
  }
}

} // namespace

mlir::FailureOr<weft::RISCVPlanningResult>
weft::planRISCVModule(mlir::ModuleOp module, RISCVCompilerOptions options) {
  mlir::PassManager manager(module.getContext());
  manager.enableVerifier(true);
  manager.addPass(createConstructRISCVProblemsPass(options));
  manager.addPass(createConstrainRISCVRepresentationsPass());
  manager.addPass(createConstrainRISCVInstructionsPass());
  manager.addPass(createConstrainRISCVResourcesPass());
  manager.addPass(createSolveRISCVProblemsPass());
  if (mlir::failed(manager.run(module)))
    return mlir::failure();

  RISCVPlanningResult result;
  llvm::raw_string_ostream output(result.assignment);
  bool found = false;
  for (riscv::AssignmentOp assignment : module.getOps<riscv::AssignmentOp>()) {
    if (found)
      output << "\n---\n\n";
    printAssignment(output, assignment);
    found = true;
  }
  if (!found) {
    module.emitError("RISC-V planning pipeline produced no assignment");
    return mlir::failure();
  }
  output.flush();
  return result;
}
