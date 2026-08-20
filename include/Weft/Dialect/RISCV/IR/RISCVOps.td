#ifndef WEFT_DIALECT_RISCV_IR_RISCVOPS_TD
#define WEFT_DIALECT_RISCV_IR_RISCVOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def WEFT_RISCV_Dialect : Dialect {
  let name = "weft_riscv";
  let cppNamespace = "::weft::riscv";
  let summary = "Transient RISC-V representation planning";
  let description = [{
    This dialect exists only inside one target-lowering invocation. A problem
    records finite physical domains and constraints derived from canonical Weft
    IR; an assignment records one complete solution. Neither operation is an
    accepted source program or a persistent algorithm authority.
  }];
}

class WeftRISCV_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_RISCV_Dialect, mnemonic, traits>;

def WeftRISCV_ProblemOp
    : WeftRISCV_Op<"problem", [Symbol, Pure]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$kernel,
    DictionaryAttr:$target,
    ArrayAttr:$candidates,
    ArrayAttr:$values,
    ArrayAttr:$operations,
    ArrayAttr:$constraints,
    DictionaryAttr:$decision_domains,
    DictionaryAttr:$resource_model,
    StrAttr:$stage
  );
  let hasVerifier = 1;
}

def WeftRISCV_AssignmentOp
    : WeftRISCV_Op<"assignment", [Symbol, Pure]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$kernel,
    DictionaryAttr:$target,
    DictionaryAttr:$candidate,
    ArrayAttr:$values,
    ArrayAttr:$operations,
    DictionaryAttr:$c_decisions,
    DictionaryAttr:$d_decisions,
    DictionaryAttr:$resources,
    StrAttr:$status
  );
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_RISCV_IR_RISCVOPS_TD
