#ifndef WEFT_DIALECT_IMEEXECUTION_IR_IMEEXECUTIONOPS_TD
#define WEFT_DIALECT_IMEEXECUTION_IR_IMEEXECUTIONOPS_TD

include "Weft/Dialect/Execution/IR/SelectedOwnerOpInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def WEFT_IMEExecution_Dialect : Dialect {
  let name = "weft_ime_execution";
  let cppNamespace = "::weft::ime_execution";
  let summary = "Owner-local selected SpacemiT IME execution configuration";
  let description = [{
    This dialect binds shared weft_execution layout groups to one selected IME
    realization. It carries only physical choices and target facts that cannot be
    reconstructed from canonical weft_kernel IR. Algorithm shapes, signedness,
    masks, pointer expressions and instruction spellings remain canonical or
    table-derived facts.
  }];
}

class WEFTIMEExecution_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_IMEExecution_Dialect, mnemonic, traits>;

def WEFTIMEExecution_ContractConfigOp
    : WEFTIMEExecution_Op<"contract_config",
        [Symbol, SelectedOwnerOpInterface]> {
  let summary = "selected IME fragment realization of one canonical contraction";
  let description = [{
    `groups` maps one canonical contraction site's roles in order: lhs, rhs,
    result. One symbol owns all three shared groups because fragment tiling is
    one indivisible physical realization. `mac_*` is the selected physical
    fragment, `vlen_bits` is the target capability fact that makes it legal,
    and `strategy` names the selected realization. Logical M/N/K, input/result
    types, axes and the vmadot spelling are deliberately recomputed from the
    canonical contraction plus selected meta bindings.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$source_node,
    DenseI64ArrayAttr:$groups,
    I64Attr:$mac_m,
    I64Attr:$mac_n,
    I64Attr:$mac_k,
    I64Attr:$vlen_bits,
    StrAttr:$strategy
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_IMEEXECUTION_IR_IMEEXECUTIONOPS_TD
