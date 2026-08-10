#ifndef WEFT_DIALECT_EXECUTION_IR_EXECUTIONOPS_TD
#define WEFT_DIALECT_EXECUTION_IR_EXECUTIONOPS_TD

include "Weft/Dialect/Layout/IR/LayoutAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def WEFT_Execution_Dialect : Dialect {
  let name = "weft_execution";
  let cppNamespace = "::weft::execution";
  let summary = "Selected RISC-V execution decisions for canonical Weft kernels";
  let description = [{
    This dialect is the single selected physical-decision layer between
    weft_kernel and generated RISC-V artifacts. It references canonical nodes;
    it never copies algorithm operations or stores alternate candidates.
  }];
}

class WEFTExecution_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Execution_Dialect, mnemonic, traits>;

def WEFTExecution_PlanOp
    : WEFTExecution_Op<"plan",
        [IsolatedFromAbove, NoRegionArguments, NoTerminator, SingleBlock,
         Symbol, SymbolTable]> {
  let summary = "one selected physical execution plan for a canonical kernel";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$kernel,
    StrAttr:$target
  );
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$sym_name `for` $kernel attr-dict-with-keyword $body";
  let hasVerifier = 1;
}

def WEFTExecution_TaskBindingOp : WEFTExecution_Op<"task_binding"> {
  let summary = "selected realization of one logical task-grid axis";
  let arguments = (ins I64Attr:$axis, StrAttr:$mapping);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def WEFTExecution_MetaBindingOp : WEFTExecution_Op<"meta_binding"> {
  let summary = "selected integer value of one canonical constexpr argument";
  let arguments = (ins I64Attr:$argument, I64Attr:$value);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def WEFTExecution_GroupOp : WEFTExecution_Op<"group"> {
  let summary = "one selected layout-equivalence group of blocked SSA values";
  let description = [{
    `owner` names an owner-local selected record nested directly in the same
    plan. The referenced symbol implements SelectedOwnerOpInterface and carries
    either one integer `group` or one non-empty `groups` back-reference. The
    shared execution dialect does not depend on a concrete RVV, Scalar, IME or
    future owner operation class.
  }];
  let arguments = (ins
    I64Attr:$id,
    I64Attr:$scope_node,
    WEFTLayout_BlockedLayoutAttr:$layout,
    FlatSymbolRefAttr:$owner
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def WEFTExecution_ValueLayoutOp : WEFTExecution_Op<"value_layout"> {
  let summary = "map one canonical block result to its selected layout group";
  let arguments = (ins
    I64Attr:$source_node,
    I64Attr:$source_result,
    I64Attr:$group
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def WEFTExecution_LayoutConversionOp
    : WEFTExecution_Op<"layout_conversion"> {
  let summary = "convert one canonical operand use into a selected target group";
  let description = [{
    The canonical source value and its native group are derived from the
    consumer operand and value_layout. Only the selected target group is
    persisted, so this record is a physical use edge rather than a copied SSA
    value or a second algorithm operation.
  }];
  let arguments = (ins
    I64Attr:$consumer_node,
    I64Attr:$consumer_operand,
    I64Attr:$group
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_EXECUTION_IR_EXECUTIONOPS_TD
