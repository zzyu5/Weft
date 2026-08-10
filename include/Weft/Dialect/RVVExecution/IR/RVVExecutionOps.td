#ifndef WEFT_DIALECT_RVVEXECUTION_IR_RVVEXECUTIONOPS_TD
#define WEFT_DIALECT_RVVEXECUTION_IR_RVVEXECUTIONOPS_TD

include "Weft/Dialect/Execution/IR/SelectedOwnerOpInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def WEFT_RVVExecution_Dialect : Dialect {
  let name = "weft_rvv_execution";
  let cppNamespace = "::weft::rvv_execution";
  let summary = "Owner-local selected RVV execution configuration";
  let description = [{
    These operations bind shared weft_execution layout groups to RVV VLA
    resources. They contain only final owner choices, never algorithm copies,
    candidate lists or intrinsic spellings.
  }];
}

class WEFTRVVExecution_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_RVVExecution_Dialect, mnemonic, traits>;

def WEFTRVVExecution_GroupConfigOp
    : WEFTRVVExecution_Op<"group_config",
        [Symbol, SelectedOwnerOpInterface]> {
  let summary = "selected VLA/LMUL ratio and register budget for one layout group";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$group,
    OptionalAttr<I64Attr>:$lane_ratio,
    I64Attr:$register_budget
  );
  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

def WEFTRVVExecution_ReductionConfigOp
    : WEFTRVVExecution_Op<"reduction_config",
        [SelectedOwnerOpInterface]> {
  let summary = "selected RVV realization of one canonical reduction";
  let arguments = (ins
    I64Attr:$source_node,
    I64Attr:$group,
    StrAttr:$strategy
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def WEFTRVVExecution_UnaryConfigOp
    : WEFTRVVExecution_Op<"unary_config",
        [SelectedOwnerOpInterface]> {
  let summary = "selected RVV realization of one canonical pointwise unary op";
  let description = [{
    The canonical operation retains the mathematical kind and types. `strategy`
    records only the final owner-local numerical realization selected for that
    operation; coefficient tables and intrinsic spellings remain emitter-static.
  }];
  let arguments = (ins
    I64Attr:$source_node,
    I64Attr:$group,
    StrAttr:$strategy
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def WEFTRVVExecution_ContractConfigOp
    : WEFTRVVExecution_Op<"contract_config",
        [SelectedOwnerOpInterface]> {
  let summary = "selected RVV realization of one canonical contraction";
  let description = [{
    `groups` maps canonical contraction roles in order: lhs, rhs, and, when the
    canonical result is blocked, result. The record references selected layout
    groups only; shapes, extents and contraction axes remain in weft_kernel.
  }];
  let arguments = (ins
    I64Attr:$source_node,
    DenseI64ArrayAttr:$groups,
    StrAttr:$strategy
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_RVVEXECUTION_IR_RVVEXECUTIONOPS_TD
