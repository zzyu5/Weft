#ifndef WEFT_DIALECT_EXECUTION_IR_EXECUTIONOPS_TD
#define WEFT_DIALECT_EXECUTION_IR_EXECUTIONOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def WEFT_Execution_Dialect : Dialect {
  let name = "weft_execution";
  let cppNamespace = "::weft::execution";
  let summary = "Selected physical decisions for canonical Weft anchors";
  let description = [{
    Records in this dialect only identify choices among legal RISC-V
    realizations.  They reference canonical anchors and never copy source
    loops, logical shapes, pointer expressions, predicates, contraction axes,
    state algebra or ABI.
  }];
}

class WeftExecution_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Execution_Dialect, mnemonic, traits>;

def WeftExecution_PlanOp
    : WeftExecution_Op<"plan", [Symbol, SingleBlock]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    FlatSymbolRefAttr:$kernel_ref,
    StrAttr:$target_triple,
    StrAttr:$march,
    StrAttr:$abi,
    I64Attr:$xlen,
    BoolAttr:$little_endian,
    BoolAttr:$has_rvv,
    I64Attr:$vlen_bits,
    I64Attr:$vector_registers
  );
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftExecution_EndOp : WeftExecution_Op<"end", [Terminator]> {
  let arguments = (ins);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftExecution_MetaBindingOp : WeftExecution_Op<"meta_binding"> {
  let arguments = (ins I64Attr:$anchor, I64Attr:$value);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftExecution_AxisPlanOp : WeftExecution_Op<"axis_plan"> {
  let arguments = (ins I64Attr:$anchor, StrAttr:$provider,
                       StrAttr:$realization, I64Attr:$sew,
                       StrAttr:$lmul, I64Attr:$unroll);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftExecution_MemoryPlanOp : WeftExecution_Op<"memory_plan"> {
  let arguments = (ins I64Attr:$anchor, StrAttr:$provider,
                       StrAttr:$strategy);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftExecution_ReducePlanOp : WeftExecution_Op<"reduce_plan"> {
  let arguments = (ins I64Attr:$anchor, StrAttr:$provider,
                       StrAttr:$strategy);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftExecution_ScanPlanOp : WeftExecution_Op<"scan_plan"> {
  let arguments = (ins I64Attr:$anchor, StrAttr:$provider,
                       StrAttr:$strategy);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftExecution_SummaryPlanOp : WeftExecution_Op<"summary_plan"> {
  let arguments = (ins I64Attr:$anchor, StrAttr:$provider,
                       StrAttr:$strategy);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftExecution_ContractPlanOp : WeftExecution_Op<"contract_plan"> {
  let arguments = (ins I64Attr:$anchor, StrAttr:$provider,
                       StrAttr:$strategy, I64Attr:$micro_m,
                       I64Attr:$micro_n, I64Attr:$micro_k);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftExecution_MathPlanOp : WeftExecution_Op<"math_plan"> {
  let arguments = (ins I64Attr:$anchor, StrAttr:$provider,
                       StrAttr:$strategy);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftExecution_PrimitivePlanOp : WeftExecution_Op<"primitive_plan"> {
  let arguments = (ins I64Attr:$anchor, StrAttr:$provider,
                       StrAttr:$strategy);
  let results = (outs);
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_EXECUTION_IR_EXECUTIONOPS_TD
