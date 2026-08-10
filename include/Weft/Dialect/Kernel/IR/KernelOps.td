#ifndef WEFT_DIALECT_KERNEL_IR_KERNELOPS_TD
#define WEFT_DIALECT_KERNEL_IR_KERNELOPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def WEFT_Kernel_Dialect : Dialect {
  let name = "weft_kernel";
  let cppNamespace = "::weft::kernel";
  let summary = "Canonical blocked kernel programs for RISC-V targets";
  let description = [{
    The Weft Kernel dialect is the standalone algorithm boundary of the Weft
    compiler. It represents task-grid blocked programs, logical block values,
    explicit pointer arithmetic, masked memory, structured reductions and
    loop-carried state. It deliberately excludes VLEN, vl, LMUL, physical
    vector registers, IME fragments and target instruction spellings.
  }];
  let useDefaultTypePrinterParser = 1;
}

class WEFTKernel_Type<string name, string typeMnemonic,
                      list<Trait> traits = []>
    : TypeDef<WEFT_Kernel_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class WEFTKernel_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Kernel_Dialect, mnemonic, traits>;

def WEFTKernel_PtrType : WEFTKernel_Type<"Ptr", "ptr"> {
  let summary = "Typed source-level kernel pointer";
  let parameters = (ins
    "::mlir::Type":$elementType,
    StringRefParameter<"logical memory space">:$addressSpace
  );
  let assemblyFormat = "`<` $elementType `,` $addressSpace `>`";
  let genVerifyDecl = 1;
}

def WEFTKernel_BlockType : WEFTKernel_Type<"Block", "block"> {
  let summary = "Logical blocked-program value, independent of hardware lanes";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "::mlir::Type":$elementType
  );
  let assemblyFormat = "`<` `[` $shape `]` `,` $elementType `>`";
  let genVerifyDecl = 1;
}

def WEFTKernel_ConstexprType : WEFTKernel_Type<"Constexpr", "constexpr"> {
  let summary = "Compile-time kernel meta-parameter";
  let parameters = (ins "::mlir::Type":$valueType);
  let assemblyFormat = "`<` $valueType `>`";
  let genVerifyDecl = 1;
}

def WEFTKernel_ReturnOp : WEFTKernel_Op<"return", [Terminator]> {
  let summary = "Terminate a Weft kernel entry";
  let arguments = (ins);
  let results = (outs);
  let hasVerifier = 1;
}

def WEFTKernel_YieldOp : WEFTKernel_Op<"yield", [Terminator]> {
  let summary = "Yield loop-carried values from a structured Weft region";
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs);
  let hasVerifier = 1;
}

def WEFTKernel_KernelOp
    : WEFTKernel_Op<"kernel", [Symbol, IsolatedFromAbove, SingleBlock]> {
  let summary = "Standalone canonical Weft kernel entry";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    I64Attr:$grid_rank,
    ArrayAttr:$arg_names,
    ArrayAttr:$arg_kinds,
    OptionalAttr<StrAttr>:$source
  );
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WEFTKernel_ConstantOp : WEFTKernel_Op<"constant", [Pure]> {
  let summary = "Scalar constant in a blocked kernel program";
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WEFTKernel_MetaValueOp : WEFTKernel_Op<"meta_value", [Pure]> {
  let summary = "Materialize a constexpr value for blocked index construction";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WEFTKernel_TaskIdOp : WEFTKernel_Op<"task_id", [Pure]> {
  let summary = "Logical task identity along one launch-grid axis";
  let arguments = (ins I64Attr:$axis);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def WEFTKernel_ArangeOp : WEFTKernel_Op<"arange", [Pure]> {
  let summary = "Construct one rank-one logical block of indices";
  let arguments = (ins Index:$start, Index:$extent);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WEFTKernel_ExpandDimsOp : WEFTKernel_Op<"expand_dims", [Pure]> {
  let summary = "Insert one singleton logical block axis without materialization";
  let arguments = (ins AnyType:$input, I64Attr:$axis);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WEFTKernel_SplatOp : WEFTKernel_Op<"splat", [Pure]> {
  let summary = "Splat one scalar over the logical shape of another block";
  let description = [{
    `shape_like` supplies only the canonical logical block shape and its dynamic
    extent provenance. The scalar value determines the result element type.
    No physical layout, vector width or owner is represented here.
  }];
  let arguments = (ins AnyType:$value, AnyType:$shape_like);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WEFTKernel_PtrAddOp : WEFTKernel_Op<"ptr_add", [Pure]> {
  let summary = "Apply scalar or blocked element offsets to a typed pointer";
  let arguments = (ins AnyType:$base, AnyType:$offset);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WEFTKernel_UnaryOp : WEFTKernel_Op<"unary", [Pure]> {
  let summary = "Closed logical pointwise unary operation";
  let arguments = (ins AnyType:$input, StrAttr:$kind);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WEFTKernel_BinaryOp : WEFTKernel_Op<"binary", [Pure]> {
  let summary = "Closed logical pointwise binary operation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$kind);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WEFTKernel_CompareOp : WEFTKernel_Op<"compare", [Pure]> {
  let summary = "Closed logical pointwise comparison";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$predicate);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WEFTKernel_CastOp : WEFTKernel_Op<"cast", [Pure]> {
  let summary = "Elementwise scalar or block cast";
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WEFTKernel_LoadOp
    : WEFTKernel_Op<"load", [MemoryEffects<[MemRead]>]> {
  let summary = "Explicit masked scalar or blocked load";
  let arguments = (ins AnyType:$pointer, AnyType:$mask, AnyType:$other);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WEFTKernel_StoreOp
    : WEFTKernel_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Explicit masked scalar or blocked store";
  let arguments = (ins AnyType:$pointer, AnyType:$value, AnyType:$mask);
  let results = (outs);
  let hasVerifier = 1;
}

def WEFTKernel_ContractOp : WEFTKernel_Op<"contract", [Pure]> {
  let summary = "Logical tensor contraction with an explicit accumulator";
  let description = [{
    Contracts paired logical axes of lhs and rhs. The result axis order is the
    non-contracted lhs axes followed by the non-contracted rhs axes. This is an
    algorithm operation: target layouts, RVV lanes and IME fragments are not
    represented here. The accumulator/result element type normally matches the
    inputs; validated widening integer forms preserve their wider accumulator
    type in the result.
  }];
  let arguments = (ins
    AnyType:$lhs,
    AnyType:$rhs,
    AnyType:$init,
    DenseI64ArrayAttr:$lhs_axes,
    DenseI64ArrayAttr:$rhs_axes,
    BoolAttr:$ordered
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WEFTKernel_ReduceOp : WEFTKernel_Op<"reduce", [Pure]> {
  let summary = "Structured reduction over one logical block axis";
  let arguments = (ins
    AnyType:$input,
    AnyType:$init,
    I64Attr:$axis,
    StrAttr:$kind,
    BoolAttr:$ordered
  );
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WEFTKernel_ForOp
    : WEFTKernel_Op<"for", [RecursiveMemoryEffects]> {
  let summary = "Structured loop with explicit SSA carried state";
  let arguments = (ins
    Index:$lower,
    Index:$upper,
    Index:$step,
    Variadic<AnyType>:$init_args
  );
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_KERNEL_IR_KERNELOPS_TD
