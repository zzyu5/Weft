#ifndef WEFT_DIALECT_KERNEL_IR_KERNELOPS_TD
#define WEFT_DIALECT_KERNEL_IR_KERNELOPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def WEFT_Kernel_Dialect : Dialect {
  let name = "weft_kernel";
  let cppNamespace = "::weft::kernel";
  let summary = "Canonical worker-local Weft kernel semantics";
  let description = [{
    The dialect is the sole algorithm representation consumed by Weft.  It
    records worker-local scalar control, one-dimensional VLA regions, logical
    blocks, predicates, state algebra and structured compute.  It deliberately
    contains no launch grid, hardware lanes, vector length, LMUL, layout,
    register tile, provider identity or instruction spelling.
  }];
  let useDefaultTypePrinterParser = 1;
}

class WeftKernel_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<WEFT_Kernel_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class WeftKernel_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Kernel_Dialect, mnemonic, traits>;

def WeftKernel_PtrType : WeftKernel_Type<"Ptr", "ptr"> {
  let summary = "Typed pointer and source-declared alias/access facts";
  let parameters = (ins
    "::mlir::Type":$elementType,
    StringRefParameter<"address space">:$addressSpace,
    StringRefParameter<"read, write, or readwrite">:$access,
    "bool":$noAlias,
    "int64_t":$alignment,
    "bool":$restrictLike
  );
  let assemblyFormat = "`<` $elementType `,` $addressSpace `,` $access `,` $noAlias `,` $alignment `,` $restrictLike `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_ConstexprType : WeftKernel_Type<"Constexpr", "constexpr"> {
  let summary = "Specialization-time scalar meta-parameter";
  let parameters = (ins "::mlir::Type":$valueType);
  let assemblyFormat = "`<` $valueType `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_BlockType : WeftKernel_Type<"Block", "block"> {
  let summary = "Static logical block axes, independent of physical layout";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "::mlir::Type":$elementType
  );
  let assemblyFormat = "`<` `[` $shape `]` `,` $elementType `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_RegionType : WeftKernel_Type<"Region", "region"> {
  let summary = "Value over the active VLA axis plus zero or more block axes";
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "::mlir::Type":$elementType
  );
  let assemblyFormat = "`<` `[` $shape `]` `,` $elementType `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_MaskedType : WeftKernel_Type<"Masked", "masked"> {
  let summary = "A value carrying first-class logical validity";
  let parameters = (ins "::mlir::Type":$valueType);
  let assemblyFormat = "`<` $valueType `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_TupleType : WeftKernel_Type<"Tuple", "tuple"> {
  let summary = "Closed heterogeneous state tuple";
  let parameters = (ins ArrayRefParameter<"::mlir::Type">:$types);
  let assemblyFormat = "`<` `[` $types `]` `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_ReturnOp : WeftKernel_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftKernel_YieldOp : WeftKernel_Op<"yield", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftKernel_ConditionOp : WeftKernel_Op<"condition", [Terminator]> {
  let arguments = (ins I1:$condition, Variadic<AnyType>:$values);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftKernel_KernelOp
    : WeftKernel_Op<"kernel", [Symbol, IsolatedFromAbove, SingleBlock]> {
  let summary = "One worker-local AOT kernel entry";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ArrayAttr:$arg_names,
    ArrayAttr:$arg_kinds,
    TypeAttr:$return_type,
    StrAttr:$source
  );
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftKernel_ConstantOp : WeftKernel_Op<"constant", [Pure]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_MetaValueOp : WeftKernel_Op<"meta_value", [Pure]> {
  let arguments = (ins WeftKernel_ConstexprType:$input);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_IfOp : WeftKernel_Op<"if", [RecursiveMemoryEffects]> {
  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$then_region,
                         SizedRegion<1>:$else_region);
  let hasVerifier = 1;
}

def WeftKernel_ForOp : WeftKernel_Op<"for", [RecursiveMemoryEffects]> {
  let arguments = (ins Index:$lower, Index:$upper, Index:$step,
                       Variadic<AnyType>:$init_args);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftKernel_WhileOp : WeftKernel_Op<"while", [RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$init_args);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$condition_region,
                         SizedRegion<1>:$body_region);
  let hasVerifier = 1;
}

def WeftKernel_VLAOp : WeftKernel_Op<"vla", [RecursiveMemoryEffects]> {
  let arguments = (ins Index:$begin, Index:$end);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftKernel_BlockAxisOp : WeftKernel_Op<"block_axis", [Pure]> {
  let arguments = (ins Index:$extent, Index:$offset);
  let results = (outs WeftKernel_BlockType:$result);
  let hasVerifier = 1;
}

def WeftKernel_FullOp : WeftKernel_Op<"full", [Pure]> {
  let arguments = (ins AnyType:$value, Variadic<Index>:$extents,
                       DenseI64ArrayAttr:$shape);
  let results = (outs WeftKernel_BlockType:$result);
  let hasVerifier = 1;
}

def WeftKernel_ExpandDimsOp : WeftKernel_Op<"expand_dims", [Pure]> {
  let arguments = (ins AnyType:$input, I64Attr:$axis);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_BroadcastToOp : WeftKernel_Op<"broadcast_to", [Pure]> {
  let arguments = (ins AnyType:$input, Variadic<Index>:$extents,
                       DenseI64ArrayAttr:$shape);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_ReshapeOp : WeftKernel_Op<"reshape", [Pure]> {
  let arguments = (ins AnyType:$input, Variadic<Index>:$extents,
                       DenseI64ArrayAttr:$shape);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_TransposeOp : WeftKernel_Op<"transpose", [Pure]> {
  let arguments = (ins AnyType:$input, DenseI64ArrayAttr:$permutation);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_PtrAddOp : WeftKernel_Op<"ptr_add", [Pure]> {
  let arguments = (ins AnyType:$base, AnyType:$offset);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_UnaryOp : WeftKernel_Op<"unary", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$kind, StrAttr:$math,
                       StrAttr:$exceptional);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_BinaryOp : WeftKernel_Op<"binary", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$kind);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_CompareOp : WeftKernel_Op<"compare", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$predicate);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_CastOp : WeftKernel_Op<"cast", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_BitcastOp : WeftKernel_Op<"bitcast", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_SelectOp : WeftKernel_Op<"select", [Pure]> {
  let arguments = (ins AnyType:$predicate, AnyType:$true_value,
                       AnyType:$false_value);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_TupleOp : WeftKernel_Op<"tuple", [Pure]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let results = (outs WeftKernel_TupleType:$result);
  let hasVerifier = 1;
}

def WeftKernel_TupleGetOp : WeftKernel_Op<"tuple_get", [Pure]> {
  let arguments = (ins WeftKernel_TupleType:$input, I64Attr:$index);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_SpecialValueOp : WeftKernel_Op<"special_value", [Pure]> {
  let arguments = (ins StrAttr:$kind);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_InvalidOp : WeftKernel_Op<"invalid", [Pure]> {
  let summary = "Explicit absence of a filled value or optional predicate";
  let arguments = (ins);
  let results = (outs NoneType:$result);
}

def WeftKernel_LoadOp : WeftKernel_Op<"load", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins AnyType:$pointer, AnyType:$where, AnyType:$other,
                       I64Attr:$alignment);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_StoreOp : WeftKernel_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins AnyType:$pointer, AnyType:$value, AnyType:$where,
                       I64Attr:$alignment);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftKernel_SortIndicesOp
    : WeftKernel_Op<"sort_indices", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Stable local ordering of an f32 pointer slice into u32 indices";
  let arguments = (ins WeftKernel_PtrType:$input,
                       WeftKernel_PtrType:$output,
                       Index:$extent,
                       StrAttr:$order,
                       StrAttr:$nan,
                       StrAttr:$tie);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftKernel_PrefetchOp : WeftKernel_Op<"prefetch", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins AnyType:$pointer, AnyType:$where,
                       StrAttr:$locality);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftKernel_AtomicAddOp
    : WeftKernel_Op<"atomic_add", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins AnyType:$pointer, AnyType:$value, AnyType:$where,
                       StrAttr:$order);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_FenceOp
    : WeftKernel_Op<"fence", [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins StrAttr:$order);
  let results = (outs);
  let hasVerifier = 1;
}

def WeftKernel_ValidOp : WeftKernel_Op<"valid", [Pure]> {
  let arguments = (ins WeftKernel_MaskedType:$input);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_FillOp : WeftKernel_Op<"fill", [Pure]> {
  let arguments = (ins WeftKernel_MaskedType:$input, AnyType:$fill_value);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_ReduceOp : WeftKernel_Op<"reduce", [Pure]> {
  let arguments = (ins AnyType:$input, AnyType:$identity, AnyType:$where,
                       StrAttr:$kind, I64Attr:$axis, StrAttr:$order,
                       TypeAttr:$acc_dtype);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_ScanOp : WeftKernel_Op<"scan", [Pure]> {
  let arguments = (ins AnyType:$input, AnyType:$identity, AnyType:$where,
                       AnyType:$segment_start, StrAttr:$kind,
                       BoolAttr:$inclusive, StrAttr:$order,
                       TypeAttr:$acc_dtype);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_SummaryFoldOp : WeftKernel_Op<"summary_fold", [Pure]> {
  let arguments = (ins AnyType:$input, AnyType:$identity, AnyType:$where,
                       AnyType:$coordinate, StrAttr:$order);
  let results = (outs AnyType:$result);
  let regions = (region SizedRegion<1>:$lift,
                         SizedRegion<1>:$merge,
                         SizedRegion<1>:$finalize);
  let hasVerifier = 1;
}

def WeftKernel_ContractOp : WeftKernel_Op<"contract", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, AnyType:$init,
                       AnyType:$where_lhs, AnyType:$where_rhs,
                       DenseI64ArrayAttr:$lhs_axes,
                       DenseI64ArrayAttr:$rhs_axes,
                       DenseI64ArrayAttr:$output_order,
                       TypeAttr:$acc_dtype, TypeAttr:$out_dtype,
                       StrAttr:$order, StrAttr:$math);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_PermuteOp : WeftKernel_Op<"permute", [Pure]> {
  let arguments = (ins AnyType:$input, DenseI64ArrayAttr:$permutation);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_LookupOp : WeftKernel_Op<"lookup", [Pure]> {
  let arguments = (ins AnyType:$table, AnyType:$indices, AnyType:$where);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_DecodeOp : WeftKernel_Op<"decode", [Pure]> {
  let arguments = (ins AnyType:$codes, AnyType:$table, AnyType:$where,
                       TypeAttr:$out_dtype);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_WidenOp : WeftKernel_Op<"widen", [Pure]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_NarrowOp : WeftKernel_Op<"narrow", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$rounding,
                       BoolAttr:$saturation);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_KERNEL_IR_KERNELOPS_TD
