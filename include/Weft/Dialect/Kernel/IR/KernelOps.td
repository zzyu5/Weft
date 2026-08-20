#ifndef WEFT_DIALECT_KERNEL_IR_KERNELOPS_TD
#define WEFT_DIALECT_KERNEL_IR_KERNELOPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def WEFT_Kernel_Dialect : Dialect {
  let name = "weft_kernel";
  let cppNamespace = "::weft::kernel";
  let summary = "Canonical Weft level, value, encoding, and operation semantics";
  let description = [{
    This dialect is the sole persistent Weft program representation. It records
    logical encodings, views, values, level ownership, ordinary ordered control,
    and explicit basic operations. It contains no hardware layout, vector width,
    register tile, fragment, instruction spelling, or built-in operator kernel.
  }];
  let useDefaultTypePrinterParser = 1;
}

class WeftKernel_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<WEFT_Kernel_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class WeftKernel_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Kernel_Dialect, mnemonic, traits>;

def WeftKernel_EncodingType : WeftKernel_Type<"Encoding", "encoding"> {
  let parameters = (ins
    StringRefParameter<"encoding family">:$family,
    StringRefParameter<"base, dense, derived_family, derived_instance, or ephemeral">:$kind,
    StringRefParameter<"fixed physical layout identity, otherwise empty">:$layoutIdentity
  );
  let assemblyFormat =
      "`<` $family `,` $kind `,` $layoutIdentity `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_ViewType : WeftKernel_Type<"View", "view"> {
  let parameters = (ins
    "::mlir::Type":$encoding,
    "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$axisIds
  );
  let assemblyFormat =
      "`<` $encoding `,` $shape `,` $axisIds `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_ValueType : WeftKernel_Type<"Value", "value"> {
  let parameters = (ins
    "::mlir::Type":$elementType,
    "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$axisIds
  );
  let assemblyFormat =
      "`<` $elementType `,` $shape `,` $axisIds `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_SliceType : WeftKernel_Type<"Slice", "slice"> {
  let parameters = (ins
    "::mlir::Type":$encoding,
    "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$axisIds
  );
  let assemblyFormat =
      "`<` $encoding `,` $shape `,` $axisIds `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_DomainType : WeftKernel_Type<"Domain", "domain"> {
  let parameters = (ins
    StringRefParameter<"axis name">:$axisName,
    "int64_t":$axisId,
    StringRefParameter<"root, rows, cols, tiles, blocks, or subs">:$relation,
    StringRefParameter<"logical extent expression">:$extent,
    StringRefParameter<"partition expression">:$partition,
    StringRefParameter<"multiplicity expression">:$multiplicity,
    StringRefParameter<"exact or tail">:$tail
  );
  let assemblyFormat =
      "`<` $axisName `,` $axisId `,` $relation `,` $extent `,` $partition `,` $multiplicity `,` $tail `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_PointType : WeftKernel_Type<"Point", "point"> {
  let parameters = (ins WeftKernel_DomainType:$domain);
  let assemblyFormat = "`<` $domain `>`";
  let genVerifyDecl = 1;
}

def WeftKernel_EncodingDeclOp
    : WeftKernel_Op<"encoding_decl", [Symbol, Pure]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$kind,
    StrAttr:$layout_identity,
    StrAttr:$bit_order,
    StrAttr:$byte_order,
    I64Attr:$alignment,
    I64Attr:$storage_bits,
    ArrayAttr:$field_names,
    ArrayAttr:$field_types,
    ArrayAttr:$field_shapes,
    ArrayAttr:$field_packing,
    DenseI64ArrayAttr:$field_bit_offsets,
    DenseI64ArrayAttr:$field_storage_bits,
    DenseI64ArrayAttr:$padding
  );
  let hasVerifier = 1;
}

def WeftKernel_DeriveOp
    : WeftKernel_Op<"derive", [Symbol, IsolatedFromAbove, SingleBlock, Pure]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       StrAttr:$source_family,
                       StrAttr:$result_family,
                       ArrayAttr:$parameters);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftKernel_DeriveYieldOp : WeftKernel_Op<"derive_yield", [Terminator]> {
  let arguments = (ins WeftKernel_ViewType:$value);
  let hasVerifier = 1;
}

def WeftKernel_KernelOp
    : WeftKernel_Op<"kernel", [Symbol, IsolatedFromAbove, SingleBlock]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       ArrayAttr:$arg_names,
                       ArrayAttr:$shape_symbols,
                       StrAttr:$source);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftKernel_ReturnOp : WeftKernel_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

def WeftKernel_RootDomainOp : WeftKernel_Op<"root_domain", [Pure]> {
  let results = (outs WeftKernel_DomainType:$result);
  let hasVerifier = 1;
}

def WeftKernel_SymbolOp : WeftKernel_Op<"symbol", [Pure]> {
  let arguments = (ins StrAttr:$name, StrAttr:$kind);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def WeftKernel_DomainOp : WeftKernel_Op<"domain", [Pure]> {
  let arguments = (ins WeftKernel_DomainType:$parent);
  let results = (outs WeftKernel_DomainType:$result);
  let hasVerifier = 1;
}

def WeftKernel_LevelOp
    : WeftKernel_Op<"level", [RecursiveMemoryEffects]> {
  let arguments = (ins WeftKernel_DomainType:$domain,
                       Variadic<AnyType>:$carried,
                       ArrayAttr:$state_names,
                       ArrayAttr:$staged_names);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$state_births,
                         SizedRegion<1>:$staged_births,
                         SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftKernel_BirthsYieldOp : WeftKernel_Op<"births_yield", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

def WeftKernel_HandoffOp : WeftKernel_Op<"handoff", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values, ArrayAttr:$names);
  let hasVerifier = 1;
}

def WeftKernel_YieldOp : WeftKernel_Op<"yield", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

def WeftKernel_ConditionOp : WeftKernel_Op<"condition", [Terminator]> {
  let arguments = (ins I1:$condition, Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

def WeftKernel_ForOp : WeftKernel_Op<"for", [RecursiveMemoryEffects]> {
  let arguments = (ins Index:$lower, Index:$upper, Index:$step,
                       Variadic<AnyType>:$init_args);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftKernel_IfOp : WeftKernel_Op<"if", [RecursiveMemoryEffects]> {
  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$then_region,
                         SizedRegion<1>:$else_region);
  let hasVerifier = 1;
}

def WeftKernel_WhileOp : WeftKernel_Op<"while", [RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<AnyType>:$init_args);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$condition_region,
                         SizedRegion<1>:$body_region);
  let hasVerifier = 1;
}

def WeftKernel_ConstantOp : WeftKernel_Op<"constant", [Pure]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_NewOp : WeftKernel_Op<"new", [Pure]> {
  let arguments = (ins Optional<AnyType>:$initial, BoolAttr:$initialized);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_MaterializeOp : WeftKernel_Op<"materialize", [Pure]> {
  let arguments = (ins AnyType:$input, OptionalAttr<StrAttr>:$engine);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_AdmitOp : WeftKernel_Op<"admit", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins AnyType:$region, OptionalAttr<StrAttr>:$engine);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_CommitOp : WeftKernel_Op<"commit", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins AnyType:$value, AnyType:$region,
                       OptionalAttr<StrAttr>:$engine);
  let hasVerifier = 1;
}

def WeftKernel_StageHandoffOp : WeftKernel_Op<"stage_handoff", [Pure]> {
  let arguments = (ins AnyType:$input, OptionalAttr<StrAttr>:$engine);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_SliceOp : WeftKernel_Op<"slice", [Pure]> {
  let arguments = (ins AnyType:$base, Variadic<AnyType>:$indices,
                       ArrayAttr:$selectors);
  let results = (outs WeftKernel_SliceType:$result);
  let hasVerifier = 1;
}

def WeftKernel_FieldOp : WeftKernel_Op<"field", [Pure]> {
  let arguments = (ins AnyType:$owner, StrAttr:$name);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_ExtractOp : WeftKernel_Op<"extract", [Pure]> {
  let arguments = (ins AnyType:$input, Variadic<AnyType>:$indices,
                       ArrayAttr:$selectors);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_UpdateOp : WeftKernel_Op<"update", [Pure]> {
  let arguments = (ins AnyType:$input, AnyType:$value,
                       Variadic<AnyType>:$indices, ArrayAttr:$selectors);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_UnaryOp : WeftKernel_Op<"unary", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$kind,
                       OptionalAttr<StrAttr>:$engine);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_BinaryOp : WeftKernel_Op<"binary", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$kind,
                       OptionalAttr<StrAttr>:$engine);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_CompareOp : WeftKernel_Op<"compare", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$predicate,
                       OptionalAttr<StrAttr>:$engine);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_CastOp : WeftKernel_Op<"cast", [Pure]> {
  let arguments = (ins AnyType:$input, OptionalAttr<StrAttr>:$engine);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

class WeftKernel_GroupedMacOp<string mnemonic>
    : WeftKernel_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, I64Attr:$group,
                       StrAttr:$overflow, OptionalAttr<StrAttr>:$engine);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_MacPairsOp : WeftKernel_GroupedMacOp<"mac_pairs">;
def WeftKernel_MacGroupsOp : WeftKernel_GroupedMacOp<"mac_groups">;

def WeftKernel_WidenOp : WeftKernel_Op<"widen", [Pure]> {
  let arguments = (ins AnyType:$input, OptionalAttr<StrAttr>:$engine);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_ReduceOp : WeftKernel_Op<"reduce", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$kind, I64Attr:$axis,
                       OptionalAttr<StrAttr>:$engine);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_Fold2Op : WeftKernel_Op<"fold2", [Pure]> {
  let arguments = (ins AnyType:$input, OptionalAttr<StrAttr>:$engine);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

class WeftKernel_ContractLikeOp<string mnemonic>
    : WeftKernel_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs,
                       DenseI64ArrayAttr:$over,
                       OptionalAttr<TypeAttr>:$acc_type,
                       OptionalAttr<StrAttr>:$engine);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_DotOp : WeftKernel_ContractLikeOp<"dot">;
def WeftKernel_ContractOp : WeftKernel_ContractLikeOp<"contract">;
def WeftKernel_OuterContractOp
    : WeftKernel_ContractLikeOp<"outer_contract">;

def WeftKernel_LookupOp : WeftKernel_Op<"lookup", [Pure]> {
  let arguments = (ins AnyType:$table, AnyType:$indices,
                       OptionalAttr<StrAttr>:$engine);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftKernel_PackOp : WeftKernel_Op<"pack", [Pure]> {
  let arguments = (ins AnyType:$view, StrAttr:$along,
                       OptionalAttr<StrAttr>:$engine);
  let results = (outs WeftKernel_ViewType:$result);
  let hasVerifier = 1;
}

def WeftKernel_InterleaveOp : WeftKernel_Op<"interleave", [Pure]> {
  let arguments = (ins WeftKernel_ViewType:$view, I64Attr:$rows,
                       OptionalAttr<StrAttr>:$engine);
  let results = (outs WeftKernel_ViewType:$result);
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_KERNEL_IR_KERNELOPS_TD
