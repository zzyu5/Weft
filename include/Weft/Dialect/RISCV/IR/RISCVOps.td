#ifndef WEFT_DIALECT_RISCV_IR_RISCVOPS_TD
#define WEFT_DIALECT_RISCV_IR_RISCVOPS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def WEFT_RISCV_Dialect : Dialect {
  let name = "weft_riscv";
  let cppNamespace = "::weft::riscv";
  let summary = "Target-aware non-SIMT RISC-V physical program";
  let description = [{
    The dialect is Weft's second and final program IR. It carries logical-axis
    preserving physical representations, explicit conversions, memory edges,
    local storage, physical loops, and selected RVV/IME local operations. It is
    a real SSA program rather than a planning record beside Canonical Kernel IR.
  }];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

class WeftRISCV_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<WEFT_RISCV_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class WeftRISCV_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<WEFT_RISCV_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class WeftRISCV_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_RISCV_Dialect, mnemonic, traits>;

def WeftRISCV_LayoutAttr : WeftRISCV_Attr<"Layout", "layout"> {
  let parameters = (ins
    StringRefParameter<"unassigned, scalar, rvv, ime, or local">:$carrier,
    "::mlir::DenseI64ArrayAttr":$axisIds,
    "::mlir::DenseI64ArrayAttr":$timeFactors,
    "::mlir::DenseI64ArrayAttr":$laneFactors,
    "::mlir::DenseI64ArrayAttr":$replicaFactors,
    "::mlir::DenseI64ArrayAttr":$fragmentFactors,
    "::mlir::DenseI64ArrayAttr":$localFactors,
    "int64_t":$sew,
    "int64_t":$lmulEighths,
    "int64_t":$vl,
    "int64_t":$registerGroups,
    StringRefParameter<"unassigned, full, tail, or mask">:$validity
  );
  let assemblyFormat = [{
    `<` $carrier `,` $axisIds `,` $timeFactors `,` $laneFactors `,`
        $replicaFactors `,` $fragmentFactors `,` $localFactors `,`
        $sew `,` $lmulEighths `,` $vl `,` $registerGroups `,` $validity `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_FragmentPackingAttr
    : WeftRISCV_Attr<"FragmentPacking", "fragment_packing"> {
  let parameters = (ins
    StringRefParameter<"target fragment packing identity">:$schema,
    "::mlir::DenseI64ArrayAttr":$axisOrder,
    "int64_t":$rowsPerTile,
    "int64_t":$columnsPerTile,
    StringRefParameter<"row_major or column_major tile order">:$tileOrder,
    StringRefParameter<"row_major or column_major element order">:$elementOrder,
    "int64_t":$storageBits,
    "int64_t":$alignment
  );
  let assemblyFormat = [{
    `<` $schema `,` $axisOrder `,` $rowsPerTile `,` $columnsPerTile `,` $tileOrder `,`
        $elementOrder `,` $storageBits `,` $alignment `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_FragmentCapabilityAttr
    : WeftRISCV_Attr<"FragmentCapability", "fragment_capability"> {
  let parameters = (ins
    StringRefParameter<"instruction identity">:$instruction,
    StringRefParameter<"lhs signedness">:$lhsSignedness,
    StringRefParameter<"rhs signedness">:$rhsSignedness,
    "int64_t":$lhsBits,
    "int64_t":$rhsBits,
    "int64_t":$accumulatorBits,
    "int64_t":$mFactor,
    "int64_t":$nFactor,
    "int64_t":$kFactor,
    "int64_t":$lhsResourceGroups,
    "int64_t":$rhsResourceGroups,
    "int64_t":$accumulatorResourceGroups,
    "::weft::riscv::FragmentPackingAttr":$lhsPacking,
    "::weft::riscv::FragmentPackingAttr":$rhsPacking,
    "::weft::riscv::FragmentPackingAttr":$accumulatorPacking,
    "int64_t":$mmaGroups,
    "int64_t":$mmaChunks,
    "::mlir::ArrayAttr":$clobbers,
    "bool":$volatileAsm,
    "bool":$memoryClobber
  );
  let assemblyFormat = [{
    `<` $instruction `,` $lhsSignedness `,` $rhsSignedness `,`
        $lhsBits `,` $rhsBits `,` $accumulatorBits `,`
        $mFactor `,` $nFactor `,` $kFactor `,`
        $lhsResourceGroups `,` $rhsResourceGroups `,`
        $accumulatorResourceGroups `,`
        $lhsPacking `,` $rhsPacking `,` $accumulatorPacking `,`
        $mmaGroups `,` $mmaChunks `,`
        $clobbers `,` $volatileAsm `,` $memoryClobber `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_TargetAttr : WeftRISCV_Attr<"Target", "target"> {
  let parameters = (ins
    StringRefParameter<"target triple">:$triple,
    StringRefParameter<"RISC-V ISA string">:$march,
    StringRefParameter<"ABI">:$abi,
    "bool":$hasRVV,
    "bool":$hasVectorF16,
    "bool":$hasIndexedMemory,
    "bool":$hasSegmentMemory,
    "bool":$hasWideningInteger,
    "bool":$hasWideningFloat,
    "int64_t":$vlenBits,
    "int64_t":$vectorRegisters,
    "int64_t":$maxPrivateStackBytes,
    "::mlir::DenseI64ArrayAttr":$supportedSEW,
    "::mlir::DenseI64ArrayAttr":$legalLMULEighths,
    "::mlir::ArrayAttr":$fragments
  );
  let assemblyFormat = [{
    `<` $triple `,` $march `,` $abi `,` $hasRVV `,` $hasVectorF16 `,`
        $hasIndexedMemory `,` $hasSegmentMemory `,` $hasWideningInteger `,`
        $hasWideningFloat `,` $vlenBits `,` $vectorRegisters `,`
        $maxPrivateStackBytes `,` $supportedSEW `,` $legalLMULEighths `,`
        $fragments `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_AccessAttr : WeftRISCV_Attr<"Access", "access"> {
  let parameters = (ins
    StringRefParameter<"unassigned, unit, strided, indexed, segment, register, or local">:$form,
    StringRefParameter<"dense, natural, grouped_layered, joined, or opaque">:$mapping,
    "int64_t":$alignment,
    "int64_t":$indexSEW,
    "int64_t":$segmentFields,
    "int64_t":$groupSize,
    "int64_t":$layerSize,
    "int64_t":$joinFields,
    "int64_t":$joinLowBits,
    "int64_t":$joinRole,
    "int64_t":$bitOffset,
    "int64_t":$storageBits,
    StringRefParameter<"none, lo_first, or hi_first">:$order
  );
  let assemblyFormat = [{
    `<` $form `,` $mapping `,` $alignment `,` $indexSEW `,`
        $segmentFields `,` $groupSize `,` $layerSize `,` $joinFields `,`
        $joinLowBits `,` $joinRole `,` $bitOffset `,`
        $storageBits `,` $order `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_ScheduleAttr : WeftRISCV_Attr<"Schedule", "schedule"> {
  let parameters = (ins
    StringRefParameter<"sequential or pipelined">:$kind,
    "int64_t":$unroll,
    "int64_t":$pipelineDepth,
    "int64_t":$bufferCount,
    "int64_t":$prefetchDistance
  );
  let assemblyFormat = [{
    `<` $kind `,` $unroll `,` $pipelineDepth `,` $bufferCount `,`
        $prefetchDistance `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_LevelAttr : WeftRISCV_Attr<"Level", "level"> {
  let parameters = (ins
    "int64_t":$domainId,
    "int64_t":$axisId,
    StringRefParameter<"rows, cols, tiles, blocks, or subs">:$relation,
    "int64_t":$stateBirthCount,
    "int64_t":$stagedBirthCount,
    "int64_t":$handoffCount,
    StringRefParameter<"ascending or descending">:$direction
  );
  let assemblyFormat = [{
    `<` $domainId `,` $axisId `,` $relation `,` $stateBirthCount `,`
        $stagedBirthCount `,` $handoffCount `,` $direction `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_ImplementationAttr
    : WeftRISCV_Attr<"Implementation", "implementation"> {
  let parameters = (ins
    StringRefParameter<"scalar, rvv, ime, or transfer">:$engine,
    StringRefParameter<"local semantic implementation family">:$family,
    StringRefParameter<"target operation class">:$operation,
    "::mlir::DenseI64ArrayAttr":$parameters
  );
  let assemblyFormat =
      "`<` $engine `,` $family `,` $operation `,` $parameters `>`";
  let genVerifyDecl = 1;
}

def WeftRISCV_LeafAttr : WeftRISCV_Attr<"Leaf", "leaf"> {
  let parameters = (ins
    StringRefParameter<"unselected, scalar, rvv, ime, or transfer">:$engine,
    StringRefParameter<"local operation family">:$family,
    StringRefParameter<"instruction identity">:$instruction,
    StringRefParameter<"terminal spelling key">:$spelling,
    "int64_t":$operandGroups,
    "int64_t":$resultGroups,
    "int64_t":$temporaryGroups,
    "int64_t":$fragmentGroups,
    StringRefParameter<"none, explicit, or tail">:$mask,
    StringRefParameter<"exact or agnostic">:$tail,
    "::mlir::DenseI64ArrayAttr":$parameters,
    "int64_t":$localBytes
  );
  let assemblyFormat = [{
    `<` $engine `,` $family `,` $instruction `,` $spelling `,`
        $operandGroups `,` $resultGroups `,` $temporaryGroups `,`
        $fragmentGroups `,` $mask `,` $tail `,` $parameters `,`
        $localBytes `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_ConversionAttr : WeftRISCV_Attr<"Conversion", "conversion"> {
  let parameters = (ins
    StringRefParameter<"splat, extract, reshape, tuple, register_to_lane, lane_to_register, rvv_to_fragment, fragment_to_rvv, local_load, or local_store">:$kind,
    StringRefParameter<"pure, read, write, or handoff">:$effect,
    "int64_t":$temporaryGroups
  );
  let assemblyFormat = "`<` $kind `,` $effect `,` $temporaryGroups `>`";
  let genVerifyDecl = 1;
}

def WeftRISCV_ValueType : WeftRISCV_Type<"Value", "value"> {
  let parameters = (ins
    "::mlir::Type":$elementType,
    "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$axisIds,
    "::weft::riscv::LayoutAttr":$layout
  );
  let assemblyFormat = "`<` $elementType `,` $shape `,` $axisIds `,` $layout `>`";
  let genVerifyDecl = 1;
}

def WeftRISCV_MemDescType : WeftRISCV_Type<"MemDesc", "memdesc"> {
  let parameters = (ins
    "::mlir::Type":$encoding,
    "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$axisIds,
    "::mlir::DenseI64ArrayAttr":$strides,
    "::mlir::DenseI64ArrayAttr":$origins,
    "int64_t":$alignment,
    StringRefParameter<"pinned, slice, or local">:$addressClass,
    StringRefParameter<"none, read, write, or readwrite">:$access,
    "int64_t":$aliasSet,
    StringRefParameter<"pinned storage layout identity">:$layoutIdentity,
    "int64_t":$storageBits,
    "int64_t":$elements,
    "int64_t":$interleaveRows
  );
  let assemblyFormat = [{
    `<` $encoding `,` $shape `,` $axisIds `,` $strides `,` $origins `,`
        $alignment `,` $addressClass `,` $access `,` $aliasSet `,`
        $layoutIdentity `,` $storageBits `,` $elements `,` $interleaveRows `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_LocalType : WeftRISCV_Type<"Local", "local"> {
  let parameters = (ins
    "::mlir::Type":$elementType,
    "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$axisIds,
    "int64_t":$sizeBytes,
    "int64_t":$alignment,
    "int64_t":$aliasSet,
    StringRefParameter<"spill, pack, pipeline, or handoff">:$purpose,
    "int64_t":$ownerDomainId,
    "int64_t":$birthId,
    "int64_t":$lifetimeEndDomainId,
    StringRefParameter<"physical storage schema">:$schema
  );
  let assemblyFormat = [{
    `<` $elementType `,` $shape `,` $axisIds `,` $sizeBytes `,` $alignment `,`
        $aliasSet `,` $purpose `,` $ownerDomainId `,` $birthId `,`
        $lifetimeEndDomainId `,` $schema `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_FragmentType : WeftRISCV_Type<"Fragment", "fragment"> {
  let parameters = (ins
    StringRefParameter<"fragment family">:$family,
    StringRefParameter<"lhs, rhs, or accumulator fragment role">:$role,
    "::weft::riscv::FragmentPackingAttr":$packing,
    "::mlir::Type":$elementType,
    "::mlir::DenseI64ArrayAttr":$shape,
    "::mlir::DenseI64ArrayAttr":$axisIds,
    "::weft::riscv::LayoutAttr":$layout,
    "int64_t":$resourceGroups
  );
  let assemblyFormat = [{
    `<` $family `,` $role `,` $packing `,` $elementType `,` $shape `,`
        $axisIds `,` $layout `,` $resourceGroups `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_WindowType : WeftRISCV_Type<"Window", "window"> {
  let parameters = (ins
    StringRefParameter<"grouped-mac, encoded-dot, or contract">:$family,
    "::mlir::Type":$lhsType,
    "::mlir::Type":$rhsType,
    "::mlir::Type":$resultType,
    "int64_t":$reductionAxis,
    "int64_t":$slots,
    "int64_t":$termsPerSlot,
    "int64_t":$resultParts,
    "::weft::riscv::LayoutAttr":$partialLayout,
    "::weft::riscv::LayoutAttr":$resultLayout,
    "int64_t":$resourceGroups
  );
  let assemblyFormat = [{
    `<` $family `,` $lhsType `,` $rhsType `,` $resultType `,` $reductionAxis `,` $slots `,`
        $termsPerSlot `,` $resultParts `,` $partialLayout `,` $resultLayout `,` $resourceGroups `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_DomainType : WeftRISCV_Type<"Domain", "domain"> {
  let parameters = (ins
    StringRefParameter<"axis name">:$axisName,
    "int64_t":$domainId,
    "int64_t":$parentDomainId,
    "int64_t":$axisId,
    StringRefParameter<"root, rows, cols, tiles, blocks, or subs">:$relation,
    StringRefParameter<"exact or tail">:$tail
  );
  let assemblyFormat = [{
    `<` $axisName `,` $domainId `,` $parentDomainId `,` $axisId `,`
        $relation `,` $tail `>`
  }];
  let genVerifyDecl = 1;
}

def WeftRISCV_PointType : WeftRISCV_Type<"Point", "point"> {
  let parameters = (ins WeftRISCV_DomainType:$domain);
  let assemblyFormat = "`<` $domain `>`";
  let genVerifyDecl = 1;
}

def WeftRISCV_EncodingDeclOp
    : WeftRISCV_Op<"encoding_decl", [Symbol, Pure]> {
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$kind,
                       StrAttr:$layout_identity, StrAttr:$bit_order,
                       StrAttr:$byte_order, I64Attr:$alignment,
                       I64Attr:$elements, I64Attr:$storage_bits,
                       ArrayAttr:$field_names, ArrayAttr:$field_types,
                       ArrayAttr:$field_shapes, ArrayAttr:$field_layouts,
                       DenseI64ArrayAttr:$field_bit_offsets,
                       DenseI64ArrayAttr:$field_storage_bits,
                       DenseI64ArrayAttr:$padding);
  let hasVerifier = 1;
}

def WeftRISCV_DerivedEncodingOp
    : WeftRISCV_Op<"derived_encoding", [Symbol, Pure]> {
  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$source_family,
                       StrAttr:$result_family, StrAttr:$layout_identity,
                       ArrayAttr:$parameter_names,
                       DenseI64ArrayAttr:$parameter_values,
                       I64Attr:$interleave_rows);
  let hasVerifier = 1;
}

def WeftRISCV_ArtifactPackOp
    : WeftRISCV_Op<"artifact_pack", [Symbol, IsolatedFromAbove]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       StrAttr:$source_family, StrAttr:$result_family,
                       StrAttr:$layout_identity, I64Attr:$record_bytes,
                       I64Attr:$elements, I64Attr:$interleave_rows);
  let regions = (region SizedRegion<1>:$size_body,
                        SizedRegion<1>:$pack_body);
  let hasVerifier = 1;
}

def WeftRISCV_ArtifactSizeYieldOp
    : WeftRISCV_Op<"artifact_size_yield", [Terminator]> {
  let arguments = (ins Index:$value);
  let hasVerifier = 1;
}

def WeftRISCV_ArtifactReturnOp
    : WeftRISCV_Op<"artifact_return", [Terminator]> {
  let hasVerifier = 1;
}

def WeftRISCV_StorageLoadOp
    : WeftRISCV_Op<"storage_load", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins WeftRISCV_MemDescType:$source, Index:$byte_index,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyInteger:$result);
  let hasVerifier = 1;
}

def WeftRISCV_StorageStoreOp
    : WeftRISCV_Op<"storage_store", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins AnyInteger:$value, WeftRISCV_MemDescType:$destination,
                       Index:$byte_index, WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_KernelOp
    : WeftRISCV_Op<"kernel", [Symbol, IsolatedFromAbove, SingleBlock]> {
  let arguments = (ins SymbolNameAttr:$sym_name, ArrayAttr:$arg_names,
                       ArrayAttr:$arg_access,
                       DenseI64ArrayAttr:$arg_alias_sets,
                       ArrayAttr:$shape_symbols, StrAttr:$source,
                       WeftRISCV_TargetAttr:$target,
                       ArrayAttr:$parameter_names,
                       DenseI64ArrayAttr:$parameter_values,
                       I64Attr:$vector_register_peak,
                       I64Attr:$fragment_register_peak,
                       I64Attr:$local_storage_bytes);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftRISCV_ReturnOp : WeftRISCV_Op<"return", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

def WeftRISCV_RootDomainOp : WeftRISCV_Op<"root_domain", [Pure]> {
  let results = (outs WeftRISCV_DomainType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_SymbolOp : WeftRISCV_Op<"symbol", [Pure]> {
  let arguments = (ins StrAttr:$name);
  let results = (outs Index:$result);
  let hasVerifier = 1;
}

def WeftRISCV_DomainOp : WeftRISCV_Op<"domain", [Pure]> {
  let arguments = (ins WeftRISCV_DomainType:$parent, Index:$extent,
                       Index:$partition, Index:$multiplicity);
  let results = (outs WeftRISCV_DomainType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LoopOp : WeftRISCV_Op<"loop", [RecursiveMemoryEffects]> {
  let arguments = (ins WeftRISCV_DomainType:$domain,
                       WeftRISCV_PointType:$parent_point,
                       Variadic<AnyType>:$carried,
                       I64Attr:$state_birth_count,
                       I64Attr:$staged_birth_count,
                       I64Attr:$handoff_count,
                       WeftRISCV_ScheduleAttr:$schedule);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

def WeftRISCV_YieldOp : WeftRISCV_Op<"yield", [Terminator]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let hasVerifier = 1;
}

def WeftRISCV_RootPointOp : WeftRISCV_Op<"root_point", [Pure]> {
  let arguments = (ins WeftRISCV_DomainType:$domain);
  let results = (outs WeftRISCV_PointType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_PhysicalPointOp : WeftRISCV_Op<"physical_point", [Pure]> {
  let arguments = (ins WeftRISCV_PointType:$parent, Index:$base,
                       Index:$active, Index:$partition);
  let results = (outs WeftRISCV_PointType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_ConstantOp : WeftRISCV_Op<"constant", [Pure]> {
  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IotaOp : WeftRISCV_Op<"iota", [Pure]> {
  let arguments = (ins I64Attr:$start, I64Attr:$end, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_NewOp : WeftRISCV_Op<"new", [Pure]> {
  let arguments = (ins Optional<AnyType>:$initial, BoolAttr:$initialized,
                       StrAttr:$placement, I64Attr:$owner_domain_id,
                       I64Attr:$birth_id, I64Attr:$lifetime_end_domain_id);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_MaterializeOp : WeftRISCV_Op<"materialize", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$placement,
                       StrAttr:$schema, I64Attr:$owner_domain_id,
                       I64Attr:$birth_id, I64Attr:$lifetime_end_domain_id);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_StagedViewOp : WeftRISCV_Op<"staged_view", [Pure]> {
  let arguments = (ins WeftRISCV_MemDescType:$source, StrAttr:$schema,
                       I64Attr:$owner_domain_id, I64Attr:$birth_id,
                       I64Attr:$lifetime_end_domain_id);
  let results = (outs WeftRISCV_MemDescType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_MemoryViewOp
    : WeftRISCV_Op<"memory_view", [Pure, AttrSizedOperandSegments]> {
  let arguments = (ins WeftRISCV_MemDescType:$base,
                       Variadic<Index>:$extents,
                       Variadic<Index>:$strides,
                       Variadic<Index>:$origins);
  let results = (outs WeftRISCV_MemDescType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_SliceOp : WeftRISCV_Op<"slice", [Pure]> {
  let arguments = (ins WeftRISCV_MemDescType:$base,
                       Variadic<AnyType>:$indices, ArrayAttr:$selectors);
  let results = (outs WeftRISCV_MemDescType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LoadOp : WeftRISCV_Op<"load", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins WeftRISCV_MemDescType:$region,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_StoreOp : WeftRISCV_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins AnyType:$value, WeftRISCV_MemDescType:$region,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_FieldOp : WeftRISCV_Op<"field", [Pure]> {
  let arguments = (ins AnyType:$owner, StrAttr:$name,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_ExtractOp : WeftRISCV_Op<"extract", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       Variadic<AnyType>:$indices, ArrayAttr:$selectors,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_UpdateOp : WeftRISCV_Op<"update", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input, AnyType:$value,
                       Variadic<AnyType>:$indices, ArrayAttr:$selectors,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_UnaryOp : WeftRISCV_Op<"unary", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$kind,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_BinaryOp : WeftRISCV_Op<"binary", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$kind,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_CompareOp : WeftRISCV_Op<"compare", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, StrAttr:$predicate,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

class WeftRISCV_ConversionOp<string mnemonic>
    : WeftRISCV_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyType:$input, WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_CastOp : WeftRISCV_ConversionOp<"cast">;

def WeftRISCV_NarrowOp : WeftRISCV_Op<"narrow", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$rounding, BoolAttr:$saturate,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_WidenOp : WeftRISCV_ConversionOp<"widen">;

def WeftRISCV_MacGroupsOp : WeftRISCV_Op<"mac_groups", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, I64Attr:$group,
                       StrAttr:$overflow, WeftRISCV_ScheduleAttr:$schedule,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_ReduceOp : WeftRISCV_Op<"reduce", [Pure]> {
  let arguments = (ins AnyType:$input, StrAttr:$kind, I64Attr:$axis,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_Fold2Op : WeftRISCV_Op<"fold2", [Pure]> {
  let arguments = (ins AnyType:$input, WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

class WeftRISCV_ContractLikeOp<string mnemonic>
    : WeftRISCV_Op<mnemonic, [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs,
                       DenseI64ArrayAttr:$over,
                       OptionalAttr<TypeAttr>:$acc_type,
                       StrAttr:$lane_operand,
                       StrAttr:$lane_memory_form,
                       WeftRISCV_ScheduleAttr:$schedule,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_DotOp : WeftRISCV_ContractLikeOp<"dot">;
def WeftRISCV_ContractOp : WeftRISCV_ContractLikeOp<"contract">;
def WeftRISCV_OuterContractOp
    : WeftRISCV_ContractLikeOp<"outer_contract">;

def WeftRISCV_LookupOp : WeftRISCV_Op<"lookup", [Pure]> {
  let arguments = (ins AnyType:$table, AnyType:$indices, StrAttr:$bounds,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_ConvertLayoutOp : WeftRISCV_Op<"convert_layout"> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       WeftRISCV_ConversionAttr:$conversion,
                       OptionalAttr<WeftRISCV_AccessAttr>:$source_access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LocalAllocOp
    : WeftRISCV_Op<"local_alloc", [MemoryEffects<[MemAlloc]>]> {
  let arguments = (ins Index:$size_bytes);
  let results = (outs WeftRISCV_LocalType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LocalCapacityGuardOp : WeftRISCV_Op<"local_capacity_guard"> {
  let arguments = (ins Index:$size_bytes, I64Attr:$limit_bytes,
                       I64Attr:$birth_id, WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_LocalBindOp : WeftRISCV_Op<"local_bind", [Pure]> {
  let arguments = (ins WeftRISCV_LocalType:$storage, Index:$elements);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LocalLoadOp
    : WeftRISCV_Op<"local_load", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins WeftRISCV_ValueType:$source, Index:$index,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_LocalStoreOp
    : WeftRISCV_Op<"local_store", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins AnyType:$value, WeftRISCV_ValueType:$destination,
                       Index:$index,
                       WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_SpillOp
    : WeftRISCV_Op<"spill", [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins AnyType:$input, WeftRISCV_LocalType:$slot,
                       WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_ReloadOp
    : WeftRISCV_Op<"reload", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins WeftRISCV_LocalType:$slot, WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IMEFragmentMMAOp
    : WeftRISCV_Op<"ime_fragment_mma", [Pure]> {
  let arguments = (ins WeftRISCV_FragmentType:$lhs,
                       WeftRISCV_FragmentType:$rhs,
                       I64Attr:$groups, I64Attr:$chunks,
                       ArrayAttr:$asm_clobbers, BoolAttr:$volatile_asm,
                       BoolAttr:$memory_clobber,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_FragmentType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IMEPackOp
    : WeftRISCV_Op<"ime_pack", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins WeftRISCV_ValueType:$input, StrAttr:$role,
                       WeftRISCV_FragmentPackingAttr:$packing,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_FragmentType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_IMEUnpackOp : WeftRISCV_Op<"ime_unpack", [Pure]> {
  let arguments = (ins WeftRISCV_FragmentType:$input,
                       WeftRISCV_ConversionAttr:$conversion,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RegisterMaterializeOp
    : WeftRISCV_Op<"register_materialize", [Pure]> {
  let arguments = (ins AnyType:$input, I64Attr:$owner_domain_id,
                       I64Attr:$birth_id, I64Attr:$lifetime_end_domain_id,
                       StrAttr:$realization);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVBitplaneMergeOp
    : WeftRISCV_Op<"rvv_bitplane_merge", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins WeftRISCV_ValueType:$low,
                       WeftRISCV_ValueType:$plane,
                       I64Attr:$insert_bit,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVGroupedMacReduceOp
    : WeftRISCV_Op<"rvv_grouped_mac_reduce", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Index:$active_terms,
                       I64Attr:$group, I64Attr:$unroll,
                       I64Attr:$reduction_axis,
                       WeftRISCV_LayoutAttr:$partial_layout,
                       WeftRISCV_AccessAttr:$lhs_access,
                       WeftRISCV_AccessAttr:$rhs_access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVGroupedMacLoadOp
    : WeftRISCV_Op<"rvv_grouped_mac_load", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Index:$group_index,
                       Index:$active_terms, I64Attr:$group,
                       I64Attr:$unroll, I64Attr:$reduction_axis,
                       StrAttr:$window_form,
                       WeftRISCV_LayoutAttr:$partial_layout,
                       WeftRISCV_AccessAttr:$lhs_access,
                       WeftRISCV_AccessAttr:$rhs_access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_WindowType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVGroupedMacStepOp
    : WeftRISCV_Op<"rvv_grouped_mac_step", [Pure]> {
  let arguments = (ins WeftRISCV_WindowType:$window,
                       WeftRISCV_ValueType:$accumulator,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVEncodedDotLoadOp
    : WeftRISCV_Op<"rvv_encoded_dot_load", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs, Index:$group_index,
                       Index:$group_count, I64Attr:$unroll,
                       I64Attr:$rhs_pairs_per_group,
                       I64Attr:$reduction_axis,
                       WeftRISCV_AccessAttr:$lhs_access,
                       WeftRISCV_AccessAttr:$rhs_access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_WindowType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVEncodedDotStepOp
    : WeftRISCV_Op<"rvv_encoded_dot_step", [Pure]> {
  let arguments = (ins WeftRISCV_WindowType:$window,
                       WeftRISCV_ValueType:$accumulator,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVWidenDotOp : WeftRISCV_Op<"rvv_widen_dot", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$lhs,
                       WeftRISCV_ValueType:$rhs,
                       DenseI64ArrayAttr:$over,
                       StrAttr:$stream_reduction,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVWidenReduceOp
    : WeftRISCV_Op<"rvv_widen_reduce", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       StrAttr:$kind,
                       I64Attr:$axis,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVPartitionedWidenReduceStoreOp
    : WeftRISCV_Op<"rvv_partitioned_widen_reduce_store",
                   [MemoryEffects<[MemWrite]>]> {
  let arguments = (ins WeftRISCV_ValueType:$input,
                       WeftRISCV_MemDescType:$destination,
                       Index:$base_index,
                       I64Attr:$partition,
                       I64Attr:$count,
                       I64Attr:$destination_axis,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let hasVerifier = 1;
}

def WeftRISCV_RVVLayeredWindowOp
    : WeftRISCV_Op<"rvv_layered_window", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins WeftRISCV_ValueType:$field,
                       WeftRISCV_PointType:$point,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$first,
                      WeftRISCV_ValueType:$second);
  let hasVerifier = 1;
}

def WeftRISCV_RVVStreamReduceOp
    : WeftRISCV_Op<"rvv_stream_reduce", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins WeftRISCV_MemDescType:$source,
                       WeftRISCV_LayoutAttr:$input_layout,
                       ArrayAttr:$kinds,
                       I64Attr:$axis,
                       WeftRISCV_AccessAttr:$access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
}

def WeftRISCV_RVVStreamDotOp
    : WeftRISCV_Op<"rvv_stream_dot", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins WeftRISCV_MemDescType:$lhs,
                       WeftRISCV_MemDescType:$rhs,
                       Index:$extent,
                       WeftRISCV_LayoutAttr:$input_layout,
                       I64Attr:$axis,
                       WeftRISCV_AccessAttr:$lhs_access,
                       WeftRISCV_AccessAttr:$rhs_access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVStreamContractOp
    : WeftRISCV_Op<"rvv_stream_contract", [MemoryEffects<[MemRead]>]> {
  let arguments = (ins WeftRISCV_MemDescType:$lhs,
                       WeftRISCV_MemDescType:$rhs,
                       Index:$extent,
                       WeftRISCV_LayoutAttr:$operand_layout,
                       WeftRISCV_LayoutAttr:$accumulator_layout,
                       I64Attr:$reduction_axis,
                       StrAttr:$stationary_operand,
                       I64Attr:$unroll,
                       WeftRISCV_AccessAttr:$lhs_access,
                       WeftRISCV_AccessAttr:$rhs_access,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVSplatOp : WeftRISCV_Op<"rvv_splat", [Pure]> {
  let arguments = (ins AnyType:$scalar, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_ProjectReductionOperandOp
    : WeftRISCV_Op<"project_reduction_operand", [Pure]> {
  let arguments = (ins WeftRISCV_ValueType:$input, Index:$reduction_index,
                       I64Attr:$reduction_axis, WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVContractStepOp : WeftRISCV_Op<"rvv_contract_step", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs,
                       WeftRISCV_ValueType:$accumulator, Index:$reduction_index,
                       I64Attr:$reduction_axis, StrAttr:$lane_operand,
                       StrAttr:$lane_memory_form,
                       WeftRISCV_LayoutAttr:$lane_load_layout,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

def WeftRISCV_RVVEncodedContractStepOp
    : WeftRISCV_Op<"rvv_encoded_contract_step", [Pure]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs,
                       WeftRISCV_ValueType:$accumulator, Index:$reduction_index,
                       I64Attr:$reduction_axis, StrAttr:$lane_operand,
                       WeftRISCV_AccessAttr:$lhs_access,
                       WeftRISCV_AccessAttr:$rhs_access,
                       WeftRISCV_LayoutAttr:$lane_load_layout,
                       WeftRISCV_LeafAttr:$leaf);
  let results = (outs WeftRISCV_ValueType:$result);
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_RISCV_IR_RISCVOPS_TD
