#ifndef WEFT_DIALECT_EXTENSION_IR_EXTENSIONOPS_TD
#define WEFT_DIALECT_EXTENSION_IR_EXTENSIONOPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def WEFT_Extension_Dialect : Dialect {
  let name = "weft_ext";
  let cppNamespace = "::weft::extension";
  let summary = "Typed local semantic extensions for canonical Weft kernels";
}

class WeftExtension_Op<string mnemonic, list<Trait> traits = []>
    : Op<WEFT_Extension_Dialect, mnemonic, traits>;

def WeftExtension_LoadF16LEOp
    : WeftExtension_Op<"load_f16_le", [MemoryEffects<[MemRead]>]> {
  let summary = "Load one little-endian f16 value and expose it as f32";
  let description = [{
    Reads one scalar f16 value from a byte pointer.  The little-endian byte
    interpretation is observable semantics.  Storage ownership, persistent
    format, surrounding traversal, and target realization remain outside this
    operation.
  }];
  let arguments = (ins AnyType:$base);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_AffineI4I8DotOp
    : WeftExtension_Op<"affine_i4_i8_dot", [MemoryEffects<[MemRead]>]> {
  let summary = "Affine packed-i4 by signed-i8 block dot";
  let description = [{
    Computes either one or four rows of sixteen independent affine dot products.
    The one-row form accepts a signed-i8 block<32>, scalar activation scale, and
    f32 block<16> init/result.  The four-row form accepts a fragment-interleaved
    signed-i8 block<128>, f32 block<4> activation scales, and f32 block<4,16>
    init/result.  packed_base addresses one local 304-byte N16 by K32 field:
    sixteen little-endian f16 scales start at byte 0, sixteen u8
    zero points start at byte 32, and packed codes start at byte 48.  For
    p = half * 8 + lane, the byte at 48 + half * 128 + n * 8 + lane supplies
    low/high unsigned-i4 values for activation positions half * 16 + lane and
    half * 16 + lane + 8.  Both codes are biased by the zero point.  The
    integer dot is scaled by activation_scale and the local f16 scale, then
    added to init[n].

    This operation defines only the local numerical relation.  Persistent
    pointer layout and outer M/N/K traversal remain explicit in Kernel IR;
    target fragments, registers, and instruction spelling are not represented
    here.
  }];
  let arguments = (ins AnyType:$activation, AnyType:$packed_base,
                       AnyType:$activation_scale, AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_SymmetricI4I8DotOp
    : WeftExtension_Op<"symmetric_i4_i8_dot", [MemoryEffects<[MemRead]>]> {
  let summary = "Symmetric packed-i4 by signed-i8 block dot";
  let description = [{
    Computes either one or four rows of sixteen independent dot products.  The
    one-row form accepts a signed-i8 block<32>, scalar activation scale, and f32
    block<16> init/result.  The four-row form accepts a fragment-interleaved
    signed-i8 block<128>, f32 block<4> activation scales, and f32 block<4,16>
    init/result.  packed_base addresses one local 288-byte N16 by K32 field:
    sixteen little-endian f16 scales start at byte 0 and packed
    codes start at byte 32.  For p = half * 8 + lane, the byte at
    32 + half * 128 + n * 8 + lane supplies low/high unsigned-i4 values for
    activation positions half * 16 + lane and half * 16 + lane + 8; each code
    is biased by eight.  The integer dot is scaled by activation_scale and the
    local f16 scale, then added to init[n].

    This operation defines only the local numerical relation.  Persistent
    pointer layout and outer M/N/K traversal remain explicit in Kernel IR;
    target fragments, registers, and instruction spelling are not represented
    here.
  }];
  let arguments = (ins AnyType:$activation, AnyType:$packed_base,
                       AnyType:$activation_scale, AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_GroupedAffineI4I8DotOp
    : WeftExtension_Op<"grouped_affine_i4_i8_dot", [Pure]> {
  let summary = "Grouped affine packed-i4 by signed-i8 block dot";
  let description = [{
    Computes one 256-element packed-i4 by signed-i8 dot.  The packed weight
    contains four 32-byte low/high-nibble pairs.  Eight logical 32-element
    groups use the explicitly supplied twelve-byte 6-bit scale/minimum table.
    Sixteen little-endian signed-i16 activation sums supply the affine minimum
    correction.  dot_scale and minimum_scale are explicit numerical operands;
    the result is added to init.

    This is a local semantic primitive.  Block pointers, persistent 144/292
    byte storage, outer reduction traversal, and the public ABI remain explicit
    in Kernel IR.  VLEN, LMUL, unroll, register organization, and instruction
    spelling belong only to target lowering.
  }];
  let arguments = (ins AnyType:$packed_weight, AnyType:$scale_min,
                       AnyType:$activation, AnyType:$activation_sum_bytes,
                       AnyType:$dot_scale, AnyType:$minimum_scale,
                       AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_SignBitI8DotOp
    : WeftExtension_Op<"sign_bit_i8_dot", [Pure]> {
  let summary = "Little-endian sign-bit by signed-i8 local dot";
  let description = [{
    Computes one thirty-two-element dot.  sign_bits contains four bytes; bit p
    is `(sign_bits[p / 8] >> (p % 8)) & 1` and contributes +activation[p]
    when set or -activation[p] when clear.  The integer sum is multiplied by
    activation_scale and sign_scale, then added to init.

    This operation owns only the observable local bit relation and arithmetic.
    Persistent block layout, the relation between multiple sub-blocks, outer
    row/reduction traversal, and all pointer arithmetic remain explicit in
    Kernel IR.  VLEN, LMUL, widening, and instruction spelling are target
    decisions.
  }];
  let arguments = (ins AnyType:$sign_bits, AnyType:$activation,
                       AnyType:$activation_scale, AnyType:$sign_scale,
                       AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_E2M1E8M0I8DotOp
    : WeftExtension_Op<"e2m1_e8m0_i8_dot", [Pure]> {
  let summary = "Packed E2M1/E8M0 by signed-i8 local dot";
  let description = [{
    Computes one thirty-two-element dot.  packed_codes contains sixteen bytes;
    its low nibbles encode elements 0..15 and its high nibbles encode elements
    16..31 using the E2M1 values
    [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6].
    exponent is an E8M0 code whose scale is 2^(exponent-127), with code zero
    represented by the f32 bit pattern 0x00400000.  The decoded dot is
    multiplied by activation_scale and added to init.

    The exponent byte, packed-code bytes, activation block, and scales are
    explicit operands.  Persistent storage, outer traversal, target lookup
    organization, and instruction spelling are not part of this operation.
  }];
  let arguments = (ins AnyType:$packed_codes, AnyType:$exponent,
                       AnyType:$activation, AnyType:$activation_scale,
                       AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_PackedI4I8DotOp
    : WeftExtension_Op<"packed_i4_i8_dot", [Pure]> {
  let summary = "Packed unsigned-i4 by signed-i8 local dot";
  let description = [{
    Computes one thirty-two-element dot.  packed_codes contains sixteen bytes;
    low nibbles encode elements 0..15 and high nibbles encode elements 16..31.
    zero_point is subtracted from every decoded code.  The integer dot is
    multiplied by dot_scale, additive_bias is added once, and the result is
    accumulated into init.

    Enclosing block storage, outer traversal, vector shape, reduction
    organization, and instruction spelling remain outside this local semantic
    primitive.
  }];
  let arguments = (ins AnyType:$packed_codes, AnyType:$activation,
                       AnyType:$zero_point, AnyType:$dot_scale,
                       AnyType:$additive_bias, AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_PackedI5I8DotOp
    : WeftExtension_Op<"packed_i5_i8_dot", [Pure]> {
  let summary = "Packed unsigned-i5 by signed-i8 local dot";
  let description = [{
    Computes one thirty-two-element dot.  low_bits contains sixteen bytes whose
    low nibbles encode elements 0..15 and high nibbles encode elements 16..31.
    high_bits contains one little-endian bit for each element and contributes
    bit four of the unsigned five-bit code.  zero_point is subtracted from each
    decoded code.  The integer dot is multiplied by dot_scale, additive_bias is
    added once, and the result is accumulated into init.

    The packed fields, activation block, affine numerical operands, and init are
    explicit.  Their enclosing block storage, outer traversal, target vector
    shape, reduction organization, and instruction spelling remain outside this
    local semantic primitive.
  }];
  let arguments = (ins AnyType:$low_bits, AnyType:$high_bits,
                       AnyType:$activation, AnyType:$zero_point,
                       AnyType:$dot_scale, AnyType:$additive_bias,
                       AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_Base3TernaryI8DotOp
    : WeftExtension_Op<"base3_ternary_i8_dot", [Pure]> {
  let summary = "Packed base-3 ternary by signed-i8 local dot";
  let description = [{
    Computes one 256-element ternary by signed-i8 dot.  codes contains a
    thirty-two-byte five-digit base-3 field followed by a sixteen-byte
    five-digit field.  high_digits contains four bytes that define the final
    sixteen values through four base-3 digits.  activation, both scales, and
    init are explicit operands.

    The 48/4-byte decode relation is local numerical semantics.  Its enclosing
    packed block, Q8 block, outer traversal, target vector shape, reduction
    organization, and instruction spelling remain outside this operation.
  }];
  let arguments = (ins AnyType:$codes, AnyType:$high_digits,
                       AnyType:$activation, AnyType:$weight_scale,
                       AnyType:$activation_scale, AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_PackedI2TernaryI8DotOp
    : WeftExtension_Op<"packed_i2_ternary_i8_dot", [Pure]> {
  let summary = "Packed two-bit ternary by signed-i8 local dot";
  let description = [{
    Computes one 256-element ternary by signed-i8 dot.  Each of sixty-four
    packed bytes contributes four two-bit fields; each field decodes as
    code - 1 and is paired with the corresponding activation element.
    Both scales and init are explicit operands.

    The local decode and dot do not own the enclosing packed block, Q8 block,
    outer traversal, target vector shape, or instruction spelling.
  }];
  let arguments = (ins AnyType:$codes, AnyType:$activation,
                       AnyType:$weight_scale, AnyType:$activation_scale,
                       AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_SignedCodebookI8DotOp
    : WeftExtension_Op<"signed_codebook_i8_dot", [Pure]> {
  let summary = "Signed codebook entry by signed-i8 local dot";
  let description = [{
    Computes one thirty-two-element codebook by signed-i8 dot.  Four code bytes
    select four eight-element entries, or eight code bytes select eight
    four-element entries.  sign_metadata contains four seven-bit indices into a
    caller-provided table of sign masks.  The selected grid values receive those
    signs, are dotted with activation, multiplied by dot_scale, and accumulated
    into init.

    The two tables, code bytes, metadata, activation, scale, and init are explicit
    operands.  Packed block layout, local-scale extraction, outer traversal,
    target gather shape, and instruction spelling remain outside this operation.
  }];
  let arguments = (ins AnyType:$codes, AnyType:$sign_metadata,
                       AnyType:$activation, AnyType:$grid_table,
                       AnyType:$sign_table, AnyType:$dot_scale,
                       AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_PackedU9U7CodebookI8DotOp
    : WeftExtension_Op<"packed_u9_u7_codebook_i8_dot", [Pure]> {
  let summary = "Packed u9/u7 codebook entry by signed-i8 local dot";
  let description = [{
    Computes one thirty-two-element signed codebook dot from four little-endian
    sixteen-bit code words.  Each word uses its low nine bits as an index into
    an eight-element grid table and its high seven bits as an index into a sign
    mask table.  The low and high nibbles of scale_byte provide the odd integer
    scales for the first and second sixteen-element halves.  The scaled integer
    dot is multiplied by dot_scale and accumulated into init.

    Packed block layout outside the eight local code bytes, outer traversal,
    target gather shape, and instruction spelling remain outside this operation.
  }];
  let arguments = (ins AnyType:$packed_codes, AnyType:$scale_byte,
                       AnyType:$activation, AnyType:$grid_table,
                       AnyType:$sign_table, AnyType:$dot_scale,
                       AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_IQ2SI8DotOp
    : WeftExtension_Op<"iq2_s_i8_dot", [Pure]> {
  let summary = "IQ2_S codebook by signed-i8 local dot";
  let description = [{
    Computes one 256-element IQ2_S by signed-i8 dot from explicit packed code,
    high-bit, sign, scale, activation, weight-scale, activation-scale, and init
    operands.  The four 8-element codebook vectors and their signs remain one
    observable local numerical relation.  Persistent block storage and outer
    traversal are not part of this operation.
  }];
  let arguments = (ins AnyType:$codes, AnyType:$high_bits,
                       AnyType:$sign_bits, AnyType:$scales,
                       AnyType:$activation, AnyType:$weight_scale,
                       AnyType:$activation_scale, AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_IQ3SI8DotOp
    : WeftExtension_Op<"iq3_s_i8_dot", [Pure]> {
  let summary = "IQ3_S codebook by signed-i8 local dot";
  let description = [{
    Computes one 256-element IQ3_S by signed-i8 dot from explicit packed code,
    high-bit, sign, scale, activation, weight-scale, activation-scale, and init
    operands.  Persistent block storage and outer traversal remain in Kernel IR.
  }];
  let arguments = (ins AnyType:$codes, AnyType:$high_bits,
                       AnyType:$sign_bits, AnyType:$scales,
                       AnyType:$activation, AnyType:$weight_scale,
                       AnyType:$activation_scale, AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_IQ1MI8DotOp
    : WeftExtension_Op<"iq1_m_i8_dot", [Pure]> {
  let summary = "IQ1_M ternary codebook by signed-i8 local dot";
  let description = [{
    Computes one 256-element IQ1_M by signed-i8 dot.  Code bytes, high/delta
    bits, packed scale bytes, activation, activation scale, and init are explicit
    operands.  The half scale reconstructed from the packed scale field is part
    of this local numerical relation.
  }];
  let arguments = (ins AnyType:$codes, AnyType:$high_delta_bits,
                       AnyType:$scales, AnyType:$activation,
                       AnyType:$activation_scale, AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_Q6KI8DotOp
    : WeftExtension_Op<"q6_k_i8_dot", [Pure]> {
  let summary = "Q6_K grouped signed-6-bit by signed-i8 local dot";
  let description = [{
    Computes one 256-element Q6_K by signed-i8 dot from explicit low-nibble,
    high-bit, signed group-scale, activation, weight-scale, activation-scale, and
    init operands.  Persistent 210/292-byte block storage and outer traversal
    remain explicit outside the operation.
  }];
  let arguments = (ins AnyType:$low_bits, AnyType:$high_bits,
                       AnyType:$group_scales, AnyType:$activation,
                       AnyType:$weight_scale, AnyType:$activation_scale,
                       AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_EXTENSION_IR_EXTENSIONOPS_TD
