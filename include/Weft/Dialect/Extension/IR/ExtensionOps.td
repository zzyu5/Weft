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

def WeftExtension_AffineI4I8ContractOp
    : WeftExtension_Op<"affine_i4_i8_contract", [Pure]> {
  let summary = "Affine packed-i4 by signed-i8 block contraction";
  let description = [{
    Computes sixteen independent affine dot products over a thirty-two-element
    signed-i8 activation block.  packed_weight[n, p] contains two unsigned-i4
    values.  For p = half * 8 + lane, its low nibble multiplies activation
    [half * 16 + lane] and its high nibble multiplies activation
    [half * 16 + lane + 8].  Both nibbles are biased by
    weight_zero_point[n].  The integer dot is scaled by activation_scale and
    weight_scale[n], then added to init[n].

    This operation defines only the local numerical relation.  Persistent
    pointer layout and outer M/N/K traversal remain explicit in Kernel IR;
    target fragments, registers, and instruction spelling are not represented
    here.
  }];
  let arguments = (ins AnyType:$activation, AnyType:$packed_weight,
                       AnyType:$activation_scale, AnyType:$weight_scale,
                       AnyType:$weight_zero_point, AnyType:$init);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

def WeftExtension_SymmetricI4I8ContractOp
    : WeftExtension_Op<"symmetric_i4_i8_contract", [Pure]> {
  let summary = "Symmetric packed-i4 by signed-i8 block contraction";
  let description = [{
    Computes sixteen independent dot products over a thirty-two-element
    signed-i8 activation block.  packed_weight[n, p] contains two unsigned-i4
    values whose numerical values are biased by eight.  For
    p = half * 8 + lane, its low nibble multiplies activation
    [half * 16 + lane] and its high nibble multiplies activation
    [half * 16 + lane + 8].  The integer dot is scaled by activation_scale and
    weight_scale[n], then added to init[n].

    This operation defines only the local numerical relation.  Persistent
    pointer layout and outer M/N/K traversal remain explicit in Kernel IR;
    target fragments, registers, and instruction spelling are not represented
    here.
  }];
  let arguments = (ins AnyType:$activation, AnyType:$packed_weight,
                       AnyType:$activation_scale, AnyType:$weight_scale,
                       AnyType:$init);
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
