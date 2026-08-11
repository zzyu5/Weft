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

#endif // WEFT_DIALECT_EXTENSION_IR_EXTENSIONOPS_TD
