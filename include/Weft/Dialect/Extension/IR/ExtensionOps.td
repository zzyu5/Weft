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

def WeftExtension_BlockScaledContractOp
    : WeftExtension_Op<"block_scaled_contract", [Pure]> {
  let summary = "Block-scaled contraction with explicit rounding and saturation";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs,
                       AnyType:$scale_a, AnyType:$scale_b, AnyType:$init,
                       StrAttr:$rounding, BoolAttr:$saturation);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
}

#endif // WEFT_DIALECT_EXTENSION_IR_EXTENSIONOPS_TD
