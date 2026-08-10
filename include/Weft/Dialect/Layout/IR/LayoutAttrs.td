#ifndef WEFT_DIALECT_LAYOUT_IR_LAYOUTATTRS_TD
#define WEFT_DIALECT_LAYOUT_IR_LAYOUTATTRS_TD

include "mlir/IR/AttrTypeBase.td"

def WEFT_Layout_Dialect : Dialect {
  let name = "weft_layout";
  let cppNamespace = "::weft::layout";
  let summary = "Composable logical-to-SIMD layouts for Weft blocked values";
  let description = [{
    The layout dialect records only owner-independent physical mapping choices.
    Logical extents and algorithm roles stay in weft_kernel; RVV LMUL/VL and IME
    fragment choices stay in their owner-local execution dialects.
  }];
  let useDefaultAttributePrinterParser = 1;
}

class WEFTLayout_Attr<string name, string attrMnemonic,
                      list<Trait> traits = []>
    : AttrDef<WEFT_Layout_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def WEFTLayout_BlockedLayoutAttr
    : WEFTLayout_Attr<"BlockedLayout", "blocked"> {
  let summary = "ordered logical axes assigned to a generic SIMD lane domain";
  let description = [{
    `order` lists logical axes from fastest to slowest traversal. `vector_axes`
    is the ordered subset distributed over a SIMD lane domain; remaining axes
    execute serially or are mapped by an owner-local layout. Shape, dynamic
    extents, reductions and broadcasts are deliberately not duplicated here.
  }];
  let parameters = (ins
    "int64_t":$rank,
    ArrayRefParameter<"int64_t">:$order,
    OptionalArrayRefParameter<"int64_t">:$vectorAxes
  );
  let assemblyFormat =
    "`<` `rank` `=` $rank `,` `order` `=` `[` $order `]` `,` "
    "`vector_axes` `=` `[` (`]`) : ($vectorAxes^ `]`)? `>`";
  let genVerifyDecl = 1;
}

#endif // WEFT_DIALECT_LAYOUT_IR_LAYOUTATTRS_TD
