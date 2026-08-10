#ifndef WEFT_DIALECT_EXECUTION_IR_SELECTEDOWNEROPINTERFACE_TD
#define WEFT_DIALECT_EXECUTION_IR_SELECTEDOWNEROPINTERFACE_TD

include "mlir/IR/OpBase.td"

def SelectedOwnerOpInterface : OpInterface<"SelectedOwnerOpInterface"> {
  let description = [{
    Marker interface for every owner-local selected-execution record. A symbol
    referenced by a shared layout group implements the same interface, while
    non-symbol records may describe final operation-local decisions. The shared
    verifier admits records and discovers group owners through this interface
    instead of depending on concrete RVV, IME, scalar, or future operation
    classes.
  }];

  let cppNamespace = "::weft::execution";
}

#endif // WEFT_DIALECT_EXECUTION_IR_SELECTEDOWNEROPINTERFACE_TD
