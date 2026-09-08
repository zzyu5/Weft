#include "Verification.h"

namespace weft::riscv::detail {

bool exactLeaf(LeafAttr leaf, llvm::StringRef engine,
               llvm::StringRef family, llvm::StringRef instruction,
               llvm::StringRef mask, llvm::StringRef tail) {
  return leaf && leaf.getEngine() == engine && leaf.getFamily() == family &&
         leaf.getInstruction() == instruction &&
         leaf.getSpelling() == instruction && leaf.getMask() == mask &&
         leaf.getTail() == tail;
}

unsigned elementBitWidth(mlir::Type type) {
  if (!type)
    return 0;
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(type))
    return integer.getWidth();
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(type))
    return floating.getWidth();
  return 0;
}

bool integerSignednessMatches(mlir::Type type, llvm::StringRef signedness) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
  return integer && !integer.isSignless() &&
         ((signedness == "signed" && integer.isSigned()) ||
          (signedness == "unsigned" && integer.isUnsigned()));
}

} // namespace weft::riscv::detail

