#include "RISCVIntrinsicC.h"

namespace weft::riscv_internal {

bool SelectedLocalImplementations::contains(LocalPrimitiveKind primitive) const {
  for (const LocalImplementation &implementation : implementations)
    if (implementation.primitive == primitive)
      return true;
  return false;
}

bool SelectedLocalImplementations::containsSymbol(
    const std::string &symbol) const {
  for (const LocalImplementation &implementation : implementations)
    if (implementation.helperSymbol == symbol)
      return true;
  return false;
}

} // namespace weft::riscv_internal
