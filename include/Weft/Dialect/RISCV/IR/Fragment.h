#ifndef WEFT_DIALECT_RISCV_IR_FRAGMENT_H
#define WEFT_DIALECT_RISCV_IR_FRAGMENT_H

#include "Weft/Dialect/RISCV/IR/RISCVDialect.h"
#include "Weft/Target/RISCVFragment.h"

namespace weft::riscv {
FragmentCapabilityAttr fragmentCapabilityAttr(
    mlir::MLIRContext *context, const RISCVFragmentCapability &capability);
bool fragmentAccumulatorTypeMatches(FragmentCapabilityAttr capability,
                                     mlir::Type element);
int64_t fragmentMMAAdditionalGroups(FragmentCapabilityAttr capability);
} // namespace weft::riscv

#endif
