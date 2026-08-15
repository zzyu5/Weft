#ifndef WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H
#define WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H

#include <cstdint>

namespace llvm {
class raw_ostream;
}

namespace weft::riscv_internal {

void emitIQ2IntrinsicCLeaves(llvm::raw_ostream &output, bool fixed,
                             bool scalable, int64_t vlenBits);
void emitIQ3IntrinsicCLeaves(llvm::raw_ostream &output, bool scalable);
void emitIQ1IntrinsicCLeaves(llvm::raw_ostream &output, bool fixed,
                             bool scalable);
void emitQ6IntrinsicCLeaves(llvm::raw_ostream &output, bool fixed,
                            bool scalable, int64_t vlenBits);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H
