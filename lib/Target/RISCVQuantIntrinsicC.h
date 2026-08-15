#ifndef WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H
#define WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H

namespace llvm {
class raw_ostream;
}

namespace weft::riscv_internal {

void emitIQ2IntrinsicCLeaves(llvm::raw_ostream &output, bool fixedLanes32,
                             bool fixedLanes64, bool scalable);
void emitIQ3IntrinsicCLeaves(llvm::raw_ostream &output, bool scalable);
void emitIQ1IntrinsicCLeaves(llvm::raw_ostream &output, bool fixed,
                             bool scalable);
void emitQ6IntrinsicCLeaves(llvm::raw_ostream &output, bool fixedLanes32,
                            bool fixedLanes64, bool scalable);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H
