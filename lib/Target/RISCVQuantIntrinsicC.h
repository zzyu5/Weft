#ifndef WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H
#define WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H

namespace llvm {
class raw_ostream;
}

namespace weft::riscv_internal {

void emitIQ2IntrinsicCLeaves(llvm::raw_ostream &output, bool fixedLanes32,
                             bool fixedLanes64, bool scalable);
void emitIQ3IntrinsicCLeaves(llvm::raw_ostream &output, bool fixedLanes64,
                             bool scalable);
void emitIQ1IntrinsicCLeaves(llvm::raw_ostream &output, bool fixedLanes32,
                             bool fixedLanes64, bool scalable);
void emitQ6IntrinsicCLeaves(llvm::raw_ostream &output, bool fixedLanes32,
                            bool fixedLanes64, bool scalable);
void emitPackedI4IntrinsicCLeaves(llvm::raw_ostream &output, bool vlen128,
                                  bool vlen256);
void emitPackedI5IntrinsicCLeaves(llvm::raw_ostream &output, bool vlen128,
                                  bool vlen256);
void emitTernaryIntrinsicCLeaves(llvm::raw_ostream &output,
                                 bool base3VLEN128, bool base3VLEN256,
                                 bool packedI2VLEN128,
                                 bool packedI2VLEN256);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H
