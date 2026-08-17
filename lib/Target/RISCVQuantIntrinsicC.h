#ifndef WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H
#define WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H

#include "RISCVPhysicalPlanning.h"

namespace llvm {
class raw_ostream;
}

namespace weft::riscv_internal {

void emitIQ2LocalImplementations(llvm::raw_ostream &output, bool registerL32,
                                 bool registerL64, bool strip);
void emitIQ3LocalImplementations(llvm::raw_ostream &output, bool registerL64,
                                 bool strip);
void emitIQ1LocalImplementations(llvm::raw_ostream &output, bool registerL32,
                                 bool registerL64, bool strip);
void emitQ6LocalImplementations(llvm::raw_ostream &output, bool registerL32,
                                bool registerL64, bool strip);
bool emitPackedI4LocalImplementation(llvm::raw_ostream &output,
                                     const LocalImplementation &implementation);
bool emitPackedI5LocalImplementation(llvm::raw_ostream &output,
                                     const LocalImplementation &implementation);
void emitPackedI3GroupedLocalImplementations(llvm::raw_ostream &output,
                                             bool registerL32,
                                             bool registerL64);
bool emitNibbleCodebookLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation);
bool emitBase3TernaryLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation);
bool emitPackedI2TernaryLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation);
void emitSignedCodebookLocalImplementations(
    llvm::raw_ostream &output, bool entry8E8M2, bool entry8E8M1,
    bool entry4E8M2, bool entry4E8M1);
void emitPackedU9U7CodebookLocalImplementations(llvm::raw_ostream &output,
                                                bool registerE8M2,
                                                bool registerE8M1);
void emitPackedU11GridDeltaLocalImplementations(llvm::raw_ostream &output,
                                                bool registerE8M2,
                                                bool registerE8M1);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H
