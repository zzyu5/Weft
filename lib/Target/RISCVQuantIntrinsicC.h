#ifndef WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H
#define WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H

#include "RISCVPhysicalPlanning.h"

namespace llvm {
class raw_ostream;
}

namespace weft::riscv_internal {

bool emitIQ2LocalImplementation(llvm::raw_ostream &output,
                                const LocalImplementation &implementation);
bool emitIQ3LocalImplementation(llvm::raw_ostream &output,
                                const LocalImplementation &implementation);
bool emitIQ1LocalImplementation(llvm::raw_ostream &output,
                                const LocalImplementation &implementation);
bool emitQ6LocalImplementation(llvm::raw_ostream &output,
                               const LocalImplementation &implementation);
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
bool emitSignedCodebookLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation);
bool emitPackedU9U7CodebookLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation);
bool emitPackedU11GridDeltaLocalImplementation(
    llvm::raw_ostream &output, const LocalImplementation &implementation);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVQUANTINTRINSICC_H
