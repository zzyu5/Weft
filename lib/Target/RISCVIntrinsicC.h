#ifndef WEFT_LIB_TARGET_RISCVINTRINSICC_H
#define WEFT_LIB_TARGET_RISCVINTRINSICC_H

#include "RISCVPhysicalPlanning.h"
#include "RISCVRVVSpelling.h"

#include <cstdint>
#include <set>
#include <string>

namespace llvm {
class raw_ostream;
}

namespace weft::riscv_internal {

struct LocalImplementationQuery {
  LocalPrimitiveKind primitive = LocalPrimitiveKind::None;
  CoreInstructionKind instruction = CoreInstructionKind::None;
  unsigned laneSpan = 0;
  unsigned rowFactor = 0;
  RVVVectorShape primaryShape;
  int sequentialK = -1;
};

class SelectedLocalImplementations {
public:
  void add(const LocalImplementation &implementation) {
    if (implementation)
      implementations.insert(implementation);
  }
  bool contains(const LocalImplementation &implementation) const {
    return implementations.find(implementation) != implementations.end();
  }
  bool contains(LocalPrimitiveKind primitive) const;
  bool contains(const LocalImplementationQuery &query) const;
  bool empty() const { return implementations.empty(); }

private:
  std::set<LocalImplementation> implementations;
};

std::string localImplementationName(const LocalImplementation &implementation);
void emitRVVLocalImplementations(llvm::raw_ostream &output,
                                 bool usesRVVSymmetricI4I8,
                                 bool usesRVVAffineI4I8,
                                 bool usesRVVSymmetricI4I8M4,
                                 bool usesRVVAffineI4I8M4,
                                 bool usesGroupedI4I8RegisterL16,
                                 bool usesGroupedI4I8RegisterL32,
                                 bool usesGroupedI4I8Strip,
                                 bool usesE2M1RegisterE8M1M2,
                                 bool usesE2M1RegisterE8MF2,
                                 bool usesE2M1Strip);
void emitQuantLocalImplementations(
    llvm::raw_ostream &output,
    const SelectedLocalImplementations &implementations);
void emitIMELocalImplementations(llvm::raw_ostream &output,
                                 bool usesIME1SymmetricI4I8,
                                 bool usesIME1AffineI4I8,
                                 bool usesIME1SymmetricI4I8M4,
                                 bool usesIME1AffineI4I8M4);
void emitIntrinsicCPrelude(llvm::raw_ostream &output,
                           const SelectedLocalImplementations &implementations);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVINTRINSICC_H
