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

class SelectedLocalImplementations {
public:
  void add(const LocalImplementation &implementation) {
    if (implementation)
      implementations.insert(implementation);
  }
  bool empty() const { return implementations.empty(); }
  template <typename Fn> void forEach(Fn &&fn) const {
    for (const LocalImplementation &implementation : implementations)
      fn(implementation);
  }

private:
  std::set<LocalImplementation> implementations;
};

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
bool emitQuantLocalImplementations(
    llvm::raw_ostream &output,
    const SelectedLocalImplementations &implementations,
    std::string &unsupportedSymbol);
void emitIMELocalImplementations(llvm::raw_ostream &output,
                                 bool usesIME1SymmetricI4I8,
                                 bool usesIME1AffineI4I8,
                                 bool usesIME1SymmetricI4I8M4,
                                 bool usesIME1AffineI4I8M4);
bool emitIntrinsicCPrelude(llvm::raw_ostream &output,
                           const SelectedLocalImplementations &implementations,
                           std::string &unsupportedSymbol);

} // namespace weft::riscv_internal

#endif // WEFT_LIB_TARGET_RISCVINTRINSICC_H
