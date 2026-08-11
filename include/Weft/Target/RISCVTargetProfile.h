#ifndef WEFT_TARGET_RISCVTARGETPROFILE_H
#define WEFT_TARGET_RISCVTARGETPROFILE_H

#include <cstdint>
#include <string>

#include "llvm/ADT/StringRef.h"

namespace weft {

struct RISCVTargetProfile {
  std::string triple;
  std::string march;
  std::string abi;
  int64_t xlen = 0;
  bool littleEndian = true;
  bool hasRVV = false;
  int64_t vlenBits = 0;
  int64_t vectorRegisters = 32;
  std::string matrixExtension;
};

bool parseRISCVTargetProfile(llvm::StringRef march, llvm::StringRef abi,
                             int64_t vlenBits, RISCVTargetProfile &profile,
                             std::string &error);

} // namespace weft

#endif // WEFT_TARGET_RISCVTARGETPROFILE_H
