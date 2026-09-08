#ifndef WEFT_TARGET_RISCVTARGETPROFILE_H
#define WEFT_TARGET_RISCVTARGETPROFILE_H

#include "Weft/Target/RISCVFragment.h"

#include <cstdint>
#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"

namespace weft {

enum class RISCVABI {
  LP64,
  LP64F,
  LP64D,
  ILP32,
  ILP32F,
  ILP32D,
};

enum class RISCVPartialCombinePolicy {
  IndependentMultilevel,
  Sequential,
};

enum class RISCVRecordAxisPolicy {
  WithinRecord,
  AcrossRecords,
};

struct RISCVTargetProfile {
  std::string triple;
  std::string march;
  std::string abi;
  RISCVABI abiKind = RISCVABI::LP64;
  int64_t xlen = 0;
  bool littleEndian = true;
  bool hasRVV = false;
  bool hasF = false;
  bool hasD = false;
  bool hasScalarF16 = false;
  bool hasVectorF16 = false;
  bool hasIndexedMemory = false;
  bool hasSegmentMemory = false;
  bool hasWideningInteger = false;
  bool hasWideningFloat = false;
  int64_t vlenBits = 0;
  int64_t vectorRegisters = 0;
  int64_t maxPrivateStackBytes = 65536;
  int64_t maxWideningCombineGroups = 2;
  std::vector<unsigned> supportedSEW;
  std::vector<int> legalLMULEighths;
  std::vector<RISCVFragmentCapability> fragmentCapabilities;
  RISCVPartialCombinePolicy partialCombinePolicy =
      RISCVPartialCombinePolicy::IndependentMultilevel;
  RISCVRecordAxisPolicy recordAxisPolicy =
      RISCVRecordAxisPolicy::WithinRecord;

  bool supportsSEW(unsigned sew) const;
  bool supportsLMULEighths(int lmulEighths) const;
  bool supportsFixedRVV() const;
  bool supportsVectorShape(unsigned sew, int lmulEighths) const;
  bool supportsIndexedVectorMemory(unsigned elementSEW,
                                   int elementLMULEighths,
                                   unsigned indexSEW,
                                   int indexLMULEighths) const;
  bool supportsSegmentVectorMemory(unsigned fields, unsigned sew,
                                   int lmulEighths) const;
};

bool parseRISCVTargetProfile(llvm::StringRef march, llvm::StringRef abi,
                             int64_t vlenBits,
                             llvm::StringRef matrixExtension,
                             int64_t maxWideningCombineGroups,
                             llvm::StringRef partialCombinePolicy,
                             llvm::StringRef recordAxisPolicy,
                             RISCVTargetProfile &profile, std::string &error);

struct RISCVNativeTarget {
  RISCVTargetProfile profile;
  std::vector<int> cpus;
};

bool queryNativeRISCVTarget(RISCVNativeTarget &target, std::string &error);

} // namespace weft

#endif // WEFT_TARGET_RISCVTARGETPROFILE_H
