#include "Weft/Target/RISCVTargetProfile.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

bool weft::RISCVTargetProfile::supportsSEW(unsigned sew) const {
  return llvm::is_contained(supportedSEW, sew);
}

bool weft::RISCVTargetProfile::supportsLMULEighths(int lmulEighths) const {
  return llvm::is_contained(legalLMULEighths, lmulEighths);
}

bool weft::RISCVTargetProfile::supportsFixedRVV() const {
  return hasRVV && vlenBits > 0 && vectorRegisters > 0;
}

bool weft::RISCVTargetProfile::supportsVectorShape(unsigned sew,
                                                   int lmulEighths) const {
  if (!supportsFixedRVV() || !supportsSEW(sew) ||
      !supportsLMULEighths(lmulEighths))
    return false;
  if (lmulEighths <= 0 ||
      (lmulEighths >= 8 && lmulEighths % 8 != 0))
    return false;
  uint64_t registerGroups =
      (static_cast<uint64_t>(lmulEighths) + 7) / 8;
  return registerGroups <= static_cast<uint64_t>(vectorRegisters);
}

bool weft::RISCVTargetProfile::supportsIndexedVectorMemory(
    unsigned elementSEW, int elementLMULEighths, unsigned indexSEW,
    int indexLMULEighths) const {
  if (!hasIndexedMemory ||
      (indexSEW != 16 && indexSEW != 32 && indexSEW != 64) ||
      !supportsVectorShape(elementSEW, elementLMULEighths) ||
      !supportsVectorShape(indexSEW, indexLMULEighths))
    return false;
  uint64_t elementLanes =
      static_cast<uint64_t>(vlenBits) * elementLMULEighths /
      (8 * static_cast<uint64_t>(elementSEW));
  uint64_t indexLanes =
      static_cast<uint64_t>(vlenBits) * indexLMULEighths /
      (8 * static_cast<uint64_t>(indexSEW));
  return elementLanes != 0 && elementLanes == indexLanes;
}

bool weft::RISCVTargetProfile::supportsSegmentVectorMemory(
    unsigned fields, unsigned sew, int lmulEighths) const {
  if (!hasSegmentMemory || fields < 2 || fields > 8 ||
      !supportsVectorShape(sew, lmulEighths))
    return false;
  if (static_cast<uint64_t>(fields) *
          static_cast<uint64_t>(lmulEighths) >
      64)
    return false;
  uint64_t registerGroups =
      (static_cast<uint64_t>(lmulEighths) + 7) / 8;
  return fields * registerGroups <=
         static_cast<uint64_t>(vectorRegisters);
}

bool weft::parseRISCVTargetProfile(llvm::StringRef march, llvm::StringRef abi,
                                   int64_t vlenBits,
                                   llvm::StringRef matrixExtension,
                                   llvm::StringRef partialCombinePolicy,
                                   llvm::StringRef recordAxisPolicy,
                                   RISCVTargetProfile &profile,
                                   std::string &error) {
  if (march.empty()) {
    error = "--march is required for RISC-V target lowering";
    return false;
  }
  if (abi.empty()) {
    error = "--abi is required for RISC-V target lowering";
    return false;
  }
  profile = RISCVTargetProfile{};
  if (march.starts_with("rv64")) {
    profile.xlen = 64;
    profile.triple = "riscv64-unknown-linux-gnu";
  } else if (march.starts_with("rv32")) {
    profile.xlen = 32;
    profile.triple = "riscv32-unknown-linux-gnu";
  } else {
    error = "--march must begin with rv32 or rv64";
    return false;
  }
  if (recordAxisPolicy == "within-record") {
    profile.recordAxisPolicy = RISCVRecordAxisPolicy::WithinRecord;
  } else if (recordAxisPolicy == "across-records") {
    profile.recordAxisPolicy = RISCVRecordAxisPolicy::AcrossRecords;
  } else {
    error =
        "--record-axis-policy must be within-record or across-records";
    return false;
  }
  std::optional<RISCVABI> parsedABI =
      llvm::StringSwitch<std::optional<RISCVABI>>(abi)
          .Case("lp64", RISCVABI::LP64)
          .Case("lp64f", RISCVABI::LP64F)
          .Case("lp64d", RISCVABI::LP64D)
          .Case("ilp32", RISCVABI::ILP32)
          .Case("ilp32f", RISCVABI::ILP32F)
          .Case("ilp32d", RISCVABI::ILP32D)
          .Default(std::nullopt);
  if (!parsedABI) {
    error = "--abi must be lp64/lp64f/lp64d or ilp32/ilp32f/ilp32d";
    return false;
  }
  bool abiIs64 = *parsedABI == RISCVABI::LP64 ||
                 *parsedABI == RISCVABI::LP64F ||
                 *parsedABI == RISCVABI::LP64D;
  if (abiIs64 != (profile.xlen == 64)) {
    error = profile.xlen == 64 ? "RV64 requires an lp64 ABI"
                               : "RV32 requires an ilp32 ABI";
    return false;
  }
  if (vlenBits < 0 || (vlenBits != 0 && vlenBits % 8 != 0)) {
    error = "--vlen-bits must be zero or a positive multiple of eight";
    return false;
  }
  llvm::SmallVector<llvm::StringRef> tokens;
  march.split(tokens, '_', -1, false);
  llvm::StringRef base = tokens.front().drop_front(4);
  if (base.empty() ||
      llvm::any_of(base, [](char extension) {
        return !llvm::StringRef("iemafdqlcbkjtpvnhg").contains(extension);
      }) ||
      (!base.contains('g') && !base.contains('i') && !base.contains('e'))) {
    error = "--march must contain a valid RISC-V base ISA";
    return false;
  }
  auto baseHas = [&](char extension) { return base.contains(extension); };
  for (size_t index = 0; index < base.size(); ++index)
    if (base.drop_front(index + 1).contains(base[index])) {
      error = "--march base ISA contains a duplicate extension";
      return false;
    }
  unsigned baseRoots = static_cast<unsigned>(baseHas('g')) +
                       static_cast<unsigned>(baseHas('i')) +
                       static_cast<unsigned>(baseHas('e'));
  if (baseRoots != 1 ||
      (baseHas('g') &&
       (baseHas('m') || baseHas('a') || baseHas('f') || baseHas('d'))) ||
      (baseHas('d') && !baseHas('f') && !baseHas('g')) ||
      (baseHas('q') && !baseHas('d') && !baseHas('g')) ||
      (profile.xlen == 64 && baseHas('e'))) {
    error = "--march base ISA has invalid extension prerequisites";
    return false;
  }
  auto hasToken = [&](llvm::StringRef extension) {
    return llvm::any_of(llvm::ArrayRef<llvm::StringRef>(tokens).drop_front(),
                        [&](llvm::StringRef token) {
                          return token == extension;
                        });
  };
  bool hasGeneral = baseHas('g');
  profile.march = march.str();
  profile.abi = abi.str();
  profile.abiKind = *parsedABI;
  profile.hasF = hasGeneral || baseHas('f');
  profile.hasD = hasGeneral || baseHas('d');
  profile.hasScalarF16 = hasToken("zfh") || hasToken("zfhmin");
  profile.hasRVV = baseHas('v');
  bool hasEmbeddedVector = llvm::any_of(
      llvm::ArrayRef<llvm::StringRef>(tokens).drop_front(),
      [](llvm::StringRef token) { return token.starts_with("zve"); });
  if (hasEmbeddedVector && !profile.hasRVV) {
    error = "the current RISC-V backend requires the full V extension, not a Zve subset";
    return false;
  }
  bool requestedVectorF16 = hasToken("zvfh") || hasToken("zvfhmin");
  if (requestedVectorF16 && !profile.hasRVV) {
    error = "zvfh/zvfhmin requires the full V extension in the current backend";
    return false;
  }
  if (requestedVectorF16 && !profile.hasF) {
    error = "zvfh/zvfhmin requires the scalar F extension";
    return false;
  }
  if (profile.hasScalarF16 && !profile.hasF) {
    error = "zfh/zfhmin requires the scalar F extension";
    return false;
  }
  profile.hasVectorF16 = requestedVectorF16;
  bool abiNeedsF = *parsedABI == RISCVABI::LP64F ||
                   *parsedABI == RISCVABI::LP64D ||
                   *parsedABI == RISCVABI::ILP32F ||
                   *parsedABI == RISCVABI::ILP32D;
  bool abiNeedsD = *parsedABI == RISCVABI::LP64D ||
                   *parsedABI == RISCVABI::ILP32D;
  if ((abiNeedsF && !profile.hasF) || (abiNeedsD && !profile.hasD)) {
    error = "--abi floating-point register convention is not present in --march";
    return false;
  }
  if (!profile.hasRVV) {
    error = "the current Weft RISC-V physical compiler requires the full V extension";
    return false;
  }
  if (vlenBits <= 0) {
    error = "the current Weft RISC-V physical compiler requires an explicit positive --vlen-bits";
    return false;
  }
  profile.vlenBits = vlenBits;
  profile.vectorRegisters = 32;
  profile.supportedSEW = {8, 16, 32, 64};
  profile.legalLMULEighths = {1, 2, 4, 8, 16, 32, 64};
  profile.hasIndexedMemory = true;
  profile.hasSegmentMemory = true;
  profile.hasWideningInteger = true;
  profile.hasWideningFloat = profile.hasF;
  if (partialCombinePolicy == "independent-multilevel") {
    profile.partialCombinePolicy =
        RISCVPartialCombinePolicy::IndependentMultilevel;
  } else if (partialCombinePolicy == "sequential") {
    profile.partialCombinePolicy = RISCVPartialCombinePolicy::Sequential;
  } else {
    error =
        "--partial-combine-policy must be independent-multilevel or sequential";
    return false;
  }
  const bool spacemitIME1 = matrixExtension == "spacemit-ime1";
  if (matrixExtension != "none" && !spacemitIME1) {
    error = "unsupported --matrix-extension value: " + matrixExtension.str();
    return false;
  }
  if (spacemitIME1) {
    if (profile.xlen != 64 || !profile.hasRVV || profile.vlenBits != 256) {
      error = "spacemit-ime1 requires RV64, RVV, and an explicit VLEN of exactly 256 bits";
      return false;
    }
  }
  if (spacemitIME1 && profile.littleEndian && profile.xlen == 64 &&
      profile.supportsFixedRVV() && profile.vlenBits == 256 &&
      profile.supportsVectorShape(8, 8) &&
      profile.supportsVectorShape(32, 8)) {
    // The board-validated leaf consumes v0/v1 as two 4x8 signed-i8 fragments
    // and produces one 4x4 i32 fragment in v2/v3.
    profile.fragmentCapabilities.push_back(RISCVFragmentCapability{
        RISCVFragmentInstruction::SpacemitIME1I8MMA,
        RISCVFragmentSignedness::Signed, RISCVFragmentSignedness::Signed,
        8, 8, 32, 4, 4, 8, 1, 1, 2});
  }
  return true;
}
