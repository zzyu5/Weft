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

bool weft::RISCVTargetProfile::supportsVLENAtLeast(unsigned bits) const {
  return hasRVV && vlenBits > 0 && static_cast<uint64_t>(vlenBits) >= bits;
}

bool weft::RISCVTargetProfile::hasMatrixExtension(
    RISCVMatrixExtension extension) const {
  return matrixExtension == extension;
}

bool weft::parseRISCVTargetProfile(llvm::StringRef march, llvm::StringRef abi,
                                   int64_t vlenBits,
                                   llvm::StringRef matrixExtension,
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
  profile.vlenBits = vlenBits;
  if (profile.hasRVV) {
    profile.vectorRegisters = 32;
    profile.supportedSEW = {8, 16, 32, 64};
    profile.legalLMULEighths = {1, 2, 4, 8, 16, 32, 64};
    profile.hasIndexedMemory = true;
    profile.hasSegmentMemory = true;
    profile.hasWideningInteger = true;
    profile.hasWideningFloat = profile.hasF;
  } else if (vlenBits != 0) {
    error = "--vlen-bits requires an ISA string with the vector extension";
    return false;
  }
  if (matrixExtension == "none") {
    profile.matrixExtension = RISCVMatrixExtension::None;
  } else if (matrixExtension == "spacemit-ime1") {
    if (profile.xlen != 64 || !profile.hasRVV || profile.vlenBits != 256) {
      error = "spacemit-ime1 requires RV64, RVV, and an explicit VLEN of exactly 256 bits";
      return false;
    }
    profile.matrixExtension = RISCVMatrixExtension::SpacemitIME1;
  } else {
    error = "unsupported --matrix-extension value: " + matrixExtension.str();
    return false;
  }
  return true;
}
