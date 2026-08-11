#include "Weft/Target/RISCVTargetProfile.h"

bool weft::parseRISCVTargetProfile(llvm::StringRef march, llvm::StringRef abi,
                                   int64_t vlenBits,
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
  if (march.starts_with("rv64")) {
    profile.xlen = 64;
    profile.triple = "riscv64-unknown-linux-gnu";
    if (!abi.starts_with("lp64")) {
      error = "RV64 requires an lp64 ABI";
      return false;
    }
  } else if (march.starts_with("rv32")) {
    profile.xlen = 32;
    profile.triple = "riscv32-unknown-linux-gnu";
    if (!abi.starts_with("ilp32")) {
      error = "RV32 requires an ilp32 ABI";
      return false;
    }
  } else {
    error = "--march must begin with rv32 or rv64";
    return false;
  }
  if (vlenBits < 0 || (vlenBits != 0 && vlenBits % 8 != 0)) {
    error = "--vlen-bits must be zero or a positive multiple of eight";
    return false;
  }
  llvm::StringRef extensions = march.drop_front(4);
  profile.march = march.str();
  profile.abi = abi.str();
  profile.hasRVV = extensions.contains('v');
  profile.vlenBits = vlenBits;
  return true;
}
