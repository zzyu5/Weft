#include "Weft/Target/RISCVFragment.h"

llvm::ArrayRef<weft::RISCVFragmentCapability> weft::riscvFragmentCapabilities() {
  static const RISCVFragmentCapability capabilities[] = {{
      RISCVFragmentInstruction::SpacemitIME1I8MMA,
      "spacemit-ime1-i8-mma", "spacemit-ime1", 64, 256,
      RISCVFragmentSignedness::Signed, RISCVFragmentSignedness::Signed,
      8, 8, 32, 4, 4, 8, 1, 1, 2,
      {"ime1-i8-m4-k8", {0, 1}, 4, 8, "row_major", "row_major", 8, 32},
      {"ime1-i8-n4-k8-transposed", {1, 0}, 4, 8,
       "row_major", "row_major", 8, 32},
      {"ime1-i32-m4-n4", {0, 1}, 4, 4, "row_major", "row_major", 32, 32},
      1, 1, {"t0", "v0", "v1", "v2", "v3"}, true, true,
      64, 64, 64, 1, 1, RISCVFragmentSignedness::Signed}};
  return capabilities;
}

const weft::RISCVFragmentCapability *
weft::findRISCVFragmentCapability(llvm::StringRef name) {
  for (const auto &capability : riscvFragmentCapabilities())
    if (capability.name == name)
      return &capability;
  return nullptr;
}
