#include "RISCVIntrinsicC.h"

namespace weft::riscv_internal {

bool SelectedLocalImplementations::contains(LocalImplementationLeaf leaf) const {
  for (const LocalImplementation &implementation : implementations)
    if (implementation.leaf == leaf)
      return true;
  return false;
}

bool SelectedLocalImplementations::contains(
    LocalImplementationLeaf leaf, unsigned semanticLanes,
    RVVVectorShape primaryShape) const {
  for (const LocalImplementation &implementation : implementations)
    if (implementation.leaf == leaf &&
        implementation.parameters.semanticLanes == semanticLanes &&
        (!primaryShape ||
         implementation.parameters.primaryShape == primaryShape))
      return true;
  return false;
}

std::string localImplementationName(const LocalImplementation &implementation) {
  const LocalImplementationParameters &parameters = implementation.parameters;
  const std::string shape = rvvShapeSuffix(parameters.primaryShape);
  const std::string registerSuffix =
      "_register_l" + std::to_string(parameters.semanticLanes) + "_e" + shape;
  switch (implementation.leaf) {
  case LocalImplementationLeaf::None:
  case LocalImplementationLeaf::RVVF32Math:
    return {};
  case LocalImplementationLeaf::RVVSymmetricI4I8N16:
    return "__weft_rvv_symmetric_i4_i8_n16_k32";
  case LocalImplementationLeaf::RVVSymmetricI4I8M4N16:
    return "__weft_rvv_symmetric_i4_i8_m4_n16_k32";
  case LocalImplementationLeaf::IME1SymmetricI4I8N16:
    return "__weft_ime1_symmetric_i4_i8_n16_k32";
  case LocalImplementationLeaf::IME1SymmetricI4I8M4N16:
    return "__weft_ime1_symmetric_i4_i8_m4_n16_k32";
  case LocalImplementationLeaf::RVVAffineI4I8N16:
    return "__weft_rvv_affine_i4_i8_n16_k32";
  case LocalImplementationLeaf::RVVAffineI4I8M4N16:
    return "__weft_rvv_affine_i4_i8_m4_n16_k32";
  case LocalImplementationLeaf::IME1AffineI4I8N16:
    return "__weft_ime1_affine_i4_i8_n16_k32";
  case LocalImplementationLeaf::IME1AffineI4I8M4N16:
    return "__weft_ime1_affine_i4_i8_m4_n16_k32";
  case LocalImplementationLeaf::RVVGroupedAffineI4I8Register:
    return "__weft_grouped_affine_i4_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVGroupedAffineI4I8Strip:
    return "__weft_grouped_affine_i4_i8_strip";
  case LocalImplementationLeaf::RVVE2M1E8M0I8RegisterMF2:
    return "__weft_e2m1_e8m0_i8_register_e8mf2";
  case LocalImplementationLeaf::RVVE2M1E8M0I8RegisterM1M2:
    return "__weft_e2m1_e8m0_i8_register_e8m1_e8m2";
  case LocalImplementationLeaf::RVVE2M1E8M0I8Strip:
    return "__weft_e2m1_e8m0_i8_strip";
  case LocalImplementationLeaf::RVVPackedI4I8Register:
    return "__weft_packed_i4_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVPackedI5I8Register:
    return "__weft_packed_i5_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVPackedI3GroupedI8Register:
    return "__weft_packed_i3_grouped_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVBase3TernaryI8Register:
    return "__weft_base3_ternary_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVPackedI2TernaryI8Register:
    return "__weft_packed_i2_ternary_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVSignedCodebook8I8Register:
    return "__weft_signed_codebook8_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVSignedCodebook4I8Register:
    return "__weft_signed_codebook4_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVPackedU9U7CodebookI8Register:
    return "__weft_packed_u9_u7_codebook_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVPackedU11GridDeltaI8Register:
    return "__weft_packed_u11_grid_delta_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVNibbleCodebookI8Register:
    return "__weft_nibble_codebook_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVIQ2SI8Register:
    return "__weft_iq2_s_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVIQ2SI8Strip:
    return "__weft_iq2_s_i8_strip";
  case LocalImplementationLeaf::RVVIQ3SI8Register:
    return "__weft_iq3_s_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVIQ3SI8Strip:
    return "__weft_iq3_s_i8_strip";
  case LocalImplementationLeaf::RVVIQ1MI8Register:
    return "__weft_iq1_m_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVIQ1MI8Strip:
    return "__weft_iq1_m_i8_strip";
  case LocalImplementationLeaf::RVVQ6KI8Register:
    return "__weft_q6_k_i8" + registerSuffix;
  case LocalImplementationLeaf::RVVQ6KI8Strip:
    return "__weft_q6_k_i8_strip";
  }
  return {};
}

} // namespace weft::riscv_internal
