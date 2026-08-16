#include "RISCVIntrinsicC.h"

namespace weft::riscv_internal {

std::string rvvShapeSuffix(const RVVVectorShape &shape) {
  if (!shape)
    return {};
  std::string lmul;
  if (shape.lmulEighths == 1)
    lmul = "mf8";
  else if (shape.lmulEighths == 2)
    lmul = "mf4";
  else if (shape.lmulEighths == 4)
    lmul = "mf2";
  else if (std::optional<unsigned> integer = rvvIntegerLMUL(shape))
    lmul = "m" + std::to_string(*integer);
  else
    return {};
  return std::to_string(shape.sew) + lmul;
}

bool SelectedLocalImplementations::contains(LocalPrimitiveKind primitive) const {
  for (const LocalImplementation &implementation : implementations)
    if (implementation.primitive == primitive)
      return true;
  return false;
}

bool SelectedLocalImplementations::contains(
    LocalPrimitiveKind primitive,
    LocalImplementationStructure structure) const {
  for (const LocalImplementation &implementation : implementations)
    if (implementation.primitive == primitive &&
        implementation.structure == structure)
      return true;
  return false;
}

bool SelectedLocalImplementations::contains(
    LocalPrimitiveKind primitive, LocalImplementationStructure structure,
    unsigned semanticLanes, RVVVectorShape primaryShape) const {
  for (const LocalImplementation &implementation : implementations)
    if (implementation.primitive == primitive &&
        implementation.structure == structure &&
        implementation.parameters.semanticLanes == semanticLanes &&
        (!primaryShape ||
         implementation.parameters.primaryShape == primaryShape))
      return true;
  return false;
}

std::string localImplementationName(const LocalImplementation &implementation) {
  const LocalImplementationParameters &parameters = implementation.parameters;
  const bool rvvRegister =
      implementation.structure ==
      LocalImplementationStructure::RVVRegisterMicrokernel;
  const bool ime = implementation.structure ==
                   LocalImplementationStructure::SpacemitIME1Fragment;
  const bool strip =
      implementation.structure == LocalImplementationStructure::RVVStripLoop;
  const bool register32E8M1OrM2 =
      rvvRegister && parameters.semanticLanes == 32 &&
      (parameters.primaryShape == RVVVectorShape{8, 8} ||
       parameters.primaryShape == RVVVectorShape{8, 16});
  const bool register32Or64E8M2 =
      rvvRegister &&
      (parameters.semanticLanes == 32 || parameters.semanticLanes == 64) &&
      parameters.primaryShape == RVVVectorShape{8, 16};
  const bool strip16E8M2 =
      strip && parameters.semanticLanes == 16 &&
      parameters.primaryShape == RVVVectorShape{8, 16};
  const std::string shape = rvvShapeSuffix(parameters.primaryShape);
  const std::string registerSuffix =
      "_register_l" + std::to_string(parameters.semanticLanes) + "_e" + shape;
  switch (implementation.primitive) {
  case LocalPrimitiveKind::None:
  case LocalPrimitiveKind::F32Math:
    return {};
  case LocalPrimitiveKind::SymmetricI4I8:
    if (parameters.semanticLanes != 16 ||
        parameters.primaryShape != RVVVectorShape{8, 8} ||
        parameters.secondaryShape != RVVVectorShape{32, 32} ||
        (parameters.rowMicrotile != 1 && parameters.rowMicrotile != 4))
      return {};
    if (ime)
      return parameters.rowMicrotile == 4
                 ? "__weft_ime1_symmetric_i4_i8_m4_n16_k32"
                 : "__weft_ime1_symmetric_i4_i8_n16_k32";
    if (rvvRegister)
      return parameters.rowMicrotile == 4
                 ? "__weft_rvv_symmetric_i4_i8_m4_n16_k32"
                 : "__weft_rvv_symmetric_i4_i8_n16_k32";
    return {};
  case LocalPrimitiveKind::AffineI4I8:
    if (parameters.semanticLanes != 16 ||
        parameters.primaryShape != RVVVectorShape{8, 8} ||
        parameters.secondaryShape != RVVVectorShape{32, 32} ||
        (parameters.rowMicrotile != 1 && parameters.rowMicrotile != 4))
      return {};
    if (ime)
      return parameters.rowMicrotile == 4
                 ? "__weft_ime1_affine_i4_i8_m4_n16_k32"
                 : "__weft_ime1_affine_i4_i8_n16_k32";
    if (rvvRegister)
      return parameters.rowMicrotile == 4
                 ? "__weft_rvv_affine_i4_i8_m4_n16_k32"
                 : "__weft_rvv_affine_i4_i8_n16_k32";
    return {};
  case LocalPrimitiveKind::GroupedAffineI4I8:
    return strip && parameters.semanticLanes == 32 &&
                   parameters.primaryShape == RVVVectorShape{8, 8} &&
                   parameters.secondaryShape == RVVVectorShape{16, 16}
               ? "__weft_grouped_affine_i4_i8_strip"
               : std::string{};
  case LocalPrimitiveKind::E2M1E8M0I8:
    if (strip && parameters.semanticLanes == 0 &&
        parameters.primaryShape == RVVVectorShape{8, 8} &&
        !parameters.secondaryShape)
      return "__weft_e2m1_e8m0_i8_strip";
    if (!rvvRegister)
      return {};
    if (parameters.semanticLanes != 32)
      return {};
    if (parameters.primaryShape == RVVVectorShape{8, 4} &&
        parameters.secondaryShape == RVVVectorShape{8, 4})
      return "__weft_e2m1_e8m0_i8_register_e8mf2";
    if (parameters.primaryShape == RVVVectorShape{8, 8} &&
        parameters.secondaryShape == RVVVectorShape{8, 16})
      return "__weft_e2m1_e8m0_i8_register_e8m1_e8m2";
    return {};
  case LocalPrimitiveKind::PackedI4I8:
    return register32E8M1OrM2
               ? "__weft_packed_i4_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::PackedI5I8:
    return register32E8M1OrM2
               ? "__weft_packed_i5_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::PackedI3GroupedI8:
    return register32Or64E8M2
               ? "__weft_packed_i3_grouped_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::Base3TernaryI8:
    return register32E8M1OrM2
               ? "__weft_base3_ternary_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::PackedI2TernaryI8:
    return register32E8M1OrM2
               ? "__weft_packed_i2_ternary_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::SignedCodebook8I8:
    return register32E8M1OrM2 && parameters.entryWidth == 8
               ? "__weft_signed_codebook8_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::SignedCodebook4I8:
    return register32E8M1OrM2 && parameters.entryWidth == 4
               ? "__weft_signed_codebook4_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::PackedU9U7CodebookI8:
    return register32E8M1OrM2 && parameters.entryWidth == 8
               ? "__weft_packed_u9_u7_codebook_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::PackedU11GridDeltaI8:
    return register32E8M1OrM2 && parameters.entryWidth == 8
               ? "__weft_packed_u11_grid_delta_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::NibbleCodebookI8:
    return register32E8M1OrM2
               ? "__weft_nibble_codebook_i8" + registerSuffix
               : std::string{};
  case LocalPrimitiveKind::IQ2SI8:
    return strip16E8M2
               ? "__weft_iq2_s_i8_strip"
               : register32Or64E8M2
                     ? "__weft_iq2_s_i8" + registerSuffix
                     : std::string{};
  case LocalPrimitiveKind::IQ3SI8:
    return strip16E8M2
               ? "__weft_iq3_s_i8_strip"
               : rvvRegister && parameters.semanticLanes == 64 &&
                         parameters.primaryShape == RVVVectorShape{8, 16}
                     ? "__weft_iq3_s_i8" + registerSuffix
                     : std::string{};
  case LocalPrimitiveKind::IQ1MI8:
    return strip16E8M2
               ? "__weft_iq1_m_i8_strip"
               : register32Or64E8M2
                     ? "__weft_iq1_m_i8" + registerSuffix
                     : std::string{};
  case LocalPrimitiveKind::Q6KI8:
    return strip16E8M2
               ? "__weft_q6_k_i8_strip"
               : register32Or64E8M2
                     ? "__weft_q6_k_i8" + registerSuffix
                     : std::string{};
  }
  return {};
}

} // namespace weft::riscv_internal
