#include "RISCVIntrinsicC.h"

namespace weft::riscv_internal {

bool SelectedLocalImplementations::contains(LocalPrimitiveKind primitive) const {
  for (const LocalImplementation &implementation : implementations)
    if (implementation.primitive == primitive)
      return true;
  return false;
}

bool SelectedLocalImplementations::contains(
    const LocalImplementationQuery &query) const {
  for (const LocalImplementation &implementation : implementations) {
    const RVVVectorShape primary = implementation.valueShapes.empty()
                                       ? RVVVectorShape{}
                                       : implementation.valueShapes.front();
    const unsigned rows =
        implementation.mapping.instruction ==
                CoreInstructionKind::SpacemitIME1MMA
            ? mappedFragmentFactor(implementation.mapping, kCoreAxisM)
            : mappedRegisterFactor(implementation.mapping, kCoreAxisM);
    const bool sequential =
        mappedAxisHasSequentialIteration(implementation.mapping, kCoreAxisK);
    const unsigned queryLanes =
        implementation.primitive == LocalPrimitiveKind::GroupedAffineI4I8
            ? mappedHardwareLaneFactor(implementation.mapping)
            : mappedLaneSpan(implementation.mapping);
    if (implementation.primitive == query.primitive &&
        (query.instruction == CoreInstructionKind::None ||
         implementation.mapping.instruction == query.instruction) &&
        (query.laneSpan == 0 || queryLanes == query.laneSpan) &&
        (query.rowFactor == 0 || rows == query.rowFactor) &&
        (!query.primaryShape || primary == query.primaryShape) &&
        (query.sequentialK < 0 || sequential == (query.sequentialK != 0)))
      return true;
  }
  return false;
}

std::string localImplementationName(const LocalImplementation &implementation) {
  const RVVVectorShape primary = implementation.valueShapes.empty()
                                     ? RVVVectorShape{}
                                     : implementation.valueShapes.front();
  const unsigned lanes = mappedLaneSpan(implementation.mapping);
  const unsigned hardwareLanes =
      mappedHardwareLaneFactor(implementation.mapping);
  const unsigned rows = implementation.mapping.instruction ==
                                CoreInstructionKind::SpacemitIME1MMA
                            ? mappedFragmentFactor(implementation.mapping,
                                                   kCoreAxisM)
                            : mappedRegisterFactor(implementation.mapping,
                                                   kCoreAxisM);
  const bool sequential =
      mappedAxisHasSequentialIteration(implementation.mapping, kCoreAxisK);
  const std::string shape = rvvShapeSuffix(primary);
  const std::string registerSuffix =
      "_register_l" + std::to_string(lanes) + "_e" + shape;
  switch (implementation.primitive) {
  case LocalPrimitiveKind::None:
  case LocalPrimitiveKind::F32Math:
    return {};
  case LocalPrimitiveKind::SymmetricI4I8:
  case LocalPrimitiveKind::AffineI4I8: {
    const bool affine = implementation.primitive == LocalPrimitiveKind::AffineI4I8;
    const bool ime = implementation.mapping.instruction ==
                     CoreInstructionKind::SpacemitIME1MMA;
    std::string name = "__weft_" + std::string(ime ? "ime1_" : "rvv_") +
                       (affine ? "affine" : "symmetric") + "_i4_i8_";
    if (rows == 4)
      name += "m4_";
    return name + "n16_k32";
  }
  case LocalPrimitiveKind::GroupedAffineI4I8:
    return sequential
               ? "__weft_grouped_affine_i4_i8_strip"
               : "__weft_grouped_affine_i4_i8_register_l" +
                     std::to_string(hardwareLanes) + "_e" + shape;
  case LocalPrimitiveKind::E2M1E8M0I8:
    if (sequential)
      return "__weft_e2m1_e8m0_i8_strip";
    return primary == RVVVectorShape{8, 4}
               ? "__weft_e2m1_e8m0_i8_register_e8mf2"
               : "__weft_e2m1_e8m0_i8_register_e8m1_e8m2";
  case LocalPrimitiveKind::PackedI4I8:
    return "__weft_packed_i4_i8" + registerSuffix;
  case LocalPrimitiveKind::PackedI5I8:
    return "__weft_packed_i5_i8" + registerSuffix;
  case LocalPrimitiveKind::PackedI3GroupedI8:
    return "__weft_packed_i3_grouped_i8" + registerSuffix;
  case LocalPrimitiveKind::Base3TernaryI8:
    return "__weft_base3_ternary_i8" + registerSuffix;
  case LocalPrimitiveKind::PackedI2TernaryI8:
    return "__weft_packed_i2_ternary_i8" + registerSuffix;
  case LocalPrimitiveKind::SignedCodebook8I8:
    return "__weft_signed_codebook8_i8" + registerSuffix;
  case LocalPrimitiveKind::SignedCodebook4I8:
    return "__weft_signed_codebook4_i8" + registerSuffix;
  case LocalPrimitiveKind::PackedU9U7CodebookI8:
    return "__weft_packed_u9_u7_codebook_i8" + registerSuffix;
  case LocalPrimitiveKind::PackedU11GridDeltaI8:
    return "__weft_packed_u11_grid_delta_i8" + registerSuffix;
  case LocalPrimitiveKind::NibbleCodebookI8:
    return "__weft_nibble_codebook_i8" + registerSuffix;
  case LocalPrimitiveKind::IQ2SI8:
    return sequential ? "__weft_iq2_s_i8_strip"
                      : "__weft_iq2_s_i8" + registerSuffix;
  case LocalPrimitiveKind::IQ3SI8:
    return sequential ? "__weft_iq3_s_i8_strip"
                      : "__weft_iq3_s_i8" + registerSuffix;
  case LocalPrimitiveKind::IQ1MI8:
    return sequential ? "__weft_iq1_m_i8_strip"
                      : "__weft_iq1_m_i8" + registerSuffix;
  case LocalPrimitiveKind::Q6KI8:
    return sequential ? "__weft_q6_k_i8_strip"
                      : "__weft_q6_k_i8" + registerSuffix;
  }
  return {};
}

} // namespace weft::riscv_internal
