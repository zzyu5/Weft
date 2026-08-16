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

llvm::StringRef intrinsicCLeafName(IntrinsicCLeaf leaf) {
  switch (leaf) {
  case IntrinsicCLeaf::None:
    return {};
  case IntrinsicCLeaf::RVVSymmetricI4I8N16K32:
    return "__weft_rvv_symmetric_i4_i8_n16_k32";
  case IntrinsicCLeaf::RVVAffineI4I8N16K32:
    return "__weft_rvv_affine_i4_i8_n16_k32";
  case IntrinsicCLeaf::IME1SymmetricI4I8N16K32:
    return "__weft_ime1_symmetric_i4_i8_n16_k32";
  case IntrinsicCLeaf::IME1AffineI4I8N16K32:
    return "__weft_ime1_affine_i4_i8_n16_k32";
  case IntrinsicCLeaf::GroupedAffineI4I8VLEN128:
    return "__weft_grouped_affine_i4_i8_vl128";
  case IntrinsicCLeaf::GroupedAffineI4I8VLEN256:
    return "__weft_grouped_affine_i4_i8_vl256";
  case IntrinsicCLeaf::GroupedAffineI4I8Scalable:
    return "__weft_grouped_affine_i4_i8_rvv";
  case IntrinsicCLeaf::E2M1E8M0I8VLEN128:
    return "__weft_e2m1_e8m0_i8_vl128";
  case IntrinsicCLeaf::E2M1E8M0I8VLEN256:
    return "__weft_e2m1_e8m0_i8_vl256";
  case IntrinsicCLeaf::E2M1E8M0I8Scalable:
    return "__weft_e2m1_e8m0_i8_rvv";
  case IntrinsicCLeaf::PackedI5I8VLEN128:
    return "__weft_packed_i5_i8_vl128";
  case IntrinsicCLeaf::PackedI5I8VLEN256:
    return "__weft_packed_i5_i8_vl256";
  case IntrinsicCLeaf::IQ2SI8FixedLanes32:
    return "__weft_iq2_s_i8_lanes32";
  case IntrinsicCLeaf::IQ2SI8FixedLanes64:
    return "__weft_iq2_s_i8_lanes64";
  case IntrinsicCLeaf::IQ2SI8Scalable:
    return "__weft_iq2_s_i8_rvv";
  case IntrinsicCLeaf::IQ3SI8Scalable:
    return "__weft_iq3_s_i8_rvv";
  case IntrinsicCLeaf::IQ1MI8Fixed:
    return "__weft_iq1_m_i8_vl128";
  case IntrinsicCLeaf::IQ1MI8Scalable:
    return "__weft_iq1_m_i8_rvv";
  case IntrinsicCLeaf::Q6KI8FixedLanes32:
    return "__weft_q6_k_i8_lanes32";
  case IntrinsicCLeaf::Q6KI8FixedLanes64:
    return "__weft_q6_k_i8_lanes64";
  case IntrinsicCLeaf::Q6KI8Scalable:
    return "__weft_q6_k_i8_rvv";
  case IntrinsicCLeaf::RVVF32M2Math:
    return {};
  }
  return {};
}

} // namespace weft::riscv_internal
