#include "RISCVQuantIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitPackedI5IntrinsicCLeaves(llvm::raw_ostream &output, bool vlen128,
                                  bool vlen256) {
  if (vlen128) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_packed_i5_i8_vl128(
    const uint8_t *low_bits, const uint8_t *high_bits,
    const uint8_t *activation_bytes, int32_t zero_point,
    float dot_scale, float additive_bias, float init) {
  const size_t vl16 = __riscv_vsetvl_e8m1(16);
  const vuint8m1_t packed = __riscv_vle8_v_u8m1(low_bits, vl16);
  const vint8m1_t low = __riscv_vreinterpret_v_u8m1_i8m1(
      __riscv_vand_vx_u8m1(packed, UINT8_C(15), vl16));
  const vint8m1_t high = __riscv_vreinterpret_v_u8m1_i8m1(
      __riscv_vsrl_vx_u8m1(packed, 4, vl16));
  const size_t vl32 = __riscv_vsetvl_e8m2(32);
  vint8m2_t codes = __riscv_vcreate_v_i8m1_i8m2(low, high);
  const vbool4_t high_mask = __riscv_vlm_v_b4(high_bits, vl32);
  codes = __riscv_vor_vx_i8m2_mu(
      high_mask, codes, codes, INT8_C(16), vl32);
  codes = __riscv_vsub_vx_i8m2(codes, zero_point, vl32);
  const vint8m2_t activation = __riscv_vle8_v_i8m2(
      (const int8_t *)(const void *)activation_bytes, vl32);
  const vint16m4_t products =
      __riscv_vwmul_vv_i16m4(codes, activation, vl32);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t integer_dot = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m4_i32m1(products, zero, vl32));
  return init + dot_scale * (float)integer_dot + additive_bias;
}

)c";
  }
  if (vlen256) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_packed_i5_i8_vl256(
    const uint8_t *low_bits, const uint8_t *high_bits,
    const uint8_t *activation_bytes, int32_t zero_point,
    float dot_scale, float additive_bias, float init) {
  const size_t vl16 = __riscv_vsetvl_e8m1(16);
  const vuint8m1_t packed = __riscv_vle8_v_u8m1(low_bits, vl16);
  vint8m1_t low = __riscv_vreinterpret_v_u8m1_i8m1(
      __riscv_vand_vx_u8m1(packed, UINT8_C(15), vl16));
  const vint8m1_t high = __riscv_vreinterpret_v_u8m1_i8m1(
      __riscv_vsrl_vx_u8m1(packed, 4, vl16));
  low = __riscv_vslideup_vx_i8m1(low, high, 16, 32);
  const size_t vl32 = __riscv_vsetvl_e8m2(32);
  vint8m2_t codes = __riscv_vlmul_ext_v_i8m1_i8m2(low);
  const vbool4_t high_mask = __riscv_vlm_v_b4(high_bits, vl32);
  codes = __riscv_vor_vx_i8m2_mu(
      high_mask, codes, codes, INT8_C(16), vl32);
  codes = __riscv_vsub_vx_i8m2(codes, zero_point, vl32);
  const vint8m2_t activation = __riscv_vle8_v_i8m2(
      (const int8_t *)(const void *)activation_bytes, vl32);
  const vint16m4_t products =
      __riscv_vwmul_vv_i16m4(codes, activation, vl32);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t integer_dot = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m4_i32m1(products, zero, vl32));
  return init + dot_scale * (float)integer_dot + additive_bias;
}

)c";
  }
}

} // namespace weft::riscv_internal
