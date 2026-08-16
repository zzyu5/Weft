#include "RISCVQuantIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitIQ3LocalImplementations(llvm::raw_ostream &output, bool registerL64,
                             bool strip) {
  if (registerL64) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_iq3_s_i8_register_l64_e8m2(
    const uint8_t *codes, const uint8_t *high_bits,
    const uint8_t *sign_bits, const uint8_t *scales,
    const uint8_t *activation_bytes, float weight_scale,
    float activation_scale, float init) {
  static const uint8_t sign_gather_indices[64] = {
      0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
      2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,
      4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5,
      6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7};
  static const uint8_t sign_masks[64] = {
      1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,
      1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,
      1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,
      1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};
  static const uint16_t high_shifts[16] = {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  const vuint8m2_t gather_indices =
      __riscv_vle8_v_u8m2(sign_gather_indices, 64);
  const vuint8m2_t masks = __riscv_vle8_v_u8m2(sign_masks, 64);
  const vuint16m1_t shifts = __riscv_vle16_v_u16m1(high_shifts, 16);
  int32_t integer_sum = 0;
  for (size_t group_pair = 0; group_pair < 4; ++group_pair) {
    const vuint8mf2_t low8 =
        __riscv_vle8_v_u8mf2(codes + group_pair * 16, 16);
    const uint16_t high_word =
        (uint16_t)high_bits[group_pair * 2] |
        ((uint16_t)high_bits[group_pair * 2 + 1] << 8);
    vuint16m1_t high = __riscv_vmv_v_x_u16m1(high_word, 16);
    high = __riscv_vand_vx_u16m1(
        __riscv_vsrl_vv_u16m1(high, shifts, 16), 1, 16);
    const vuint16m1_t low = __riscv_vwcvtu_x_x_v_u16m1(low8, 16);
    const vuint16m1_t offsets = __riscv_vor_vv_u16m1(
        __riscv_vsll_vx_u16m1(low, 2, 16),
        __riscv_vsll_vx_u16m1(high, 10, 16), 16);
    const vuint32m2_t packed_grid = __riscv_vluxei16_v_u32m2(
        (const uint32_t *)(const void *)__weft_iq3_s_grid, offsets, 16);
    const vuint8m2_t grid =
        __riscv_vreinterpret_v_u32m2_u8m2(packed_grid);
    const vuint8mf4_t packed_signs =
        __riscv_vle8_v_u8mf4(sign_bits + group_pair * 8, 8);
    const vuint8m2_t sign_source =
        __riscv_vlmul_ext_v_u8mf4_u8m2(packed_signs);
    const vuint8m2_t sign_bytes =
        __riscv_vrgather_vv_u8m2(sign_source, gather_indices, 64);
    const vbool4_t negative = __riscv_vmsne_vx_u8m2_b4(
        __riscv_vand_vv_u8m2(sign_bytes, masks, 64), 0, 64);
    const vint8m2_t q8 =
        __riscv_vle8_v_i8m2(activation + group_pair * 64, 64);
    const vint8m2_t signed_q8 =
        __riscv_vrsub_vx_i8m2_mu(negative, q8, q8, 0, 64);
    const vint16m4_t product =
        __riscv_vwmulsu_vv_i16m4(signed_q8, grid, 64);
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    const int32_t low_sum = __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m2_i32m1(
            __riscv_vget_v_i16m4_i16m2(product, 0), zero, 32));
    const int32_t high_sum = __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m2_i32m1(
            __riscv_vget_v_i16m4_i16m2(product, 1), zero, 32));
    const uint8_t packed_scale = scales[group_pair];
    integer_sum += low_sum * (1 + 2 * (packed_scale & 15)) +
                   high_sum * (1 + 2 * (packed_scale >> 4));
  }
  return init + (float)integer_sum * weight_scale * activation_scale;
}
)c";
  }
  if (strip) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_iq3_s_i8_strip(
    const uint8_t *codes, const uint8_t *high_bits,
    const uint8_t *sign_bits, const uint8_t *scales,
    const uint8_t *activation_bytes, float weight_scale,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  int8_t decoded[32];
  int32_t integer_sum = 0;
  for (size_t group = 0; group < 8; ++group) {
    for (size_t vector = 0; vector < 8; ++vector) {
      const uint16_t index = (uint16_t)codes[group * 8 + vector] |
          (uint16_t)(((uint16_t)high_bits[group] << (8 - vector)) & 0x100);
      const int8_t *grid = (const int8_t *)(const void *)&__weft_iq3_s_grid[index];
      const uint8_t signs = sign_bits[group * 4 + vector / 2];
      const size_t sign_base = (vector & 1) * 4;
      for (size_t lane = 0; lane < 4; ++lane)
        decoded[vector * 4 + lane] =
            (signs & (UINT8_C(1) << (sign_base + lane))) ? -grid[lane]
                                                         : grid[lane];
    }
    const int32_t dot = __weft_i8_dot(decoded, activation + group * 32, 32);
    const uint8_t packed_scale = scales[group / 2];
    const int32_t scale =
        1 + 2 * ((group & 1) ? (packed_scale >> 4) : (packed_scale & 15));
    integer_sum += dot * scale;
  }
  return init + (float)integer_sum * weight_scale * activation_scale;
}
)c";
  }
}

} // namespace weft::riscv_internal
