#include "RISCVQuantIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitIQ2LocalImplementations(llvm::raw_ostream &output, bool registerL32,
                             bool registerL64, bool strip) {
  if (registerL32 || registerL64) {
    output << R"c(static const uint8_t __attribute__((unused))
__weft_sign_gather_indices_64[64] = {
    0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,
    2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,
    4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5,
    6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7};
static const uint8_t __attribute__((unused))
__weft_sign_bit_masks_64[64] = {
    1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,
    1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,
    1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128,
    1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128};

)c";
  }
  if (registerL64) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_iq2_s_i8_register_l64_e8m2(
    const uint8_t *codes, const uint8_t *high_bits,
    const uint8_t *sign_bits, const uint8_t *scales,
    const uint8_t *activation_bytes, float weight_scale,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;

    const uint16_t gather_qh[8] = {0, 0, 0, 0, 1, 1, 1, 1};
    const uint16_t shift_qh[8] = {11, 9, 7, 5, 11, 9, 7, 5};
    const vuint16mf2_t gather =
        __riscv_vle16_v_u16mf2(gather_qh, 8);
    const vuint16mf2_t shifts =
        __riscv_vle16_v_u16mf2(shift_qh, 8);
    const vuint8m2_t sign_indices =
        __riscv_vle8_v_u8m2(__weft_sign_gather_indices_64, 64);
    const vuint8m2_t sign_masks =
        __riscv_vle8_v_u8m2(__weft_sign_bit_masks_64, 64);
    int32_t integer_sum = 0;
    for (size_t group = 0; group < 4; ++group) {
      const vuint8mf4_t code8 =
          __riscv_vle8_v_u8mf4(codes + group * 8, 8);
      const uint16_t high = (uint16_t)high_bits[group * 2] |
                            ((uint16_t)high_bits[group * 2 + 1] << 8);
      const vuint8mf8_t high8 =
          __riscv_vle8_v_u8mf8((const uint8_t *)(const void *)&high, 2);
      const vuint16mf4_t high16 =
          __riscv_vwcvtu_x_x_v_u16mf4(high8, 2);
      vuint16mf2_t expanded = __riscv_vrgather_vv_u16mf2(
          __riscv_vlmul_ext_v_u16mf4_u16mf2(high16), gather, 8);
      expanded = __riscv_vand_vx_u16mf2(
          __riscv_vsll_vv_u16mf2(expanded, shifts, 8), 0x1800, 8);
      vuint16mf2_t offsets = __riscv_vor_vv_u16mf2(
          __riscv_vsll_vx_u16mf2(
              __riscv_vwcvtu_x_x_v_u16mf2(code8, 8), 3, 8),
          expanded, 8);
      const vint8m2_t grid = __riscv_vreinterpret_v_u8m2_i8m2(
          __riscv_vreinterpret_v_u64m2_u8m2(
              __riscv_vluxei16_v_u64m2(
                  (const uint64_t *)(const void *)__weft_iq2_s_grid,
                  offsets, 8)));
      const vuint8mf4_t packed_signs =
          __riscv_vle8_v_u8mf4(sign_bits + group * 8, 8);
      const vuint8m2_t expanded_signs = __riscv_vrgather_vv_u8m2(
          __riscv_vlmul_ext_v_u8mf4_u8m2(packed_signs), sign_indices, 64);
      const vbool4_t negative = __riscv_vmsne_vx_u8m2_b4(
          __riscv_vand_vv_u8m2(expanded_signs, sign_masks, 64), 0, 64);
      const vint8m2_t q8 =
          __riscv_vle8_v_i8m2(activation + group * 64, 64);
      const vint8m2_t signed_q8 =
          __riscv_vrsub_vx_i8m2_mu(negative, q8, q8, 0, 64);
      const vint16m4_t product =
          __riscv_vwmul_vv_i16m4(grid, signed_q8, 64);
      const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
      const int32_t partial0 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(
              __riscv_vget_v_i16m4_i16m1(product, 0), zero, 16));
      const int32_t partial1 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(
              __riscv_vget_v_i16m4_i16m1(product, 1), zero, 16));
      const int32_t partial2 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(
              __riscv_vget_v_i16m4_i16m1(product, 2), zero, 16));
      const int32_t partial3 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(
              __riscv_vget_v_i16m4_i16m1(product, 3), zero, 16));
      const uint8_t scale0 = scales[group * 2];
      const uint8_t scale1 = scales[group * 2 + 1];
      integer_sum += partial0 * (1 + 2 * (scale0 & 15));
      integer_sum += partial1 * (1 + 2 * (scale0 >> 4));
      integer_sum += partial2 * (1 + 2 * (scale1 & 15));
      integer_sum += partial3 * (1 + 2 * (scale1 >> 4));
    }
    return init + 0.125f * (float)integer_sum *
                      weight_scale * activation_scale;
}
)c";
  }
  if (registerL32) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_iq2_s_i8_register_l32_e8m2(
    const uint8_t *codes, const uint8_t *high_bits,
    const uint8_t *sign_bits, const uint8_t *scales,
    const uint8_t *activation_bytes, float weight_scale,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  const size_t vl32 = __riscv_vsetvl_e8m2(32);
  const vuint8m2_t lane = __riscv_vid_v_u8m2(vl32);
  const vuint8m2_t sign_source_index = __riscv_vsrl_vx_u8m2(lane, 3, vl32);
  const vuint8m2_t sign_bit = __riscv_vsll_vv_u8m2(
      __riscv_vmv_v_x_u8m2(1, vl32),
      __riscv_vand_vx_u8m2(lane, 7, vl32), vl32);
  int32_t integer_sum = 0;
  for (size_t group = 0; group < 8; ++group) {
    uint16_t byte_offsets[4];
    for (size_t vector = 0; vector < 4; ++vector)
      byte_offsets[vector] = (uint16_t)((codes[group * 4 + vector] |
          (((uint16_t)high_bits[group] << (8 - 2 * vector)) & 0x300)) * 8);
    const vuint16mf2_t offsets =
        __riscv_vle16_v_u16mf2(byte_offsets, 4);
    const vuint64m2_t packed = __riscv_vluxei16_v_u64m2(
        (const uint64_t *)(const void *)__weft_iq2_s_grid, offsets, 4);
    const vint8m2_t grid = __riscv_vreinterpret_v_u8m2_i8m2(
        __riscv_vreinterpret_v_u64m2_u8m2(packed));
    const vuint8mf4_t packed_signs =
        __riscv_vle8_v_u8mf4(sign_bits + group * 4, 4);
    const vuint8m2_t expanded_signs = __riscv_vrgather_vv_u8m2(
        __riscv_vlmul_ext_v_u8mf4_u8m2(packed_signs), sign_source_index,
        vl32);
    const vbool4_t negative = __riscv_vmsne_vx_u8m2_b4(
        __riscv_vand_vv_u8m2(expanded_signs, sign_bit, vl32), 0, vl32);
    const vint8m2_t q8 = __riscv_vle8_v_i8m2(
        activation + group * 32, vl32);
    const vint8m2_t signed_q8 =
        __riscv_vrsub_vx_i8m2_mu(negative, q8, q8, 0, vl32);
    const vint16m4_t products =
        __riscv_vwmul_vv_i16m4(grid, signed_q8, vl32);
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    const int32_t first = __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m2_i32m1(
            __riscv_vget_v_i16m4_i16m2(products, 0), zero, 16));
    const int32_t second = __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m2_i32m1(
            __riscv_vget_v_i16m4_i16m2(products, 1), zero, 16));
    integer_sum += first * (1 + 2 * (scales[group] & 15));
    integer_sum += second * (1 + 2 * (scales[group] >> 4));
  }
  return init + 0.125f * (float)integer_sum * weight_scale * activation_scale;
}
)c";
  }
  if (strip) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_iq2_s_i8_strip(
    const uint8_t *codes, const uint8_t *high_bits,
    const uint8_t *sign_bits, const uint8_t *scales,
    const uint8_t *activation_bytes, float weight_scale,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  int32_t integer_sum = 0;
  int8_t decoded[32];
  for (size_t group = 0; group < 8; ++group) {
    for (size_t vector = 0; vector < 4; ++vector) {
      const uint16_t index = (uint16_t)codes[group * 4 + vector] |
          (uint16_t)(((uint16_t)high_bits[group] << (8 - 2 * vector)) & 0x300);
      const int8_t *grid =
          (const int8_t *)(const void *)&__weft_iq2_s_grid[index];
      const uint8_t signs = sign_bits[group * 4 + vector];
      for (size_t lane = 0; lane < 8; ++lane)
        decoded[vector * 8 + lane] =
            signs & (UINT8_C(1) << lane) ? -grid[lane] : grid[lane];
    }
    const int32_t first = __weft_i8_dot(decoded, activation + group * 32, 16);
    const int32_t second =
        __weft_i8_dot(decoded + 16, activation + group * 32 + 16, 16);
    integer_sum += first * (1 + 2 * (scales[group] & UINT8_C(15)));
    integer_sum += second * (1 + 2 * (scales[group] >> 4));
  }
  return init + 0.125f * (float)integer_sum * weight_scale * activation_scale;
}
)c";
  }
}

} // namespace weft::riscv_internal
