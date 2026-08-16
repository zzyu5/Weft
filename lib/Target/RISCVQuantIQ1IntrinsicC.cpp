#include "RISCVQuantIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitIQ1IntrinsicCLeaves(llvm::raw_ostream &output, bool fixedLanes32,
                             bool fixedLanes64, bool scalable) {
  if (fixedLanes32) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_iq1_m_i8_lanes32(
    const uint8_t *codes, const uint8_t *high_delta_bits,
    const uint8_t *scales, const uint8_t *activation_bytes,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  const uint16_t *packed_scales = (const uint16_t *)(const void *)scales;
  const uint16_t scale_bits =
      (packed_scales[0] >> 12) | ((packed_scales[1] >> 8) & 0x00f0) |
      ((packed_scales[2] >> 4) & 0x0f00) | (packed_scales[3] & 0xf000);
  const float block_scale = (float)__weft_bitcast_u16_f16(scale_bits);
  vint32m4_t grid_accumulator = __riscv_vmv_v_x_i32m4(0, 16);
  vint32m4_t delta_accumulator = __riscv_vmv_v_x_i32m4(0, 16);
  const uint16_t index_shift_values[16] = {
      8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4, 8, 4};
  const vuint16m2_t index_shifts =
      __riscv_vle16_v_u16m2(index_shift_values, 16);
  const uint16_t delta_mask_values[16] = {
      0x08, 0x80, 0x08, 0x80, 0x08, 0x80, 0x08, 0x80,
      0x08, 0x80, 0x08, 0x80, 0x08, 0x80, 0x08, 0x80};
  const vuint16m2_t delta_masks =
      __riscv_vle16_v_u16m2(delta_mask_values, 16);
  for (size_t half = 0; half < 2; ++half) {
    const vuint8mf2_t high8 =
        __riscv_vle8_v_u8mf2(high_delta_bits + half * 8, 8);
    const vuint16m1_t high_low = __riscv_vzext_vf2_u16m1(high8, 8);
    const vuint16m1_t high_high =
        __riscv_vsll_vx_u16m1(high_low, 8, 8);
    const vuint16m2_t high = __riscv_vzext_vf2_u16m2(
        __riscv_vreinterpret_v_u16m1_u8m1(
            __riscv_vor_vv_u16m1(high_low, high_high, 8)), 16);
    const vuint16m2_t low = __riscv_vzext_vf2_u16m2(
        __riscv_vle8_v_u8m1(codes + half * 16, 16), 16);
    vuint16m2_t indices = __riscv_vor_vv_u16m2(
        low,
        __riscv_vand_vx_u16m2(
            __riscv_vsll_vv_u16m2(high, index_shifts, 16), 0x700, 16),
        16);
    indices = __riscv_vsll_vx_u16m2(indices, 3, 16);
    const vbool8_t negative = __riscv_vmsgtu_vx_u16m2_b8(
        __riscv_vand_vv_u16m2(high, delta_masks, 16), 0, 16);
    const vint64m8_t positive =
        __riscv_vmv_v_x_i64m8(INT64_C(0x0101010101010101), 16);
    const vint8m8_t delta = __riscv_vreinterpret_v_i64m8_i8m8(
        __riscv_vmerge_vxm_i64m8(positive, INT64_C(-1), negative, 16));
    for (size_t quarter = 0; quarter < 2; ++quarter) {
      const vint8m4_t grid = __riscv_vreinterpret_v_i64m4_i8m4(
          __riscv_vreinterpret_v_u64m4_i64m4(
              __riscv_vluxei16_v_u64m4(
                  (const uint64_t *)(const void *)__weft_iq1_m_grid,
                  __weft_get_u16m2_u16m1(indices, quarter), 8)));
      for (size_t pair = 0; pair < 2; ++pair) {
        const size_t segment = quarter * 2 + pair;
        const vint8m2_t q8 = __riscv_vle8_v_i8m2(
            activation + half * 128 + segment * 32, 32);
        const vint16m4_t grid_product = __riscv_vwmul_vv_i16m4(
            __weft_get_i8m4_i8m2(grid, pair), q8, 32);
        const vint16m4_t delta_product = __riscv_vwmul_vv_i16m4(
            __weft_get_i8m8_i8m2(delta, segment), q8, 32);
        const uint16_t scale_word = packed_scales[half * 2 + segment / 2];
        const unsigned shift = 6 * (segment % 2);
        const int16_t first_scale = 1 + 2 * ((scale_word >> shift) & 7);
        const int16_t second_scale =
            1 + 2 * ((scale_word >> (shift + 3)) & 7);
        grid_accumulator = __riscv_vwmacc_vx_i32m4(
            grid_accumulator, first_scale,
            __riscv_vget_v_i16m4_i16m2(grid_product, 0), 16);
        grid_accumulator = __riscv_vwmacc_vx_i32m4(
            grid_accumulator, second_scale,
            __riscv_vget_v_i16m4_i16m2(grid_product, 1), 16);
        delta_accumulator = __riscv_vwmacc_vx_i32m4(
            delta_accumulator, first_scale,
            __riscv_vget_v_i16m4_i16m2(delta_product, 0), 16);
        delta_accumulator = __riscv_vwmacc_vx_i32m4(
            delta_accumulator, second_scale,
            __riscv_vget_v_i16m4_i16m2(delta_product, 1), 16);
      }
    }
  }
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t grid_sum = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vredsum_vs_i32m4_i32m1(grid_accumulator, zero, 16));
  const int32_t delta_sum = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vredsum_vs_i32m4_i32m1(delta_accumulator, zero, 16));
  return init + block_scale * activation_scale *
                    ((float)grid_sum + 0.125f * (float)delta_sum);
}
)c";
  }
  if (fixedLanes64) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_iq1_m_i8_lanes64(
    const uint8_t *codes, const uint8_t *high_delta_bits,
    const uint8_t *scales, const uint8_t *activation_bytes,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  const uint16_t scale_words[4] = {
      (uint16_t)scales[0] | ((uint16_t)scales[1] << 8),
      (uint16_t)scales[2] | ((uint16_t)scales[3] << 8),
      (uint16_t)scales[4] | ((uint16_t)scales[5] << 8),
      (uint16_t)scales[6] | ((uint16_t)scales[7] << 8)};
  const uint16_t scale_bits =
      (scale_words[0] >> 12) | ((scale_words[1] >> 8) & 0x00f0) |
      ((scale_words[2] >> 4) & 0x0f00) | (scale_words[3] & 0xf000);
  vint32m2_t grid_accumulator = __riscv_vmv_v_x_i32m2(0, 16);
  vint32m2_t delta_accumulator = __riscv_vmv_v_x_i32m2(0, 16);
  for (size_t half = 0; half < 2; ++half) {
    const vuint8mf4_t high8 =
        __riscv_vle8_v_u8mf4(high_delta_bits + half * 8, 8);
    const vuint16mf2_t high16 = __riscv_vzext_vf2_u16mf2(high8, 8);
    const vuint16mf2_t high16_shifted =
        __riscv_vsll_vx_u16mf2(high16, 8, 8);
    const vuint16m1_t high = __riscv_vzext_vf2_u16m1(
        __riscv_vreinterpret_v_u16mf2_u8mf2(
            __riscv_vor_vv_u16mf2(high16, high16_shifted, 8)),
        16);
    const vuint16m1_t low = __riscv_vzext_vf2_u16m1(
        __riscv_vle8_v_u8mf2(codes + half * 16, 16), 16);
    const vuint16m1_t shifts = __riscv_vreinterpret_v_u32m1_u16m1(
        __riscv_vmv_v_x_u32m1(UINT32_C(0x00040008), 8));
    vuint16m1_t indices = __riscv_vor_vv_u16m1(
        low,
        __riscv_vand_vx_u16m1(
            __riscv_vsll_vv_u16m1(high, shifts, 16), UINT16_C(0x700), 16),
        16);
    indices = __riscv_vsll_vx_u16m1(indices, 3, 16);
    const vint8m4_t grid = __riscv_vreinterpret_v_i64m4_i8m4(
        __riscv_vreinterpret_v_u64m4_i64m4(__riscv_vluxei16_v_u64m4(
            (const uint64_t *)(const void *)__weft_iq1_m_grid, indices, 16)));
    const vuint16m1_t delta_masks = __riscv_vreinterpret_v_u32m1_u16m1(
        __riscv_vmv_v_x_u32m1(UINT32_C(0x00800008), 8));
    const vbool16_t negative = __riscv_vmsgtu_vx_u16m1_b16(
        __riscv_vand_vv_u16m1(high, delta_masks, 16), 0, 16);
    const vint64m4_t positive =
        __riscv_vmv_v_x_i64m4(INT64_C(0x0101010101010101), 16);
    const vint8m4_t delta = __riscv_vreinterpret_v_i64m4_i8m4(
        __riscv_vmerge_vxm_i64m4(positive, INT64_C(-1), negative, 16));
    const vint8m4_t q8 =
        __riscv_vle8_v_i8m4(activation + half * 128, 128);
    const vint16m8_t grid_product =
        __riscv_vwmul_vv_i16m8(grid, q8, 128);
    const vint16m8_t delta_product =
        __riscv_vwmul_vv_i16m8(delta, q8, 128);
    const uint16_t first_word = scale_words[half * 2];
    const uint16_t second_word = scale_words[half * 2 + 1];
    const int16_t scale0 = 1 + 2 * ((first_word >> 0) & 7);
    const int16_t scale1 = 1 + 2 * ((first_word >> 3) & 7);
    const int16_t scale2 = 1 + 2 * ((first_word >> 6) & 7);
    const int16_t scale3 = 1 + 2 * ((first_word >> 9) & 7);
    const int16_t scale4 = 1 + 2 * ((second_word >> 0) & 7);
    const int16_t scale5 = 1 + 2 * ((second_word >> 3) & 7);
    const int16_t scale6 = 1 + 2 * ((second_word >> 6) & 7);
    const int16_t scale7 = 1 + 2 * ((second_word >> 9) & 7);
#define __WEFT_IQ1_M_ACCUMULATE(SEGMENT, SCALE)                         \
    grid_accumulator = __riscv_vwmacc_vx_i32m2(                        \
        grid_accumulator, SCALE,                                        \
        __riscv_vget_v_i16m8_i16m1(grid_product, SEGMENT), 16);        \
    delta_accumulator = __riscv_vwmacc_vx_i32m2(                       \
        delta_accumulator, SCALE,                                       \
        __riscv_vget_v_i16m8_i16m1(delta_product, SEGMENT), 16)
    __WEFT_IQ1_M_ACCUMULATE(0, scale0);
    __WEFT_IQ1_M_ACCUMULATE(1, scale1);
    __WEFT_IQ1_M_ACCUMULATE(2, scale2);
    __WEFT_IQ1_M_ACCUMULATE(3, scale3);
    __WEFT_IQ1_M_ACCUMULATE(4, scale4);
    __WEFT_IQ1_M_ACCUMULATE(5, scale5);
    __WEFT_IQ1_M_ACCUMULATE(6, scale6);
    __WEFT_IQ1_M_ACCUMULATE(7, scale7);
#undef __WEFT_IQ1_M_ACCUMULATE
  }
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t grid_sum = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vredsum_vs_i32m2_i32m1(grid_accumulator, zero, 16));
  const int32_t delta_sum = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vredsum_vs_i32m2_i32m1(delta_accumulator, zero, 16));
  return init + (float)__weft_bitcast_u16_f16(scale_bits) *
                    activation_scale *
                    ((float)grid_sum + 0.125f * (float)delta_sum);
}
)c";
  }
  if (scalable) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_iq1_m_i8_rvv(
    const uint8_t *codes, const uint8_t *high_delta_bits,
    const uint8_t *scales, const uint8_t *activation_bytes,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  uint16_t scale_words[4];
  for (size_t word = 0; word < 4; ++word)
    scale_words[word] = (uint16_t)scales[2 * word] |
                        ((uint16_t)scales[2 * word + 1] << 8);
  const uint16_t scale_bits =
      (scale_words[0] >> 12) | ((scale_words[1] >> 8) & 0x00f0) |
      ((scale_words[2] >> 4) & 0x0f00) | (scale_words[3] & 0xf000);
  int32_t grid_sum = 0;
  int32_t delta_sum = 0;
  int8_t grid_values[16];
  int8_t delta_values[16];
  for (size_t group = 0; group < 8; ++group) {
    for (size_t vector = 0; vector < 4; ++vector) {
      const uint8_t high = high_delta_bits[group * 2 + vector / 2];
      const uint16_t index = (uint16_t)codes[group * 4 + vector] |
          (uint16_t)(((uint16_t)high << (8 - 4 * (vector & 1))) & 0x700);
      const int8_t *grid =
          (const int8_t *)(const void *)&__weft_iq1_m_grid[index];
      const int8_t delta =
          high & ((vector & 1) ? UINT8_C(0x80) : UINT8_C(0x08)) ? -1 : 1;
      for (size_t lane = 0; lane < 8; ++lane) {
        const size_t local = (vector % 2) * 8 + lane;
        grid_values[local] = grid[lane];
        delta_values[local] = delta;
      }
      if ((vector & 1) != 0) {
        const size_t half = vector / 2;
        const int32_t grid_dot = __weft_i8_dot(
            grid_values, activation + group * 32 + half * 16, 16);
        const int32_t delta_dot = __weft_i8_dot(
            delta_values, activation + group * 32 + half * 16, 16);
        const unsigned shift = 6 * (group & 1);
        const int32_t scale = 1 + 2 *
            ((scale_words[group / 2] >> (shift + half * 3)) & 7);
        grid_sum += grid_dot * scale;
        delta_sum += delta_dot * scale;
      }
    }
  }
  return init + __weft_bitcast_u16_f16(scale_bits) * activation_scale *
                    ((float)grid_sum + 0.125f * (float)delta_sum);
}
)c";
  }
}

} // namespace weft::riscv_internal
