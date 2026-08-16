#include "RISCVIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitRVVLocalImplementations(llvm::raw_ostream &output,
                             bool usesRVVSymmetricI4I8,
                             bool usesRVVAffineI4I8,
                             bool usesRVVSymmetricI4I8M4,
                             bool usesRVVAffineI4I8M4,
                                 bool usesGroupedI4I8Strip,
                                 bool usesE2M1RegisterE8M1M2,
                                 bool usesE2M1RegisterE8MF2,
                                 bool usesE2M1Strip) {
  if (usesRVVSymmetricI4I8) {
    output << R"c(static inline __attribute__((always_inline, unused)) void
__weft_rvv_symmetric_i4_i8_n16_k32(
    float activation_scale, const int8_t *activation_code,
    const uint8_t *packed_weight, float *accumulator) {
  const size_t vl = __riscv_vsetvl_e8m1(16);
  vint32m4_t integer_dot = __riscv_vmv_v_x_i32m4(0, vl);
  for (size_t half = 0; half < 2; ++half) {
    const uint8_t *codes = packed_weight + 32 + half * 128;
    const int8_t *activation = activation_code + half * 16;
    for (size_t byte = 0; byte < 8; ++byte) {
      const vuint8m1_t packed =
          __riscv_vlse8_v_u8m1(codes + byte, 8, vl);
      const vuint8m1_t low =
          __riscv_vand_vx_u8m1(packed, UINT8_C(15), vl);
      const vuint8m1_t high = __riscv_vsrl_vx_u8m1(packed, 4, vl);
      const vint16m2_t low_centered = __riscv_vsub_vx_i16m2(
          __riscv_vreinterpret_v_u16m2_i16m2(
              __riscv_vzext_vf2_u16m2(low, vl)),
          8, vl);
      const vint16m2_t high_centered = __riscv_vsub_vx_i16m2(
          __riscv_vreinterpret_v_u16m2_i16m2(
              __riscv_vzext_vf2_u16m2(high, vl)),
          8, vl);
      integer_dot = __riscv_vwmacc_vx_i32m4(
          integer_dot, activation[byte], low_centered, vl);
      integer_dot = __riscv_vwmacc_vx_i32m4(
          integer_dot, activation[byte + 8], high_centered, vl);
    }
  }
  const vfloat16m2_t packed_scale = __riscv_vle16_v_f16m2(
      (const _Float16 *)(const void *)packed_weight, vl);
  const vfloat32m4_t weight_scale =
      __riscv_vfwcvt_f_f_v_f32m4(packed_scale, vl);
  vfloat32m4_t contribution =
      __riscv_vfcvt_f_x_v_f32m4(integer_dot, vl);
  contribution =
      __riscv_vfmul_vf_f32m4(contribution, activation_scale, vl);
  vfloat32m4_t result = __riscv_vle32_v_f32m4(accumulator, vl);
  result = __riscv_vfmacc_vv_f32m4(
      result, contribution, weight_scale, vl);
  __riscv_vse32_v_f32m4(accumulator, result, vl);
}

)c";
  }
  if (usesRVVAffineI4I8) {
    output << R"c(
static inline __attribute__((always_inline, unused)) void
__weft_rvv_affine_i4_i8_n16_k32(
    float activation_scale, const int8_t *activation_code,
    const uint8_t *packed_weight, float *accumulator) {
  const size_t vl = __riscv_vsetvl_e8m1(16);
  const vuint16m2_t zero_unsigned = __riscv_vzext_vf2_u16m2(
      __riscv_vle8_v_u8m1(packed_weight + 32, vl), vl);
  const vint16m2_t zero_point =
      __riscv_vreinterpret_v_u16m2_i16m2(zero_unsigned);
  vint32m4_t integer_dot = __riscv_vmv_v_x_i32m4(0, vl);
  for (size_t half = 0; half < 2; ++half) {
    const uint8_t *codes = packed_weight + 48 + half * 128;
    const int8_t *activation = activation_code + half * 16;
    for (size_t byte = 0; byte < 8; ++byte) {
      const vuint8m1_t packed =
          __riscv_vlse8_v_u8m1(codes + byte, 8, vl);
      const vuint8m1_t low =
          __riscv_vand_vx_u8m1(packed, UINT8_C(15), vl);
      const vuint8m1_t high = __riscv_vsrl_vx_u8m1(packed, 4, vl);
      const vint16m2_t low_centered = __riscv_vsub_vv_i16m2(
          __riscv_vreinterpret_v_u16m2_i16m2(
              __riscv_vzext_vf2_u16m2(low, vl)),
          zero_point, vl);
      const vint16m2_t high_centered = __riscv_vsub_vv_i16m2(
          __riscv_vreinterpret_v_u16m2_i16m2(
              __riscv_vzext_vf2_u16m2(high, vl)),
          zero_point, vl);
      integer_dot = __riscv_vwmacc_vx_i32m4(
          integer_dot, activation[byte], low_centered, vl);
      integer_dot = __riscv_vwmacc_vx_i32m4(
          integer_dot, activation[byte + 8], high_centered, vl);
    }
  }
  const vfloat16m2_t packed_scale = __riscv_vle16_v_f16m2(
      (const _Float16 *)(const void *)packed_weight, vl);
  const vfloat32m4_t weight_scale =
      __riscv_vfwcvt_f_f_v_f32m4(packed_scale, vl);
  vfloat32m4_t contribution =
      __riscv_vfcvt_f_x_v_f32m4(integer_dot, vl);
  contribution =
      __riscv_vfmul_vf_f32m4(contribution, activation_scale, vl);
  vfloat32m4_t result = __riscv_vle32_v_f32m4(accumulator, vl);
  result = __riscv_vfmacc_vv_f32m4(
      result, contribution, weight_scale, vl);
  __riscv_vse32_v_f32m4(accumulator, result, vl);
}

)c";
  }
  if (usesRVVSymmetricI4I8M4) {
    output << R"c(
static inline __attribute__((always_inline, unused)) void
__weft_rvv_symmetric_i4_i8_m4_n16_k32(
    const float *activation_scale, const int8_t *activation_code,
    const uint8_t *packed_weight, float *accumulator) {
  const size_t vl = __riscv_vsetvl_e8m1(16);
  vint32m4_t dot0 = __riscv_vmv_v_x_i32m4(0, vl);
  vint32m4_t dot1 = __riscv_vmv_v_x_i32m4(0, vl);
  vint32m4_t dot2 = __riscv_vmv_v_x_i32m4(0, vl);
  vint32m4_t dot3 = __riscv_vmv_v_x_i32m4(0, vl);
  for (size_t half = 0; half < 2; ++half) {
    const uint8_t *codes = packed_weight + 32 + half * 128;
    const size_t first_fragment = 2 * half;
    for (size_t byte = 0; byte < 8; ++byte) {
      const vuint8m1_t packed =
          __riscv_vlse8_v_u8m1(codes + byte, 8, vl);
      const vint16m2_t low = __riscv_vsub_vx_i16m2(
          __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(
              __riscv_vand_vx_u8m1(packed, UINT8_C(15), vl), vl)),
          8, vl);
      const vint16m2_t high = __riscv_vsub_vx_i16m2(
          __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(
              __riscv_vsrl_vx_u8m1(packed, 4, vl), vl)),
          8, vl);
      const size_t low_base = first_fragment * 32 + byte;
      const size_t high_base = low_base + 32;
      dot0 = __riscv_vwmacc_vx_i32m4(dot0, activation_code[low_base], low, vl);
      dot0 = __riscv_vwmacc_vx_i32m4(dot0, activation_code[high_base], high, vl);
      dot1 = __riscv_vwmacc_vx_i32m4(dot1, activation_code[low_base + 8], low, vl);
      dot1 = __riscv_vwmacc_vx_i32m4(dot1, activation_code[high_base + 8], high, vl);
      dot2 = __riscv_vwmacc_vx_i32m4(dot2, activation_code[low_base + 16], low, vl);
      dot2 = __riscv_vwmacc_vx_i32m4(dot2, activation_code[high_base + 16], high, vl);
      dot3 = __riscv_vwmacc_vx_i32m4(dot3, activation_code[low_base + 24], low, vl);
      dot3 = __riscv_vwmacc_vx_i32m4(dot3, activation_code[high_base + 24], high, vl);
    }
  }
  const vfloat32m4_t weight_scale = __riscv_vfwcvt_f_f_v_f32m4(
      __riscv_vle16_v_f16m2((const _Float16 *)(const void *)packed_weight, vl),
      vl);
#define __WEFT_RVV_ACCUMULATE_M4(ROW, DOT)                                  \
  do {                                                                       \
    vfloat32m4_t contribution = __riscv_vfcvt_f_x_v_f32m4(DOT, vl);         \
    contribution = __riscv_vfmul_vf_f32m4(                                  \
        contribution, activation_scale[ROW], vl);                            \
    vfloat32m4_t result =                                                    \
        __riscv_vle32_v_f32m4(accumulator + (ROW) * 16, vl);                 \
    result = __riscv_vfmacc_vv_f32m4(result, contribution, weight_scale, vl);\
    __riscv_vse32_v_f32m4(accumulator + (ROW) * 16, result, vl);             \
  } while (0)
  __WEFT_RVV_ACCUMULATE_M4(0, dot0);
  __WEFT_RVV_ACCUMULATE_M4(1, dot1);
  __WEFT_RVV_ACCUMULATE_M4(2, dot2);
  __WEFT_RVV_ACCUMULATE_M4(3, dot3);
#undef __WEFT_RVV_ACCUMULATE_M4
}

)c";
  }
  if (usesRVVAffineI4I8M4) {
    output << R"c(
static inline __attribute__((always_inline, unused)) void
__weft_rvv_affine_i4_i8_m4_n16_k32(
    const float *activation_scale, const int8_t *activation_code,
    const uint8_t *packed_weight, float *accumulator) {
  const size_t vl = __riscv_vsetvl_e8m1(16);
  const vint16m2_t zero_point = __riscv_vreinterpret_v_u16m2_i16m2(
      __riscv_vzext_vf2_u16m2(
          __riscv_vle8_v_u8m1(packed_weight + 32, vl), vl));
  vint32m4_t dot0 = __riscv_vmv_v_x_i32m4(0, vl);
  vint32m4_t dot1 = __riscv_vmv_v_x_i32m4(0, vl);
  vint32m4_t dot2 = __riscv_vmv_v_x_i32m4(0, vl);
  vint32m4_t dot3 = __riscv_vmv_v_x_i32m4(0, vl);
  for (size_t half = 0; half < 2; ++half) {
    const uint8_t *codes = packed_weight + 48 + half * 128;
    const size_t first_fragment = 2 * half;
    for (size_t byte = 0; byte < 8; ++byte) {
      const vuint8m1_t packed =
          __riscv_vlse8_v_u8m1(codes + byte, 8, vl);
      const vint16m2_t low = __riscv_vsub_vv_i16m2(
          __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(
              __riscv_vand_vx_u8m1(packed, UINT8_C(15), vl), vl)),
          zero_point, vl);
      const vint16m2_t high = __riscv_vsub_vv_i16m2(
          __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vzext_vf2_u16m2(
              __riscv_vsrl_vx_u8m1(packed, 4, vl), vl)),
          zero_point, vl);
      const size_t low_base = first_fragment * 32 + byte;
      const size_t high_base = low_base + 32;
      dot0 = __riscv_vwmacc_vx_i32m4(dot0, activation_code[low_base], low, vl);
      dot0 = __riscv_vwmacc_vx_i32m4(dot0, activation_code[high_base], high, vl);
      dot1 = __riscv_vwmacc_vx_i32m4(dot1, activation_code[low_base + 8], low, vl);
      dot1 = __riscv_vwmacc_vx_i32m4(dot1, activation_code[high_base + 8], high, vl);
      dot2 = __riscv_vwmacc_vx_i32m4(dot2, activation_code[low_base + 16], low, vl);
      dot2 = __riscv_vwmacc_vx_i32m4(dot2, activation_code[high_base + 16], high, vl);
      dot3 = __riscv_vwmacc_vx_i32m4(dot3, activation_code[low_base + 24], low, vl);
      dot3 = __riscv_vwmacc_vx_i32m4(dot3, activation_code[high_base + 24], high, vl);
    }
  }
  const vfloat32m4_t weight_scale = __riscv_vfwcvt_f_f_v_f32m4(
      __riscv_vle16_v_f16m2((const _Float16 *)(const void *)packed_weight, vl),
      vl);
#define __WEFT_RVV_AFFINE_ACCUMULATE_M4(ROW, DOT)                           \
  do {                                                                       \
    vfloat32m4_t contribution = __riscv_vfcvt_f_x_v_f32m4(DOT, vl);         \
    contribution = __riscv_vfmul_vf_f32m4(                                  \
        contribution, activation_scale[ROW], vl);                            \
    vfloat32m4_t result =                                                    \
        __riscv_vle32_v_f32m4(accumulator + (ROW) * 16, vl);                 \
    result = __riscv_vfmacc_vv_f32m4(result, contribution, weight_scale, vl);\
    __riscv_vse32_v_f32m4(accumulator + (ROW) * 16, result, vl);             \
  } while (0)
  __WEFT_RVV_AFFINE_ACCUMULATE_M4(0, dot0);
  __WEFT_RVV_AFFINE_ACCUMULATE_M4(1, dot1);
  __WEFT_RVV_AFFINE_ACCUMULATE_M4(2, dot2);
  __WEFT_RVV_AFFINE_ACCUMULATE_M4(3, dot3);
#undef __WEFT_RVV_AFFINE_ACCUMULATE_M4
}

)c";
  }
  if (usesE2M1RegisterE8M1M2 || usesE2M1RegisterE8MF2 || usesE2M1Strip) {
    output << R"c(static const int8_t __attribute__((unused))
__weft_e2m1_doubled[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

static inline __attribute__((always_inline, unused)) float
__weft_e8m0_half(uint8_t exponent) {
  const uint32_t bits = exponent < UINT8_C(2)
                            ? (UINT32_C(0x00200000) << exponent)
                            : ((uint32_t)(exponent - UINT8_C(1)) << 23);
  return __weft_bitcast_u32_f32(bits);
}

)c";
  }
  if (usesE2M1Strip) {
    output << R"c(
static inline __attribute__((always_inline, unused)) int32_t
__weft_local_i8_dot(const int8_t *lhs, const int8_t *rhs, size_t count) {
  int32_t result = 0;
  size_t offset = 0;
  while (offset < count) {
    const size_t vl = __riscv_vsetvl_e8m1(count - offset);
    const vint8m1_t left = __riscv_vle8_v_i8m1(lhs + offset, vl);
    const vint8m1_t right = __riscv_vle8_v_i8m1(rhs + offset, vl);
    const vint16m2_t products = __riscv_vwmul_vv_i16m2(left, right, vl);
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    result += __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m2_i32m1(products, zero, vl));
    offset += vl;
  }
  return result;
}

)c";
  }
  if (usesE2M1RegisterE8M1M2) {
    output << R"c(
static inline __attribute__((always_inline, unused)) float
__weft_e2m1_e8m0_i8_register_e8m1_e8m2(
    const uint8_t *packed_codes, uint8_t exponent,
    const uint8_t *activation_bytes, float activation_scale, float init) {
  const size_t vl16 = __riscv_vsetvl_e8m1(16);
  const vint8m2_t table =
      __riscv_vle8_v_i8m2(__weft_e2m1_doubled, vl16);
  const vuint8m1_t packed =
      __riscv_vle8_v_u8m1(packed_codes, vl16);
  const vuint8m1_t low =
      __riscv_vand_vx_u8m1(packed, UINT8_C(15), vl16);
  const vuint8m1_t high = __riscv_vsrl_vx_u8m1(packed, 4, vl16);
  const vuint8m2_t indices =
      __riscv_vcreate_v_u8m1_u8m2(low, high);
  const size_t vl32 = __riscv_vsetvl_e8m2(32);
  const vint8m2_t activation = __riscv_vle8_v_i8m2(
      (const int8_t *)(const void *)activation_bytes, vl32);
  const vint8m2_t decoded =
      __riscv_vrgather_vv_i8m2(table, indices, vl32);
  const vint16m4_t products =
      __riscv_vwmul_vv_i16m4(activation, decoded, vl32);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const vint32m1_t reduced =
      __riscv_vwredsum_vs_i16m4_i32m1(products, zero, vl32);
  const int32_t dot = __riscv_vmv_x_s_i32m1_i32(reduced);
  return init + (float)dot * __weft_e8m0_half(exponent) * activation_scale;
}

)c";
  }
  if (usesE2M1RegisterE8MF2) {
    output << R"c(
static inline __attribute__((always_inline, unused)) float
__weft_e2m1_e8m0_i8_register_e8mf2(
    const uint8_t *packed_codes, uint8_t exponent,
    const uint8_t *activation_bytes, float activation_scale, float init) {
  const size_t vl = __riscv_vsetvl_e8mf2(16);
  const vint8mf2_t table =
      __riscv_vle8_v_i8mf2(__weft_e2m1_doubled, vl);
  const vuint8mf2_t packed =
      __riscv_vle8_v_u8mf2(packed_codes, vl);
  const vuint8mf2_t low =
      __riscv_vand_vx_u8mf2(packed, UINT8_C(15), vl);
  const vuint8mf2_t high = __riscv_vsrl_vx_u8mf2(packed, 4, vl);
  const vint8mf2_t decoded_low =
      __riscv_vrgather_vv_i8mf2(table, low, vl);
  const vint8mf2_t decoded_high =
      __riscv_vrgather_vv_i8mf2(table, high, vl);
  const int8_t *activation =
      (const int8_t *)(const void *)activation_bytes;
  const vint8mf2_t activation_low =
      __riscv_vle8_v_i8mf2(activation, vl);
  const vint8mf2_t activation_high =
      __riscv_vle8_v_i8mf2(activation + 16, vl);
  vint16m1_t products =
      __riscv_vwmul_vv_i16m1(activation_low, decoded_low, vl);
  products = __riscv_vwmacc_vv_i16m1(
      products, activation_high, decoded_high, vl);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const vint32m1_t reduced =
      __riscv_vwredsum_vs_i16m1_i32m1(products, zero, vl);
  const int32_t dot = __riscv_vmv_x_s_i32m1_i32(reduced);
  return init + (float)dot * __weft_e8m0_half(exponent) * activation_scale;
}

)c";
  }
  if (usesE2M1Strip) {
    output << R"c(
static inline __attribute__((always_inline, unused)) float
__weft_e2m1_e8m0_i8_strip(
    const uint8_t *packed_codes, uint8_t exponent,
    const uint8_t *activation_bytes, float activation_scale, float init) {
  int8_t decoded[32];
  for (size_t byte = 0; byte < 16; ++byte) {
    decoded[byte] = __weft_e2m1_doubled[packed_codes[byte] & UINT8_C(15)];
    decoded[16 + byte] = __weft_e2m1_doubled[packed_codes[byte] >> 4];
  }
  const int32_t dot = __weft_local_i8_dot(
      decoded, (const int8_t *)(const void *)activation_bytes, 32);
  return init + (float)dot * __weft_e8m0_half(exponent) * activation_scale;
}

)c";
  }
  if (usesGroupedI4I8Strip) {
    output << R"c(
static inline __attribute__((always_inline, unused)) int32_t
__weft_grouped_nibble_i8_dot(
    const uint8_t *packed, const int8_t *activation, bool high) {
  int32_t result = 0;
  size_t offset = 0;
  while (offset < 32) {
    const size_t vl = __riscv_vsetvl_e8m1(32 - offset);
    const vuint8m1_t bytes = __riscv_vle8_v_u8m1(packed + offset, vl);
    const vuint8m1_t fields = high
        ? __riscv_vsrl_vx_u8m1(bytes, 4, vl)
        : __riscv_vand_vx_u8m1(bytes, UINT8_C(15), vl);
    const vint8m1_t codes = __riscv_vreinterpret_v_u8m1_i8m1(fields);
    const vint8m1_t values =
        __riscv_vle8_v_i8m1(activation + offset, vl);
    const vint16m2_t products =
        __riscv_vwmul_vv_i16m2(codes, values, vl);
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    result += __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m2_i32m1(products, zero, vl));
    offset += vl;
  }
  return result;
}

static inline __attribute__((always_inline, unused)) float
__weft_grouped_affine_i4_i8_strip(
    const uint8_t *packed_weight, const uint8_t *scale_min,
    const uint8_t *activation_bytes, const uint8_t *activation_sum_bytes,
    float dot_scale, float minimum_scale, float init) {
  uint8_t scale[8];
  uint8_t minimum[8];
  for (size_t group = 0; group < 4; ++group) {
    scale[group] = scale_min[group] & UINT8_C(63);
    minimum[group] = scale_min[group + 4] & UINT8_C(63);
    scale[group + 4] = (scale_min[group + 8] & UINT8_C(15)) |
                       ((scale_min[group] >> 6) << 4);
    minimum[group + 4] = (scale_min[group + 8] >> 4) |
                         ((scale_min[group + 4] >> 6) << 4);
  }
  const int8_t *activation =
      (const int8_t *)(const void *)activation_bytes;
  const int16_t *activation_sum =
      (const int16_t *)(const void *)activation_sum_bytes;
  int32_t weighted_dot = 0;
  int32_t weighted_minimum = 0;
  for (size_t group = 0; group < 8; ++group) {
    const uint8_t *packed = packed_weight + (group / 2) * 32;
    const bool high = (group & 1) != 0;
    weighted_dot += scale[group] * __weft_grouped_nibble_i8_dot(
        packed, activation + group * 32, high);
    weighted_minimum += minimum[group] *
                        (activation_sum[2 * group] +
                         activation_sum[2 * group + 1]);
  }
  return init + dot_scale * (float)weighted_dot -
         minimum_scale * (float)weighted_minimum;
}

)c";
  }
}

} // namespace weft::riscv_internal
