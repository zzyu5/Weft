#include "RISCVIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitRVVLocalImplementations(llvm::raw_ostream &output,
    const std::optional<LocalMicrokernelSchedule> &symmetricI4I8,
    const std::optional<LocalMicrokernelSchedule> &affineI4I8,
    const std::optional<LocalMicrokernelSchedule> &symmetricI4I8M4,
    const std::optional<LocalMicrokernelSchedule> &affineI4I8M4,
                                 bool usesGroupedI4I8RegisterL16,
                                 bool usesGroupedI4I8RegisterL32,
                                 bool usesGroupedI4I8Strip,
                                 bool usesE2M1RegisterE8M1M2,
                                 bool usesE2M1RegisterE8MF2,
                                 bool usesE2M1Strip) {
  const bool usesRVVSymmetricI4I8 = symmetricI4I8.has_value();
  const bool usesRVVAffineI4I8 = affineI4I8.has_value();
  const bool usesRVVSymmetricI4I8M4 = symmetricI4I8M4.has_value();
  const bool usesRVVAffineI4I8M4 = affineI4I8M4.has_value();
  if (usesRVVSymmetricI4I8) {
    output << R"c(static inline __attribute__((always_inline, unused)) void
__weft_rvv_symmetric_i4_i8_n16_k32(
    float activation_scale, const int8_t *activation_code,
    const uint8_t *packed_weight, float *accumulator) {
  const size_t vl = __riscv_vsetvl_e8m1(16);
  vint32m4_t integer_dot = __riscv_vmv_v_x_i32m4(0, vl);
  for (size_t half = 0; half < )c"
           << symmetricI4I8->decode.chunksPerStep << R"c(; ++half) {
    const uint8_t *codes = packed_weight + 32 + half * 128;
    const int8_t *activation = activation_code + half * 16;
    for (size_t byte = 0; byte < )c"
           << symmetricI4I8->decode.groupsPerChunk / 2 << R"c(; ++byte) {
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
  for (size_t half = 0; half < )c"
           << affineI4I8->decode.chunksPerStep << R"c(; ++half) {
    const uint8_t *codes = packed_weight + 48 + half * 128;
    const int8_t *activation = activation_code + half * 16;
    for (size_t byte = 0; byte < )c"
           << affineI4I8->decode.groupsPerChunk / 2 << R"c(; ++byte) {
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
  for (size_t half = 0; half < )c"
           << symmetricI4I8M4->decode.chunksPerStep << R"c(; ++half) {
    const uint8_t *codes = packed_weight + 32 + half * 128;
    const size_t first_fragment = 2 * half;
    for (size_t byte = 0; byte < )c"
           << symmetricI4I8M4->decode.groupsPerChunk / 2 << R"c(; ++byte) {
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
  for (size_t half = 0; half < )c"
           << affineI4I8M4->decode.chunksPerStep << R"c(; ++half) {
    const uint8_t *codes = packed_weight + 48 + half * 128;
    const size_t first_fragment = 2 * half;
    for (size_t byte = 0; byte < )c"
           << affineI4I8M4->decode.groupsPerChunk / 2 << R"c(; ++byte) {
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
  const size_t vl = __riscv_vsetvl_e8m1(16);
  const vint8m1_t table =
      __riscv_vle8_v_i8m1(__weft_e2m1_doubled, vl);
  const vuint8m1_t packed =
      __riscv_vle8_v_u8m1(packed_codes, vl);
  const vuint8m1_t low =
      __riscv_vand_vx_u8m1(packed, UINT8_C(15), vl);
  const vuint8m1_t high = __riscv_vsrl_vx_u8m1(packed, 4, vl);
  const vint8m1_t decoded_low =
      __riscv_vrgather_vv_i8m1(table, low, vl);
  const vint8m1_t decoded_high =
      __riscv_vrgather_vv_i8m1(table, high, vl);
  const int8_t *activation =
      (const int8_t *)(const void *)activation_bytes;
  const vint8m1_t activation_low =
      __riscv_vle8_v_i8m1(activation, vl);
  const vint8m1_t activation_high =
      __riscv_vle8_v_i8m1(activation + 16, vl);
  vint16m2_t products =
      __riscv_vwmul_vv_i16m2(activation_low, decoded_low, vl);
  products = __riscv_vwmacc_vv_i16m2(
      products, activation_high, decoded_high, vl);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const vint32m1_t reduced =
      __riscv_vwredsum_vs_i16m2_i32m1(products, zero, vl);
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
  if (usesGroupedI4I8RegisterL16) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_grouped_affine_i4_i8_register_l16_e8m1(
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

  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  float sum = init;
  float temporary;
  float second;
  const uint8_t *q40;
  const uint8_t *q41;
  const uint8_t *q42;
  const uint8_t *q43;
  const int8_t *q80;
  const int8_t *q81;
  const int8_t *q82;
  const int8_t *q83;
  int s0;
  int s1;
  int s2;
  int s3;

  __asm__ volatile(
      "vsetivli zero, 4, e32, m1, ta, ma\n\t"
      "vmv.v.x v16, zero\n\t"
      "vsetivli zero, 8, e16, m1, ta, ma\n\t"
      "vle32.v v2, (%[bsums])\n\t"
      "vnsrl.wi v0, v2, 0\n\t"
      "vnsrl.wi v1, v2, 16\n\t"
      "vadd.vv v2, v0, v1\n\t"
      "vle8.v v3, (%[mins])\n\t"
      "vzext.vf2 v4, v3\n\t"
      "vwmul.vv v6, v4, v2\n\t"
      "vsetivli zero, 4, e32, m1, ta, ma\n\t"
      "vredsum.vs v0, v6, v16\n\t"
      "vredsum.vs v0, v7, v0\n\t"
      "vfcvt.f.x.v v0, v0\n\t"
      "vfmv.f.s %[temporary], v0\n\t"
      "vsetivli zero, 16, e8, m1, ta, ma\n\t"
      "vle8.v v0, (%[packed])\n\t"
      "fnmsub.s %[sum], %[minimum_scale], %[temporary], %[sum]\n\t"
      "addi %[q40], %[packed], 64\n\t"
      "addi %[q41], %[packed], 16\n\t"
      "addi %[q42], %[packed], 32\n\t"
      "addi %[q43], %[packed], 48\n\t"
      "addi %[q80], %[activation], 64\n\t"
      "vle8.v v1, (%[q41])\n\t"
      "vle8.v v2, (%[q42])\n\t"
      "addi %[q81], %[activation], 16\n\t"
      "addi %[q41], %[q41], 64\n\t"
      "addi %[q82], %[activation], 32\n\t"
      "vle8.v v3, (%[q43])\n\t"
      "vle8.v v8, (%[activation])\n\t"
      "addi %[q42], %[q42], 64\n\t"
      "addi %[q83], %[activation], 48\n\t"
      "addi %[q43], %[q43], 64\n\t"
      "vsrl.vi v4, v0, 4\n\t"
      "vle8.v v9, (%[q81])\n\t"
      "vle8.v v10, (%[q82])\n\t"
      "vand.vi v0, v0, 0xF\n\t"
      "addi %[q81], %[q81], 64\n\t"
      "vsrl.vi v5, v1, 4\n\t"
      "addi %[q82], %[q82], 64\n\t"
      "vle8.v v11, (%[q83])\n\t"
      "vle8.v v12, (%[q80])\n\t"
      "vand.vi v1, v1, 0xF\n\t"
      "addi %[q83], %[q83], 64\n\t"
      "vsrl.vi v6, v2, 4\n\t"
      "addi %[q80], %[q80], 64\n\t"
      "vle8.v v13, (%[q81])\n\t"
      "vle8.v v14, (%[q82])\n\t"
      "vand.vi v2, v2, 0xF\n\t"
      "addi %[q81], %[q81], 64\n\t"
      "vsrl.vi v7, v3, 4\n\t"
      "addi %[q82], %[q82], 64\n\t"
      "vwmul.vv v16, v0, v8\n\t"
      "vle8.v v15, (%[q83])\n\t"
      "vle8.v v0, (%[q40])\n\t"
      "vand.vi v3, v3, 0xF\n\t"
      "addi %[q83], %[q83], 64\n\t"
      "vwmul.vv v24, v2, v12\n\t"
      "vwmul.vv v20, v4, v10\n\t"
      "vwmul.vv v28, v6, v14\n\t"
      "vwmacc.vv v16, v1, v9\n\t"
      "vle8.v v1, (%[q41])\n\t"
      "vle8.v v2, (%[q42])\n\t"
      "vwmacc.vv v24, v3, v13\n\t"
      "vwmacc.vv v20, v5, v11\n\t"
      "vwmacc.vv v28, v7, v15\n\t"
      "addi %[q40], %[q80], 64\n\t"
      "addi %[q41], %[q81], 64\n\t"
      "vle8.v v3, (%[q43])\n\t"
      "vle8.v v8, (%[q80])\n\t"
      "addi %[q42], %[q82], 64\n\t"
      "addi %[q43], %[q83], 64\n\t"
      "vsrl.vi v4, v0, 4\n\t"
      "vle8.v v9, (%[q81])\n\t"
      "vle8.v v10, (%[q82])\n\t"
      "vand.vi v0, v0, 0xF\n\t"
      "vsrl.vi v5, v1, 4\n\t"
      "vsrl.vi v7, v3, 4\n\t"
      "vand.vi v3, v3, 0xF\n\t"
      "vle8.v v11, (%[q83])\n\t"
      "vle8.v v12, (%[q40])\n\t"
      "vand.vi v1, v1, 0xF\n\t"
      "vsrl.vi v6, v2, 4\n\t"
      "vand.vi v2, v2, 0xF\n\t"
      "vwmul.vv v18, v0, v8\n\t"
      "vle8.v v13, (%[q41])\n\t"
      "vle8.v v14, (%[q42])\n\t"
      "vwmul.vv v26, v2, v12\n\t"
      "vwmul.vv v22, v4, v10\n\t"
      "vwmul.vv v30, v6, v14\n\t"
      "vwmacc.vv v18, v1, v9\n\t"
      "vle8.v v15, (%[q43])\n\t"
      "vwmacc.vv v26, v3, v13\n\t"
      "vwmacc.vv v22, v5, v11\n\t"
      "vwmacc.vv v30, v7, v15\n\t"
      "vmv.v.x v0, zero\n\t"
      "vsetivli zero, 16, e16, m2, ta, ma\n\t"
      "vwredsum.vs v4, v16, v0\n\t"
      "lbu %[s0], 0(%[scale])\n\t"
      "vwredsum.vs v5, v20, v0\n\t"
      "lbu %[s1], 1(%[scale])\n\t"
      "vwredsum.vs v6, v24, v0\n\t"
      "lbu %[s2], 2(%[scale])\n\t"
      "vwredsum.vs v7, v28, v0\n\t"
      "lbu %[s3], 3(%[scale])\n\t"
      "vwredsum.vs v8, v18, v0\n\t"
      "lbu %[q40], 4(%[scale])\n\t"
      "vwredsum.vs v9, v22, v0\n\t"
      "lbu %[q41], 5(%[scale])\n\t"
      "vwredsum.vs v10, v26, v0\n\t"
      "lbu %[q42], 6(%[scale])\n\t"
      "vwredsum.vs v11, v30, v0\n\t"
      "lbu %[q43], 7(%[scale])\n\t"
      "vsetivli zero, 4, e32, m1, ta, ma\n\t"
      "vmul.vx v0, v4, %[s0]\n\t"
      "vmul.vx v1, v8, %[q40]\n\t"
      "vmacc.vx v0, %[s1], v5\n\t"
      "vmacc.vx v1, %[q41], v9\n\t"
      "vmacc.vx v0, %[s2], v6\n\t"
      "vmacc.vx v1, %[q42], v10\n\t"
      "vmacc.vx v0, %[s3], v7\n\t"
      "vmacc.vx v1, %[q43], v11\n\t"
      "vfcvt.f.x.v v0, v0\n\t"
      "vfcvt.f.x.v v1, v1\n\t"
      "vfmv.f.s %[second], v0\n\t"
      "vfmv.f.s %[temporary], v1\n\t"
      "fadd.s %[second], %[second], %[temporary]\n\t"
      "fmadd.s %[sum], %[dot_scale], %[second], %[sum]"
      : [temporary] "=&f"(temporary), [sum] "+&f"(sum),
        [second] "=&f"(second), [s0] "=&r"(s0), [s1] "=&r"(s1),
        [s2] "=&r"(s2), [s3] "=&r"(s3), [q40] "=&r"(q40),
        [q41] "=&r"(q41), [q42] "=&r"(q42), [q43] "=&r"(q43),
        [q80] "=&r"(q80), [q81] "=&r"(q81), [q82] "=&r"(q82),
        [q83] "=&r"(q83)
      : [dot_scale] "f"(dot_scale), [activation] "r"(activation),
        [packed] "r"(packed_weight), [scale] "r"(scale),
        [bsums] "r"(activation_sum_bytes), [mins] "r"(minimum),
        [minimum_scale] "f"(minimum_scale)
      : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
        "v15", "v16", "v17", "v18", "v19", "v20", "v21",
        "v22", "v23", "v24", "v25", "v26", "v27", "v28",
        "v29", "v30", "v31");
  return sum;
}

)c";
  }
  if (usesGroupedI4I8RegisterL32) {
    output << R"c(
static inline __attribute__((always_inline, unused)) float
__weft_grouped_affine_i4_i8_register_l32_e8m1(
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
  const size_t vl = __riscv_vsetvl_e8m1(32);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  int32_t weighted_dot = 0;
  int32_t weighted_minimum = 0;
  for (size_t pair = 0; pair < 4; ++pair) {
    const vuint8m1_t packed =
        __riscv_vle8_v_u8m1(packed_weight + pair * 32, vl);
    const vint8m1_t low = __riscv_vreinterpret_v_u8m1_i8m1(
        __riscv_vand_vx_u8m1(packed, UINT8_C(15), vl));
    const vint8m1_t high = __riscv_vreinterpret_v_u8m1_i8m1(
        __riscv_vsrl_vx_u8m1(packed, 4, vl));
    const vint8m1_t activation_low =
        __riscv_vle8_v_i8m1(activation + pair * 64, vl);
    const vint8m1_t activation_high =
        __riscv_vle8_v_i8m1(activation + pair * 64 + 32, vl);
    const vint16m2_t products_low =
        __riscv_vwmul_vv_i16m2(low, activation_low, vl);
    const vint16m2_t products_high =
        __riscv_vwmul_vv_i16m2(high, activation_high, vl);
    const int32_t dot_low = __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m2_i32m1(products_low, zero, vl));
    const int32_t dot_high = __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m2_i32m1(products_high, zero, vl));
    weighted_dot += scale[2 * pair] * dot_low;
    weighted_dot += scale[2 * pair + 1] * dot_high;
    weighted_minimum +=
        minimum[2 * pair] *
        (activation_sum[4 * pair] + activation_sum[4 * pair + 1]);
    weighted_minimum +=
        minimum[2 * pair + 1] *
        (activation_sum[4 * pair + 2] + activation_sum[4 * pair + 3]);
  }
  return init + dot_scale * (float)weighted_dot -
         minimum_scale * (float)weighted_minimum;
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
