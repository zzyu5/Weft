#include "RISCVQuantIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitTernaryLocalImplementations(llvm::raw_ostream &output,
                                 bool base3E8M2, bool base3E8M1,
                                 bool packedI2E8M2,
                                 bool packedI2E8M1) {
  if (base3E8M2) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_base3_ternary_i8_register_l32_e8m2(
    const uint8_t *codes, const uint8_t *high_digits,
    const uint8_t *activation_bytes, float weight_scale,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  const uint8_t powers[16] = {
      1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};

  const size_t vl32 = __riscv_vsetvl_e8m2(32);
  const vuint8m2_t packed32 = __riscv_vle8_v_u8m2(codes, vl32);
  const vuint16m4_t digit0 = __riscv_vsrl_vx_u16m4(
      __riscv_vwmulu_vx_u16m4(packed32, 3, vl32), 8, vl32);
  const vint16m4_t activation0 = __riscv_vwcvt_x_x_v_i16m4(
      __riscv_vle8_v_i8m2(activation, vl32), vl32);
  vint16m4_t sum32 = __riscv_vmul_vv_i16m4(
      __riscv_vreinterpret_v_u16m4_i16m4(
          __riscv_vsub_vx_u16m4(digit0, 1, vl32)),
      activation0, vl32);
  uint8_t power = 3;
  for (size_t digit = 1; digit < 5; ++digit) {
    const vuint16m4_t decoded = __riscv_vsrl_vx_u16m4(
        __riscv_vwmulu_vx_u16m4(
            __riscv_vmul_vx_u8m2(packed32, power, vl32), 3, vl32),
        8, vl32);
    const vint16m4_t values = __riscv_vwcvt_x_x_v_i16m4(
        __riscv_vle8_v_i8m2(activation + digit * 32, vl32), vl32);
    sum32 = __riscv_vmacc_vv_i16m4(
        sum32,
        __riscv_vreinterpret_v_u16m4_i16m4(
            __riscv_vsub_vx_u16m4(decoded, 1, vl32)),
        values, vl32);
    power = (uint8_t)(power * 3);
  }

  const size_t vl16 = __riscv_vsetvl_e8m1(16);
  const vuint8m1_t packed16 = __riscv_vle8_v_u8m1(codes + 32, vl16);
  const vuint16m2_t tail0 = __riscv_vsrl_vx_u16m2(
      __riscv_vwmulu_vx_u16m2(packed16, 3, vl16), 8, vl16);
  const vint16m2_t tailActivation0 = __riscv_vwcvt_x_x_v_i16m2(
      __riscv_vle8_v_i8m1(activation + 160, vl16), vl16);
  vint16m2_t sum16 = __riscv_vmul_vv_i16m2(
      __riscv_vreinterpret_v_u16m2_i16m2(
          __riscv_vsub_vx_u16m2(tail0, 1, vl16)),
      tailActivation0, vl16);
  power = 3;
  for (size_t digit = 1; digit < 5; ++digit) {
    const vuint16m2_t decoded = __riscv_vsrl_vx_u16m2(
        __riscv_vwmulu_vx_u16m2(
            __riscv_vmul_vx_u8m1(packed16, power, vl16), 3, vl16),
        8, vl16);
    const vint16m2_t values = __riscv_vwcvt_x_x_v_i16m2(
        __riscv_vle8_v_i8m1(activation + 160 + digit * 16, vl16), vl16);
    sum16 = __riscv_vmacc_vv_i16m2(
        sum16,
        __riscv_vreinterpret_v_u16m2_i16m2(
            __riscv_vsub_vx_u16m2(decoded, 1, vl16)),
        values, vl16);
    power = (uint8_t)(power * 3);
  }

  uint32_t packed_high = (uint32_t)high_digits[0] |
      ((uint32_t)high_digits[1] << 8) |
      ((uint32_t)high_digits[2] << 16) |
      ((uint32_t)high_digits[3] << 24);
  __asm__ __volatile__("" : "+r"(packed_high));
  const vuint8m1_t high = __riscv_vreinterpret_v_u32m1_u8m1(
      __riscv_vmv_v_x_u32m1(packed_high, 4));
  const vuint8m1_t high_power = __riscv_vle8_v_u8m1(powers, vl16);
  const vuint16m2_t high_decoded = __riscv_vsrl_vx_u16m2(
      __riscv_vwmulu_vx_u16m2(
          __riscv_vmul_vv_u8m1(high, high_power, vl16), 3, vl16),
      8, vl16);
  const vint16m2_t high_activation = __riscv_vwcvt_x_x_v_i16m2(
      __riscv_vle8_v_i8m1(activation + 240, vl16), vl16);
  const vint16m2_t high_sum = __riscv_vmul_vv_i16m2(
      __riscv_vreinterpret_v_u16m2_i16m2(
          __riscv_vsub_vx_u16m2(high_decoded, 1, vl16)),
      high_activation, vl16);

  vint16m2_t combined = __riscv_vadd_vv_i16m2(
      __riscv_vget_v_i16m4_i16m2(sum32, 0),
      __riscv_vget_v_i16m4_i16m2(sum32, 1), vl16);
  combined = __riscv_vadd_vv_i16m2(combined, sum16, vl16);
  combined = __riscv_vadd_vv_i16m2(combined, high_sum, vl16);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t integer_sum = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m2_i32m1(combined, zero, vl16));
  return init + (float)integer_sum * weight_scale * activation_scale;
}
)c";
  }

  if (base3E8M1) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_base3_ternary_i8_register_l32_e8m1(
    const uint8_t *codes, const uint8_t *high_digits,
    const uint8_t *activation_bytes, float weight_scale,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  const uint8_t powers[16] = {
      1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};

  const size_t vl32 = __riscv_vsetvl_e8m1(32);
  const vuint8m1_t packed32 = __riscv_vle8_v_u8m1(codes, vl32);
  const vuint16m2_t digit0 = __riscv_vsrl_vx_u16m2(
      __riscv_vwmulu_vx_u16m2(packed32, 3, vl32), 8, vl32);
  const vint16m2_t activation0 = __riscv_vwcvt_x_x_v_i16m2(
      __riscv_vle8_v_i8m1(activation, vl32), vl32);
  vint16m2_t sum32 = __riscv_vmul_vv_i16m2(
      __riscv_vreinterpret_v_u16m2_i16m2(
          __riscv_vsub_vx_u16m2(digit0, 1, vl32)),
      activation0, vl32);
  uint8_t power = 3;
  for (size_t digit = 1; digit < 5; ++digit) {
    const vuint16m2_t decoded = __riscv_vsrl_vx_u16m2(
        __riscv_vwmulu_vx_u16m2(
            __riscv_vmul_vx_u8m1(packed32, power, vl32), 3, vl32),
        8, vl32);
    const vint16m2_t values = __riscv_vwcvt_x_x_v_i16m2(
        __riscv_vle8_v_i8m1(activation + digit * 32, vl32), vl32);
    sum32 = __riscv_vmacc_vv_i16m2(
        sum32,
        __riscv_vreinterpret_v_u16m2_i16m2(
            __riscv_vsub_vx_u16m2(decoded, 1, vl32)),
        values, vl32);
    power = (uint8_t)(power * 3);
  }

  const size_t vl16 = __riscv_vsetvl_e8mf2(16);
  const vuint8mf2_t packed16 = __riscv_vle8_v_u8mf2(codes + 32, vl16);
  const vuint16m1_t tail0 = __riscv_vsrl_vx_u16m1(
      __riscv_vwmulu_vx_u16m1(packed16, 3, vl16), 8, vl16);
  const vint16m1_t tailActivation0 = __riscv_vwcvt_x_x_v_i16m1(
      __riscv_vle8_v_i8mf2(activation + 160, vl16), vl16);
  vint16m1_t sum16 = __riscv_vmul_vv_i16m1(
      __riscv_vreinterpret_v_u16m1_i16m1(
          __riscv_vsub_vx_u16m1(tail0, 1, vl16)),
      tailActivation0, vl16);
  power = 3;
  for (size_t digit = 1; digit < 5; ++digit) {
    const vuint16m1_t decoded = __riscv_vsrl_vx_u16m1(
        __riscv_vwmulu_vx_u16m1(
            __riscv_vmul_vx_u8mf2(packed16, power, vl16), 3, vl16),
        8, vl16);
    const vint16m1_t values = __riscv_vwcvt_x_x_v_i16m1(
        __riscv_vle8_v_i8mf2(activation + 160 + digit * 16, vl16), vl16);
    sum16 = __riscv_vmacc_vv_i16m1(
        sum16,
        __riscv_vreinterpret_v_u16m1_i16m1(
            __riscv_vsub_vx_u16m1(decoded, 1, vl16)),
        values, vl16);
    power = (uint8_t)(power * 3);
  }

  uint32_t packed_high = (uint32_t)high_digits[0] |
      ((uint32_t)high_digits[1] << 8) |
      ((uint32_t)high_digits[2] << 16) |
      ((uint32_t)high_digits[3] << 24);
  __asm__ __volatile__("" : "+r"(packed_high));
  const vuint8mf2_t high = __riscv_vreinterpret_v_u32mf2_u8mf2(
      __riscv_vmv_v_x_u32mf2(packed_high, 4));
  const vuint8mf2_t high_power = __riscv_vle8_v_u8mf2(powers, vl16);
  const vuint16m1_t high_decoded = __riscv_vsrl_vx_u16m1(
      __riscv_vwmulu_vx_u16m1(
          __riscv_vmul_vv_u8mf2(high, high_power, vl16), 3, vl16),
      8, vl16);
  const vint16m1_t high_activation = __riscv_vwcvt_x_x_v_i16m1(
      __riscv_vle8_v_i8mf2(activation + 240, vl16), vl16);
  const vint16m1_t high_sum = __riscv_vmul_vv_i16m1(
      __riscv_vreinterpret_v_u16m1_i16m1(
          __riscv_vsub_vx_u16m1(high_decoded, 1, vl16)),
      high_activation, vl16);

  vint16m1_t combined = __riscv_vadd_vv_i16m1(
      __riscv_vget_v_i16m2_i16m1(sum32, 0),
      __riscv_vget_v_i16m2_i16m1(sum32, 1), vl16);
  combined = __riscv_vadd_vv_i16m1(combined, sum16, vl16);
  combined = __riscv_vadd_vv_i16m1(combined, high_sum, vl16);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t integer_sum = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m1_i32m1(combined, zero, vl16));
  return init + (float)integer_sum * weight_scale * activation_scale;
}
)c";
  }

  if (packedI2E8M2) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_packed_i2_ternary_i8_register_l32_e8m2(
    const uint8_t *codes, const uint8_t *activation_bytes,
    float weight_scale, float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  int32_t integer_sum = 0;
  for (size_t half = 0; half < 2; ++half) {
    const size_t vl = __riscv_vsetvl_e8m2(32);
    const vuint8m2_t packed =
        __riscv_vle8_v_u8m2(codes + half * 32, vl);
    vint16m4_t accumulator = __riscv_vmv_v_x_i16m4(0, 32);
    for (size_t field = 0; field < 4; ++field) {
      vuint8m2_t unpacked =
          __riscv_vsrl_vx_u8m2(packed, 2 * field, vl);
      if (field != 3)
        unpacked = __riscv_vand_vx_u8m2(unpacked, 3, vl);
      const vint8m2_t ternary = __riscv_vsub_vx_i8m2(
          __riscv_vreinterpret_v_u8m2_i8m2(unpacked), 1, vl);
      const vint8m2_t values = __riscv_vle8_v_i8m2(
          activation + half * 128 + field * 32, vl);
      accumulator = __riscv_vwmacc_vv_i16m4(
          accumulator, ternary, values, vl);
    }
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    integer_sum += __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m4_i32m1(accumulator, zero, 32));
  }
  return init + (float)integer_sum * weight_scale * activation_scale;
}
)c";
  }

  if (packedI2E8M1) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_packed_i2_ternary_i8_register_l32_e8m1(
    const uint8_t *codes, const uint8_t *activation_bytes,
    float weight_scale, float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  int32_t integer_sum = 0;
  for (size_t half = 0; half < 2; ++half) {
    const size_t vl = __riscv_vsetvl_e8m1(32);
    const vuint8m1_t packed =
        __riscv_vle8_v_u8m1(codes + half * 32, vl);
    vint16m2_t accumulator = __riscv_vmv_v_x_i16m2(0, 32);
    for (size_t field = 0; field < 4; ++field) {
      vuint8m1_t unpacked =
          __riscv_vsrl_vx_u8m1(packed, 2 * field, vl);
      if (field != 3)
        unpacked = __riscv_vand_vx_u8m1(unpacked, 3, vl);
      const vint8m1_t ternary = __riscv_vsub_vx_i8m1(
          __riscv_vreinterpret_v_u8m1_i8m1(unpacked), 1, vl);
      const vint8m1_t values = __riscv_vle8_v_i8m1(
          activation + half * 128 + field * 32, vl);
      accumulator = __riscv_vwmacc_vv_i16m2(
          accumulator, ternary, values, vl);
    }
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    integer_sum += __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m2_i32m1(accumulator, zero, 32));
  }
  return init + (float)integer_sum * weight_scale * activation_scale;
}
)c";
  }
}

} // namespace weft::riscv_internal
