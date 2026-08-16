#include "RISCVQuantIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitQ6LocalImplementations(llvm::raw_ostream &output, bool registerL32,
                            bool registerL64, bool strip) {
  if (registerL64) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_q6_k_i8_register_l64_e8m2(
    const uint8_t *low_bits, const uint8_t *high_bits,
    const uint8_t *group_scale_bytes, const uint8_t *activation_bytes,
    float weight_scale, float activation_scale, float init) {
  const int8_t *group_scales =
      (const int8_t *)(const void *)group_scale_bytes;
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  const float combined_scale = weight_scale * activation_scale;
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    int32_t integer_sum = 0;
    for (size_t half = 0; half < 2; ++half) {
      const uint8_t *q6 = low_bits + half * 64;
      const uint8_t *qh = high_bits + half * 32;
      const int8_t *q8v = activation + half * 128;
      const int8_t *sc = group_scales + half * 8;
      const size_t vl32 = __riscv_vsetvl_e8m1(32);
      const vuint8m1_t qh_value = __riscv_vle8_v_u8m1(qh, vl32);
      const vuint8m1_t q6_0 = __riscv_vle8_v_u8m1(q6, vl32);
      const vuint8m1_t q6_1 = __riscv_vle8_v_u8m1(q6 + 32, vl32);
      const vuint8m1_t low_0 = __riscv_vand_vx_u8m1(q6_0, 0x0f, vl32);
      const vuint8m1_t low_1 = __riscv_vand_vx_u8m1(q6_1, 0x0f, vl32);
      const vuint8m1_t high_0 = __riscv_vsrl_vx_u8m1(q6_0, 4, vl32);
      const vuint8m1_t high_1 = __riscv_vsrl_vx_u8m1(q6_1, 4, vl32);
      const vuint8m1_t qh_0 = __riscv_vand_vx_u8m1(qh_value, 3, vl32);
      const vuint8m1_t qh_1 = __riscv_vand_vx_u8m1(
          __riscv_vsrl_vx_u8m1(qh_value, 2, vl32), 3, vl32);
      const vuint8m1_t qh_2 = __riscv_vand_vx_u8m1(
          __riscv_vsrl_vx_u8m1(qh_value, 4, vl32), 3, vl32);
      const vuint8m1_t qh_3 = __riscv_vand_vx_u8m1(
          __riscv_vsrl_vx_u8m1(qh_value, 6, vl32), 3, vl32);
      const vint8m1_t value_0 = __riscv_vsub_vx_i8m1(
          __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(
              low_0, __riscv_vsll_vx_u8m1(qh_0, 4, vl32), vl32)),
          32, vl32);
      const vint8m1_t value_1 = __riscv_vsub_vx_i8m1(
          __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(
              low_1, __riscv_vsll_vx_u8m1(qh_1, 4, vl32), vl32)),
          32, vl32);
      const vint8m1_t value_2 = __riscv_vsub_vx_i8m1(
          __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(
              high_0, __riscv_vsll_vx_u8m1(qh_2, 4, vl32), vl32)),
          32, vl32);
      const vint8m1_t value_3 = __riscv_vsub_vx_i8m1(
          __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vor_vv_u8m1(
              high_1, __riscv_vsll_vx_u8m1(qh_3, 4, vl32), vl32)),
          32, vl32);
      const vint16m2_t product_0 = __riscv_vwmul_vv_i16m2(
          value_0, __riscv_vle8_v_i8m1(q8v, vl32), vl32);
      const vint16m2_t product_1 = __riscv_vwmul_vv_i16m2(
          value_1, __riscv_vle8_v_i8m1(q8v + 32, vl32), vl32);
      const vint16m2_t product_2 = __riscv_vwmul_vv_i16m2(
          value_2, __riscv_vle8_v_i8m1(q8v + 64, vl32), vl32);
      const vint16m2_t product_3 = __riscv_vwmul_vv_i16m2(
          value_3, __riscv_vle8_v_i8m1(q8v + 96, vl32), vl32);
      const size_t vl16 = __riscv_vsetvl_e16m1(16);
      const vint32m2_t scaled_0 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(product_0, 0), sc[0], vl16);
      const vint32m2_t scaled_1 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(product_0, 1), sc[1], vl16);
      const vint32m2_t scaled_2 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(product_1, 0), sc[2], vl16);
      const vint32m2_t scaled_3 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(product_1, 1), sc[3], vl16);
      const vint32m2_t scaled_4 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(product_2, 0), sc[4], vl16);
      const vint32m2_t scaled_5 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(product_2, 1), sc[5], vl16);
      const vint32m2_t scaled_6 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(product_3, 0), sc[6], vl16);
      const vint32m2_t scaled_7 = __riscv_vwmul_vx_i32m2(
          __riscv_vget_v_i16m2_i16m1(product_3, 1), sc[7], vl16);
      const vint32m1_t sum_0 = __riscv_vredsum_vs_i32m2_i32m1(
          __riscv_vadd_vv_i32m2(scaled_0, scaled_1, vl16), zero, vl16);
      const vint32m1_t sum_1 = __riscv_vredsum_vs_i32m2_i32m1(
          __riscv_vadd_vv_i32m2(scaled_2, scaled_3, vl16), sum_0, vl16);
      const vint32m1_t sum_2 = __riscv_vredsum_vs_i32m2_i32m1(
          __riscv_vadd_vv_i32m2(scaled_4, scaled_5, vl16), sum_1, vl16);
      const vint32m1_t sum_3 = __riscv_vredsum_vs_i32m2_i32m1(
          __riscv_vadd_vv_i32m2(scaled_6, scaled_7, vl16), sum_2, vl16);
      integer_sum += __riscv_vmv_x_s_i32m1_i32(sum_3);
    }
    return init + combined_scale * (float)integer_sum;
}
)c";
  }
  if (registerL32) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_q6_k_i8_register_l32_e8m2(
    const uint8_t *low_bits, const uint8_t *high_bits,
    const uint8_t *group_scale_bytes, const uint8_t *activation_bytes,
    float weight_scale, float activation_scale, float init) {
  const int8_t *group_scales =
      (const int8_t *)(const void *)group_scale_bytes;
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  const float combined_scale = weight_scale * activation_scale;
  const uint8_t *low = low_bits;
  const uint8_t *high = high_bits;
  const int8_t *scale = group_scales;
  const int8_t *q8 = activation;
  int low_high;
  float temporary;
  float result = init;
  for (size_t half = 0; half < 2; ++half) {
    __asm__ volatile(
        "addi %[low_high], %[low], 32\n\t"
        "ld t0, 0(%[scale])\n\t"
        "addi %[scale], %[scale], 8\n\t"
        "slli t6, t0, 1 * 8\n\t"
        "lb zero, 0(%[low])\n\t"
        "slli t5, t0, 2 * 8\n\t"
        "slli t4, t0, 3 * 8\n\t"
        "lb zero, 0(%[low_high])\n\t"
        "slli t3, t0, 4 * 8\n\t"
        "slli t2, t0, 5 * 8\n\t"
        "lb zero, 0(%[high])\n\t"
        "lb zero, 31(%[low_high])\n\t"
        "slli t1, t0, 6 * 8\n\t"
        "srai a7, t0, 56\n\t"
        "vsetvli zero, %[vl32], e8, m2\n\t"
        "vle8.v v8, (%[low])\n\t"
        "srai t6, t6, 56\n\t"
        "srai t5, t5, 56\n\t"
        "srai t4, t4, 56\n\t"
        "srai t3, t3, 56\n\t"
        "vle8.v v10, (%[low_high])\n\t"
        "addi %[low], %[low], 64\n\t"
        "slli t0, t0, 7 * 8\n\t"
        "srai t2, t2, 56\n\t"
        "srai t1, t1, 56\n\t"
        "srai t0, t0, 56\n\t"
        "vle8.v v4, (%[high])\n\t"
        "vsrl.vi v12, v8, 4\n\t"
        "vsrl.vi v14, v10, 4\n\t"
        "lb zero, 0(%[q8])\n\t"
        "vand.vi v8, v8, 0xF\n\t"
        "vand.vi v10, v10, 0xF\n\t"
        "lb zero, 32(%[q8])\n\t"
        "vsll.vi v0, v4, 4\n\t"
        "vsll.vi v2, v4, 2\n\t"
        "lb zero, 64(%[q8])\n\t"
        "vsrl.vi v6, v4, 2\n\t"
        "lb zero, 96(%[q8])\n\t"
        "vand.vx v0, v0, %[mask]\n\t"
        "vand.vx v2, v2, %[mask]\n\t"
        "vand.vx v4, v4, %[mask]\n\t"
        "vand.vx v6, v6, %[mask]\n\t"
        "vor.vv v8, v8, v0\n\t"
        "vor.vv v10, v10, v2\n\t"
        "vor.vv v12, v12, v4\n\t"
        "vor.vv v14, v14, v6\n\t"
        "lb zero, 127(%[q8])\n\t"
        "vsetvli zero, %[vl128], e8, m8\n\t"
        "vle8.v v0, (%[q8])\n\t"
        "vsub.vx v8, v8, %[vl32]\n\t"
        "vsetvli zero, %[vl64], e8, m4\n\t"
        "vwmul.vv v16, v0, v8\n\t"
        "vwmul.vv v24, v4, v12\n\t"
        "vsetivli zero, 16, e16, m2\n\t"
        "vmv.v.x v0, zero\n\t"
        "vwredsum.vs v10, v16, v0\n\t"
        "vwredsum.vs v9, v18, v0\n\t"
        "vwredsum.vs v8, v20, v0\n\t"
        "vwredsum.vs v7, v22, v0\n\t"
        "vwredsum.vs v11, v24, v0\n\t"
        "vwredsum.vs v12, v26, v0\n\t"
        "vwredsum.vs v13, v28, v0\n\t"
        "vwredsum.vs v14, v30, v0\n\t"
        "vsetivli zero, 4, e32, m1\n\t"
        "vmul.vx v0, v10, t0\n\t"
        "vmul.vx v1, v9, t1\n\t"
        "vmacc.vx v0, t2, v8\n\t"
        "vmacc.vx v1, t3, v7\n\t"
        "vmacc.vx v0, t4, v11\n\t"
        "vmacc.vx v1, t5, v12\n\t"
        "vmacc.vx v0, t6, v13\n\t"
        "vmacc.vx v1, a7, v14\n\t"
        "vadd.vv v0, v0, v1\n\t"
        "vfcvt.f.x.v v0, v0\n\t"
        "vfmv.f.s %[temporary], v0\n\t"
        "fmadd.s %[result], %[combined_scale], %[temporary], %[result]"
        : [low] "+&r"(low), [low_high] "=&r"(low_high),
          [scale] "+&r"(scale), [result] "+&f"(result),
          [temporary] "=&f"(temporary)
        : [high] "r"(high), [q8] "r"(q8), [vl32] "r"(32),
          [vl64] "r"(64), [vl128] "r"(128), [mask] "r"(0x30),
          [combined_scale] "f"(combined_scale)
        : "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
          "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
          "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30",
          "v31", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a7");
    high += 32;
    q8 += 128;
  }
  return result;
}
)c";
  }
  if (strip) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_q6_k_i8_strip(
    const uint8_t *low_bits, const uint8_t *high_bits,
    const uint8_t *group_scale_bytes, const uint8_t *activation_bytes,
    float weight_scale, float activation_scale, float init) {
  const int8_t *group_scales =
      (const int8_t *)(const void *)group_scale_bytes;
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  int8_t decoded[32];
  int32_t integer_sum = 0;
  for (size_t half = 0; half < 2; ++half) {
    for (size_t quarter = 0; quarter < 4; ++quarter) {
      for (size_t lane = 0; lane < 32; ++lane) {
        const uint8_t low0 = low_bits[half * 64 + lane];
        const uint8_t low1 = low_bits[half * 64 + 32 + lane];
        const uint8_t high = high_bits[half * 32 + lane];
        uint8_t code;
        if (quarter == 0)
          code = (low0 & UINT8_C(15)) | (((high >> 0) & 3) << 4);
        else if (quarter == 1)
          code = (low1 & UINT8_C(15)) | (((high >> 2) & 3) << 4);
        else if (quarter == 2)
          code = (low0 >> 4) | (((high >> 4) & 3) << 4);
        else
          code = (low1 >> 4) | (((high >> 6) & 3) << 4);
        decoded[lane] = (int8_t)code - 32;
      }
      const size_t base = half * 128 + quarter * 32;
      const int32_t first = __weft_i8_dot(decoded, activation + base, 16);
      const int32_t second =
          __weft_i8_dot(decoded + 16, activation + base + 16, 16);
      integer_sum += first * group_scales[base / 16];
      integer_sum += second * group_scales[base / 16 + 1];
    }
  }
  return init + (float)integer_sum * weight_scale * activation_scale;
}
)c";
  }
}

} // namespace weft::riscv_internal
