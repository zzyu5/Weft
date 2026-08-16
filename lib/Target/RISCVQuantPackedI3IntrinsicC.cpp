#include "RISCVQuantIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitPackedI3GroupedIntrinsicCLeaves(llvm::raw_ostream &output,
                                         bool vlen128, bool vlen256) {
  if (!vlen128 && !vlen256)
    return;
  output << R"c(static inline __attribute__((always_inline, unused)) int32_t
__weft_packed_i3_group_scale(const uint8_t *scales, size_t group) {
  const size_t quarter = group / 4;
  const size_t lane = group % 4;
  const uint8_t base = scales[(quarter % 2) * 4 + lane];
  const uint8_t low = quarter >= 2 ? base >> 4 : base & UINT8_C(15);
  const uint8_t high =
      (scales[8 + lane] >> (uint8_t)(quarter * 2)) & UINT8_C(3);
  return (int32_t)(low | (uint8_t)(high << 4)) - 32;
}

)c";

  if (vlen128) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_packed_i3_grouped_i8_vl128(
    const uint8_t *low_bits, const uint8_t *high_bits,
    const uint8_t *scales, const uint8_t *activation_bytes,
    float weight_scale, float activation_scale, float init) {
  int32_t integer_sum = 0;
  const size_t vl16 = __riscv_vsetvl_e8m1(16);
#pragma GCC unroll 16
  for (size_t group = 0; group < 16; ++group) {
    const size_t half = group / 8;
    const size_t within_half = group % 8;
    const size_t field = within_half / 2;
    const size_t lane_group = within_half % 2;
    const uint8_t low_shift = (uint8_t)(field * 2);
    const uint8_t high_mask = (uint8_t)(1u << (half * 4 + field));
    const vuint8m1_t packed = __riscv_vle8_v_u8m1(
        low_bits + half * 32 + lane_group * 16, vl16);
    const vuint8m1_t high = __riscv_vle8_v_u8m1(
        high_bits + lane_group * 16, vl16);
    const vuint8m1_t low = __riscv_vand_vx_u8m1(
        __riscv_vsrl_vx_u8m1(packed, low_shift, vl16), 3, vl16);
    vint8m1_t code = __riscv_vreinterpret_v_u8m1_i8m1(low);
    const vbool8_t missing = __riscv_vmseq_vx_u8m1_b8(
        __riscv_vand_vx_u8m1(high, high_mask, vl16), 0, vl16);
    code = __riscv_vsub_vx_i8m1_mu(missing, code, code, 4, vl16);
    const vint8m1_t activation = __riscv_vle8_v_i8m1(
        (const int8_t *)(const void *)(activation_bytes + group * 16), vl16);
    const vint16m2_t product =
        __riscv_vwmul_vv_i16m2(code, activation, vl16);
    const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
    const int32_t dot = __riscv_vmv_x_s_i32m1_i32(
        __riscv_vwredsum_vs_i16m2_i32m1(product, zero, vl16));
    integer_sum += __weft_packed_i3_group_scale(scales, group) * dot;
  }
  return init + weight_scale * activation_scale * (float)integer_sum;
}

)c";
  }

  if (vlen256) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_packed_i3_grouped_i8_vl256(
    const uint8_t *low_bits, const uint8_t *high_bits,
    const uint8_t *scales, const uint8_t *activation_bytes,
    float weight_scale, float activation_scale, float init) {
  int32_t integer_sum = 0;
  const size_t vl16 = __riscv_vsetvl_e16m1(16);
  const size_t vl32 = __riscv_vsetvl_e8m1(32);
#pragma GCC unroll 2
  for (size_t half = 0; half < 2; ++half) {
    const vuint8m1_t packed =
        __riscv_vle8_v_u8m1(low_bits + half * 32, vl32);
    const vuint8m1_t high = __riscv_vle8_v_u8m1(high_bits, vl32);
#pragma GCC unroll 4
    for (size_t field = 0; field < 4; ++field) {
      const size_t group = half * 8 + field * 2;
      const uint8_t low_shift = (uint8_t)(field * 2);
      const uint8_t high_mask = (uint8_t)(1u << (half * 4 + field));
      const vuint8m1_t low = __riscv_vand_vx_u8m1(
          __riscv_vsrl_vx_u8m1(packed, low_shift, vl32), 3, vl32);
      vint8m1_t code = __riscv_vreinterpret_v_u8m1_i8m1(low);
      const vbool8_t missing = __riscv_vmseq_vx_u8m1_b8(
          __riscv_vand_vx_u8m1(high, high_mask, vl32), 0, vl32);
      code = __riscv_vsub_vx_i8m1_mu(missing, code, code, 4, vl32);
      const vint8m1_t activation = __riscv_vle8_v_i8m1(
          (const int8_t *)(const void *)(activation_bytes + group * 16), vl32);
      const vint16m2_t product =
          __riscv_vwmul_vv_i16m2(code, activation, vl32);
      const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
      const int32_t dot0 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(
              __riscv_vget_v_i16m2_i16m1(product, 0), zero, vl16));
      const int32_t dot1 = __riscv_vmv_x_s_i32m1_i32(
          __riscv_vwredsum_vs_i16m1_i32m1(
              __riscv_vget_v_i16m2_i16m1(product, 1), zero, vl16));
      integer_sum += __weft_packed_i3_group_scale(scales, group) * dot0;
      integer_sum +=
          __weft_packed_i3_group_scale(scales, group + 1) * dot1;
    }
  }
  return init + weight_scale * activation_scale * (float)integer_sum;
}

)c";
  }
}

} // namespace weft::riscv_internal
