#include "RISCVQuantIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitPackedI4IntrinsicCLeaves(llvm::raw_ostream &output, bool vlen128,
                                  bool vlen256) {
  if (vlen128) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_packed_i4_i8_vl128(
    const uint8_t *packed_codes, const uint8_t *activation_bytes,
    int32_t zero_point, float dot_scale, float additive_bias, float init) {
  const size_t vl16 = __riscv_vsetvl_e8m1(16);
  const vuint8m1_t packed = __riscv_vle8_v_u8m1(packed_codes, vl16);
  const vint8m1_t low = __riscv_vreinterpret_v_u8m1_i8m1(
      __riscv_vand_vx_u8m1(packed, UINT8_C(15), vl16));
  const vint8m1_t high = __riscv_vreinterpret_v_u8m1_i8m1(
      __riscv_vsrl_vx_u8m1(packed, 4, vl16));
  const size_t vl32 = __riscv_vsetvl_e8m2(32);
  vint8m2_t codes = __riscv_vcreate_v_i8m1_i8m2(low, high);
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
__weft_packed_i4_i8_vl256(
    const uint8_t *packed_codes, const uint8_t *activation_bytes,
    int32_t zero_point, float dot_scale, float additive_bias, float init) {
  const size_t vl16 = __riscv_vsetvl_e8m1(16);
  const vuint8m1_t packed = __riscv_vle8_v_u8m1(packed_codes, vl16);
  vint8m1_t codes = __riscv_vreinterpret_v_u8m1_i8m1(
      __riscv_vand_vx_u8m1(packed, UINT8_C(15), vl16));
  const vint8m1_t high = __riscv_vreinterpret_v_u8m1_i8m1(
      __riscv_vsrl_vx_u8m1(packed, 4, vl16));
  codes = __riscv_vslideup_vx_i8m1(codes, high, 16, 32);
  const size_t vl32 = __riscv_vsetvl_e8m1(32);
  codes = __riscv_vsub_vx_i8m1(codes, zero_point, vl32);
  const vint8m1_t activation = __riscv_vle8_v_i8m1(
      (const int8_t *)(const void *)activation_bytes, vl32);
  const vint16m2_t products =
      __riscv_vwmul_vv_i16m2(codes, activation, vl32);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t integer_dot = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m2_i32m1(products, zero, vl32));
  return init + dot_scale * (float)integer_dot + additive_bias;
}

)c";
  }
}

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
  const size_t vl32 = __riscv_vsetvl_e8m1(32);
  vint8m1_t codes = low;
  const vbool8_t high_mask = __riscv_vlm_v_b8(high_bits, vl32);
  codes = __riscv_vor_vx_i8m1_mu(
      high_mask, codes, codes, INT8_C(16), vl32);
  codes = __riscv_vsub_vx_i8m1(codes, zero_point, vl32);
  const vint8m1_t activation = __riscv_vle8_v_i8m1(
      (const int8_t *)(const void *)activation_bytes, vl32);
  const vint16m2_t products =
      __riscv_vwmul_vv_i16m2(codes, activation, vl32);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t integer_dot = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m2_i32m1(products, zero, vl32));
  return init + dot_scale * (float)integer_dot + additive_bias;
}

)c";
  }
}

void emitNibbleCodebookIntrinsicCLeaves(llvm::raw_ostream &output,
                                        bool vlen128, bool vlen256) {
  if (vlen128) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_nibble_codebook_i8_vl128(
    const uint8_t *packed_codes, const uint8_t *table_bytes,
    const uint8_t *activation_bytes, float dot_scale, float init) {
  const size_t vl16 = __riscv_vsetvl_e8m1(16);
  const vuint8m1_t packed = __riscv_vle8_v_u8m1(packed_codes, vl16);
  const vint8m2_t table = __riscv_vle8_v_i8m2(
      (const int8_t *)(const void *)table_bytes, vl16);
  const vuint8m1_t low_index =
      __riscv_vand_vx_u8m1(packed, UINT8_C(15), vl16);
  const vuint8m1_t high_index =
      __riscv_vsrl_vx_u8m1(packed, 4, vl16);
  const size_t vl32 = __riscv_vsetvl_e8m2(32);
  const vuint8m2_t index =
      __riscv_vcreate_v_u8m1_u8m2(low_index, high_index);
  const vint8m2_t decoded =
      __riscv_vrgather_vv_i8m2(table, index, vl32);
  const vint8m2_t activation = __riscv_vle8_v_i8m2(
      (const int8_t *)(const void *)activation_bytes, vl32);
  const vint16m4_t product =
      __riscv_vwmul_vv_i16m4(decoded, activation, vl32);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t integer_dot = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m4_i32m1(product, zero, vl32));
  return init + dot_scale * (float)integer_dot;
}

)c";
  }

  if (vlen256) {
    output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_nibble_codebook_i8_vl256(
    const uint8_t *packed_codes, const uint8_t *table_bytes,
    const uint8_t *activation_bytes, float dot_scale, float init) {
  const size_t vl16 = __riscv_vsetvl_e8mf2(16);
  const vuint8mf2_t packed = __riscv_vle8_v_u8mf2(packed_codes, vl16);
  const vint8mf2_t table = __riscv_vle8_v_i8mf2(
      (const int8_t *)(const void *)table_bytes, vl16);
  const vuint8mf2_t low_index =
      __riscv_vand_vx_u8mf2(packed, UINT8_C(15), vl16);
  const vuint8mf2_t high_index =
      __riscv_vsrl_vx_u8mf2(packed, 4, vl16);
  const vint8mf2_t low =
      __riscv_vrgather_vv_i8mf2(table, low_index, vl16);
  const vint8mf2_t high =
      __riscv_vrgather_vv_i8mf2(table, high_index, vl16);
  const vint8mf2_t activation_low = __riscv_vle8_v_i8mf2(
      (const int8_t *)(const void *)activation_bytes, vl16);
  const vint8mf2_t activation_high = __riscv_vle8_v_i8mf2(
      (const int8_t *)(const void *)(activation_bytes + 16), vl16);
  const vint16m1_t product = __riscv_vadd_vv_i16m1(
      __riscv_vwmul_vv_i16m1(low, activation_low, vl16),
      __riscv_vwmul_vv_i16m1(high, activation_high, vl16), vl16);
  const vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
  const int32_t integer_dot = __riscv_vmv_x_s_i32m1_i32(
      __riscv_vwredsum_vs_i16m1_i32m1(product, zero, vl16));
  return init + dot_scale * (float)integer_dot;
}

)c";
  }
}

} // namespace weft::riscv_internal
