#include "RISCVIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitPrelude(llvm::raw_ostream &output, bool usesExp,
                 bool usesRVVSymmetricI4I8, bool usesRVVAffineI4I8,
                 bool usesRVVSymmetricI4I8M4, bool usesRVVAffineI4I8M4,
                 bool usesIME1SymmetricI4I8, bool usesIME1AffineI4I8,
                 bool usesIME1SymmetricI4I8M4, bool usesIME1AffineI4I8M4,
                 bool usesGroupedI4I8VLEN128,
                 bool usesGroupedI4I8VLEN256,
                 bool usesGroupedI4I8Scalable,
                 bool usesE2M1VLEN128, bool usesE2M1VLEN256,
                 bool usesE2M1Scalable) {
  output << R"c(#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

static inline __attribute__((unused)) int8_t __weft_bitcast_u8_i8(uint8_t bits) {
  union { uint8_t u; int8_t i; } value = { .u = bits };
  return value.i;
}

static inline __attribute__((unused)) int16_t __weft_bitcast_u16_i16(uint16_t bits) {
  union { uint16_t u; int16_t i; } value = { .u = bits };
  return value.i;
}

static inline __attribute__((unused)) _Float16 __weft_bitcast_u16_f16(uint16_t bits) {
  union { uint16_t u; _Float16 f; } value = { .u = bits };
  return value.f;
}

static inline __attribute__((unused)) uint16_t __weft_bitcast_f16_u16(_Float16 source) {
  union { uint16_t u; _Float16 f; } value = { .f = source };
  return value.u;
}

static inline __attribute__((unused)) uint32_t __weft_bitcast_f32_u32(float source) {
  union { uint32_t u; float f; } value = { .f = source };
  return value.u;
}

static inline __attribute__((unused)) float __weft_bitcast_u32_f32(uint32_t bits) {
  union { uint32_t u; float f; } value = { .u = bits };
  return value.f;
}

static inline __attribute__((unused)) float
__weft_load_f16_le(const uint8_t *bytes) {
  const uint16_t bits =
      (uint16_t)bytes[0] | ((uint16_t)bytes[1] << UINT16_C(8));
  return (float)__weft_bitcast_u16_f16(bits);
}

)c";
  emitRVVIntrinsicCLeaves(output, usesRVVSymmetricI4I8,
                          usesRVVAffineI4I8, usesRVVSymmetricI4I8M4,
                          usesRVVAffineI4I8M4, usesGroupedI4I8VLEN128,
                          usesGroupedI4I8VLEN256, usesGroupedI4I8Scalable,
                          usesE2M1VLEN128, usesE2M1VLEN256,
                          usesE2M1Scalable);
  emitIMEIntrinsicCLeaves(output, usesIME1SymmetricI4I8,
                          usesIME1AffineI4I8, usesIME1SymmetricI4I8M4,
                          usesIME1AffineI4I8M4);
  if (!usesExp)
    return;
  output << R"c(static inline __attribute__((unused)) vfloat32m2_t __weft_exp_f32m2(
    vfloat32m2_t x, size_t vl) {
  const vfloat32m2_t r = __riscv_vfmv_v_f_f32m2(0x1.8p23f, vl);
  const vfloat32m2_t z =
      __riscv_vfmacc_vf_f32m2(r, 0x1.715476p+0f, x, vl);
  const vfloat32m2_t n = __riscv_vfsub_vv_f32m2(z, r, vl);
  const vfloat32m2_t b0 =
      __riscv_vfnmsac_vf_f32m2(x, 0x1.62e4p-1f, n, vl);
  const vfloat32m2_t b =
      __riscv_vfnmsac_vf_f32m2(b0, 0x1.7f7d1cp-20f, n, vl);
  const vuint32m2_t z_bits = __riscv_vreinterpret_v_f32m2_u32m2(z);
  const vuint32m2_t e = __riscv_vsll_vx_u32m2(z_bits, 23, vl);
  const vuint32m2_t k_bits =
      __riscv_vadd_vx_u32m2(e, UINT32_C(0x3f800000), vl);
  const vfloat32m2_t k = __riscv_vreinterpret_v_u32m2_f32m2(k_bits);
  const vfloat32m2_t abs_n = __riscv_vfabs_v_f32m2(n, vl);
  const vbool16_t extreme =
      __riscv_vmfgt_vf_f32m2_b16(abs_n, 126.0f, vl);
  const vfloat32m2_t u = __riscv_vfmul_vv_f32m2(b, b, vl);
  const vfloat32m2_t j0 =
      __riscv_vfmul_vf_f32m2(b, 0x1.ffffecp-1f, vl);
  const vfloat32m2_t j1 = __riscv_vfmacc_vf_f32m2(
      __riscv_vfmv_v_f_f32m2(0x1.fffdb6p-2f, vl),
      0x1.555e66p-3f, b, vl);
  const vfloat32m2_t j2 = __riscv_vfmacc_vf_f32m2(
      __riscv_vfmv_v_f_f32m2(0x1.573e2ep-5f, vl),
      0x1.0e4020p-7f, b, vl);
  const vfloat32m2_t j3 = __riscv_vfmacc_vv_f32m2(j1, j2, u, vl);
  const vfloat32m2_t j = __riscv_vfmacc_vv_f32m2(j0, j3, u, vl);
  const vfloat32m2_t fast = __riscv_vfmacc_vv_f32m2(k, k, j, vl);
  if (__riscv_vcpop_m_b16(extreme, vl) == 0)
    return fast;

  const vbool16_t negative =
      __riscv_vmfle_vf_f32m2_b16(n, 0.0f, vl);
  const vuint32m2_t d = __riscv_vmerge_vxm_u32m2(
      __riscv_vmv_v_x_u32m2(0, vl), UINT32_C(0x82000000), negative, vl);
  const vfloat32m2_t s1 = __riscv_vreinterpret_v_u32m2_f32m2(
      __riscv_vadd_vx_u32m2(d, UINT32_C(0x7f000000), vl));
  const vfloat32m2_t s2 = __riscv_vreinterpret_v_u32m2_f32m2(
      __riscv_vsub_vv_u32m2(e, d, vl));
  const vfloat32m2_t corrected = __riscv_vfmul_vv_f32m2(
      __riscv_vfmacc_vv_f32m2(s2, s2, j, vl), s1, vl);
  const vfloat32m2_t bounded =
      __riscv_vmerge_vvm_f32m2(fast, corrected, extreme, vl);
  const vbool16_t overflow = __riscv_vmfgt_vf_f32m2_b16(
      __riscv_vfabs_v_f32m2(n, vl), 192.0f, vl);
  return __riscv_vmerge_vvm_f32m2(
      bounded, __riscv_vfmul_vv_f32m2(s1, s1, vl), overflow, vl);
}

static inline __attribute__((always_inline, unused)) vfloat32m2_t
__weft_tanh_f32m2(vfloat32m2_t x, size_t vl) {
  const vfloat32m2_t negative_twice =
      __riscv_vfmul_vf_f32m2(x, -2.0f, vl);
  const vfloat32m2_t exponential =
      __weft_exp_f32m2(negative_twice, vl);
  const vfloat32m2_t denominator =
      __riscv_vfadd_vf_f32m2(exponential, 1.0f, vl);
  const vfloat32m2_t fraction =
      __riscv_vfrdiv_vf_f32m2(denominator, 2.0f, vl);
  return __riscv_vfsub_vf_f32m2(fraction, 1.0f, vl);
}

static inline __attribute__((always_inline, unused)) vfloat32m2_t
__weft_sin_f32m2(vfloat32m2_t x, size_t vl) {
  float lanes[vl];
  __riscv_vse32_v_f32m2(lanes, x, vl);
  for (size_t lane = 0; lane < vl; ++lane)
    lanes[lane] = sinf(lanes[lane]);
  return __riscv_vle32_v_f32m2(lanes, vl);
}

static inline __attribute__((always_inline, unused)) vfloat32m2_t
__weft_cos_f32m2(vfloat32m2_t x, size_t vl) {
  float lanes[vl];
  __riscv_vse32_v_f32m2(lanes, x, vl);
  for (size_t lane = 0; lane < vl; ++lane)
    lanes[lane] = cosf(lanes[lane]);
  return __riscv_vle32_v_f32m2(lanes, vl);
}

static inline __attribute__((always_inline, unused)) void
__weft_online_summary_merge_f32(
    float *maximum, float *sum, float strip_maximum, float strip_sum) {
  if (*maximum == -INFINITY) {
    *maximum = strip_maximum;
    *sum = strip_sum;
    return;
  }
  if (strip_maximum <= *maximum) {
    *sum += strip_sum * expf(strip_maximum - *maximum);
    return;
  }
  *sum = *sum * expf(*maximum - strip_maximum) + strip_sum;
  *maximum = strip_maximum;
}

)c";
}

void emitIntrinsicCPrelude(llvm::raw_ostream &output,
                           const SelectedIntrinsicCLeaves &leaves) {
  bool usesExp = leaves.contains(IntrinsicCLeaf::RVVF32M2Math);
  bool usesRVVSymmetricI4I8 =
      leaves.contains(IntrinsicCLeaf::RVVSymmetricI4I8N16K32);
  bool usesRVVAffineI4I8 =
      leaves.contains(IntrinsicCLeaf::RVVAffineI4I8N16K32);
  bool usesRVVSymmetricI4I8M4 =
      leaves.contains(IntrinsicCLeaf::RVVSymmetricI4I8M4N16K32);
  bool usesRVVAffineI4I8M4 =
      leaves.contains(IntrinsicCLeaf::RVVAffineI4I8M4N16K32);
  bool usesIME1SymmetricI4I8 =
      leaves.contains(IntrinsicCLeaf::IME1SymmetricI4I8N16K32);
  bool usesIME1AffineI4I8 =
      leaves.contains(IntrinsicCLeaf::IME1AffineI4I8N16K32);
  bool usesIME1SymmetricI4I8M4 =
      leaves.contains(IntrinsicCLeaf::IME1SymmetricI4I8M4N16K32);
  bool usesIME1AffineI4I8M4 =
      leaves.contains(IntrinsicCLeaf::IME1AffineI4I8M4N16K32);
  bool usesGroupedI4I8VLEN128 =
      leaves.contains(IntrinsicCLeaf::GroupedAffineI4I8VLEN128);
  bool usesGroupedI4I8VLEN256 =
      leaves.contains(IntrinsicCLeaf::GroupedAffineI4I8VLEN256);
  bool usesGroupedI4I8Scalable =
      leaves.contains(IntrinsicCLeaf::GroupedAffineI4I8Scalable);
  bool usesE2M1VLEN128 =
      leaves.contains(IntrinsicCLeaf::E2M1E8M0I8VLEN128);
  bool usesE2M1VLEN256 =
      leaves.contains(IntrinsicCLeaf::E2M1E8M0I8VLEN256);
  bool usesE2M1Scalable =
      leaves.contains(IntrinsicCLeaf::E2M1E8M0I8Scalable);
  emitPrelude(output, usesExp, usesRVVSymmetricI4I8, usesRVVAffineI4I8,
              usesRVVSymmetricI4I8M4, usesRVVAffineI4I8M4,
              usesIME1SymmetricI4I8, usesIME1AffineI4I8,
              usesIME1SymmetricI4I8M4, usesIME1AffineI4I8M4,
              usesGroupedI4I8VLEN128, usesGroupedI4I8VLEN256,
              usesGroupedI4I8Scalable,
              usesE2M1VLEN128, usesE2M1VLEN256, usesE2M1Scalable);
  emitQuantIntrinsicCLeaves(output, leaves);
}

} // namespace weft::riscv_internal
