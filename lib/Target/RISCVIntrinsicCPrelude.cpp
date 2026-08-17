#include "RISCVIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitPrelude(llvm::raw_ostream &output, bool usesExp,
                 bool usesRVVSymmetricI4I8, bool usesRVVAffineI4I8,
                 bool usesRVVSymmetricI4I8M4, bool usesRVVAffineI4I8M4,
                 bool usesIME1SymmetricI4I8, bool usesIME1AffineI4I8,
                 bool usesIME1SymmetricI4I8M4, bool usesIME1AffineI4I8M4,
                 bool usesGroupedI4I8RegisterL16,
                 bool usesGroupedI4I8RegisterL32,
                 bool usesGroupedI4I8Strip,
                 bool usesE2M1RegisterE8M1M2,
                 bool usesE2M1RegisterE8MF2,
                 bool usesE2M1Strip) {
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

static inline __attribute__((unused)) float
__weft_load_f16_le_aligned(const uint8_t *bytes) {
  const uint8_t *aligned =
      (const uint8_t *)__builtin_assume_aligned(bytes, 2);
  uint16_t bits;
  __builtin_memcpy(&bits, aligned, sizeof(bits));
  return (float)__weft_bitcast_u16_f16(bits);
}

)c";
  emitRVVLocalImplementations(
      output, usesRVVSymmetricI4I8, usesRVVAffineI4I8,
      usesRVVSymmetricI4I8M4, usesRVVAffineI4I8M4,
      usesGroupedI4I8RegisterL16, usesGroupedI4I8RegisterL32,
      usesGroupedI4I8Strip, usesE2M1RegisterE8M1M2,
      usesE2M1RegisterE8MF2, usesE2M1Strip);
  emitIMELocalImplementations(output, usesIME1SymmetricI4I8,
                              usesIME1AffineI4I8,
                              usesIME1SymmetricI4I8M4,
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

bool emitIntrinsicCPrelude(llvm::raw_ostream &output,
                           const SelectedLocalImplementations &implementations,
                           std::string &unsupportedSymbol) {
  bool usesExp = false;
  bool usesRVVSymmetricI4I8 = false;
  bool usesRVVAffineI4I8 = false;
  bool usesRVVSymmetricI4I8M4 = false;
  bool usesRVVAffineI4I8M4 = false;
  bool usesIME1SymmetricI4I8 = false;
  bool usesIME1AffineI4I8 = false;
  bool usesIME1SymmetricI4I8M4 = false;
  bool usesIME1AffineI4I8M4 = false;
  bool usesGroupedI4I8RegisterL16 = false;
  bool usesGroupedI4I8RegisterL32 = false;
  bool usesGroupedI4I8Strip = false;
  bool usesE2M1RegisterE8M1M2 = false;
  bool usesE2M1RegisterE8MF2 = false;
  bool usesE2M1Strip = false;
  bool supported = true;
  implementations.forEach([&](const LocalImplementation &implementation) {
    if (!supported)
      return;
    const std::string &symbol = implementation.helperSymbol;
    auto match = [&](const char *expected, bool &flag) {
      if (symbol != expected)
        return false;
      flag = true;
      return true;
    };
    switch (implementation.primitive) {
    case LocalPrimitiveKind::F32Math:
      supported = symbol.empty();
      usesExp = supported;
      break;
    case LocalPrimitiveKind::SymmetricI4I8:
    case LocalPrimitiveKind::AffineI4I8:
      supported =
          match("__weft_rvv_symmetric_i4_i8_n16_k32",
                usesRVVSymmetricI4I8) ||
          match("__weft_rvv_affine_i4_i8_n16_k32", usesRVVAffineI4I8) ||
          match("__weft_rvv_symmetric_i4_i8_m4_n16_k32",
                usesRVVSymmetricI4I8M4) ||
          match("__weft_rvv_affine_i4_i8_m4_n16_k32",
                usesRVVAffineI4I8M4) ||
          match("__weft_ime1_symmetric_i4_i8_n16_k32",
                usesIME1SymmetricI4I8) ||
          match("__weft_ime1_affine_i4_i8_n16_k32", usesIME1AffineI4I8) ||
          match("__weft_ime1_symmetric_i4_i8_m4_n16_k32",
                usesIME1SymmetricI4I8M4) ||
          match("__weft_ime1_affine_i4_i8_m4_n16_k32",
                usesIME1AffineI4I8M4);
      break;
    case LocalPrimitiveKind::GroupedAffineI4I8:
      supported =
          match("__weft_grouped_affine_i4_i8_register_l16_e8m1",
                usesGroupedI4I8RegisterL16) ||
          match("__weft_grouped_affine_i4_i8_register_l32_e8m1",
                usesGroupedI4I8RegisterL32) ||
          match("__weft_grouped_affine_i4_i8_strip", usesGroupedI4I8Strip);
      break;
    case LocalPrimitiveKind::E2M1E8M0I8:
      supported =
          match("__weft_e2m1_e8m0_i8_register_e8m1_e8m2",
                usesE2M1RegisterE8M1M2) ||
          match("__weft_e2m1_e8m0_i8_register_e8mf2",
                usesE2M1RegisterE8MF2) ||
          match("__weft_e2m1_e8m0_i8_strip", usesE2M1Strip);
      break;
    case LocalPrimitiveKind::PackedI4I8:
    case LocalPrimitiveKind::PackedI5I8:
    case LocalPrimitiveKind::PackedI3GroupedI8:
    case LocalPrimitiveKind::Base3TernaryI8:
    case LocalPrimitiveKind::PackedI2TernaryI8:
    case LocalPrimitiveKind::SignedCodebook8I8:
    case LocalPrimitiveKind::SignedCodebook4I8:
    case LocalPrimitiveKind::PackedU9U7CodebookI8:
    case LocalPrimitiveKind::PackedU11GridDeltaI8:
    case LocalPrimitiveKind::NibbleCodebookI8:
    case LocalPrimitiveKind::IQ2SI8:
    case LocalPrimitiveKind::IQ3SI8:
    case LocalPrimitiveKind::IQ1MI8:
    case LocalPrimitiveKind::Q6KI8:
      break;
    case LocalPrimitiveKind::None:
      supported = false;
      break;
    }
    if (!supported)
      unsupportedSymbol = symbol.empty() ? "<unnamed local implementation>"
                                         : symbol;
  });
  if (!supported)
    return false;
  emitPrelude(output, usesExp, usesRVVSymmetricI4I8, usesRVVAffineI4I8,
              usesRVVSymmetricI4I8M4, usesRVVAffineI4I8M4,
              usesIME1SymmetricI4I8, usesIME1AffineI4I8,
              usesIME1SymmetricI4I8M4, usesIME1AffineI4I8M4,
              usesGroupedI4I8RegisterL16, usesGroupedI4I8RegisterL32,
              usesGroupedI4I8Strip, usesE2M1RegisterE8M1M2,
              usesE2M1RegisterE8MF2, usesE2M1Strip);
  return emitQuantLocalImplementations(output, implementations,
                                       unsupportedSymbol);
}

} // namespace weft::riscv_internal
