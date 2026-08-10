#include <stddef.h>
#include <stdint.h>

static float fp16_to_fp32(uint16_t value) {
  const uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x03ffu;
  uint32_t bits;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 113;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }

  union {
    uint32_t bits;
    float value;
  } converted = {bits};
  return converted.value;
}

static size_t minimum(size_t a, size_t b) {
  return a < b ? a : b;
}

void weft_reference_blocked_gemm_f16_f32(
    const uint16_t *a,
    const uint16_t *b,
    float *c,
    size_t m,
    size_t n,
    size_t k,
    size_t lda,
    size_t ldb,
    size_t ldc,
    size_t bm,
    size_t bn,
    size_t bk) {
  for (size_t row = 0; row < m; ++row) {
    for (size_t col = 0; col < n; ++col) {
      c[row * ldc + col] = 0.0f;
    }
  }

  for (size_t m0 = 0; m0 < m; m0 += bm) {
    for (size_t n0 = 0; n0 < n; n0 += bn) {
      for (size_t k0 = 0; k0 < k; k0 += bk) {
        for (size_t row = m0; row < minimum(m0 + bm, m); ++row) {
          for (size_t col = n0; col < minimum(n0 + bn, n); ++col) {
            float acc = c[row * ldc + col];
            for (size_t inner = k0; inner < minimum(k0 + bk, k); ++inner) {
              acc += fp16_to_fp32(a[row * lda + inner]) *
                     fp16_to_fp32(b[inner * ldb + col]);
            }
            c[row * ldc + col] = acc;
          }
        }
      }
    }
  }
}
