#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

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
    size_t bk);

int main(void) {
  static const uint16_t a[] = {
      0x3c00u, 0x3800u, 0xbc00u,
      0x4000u, 0xb800u, 0x3c00u,
  };
  static const uint16_t b[] = {
      0x3c00u, 0x4000u,
      0x3800u, 0xbc00u,
      0x4000u, 0x3400u,
  };
  static const float expected[] = {-0.75f, 1.25f, 3.75f, 4.75f};
  float c[4];

  weft_reference_blocked_gemm_f16_f32(a, b, c, 2, 2, 3, 3, 2, 2, 2, 2, 2);
  for (size_t i = 0; i < 4; ++i) {
    if (fabsf(c[i] - expected[i]) > 1.0e-6f) {
      return 1;
    }
  }

  puts("PASS blocked_gemm_f16_f32");
  return 0;
}
