#include "../common/ggml_quant.h"

#include <math.h>
#include <stdio.h>

int main(void) {
  block_q4_K x = {
      .d = 0x3800u,
      .dmin = 0x3400u,
      .scales = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 0x21u, 0x43u, 0x65u, 0x07u},
  };
  block_q8_K y = {.d = 0.125f};

  for (int i = 0; i < GGML_QK_K / 2; ++i) {
    x.qs[i] = (uint8_t)(((3 * i + 1) & 0x0f) |
                        (((5 * i + 2) & 0x0f) << 4));
  }
  for (int i = 0; i < GGML_QK_K; ++i) {
    y.qs[i] = (int8_t)((i % 13) - 6);
  }
  for (int group = 0; group < GGML_QK_K / 16; ++group) {
    int sum = 0;
    for (int i = 0; i < 16; ++i) {
      sum += y.qs[group * 16 + i];
    }
    y.bsums[group] = (int16_t)sum;
  }

  float expected = 0.0f;
  for (int i = 0; i < GGML_QK_K; ++i) {
    const int group = i / 64;
    const int offset = i % 64;
    const uint8_t packed = x.qs[group * 32 + offset % 32];
    const int quant = offset < 32 ? packed & 0x0f : packed >> 4;
    uint8_t scale;
    uint8_t minimum;
    ggml_ref_get_scale_min_k4(i / 32, x.scales, &scale, &minimum);
    const float x_value = 0.5f * (float)scale * (float)quant -
                          0.25f * (float)minimum;
    expected += x_value * (0.125f * (float)y.qs[i]);
  }

  const float actual = ggml_source_q4_K_q8_K(&x, &y, 1);
  const float tolerance = 1.0e-5f * fmaxf(1.0f, fabsf(expected));
  if (fabsf(actual - expected) > tolerance) {
    return 1;
  }
  puts("PASS q4_K_q8_K_block_dot");
  return 0;
}
