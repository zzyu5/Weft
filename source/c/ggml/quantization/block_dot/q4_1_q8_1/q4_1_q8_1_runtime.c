#include "../common/ggml_quant.h"

#include <math.h>
#include <stdio.h>

int main(void) {
  block_q4_1 x = {.d = 0x3800u, .m = 0xb800u};
  block_q8_1 y = {.d = 0x3400u, .s = 0x4800u};

  for (int i = 0; i < GGML_QK32 / 2; ++i) {
    x.qs[i] = (uint8_t)(((3 * i) & 0x0f) | (((5 * i + 1) & 0x0f) << 4));
  }
  for (int i = 0; i < GGML_QK32; ++i) {
    y.qs[i] = 1;
  }

  float expected = 0.0f;
  for (int i = 0; i < GGML_QK32 / 2; ++i) {
    const float low = 0.5f * (float)(x.qs[i] & 0x0f) - 0.5f;
    const float high = 0.5f * (float)(x.qs[i] >> 4) - 0.5f;
    expected += low * 0.25f;
    expected += high * 0.25f;
  }

  const float actual = ggml_source_q4_1_q8_1(&x, &y, 1);
  if (fabsf(actual - expected) > 1.0e-6f) {
    return 1;
  }
  puts("PASS q4_1_q8_1_block_dot");
  return 0;
}
