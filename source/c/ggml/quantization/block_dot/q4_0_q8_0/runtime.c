#include "../common/ggml_quant.h"

#include <math.h>
#include <stdio.h>

int main(void) {
  block_q4_0 x = {.d = 0x3c00u};
  block_q8_0 y = {.d = 0x3800u};

  for (int i = 0; i < GGML_QK32 / 2; ++i) {
    x.qs[i] = (uint8_t)((i & 0x0f) | ((15 - i) << 4));
  }
  for (int i = 0; i < GGML_QK32; ++i) {
    y.qs[i] = (int8_t)((i % 7) - 3);
  }

  float expected = 0.0f;
  for (int i = 0; i < GGML_QK32 / 2; ++i) {
    const float low = (float)((x.qs[i] & 0x0f) - 8);
    const float high = (float)((x.qs[i] >> 4) - 8);
    expected += low * (float)y.qs[i] * 0.5f;
    expected += high * (float)y.qs[i + GGML_QK32 / 2] * 0.5f;
  }

  const float actual = weft_reference_q4_0_q8_0(&x, &y, 1);
  if (fabsf(actual - expected) > 1.0e-6f) {
    return 1;
  }
  puts("PASS q4_0_q8_0_block_dot");
  return 0;
}
