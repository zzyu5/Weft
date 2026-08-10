#include "../common/ggml_quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
  block_q5_1 x = {
      .d = 0x3400u,
      .m = 0xb800u,
      .qh = {0x96u, 0x69u, 0x3cu, 0xc3u},
  };
  block_q8_1 y = {.d = 0x3400u, .s = 0x4800u};

  for (int i = 0; i < GGML_QK32 / 2; ++i) {
    x.qs[i] = (uint8_t)(((5 * i + 2) & 0x0f) | (((3 * i) & 0x0f) << 4));
  }
  for (int i = 0; i < GGML_QK32; ++i) {
    y.qs[i] = 1;
  }

  uint32_t high_bits;
  memcpy(&high_bits, x.qh, sizeof(high_bits));
  float expected = 0.0f;
  for (int i = 0; i < GGML_QK32 / 2; ++i) {
    const int low_bit = (int)(((high_bits >> i) << 4) & 0x10u);
    const int high_bit = (int)((high_bits >> (i + 12)) & 0x10u);
    const float low = 0.25f * (float)((x.qs[i] & 0x0f) | low_bit) - 0.5f;
    const float high = 0.25f * (float)((x.qs[i] >> 4) | high_bit) - 0.5f;
    expected += low * 0.25f;
    expected += high * 0.25f;
  }

  const float actual = ggml_source_q5_1_q8_1(&x, &y, 1);
  if (fabsf(actual - expected) > 1.0e-6f) {
    return 1;
  }
  puts("PASS q5_1_q8_1_block_dot");
  return 0;
}
