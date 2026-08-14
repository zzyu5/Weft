#include "../common/ggml_quant.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
  block_q5_0 x = {.d = 0x3400u, .qh = {0xa5u, 0x5au, 0x3cu, 0xc3u}};
  block_q8_0 y = {.d = 0x3800u};

  for (int i = 0; i < GGML_QK32 / 2; ++i) {
    x.qs[i] = (uint8_t)(((7 * i) & 0x0f) | (((2 * i + 3) & 0x0f) << 4));
  }
  for (int i = 0; i < GGML_QK32; ++i) {
    y.qs[i] = (int8_t)((i % 9) - 4);
  }

  uint32_t high_bits;
  memcpy(&high_bits, x.qh, sizeof(high_bits));
  float expected = 0.0f;
  for (int i = 0; i < GGML_QK32 / 2; ++i) {
    const int low_bit = (int)(((high_bits >> i) << 4) & 0x10u);
    const int high_bit = (int)((high_bits >> (i + 12)) & 0x10u);
    const float low = (float)(((x.qs[i] & 0x0f) | low_bit) - 16);
    const float high = (float)(((x.qs[i] >> 4) | high_bit) - 16);
    expected += low * (float)y.qs[i] * 0.125f;
    expected += high * (float)y.qs[i + GGML_QK32 / 2] * 0.125f;
  }

  const float actual =
      q5_0_q8_0((const uint8_t *)&x, (const uint8_t *)&y, 1);
  if (fabsf(actual - expected) > 1.0e-6f) {
    return 1;
  }
  puts("PASS q5_0_q8_0_block_dot");
  return 0;
}
