#include "../common/ggml_quant.h"

#include <math.h>
#include <stdio.h>

int main(void) {
  block_q8_0 x = {.d = 0x3800u};
  block_q8_0 y = {.d = 0x3400u};

  int integer_sum = 0;
  for (int i = 0; i < GGML_QK32; ++i) {
    x.qs[i] = (int8_t)((i % 11) - 5);
    y.qs[i] = (int8_t)(((3 * i) % 13) - 6);
    integer_sum += x.qs[i] * y.qs[i];
  }

  const float expected = (float)integer_sum * 0.5f * 0.25f;
  const float actual = ggml_source_q8_0_q8_0(&x, &y, 1);
  if (fabsf(actual - expected) > 1.0e-6f) {
    return 1;
  }
  puts("PASS q8_0_q8_0_block_dot");
  return 0;
}
