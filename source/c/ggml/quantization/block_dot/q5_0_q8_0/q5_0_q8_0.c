/* Adapted from ggml generic quantized dot code. See source/c/ggml/LICENSE. */

#include "../common/ggml_quant.h"

#include <string.h>

float ggml_source_q5_0_q8_0(
    const block_q5_0 *x,
    const block_q8_0 *y,
    size_t blocks) {
  float sum = 0.0f;
  for (size_t block = 0; block < blocks; ++block) {
    uint32_t high_bits;
    memcpy(&high_bits, x[block].qh, sizeof(high_bits));

    int low_sum = 0;
    int high_sum = 0;
    for (int i = 0; i < GGML_QK32 / 2; ++i) {
      const uint8_t low_bit =
          (uint8_t)(((high_bits >> i) << 4) & 0x10u);
      const uint8_t high_bit =
          (uint8_t)((high_bits >> (i + 12)) & 0x10u);
      const int low = ((x[block].qs[i] & 0x0f) | low_bit) - 16;
      const int high = ((x[block].qs[i] >> 4) | high_bit) - 16;
      low_sum += low * y[block].qs[i];
      high_sum += high * y[block].qs[i + GGML_QK32 / 2];
    }
    sum += ggml_ref_fp16_to_fp32(x[block].d) *
           ggml_ref_fp16_to_fp32(y[block].d) *
           (float)(low_sum + high_sum);
  }
  return sum;
}
