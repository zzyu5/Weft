/* Adapted from ggml generic quantized dot code. See source/c/ggml/LICENSE. */

#include "../common/ggml_quant.h"

float ggml_source_q4_0_q8_0(
    const block_q4_0 *x,
    const block_q8_0 *y,
    size_t blocks) {
  float sum = 0.0f;
  for (size_t block = 0; block < blocks; ++block) {
    int low_sum = 0;
    int high_sum = 0;
    for (int i = 0; i < GGML_QK32 / 2; ++i) {
      const int low = (x[block].qs[i] & 0x0f) - 8;
      const int high = (x[block].qs[i] >> 4) - 8;
      low_sum += low * y[block].qs[i];
      high_sum += high * y[block].qs[i + GGML_QK32 / 2];
    }
    sum += (float)(low_sum + high_sum) *
           ggml_ref_fp16_to_fp32(x[block].d) *
           ggml_ref_fp16_to_fp32(y[block].d);
  }
  return sum;
}
