/* Adapted from ggml generic quantized dot code. See source/c/ggml/LICENSE. */

#include "../common/ggml_quant.h"

float weft_reference_q8_0_q8_0(
    const block_q8_0 *x,
    const block_q8_0 *y,
    size_t blocks) {
  float sum = 0.0f;
  for (size_t block = 0; block < blocks; ++block) {
    int integer_sum = 0;
    for (int i = 0; i < GGML_QK32; ++i) {
      integer_sum += x[block].qs[i] * y[block].qs[i];
    }
    sum += (float)integer_sum * ggml_ref_fp16_to_fp32(x[block].d) *
           ggml_ref_fp16_to_fp32(y[block].d);
  }
  return sum;
}
