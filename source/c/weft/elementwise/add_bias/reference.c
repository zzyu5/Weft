#include <stddef.h>

void weft_reference_add_bias_f32(
    const float *x,
    const float *bias,
    float *y,
    size_t n) {
  for (size_t i = 0; i < n; ++i) {
    y[i] = x[i] + bias[i];
  }
}
