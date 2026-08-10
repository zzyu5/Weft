#include <math.h>
#include <stddef.h>

void weft_reference_rms_norm_f32(
    const float *x,
    const float *weight,
    float *y,
    size_t rows,
    size_t cols,
    size_t stride,
    float eps) {
  for (size_t row = 0; row < rows; ++row) {
    float sum_sq = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
      const float value = x[row * stride + col];
      sum_sq += value * value;
    }

    const float scale = 1.0f / sqrtf(sum_sq / (float)cols + eps);
    for (size_t col = 0; col < cols; ++col) {
      y[row * stride + col] = x[row * stride + col] * scale * weight[col];
    }
  }
}
