#include <math.h>
#include <stddef.h>
#include <stdio.h>

void weft_reference_rms_norm_f32(
    const float *x,
    const float *weight,
    float *y,
    size_t rows,
    size_t cols,
    size_t stride,
    float eps);

int main(void) {
  enum { ROWS = 2, COLS = 8, STRIDE = 8 };
  float x[ROWS * STRIDE];
  float weight[COLS];
  float y[ROWS * STRIDE];
  const float eps = 1.0e-5f;

  for (size_t i = 0; i < ROWS * STRIDE; ++i) {
    x[i] = (float)((int)(i % 9) - 4) * 0.25f;
  }
  for (size_t i = 0; i < COLS; ++i) {
    weight[i] = 0.75f + (float)i * 0.0625f;
  }

  weft_reference_rms_norm_f32(x, weight, y, ROWS, COLS, STRIDE, eps);
  for (size_t row = 0; row < ROWS; ++row) {
    float sum_sq = 0.0f;
    for (size_t col = 0; col < COLS; ++col) {
      const float value = x[row * STRIDE + col];
      sum_sq += value * value;
    }
    const float scale = 1.0f / sqrtf(sum_sq / (float)COLS + eps);
    for (size_t col = 0; col < COLS; ++col) {
      const float expected = x[row * STRIDE + col] * scale * weight[col];
      if (fabsf(y[row * STRIDE + col] - expected) > 1.0e-6f) {
        return 1;
      }
    }
  }

  puts("PASS rms_norm_f32");
  return 0;
}
