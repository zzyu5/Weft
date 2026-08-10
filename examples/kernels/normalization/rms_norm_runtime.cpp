#include <cmath>
#include <cstddef>
#include <cstdio>

extern "C" void rms_norm_worker(
    const float *x,
    const float *weight,
    float *y,
    std::size_t row_begin,
    std::size_t row_end,
    std::size_t cols,
    std::size_t stride,
    float eps);

int main() {
  constexpr std::size_t rows = 3;
  constexpr std::size_t cols = 17;
  constexpr float eps = 1.0e-5f;
  float x[rows * cols];
  float weight[cols];
  float y[rows * cols];
  for (std::size_t i = 0; i < rows * cols; ++i) {
    x[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125f;
    y[i] = 0.0f;
  }
  for (std::size_t i = 0; i < cols; ++i)
    weight[i] = 0.75f + static_cast<float>(i) * 0.03125f;

  rms_norm_worker(x, weight, y, 0, rows, cols, cols, eps);
  for (std::size_t row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (std::size_t col = 0; col < cols; ++col) {
      const float value = x[row * cols + col];
      sum += value * value;
    }
    const float scale = 1.0f / std::sqrt(sum / static_cast<float>(cols) + eps);
    for (std::size_t col = 0; col < cols; ++col) {
      const std::size_t index = row * cols + col;
      const float expected = x[index] * scale * weight[col];
      if (std::fabs(y[index] - expected) > 1.0e-6f)
        return 1;
    }
  }
  std::puts("PASS weft rms_norm_worker");
  return 0;
}
