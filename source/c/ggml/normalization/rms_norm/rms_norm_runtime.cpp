#include <cmath>
#include <cstddef>
#include <cstdio>

extern "C" void ggml_source_rms_norm_f32(
    const float *src,
    const float *weight,
    float *dst,
    std::size_t rows,
    std::size_t columns,
    float eps);

int main() {
  constexpr std::size_t rows = 2;
  constexpr std::size_t columns = 16;
  constexpr float eps = 1.0e-5f;
  float src[rows * columns];
  float weight[columns];
  float dst[rows * columns];

  for (std::size_t i = 0; i < rows * columns; ++i) {
    src[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.25f;
  }
  for (std::size_t i = 0; i < columns; ++i) {
    weight[i] = 0.75f + static_cast<float>(i) * 0.03125f;
  }

  ggml_source_rms_norm_f32(src, weight, dst, rows, columns, eps);
  for (std::size_t row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (std::size_t column = 0; column < columns; ++column) {
      const float value = src[row * columns + column];
      sum += static_cast<double>(value * value);
    }
    const float scale =
        1.0f / std::sqrt(static_cast<float>(sum / columns) + eps);
    for (std::size_t column = 0; column < columns; ++column) {
      const std::size_t index = row * columns + column;
      const float expected = src[index] * scale * weight[column];
      if (std::fabs(dst[index] - expected) > 1.0e-6f) {
        return 1;
      }
    }
  }

  std::puts("PASS ggml_rms_norm_f32");
  return 0;
}
