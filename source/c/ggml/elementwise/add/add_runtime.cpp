#include <cmath>
#include <cstddef>
#include <cstdio>

extern "C" void ggml_source_add_f32(
    const float *src0,
    const float *src1,
    float *dst,
    std::size_t rows,
    std::size_t columns);

int main() {
  constexpr std::size_t rows = 4;
  constexpr std::size_t columns = 16;
  float src0[rows * columns];
  float bias[columns];
  float dst[rows * columns];

  for (std::size_t i = 0; i < rows * columns; ++i) {
    src0[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25f;
  }
  for (std::size_t i = 0; i < columns; ++i) {
    bias[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.5f;
  }

  ggml_source_add_f32(src0, bias, dst, rows, columns);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      const std::size_t index = row * columns + column;
      if (std::fabs(dst[index] - (src0[index] + bias[column])) > 1.0e-6f) {
        return 1;
      }
    }
  }

  std::puts("PASS ggml_add_f32");
  return 0;
}
