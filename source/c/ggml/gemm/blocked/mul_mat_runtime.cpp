#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

extern "C" void ggml_source_mul_mat_f16_f16(
    const std::uint16_t *src0,
    const std::uint16_t *src1,
    float *dst,
    std::size_t rows,
    std::size_t columns,
    std::size_t inner);

int main() {
  constexpr std::size_t rows = 16;
  constexpr std::size_t columns = 16;
  constexpr std::size_t inner = 16;
  static const std::uint16_t half_values[] = {
      0x3c00u, 0x3800u, 0xbc00u, 0x4000u,
  };
  static const float float_values[] = {1.0f, 0.5f, -1.0f, 2.0f};

  std::uint16_t src0[rows * inner];
  std::uint16_t src1[inner * columns];
  float src0_f32[rows * inner];
  float src1_f32[inner * columns];
  float dst[rows * columns];

  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t k = 0; k < inner; ++k) {
      const std::size_t index = row * inner + k;
      const std::size_t choice = (row + 3 * k + 1) % 4;
      src0[index] = half_values[choice];
      src0_f32[index] = float_values[choice];
    }
  }
  for (std::size_t column = 0; column < columns; ++column) {
    for (std::size_t k = 0; k < inner; ++k) {
      const std::size_t index = column * inner + k;
      const std::size_t choice = (5 * column + 3 * k + 2) % 4;
      src1[index] = half_values[choice];
      src1_f32[index] = float_values[choice];
    }
  }

  ggml_source_mul_mat_f16_f16(src0, src1, dst, rows, columns, inner);
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t column = 0; column < columns; ++column) {
      float expected = 0.0f;
      for (std::size_t k = 0; k < inner; ++k) {
        expected += src0_f32[row * inner + k] *
                    src1_f32[column * inner + k];
      }
      if (std::fabs(dst[column * rows + row] - expected) > 1.0e-5f) {
        return 1;
      }
    }
  }

  std::puts("PASS ggml_mul_mat_f16_f16");
  return 0;
}
