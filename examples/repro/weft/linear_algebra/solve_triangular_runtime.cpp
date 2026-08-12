#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

extern "C" void solve_lower_triangular_f32(
    const float *, const float *, float *, size_t, size_t, size_t);

static double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int main() {
  constexpr size_t batches = 8;
  constexpr size_t rows = 256;
  constexpr size_t rhs_columns = 64;
  const size_t matrix_elements = batches * rows * rows;
  const size_t rhs_elements = batches * rows * rhs_columns;
  std::vector<float> matrix(matrix_elements, 0.0f), rhs(rhs_elements);
  std::vector<float> solution(rhs_elements), expected(rhs_elements);
  for (size_t batch = 0; batch < batches; ++batch) {
    for (size_t row = 0; row < rows; ++row) {
      for (size_t inner = 0; inner <= row; ++inner)
        matrix[(batch * rows + row) * rows + inner] =
            inner == row ? 2.0f + static_cast<float>(row % 7) * 0.1f
                         : static_cast<float>(static_cast<int>((row + inner) % 9) - 4) * 0.0002f;
    }
  }
  for (size_t i = 0; i < rhs_elements; ++i)
    rhs[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.01f;
  for (size_t batch = 0; batch < batches; ++batch) {
    for (size_t row = 0; row < rows; ++row) {
      for (size_t column = 0; column < rhs_columns; ++column) {
        float residual = rhs[(batch * rows + row) * rhs_columns + column];
        for (size_t inner = 0; inner < row; ++inner)
          residual -= matrix[(batch * rows + row) * rows + inner] *
                      expected[(batch * rows + inner) * rhs_columns + column];
        expected[(batch * rows + row) * rhs_columns + column] =
            residual / matrix[(batch * rows + row) * rows + row];
      }
    }
  }
  solve_lower_triangular_f32(matrix.data(), rhs.data(), solution.data(), batches,
                             rows, rhs_columns);
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (size_t i = 0; i < solution.size(); ++i) {
    const float error = std::abs(solution[i] - expected[i]);
    max_abs = std::max(max_abs, error);
    max_rel = std::max(max_rel, error / std::max(1.0e-6f, std::abs(expected[i])));
  }
  if (max_abs > 1.0e-5f && max_rel > 1.0e-4f) {
    std::fprintf(stderr, "triangular solve mismatch\n");
    return 1;
  }
  std::vector<unsigned char> eviction(64 * 1024 * 1024, 1);
  std::vector<double> samples;
  for (int repetition = 0; repetition < 10; ++repetition) {
    for (size_t i = 0; i < eviction.size(); i += 64)
      eviction[i] = static_cast<unsigned char>(eviction[i] + 1);
    const auto begin = std::chrono::steady_clock::now();
    solve_lower_triangular_f32(matrix.data(), rhs.data(), solution.data(), batches,
                               rows, rhs_columns);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double operations = static_cast<double>(batches * rhs_columns * rows * rows);
  std::printf("kernel=solve_lower_triangular_f32\n");
  std::printf("model_shape=batches=8;rows=256;rhs_columns=64\n");
  std::printf("max_absolute_error=%.9g\n", max_abs);
  std::printf("max_relative_error=%.9g\n", max_rel);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / (milliseconds * 1.0e6));
  return 0;
}
