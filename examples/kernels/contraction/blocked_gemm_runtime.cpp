#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void gemm_worker(
    const std::uint16_t *a,
    const std::uint16_t *b,
    float *c,
    std::size_t m_begin,
    std::size_t m_end,
    std::size_t n,
    std::size_t k,
    std::size_t lda,
    std::size_t ldb,
    std::size_t ldc);

int main() {
  constexpr std::size_t m = 17;
  constexpr std::size_t n = 18;
  constexpr std::size_t k = 19;
  static const std::uint16_t half_values[] = {
      0x3c00u, 0x3800u, 0xbc00u, 0x4000u,
  };
  static const float float_values[] = {1.0f, 0.5f, -1.0f, 2.0f};
  std::vector<std::uint16_t> a(m * k);
  std::vector<std::uint16_t> b(k * n);
  std::vector<float> a32(m * k);
  std::vector<float> b32(k * n);
  std::vector<float> c(m * n, 0.0f);
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t inner = 0; inner < k; ++inner) {
      const std::size_t index = row * k + inner;
      const std::size_t choice = (row + 3 * inner + 1) % 4;
      a[index] = half_values[choice];
      a32[index] = float_values[choice];
    }
  }
  for (std::size_t inner = 0; inner < k; ++inner) {
    for (std::size_t column = 0; column < n; ++column) {
      const std::size_t index = inner * n + column;
      const std::size_t choice = (5 * column + 3 * inner + 2) % 4;
      b[index] = half_values[choice];
      b32[index] = float_values[choice];
    }
  }

  gemm_worker(a.data(), b.data(), c.data(), 0, m, n, k, k, n, n);
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t column = 0; column < n; ++column) {
      float expected = 0.0f;
      for (std::size_t inner = 0; inner < k; ++inner)
        expected += a32[row * k + inner] * b32[inner * n + column];
      if (std::fabs(c[row * n + column] - expected) > 1.0e-5f)
        return 1;
    }
  }
  std::puts("PASS weft gemm_worker");
  return 0;
}
