#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr std::size_t kN = 4096;
constexpr std::size_t kK = 4096;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
constexpr _Float16 kHalfValues[] = {
    static_cast<_Float16>(1.0F),
    static_cast<_Float16>(0.5F),
    static_cast<_Float16>(-1.0F),
    static_cast<_Float16>(2.0F),
};
constexpr float kFloatValues[] = {1.0F, 0.5F, -1.0F, 2.0F};
volatile std::uint64_t evictionSink;

std::size_t phaseRows(const char *phase) {
  if (std::strcmp(phase, "decode") == 0)
    return 1;
  if (std::strcmp(phase, "prefill") == 0)
    return 128;
  return 0;
}

std::size_t repetitions(const char *text) {
  char *end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (*text == '\0' || *end != '\0' || value == 0)
    return 0;
  return static_cast<std::size_t>(value);
}

void evict(std::vector<std::uint8_t> &buffer) {
  for (std::size_t index = 0; index < buffer.size(); index += 64) {
    buffer[index] = static_cast<std::uint8_t>(buffer[index] + 1U);
    evictionSink += buffer[index];
  }
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  return samples.size() % 2 == 0
             ? 0.5 * (samples[middle - 1] + samples[middle])
             : samples[middle];
}

float reference(std::size_t row, std::size_t column) {
  float result = 0.0F;
  for (std::size_t inner = 0; inner < kK; ++inner) {
    const std::size_t aChoice = (row + 3 * inner + 1) % 4;
    const std::size_t bChoice = (5 * column + 3 * inner + 2) % 4;
    result += kFloatValues[aChoice] * kFloatValues[bChoice];
  }
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <decode|prefill> <repetitions>\n", argv[0]);
    return 2;
  }
  const std::size_t m = phaseRows(argv[1]);
  const std::size_t repeatCount = repetitions(argv[2]);
  if (m == 0 || repeatCount == 0) {
    std::fprintf(stderr, "invalid phase or repetitions\n");
    return 2;
  }

  std::vector<float> a(m * kK);
  std::vector<_Float16> b(kK * kN);
  std::vector<float> c(m * kN, 0.0F);
  std::vector<_Float16> aF16(m * kK);
  for (std::size_t row = 0; row < m; ++row)
    for (std::size_t inner = 0; inner < kK; ++inner)
      a[row * kK + inner] = kFloatValues[(row + 3 * inner + 1) % 4];
  for (std::size_t column = 0; column < kN; ++column)
    for (std::size_t inner = 0; inner < kK; ++inner)
      b[column * kK + inner] =
          kHalfValues[(5 * column + 3 * inner + 2) % 4];

  gemm_worker(a.data(), b.data(), c.data(), aF16.data(), 0, m, kN, kK, kK,
              kN, kN);
  const std::size_t sampleRows[] = {0, m / 2, m - 1};
  const std::size_t sampleColumns[] = {0, 1, kN / 2, kN - 1};
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t row : sampleRows) {
    for (std::size_t column : sampleColumns) {
      const double expected = reference(row, column);
      const double actual = c[row * kN + column];
      const double absolute = std::fabs(actual - expected);
      const double relative = absolute / std::fmax(1.0, std::fabs(expected));
      maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
      maxRelativeError = std::fmax(maxRelativeError, relative);
      if (relative > 1.0e-5) {
        std::fprintf(
            stderr,
            "mismatch at (%zu,%zu): weft=%g reference=%g relative=%g\n",
            row,
            column,
            actual,
            expected,
            relative);
        return 1;
      }
    }
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(repeatCount);
  for (std::size_t repetition = 0; repetition < repeatCount; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    gemm_worker(a.data(), b.data(), c.data(), aF16.data(), 0, m, kN, kK, kK,
                kN, kN);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  const double operations = 2.0 * static_cast<double>(m) *
                            static_cast<double>(kN) * static_cast<double>(kK);
  std::printf("kernel=f16_dense_projection\n");
  std::printf("model_shape=Llama-8B.hidden_projection\n");
  std::printf("phase=%s\n", argv[1]);
  std::printf("M=%zu\nN=%zu\nK=%zu\n", m, kN, kK);
  std::printf("timing=cold-cache-64MiB-eviction-median\n");
  std::printf("repetitions=%zu\n", repeatCount);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("weft_ms=%.6f\n", milliseconds);
  std::printf("weft_gop_s=%.6f\n", operations / milliseconds / 1.0e6);
  return 0;
}
