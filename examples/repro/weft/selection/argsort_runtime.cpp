#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

namespace {

constexpr std::size_t kRows = 8;
constexpr std::size_t kColumns = 32000;
constexpr std::size_t kElements = kRows * kColumns;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

void evict(std::vector<std::uint8_t> &buffer) {
  for (std::size_t index = 0; index < buffer.size(); index += 64) {
    buffer[index] = static_cast<std::uint8_t>(buffer[index] + 1U);
    evictionSink += buffer[index];
  }
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2U];
}

} // namespace

int main() {
  std::vector<float> values(kElements);
  std::vector<std::uint32_t> expected(kElements);
  std::vector<std::uint32_t> actual(kElements);
  for (std::size_t row = 0; row < kRows; ++row) {
    std::uint32_t *order = expected.data() + row * kColumns;
    std::iota(order, order + kColumns, 0U);
    for (std::size_t column = 0; column < kColumns; ++column)
      values[row * kColumns + column] =
          static_cast<float>((row * 104729U + column * 65537U) % 16777213U) +
          static_cast<float>(column) / 65536.0F;
    values[row * kColumns + 3] = -0.0F;
    values[row * kColumns + 7] = 0.0F;
    values[row * kColumns + 11] = values[row * kColumns + 13];
    std::sort(order, order + kColumns, [&](std::uint32_t lhs, std::uint32_t rhs) {
      const float left = values[row * kColumns + lhs];
      const float right = values[row * kColumns + rhs];
      return left < right || (left == right && lhs < rhs);
    });
  }

  std::vector<std::uint32_t> scratch(
      argsort_f32__scratch_elements(kColumns), 0);
  argsort_f32(values.data(), actual.data(), scratch.data(), 0, kRows, kColumns, kColumns,
              kColumns);
  if (actual != expected) {
    for (std::size_t index = 0; index < actual.size(); ++index)
      if (actual[index] != expected[index]) {
        std::fprintf(stderr,
                     "argsort_f32 mismatch at %zu: actual=%u expected=%u\n",
                     index, actual[index], expected[index]);
        return 1;
      }
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(5);
  for (int repetition = 0; repetition < 5; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    argsort_f32(values.data(), actual.data(), scratch.data(), 0, kRows, kColumns, kColumns,
                kColumns);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  std::printf("kernel=argsort_f32\n");
  std::printf("model_shape=batched_logits[rows=8,vocab=32000]\n");
  const double milliseconds = median(samples);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("million_values_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
