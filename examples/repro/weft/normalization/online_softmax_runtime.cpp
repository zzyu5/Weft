#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kRows = 32 * 128;
constexpr std::size_t kColumns = 128;
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
  const std::size_t middle = samples.size() / 2;
  return samples.size() % 2 == 0
             ? 0.5 * (samples[middle - 1] + samples[middle])
             : samples[middle];
}

} // namespace

int main() {
  std::vector<float> input(kElements);
  std::vector<float> output(kElements);
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] =
        static_cast<float>(static_cast<int>(index % 31) - 15) / 16.0F;

  softmax_f32(input.data(), output.data(), 0, kRows, kColumns, kColumns);
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t row = 0; row < kRows; ++row) {
    const float *source = input.data() + row * kColumns;
    const float *result = output.data() + row * kColumns;
    double maximum = source[0];
    for (std::size_t column = 1; column < kColumns; ++column)
      maximum = std::max(maximum, static_cast<double>(source[column]));
    double sum = 0.0;
    for (std::size_t column = 0; column < kColumns; ++column)
      sum += std::exp(static_cast<double>(source[column]) - maximum);
    for (std::size_t column = 0; column < kColumns; ++column) {
      const double reference =
          std::exp(static_cast<double>(source[column]) - maximum) / sum;
      const double error = std::abs(static_cast<double>(result[column]) - reference);
      maxAbsoluteError = std::max(maxAbsoluteError, error);
      maxRelativeError = std::max(
          maxRelativeError, error / std::max(std::abs(reference), 1.0e-12));
    }
  }
  if (maxAbsoluteError > 3.0e-6 && maxRelativeError > 3.0e-5) {
    std::fprintf(stderr, "Softmax mismatch: max_abs=%g max_rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    softmax_f32(input.data(), output.data(), 0, kRows, kColumns, kColumns);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  std::printf("kernel=softmax_f32\n");
  std::printf("model_shape=attention[heads=32,Q=128,K=128]\n");
  std::printf("rows=%zu\n", kRows);
  std::printf("columns=%zu\n", kColumns);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
