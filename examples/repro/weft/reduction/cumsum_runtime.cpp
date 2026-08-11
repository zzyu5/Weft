#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void cumsum_f32(const float *x, float *y, std::size_t row_begin,
                           std::size_t row_end, std::size_t cols,
                           std::size_t stride);

namespace {

constexpr std::size_t kHeads = 64;
constexpr std::size_t kTokens = 4096;
constexpr std::size_t kElements = kHeads * kTokens;
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
    input[index] = static_cast<float>(static_cast<int>(index % 251) - 125) /
                   16384.0F;

  cumsum_f32(input.data(), output.data(), 0, kHeads, kTokens, kTokens);
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t row = 0; row < kHeads; ++row) {
    double prefix = 0.0;
    for (std::size_t column = 0; column < kTokens; ++column) {
      const std::size_t index = row * kTokens + column;
      prefix += input[index];
      const double error = std::abs(static_cast<double>(output[index]) - prefix);
      maxAbsoluteError = std::max(maxAbsoluteError, error);
      maxRelativeError = std::max(
          maxRelativeError, error / std::max(std::abs(prefix), 1.0e-12));
    }
  }
  if (maxAbsoluteError > 2.0e-5 && maxRelativeError > 2.0e-5) {
    std::fprintf(stderr, "cumsum mismatch: max_abs=%g max_rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    cumsum_f32(input.data(), output.data(), 0, kHeads, kTokens, kTokens);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  std::printf("kernel=cumsum_f32\n");
  std::printf("model_shape=mamba_dt[sequences=1,heads=64,tokens=4096]\n");
  std::printf("elements=%zu\n", kElements);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
