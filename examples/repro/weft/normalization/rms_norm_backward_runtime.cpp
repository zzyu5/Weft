#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void rms_norm_backward_f32(
    const float *gradient, const float *source, float *output,
    std::size_t row_begin, std::size_t row_end, std::size_t columns,
    std::size_t row_stride, float eps);

namespace {

constexpr std::size_t kRows = 512;
constexpr std::size_t kColumns = 4096;
constexpr std::size_t kElements = kRows * kColumns;
constexpr float kEpsilon = 1.0e-5F;
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
  std::vector<float> gradient(kElements);
  std::vector<float> source(kElements);
  std::vector<float> expected(kElements);
  std::vector<float> output(kElements);
  for (std::size_t index = 0; index < kElements; ++index) {
    source[index] =
        static_cast<float>(static_cast<int>(index % 257U) - 128) / 128.0F;
    gradient[index] =
        static_cast<float>(static_cast<int>((index * 17U) % 251U) - 125) /
        256.0F;
  }
  for (std::size_t row = 0; row < kRows; ++row) {
    float sumXX = 0.0F;
    float sumXDZ = 0.0F;
    for (std::size_t column = 0; column < kColumns; ++column) {
      const float x = source[row * kColumns + column];
      const float dz = gradient[row * kColumns + column];
      sumXX += x * x;
      sumXDZ += x * dz;
    }
    const float meanEps = sumXX / static_cast<float>(kColumns) + kEpsilon;
    const float sumEps = sumXX + kEpsilon * static_cast<float>(kColumns);
    const float reciprocalRms = 1.0F / std::sqrt(meanEps);
    const float sourceScale = -sumXDZ / sumEps;
    for (std::size_t column = 0; column < kColumns; ++column) {
      const std::size_t index = row * kColumns + column;
      expected[index] =
          (gradient[index] + source[index] * sourceScale) * reciprocalRms;
    }
  }

  rms_norm_backward_f32(gradient.data(), source.data(), output.data(), 0, kRows,
                        kColumns, kColumns, kEpsilon);
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t index = 0; index < kElements; ++index) {
    const double absolute =
        std::fabs(static_cast<double>(output[index]) - expected[index]);
    const double relative = absolute / std::fmax(1.0, std::fabs(expected[index]));
    maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
    maxRelativeError = std::fmax(maxRelativeError, relative);
  }
  if (maxRelativeError > 2.0e-5) {
    std::fprintf(stderr, "rms_norm_backward_f32 mismatch: abs=%g rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    rms_norm_backward_f32(gradient.data(), source.data(), output.data(), 0,
                          kRows, kColumns, kColumns, kEpsilon);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=rms_norm_backward_f32\n");
  std::printf("model_shape=training_hidden[rows=512,hidden=4096]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1.0e3);
  return 0;
}
