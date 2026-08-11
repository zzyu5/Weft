#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void out_product_f32(
    const float *lhs, const float *rhs, float *output, std::size_t rows,
    std::size_t columns, std::size_t samples, std::size_t lhs_sample_stride,
    std::size_t rhs_sample_stride, std::size_t output_row_stride);

namespace {

constexpr std::size_t kRows = 2048;
constexpr std::size_t kColumns = 5632;
constexpr std::size_t kSamples = 32;
constexpr std::size_t kOutputElements = kRows * kColumns;
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

float reference(const std::vector<float> &lhs, const std::vector<float> &rhs,
                std::size_t row, std::size_t column) {
  float value = 0.0F;
  for (std::size_t sample = 0; sample < kSamples; ++sample)
    value += lhs[sample * kRows + row] * rhs[sample * kColumns + column];
  return value;
}

} // namespace

int main() {
  std::vector<float> lhs(kSamples * kRows);
  std::vector<float> rhs(kSamples * kColumns);
  std::vector<float> output(kOutputElements);
  for (std::size_t index = 0; index < lhs.size(); ++index)
    lhs[index] =
        static_cast<float>(static_cast<int>(index % 127U) - 63) / 64.0F;
  for (std::size_t index = 0; index < rhs.size(); ++index)
    rhs[index] =
        static_cast<float>(static_cast<int>((index * 3U) % 131U) - 65) /
        128.0F;

  out_product_f32(lhs.data(), rhs.data(), output.data(), kRows, kColumns,
                  kSamples, kRows, kColumns, kColumns);
  const std::size_t sampleRows[] = {0, kRows / 2U, kRows - 1U};
  const std::size_t sampleColumns[] = {0, 1, kColumns / 2U, kColumns - 1U};
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t row : sampleRows)
    for (std::size_t column : sampleColumns) {
      const double expected = reference(lhs, rhs, row, column);
      const double actual = output[row * kColumns + column];
      const double absolute = std::fabs(actual - expected);
      const double relative = absolute / std::fmax(1.0, std::fabs(expected));
      maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
      maxRelativeError = std::fmax(maxRelativeError, relative);
    }
  if (maxRelativeError > 1.0e-5) {
    std::fprintf(stderr, "out_product_f32 mismatch: abs=%g rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(3);
  for (int repetition = 0; repetition < 3; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    out_product_f32(lhs.data(), rhs.data(), output.data(), kRows, kColumns,
                    kSamples, kRows, kColumns, kColumns);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double operations =
      2.0 * static_cast<double>(kRows * kColumns * kSamples);
  std::printf("kernel=out_product_f32\n");
  std::printf(
      "model_shape=TinyLlama_linear_grad[rows=2048,columns=5632,samples=32]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / milliseconds / 1.0e6);
  return 0;
}
