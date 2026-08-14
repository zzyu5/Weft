#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kRows = 128;
constexpr std::size_t kColumns = 4096;
constexpr float kEpsilon = 1.0e-5F;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink;

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

void reference(const std::vector<float> &source,
               const std::vector<float> &weight,
               std::vector<float> &output) {
  for (std::size_t row = 0; row < kRows; ++row) {
    float sum = 0.0F;
    for (std::size_t column = 0; column < kColumns; ++column) {
      const float value = source[row * kColumns + column];
      sum += value * value;
    }
    const float inverse =
        1.0F / std::sqrt(sum / static_cast<float>(kColumns) + kEpsilon);
    for (std::size_t column = 0; column < kColumns; ++column)
      output[row * kColumns + column] =
          source[row * kColumns + column] * inverse * weight[column];
  }
}

} // namespace

int main() {
  const std::size_t elements = kRows * kColumns;
  std::vector<float> source(elements);
  std::vector<float> weight(kColumns);
  std::vector<float> primary(elements);
  std::vector<float> equivalent(elements);
  std::vector<float> expected(elements);
  for (std::size_t index = 0; index < elements; ++index)
    source[index] =
        static_cast<float>(static_cast<int>((index * 13U) % 251U) - 125) /
        128.0F;
  for (std::size_t column = 0; column < kColumns; ++column)
    weight[column] =
        0.5F + static_cast<float>((column * 7U) % 97U) / 128.0F;

  reference(source, weight, expected);
  rms_norm_mul_f32(source.data(), weight.data(), primary.data(), 0, kRows,
                   kColumns, kColumns, kColumns, kEpsilon);
  rms_norm_mul_f32_equivalent(source.data(), weight.data(), equivalent.data(),
                              0, kRows, kColumns, kColumns, kColumns,
                              kEpsilon);

  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  double equivalentMaxAbsoluteError = 0.0;
  double equivalentMaxRelativeError = 0.0;
  for (std::size_t index = 0; index < elements; ++index) {
    const double absolute = std::fabs(primary[index] - expected[index]);
    const double relative = absolute / std::fmax(1.0, std::fabs(expected[index]));
    maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
    maxRelativeError = std::fmax(maxRelativeError, relative);
    equivalentMaxAbsoluteError = std::fmax(
        equivalentMaxAbsoluteError,
        static_cast<double>(std::fabs(equivalent[index] - expected[index])));
    equivalentMaxRelativeError = std::fmax(
        equivalentMaxRelativeError,
        static_cast<double>(std::fabs(equivalent[index] - expected[index])) /
            std::fmax(1.0, std::fabs(expected[index])));
  }
  if (maxRelativeError > 2.0e-5 || equivalentMaxRelativeError > 2.0e-5) {
    std::fprintf(stderr,
                 "rms_norm_mul_f32 mismatch: abs=%g rel=%g equivalent=%g/%g\n",
                 maxAbsoluteError, maxRelativeError,
                 equivalentMaxAbsoluteError, equivalentMaxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> primarySamples;
  std::vector<double> equivalentSamples;
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    auto begin = std::chrono::steady_clock::now();
    rms_norm_mul_f32(source.data(), weight.data(), primary.data(), 0, kRows,
                     kColumns, kColumns, kColumns, kEpsilon);
    auto end = std::chrono::steady_clock::now();
    primarySamples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
    evict(eviction);
    begin = std::chrono::steady_clock::now();
    rms_norm_mul_f32_equivalent(source.data(), weight.data(), equivalent.data(),
                                0, kRows, kColumns, kColumns, kColumns,
                                kEpsilon);
    end = std::chrono::steady_clock::now();
    equivalentSamples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double primaryMilliseconds = median(primarySamples);
  const double equivalentMilliseconds = median(equivalentSamples);
  std::printf("kernel=rms_norm_mul_f32\n");
  std::printf("model_shape=hidden[128x4096];weight[4096]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("equivalent_max_absolute_error=%.9g\n",
              equivalentMaxAbsoluteError);
  std::printf("equivalent_max_relative_error=%.9g\n",
              equivalentMaxRelativeError);
  std::printf("primary_median_ms=%.6f\n", primaryMilliseconds);
  std::printf("equivalent_median_ms=%.6f\n", equivalentMilliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(elements) / primaryMilliseconds / 1000.0);
  return 0;
}
