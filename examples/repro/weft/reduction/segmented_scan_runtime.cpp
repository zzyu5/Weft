#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void segmented_inclusive_scan_f32(
    const float *values, const std::uint8_t *segment_starts, float *output,
    std::size_t begin, std::size_t end);
extern "C" void segmented_inclusive_scan_f32_equivalent(
    const float *values, const std::uint8_t *segment_starts, float *output,
    std::size_t begin, std::size_t end);

namespace {

constexpr std::size_t kElements = 4U * 1024U * 1024U;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

using Kernel = void (*)(const float *, const std::uint8_t *, float *,
                        std::size_t, std::size_t);

void evict(std::vector<std::uint8_t> &buffer) {
  for (std::size_t index = 0; index < buffer.size(); index += 64) {
    buffer[index] = static_cast<std::uint8_t>(buffer[index] + 1U);
    evictionSink += buffer[index];
  }
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

double run(Kernel kernel, const std::vector<float> &values,
           const std::vector<std::uint8_t> &starts, std::vector<float> &output,
           std::vector<std::uint8_t> &eviction) {
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    kernel(values.data(), starts.data(), output.data(), 0, kElements);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  return median(samples);
}

bool compare(const std::vector<float> &actual,
             const std::vector<float> &expected, double &maxAbsolute,
             double &maxRelative) {
  maxAbsolute = 0.0;
  maxRelative = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double error = std::abs(static_cast<double>(actual[index]) -
                                  static_cast<double>(expected[index]));
    maxAbsolute = std::max(maxAbsolute, error);
    maxRelative = std::max(
        maxRelative,
        error / std::max(std::abs(static_cast<double>(expected[index])), 1.0e-6));
  }
  return maxAbsolute <= 3.0e-4 || maxRelative <= 3.0e-4;
}

} // namespace

int main() {
  std::vector<float> values(kElements);
  std::vector<std::uint8_t> starts(kElements, 0);
  std::vector<float> expected(kElements);
  for (std::size_t index = 0; index < kElements; ++index)
    values[index] =
        static_cast<float>(static_cast<int>((index * 37U) % 101U) - 50) /
        4096.0F;

  std::size_t position = 0;
  std::size_t segment = 0;
  while (position < kElements) {
    starts[position] = 1;
    const std::size_t length =
        1U + ((segment * 17U + segment * segment * 3U) % 257U);
    position = std::min(kElements, position + length);
    ++segment;
  }

  float carry = 0.0F;
  for (std::size_t index = 0; index < kElements; ++index) {
    if (starts[index] != 0)
      carry = 0.0F;
    carry += values[index];
    expected[index] = carry;
  }

  std::vector<float> primary(kElements, 0.0F);
  std::vector<float> equivalent(kElements, 0.0F);
  segmented_inclusive_scan_f32(values.data(), starts.data(), primary.data(), 0,
                               kElements);
  segmented_inclusive_scan_f32_equivalent(
      values.data(), starts.data(), equivalent.data(), 0, kElements);
  double primaryAbsolute = 0.0;
  double primaryRelative = 0.0;
  double equivalentAbsolute = 0.0;
  double equivalentRelative = 0.0;
  if (!compare(primary, expected, primaryAbsolute, primaryRelative) ||
      !compare(equivalent, expected, equivalentAbsolute, equivalentRelative)) {
    std::fprintf(stderr,
                 "segmented scan mismatch: primary_abs=%g primary_rel=%g "
                 "equivalent_abs=%g equivalent_rel=%g\n",
                 primaryAbsolute, primaryRelative, equivalentAbsolute,
                 equivalentRelative);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  const double primaryMilliseconds =
      run(segmented_inclusive_scan_f32, values, starts, primary, eviction);
  const double equivalentMilliseconds = run(
      segmented_inclusive_scan_f32_equivalent, values, starts, equivalent,
      eviction);
  std::printf("kernel=segmented_inclusive_scan_f32\n");
  std::printf("model_shape=ragged_state[elements=%zu,segments=%zu]\n",
              kElements, segment);
  std::printf("max_absolute_error=%.9g\n", primaryAbsolute);
  std::printf("max_relative_error=%.9g\n", primaryRelative);
  std::printf("primary_median_ms=%.6f\n", primaryMilliseconds);
  std::printf("equivalent_median_ms=%.6f\n", equivalentMilliseconds);
  std::printf("primary_melements_s=%.6f\n",
              static_cast<double>(kElements) / primaryMilliseconds / 1000.0);
  std::printf("equivalent_melements_s=%.6f\n",
              static_cast<double>(kElements) / equivalentMilliseconds / 1000.0);
  return 0;
}
