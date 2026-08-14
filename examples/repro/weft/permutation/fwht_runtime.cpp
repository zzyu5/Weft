#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kBatches = 1024;
constexpr std::size_t kExtent = 4096;
constexpr std::size_t kStages = 12;
constexpr std::size_t kElements = kBatches * kExtent;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

void reference(std::vector<float> &data) {
  for (std::size_t batch = 0; batch < kBatches; ++batch)
    for (std::size_t half = 1; half < kExtent; half *= 2)
      for (std::size_t block = 0; block < kExtent; block += 2 * half)
        for (std::size_t lane = 0; lane < half; ++lane) {
          const std::size_t leftIndex = batch * kExtent + block + lane;
          const std::size_t rightIndex = leftIndex + half;
          const float left = data[leftIndex];
          const float right = data[rightIndex];
          data[leftIndex] = left + right;
          data[rightIndex] = left - right;
        }
}

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

} // namespace

int main() {
  std::vector<float> input(kElements);
  for (std::size_t index = 0; index < kElements; ++index)
    input[index] =
        static_cast<float>(static_cast<int>((index * 17U) % 251U) - 125) /
        4096.0F;
  std::vector<float> expected = input;
  std::vector<float> output = input;
  reference(expected);
  fwht_f32(output.data(), kBatches, kExtent, kStages);
  double maxAbsolute = 0.0;
  double maxRelative = 0.0;
  for (std::size_t index = 0; index < kElements; ++index) {
    const double error = std::abs(static_cast<double>(output[index]) -
                                  static_cast<double>(expected[index]));
    maxAbsolute = std::max(maxAbsolute, error);
    maxRelative = std::max(
        maxRelative,
        error / std::max(std::abs(static_cast<double>(expected[index])), 1.0e-6));
  }
  if (maxAbsolute != 0.0 || maxRelative != 0.0) {
    std::fprintf(stderr, "FWHT mismatch: abs=%g rel=%g\n", maxAbsolute,
                 maxRelative);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(5);
  for (int repetition = 0; repetition < 5; ++repetition) {
    output = input;
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    fwht_f32(output.data(), kBatches, kExtent, kStages);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double butterflies =
      static_cast<double>(kElements) * std::log2(static_cast<double>(kExtent)) /
      2.0;
  std::printf("kernel=fwht_f32\n");
  std::printf("model_shape=hadamard_projection[batches=%zu,extent=%zu]\n",
              kBatches, kExtent);
  std::printf("max_absolute_error=%.9g\n", maxAbsolute);
  std::printf("max_relative_error=%.9g\n", maxRelative);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", 2.0 * butterflies / milliseconds / 1.0e6);
  return 0;
}
