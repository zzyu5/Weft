#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void interleaved_rope_f32(
    const float *, const float *, float *, std::size_t, std::size_t, std::size_t,
    std::size_t, std::size_t);

namespace {
constexpr std::size_t kTokens = 128;
constexpr std::size_t kHeads = 32;
constexpr std::size_t kPairs = 64;
constexpr std::size_t kHeadStride = 2 * kPairs;
constexpr std::size_t kTokenStride = kHeads * kHeadStride;
constexpr std::size_t kElements = kTokens * kTokenStride;
constexpr std::size_t kEvictionBytes = 64U << 20U;
volatile std::uint64_t evictionSink = 0;
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
  std::vector<float> source(kElements);
  std::vector<float> angles(2 * kTokens * kPairs);
  std::vector<float> output(kElements);
  std::vector<float> reference(kElements);
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] = static_cast<float>(static_cast<int>((index * 13) % 257) - 128) /
                    256.0F;
  for (std::size_t token = 0; token < kTokens; ++token)
    for (std::size_t pair = 0; pair < kPairs; ++pair) {
      const float angle = static_cast<float>(token) *
                          std::pow(10000.0F, -static_cast<float>(pair) / kPairs);
      angles[2 * (token * kPairs + pair)] = std::cos(angle);
      angles[2 * (token * kPairs + pair) + 1] = std::sin(angle);
    }
  for (std::size_t token = 0; token < kTokens; ++token)
    for (std::size_t head = 0; head < kHeads; ++head)
      for (std::size_t pair = 0; pair < kPairs; ++pair) {
        const std::size_t base = token * kTokenStride + head * kHeadStride + 2 * pair;
        const float c = angles[2 * (token * kPairs + pair)];
        const float s = angles[2 * (token * kPairs + pair) + 1];
        reference[base] = source[base] * c - source[base + 1] * s;
        reference[base + 1] = source[base] * s + source[base + 1] * c;
      }
  interleaved_rope_f32(source.data(), angles.data(), output.data(), kTokens,
                       kHeads, kPairs, kTokenStride, kHeadStride);
  double maximum = 0.0;
  for (std::size_t index = 0; index < output.size(); ++index)
    maximum = std::max(maximum, std::abs(static_cast<double>(output[index]) -
                                         reference[index]));
  if (maximum > 1.0e-6) {
    std::fprintf(stderr, "interleaved rope mismatch: %g\n", maximum);
    return 1;
  }
  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  for (int repetition = 0; repetition < 7; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    interleaved_rope_f32(source.data(), angles.data(), output.data(), kTokens,
                         kHeads, kPairs, kTokenStride, kHeadStride);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=interleaved_rope_f32\n");
  std::printf("model_shape=rope[tokens=%zu;heads=%zu;D=%zu]\n", kTokens, kHeads,
              2 * kPairs);
  std::printf("max_absolute_error=%.9g\n", maximum);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("logical_gbs=%.6f\n",
              static_cast<double>(kElements) * 3.0 * sizeof(float) /
                  milliseconds / 1.0e6);
  return 0;
}
