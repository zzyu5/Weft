#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
constexpr std::size_t kTokens = 2048;
constexpr std::size_t kChannels = 256;
constexpr std::size_t kTaps = 7;
constexpr std::size_t kDilation = 2;
constexpr std::size_t kPadding = (kTaps - 1) * kDilation;
constexpr std::size_t kElements = kTokens * kChannels;
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
  std::vector<float> source((kTokens + kPadding) * kChannels, 0.0F);
  std::vector<float> weight(kChannels * kTaps);
  std::vector<float> output(kElements, 0.0F);
  std::vector<float> reference(kElements, 0.0F);
  for (std::size_t index = 0; index < kElements; ++index)
    source[kPadding * kChannels + index] =
        static_cast<float>(static_cast<int>((index * 31) % 251) - 125) /
        512.0F;
  for (std::size_t index = 0; index < weight.size(); ++index)
    weight[index] = static_cast<float>(static_cast<int>((index * 7) % 29) - 14) /
                    128.0F;
  for (std::size_t token = 0; token < kTokens; ++token)
    for (std::size_t channel = 0; channel < kChannels; ++channel)
      for (std::size_t tap = 0; tap < kTaps; ++tap)
        if (token >= tap * kDilation)
          reference[token * kChannels + channel] +=
              source[(token + kPadding - tap * kDilation) * kChannels +
                     channel] *
              weight[channel * kTaps + tap];
  dilated_causal_conv1d_f32(source.data(), weight.data(), output.data(), kTokens,
                            kChannels, kTaps, kDilation);
  double maximum = 0.0;
  for (std::size_t index = 0; index < output.size(); ++index)
    maximum = std::max(maximum, std::abs(static_cast<double>(output[index]) -
                                         reference[index]));
  if (maximum > 2.0e-6) {
    std::fprintf(stderr, "dilated causal conv mismatch: %g\n", maximum);
    return 1;
  }
  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  for (int repetition = 0; repetition < 5; ++repetition) {
    std::fill(output.begin(), output.end(), 0.0F);
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    dilated_causal_conv1d_f32(source.data(), weight.data(), output.data(), kTokens,
                              kChannels, kTaps, kDilation);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=dilated_causal_conv1d_f32\n");
  std::printf("model_shape=causal_conv[tokens=%zu;channels=%zu;taps=%zu;dilation=%zu;padding=%zu]\n",
              kTokens, kChannels, kTaps, kDilation, kPadding);
  std::printf("max_absolute_error=%.9g\n", maximum);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gflops=%.6f\n",
              static_cast<double>(kElements) * kTaps * 2.0 / milliseconds / 1.0e6);
  return 0;
}
