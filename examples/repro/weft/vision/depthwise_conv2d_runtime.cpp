#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kBatch = 1;
constexpr std::size_t kHeight = 112;
constexpr std::size_t kWidth = 112;
constexpr std::size_t kChannels = 32;
constexpr std::size_t kKernel = 3;
constexpr std::size_t kPadding = 1;
constexpr std::size_t kOutputHeight = 112;
constexpr std::size_t kOutputWidth = 112;
constexpr std::size_t kOutputElements =
    kBatch * kOutputHeight * kOutputWidth * kChannels;
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
  return samples[samples.size() / 2];
}

} // namespace

int main() {
  std::vector<float> source(kBatch * kHeight * kWidth * kChannels);
  std::vector<float> weight(kKernel * kKernel * kChannels);
  std::vector<float> expected(kOutputElements);
  std::vector<float> output(kOutputElements);
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] =
        static_cast<float>(static_cast<int>(index % 257U) - 128) / 256.0F;
  for (std::size_t index = 0; index < weight.size(); ++index)
    weight[index] =
        static_cast<float>(static_cast<int>(index % 31U) - 15) / 64.0F;

  for (std::size_t y = 0; y < kOutputHeight; ++y)
    for (std::size_t x = 0; x < kOutputWidth; ++x)
      for (std::size_t channel = 0; channel < kChannels; ++channel) {
        float value = 0.0F;
        for (std::size_t ky = 0; ky < kKernel; ++ky)
          for (std::size_t kx = 0; kx < kKernel; ++kx) {
            const std::size_t paddedY = y + ky;
            const std::size_t paddedX = x + kx;
            if (paddedY < kPadding || paddedY >= kHeight + kPadding ||
                paddedX < kPadding || paddedX >= kWidth + kPadding)
              continue;
            const std::size_t sourceY = paddedY - kPadding;
            const std::size_t sourceX = paddedX - kPadding;
            value += source[(sourceY * kWidth + sourceX) * kChannels +
                            channel] *
                     weight[(ky * kKernel + kx) * kChannels + channel];
          }
        expected[(y * kOutputWidth + x) * kChannels + channel] = value;
      }

  depthwise_conv2d_f32(
      source.data(), weight.data(), output.data(), kBatch, kHeight, kWidth,
      kChannels, kOutputHeight, kOutputWidth, kKernel, kKernel, 1, 1,
      kPadding, kPadding, 1, 1);
  double maxAbsoluteError = 0.0;
  for (std::size_t index = 0; index < output.size(); ++index)
    maxAbsoluteError = std::fmax(
        maxAbsoluteError, std::fabs(static_cast<double>(output[index]) -
                                    static_cast<double>(expected[index])));
  if (maxAbsoluteError > 1.0e-6) {
    std::fprintf(stderr, "depthwise_conv2d_f32 mismatch: max_abs=%g\n",
                 maxAbsoluteError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    depthwise_conv2d_f32(
        source.data(), weight.data(), output.data(), kBatch, kHeight, kWidth,
        kChannels, kOutputHeight, kOutputWidth, kKernel, kKernel, 1, 1,
        kPadding, kPadding, 1, 1);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=depthwise_conv2d_f32\n");
  std::printf("model_shape=mobilenet[N=1,H=112,W=112,C=32,K=3,pad=1]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n",
              2.0 * static_cast<double>(kOutputElements * kKernel * kKernel) /
                  milliseconds / 1.0e6);
  return 0;
}
