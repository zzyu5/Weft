#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kBatch = 1;
constexpr std::size_t kChannels = 64;
constexpr std::size_t kInputHeight = 112;
constexpr std::size_t kInputWidth = 112;
constexpr std::size_t kKernel = 2;
constexpr std::size_t kStride = 2;
constexpr std::size_t kOutputHeight = kInputHeight / kStride;
constexpr std::size_t kOutputWidth = kInputWidth / kStride;
constexpr std::size_t kInputElements =
    kBatch * kChannels * kInputHeight * kInputWidth;
constexpr std::size_t kOutputElements =
    kBatch * kChannels * kOutputHeight * kOutputWidth;
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
  std::vector<float> source(kInputElements);
  std::vector<float> expected(kOutputElements);
  std::vector<float> output(kOutputElements);
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] =
        static_cast<float>(static_cast<int>((index * 811U) % 65521U) - 32760) /
        1024.0F;

  for (std::size_t channel = 0; channel < kChannels; ++channel)
    for (std::size_t outputY = 0; outputY < kOutputHeight; ++outputY)
      for (std::size_t outputX = 0; outputX < kOutputWidth; ++outputX) {
        float value = -__builtin_inff();
        for (std::size_t kernelY = 0; kernelY < kKernel; ++kernelY)
          for (std::size_t kernelX = 0; kernelX < kKernel; ++kernelX) {
            const std::size_t sourceY = outputY * kStride + kernelY;
            const std::size_t sourceX = outputX * kStride + kernelX;
            value = std::max(
                value,
                source[(channel * kInputHeight + sourceY) * kInputWidth +
                       sourceX]);
          }
        expected[(channel * kOutputHeight + outputY) * kOutputWidth +
                 outputX] = value;
      }

  max_pool2d_f32(source.data(), output.data(), kBatch, kChannels,
                 kInputHeight, kInputWidth, kOutputHeight, kOutputWidth,
                 kKernel, kKernel, kStride, kStride);
  if (output != expected) {
    std::fprintf(stderr, "max_pool2d_f32 result mismatch\n");
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    max_pool2d_f32(source.data(), output.data(), kBatch, kChannels,
                   kInputHeight, kInputWidth, kOutputHeight, kOutputWidth,
                   kKernel, kKernel, kStride, kStride);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=max_pool2d_f32\n");
  std::printf("model_shape=vision_pool[N=1,C=64,H=112,W=112,K=2,S=2]\n");
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gbytes_s=%.6f\n",
              static_cast<double>((kInputElements + kOutputElements) *
                                  sizeof(float)) /
                  milliseconds / 1.0e6);
  return 0;
}
