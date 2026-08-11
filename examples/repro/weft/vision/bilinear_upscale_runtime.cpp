#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void bilinear_upscale_f32(
    const float *source, float *output, std::size_t batch,
    std::size_t input_height, std::size_t input_width, std::size_t channels,
    std::size_t output_height, std::size_t output_width);

namespace {

constexpr std::size_t kBatch = 1;
constexpr std::size_t kInputHeight = 128;
constexpr std::size_t kInputWidth = 128;
constexpr std::size_t kChannels = 64;
constexpr std::size_t kOutputHeight = 256;
constexpr std::size_t kOutputWidth = 256;
constexpr std::size_t kInputElements =
    kBatch * kInputHeight * kInputWidth * kChannels;
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

std::size_t clampIndex(float value, std::size_t extent) {
  const float clamped = std::fmin(
      std::fmax(value, 0.0F), static_cast<float>(extent - 1U));
  return static_cast<std::size_t>(clamped);
}

} // namespace

int main() {
  std::vector<float> source(kInputElements);
  std::vector<float> expected(kOutputElements);
  std::vector<float> output(kOutputElements);
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] =
        static_cast<float>(static_cast<int>(index % 1021U) - 510) / 512.0F;

  const float scaleY =
      static_cast<float>(kOutputHeight) / static_cast<float>(kInputHeight);
  const float scaleX =
      static_cast<float>(kOutputWidth) / static_cast<float>(kInputWidth);
  for (std::size_t outputY = 0; outputY < kOutputHeight; ++outputY) {
    const float sourceY =
        (static_cast<float>(outputY) + 0.5F) / scaleY - 0.5F;
    const float floorY = std::floor(sourceY);
    const std::size_t y0 = clampIndex(floorY, kInputHeight);
    const std::size_t y1 = clampIndex(floorY + 1.0F, kInputHeight);
    const float dy =
        std::fmin(std::fmax(sourceY - static_cast<float>(y0), 0.0F), 1.0F);
    for (std::size_t outputX = 0; outputX < kOutputWidth; ++outputX) {
      const float sourceX =
          (static_cast<float>(outputX) + 0.5F) / scaleX - 0.5F;
      const float floorX = std::floor(sourceX);
      const std::size_t x0 = clampIndex(floorX, kInputWidth);
      const std::size_t x1 = clampIndex(floorX + 1.0F, kInputWidth);
      const float dx = std::fmin(
          std::fmax(sourceX - static_cast<float>(x0), 0.0F), 1.0F);
      for (std::size_t channel = 0; channel < kChannels; ++channel) {
        const float a =
            source[(y0 * kInputWidth + x0) * kChannels + channel];
        const float b =
            source[(y0 * kInputWidth + x1) * kChannels + channel];
        const float c =
            source[(y1 * kInputWidth + x0) * kChannels + channel];
        const float d =
            source[(y1 * kInputWidth + x1) * kChannels + channel];
        expected[(outputY * kOutputWidth + outputX) * kChannels + channel] =
            a * (1.0F - dx) * (1.0F - dy) +
            b * dx * (1.0F - dy) + c * (1.0F - dx) * dy + d * dx * dy;
      }
    }
  }

  bilinear_upscale_f32(source.data(), output.data(), kBatch, kInputHeight,
                       kInputWidth, kChannels, kOutputHeight, kOutputWidth);
  double maxAbsoluteError = 0.0;
  for (std::size_t index = 0; index < output.size(); ++index)
    maxAbsoluteError = std::fmax(
        maxAbsoluteError, std::fabs(static_cast<double>(output[index]) -
                                    static_cast<double>(expected[index])));
  if (maxAbsoluteError > 1.0e-6) {
    std::fprintf(stderr, "bilinear_upscale_f32 mismatch: max_abs=%g\n",
                 maxAbsoluteError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(5);
  for (int repetition = 0; repetition < 5; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    bilinear_upscale_f32(source.data(), output.data(), kBatch, kInputHeight,
                         kInputWidth, kChannels, kOutputHeight, kOutputWidth);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=bilinear_upscale_f32\n");
  std::printf("model_shape=vision_upscale[N=1,H=128,W=128,C=64,to=256x256]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gbytes_s=%.6f\n",
              static_cast<double>((4U * kOutputElements + kOutputElements) *
                                  sizeof(float)) /
                  milliseconds / 1.0e6);
  return 0;
}
