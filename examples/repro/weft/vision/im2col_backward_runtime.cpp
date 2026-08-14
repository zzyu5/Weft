#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kBatch = 1;
constexpr std::size_t kInputHeight = 64;
constexpr std::size_t kInputWidth = 64;
constexpr std::size_t kInputChannels = 256;
constexpr std::size_t kKernelHeight = 3;
constexpr std::size_t kKernelWidth = 3;
constexpr std::size_t kOutputHeight = 64;
constexpr std::size_t kOutputWidth = 64;
constexpr std::size_t kPadding = 1;
constexpr std::size_t kKernelPlane = kKernelHeight * kKernelWidth;
constexpr std::size_t kColumnStride = kInputChannels * kKernelPlane;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

std::size_t gradientIndex(std::size_t image, std::size_t outputY,
                          std::size_t outputX, std::size_t inputChannel,
                          std::size_t kernelY, std::size_t kernelX) {
  const std::size_t window = kernelY * kKernelWidth + kernelX;
  return inputChannel * kKernelPlane + window +
         kColumnStride *
             (outputX +
              kOutputWidth * (outputY + kOutputHeight * image));
}

std::size_t outputIndex(std::size_t image, std::size_t inputChannel,
                        std::size_t inputY, std::size_t inputX) {
  return inputX +
         kInputWidth *
             (inputY + kInputHeight *
                           (inputChannel + kInputChannels * image));
}

float reference(const std::vector<float> &gradient, std::size_t image,
                std::size_t inputChannel, std::size_t inputY,
                std::size_t inputX) {
  float value = 0.0F;
  for (std::size_t kernelY = 0; kernelY < kKernelHeight; ++kernelY)
    for (std::size_t kernelX = 0; kernelX < kKernelWidth; ++kernelX) {
      const std::size_t offsetY = kernelY;
      const std::size_t offsetX = kernelX;
      if (inputY + kPadding < offsetY || inputX + kPadding < offsetX)
        continue;
      const std::size_t outputY = inputY + kPadding - offsetY;
      const std::size_t outputX = inputX + kPadding - offsetX;
      if (outputY >= kOutputHeight || outputX >= kOutputWidth)
        continue;
      value += gradient[gradientIndex(image, outputY, outputX, inputChannel,
                                     kernelY, kernelX)];
    }
  return value;
}

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
  const std::size_t gradientElements =
      kBatch * kOutputHeight * kOutputWidth * kColumnStride;
  const std::size_t outputElements =
      kBatch * kInputHeight * kInputWidth * kInputChannels;
  std::vector<float> gradient(gradientElements);
  std::vector<float> expected(outputElements);
  std::vector<float> output(outputElements);
  for (std::size_t index = 0; index < gradient.size(); ++index)
    gradient[index] =
        static_cast<float>(static_cast<int>((index * 11U) % 251U) - 125) /
        256.0F;
  for (std::size_t image = 0; image < kBatch; ++image)
    for (std::size_t channel = 0; channel < kInputChannels; ++channel)
      for (std::size_t y = 0; y < kInputHeight; ++y)
        for (std::size_t x = 0; x < kInputWidth; ++x)
          expected[outputIndex(image, channel, y, x)] =
              reference(gradient, image, channel, y, x);

  im2col_backward_stride1_f32(
      gradient.data(), output.data(), kBatch, kInputHeight, kInputWidth,
      kInputChannels, kKernelHeight, kKernelWidth, kOutputHeight, kOutputWidth,
      kPadding, kPadding, 1, 1);
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t index = 0; index < outputElements; ++index) {
    const double absolute = std::fabs(output[index] - expected[index]);
    const double relative = absolute / std::fmax(1.0, std::fabs(expected[index]));
    maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
    maxRelativeError = std::fmax(maxRelativeError, relative);
  }
  if (maxRelativeError > 1.0e-6) {
    std::fprintf(stderr,
                 "im2col_backward_stride1_f32 mismatch: abs=%g rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(5);
  for (int repetition = 0; repetition < 5; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    im2col_backward_stride1_f32(
        gradient.data(), output.data(), kBatch, kInputHeight, kInputWidth,
        kInputChannels, kKernelHeight, kKernelWidth, kOutputHeight,
        kOutputWidth, kPadding, kPadding, 1, 1);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=im2col_backward_stride1_f32\n");
  std::printf(
      "model_shape=SAM_im2col_backward[N=1,H=64,W=64,IC=256,K=3x3,pad=1,stride=1]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("moutput_s=%.6f\n",
              static_cast<double>(outputElements) / milliseconds / 1.0e3);
  return 0;
}
