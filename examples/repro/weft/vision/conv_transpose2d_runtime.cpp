#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kBatch = 1;
constexpr std::size_t kInputHeight = 16;
constexpr std::size_t kInputWidth = 16;
constexpr std::size_t kInputChannels = 256;
constexpr std::size_t kOutputChannels = 128;
constexpr std::size_t kKernelHeight = 2;
constexpr std::size_t kKernelWidth = 2;
constexpr std::size_t kStride = 2;
constexpr std::size_t kOutputHeight =
    (kInputHeight - 1) * kStride + kKernelHeight;
constexpr std::size_t kOutputWidth =
    (kInputWidth - 1) * kStride + kKernelWidth;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

std::size_t inputIndex(std::size_t image, std::size_t channel,
                       std::size_t y, std::size_t x) {
  return x + kInputWidth *
                 (y + kInputHeight * (channel + kInputChannels * image));
}

std::size_t weightIndex(std::size_t inputChannel,
                        std::size_t outputChannel, std::size_t y,
                        std::size_t x) {
  return x + kKernelWidth *
                 (y + kKernelHeight *
                          (outputChannel + kOutputChannels * inputChannel));
}

std::size_t outputIndex(std::size_t image, std::size_t outputChannel,
                        std::size_t y, std::size_t x) {
  return x + kOutputWidth *
                 (y + kOutputHeight *
                          (outputChannel + kOutputChannels * image));
}

float reference(const std::vector<float> &source,
                const std::vector<float> &weight, std::size_t image,
                std::size_t outputChannel, std::size_t outputY,
                std::size_t outputX) {
  float value = 0.0F;
  for (std::size_t inputChannel = 0; inputChannel < kInputChannels;
       ++inputChannel)
    for (std::size_t kernelY = 0; kernelY < kKernelHeight; ++kernelY)
      for (std::size_t kernelX = 0; kernelX < kKernelWidth; ++kernelX) {
        if (outputY < kernelY || outputX < kernelX)
          continue;
        const std::size_t relativeY = outputY - kernelY;
        const std::size_t relativeX = outputX - kernelX;
        if (relativeY % kStride != 0 || relativeX % kStride != 0)
          continue;
        const std::size_t inputY = relativeY / kStride;
        const std::size_t inputX = relativeX / kStride;
        if (inputY >= kInputHeight || inputX >= kInputWidth)
          continue;
        value += source[inputIndex(image, inputChannel, inputY, inputX)] *
                 weight[weightIndex(inputChannel, outputChannel, kernelY,
                                    kernelX)];
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
  const std::size_t inputElements =
      kBatch * kInputChannels * kInputHeight * kInputWidth;
  const std::size_t weightElements = kInputChannels * kOutputChannels *
                                     kKernelHeight * kKernelWidth;
  const std::size_t outputElements =
      kBatch * kOutputChannels * kOutputHeight * kOutputWidth;
  std::vector<float> source(inputElements);
  std::vector<float> weight(weightElements);
  std::vector<float> output(outputElements);
  std::vector<float> packedSource(
      conv_transpose2d_p0_f32__packed_source_elements(
          kBatch, kInputHeight, kInputWidth, kInputChannels));
  std::vector<float> packedWeight(
      conv_transpose2d_p0_f32__packed_weight_elements(
          kInputChannels, kOutputChannels, kKernelHeight, kKernelWidth));
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] =
        static_cast<float>(static_cast<int>(index % 127U) - 63) / 256.0F;
  for (std::size_t index = 0; index < weight.size(); ++index)
    weight[index] =
        static_cast<float>(static_cast<int>((index * 7U) % 131U) - 65) /
        512.0F;

  conv_transpose2d_p0_f32(
      source.data(), weight.data(), output.data(), packedSource.data(),
      packedWeight.data(), kBatch, kInputHeight, kInputWidth, kInputChannels,
      kOutputChannels, kKernelHeight, kKernelWidth, kStride);
  const std::size_t sampleChannels[] = {0, 63, 127};
  const std::size_t sampleY[] = {0, 1, 15, 31};
  const std::size_t sampleX[] = {0, 1, 16, 31};
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t channel : sampleChannels)
    for (std::size_t y : sampleY)
      for (std::size_t x : sampleX) {
        const double expected = reference(source, weight, 0, channel, y, x);
        const double actual = output[outputIndex(0, channel, y, x)];
        const double absolute = std::fabs(actual - expected);
        const double relative = absolute / std::fmax(1.0, std::fabs(expected));
        maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
        maxRelativeError = std::fmax(maxRelativeError, relative);
      }
  if (maxRelativeError > 5.0e-4) {
    std::fprintf(stderr, "conv_transpose2d_p0_f32 mismatch: abs=%g rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(5);
  for (int repetition = 0; repetition < 5; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    conv_transpose2d_p0_f32(
        source.data(), weight.data(), output.data(), packedSource.data(),
        packedWeight.data(), kBatch, kInputHeight, kInputWidth, kInputChannels,
        kOutputChannels, kKernelHeight, kKernelWidth, kStride);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double operations =
      2.0 * static_cast<double>(kBatch * kInputHeight * kInputWidth *
                                kInputChannels * kOutputChannels *
                                kKernelHeight * kKernelWidth);
  std::printf("kernel=conv_transpose2d_p0_f32\n");
  std::printf(
      "model_shape=decoder_upsample[N=1,H=16,W=16,IC=256,OC=128,K=2x2,stride=2]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / milliseconds / 1.0e6);
  return 0;
}
