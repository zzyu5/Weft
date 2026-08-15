#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kBatch = 1;
constexpr std::size_t kInputDepth = 8;
constexpr std::size_t kInputHeight = 32;
constexpr std::size_t kInputWidth = 32;
constexpr std::size_t kInputChannels = 32;
constexpr std::size_t kOutputChannels = 64;
constexpr std::size_t kKernelDepth = 3;
constexpr std::size_t kKernelHeight = 3;
constexpr std::size_t kKernelWidth = 3;
constexpr std::size_t kOutputDepth = 8;
constexpr std::size_t kOutputHeight = 32;
constexpr std::size_t kOutputWidth = 32;
constexpr std::size_t kPadding = 1;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink;

std::size_t inputIndex(std::size_t image, std::size_t channel,
                       std::size_t z, std::size_t y, std::size_t x) {
  return x + kInputWidth *
                 (y + kInputHeight *
                          (z + kInputDepth *
                                   (channel + kInputChannels * image)));
}

std::size_t weightIndex(std::size_t outputChannel,
                        std::size_t inputChannel, std::size_t z,
                        std::size_t y, std::size_t x) {
  return x + kKernelWidth *
                 (y + kKernelHeight *
                          (z + kKernelDepth *
                                   (inputChannel +
                                    kInputChannels * outputChannel)));
}

std::size_t outputIndex(std::size_t image, std::size_t outputChannel,
                        std::size_t z, std::size_t y, std::size_t x) {
  return x + kOutputWidth *
                 (y + kOutputHeight *
                          (z + kOutputDepth *
                                   (outputChannel +
                                    kOutputChannels * image)));
}

float reference(const std::vector<float> &source,
                const std::vector<float> &weight, std::size_t image,
                std::size_t outputChannel, std::size_t outputZ,
                std::size_t outputY, std::size_t outputX) {
  float result = 0.0F;
  for (std::size_t inputChannel = 0; inputChannel < kInputChannels;
       ++inputChannel)
    for (std::size_t kernelZ = 0; kernelZ < kKernelDepth; ++kernelZ)
      for (std::size_t kernelY = 0; kernelY < kKernelHeight; ++kernelY)
        for (std::size_t kernelX = 0; kernelX < kKernelWidth; ++kernelX) {
          const std::ptrdiff_t inputZ =
              static_cast<std::ptrdiff_t>(outputZ + kernelZ) - kPadding;
          const std::ptrdiff_t inputY =
              static_cast<std::ptrdiff_t>(outputY + kernelY) - kPadding;
          const std::ptrdiff_t inputX =
              static_cast<std::ptrdiff_t>(outputX + kernelX) - kPadding;
          if (inputZ < 0 || inputY < 0 || inputX < 0 ||
              inputZ >= static_cast<std::ptrdiff_t>(kInputDepth) ||
              inputY >= static_cast<std::ptrdiff_t>(kInputHeight) ||
              inputX >= static_cast<std::ptrdiff_t>(kInputWidth))
            continue;
          result += source[inputIndex(image, inputChannel, inputZ, inputY,
                                      inputX)] *
                    weight[weightIndex(outputChannel, inputChannel, kernelZ,
                                       kernelY, kernelX)];
        }
  return result;
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
  const std::size_t inputElements = kBatch * kInputChannels * kInputDepth *
                                    kInputHeight * kInputWidth;
  const std::size_t weightElements =
      kOutputChannels * kInputChannels * kKernelDepth * kKernelHeight *
      kKernelWidth;
  const std::size_t outputElements = kBatch * kOutputChannels * kOutputDepth *
                                     kOutputHeight * kOutputWidth;
  std::vector<float> source(inputElements);
  std::vector<float> weight(weightElements);
  std::vector<float> output(outputElements);
  std::vector<float> packedPatches(
      conv3d_f32__packed_patches_elements(
          kBatch, kInputChannels, kKernelDepth, kKernelHeight, kKernelWidth,
          kOutputDepth, kOutputHeight, kOutputWidth));
  std::vector<float> packedWeight(conv3d_f32__packed_weight_elements(
      kInputChannels, kOutputChannels, kKernelDepth, kKernelHeight,
      kKernelWidth));
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] =
        static_cast<float>(static_cast<int>((index * 7U) % 127U) - 63) /
        256.0F;
  for (std::size_t index = 0; index < weight.size(); ++index)
    weight[index] =
        static_cast<float>(static_cast<int>((index * 11U) % 131U) - 65) /
        512.0F;

  conv3d_f32(
      source.data(), weight.data(), output.data(), packedPatches.data(),
      packedWeight.data(), kBatch, kInputDepth, kInputHeight, kInputWidth,
      kInputChannels, kOutputChannels, kKernelDepth, kKernelHeight,
      kKernelWidth, kOutputDepth, kOutputHeight, kOutputWidth, 1, 1, 1,
      kPadding, kPadding, kPadding, 1, 1, 1);
  const std::size_t sampleChannels[] = {0, kOutputChannels / 2,
                                        kOutputChannels - 1};
  const std::size_t sampleZ[] = {0, kOutputDepth / 2, kOutputDepth - 1};
  const std::size_t sampleY[] = {0, kOutputHeight / 2, kOutputHeight - 1};
  const std::size_t sampleX[] = {0, 1, kOutputWidth / 2,
                                 kOutputWidth - 1};
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t channel : sampleChannels)
    for (std::size_t z : sampleZ)
      for (std::size_t y : sampleY)
        for (std::size_t x : sampleX) {
          const double expected = reference(source, weight, 0, channel, z, y, x);
          const double actual = output[outputIndex(0, channel, z, y, x)];
          const double absolute = std::fabs(actual - expected);
          const double relative = absolute / std::fmax(1.0, std::fabs(expected));
          maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
          maxRelativeError = std::fmax(maxRelativeError, relative);
        }
  if (maxRelativeError > 5.0e-4) {
    std::fprintf(stderr, "conv3d_f32 mismatch: abs=%g rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  for (int repetition = 0; repetition < 3; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    conv3d_f32(
        source.data(), weight.data(), output.data(), packedPatches.data(),
        packedWeight.data(), kBatch, kInputDepth, kInputHeight, kInputWidth,
        kInputChannels, kOutputChannels, kKernelDepth, kKernelHeight,
        kKernelWidth, kOutputDepth, kOutputHeight, kOutputWidth, 1, 1, 1,
        kPadding, kPadding, kPadding, 1, 1, 1);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double operations =
      2.0 * static_cast<double>(packedPatches.size() * kOutputChannels);
  std::printf("kernel=conv3d_f32\n");
  std::printf(
      "model_shape=video_conv3d[N=1,D=8,H=32,W=32,IC=32,OC=64,K=3x3x3,pad=1]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / milliseconds / 1.0e6);
  return 0;
}
