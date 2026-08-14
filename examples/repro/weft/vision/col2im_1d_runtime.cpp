#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kInputSteps = 1024;
constexpr std::size_t kKernel = 5;
constexpr std::size_t kOutputChannels = 256;
constexpr std::size_t kStride = 2;
constexpr std::size_t kPadding = 2;
constexpr std::size_t kOutputSteps =
    (kInputSteps - 1U) * kStride + kKernel - 2U * kPadding;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink;

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
  const std::size_t kernelChannels = kKernel * kOutputChannels;
  std::vector<float> columns(kernelChannels * kInputSteps);
  std::vector<float> primary(kOutputChannels * kOutputSteps);
  std::vector<float> equivalent(kOutputChannels * kOutputSteps);
  std::vector<float> expected(kOutputChannels * kOutputSteps);
  for (std::size_t index = 0; index < columns.size(); ++index)
    columns[index] =
        static_cast<float>(static_cast<int>((index * 17U) % 127U) - 63) /
        64.0F;
  for (std::size_t channel = 0; channel < kOutputChannels; ++channel)
    for (std::size_t outputStep = 0; outputStep < kOutputSteps; ++outputStep) {
      const std::ptrdiff_t absolute =
          static_cast<std::ptrdiff_t>(outputStep + kPadding);
      float accumulator = 0.0F;
      for (std::size_t inputStep = 0; inputStep < kInputSteps; ++inputStep) {
        const std::ptrdiff_t kernelStep =
            absolute - static_cast<std::ptrdiff_t>(inputStep * kStride);
        if (kernelStep < 0 ||
            kernelStep >= static_cast<std::ptrdiff_t>(kKernel))
          continue;
        accumulator += columns[channel * kKernel + kernelStep +
                               inputStep * kernelChannels];
      }
      expected[channel * kOutputSteps + outputStep] = accumulator;
    }

  col2im_1d_f32(columns.data(), primary.data(), kInputSteps, kKernel,
                kOutputChannels, kStride, kPadding);
  col2im_1d_f32_equivalent(columns.data(), equivalent.data(), kInputSteps,
                           kKernel, kOutputChannels, kStride, kPadding);
  double maxAbsoluteError = 0.0;
  double equivalentMaxAbsoluteError = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    maxAbsoluteError = std::fmax(
        maxAbsoluteError,
        static_cast<double>(std::fabs(primary[index] - expected[index])));
    equivalentMaxAbsoluteError = std::fmax(
        equivalentMaxAbsoluteError,
        static_cast<double>(std::fabs(equivalent[index] - expected[index])));
  }
  if (maxAbsoluteError > 1.0e-6 || equivalentMaxAbsoluteError > 1.0e-6) {
    std::fprintf(stderr, "col2im_1d_f32 mismatch\n");
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> primarySamples;
  std::vector<double> equivalentSamples;
  for (int repetition = 0; repetition < 7; ++repetition) {
    evict(eviction);
    auto begin = std::chrono::steady_clock::now();
    col2im_1d_f32(columns.data(), primary.data(), kInputSteps, kKernel,
                  kOutputChannels, kStride, kPadding);
    auto end = std::chrono::steady_clock::now();
    primarySamples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
    evict(eviction);
    begin = std::chrono::steady_clock::now();
    col2im_1d_f32_equivalent(columns.data(), equivalent.data(), kInputSteps,
                             kKernel, kOutputChannels, kStride, kPadding);
    end = std::chrono::steady_clock::now();
    equivalentSamples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double primaryMilliseconds = median(primarySamples);
  const double equivalentMilliseconds = median(equivalentSamples);
  std::printf("kernel=col2im_1d_f32\n");
  std::printf(
      "model_shape=audio_col2im[Tin=1024,K=5,OC=256,stride=2,pad=2]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("equivalent_max_absolute_error=%.9g\n",
              equivalentMaxAbsoluteError);
  std::printf("primary_median_ms=%.6f\n", primaryMilliseconds);
  std::printf("equivalent_median_ms=%.6f\n", equivalentMilliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(expected.size()) / primaryMilliseconds /
                  1000.0);
  return 0;
}
