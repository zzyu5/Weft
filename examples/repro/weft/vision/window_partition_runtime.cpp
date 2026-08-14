#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kHeight = 64;
constexpr std::size_t kWidth = 64;
constexpr std::size_t kChannels = 768;
constexpr std::size_t kWindow = 14;
constexpr std::size_t kWindowsY = (kHeight + kWindow - 1U) / kWindow;
constexpr std::size_t kWindowsX = (kWidth + kWindow - 1U) / kWindow;
constexpr std::size_t kInputElements = kHeight * kWidth * kChannels;
constexpr std::size_t kOutputElements =
    kWindowsY * kWindowsX * kWindow * kWindow * kChannels;
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
  return samples[samples.size() / 2U];
}

} // namespace

int main() {
  std::vector<float> source(kInputElements);
  std::vector<float> expected(kOutputElements, 0.0F);
  std::vector<float> output(kOutputElements);
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] =
        static_cast<float>(static_cast<int>(index % 1021U) - 510) / 512.0F;
  for (std::size_t windowY = 0; windowY < kWindowsY; ++windowY)
    for (std::size_t windowX = 0; windowX < kWindowsX; ++windowX)
      for (std::size_t localY = 0; localY < kWindow; ++localY)
        for (std::size_t localX = 0; localX < kWindow; ++localX) {
          const std::size_t sourceY = windowY * kWindow + localY;
          const std::size_t sourceX = windowX * kWindow + localX;
          if (sourceY >= kHeight || sourceX >= kWidth)
            continue;
          const std::size_t windowIndex = windowY * kWindowsX + windowX;
          std::copy_n(
              source.data() + (sourceY * kWidth + sourceX) * kChannels,
              kChannels,
              expected.data() +
                  ((windowIndex * kWindow + localY) * kWindow + localX) *
                      kChannels);
        }

  window_partition_f32(source.data(), output.data(), kHeight, kWidth, kChannels,
                       kWindow, kWindowsY, kWindowsX);
  if (output != expected) {
    std::fprintf(stderr, "window_partition_f32 result mismatch\n");
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    window_partition_f32(source.data(), output.data(), kHeight, kWidth,
                         kChannels, kWindow, kWindowsY, kWindowsX);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=window_partition_f32\n");
  std::printf("model_shape=SAM[C=768,H=64,W=64,window=14]\n");
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gbytes_s=%.6f\n",
              static_cast<double>((kInputElements + kOutputElements) *
                                  sizeof(float)) /
                  milliseconds / 1.0e6);
  return 0;
}
