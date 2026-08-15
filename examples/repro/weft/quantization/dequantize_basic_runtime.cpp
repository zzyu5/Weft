#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef WEFT_DEQUANT_KIND
#error "WEFT_DEQUANT_KIND is required"
#endif

#ifndef WEFT_DEQUANT_ENTRY
#error "WEFT_DEQUANT_ENTRY is required"
#endif

#define WEFT_STRINGIFY_INNER(value) #value
#define WEFT_STRINGIFY(value) WEFT_STRINGIFY_INNER(value)

namespace {

constexpr std::size_t kRows = 1024;
constexpr std::size_t kColumns = 4096;
constexpr std::size_t kBlockSize = 32;
#if WEFT_DEQUANT_KIND == 1
constexpr std::size_t kBlockBytes = 20;
#elif WEFT_DEQUANT_KIND == 2
constexpr std::size_t kBlockBytes = 34;
#else
constexpr std::size_t kBlockBytes = 18;
#endif
constexpr std::size_t kBlocksPerRow = kColumns / kBlockSize;
constexpr std::size_t kInputStride = kBlocksPerRow * kBlockBytes;
constexpr std::size_t kElements = kRows * kColumns;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

void storeF16(std::vector<std::uint8_t> &bytes, std::size_t offset,
              float value) {
  const _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  bytes[offset] = static_cast<std::uint8_t>(bits);
  bytes[offset + 1] = static_cast<std::uint8_t>(bits >> 8);
}

void evict(std::vector<std::uint8_t> &buffer) {
  for (std::size_t index = 0; index < buffer.size(); index += 64) {
    buffer[index] = static_cast<std::uint8_t>(buffer[index] + 1U);
    evictionSink += buffer[index];
  }
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  return samples.size() % 2 == 0
             ? 0.5 * (samples[middle - 1] + samples[middle])
             : samples[middle];
}

} // namespace

int main() {
  std::vector<std::uint8_t> packed(kRows * kInputStride);
  std::vector<float> output(kElements);
  std::vector<float> reference(kElements);

  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
      const std::size_t inputBase = row * kInputStride + block * kBlockBytes;
      const std::size_t outputBase = row * kColumns + block * kBlockSize;
      const float scaleValue =
          static_cast<float>(1 + ((row * 17 + block * 13) % 31)) / 64.0F;
      storeF16(packed, inputBase, scaleValue);
      const float scale = scaleValue;
#if WEFT_DEQUANT_KIND == 1
      const float minimumValue =
          -static_cast<float>(1 + ((row * 5 + block * 3) % 7)) / 8.0F;
      storeF16(packed, inputBase + 2, minimumValue);
      for (std::size_t member = 0; member < 16; ++member) {
        const std::uint8_t low = static_cast<std::uint8_t>(
            (row * 3 + block * 5 + member * 7) & 15U);
        const std::uint8_t high = static_cast<std::uint8_t>(
            (row * 11 + block * 3 + member * 13 + 1) & 15U);
        packed[inputBase + 4 + member] =
            static_cast<std::uint8_t>(low | (high << 4));
        reference[outputBase + member] = scale * low + minimumValue;
        reference[outputBase + 16 + member] = scale * high + minimumValue;
      }
#elif WEFT_DEQUANT_KIND == 2
      for (std::size_t member = 0; member < 32; ++member) {
        const std::int8_t code = static_cast<std::int8_t>(
            static_cast<int>((row * 19 + block * 11 + member * 7) % 255) -
            127);
        packed[inputBase + 2 + member] = static_cast<std::uint8_t>(code);
        reference[outputBase + member] = scale * static_cast<float>(code);
      }
#else
      for (std::size_t member = 0; member < 16; ++member) {
        const std::uint8_t low = static_cast<std::uint8_t>(
            (row * 3 + block * 5 + member * 7) & 15U);
        const std::uint8_t high = static_cast<std::uint8_t>(
            (row * 11 + block * 3 + member * 13 + 1) & 15U);
        packed[inputBase + 2 + member] =
            static_cast<std::uint8_t>(low | (high << 4));
        reference[outputBase + member] =
            scale * static_cast<float>(static_cast<int>(low) - 8);
        reference[outputBase + 16 + member] =
            scale * static_cast<float>(static_cast<int>(high) - 8);
      }
#endif
    }
  }

  auto invoke = [&]() {
    WEFT_DEQUANT_ENTRY(packed.data(), output.data(), kRows, kBlocksPerRow,
                       kInputStride, kColumns);
  };
  invoke();

  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t index = 0; index < kElements; ++index) {
    const double absolute = std::fabs(static_cast<double>(output[index]) -
                                      static_cast<double>(reference[index]));
    const double relative =
        absolute / std::max(1.0, std::fabs(static_cast<double>(reference[index])));
    maxAbsoluteError = std::max(maxAbsoluteError, absolute);
    maxRelativeError = std::max(maxRelativeError, relative);
  }
  if (maxAbsoluteError != 0.0) {
    std::fprintf(stderr, "%s mismatch: max_absolute_error=%.9g\n",
                 WEFT_STRINGIFY(WEFT_DEQUANT_ENTRY), maxAbsoluteError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    invoke();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  std::printf("kernel=%s\n", WEFT_STRINGIFY(WEFT_DEQUANT_ENTRY));
  std::printf("model_shape=N=1024;K=4096\n");
  std::printf("elements=%zu\n", kElements);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
