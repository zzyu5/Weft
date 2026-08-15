#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef WEFT_FP4_DEQUANT_KIND
#error "WEFT_FP4_DEQUANT_KIND is required"
#endif

#ifndef WEFT_FP4_DEQUANT_ENTRY
#error "WEFT_FP4_DEQUANT_ENTRY is required"
#endif

#define WEFT_STRINGIFY_INNER(value) #value
#define WEFT_STRINGIFY(value) WEFT_STRINGIFY_INNER(value)

namespace {

constexpr std::size_t kRows = 1024;
constexpr std::size_t kColumns = 4096;
#if WEFT_FP4_DEQUANT_KIND == 1
constexpr std::size_t kBlockSize = 64;
constexpr std::size_t kBlockBytes = 36;
#else
constexpr std::size_t kBlockSize = 32;
constexpr std::size_t kBlockBytes = 17;
#endif
constexpr std::size_t kBlocksPerRow = kColumns / kBlockSize;
constexpr std::size_t kInputStride = kBlocksPerRow * kBlockBytes;
constexpr std::size_t kElements = kRows * kColumns;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
constexpr std::int8_t kCodebook[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};
volatile std::uint64_t evictionSink = 0;

[[maybe_unused]] float e8m0Half(std::uint8_t exponent) {
  const std::uint32_t bits = exponent < 2
                                 ? UINT32_C(0x00200000) << exponent
                                 : static_cast<std::uint32_t>(exponent - 1) << 23;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

[[maybe_unused]] float ue4m3Half(std::uint8_t encoded) {
  if (encoded == 0 || encoded == 127)
    return 0.0F;
  const std::uint8_t exponent = static_cast<std::uint8_t>((encoded >> 3) & 15U);
  const std::uint8_t mantissa = static_cast<std::uint8_t>(encoded & 7U);
  if (exponent == 0)
    return static_cast<float>(mantissa) * 0.0009765625F;
  const std::uint32_t bits =
      (static_cast<std::uint32_t>(exponent) + 119U) << 23 |
      static_cast<std::uint32_t>(mantissa) << 20;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
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
  std::uint8_t codebookBytes[16];
  std::memcpy(codebookBytes, kCodebook, sizeof(codebookBytes));

  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
      const std::size_t inputBase = row * kInputStride + block * kBlockBytes;
      const std::size_t outputBase = row * kColumns + block * kBlockSize;
#if WEFT_FP4_DEQUANT_KIND == 1
      for (std::size_t subblock = 0; subblock < 4; ++subblock) {
        const std::uint8_t scaleBits = static_cast<std::uint8_t>(
            ((1 + ((row + block + subblock) % 14)) << 3) |
            ((row * 3 + block * 5 + subblock * 7) & 7U));
        packed[inputBase + subblock] = scaleBits;
        const float scale = ue4m3Half(scaleBits);
        for (std::size_t member = 0; member < 8; ++member) {
          const std::uint8_t low = static_cast<std::uint8_t>(
              (row * 3 + block * 5 + subblock * 7 + member * 11) & 15U);
          const std::uint8_t high = static_cast<std::uint8_t>(
              (row * 13 + block * 3 + subblock * 5 + member * 7 + 1) & 15U);
          packed[inputBase + 4 + subblock * 8 + member] =
              static_cast<std::uint8_t>(low | (high << 4));
          reference[outputBase + subblock * 16 + member] =
              scale * kCodebook[low];
          reference[outputBase + subblock * 16 + 8 + member] =
              scale * kCodebook[high];
        }
      }
#else
      const std::uint8_t exponent = static_cast<std::uint8_t>(
          118 + ((row * 3 + block * 5) % 20));
      packed[inputBase] = exponent;
      const float scale = e8m0Half(exponent);
      for (std::size_t member = 0; member < 16; ++member) {
        const std::uint8_t low = static_cast<std::uint8_t>(
            (row * 3 + block * 5 + member * 7) & 15U);
        const std::uint8_t high = static_cast<std::uint8_t>(
            (row * 11 + block * 3 + member * 13 + 1) & 15U);
        packed[inputBase + 1 + member] =
            static_cast<std::uint8_t>(low | (high << 4));
        reference[outputBase + member] = scale * kCodebook[low];
        reference[outputBase + 16 + member] = scale * kCodebook[high];
      }
#endif
    }
  }

  auto invoke = [&]() {
    WEFT_FP4_DEQUANT_ENTRY(packed.data(), codebookBytes, output.data(), kRows,
                           kBlocksPerRow, kInputStride, kColumns);
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
                 WEFT_STRINGIFY(WEFT_FP4_DEQUANT_ENTRY), maxAbsoluteError);
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
  std::printf("kernel=%s\n", WEFT_STRINGIFY(WEFT_FP4_DEQUANT_ENTRY));
  std::printf("model_shape=N=1024;K=4096\n");
  std::printf("elements=%zu\n", kElements);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
