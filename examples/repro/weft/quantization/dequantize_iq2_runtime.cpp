#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef WEFT_IQ2_DEQUANT_KIND
#error "WEFT_IQ2_DEQUANT_KIND is required"
#endif

#ifndef WEFT_IQ2_DEQUANT_ENTRY
#error "WEFT_IQ2_DEQUANT_ENTRY is required"
#endif

#define WEFT_STRINGIFY_INNER(value) #value
#define WEFT_STRINGIFY(value) WEFT_STRINGIFY_INNER(value)

namespace {

constexpr std::size_t kRows = 1024;
constexpr std::size_t kColumns = 4096;
constexpr std::size_t kBlockSize = 256;
#if WEFT_IQ2_DEQUANT_KIND == 0
constexpr std::size_t kBlockBytes = 66;
constexpr std::size_t kGridRows = 256;
#elif WEFT_IQ2_DEQUANT_KIND == 1
constexpr std::size_t kBlockBytes = 74;
constexpr std::size_t kGridRows = 512;
#elif WEFT_IQ2_DEQUANT_KIND == 2
constexpr std::size_t kBlockBytes = 82;
constexpr std::size_t kGridRows = 1024;
#else
#error "unsupported WEFT_IQ2_DEQUANT_KIND"
#endif
constexpr std::size_t kBlocksPerRow = kColumns / kBlockSize;
constexpr std::size_t kInputStride = kBlocksPerRow * kBlockBytes;
constexpr std::size_t kElements = kRows * kColumns;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

float storeF16(std::vector<std::uint8_t> &bytes, std::size_t offset,
               float value) {
  const _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  bytes[offset] = static_cast<std::uint8_t>(bits);
  bytes[offset + 1] = static_cast<std::uint8_t>(bits >> 8);
  return static_cast<float>(half);
}

[[maybe_unused]] void storeU16(std::vector<std::uint8_t> &bytes,
                               std::size_t offset, std::uint16_t value) {
  bytes[offset] = static_cast<std::uint8_t>(value);
  bytes[offset + 1] = static_cast<std::uint8_t>(value >> 8);
}

[[maybe_unused]] void storeU32(std::vector<std::uint8_t> &bytes,
                               std::size_t offset, std::uint32_t value) {
  bytes[offset] = static_cast<std::uint8_t>(value);
  bytes[offset + 1] = static_cast<std::uint8_t>(value >> 8);
  bytes[offset + 2] = static_cast<std::uint8_t>(value >> 16);
  bytes[offset + 3] = static_cast<std::uint8_t>(value >> 24);
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
  std::vector<std::uint8_t> grid(kGridRows * 8);
  std::vector<std::uint8_t> signTable(128);
  std::vector<float> output(kElements);
  std::vector<float> reference(kElements);

  for (std::size_t index = 0; index < kGridRows; ++index) {
    for (std::size_t lane = 0; lane < 8; ++lane) {
      grid[index * 8 + lane] = static_cast<std::uint8_t>(
          1U + 2U * ((index * 5U + lane * 3U) % 15U));
    }
  }
  for (std::size_t index = 0; index < signTable.size(); ++index)
    signTable[index] = static_cast<std::uint8_t>(index * 37U + 13U);

  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
      const std::size_t inputBase = row * kInputStride + block * kBlockBytes;
      const std::size_t outputBase = row * kColumns + block * kBlockSize;
      const float d = storeF16(
          packed, inputBase,
          static_cast<float>(1U + ((row * 17U + block * 13U) % 31U)) /
              128.0F);
      for (std::size_t group = 0; group < 8; ++group) {
#if WEFT_IQ2_DEQUANT_KIND == 0
        const std::uint8_t localScale = static_cast<std::uint8_t>(
            (row * 3U + block * 5U + group * 7U) & 15U);
        std::uint32_t signMetadata =
            static_cast<std::uint32_t>(localScale) << 28;
        for (std::size_t field = 0; field < 4; ++field) {
          const std::size_t gridIndex =
              (row * 11U + block * 7U + group * 17U + field * 29U) & 255U;
          packed[inputBase + 2 + group * 8 + field] =
              static_cast<std::uint8_t>(gridIndex);
          const std::size_t signIndex =
              (row * 13U + block * 3U + group * 19U + field * 23U) & 127U;
          signMetadata |= static_cast<std::uint32_t>(signIndex) << (7 * field);
          const float scale = d * (0.5F + localScale) * 0.25F;
          const std::uint8_t signs = signTable[signIndex];
          for (std::size_t lane = 0; lane < 8; ++lane) {
            const float factor = (signs >> lane) & 1U ? -1.0F : 1.0F;
            reference[outputBase + group * 32 + field * 8 + lane] =
                scale * grid[gridIndex * 8 + lane] * factor;
          }
        }
        storeU32(packed, inputBase + 6 + group * 8, signMetadata);
#elif WEFT_IQ2_DEQUANT_KIND == 1
        const std::uint8_t lowScale = static_cast<std::uint8_t>(
            (row * 3U + block * 5U + group * 7U) & 15U);
        const std::uint8_t highScale = static_cast<std::uint8_t>(
            (row * 11U + block * 13U + group * 17U + 1U) & 15U);
        packed[inputBase + 66 + group] =
            static_cast<std::uint8_t>(lowScale | (highScale << 4));
        for (std::size_t field = 0; field < 4; ++field) {
          const std::size_t gridIndex =
              (row * 11U + block * 7U + group * 17U + field * 29U) & 511U;
          const std::size_t signIndex =
              (row * 13U + block * 3U + group * 19U + field * 23U) & 127U;
          storeU16(packed, inputBase + 2 + (group * 4 + field) * 2,
                   static_cast<std::uint16_t>(gridIndex | (signIndex << 9)));
          const std::uint8_t localScale = field < 2 ? lowScale : highScale;
          const float scale = d * (0.5F + localScale) * 0.25F;
          const std::uint8_t signs = signTable[signIndex];
          for (std::size_t lane = 0; lane < 8; ++lane) {
            const float factor = (signs >> lane) & 1U ? -1.0F : 1.0F;
            reference[outputBase + group * 32 + field * 8 + lane] =
                scale * grid[gridIndex * 8 + lane] * factor;
          }
        }
#else
        const std::uint8_t lowScale = static_cast<std::uint8_t>(
            (row * 3U + block * 5U + group * 7U) & 15U);
        const std::uint8_t highScale = static_cast<std::uint8_t>(
            (row * 11U + block * 13U + group * 17U + 1U) & 15U);
        packed[inputBase + 74 + group] =
            static_cast<std::uint8_t>(lowScale | (highScale << 4));
        std::uint8_t highBits = 0;
        for (std::size_t field = 0; field < 4; ++field) {
          const std::size_t gridIndex =
              (row * 11U + block * 7U + group * 17U + field * 29U) & 1023U;
          highBits |= static_cast<std::uint8_t>((gridIndex >> 8) << (2 * field));
          packed[inputBase + 2 + group * 4 + field] =
              static_cast<std::uint8_t>(gridIndex);
          const std::uint8_t signs = static_cast<std::uint8_t>(
              row * 13U + block * 3U + group * 19U + field * 23U);
          packed[inputBase + 34 + group * 4 + field] = signs;
          const std::uint8_t localScale = field < 2 ? lowScale : highScale;
          const float scale = d * (0.5F + localScale) * 0.25F;
          for (std::size_t lane = 0; lane < 8; ++lane) {
            const float factor = (signs >> lane) & 1U ? -1.0F : 1.0F;
            reference[outputBase + group * 32 + field * 8 + lane] =
                scale * grid[gridIndex * 8 + lane] * factor;
          }
        }
        packed[inputBase + 66 + group] = highBits;
#endif
      }
    }
  }

  auto invoke = [&]() {
#if WEFT_IQ2_DEQUANT_KIND == 2
    WEFT_IQ2_DEQUANT_ENTRY(packed.data(), grid.data(), output.data(), kRows,
                           kBlocksPerRow, kInputStride, kColumns);
#else
    WEFT_IQ2_DEQUANT_ENTRY(packed.data(), grid.data(), signTable.data(),
                           output.data(), kRows, kBlocksPerRow, kInputStride,
                           kColumns);
#endif
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
                 WEFT_STRINGIFY(WEFT_IQ2_DEQUANT_ENTRY), maxAbsoluteError);
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
  std::printf("kernel=%s\n", WEFT_STRINGIFY(WEFT_IQ2_DEQUANT_ENTRY));
  std::printf("model_shape=N=1024;K=4096\n");
  std::printf("elements=%zu\n", kElements);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
