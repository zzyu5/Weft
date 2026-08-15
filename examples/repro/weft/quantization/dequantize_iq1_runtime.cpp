#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef WEFT_IQ1_DEQUANT_KIND
#error "WEFT_IQ1_DEQUANT_KIND is required"
#endif

#ifndef WEFT_IQ1_DEQUANT_ENTRY
#error "WEFT_IQ1_DEQUANT_ENTRY is required"
#endif

#define WEFT_STRINGIFY_INNER(value) #value
#define WEFT_STRINGIFY(value) WEFT_STRINGIFY_INNER(value)

namespace {

constexpr std::size_t kRows = 1024;
constexpr std::size_t kColumns = 4096;
constexpr std::size_t kBlockSize = 256;
#if WEFT_IQ1_DEQUANT_KIND == 0
constexpr std::size_t kBlockBytes = 50;
#elif WEFT_IQ1_DEQUANT_KIND == 1
constexpr std::size_t kBlockBytes = 56;
#else
#error "unsupported WEFT_IQ1_DEQUANT_KIND"
#endif
constexpr std::size_t kGridRows = 2048;
constexpr std::size_t kBlocksPerRow = kColumns / kBlockSize;
constexpr std::size_t kInputStride = kBlocksPerRow * kBlockBytes;
constexpr std::size_t kElements = kRows * kColumns;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

[[maybe_unused]] float storeF16(std::vector<std::uint8_t> &bytes,
                                std::size_t offset, float value) {
  const _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  bytes[offset] = static_cast<std::uint8_t>(bits);
  bytes[offset + 1] = static_cast<std::uint8_t>(bits >> 8);
  return static_cast<float>(half);
}

void storeU16(std::vector<std::uint8_t> &bytes, std::size_t offset,
              std::uint16_t value) {
  bytes[offset] = static_cast<std::uint8_t>(value);
  bytes[offset + 1] = static_cast<std::uint8_t>(value >> 8);
}

[[maybe_unused]] std::uint16_t f16Bits(float value) {
  const _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

[[maybe_unused]] float f16Value(std::uint16_t bits) {
  _Float16 half = 0;
  std::memcpy(&half, &bits, sizeof(bits));
  return static_cast<float>(half);
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
  std::vector<float> output(kElements);
  std::vector<float> reference(kElements);

  for (std::size_t index = 0; index < kGridRows; ++index) {
    for (std::size_t lane = 0; lane < 8; ++lane) {
      const std::int8_t value = static_cast<std::int8_t>(
          static_cast<int>((index * 7U + lane * 5U) % 15U) - 7);
      grid[index * 8 + lane] = static_cast<std::uint8_t>(value);
    }
  }

  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
      const std::size_t inputBase = row * kInputStride + block * kBlockBytes;
      const std::size_t outputBase = row * kColumns + block * kBlockSize;
#if WEFT_IQ1_DEQUANT_KIND == 0
      const float d = storeF16(
          packed, inputBase,
          static_cast<float>(1U + ((row * 17U + block * 13U) % 31U)) /
              128.0F);
      for (std::size_t group = 0; group < 8; ++group) {
        const std::uint16_t localScale = static_cast<std::uint16_t>(
            (row * 3U + block * 5U + group * 7U) & 7U);
        const bool negativeDelta = ((row + block + group) & 1U) != 0;
        std::uint16_t metadata = static_cast<std::uint16_t>(localScale << 12);
        if (negativeDelta)
          metadata |= UINT16_C(0x8000);
        for (std::size_t field = 0; field < 4; ++field) {
          const std::size_t gridIndex =
              (row * 11U + block * 7U + group * 17U + field * 29U) & 2047U;
          packed[inputBase + 2 + group * 4 + field] =
              static_cast<std::uint8_t>(gridIndex);
          metadata |= static_cast<std::uint16_t>((gridIndex >> 8)
                                                 << (3 * field));
          const float scale = d * (1.0F + 2.0F * localScale);
          const float delta = negativeDelta ? -0.125F : 0.125F;
          for (std::size_t lane = 0; lane < 8; ++lane) {
            const std::int8_t code =
                static_cast<std::int8_t>(grid[gridIndex * 8 + lane]);
            reference[outputBase + group * 32 + field * 8 + lane] =
                scale * (static_cast<float>(code) + delta);
          }
        }
        storeU16(packed, inputBase + 34 + group * 2, metadata);
      }
#else
      const float requestedScale =
          static_cast<float>(1U + ((row * 17U + block * 13U) % 31U)) /
          128.0F;
      const std::uint16_t scaleBits = f16Bits(requestedScale);
      const float d = f16Value(scaleBits);
      std::uint16_t scaleWords[4] = {};
      scaleWords[0] = static_cast<std::uint16_t>((scaleBits & 0x000FU) << 12);
      scaleWords[1] = static_cast<std::uint16_t>((scaleBits & 0x00F0U) << 8);
      scaleWords[2] = static_cast<std::uint16_t>((scaleBits & 0x0F00U) << 4);
      scaleWords[3] = static_cast<std::uint16_t>(scaleBits & 0xF000U);
      for (std::size_t group = 0; group < 8; ++group) {
        const std::uint16_t lowScale = static_cast<std::uint16_t>(
            (row * 3U + block * 5U + group * 7U) & 7U);
        const std::uint16_t highScale = static_cast<std::uint16_t>(
            (row * 11U + block * 13U + group * 17U + 1U) & 7U);
        const std::size_t word = group / 2;
        const std::size_t shift = 6 * (group % 2);
        scaleWords[word] |= static_cast<std::uint16_t>(lowScale << shift);
        scaleWords[word] |= static_cast<std::uint16_t>(highScale << (shift + 3));
        std::uint8_t highBytes[2] = {};
        for (std::size_t field = 0; field < 4; ++field) {
          const std::size_t gridIndex =
              (row * 11U + block * 7U + group * 17U + field * 29U) & 2047U;
          packed[inputBase + group * 4 + field] =
              static_cast<std::uint8_t>(gridIndex);
          const std::size_t highByte = field / 2;
          const std::size_t highShift = 4 * (field % 2);
          highBytes[highByte] |= static_cast<std::uint8_t>(
              (gridIndex >> 8) << highShift);
          const bool negativeDelta = ((row + block + group + field) & 1U) != 0;
          if (negativeDelta)
            highBytes[highByte] |= static_cast<std::uint8_t>(1U
                                                            << (highShift + 3));
          const float local = field < 2 ? 1.0F + 2.0F * lowScale
                                        : 1.0F + 2.0F * highScale;
          const float delta = negativeDelta ? -0.125F : 0.125F;
          for (std::size_t lane = 0; lane < 8; ++lane) {
            const std::int8_t code =
                static_cast<std::int8_t>(grid[gridIndex * 8 + lane]);
            reference[outputBase + group * 32 + field * 8 + lane] =
                d * local * (static_cast<float>(code) + delta);
          }
        }
        packed[inputBase + 32 + group * 2] = highBytes[0];
        packed[inputBase + 33 + group * 2] = highBytes[1];
      }
      for (std::size_t word = 0; word < 4; ++word)
        storeU16(packed, inputBase + 48 + word * 2, scaleWords[word]);
#endif
    }
  }

  auto invoke = [&]() {
    WEFT_IQ1_DEQUANT_ENTRY(packed.data(), grid.data(), output.data(), kRows,
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
                 WEFT_STRINGIFY(WEFT_IQ1_DEQUANT_ENTRY), maxAbsoluteError);
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
  std::printf("kernel=%s\n", WEFT_STRINGIFY(WEFT_IQ1_DEQUANT_ENTRY));
  std::printf("model_shape=N=1024;K=4096\n");
  std::printf("elements=%zu\n", kElements);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
