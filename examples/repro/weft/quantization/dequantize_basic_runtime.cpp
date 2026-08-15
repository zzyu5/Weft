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
#if WEFT_DEQUANT_KIND == 5
constexpr std::size_t kBlockSize = 128;
#elif WEFT_DEQUANT_KIND == 6 || WEFT_DEQUANT_KIND == 7
constexpr std::size_t kBlockSize = 256;
#else
constexpr std::size_t kBlockSize = 32;
#endif
#if WEFT_DEQUANT_KIND == 1
constexpr std::size_t kBlockBytes = 20;
#elif WEFT_DEQUANT_KIND == 2
constexpr std::size_t kBlockBytes = 34;
#elif WEFT_DEQUANT_KIND == 3
constexpr std::size_t kBlockBytes = 22;
#elif WEFT_DEQUANT_KIND == 4
constexpr std::size_t kBlockBytes = 24;
#elif WEFT_DEQUANT_KIND == 6
constexpr std::size_t kBlockBytes = 66;
#elif WEFT_DEQUANT_KIND == 7
constexpr std::size_t kBlockBytes = 54;
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
#if WEFT_DEQUANT_KIND == 6
      storeF16(packed, inputBase + 64, scaleValue);
#elif WEFT_DEQUANT_KIND == 7
      storeF16(packed, inputBase + 52, scaleValue);
#else
      storeF16(packed, inputBase, scaleValue);
#endif
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
#elif WEFT_DEQUANT_KIND == 3 || WEFT_DEQUANT_KIND == 4
#if WEFT_DEQUANT_KIND == 4
      constexpr std::size_t kHighOffset = 4;
      constexpr std::size_t kCodeOffset = 8;
      const float minimumValue =
          -static_cast<float>(1 + ((row * 5 + block * 3) % 7)) / 8.0F;
      storeF16(packed, inputBase + 2, minimumValue);
#else
      constexpr std::size_t kHighOffset = 2;
      constexpr std::size_t kCodeOffset = 6;
#endif
      std::uint32_t highBits = 0;
      for (std::size_t member = 0; member < 16; ++member) {
        const std::uint8_t low = static_cast<std::uint8_t>(
            (row * 3 + block * 5 + member * 7) & 31U);
        const std::uint8_t high = static_cast<std::uint8_t>(
            (row * 11 + block * 3 + member * 13 + 1) & 31U);
        packed[inputBase + kCodeOffset + member] = static_cast<std::uint8_t>(
            (low & 15U) | ((high & 15U) << 4));
        highBits |= static_cast<std::uint32_t>(low >> 4) << member;
        highBits |= static_cast<std::uint32_t>(high >> 4) << (16 + member);
#if WEFT_DEQUANT_KIND == 4
        reference[outputBase + member] = scale * low + minimumValue;
        reference[outputBase + 16 + member] = scale * high + minimumValue;
#else
        reference[outputBase + member] =
            scale * static_cast<float>(static_cast<int>(low) - 16);
        reference[outputBase + 16 + member] =
            scale * static_cast<float>(static_cast<int>(high) - 16);
#endif
      }
      for (std::size_t byte = 0; byte < 4; ++byte)
        packed[inputBase + kHighOffset + byte] =
            static_cast<std::uint8_t>(highBits >> (8 * byte));
#elif WEFT_DEQUANT_KIND == 5
      for (std::size_t byte = 0; byte < 16; ++byte) {
        const std::uint8_t signs = static_cast<std::uint8_t>(
            row * 7 + block * 11 + byte * 29 + 0x5aU);
        packed[inputBase + 2 + byte] = signs;
        for (std::size_t bit = 0; bit < 8; ++bit)
          reference[outputBase + byte * 8 + bit] =
              scale * ((signs >> bit) & 1U ? 1.0F : -1.0F);
      }
#elif WEFT_DEQUANT_KIND == 6
      for (std::size_t half = 0; half < 2; ++half) {
        for (std::size_t member = 0; member < 32; ++member) {
          std::uint8_t packedCodes = 0;
          for (std::size_t field = 0; field < 4; ++field) {
            const std::uint8_t code = static_cast<std::uint8_t>(
                (row * 3 + block * 5 + half * 7 + member * 11 + field) %
                3);
            packedCodes |= static_cast<std::uint8_t>(code << (2 * field));
            reference[outputBase + half * 128 + field * 32 + member] =
                scale * static_cast<float>(static_cast<int>(code) - 1);
          }
          packed[inputBase + half * 32 + member] = packedCodes;
        }
      }
#elif WEFT_DEQUANT_KIND == 7
      constexpr std::uint8_t powers[5] = {1, 3, 9, 27, 81};
      for (std::size_t byte = 0; byte < 48; ++byte)
        packed[inputBase + byte] = static_cast<std::uint8_t>(
            row * 7 + block * 11 + byte * 29 + 0x5aU);
      for (std::size_t byte = 0; byte < 4; ++byte)
        packed[inputBase + 48 + byte] = static_cast<std::uint8_t>(
            row * 13 + block * 3 + byte * 37 + 0x2dU);
      std::size_t outputIndex = outputBase;
      for (std::size_t begin = 0; begin < 48; begin += 32) {
        const std::size_t count = std::min<std::size_t>(32, 48 - begin);
        for (std::uint8_t power : powers) {
          for (std::size_t member = 0; member < count; ++member) {
            const std::uint8_t encoded = static_cast<std::uint8_t>(
                packed[inputBase + begin + member] * power);
            const std::uint16_t ternary =
                static_cast<std::uint16_t>(encoded) * 3U >> 8;
            reference[outputIndex++] =
                scale * (static_cast<float>(ternary) - 1.0F);
          }
        }
      }
      for (std::size_t digit = 0; digit < 4; ++digit) {
        for (std::size_t member = 0; member < 4; ++member) {
          const std::uint8_t encoded = static_cast<std::uint8_t>(
              packed[inputBase + 48 + member] * powers[digit]);
          const std::uint16_t ternary =
              static_cast<std::uint16_t>(encoded) * 3U >> 8;
          reference[outputIndex++] =
              scale * (static_cast<float>(ternary) - 1.0F);
        }
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
