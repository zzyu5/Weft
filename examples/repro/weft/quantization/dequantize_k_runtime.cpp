#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef WEFT_K_DEQUANT_KIND
#error "WEFT_K_DEQUANT_KIND is required"
#endif

#ifndef WEFT_K_DEQUANT_ENTRY
#error "WEFT_K_DEQUANT_ENTRY is required"
#endif

#define WEFT_STRINGIFY_INNER(value) #value
#define WEFT_STRINGIFY(value) WEFT_STRINGIFY_INNER(value)

namespace {

constexpr std::size_t kRows = 1024;
constexpr std::size_t kColumns = 4096;
constexpr std::size_t kBlockSize = 256;
#if WEFT_K_DEQUANT_KIND == 2
constexpr std::size_t kBlockBytes = 84;
#elif WEFT_K_DEQUANT_KIND == 3
constexpr std::size_t kBlockBytes = 110;
#elif WEFT_K_DEQUANT_KIND == 5
constexpr std::size_t kBlockBytes = 176;
#elif WEFT_K_DEQUANT_KIND == 6
constexpr std::size_t kBlockBytes = 210;
#else
constexpr std::size_t kBlockBytes = 144;
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
#if WEFT_K_DEQUANT_KIND == 2
      const float d =
          static_cast<float>(1 + ((row * 7 + block * 3) % 15)) / 64.0F;
      const float dmin =
          static_cast<float>(1 + ((row * 5 + block * 11) % 7)) / 32.0F;
      storeF16(packed, inputBase + 80, d);
      storeF16(packed, inputBase + 82, dmin);
      for (std::size_t half = 0; half < 2; ++half) {
        for (std::size_t field = 0; field < 4; ++field) {
          for (std::size_t laneGroup = 0; laneGroup < 2; ++laneGroup) {
            const std::size_t metadataIndex =
                half * 8 + field * 2 + laneGroup;
            const std::uint8_t scale = static_cast<std::uint8_t>(
                1 + ((row * 3 + block * 5 + metadataIndex * 7) % 15));
            const std::uint8_t minimum = static_cast<std::uint8_t>(
                1 + ((row * 11 + block * 13 + metadataIndex * 5) % 15));
            packed[inputBase + metadataIndex] =
                static_cast<std::uint8_t>(scale | (minimum << 4));
            for (std::size_t member = 0; member < 16; ++member) {
              const std::uint8_t code = static_cast<std::uint8_t>(
                  (row * 3 + block * 5 + half * 7 + field * 11 +
                   laneGroup * 13 + member) &
                  3U);
              packed[inputBase + 16 + half * 32 + laneGroup * 16 + member] |=
                  static_cast<std::uint8_t>(code << (2 * field));
              reference[outputBase + half * 128 + field * 32 +
                        laneGroup * 16 + member] =
                  d * scale * code - dmin * minimum;
            }
          }
        }
      }
#elif WEFT_K_DEQUANT_KIND == 3
      const float d =
          static_cast<float>(1 + ((row * 7 + block * 3) % 15)) / 128.0F;
      storeF16(packed, inputBase + 108, d);
      std::uint8_t scales[16];
      for (std::size_t index = 0; index < 16; ++index)
        scales[index] = static_cast<std::uint8_t>(
            (row * 3 + block * 5 + index * 7) & 63U);
      for (std::size_t index = 0; index < 16; ++index) {
        const std::size_t quarter = index / 4;
        const std::size_t lane = index % 4;
        const std::size_t base = inputBase + 96 + (quarter % 2) * 4 + lane;
        if (quarter < 2)
          packed[base] |= scales[index] & 15U;
        else
          packed[base] |= static_cast<std::uint8_t>((scales[index] & 15U) << 4);
        packed[inputBase + 104 + lane] |= static_cast<std::uint8_t>(
            (scales[index] >> 4) << (2 * quarter));
      }
      for (std::size_t half = 0; half < 2; ++half) {
        for (std::size_t field = 0; field < 4; ++field) {
          for (std::size_t laneGroup = 0; laneGroup < 2; ++laneGroup) {
            const std::size_t scaleIndex =
                half * 8 + field * 2 + laneGroup;
            for (std::size_t member = 0; member < 16; ++member) {
              const std::int8_t code = static_cast<std::int8_t>(
                  static_cast<int>((row * 3 + block * 5 + half * 7 +
                                    field * 11 + laneGroup * 13 + member) %
                                   8) -
                  4);
              const std::uint8_t low = static_cast<std::uint8_t>(
                  code < 0 ? code + 4 : code);
              packed[inputBase + 32 + half * 32 + laneGroup * 16 + member] |=
                  static_cast<std::uint8_t>(low << (2 * field));
              if (code >= 0)
                packed[inputBase + laneGroup * 16 + member] |=
                    static_cast<std::uint8_t>(1U << (half * 4 + field));
              reference[outputBase + half * 128 + field * 32 +
                        laneGroup * 16 + member] =
                  d * (static_cast<int>(scales[scaleIndex]) - 32) * code;
            }
          }
        }
      }
#elif WEFT_K_DEQUANT_KIND == 6
      const float d =
          static_cast<float>(1 + ((row * 7 + block * 3) % 15)) / 128.0F;
      storeF16(packed, inputBase + 208, d);
      std::int8_t scales[16];
      for (std::size_t index = 0; index < 16; ++index) {
        scales[index] = static_cast<std::int8_t>(
            static_cast<int>((row * 5 + block * 7 + index * 11) % 31) - 15);
        packed[inputBase + 192 + index] =
            static_cast<std::uint8_t>(scales[index]);
      }
      for (std::size_t half = 0; half < 2; ++half) {
        for (std::size_t member = 0; member < 32; ++member) {
          std::uint8_t codes[4];
          for (std::size_t field = 0; field < 4; ++field)
            codes[field] = static_cast<std::uint8_t>(
                (row * 3 + block * 5 + half * 7 + member * 11 + field * 13) &
                63U);
          packed[inputBase + half * 64 + member] = static_cast<std::uint8_t>(
              (codes[0] & 15U) | ((codes[2] & 15U) << 4));
          packed[inputBase + half * 64 + 32 + member] =
              static_cast<std::uint8_t>((codes[1] & 15U) |
                                        ((codes[3] & 15U) << 4));
          packed[inputBase + 128 + half * 32 + member] =
              static_cast<std::uint8_t>((codes[0] >> 4) |
                                        ((codes[1] >> 4) << 2) |
                                        ((codes[2] >> 4) << 4) |
                                        ((codes[3] >> 4) << 6));
          const std::size_t scaleLane = member / 16;
          for (std::size_t field = 0; field < 4; ++field) {
            const std::size_t scaleIndex =
                half * 8 + scaleLane + field * 2;
            reference[outputBase + half * 128 + field * 32 + member] =
                d * scales[scaleIndex] *
                static_cast<float>(static_cast<int>(codes[field]) - 32);
          }
        }
      }
#else
      const float d =
          static_cast<float>(1 + ((row * 7 + block * 3) % 15)) / 64.0F;
      const float dmin =
          static_cast<float>(1 + ((row * 5 + block * 11) % 7)) / 32.0F;
      storeF16(packed, inputBase, d);
      storeF16(packed, inputBase + 2, dmin);

      std::uint8_t scales[8];
      std::uint8_t minima[8];
      std::uint8_t metadata[12] = {};
      for (std::size_t group = 0; group < 8; ++group) {
        scales[group] = static_cast<std::uint8_t>(
            1 + ((row * 3 + block * 5 + group * 7) % 63));
        minima[group] = static_cast<std::uint8_t>(
            1 + ((row * 11 + block * 13 + group * 5) % 63));
      }
      for (std::size_t group = 0; group < 4; ++group) {
        metadata[group] = scales[group];
        metadata[4 + group] = minima[group];
      }
      for (std::size_t group = 4; group < 8; ++group) {
        metadata[group - 4] |= static_cast<std::uint8_t>(
            (scales[group] >> 4) << 6);
        metadata[group] |=
            static_cast<std::uint8_t>((minima[group] >> 4) << 6);
        metadata[4 + group] = static_cast<std::uint8_t>(
            (scales[group] & 15U) | ((minima[group] & 15U) << 4));
      }
      std::memcpy(packed.data() + inputBase + 4, metadata, sizeof(metadata));

      for (std::size_t pair = 0; pair < 4; ++pair) {
        const std::size_t lowGroup = 2 * pair;
        const std::size_t highGroup = lowGroup + 1;
        for (std::size_t member = 0; member < 32; ++member) {
          const std::uint8_t low = static_cast<std::uint8_t>(
              (row * 3 + block * 5 + pair * 7 + member * 11) &
#if WEFT_K_DEQUANT_KIND == 5
              31U);
#else
              15U);
#endif
          const std::uint8_t high = static_cast<std::uint8_t>(
              (row * 13 + block * 3 + pair * 5 + member * 7 + 1) &
#if WEFT_K_DEQUANT_KIND == 5
              31U);
          packed[inputBase + 16 + member] |= static_cast<std::uint8_t>(
              ((low >> 4) << (2 * pair)) |
              ((high >> 4) << (2 * pair + 1)));
          packed[inputBase + 48 + pair * 32 + member] =
              static_cast<std::uint8_t>((low & 15U) | ((high & 15U) << 4));
#else
              15U);
          packed[inputBase + 16 + pair * 32 + member] =
              static_cast<std::uint8_t>(low | (high << 4));
#endif
          reference[outputBase + lowGroup * 32 + member] =
              d * scales[lowGroup] * low - dmin * minima[lowGroup];
          reference[outputBase + highGroup * 32 + member] =
              d * scales[highGroup] * high - dmin * minima[highGroup];
        }
      }
#endif
    }
  }

  auto invoke = [&]() {
    WEFT_K_DEQUANT_ENTRY(packed.data(), output.data(), kRows, kBlocksPerRow,
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
                 WEFT_STRINGIFY(WEFT_K_DEQUANT_ENTRY), maxAbsoluteError);
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
  std::printf("kernel=%s\n", WEFT_STRINGIFY(WEFT_K_DEQUANT_ENTRY));
  std::printf("model_shape=N=1024;K=4096\n");
  std::printf("elements=%zu\n", kElements);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
