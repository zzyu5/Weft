#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef WEFT_IQ_LOWBIT_DOT_KIND
#error "WEFT_IQ_LOWBIT_DOT_KIND is required"
#endif

#ifndef WEFT_IQ_LOWBIT_DOT_ENTRY
#error "WEFT_IQ_LOWBIT_DOT_ENTRY is required"
#endif

#define WEFT_STRINGIFY_INNER(value) #value
#define WEFT_STRINGIFY(value) WEFT_STRINGIFY_INNER(value)

namespace {

constexpr std::size_t kRows = 14336;
constexpr std::size_t kElements = 4096;
constexpr std::size_t kBlockSize = 256;
#if WEFT_IQ_LOWBIT_DOT_KIND == 0
constexpr std::size_t kWeightBytes = 66;
constexpr std::size_t kGridRows = 256;
constexpr std::size_t kGridWidth = 8;
#elif WEFT_IQ_LOWBIT_DOT_KIND == 1
constexpr std::size_t kWeightBytes = 74;
constexpr std::size_t kGridRows = 512;
constexpr std::size_t kGridWidth = 8;
#elif WEFT_IQ_LOWBIT_DOT_KIND == 2
constexpr std::size_t kWeightBytes = 98;
constexpr std::size_t kGridRows = 256;
constexpr std::size_t kGridWidth = 4;
#elif WEFT_IQ_LOWBIT_DOT_KIND == 3
constexpr std::size_t kWeightBytes = 50;
constexpr std::size_t kGridRows = 2048;
constexpr std::size_t kGridWidth = 8;
#else
#error "unsupported WEFT_IQ_LOWBIT_DOT_KIND"
#endif
constexpr std::size_t kActivationBytes = 292;
constexpr std::size_t kBlocks = kElements / kBlockSize;
constexpr std::size_t kWeightRowBytes = kBlocks * kWeightBytes;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

float writeHalf(std::uint8_t *bytes, float value) {
  const _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  bytes[0] = static_cast<std::uint8_t>(bits);
  bytes[1] = static_cast<std::uint8_t>(bits >> 8);
  return static_cast<float>(half);
}

float halfToFloat(const std::uint8_t *bytes) {
  const std::uint16_t bits = static_cast<std::uint16_t>(bytes[0]) |
                             static_cast<std::uint16_t>(bytes[1]) << 8;
  _Float16 half = 0;
  std::memcpy(&half, &bits, sizeof(bits));
  return static_cast<float>(half);
}

void writeFloat(std::uint8_t *bytes, float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bytes[0] = static_cast<std::uint8_t>(bits);
  bytes[1] = static_cast<std::uint8_t>(bits >> 8);
  bytes[2] = static_cast<std::uint8_t>(bits >> 16);
  bytes[3] = static_cast<std::uint8_t>(bits >> 24);
}

float readFloat(const std::uint8_t *bytes) {
  const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                             static_cast<std::uint32_t>(bytes[1]) << 8 |
                             static_cast<std::uint32_t>(bytes[2]) << 16 |
                             static_cast<std::uint32_t>(bytes[3]) << 24;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void writeU16(std::uint8_t *bytes, std::uint16_t value) {
  bytes[0] = static_cast<std::uint8_t>(value);
  bytes[1] = static_cast<std::uint8_t>(value >> 8);
}

[[maybe_unused]] std::uint16_t readU16(const std::uint8_t *bytes) {
  return static_cast<std::uint16_t>(bytes[0]) |
         static_cast<std::uint16_t>(bytes[1]) << 8;
}

[[maybe_unused]] void writeU32(std::uint8_t *bytes, std::uint32_t value) {
  bytes[0] = static_cast<std::uint8_t>(value);
  bytes[1] = static_cast<std::uint8_t>(value >> 8);
  bytes[2] = static_cast<std::uint8_t>(value >> 16);
  bytes[3] = static_cast<std::uint8_t>(value >> 24);
}

[[maybe_unused]] std::uint32_t readU32(const std::uint8_t *bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         static_cast<std::uint32_t>(bytes[1]) << 8 |
         static_cast<std::uint32_t>(bytes[2]) << 16 |
         static_cast<std::uint32_t>(bytes[3]) << 24;
}

float referenceBlock(const std::uint8_t *x, const std::uint8_t *y,
                     const std::vector<std::uint8_t> &grid,
                     const std::vector<std::uint8_t> &signTable) {
  (void)signTable;
  const float d = halfToFloat(x);
  const float dy = readFloat(y);
  float result = 0.0F;
  for (std::size_t group = 0; group < 8; ++group) {
#if WEFT_IQ_LOWBIT_DOT_KIND == 0
    const std::uint32_t metadata = readU32(x + 6 + group * 8);
    const float scale = d * dy * 0.125F *
                        static_cast<float>(1U + 2U * (metadata >> 28));
    for (std::size_t field = 0; field < 4; ++field) {
      const std::size_t gridIndex = x[2 + group * 8 + field];
      const std::uint8_t signs =
          signTable[(metadata >> (7 * field)) & 127U];
      for (std::size_t lane = 0; lane < 8; ++lane) {
        const int code = grid[gridIndex * 8 + lane] *
                         (((signs >> lane) & 1U) ? -1 : 1);
        result += scale * code *
                  static_cast<std::int8_t>(y[4 + group * 32 + field * 8 + lane]);
      }
    }
#elif WEFT_IQ_LOWBIT_DOT_KIND == 1
    const std::uint8_t scales = x[66 + group];
    for (std::size_t field = 0; field < 4; ++field) {
      const std::uint16_t packed = readU16(x + 2 + (group * 4 + field) * 2);
      const std::size_t gridIndex = packed & 511U;
      const std::uint8_t signs = signTable[packed >> 9];
      const unsigned localScale =
          field < 2 ? scales & 15U : scales >> 4;
      const float scale = d * dy * 0.125F * (1U + 2U * localScale);
      for (std::size_t lane = 0; lane < 8; ++lane) {
        const int code = grid[gridIndex * 8 + lane] *
                         (((signs >> lane) & 1U) ? -1 : 1);
        result += scale * code *
                  static_cast<std::int8_t>(y[4 + group * 32 + field * 8 + lane]);
      }
    }
#elif WEFT_IQ_LOWBIT_DOT_KIND == 2
    const std::uint32_t metadata = readU32(x + 66 + group * 4);
    const float scale = d * dy * 0.25F *
                        static_cast<float>(1U + 2U * (metadata >> 28));
    for (std::size_t field = 0; field < 4; ++field) {
      const std::uint8_t signs =
          signTable[(metadata >> (7 * field)) & 127U];
      for (std::size_t subrow = 0; subrow < 2; ++subrow) {
        const std::size_t gridIndex =
            x[2 + group * 8 + field * 2 + subrow];
        for (std::size_t lane = 0; lane < 4; ++lane) {
          const std::size_t signLane = subrow * 4 + lane;
          const int code = grid[gridIndex * 4 + lane] *
                           (((signs >> signLane) & 1U) ? -1 : 1);
          result += scale * code * static_cast<std::int8_t>(
                                       y[4 + group * 32 + field * 8 + signLane]);
        }
      }
    }
#else
    const std::uint16_t metadata = readU16(x + 34 + group * 2);
    const float localScale = static_cast<float>(1U + 2U * ((metadata >> 12) & 7U));
    const float delta = (metadata & 0x8000U) ? -0.125F : 0.125F;
    for (std::size_t field = 0; field < 4; ++field) {
      const std::size_t gridIndex =
          x[2 + group * 4 + field] |
          static_cast<std::size_t>((metadata >> (3 * field)) & 7U) << 8;
      for (std::size_t lane = 0; lane < 8; ++lane) {
        const auto code = static_cast<std::int8_t>(grid[gridIndex * 8 + lane]);
        result += d * dy * localScale * (static_cast<float>(code) + delta) *
                  static_cast<std::int8_t>(
                      y[4 + group * 32 + field * 8 + lane]);
      }
    }
#endif
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
  std::vector<std::uint8_t> weight(kRows * kWeightRowBytes);
  std::vector<std::uint8_t> activation(kBlocks * kActivationBytes);
  std::vector<std::uint8_t> grid(kGridRows * kGridWidth);
  std::vector<std::uint8_t> signTable(128);
  std::vector<float> output(kRows);

  for (std::size_t index = 0; index < kGridRows; ++index) {
    for (std::size_t lane = 0; lane < kGridWidth; ++lane) {
#if WEFT_IQ_LOWBIT_DOT_KIND == 3
      const std::int8_t value = static_cast<std::int8_t>(
          static_cast<int>((index * 7U + lane * 5U) % 15U) - 7);
      grid[index * kGridWidth + lane] = static_cast<std::uint8_t>(value);
#else
      grid[index * kGridWidth + lane] = static_cast<std::uint8_t>(
          1U + 2U * ((index * 5U + lane * 3U) % 15U));
#endif
    }
  }
  for (std::size_t index = 0; index < signTable.size(); ++index)
    signTable[index] = static_cast<std::uint8_t>(index * 37U + 13U);

  for (std::size_t block = 0; block < kBlocks; ++block) {
    std::uint8_t *y = activation.data() + block * kActivationBytes;
    writeFloat(y, 0.0078125F);
    for (std::size_t lane = 0; lane < 256; ++lane) {
      const int code = static_cast<int>((block * 37U + lane * 11U) % 127U) - 63;
      y[4 + lane] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
    }
    for (std::size_t group = 0; group < 16; ++group) {
      std::int32_t sum = 0;
      for (std::size_t lane = 0; lane < 16; ++lane)
        sum += static_cast<std::int8_t>(y[4 + group * 16 + lane]);
      writeU16(y + 260 + group * 2,
               static_cast<std::uint16_t>(static_cast<std::int16_t>(sum)));
    }
  }

  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t block = 0; block < kBlocks; ++block) {
      std::uint8_t *x =
          weight.data() + row * kWeightRowBytes + block * kWeightBytes;
      writeHalf(x, static_cast<float>(1U + ((row + block) % 31U)) / 128.0F);
      for (std::size_t group = 0; group < 8; ++group) {
#if WEFT_IQ_LOWBIT_DOT_KIND == 0
        const std::uint8_t localScale = static_cast<std::uint8_t>(
            (row * 3U + block * 5U + group * 7U) & 15U);
        std::uint32_t metadata = static_cast<std::uint32_t>(localScale) << 28;
        for (std::size_t field = 0; field < 4; ++field) {
          const std::size_t gridIndex =
              (row * 11U + block * 7U + group * 17U + field * 29U) & 255U;
          x[2 + group * 8 + field] = static_cast<std::uint8_t>(gridIndex);
          const std::size_t signIndex =
              (row * 13U + block * 3U + group * 19U + field * 23U) & 127U;
          metadata |= static_cast<std::uint32_t>(signIndex) << (7 * field);
        }
        writeU32(x + 6 + group * 8, metadata);
#elif WEFT_IQ_LOWBIT_DOT_KIND == 1
        const std::uint8_t lowScale = static_cast<std::uint8_t>(
            (row * 3U + block * 5U + group * 7U) & 15U);
        const std::uint8_t highScale = static_cast<std::uint8_t>(
            (row * 11U + block * 13U + group * 17U + 1U) & 15U);
        x[66 + group] = static_cast<std::uint8_t>(lowScale | (highScale << 4));
        for (std::size_t field = 0; field < 4; ++field) {
          const std::size_t gridIndex =
              (row * 11U + block * 7U + group * 17U + field * 29U) & 511U;
          const std::size_t signIndex =
              (row * 13U + block * 3U + group * 19U + field * 23U) & 127U;
          writeU16(x + 2 + (group * 4 + field) * 2,
                   static_cast<std::uint16_t>(gridIndex | (signIndex << 9)));
        }
#elif WEFT_IQ_LOWBIT_DOT_KIND == 2
        const std::uint8_t localScale = static_cast<std::uint8_t>(
            (row * 3U + block * 5U + group * 7U) & 15U);
        std::uint32_t metadata = static_cast<std::uint32_t>(localScale) << 28;
        for (std::size_t field = 0; field < 4; ++field) {
          const std::size_t signIndex =
              (row * 13U + block * 3U + group * 19U + field * 23U) & 127U;
          metadata |= static_cast<std::uint32_t>(signIndex) << (7 * field);
          for (std::size_t subrow = 0; subrow < 2; ++subrow) {
            const std::size_t gridIndex =
                (row * 11U + block * 7U + group * 17U + field * 29U +
                 subrow * 31U) &
                255U;
            x[2 + group * 8 + field * 2 + subrow] =
                static_cast<std::uint8_t>(gridIndex);
          }
        }
        writeU32(x + 66 + group * 4, metadata);
#else
        const std::uint16_t localScale = static_cast<std::uint16_t>(
            (row * 3U + block * 5U + group * 7U) & 7U);
        std::uint16_t metadata = static_cast<std::uint16_t>(localScale << 12);
        if ((row + block + group) & 1U)
          metadata |= UINT16_C(0x8000);
        for (std::size_t field = 0; field < 4; ++field) {
          const std::size_t gridIndex =
              (row * 11U + block * 7U + group * 17U + field * 29U) & 2047U;
          x[2 + group * 4 + field] = static_cast<std::uint8_t>(gridIndex);
          metadata |= static_cast<std::uint16_t>((gridIndex >> 8)
                                                 << (3 * field));
        }
        writeU16(x + 34 + group * 2, metadata);
#endif
      }
    }
  }

  auto invoke = [&]() {
    for (std::size_t row = 0; row < kRows; ++row) {
#if WEFT_IQ_LOWBIT_DOT_KIND == 3
      output[row] = WEFT_IQ_LOWBIT_DOT_ENTRY(
          weight.data() + row * kWeightRowBytes, activation.data(), grid.data(),
          kBlocks);
#else
      output[row] = WEFT_IQ_LOWBIT_DOT_ENTRY(
          weight.data() + row * kWeightRowBytes, activation.data(), grid.data(),
          signTable.data(), kBlocks);
#endif
    }
  };
  invoke();
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t row = 0; row < kRows; ++row) {
    float expected = 0.0F;
    for (std::size_t block = 0; block < kBlocks; ++block)
      expected += referenceBlock(
          weight.data() + row * kWeightRowBytes + block * kWeightBytes,
          activation.data() + block * kActivationBytes, grid, signTable);
    const double absolute = std::fabs(static_cast<double>(output[row]) - expected);
    const double relative = absolute / std::fmax(1.0, std::fabs(expected));
    maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
    maxRelativeError = std::fmax(maxRelativeError, relative);
  }
  if (maxRelativeError > 2.0e-4) {
    std::fprintf(stderr, "%s mismatch: abs=%g rel=%g\n",
                 WEFT_STRINGIFY(WEFT_IQ_LOWBIT_DOT_ENTRY), maxAbsoluteError,
                 maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(3);
  for (int repetition = 0; repetition < 3; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    invoke();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double operations = 2.0 * static_cast<double>(kRows * kElements);
  std::printf("kernel=%s\n", WEFT_STRINGIFY(WEFT_IQ_LOWBIT_DOT_ENTRY));
  std::printf("model_shape=M=1;N=14336;K=4096\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / milliseconds / 1.0e6);
  return 0;
}
