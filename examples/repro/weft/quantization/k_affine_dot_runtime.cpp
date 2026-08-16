#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef WEFT_K_AFFINE_DOT_KIND
#error "WEFT_K_AFFINE_DOT_KIND is required"
#endif

#ifndef WEFT_K_AFFINE_DOT_ENTRY
#error "WEFT_K_AFFINE_DOT_ENTRY is required"
#endif

#define WEFT_STRINGIFY_INNER(value) #value
#define WEFT_STRINGIFY(value) WEFT_STRINGIFY_INNER(value)

namespace {

constexpr std::size_t kRows = 14336;
constexpr std::size_t kElements = 4096;
constexpr std::size_t kBlockSize = 256;
#if WEFT_K_AFFINE_DOT_KIND == 2
constexpr std::size_t kWeightBytes = 84;
#elif WEFT_K_AFFINE_DOT_KIND == 3
constexpr std::size_t kWeightBytes = 110;
#elif WEFT_K_AFFINE_DOT_KIND == 5
constexpr std::size_t kWeightBytes = 176;
#else
#error "unsupported WEFT_K_AFFINE_DOT_KIND"
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

void writeI16(std::uint8_t *bytes, std::int16_t value) {
  const std::uint16_t bits = static_cast<std::uint16_t>(value);
  bytes[0] = static_cast<std::uint8_t>(bits);
  bytes[1] = static_cast<std::uint8_t>(bits >> 8);
}

[[maybe_unused]] std::int16_t readI16(const std::uint8_t *bytes) {
  return static_cast<std::int16_t>(
      static_cast<std::uint16_t>(bytes[0]) |
      static_cast<std::uint16_t>(bytes[1]) << 8);
}

[[maybe_unused]] std::uint8_t q3Scale(const std::uint8_t *scales,
                                      std::size_t index) {
  const std::size_t quarter = index / 4;
  const std::size_t lane = index % 4;
  const std::uint8_t base = scales[(quarter % 2) * 4 + lane];
  const std::uint8_t low = quarter < 2 ? base & 15U : base >> 4;
  const std::uint8_t high =
      static_cast<std::uint8_t>((scales[8 + lane] >> (2 * quarter)) & 3U);
  return static_cast<std::uint8_t>(low | (high << 4));
}

[[maybe_unused]] void k4ScaleMin(const std::uint8_t *metadata,
                                 std::size_t group, std::uint8_t &scale,
                                 std::uint8_t &minimum) {
  if (group < 4) {
    scale = static_cast<std::uint8_t>(metadata[group] & 63U);
    minimum = static_cast<std::uint8_t>(metadata[4 + group] & 63U);
    return;
  }
  scale = static_cast<std::uint8_t>(
      (metadata[8 + group - 4] & 15U) | ((metadata[group - 4] >> 6) << 4));
  minimum = static_cast<std::uint8_t>(
      (metadata[8 + group - 4] >> 4) | ((metadata[group] >> 6) << 4));
}

float referenceBlock(const std::uint8_t *x, const std::uint8_t *y) {
  const float activationScale = readFloat(y);
  float result = 0.0F;
#if WEFT_K_AFFINE_DOT_KIND == 2
  const float d = halfToFloat(x + 80);
  const float dmin = halfToFloat(x + 82);
  for (std::size_t group = 0; group < 16; ++group) {
    const std::size_t half = group / 8;
    const std::size_t field = (group % 8) / 2;
    const std::size_t laneGroup = group % 2;
    std::int32_t qsum = 0;
    for (std::size_t lane = 0; lane < 16; ++lane) {
      const std::uint8_t packed = x[16 + half * 32 + laneGroup * 16 + lane];
      const int code = (packed >> (2 * field)) & 3U;
      qsum += code * static_cast<std::int8_t>(y[4 + group * 16 + lane]);
    }
    const std::uint8_t metadata = x[group];
    result += activationScale *
              (d * static_cast<float>(metadata & 15U) * qsum -
               dmin * static_cast<float>(metadata >> 4) *
                   readI16(y + 260 + group * 2));
  }
#elif WEFT_K_AFFINE_DOT_KIND == 3
  const float scaleProduct = halfToFloat(x + 108) * activationScale;
  for (std::size_t group = 0; group < 16; ++group) {
    const std::size_t half = group / 8;
    const std::size_t field = (group % 8) / 2;
    const std::size_t laneGroup = group % 2;
    std::int32_t qsum = 0;
    for (std::size_t lane = 0; lane < 16; ++lane) {
      const std::uint8_t packed = x[32 + half * 32 + laneGroup * 16 + lane];
      const int low = (packed >> (2 * field)) & 3U;
      const bool present =
          ((x[laneGroup * 16 + lane] >> (half * 4 + field)) & 1U) != 0;
      const int code = low - (present ? 0 : 4);
      qsum += code * static_cast<std::int8_t>(y[4 + group * 16 + lane]);
    }
    result += scaleProduct *
              (static_cast<int>(q3Scale(x + 96, group)) - 32) * qsum;
  }
#else
  const float d = halfToFloat(x);
  const float dmin = halfToFloat(x + 2);
  for (std::size_t group = 0; group < 8; ++group) {
    const std::size_t pair = group / 2;
    const bool high = (group & 1U) != 0;
    std::uint8_t scale = 0;
    std::uint8_t minimum = 0;
    k4ScaleMin(x + 4, group, scale, minimum);
    std::int32_t qsum = 0;
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const std::uint8_t packed = x[48 + pair * 32 + lane];
      const std::uint8_t highBits = x[16 + lane];
      const unsigned bit = 2 * pair + (high ? 1U : 0U);
      const int code = (high ? packed >> 4 : packed & 15U) |
                       (((highBits >> bit) & 1U) << 4);
      qsum += code * static_cast<std::int8_t>(y[4 + group * 32 + lane]);
    }
    const std::int32_t activationSum =
        readI16(y + 260 + group * 4) + readI16(y + 262 + group * 4);
    result += activationScale *
              (d * static_cast<float>(scale) * qsum -
               dmin * static_cast<float>(minimum) * activationSum);
  }
#endif
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
  std::vector<float> output(kRows);

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
      writeI16(y + 260 + group * 2, static_cast<std::int16_t>(sum));
    }
  }

  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t block = 0; block < kBlocks; ++block) {
      std::uint8_t *x =
          weight.data() + row * kWeightRowBytes + block * kWeightBytes;
#if WEFT_K_AFFINE_DOT_KIND == 2
      writeHalf(x + 80,
                static_cast<float>(1U + ((row * 7U + block * 3U) % 15U)) /
                    64.0F);
      writeHalf(x + 82,
                static_cast<float>(1U + ((row * 5U + block * 11U) % 7U)) /
                    32.0F);
      for (std::size_t group = 0; group < 16; ++group) {
        const std::uint8_t scale = static_cast<std::uint8_t>(
            1U + ((row * 3U + block * 5U + group * 7U) % 15U));
        const std::uint8_t minimum = static_cast<std::uint8_t>(
            1U + ((row * 11U + block * 13U + group * 5U) % 15U));
        x[group] = static_cast<std::uint8_t>(scale | (minimum << 4));
      }
      for (std::size_t half = 0; half < 2; ++half)
        for (std::size_t laneGroup = 0; laneGroup < 2; ++laneGroup)
          for (std::size_t lane = 0; lane < 16; ++lane)
            for (std::size_t field = 0; field < 4; ++field) {
              const std::uint8_t code = static_cast<std::uint8_t>(
                  (row * 3U + block * 5U + half * 7U + field * 11U +
                   laneGroup * 13U + lane) &
                  3U);
              x[16 + half * 32 + laneGroup * 16 + lane] |=
                  static_cast<std::uint8_t>(code << (2 * field));
            }
#elif WEFT_K_AFFINE_DOT_KIND == 3
      writeHalf(x + 108,
                static_cast<float>(1U + ((row * 7U + block * 3U) % 15U)) /
                    128.0F);
      std::uint8_t scales[16];
      for (std::size_t index = 0; index < 16; ++index)
        scales[index] = static_cast<std::uint8_t>(
            (row * 3U + block * 5U + index * 7U) & 63U);
      for (std::size_t index = 0; index < 16; ++index) {
        const std::size_t quarter = index / 4;
        const std::size_t lane = index % 4;
        std::uint8_t &base = x[96 + (quarter % 2) * 4 + lane];
        if (quarter < 2)
          base |= scales[index] & 15U;
        else
          base |= static_cast<std::uint8_t>((scales[index] & 15U) << 4);
        x[104 + lane] |= static_cast<std::uint8_t>(
            (scales[index] >> 4) << (2 * quarter));
      }
      for (std::size_t half = 0; half < 2; ++half)
        for (std::size_t field = 0; field < 4; ++field)
          for (std::size_t laneGroup = 0; laneGroup < 2; ++laneGroup)
            for (std::size_t lane = 0; lane < 16; ++lane) {
              const int code = static_cast<int>(
                                   (row * 3U + block * 5U + half * 7U +
                                    field * 11U + laneGroup * 13U + lane) %
                                   8U) -
                               4;
              const std::uint8_t low =
                  static_cast<std::uint8_t>(code < 0 ? code + 4 : code);
              x[32 + half * 32 + laneGroup * 16 + lane] |=
                  static_cast<std::uint8_t>(low << (2 * field));
              if (code >= 0)
                x[laneGroup * 16 + lane] |=
                    static_cast<std::uint8_t>(1U << (half * 4 + field));
            }
#else
      writeHalf(x,
                static_cast<float>(1U + ((row * 7U + block * 3U) % 15U)) /
                    64.0F);
      writeHalf(x + 2,
                static_cast<float>(1U + ((row * 5U + block * 11U) % 7U)) /
                    32.0F);
      std::uint8_t scales[8];
      std::uint8_t minima[8];
      std::uint8_t metadata[12] = {};
      for (std::size_t group = 0; group < 8; ++group) {
        scales[group] = static_cast<std::uint8_t>(
            1U + ((row * 3U + block * 5U + group * 7U) % 63U));
        minima[group] = static_cast<std::uint8_t>(
            1U + ((row * 11U + block * 13U + group * 5U) % 63U));
      }
      for (std::size_t group = 0; group < 4; ++group) {
        metadata[group] = scales[group];
        metadata[4 + group] = minima[group];
      }
      for (std::size_t group = 4; group < 8; ++group) {
        metadata[group - 4] |= static_cast<std::uint8_t>(
            (scales[group] >> 4) << 6);
        metadata[group] |= static_cast<std::uint8_t>((minima[group] >> 4) << 6);
        metadata[4 + group] = static_cast<std::uint8_t>(
            (scales[group] & 15U) | ((minima[group] & 15U) << 4));
      }
      std::memcpy(x + 4, metadata, sizeof(metadata));
      for (std::size_t pair = 0; pair < 4; ++pair)
        for (std::size_t lane = 0; lane < 32; ++lane) {
          const std::uint8_t low = static_cast<std::uint8_t>(
              (row * 3U + block * 5U + pair * 7U + lane * 11U) & 31U);
          const std::uint8_t high = static_cast<std::uint8_t>(
              (row * 13U + block * 3U + pair * 5U + lane * 7U + 1U) & 31U);
          x[16 + lane] |= static_cast<std::uint8_t>(
              ((low >> 4) << (2 * pair)) |
              ((high >> 4) << (2 * pair + 1)));
          x[48 + pair * 32 + lane] = static_cast<std::uint8_t>(
              (low & 15U) | ((high & 15U) << 4));
        }
#endif
    }
  }

  auto invoke = [&]() {
    for (std::size_t row = 0; row < kRows; ++row)
      output[row] = WEFT_K_AFFINE_DOT_ENTRY(
          weight.data() + row * kWeightRowBytes, activation.data(), kBlocks);
  };
  invoke();
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t row = 0; row < kRows; ++row) {
    float expected = 0.0F;
    for (std::size_t block = 0; block < kBlocks; ++block)
      expected += referenceBlock(
          weight.data() + row * kWeightRowBytes + block * kWeightBytes,
          activation.data() + block * kActivationBytes);
    const double absolute = std::fabs(static_cast<double>(output[row]) - expected);
    const double relative = absolute / std::fmax(1.0, std::fabs(expected));
    maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
    maxRelativeError = std::fmax(maxRelativeError, relative);
  }
  if (maxRelativeError > 1.0e-5) {
    std::fprintf(stderr, "%s mismatch: abs=%g rel=%g\n",
                 WEFT_STRINGIFY(WEFT_K_AFFINE_DOT_ENTRY), maxAbsoluteError,
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
  std::printf("kernel=%s\n", WEFT_STRINGIFY(WEFT_K_AFFINE_DOT_ENTRY));
  std::printf("model_shape=M=1;N=14336;K=4096\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / milliseconds / 1.0e6);
  return 0;
}
