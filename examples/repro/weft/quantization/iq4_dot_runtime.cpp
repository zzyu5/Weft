#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef WEFT_IQ4_DOT_KIND
#error "WEFT_IQ4_DOT_KIND is required"
#endif

#ifndef WEFT_IQ4_DOT_ENTRY
#error "WEFT_IQ4_DOT_ENTRY is required"
#endif

#define WEFT_STRINGIFY_INNER(value) #value
#define WEFT_STRINGIFY(value) WEFT_STRINGIFY_INNER(value)

namespace {

constexpr std::size_t kRows = 14336;
constexpr std::size_t kElements = 4096;
#if WEFT_IQ4_DOT_KIND == 0
constexpr std::size_t kBlockSize = 32;
constexpr std::size_t kWeightBytes = 18;
constexpr std::size_t kActivationBytes = 34;
#elif WEFT_IQ4_DOT_KIND == 1
constexpr std::size_t kBlockSize = 256;
constexpr std::size_t kWeightBytes = 136;
constexpr std::size_t kActivationBytes = 292;
#else
#error "unsupported WEFT_IQ4_DOT_KIND"
#endif
constexpr std::size_t kBlocks = kElements / kBlockSize;
constexpr std::size_t kWeightRowBytes = kBlocks * kWeightBytes;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
constexpr std::int8_t kCodebook[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,   25,  38,  53,  69,  89,  113,
};
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
  std::uint16_t bits = static_cast<std::uint16_t>(bytes[0]) |
                       static_cast<std::uint16_t>(bytes[1]) << 8;
  _Float16 half = 0;
  std::memcpy(&half, &bits, sizeof(bits));
  return static_cast<float>(half);
}

[[maybe_unused]] void writeFloat(std::uint8_t *bytes, float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bytes[0] = static_cast<std::uint8_t>(bits);
  bytes[1] = static_cast<std::uint8_t>(bits >> 8);
  bytes[2] = static_cast<std::uint8_t>(bits >> 16);
  bytes[3] = static_cast<std::uint8_t>(bits >> 24);
}

[[maybe_unused]] float readFloat(const std::uint8_t *bytes) {
  const std::uint32_t bits = static_cast<std::uint32_t>(bytes[0]) |
                             static_cast<std::uint32_t>(bytes[1]) << 8 |
                             static_cast<std::uint32_t>(bytes[2]) << 16 |
                             static_cast<std::uint32_t>(bytes[3]) << 24;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float reference(const std::vector<std::uint8_t> &weight,
                const std::vector<std::uint8_t> &activation,
                std::size_t row) {
  float result = 0.0F;
  for (std::size_t block = 0; block < kBlocks; ++block) {
    const std::uint8_t *x =
        weight.data() + row * kWeightRowBytes + block * kWeightBytes;
    const std::uint8_t *y = activation.data() + block * kActivationBytes;
#if WEFT_IQ4_DOT_KIND == 0
    std::int32_t sum = 0;
    for (std::size_t lane = 0; lane < 16; ++lane) {
      const std::uint8_t packed = x[2 + lane];
      sum += kCodebook[packed & 15U] * static_cast<std::int8_t>(y[2 + lane]);
      sum += kCodebook[packed >> 4] * static_cast<std::int8_t>(y[18 + lane]);
    }
    result += halfToFloat(x) * halfToFloat(y) * static_cast<float>(sum);
#else
    const std::uint16_t scalesHigh = static_cast<std::uint16_t>(x[2]) |
                                     static_cast<std::uint16_t>(x[3]) << 8;
    for (std::size_t group = 0; group < 8; ++group) {
      const std::uint8_t scalesLow = x[4 + group / 2];
      const std::uint8_t scaleCode = static_cast<std::uint8_t>(
          ((scalesLow >> (4 * (group % 2))) & 15U) |
          (((scalesHigh >> (2 * group)) & 3U) << 4));
      std::int32_t sum = 0;
      for (std::size_t lane = 0; lane < 16; ++lane) {
        const std::uint8_t packed = x[8 + group * 16 + lane];
        sum += kCodebook[packed & 15U] *
               static_cast<std::int8_t>(y[4 + group * 32 + lane]);
        sum += kCodebook[packed >> 4] *
               static_cast<std::int8_t>(y[20 + group * 32 + lane]);
      }
      result += halfToFloat(x) * readFloat(y) *
                static_cast<float>(static_cast<int>(scaleCode) - 32) *
                static_cast<float>(sum);
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
  std::vector<float> output(kRows);
  std::uint8_t codebook[16];
  std::memcpy(codebook, kCodebook, sizeof(codebook));

  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t block = 0; block < kBlocks; ++block) {
      std::uint8_t *x =
          weight.data() + row * kWeightRowBytes + block * kWeightBytes;
      writeHalf(x, static_cast<float>(1U + ((row + block) % 31U)) / 4096.0F);
#if WEFT_IQ4_DOT_KIND == 0
      for (std::size_t lane = 0; lane < 16; ++lane) {
        const std::uint8_t low = static_cast<std::uint8_t>(
            (row * 3U + block * 5U + lane * 7U) & 15U);
        const std::uint8_t high = static_cast<std::uint8_t>(
            (row * 11U + block * 13U + lane * 5U + 1U) & 15U);
        x[2 + lane] = static_cast<std::uint8_t>(low | (high << 4));
      }
#else
      std::uint16_t scalesHigh = 0;
      for (std::size_t group = 0; group < 8; ++group) {
        const std::uint8_t scaleCode = static_cast<std::uint8_t>(
            (row * 3U + block * 5U + group * 7U) & 63U);
        x[4 + group / 2] |= static_cast<std::uint8_t>(
            (scaleCode & 15U) << (4 * (group % 2)));
        scalesHigh |= static_cast<std::uint16_t>(scaleCode >> 4)
                      << (2 * group);
        for (std::size_t lane = 0; lane < 16; ++lane) {
          const std::uint8_t low = static_cast<std::uint8_t>(
              (row * 3U + block * 5U + group * 11U + lane * 7U) & 15U);
          const std::uint8_t high = static_cast<std::uint8_t>(
              (row * 13U + block * 3U + group * 5U + lane * 11U + 1U) & 15U);
          x[8 + group * 16 + lane] =
              static_cast<std::uint8_t>(low | (high << 4));
        }
      }
      x[2] = static_cast<std::uint8_t>(scalesHigh);
      x[3] = static_cast<std::uint8_t>(scalesHigh >> 8);
#endif
    }
  }
  for (std::size_t block = 0; block < kBlocks; ++block) {
    std::uint8_t *y = activation.data() + block * kActivationBytes;
#if WEFT_IQ4_DOT_KIND == 0
    writeHalf(y, (block & 1U) == 0 ? 0.5F : 1.0F);
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const int code = static_cast<int>((block * 37U + lane * 11U) % 255U) - 127;
      y[2 + lane] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
    }
#else
    writeFloat(y, 0.0078125F);
    for (std::size_t lane = 0; lane < 256; ++lane) {
      const int code = static_cast<int>((block * 37U + lane * 11U) % 127U) - 63;
      y[4 + lane] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
    }
#endif
  }

  auto invoke = [&]() {
    WEFT_IQ4_DOT_ENTRY(weight.data(), activation.data(), codebook, output.data(),
                       0, kRows, kElements);
  };
  invoke();
  const std::size_t sampleRows[] = {0, kRows / 2, kRows - 1};
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t row : sampleRows) {
    const double expected = reference(weight, activation, row);
    const double actual = output[row];
    const double absolute = std::fabs(actual - expected);
    const double relative = absolute / std::fmax(1.0, std::fabs(expected));
    maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
    maxRelativeError = std::fmax(maxRelativeError, relative);
  }
  if (maxRelativeError > 1.0e-5) {
    std::fprintf(stderr, "%s mismatch: abs=%g rel=%g\n",
                 WEFT_STRINGIFY(WEFT_IQ4_DOT_ENTRY), maxAbsoluteError,
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
  std::printf("kernel=%s\n", WEFT_STRINGIFY(WEFT_IQ4_DOT_ENTRY));
  std::printf("model_shape=M=1;N=14336;K=4096\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / milliseconds / 1.0e6);
  return 0;
}
