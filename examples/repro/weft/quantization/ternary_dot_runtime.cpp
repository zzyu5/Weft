#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef WEFT_TERNARY_DOT_KIND
#error "WEFT_TERNARY_DOT_KIND is required"
#endif

#ifndef WEFT_TERNARY_DOT_ENTRY
#error "WEFT_TERNARY_DOT_ENTRY is required"
#endif

#define WEFT_STRINGIFY_INNER(value) #value
#define WEFT_STRINGIFY(value) WEFT_STRINGIFY_INNER(value)

namespace {

constexpr std::size_t kRows = 14336;
constexpr std::size_t kElements = 4096;
constexpr std::size_t kBlockSize = 256;
#if WEFT_TERNARY_DOT_KIND == 0
constexpr std::size_t kWeightBytes = 54;
#elif WEFT_TERNARY_DOT_KIND == 1
constexpr std::size_t kWeightBytes = 66;
#else
#error "unsupported WEFT_TERNARY_DOT_KIND"
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

[[maybe_unused]] int ternaryDigit(std::uint8_t packed, unsigned power) {
  const std::uint8_t encoded = static_cast<std::uint8_t>(packed * power);
  return static_cast<int>((static_cast<unsigned>(encoded) * 3U) >> 8) - 1;
}

float reference(const std::vector<std::uint8_t> &weight,
                const std::vector<std::uint8_t> &activation,
                std::size_t row) {
  float result = 0.0F;
  for (std::size_t block = 0; block < kBlocks; ++block) {
    const std::uint8_t *x =
        weight.data() + row * kWeightRowBytes + block * kWeightBytes;
    const std::uint8_t *y = activation.data() + block * kActivationBytes;
    std::int32_t sum = 0;
#if WEFT_TERNARY_DOT_KIND == 0
    unsigned power = 1;
    for (std::size_t digit = 0; digit < 5; ++digit, power *= 3)
      for (std::size_t lane = 0; lane < 32; ++lane)
        sum += ternaryDigit(x[lane], power) *
               static_cast<std::int8_t>(y[4 + digit * 32 + lane]);
    power = 1;
    for (std::size_t digit = 0; digit < 5; ++digit, power *= 3)
      for (std::size_t lane = 0; lane < 16; ++lane)
        sum += ternaryDigit(x[32 + lane], power) *
               static_cast<std::int8_t>(y[164 + digit * 16 + lane]);
    power = 1;
    for (std::size_t digit = 0; digit < 4; ++digit, power *= 3)
      for (std::size_t lane = 0; lane < 4; ++lane)
        sum += ternaryDigit(x[48 + lane], power) *
               static_cast<std::int8_t>(y[244 + digit * 4 + lane]);
    result += halfToFloat(x + 52) * readFloat(y) * static_cast<float>(sum);
#else
    for (std::size_t half = 0; half < 2; ++half) {
      for (std::size_t lane = 0; lane < 32; ++lane) {
        const std::uint8_t packed = x[half * 32 + lane];
        for (std::size_t field = 0; field < 4; ++field) {
          const int code = static_cast<int>((packed >> (2 * field)) & 3U) - 1;
          sum += code * static_cast<std::int8_t>(
                            y[4 + half * 128 + field * 32 + lane]);
        }
      }
    }
    result += halfToFloat(x + 64) * readFloat(y) * static_cast<float>(sum);
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

  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t block = 0; block < kBlocks; ++block) {
      std::uint8_t *x =
          weight.data() + row * kWeightRowBytes + block * kWeightBytes;
#if WEFT_TERNARY_DOT_KIND == 0
      for (std::size_t byte = 0; byte < 52; ++byte)
        x[byte] = static_cast<std::uint8_t>(
            row * 13U + block * 17U + byte * 29U);
      writeHalf(x + 52,
                static_cast<float>(1U + ((row + block) % 31U)) / 128.0F);
#else
      for (std::size_t byte = 0; byte < 64; ++byte)
        x[byte] = static_cast<std::uint8_t>(
            row * 13U + block * 17U + byte * 29U);
      writeHalf(x + 64,
                static_cast<float>(1U + ((row + block) % 31U)) / 128.0F);
#endif
    }
  }
  for (std::size_t block = 0; block < kBlocks; ++block) {
    std::uint8_t *y = activation.data() + block * kActivationBytes;
    writeFloat(y, 0.0078125F);
    for (std::size_t lane = 0; lane < 256; ++lane) {
      const int code = static_cast<int>((block * 37U + lane * 11U) % 127U) - 63;
      y[4 + lane] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
    }
  }

  auto invoke = [&]() {
    for (std::size_t row = 0; row < kRows; ++row)
      output[row] = WEFT_TERNARY_DOT_ENTRY(
          weight.data() + row * kWeightRowBytes, activation.data(), kBlocks);
  };
  invoke();
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t row = 0; row < kRows; ++row) {
    const double expected = reference(weight, activation, row);
    const double actual = output[row];
    const double absolute = std::fabs(actual - expected);
    const double relative = absolute / std::fmax(1.0, std::fabs(expected));
    maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
    maxRelativeError = std::fmax(maxRelativeError, relative);
  }
  if (maxRelativeError > 1.0e-5) {
    std::fprintf(stderr, "%s mismatch: abs=%g rel=%g\n",
                 WEFT_STRINGIFY(WEFT_TERNARY_DOT_ENTRY), maxAbsoluteError,
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
  std::printf("kernel=%s\n", WEFT_STRINGIFY(WEFT_TERNARY_DOT_ENTRY));
  std::printf("model_shape=M=1;N=14336;K=4096\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / milliseconds / 1.0e6);
  return 0;
}
