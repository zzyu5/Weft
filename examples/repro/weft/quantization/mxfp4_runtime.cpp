#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void mxfp4_q8_0_rows(
    const std::uint8_t *weight, const std::uint8_t *activation, float *output,
    std::size_t row_begin, std::size_t row_end, std::size_t elements);

namespace {

constexpr std::size_t kRows = 14336;
constexpr std::size_t kElements = 4096;
constexpr std::size_t kBlock = 32;
constexpr std::size_t kWeightBytes = 17;
constexpr std::size_t kQ8Bytes = 34;
constexpr std::size_t kBlocks = kElements / kBlock;
constexpr std::size_t kWeightRowBytes = kBlocks * kWeightBytes;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
constexpr std::int8_t kE2M1Doubled[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
volatile std::uint64_t evictionSink = 0;

float halfToFloat(const std::uint8_t *bytes) {
  union {
    std::uint16_t bits;
    _Float16 value;
  } decoded{static_cast<std::uint16_t>(
      static_cast<std::uint16_t>(bytes[0]) |
      static_cast<std::uint16_t>(bytes[1]) << 8)};
  return static_cast<float>(decoded.value);
}

void writeHalf(std::uint8_t *bytes, std::uint16_t bits) {
  bytes[0] = static_cast<std::uint8_t>(bits);
  bytes[1] = static_cast<std::uint8_t>(bits >> 8);
}

float e8m0Half(std::uint8_t exponent) {
  const std::uint32_t bits =
      exponent < 2 ? UINT32_C(0x00200000) << exponent
                   : static_cast<std::uint32_t>(exponent - 1U) << 23;
  union {
    std::uint32_t bits;
    float value;
  } decoded{bits};
  return decoded.value;
}

float reference(const std::vector<std::uint8_t> &weight,
                const std::vector<std::uint8_t> &activation,
                std::size_t row) {
  float result = 0.0F;
  for (std::size_t block = 0; block < kBlocks; ++block) {
    const std::uint8_t *mx =
        weight.data() + row * kWeightRowBytes + block * kWeightBytes;
    const std::uint8_t *q8 = activation.data() + block * kQ8Bytes;
    std::int32_t integerSum = 0;
    for (std::size_t lane = 0; lane < kBlock; ++lane) {
      const std::uint8_t packed = mx[1 + lane % 16];
      const std::uint8_t code =
          lane < 16 ? packed & 15U : static_cast<std::uint8_t>(packed >> 4);
      integerSum += static_cast<std::int32_t>(kE2M1Doubled[code]) *
                    static_cast<std::int8_t>(q8[2 + lane]);
    }
    result += static_cast<float>(integerSum) * e8m0Half(mx[0]) *
              halfToFloat(q8);
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
  std::vector<std::uint8_t> activation(kBlocks * kQ8Bytes);
  std::vector<float> output(kRows);
  for (std::size_t row = 0; row < kRows; ++row)
    for (std::size_t block = 0; block < kBlocks; ++block) {
      std::uint8_t *mx =
          weight.data() + row * kWeightRowBytes + block * kWeightBytes;
      mx[0] = static_cast<std::uint8_t>(126U + ((row + block) % 3U));
      for (std::size_t byte = 0; byte < 16; ++byte) {
        const std::uint8_t low =
            static_cast<std::uint8_t>((row + block + byte) % 16U);
        const std::uint8_t high =
            static_cast<std::uint8_t>((row * 3U + block + byte * 5U) % 16U);
        mx[1 + byte] = static_cast<std::uint8_t>(low | (high << 4));
      }
    }
  for (std::size_t block = 0; block < kBlocks; ++block) {
    std::uint8_t *q8 = activation.data() + block * kQ8Bytes;
    writeHalf(q8, (block & 1U) == 0 ? UINT16_C(0x3800)
                                    : UINT16_C(0x3c00));
    for (std::size_t lane = 0; lane < kBlock; ++lane) {
      const int code = static_cast<int>((block * 37U + lane * 11U) % 255U) - 127;
      q8[2 + lane] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
    }
  }

  mxfp4_q8_0_rows(weight.data(), activation.data(), output.data(), 0, kRows,
                   kElements);
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
    std::fprintf(stderr, "mxfp4_q8_0_rows mismatch: abs=%g rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(3);
  for (int repetition = 0; repetition < 3; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    mxfp4_q8_0_rows(weight.data(), activation.data(), output.data(), 0, kRows,
                     kElements);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double operations = 2.0 * static_cast<double>(kRows * kElements);
  std::printf("kernel=mxfp4_q8_0_rows\n");
  std::printf("model_shape=Llama_FFN[N=14336,K=4096]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / milliseconds / 1.0e6);
  return 0;
}
