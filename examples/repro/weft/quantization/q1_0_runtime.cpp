#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void q1_0_q8_0_rows(
    const std::uint8_t *weight, const std::uint8_t *activation, float *output,
    std::size_t row_begin, std::size_t row_end, std::size_t elements);

namespace {

constexpr std::size_t kRows = 14336;
constexpr std::size_t kElements = 4096;
constexpr std::size_t kQ1Block = 128;
constexpr std::size_t kQ8Block = 32;
constexpr std::size_t kQ1Bytes = 18;
constexpr std::size_t kQ8Bytes = 34;
constexpr std::size_t kBlocks = kElements / kQ1Block;
constexpr std::size_t kWeightRowBytes = kBlocks * kQ1Bytes;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
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

float reference(const std::vector<std::uint8_t> &weight,
                const std::vector<std::uint8_t> &activation,
                std::size_t row) {
  float result = 0.0F;
  for (std::size_t block = 0; block < kBlocks; ++block) {
    const std::uint8_t *q1 =
        weight.data() + row * kWeightRowBytes + block * kQ1Bytes;
    const float q1Scale = halfToFloat(q1);
    for (std::size_t sub = 0; sub < 4; ++sub) {
      const std::uint8_t *q8 =
          activation.data() + (block * 4 + sub) * kQ8Bytes;
      std::int32_t integerSum = 0;
      for (std::size_t lane = 0; lane < kQ8Block; ++lane) {
        const bool positive =
            ((q1[2 + sub * 4 + lane / 8] >> (lane % 8)) & 1U) != 0;
        const auto code = static_cast<std::int8_t>(q8[2 + lane]);
        integerSum += positive ? code : -static_cast<std::int32_t>(code);
      }
      result += static_cast<float>(integerSum) * halfToFloat(q8) * q1Scale;
    }
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
  std::vector<std::uint8_t> activation((kElements / kQ8Block) * kQ8Bytes);
  std::vector<float> output(kRows);
  for (std::size_t row = 0; row < kRows; ++row)
    for (std::size_t block = 0; block < kBlocks; ++block) {
      std::uint8_t *q1 =
          weight.data() + row * kWeightRowBytes + block * kQ1Bytes;
      writeHalf(q1, ((row + block) & 1U) == 0 ? UINT16_C(0x3800)
                                              : UINT16_C(0x3c00));
      for (std::size_t byte = 0; byte < 16; ++byte)
        q1[2 + byte] = static_cast<std::uint8_t>(
            (row * 13U + block * 17U + byte * 29U) & 0xffU);
    }
  for (std::size_t block = 0; block < kElements / kQ8Block; ++block) {
    std::uint8_t *q8 = activation.data() + block * kQ8Bytes;
    writeHalf(q8, (block & 1U) == 0 ? UINT16_C(0x3800)
                                    : UINT16_C(0x3c00));
    for (std::size_t lane = 0; lane < kQ8Block; ++lane) {
      const int code = static_cast<int>((block * 31U + lane * 7U) % 255U) - 127;
      q8[2 + lane] = static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
    }
  }

  q1_0_q8_0_rows(weight.data(), activation.data(), output.data(), 0, kRows,
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
    std::fprintf(stderr, "q1_0_q8_0_rows mismatch: abs=%g rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(3);
  for (int repetition = 0; repetition < 3; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    q1_0_q8_0_rows(weight.data(), activation.data(), output.data(), 0, kRows,
                    kElements);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double operations = 2.0 * static_cast<double>(kRows * kElements);
  std::printf("kernel=q1_0_q8_0_rows\n");
  std::printf("model_shape=Llama_FFN[N=14336,K=4096]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / milliseconds / 1.0e6);
  return 0;
}
