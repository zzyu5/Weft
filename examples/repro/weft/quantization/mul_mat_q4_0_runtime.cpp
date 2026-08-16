#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr std::size_t kN = 4096;
constexpr std::size_t kK = 4096;
constexpr std::size_t kBlock = 32;
constexpr std::size_t kWeightBlockBytes = 18;
constexpr std::size_t kActivationBlockBytes = 34;
constexpr std::size_t kBlocks = kK / kBlock;
constexpr std::size_t kWeightRowBytes = kBlocks * kWeightBlockBytes;
constexpr std::size_t kActivationRowBytes = kBlocks * kActivationBlockBytes;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink;

std::size_t phaseRows(const char *phase) {
  if (std::strcmp(phase, "decode") == 0)
    return 1;
  if (std::strcmp(phase, "prefill") == 0)
    return 128;
  return 0;
}

std::size_t parseRepetitions(const char *text) {
  char *end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (*text == '\0' || *end != '\0' || value == 0)
    return 0;
  return static_cast<std::size_t>(value);
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

void writeHalf(std::uint8_t *bytes, float value) {
  const _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  bytes[0] = static_cast<std::uint8_t>(bits);
  bytes[1] = static_cast<std::uint8_t>(bits >> 8);
}

float readHalf(const std::uint8_t *bytes) {
  const std::uint16_t bits = static_cast<std::uint16_t>(bytes[0]) |
                             static_cast<std::uint16_t>(bytes[1]) << 8;
  _Float16 half = 0;
  std::memcpy(&half, &bits, sizeof(bits));
  return static_cast<float>(half);
}

void initializeWeightRow(std::uint8_t *row) {
  for (std::size_t block = 0; block < kBlocks; ++block) {
    float source[kBlock];
    float absoluteMaximum = 0.0F;
    float signedMaximum = 0.0F;
    for (std::size_t lane = 0; lane < kBlock; ++lane) {
      const std::size_t inner = block * kBlock + lane;
      source[lane] =
          static_cast<float>(static_cast<int>(inner % 31) - 15) / 16.0F;
      if (absoluteMaximum < std::fabs(source[lane])) {
        absoluteMaximum = std::fabs(source[lane]);
        signedMaximum = source[lane];
      }
    }
    const float scale = signedMaximum / -8.0F;
    const float inverse = scale == 0.0F ? 0.0F : 1.0F / scale;
    std::uint8_t *packed = row + block * kWeightBlockBytes;
    writeHalf(packed, scale);
    for (std::size_t lane = 0; lane < kBlock / 2; ++lane) {
      const auto low = static_cast<std::uint8_t>(std::min(
          15, static_cast<int>(source[lane] * inverse + 8.5F)));
      const auto high = static_cast<std::uint8_t>(std::min(
          15, static_cast<int>(source[lane + 16] * inverse + 8.5F)));
      packed[2 + lane] = static_cast<std::uint8_t>(low | (high << 4));
    }
  }
}

std::int8_t expectedQ8(float value, float inverse) {
  const float rounded = std::nearbyint(value * inverse);
  return static_cast<std::int8_t>(
      std::max(-128.0F, std::min(127.0F, rounded)));
}

bool validateQuantizedActivation(const std::vector<float> &activation,
                                 const std::vector<std::uint8_t> &workspace,
                                 std::size_t rows) {
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < kBlocks; ++block) {
      const float *input = activation.data() + row * kK + block * kBlock;
      const std::uint8_t *packed =
          workspace.data() + row * kActivationRowBytes +
          block * kActivationBlockBytes;
      float maximum = 0.0F;
      for (std::size_t lane = 0; lane < kBlock; ++lane)
        maximum = std::max(maximum, std::fabs(input[lane]));
      const float scale = maximum / 127.0F;
      const float inverse = maximum == 0.0F ? 0.0F : 1.0F / scale;
      const _Float16 expectedScale = static_cast<_Float16>(scale);
      if (readHalf(packed) != static_cast<float>(expectedScale))
        return false;
      for (std::size_t lane = 0; lane < kBlock; ++lane) {
        const auto actual = static_cast<std::int8_t>(packed[2 + lane]);
        if (actual != expectedQ8(input[lane], inverse))
          return false;
      }
    }
  }
  return true;
}

float reference(const std::uint8_t *weight,
                const std::uint8_t *activation) {
  float result = 0.0F;
  for (std::size_t block = 0; block < kBlocks; ++block) {
    const std::uint8_t *x = weight + block * kWeightBlockBytes;
    const std::uint8_t *y = activation + block * kActivationBlockBytes;
    std::int32_t integerSum = 0;
    for (std::size_t lane = 0; lane < 16; ++lane) {
      const std::uint8_t packed = x[2 + lane];
      integerSum += (static_cast<int>(packed & 15U) - 8) *
                    static_cast<std::int8_t>(y[2 + lane]);
      integerSum += (static_cast<int>(packed >> 4) - 8) *
                    static_cast<std::int8_t>(y[18 + lane]);
    }
    result += readHalf(x) * readHalf(y) * static_cast<float>(integerSum);
  }
  return result;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <decode|prefill> <repetitions>\n", argv[0]);
    return 2;
  }
  const std::size_t rows = phaseRows(argv[1]);
  const std::size_t repeatCount = parseRepetitions(argv[2]);
  if (rows == 0 || repeatCount == 0) {
    std::fprintf(stderr, "invalid phase or repetitions\n");
    return 2;
  }

  std::vector<std::uint8_t> weight(kN * kWeightRowBytes);
  std::vector<float> activation(rows * kK);
  std::vector<float> output(rows * kN, 0.0F);
  std::vector<std::uint8_t> workspace(rows * kActivationRowBytes);
  initializeWeightRow(weight.data());
  for (std::size_t column = 1; column < kN; ++column)
    std::memcpy(weight.data() + column * kWeightRowBytes, weight.data(),
                kWeightRowBytes);
  for (std::size_t index = 0; index < activation.size(); ++index)
    activation[index] =
        static_cast<float>(static_cast<int>(index % 17) - 8) / 9.0F;

  mul_mat_q4_0(weight.data(), activation.data(), output.data(),
               workspace.data(), 0, rows, kN, kK);
  if (!validateQuantizedActivation(activation, workspace, rows)) {
    std::fprintf(stderr, "generated Q8_0 workspace differs from DSL semantics\n");
    return 1;
  }

  const std::size_t sampleRows[] = {0, rows / 2, rows - 1};
  const std::size_t sampleColumns[] = {0, 1, kN / 2, kN - 1};
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t row : sampleRows) {
    for (std::size_t column : sampleColumns) {
      const float expected = reference(
          weight.data() + column * kWeightRowBytes,
          workspace.data() + row * kActivationRowBytes);
      const float actual = output[row * kN + column];
      const double absolute = std::fabs(static_cast<double>(actual) - expected);
      const double relative = absolute / std::fmax(1.0, std::fabs(expected));
      maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
      maxRelativeError = std::fmax(maxRelativeError, relative);
      if (relative > 1.0e-5) {
        std::fprintf(stderr,
                     "mismatch at (%zu,%zu): weft=%g reference=%g relative=%g\n",
                     row, column, actual, expected, relative);
        return 1;
      }
    }
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(repeatCount);
  for (std::size_t repetition = 0; repetition < repeatCount; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    mul_mat_q4_0(weight.data(), activation.data(), output.data(),
                 workspace.data(), 0, rows, kN, kK);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }

  const double microseconds = median(samples);
  const double operations = 2.0 * static_cast<double>(rows) *
                            static_cast<double>(kN) * static_cast<double>(kK);
  std::printf("family=mul_mat\n");
  std::printf("kernel=mul_mat_q4_0\n");
  std::printf("phase=%s\n", argv[1]);
  std::printf("implementation=weft_core_local_q8_0_rvv\n");
  std::printf("model_shape=Llama-8B.hidden_projection\n");
  std::printf("M=%zu\nN=%zu\nK=%zu\n", rows, kN, kK);
  std::printf("cold_protocol=64MiB-evict-then-single-kernel\n");
  std::printf("repetitions=%zu\n", repeatCount);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("cold_median_us=%.3f\n", microseconds);
  std::printf("cold_gop_s=%.6f\n", operations / microseconds / 1.0e3);
  std::printf("output_sample=%.9g\n", output[0]);
  return 0;
}
