#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef WEFT_MUL_MAT_KIND
#error "WEFT_MUL_MAT_KIND is required"
#endif

#ifndef WEFT_MUL_MAT_ENTRY
#error "WEFT_MUL_MAT_ENTRY is required"
#endif

#define WEFT_STRINGIFY_INNER(value) #value
#define WEFT_STRINGIFY(value) WEFT_STRINGIFY_INNER(value)

namespace {

constexpr std::size_t kN = 4096;
constexpr std::size_t kK = 4096;
constexpr std::size_t kBlock = 32;
#if WEFT_MUL_MAT_KIND == 0
constexpr std::size_t kWeightBlockBytes = 18;
constexpr std::size_t kActivationBlockBytes = 34;
#elif WEFT_MUL_MAT_KIND == 1
constexpr std::size_t kWeightBlockBytes = 20;
constexpr std::size_t kActivationBlockBytes = 36;
#elif WEFT_MUL_MAT_KIND == 2
constexpr std::size_t kWeightBlockBytes = 22;
constexpr std::size_t kActivationBlockBytes = 34;
#elif WEFT_MUL_MAT_KIND == 3
constexpr std::size_t kWeightBlockBytes = 24;
constexpr std::size_t kActivationBlockBytes = 36;
#elif WEFT_MUL_MAT_KIND == 4
constexpr std::size_t kWeightBlockBytes = 34;
constexpr std::size_t kActivationBlockBytes = 34;
#else
#error "unsupported WEFT_MUL_MAT_KIND"
#endif
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
#if WEFT_MUL_MAT_KIND == 0
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
#elif WEFT_MUL_MAT_KIND == 1
    std::uint8_t *packed = row + block * kWeightBlockBytes;
    writeHalf(packed, 0.125F);
    writeHalf(packed + 2, -0.25F);
    for (std::size_t lane = 0; lane < 16; ++lane) {
      const std::uint8_t low = static_cast<std::uint8_t>((3 * lane + block) & 15U);
      const std::uint8_t high =
          static_cast<std::uint8_t>((5 * lane + block + 1) & 15U);
      packed[4 + lane] = static_cast<std::uint8_t>(low | (high << 4));
    }
#elif WEFT_MUL_MAT_KIND == 2 || WEFT_MUL_MAT_KIND == 3
    std::uint8_t *packed = row + block * kWeightBlockBytes;
    writeHalf(packed, 0.0625F);
#if WEFT_MUL_MAT_KIND == 2
    constexpr std::size_t highOffset = 2;
    constexpr std::size_t codeOffset = 6;
#else
    writeHalf(packed + 2, -0.125F);
    constexpr std::size_t highOffset = 4;
    constexpr std::size_t codeOffset = 8;
#endif
    for (std::size_t byte = 0; byte < 4; ++byte)
      packed[highOffset + byte] =
          static_cast<std::uint8_t>(0x96U ^ (block + byte));
    for (std::size_t lane = 0; lane < 16; ++lane) {
      const std::uint8_t low = static_cast<std::uint8_t>((lane + block) & 15U);
      const std::uint8_t high =
          static_cast<std::uint8_t>((7 * lane + block + 3) & 15U);
      packed[codeOffset + lane] =
          static_cast<std::uint8_t>(low | (high << 4));
    }
#else
    std::uint8_t *packed = row + block * kWeightBlockBytes;
    writeHalf(packed, 0.125F);
    for (std::size_t lane = 0; lane < 32; ++lane)
      packed[2 + lane] = static_cast<std::uint8_t>(
          static_cast<std::int8_t>((lane + 3 * block) % 31 - 15));
#endif
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
#if WEFT_MUL_MAT_KIND == 1 || WEFT_MUL_MAT_KIND == 3
      std::int32_t codeSum = 0;
#endif
      for (std::size_t lane = 0; lane < kBlock; ++lane) {
        const std::size_t codeOffset =
            WEFT_MUL_MAT_KIND == 1 || WEFT_MUL_MAT_KIND == 3 ? 4 : 2;
        const auto actual =
            static_cast<std::int8_t>(packed[codeOffset + lane]);
        const std::int8_t expected = expectedQ8(input[lane], inverse);
        if (actual != expected)
          return false;
#if WEFT_MUL_MAT_KIND == 1 || WEFT_MUL_MAT_KIND == 3
        codeSum += expected;
#endif
      }
#if WEFT_MUL_MAT_KIND == 1 || WEFT_MUL_MAT_KIND == 3
      const _Float16 expectedSumScale =
          static_cast<_Float16>(static_cast<float>(codeSum) * scale);
      if (readHalf(packed + 2) != static_cast<float>(expectedSumScale))
        return false;
#endif
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
#if WEFT_MUL_MAT_KIND == 0 || WEFT_MUL_MAT_KIND == 1
    for (std::size_t lane = 0; lane < 16; ++lane) {
      const std::size_t weightCodeOffset = WEFT_MUL_MAT_KIND == 0 ? 2 : 4;
      const std::size_t activationCodeOffset = WEFT_MUL_MAT_KIND == 0 ? 2 : 4;
      const std::uint8_t packed = x[weightCodeOffset + lane];
      const int zero = WEFT_MUL_MAT_KIND == 0 ? -8 : 0;
      integerSum += (static_cast<int>(packed & 15U) + zero) *
                    static_cast<std::int8_t>(y[activationCodeOffset + lane]);
      integerSum += (static_cast<int>(packed >> 4) + zero) *
                    static_cast<std::int8_t>(
                        y[activationCodeOffset + 16 + lane]);
    }
    result += readHalf(x) * readHalf(y) * static_cast<float>(integerSum);
#if WEFT_MUL_MAT_KIND == 1
    result += readHalf(x + 2) * readHalf(y + 2);
#endif
#elif WEFT_MUL_MAT_KIND == 2 || WEFT_MUL_MAT_KIND == 3
#if WEFT_MUL_MAT_KIND == 2
    constexpr std::size_t highOffset = 2;
    constexpr std::size_t codeOffset = 6;
    constexpr std::size_t activationCodeOffset = 2;
#else
    constexpr std::size_t highOffset = 4;
    constexpr std::size_t codeOffset = 8;
    constexpr std::size_t activationCodeOffset = 4;
#endif
    const std::uint32_t highBits =
        static_cast<std::uint32_t>(x[highOffset]) |
        static_cast<std::uint32_t>(x[highOffset + 1]) << 8 |
        static_cast<std::uint32_t>(x[highOffset + 2]) << 16 |
        static_cast<std::uint32_t>(x[highOffset + 3]) << 24;
    for (std::size_t lane = 0; lane < 32; ++lane) {
      const std::uint8_t packed = x[codeOffset + lane % 16];
      const unsigned shift = lane < 16 ? 0 : 4;
      const int low = (packed >> shift) & 15U;
      const int high = (highBits >> lane) & 1U;
      const int value = low | (high << 4);
      integerSum += (value + (WEFT_MUL_MAT_KIND == 2 ? -16 : 0)) *
                    static_cast<std::int8_t>(y[activationCodeOffset + lane]);
    }
    result += readHalf(x) * readHalf(y) * static_cast<float>(integerSum);
#if WEFT_MUL_MAT_KIND == 3
    result += readHalf(x + 2) * readHalf(y + 2);
#endif
#else
    for (std::size_t lane = 0; lane < 32; ++lane)
      integerSum += static_cast<std::int8_t>(x[2 + lane]) *
                    static_cast<std::int8_t>(y[2 + lane]);
    result += readHalf(x) * readHalf(y) * static_cast<float>(integerSum);
#endif
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

  WEFT_MUL_MAT_ENTRY(weight.data(), activation.data(), output.data(),
                     workspace.data(), 0, rows, kN, kK);
  if (!validateQuantizedActivation(activation, workspace, rows)) {
    std::fprintf(stderr, "generated Q8 workspace differs from DSL semantics\n");
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
    WEFT_MUL_MAT_ENTRY(weight.data(), activation.data(), output.data(),
                       workspace.data(), 0, rows, kN, kK);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }

  const double microseconds = median(samples);
  const double operations = 2.0 * static_cast<double>(rows) *
                            static_cast<double>(kN) * static_cast<double>(kK);
  std::printf("family=mul_mat\n");
  std::printf("kernel=%s\n", WEFT_STRINGIFY(WEFT_MUL_MAT_ENTRY));
  std::printf("phase=%s\n", argv[1]);
  std::printf("implementation=weft_core_local_q8_rvv\n");
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
