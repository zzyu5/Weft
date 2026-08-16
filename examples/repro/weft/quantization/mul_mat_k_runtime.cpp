#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef WEFT_MUL_MAT_K_KIND
#error "WEFT_MUL_MAT_K_KIND is required"
#endif
#ifndef WEFT_MUL_MAT_ENTRY
#error "WEFT_MUL_MAT_ENTRY is required"
#endif
#ifndef WEFT_GGML_DOT
#error "WEFT_GGML_DOT is required"
#endif

#if WEFT_MUL_MAT_K_KIND >= 14 && WEFT_MUL_MAT_K_KIND <= 17
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#endif

#define WEFT_STRINGIFY_INNER(value) #value
#define WEFT_STRINGIFY(value) WEFT_STRINGIFY_INNER(value)

extern "C" void WEFT_GGML_DOT(int n, float *result, std::size_t resultStride,
                               const void *weight, std::size_t weightStride,
                               const void *activation,
                               std::size_t activationStride, int rows);

namespace {

constexpr std::size_t kN = 4096;
constexpr std::size_t kK = 4096;
constexpr std::size_t kBlock = 256;
constexpr std::size_t kBlocks = kK / kBlock;
constexpr std::size_t kActivationBlockBytes = 292;
constexpr std::size_t kActivationRowBytes = kBlocks * kActivationBlockBytes;
#if WEFT_MUL_MAT_K_KIND == 2
constexpr std::size_t kWeightBlockBytes = 84;
#elif WEFT_MUL_MAT_K_KIND == 3
constexpr std::size_t kWeightBlockBytes = 110;
#elif WEFT_MUL_MAT_K_KIND == 4
constexpr std::size_t kWeightBlockBytes = 144;
#elif WEFT_MUL_MAT_K_KIND == 5
constexpr std::size_t kWeightBlockBytes = 176;
#elif WEFT_MUL_MAT_K_KIND == 6
constexpr std::size_t kWeightBlockBytes = 210;
#elif WEFT_MUL_MAT_K_KIND == 7
constexpr std::size_t kWeightBlockBytes = 54;
#elif WEFT_MUL_MAT_K_KIND == 8
constexpr std::size_t kWeightBlockBytes = 66;
#elif WEFT_MUL_MAT_K_KIND == 10
constexpr std::size_t kWeightBlockBytes = 82;
#elif WEFT_MUL_MAT_K_KIND == 11
constexpr std::size_t kWeightBlockBytes = 110;
#elif WEFT_MUL_MAT_K_KIND == 12
constexpr std::size_t kWeightBlockBytes = 56;
#elif WEFT_MUL_MAT_K_KIND == 13
constexpr std::size_t kWeightBlockBytes = 136;
#elif WEFT_MUL_MAT_K_KIND == 14
constexpr std::size_t kWeightBlockBytes = 66;
#elif WEFT_MUL_MAT_K_KIND == 15
constexpr std::size_t kWeightBlockBytes = 74;
#elif WEFT_MUL_MAT_K_KIND == 16
constexpr std::size_t kWeightBlockBytes = 98;
#elif WEFT_MUL_MAT_K_KIND == 17
constexpr std::size_t kWeightBlockBytes = 50;
#else
#error "unsupported WEFT_MUL_MAT_K_KIND"
#endif
constexpr std::size_t kWeightRowBytes = kBlocks * kWeightBlockBytes;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink;
[[maybe_unused]] constexpr std::int8_t kIQ4Codebook[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,   25,  38,  53,  69,  89,  113,
};

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

[[maybe_unused]] void writeHalf(std::uint8_t *bytes, float value) {
  const _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  bytes[0] = static_cast<std::uint8_t>(bits);
  bytes[1] = static_cast<std::uint8_t>(bits >> 8);
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

std::int16_t readI16(const std::uint8_t *bytes) {
  return static_cast<std::int16_t>(
      static_cast<std::uint16_t>(bytes[0]) |
      static_cast<std::uint16_t>(bytes[1]) << 8);
}

void initializeWeightRow(std::uint8_t *row) {
  for (std::size_t block = 0; block < kBlocks; ++block) {
    std::uint8_t *packed = row + block * kWeightBlockBytes;
    for (std::size_t byte = 0; byte < kWeightBlockBytes; ++byte)
      packed[byte] = static_cast<std::uint8_t>(
          block * 17U + byte * 29U + (byte >> 2));
#if WEFT_MUL_MAT_K_KIND == 2
    writeHalf(packed + 80, 0.015625F);
    writeHalf(packed + 82, 0.0078125F);
#elif WEFT_MUL_MAT_K_KIND == 3
    writeHalf(packed + 108, 0.015625F);
#elif WEFT_MUL_MAT_K_KIND == 4 || WEFT_MUL_MAT_K_KIND == 5
    writeHalf(packed, 0.015625F);
    writeHalf(packed + 2, 0.0078125F);
#elif WEFT_MUL_MAT_K_KIND == 6
    writeHalf(packed + 208, 0.015625F);
#elif WEFT_MUL_MAT_K_KIND == 7
    writeHalf(packed + 52, 0.015625F);
#elif WEFT_MUL_MAT_K_KIND == 8
    writeHalf(packed + 64, 0.015625F);
#elif WEFT_MUL_MAT_K_KIND == 10 || WEFT_MUL_MAT_K_KIND == 11 || \
    WEFT_MUL_MAT_K_KIND == 13 || WEFT_MUL_MAT_K_KIND == 14 || \
    WEFT_MUL_MAT_K_KIND == 15 || WEFT_MUL_MAT_K_KIND == 16 || \
    WEFT_MUL_MAT_K_KIND == 17
    writeHalf(packed, 0.015625F);
#else
    for (std::size_t byte = 48; byte < 56; ++byte)
      packed[byte] = 0;
    packed[55] = 0x3cU;
#endif
  }
}

std::int8_t expectedCode(float value, float inverse) {
  const float rounded = std::nearbyint(value * inverse);
  return static_cast<std::int8_t>(
      std::max(-128.0F, std::min(127.0F, rounded)));
}

bool validateWorkspace(const std::vector<float> &activation,
                       const std::vector<std::uint8_t> &workspace,
                       std::size_t rows) {
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < kBlocks; ++block) {
      const float *input = activation.data() + row * kK + block * kBlock;
      const std::uint8_t *packed =
          workspace.data() + row * kActivationRowBytes +
          block * kActivationBlockBytes;
      std::size_t maximumIndex = 0;
      float maximum = std::fabs(input[0]);
      for (std::size_t lane = 1; lane < kBlock; ++lane) {
        const float magnitude = std::fabs(input[lane]);
        if (magnitude > maximum) {
          maximum = magnitude;
          maximumIndex = lane;
        }
      }
      float inverse = 0.0F;
      float scale = 0.0F;
      if (maximum != 0.0F) {
        inverse = -127.0F / input[maximumIndex];
        scale = 1.0F / inverse;
      }
      if (readFloat(packed) != scale)
        return false;
      for (std::size_t group = 0; group < 16; ++group) {
        std::int32_t sum = 0;
        for (std::size_t lane = 0; lane < 16; ++lane) {
          const std::size_t index = group * 16 + lane;
          const std::int8_t expected = expectedCode(input[index], inverse);
          if (static_cast<std::int8_t>(packed[4 + index]) != expected)
            return false;
          sum += expected;
        }
        if (readI16(packed + 260 + group * 2) != sum)
          return false;
      }
    }
  }
  return true;
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
        static_cast<float>(static_cast<int>(index % 127) - 63) / 64.0F;

#if WEFT_MUL_MAT_K_KIND == 13
  WEFT_MUL_MAT_ENTRY(
      weight.data(), activation.data(),
      reinterpret_cast<const std::uint8_t *>(kIQ4Codebook), output.data(),
      workspace.data(), 0, rows, kN, kK);
#elif WEFT_MUL_MAT_K_KIND == 14
  WEFT_MUL_MAT_ENTRY(
      weight.data(), activation.data(),
      reinterpret_cast<const std::uint8_t *>(iq2xxs_grid),
      reinterpret_cast<const std::uint8_t *>(ksigns_iq2xs), output.data(),
      workspace.data(), 0, rows, kN, kK);
#elif WEFT_MUL_MAT_K_KIND == 15
  WEFT_MUL_MAT_ENTRY(
      weight.data(), activation.data(),
      reinterpret_cast<const std::uint8_t *>(iq2xs_grid),
      reinterpret_cast<const std::uint8_t *>(ksigns_iq2xs), output.data(),
      workspace.data(), 0, rows, kN, kK);
#elif WEFT_MUL_MAT_K_KIND == 16
  WEFT_MUL_MAT_ENTRY(
      weight.data(), activation.data(),
      reinterpret_cast<const std::uint8_t *>(iq3xxs_grid),
      reinterpret_cast<const std::uint8_t *>(ksigns_iq2xs), output.data(),
      workspace.data(), 0, rows, kN, kK);
#elif WEFT_MUL_MAT_K_KIND == 17
  WEFT_MUL_MAT_ENTRY(
      weight.data(), activation.data(),
      reinterpret_cast<const std::uint8_t *>(iq1s_grid), output.data(),
      workspace.data(), 0, rows, kN, kK);
#else
  WEFT_MUL_MAT_ENTRY(weight.data(), activation.data(), output.data(),
                     workspace.data(), 0, rows, kN, kK);
#endif
  if (!validateWorkspace(activation, workspace, rows)) {
    std::fprintf(stderr, "generated Q8_K workspace differs from DSL semantics\n");
    return 1;
  }

  const std::size_t sampleRows[] = {0, rows / 2, rows - 1};
  const std::size_t sampleColumns[] = {0, 1, kN / 2, kN - 1};
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t row : sampleRows) {
    for (std::size_t column : sampleColumns) {
      float expected = 0.0F;
      WEFT_GGML_DOT(
          static_cast<int>(kK), &expected, 0,
          weight.data() + column * kWeightRowBytes, 0,
          workspace.data() + row * kActivationRowBytes, 0, 1);
      const float actual = output[row * kN + column];
      const double absolute = std::fabs(static_cast<double>(actual) - expected);
      const double relative = absolute / std::fmax(1.0, std::fabs(expected));
      maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
      maxRelativeError = std::fmax(maxRelativeError, relative);
      if (relative > 2.0e-4) {
        std::fprintf(stderr,
                     "mismatch at (%zu,%zu): weft=%g ggml=%g relative=%g\n",
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
#if WEFT_MUL_MAT_K_KIND == 13
    WEFT_MUL_MAT_ENTRY(
        weight.data(), activation.data(),
        reinterpret_cast<const std::uint8_t *>(kIQ4Codebook), output.data(),
        workspace.data(), 0, rows, kN, kK);
#elif WEFT_MUL_MAT_K_KIND == 14
    WEFT_MUL_MAT_ENTRY(
        weight.data(), activation.data(),
        reinterpret_cast<const std::uint8_t *>(iq2xxs_grid),
        reinterpret_cast<const std::uint8_t *>(ksigns_iq2xs), output.data(),
        workspace.data(), 0, rows, kN, kK);
#elif WEFT_MUL_MAT_K_KIND == 15
    WEFT_MUL_MAT_ENTRY(
        weight.data(), activation.data(),
        reinterpret_cast<const std::uint8_t *>(iq2xs_grid),
        reinterpret_cast<const std::uint8_t *>(ksigns_iq2xs), output.data(),
        workspace.data(), 0, rows, kN, kK);
#elif WEFT_MUL_MAT_K_KIND == 16
    WEFT_MUL_MAT_ENTRY(
        weight.data(), activation.data(),
        reinterpret_cast<const std::uint8_t *>(iq3xxs_grid),
        reinterpret_cast<const std::uint8_t *>(ksigns_iq2xs), output.data(),
        workspace.data(), 0, rows, kN, kK);
#elif WEFT_MUL_MAT_K_KIND == 17
    WEFT_MUL_MAT_ENTRY(
        weight.data(), activation.data(),
        reinterpret_cast<const std::uint8_t *>(iq1s_grid), output.data(),
        workspace.data(), 0, rows, kN, kK);
#else
    WEFT_MUL_MAT_ENTRY(weight.data(), activation.data(), output.data(),
                       workspace.data(), 0, rows, kN, kK);
#endif
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
  std::printf("implementation=weft_core_local_q8_K_rvv\n");
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
