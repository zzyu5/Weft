#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void get_rows_q4_k(const std::uint8_t *packed_rows,
                               const std::int32_t *row_indices, float *output,
                               std::size_t tokens,
                               std::size_t blocks_per_row,
                               std::size_t row_stride_bytes,
                               std::size_t output_stride);

namespace {

constexpr std::size_t kVocabulary = 128256;
constexpr std::size_t kHidden = 4096;
constexpr std::size_t kTokens = 128;
constexpr std::size_t kBlockElements = 256;
constexpr std::size_t kBlockBytes = 144;
constexpr std::size_t kBlocksPerRow = kHidden / kBlockElements;
constexpr std::size_t kRowBytes = kBlocksPerRow * kBlockBytes;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink;

std::size_t repetitions(const char *text) {
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

void storeU16(std::uint8_t *destination, std::uint16_t value) {
  destination[0] = static_cast<std::uint8_t>(value);
  destination[1] = static_cast<std::uint8_t>(value >> 8);
}

float loadHalf(const std::uint8_t *source) {
  union HalfBits {
    std::uint16_t bits;
    _Float16 value;
  } half{static_cast<std::uint16_t>(source[0] |
                                    (std::uint16_t{source[1]} << 8))};
  return static_cast<float>(half.value);
}

void setScaleMinimum(std::uint8_t *packed, std::size_t group,
                     std::uint8_t scale, std::uint8_t minimum) {
  if (group < 4) {
    packed[group] = static_cast<std::uint8_t>((packed[group] & 0xc0U) |
                                              (scale & 0x3fU));
    packed[4 + group] = static_cast<std::uint8_t>(
        (packed[4 + group] & 0xc0U) | (minimum & 0x3fU));
    return;
  }
  const std::size_t high = group - 4;
  packed[8 + high] = static_cast<std::uint8_t>(
      (packed[8 + high] & 0xf0U) | (scale & 0x0fU));
  packed[high] = static_cast<std::uint8_t>(
      (packed[high] & 0x3fU) | ((scale >> 4) << 6));
  packed[8 + high] = static_cast<std::uint8_t>(
      (packed[8 + high] & 0x0fU) | ((minimum & 0x0fU) << 4));
  packed[4 + high] = static_cast<std::uint8_t>(
      (packed[4 + high] & 0x3fU) | ((minimum >> 4) << 6));
}

void getScaleMinimum(const std::uint8_t *packed, std::size_t group,
                     std::uint8_t &scale, std::uint8_t &minimum) {
  if (group < 4) {
    scale = packed[group] & 0x3fU;
    minimum = packed[4 + group] & 0x3fU;
    return;
  }
  scale = static_cast<std::uint8_t>(
      (packed[4 + group] & 0x0fU) | ((packed[group - 4] >> 6) << 4));
  minimum = static_cast<std::uint8_t>(
      (packed[4 + group] >> 4) | ((packed[group] >> 6) << 4));
}

void initializeRow(std::uint8_t *row, std::size_t rowIndex) {
  for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
    std::uint8_t *packedBlock = row + block * kBlockBytes;
    storeU16(packedBlock, (block + rowIndex) % 2 == 0 ? 0x3c00U : 0x3800U);
    storeU16(packedBlock + 2, 0x3800U);
    for (std::size_t group = 0; group < 8; ++group) {
      const std::uint8_t scale =
          static_cast<std::uint8_t>(1 + (rowIndex + 3 * block + group) % 31);
      const std::uint8_t minimum =
          static_cast<std::uint8_t>((5 * rowIndex + block + 7 * group) % 29);
      setScaleMinimum(packedBlock + 4, group, scale, minimum);
    }
    for (std::size_t pair = 0; pair < 4; ++pair) {
      for (std::size_t member = 0; member < 32; ++member) {
        const std::uint8_t low = static_cast<std::uint8_t>(
            (rowIndex + block + 3 * pair + member) & 0x0fU);
        const std::uint8_t high = static_cast<std::uint8_t>(
            (7 * rowIndex + 5 * block + pair + 3 * member) & 0x0fU);
        packedBlock[16 + pair * 32 + member] =
            static_cast<std::uint8_t>(low | (high << 4));
      }
    }
  }
}

float reference(const std::uint8_t *row, std::size_t column) {
  const std::size_t block = column / kBlockElements;
  const std::size_t withinBlock = column % kBlockElements;
  const std::size_t group = withinBlock / 32;
  const std::size_t member = withinBlock % 32;
  const std::uint8_t *packedBlock = row + block * kBlockBytes;
  std::uint8_t scale = 0;
  std::uint8_t minimum = 0;
  getScaleMinimum(packedBlock + 4, group, scale, minimum);
  const std::size_t pair = group / 2;
  const std::uint8_t packed = packedBlock[16 + pair * 32 + member];
  const std::uint8_t code =
      group % 2 == 0 ? packed & 0x0fU : packed >> 4;
  const float scaled = loadHalf(packedBlock) * static_cast<float>(scale) *
                       static_cast<float>(code);
  const float bias =
      loadHalf(packedBlock + 2) * static_cast<float>(minimum);
  return scaled - bias;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <repetitions>\n", argv[0]);
    return 2;
  }
  const std::size_t repeatCount = repetitions(argv[1]);
  if (repeatCount == 0) {
    std::fprintf(stderr, "invalid repetitions\n");
    return 2;
  }

  std::vector<std::uint8_t> packedRows(kVocabulary * kRowBytes, 0);
  std::vector<std::int32_t> rowIndices(kTokens);
  std::vector<float> output(kTokens * kHidden, 0.0F);
  for (std::size_t token = 0; token < kTokens; ++token) {
    const std::size_t row = (997 * token) % kVocabulary;
    rowIndices[token] = static_cast<std::int32_t>(row);
    initializeRow(packedRows.data() + row * kRowBytes, row);
  }

  get_rows_q4_k(packedRows.data(), rowIndices.data(), output.data(), kTokens,
                kBlocksPerRow, kRowBytes, kHidden);
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t token = 0; token < kTokens; ++token) {
    const std::uint8_t *row =
        packedRows.data() + static_cast<std::size_t>(rowIndices[token]) * kRowBytes;
    for (std::size_t column = 0; column < kHidden; ++column) {
      const double expected = reference(row, column);
      const double actual = output[token * kHidden + column];
      const double absolute = std::abs(actual - expected);
      const double relative = absolute / std::max(1.0, std::abs(expected));
      maxAbsoluteError = std::max(maxAbsoluteError, absolute);
      maxRelativeError = std::max(maxRelativeError, relative);
    }
  }
  if (maxAbsoluteError > 1.0e-4 && maxRelativeError > 1.0e-6) {
    std::fprintf(stderr, "get_rows mismatch: max_abs=%g max_rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(repeatCount);
  for (std::size_t repetition = 0; repetition < repeatCount; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    get_rows_q4_k(packedRows.data(), rowIndices.data(), output.data(), kTokens,
                  kBlocksPerRow, kRowBytes, kHidden);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  std::printf("kernel=get_rows_q4_k\n");
  std::printf("model_shape=embedding[vocab=128256,hidden=4096,tokens=128]\n");
  std::printf("timing=cold-cache-64MiB-eviction-median\n");
  std::printf("repetitions=%zu\n", repeatCount);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("weft_ms=%.6f\n", milliseconds);
  return 0;
}
