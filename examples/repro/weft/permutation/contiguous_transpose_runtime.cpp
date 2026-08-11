#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void transpose_f32(const float *source, float *destination,
                               std::size_t rows, std::size_t columns,
                               std::size_t source_stride,
                               std::size_t destination_stride);

namespace {

constexpr std::size_t kRows = 128;
constexpr std::size_t kColumns = 4096;
constexpr std::size_t kElements = kRows * kColumns;
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

  std::vector<float> source(kElements);
  std::vector<float> destination(kElements, 0.0F);
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] = static_cast<float>(static_cast<int>(index % 8191) - 4095) /
                    256.0F;

  transpose_f32(source.data(), destination.data(), kRows, kColumns, kColumns,
                kRows);
  double maxAbsoluteError = 0.0;
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t column = 0; column < kColumns; ++column) {
      const double expected = source[row * kColumns + column];
      const double actual = destination[column * kRows + row];
      maxAbsoluteError = std::max(maxAbsoluteError, std::abs(actual - expected));
    }
  }
  if (maxAbsoluteError != 0.0) {
    std::fprintf(stderr, "transpose mismatch: max_abs=%g\n", maxAbsoluteError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(repeatCount);
  for (std::size_t repetition = 0; repetition < repeatCount; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    transpose_f32(source.data(), destination.data(), kRows, kColumns, kColumns,
                  kRows);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  const double bytes = 2.0 * static_cast<double>(kElements * sizeof(float));
  std::printf("kernel=contiguous_transpose_f32\n");
  std::printf("model_shape=attention_activation[128,4096]\n");
  std::printf("rows=%zu\ncolumns=%zu\n", kRows, kColumns);
  std::printf("timing=cold-cache-64MiB-eviction-median\n");
  std::printf("repetitions=%zu\n", repeatCount);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("weft_ms=%.6f\n", milliseconds);
  std::printf("weft_gb_s=%.6f\n", bytes / milliseconds / 1.0e6);
  return 0;
}
