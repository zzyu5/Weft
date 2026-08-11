#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" void dequantize_iq4_nl(
    const std::uint8_t *packed, const std::uint8_t *codebook, float *output,
    std::size_t rows, std::size_t blocks_per_row,
    std::size_t input_stride_bytes, std::size_t output_stride);

namespace {

constexpr std::size_t kRows = 1024;
constexpr std::size_t kColumns = 4096;
constexpr std::size_t kBlockSize = 32;
constexpr std::size_t kBlockBytes = 18;
constexpr std::size_t kBlocksPerRow = kColumns / kBlockSize;
constexpr std::size_t kInputStride = kBlocksPerRow * kBlockBytes;
constexpr std::size_t kElements = kRows * kColumns;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
constexpr std::int8_t kCodebook[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1,    13,   25,  38,  53,  69,  89,  113,
};
volatile std::uint64_t evictionSink = 0;

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

int main() {
  std::vector<std::uint8_t> packed(kRows * kInputStride);
  std::vector<float> output(kElements);
  std::vector<float> reference(kElements);
  std::uint8_t codebookBytes[16];
  std::memcpy(codebookBytes, kCodebook, sizeof(codebookBytes));

  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t block = 0; block < kBlocksPerRow; ++block) {
      const std::size_t inputOffset = row * kInputStride + block * kBlockBytes;
      const float scaleValue =
          0.0078125F * static_cast<float>(1 + ((row * 17 + block * 13) % 63));
      const _Float16 halfScale = static_cast<_Float16>(scaleValue);
      std::uint16_t scaleBits = 0;
      std::memcpy(&scaleBits, &halfScale, sizeof(scaleBits));
      packed[inputOffset] = static_cast<std::uint8_t>(scaleBits);
      packed[inputOffset + 1] = static_cast<std::uint8_t>(scaleBits >> 8);
      const float scale = static_cast<float>(halfScale);
      const std::size_t outputOffset = row * kColumns + block * kBlockSize;
      for (std::size_t member = 0; member < 16; ++member) {
        const std::uint8_t low = static_cast<std::uint8_t>(
            (row * 3 + block * 5 + member * 7) & 15U);
        const std::uint8_t high = static_cast<std::uint8_t>(
            (row * 11 + block * 3 + member * 13 + 1) & 15U);
        packed[inputOffset + 2 + member] =
            static_cast<std::uint8_t>(low | (high << 4));
        reference[outputOffset + member] =
            scale * static_cast<float>(kCodebook[low]);
        reference[outputOffset + 16 + member] =
            scale * static_cast<float>(kCodebook[high]);
      }
    }
  }

  dequantize_iq4_nl(packed.data(), codebookBytes, output.data(), kRows,
                    kBlocksPerRow, kInputStride, kColumns);
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t index = 0; index < kElements; ++index) {
    const double absolute = std::fabs(static_cast<double>(output[index]) -
                                      static_cast<double>(reference[index]));
    const double relative =
        absolute / std::max(1.0, std::fabs(static_cast<double>(reference[index])));
    maxAbsoluteError = std::max(maxAbsoluteError, absolute);
    maxRelativeError = std::max(maxRelativeError, relative);
  }
  if (maxAbsoluteError != 0.0) {
    std::fprintf(stderr, "IQ4_NL decode mismatch: max_absolute_error=%.9g\n",
                 maxAbsoluteError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    dequantize_iq4_nl(packed.data(), codebookBytes, output.data(), kRows,
                      kBlocksPerRow, kInputStride, kColumns);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  std::printf("kernel=dequantize_iq4_nl\n");
  std::printf("model_shape=N=1024;K=4096\n");
  std::printf("elements=%zu\n", kElements);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
