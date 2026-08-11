#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

extern "C" void quantize_q8_0(const float *x, std::int8_t *output,
                              std::size_t block_begin,
                              std::size_t block_end);

namespace {

constexpr std::size_t kRows = 128;
constexpr std::size_t kColumns = 14336;
constexpr std::size_t kBlockSize = 32;
constexpr std::size_t kBlockBytes = 34;
constexpr std::size_t kElements = kRows * kColumns;
constexpr std::size_t kBlocks = kElements / kBlockSize;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
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
  std::vector<float> input(kElements);
  std::vector<std::int8_t> output(kBlocks * kBlockBytes);
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<float>(static_cast<int>(index % 4093) - 2046) /
                   257.0F;

  quantize_q8_0(input.data(), output.data(), 0, kBlocks);
  for (std::size_t block = 0; block < kBlocks; ++block) {
    float maximum = 0.0F;
    for (std::size_t lane = 0; lane < kBlockSize; ++lane)
      maximum = std::max(maximum,
                         std::fabs(input[block * kBlockSize + lane]));
    const float scale = maximum / 127.0F;
    const float inverse = scale == 0.0F ? 0.0F : 1.0F / scale;
    const _Float16 half = static_cast<_Float16>(scale);
    std::uint16_t bits = 0;
    std::memcpy(&bits, &half, sizeof(bits));
    const std::size_t offset = block * kBlockBytes;
    if (static_cast<std::uint8_t>(output[offset]) !=
            static_cast<std::uint8_t>(bits) ||
        static_cast<std::uint8_t>(output[offset + 1]) !=
            static_cast<std::uint8_t>(bits >> 8)) {
      std::fprintf(stderr, "q8_0 scale mismatch at block=%zu\n", block);
      return 1;
    }
    for (std::size_t lane = 0; lane < kBlockSize; ++lane) {
      const auto expected = static_cast<std::int8_t>(std::nearbyint(
          static_cast<double>(input[block * kBlockSize + lane] * inverse)));
      if (output[offset + 2 + lane] != expected) {
        std::fprintf(stderr, "q8_0 code mismatch at block=%zu lane=%zu\n", block,
                     lane);
        return 1;
      }
    }
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    quantize_q8_0(input.data(), output.data(), 0, kBlocks);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  std::printf("kernel=quantize_q8_0\n");
  std::printf("model_shape=M=128;K=14336\n");
  std::printf("elements=%zu\n", kElements);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
