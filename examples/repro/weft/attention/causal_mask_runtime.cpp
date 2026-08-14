#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kHeads = 32;
constexpr std::size_t kQueries = 128;
constexpr std::size_t kKeys = 2048;
constexpr std::size_t kPast = kKeys - kQueries;
constexpr std::size_t kRows = kHeads * kQueries;
constexpr std::size_t kElements = kRows * kKeys;
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
  std::vector<float> scores(kElements);
  for (std::size_t index = 0; index < scores.size(); ++index)
    scores[index] = static_cast<float>(static_cast<int>(index % 4093) - 2046) /
                    128.0F;

  causal_mask_f32(scores.data(), 0, kRows, kKeys, kKeys, kQueries, kPast);
  for (std::size_t row = 0; row < kRows; ++row) {
    const std::size_t query = row % kQueries;
    for (std::size_t column = 0; column < kKeys; ++column) {
      const std::size_t index = row * kKeys + column;
      const bool masked = column > kPast + query;
      if ((masked && !std::isinf(scores[index])) ||
          (!masked && std::isinf(scores[index]))) {
        std::fprintf(stderr, "causal mask mismatch at row=%zu column=%zu\n", row,
                     column);
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
    causal_mask_f32(scores.data(), 0, kRows, kKeys, kKeys, kQueries, kPast);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  std::printf("kernel=causal_mask_f32\n");
  std::printf("model_shape=attention[heads=32,Q=128,K=2048]\n");
  std::printf("elements=%zu\n", kElements);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
