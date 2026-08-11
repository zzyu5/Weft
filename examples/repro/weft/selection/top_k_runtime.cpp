#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void top_k_f32(const float *scores, std::uint32_t *indices,
                          std::size_t row_begin, std::size_t row_end,
                          std::size_t columns, std::size_t k,
                          std::size_t score_stride,
                          std::size_t index_stride);
extern "C" void top_k_f32_equivalent(
    const float *scores, std::uint32_t *indices, std::size_t row_begin,
    std::size_t row_end, std::size_t columns, std::size_t k,
    std::size_t score_stride, std::size_t index_stride);

namespace {

constexpr std::size_t kTokens = 512;
constexpr std::size_t kExperts = 256;
constexpr std::size_t kSelected = 8;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

using Kernel = void (*)(const float *, std::uint32_t *, std::size_t,
                        std::size_t, std::size_t, std::size_t, std::size_t,
                        std::size_t);

void evict(std::vector<std::uint8_t> &buffer) {
  for (std::size_t index = 0; index < buffer.size(); index += 64) {
    buffer[index] = static_cast<std::uint8_t>(buffer[index] + 1U);
    evictionSink += buffer[index];
  }
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void reference(const std::vector<float> &scores,
               std::vector<std::uint32_t> &indices) {
  for (std::size_t row = 0; row < kTokens; ++row) {
    for (std::size_t rank = 0; rank < kSelected; ++rank) {
      float best = -__builtin_inff();
      std::uint32_t bestIndex = 0;
      for (std::size_t column = 0; column < kExperts; ++column) {
        bool unused = true;
        for (std::size_t previous = 0; previous < rank; ++previous)
          unused &= indices[row * kSelected + previous] != column;
        const float value = scores[row * kExperts + column];
        if (unused &&
            (value > best || (value == best && column < bestIndex))) {
          best = value;
          bestIndex = static_cast<std::uint32_t>(column);
        }
      }
      indices[row * kSelected + rank] = bestIndex;
    }
  }
}

double run(Kernel kernel, const std::vector<float> &scores,
           std::vector<std::uint32_t> &indices,
           std::vector<std::uint8_t> &eviction) {
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    kernel(scores.data(), indices.data(), 0, kTokens, kExperts, kSelected,
           kExperts, kSelected);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  return median(samples);
}

} // namespace

int main() {
  std::vector<float> scores(kTokens * kExperts);
  for (std::size_t row = 0; row < kTokens; ++row)
    for (std::size_t column = 0; column < kExperts; ++column)
      scores[row * kExperts + column] =
          static_cast<float>((row * 977U + column * 811U) % 65521U) /
              1024.0F +
          static_cast<float>(column) / 1048576.0F;

  std::vector<std::uint32_t> expected(kTokens * kSelected);
  std::vector<std::uint32_t> actual(kTokens * kSelected);
  std::vector<std::uint32_t> equivalent(kTokens * kSelected);
  reference(scores, expected);
  top_k_f32(scores.data(), actual.data(), 0, kTokens, kExperts, kSelected,
            kExperts, kSelected);
  top_k_f32_equivalent(scores.data(), equivalent.data(), 0, kTokens, kExperts,
                       kSelected, kExperts, kSelected);
  if (actual != expected || equivalent != expected) {
    std::fprintf(stderr, "top_k_f32 result mismatch\n");
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  const double primary = run(top_k_f32, scores, actual, eviction);
  const double alternate =
      run(top_k_f32_equivalent, scores, equivalent, eviction);
  std::printf("kernel=top_k_f32\n");
  std::printf("model_shape=moe_router[tokens=512,experts=256,k=8]\n");
  std::printf("primary_median_ms=%.6f\n", primary);
  std::printf("equivalent_median_ms=%.6f\n", alternate);
  return 0;
}
