#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

namespace {

constexpr std::size_t kRows = 64;
constexpr std::size_t kVocabulary = 256;
constexpr float kThreshold = 0.9F;
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
  return samples[samples.size() / 2];
}

} // namespace

int main() {
  std::vector<float> probabilities(kRows * kVocabulary);
  std::vector<float> uniforms(kRows);
  for (std::size_t row = 0; row < kRows; ++row) {
    double total = 0.0;
    for (std::size_t token = 0; token < kVocabulary; ++token) {
      const float value =
          0.1F + static_cast<float>((row * 977U + token * 811U) % 65521U) /
                     65536.0F;
      probabilities[row * kVocabulary + token] = value;
      total += value;
    }
    for (std::size_t token = 0; token < kVocabulary; ++token)
      probabilities[row * kVocabulary + token] =
          static_cast<float>(probabilities[row * kVocabulary + token] / total);
    uniforms[row] =
        static_cast<float>((row * 104729U + 15485863U) % 1000003U) /
        1000003.0F;
  }

  std::vector<std::uint32_t> expectedIndices(kRows * kVocabulary);
  std::vector<std::uint32_t> expectedCounts(kRows);
  std::vector<std::uint32_t> expectedSamples(kRows);
  for (std::size_t row = 0; row < kRows; ++row) {
    auto first = expectedIndices.begin() + row * kVocabulary;
    std::iota(first, first + kVocabulary, 0U);
    std::sort(first, first + kVocabulary, [&](std::uint32_t lhs,
                                              std::uint32_t rhs) {
      const float left = probabilities[row * kVocabulary + lhs];
      const float right = probabilities[row * kVocabulary + rhs];
      return left > right || (left == right && lhs < rhs);
    });
    float prefix = 0.0F;
    expectedCounts[row] = static_cast<std::uint32_t>(kVocabulary);
    for (std::size_t rank = 0; rank < kVocabulary; ++rank) {
      prefix += probabilities[row * kVocabulary + first[rank]];
      if (prefix >= kThreshold) {
        expectedCounts[row] = static_cast<std::uint32_t>(rank + 1);
        break;
      }
    }
    float nucleusMass = 0.0F;
    for (std::size_t rank = 0; rank < expectedCounts[row]; ++rank)
      nucleusMass += probabilities[row * kVocabulary + first[rank]];
    const float draw = uniforms[row] * nucleusMass;
    float samplePrefix = 0.0F;
    expectedSamples[row] = first[0];
    for (std::size_t rank = 0; rank < expectedCounts[row]; ++rank) {
      samplePrefix += probabilities[row * kVocabulary + first[rank]];
      if (samplePrefix >= draw) {
        expectedSamples[row] = first[rank];
        break;
      }
    }
  }

  std::vector<std::uint32_t> indices(
      top_p_nucleus_f32__sorted_indices_elements(kRows, kVocabulary), 0);
  std::vector<float> sortedProbabilities(
      top_p_nucleus_f32__sorted_probabilities_elements(kRows, kVocabulary),
      0.0F);
  std::vector<std::uint32_t> sortScratch(
      top_p_nucleus_f32__sort_scratch_elements(kVocabulary), 0);
  std::vector<std::uint32_t> counts(kRows, 0);
  std::vector<std::uint32_t> sampledTokens(kRows, 0);
  top_p_nucleus_f32(probabilities.data(), indices.data(),
                    sortedProbabilities.data(), sortScratch.data(),
                    uniforms.data(), counts.data(),
                    sampledTokens.data(), kRows, kVocabulary, kThreshold);
  if (indices != expectedIndices || counts != expectedCounts ||
      sampledTokens != expectedSamples) {
    std::fprintf(stderr, "top-p nucleus order, cutoff, or sample mismatch\n");
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(5);
  for (int repetition = 0; repetition < 5; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    top_p_nucleus_f32(probabilities.data(), indices.data(),
                      sortedProbabilities.data(), sortScratch.data(),
                      uniforms.data(), counts.data(),
                      sampledTokens.data(), kRows, kVocabulary, kThreshold);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=top_p_nucleus_f32\n");
  std::printf("model_shape=decoder_sampling[rows=%zu,vocabulary=%zu,p=%.2f]\n",
              kRows, kVocabulary, kThreshold);
  std::printf("order_mismatches=0\n");
  std::printf("cutoff_mismatches=0\n");
  std::printf("sample_mismatches=0\n");
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("million_values_s=%.6f\n",
              static_cast<double>(kRows * kVocabulary) / milliseconds /
                  1000.0);
  return 0;
}
