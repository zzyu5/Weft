#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void weighted_embedding_bag_f32(
    const float *table, const std::uint32_t *indices, const float *weights,
    const std::uint32_t *offsets, float *output, std::size_t bag_begin,
    std::size_t bag_end, std::size_t embedding_dim, std::size_t table_stride,
    std::size_t output_stride);
extern "C" void weighted_embedding_bag_f32_equivalent(
    const float *table, const std::uint32_t *indices, const float *weights,
    const std::uint32_t *offsets, float *output, std::size_t bag_begin,
    std::size_t bag_end, std::size_t embedding_dim, std::size_t table_stride,
    std::size_t output_stride);

namespace {

constexpr std::size_t kBags = 2048;
constexpr std::size_t kVocabulary = 24576;
constexpr std::size_t kEmbedding = 768;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

using Kernel = void (*)(const float *, const std::uint32_t *, const float *,
                        const std::uint32_t *, float *, std::size_t,
                        std::size_t, std::size_t, std::size_t, std::size_t);

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

double run(Kernel kernel, const std::vector<float> &table,
           const std::vector<std::uint32_t> &indices,
           const std::vector<float> &weights,
           const std::vector<std::uint32_t> &offsets,
           std::vector<float> &output, std::vector<std::uint8_t> &eviction) {
  std::vector<double> samples;
  samples.reserve(5);
  for (int repetition = 0; repetition < 5; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    kernel(table.data(), indices.data(), weights.data(), offsets.data(),
           output.data(), 0, kBags, kEmbedding, kEmbedding, kEmbedding);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  return median(samples);
}

bool compare(const std::vector<float> &actual,
             const std::vector<float> &expected, double &maxAbsolute,
             double &maxRelative) {
  maxAbsolute = 0.0;
  maxRelative = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double error = std::abs(static_cast<double>(actual[index]) -
                                  static_cast<double>(expected[index]));
    maxAbsolute = std::max(maxAbsolute, error);
    maxRelative = std::max(
        maxRelative,
        error / std::max(std::abs(static_cast<double>(expected[index])), 1.0e-6));
  }
  return maxAbsolute <= 3.0e-4 || maxRelative <= 3.0e-4;
}

} // namespace

int main() {
  std::vector<std::uint32_t> offsets(kBags + 1, 0);
  for (std::size_t bag = 0; bag < kBags; ++bag)
    offsets[bag + 1] =
        offsets[bag] + static_cast<std::uint32_t>((bag * 13U) % 31U);
  const std::size_t entries = offsets.back();

  std::vector<float> table(kVocabulary * kEmbedding);
  std::vector<std::uint32_t> indices(entries);
  std::vector<float> weights(entries);
  for (std::size_t index = 0; index < table.size(); ++index)
    table[index] =
        static_cast<float>(static_cast<int>((index * 17U) % 257U) - 128) /
        512.0F;
  for (std::size_t entry = 0; entry < entries; ++entry) {
    indices[entry] =
        static_cast<std::uint32_t>((entry * 7919U + entry / 7U) % kVocabulary);
    weights[entry] =
        static_cast<float>(static_cast<int>((entry * 29U) % 101U) - 50) /
        128.0F;
  }

  std::vector<float> expected(kBags * kEmbedding, 0.0F);
  for (std::size_t bag = 0; bag < kBags; ++bag)
    for (std::size_t entry = offsets[bag]; entry < offsets[bag + 1]; ++entry)
      for (std::size_t channel = 0; channel < kEmbedding; ++channel)
        expected[bag * kEmbedding + channel] +=
            weights[entry] * table[indices[entry] * kEmbedding + channel];

  std::vector<float> primary(expected.size(), 0.0F);
  std::vector<float> equivalent(expected.size(), 0.0F);
  weighted_embedding_bag_f32(table.data(), indices.data(), weights.data(),
                             offsets.data(), primary.data(), 0, kBags,
                             kEmbedding, kEmbedding, kEmbedding);
  weighted_embedding_bag_f32_equivalent(
      table.data(), indices.data(), weights.data(), offsets.data(),
      equivalent.data(), 0, kBags, kEmbedding, kEmbedding, kEmbedding);
  double primaryAbsolute = 0.0;
  double primaryRelative = 0.0;
  double equivalentAbsolute = 0.0;
  double equivalentRelative = 0.0;
  if (!compare(primary, expected, primaryAbsolute, primaryRelative) ||
      !compare(equivalent, expected, equivalentAbsolute, equivalentRelative)) {
    std::fprintf(stderr,
                 "weighted embedding bag mismatch: primary_abs=%g "
                 "primary_rel=%g equivalent_abs=%g equivalent_rel=%g\n",
                 primaryAbsolute, primaryRelative, equivalentAbsolute,
                 equivalentRelative);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  const double primaryMilliseconds =
      run(weighted_embedding_bag_f32, table, indices, weights, offsets, primary,
          eviction);
  const double equivalentMilliseconds = run(
      weighted_embedding_bag_f32_equivalent, table, indices, weights, offsets,
      equivalent, eviction);
  const double bytes = static_cast<double>(entries) *
                           static_cast<double>(kEmbedding * sizeof(float) +
                                               sizeof(float) +
                                               sizeof(std::uint32_t)) +
                       static_cast<double>(expected.size() * sizeof(float));
  std::printf("kernel=weighted_embedding_bag_f32\n");
  std::printf(
      "model_shape=recommendation_embedding[bags=%zu,entries=%zu,vocab=%zu,dim=%zu]\n",
      kBags, entries, kVocabulary, kEmbedding);
  std::printf("max_absolute_error=%.9g\n", primaryAbsolute);
  std::printf("max_relative_error=%.9g\n", primaryRelative);
  std::printf("primary_median_ms=%.6f\n", primaryMilliseconds);
  std::printf("equivalent_median_ms=%.6f\n", equivalentMilliseconds);
  std::printf("primary_logical_gbs=%.6f\n",
              bytes / primaryMilliseconds / 1.0e6);
  std::printf("equivalent_logical_gbs=%.6f\n",
              bytes / equivalentMilliseconds / 1.0e6);
  return 0;
}
