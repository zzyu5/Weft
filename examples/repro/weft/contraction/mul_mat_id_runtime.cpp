#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kExperts = 8;
constexpr std::size_t kTokens = 64;
constexpr std::size_t kSlots = 2;
constexpr std::size_t kRows = 4096;
constexpr std::size_t kInner = 4096;
constexpr std::size_t kWeightElements = kExperts * kRows * kInner;
constexpr std::size_t kActivationElements = kTokens * kSlots * kInner;
constexpr std::size_t kOutputElements = kTokens * kSlots * kRows;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
constexpr float kValues[] = {1.0F, 0.5F, -1.0F, 2.0F};
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

float reference(const std::vector<float> &weights,
                const std::vector<float> &activations,
                const std::vector<std::uint32_t> &ids, std::size_t token,
                std::size_t slot, std::size_t row) {
  const std::size_t expert = ids[token * kSlots + slot];
  const float *weight =
      weights.data() + (expert * kRows + row) * kInner;
  const float *activation =
      activations.data() + (token * kSlots + slot) * kInner;
  float value = 0.0F;
  for (std::size_t inner = 0; inner < kInner; ++inner)
    value += weight[inner] * activation[inner];
  return value;
}

} // namespace

int main() {
  std::vector<float> weights(kWeightElements);
  std::vector<float> activations(kActivationElements);
  std::vector<std::uint32_t> ids(kTokens * kSlots);
  std::vector<float> output(kOutputElements);
  std::vector<std::uint32_t> expertCounts(kExperts);
  std::vector<std::uint32_t> expertOffsets(kExperts + 1U);
  std::vector<std::uint32_t> expertCursors(kExperts);
  std::vector<std::uint32_t> expertItems(kTokens * kSlots);
  for (std::size_t expert = 0; expert < kExperts; ++expert)
    for (std::size_t row = 0; row < kRows; ++row)
      for (std::size_t inner = 0; inner < kInner; ++inner)
        weights[(expert * kRows + row) * kInner + inner] =
            kValues[(expert + row + 3U * inner + 1U) % 4U];
  for (std::size_t token = 0; token < kTokens; ++token)
    for (std::size_t slot = 0; slot < kSlots; ++slot) {
      ids[token * kSlots + slot] =
          static_cast<std::uint32_t>((token * 5U + slot * 3U) % kExperts);
      for (std::size_t inner = 0; inner < kInner; ++inner)
        activations[(token * kSlots + slot) * kInner + inner] =
            kValues[(token + 5U * slot + 3U * inner + 2U) % 4U];
    }

  mul_mat_id_f32(
      weights.data(), activations.data(), ids.data(), output.data(),
      expertCounts.data(), expertOffsets.data(), expertCursors.data(),
      expertItems.data(), kExperts, kTokens, kSlots, kRows, kInner,
      kRows * kInner, kInner, kInner, kSlots * kInner, kSlots, kRows,
      kSlots * kRows);
  const std::size_t sampleTokens[] = {0, kTokens / 2, kTokens - 1};
  const std::size_t sampleSlots[] = {0, kSlots - 1};
  const std::size_t sampleRows[] = {0, 1, kRows / 2, kRows - 1};
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t token : sampleTokens)
    for (std::size_t slot : sampleSlots)
      for (std::size_t row : sampleRows) {
        const double expected =
            reference(weights, activations, ids, token, slot, row);
        const double actual =
            output[(token * kSlots + slot) * kRows + row];
        const double absolute = std::fabs(actual - expected);
        const double relative = absolute / std::fmax(1.0, std::fabs(expected));
        maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
        maxRelativeError = std::fmax(maxRelativeError, relative);
      }
  if (maxRelativeError > 1.0e-5) {
    std::fprintf(stderr, "mul_mat_id_f32 mismatch: abs=%g rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(3);
  for (int repetition = 0; repetition < 3; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    mul_mat_id_f32(
        weights.data(), activations.data(), ids.data(), output.data(),
        expertCounts.data(), expertOffsets.data(), expertCursors.data(),
        expertItems.data(), kExperts, kTokens, kSlots, kRows, kInner,
        kRows * kInner, kInner, kInner, kSlots * kInner, kSlots, kRows,
        kSlots * kRows);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double operations =
      2.0 * static_cast<double>(kTokens * kSlots * kRows * kInner);
  std::printf("kernel=mul_mat_id_f32\n");
  std::printf("model_shape=moe_indexed_contract[tokens=64,slots=2,experts=8,N=4096,K=4096]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / milliseconds / 1.0e6);
  return 0;
}
