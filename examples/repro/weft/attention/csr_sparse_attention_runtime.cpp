#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t kRows = 2048;
constexpr std::size_t kKeys = 8192;
constexpr std::size_t kDimension = 128;
constexpr float kScale = 0.08838834764831845F;
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
  std::vector<std::uint32_t> rowOffsets(kRows + 1, 0);
  for (std::size_t row = 0; row < kRows; ++row)
    rowOffsets[row + 1] = rowOffsets[row] +
                          static_cast<std::uint32_t>(24U + (row * 17U) % 49U);
  const std::size_t edges = rowOffsets.back();
  std::vector<std::uint32_t> keyIndices(edges);
  for (std::size_t edge = 0; edge < edges; ++edge)
    keyIndices[edge] =
        static_cast<std::uint32_t>((edge * 7919U + edge / 31U) % kKeys);

  std::vector<float> query(kRows * kDimension);
  std::vector<float> key(kKeys * kDimension);
  std::vector<float> value(kKeys * kDimension);
  for (std::size_t index = 0; index < query.size(); ++index)
    query[index] =
        static_cast<float>(static_cast<int>((index * 13U) % 101U) - 50) /
        256.0F;
  for (std::size_t index = 0; index < key.size(); ++index) {
    key[index] =
        static_cast<float>(static_cast<int>((index * 19U) % 107U) - 53) /
        256.0F;
    value[index] =
        static_cast<float>(static_cast<int>((index * 23U) % 113U) - 56) /
        128.0F;
  }

  std::vector<float> expected(kRows * kDimension, 0.0F);
  std::vector<float> scores(73);
  for (std::size_t row = 0; row < kRows; ++row) {
    const std::size_t begin = rowOffsets[row];
    const std::size_t end = rowOffsets[row + 1];
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t edge = begin; edge < end; ++edge) {
      float score = 0.0F;
      const std::size_t keyRow = keyIndices[edge] * kDimension;
      for (std::size_t dimension = 0; dimension < kDimension; ++dimension)
        score += query[row * kDimension + dimension] *
                 key[keyRow + dimension];
      score *= kScale;
      scores[edge - begin] = score;
      maximum = std::max(maximum, score);
    }
    float total = 0.0F;
    for (std::size_t edge = begin; edge < end; ++edge) {
      const float weight = std::exp(scores[edge - begin] - maximum);
      total += weight;
      const std::size_t keyRow = keyIndices[edge] * kDimension;
      for (std::size_t dimension = 0; dimension < kDimension; ++dimension)
        expected[row * kDimension + dimension] +=
            weight * value[keyRow + dimension];
    }
    for (std::size_t dimension = 0; dimension < kDimension; ++dimension)
      expected[row * kDimension + dimension] /= total;
  }

  std::vector<float> output(expected.size(), 0.0F);
  std::vector<float> scratch(
      csr_sparse_attention_f32__accumulator_scratch_elements(kDimension),
      0.0F);
  csr_sparse_attention_f32(
      query.data(), key.data(), value.data(), rowOffsets.data(),
      keyIndices.data(), output.data(), scratch.data(), 0, kRows, kDimension,
      kScale);
  double maxAbsolute = 0.0;
  double maxRelative = 0.0;
  for (std::size_t index = 0; index < output.size(); ++index) {
    const double error = std::abs(static_cast<double>(output[index]) -
                                  static_cast<double>(expected[index]));
    maxAbsolute = std::max(maxAbsolute, error);
    maxRelative = std::max(
        maxRelative,
        error / std::max(std::abs(static_cast<double>(expected[index])), 1.0e-6));
  }
  if (maxAbsolute > 3.0e-4 && maxRelative > 3.0e-3) {
    std::fprintf(stderr, "csr sparse attention mismatch: abs=%g rel=%g\n",
                 maxAbsolute, maxRelative);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(5);
  for (int repetition = 0; repetition < 5; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    csr_sparse_attention_f32(
        query.data(), key.data(), value.data(), rowOffsets.data(),
        keyIndices.data(), output.data(), scratch.data(), 0, kRows, kDimension,
        kScale);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double operations =
      static_cast<double>(edges) * static_cast<double>(kDimension) * 4.0;
  std::printf("kernel=csr_sparse_attention_f32\n");
  std::printf("model_shape=sparse_attention[rows=%zu,keys=%zu,edges=%zu,dim=%zu]\n",
              kRows, kKeys, edges, kDimension);
  std::printf("max_absolute_error=%.9g\n", maxAbsolute);
  std::printf("max_relative_error=%.9g\n", maxRelative);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / milliseconds / 1.0e6);
  return 0;
}
