#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr std::size_t kQueryHeads = 32;
constexpr std::size_t kKeyValueHeads = 8;
constexpr std::size_t kQueries = 128;
constexpr std::size_t kKeys = 128;
constexpr std::size_t kHeadDimension = 128;
constexpr std::size_t kGroupSize = kQueryHeads / kKeyValueHeads;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
constexpr std::uint16_t kHalfValues[] = {
    0x3000U,
    0x2c00U,
    0xb000U,
    0x3400U,
};
constexpr float kFloatValues[] = {0.125F, 0.0625F, -0.125F, 0.25F};
constexpr std::uint16_t kHalfZero = 0x0000U;
constexpr std::uint16_t kHalfNegativeInfinity = 0xfc00U;
volatile std::uint64_t evictionSink;

_Float16 halfFromBits(std::uint16_t bits) {
  static_assert(sizeof(_Float16) == sizeof(bits));
  _Float16 value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

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

float referenceElement(std::size_t queryHead, std::size_t queryIndex,
                       std::size_t outputDimension) {
  const std::size_t keyValueHead = queryHead / kGroupSize;
  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDimension));
  float maximum = -INFINITY;
  float total = 0.0F;
  float accumulated = 0.0F;
  for (std::size_t keyIndex = 0; keyIndex <= queryIndex; ++keyIndex) {
    float dot = 0.0F;
    for (std::size_t dimension = 0; dimension < kHeadDimension; ++dimension) {
      const float queryElement =
          kFloatValues[(queryHead + queryIndex + 3 * dimension + 1) % 4];
      const float keyElement =
          kFloatValues[(keyValueHead + 5 * keyIndex + dimension + 2) % 4];
      dot += queryElement * keyElement;
    }
    const float score = dot * scale;
    const float nextMaximum = std::max(maximum, score);
    const float oldWeight = std::exp(maximum - nextMaximum);
    const float newWeight = std::exp(score - nextMaximum);
    const float valueElement =
        kFloatValues[(keyValueHead + 7 * keyIndex +
                      3 * outputDimension + 3) % 4];
    accumulated = accumulated * oldWeight + valueElement * newWeight;
    total = total * oldWeight + newWeight;
    maximum = nextMaximum;
  }
  return accumulated / total;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <repetitions>\n", argv[0]);
    return 2;
  }
  const std::size_t repeatCount = repetitions(argv[1]);
  if (repeatCount == 0) {
    std::fprintf(stderr, "invalid repetition count\n");
    return 2;
  }

  const std::size_t queryElements =
      kQueryHeads * kQueries * kHeadDimension;
  const std::size_t keyValueElements =
      kKeyValueHeads * kKeys * kHeadDimension;
  std::vector<float> query(queryElements);
  std::vector<_Float16> key(keyValueElements);
  std::vector<_Float16> value(keyValueElements);
  std::vector<_Float16> mask(kQueries * kKeys);
  std::vector<float> output(queryElements, 0.0F);
  std::vector<_Float16> queryScratch(
      online_flash_attention_f32_f16__query_scratch_elements(kHeadDimension));
  std::vector<_Float16> accumulatorScratch(
      online_flash_attention_f32_f16__accumulator_scratch_elements(
          kHeadDimension));

  for (std::size_t head = 0; head < kQueryHeads; ++head)
    for (std::size_t row = 0; row < kQueries; ++row)
      for (std::size_t dimension = 0; dimension < kHeadDimension; ++dimension)
        query[(head * kQueries + row) * kHeadDimension + dimension] =
            kFloatValues[(head + row + 3 * dimension + 1) % 4];
  for (std::size_t head = 0; head < kKeyValueHeads; ++head)
    for (std::size_t row = 0; row < kKeys; ++row)
      for (std::size_t dimension = 0; dimension < kHeadDimension; ++dimension) {
        const std::size_t offset =
            (head * kKeys + row) * kHeadDimension + dimension;
        key[offset] = halfFromBits(
            kHalfValues[(head + 5 * row + dimension + 2) % 4]);
        value[offset] = halfFromBits(
            kHalfValues[(head + 7 * row + 3 * dimension + 3) % 4]);
      }
  for (std::size_t queryIndex = 0; queryIndex < kQueries; ++queryIndex)
    for (std::size_t keyIndex = 0; keyIndex < kKeys; ++keyIndex)
      mask[queryIndex * kKeys + keyIndex] = halfFromBits(
          keyIndex <= queryIndex ? kHalfZero : kHalfNegativeInfinity);

  const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDimension));
  online_flash_attention_f32_f16(
      query.data(), key.data(), value.data(), mask.data(), output.data(),
      queryScratch.data(), accumulatorScratch.data(), 0, kQueryHeads, kQueries,
      kKeys, kHeadDimension, kGroupSize, scale);

  const std::size_t sampleHeads[] = {0, kQueryHeads / 2, kQueryHeads - 1};
  const std::size_t sampleQueries[] = {0, 1, kQueries / 2, kQueries - 1};
  const std::size_t sampleDimensions[] = {0, 1, kHeadDimension / 2,
                                          kHeadDimension - 1};
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t head : sampleHeads) {
    for (std::size_t queryIndex : sampleQueries) {
      for (std::size_t dimension : sampleDimensions) {
        const double expected = referenceElement(head, queryIndex, dimension);
        const double actual =
            output[(head * kQueries + queryIndex) * kHeadDimension + dimension];
        const double absolute = std::fabs(actual - expected);
        const double relative = absolute / std::max(std::fabs(expected), 1.0e-7);
        maxAbsoluteError = std::max(maxAbsoluteError, absolute);
        maxRelativeError = std::max(maxRelativeError, relative);
        if (absolute > 1.0e-3 && relative > 2.0e-2) {
          std::fprintf(stderr,
                       "mismatch at (%zu,%zu,%zu): weft=%g reference=%g "
                       "absolute=%g relative=%g\n",
                       head, queryIndex, dimension, actual, expected, absolute,
                       relative);
          return 1;
        }
      }
    }
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(repeatCount);
  for (std::size_t repetition = 0; repetition < repeatCount; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    online_flash_attention_f32_f16(
        query.data(), key.data(), value.data(), mask.data(), output.data(),
        queryScratch.data(), accumulatorScratch.data(), 0, kQueryHeads,
        kQueries, kKeys, kHeadDimension, kGroupSize, scale);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  std::printf("kernel=flash_attention_f32_f16\n");
  std::printf("model_shape=Llama-8B.attention\n");
  std::printf("Q=%zu\nKV=%zu\nH=%zu\nHkv=%zu\nD=%zu\n", kQueries, kKeys,
              kQueryHeads, kKeyValueHeads, kHeadDimension);
  std::printf("timing=cold-cache-64MiB-eviction-median\n");
  std::printf("repetitions=%zu\n", repeatCount);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("weft_ms=%.6f\n", milliseconds);
  return 0;
}
