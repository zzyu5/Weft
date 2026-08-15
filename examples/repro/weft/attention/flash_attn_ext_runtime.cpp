#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kOuter = 2;
constexpr std::size_t kKeyOuter = 1;
constexpr std::size_t kValueOuter = 2;
constexpr std::size_t kMaskOuter = 1;
constexpr std::size_t kQueryHeads = 32;
constexpr std::size_t kKeyHeads = 8;
constexpr std::size_t kValueHeads = 8;
constexpr std::size_t kMaskHeads = 4;
constexpr std::size_t kQueries = 64;
constexpr std::size_t kKeys = 512;
constexpr std::size_t kKeyDimension = 128;
constexpr std::size_t kValueDimension = 128;
constexpr float kMaximumBias = 8.0F;
constexpr float kSoftcap = 30.0F;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink;

void evict(std::vector<std::uint8_t> &buffer) {
  for (std::size_t index = 0; index < buffer.size(); index += 64) {
    buffer[index] = static_cast<std::uint8_t>(buffer[index] + 1U);
    evictionSink += buffer[index];
  }
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2U];
}

float slopeFor(std::size_t head) {
  std::size_t headPowerOfTwo = 1;
  while (headPowerOfTwo * 2U <= kQueryHeads)
    headPowerOfTwo *= 2U;
  const float base = std::exp(-kMaximumBias * std::log(2.0F) /
                              static_cast<float>(headPowerOfTwo));
  const float extraBase =
      std::exp(-(kMaximumBias / 2.0F) * std::log(2.0F) /
               static_cast<float>(headPowerOfTwo));
  return head < headPowerOfTwo
             ? std::pow(base, static_cast<float>(head + 1U))
             : std::pow(extraBase,
                        static_cast<float>(2U * (head - headPowerOfTwo) + 1U));
}

void referenceRow(const std::vector<float> &query,
                  const std::vector<_Float16> &key,
                  const std::vector<_Float16> &value,
                  const std::vector<_Float16> &mask,
                  const std::vector<float> &sinks, std::size_t outer,
                  std::size_t head, std::size_t queryIndex,
                  std::vector<float> &result) {
  const std::size_t keyHead = head / (kQueryHeads / kKeyHeads);
  const std::size_t valueHead = head / (kQueryHeads / kValueHeads);
  const std::size_t maskHead = head % kMaskHeads;
  const std::size_t keyOuter = outer / (kOuter / kKeyOuter);
  const std::size_t valueOuter = outer / (kOuter / kValueOuter);
  const std::size_t maskOuter = outer % kMaskOuter;
  const std::size_t queryBase =
      ((outer * kQueryHeads + head) * kQueries + queryIndex) * kKeyDimension;
  const float scale = 1.0F / std::sqrt(static_cast<float>(kKeyDimension));
  const float slope = slopeFor(head);
  float maximum = -INFINITY;
  float total = 0.0F;
  std::fill(result.begin(), result.end(), 0.0F);
  for (std::size_t keyIndex = 0; keyIndex < kKeys; ++keyIndex) {
    const float maskValue = static_cast<float>(
        mask[((maskOuter * kMaskHeads + maskHead) * kQueries + queryIndex) *
                 kKeys +
             keyIndex]);
    if (std::isinf(maskValue) && maskValue < 0.0F)
      continue;
    const std::size_t keyBase =
        ((keyOuter * kKeyHeads + keyHead) * kKeys + keyIndex) * kKeyDimension;
    float dot = 0.0F;
    for (std::size_t dimension = 0; dimension < kKeyDimension; ++dimension)
      dot += query[queryBase + dimension] *
             static_cast<float>(key[keyBase + dimension]);
    float score = kSoftcap * std::tanh(dot * scale / kSoftcap);
    score += maskValue * slope;
    const float nextMaximum = std::max(maximum, score);
    const float oldWeight = std::exp(maximum - nextMaximum);
    const float newWeight = std::exp(score - nextMaximum);
    const std::size_t valueBase =
        ((valueOuter * kValueHeads + valueHead) * kKeys + keyIndex) *
        kValueDimension;
    for (std::size_t dimension = 0; dimension < kValueDimension; ++dimension)
      result[dimension] =
          result[dimension] * oldWeight +
          static_cast<float>(value[valueBase + dimension]) * newWeight;
    total = total * oldWeight + newWeight;
    maximum = nextMaximum;
  }
  const float sink = sinks[head];
  const float nextMaximum = std::max(maximum, sink);
  const float oldWeight = std::exp(maximum - nextMaximum);
  const float sinkWeight = std::exp(sink - nextMaximum);
  for (float &element : result)
    element *= oldWeight;
  total = total * oldWeight + sinkWeight;
  for (float &element : result)
    element /= total;
}

} // namespace

int main() {
  const std::size_t queryElements =
      kOuter * kQueryHeads * kQueries * kKeyDimension;
  const std::size_t keyElements =
      kKeyOuter * kKeyHeads * kKeys * kKeyDimension;
  const std::size_t valueElements =
      kValueOuter * kValueHeads * kKeys * kValueDimension;
  const std::size_t maskElements =
      kMaskOuter * kMaskHeads * kQueries * kKeys;
  const std::size_t outputElements =
      kOuter * kQueries * kQueryHeads * kValueDimension;
  std::vector<float> query(queryElements);
  std::vector<_Float16> key(keyElements);
  std::vector<_Float16> value(valueElements);
  std::vector<_Float16> mask(maskElements);
  std::vector<float> sinks(kQueryHeads);
  std::vector<float> output(outputElements);
  std::vector<float> queryScratch(
      flash_attn_ext_f32_f16__query_scratch_elements(kKeyDimension));
  std::vector<float> accumulatorScratch(
      flash_attn_ext_f32_f16__accumulator_scratch_elements(kValueDimension));
  for (std::size_t index = 0; index < query.size(); ++index)
    query[index] =
        static_cast<float>(static_cast<int>((index * 7U) % 61U) - 30) /
        64.0F;
  for (std::size_t index = 0; index < key.size(); ++index)
    key[index] = static_cast<_Float16>(
        static_cast<float>(static_cast<int>((index * 11U) % 47U) - 23) /
        64.0F);
  for (std::size_t index = 0; index < value.size(); ++index)
    value[index] = static_cast<_Float16>(
        static_cast<float>(static_cast<int>((index * 13U) % 53U) - 26) /
        32.0F);
  for (std::size_t outer = 0; outer < kMaskOuter; ++outer)
    for (std::size_t head = 0; head < kMaskHeads; ++head)
      for (std::size_t queryIndex = 0; queryIndex < kQueries; ++queryIndex)
        for (std::size_t keyIndex = 0; keyIndex < kKeys; ++keyIndex) {
          const bool active = keyIndex <= queryIndex + 448U;
          const float bias = active
                                 ? -static_cast<float>(keyIndex % 17U) / 32.0F
                                 : -INFINITY;
          mask[((outer * kMaskHeads + head) * kQueries + queryIndex) * kKeys +
               keyIndex] = static_cast<_Float16>(bias);
        }
  for (std::size_t head = 0; head < kQueryHeads; ++head)
    sinks[head] = -0.5F + static_cast<float>(head % 7U) / 16.0F;

  const float scale = 1.0F / std::sqrt(static_cast<float>(kKeyDimension));
  const std::size_t queryRowStride = kKeyDimension;
  const std::size_t queryHeadStride = kQueries * queryRowStride;
  const std::size_t queryOuterStride = kQueryHeads * queryHeadStride;
  const std::size_t keyRowStride = kKeyDimension;
  const std::size_t keyHeadStride = kKeys * keyRowStride;
  const std::size_t keyOuterStride = kKeyHeads * keyHeadStride;
  const std::size_t valueRowStride = kValueDimension;
  const std::size_t valueHeadStride = kKeys * valueRowStride;
  const std::size_t valueOuterStride = kValueHeads * valueHeadStride;
  const std::size_t maskRowStride = kKeys;
  const std::size_t maskHeadStride = kQueries * maskRowStride;
  const std::size_t maskOuterStride = kMaskHeads * maskHeadStride;
  const std::size_t outputHeadStride = kValueDimension;
  const std::size_t outputQueryStride = kQueryHeads * outputHeadStride;
  const std::size_t outputOuterStride = kQueries * outputQueryStride;

  flash_attn_ext_f32_f16(
      query.data(), key.data(), value.data(), mask.data(), sinks.data(),
      output.data(), queryScratch.data(), accumulatorScratch.data(), 0, kOuter,
      kOuter, kKeyOuter, kValueOuter, kMaskOuter, kQueryHeads, kKeyHeads,
      kValueHeads, kMaskHeads, kQueries, kKeys, kKeyDimension, kValueDimension,
      queryOuterStride, queryHeadStride, queryRowStride, keyOuterStride,
      keyHeadStride, keyRowStride, valueOuterStride, valueHeadStride,
      valueRowStride, maskOuterStride, maskHeadStride, maskRowStride,
      outputOuterStride, outputQueryStride, outputHeadStride, scale,
      kMaximumBias, kSoftcap);

  const std::size_t sampleOuter[] = {0, kOuter - 1};
  const std::size_t sampleHeads[] = {0, kQueryHeads / 2, kQueryHeads - 1};
  const std::size_t sampleQueries[] = {0, kQueries / 2, kQueries - 1};
  const std::size_t sampleDimensions[] = {0, 1, kValueDimension / 2,
                                          kValueDimension - 1};
  std::vector<float> reference(kValueDimension);
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t outer : sampleOuter)
    for (std::size_t head : sampleHeads)
      for (std::size_t queryIndex : sampleQueries) {
        referenceRow(query, key, value, mask, sinks, outer, head, queryIndex,
                     reference);
        const std::size_t outputBase =
            outer * outputOuterStride + queryIndex * outputQueryStride +
            head * outputHeadStride;
        for (std::size_t dimension : sampleDimensions) {
          const double absolute =
              std::fabs(output[outputBase + dimension] - reference[dimension]);
          const double relative =
              absolute / std::fmax(1.0, std::fabs(reference[dimension]));
          maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
          maxRelativeError = std::fmax(maxRelativeError, relative);
        }
      }
  if (maxRelativeError > 5.0e-4) {
    std::fprintf(stderr, "flash_attn_ext_f32_f16 mismatch: abs=%g rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  for (int repetition = 0; repetition < 3; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    flash_attn_ext_f32_f16(
        query.data(), key.data(), value.data(), mask.data(), sinks.data(),
        output.data(), queryScratch.data(), accumulatorScratch.data(), 0,
        kOuter, kOuter, kKeyOuter, kValueOuter, kMaskOuter, kQueryHeads,
        kKeyHeads, kValueHeads, kMaskHeads, kQueries, kKeys, kKeyDimension,
        kValueDimension, queryOuterStride, queryHeadStride, queryRowStride,
        keyOuterStride, keyHeadStride, keyRowStride, valueOuterStride,
        valueHeadStride, valueRowStride, maskOuterStride, maskHeadStride,
        maskRowStride, outputOuterStride, outputQueryStride, outputHeadStride,
        scale, kMaximumBias, kSoftcap);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double operations = 2.0 * static_cast<double>(
      kOuter * kQueryHeads * kQueries * kKeys *
      (kKeyDimension + kValueDimension));
  std::printf("kernel=flash_attn_ext_f32_f16\n");
  std::printf(
      "model_shape=GQA[B=2,Q=64,KV=512,H=32,Hkv=8,Dk=128,Dv=128,softcap=30,sinks]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / milliseconds / 1.0e6);
  return 0;
}
