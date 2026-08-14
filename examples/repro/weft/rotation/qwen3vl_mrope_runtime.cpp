#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kTokens = 128;
constexpr std::size_t kHeads = 32;
constexpr std::size_t kHeadDimension = 128;
constexpr std::size_t kRotaryDimension = 96;
constexpr std::size_t kHalfDimension = kRotaryDimension / 2U;
constexpr std::size_t kSectionT = 15;
constexpr std::size_t kSectionH = 16;
constexpr std::size_t kSectionW = 16;
constexpr std::size_t kSectionE = 1;
constexpr float kThetaScale = 0.865964323F;
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

} // namespace

int main() {
  const std::size_t tokenStride = kHeads * kHeadDimension;
  const std::size_t elements = kTokens * tokenStride;
  std::vector<float> source(elements);
  std::vector<float> output(elements);
  std::vector<float> expected(elements);
  std::vector<std::int32_t> positions(4U * kTokens);
  std::vector<float> angleCache(2U * kHalfDimension);
  for (std::size_t index = 0; index < elements; ++index)
    source[index] =
        static_cast<float>(static_cast<int>((index * 11U) % 257U) - 128) /
        128.0F;
  for (std::size_t token = 0; token < kTokens; ++token) {
    positions[token] = static_cast<std::int32_t>(token);
    positions[kTokens + token] = static_cast<std::int32_t>(token / 8U);
    positions[2U * kTokens + token] = static_cast<std::int32_t>(token % 8U);
    positions[3U * kTokens + token] = static_cast<std::int32_t>(token / 32U);
  }

  const std::size_t sectionTotal =
      kSectionT + kSectionH + kSectionW + kSectionE;
  for (std::size_t token = 0; token < kTokens; ++token) {
    float thetaT = static_cast<float>(positions[token]);
    float thetaH = static_cast<float>(positions[kTokens + token]);
    float thetaW = static_cast<float>(positions[2U * kTokens + token]);
    float thetaE = static_cast<float>(positions[3U * kTokens + token]);
    std::vector<float> cosine(kHalfDimension);
    std::vector<float> sine(kHalfDimension);
    for (std::size_t pair = 0; pair < kHalfDimension; ++pair) {
      const std::size_t sector = pair % sectionTotal;
      float theta = thetaE;
      if (sector % 3U == 0U && sector < 3U * kSectionT)
        theta = thetaT;
      if (sector % 3U == 1U && sector < 3U * kSectionH)
        theta = thetaH;
      if (sector % 3U == 2U && sector < 3U * kSectionW)
        theta = thetaW;
      cosine[pair] = std::cos(theta);
      sine[pair] = std::sin(theta);
      thetaT *= kThetaScale;
      thetaH *= kThetaScale;
      thetaW *= kThetaScale;
      thetaE *= kThetaScale;
    }
    for (std::size_t head = 0; head < kHeads; ++head) {
      const std::size_t base = token * tokenStride + head * kHeadDimension;
      for (std::size_t pair = 0; pair < kHalfDimension; ++pair) {
        const float first = source[base + pair];
        const float second = source[base + kHalfDimension + pair];
        expected[base + pair] = first * cosine[pair] - second * sine[pair];
        expected[base + kHalfDimension + pair] =
            first * sine[pair] + second * cosine[pair];
      }
      for (std::size_t coordinate = kRotaryDimension;
           coordinate < kHeadDimension; ++coordinate)
        expected[base + coordinate] = source[base + coordinate];
    }
  }

  qwen3vl_mrope_f32(
      source.data(), positions.data(), output.data(), angleCache.data(), kTokens,
      kHeads, kHeadDimension, kRotaryDimension, kHeadDimension, tokenStride,
      kSectionT, kSectionH, kSectionW, kSectionE, kThetaScale);
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t index = 0; index < elements; ++index) {
    const double absolute = std::fabs(output[index] - expected[index]);
    const double relative = absolute / std::fmax(1.0, std::fabs(expected[index]));
    maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
    maxRelativeError = std::fmax(maxRelativeError, relative);
  }
  if (maxAbsoluteError > 2.0e-6) {
    std::fprintf(stderr, "qwen3vl_mrope_f32 mismatch: %g\n", maxAbsoluteError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  for (int repetition = 0; repetition < 7; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    qwen3vl_mrope_f32(
        source.data(), positions.data(), output.data(), angleCache.data(),
        kTokens, kHeads, kHeadDimension, kRotaryDimension, kHeadDimension,
        tokenStride, kSectionT, kSectionH, kSectionW, kSectionE, kThetaScale);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=qwen3vl_mrope_f32\n");
  std::printf(
      "model_shape=Qwen3VL[tokens=128,heads=32,D=128,rotary=96,sections=15/16/16/1]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gb_s=%.6f\n",
              2.0 * static_cast<double>(elements * sizeof(float)) /
                  milliseconds / 1.0e6);
  return 0;
}
