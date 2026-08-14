#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr std::size_t kTokens = 128;
constexpr std::size_t kHeads = 32;
constexpr std::size_t kHeadDimension = 128;
constexpr std::size_t kHalfDimension = kHeadDimension / 2;
constexpr std::size_t kTokenStride = kHeads * kHeadDimension;
constexpr std::size_t kElements = kTokens * kTokenStride;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink;

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

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <repetitions>\n", argv[0]);
    return 2;
  }
  const std::size_t repeatCount = repetitions(argv[1]);
  if (repeatCount == 0) {
    std::fprintf(stderr, "invalid repetitions\n");
    return 2;
  }

  std::vector<float> source(kElements);
  std::vector<float> destination(kElements, 0.0F);
  std::vector<float> reference(kElements, 0.0F);
  std::vector<float> angleCache(kHeadDimension, 0.0F);
  std::vector<std::int32_t> positions(kTokens);
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] = static_cast<float>(static_cast<int>(index % 31) - 15) /
                    16.0F;
  for (std::size_t token = 0; token < kTokens; ++token)
    positions[token] = static_cast<std::int32_t>(token);

  const float thetaScale =
      std::pow(10000.0F, -2.0F / static_cast<float>(kHeadDimension));
  rope_neox_f32(source.data(), positions.data(), destination.data(),
                angleCache.data(), kTokens, kHeads, kHalfDimension,
                kHeadDimension, kTokenStride, thetaScale);

  for (std::size_t token = 0; token < kTokens; ++token) {
    float theta = static_cast<float>(positions[token]);
    for (std::size_t pair = 0; pair < kHalfDimension; ++pair) {
      const float cosine = std::cos(theta);
      const float sine = std::sin(theta);
      for (std::size_t head = 0; head < kHeads; ++head) {
        const std::size_t first = token * kTokenStride +
                                  head * kHeadDimension + pair;
        const std::size_t second = first + kHalfDimension;
        reference[first] = source[first] * cosine - source[second] * sine;
        reference[second] = source[first] * sine + source[second] * cosine;
      }
      theta *= thetaScale;
    }
  }

  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t index = 0; index < kElements; ++index) {
    const double absolute = std::abs(
        static_cast<double>(destination[index]) - reference[index]);
    const double relative =
        absolute / std::max(1.0, std::abs(static_cast<double>(reference[index])));
    maxAbsoluteError = std::max(maxAbsoluteError, absolute);
    maxRelativeError = std::max(maxRelativeError, relative);
  }
  if (maxAbsoluteError > 2.0e-6 && maxRelativeError > 2.0e-6) {
    std::fprintf(stderr, "RoPE mismatch: max_abs=%g max_rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(repeatCount);
  for (std::size_t repetition = 0; repetition < repeatCount; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    rope_neox_f32(source.data(), positions.data(), destination.data(),
                  angleCache.data(), kTokens, kHeads, kHalfDimension,
                  kHeadDimension, kTokenStride, thetaScale);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  std::printf("kernel=rope_neox_f32\n");
  std::printf("model_shape=q[head_dim=128,heads=32,tokens=128]\n");
  std::printf("timing=cold-cache-64MiB-eviction-median\n");
  std::printf("repetitions=%zu\n", repeatCount);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("weft_ms=%.6f\n", milliseconds);
  return 0;
}
