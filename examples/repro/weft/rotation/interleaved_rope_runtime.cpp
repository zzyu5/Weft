#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <riscv_vector.h>
#include <vector>

namespace {
constexpr std::size_t kTokens = 128;
constexpr std::size_t kHeads = 32;
constexpr std::size_t kPairs = 64;
constexpr std::size_t kHeadStride = 2 * kPairs;
constexpr std::size_t kTokenStride = kHeads * kHeadStride;
constexpr std::size_t kElements = kTokens * kTokenStride;
constexpr std::size_t kEvictionBytes = 64U << 20U;
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
using Kernel = void (*)(const float *, const float *, float *, std::size_t,
                        std::size_t, std::size_t, std::size_t, std::size_t);
void interleaved_rope_f32_strided_baseline(
    const float *source, const float *angles, float *destination,
    std::size_t tokens, std::size_t heads, std::size_t pairs,
    std::size_t tokenStride, std::size_t headStride) {
  constexpr std::ptrdiff_t kStride = 2 * sizeof(float);
  for (std::size_t token = 0; token < tokens; ++token) {
    const float *tokenSource = source + token * tokenStride;
    float *tokenDestination = destination + token * tokenStride;
    const float *tokenAngles = angles + 2 * token * pairs;
    for (std::size_t head = 0; head < heads; ++head) {
      const float *headSource = tokenSource + head * headStride;
      float *headDestination = tokenDestination + head * headStride;
      for (std::size_t offset = 0; offset < pairs;) {
        const std::size_t vl = __riscv_vsetvl_e32m4(pairs - offset);
        const vfloat32m4_t real = __riscv_vlse32_v_f32m4(
            headSource + 2 * offset, kStride, vl);
        const vfloat32m4_t imag = __riscv_vlse32_v_f32m4(
            headSource + 2 * offset + 1, kStride, vl);
        const vfloat32m4_t cosine = __riscv_vlse32_v_f32m4(
            tokenAngles + 2 * offset, kStride, vl);
        const vfloat32m4_t sine = __riscv_vlse32_v_f32m4(
            tokenAngles + 2 * offset + 1, kStride, vl);
        const vfloat32m4_t rotatedReal = __riscv_vfsub_vv_f32m4(
            __riscv_vfmul_vv_f32m4(real, cosine, vl),
            __riscv_vfmul_vv_f32m4(imag, sine, vl), vl);
        const vfloat32m4_t rotatedImag = __riscv_vfadd_vv_f32m4(
            __riscv_vfmul_vv_f32m4(real, sine, vl),
            __riscv_vfmul_vv_f32m4(imag, cosine, vl), vl);
        __riscv_vsse32_v_f32m4(headDestination + 2 * offset, kStride,
                               rotatedReal, vl);
        __riscv_vsse32_v_f32m4(headDestination + 2 * offset + 1, kStride,
                               rotatedImag, vl);
        offset += vl;
      }
    }
  }
}
double run(Kernel kernel, const std::vector<float> &source,
           const std::vector<float> &angles, std::vector<float> &output,
           std::vector<std::uint8_t> &eviction) {
  std::vector<double> samples;
  for (int repetition = 0; repetition < 7; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    kernel(source.data(), angles.data(), output.data(), kTokens, kHeads, kPairs,
           kTokenStride, kHeadStride);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  return median(samples);
}
} // namespace

int main() {
  std::vector<float> source(kElements);
  std::vector<float> angles(2 * kTokens * kPairs);
  std::vector<float> output(kElements);
  std::vector<float> baseline(kElements);
  std::vector<float> reference(kElements);
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] = static_cast<float>(static_cast<int>((index * 13) % 257) - 128) /
                    256.0F;
  for (std::size_t token = 0; token < kTokens; ++token)
    for (std::size_t pair = 0; pair < kPairs; ++pair) {
      const float angle = static_cast<float>(token) *
                          std::pow(10000.0F, -static_cast<float>(pair) / kPairs);
      angles[2 * (token * kPairs + pair)] = std::cos(angle);
      angles[2 * (token * kPairs + pair) + 1] = std::sin(angle);
    }
  for (std::size_t token = 0; token < kTokens; ++token)
    for (std::size_t head = 0; head < kHeads; ++head)
      for (std::size_t pair = 0; pair < kPairs; ++pair) {
        const std::size_t base = token * kTokenStride + head * kHeadStride + 2 * pair;
        const float c = angles[2 * (token * kPairs + pair)];
        const float s = angles[2 * (token * kPairs + pair) + 1];
        reference[base] = source[base] * c - source[base + 1] * s;
        reference[base + 1] = source[base] * s + source[base + 1] * c;
      }
  interleaved_rope_f32(source.data(), angles.data(), output.data(), kTokens,
                       kHeads, kPairs, kTokenStride, kHeadStride);
  interleaved_rope_f32_strided_baseline(
      source.data(), angles.data(), baseline.data(), kTokens, kHeads, kPairs,
      kTokenStride, kHeadStride);
  double maximum = 0.0;
  double baselineMaximum = 0.0;
  for (std::size_t index = 0; index < output.size(); ++index) {
    maximum = std::max(maximum, std::abs(static_cast<double>(output[index]) -
                                         reference[index]));
    baselineMaximum =
        std::max(baselineMaximum,
                 std::abs(static_cast<double>(baseline[index]) -
                          reference[index]));
  }
  if (maximum > 1.0e-6 || baselineMaximum > 1.0e-6) {
    std::fprintf(stderr, "interleaved rope mismatch: %g %g\n", maximum,
                 baselineMaximum);
    return 1;
  }
  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  const double milliseconds =
      run(interleaved_rope_f32, source, angles, output, eviction);
  const double baselineMilliseconds = run(
      interleaved_rope_f32_strided_baseline, source, angles, baseline, eviction);
  const double bytes = static_cast<double>(kElements) * 3.0 * sizeof(float);
  std::printf("kernel=interleaved_rope_f32\n");
  std::printf("model_shape=rope[tokens=%zu;heads=%zu;D=%zu]\n", kTokens, kHeads,
              2 * kPairs);
  std::printf("max_absolute_error=%.9g\n", maximum);
  std::printf("strided_baseline_max_absolute_error=%.9g\n", baselineMaximum);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("strided_baseline_median_ms=%.6f\n", baselineMilliseconds);
  std::printf("logical_gbs=%.6f\n", bytes / milliseconds / 1.0e6);
  std::printf("strided_baseline_logical_gbs=%.6f\n",
              bytes / baselineMilliseconds / 1.0e6);
  return 0;
}
