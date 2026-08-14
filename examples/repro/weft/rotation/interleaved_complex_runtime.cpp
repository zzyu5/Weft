#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void interleaved_complex_mul_f32(const float *, const float *,
                                               float *, std::size_t);
extern "C" void interleaved_complex_mul_f32_equivalent(
    const float *, const float *, float *, std::size_t);

namespace {
constexpr std::size_t kPairs = 1U << 20U;
constexpr std::size_t kEvictionBytes = 64U << 20U;
volatile std::uint64_t evictionSink = 0;
using Kernel = void (*)(const float *, const float *, float *, std::size_t);

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

double run(Kernel kernel, const std::vector<float> &lhs,
           const std::vector<float> &rhs, std::vector<float> &output,
           std::vector<std::uint8_t> &eviction) {
  std::vector<double> samples;
  for (int repetition = 0; repetition < 7; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    kernel(lhs.data(), rhs.data(), output.data(), kPairs);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  return median(samples);
}
} // namespace

int main() {
  std::vector<float> lhs(2 * kPairs);
  std::vector<float> rhs(2 * kPairs);
  std::vector<float> reference(2 * kPairs);
  std::vector<float> primary(2 * kPairs);
  std::vector<float> equivalent(2 * kPairs);
  for (std::size_t index = 0; index < 2 * kPairs; ++index) {
    lhs[index] = static_cast<float>(static_cast<int>((index * 17) % 251) - 125) /
                 128.0F;
    rhs[index] = static_cast<float>(static_cast<int>((index * 29) % 257) - 128) /
                 256.0F;
  }
  for (std::size_t pair = 0; pair < kPairs; ++pair) {
    const float ar = lhs[2 * pair];
    const float ai = lhs[2 * pair + 1];
    const float br = rhs[2 * pair];
    const float bi = rhs[2 * pair + 1];
    reference[2 * pair] = ar * br - ai * bi;
    reference[2 * pair + 1] = ar * bi + ai * br;
  }
  interleaved_complex_mul_f32(lhs.data(), rhs.data(), primary.data(), kPairs);
  interleaved_complex_mul_f32_equivalent(lhs.data(), rhs.data(),
                                         equivalent.data(), kPairs);
  double maximum = 0.0;
  double equivalentMaximum = 0.0;
  for (std::size_t index = 0; index < reference.size(); ++index) {
    maximum = std::max(maximum, std::abs(static_cast<double>(primary[index]) -
                                         reference[index]));
    equivalentMaximum =
        std::max(equivalentMaximum,
                 std::abs(static_cast<double>(equivalent[index]) -
                          reference[index]));
  }
  if (maximum > 1.0e-6 || equivalentMaximum > 1.0e-6) {
    std::fprintf(stderr, "interleaved complex mismatch: %g %g\n", maximum,
                 equivalentMaximum);
    return 1;
  }
  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  const double milliseconds = run(interleaved_complex_mul_f32, lhs, rhs, primary,
                                  eviction);
  const double equivalentMilliseconds = run(
      interleaved_complex_mul_f32_equivalent, lhs, rhs, equivalent, eviction);
  const double bytes = static_cast<double>(kPairs) * 6.0 * sizeof(float);
  std::printf("kernel=interleaved_complex_mul_f32\n");
  std::printf("model_shape=complex[pairs=%zu]\n", kPairs);
  std::printf("max_absolute_error=%.9g\n", maximum);
  std::printf("equivalent_max_absolute_error=%.9g\n", equivalentMaximum);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("equivalent_median_ms=%.6f\n", equivalentMilliseconds);
  std::printf("logical_gbs=%.6f\n", bytes / milliseconds / 1.0e6);
  return 0;
}
