#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
constexpr std::size_t kElements = 1U << 20U;
constexpr std::size_t kEvictionBytes = 64U << 20U;
volatile std::uint64_t evictionSink = 0;
using Kernel = void (*)(const std::uint8_t *, const float *, const float *,
                        const float *, float *, std::size_t);
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
double run(Kernel kernel, const std::vector<std::uint8_t> &codes,
           const std::vector<float> &table, const std::vector<float> &scale,
           const std::vector<float> &bias, std::vector<float> &output,
           std::vector<std::uint8_t> &eviction) {
  std::vector<double> samples;
  for (int repetition = 0; repetition < 7; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    kernel(codes.data(), table.data(), scale.data(), bias.data(), output.data(),
           kElements);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  return median(samples);
}
} // namespace
int main() {
  std::vector<std::uint8_t> codes(kElements);
  std::vector<float> table(16);
  std::vector<float> scale(kElements);
  std::vector<float> bias(kElements);
  std::vector<float> primary(kElements);
  std::vector<float> equivalent(kElements);
  for (std::size_t index = 0; index < 16; ++index)
    table[index] = static_cast<float>(static_cast<int>(index) - 8) / 8.0F;
  for (std::size_t index = 0; index < kElements; ++index) {
    codes[index] = static_cast<std::uint8_t>((index * 13) & 15U);
    scale[index] = 0.5F + static_cast<float>(index % 31) / 64.0F;
    bias[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 32.0F;
  }
  codebook_lookup_affine_f32(codes.data(), table.data(), scale.data(), bias.data(),
                             primary.data(), kElements);
  codebook_lookup_affine_f32_equivalent(
      codes.data(), table.data(), scale.data(), bias.data(), equivalent.data(),
      kElements);
  double maximum = 0.0;
  double equivalentMaximum = 0.0;
  for (std::size_t index = 0; index < kElements; ++index) {
    const float reference = table[codes[index]] * scale[index] + bias[index];
    maximum = std::max(maximum, std::abs(static_cast<double>(primary[index]) -
                                         reference));
    equivalentMaximum = std::max(
        equivalentMaximum,
        std::abs(static_cast<double>(equivalent[index]) - reference));
  }
  if (maximum > 1.0e-6 || equivalentMaximum > 1.0e-6) {
    std::fprintf(stderr, "codebook lookup mismatch: %g %g\n", maximum,
                 equivalentMaximum);
    return 1;
  }
  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  const double milliseconds =
      run(codebook_lookup_affine_f32, codes, table, scale, bias, primary, eviction);
  const double equivalentMilliseconds = run(
      codebook_lookup_affine_f32_equivalent, codes, table, scale, bias, equivalent,
      eviction);
  std::printf("kernel=codebook_lookup_affine_f32\n");
  std::printf("model_shape=codebook[elements=%zu;entries=16]\n", kElements);
  std::printf("max_absolute_error=%.9g\n", maximum);
  std::printf("equivalent_max_absolute_error=%.9g\n", equivalentMaximum);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("equivalent_median_ms=%.6f\n", equivalentMilliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1.0e3);
  return 0;
}
