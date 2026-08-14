#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kRows = 128;
constexpr std::size_t kDimension = 1281;
constexpr float kMaximumPeriod = 10000.0F;
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
  const std::size_t half = kDimension / 2U;
  std::vector<float> timesteps(kRows);
  std::vector<float> output(kRows * kDimension, -1.0F);
  std::vector<float> reference(kRows * kDimension, -1.0F);
  for (std::size_t row = 0; row < kRows; ++row)
    timesteps[row] = static_cast<float>((row * 17U) % 1000U) / 7.0F;
  for (std::size_t row = 0; row < kRows; ++row) {
    for (std::size_t coordinate = 0; coordinate < half; ++coordinate) {
      const float frequency = std::exp(
          -std::log(kMaximumPeriod) * static_cast<float>(coordinate) /
          static_cast<float>(half));
      const float angle = timesteps[row] * frequency;
      reference[row * kDimension + coordinate] = std::cos(angle);
      reference[row * kDimension + half + coordinate] = std::sin(angle);
    }
    reference[row * kDimension + 2U * half] = 0.0F;
  }

  timestep_embedding_f32(timesteps.data(), output.data(), 0, kRows,
                         kDimension, kMaximumPeriod, kDimension);
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t index = 0; index < output.size(); ++index) {
    const double absolute = std::fabs(output[index] - reference[index]);
    const double relative = absolute / std::fmax(1.0, std::fabs(reference[index]));
    maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
    maxRelativeError = std::fmax(maxRelativeError, relative);
  }
  if (maxAbsoluteError > 3.0e-5) {
    std::fprintf(stderr, "timestep_embedding_f32 mismatch: %g\n",
                 maxAbsoluteError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  for (int repetition = 0; repetition < 7; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    timestep_embedding_f32(timesteps.data(), output.data(), 0, kRows,
                           kDimension, kMaximumPeriod, kDimension);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=timestep_embedding_f32\n");
  std::printf("model_shape=diffusion_timestep[rows=128,dim=1281]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(output.size()) / milliseconds / 1000.0);
  return 0;
}
