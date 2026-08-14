#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

static double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int main() {
  constexpr size_t batches = 1;
  constexpr size_t channels = 320;
  constexpr size_t height = 64;
  constexpr size_t width = 64;
  constexpr size_t groups = 32;
  constexpr float epsilon = 1.0e-5f;
  const size_t plane = height * width;
  const size_t elements = batches * channels * plane;
  std::vector<float> source(elements), output(elements), expected(elements);
  for (size_t i = 0; i < elements; ++i)
    source[i] = static_cast<float>(static_cast<int>(i % 251) - 125) * 0.002f;
  const size_t channels_per_group = (channels + groups - 1) / groups;
  for (size_t batch = 0; batch < batches; ++batch) {
    for (size_t group = 0; group < groups; ++group) {
      const size_t channel_begin = group * channels_per_group;
      const size_t channel_end = std::min(channels, channel_begin + channels_per_group);
      const size_t begin = batch * channels * plane + channel_begin * plane;
      const size_t end = batch * channels * plane + channel_end * plane;
      double total = 0.0;
      for (size_t i = begin; i < end; ++i)
        total += source[i];
      const float mean = static_cast<float>(total / static_cast<double>(end - begin));
      double squared = 0.0;
      for (size_t i = begin; i < end; ++i) {
        expected[i] = source[i] - mean;
        squared += static_cast<double>(expected[i]) * expected[i];
      }
      const float scale = 1.0f / std::sqrt(static_cast<float>(squared / (end - begin)) + epsilon);
      for (size_t i = begin; i < end; ++i)
        expected[i] *= scale;
    }
  }
  group_norm_f32(source.data(), output.data(), batches, channels, height, width,
                 groups, epsilon);
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (size_t i = 0; i < output.size(); ++i) {
    const float error = std::abs(output[i] - expected[i]);
    max_abs = std::max(max_abs, error);
    max_rel = std::max(max_rel, error / std::max(1.0e-6f, std::abs(expected[i])));
  }
  if (max_abs > 2.0e-4f || max_rel > 2.0e-3f) {
    std::fprintf(stderr, "group norm mismatch\n");
    return 1;
  }
  std::vector<unsigned char> eviction(64 * 1024 * 1024, 1);
  std::vector<double> samples;
  for (int repetition = 0; repetition < 10; ++repetition) {
    for (size_t i = 0; i < eviction.size(); i += 64)
      eviction[i] = static_cast<unsigned char>(eviction[i] + 1);
    const auto begin = std::chrono::steady_clock::now();
    group_norm_f32(source.data(), output.data(), batches, channels, height, width,
                   groups, epsilon);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=group_norm_f32\n");
  std::printf("model_shape=N=1;C=320;H=64;W=64;groups=32\n");
  std::printf("max_absolute_error=%.9g\n", max_abs);
  std::printf("max_relative_error=%.9g\n", max_rel);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n", elements / (milliseconds * 1000.0));
  return 0;
}
