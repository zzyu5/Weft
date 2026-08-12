#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

extern "C" void sam_add_relative_position_f32(
    const float *, const float *, const float *, float *, size_t, size_t, size_t,
    size_t);

static double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int main() {
  constexpr size_t patches = 64;
  constexpr size_t query_height = 8;
  constexpr size_t query_width = 8;
  constexpr size_t key_size = 14;
  const size_t query_plane = query_height * query_width;
  const size_t score_stride = query_plane * key_size * key_size;
  const size_t score_elements = patches * score_stride;
  const size_t relative_elements = patches * query_plane * key_size;
  std::vector<float> scores(score_elements), relative_width(relative_elements);
  std::vector<float> relative_height(relative_elements), output(score_elements);
  std::vector<float> expected(score_elements);
  for (size_t i = 0; i < scores.size(); ++i)
    scores[i] = static_cast<float>(static_cast<int>(i % 101) - 50) * 0.002f;
  for (size_t i = 0; i < relative_elements; ++i) {
    relative_width[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.001f;
    relative_height[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.001f;
  }
  expected = scores;
  for (size_t patch = 0; patch < patches; ++patch) {
    for (size_t query_y = 0; query_y < query_height; ++query_y) {
      for (size_t query_x = 0; query_x < query_width; ++query_x) {
        const size_t query = query_y * query_width + query_x;
        const size_t score_base = patch * score_stride + query * key_size * key_size;
        const size_t relative_base = (patch * query_plane + query) * key_size;
        for (size_t key = 0; key < key_size; ++key) {
          for (size_t key_y = 0; key_y < key_size; ++key_y)
            expected[score_base + key_y * key_size + key] += relative_width[relative_base + key];
          for (size_t key_x = 0; key_x < key_size; ++key_x)
            expected[score_base + key * key_size + key_x] += relative_height[relative_base + key];
        }
      }
    }
  }
  sam_add_relative_position_f32(scores.data(), relative_width.data(),
                                relative_height.data(), output.data(), patches,
                                query_height, query_width, key_size);
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (size_t i = 0; i < output.size(); ++i) {
    const float error = std::abs(output[i] - expected[i]);
    max_abs = std::max(max_abs, error);
    max_rel = std::max(max_rel, error / std::max(1.0e-6f, std::abs(expected[i])));
  }
  if (max_abs > 1.0e-6f || max_rel > 1.0e-3f) {
    std::fprintf(stderr, "sam relative position mismatch\n");
    return 1;
  }
  std::vector<unsigned char> eviction(64 * 1024 * 1024, 1);
  std::vector<double> samples;
  for (int repetition = 0; repetition < 10; ++repetition) {
    for (size_t i = 0; i < eviction.size(); i += 64)
      eviction[i] = static_cast<unsigned char>(eviction[i] + 1);
    const auto begin = std::chrono::steady_clock::now();
    sam_add_relative_position_f32(scores.data(), relative_width.data(),
                                  relative_height.data(), output.data(), patches,
                                  query_height, query_width, key_size);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double bytes = static_cast<double>(score_elements * sizeof(float) * 2 +
                                           relative_elements * sizeof(float) * 2);
  std::printf("kernel=sam_add_relative_position_f32\n");
  std::printf("model_shape=patches=64;query=8x8;key=14\n");
  std::printf("max_absolute_error=%.9g\n", max_abs);
  std::printf("max_relative_error=%.9g\n", max_rel);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gbytes_s=%.6f\n", bytes / (milliseconds * 1.0e6));
  return 0;
}
