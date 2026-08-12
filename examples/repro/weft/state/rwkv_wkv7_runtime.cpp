#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

extern "C" void rwkv_wkv7_f32(
    const float *, const float *, const float *, const float *, const float *,
    const float *, const float *, float *, float *, size_t, size_t, size_t);

static double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

int main() {
  constexpr size_t tokens = 128;
  constexpr size_t heads = 32;
  constexpr size_t width = 64;
  const size_t stream_elements = tokens * heads * width;
  const size_t state_elements = heads * width * width;
  std::vector<float> key(stream_elements), value(stream_elements), receptance(stream_elements);
  std::vector<float> decay(stream_elements), a(stream_elements), b(stream_elements);
  std::vector<float> initial(state_elements), output(stream_elements), state(state_elements);
  std::vector<float> expected_output(stream_elements), expected_state(state_elements);
  for (size_t i = 0; i < stream_elements; ++i) {
    key[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.002f;
    value[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.002f;
    receptance[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.002f;
    decay[i] = 0.97f + static_cast<float>(i % 7) * 0.003f;
    a[i] = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.001f;
    b[i] = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.001f;
  }
  for (size_t i = 0; i < state_elements; ++i)
    initial[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.001f;
  expected_state = initial;
  for (size_t token = 0; token < tokens; ++token) {
    for (size_t head = 0; head < heads; ++head) {
      const size_t th = (token * heads + head) * width;
      const size_t sh = head * width * width;
      for (size_t row = 0; row < width; ++row) {
        float sa = 0.0f;
        for (size_t column = 0; column < width; ++column)
          sa += a[th + column] * expected_state[sh + row * width + column];
        float result = 0.0f;
        for (size_t column = 0; column < width; ++column) {
          const size_t si = sh + row * width + column;
          const float next = expected_state[si] * decay[th + column] +
                             value[th + row] * key[th + column] + sa * b[th + column];
          expected_state[si] = next;
          result += next * receptance[th + column];
        }
        expected_output[th + row] = result;
      }
    }
  }
  rwkv_wkv7_f32(key.data(), value.data(), receptance.data(), decay.data(), a.data(),
                 b.data(), initial.data(), output.data(), state.data(), tokens, heads, width);
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  auto compare = [&](const std::vector<float> &actual,
                     const std::vector<float> &expected) {
    for (size_t i = 0; i < actual.size(); ++i) {
      const float error = std::abs(actual[i] - expected[i]);
      max_abs = std::max(max_abs, error);
      max_rel = std::max(max_rel, error / std::max(1.0e-6f, std::abs(expected[i])));
    }
  };
  compare(output, expected_output);
  compare(state, expected_state);
  if (max_abs > 2.0e-4f || max_rel > 2.0e-3f) {
    std::fprintf(stderr, "rwkv wkv7 mismatch\n");
    return 1;
  }
  std::vector<unsigned char> eviction(64 * 1024 * 1024, 1);
  std::vector<double> samples;
  for (int repetition = 0; repetition < 10; ++repetition) {
    for (size_t i = 0; i < eviction.size(); i += 64)
      eviction[i] = static_cast<unsigned char>(eviction[i] + 1);
    const auto begin = std::chrono::steady_clock::now();
    rwkv_wkv7_f32(key.data(), value.data(), receptance.data(), decay.data(), a.data(),
                   b.data(), initial.data(), output.data(), state.data(), tokens, heads, width);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double operations = 10.0 * static_cast<double>(tokens * heads * width * width);
  std::printf("kernel=rwkv_wkv7_f32\n");
  std::printf("model_shape=tokens=128;heads=32;width=64\n");
  std::printf("max_absolute_error=%.9g\n", max_abs);
  std::printf("max_relative_error=%.9g\n", max_rel);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / (milliseconds * 1.0e6));
  return 0;
}
