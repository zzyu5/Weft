#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

extern "C" void gated_delta_net_kda_f32(
    const float *, const float *, const float *, const float *, const float *,
    const float *, float *, float *, size_t, size_t, size_t);

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
  std::vector<float> query(stream_elements), key(stream_elements), value(stream_elements);
  std::vector<float> gate(stream_elements), beta(tokens * heads), initial(state_elements);
  std::vector<float> output(stream_elements), state(state_elements);
  std::vector<float> expected_output(stream_elements), expected_state(state_elements);
  for (size_t i = 0; i < stream_elements; ++i) {
    query[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.003f;
    key[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.003f;
    value[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.003f;
    gate[i] = -0.003f * static_cast<float>(1 + i % 5);
  }
  for (size_t i = 0; i < beta.size(); ++i)
    beta[i] = 0.08f + static_cast<float>(i % 7) * 0.005f;
  for (size_t i = 0; i < state_elements; ++i)
    initial[i] = static_cast<float>(static_cast<int>(i % 29) - 14) * 0.001f;
  expected_state = initial;
  const float inverse_root = 1.0f / std::sqrt(static_cast<float>(width));
  for (size_t token = 0; token < tokens; ++token) {
    for (size_t head = 0; head < heads; ++head) {
      const size_t th = (token * heads + head) * width;
      const size_t sh = head * width * width;
      for (size_t row = 0; row < width; ++row)
        for (size_t column = 0; column < width; ++column)
          expected_state[sh + row * width + column] *= std::exp(gate[th + column]);
      for (size_t row = 0; row < width; ++row) {
        float projection = 0.0f;
        for (size_t column = 0; column < width; ++column)
          projection += expected_state[sh + row * width + column] * key[th + column];
        const float delta = (value[th + row] - projection) * beta[token * heads + head];
        float result = 0.0f;
        for (size_t column = 0; column < width; ++column) {
          const size_t si = sh + row * width + column;
          expected_state[si] += delta * key[th + column];
          result += expected_state[si] * query[th + column];
        }
        expected_output[th + row] = result * inverse_root;
      }
    }
  }
  gated_delta_net_kda_f32(query.data(), key.data(), value.data(), gate.data(), beta.data(),
                          initial.data(), output.data(), state.data(), tokens, heads, width);
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
  if (max_abs > 4.0e-4f || max_rel > 3.0e-3f) {
    std::fprintf(stderr, "gated delta net mismatch\n");
    return 1;
  }
  std::vector<unsigned char> eviction(64 * 1024 * 1024, 1);
  std::vector<double> samples;
  for (int repetition = 0; repetition < 10; ++repetition) {
    for (size_t i = 0; i < eviction.size(); i += 64)
      eviction[i] = static_cast<unsigned char>(eviction[i] + 1);
    const auto begin = std::chrono::steady_clock::now();
    gated_delta_net_kda_f32(query.data(), key.data(), value.data(), gate.data(), beta.data(),
                            initial.data(), output.data(), state.data(), tokens, heads, width);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double operations = 13.0 * static_cast<double>(tokens * heads * width * width);
  std::printf("kernel=gated_delta_net_kda_f32\n");
  std::printf("model_shape=tokens=128;heads=32;width=64\n");
  std::printf("max_absolute_error=%.9g\n", max_abs);
  std::printf("max_relative_error=%.9g\n", max_rel);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / (milliseconds * 1.0e6));
  return 0;
}
