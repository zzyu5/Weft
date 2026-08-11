#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void rwkv_wkv6_f32(
    const float *k, const float *v, const float *r, const float *time_decay,
    const float *time_first, const float *initial_state, float *output,
    float *final_state, std::size_t tokens, std::size_t heads,
    std::size_t width);

namespace {

constexpr std::size_t kTokens = 128;
constexpr std::size_t kHeads = 32;
constexpr std::size_t kWidth = 64;
constexpr std::size_t kTokenElements = kTokens * kHeads * kWidth;
constexpr std::size_t kStateElements = kHeads * kWidth * kWidth;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
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

float inputValue(std::size_t index, std::size_t modulus, float scale) {
  return static_cast<float>(static_cast<int>(index % modulus) -
                            static_cast<int>(modulus / 2U)) /
         scale;
}

void reference(const std::vector<float> &k, const std::vector<float> &v,
               const std::vector<float> &r,
               const std::vector<float> &timeDecay,
               const std::vector<float> &timeFirst,
               const std::vector<float> &initialState,
               std::vector<float> &output, std::vector<float> &state) {
  state = initialState;
  for (std::size_t token = 0; token < kTokens; ++token)
    for (std::size_t head = 0; head < kHeads; ++head) {
      const std::size_t tokenHead = (token * kHeads + head) * kWidth;
      const std::size_t stateHead = head * kWidth * kWidth;
      for (std::size_t column = 0; column < kWidth; ++column) {
        float value = 0.0F;
        for (std::size_t row = 0; row < kWidth; ++row) {
          const std::size_t stateOffset =
              stateHead + row * kWidth + column;
          const float kv = v[tokenHead + column] * k[tokenHead + row];
          value += (state[stateOffset] +
                    kv * timeFirst[head * kWidth + row]) *
                   r[tokenHead + row];
          state[stateOffset] =
              state[stateOffset] * timeDecay[tokenHead + row] + kv;
        }
        output[tokenHead + column] = value;
      }
    }
}

} // namespace

int main() {
  std::vector<float> k(kTokenElements);
  std::vector<float> v(kTokenElements);
  std::vector<float> r(kTokenElements);
  std::vector<float> timeDecay(kTokenElements);
  std::vector<float> timeFirst(kHeads * kWidth);
  std::vector<float> initialState(kStateElements);
  for (std::size_t index = 0; index < kTokenElements; ++index) {
    k[index] = inputValue(index, 29, 64.0F);
    v[index] = inputValue(index * 3U + 1U, 31, 64.0F);
    r[index] = inputValue(index * 5U + 2U, 37, 128.0F);
    timeDecay[index] = 0.90F +
                       static_cast<float>(index % 17U) / 200.0F;
  }
  for (std::size_t index = 0; index < timeFirst.size(); ++index)
    timeFirst[index] = inputValue(index, 19, 128.0F);
  for (std::size_t index = 0; index < initialState.size(); ++index)
    initialState[index] = inputValue(index, 23, 256.0F);

  std::vector<float> expectedOutput(kTokenElements);
  std::vector<float> expectedState;
  std::vector<float> output(kTokenElements);
  std::vector<float> state(kStateElements);
  reference(k, v, r, timeDecay, timeFirst, initialState, expectedOutput,
            expectedState);
  rwkv_wkv6_f32(k.data(), v.data(), r.data(), timeDecay.data(),
                timeFirst.data(), initialState.data(), output.data(),
                state.data(), kTokens, kHeads, kWidth);
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  auto compare = [&](const std::vector<float> &actual,
                     const std::vector<float> &expected) {
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const double absolute = std::fabs(static_cast<double>(actual[index]) -
                                        static_cast<double>(expected[index]));
      const double relative = absolute /
                              std::fmax(1.0, std::fabs(expected[index]));
      maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
      maxRelativeError = std::fmax(maxRelativeError, relative);
    }
  };
  compare(output, expectedOutput);
  compare(state, expectedState);
  if (maxRelativeError > 2.0e-5) {
    std::fprintf(stderr, "rwkv_wkv6_f32 mismatch: abs=%g rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    rwkv_wkv6_f32(k.data(), v.data(), r.data(), timeDecay.data(),
                  timeFirst.data(), initialState.data(), output.data(),
                  state.data(), kTokens, kHeads, kWidth);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=rwkv_wkv6_f32\n");
  std::printf("model_shape=rwkv6[tokens=128,heads=32,head_size=64]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  return 0;
}
