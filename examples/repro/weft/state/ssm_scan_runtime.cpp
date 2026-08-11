#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void ssm_scan_f32(
    const float *initial_state, const std::uint32_t *state_ids, const float *x,
    const float *dt, const float *decay, const float *b, const float *c,
    float *output, float *final_state, std::size_t sequences,
    std::size_t tokens, std::size_t heads, std::size_t groups,
    std::size_t dimensions, std::size_t state_width);

namespace {

constexpr std::size_t kSequences = 1;
constexpr std::size_t kTokens = 128;
constexpr std::size_t kHeads = 32;
constexpr std::size_t kGroups = 8;
constexpr std::size_t kDimensions = 64;
constexpr std::size_t kStateWidth = 16;
constexpr std::size_t kStateElements =
    kSequences * kHeads * kDimensions * kStateWidth;
constexpr std::size_t kOutputElements =
    kSequences * kTokens * kHeads * kDimensions;
constexpr std::size_t kParameterElements =
    kSequences * kTokens * kGroups * kStateWidth;
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

void reference(const std::vector<float> &initialState,
               const std::vector<float> &x, const std::vector<float> &dt,
               const std::vector<float> &decay, const std::vector<float> &b,
               const std::vector<float> &c, std::vector<float> &output,
               std::vector<float> &state) {
  state = initialState;
  for (std::size_t token = 0; token < kTokens; ++token)
    for (std::size_t head = 0; head < kHeads; ++head) {
      const float positiveDt = std::log1p(std::exp(dt[token * kHeads + head]));
      const float decayValue = std::exp(positiveDt * decay[head]);
      const std::size_t group = head / (kHeads / kGroups);
      const std::size_t parameterBase =
          (token * kGroups + group) * kStateWidth;
      for (std::size_t dimension = 0; dimension < kDimensions; ++dimension) {
        const std::size_t stateBase =
            (head * kDimensions + dimension) * kStateWidth;
        const float xdt =
            x[(token * kHeads + head) * kDimensions + dimension] * positiveDt;
        float value = 0.0F;
        for (std::size_t stateIndex = 0; stateIndex < kStateWidth;
             ++stateIndex) {
          const std::size_t offset = stateBase + stateIndex;
          state[offset] = state[offset] * decayValue +
                          b[parameterBase + stateIndex] * xdt;
          value += state[offset] * c[parameterBase + stateIndex];
        }
        output[(token * kHeads + head) * kDimensions + dimension] = value;
      }
    }
}

} // namespace

int main() {
  std::vector<float> initialState(kStateElements);
  std::vector<std::uint32_t> stateIds(kSequences, 0);
  std::vector<float> x(kOutputElements);
  std::vector<float> dt(kSequences * kTokens * kHeads);
  std::vector<float> decay(kHeads);
  std::vector<float> b(kParameterElements);
  std::vector<float> c(kParameterElements);
  for (std::size_t index = 0; index < initialState.size(); ++index)
    initialState[index] = inputValue(index, 31, 128.0F);
  for (std::size_t index = 0; index < x.size(); ++index)
    x[index] = inputValue(index, 67, 64.0F);
  for (std::size_t index = 0; index < dt.size(); ++index)
    dt[index] = inputValue(index, 29, 32.0F);
  for (std::size_t head = 0; head < kHeads; ++head)
    decay[head] = -0.01F * static_cast<float>(head + 1U);
  for (std::size_t index = 0; index < b.size(); ++index) {
    b[index] = inputValue(index, 23, 128.0F);
    c[index] = inputValue(index * 3U + 1U, 37, 64.0F);
  }

  std::vector<float> expectedOutput(kOutputElements);
  std::vector<float> expectedState;
  std::vector<float> output(kOutputElements);
  std::vector<float> state(kStateElements);
  reference(initialState, x, dt, decay, b, c, expectedOutput, expectedState);
  ssm_scan_f32(initialState.data(), stateIds.data(), x.data(), dt.data(),
               decay.data(), b.data(), c.data(), output.data(), state.data(),
               kSequences, kTokens, kHeads, kGroups, kDimensions, kStateWidth);
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
    std::fprintf(stderr, "ssm_scan_f32 mismatch: abs=%g rel=%g\n",
                 maxAbsoluteError, maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    ssm_scan_f32(initialState.data(), stateIds.data(), x.data(), dt.data(),
                 decay.data(), b.data(), c.data(), output.data(), state.data(),
                 kSequences, kTokens, kHeads, kGroups, kDimensions,
                 kStateWidth);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=ssm_scan_f32\n");
  std::printf("model_shape=mamba2[sequences=1,tokens=128,heads=32,dim=64,state=16,groups=8]\n");
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  return 0;
}
