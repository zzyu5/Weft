#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void ssm_conv_f32(
    const float *state, const float *weight, float *output,
    std::size_t sequences, std::size_t tokens, std::size_t channels,
    std::size_t taps, std::size_t state_channel_stride,
    std::size_t state_sequence_stride, std::size_t weight_channel_stride,
    std::size_t output_token_stride, std::size_t output_sequence_stride);
extern "C" void ssm_conv_f32_equivalent(
    const float *state, const float *weight, float *output,
    std::size_t sequences, std::size_t tokens, std::size_t channels,
    std::size_t taps, std::size_t state_channel_stride,
    std::size_t state_sequence_stride, std::size_t weight_channel_stride,
    std::size_t output_token_stride, std::size_t output_sequence_stride);

namespace {

constexpr std::size_t kSequences = 1;
constexpr std::size_t kTokens = 128;
constexpr std::size_t kChannels = 2048;
constexpr std::size_t kTaps = 4;
constexpr std::size_t kStateTokens = kTokens + kTaps - 1;
constexpr std::size_t kOutputElements = kSequences * kTokens * kChannels;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

using Kernel = decltype(&ssm_conv_f32);

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

double run(Kernel kernel, const std::vector<float> &state,
           const std::vector<float> &weight, std::vector<float> &output,
           std::vector<std::uint8_t> &eviction) {
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    kernel(state.data(), weight.data(), output.data(), kSequences, kTokens,
           kChannels, kTaps, kStateTokens, kChannels * kStateTokens, kTaps,
           kChannels, kTokens * kChannels);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  return median(samples);
}

} // namespace

int main() {
  std::vector<float> state(kSequences * kChannels * kStateTokens);
  std::vector<float> weight(kChannels * kTaps);
  std::vector<float> expected(kOutputElements);
  std::vector<float> output(kOutputElements);
  std::vector<float> equivalent(kOutputElements);
  for (std::size_t index = 0; index < state.size(); ++index)
    state[index] =
        static_cast<float>(static_cast<int>(index % 257U) - 128) / 256.0F;
  for (std::size_t index = 0; index < weight.size(); ++index)
    weight[index] =
        static_cast<float>(static_cast<int>(index % 31U) - 15) / 32.0F;

  for (std::size_t token = 0; token < kTokens; ++token)
    for (std::size_t channel = 0; channel < kChannels; ++channel) {
      float value = 0.0F;
      for (std::size_t tap = 0; tap < kTaps; ++tap)
        value += state[channel * kStateTokens + token + tap] *
                 weight[channel * kTaps + tap];
      expected[token * kChannels + channel] = value;
    }

  ssm_conv_f32(state.data(), weight.data(), output.data(), kSequences, kTokens,
               kChannels, kTaps, kStateTokens, kChannels * kStateTokens, kTaps,
               kChannels, kTokens * kChannels);
  ssm_conv_f32_equivalent(
      state.data(), weight.data(), equivalent.data(), kSequences, kTokens,
      kChannels, kTaps, kStateTokens, kChannels * kStateTokens, kTaps,
      kChannels, kTokens * kChannels);
  double maxAbsoluteError = 0.0;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    maxAbsoluteError = std::fmax(
        maxAbsoluteError, std::fabs(static_cast<double>(output[index]) -
                                    static_cast<double>(expected[index])));
    maxAbsoluteError = std::fmax(
        maxAbsoluteError, std::fabs(static_cast<double>(equivalent[index]) -
                                    static_cast<double>(expected[index])));
  }
  if (maxAbsoluteError != 0.0) {
    std::fprintf(stderr, "ssm_conv_f32 mismatch: max_abs=%g\n",
                 maxAbsoluteError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  const double primary = run(ssm_conv_f32, state, weight, output, eviction);
  const double alternate =
      run(ssm_conv_f32_equivalent, state, weight, equivalent, eviction);
  std::printf("kernel=ssm_conv_f32\n");
  std::printf("model_shape=mamba_short_conv[sequences=1,tokens=128,channels=2048,taps=4]\n");
  std::printf("primary_median_ms=%.6f\n", primary);
  std::printf("equivalent_median_ms=%.6f\n", alternate);
  std::printf("gop_s=%.6f\n",
              2.0 * static_cast<double>(kOutputElements * kTaps) / primary /
                  1.0e6);
  return 0;
}
