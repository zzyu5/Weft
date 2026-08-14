#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kElements = 16U * 1024U * 1024U;
constexpr float kLearningRate = 1.0e-3F;
constexpr float kBeta1 = 0.9F;
constexpr float kBeta2 = 0.999F;
constexpr float kInverseBias1 = 10.0F;
constexpr float kInverseBias2 = 1000.0F;
constexpr float kEpsilon = 1.0e-8F;
constexpr float kWeightDecay = 0.01F;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

using Kernel = void (*)(float *, const float *, float *, float *, std::size_t,
                        std::size_t, float, float, float, float, float, float,
                        float);

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

double run(Kernel kernel, const std::vector<float> &initialParameters,
           const std::vector<float> &gradients,
           const std::vector<float> &initialFirst,
           const std::vector<float> &initialSecond,
           std::vector<std::uint8_t> &eviction) {
  std::vector<float> parameters(kElements);
  std::vector<float> first(kElements);
  std::vector<float> second(kElements);
  std::vector<double> samples;
  samples.reserve(5);
  for (int repetition = 0; repetition < 5; ++repetition) {
    parameters = initialParameters;
    first = initialFirst;
    second = initialSecond;
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    kernel(parameters.data(), gradients.data(), first.data(), second.data(), 0,
           kElements, kLearningRate, kBeta1, kBeta2, kInverseBias1,
           kInverseBias2, kEpsilon, kWeightDecay);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  return median(samples);
}

} // namespace

int main() {
  std::vector<float> initialParameters(kElements);
  std::vector<float> gradients(kElements);
  std::vector<float> initialFirst(kElements);
  std::vector<float> initialSecond(kElements);
  for (std::size_t index = 0; index < kElements; ++index) {
    initialParameters[index] =
        static_cast<float>(static_cast<int>((index * 17U) % 257U) - 128) /
        256.0F;
    gradients[index] =
        static_cast<float>(static_cast<int>((index * 23U) % 113U) - 56) /
        4096.0F;
    initialFirst[index] =
        static_cast<float>(static_cast<int>((index * 29U) % 97U) - 48) /
        8192.0F;
    initialSecond[index] =
        static_cast<float>((index * 31U) % 89U + 1U) / 65536.0F;
  }
  std::vector<float> expectedParameters = initialParameters;
  std::vector<float> expectedFirst = initialFirst;
  std::vector<float> expectedSecond = initialSecond;
  for (std::size_t index = 0; index < kElements; ++index) {
    const float gradient = gradients[index];
    const float nextFirst =
        kBeta1 * expectedFirst[index] + (1.0F - kBeta1) * gradient;
    const float nextSecond = kBeta2 * expectedSecond[index] +
                             (1.0F - kBeta2) * gradient * gradient;
    const float update = nextFirst * kInverseBias1 /
                         (std::sqrt(nextSecond * kInverseBias2) + kEpsilon);
    expectedParameters[index] -=
        kLearningRate * (update + kWeightDecay * expectedParameters[index]);
    expectedFirst[index] = nextFirst;
    expectedSecond[index] = nextSecond;
  }

  std::vector<float> primaryParameters = initialParameters;
  std::vector<float> primaryFirst = initialFirst;
  std::vector<float> primarySecond = initialSecond;
  std::vector<float> equivalentParameters = initialParameters;
  std::vector<float> equivalentFirst = initialFirst;
  std::vector<float> equivalentSecond = initialSecond;
  adamw_f32(primaryParameters.data(), gradients.data(), primaryFirst.data(),
            primarySecond.data(), 0, kElements, kLearningRate, kBeta1, kBeta2,
            kInverseBias1, kInverseBias2, kEpsilon, kWeightDecay);
  adamw_f32_equivalent(
      equivalentParameters.data(), gradients.data(), equivalentFirst.data(),
      equivalentSecond.data(), 0, kElements, kLearningRate, kBeta1, kBeta2,
      kInverseBias1, kInverseBias2, kEpsilon, kWeightDecay);
  double maxAbsolute = 0.0;
  double maxRelative = 0.0;
  auto compare = [&](const std::vector<float> &actual,
                     const std::vector<float> &expected) {
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const double error = std::abs(static_cast<double>(actual[index]) -
                                    static_cast<double>(expected[index]));
      maxAbsolute = std::max(maxAbsolute, error);
      maxRelative = std::max(
          maxRelative,
          error / std::max(std::abs(static_cast<double>(expected[index])), 1.0e-6));
    }
  };
  compare(primaryParameters, expectedParameters);
  compare(primaryFirst, expectedFirst);
  compare(primarySecond, expectedSecond);
  compare(equivalentParameters, expectedParameters);
  compare(equivalentFirst, expectedFirst);
  compare(equivalentSecond, expectedSecond);
  if (maxAbsolute > 2.0e-6 && maxRelative > 2.0e-5) {
    std::fprintf(stderr, "AdamW mismatch: abs=%g rel=%g\n", maxAbsolute,
                 maxRelative);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  const double primaryMilliseconds =
      run(adamw_f32, initialParameters, gradients, initialFirst, initialSecond,
          eviction);
  const double equivalentMilliseconds = run(
      adamw_f32_equivalent, initialParameters, gradients, initialFirst,
      initialSecond, eviction);
  const double bytes = static_cast<double>(kElements) * sizeof(float) * 7.0;
  std::printf("kernel=adamw_f32\n");
  std::printf("model_shape=optimizer_state[parameters=%zu]\n", kElements);
  std::printf("max_absolute_error=%.9g\n", maxAbsolute);
  std::printf("max_relative_error=%.9g\n", maxRelative);
  std::printf("primary_median_ms=%.6f\n", primaryMilliseconds);
  std::printf("equivalent_median_ms=%.6f\n", equivalentMilliseconds);
  std::printf("primary_logical_gbs=%.6f\n",
              bytes / primaryMilliseconds / 1.0e6);
  std::printf("equivalent_logical_gbs=%.6f\n",
              bytes / equivalentMilliseconds / 1.0e6);
  return 0;
}
