#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t kElements = 128256;
constexpr std::size_t kLogicalElements = 127997;
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
  const std::size_t middle = samples.size() / 2;
  return samples.size() % 2 == 0
             ? 0.5 * (samples[middle - 1] + samples[middle])
             : samples[middle];
}

} // namespace

int main() {
  std::vector<float> input(kElements);
  for (std::size_t index = 0; index < kLogicalElements; ++index)
    input[index] =
        static_cast<float>(static_cast<int>(index % 1021) - 510) / 256.0F;
  input[7919] = 7.0F;
  std::fill(input.begin() + kLogicalElements, input.end(), 1000.0F);

  float output[2] = {};
  online_softmax_summary(input.data(), output, kElements, kLogicalElements);

  double referenceMaximum = -std::numeric_limits<double>::infinity();
  for (std::size_t index = 0; index < kLogicalElements; ++index)
    referenceMaximum = std::max(referenceMaximum,
                                static_cast<double>(input[index]));
  double referenceSum = 0.0;
  for (std::size_t index = 0; index < kLogicalElements; ++index)
    referenceSum +=
        std::exp(static_cast<double>(input[index]) - referenceMaximum);

  const double maximumError =
      std::abs(static_cast<double>(output[0]) - referenceMaximum);
  const double sumRelativeError =
      std::abs(static_cast<double>(output[1]) - referenceSum) / referenceSum;
  if (maximumError != 0.0 || sumRelativeError > 3.0e-4) {
    std::fprintf(stderr,
                 "online summary mismatch: max_abs=%g sum_rel=%g got=(%g,%g)\n",
                 maximumError, sumRelativeError, output[0], output[1]);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    online_softmax_summary(input.data(), output, kElements, kLogicalElements);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  std::printf("kernel=online_softmax_summary\n");
  std::printf("model_shape=logits[vocab=128256,logical=127997]\n");
  std::printf("elements=%zu\n", kElements);
  std::printf("logical_elements=%zu\n", kLogicalElements);
  std::printf("maximum_absolute_error=%.9g\n", maximumError);
  std::printf("sum_relative_error=%.9g\n", sumRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
