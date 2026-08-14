#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t kRows = 128;
constexpr std::size_t kClasses = 32768;
constexpr std::size_t kElements = kRows * kClasses;
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

} // namespace

int main() {
  std::vector<float> logits(kElements);
  std::vector<std::uint32_t> labels(kRows);
  for (std::size_t index = 0; index < kElements; ++index)
    logits[index] =
        static_cast<float>(static_cast<int>((index * 17U) % 257U) - 128) /
        32.0F;
  for (std::size_t row = 0; row < kRows; ++row)
    labels[row] = static_cast<std::uint32_t>((row * 7919U) % kClasses);

  std::vector<float> expectedGradient(kElements);
  std::vector<float> expectedLoss(kRows);
  for (std::size_t row = 0; row < kRows; ++row) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::size_t cls = 0; cls < kClasses; ++cls)
      maximum = std::max(maximum, logits[row * kClasses + cls]);
    double total = 0.0;
    for (std::size_t cls = 0; cls < kClasses; ++cls)
      total += std::exp(static_cast<double>(logits[row * kClasses + cls] -
                                            maximum));
    expectedLoss[row] = static_cast<float>(
        std::log(total) + maximum - logits[row * kClasses + labels[row]]);
    for (std::size_t cls = 0; cls < kClasses; ++cls)
      expectedGradient[row * kClasses + cls] =
          static_cast<float>(std::exp(logits[row * kClasses + cls] - maximum) /
                             total) -
          (cls == labels[row] ? 1.0F : 0.0F);
  }

  std::vector<float> gradient(kElements, 0.0F);
  std::vector<float> loss(kRows, 0.0F);
  cross_entropy_loss_gradient_f32(logits.data(), labels.data(), gradient.data(),
                                  loss.data(), 0, kRows, kClasses);
  double maxAbsolute = 0.0;
  double maxRelative = 0.0;
  for (std::size_t index = 0; index < kElements; ++index) {
    const double error = std::abs(static_cast<double>(gradient[index]) -
                                  static_cast<double>(expectedGradient[index]));
    maxAbsolute = std::max(maxAbsolute, error);
    maxRelative = std::max(
        maxRelative, error /
                         std::max(std::abs(static_cast<double>(expectedGradient[index])),
                                  1.0e-6));
  }
  for (std::size_t row = 0; row < kRows; ++row) {
    const double error =
        std::abs(static_cast<double>(loss[row]) - expectedLoss[row]);
    maxAbsolute = std::max(maxAbsolute, error);
    maxRelative = std::max(
        maxRelative,
        error / std::max(std::abs(static_cast<double>(expectedLoss[row])), 1.0e-6));
  }
  if (maxAbsolute > 2.0e-3 && maxRelative > 2.0e-3) {
    std::fprintf(stderr, "cross entropy mismatch: abs=%g rel=%g\n", maxAbsolute,
                 maxRelative);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(5);
  for (int repetition = 0; repetition < 5; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    cross_entropy_loss_gradient_f32(
        logits.data(), labels.data(), gradient.data(), loss.data(), 0, kRows,
        kClasses);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=cross_entropy_loss_gradient_f32\n");
  std::printf("model_shape=language_model[tokens=%zu,vocabulary=%zu]\n", kRows,
              kClasses);
  std::printf("max_absolute_error=%.9g\n", maxAbsolute);
  std::printf("max_relative_error=%.9g\n", maxRelative);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
