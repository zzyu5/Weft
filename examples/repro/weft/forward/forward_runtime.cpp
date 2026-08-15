#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#ifndef WEFT_FORWARD_KIND
#error "WEFT_FORWARD_KIND is required"
#endif

#ifndef WEFT_FORWARD_ENTRY
#error "WEFT_FORWARD_ENTRY is required"
#endif

#define WEFT_STRINGIFY_INNER(value) #value
#define WEFT_STRINGIFY(value) WEFT_STRINGIFY_INNER(value)

namespace {

constexpr std::size_t kRows = 128;
[[maybe_unused]] constexpr std::size_t kHidden = 4096;
[[maybe_unused]] constexpr std::size_t kFFN = 14336;
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

float sourceValue(std::size_t index) {
  return static_cast<float>(static_cast<int>(index % 61) - 30) / 32.0F;
}

float rhsValue(std::size_t index) {
  return static_cast<float>(static_cast<int>(index % 29) - 14) / 17.0F;
}

} // namespace

int main() {
#if WEFT_FORWARD_KIND == 5
  constexpr std::size_t elements = kRows * kFFN;
#elif WEFT_FORWARD_KIND == 6
  constexpr std::size_t elements = kRows * kHidden;
#elif WEFT_FORWARD_KIND == 8
  constexpr std::size_t elements = kRows * kHidden;
#elif WEFT_FORWARD_KIND == 9
  constexpr std::size_t elements = kRows * kHidden;
#elif WEFT_FORWARD_KIND == 10
  constexpr std::size_t elements = kRows * kHidden;
#else
  constexpr std::size_t elements = kRows * kHidden;
#endif

  std::vector<float> lhs(elements);
  std::vector<float> rhs(elements);
  std::vector<float> output(elements, -123.0F);
  std::vector<float> reference(elements, 0.0F);
  for (std::size_t index = 0; index < elements; ++index) {
    lhs[index] = sourceValue(index);
    rhs[index] = rhsValue(index);
  }
#if WEFT_FORWARD_KIND == 3
  for (std::size_t index = 0; index < elements; ++index)
    rhs[index] = static_cast<float>((index % 17) + 1) / 9.0F;
#endif

  auto invoke = [&]() {
#if WEFT_FORWARD_KIND == 0
    WEFT_FORWARD_ENTRY(lhs.data(), rhs.data(), output.data(), 0, elements);
#elif WEFT_FORWARD_KIND == 1
    WEFT_FORWARD_ENTRY(lhs.data(), rhs.data(), output.data(), 0, elements);
#elif WEFT_FORWARD_KIND == 2
    WEFT_FORWARD_ENTRY(lhs.data(), rhs.data(), output.data(), 0, elements);
#elif WEFT_FORWARD_KIND == 3
    WEFT_FORWARD_ENTRY(lhs.data(), rhs.data(), output.data(), 0, elements);
#elif WEFT_FORWARD_KIND == 4
    WEFT_FORWARD_ENTRY(lhs.data(), output.data(), 0.5F, 0, elements);
#elif WEFT_FORWARD_KIND == 5
    WEFT_FORWARD_ENTRY(lhs.data(), output.data(), 0, elements);
#elif WEFT_FORWARD_KIND == 6
    WEFT_FORWARD_ENTRY(lhs.data(), output.data(), 0, kRows, kHidden, kHidden);
#elif WEFT_FORWARD_KIND == 7
    WEFT_FORWARD_ENTRY(lhs.data(), output.data(), 0, elements);
#elif WEFT_FORWARD_KIND == 8
    WEFT_FORWARD_ENTRY(lhs.data(), output.data(), 0, kRows, kHidden, kHidden);
#elif WEFT_FORWARD_KIND == 9
    WEFT_FORWARD_ENTRY(lhs.data(), rhs.data(), output.data(), 0, kRows, 64,
                       kHidden);
#elif WEFT_FORWARD_KIND == 10
    WEFT_FORWARD_ENTRY(lhs.data(), rhs.data(), output.data(), 0, kRows, 2048,
                       2048);
#else
#error "unsupported WEFT_FORWARD_KIND"
#endif
  };

  invoke();
#if WEFT_FORWARD_KIND == 0
  for (std::size_t index = 0; index < elements; ++index)
    reference[index] = lhs[index] + rhs[index];
#elif WEFT_FORWARD_KIND == 1
  for (std::size_t index = 0; index < elements; ++index)
    reference[index] = lhs[index] - rhs[index];
#elif WEFT_FORWARD_KIND == 2
  for (std::size_t index = 0; index < elements; ++index)
    reference[index] = lhs[index] * rhs[index];
#elif WEFT_FORWARD_KIND == 3
  for (std::size_t index = 0; index < elements; ++index)
    reference[index] = lhs[index] / rhs[index];
#elif WEFT_FORWARD_KIND == 4
  for (std::size_t index = 0; index < elements; ++index)
    reference[index] = lhs[index] * 0.5F;
#elif WEFT_FORWARD_KIND == 5
  for (std::size_t index = 0; index < elements; ++index) {
    const float value = lhs[index];
    reference[index] =
        0.5F * value *
        (1.0F + std::tanh(0.7978845608028654F * value *
                          (1.0F + 0.044715F * value * value)));
  }
#elif WEFT_FORWARD_KIND == 6
  reference.resize(kRows);
  for (std::size_t row = 0; row < kRows; ++row)
    for (std::size_t column = 0; column < kHidden; ++column)
      reference[row] += lhs[row * kHidden + column];
#elif WEFT_FORWARD_KIND == 7
  reference = lhs;
#elif WEFT_FORWARD_KIND == 8
  for (std::size_t row = 0; row < kRows; ++row)
    std::copy_n(lhs.begin(), kHidden, reference.begin() + row * kHidden);
#elif WEFT_FORWARD_KIND == 9
  for (std::size_t row = 0; row < 64; ++row) {
    std::copy_n(lhs.begin() + row * kHidden, kHidden,
                reference.begin() + row * kHidden);
    std::copy_n(rhs.begin() + row * kHidden, kHidden,
                reference.begin() + (row + 64) * kHidden);
  }
#elif WEFT_FORWARD_KIND == 10
  for (std::size_t row = 0; row < kRows; ++row) {
    std::copy_n(lhs.begin() + row * 2048, 2048,
                reference.begin() + row * kHidden);
    std::copy_n(rhs.begin() + row * 2048, 2048,
                reference.begin() + row * kHidden + 2048);
  }
#endif

  const std::size_t outputElements =
      WEFT_FORWARD_KIND == 6 ? kRows : elements;
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t index = 0; index < outputElements; ++index) {
    const double expected = reference[index];
    const double error =
        std::abs(static_cast<double>(output[index]) - expected);
    maxAbsoluteError = std::max(maxAbsoluteError, error);
    maxRelativeError =
        std::max(maxRelativeError,
                 error / std::max(std::abs(expected), 1.0e-12));
  }
  if (maxAbsoluteError > 3.0e-4 && maxRelativeError > 3.0e-4) {
    std::fprintf(stderr, "%s mismatch: max_abs=%g max_rel=%g\n",
                 WEFT_STRINGIFY(WEFT_FORWARD_ENTRY), maxAbsoluteError,
                 maxRelativeError);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    invoke();
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  std::printf("kernel=%s\n", WEFT_STRINGIFY(WEFT_FORWARD_ENTRY));
#if WEFT_FORWARD_KIND == 5
  std::printf("model_shape=ffn[128,14336]\n");
#elif WEFT_FORWARD_KIND == 8
  std::printf("model_shape=norm_weight[4096]->hidden[128,4096]\n");
#elif WEFT_FORWARD_KIND == 9
  std::printf("model_shape=hidden[64+64,4096]\n");
#elif WEFT_FORWARD_KIND == 10
  std::printf("model_shape=hidden[128,2048+2048]\n");
#else
  std::printf("model_shape=hidden[128,4096]\n");
#endif
  std::printf("elements=%zu\n", elements);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(elements) / milliseconds / 1000.0);
  return 0;
}
