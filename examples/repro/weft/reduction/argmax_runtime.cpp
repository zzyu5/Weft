#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void argmax_f32(const float *x, std::uint32_t *indices,
                           std::size_t row_begin, std::size_t row_end,
                           std::size_t cols, std::size_t stride);

namespace {

constexpr std::size_t kRows = 128;
constexpr std::size_t kVocabulary = 128256;
constexpr std::size_t kElements = kRows * kVocabulary;
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
  std::vector<std::uint32_t> output(kRows);
  std::vector<std::uint32_t> expected(kRows);
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<float>(static_cast<int>(index % 4093) - 2046) /
                   128.0F;
  for (std::size_t row = 0; row < kRows; ++row) {
    const std::size_t first = (row * 7919U + 123U) % (kVocabulary / 2U);
    input[row * kVocabulary + first] = 1000.0F;
    input[row * kVocabulary + first + kVocabulary / 2U] = 1000.0F;
    expected[row] = static_cast<std::uint32_t>(first);
  }

  argmax_f32(input.data(), output.data(), 0, kRows, kVocabulary, kVocabulary);
  for (std::size_t row = 0; row < kRows; ++row) {
    if (output[row] != expected[row]) {
      std::fprintf(stderr, "argmax mismatch at row=%zu: got=%u expected=%u\n",
                   row, output[row], expected[row]);
      return 1;
    }
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    argmax_f32(input.data(), output.data(), 0, kRows, kVocabulary,
               kVocabulary);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  std::printf("kernel=argmax_f32\n");
  std::printf("model_shape=logits[batch=128,vocab=128256]\n");
  std::printf("elements=%zu\n", kElements);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
