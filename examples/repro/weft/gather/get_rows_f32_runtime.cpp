#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void get_rows_f32(const float *table, const std::uint32_t *indices,
                             float *output, std::size_t token_begin,
                             std::size_t token_end, std::size_t hidden,
                             std::size_t table_stride,
                             std::size_t output_stride);

namespace {

constexpr std::size_t kVocabulary = 32000;
constexpr std::size_t kHidden = 2048;
constexpr std::size_t kTokens = 128;
constexpr std::size_t kOutputElements = kTokens * kHidden;
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
  std::vector<float> table(kVocabulary * kHidden);
  std::vector<std::uint32_t> indices(kTokens);
  std::vector<float> output(kOutputElements);
  for (std::size_t token = 0; token < kTokens; ++token) {
    const std::uint32_t row =
        static_cast<std::uint32_t>((token * 997U) % kVocabulary);
    indices[token] = row;
    for (std::size_t column = 0; column < kHidden; ++column)
      table[static_cast<std::size_t>(row) * kHidden + column] =
          static_cast<float>(static_cast<int>((row + column) % 4093) - 2046) /
          1024.0F;
  }

  get_rows_f32(table.data(), indices.data(), output.data(), 0, kTokens, kHidden,
               kHidden, kHidden);
  for (std::size_t token = 0; token < kTokens; ++token) {
    const std::size_t row = indices[token];
    for (std::size_t column = 0; column < kHidden; ++column) {
      const float expected = table[row * kHidden + column];
      if (output[token * kHidden + column] != expected) {
        std::fprintf(stderr, "get_rows_f32 mismatch at token=%zu column=%zu\n",
                     token, column);
        return 1;
      }
    }
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    get_rows_f32(table.data(), indices.data(), output.data(), 0, kTokens,
                 kHidden, kHidden, kHidden);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  std::printf("kernel=get_rows_f32\n");
  std::printf("model_shape=TinyLlama_embedding[vocab=32000,hidden=2048,tokens=128]\n");
  std::printf("elements=%zu\n", kOutputElements);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gbytes_s=%.6f\n",
              static_cast<double>(kOutputElements * sizeof(float) * 2U) /
                  milliseconds / 1.0e6);
  return 0;
}
