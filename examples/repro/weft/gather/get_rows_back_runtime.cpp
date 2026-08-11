#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void get_rows_back_f32(
    const float *source, const std::uint32_t *indices, float *output,
    std::size_t rows, std::size_t items, std::size_t hidden,
    std::size_t source_stride, std::size_t output_stride);

namespace {

constexpr std::size_t kRows = 32000;
constexpr std::size_t kItems = 128;
constexpr std::size_t kHidden = 2048;
constexpr std::size_t kOutputElements = kRows * kHidden;
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
  std::vector<float> source(kItems * kHidden);
  std::vector<std::uint32_t> indices(kItems);
  std::vector<float> expected(kOutputElements, 0.0F);
  std::vector<float> output(kOutputElements);
  for (std::size_t item = 0; item < kItems; ++item) {
    indices[item] = static_cast<std::uint32_t>((item * 7919U) % 97U);
    for (std::size_t column = 0; column < kHidden; ++column) {
      const float value =
          static_cast<float>(static_cast<int>((item + column) % 257U) - 128) /
          256.0F;
      source[item * kHidden + column] = value;
      expected[static_cast<std::size_t>(indices[item]) * kHidden + column] +=
          value;
    }
  }

  get_rows_back_f32(source.data(), indices.data(), output.data(), kRows, kItems,
                    kHidden, kHidden, kHidden);
  if (output != expected) {
    std::fprintf(stderr, "get_rows_back_f32 result mismatch\n");
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(5);
  for (int repetition = 0; repetition < 5; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    get_rows_back_f32(source.data(), indices.data(), output.data(), kRows,
                      kItems, kHidden, kHidden, kHidden);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=get_rows_back_f32\n");
  std::printf("model_shape=embedding_backward[rows=32000,items=128,hidden=2048]\n");
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("zero_fill_gbytes_s=%.6f\n",
              static_cast<double>(kOutputElements * sizeof(float)) /
                  milliseconds / 1.0e6);
  return 0;
}
