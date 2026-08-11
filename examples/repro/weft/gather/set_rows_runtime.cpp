#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void set_rows_f32(
    float *destination, const float *source, const std::uint32_t *indices,
    std::size_t groups, std::size_t updates, std::size_t width,
    std::size_t destination_group_stride,
    std::size_t destination_row_stride, std::size_t source_group_stride,
    std::size_t source_row_stride);

namespace {

constexpr std::size_t kGroups = 8;
constexpr std::size_t kCapacity = 4096;
constexpr std::size_t kUpdates = 128;
constexpr std::size_t kWidth = 128;
constexpr std::size_t kDestinationElements = kGroups * kCapacity * kWidth;
constexpr std::size_t kSourceElements = kGroups * kUpdates * kWidth;
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
  return samples[samples.size() / 2U];
}

} // namespace

int main() {
  std::vector<float> source(kSourceElements);
  std::vector<std::uint32_t> indices(kUpdates);
  std::vector<float> expected(kDestinationElements, -7.0F);
  std::vector<float> destination(kDestinationElements, -7.0F);
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] =
        static_cast<float>(static_cast<int>(index % 257U) - 128) / 128.0F;
  for (std::size_t update = 0; update < kUpdates; ++update)
    indices[update] = static_cast<std::uint32_t>((update * 31U + 17U) % kCapacity);
  for (std::size_t group = 0; group < kGroups; ++group)
    for (std::size_t update = 0; update < kUpdates; ++update)
      std::copy_n(source.data() + (group * kUpdates + update) * kWidth, kWidth,
                  expected.data() +
                      (group * kCapacity + indices[update]) * kWidth);

  set_rows_f32(destination.data(), source.data(), indices.data(), kGroups,
               kUpdates, kWidth, kCapacity * kWidth, kWidth,
               kUpdates * kWidth, kWidth);
  if (destination != expected) {
    std::fprintf(stderr, "set_rows_f32 result mismatch\n");
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    set_rows_f32(destination.data(), source.data(), indices.data(), kGroups,
                 kUpdates, kWidth, kCapacity * kWidth, kWidth,
                 kUpdates * kWidth, kWidth);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=set_rows_f32\n");
  std::printf(
      "model_shape=kv_cache[heads=8,capacity=4096,updates=128,head_dim=128]\n");
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gbytes_s=%.6f\n",
              static_cast<double>(kSourceElements * sizeof(float) * 2U) /
                  milliseconds / 1.0e6);
  return 0;
}
