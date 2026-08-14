#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kTokens = 128;
constexpr std::size_t kSlots = 8;
constexpr std::size_t kExperts = 256;
constexpr std::size_t kHidden = 4096;
constexpr std::size_t kElements = kTokens * kSlots * kHidden;
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
  std::vector<float> a(kElements);
  std::vector<float> b(kExperts * kHidden);
  std::vector<std::uint32_t> ids(kTokens * kSlots);
  std::vector<float> output(kElements);
  for (std::size_t index = 0; index < a.size(); ++index)
    a[index] = static_cast<float>(static_cast<int>(index % 257U) - 128) /
               128.0F;
  for (std::size_t index = 0; index < b.size(); ++index)
    b[index] = static_cast<float>(static_cast<int>(index % 127U) - 63) /
               64.0F;
  for (std::size_t token = 0; token < kTokens; ++token)
    for (std::size_t slot = 0; slot < kSlots; ++slot)
      ids[token * kSlots + slot] =
          static_cast<std::uint32_t>((token * 37U + slot * 53U) % kExperts);

  add_id_f32(a.data(), b.data(), ids.data(), output.data(), 0, kTokens,
             kSlots, kHidden, kHidden, kSlots * kHidden, kHidden, kSlots,
             kHidden, kSlots * kHidden);
  for (std::size_t token = 0; token < kTokens; ++token)
    for (std::size_t slot = 0; slot < kSlots; ++slot)
      for (std::size_t column = 0; column < kHidden; ++column) {
        const std::size_t offset =
            token * kSlots * kHidden + slot * kHidden + column;
        const float expected =
            a[offset] + b[static_cast<std::size_t>(ids[token * kSlots + slot]) *
                              kHidden +
                          column];
        if (output[offset] != expected) {
          std::fprintf(stderr,
                       "add_id_f32 mismatch at token=%zu slot=%zu column=%zu\n",
                       token, slot, column);
          return 1;
        }
      }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    add_id_f32(a.data(), b.data(), ids.data(), output.data(), 0, kTokens,
               kSlots, kHidden, kHidden, kSlots * kHidden, kHidden, kSlots,
               kHidden, kSlots * kHidden);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=add_id_f32\n");
  std::printf("model_shape=moe_combine[tokens=128,slots=8,experts=256,hidden=4096]\n");
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gbytes_s=%.6f\n",
              static_cast<double>(kElements * sizeof(float) * 3U) /
                  milliseconds / 1.0e6);
  return 0;
}
