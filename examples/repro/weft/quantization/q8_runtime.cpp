#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef WEFT_QUANTIZE_KIND
#error "WEFT_QUANTIZE_KIND is required"
#endif

#ifndef WEFT_QUANTIZE_ENTRY
#error "WEFT_QUANTIZE_ENTRY is required"
#endif

#define WEFT_STRINGIFY_INNER(value) #value
#define WEFT_STRINGIFY(value) WEFT_STRINGIFY_INNER(value)

namespace {

constexpr std::size_t kRows = 128;
constexpr std::size_t kColumns = 14336;
constexpr std::size_t kElements = kRows * kColumns;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

#if WEFT_QUANTIZE_KIND == 2
constexpr std::size_t kBlockSize = 256;
constexpr std::size_t kBlockBytes = 292;
#elif WEFT_QUANTIZE_KIND == 1
constexpr std::size_t kBlockSize = 32;
constexpr std::size_t kBlockBytes = 36;
#else
constexpr std::size_t kBlockSize = 32;
constexpr std::size_t kBlockBytes = 34;
#endif

constexpr std::size_t kBlocks = kElements / kBlockSize;

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

[[maybe_unused]] void storeF16(std::vector<std::uint8_t> &bytes,
                               std::size_t offset, float value) {
  const _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  bytes[offset] = static_cast<std::uint8_t>(bits);
  bytes[offset + 1] = static_cast<std::uint8_t>(bits >> 8);
}

[[maybe_unused]] void storeF32(std::vector<std::uint8_t> &bytes,
                               std::size_t offset, float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  for (std::size_t byte = 0; byte < 4; ++byte)
    bytes[offset + byte] = static_cast<std::uint8_t>(bits >> (8 * byte));
}

[[maybe_unused]] void storeI16(std::vector<std::uint8_t> &bytes,
                               std::size_t offset, std::int16_t value) {
  const std::uint16_t bits = static_cast<std::uint16_t>(value);
  bytes[offset] = static_cast<std::uint8_t>(bits);
  bytes[offset + 1] = static_cast<std::uint8_t>(bits >> 8);
}

std::int8_t quantize(float value) {
  const long rounded = std::lrint(static_cast<double>(value));
  return static_cast<std::int8_t>(std::max(-128L, std::min(127L, rounded)));
}

} // namespace

int main() {
  std::vector<float> input(kElements);
  std::vector<std::uint8_t> output(kBlocks * kBlockBytes);
  std::vector<std::uint8_t> reference(kBlocks * kBlockBytes, 0);
  for (std::size_t index = 0; index < input.size(); ++index)
    input[index] = static_cast<float>(static_cast<int>(index % 4093) - 2046) /
                   257.0F;

  auto invoke = [&]() {
#if WEFT_QUANTIZE_KIND == 0
    WEFT_QUANTIZE_ENTRY(
        input.data(), reinterpret_cast<std::int8_t *>(output.data()), 0, kBlocks);
#else
    WEFT_QUANTIZE_ENTRY(input.data(), output.data(), 0, kBlocks);
#endif
  };
  invoke();

  for (std::size_t block = 0; block < kBlocks; ++block) {
    const std::size_t inputBase = block * kBlockSize;
    const std::size_t outputBase = block * kBlockBytes;
#if WEFT_QUANTIZE_KIND == 2
    float maximum = 0.0F;
    float extreme = 0.0F;
    for (std::size_t lane = 0; lane < kBlockSize; ++lane) {
      const float magnitude = std::fabs(input[inputBase + lane]);
      if (magnitude > maximum) {
        maximum = magnitude;
        extreme = input[inputBase + lane];
      }
    }
    const float inverse = maximum == 0.0F ? 0.0F : -127.0F / extreme;
    const float scale = maximum == 0.0F ? 0.0F : 1.0F / inverse;
    storeF32(reference, outputBase, scale);
    for (std::size_t group = 0; group < 16; ++group) {
      std::int32_t sum = 0;
      for (std::size_t lane = 0; lane < 16; ++lane) {
        const std::int8_t code =
            quantize(input[inputBase + group * 16 + lane] * inverse);
        reference[outputBase + 4 + group * 16 + lane] =
            static_cast<std::uint8_t>(code);
        sum += code;
      }
      storeI16(reference, outputBase + 260 + group * 2,
               static_cast<std::int16_t>(sum));
    }
#else
    float maximum = 0.0F;
    for (std::size_t lane = 0; lane < kBlockSize; ++lane)
      maximum =
          std::max(maximum, std::fabs(input[inputBase + lane]));
    const float scale = maximum / 127.0F;
    const float inverse = scale == 0.0F ? 0.0F : 1.0F / scale;
    storeF16(reference, outputBase, scale);
#if WEFT_QUANTIZE_KIND == 1
    std::int32_t sum = 0;
#endif
    for (std::size_t lane = 0; lane < kBlockSize; ++lane) {
      const std::int8_t code = quantize(input[inputBase + lane] * inverse);
#if WEFT_QUANTIZE_KIND == 1
      reference[outputBase + 4 + lane] = static_cast<std::uint8_t>(code);
#else
      reference[outputBase + 2 + lane] = static_cast<std::uint8_t>(code);
#endif
#if WEFT_QUANTIZE_KIND == 1
      sum += code;
#endif
    }
#if WEFT_QUANTIZE_KIND == 1
    storeF16(reference, outputBase + 2, static_cast<float>(sum) * scale);
#endif
#endif
  }

  for (std::size_t index = 0; index < output.size(); ++index) {
    if (output[index] != reference[index]) {
      std::fprintf(stderr,
                   "%s byte mismatch at index=%zu: actual=%u expected=%u\n",
                   WEFT_STRINGIFY(WEFT_QUANTIZE_ENTRY), index,
                   static_cast<unsigned>(output[index]),
                   static_cast<unsigned>(reference[index]));
      return 1;
    }
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
  std::printf("kernel=%s\n", WEFT_STRINGIFY(WEFT_QUANTIZE_ENTRY));
  std::printf("model_shape=M=128;K=14336\n");
  std::printf("elements=%zu\n", kElements);
  std::printf("max_absolute_error=0\n");
  std::printf("max_relative_error=0\n");
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kElements) / milliseconds / 1000.0);
  return 0;
}
