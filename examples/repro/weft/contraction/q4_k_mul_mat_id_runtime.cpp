#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr std::size_t kExperts = 8;
constexpr std::size_t kTokens = 64;
constexpr std::size_t kSlots = 2;
constexpr std::size_t kRows = 4096;
constexpr std::size_t kInner = 4096;
constexpr std::size_t kBlockExtent = 32;
constexpr std::size_t kColumnTile = 16;
constexpr std::size_t kPackedBlockBytes = 304;
constexpr std::size_t kBlocks = kInner / kBlockExtent;
constexpr std::size_t kRowTiles = kRows / kColumnTile;
constexpr std::size_t kExpertPackedBytes =
    kRowTiles * kBlocks * kPackedBlockBytes;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink;

std::uint16_t f16Bits(float value) {
  _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

float f16Value(const std::uint8_t *address) {
  _Float16 half;
  std::memcpy(&half, address, sizeof(half));
  return static_cast<float>(half);
}

std::vector<std::uint8_t> makePackedWeights() {
  std::vector<std::uint8_t> packed(kExperts * kExpertPackedBytes);
  for (std::size_t expert = 0; expert < kExperts; ++expert)
    for (std::size_t rowTile = 0; rowTile < kRowTiles; ++rowTile)
      for (std::size_t block = 0; block < kBlocks; ++block) {
        std::uint8_t *base =
            packed.data() + expert * kExpertPackedBytes +
            (rowTile * kBlocks + block) * kPackedBlockBytes;
        for (std::size_t lane = 0; lane < kColumnTile; ++lane) {
          const float scale =
              static_cast<float>(1U + ((expert + rowTile + block + lane) % 4U)) /
              64.0F;
          const std::uint16_t bits = f16Bits(scale);
          std::memcpy(base + 2U * lane, &bits, sizeof(bits));
          base[32U + lane] =
              static_cast<std::uint8_t>(6U + ((expert + lane) % 5U));
          for (std::size_t byte = 0; byte < 8U; ++byte) {
            const std::uint8_t low = static_cast<std::uint8_t>(
                (expert + rowTile + block + lane + byte) & 15U);
            const std::uint8_t high = static_cast<std::uint8_t>(
                (3U * expert + rowTile + 5U * block + lane + 7U * byte + 1U) &
                15U);
            base[48U + lane * 8U + byte] =
                static_cast<std::uint8_t>(low | (high << 4U));
            base[48U + 128U + lane * 8U + byte] =
                static_cast<std::uint8_t>(high | (low << 4U));
          }
        }
      }
  return packed;
}

void referenceQuantize(const std::vector<float> &activation,
                       std::vector<float> &scale,
                       std::vector<std::int8_t> &code) {
  for (std::size_t item = 0; item < kTokens * kSlots; ++item)
    for (std::size_t block = 0; block < kBlocks; ++block) {
      float maximum = 0.0F;
      for (std::size_t k = 0; k < kBlockExtent; ++k)
        maximum = std::max(
            maximum,
            std::fabs(activation[item * kInner + block * kBlockExtent + k]));
      scale[item * kBlocks + block] = maximum / 127.0F;
      const float inverse =
          maximum == 0.0F ? 0.0F : 1.0F / scale[item * kBlocks + block];
      for (std::size_t k = 0; k < kBlockExtent; ++k) {
        const float value = std::nearbyint(
            activation[item * kInner + block * kBlockExtent + k] * inverse);
        code[item * kInner + block * kBlockExtent + k] =
            static_cast<std::int8_t>(
                std::max(-128.0F, std::min(127.0F, value)));
      }
    }
}

float referenceElement(const std::vector<std::uint8_t> &packed,
                       const std::vector<float> &scale,
                       const std::vector<std::int8_t> &code,
                       const std::vector<std::uint32_t> &ids,
                       std::size_t token, std::size_t slot, std::size_t row) {
  const std::size_t item = token * kSlots + slot;
  const std::size_t expert = ids[item];
  const std::size_t rowTile = row / kColumnTile;
  const std::size_t lane = row % kColumnTile;
  float accumulator = 0.0F;
  for (std::size_t block = 0; block < kBlocks; ++block) {
    const std::uint8_t *weight =
        packed.data() + expert * kExpertPackedBytes +
        (rowTile * kBlocks + block) * kPackedBlockBytes;
    const float weightScale = f16Value(weight + 2U * lane);
    const std::int32_t zeroPoint = weight[32U + lane];
    std::int32_t dot = 0;
    for (std::size_t half = 0; half < 2U; ++half)
      for (std::size_t byte = 0; byte < 8U; ++byte) {
        const std::uint8_t packedValue =
            weight[48U + half * 128U + lane * 8U + byte];
        const std::size_t lowIndex = half * 16U + byte;
        dot += (static_cast<std::int32_t>(packedValue & 15U) - zeroPoint) *
               code[item * kInner + block * kBlockExtent + lowIndex];
        dot += (static_cast<std::int32_t>(packedValue >> 4U) - zeroPoint) *
               code[item * kInner + block * kBlockExtent + lowIndex + 8U];
      }
    accumulator += static_cast<float>(dot) * scale[item * kBlocks + block] *
                   weightScale;
  }
  return accumulator;
}

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

struct RuntimeArguments {
  std::size_t repetitions;
  const char *hardware;
  const char *scope;
};

RuntimeArguments runtimeArguments(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <repetitions> <hardware> <scope>\n", argv[0]);
    std::exit(2);
  }
  char *end = nullptr;
  const unsigned long value = std::strtoul(argv[1], &end, 10);
  if (*argv[1] == '\0' || *end != '\0' || value == 0)
    std::exit(2);
  return {static_cast<std::size_t>(value), argv[2], argv[3]};
}

} // namespace

int main(int argc, char **argv) {
  const RuntimeArguments runtime = runtimeArguments(argc, argv);
  const std::size_t items = kTokens * kSlots;
  std::vector<float> activation(items * kInner);
  std::vector<std::uint32_t> ids(items);
  for (std::size_t item = 0; item < items; ++item) {
    ids[item] = static_cast<std::uint32_t>((item * 5U + item / 3U) % kExperts);
    for (std::size_t k = 0; k < kInner; ++k) {
      const int value = static_cast<int>((item * 19U + k * 37U) % 257U) - 128;
      activation[item * kInner + k] =
          static_cast<float>(value) / 96.0F +
          static_cast<float>(static_cast<int>(k % 7U) - 3) * 0.0005F;
    }
  }
  const std::vector<std::uint8_t> packed = makePackedWeights();
  std::vector<float> activationScale(items * kBlocks);
  std::vector<std::int8_t> activationCode(items * kInner);
  std::vector<float> referenceScale(items * kBlocks);
  std::vector<std::int8_t> referenceCode(items * kInner);
  std::vector<float> output(items * kRows);
  std::vector<std::uint32_t> expertCounts(kExperts);
  std::vector<std::uint32_t> expertOffsets(kExperts + 1U);
  std::vector<std::uint32_t> expertCursors(kExperts);
  std::vector<std::uint32_t> expertItems(items);
  referenceQuantize(activation, referenceScale, referenceCode);

  q4_k_mul_mat_id(
      activation.data(), packed.data(), ids.data(), output.data(),
      activationScale.data(), activationCode.data(), expertCounts.data(),
      expertOffsets.data(), expertCursors.data(), expertItems.data(), kExperts,
      kTokens, kSlots, kRows, kInner, kExpertPackedBytes, kInner,
      kSlots * kInner, 1, kSlots, kRows, kSlots * kRows);

  std::size_t codeMismatches = 0;
  float scaleError = 0.0F;
  for (std::size_t index = 0; index < activationScale.size(); ++index)
    scaleError =
        std::max(scaleError, std::fabs(activationScale[index] - referenceScale[index]));
  for (std::size_t index = 0; index < activationCode.size(); ++index)
    codeMismatches += activationCode[index] != referenceCode[index];
  const std::size_t sampleTokens[] = {0, kTokens / 2, kTokens - 1};
  const std::size_t sampleSlots[] = {0, kSlots - 1};
  const std::size_t sampleRows[] = {0, 1, kRows / 2, kRows - 1};
  double maxAbsoluteError = 0.0;
  double maxRelativeError = 0.0;
  for (std::size_t token : sampleTokens)
    for (std::size_t slot : sampleSlots)
      for (std::size_t row : sampleRows) {
        const double expected = referenceElement(
            packed, referenceScale, referenceCode, ids, token, slot, row);
        const double actual = output[(token * kSlots + slot) * kRows + row];
        const double absolute = std::fabs(actual - expected);
        const double relative = absolute / std::fmax(1.0, std::fabs(expected));
        maxAbsoluteError = std::fmax(maxAbsoluteError, absolute);
        maxRelativeError = std::fmax(maxRelativeError, relative);
      }
  if (codeMismatches != 0 || scaleError != 0.0F ||
      maxRelativeError > 5.0e-4) {
    std::fprintf(stderr, "q4_k_mul_mat_id mismatch\n");
    return 1;
  }

  for (int warmup = 0; warmup < 3; ++warmup)
    q4_k_mul_mat_id(
        activation.data(), packed.data(), ids.data(), output.data(),
        activationScale.data(), activationCode.data(), expertCounts.data(),
        expertOffsets.data(), expertCursors.data(), expertItems.data(), kExperts,
        kTokens, kSlots, kRows, kInner, kExpertPackedBytes, kInner,
        kSlots * kInner, 1, kSlots, kRows, kSlots * kRows);
  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  for (std::size_t repetition = 0; repetition < runtime.repetitions;
       ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    q4_k_mul_mat_id(
        activation.data(), packed.data(), ids.data(), output.data(),
        activationScale.data(), activationCode.data(), expertCounts.data(),
        expertOffsets.data(), expertCursors.data(), expertItems.data(), kExperts,
        kTokens, kSlots, kRows, kInner, kExpertPackedBytes, kInner,
        kSlots * kInner, 1, kSlots, kRows, kSlots * kRows);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double operations =
      2.0 * static_cast<double>(items * kRows * kInner);
  std::printf("kernel=q4_k_mul_mat_id\n");
  std::printf("hardware=%s\n", runtime.hardware);
  std::printf(
      "model_shape=MoE[tokens=64,slots=2,experts=8,N=4096,K=4096]\n");
  std::printf("persistent_weight=q4_k_n16_k32_304b\n");
  std::printf("scope=%s\n", runtime.scope);
  std::printf("activation_code_mismatches=%zu\n", codeMismatches);
  std::printf("activation_scale_max_absolute_error=%.9g\n", scaleError);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / milliseconds / 1.0e6);
  return 0;
}
