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

constexpr std::size_t kRows = 1;
constexpr std::size_t kColumns = 4096;
constexpr std::size_t kInner = 4096;
constexpr std::size_t kBlockExtent = 32;
constexpr std::size_t kColumnTile = 16;
constexpr std::size_t kPackedBlockBytes = 288;

struct CanonicalQ40Block {
  std::uint16_t d;
  std::uint8_t qs[16];
};

static_assert(sizeof(CanonicalQ40Block) == 18);

std::uint16_t f16_bits(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000);
  std::int32_t exponent = static_cast<std::int32_t>((bits >> 23) & 0xff) - 127 + 15;
  std::uint32_t mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    mantissa |= 0x800000;
    const std::uint32_t shift = static_cast<std::uint32_t>(14 - exponent);
    const std::uint32_t rounded =
        (mantissa + ((UINT32_C(1) << (shift - 1)) - 1) +
         ((mantissa >> shift) & 1)) >>
        shift;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  std::uint32_t rounded =
      mantissa + 0x0fff + ((mantissa >> 13) & UINT32_C(1));
  if ((rounded & 0x800000) != 0) {
    rounded = 0;
    ++exponent;
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint16_t>(exponent) << 10) |
      static_cast<std::uint16_t>(rounded >> 13));
}

float f16_value(const std::uint8_t *bytes) {
  std::uint16_t bits;
  std::memcpy(&bits, bytes, sizeof(bits));
  const float magnitude =
      std::ldexp(1.0F + static_cast<float>(bits & 0x03ff) / 1024.0F,
                 static_cast<int>((bits >> 10) & 0x1f) - 15);
  return (bits & 0x8000) == 0 ? magnitude : -magnitude;
}

std::vector<CanonicalQ40Block> make_canonical_weights() {
  constexpr std::size_t blocks = kInner / kBlockExtent;
  std::vector<CanonicalQ40Block> weights(kColumns * blocks);
  for (std::size_t column = 0; column < kColumns; ++column) {
    for (std::size_t block = 0; block < blocks; ++block) {
      CanonicalQ40Block &weight = weights[column * blocks + block];
      const float scale = ((column + block) & 1) == 0 ? -0.03125F : -0.015625F;
      weight.d = f16_bits(scale);
      for (std::size_t byte = 0; byte < 16; ++byte) {
        const std::uint8_t low = static_cast<std::uint8_t>(
            (column * 5 + block * 3 + byte * 7) & 15U);
        const std::uint8_t high = static_cast<std::uint8_t>(
            (column * 11 + block * 13 + byte * 3 + 1) & 15U);
        weight.qs[byte] = static_cast<std::uint8_t>(low | (high << 4));
      }
    }
  }
  return weights;
}

std::vector<std::uint8_t> repack_q4_0_16x32(
    const std::vector<CanonicalQ40Block> &canonical) {
  constexpr std::size_t blocks = kInner / kBlockExtent;
  std::vector<std::uint8_t> packed((kColumns / kColumnTile) * blocks *
                                   kPackedBlockBytes);
  for (std::size_t columnGroup = 0; columnGroup < kColumns / kColumnTile;
       ++columnGroup) {
    for (std::size_t block = 0; block < blocks; ++block) {
      std::uint8_t *destination =
          packed.data() +
          (columnGroup * blocks + block) * kPackedBlockBytes;
      for (std::size_t lane = 0; lane < kColumnTile; ++lane) {
        const CanonicalQ40Block &source =
            canonical[(columnGroup * kColumnTile + lane) * blocks + block];
        std::memcpy(destination + lane * 2, &source.d, sizeof(source.d));
        for (std::size_t byte = 0; byte < 8; ++byte) {
          destination[32 + lane * 8 + byte] = static_cast<std::uint8_t>(
              (source.qs[byte] & 15) |
              ((source.qs[byte + 8] & 15) << 4));
          destination[32 + 128 + lane * 8 + byte] =
              static_cast<std::uint8_t>(((source.qs[byte] >> 4) & 15) |
                                        (source.qs[byte + 8] & 0xf0));
        }
      }
    }
  }
  return packed;
}

void reference_quantize(const std::vector<float> &activation,
                        std::vector<float> &scale,
                        std::vector<std::int8_t> &code) {
  for (std::size_t block = 0; block < kInner / kBlockExtent; ++block) {
    float maximum = 0.0F;
    for (std::size_t k = 0; k < kBlockExtent; ++k)
      maximum = std::max(
          maximum, std::fabs(activation[block * kBlockExtent + k]));
    scale[block] = maximum / 127.0F;
    const float inverse = 1.0F / scale[block];
    for (std::size_t k = 0; k < kBlockExtent; ++k) {
      const float rounded =
          std::nearbyint(activation[block * kBlockExtent + k] * inverse);
      code[block * kBlockExtent + k] = static_cast<std::int8_t>(
          std::max(-128.0F, std::min(127.0F, rounded)));
    }
  }
}

void reference_projection(const std::vector<std::uint8_t> &packed,
                          const std::vector<float> &scale,
                          const std::vector<std::int8_t> &code,
                          std::vector<float> &output) {
  constexpr std::size_t blocks = kInner / kBlockExtent;
  for (std::size_t column = 0; column < kColumns; ++column) {
    const std::size_t group = column / kColumnTile;
    const std::size_t lane = column % kColumnTile;
    float accumulator = 0.0F;
    for (std::size_t block = 0; block < blocks; ++block) {
      const std::uint8_t *weight =
          packed.data() + (group * blocks + block) * kPackedBlockBytes;
      const float weightScale = f16_value(weight + lane * 2);
      std::int32_t dot = 0;
      for (std::size_t byte = 0; byte < 16; ++byte) {
        const std::size_t half = byte / 8;
        const std::size_t within = byte % 8;
        const std::uint8_t packedValue =
            weight[32 + half * 128 + lane * 8 + within];
        const std::size_t lowIndex = half * 16 + within;
        dot += (static_cast<std::int32_t>(packedValue & 15) - 8) *
               code[block * kBlockExtent + lowIndex];
        dot += (static_cast<std::int32_t>(packedValue >> 4) - 8) *
               code[block * kBlockExtent + lowIndex + 8];
      }
      accumulator += static_cast<float>(dot) * scale[block] * weightScale;
    }
    output[column] = accumulator;
  }
}

void evict(std::vector<std::uint8_t> &buffer) {
  volatile std::uint64_t sum = 0;
  for (std::size_t offset = 0; offset < buffer.size(); offset += 64)
    sum += buffer[offset];
  if (sum == UINT64_MAX)
    std::fprintf(stderr, "unreachable eviction sum\n");
}

struct RuntimeArguments {
  std::size_t repetitions;
  const char *hardware;
  const char *scope;
};

RuntimeArguments runtime_arguments_from(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <repetitions> <hardware> <scope>\n", argv[0]);
    std::exit(2);
  }
  char *end = nullptr;
  const long value = std::strtol(argv[1], &end, 10);
  if (*argv[1] == '\0' || *end != '\0' || value <= 0)
    std::exit(2);
  return {static_cast<std::size_t>(value), argv[2], argv[3]};
}

} // namespace

int main(int argc, char **argv) {
  const RuntimeArguments runtime = runtime_arguments_from(argc, argv);
  std::vector<float> activation(kInner);
  for (std::size_t k = 0; k < kInner; ++k) {
    const int value = static_cast<int>((k * 37 + (k / 17) * 11) % 257) - 128;
    activation[k] = static_cast<float>(value) / 96.0F +
                    static_cast<float>(static_cast<int>(k % 7) - 3) * 0.0005F;
  }

  const std::vector<CanonicalQ40Block> canonical = make_canonical_weights();
  const std::vector<std::uint8_t> packed = repack_q4_0_16x32(canonical);
  std::vector<float> scale(kInner / kBlockExtent);
  std::vector<std::int8_t> code(kInner);
  std::vector<float> output(kColumns);
  std::vector<float> referenceScale(kInner / kBlockExtent);
  std::vector<std::int8_t> referenceCode(kInner);
  std::vector<float> reference(kColumns);

  reference_quantize(activation, referenceScale, referenceCode);
  reference_projection(packed, referenceScale, referenceCode, reference);
  q4_0_projection_ime(activation.data(), packed.data(), output.data(),
                      scale.data(), code.data(), 0, kRows, kColumns, kInner);

  std::size_t codeMismatches = 0;
  float scaleError = 0.0F;
  for (std::size_t block = 0; block < scale.size(); ++block)
    scaleError = std::max(
        scaleError, std::fabs(scale[block] - referenceScale[block]));
  for (std::size_t k = 0; k < code.size(); ++k)
    codeMismatches += code[k] != referenceCode[k];

  float maxAbsoluteError = 0.0F;
  float maxRelativeError = 0.0F;
  for (std::size_t column = 0; column < kColumns; ++column) {
    const float absolute = std::fabs(output[column] - reference[column]);
    const float relative =
        absolute / std::max(1.0F, std::fabs(reference[column]));
    maxAbsoluteError = std::max(maxAbsoluteError, absolute);
    maxRelativeError = std::max(maxRelativeError, relative);
  }
  for (std::size_t warmup = 0; warmup < 3; ++warmup)
    q4_0_projection_ime(activation.data(), packed.data(), output.data(),
                        scale.data(), code.data(), 0, kRows, kColumns, kInner);

  std::vector<std::uint8_t> eviction(64U * 1024U * 1024U, 1);
  std::vector<double> milliseconds;
  milliseconds.reserve(runtime.repetitions);
  for (std::size_t repetition = 0; repetition < runtime.repetitions;
       ++repetition) {
    evict(eviction);
    const auto start = std::chrono::steady_clock::now();
    q4_0_projection_ime(activation.data(), packed.data(), output.data(),
                        scale.data(), code.data(), 0, kRows, kColumns, kInner);
    const auto end = std::chrono::steady_clock::now();
    milliseconds.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }
  std::sort(milliseconds.begin(), milliseconds.end());
  const double medianMs = milliseconds[milliseconds.size() / 2];
  const double operations = 2.0 * kRows * kColumns * kInner;

  std::printf("kernel=q4_0_projection_ime\n");
  std::printf("hardware=%s\n", runtime.hardware);
  std::printf("M=%zu,N=%zu,K=%zu\n", kRows, kColumns, kInner);
  std::printf("persistent_weight=q4_0_n16_k32_288b\n");
  std::printf("scope=%s\n", runtime.scope);
  std::printf("cold_protocol=64MiB-evict-then-single-kernel\n");
  std::printf("activation_code_mismatches=%zu\n", codeMismatches);
  std::printf("activation_scale_max_absolute_error=%.9g\n", scaleError);
  std::printf("max_absolute_error=%.9g\n", maxAbsoluteError);
  std::printf("max_relative_error=%.9g\n", maxRelativeError);
  std::printf("repetitions=%zu\n", runtime.repetitions);
  std::printf("median_ms=%.6f\n", medianMs);
  std::printf("gop_s=%.6f\n", operations / (medianMs * 1.0e6));

  return codeMismatches == 0 && scaleError == 0.0F &&
                 std::isfinite(maxAbsoluteError) && maxAbsoluteError <= 0.05F
             ? 0
             : 1;
}
