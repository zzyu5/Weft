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

constexpr std::size_t kColumns = 4096;
constexpr std::size_t kInner = 4096;
constexpr std::size_t kBlockExtent = 32;
constexpr std::size_t kColumnTile = 16;
constexpr std::size_t kPackedBlockBytes = 304;

struct CanonicalQ41Block {
  std::uint16_t d;
  std::uint16_t m;
  std::uint8_t qs[16];
};

static_assert(sizeof(CanonicalQ41Block) == 20);

std::size_t phase_rows(const char *phase) {
  if (std::strcmp(phase, "decode") == 0)
    return 1;
  if (std::strcmp(phase, "prefill") == 0)
    return 128;
  return 0;
}

std::uint16_t f16_bits(float value) {
  const _Float16 half = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

float f16_value(const std::uint8_t *bytes) {
  std::uint16_t bits = 0;
  std::memcpy(&bits, bytes, sizeof(bits));
  _Float16 half = 0;
  std::memcpy(&half, &bits, sizeof(bits));
  return static_cast<float>(half);
}

std::vector<CanonicalQ41Block> make_canonical_weights() {
  constexpr std::size_t blocks = kInner / kBlockExtent;
  std::vector<CanonicalQ41Block> weights(kColumns * blocks);
  for (std::size_t column = 0; column < kColumns; ++column) {
    for (std::size_t block = 0; block < blocks; ++block) {
      CanonicalQ41Block &weight = weights[column * blocks + block];
      const float scale = ((column + block) & 1U) == 0 ? 0.03125F : 0.015625F;
      weight.d = f16_bits(scale);
      weight.m = f16_bits(-2.0F * scale);
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

std::vector<std::uint8_t> repack_q4_1_16x32(
    const std::vector<CanonicalQ41Block> &canonical) {
  constexpr std::size_t blocks = kInner / kBlockExtent;
  std::vector<std::uint8_t> packed(
      q4_1_projection_ime__packed_weight_elements(kColumns, kInner));
  for (std::size_t column_group = 0;
       column_group < kColumns / kColumnTile; ++column_group) {
    for (std::size_t block = 0; block < blocks; ++block) {
      std::uint8_t *destination =
          packed.data() +
          (column_group * blocks + block) * kPackedBlockBytes;
      for (std::size_t lane = 0; lane < kColumnTile; ++lane) {
        const CanonicalQ41Block &source =
            canonical[(column_group * kColumnTile + lane) * blocks + block];
        std::memcpy(destination + lane * 2, &source.d, sizeof(source.d));
        const float scale = f16_value(
            reinterpret_cast<const std::uint8_t *>(&source.d));
        const float minimum = f16_value(
            reinterpret_cast<const std::uint8_t *>(&source.m));
        destination[32 + lane] = static_cast<std::uint8_t>(
            std::nearbyint(-minimum / scale));
        for (std::size_t byte = 0; byte < 8; ++byte) {
          destination[48 + lane * 8 + byte] = static_cast<std::uint8_t>(
              (source.qs[byte] & 15U) |
              ((source.qs[byte + 8] & 15U) << 4));
          destination[48 + 128 + lane * 8 + byte] =
              static_cast<std::uint8_t>(((source.qs[byte] >> 4) & 15U) |
                                        (source.qs[byte + 8] & 0xf0U));
        }
      }
    }
  }
  return packed;
}

void reference_quantize(const std::vector<float> &activation,
                        std::vector<float> &scale,
                        std::vector<std::int8_t> &code,
                        std::size_t rows) {
  constexpr std::size_t blocks = kInner / kBlockExtent;
  const std::size_t full_rows = (rows / 4) * 4;
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t block = 0; block < blocks; ++block) {
      const std::size_t base = row * kInner + block * kBlockExtent;
      float maximum = 0.0F;
      for (std::size_t lane = 0; lane < kBlockExtent; ++lane)
        maximum = std::max(maximum, std::fabs(activation[base + lane]));
      const std::size_t row_group = row - row % 4;
      const std::size_t scale_index =
          row < full_rows ? row_group * blocks + block * 4 + row % 4
                          : row * blocks + block;
      scale[scale_index] = maximum / 127.0F;
      const float inverse = 1.0F / scale[scale_index];
      for (std::size_t lane = 0; lane < kBlockExtent; ++lane) {
        const float rounded = std::nearbyint(activation[base + lane] * inverse);
        const std::size_t code_index =
            row < full_rows
                ? row_group * kInner + block * 128 + (lane / 8) * 32 +
                      (row % 4) * 8 + lane % 8
                : base + lane;
        code[code_index] = static_cast<std::int8_t>(
            std::max(-128.0F, std::min(127.0F, rounded)));
      }
    }
  }
}

float reference_value(const std::vector<std::uint8_t> &packed,
                      const std::vector<float> &scale,
                      const std::vector<std::int8_t> &code,
                      std::size_t row, std::size_t column) {
  constexpr std::size_t blocks = kInner / kBlockExtent;
  const std::size_t column_group = column / kColumnTile;
  const std::size_t lane = column % kColumnTile;
  const std::size_t full_rows = (scale.size() / blocks / 4) * 4;
  const std::size_t row_group = row - row % 4;
  float accumulator = 0.0F;
  for (std::size_t block = 0; block < blocks; ++block) {
    const std::uint8_t *weight =
        packed.data() +
        (column_group * blocks + block) * kPackedBlockBytes;
    const float weight_scale = f16_value(weight + lane * 2);
    const std::int32_t zero_point = weight[32 + lane];
    std::int32_t dot = 0;
    for (std::size_t byte = 0; byte < 16; ++byte) {
      const std::size_t half = byte / 8;
      const std::size_t within = byte % 8;
      const std::uint8_t packed_value =
          weight[48 + half * 128 + lane * 8 + within];
      const std::size_t low_index = half * 16 + within;
      const auto activation_code = [&](std::size_t local) {
        const std::size_t index =
            row < full_rows
                ? row_group * kInner + block * 128 + (local / 8) * 32 +
                      (row % 4) * 8 + local % 8
                : row * kInner + block * kBlockExtent + local;
        return code[index];
      };
      dot += (static_cast<std::int32_t>(packed_value & 15U) - zero_point) *
             activation_code(low_index);
      dot += (static_cast<std::int32_t>(packed_value >> 4) - zero_point) *
             activation_code(low_index + 8);
    }
    const std::size_t scale_index =
        row < full_rows ? row_group * blocks + block * 4 + row % 4
                        : row * blocks + block;
    accumulator += dot * scale[scale_index] * weight_scale;
  }
  return accumulator;
}

void evict(std::vector<std::uint8_t> &buffer) {
  volatile std::uint64_t sum = 0;
  for (std::size_t offset = 0; offset < buffer.size(); offset += 64)
    sum += buffer[offset];
  if (sum == UINT64_MAX)
    std::fprintf(stderr, "unreachable eviction sum\n");
}

struct RuntimeArguments {
  const char *phase;
  std::size_t rows;
  std::size_t repetitions;
  const char *hardware;
  const char *scope;
};

RuntimeArguments runtime_arguments_from(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr,
                 "usage: %s <decode|prefill> <repetitions> <hardware> <scope>\n",
                 argv[0]);
    std::exit(2);
  }
  const std::size_t rows = phase_rows(argv[1]);
  char *end = nullptr;
  const unsigned long repetitions = std::strtoul(argv[2], &end, 10);
  if (rows == 0 || *argv[2] == '\0' || *end != '\0' || repetitions == 0)
    std::exit(2);
  return {argv[1], rows, static_cast<std::size_t>(repetitions), argv[3],
          argv[4]};
}

} // namespace

int main(int argc, char **argv) {
  const RuntimeArguments runtime = runtime_arguments_from(argc, argv);
  constexpr std::size_t blocks = kInner / kBlockExtent;
  std::vector<float> activation(runtime.rows * kInner);
  for (std::size_t index = 0; index < activation.size(); ++index) {
    const int value =
        static_cast<int>((index * 37 + (index / 17) * 11) % 257) - 128;
    activation[index] = static_cast<float>(value) / 96.0F +
                        static_cast<float>(static_cast<int>(index % 7) - 3) *
                            0.0005F;
  }

  const std::vector<CanonicalQ41Block> canonical = make_canonical_weights();
  const std::vector<std::uint8_t> packed = repack_q4_1_16x32(canonical);
  std::vector<float> scale(
      q4_1_projection_ime__activation_scale_elements(runtime.rows, kInner));
  std::vector<std::int8_t> code(
      q4_1_projection_ime__activation_code_elements(runtime.rows, kInner));
  std::vector<float> output(runtime.rows * kColumns);
  std::vector<float> reference_scale(runtime.rows * blocks);
  std::vector<std::int8_t> reference_code(runtime.rows * kInner);

  reference_quantize(activation, reference_scale, reference_code, runtime.rows);
  q4_1_projection_ime(activation.data(), packed.data(), output.data(),
                      scale.data(), code.data(), 0, runtime.rows, kColumns,
                      kInner);

  std::size_t code_mismatches = 0;
  float scale_error = 0.0F;
  for (std::size_t index = 0; index < scale.size(); ++index)
    scale_error =
        std::max(scale_error, std::fabs(scale[index] - reference_scale[index]));
  for (std::size_t index = 0; index < code.size(); ++index)
    code_mismatches += code[index] != reference_code[index];

  const std::size_t sample_rows[] = {0, runtime.rows / 2, runtime.rows - 1};
  const std::size_t sample_columns[] = {0, 1, kColumns / 2, kColumns - 1};
  float max_absolute_error = 0.0F;
  float max_relative_error = 0.0F;
  for (std::size_t row : sample_rows) {
    for (std::size_t column : sample_columns) {
      const float expected =
          reference_value(packed, reference_scale, reference_code, row, column);
      const float actual = output[row * kColumns + column];
      const float absolute = std::fabs(actual - expected);
      const float relative = absolute / std::max(1.0F, std::fabs(expected));
      max_absolute_error = std::max(max_absolute_error, absolute);
      max_relative_error = std::max(max_relative_error, relative);
    }
  }

  for (std::size_t warmup = 0; warmup < 3; ++warmup)
    q4_1_projection_ime(activation.data(), packed.data(), output.data(),
                        scale.data(), code.data(), 0, runtime.rows, kColumns,
                        kInner);

  std::vector<std::uint8_t> eviction(64U * 1024U * 1024U, 1);
  std::vector<double> milliseconds;
  milliseconds.reserve(runtime.repetitions);
  for (std::size_t repetition = 0; repetition < runtime.repetitions;
       ++repetition) {
    evict(eviction);
    const auto start = std::chrono::steady_clock::now();
    q4_1_projection_ime(activation.data(), packed.data(), output.data(),
                        scale.data(), code.data(), 0, runtime.rows, kColumns,
                        kInner);
    const auto end = std::chrono::steady_clock::now();
    milliseconds.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }
  std::sort(milliseconds.begin(), milliseconds.end());
  const double median_ms = milliseconds[milliseconds.size() / 2];
  const double operations = 2.0 * runtime.rows * kColumns * kInner;

  std::printf("kernel=q4_1_projection_ime\n");
  std::printf("phase=%s\n", runtime.phase);
  std::printf("hardware=%s\n", runtime.hardware);
  std::printf("M=%zu,N=%zu,K=%zu\n", runtime.rows, kColumns, kInner);
  std::printf("persistent_weight=affine_i4_n16_k32_304b\n");
  std::printf("scope=%s\n", runtime.scope);
  std::printf("cold_protocol=64MiB-evict-then-single-kernel\n");
  std::printf("activation_code_mismatches=%zu\n", code_mismatches);
  std::printf("activation_scale_max_absolute_error=%.9g\n", scale_error);
  std::printf("max_absolute_error=%.9g\n", max_absolute_error);
  std::printf("max_relative_error=%.9g\n", max_relative_error);
  std::printf("repetitions=%zu\n", runtime.repetitions);
  std::printf("median_ms=%.6f\n", median_ms);
  std::printf("gop_s=%.6f\n", operations / (median_ms * 1.0e6));

  return code_mismatches == 0 && scale_error == 0.0F &&
                 std::isfinite(max_absolute_error) &&
                 max_absolute_error <= 0.05F
             ? 0
             : 1;
}
