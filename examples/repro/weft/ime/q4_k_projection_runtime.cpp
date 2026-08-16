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

constexpr size_t kColumns = 4096;
constexpr size_t kInner = 4096;
constexpr size_t kQ4KExtent = 256;
constexpr size_t kBlockExtent = 32;
constexpr size_t kColumnTile = 16;
constexpr size_t kPackedBlockBytes = 304;

struct CanonicalQ4KBlock {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[12];
  uint8_t qs[128];
};

static_assert(sizeof(CanonicalQ4KBlock) == 144);

size_t phase_rows(const char *phase) {
  if (std::strcmp(phase, "decode") == 0)
    return 1;
  if (std::strcmp(phase, "prefill") == 0)
    return 128;
  return 0;
}

uint16_t f16_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    mantissa |= 0x800000;
    const uint32_t shift = static_cast<uint32_t>(14 - exponent);
    const uint32_t rounded =
        (mantissa + ((UINT32_C(1) << (shift - 1)) - 1) +
         ((mantissa >> shift) & 1)) >>
        shift;
    return static_cast<uint16_t>(sign | rounded);
  }
  uint32_t rounded =
      mantissa + 0x0fff + ((mantissa >> 13) & UINT32_C(1));
  if ((rounded & 0x800000) != 0) {
    rounded = 0;
    ++exponent;
  }
  return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) |
                               static_cast<uint16_t>(rounded >> 13));
}

float f16_value(const uint8_t *bytes) {
  uint16_t bits;
  std::memcpy(&bits, bytes, sizeof(bits));
  const float magnitude =
      std::ldexp(1.0f + static_cast<float>(bits & 0x03ff) / 1024.0f,
                 static_cast<int>((bits >> 10) & 0x1f) - 15);
  return (bits & 0x8000) == 0 ? magnitude : -magnitude;
}

void get_scale_min(size_t group, const uint8_t *packed, uint8_t &scale,
                   uint8_t &minimum) {
  if (group < 4) {
    scale = packed[group] & 63;
    minimum = packed[group + 4] & 63;
    return;
  }
  scale = (packed[group + 4] & 15) | ((packed[group - 4] >> 6) << 4);
  minimum = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4);
}

std::vector<CanonicalQ4KBlock> make_canonical_weights() {
  constexpr size_t superblocks = kInner / kQ4KExtent;
  std::vector<CanonicalQ4KBlock> weights(kColumns * superblocks);
  for (size_t column = 0; column < kColumns; ++column) {
    for (size_t block = 0; block < superblocks; ++block) {
      CanonicalQ4KBlock &weight = weights[column * superblocks + block];
      const float delta = ((column + block) & 1) == 0 ? 0.03125f : 0.015625f;
      weight.d = f16_bits(delta);
      weight.dmin = f16_bits(delta);
      uint8_t scale[8];
      uint8_t minimum[8];
      for (size_t group = 0; group < 8; ++group) {
        scale[group] = static_cast<uint8_t>(1 + (group % 4));
        minimum[group] = static_cast<uint8_t>(2 * scale[group]);
      }
      for (size_t group = 0; group < 4; ++group) {
        weight.scales[group] = static_cast<uint8_t>(
            scale[group] | ((scale[group + 4] >> 4) << 6));
        weight.scales[group + 4] = static_cast<uint8_t>(
            minimum[group] | ((minimum[group + 4] >> 4) << 6));
        weight.scales[group + 8] = static_cast<uint8_t>(
            (scale[group + 4] & 15) | ((minimum[group + 4] & 15) << 4));
      }
      for (size_t byte = 0; byte < sizeof(weight.qs); ++byte) {
        const uint8_t low =
            static_cast<uint8_t>((column * 5 + block * 3 + byte * 7) & 15);
        const uint8_t high =
            static_cast<uint8_t>((column * 11 + block * 13 + byte * 3 + 1) & 15);
        weight.qs[byte] = static_cast<uint8_t>(low | (high << 4));
      }
    }
  }
  return weights;
}

std::vector<uint8_t> repack_q4_k_16x32(
    const std::vector<CanonicalQ4KBlock> &canonical) {
  constexpr size_t superblocks = kInner / kQ4KExtent;
  constexpr size_t blocks = kInner / kBlockExtent;
  std::vector<uint8_t> packed(
      q4_k_projection_ime__packed_weight_elements(kColumns, kInner));
  for (size_t column_group = 0; column_group < kColumns / kColumnTile;
       ++column_group) {
    for (size_t superblock = 0; superblock < superblocks; ++superblock) {
      for (size_t group = 0; group < kQ4KExtent / kBlockExtent; ++group) {
        uint8_t *destination =
            packed.data() +
            ((column_group * blocks + superblock * 8 + group) *
             kPackedBlockBytes);
        for (size_t lane = 0; lane < kColumnTile; ++lane) {
          const CanonicalQ4KBlock &source =
              canonical[(column_group * kColumnTile + lane) * superblocks +
                        superblock];
          uint8_t scale;
          uint8_t minimum;
          get_scale_min(group, source.scales, scale, minimum);
          const float delta = f16_value(reinterpret_cast<const uint8_t *>(&source.d));
          const float delta_min =
              f16_value(reinterpret_cast<const uint8_t *>(&source.dmin));
          const float affine_delta = delta * scale;
          const float affine_minimum = delta_min * minimum;
          const float zero_point = std::nearbyint(affine_minimum / affine_delta);
          const uint8_t zp = static_cast<uint8_t>(
              std::max(0.0f, std::min(15.0f, zero_point)));
          const uint16_t delta_bits = f16_bits(affine_delta);
          std::memcpy(destination + lane * 2, &delta_bits, sizeof(delta_bits));
          destination[32 + lane] = zp;

          const uint8_t *source_quants = source.qs + (group / 2) * 32;
          uint8_t logical[16];
          if ((group & 1) == 0) {
            for (size_t byte = 0; byte < 16; ++byte)
              logical[byte] = static_cast<uint8_t>(
                  (source_quants[byte] & 15) |
                  ((source_quants[byte + 16] & 15) << 4));
          } else {
            for (size_t byte = 0; byte < 16; ++byte)
              logical[byte] = static_cast<uint8_t>(
                  ((source_quants[byte] & 0xf0) >> 4) |
                  (source_quants[byte + 16] & 0xf0));
          }
          for (size_t byte = 0; byte < 8; ++byte) {
            destination[48 + lane * 8 + byte] = static_cast<uint8_t>(
                (logical[byte] & 15) | ((logical[byte + 8] & 15) << 4));
            destination[48 + 128 + lane * 8 + byte] =
                static_cast<uint8_t>(((logical[byte] & 0xf0) >> 4) |
                                     (logical[byte + 8] & 0xf0));
          }
        }
      }
    }
  }
  return packed;
}

void reference_quantize(const std::vector<float> &activation,
                        std::vector<float> &scale,
                        std::vector<int8_t> &code, size_t rows) {
  constexpr size_t blocks = kInner / kBlockExtent;
  const size_t full_rows = (rows / 4) * 4;
  for (size_t row = 0; row < rows; ++row) {
    for (size_t block = 0; block < blocks; ++block) {
      const size_t base = row * kInner + block * kBlockExtent;
      float maximum = 0.0f;
      for (size_t lane = 0; lane < kBlockExtent; ++lane)
        maximum = std::max(maximum, std::fabs(activation[base + lane]));
      const size_t row_group = row - row % 4;
      const size_t scale_index =
          row < full_rows ? row_group * blocks + block * 4 + row % 4
                          : row * blocks + block;
      scale[scale_index] = maximum / 127.0f;
      const float inverse = 1.0f / scale[scale_index];
      for (size_t lane = 0; lane < kBlockExtent; ++lane) {
        const float rounded = std::nearbyint(activation[base + lane] * inverse);
        const size_t code_index =
            row < full_rows
                ? row_group * kInner + block * 128 + (lane / 8) * 32 +
                      (row % 4) * 8 + lane % 8
                : base + lane;
        code[code_index] = static_cast<int8_t>(
            std::max(-128.0f, std::min(127.0f, rounded)));
      }
    }
  }
}

float reference_value(const std::vector<uint8_t> &packed,
                      const std::vector<float> &scale,
                      const std::vector<int8_t> &code, size_t row,
                      size_t column) {
  constexpr size_t blocks = kInner / kBlockExtent;
  const size_t group = column / kColumnTile;
  const size_t lane = column % kColumnTile;
  const size_t full_rows = (scale.size() / blocks / 4) * 4;
  const size_t row_group = row - row % 4;
  float accumulator = 0.0f;
  for (size_t block = 0; block < blocks; ++block) {
    const uint8_t *weight =
        packed.data() + (group * blocks + block) * kPackedBlockBytes;
    const float weight_scale = f16_value(weight + lane * 2);
    const int32_t zero_point = weight[32 + lane];
    int32_t dot = 0;
    for (size_t byte = 0; byte < 16; ++byte) {
      const size_t half = byte / 8;
      const size_t within = byte % 8;
      const uint8_t packed_value =
          weight[48 + half * 128 + lane * 8 + within];
      const size_t low_index = half * 16 + within;
      const auto activation_code = [&](size_t local) {
        const size_t index =
            row < full_rows
                ? row_group * kInner + block * 128 + (local / 8) * 32 +
                      (row % 4) * 8 + local % 8
                : row * kInner + block * kBlockExtent + local;
        return code[index];
      };
      dot += (static_cast<int32_t>(packed_value & 15) - zero_point) *
             activation_code(low_index);
      dot += (static_cast<int32_t>(packed_value >> 4) - zero_point) *
             activation_code(low_index + 8);
    }
    const size_t scale_index =
        row < full_rows ? row_group * blocks + block * 4 + row % 4
                        : row * blocks + block;
    accumulator +=
        static_cast<float>(dot) * scale[scale_index] * weight_scale;
  }
  return accumulator;
}

void evict(std::vector<uint8_t> &buffer) {
  volatile uint64_t sum = 0;
  for (size_t offset = 0; offset < buffer.size(); offset += 64)
    sum += buffer[offset];
  if (sum == UINT64_MAX)
    std::fprintf(stderr, "unreachable eviction sum\n");
}

struct RuntimeArguments {
  const char *phase;
  size_t rows;
  size_t repetitions;
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
  const size_t rows = phase_rows(argv[1]);
  char *end = nullptr;
  const unsigned long repetitions = std::strtoul(argv[2], &end, 10);
  if (rows == 0 || *argv[2] == '\0' || *end != '\0' || repetitions == 0)
    std::exit(2);
  return {argv[1], rows, static_cast<size_t>(repetitions), argv[3], argv[4]};
}

} // namespace

int main(int argc, char **argv) {
  const RuntimeArguments runtime = runtime_arguments_from(argc, argv);
  constexpr size_t blocks = kInner / kBlockExtent;
  std::vector<float> activation(runtime.rows * kInner);
  for (size_t index = 0; index < activation.size(); ++index) {
    const int value =
        static_cast<int>((index * 37 + (index / 17) * 11) % 257) - 128;
    activation[index] =
        static_cast<float>(value) / 96.0f +
        static_cast<float>(static_cast<int>(index % 7) - 3) * 0.0005f;
  }

  const std::vector<CanonicalQ4KBlock> canonical = make_canonical_weights();
  const std::vector<uint8_t> packed = repack_q4_k_16x32(canonical);
  std::vector<float> scale(
      q4_k_projection_ime__activation_scale_elements(runtime.rows, kInner));
  std::vector<int8_t> code(
      q4_k_projection_ime__activation_code_elements(runtime.rows, kInner));
  std::vector<float> output(runtime.rows * kColumns);
  std::vector<float> reference_scale(runtime.rows * blocks);
  std::vector<int8_t> reference_code(runtime.rows * kInner);

  reference_quantize(activation, reference_scale, reference_code, runtime.rows);
  q4_k_projection_ime(activation.data(), packed.data(), output.data(),
                      scale.data(), code.data(), 0, runtime.rows, kColumns,
                      kInner);

  size_t code_mismatches = 0;
  float scale_error = 0.0f;
  for (size_t block = 0; block < scale.size(); ++block)
    scale_error = std::max(scale_error,
                           std::fabs(scale[block] - reference_scale[block]));
  for (size_t k = 0; k < code.size(); ++k)
    code_mismatches += code[k] != reference_code[k];

  const size_t sample_rows[] = {0, runtime.rows / 2, runtime.rows - 1};
  const size_t sample_columns[] = {0, 1, kColumns / 2, kColumns - 1};
  float max_absolute_error = 0.0f;
  float max_relative_error = 0.0f;
  for (size_t row : sample_rows) {
    for (size_t column : sample_columns) {
      const float expected =
          reference_value(packed, reference_scale, reference_code, row, column);
      const float actual = output[row * kColumns + column];
      const float absolute = std::fabs(actual - expected);
      const float relative = absolute / std::max(1.0f, std::fabs(expected));
      max_absolute_error = std::max(max_absolute_error, absolute);
      max_relative_error = std::max(max_relative_error, relative);
    }
  }

  for (size_t warmup = 0; warmup < 3; ++warmup)
    q4_k_projection_ime(activation.data(), packed.data(), output.data(),
                        scale.data(), code.data(), 0, runtime.rows, kColumns,
                        kInner);

  std::vector<uint8_t> eviction(64 * 1024 * 1024, 1);
  std::vector<double> milliseconds;
  milliseconds.reserve(runtime.repetitions);
  for (size_t repetition = 0; repetition < runtime.repetitions; ++repetition) {
    evict(eviction);
    const auto start = std::chrono::steady_clock::now();
    q4_k_projection_ime(activation.data(), packed.data(), output.data(),
                        scale.data(), code.data(), 0, runtime.rows, kColumns,
                        kInner);
    const auto end = std::chrono::steady_clock::now();
    milliseconds.push_back(
        std::chrono::duration<double, std::milli>(end - start).count());
  }
  std::sort(milliseconds.begin(), milliseconds.end());
  const double median_ms = milliseconds[milliseconds.size() / 2];
  const double operations = 2.0 * runtime.rows * kColumns * kInner;

  std::printf("kernel=q4_k_projection_ime\n");
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

  return code_mismatches == 0 && scale_error == 0.0f &&
                 std::isfinite(max_absolute_error) &&
                 max_absolute_error <= 0.01f
             ? 0
             : 1;
}
