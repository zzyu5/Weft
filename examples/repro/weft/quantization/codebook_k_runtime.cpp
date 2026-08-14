#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#ifndef WEFT_QUANT_KIND
#error "WEFT_QUANT_KIND must name one K-quant dot kernel"
#endif

#if WEFT_QUANT_KIND == 2
using WeightBlock = block_iq2_s;
static constexpr const char *kernel_name = "iq2_S_q8_K";
#elif WEFT_QUANT_KIND == 3
using WeightBlock = block_iq3_s;
static constexpr const char *kernel_name = "iq3_S_q8_K";
#elif WEFT_QUANT_KIND == 1
using WeightBlock = block_iq1_m;
static constexpr const char *kernel_name = "iq1_M_q8_K";
#elif WEFT_QUANT_KIND == 6
using WeightBlock = block_q6_K;
static constexpr const char *kernel_name = "q6_K_q8_K";
#else
#error "unsupported WEFT_QUANT_KIND"
#endif

static double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

static __attribute__((unused)) uint16_t half_bits(float value) {
  _Float16 half = static_cast<_Float16>(value);
  uint16_t bits;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

static float half_value(ggml_half bits) {
  _Float16 half;
  std::memcpy(&half, &bits, sizeof(bits));
  return static_cast<float>(half);
}

static float run_kernel(const WeightBlock *weight, const block_q8_K *activation,
                        size_t blocks) {
#if WEFT_QUANT_KIND == 2
  return iq2_S_q8_K(reinterpret_cast<const uint8_t *>(weight),
                    reinterpret_cast<const uint8_t *>(activation), blocks);
#elif WEFT_QUANT_KIND == 3
  return iq3_S_q8_K(reinterpret_cast<const uint8_t *>(weight),
                    reinterpret_cast<const uint8_t *>(activation), blocks);
#elif WEFT_QUANT_KIND == 1
  return iq1_M_q8_K(reinterpret_cast<const uint8_t *>(weight),
                    reinterpret_cast<const uint8_t *>(activation), blocks);
#else
  return q6_K_q8_K(reinterpret_cast<const uint8_t *>(weight),
                   reinterpret_cast<const uint8_t *>(activation), blocks);
#endif
}

static float reference_block(const WeightBlock &x, const block_q8_K &y) {
#if WEFT_QUANT_KIND == 2
  const int8_t *q8 = y.qs;
  const uint8_t *codes = x.qs;
  const uint8_t *signs = x.qs + 32;
  int32_t sum = 0;
  for (size_t group = 0; group < 8; ++group) {
    int32_t first = 0;
    int32_t second = 0;
    for (size_t vector = 0; vector < 4; ++vector) {
      const uint16_t index = codes[group * 4 + vector] |
          ((static_cast<uint16_t>(x.qh[group]) << (8 - 2 * vector)) & 0x300);
      const int8_t *grid = reinterpret_cast<const int8_t *>(iq2s_grid + index);
      for (size_t lane = 0; lane < 8; ++lane) {
        const int sign = signs[group * 4 + vector] & (1u << lane) ? -1 : 1;
        const int product = q8[group * 32 + vector * 8 + lane] * grid[lane] * sign;
        (vector < 2 ? first : second) += product;
      }
    }
    sum += first * (1 + 2 * (x.scales[group] & 15));
    sum += second * (1 + 2 * (x.scales[group] >> 4));
  }
  return 0.125f * half_value(x.d) * y.d * static_cast<float>(sum);
#elif WEFT_QUANT_KIND == 3
  int32_t sum = 0;
  for (size_t group = 0; group < 8; ++group) {
    int32_t dot = 0;
    for (size_t vector = 0; vector < 8; ++vector) {
      const uint16_t index = x.qs[group * 8 + vector] |
          ((static_cast<uint16_t>(x.qh[group]) << (8 - vector)) & 0x100);
      const int8_t *grid = reinterpret_cast<const int8_t *>(iq3s_grid + index);
      const uint8_t signs = x.signs[group * 4 + vector / 2];
      for (size_t lane = 0; lane < 4; ++lane) {
        const size_t sign_bit = (vector & 1) * 4 + lane;
        const int sign = signs & (1u << sign_bit) ? -1 : 1;
        dot += y.qs[group * 32 + vector * 4 + lane] * grid[lane] * sign;
      }
    }
    const uint8_t packed = x.scales[group / 2];
    sum += dot * (1 + 2 * ((group & 1) ? packed >> 4 : packed & 15));
  }
  return half_value(x.d) * y.d * static_cast<float>(sum);
#elif WEFT_QUANT_KIND == 1
  const uint16_t *sc = reinterpret_cast<const uint16_t *>(x.scales);
  const uint16_t scale_bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                              ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
  int32_t grid_sum = 0;
  int32_t delta_sum = 0;
  for (size_t group = 0; group < 8; ++group) {
    int32_t grid_half[2] = {0, 0};
    int32_t delta_half[2] = {0, 0};
    const uint8_t *high = x.qh + group * 2;
    for (size_t vector = 0; vector < 4; ++vector) {
      const uint16_t index = x.qs[group * 4 + vector] |
          ((static_cast<uint16_t>(high[vector / 2]) <<
            (8 - 4 * (vector % 2))) & 0x700);
      const int8_t *grid = reinterpret_cast<const int8_t *>(iq1s_grid + index);
      const int delta = high[vector / 2] & ((vector & 1) ? 0x80 : 0x08) ? -1 : 1;
      for (size_t lane = 0; lane < 8; ++lane) {
        const int q8 = y.qs[group * 32 + vector * 8 + lane];
        grid_half[vector / 2] += q8 * grid[lane];
        delta_half[vector / 2] += q8 * delta;
      }
    }
    const unsigned shift = 6 * (group % 2);
    const int first_scale = 1 + 2 * ((sc[group / 2] >> shift) & 7);
    const int second_scale = 1 + 2 * ((sc[group / 2] >> (shift + 3)) & 7);
    grid_sum += grid_half[0] * first_scale + grid_half[1] * second_scale;
    delta_sum += delta_half[0] * first_scale + delta_half[1] * second_scale;
  }
  return half_value(scale_bits) * y.d *
         (static_cast<float>(grid_sum) + 0.125f * static_cast<float>(delta_sum));
#else
  int32_t sum = 0;
  for (size_t half = 0; half < 2; ++half) {
    for (size_t lane = 0; lane < 32; ++lane) {
      const uint8_t low0 = x.ql[half * 64 + lane];
      const uint8_t low1 = x.ql[half * 64 + 32 + lane];
      const uint8_t high = x.qh[half * 32 + lane];
      const int decoded[4] = {
          static_cast<int>((low0 & 15) | (((high >> 0) & 3) << 4)) - 32,
          static_cast<int>((low1 & 15) | (((high >> 2) & 3) << 4)) - 32,
          static_cast<int>((low0 >> 4) | (((high >> 4) & 3) << 4)) - 32,
          static_cast<int>((low1 >> 4) | (((high >> 6) & 3) << 4)) - 32,
      };
      for (size_t quarter = 0; quarter < 4; ++quarter) {
        const size_t index = half * 128 + quarter * 32 + lane;
        sum += decoded[quarter] * y.qs[index] * x.scales[index / 16];
      }
    }
  }
  return half_value(x.d) * y.d * static_cast<float>(sum);
#endif
}

int main() {
  constexpr size_t blocks = 4096 / 256;
  constexpr size_t rows = 14336;
  std::vector<WeightBlock> weight(rows * blocks);
  std::vector<block_q8_K> activation(blocks);
  for (size_t block = 0; block < blocks; ++block) {
    activation[block].d = 0.0078125f;
    for (size_t i = 0; i < 256; ++i)
      activation[block].qs[i] = static_cast<int8_t>(static_cast<int>((i * 17 + block * 13) % 63) - 31);
  }
  for (size_t i = 0; i < weight.size(); ++i) {
    std::memset(&weight[i], 0, sizeof(WeightBlock));
#if WEFT_QUANT_KIND == 2
    weight[i].d = static_cast<ggml_half>(half_bits(0.015625f));
    for (size_t j = 0; j < 32; ++j) {
      weight[i].qs[j] = static_cast<uint8_t>((i * 11 + j * 7) & 255);
      weight[i].qs[32 + j] = static_cast<uint8_t>(i * 5 + j * 29);
    }
    for (size_t j = 0; j < 8; ++j) {
      weight[i].qh[j] = static_cast<uint8_t>(i + j * 37);
      weight[i].scales[j] = static_cast<uint8_t>((j & 7) | ((7 - (j & 7)) << 4));
    }
#elif WEFT_QUANT_KIND == 3
    weight[i].d = static_cast<ggml_half>(half_bits(0.015625f));
    for (size_t j = 0; j < 64; ++j)
      weight[i].qs[j] = static_cast<uint8_t>(i * 11 + j * 7);
    for (size_t j = 0; j < 8; ++j)
      weight[i].qh[j] = static_cast<uint8_t>(i + j * 37);
    for (size_t j = 0; j < 32; ++j)
      weight[i].signs[j] = static_cast<uint8_t>(i * 5 + j * 29);
    for (size_t j = 0; j < 4; ++j)
      weight[i].scales[j] = static_cast<uint8_t>((j + 1) | ((7 - j) << 4));
#elif WEFT_QUANT_KIND == 1
    for (size_t j = 0; j < 32; ++j)
      weight[i].qs[j] = static_cast<uint8_t>(i * 11 + j * 7);
    for (size_t j = 0; j < 16; ++j)
      weight[i].qh[j] = static_cast<uint8_t>(i * 5 + j * 29);
    uint16_t sc[4] = {0x0000, 0x0000, 0x0000, 0x3c00};
    std::memcpy(weight[i].scales, sc, sizeof(sc));
#else
    weight[i].d = static_cast<ggml_half>(half_bits(0.015625f));
    for (size_t j = 0; j < 128; ++j)
      weight[i].ql[j] = static_cast<uint8_t>(i * 11 + j * 7);
    for (size_t j = 0; j < 64; ++j)
      weight[i].qh[j] = static_cast<uint8_t>(i * 5 + j * 29);
    for (size_t j = 0; j < 16; ++j)
      weight[i].scales[j] = static_cast<int8_t>(static_cast<int>(j % 7) - 3);
#endif
  }
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (size_t row = 0; row < rows; ++row) {
    float expected = 0.0f;
    for (size_t block = 0; block < blocks; ++block)
      expected += reference_block(weight[row * blocks + block], activation[block]);
    const float actual = run_kernel(weight.data() + row * blocks, activation.data(), blocks);
    const float error = std::abs(actual - expected);
    max_abs = std::max(max_abs, error);
    max_rel = std::max(max_rel, error / std::max(1.0e-6f, std::abs(expected)));
  }
  if (max_abs > 2.0e-3f || max_rel > 2.0e-4f) {
    std::fprintf(stderr, "%s mismatch\n", kernel_name);
    return 1;
  }
  std::vector<unsigned char> eviction(64 * 1024 * 1024, 1);
  std::vector<double> samples;
  volatile float sink = 0.0f;
  for (int repetition = 0; repetition < 3; ++repetition) {
    for (size_t i = 0; i < eviction.size(); i += 64)
      eviction[i] = static_cast<unsigned char>(eviction[i] + 1);
    const auto begin = std::chrono::steady_clock::now();
    float total = 0.0f;
    for (size_t row = 0; row < rows; ++row)
      total += run_kernel(weight.data() + row * blocks, activation.data(), blocks);
    const auto end = std::chrono::steady_clock::now();
    sink = total;
    samples.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  const double operations = 2.0 * static_cast<double>(rows) * 4096.0;
  std::printf("kernel=%s\n", kernel_name);
  std::printf("model_shape=M=1;N=14336;K=4096\n");
  std::printf("correctness_scope=full_rows\n");
  std::printf("max_absolute_error=%.9g\n", max_abs);
  std::printf("max_relative_error=%.9g\n", max_rel);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gop_s=%.6f\n", operations / (milliseconds * 1.0e6));
  return sink == 0.1234567f;
}
