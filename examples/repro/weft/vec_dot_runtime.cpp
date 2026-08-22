#include "quants.h"

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#ifndef WEFT_VEC_DOT_FORMAT
#error "WEFT_VEC_DOT_FORMAT must select one quantized vec-dot kernel"
#endif

namespace {

#if WEFT_VEC_DOT_FORMAT == 0
constexpr int kElements = 128;
#elif WEFT_VEC_DOT_FORMAT == 23
constexpr int kElements = 64;
#elif (WEFT_VEC_DOT_FORMAT >= 6 && WEFT_VEC_DOT_FORMAT <= 17) || \
    (WEFT_VEC_DOT_FORMAT >= 19 && WEFT_VEC_DOT_FORMAT <= 21)
constexpr int kElements = 256;
#else
constexpr int kElements = 32;
#endif

std::vector<float> grid64(const std::uint64_t *table, std::size_t entries) {
  std::vector<float> result(entries * 8);
  for (std::size_t entry = 0; entry < entries; ++entry) {
    for (std::size_t lane = 0; lane < 8; ++lane) {
      result[entry * 8 + lane] = static_cast<float>(
          static_cast<std::int8_t>((table[entry] >> (8 * lane)) & 0xffU));
    }
  }
  return result;
}

std::vector<float> grid32(const std::uint32_t *table, std::size_t entries) {
  std::vector<float> result(entries * 4);
  for (std::size_t entry = 0; entry < entries; ++entry) {
    for (std::size_t lane = 0; lane < 4; ++lane) {
      result[entry * 4 + lane] = static_cast<float>(
          static_cast<std::int8_t>((table[entry] >> (8 * lane)) & 0xffU));
    }
  }
  return result;
}

std::vector<float> sign_table() {
  std::vector<float> result(128 * 8);
  for (std::size_t sign = 0; sign < 128; ++sign) {
    for (std::size_t lane = 0; lane < 8; ++lane) {
      result[sign * 8 + lane] =
          (ksigns_iq2xs[sign] & kmask_iq2xs[lane]) ? -1.0f : 1.0f;
    }
  }
  return result;
}

std::vector<float> f16_table() {
  std::vector<float> result(65536);
  for (std::uint32_t bits = 0; bits < 65536; ++bits) {
    const std::uint16_t packed = static_cast<std::uint16_t>(bits);
    _Float16 value;
    std::memcpy(&value, &packed, sizeof(value));
    result[bits] = static_cast<float>(value);
  }
  return result;
}

std::vector<float> e8m0_table() {
  std::vector<float> result(256);
  for (std::uint32_t value = 0; value < 256; ++value) {
    const std::uint32_t bits =
        value < 2 ? 0x00200000U << value : (value - 1U) << 23U;
    std::memcpy(&result[value], &bits, sizeof(bits));
  }
  return result;
}

std::vector<float> ue4m3_table() {
  std::vector<float> result(256);
  for (std::uint32_t value = 0; value < 256; ++value) {
    if (value == 0 || value == 0x7fU) {
      result[value] = 0.0f;
      continue;
    }
    const int exponent = static_cast<int>((value >> 3U) & 0xfU);
    const int mantissa = static_cast<int>(value & 7U);
    const float raw = exponent == 0
                          ? __builtin_ldexpf(static_cast<float>(mantissa), -9)
                          : __builtin_ldexpf(1.0f + static_cast<float>(mantissa) / 8.0f,
                                             exponent - 7);
    result[value] = raw * 0.5f;
  }
  return result;
}

} // namespace

#if WEFT_VEC_DOT_FORMAT == 0
using selected_weight = block_q1_0;
using selected_activation = block_q8_0;
#define selected_wqk 128
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q1_0_q8_0_generic
extern "C" void quantized_vec_dot_q1_0_q8_0(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q1_0_q8_0(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 1
using selected_weight = block_q4_0;
using selected_activation = block_q8_0;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q4_0_q8_0_generic
extern "C" void quantized_vec_dot_q4_0_q8_0(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q4_0_q8_0(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 2
using selected_weight = block_q4_1;
using selected_activation = block_q8_1;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q4_1_q8_1_generic
extern "C" void quantized_vec_dot_q4_1_q8_1(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q4_1_q8_1(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 3
using selected_weight = block_q5_0;
using selected_activation = block_q8_0;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q5_0_q8_0_generic
extern "C" void quantized_vec_dot_q5_0_q8_0(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q5_0_q8_0(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 4
using selected_weight = block_q5_1;
using selected_activation = block_q8_1;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q5_1_q8_1_generic
extern "C" void quantized_vec_dot_q5_1_q8_1(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q5_1_q8_1(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 5
using selected_weight = block_q8_0;
using selected_activation = block_q8_0;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q8_0_q8_0_generic
extern "C" void quantized_vec_dot_q8_0_q8_0(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q8_0_q8_0(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 6
using selected_weight = block_q2_K;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q2_K_q8_K_generic
extern "C" void quantized_vec_dot_q2_k_q8_k(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q2_k_q8_k(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 7
using selected_weight = block_q3_K;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q3_K_q8_K_generic
extern "C" void quantized_vec_dot_q3_k_q8_k(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q3_k_q8_k(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 8
using selected_weight = block_q4_K;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q4_K_q8_K_generic
extern "C" void quantized_vec_dot_q4_k_q8_k(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q4_k_q8_k(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 9
using selected_weight = block_q5_K;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q5_K_q8_K_generic
extern "C" void quantized_vec_dot_q5_k_q8_k(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q5_k_q8_k(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 10
using selected_weight = block_q6_K;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q6_K_q8_K_generic
extern "C" void quantized_vec_dot_q6_k_q8_k(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_q6_k_q8_k(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 11
using selected_weight = block_iq1_s;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq1_s_q8_K_generic
extern "C" void quantized_vec_dot_iq1_s_q8_k(const std::uint8_t *, const std::uint8_t *, const float *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq1_s_q8_k(w, x, iq1.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 12
using selected_weight = block_iq1_m;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq1_m_q8_K_generic
extern "C" void quantized_vec_dot_iq1_m_q8_k(const std::uint8_t *, const std::uint8_t *, const float *, const float *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq1_m_q8_k(w, x, iq1.data(), fp16.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 13
using selected_weight = block_iq2_s;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq2_s_q8_K_generic
extern "C" void quantized_vec_dot_iq2_s_q8_k(const std::uint8_t *, const std::uint8_t *, const float *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq2_s_q8_k(w, x, iq2s.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 14
using selected_weight = block_iq2_xs;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq2_xs_q8_K_generic
extern "C" void quantized_vec_dot_iq2_xs_q8_k(const std::uint8_t *, const std::uint8_t *, const float *, const float *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq2_xs_q8_k(w, x, iq2xs.data(), signs.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 15
using selected_weight = block_iq2_xxs;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq2_xxs_q8_K_generic
extern "C" void quantized_vec_dot_iq2_xxs_q8_k(const std::uint8_t *, const std::uint8_t *, const float *, const float *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq2_xxs_q8_k(w, x, iq2xxs.data(), signs.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 16
using selected_weight = block_iq3_s;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq3_s_q8_K_generic
extern "C" void quantized_vec_dot_iq3_s_q8_k(const std::uint8_t *, const std::uint8_t *, const float *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq3_s_q8_k(w, x, iq3s.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 17
using selected_weight = block_iq3_xxs;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq3_xxs_q8_K_generic
extern "C" void quantized_vec_dot_iq3_xxs_q8_k(const std::uint8_t *, const std::uint8_t *, const float *, const float *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq3_xxs_q8_k(w, x, iq3xxs.data(), signs.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 18
using selected_weight = block_iq4_nl;
using selected_activation = block_q8_0;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_iq4_nl_q8_0_generic
extern "C" void quantized_vec_dot_iq4_nl_q8_0(const std::uint8_t *, const std::uint8_t *, const float *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq4_nl_q8_0(w, x, iq4.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 19
using selected_weight = block_iq4_xs;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq4_xs_q8_K_generic
extern "C" void quantized_vec_dot_iq4_xs_q8_k(const std::uint8_t *, const std::uint8_t *, const float *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_iq4_xs_q8_k(w, x, iq4.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 20
using selected_weight = block_tq1_0;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_tq1_0_q8_K_generic
extern "C" void quantized_vec_dot_tq1_0_q8_k(const std::uint8_t *, const std::uint8_t *, const std::uint32_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_tq1_0_q8_k(w, x, powers, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 21
using selected_weight = block_tq2_0;
using selected_activation = block_q8_K;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_tq2_0_q8_K_generic
extern "C" void quantized_vec_dot_tq2_0_q8_k(const std::uint8_t *, const std::uint8_t *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_tq2_0_q8_k(w, x, y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 22
using selected_weight = block_mxfp4;
using selected_activation = block_q8_0;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_mxfp4_q8_0_generic
extern "C" void quantized_vec_dot_mxfp4_q8_0(const std::uint8_t *, const std::uint8_t *, const float *, const float *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_mxfp4_q8_0(w, x, fp4.data(), e8m0.data(), y, kElements)
#elif WEFT_VEC_DOT_FORMAT == 23
using selected_weight = block_nvfp4;
using selected_activation = block_q8_0;
#define selected_wqk 64
#define selected_xqk 32
#define selected_reference ggml_vec_dot_nvfp4_q8_0
extern "C" void quantized_vec_dot_nvfp4_q8_0(const std::uint8_t *, const std::uint8_t *, const float *, const float *, float *, std::size_t);
#define selected_call(w, x, y) quantized_vec_dot_nvfp4_q8_0(w, x, fp4.data(), ue4m3.data(), y, kElements)
#else
#error "unknown WEFT_VEC_DOT_FORMAT"
#endif

int main() {
  std::mt19937 generator(0x56444f54U + WEFT_VEC_DOT_FORMAT);
  std::uniform_int_distribution<unsigned> bytes(0, 255);
  std::vector<selected_weight> weights(kElements / selected_wqk);
  std::vector<selected_activation> activation(kElements / selected_xqk);
  float expected = 0.0f;
  bool found = false;
  for (int attempt = 0; attempt < 4096; ++attempt) {
    auto *weight_bytes = reinterpret_cast<std::uint8_t *>(weights.data());
    auto *activation_bytes = reinterpret_cast<std::uint8_t *>(activation.data());
    for (std::size_t i = 0; i < weights.size() * sizeof(selected_weight); ++i)
      weight_bytes[i] = static_cast<std::uint8_t>(bytes(generator));
    for (std::size_t i = 0; i < activation.size() * sizeof(selected_activation); ++i)
      activation_bytes[i] = static_cast<std::uint8_t>(bytes(generator));
    selected_reference(kElements, &expected, 0, weights.data(), 0,
                       activation.data(), 0, 1);
    if (__builtin_isfinite(expected)) {
      found = true;
      break;
    }
  }
  if (!found) {
    std::fprintf(stderr, "failed to generate finite random encoded data\n");
    return 2;
  }

  const std::vector<float> iq1 = grid64(iq1s_grid, 2048);
  const std::vector<float> iq2xxs = grid64(iq2xxs_grid, 256);
  const std::vector<float> iq2xs = grid64(iq2xs_grid, 512);
  const std::vector<float> iq2s = grid64(iq2s_grid, 1024);
  const std::vector<float> iq3xxs = grid32(iq3xxs_grid, 256);
  const std::vector<float> iq3s = grid32(iq3s_grid, 512);
  const std::vector<float> signs = sign_table();
  const std::vector<float> fp16 = f16_table();
  const std::vector<float> e8m0 = e8m0_table();
  const std::vector<float> ue4m3 = ue4m3_table();
  std::vector<float> iq4(16);
  std::vector<float> fp4(16);
  for (std::size_t index = 0; index < 16; ++index) {
    iq4[index] = static_cast<float>(kvalues_iq4nl[index]);
    fp4[index] = static_cast<float>(kvalues_mxfp4[index]);
  }
  const std::uint32_t powers[5] = {1, 3, 9, 27, 81};
  (void)iq1;
  (void)iq2xxs;
  (void)iq2xs;
  (void)iq2s;
  (void)iq3xxs;
  (void)iq3s;
  (void)signs;
  (void)fp16;
  (void)e8m0;
  (void)ue4m3;
  (void)iq4;
  (void)fp4;
  (void)powers;

  float actual = 0.0f;
  selected_call(reinterpret_cast<const std::uint8_t *>(weights.data()),
                reinterpret_cast<const std::uint8_t *>(activation.data()),
                &actual);
  if (std::memcmp(&actual, &expected, sizeof(float)) != 0) {
    std::fprintf(stderr, "mismatch: actual=%a expected=%a\n", actual, expected);
    return 1;
  }
  std::printf("bit_exact=yes elements=%d\n", kElements);
  return 0;
}
