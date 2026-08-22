#include "ggml-quants.h"
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

#ifndef WEFT_MUL_MAT_FORMAT
#error "WEFT_MUL_MAT_FORMAT must select one production MUL_MAT kernel"
#endif

namespace {

constexpr std::size_t kM = 2;
constexpr std::size_t kN = 2;

[[maybe_unused]] std::vector<float> grid64(const std::uint64_t *table, std::size_t entries) {
  std::vector<float> result(entries * 8);
  for (std::size_t entry = 0; entry < entries; ++entry)
    for (std::size_t lane = 0; lane < 8; ++lane)
      result[entry * 8 + lane] = static_cast<float>(
          static_cast<std::int8_t>((table[entry] >> (8 * lane)) & 0xffU));
  return result;
}

[[maybe_unused]] std::vector<float> grid32(const std::uint32_t *table, std::size_t entries) {
  std::vector<float> result(entries * 4);
  for (std::size_t entry = 0; entry < entries; ++entry)
    for (std::size_t lane = 0; lane < 4; ++lane)
      result[entry * 4 + lane] = static_cast<float>(
          static_cast<std::int8_t>((table[entry] >> (8 * lane)) & 0xffU));
  return result;
}

[[maybe_unused]] std::vector<float> sign_table() {
  std::vector<float> result(128 * 8);
  for (std::size_t sign = 0; sign < 128; ++sign)
    for (std::size_t lane = 0; lane < 8; ++lane)
      result[sign * 8 + lane] =
          (ksigns_iq2xs[sign] & kmask_iq2xs[lane]) ? -1.0f : 1.0f;
  return result;
}

[[maybe_unused]] std::vector<float> f16_table() {
  std::vector<float> result(65536);
  for (std::uint32_t bits = 0; bits < 65536; ++bits) {
    const std::uint16_t packed = static_cast<std::uint16_t>(bits);
    _Float16 value;
    std::memcpy(&value, &packed, sizeof(value));
    result[bits] = static_cast<float>(value);
  }
  return result;
}

[[maybe_unused]] std::vector<float> e8m0_table() {
  std::vector<float> result(256);
  for (std::uint32_t value = 0; value < 256; ++value) {
    const std::uint32_t bits =
        value < 2 ? 0x00200000U << value : (value - 1U) << 23U;
    std::memcpy(&result[value], &bits, sizeof(bits));
  }
  return result;
}

[[maybe_unused]] std::vector<float> ue4m3_table() {
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

std::vector<float> activation_values(std::size_t count) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index)
    result[index] = static_cast<float>(static_cast<int>((index * 37U) % 251U) - 125) /
                    17.0f;
  return result;
}

} // namespace

#if WEFT_MUL_MAT_FORMAT == 0
constexpr int kElements = 32;
extern "C" void production_mul_mat_f16(const _Float16 *, const float *, _Float16 *,
                                        float *, std::size_t, std::size_t, std::size_t);
#elif WEFT_MUL_MAT_FORMAT == 1
using selected_weight = block_q1_0;
using selected_activation = block_q8_0;
constexpr int kElements = 128;
#define selected_wqk 128
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q1_0_q8_0_generic
#define selected_quantize quantize_row_q8_0_ref
extern "C" void production_mul_mat_q1_0(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q1_0(w, x, xq, y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 2
using selected_weight = block_q4_0;
using selected_activation = block_q8_0;
constexpr int kElements = 32;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q4_0_q8_0_generic
#define selected_quantize quantize_row_q8_0_ref
extern "C" void production_mul_mat_q4_0(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q4_0(w, x, xq, y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 3
using selected_weight = block_q4_1;
using selected_activation = block_q8_1;
constexpr int kElements = 32;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q4_1_q8_1_generic
#define selected_quantize quantize_row_q8_1_ref
extern "C" void production_mul_mat_q4_1(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q4_1(w, x, xq, y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 4
using selected_weight = block_q5_0;
using selected_activation = block_q8_0;
constexpr int kElements = 32;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q5_0_q8_0_generic
#define selected_quantize quantize_row_q8_0_ref
extern "C" void production_mul_mat_q5_0(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q5_0(w, x, xq, y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 5
using selected_weight = block_q5_1;
using selected_activation = block_q8_1;
constexpr int kElements = 32;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q5_1_q8_1_generic
#define selected_quantize quantize_row_q8_1_ref
extern "C" void production_mul_mat_q5_1(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q5_1(w, x, xq, y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 6
using selected_weight = block_q8_0;
using selected_activation = block_q8_0;
constexpr int kElements = 32;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_q8_0_q8_0_generic
#define selected_quantize quantize_row_q8_0_ref
extern "C" void production_mul_mat_q8_0(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q8_0(w, x, xq, y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 7
using selected_weight = block_q2_K;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q2_K_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_q2_k(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q2_k(w, x, xq, y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 8
using selected_weight = block_q3_K;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q3_K_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_q3_k(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q3_k(w, x, xq, y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 9
using selected_weight = block_q4_K;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q4_K_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_q4_k(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q4_k(w, x, xq, y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 10
using selected_weight = block_q5_K;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q5_K_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_q5_k(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q5_k(w, x, xq, y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 11
using selected_weight = block_q6_K;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_q6_K_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_q6_k(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_q6_k(w, x, xq, y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 12
using selected_weight = block_iq1_s;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq1_s_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_iq1_s(const std::uint8_t *, const float *, std::uint8_t *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq1_s(w, x, xq, iq1.data(), y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 13
using selected_weight = block_iq1_m;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq1_m_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_iq1_m(const std::uint8_t *, const float *, std::uint8_t *, const float *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq1_m(w, x, xq, iq1.data(), fp16.data(), y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 14
using selected_weight = block_iq2_s;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq2_s_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_iq2_s(const std::uint8_t *, const float *, std::uint8_t *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq2_s(w, x, xq, iq2s.data(), y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 15
using selected_weight = block_iq2_xs;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq2_xs_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_iq2_xs(const std::uint8_t *, const float *, std::uint8_t *, const float *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq2_xs(w, x, xq, iq2xs.data(), signs.data(), y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 16
using selected_weight = block_iq2_xxs;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq2_xxs_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_iq2_xxs(const std::uint8_t *, const float *, std::uint8_t *, const float *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq2_xxs(w, x, xq, iq2xxs.data(), signs.data(), y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 17
using selected_weight = block_iq3_s;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq3_s_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_iq3_s(const std::uint8_t *, const float *, std::uint8_t *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq3_s(w, x, xq, iq3s.data(), y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 18
using selected_weight = block_iq3_xxs;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq3_xxs_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_iq3_xxs(const std::uint8_t *, const float *, std::uint8_t *, const float *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq3_xxs(w, x, xq, iq3xxs.data(), signs.data(), y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 19
using selected_weight = block_iq4_nl;
using selected_activation = block_q8_0;
constexpr int kElements = 32;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_iq4_nl_q8_0_generic
#define selected_quantize quantize_row_q8_0_ref
extern "C" void production_mul_mat_iq4_nl(const std::uint8_t *, const float *, std::uint8_t *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq4_nl(w, x, xq, iq4.data(), y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 20
using selected_weight = block_iq4_xs;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_iq4_xs_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_iq4_xs(const std::uint8_t *, const float *, std::uint8_t *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_iq4_xs(w, x, xq, iq4.data(), y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 21
using selected_weight = block_tq1_0;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_tq1_0_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_tq1_0(const std::uint8_t *, const float *, std::uint8_t *, const std::uint32_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_tq1_0(w, x, xq, powers, y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 22
using selected_weight = block_tq2_0;
using selected_activation = block_q8_K;
constexpr int kElements = 256;
#define selected_wqk 256
#define selected_xqk 256
#define selected_reference ggml_vec_dot_tq2_0_q8_K_generic
#define selected_quantize quantize_row_q8_K_ref
extern "C" void production_mul_mat_tq2_0(const std::uint8_t *, const float *, std::uint8_t *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_tq2_0(w, x, xq, y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 23
using selected_weight = block_mxfp4;
using selected_activation = block_q8_0;
constexpr int kElements = 32;
#define selected_wqk 32
#define selected_xqk 32
#define selected_reference ggml_vec_dot_mxfp4_q8_0_generic
#define selected_quantize quantize_row_q8_0_ref
extern "C" void production_mul_mat_mxfp4(const std::uint8_t *, const float *, std::uint8_t *, const float *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_mxfp4(w, x, xq, fp4.data(), e8m0.data(), y, kN, kElements, kM)
#elif WEFT_MUL_MAT_FORMAT == 24
using selected_weight = block_nvfp4;
using selected_activation = block_q8_0;
constexpr int kElements = 64;
#define selected_wqk 64
#define selected_xqk 32
#define selected_reference ggml_vec_dot_nvfp4_q8_0
#define selected_quantize quantize_row_q8_0_ref
extern "C" void production_mul_mat_nvfp4(const std::uint8_t *, const float *, std::uint8_t *, const float *, const float *, float *, std::size_t, std::size_t, std::size_t);
#define selected_call(w, x, xq, y) production_mul_mat_nvfp4(w, x, xq, fp4.data(), ue4m3.data(), y, kN, kElements, kM)
#else
#error "unknown WEFT_MUL_MAT_FORMAT"
#endif

int main() {
  const std::vector<float> activation = activation_values(kM * kElements);
  std::vector<float> expected(kM * kN, 0.0f);
  std::vector<float> actual(kM * kN, 0.0f);

#if WEFT_MUL_MAT_FORMAT == 0
  std::vector<_Float16> weights(kN * kElements);
  std::vector<_Float16> expected_workspace(kM * kElements);
  std::vector<_Float16> actual_workspace(kM * kElements);
  for (std::size_t index = 0; index < weights.size(); ++index)
    weights[index] = static_cast<_Float16>(
        static_cast<float>(static_cast<int>((index * 19U) % 127U) - 63) / 13.0f);
  for (std::size_t index = 0; index < expected_workspace.size(); ++index)
    expected_workspace[index] = static_cast<_Float16>(activation[index]);
  for (std::size_t row = 0; row < kM; ++row)
    for (std::size_t column = 0; column < kN; ++column)
      for (int k = 0; k < kElements; ++k)
        expected[row * kN + column] +=
            static_cast<float>(weights[column * kElements + k]) *
            static_cast<float>(expected_workspace[row * kElements + k]);
  production_mul_mat_f16(weights.data(), activation.data(), actual_workspace.data(),
                          actual.data(), kN, kElements, kM);
  if (actual_workspace != expected_workspace) {
    std::fprintf(stderr, "f16 staging mismatch\n");
    return 1;
  }
#else
  std::vector<selected_activation> expected_workspace(kM * kElements / selected_xqk);
  std::vector<selected_activation> actual_workspace(kM * kElements / selected_xqk);
  for (std::size_t row = 0; row < kM; ++row)
    selected_quantize(activation.data() + row * kElements,
                      expected_workspace.data() + row * kElements / selected_xqk,
                      kElements);

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

  std::mt19937 generator(0x4d554c4dU + WEFT_MUL_MAT_FORMAT);
  std::uniform_int_distribution<unsigned> bytes(0, 255);
  std::vector<selected_weight> weights(kN * kElements / selected_wqk);
  bool found = false;
  for (int attempt = 0; attempt < 4096; ++attempt) {
    auto *raw = reinterpret_cast<std::uint8_t *>(weights.data());
    for (std::size_t index = 0; index < weights.size() * sizeof(selected_weight); ++index)
      raw[index] = static_cast<std::uint8_t>(bytes(generator));
    found = true;
    for (std::size_t row = 0; row < kM; ++row) {
      for (std::size_t column = 0; column < kN; ++column) {
        selected_reference(
            kElements, &expected[row * kN + column], 0,
            weights.data() + column * kElements / selected_wqk, 0,
            expected_workspace.data() + row * kElements / selected_xqk, 0, 1);
        found = found && __builtin_isfinite(expected[row * kN + column]);
      }
    }
    if (found)
      break;
  }
  if (!found) {
    std::fprintf(stderr, "failed to generate finite random encoded weights\n");
    return 2;
  }
  selected_call(reinterpret_cast<const std::uint8_t *>(weights.data()),
                activation.data(),
                reinterpret_cast<std::uint8_t *>(actual_workspace.data()),
                actual.data());
  if (std::memcmp(actual_workspace.data(), expected_workspace.data(),
                  actual_workspace.size() * sizeof(selected_activation)) != 0) {
    std::fprintf(stderr, "activation workspace mismatch\n");
    return 1;
  }
#endif

  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (std::memcmp(&actual[index], &expected[index], sizeof(float)) != 0) {
      std::fprintf(stderr, "output mismatch at %zu: actual=%a expected=%a\n",
                   index, actual[index], expected[index]);
      return 1;
    }
  }
  std::printf("bit_exact=yes M=%zu N=%zu K=%d\n", kM, kN, kElements);
  return 0;
}
