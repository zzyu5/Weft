#include "ggml-quants.h"

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

#ifndef WEFT_ROW_FORMAT
#error "WEFT_ROW_FORMAT must select one row-dequantization kernel"
#endif

namespace {

constexpr std::size_t kElements = 256;

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
    std::uint32_t bits =
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

template <typename Block, typename Reference>
std::vector<Block> finite_random_records(Reference reference,
                                         std::size_t block_elements) {
  std::mt19937 generator(0x57454654U + WEFT_ROW_FORMAT);
  std::uniform_int_distribution<unsigned> bytes(0, 255);
  std::vector<Block> records(kElements / block_elements);
  std::vector<float> probe(kElements);
  for (int attempt = 0; attempt < 4096; ++attempt) {
    auto *raw = reinterpret_cast<std::uint8_t *>(records.data());
    for (std::size_t index = 0; index < records.size() * sizeof(Block); ++index)
      raw[index] = static_cast<std::uint8_t>(bytes(generator));
    reference(records.data(), probe.data(), kElements);
    if (std::all_of(probe.begin(), probe.end(), [](float value) {
          return __builtin_isfinite(value);
        })) {
      return records;
    }
  }
  std::fprintf(stderr, "failed to generate finite random encoded data\n");
  std::exit(2);
}

} // namespace

#if WEFT_ROW_FORMAT == 0
using selected_block = block_q1_0;
#define selected_reference dequantize_row_q1_0
#define selected_qk 128
extern "C" void row_dequantize_q1_0(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 1
using selected_block = block_q4_0;
#define selected_reference dequantize_row_q4_0
#define selected_qk 32
extern "C" void row_dequantize_q4_0(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 2
using selected_block = block_q4_1;
#define selected_reference dequantize_row_q4_1
#define selected_qk 32
extern "C" void row_dequantize_q4_1(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 3
using selected_block = block_q5_0;
#define selected_reference dequantize_row_q5_0
#define selected_qk 32
extern "C" void row_dequantize_q5_0(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 4
using selected_block = block_q5_1;
#define selected_reference dequantize_row_q5_1
#define selected_qk 32
extern "C" void row_dequantize_q5_1(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 5
using selected_block = block_q8_0;
#define selected_reference dequantize_row_q8_0
#define selected_qk 32
extern "C" void row_dequantize_q8_0(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 6
using selected_block = block_q2_K;
#define selected_reference dequantize_row_q2_K
#define selected_qk 256
extern "C" void row_dequantize_q2_k(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 7
using selected_block = block_q3_K;
#define selected_reference dequantize_row_q3_K
#define selected_qk 256
extern "C" void row_dequantize_q3_k(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 8
using selected_block = block_q4_K;
#define selected_reference dequantize_row_q4_K
#define selected_qk 256
extern "C" void row_dequantize_q4_k(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 9
using selected_block = block_q5_K;
#define selected_reference dequantize_row_q5_K
#define selected_qk 256
extern "C" void row_dequantize_q5_k(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 10
using selected_block = block_q6_K;
#define selected_reference dequantize_row_q6_K
#define selected_qk 256
extern "C" void row_dequantize_q6_k(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 11
using selected_block = block_iq1_s;
#define selected_reference dequantize_row_iq1_s
#define selected_qk 256
extern "C" void row_dequantize_iq1_s(const std::uint8_t *, const float *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 12
using selected_block = block_iq1_m;
#define selected_reference dequantize_row_iq1_m
#define selected_qk 256
extern "C" void row_dequantize_iq1_m(const std::uint8_t *, const float *, const float *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 13
using selected_block = block_iq2_s;
#define selected_reference dequantize_row_iq2_s
#define selected_qk 256
extern "C" void row_dequantize_iq2_s(const std::uint8_t *, const float *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 14
using selected_block = block_iq2_xs;
#define selected_reference dequantize_row_iq2_xs
#define selected_qk 256
extern "C" void row_dequantize_iq2_xs(const std::uint8_t *, const float *, const float *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 15
using selected_block = block_iq2_xxs;
#define selected_reference dequantize_row_iq2_xxs
#define selected_qk 256
extern "C" void row_dequantize_iq2_xxs(const std::uint8_t *, const float *, const float *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 16
using selected_block = block_iq3_s;
#define selected_reference dequantize_row_iq3_s
#define selected_qk 256
extern "C" void row_dequantize_iq3_s(const std::uint8_t *, const float *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 17
using selected_block = block_iq3_xxs;
#define selected_reference dequantize_row_iq3_xxs
#define selected_qk 256
extern "C" void row_dequantize_iq3_xxs(const std::uint8_t *, const float *, const float *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 18
using selected_block = block_iq4_nl;
#define selected_reference dequantize_row_iq4_nl
#define selected_qk 32
extern "C" void row_dequantize_iq4_nl(const std::uint8_t *, const float *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 19
using selected_block = block_iq4_xs;
#define selected_reference dequantize_row_iq4_xs
#define selected_qk 256
extern "C" void row_dequantize_iq4_xs(const std::uint8_t *, const float *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 20
using selected_block = block_tq1_0;
#define selected_reference dequantize_row_tq1_0
#define selected_qk 256
extern "C" void row_dequantize_tq1_0(const std::uint8_t *, const std::uint32_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 21
using selected_block = block_tq2_0;
#define selected_reference dequantize_row_tq2_0
#define selected_qk 256
extern "C" void row_dequantize_tq2_0(const std::uint8_t *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 22
using selected_block = block_mxfp4;
#define selected_reference dequantize_row_mxfp4
#define selected_qk 32
extern "C" void row_dequantize_mxfp4(const std::uint8_t *, const float *, const float *, float *, std::size_t);
#elif WEFT_ROW_FORMAT == 23
using selected_block = block_nvfp4;
#define selected_reference dequantize_row_nvfp4
#define selected_qk 64
extern "C" void row_dequantize_nvfp4(const std::uint8_t *, const float *, const float *, float *, std::size_t);
#else
#error "unknown WEFT_ROW_FORMAT"
#endif

int main() {
  auto records =
      finite_random_records<selected_block>(selected_reference, selected_qk);
  std::vector<float> expected(kElements);
  std::vector<float> actual(kElements);
  selected_reference(records.data(), expected.data(), kElements);

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
  (void)powers;

#if WEFT_ROW_FORMAT == 0
  row_dequantize_q1_0(reinterpret_cast<const std::uint8_t *>(records.data()), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 1
  row_dequantize_q4_0(reinterpret_cast<const std::uint8_t *>(records.data()), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 2
  row_dequantize_q4_1(reinterpret_cast<const std::uint8_t *>(records.data()), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 3
  row_dequantize_q5_0(reinterpret_cast<const std::uint8_t *>(records.data()), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 4
  row_dequantize_q5_1(reinterpret_cast<const std::uint8_t *>(records.data()), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 5
  row_dequantize_q8_0(reinterpret_cast<const std::uint8_t *>(records.data()), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 6
  row_dequantize_q2_k(reinterpret_cast<const std::uint8_t *>(records.data()), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 7
  row_dequantize_q3_k(reinterpret_cast<const std::uint8_t *>(records.data()), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 8
  row_dequantize_q4_k(reinterpret_cast<const std::uint8_t *>(records.data()), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 9
  row_dequantize_q5_k(reinterpret_cast<const std::uint8_t *>(records.data()), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 10
  row_dequantize_q6_k(reinterpret_cast<const std::uint8_t *>(records.data()), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 11
  row_dequantize_iq1_s(reinterpret_cast<const std::uint8_t *>(records.data()), iq1.data(), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 12
  row_dequantize_iq1_m(reinterpret_cast<const std::uint8_t *>(records.data()), iq1.data(), fp16.data(), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 13
  row_dequantize_iq2_s(reinterpret_cast<const std::uint8_t *>(records.data()), iq2s.data(), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 14
  row_dequantize_iq2_xs(reinterpret_cast<const std::uint8_t *>(records.data()), iq2xs.data(), signs.data(), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 15
  row_dequantize_iq2_xxs(reinterpret_cast<const std::uint8_t *>(records.data()), iq2xxs.data(), signs.data(), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 16
  row_dequantize_iq3_s(reinterpret_cast<const std::uint8_t *>(records.data()), iq3s.data(), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 17
  row_dequantize_iq3_xxs(reinterpret_cast<const std::uint8_t *>(records.data()), iq3xxs.data(), signs.data(), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 18
  row_dequantize_iq4_nl(reinterpret_cast<const std::uint8_t *>(records.data()), iq4.data(), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 19
  row_dequantize_iq4_xs(reinterpret_cast<const std::uint8_t *>(records.data()), iq4.data(), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 20
  row_dequantize_tq1_0(reinterpret_cast<const std::uint8_t *>(records.data()), powers, actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 21
  row_dequantize_tq2_0(reinterpret_cast<const std::uint8_t *>(records.data()), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 22
  row_dequantize_mxfp4(reinterpret_cast<const std::uint8_t *>(records.data()), fp4.data(), e8m0.data(), actual.data(), kElements);
#elif WEFT_ROW_FORMAT == 23
  row_dequantize_nvfp4(reinterpret_cast<const std::uint8_t *>(records.data()), fp4.data(), ue4m3.data(), actual.data(), kElements);
#endif

  for (std::size_t index = 0; index < kElements; ++index) {
    if (std::memcmp(&actual[index], &expected[index], sizeof(float)) != 0) {
      std::fprintf(stderr, "mismatch at %zu: actual=%a expected=%a\n", index,
                   actual[index], expected[index]);
      return 1;
    }
  }
  std::printf("bit_exact=yes elements=%zu\n", kElements);
  return 0;
}
