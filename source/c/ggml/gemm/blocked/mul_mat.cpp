/*
 * Standalone F16 x F16 extraction of ggml_compute_forward_mul_mat_one_chunk.
 * It preserves ggml's 16 x 16 output block traversal without tensor/threadpool
 * dependencies. Distributed under the MIT license in source/c/ggml/LICENSE.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

static float fp16_to_fp32(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fu;
  std::uint32_t mantissa = value & 0x03ffu;
  std::uint32_t bits;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 113;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }

  float converted;
  std::memcpy(&converted, &bits, sizeof(converted));
  return converted;
}

extern "C" void ggml_source_mul_mat_f16_f16(
    const std::uint16_t *src0,
    const std::uint16_t *src1,
    float *dst,
    std::size_t rows,
    std::size_t columns,
    std::size_t inner) {
  constexpr std::size_t block_rows = 16;
  constexpr std::size_t block_columns = 16;

  for (std::size_t column_block = 0; column_block < columns;
       column_block += block_columns) {
    for (std::size_t row_block = 0; row_block < rows;
         row_block += block_rows) {
      const std::size_t column_end =
          std::min(column_block + block_columns, columns);
      const std::size_t row_end = std::min(row_block + block_rows, rows);
      for (std::size_t column = column_block; column < column_end; ++column) {
        for (std::size_t row = row_block; row < row_end; ++row) {
          float sum = 0.0f;
          for (std::size_t k = 0; k < inner; ++k) {
            sum += fp16_to_fp32(src0[row * inner + k]) *
                   fp16_to_fp32(src1[column * inner + k]);
          }
          dst[column * rows + row] = sum;
        }
      }
    }
  }
}
