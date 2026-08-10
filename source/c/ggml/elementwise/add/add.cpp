/*
 * Standalone extraction of the F32 add path in ggml-cpu/binary-ops.cpp.
 * Distributed under the MIT license in source/c/ggml/LICENSE.
 */

#include <cstddef>

static inline float op_add(float a, float b) {
  return a + b;
}

extern "C" void ggml_source_add_f32(
    const float *src0,
    const float *src1,
    float *dst,
    std::size_t rows,
    std::size_t columns) {
  for (std::size_t row = 0; row < rows; ++row) {
    const std::size_t offset = row * columns;
    for (std::size_t column = 0; column < columns; ++column) {
      dst[offset + column] = op_add(src0[offset + column], src1[column]);
    }
  }
}
