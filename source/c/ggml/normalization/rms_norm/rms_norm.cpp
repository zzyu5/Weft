/*
 * Standalone extraction of the F32 RMSNorm + MUL loop in ggml-cpu/ops.cpp.
 * Distributed under the MIT license in source/c/ggml/LICENSE.
 */

#include <cmath>
#include <cstddef>

extern "C" void ggml_source_rms_norm_f32(
    const float *src,
    const float *weight,
    float *dst,
    std::size_t rows,
    std::size_t columns,
    float eps) {
  for (std::size_t row = 0; row < rows; ++row) {
    const float *x = src + row * columns;
    float *y = dst + row * columns;

    double sum = 0.0;
    for (std::size_t column = 0; column < columns; ++column) {
      sum += static_cast<double>(x[column] * x[column]);
    }

    const float mean = static_cast<float>(sum / static_cast<double>(columns));
    const float scale = 1.0f / std::sqrt(mean + eps);
    for (std::size_t column = 0; column < columns; ++column) {
      y[column] = x[column] * scale * weight[column];
    }
  }
}
