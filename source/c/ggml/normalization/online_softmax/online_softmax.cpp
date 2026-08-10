/*
 * Standalone extraction of the online M/S update in ggml flash-attention.
 * Distributed under the MIT license in source/c/ggml/LICENSE.
 */

#include <cmath>
#include <cstddef>
#include <limits>

extern "C" void ggml_source_online_softmax_f32(
    const float *src,
    float *dst,
    std::size_t count) {
  float maximum = -std::numeric_limits<float>::infinity();
  float sum = 0.0f;

  for (std::size_t i = 0; i < count; ++i) {
    const float score = src[i];
    const float old_maximum = maximum;
    float maximum_scale = 1.0f;
    float value_scale = 1.0f;

    if (score > maximum) {
      maximum = score;
      maximum_scale = std::exp(old_maximum - maximum);
    } else {
      value_scale = std::exp(score - maximum);
    }
    sum = sum * maximum_scale + value_scale;
  }

  for (std::size_t i = 0; i < count; ++i) {
    dst[i] = std::exp(src[i] - maximum) / sum;
  }
}
