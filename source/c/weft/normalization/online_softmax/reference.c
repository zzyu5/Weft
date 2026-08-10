#include <math.h>
#include <stddef.h>

void weft_reference_online_softmax_f32(
    const float *x,
    float *y,
    size_t n) {
  float maximum = -INFINITY;
  float scaled_sum = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    const float next_maximum = fmaxf(maximum, x[i]);
    scaled_sum = scaled_sum * expf(maximum - next_maximum) +
                 expf(x[i] - next_maximum);
    maximum = next_maximum;
  }

  for (size_t i = 0; i < n; ++i) {
    y[i] = expf(x[i] - maximum) / scaled_sum;
  }
}
