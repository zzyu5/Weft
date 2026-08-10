#include <math.h>
#include <stddef.h>
#include <stdio.h>

void weft_reference_add_bias_f32(
    const float *x,
    const float *bias,
    float *y,
    size_t n);

int main(void) {
  enum { N = 32 };
  float x[N];
  float bias[N];
  float y[N];

  for (size_t i = 0; i < N; ++i) {
    x[i] = (float)((int)i - 11) * 0.25f;
    bias[i] = (float)((int)(i % 5) - 2) * 0.5f;
  }

  weft_reference_add_bias_f32(x, bias, y, N);
  for (size_t i = 0; i < N; ++i) {
    if (fabsf(y[i] - (x[i] + bias[i])) > 1.0e-6f) {
      return 1;
    }
  }

  puts("PASS add_bias_f32");
  return 0;
}
