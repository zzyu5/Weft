#include <math.h>
#include <stddef.h>
#include <stdio.h>

void weft_reference_online_softmax_f32(
    const float *x,
    float *y,
    size_t n);

int main(void) {
  static const float x[] = {-4.0f, -1.5f, 0.0f, 2.0f, 1.25f, -0.5f, 3.0f, 0.75f};
  enum { N = sizeof(x) / sizeof(x[0]) };
  float y[N];

  weft_reference_online_softmax_f32(x, y, N);

  float sum = 0.0f;
  for (size_t i = 0; i < N; ++i) {
    if (!(y[i] > 0.0f && y[i] < 1.0f)) {
      return 1;
    }
    sum += y[i];
  }
  if (fabsf(sum - 1.0f) > 1.0e-6f) {
    return 1;
  }

  puts("PASS online_softmax_f32");
  return 0;
}
