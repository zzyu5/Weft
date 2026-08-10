#include <cmath>
#include <cstddef>
#include <cstdio>

extern "C" void online_softmax_f32(
    const float *x,
    float *y,
    std::size_t n);

int main() {
  constexpr std::size_t count = 9;
  const float x[count] = {1.0f, -2.0f, 3.0f, 0.5f, 4.0f,
                          -1.0f, 2.0f, 2.5f, -0.25f};
  float y[count] = {};

  online_softmax_f32(x, y, count);
  float maximum = x[0];
  for (std::size_t i = 1; i < count; ++i)
    maximum = maximum > x[i] ? maximum : x[i];
  float sum = 0.0f;
  for (float value : x)
    sum += std::exp(value - maximum);
  for (std::size_t i = 0; i < count; ++i) {
    const float expected = std::exp(x[i] - maximum) / sum;
    if (std::fabs(y[i] - expected) > 1.0e-6f)
      return 1;
  }
  std::puts("PASS weft online_softmax_f32");
  return 0;
}
