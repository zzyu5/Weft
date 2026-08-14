#include <cmath>
#include <cstddef>
#include <cstdio>

int main() {
  constexpr std::size_t count = 16;
  constexpr std::size_t begin = 3;
  constexpr std::size_t end = 13;
  float x[count];
  float bias[count];
  float y[count];
  for (std::size_t i = 0; i < count; ++i) {
    x[i] = static_cast<float>(i) * 0.25f - 1.0f;
    bias[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.5f;
    y[i] = -123.0f;
  }

  add_bias(x, bias, y, begin, end);
  for (std::size_t i = 0; i < count; ++i) {
    const float expected = i >= begin && i < end ? x[i] + bias[i] : -123.0f;
    if (std::fabs(y[i] - expected) > 1.0e-6f)
      return 1;
  }
  std::puts("PASS weft add_bias");
  return 0;
}
