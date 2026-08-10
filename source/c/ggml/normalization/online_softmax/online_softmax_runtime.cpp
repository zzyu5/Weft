#include <cmath>
#include <cstddef>
#include <cstdio>

extern "C" void ggml_source_online_softmax_f32(
    const float *src,
    float *dst,
    std::size_t count);

int main() {
  static const float src[] = {
      -4.0f, -1.5f, 0.0f, 2.0f, 1.25f, -0.5f, 3.0f, 0.75f,
  };
  constexpr std::size_t count = sizeof(src) / sizeof(src[0]);
  float dst[count];

  ggml_source_online_softmax_f32(src, dst, count);
  float sum = 0.0f;
  for (float value : dst) {
    if (!(value > 0.0f && value < 1.0f)) {
      return 1;
    }
    sum += value;
  }
  if (std::fabs(sum - 1.0f) > 1.0e-6f) {
    return 1;
  }

  std::puts("PASS ggml_online_softmax_f32");
  return 0;
}
