#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void gemm_f32(const float *A, const float *B, float *C,
                          std::size_t M, std::size_t K, std::size_t N);

#if defined(WEFT_TARGET_K1)
#define WEFT_TARGET_NAME "K1/X60"
#else
#define WEFT_TARGET_NAME "SG2044"
#endif

namespace {
constexpr std::size_t kM = 128;
constexpr std::size_t kK = 4096;
constexpr std::size_t kN = 4096;
constexpr std::size_t kFlushBytes = 64U * 1024U * 1024U;
volatile std::uint64_t flush_sink = 0;

std::size_t parse_repetitions(const char *text) {
  char *end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  return *text != '\0' && *end == '\0' && value != 0
             ? static_cast<std::size_t>(value)
             : 0;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  return values.size() & 1U ? values[middle]
                            : 0.5 * (values[middle - 1] + values[middle]);
}

void evict_cache(std::vector<std::uint8_t> &buffer) {
  for (std::size_t offset = 0; offset < buffer.size(); offset += 64) {
    buffer[offset] = static_cast<std::uint8_t>(buffer[offset] + 1U);
    flush_sink += buffer[offset];
  }
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <repetitions>\n", argv[0]);
    return 2;
  }
  const std::size_t repetitions = parse_repetitions(argv[1]);
  if (repetitions == 0)
    return 2;
  std::vector<float> lhs(kM * kK, 1.0F / 4096.0F);
  std::vector<float> rhs(kK * kN, 1.0F);
  std::vector<float> output(kM * kN, 0.0F);
  gemm_f32(lhs.data(), rhs.data(), output.data(), kM, kK, kN);
  const float expected = 1.0F;
  for (std::size_t index = 0; index < output.size(); ++index) {
    if (std::memcmp(&expected, &output[index], sizeof(float)) != 0) {
      std::fprintf(stderr,
                   "numeric mismatch index=%zu expected=%.9g actual=%.9g\n",
                   index, expected, output[index]);
      return 1;
    }
  }
  std::vector<std::uint8_t> flush(kFlushBytes, 1);
  std::vector<double> samples;
  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    std::fill(output.begin(), output.end(), 0.0F);
    evict_cache(flush);
    const auto begin = std::chrono::steady_clock::now();
    gemm_f32(lhs.data(), rhs.data(), output.data(), kM, kK, kN);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::micro>(end - begin).count());
  }
  const double median_us = median(samples);
  const double operations =
      2.0 * static_cast<double>(kM) * kK * static_cast<double>(kN);
  std::printf("kernel=gemm_f32\ntarget=%s\nM=%zu\nN=%zu\nK=%zu\n",
              WEFT_TARGET_NAME, kM, kN, kK);
  std::printf("numeric=bit-exact\nrepetitions=%zu\n", repetitions);
  std::printf("cold_median_us=%.3f\ncold_gop_s=%.6f\n", median_us,
              operations / median_us / 1.0e3);
  return 0;
}
