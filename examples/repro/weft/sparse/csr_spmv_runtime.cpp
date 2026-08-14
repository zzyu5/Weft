#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kRows = 65536;
constexpr std::size_t kColumns = 65536;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

void evict(std::vector<std::uint8_t> &buffer) {
  for (std::size_t index = 0; index < buffer.size(); index += 64) {
    buffer[index] = static_cast<std::uint8_t>(buffer[index] + 1U);
    evictionSink += buffer[index];
  }
}

double median(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

} // namespace

int main() {
  std::vector<std::uint32_t> rowOffsets(kRows + 1, 0);
  for (std::size_t row = 0; row < kRows; ++row) {
    const std::size_t rowLength = (row % 97U == 0)
                                      ? 0U
                                      : 16U + ((row * 29U + row / 11U) % 65U);
    rowOffsets[row + 1] =
        rowOffsets[row] + static_cast<std::uint32_t>(rowLength);
  }
  const std::size_t nonzeros = rowOffsets.back();
  std::vector<std::uint32_t> columns(nonzeros);
  std::vector<float> matrix(nonzeros);
  std::vector<float> vector(kColumns);
  for (std::size_t index = 0; index < nonzeros; ++index) {
    columns[index] = static_cast<std::uint32_t>(
        (index * 7919U + (index / 47U) * 104729U) % kColumns);
    matrix[index] =
        static_cast<float>(static_cast<int>((index * 31U) % 127U) - 63) /
        1024.0F;
  }
  for (std::size_t column = 0; column < kColumns; ++column)
    vector[column] =
        static_cast<float>(static_cast<int>((column * 17U) % 113U) - 56) /
        512.0F;

  std::vector<float> expected(kRows, 0.0F);
  for (std::size_t row = 0; row < kRows; ++row)
    for (std::size_t entry = rowOffsets[row]; entry < rowOffsets[row + 1];
         ++entry)
      expected[row] += matrix[entry] * vector[columns[entry]];

  std::vector<float> output(kRows, 0.0F);
  csr_spmv_f32(rowOffsets.data(), columns.data(), matrix.data(), vector.data(),
               output.data(), 0, kRows);
  double maxAbsolute = 0.0;
  double maxRelative = 0.0;
  for (std::size_t row = 0; row < kRows; ++row) {
    const double error = std::abs(static_cast<double>(output[row]) -
                                  static_cast<double>(expected[row]));
    maxAbsolute = std::max(maxAbsolute, error);
    maxRelative = std::max(
        maxRelative,
        error / std::max(std::abs(static_cast<double>(expected[row])), 1.0e-6));
  }
  if (maxAbsolute > 3.0e-5 && maxRelative > 3.0e-4) {
    std::fprintf(stderr, "csr spmv mismatch: max_abs=%g max_rel=%g\n",
                 maxAbsolute, maxRelative);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    csr_spmv_f32(rowOffsets.data(), columns.data(), matrix.data(), vector.data(),
                 output.data(), 0, kRows);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=csr_spmv_f32\n");
  std::printf("model_shape=sparse_graph[rows=%zu,columns=%zu,nnz=%zu]\n",
              kRows, kColumns, nonzeros);
  std::printf("max_absolute_error=%.9g\n", maxAbsolute);
  std::printf("max_relative_error=%.9g\n", maxRelative);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("gflops=%.6f\n",
              2.0 * static_cast<double>(nonzeros) / milliseconds / 1.0e6);
  return 0;
}
