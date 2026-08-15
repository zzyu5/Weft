#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

constexpr std::size_t kBegin = 1024;
constexpr std::size_t kEnd = 128256;
constexpr std::size_t kSequenceLength = 127997;
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
  const std::size_t middle = samples.size() / 2;
  return samples.size() % 2 == 0
             ? 0.5 * (samples[middle - 1] + samples[middle])
             : samples[middle];
}

} // namespace

int main() {
  std::vector<float> input(kEnd);
  for (std::size_t index = 0; index < kSequenceLength; ++index)
    input[index] =
        static_cast<float>(static_cast<int>(index % 4093) - 2046) / 128.0F;
  input[7919] = 123.0F;
  std::fill(input.begin() + kSequenceLength, input.end(), 1000.0F);

  float output = 0.0F;
  predicate_reduce(input.data(), &output, kBegin, kEnd, kSequenceLength);
  float reference = input[kBegin];
  for (std::size_t index = kBegin + 1; index < kSequenceLength; ++index)
    reference = std::max(reference, input[index]);
  if (output != reference) {
    std::fprintf(stderr, "predicate reduce mismatch: got=%g expected=%g\n",
                 output, reference);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(10);
  for (int repetition = 0; repetition < 10; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    predicate_reduce(input.data(), &output, kBegin, kEnd, kSequenceLength);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }

  const double milliseconds = median(samples);
  std::printf("kernel=predicate_reduce\n");
  std::printf("model_shape=logits[begin=1024,end=128256,logical=127997]\n");
  std::printf("elements=%zu\n", kEnd - kBegin);
  std::printf("result=%g\n", output);
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("melements_s=%.6f\n",
              static_cast<double>(kEnd - kBegin) / milliseconds / 1000.0);
  return 0;
}
