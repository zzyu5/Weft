#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t kCandidates = 4096;
constexpr std::size_t kSelected = 256;
constexpr float kThreshold = 0.5F;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

float overlap(const float *a, const float *b) {
  const float width =
      std::max(0.0F, std::min(a[2], b[2]) - std::max(a[0], b[0]));
  const float height =
      std::max(0.0F, std::min(a[3], b[3]) - std::max(a[1], b[1]));
  const float intersection = width * height;
  const float areaA = std::max(0.0F, a[2] - a[0]) *
                      std::max(0.0F, a[3] - a[1]);
  const float areaB = std::max(0.0F, b[2] - b[0]) *
                      std::max(0.0F, b[3] - b[1]);
  return intersection / (areaA + areaB - intersection);
}

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
  std::vector<float> boxes(kCandidates * 4);
  std::vector<float> scores(kCandidates);
  for (std::size_t candidate = 0; candidate < kCandidates; ++candidate) {
    const std::size_t cluster = candidate % 512U;
    const float x = static_cast<float>((cluster * 13U) % 256U) +
                    static_cast<float>(candidate % 7U) * 0.35F;
    const float y = static_cast<float>((cluster * 29U) % 256U) +
                    static_cast<float>(candidate % 5U) * 0.4F;
    boxes[candidate * 4 + 0] = x;
    boxes[candidate * 4 + 1] = y;
    boxes[candidate * 4 + 2] = x + 12.0F + candidate % 19U;
    boxes[candidate * 4 + 3] = y + 11.0F + candidate % 17U;
    scores[candidate] =
        static_cast<float>((candidate * 811U) % 65521U) / 65536.0F +
        static_cast<float>(kCandidates - candidate) / 1.0e9F;
  }

  std::vector<std::uint8_t> expectedSuppressed(kCandidates, 0);
  std::vector<std::uint32_t> expected(kSelected, 0);
  for (std::size_t rank = 0; rank < kSelected; ++rank) {
    float bestScore = -std::numeric_limits<float>::infinity();
    std::size_t bestIndex = 0;
    for (std::size_t candidate = 0; candidate < kCandidates; ++candidate)
      if (expectedSuppressed[candidate] == 0 &&
          (scores[candidate] > bestScore ||
           (scores[candidate] == bestScore && candidate < bestIndex))) {
        bestScore = scores[candidate];
        bestIndex = candidate;
      }
    expected[rank] = static_cast<std::uint32_t>(bestIndex);
    expectedSuppressed[bestIndex] = 1;
    for (std::size_t candidate = 0; candidate < kCandidates; ++candidate)
      if (expectedSuppressed[candidate] == 0 &&
          overlap(boxes.data() + bestIndex * 4,
                  boxes.data() + candidate * 4) > kThreshold)
        expectedSuppressed[candidate] = 1;
  }

  std::vector<std::uint8_t> suppressed(
      greedy_nms_f32__suppressed_elements(kCandidates), 0);
  std::vector<std::uint32_t> selected(kSelected, 0);
  greedy_nms_f32(boxes.data(), scores.data(), suppressed.data(), selected.data(),
                 kCandidates, kSelected, kThreshold);
  if (selected != expected) {
    std::fprintf(stderr, "greedy NMS selected index mismatch\n");
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  std::vector<double> samples;
  samples.reserve(5);
  for (int repetition = 0; repetition < 5; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    greedy_nms_f32(boxes.data(), scores.data(), suppressed.data(),
                   selected.data(), kCandidates, kSelected, kThreshold);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  const double milliseconds = median(samples);
  std::printf("kernel=greedy_nms_f32\n");
  std::printf("model_shape=detection_postprocess[candidates=%zu,max_selected=%zu]\n",
              kCandidates, kSelected);
  std::printf("selected_index_mismatches=0\n");
  std::printf("median_ms=%.6f\n", milliseconds);
  std::printf("million_pair_checks_s=%.6f\n",
              static_cast<double>(kCandidates * kSelected * 2U) /
                  milliseconds / 1000.0);
  return 0;
}
