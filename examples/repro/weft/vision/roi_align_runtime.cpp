#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void roi_align_f32(
    const float *source, const float *boxes,
    const std::uint32_t *batch_indices, float *output, std::size_t box_begin,
    std::size_t box_end, std::size_t input_height, std::size_t input_width,
    std::size_t channels, std::size_t pooled_height,
    std::size_t pooled_width, std::size_t samples_y, std::size_t samples_x,
    float spatial_scale);
extern "C" void roi_align_f32_equivalent(
    const float *source, const float *boxes,
    const std::uint32_t *batch_indices, float *output, std::size_t box_begin,
    std::size_t box_end, std::size_t input_height, std::size_t input_width,
    std::size_t channels, std::size_t pooled_height,
    std::size_t pooled_width, std::size_t samples_y, std::size_t samples_x,
    float spatial_scale);

namespace {

constexpr std::size_t kBatch = 2;
constexpr std::size_t kHeight = 64;
constexpr std::size_t kWidth = 64;
constexpr std::size_t kChannels = 256;
constexpr std::size_t kBoxes = 256;
constexpr std::size_t kPooled = 7;
constexpr std::size_t kSamples = 2;
constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;
volatile std::uint64_t evictionSink = 0;

using Kernel = void (*)(const float *, const float *, const std::uint32_t *,
                        float *, std::size_t, std::size_t, std::size_t,
                        std::size_t, std::size_t, std::size_t, std::size_t,
                        std::size_t, std::size_t, float);

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

double run(Kernel kernel, const std::vector<float> &source,
           const std::vector<float> &boxes,
           const std::vector<std::uint32_t> &batchIndices,
           std::vector<float> &output, std::vector<std::uint8_t> &eviction) {
  std::vector<double> samples;
  samples.reserve(5);
  for (int repetition = 0; repetition < 5; ++repetition) {
    evict(eviction);
    const auto begin = std::chrono::steady_clock::now();
    kernel(source.data(), boxes.data(), batchIndices.data(), output.data(), 0,
           kBoxes, kHeight, kWidth, kChannels, kPooled, kPooled, kSamples,
           kSamples, 1.0F);
    const auto end = std::chrono::steady_clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(end - begin).count());
  }
  return median(samples);
}

} // namespace

int main() {
  std::vector<float> source(kBatch * kHeight * kWidth * kChannels);
  for (std::size_t index = 0; index < source.size(); ++index)
    source[index] =
        static_cast<float>(static_cast<int>((index * 17U) % 257U) - 128) /
        1024.0F;
  std::vector<float> boxes(kBoxes * 4);
  std::vector<std::uint32_t> batchIndices(kBoxes);
  for (std::size_t box = 0; box < kBoxes; ++box) {
    const float x1 = static_cast<float>((box * 13U) % 32U) + 0.25F;
    const float y1 = static_cast<float>((box * 19U) % 32U) + 0.5F;
    boxes[box * 4 + 0] = x1;
    boxes[box * 4 + 1] = y1;
    boxes[box * 4 + 2] = std::min(63.0F, x1 + 8.0F + box % 21U);
    boxes[box * 4 + 3] = std::min(63.0F, y1 + 7.0F + box % 23U);
    batchIndices[box] = static_cast<std::uint32_t>(box % kBatch);
  }

  const std::size_t outputElements =
      kBoxes * kPooled * kPooled * kChannels;
  std::vector<float> primary(outputElements, 0.0F);
  std::vector<float> equivalent(outputElements, 0.0F);
  roi_align_f32(source.data(), boxes.data(), batchIndices.data(), primary.data(),
                0, kBoxes, kHeight, kWidth, kChannels, kPooled, kPooled,
                kSamples, kSamples, 1.0F);
  roi_align_f32_equivalent(
      source.data(), boxes.data(), batchIndices.data(), equivalent.data(), 0,
      kBoxes, kHeight, kWidth, kChannels, kPooled, kPooled, kSamples, kSamples,
      1.0F);
  double maxAbsolute = 0.0;
  double maxRelative = 0.0;
  for (std::size_t index = 0; index < outputElements; ++index) {
    const double error = std::abs(static_cast<double>(primary[index]) -
                                  static_cast<double>(equivalent[index]));
    maxAbsolute = std::max(maxAbsolute, error);
    maxRelative = std::max(
        maxRelative,
        error / std::max(std::abs(static_cast<double>(primary[index])), 1.0e-6));
  }
  if (maxAbsolute != 0.0 || maxRelative != 0.0) {
    std::fprintf(stderr, "roi align equivalent mismatch: abs=%g rel=%g\n",
                 maxAbsolute, maxRelative);
    return 1;
  }

  std::vector<std::uint8_t> eviction(kEvictionBytes, 1);
  const double primaryMilliseconds =
      run(roi_align_f32, source, boxes, batchIndices, primary, eviction);
  const double equivalentMilliseconds = run(
      roi_align_f32_equivalent, source, boxes, batchIndices, equivalent,
      eviction);
  const double bytes =
      static_cast<double>(outputElements) * sizeof(float) * 17.0;
  std::printf("kernel=roi_align_f32\n");
  std::printf(
      "model_shape=detection_roi[N=%zu,C=%zu,H=%zu,W=%zu,boxes=%zu,pooled=%zux%zu,samples=%zux%zu]\n",
      kBatch, kChannels, kHeight, kWidth, kBoxes, kPooled, kPooled, kSamples,
      kSamples);
  std::printf("max_absolute_error=%.9g\n", maxAbsolute);
  std::printf("max_relative_error=%.9g\n", maxRelative);
  std::printf("primary_median_ms=%.6f\n", primaryMilliseconds);
  std::printf("equivalent_median_ms=%.6f\n", equivalentMilliseconds);
  std::printf("primary_logical_gbs=%.6f\n",
              bytes / primaryMilliseconds / 1.0e6);
  std::printf("equivalent_logical_gbs=%.6f\n",
              bytes / equivalentMilliseconds / 1.0e6);
  return 0;
}
