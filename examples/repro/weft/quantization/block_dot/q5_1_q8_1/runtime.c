#include "../common/ggml_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
  COLUMNS = 14336,
  INNER = 4096,
  BLOCKS = INNER / GGML_QK32,
  REPETITIONS = 10,
  EVICTION_BYTES = 64 * 1024 * 1024,
};

static volatile uint64_t eviction_sink;

static double now_seconds(void) {
  struct timespec value;
  clock_gettime(CLOCK_MONOTONIC, &value);
  return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static void evict(uint8_t *buffer) {
  for (size_t index = 0; index < EVICTION_BYTES; index += 64) {
    buffer[index] = (uint8_t)(buffer[index] + 1u);
    eviction_sink += buffer[index];
  }
}

static int compare_double(const void *lhs, const void *rhs) {
  const double left = *(const double *)lhs;
  const double right = *(const double *)rhs;
  return (left > right) - (left < right);
}

static float reference_block(const block_q5_1 *weight,
                             const block_q8_1 *activation) {
  const uint32_t high_bits =
      (uint32_t)weight->qh[0] | ((uint32_t)weight->qh[1] << 8) |
      ((uint32_t)weight->qh[2] << 16) | ((uint32_t)weight->qh[3] << 24);
  float result = 0.0f;
  for (size_t lane = 0; lane < GGML_QK32 / 2; ++lane) {
    const int32_t low_bit = (int32_t)(((high_bits >> lane) << 4) & 16u);
    const int32_t high_bit =
        (int32_t)((high_bits >> (lane + 12)) & 16u);
    const float low =
        0.25f * (float)((weight->qs[lane] & 15u) | low_bit) - 0.5f;
    const float high =
        0.25f * (float)((weight->qs[lane] >> 4) | high_bit) - 0.5f;
    result += low * 0.25f * activation->qs[lane];
    result += high * 0.25f * activation->qs[lane + GGML_QK32 / 2];
  }
  return result;
}

static void run_projection(const block_q5_1 *weights,
                           const block_q8_1 *activation, float *output) {
  for (size_t column = 0; column < COLUMNS; ++column)
    output[column] = q5_1_q8_1(
        (const uint8_t *)(const void *)(weights + column * BLOCKS),
        (const uint8_t *)(const void *)activation, BLOCKS);
}

int main(void) {
  block_q5_1 *weights =
      (block_q5_1 *)malloc((size_t)COLUMNS * BLOCKS * sizeof(*weights));
  block_q8_1 *activation =
      (block_q8_1 *)malloc((size_t)BLOCKS * sizeof(*activation));
  float *output = (float *)malloc((size_t)COLUMNS * sizeof(*output));
  uint8_t *eviction = (uint8_t *)malloc(EVICTION_BYTES);
  if (weights == NULL || activation == NULL || output == NULL ||
      eviction == NULL)
    return 1;

  block_q5_1 weight = {
      .d = 0x3400u,
      .m = 0xb800u,
      .qh = {0x96u, 0x69u, 0x3cu, 0xc3u},
  };
  block_q8_1 input = {.d = 0x3400u, .s = 0x4800u};
  for (size_t lane = 0; lane < GGML_QK32 / 2; ++lane)
    weight.qs[lane] = (uint8_t)(((5 * lane + 2) & 15u) |
                                (((3 * lane) & 15u) << 4));
  for (size_t lane = 0; lane < GGML_QK32; ++lane)
    input.qs[lane] = 1;
  for (size_t index = 0; index < (size_t)COLUMNS * BLOCKS; ++index)
    weights[index] = weight;
  for (size_t block = 0; block < BLOCKS; ++block)
    activation[block] = input;
  memset(eviction, 1, EVICTION_BYTES);

  run_projection(weights, activation, output);
  const float expected = (float)BLOCKS * reference_block(&weight, &input);
  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  for (size_t column = 0; column < COLUMNS; ++column) {
    const double absolute = fabs((double)output[column] - expected);
    const double relative = absolute / fmax(1.0, fabs((double)expected));
    max_absolute_error = fmax(max_absolute_error, absolute);
    max_relative_error = fmax(max_relative_error, relative);
  }

  double samples[REPETITIONS];
  for (size_t repetition = 0; repetition < REPETITIONS; ++repetition) {
    evict(eviction);
    const double begin = now_seconds();
    run_projection(weights, activation, output);
    samples[repetition] = (now_seconds() - begin) * 1000.0;
  }
  qsort(samples, REPETITIONS, sizeof(samples[0]), compare_double);
  const double median_ms =
      (samples[REPETITIONS / 2 - 1] + samples[REPETITIONS / 2]) * 0.5;
  const double operations = 2.0 * COLUMNS * INNER;

  printf("kernel=q5_1_q8_1\n");
  printf("model_shape=M=1;N=%d;K=%d\n", COLUMNS, INNER);
  printf("correctness_scope=full_rows\n");
  printf("repetitions=%d\n", REPETITIONS);
  printf("max_absolute_error=%.9g\n", max_absolute_error);
  printf("max_relative_error=%.9g\n", max_relative_error);
  printf("median_ms=%.6f\n", median_ms);
  printf("gop_s=%.6f\n", operations / (median_ms * 1.0e6));

  free(weights);
  free(activation);
  free(output);
  free(eviction);
  return max_relative_error <= 1.0e-6 ? 0 : 1;
}
