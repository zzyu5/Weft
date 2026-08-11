#define _POSIX_C_SOURCE 200809L

#include "../common/ggml_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void ggml_vec_dot_q4_K_q8_K(
    int n,
    float *s,
    size_t bs,
    const void *vx,
    size_t bx,
    const void *vy,
    size_t by,
    int nrc);

typedef struct {
  const char *argument;
  const char *tensor;
  size_t n;
  size_t k;
} projection;

static const projection projections[] = {
    {"attn_q", "blk.0.attn_q.weight", 4096, 4096},
    {"attn_k", "blk.0.attn_k.weight", 1024, 4096},
    {"attn_output", "blk.0.attn_output.weight", 4096, 4096},
    {"ffn_gate", "blk.0.ffn_gate.weight", 14336, 4096},
    {"ffn_up", "blk.0.ffn_up.weight", 14336, 4096},
};

static const projection *select_projection(const char *argument) {
  for (size_t i = 0; i < sizeof(projections) / sizeof(projections[0]); ++i) {
    if (strcmp(argument, projections[i].argument) == 0) {
      return &projections[i];
    }
  }
  return NULL;
}

static size_t select_m(const char *phase) {
  if (strcmp(phase, "decode") == 0) {
    return 1;
  }
  if (strcmp(phase, "prefill") == 0) {
    return 128;
  }
  return 0;
}

static size_t parse_repetitions(const char *argument) {
  char *end = NULL;
  const unsigned long value = strtoul(argument, &end, 10);
  if (*argument == '\0' || *end != '\0' || value == 0) {
    return 0;
  }
  return (size_t)value;
}

static double now_seconds(void) {
  struct timespec value;
  clock_gettime(CLOCK_MONOTONIC, &value);
  return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static void initialize_weights(block_q4_K *weights, size_t count) {
  static const uint8_t scales[GGML_K_SCALE_SIZE] = {
      1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 0x21u, 0x43u, 0x65u, 0x07u,
  };
  for (size_t block = 0; block < count; ++block) {
    block_q4_K *value = &weights[block];
    value->d = 0x3800u;
    value->dmin = 0x3400u;
    memcpy(value->scales, scales, sizeof(scales));
    for (size_t i = 0; i < GGML_QK_K / 2; ++i) {
      const uint8_t low = (uint8_t)((3 * i + block + 1) & 0x0f);
      const uint8_t high = (uint8_t)((5 * i + 3 * block + 2) & 0x0f);
      value->qs[i] = (uint8_t)(low | (uint8_t)(high << 4));
    }
  }
}

static void initialize_activations(block_q8_K *activations, size_t count) {
  for (size_t block = 0; block < count; ++block) {
    block_q8_K *value = &activations[block];
    value->d = 0.125f;
    for (size_t i = 0; i < GGML_QK_K; ++i) {
      value->qs[i] = (int8_t)((i + 5 * block) % 17 - 8);
    }
    for (size_t group = 0; group < GGML_QK_K / 16; ++group) {
      int sum = 0;
      for (size_t i = 0; i < 16; ++i) {
        sum += value->qs[group * 16 + i];
      }
      value->bsums[group] = (int16_t)sum;
    }
  }
}

static void run_weft(
    const block_q4_K *weights,
    const block_q8_K *activations,
    float *output,
    size_t m,
    size_t n,
    size_t k_blocks) {
  for (size_t row = 0; row < m; ++row) {
    const block_q8_K *activation = activations + row * k_blocks;
    for (size_t column = 0; column < n; ++column) {
      const block_q4_K *weight = weights + column * k_blocks;
      output[row * n + column] = q4_K_q8_K(weight, activation, k_blocks);
    }
  }
}

static void run_ggml(
    const block_q4_K *weights,
    const block_q8_K *activations,
    float *output,
    size_t m,
    size_t n,
    size_t k,
    size_t k_blocks) {
  for (size_t row = 0; row < m; ++row) {
    const block_q8_K *activation = activations + row * k_blocks;
    for (size_t column = 0; column < n; ++column) {
      const block_q4_K *weight = weights + column * k_blocks;
      ggml_vec_dot_q4_K_q8_K(
          (int)k,
          &output[row * n + column],
          0,
          weight,
          0,
          activation,
          0,
          1);
    }
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(
        stderr,
        "usage: %s <attn_q|attn_k|attn_output|ffn_gate|ffn_up> "
        "<decode|prefill> <repetitions>\n",
        argv[0]);
    return 2;
  }

  const projection *selected = select_projection(argv[1]);
  if (selected == NULL) {
    fprintf(stderr, "unsupported DeepSeek 8B Q4_K tensor: %s\n", argv[1]);
    return 2;
  }
  const size_t m = select_m(argv[2]);
  if (m == 0) {
    fprintf(stderr, "unsupported phase: %s\n", argv[2]);
    return 2;
  }
  const size_t repetitions = parse_repetitions(argv[3]);
  if (repetitions == 0) {
    fprintf(stderr, "repetitions must be a positive integer\n");
    return 2;
  }

  const size_t n = selected->n;
  const size_t k = selected->k;
  const size_t k_blocks = k / GGML_QK_K;
  const size_t weight_blocks = n * k_blocks;
  const size_t activation_blocks = m * k_blocks;
  const size_t output_elements = m * n;

  block_q4_K *weights = malloc(weight_blocks * sizeof(*weights));
  block_q8_K *activations = malloc(activation_blocks * sizeof(*activations));
  float *weft_output = malloc(output_elements * sizeof(*weft_output));
  float *ggml_output = malloc(output_elements * sizeof(*ggml_output));
  if (weights == NULL || activations == NULL || weft_output == NULL ||
      ggml_output == NULL) {
    fprintf(stderr, "failed to allocate model-scale projection buffers\n");
    free(weights);
    free(activations);
    free(weft_output);
    free(ggml_output);
    return 1;
  }

  initialize_weights(weights, weight_blocks);
  initialize_activations(activations, activation_blocks);

  run_weft(weights, activations, weft_output, m, n, k_blocks);
  run_ggml(weights, activations, ggml_output, m, n, k, k_blocks);

  double max_absolute_error = 0.0;
  double max_relative_error = 0.0;
  for (size_t i = 0; i < output_elements; ++i) {
    const double absolute_error = fabs((double)weft_output[i] - ggml_output[i]);
    const double scale = fmax(1.0, fabs((double)ggml_output[i]));
    const double relative_error = absolute_error / scale;
    if (absolute_error > max_absolute_error) {
      max_absolute_error = absolute_error;
    }
    if (relative_error > max_relative_error) {
      max_relative_error = relative_error;
    }
    if (relative_error > 1.0e-4) {
      fprintf(
          stderr,
          "mismatch at output[%zu]: weft=%g ggml=%g relative_error=%g\n",
          i,
          weft_output[i],
          ggml_output[i],
          relative_error);
      free(weights);
      free(activations);
      free(weft_output);
      free(ggml_output);
      return 1;
    }
  }

  double weft_total = 0.0;
  double ggml_total = 0.0;
  for (size_t repetition = 0; repetition < repetitions; ++repetition) {
    double begin = now_seconds();
    run_weft(weights, activations, weft_output, m, n, k_blocks);
    weft_total += now_seconds() - begin;

    begin = now_seconds();
    run_ggml(weights, activations, ggml_output, m, n, k, k_blocks);
    ggml_total += now_seconds() - begin;
  }

  const double weft_seconds = weft_total / (double)repetitions;
  const double ggml_seconds = ggml_total / (double)repetitions;
  const double operations = 2.0 * (double)m * (double)n * (double)k;

  printf("kernel=q4_K_q8_K\n");
  printf("model=DeepSeek-R1-Distill-Llama-8B-Q4_K_M\n");
  printf("tensor=%s\n", selected->tensor);
  printf("phase=%s\n", argv[2]);
  printf("M=%zu\nN=%zu\nK=%zu\n", m, n, k);
  printf("scope=single-thread-full-projection-block-dot\n");
  printf("repetitions=%zu\n", repetitions);
  printf("max_absolute_error=%.9g\n", max_absolute_error);
  printf("max_relative_error=%.9g\n", max_relative_error);
  printf("weft_ms=%.6f\n", weft_seconds * 1.0e3);
  printf("ggml_ms=%.6f\n", ggml_seconds * 1.0e3);
  printf("weft_gop_s=%.6f\n", operations / weft_seconds / 1.0e9);
  printf("ggml_gop_s=%.6f\n", operations / ggml_seconds / 1.0e9);
  printf("weft_over_ggml=%.6f\n", ggml_seconds / weft_seconds);

  free(weights);
  free(activations);
  free(weft_output);
  free(ggml_output);
  return 0;
}
