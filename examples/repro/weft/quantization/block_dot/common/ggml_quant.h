#ifndef GGML_SOURCE_QUANT_H
#define GGML_SOURCE_QUANT_H

#include <stddef.h>
#include <stdint.h>

enum {
  GGML_QK32 = 32,
  GGML_QK_K = 256,
  GGML_K_SCALE_SIZE = 12,
};

typedef uint16_t ggml_half;

typedef struct {
  ggml_half d;
  uint8_t qs[GGML_QK32 / 2];
} block_q4_0;

typedef struct {
  ggml_half d;
  ggml_half m;
  uint8_t qs[GGML_QK32 / 2];
} block_q4_1;

typedef struct {
  ggml_half d;
  uint8_t qh[4];
  uint8_t qs[GGML_QK32 / 2];
} block_q5_0;

typedef struct {
  ggml_half d;
  ggml_half m;
  uint8_t qh[4];
  uint8_t qs[GGML_QK32 / 2];
} block_q5_1;

typedef struct {
  ggml_half d;
  int8_t qs[GGML_QK32];
} block_q8_0;

typedef struct {
  ggml_half d;
  ggml_half s;
  int8_t qs[GGML_QK32];
} block_q8_1;

typedef struct {
  ggml_half d;
  ggml_half dmin;
  uint8_t scales[GGML_K_SCALE_SIZE];
  uint8_t qs[GGML_QK_K / 2];
} block_q4_K;

typedef struct {
  float d;
  int8_t qs[GGML_QK_K];
  int16_t bsums[GGML_QK_K / 16];
} block_q8_K;

_Static_assert(sizeof(block_q4_0) == 18, "unexpected block_q4_0 layout");
_Static_assert(sizeof(block_q4_1) == 20, "unexpected block_q4_1 layout");
_Static_assert(sizeof(block_q5_0) == 22, "unexpected block_q5_0 layout");
_Static_assert(sizeof(block_q5_1) == 24, "unexpected block_q5_1 layout");
_Static_assert(sizeof(block_q8_0) == 34, "unexpected block_q8_0 layout");
_Static_assert(sizeof(block_q8_1) == 36, "unexpected block_q8_1 layout");
_Static_assert(sizeof(block_q4_K) == 144, "unexpected block_q4_K layout");
_Static_assert(sizeof(block_q8_K) == 292, "unexpected block_q8_K layout");

static inline void repro_get_scale_min_k4(
    int index,
    const uint8_t *packed,
    uint8_t *scale,
    uint8_t *minimum) {
  if (index < 4) {
    *scale = packed[index] & 63u;
    *minimum = packed[index + 4] & 63u;
  } else {
    *scale = (packed[index + 4] & 0x0fu) |
             ((packed[index - 4] >> 6) << 4);
    *minimum = (packed[index + 4] >> 4) |
               ((packed[index] >> 6) << 4);
  }
}

#endif
