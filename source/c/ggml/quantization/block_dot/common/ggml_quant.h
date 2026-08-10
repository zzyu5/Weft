#ifndef WEFT_SOURCE_GGML_QUANT_H
#define WEFT_SOURCE_GGML_QUANT_H

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

static inline float ggml_ref_fp16_to_fp32(ggml_half value) {
  const uint32_t sign = (uint32_t)(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x03ffu;
  uint32_t bits;

  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      exponent = 113;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --exponent;
      }
      mantissa &= 0x03ffu;
      bits = sign | (exponent << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 112) << 23) | (mantissa << 13);
  }

  union {
    uint32_t bits;
    float value;
  } converted = {bits};
  return converted.value;
}

static inline void ggml_ref_get_scale_min_k4(
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

float weft_reference_q4_0_q8_0(
    const block_q4_0 *x,
    const block_q8_0 *y,
    size_t blocks);
float weft_reference_q4_1_q8_1(
    const block_q4_1 *x,
    const block_q8_1 *y,
    size_t blocks);
float weft_reference_q5_0_q8_0(
    const block_q5_0 *x,
    const block_q8_0 *y,
    size_t blocks);
float weft_reference_q5_1_q8_1(
    const block_q5_1 *x,
    const block_q8_1 *y,
    size_t blocks);
float weft_reference_q8_0_q8_0(
    const block_q8_0 *x,
    const block_q8_0 *y,
    size_t blocks);
float weft_reference_q4_K_q8_K(
    const block_q4_K *x,
    const block_q8_K *y,
    size_t blocks);

#endif
