/* Adapted from ggml generic quantized dot code. See source/c/ggml/LICENSE. */

#include "../common/ggml_quant.h"

#include <string.h>

float ggml_source_q4_K_q8_K(
    const block_q4_K *x,
    const block_q8_K *y,
    size_t blocks) {
  static const uint32_t mask_6bit = 0x3f3f3f3fu;
  static const uint32_t mask_nibble = 0x0f0f0f0fu;
  static const uint32_t mask_2bit = 0x03030303u;

  float lane_sums[8] = {0.0f};
  float minimum_sum = 0.0f;

  for (size_t block = 0; block < blocks; ++block) {
    int8_t decoded[GGML_QK_K];
    int32_t lane_acc[8] = {0};
    uint32_t unpacked[4] = {0};
    const uint8_t *packed_q4 = x[block].qs;

    int8_t *next = decoded;
    for (int group = 0; group < GGML_QK_K / 64; ++group) {
      for (int i = 0; i < 32; ++i) {
        next[i] = (int8_t)(packed_q4[i] & 0x0f);
      }
      next += 32;
      for (int i = 0; i < 32; ++i) {
        next[i] = (int8_t)(packed_q4[i] >> 4);
      }
      next += 32;
      packed_q4 += 32;
    }

    memcpy(unpacked, x[block].scales, GGML_K_SCALE_SIZE);
    unpacked[3] = ((unpacked[2] >> 4) & mask_nibble) |
                  (((unpacked[1] >> 6) & mask_2bit) << 4);
    const uint32_t saved = unpacked[1] & mask_6bit;
    unpacked[1] = (unpacked[2] & mask_nibble) |
                  (((unpacked[0] >> 6) & mask_2bit) << 4);
    unpacked[2] = saved;
    unpacked[0] &= mask_6bit;

    const uint8_t *scales = (const uint8_t *)&unpacked[0];
    const uint8_t *minimums = (const uint8_t *)&unpacked[2];

    int integer_minimum_sum = 0;
    for (int group = 0; group < GGML_QK_K / 16; ++group) {
      integer_minimum_sum += y[block].bsums[group] * minimums[group / 2];
    }

    const int8_t *q8 = y[block].qs;
    const int8_t *q4 = decoded;
    int scale_index = 0;
    for (int group = 0; group < GGML_QK_K / 32; ++group) {
      const int scale = scales[scale_index++];
      for (int chunk = 0; chunk < 4; ++chunk) {
        for (int lane = 0; lane < 8; ++lane) {
          lane_acc[lane] += scale * q8[lane] * q4[lane];
        }
        q8 += 8;
        q4 += 8;
      }
    }

    const float scale = ggml_ref_fp16_to_fp32(x[block].d) * y[block].d;
    for (int lane = 0; lane < 8; ++lane) {
      lane_sums[lane] += scale * (float)lane_acc[lane];
    }
    minimum_sum -= ggml_ref_fp16_to_fp32(x[block].dmin) * y[block].d *
                   (float)integer_minimum_sum;
  }

  for (int lane = 0; lane < 8; ++lane) {
    minimum_sum += lane_sums[lane];
  }
  return minimum_sum;
}
