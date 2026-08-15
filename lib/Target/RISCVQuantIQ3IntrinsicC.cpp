#include "RISCVQuantIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitIQ3IntrinsicCLeaves(llvm::raw_ostream &output, bool scalable) {
  if (!scalable)
    return;
  output << R"c(static inline __attribute__((always_inline, unused)) float
__weft_iq3_s_i8_rvv(
    const uint8_t *codes, const uint8_t *high_bits,
    const uint8_t *sign_bits, const uint8_t *scales,
    const uint8_t *activation_bytes, float weight_scale,
    float activation_scale, float init) {
  const int8_t *activation = (const int8_t *)(const void *)activation_bytes;
  int8_t decoded[32];
  int32_t integer_sum = 0;
  for (size_t group = 0; group < 8; ++group) {
    for (size_t vector = 0; vector < 8; ++vector) {
      const uint16_t index = (uint16_t)codes[group * 8 + vector] |
          (uint16_t)(((uint16_t)high_bits[group] << (8 - vector)) & 0x100);
      const int8_t *grid = (const int8_t *)(const void *)&__weft_iq3_s_grid[index];
      const uint8_t signs = sign_bits[group * 4 + vector / 2];
      const size_t sign_base = (vector & 1) * 4;
      for (size_t lane = 0; lane < 4; ++lane)
        decoded[vector * 4 + lane] =
            (signs & (UINT8_C(1) << (sign_base + lane))) ? -grid[lane]
                                                         : grid[lane];
    }
    const int32_t dot = __weft_i8_dot(decoded, activation + group * 32, 32);
    const uint8_t packed_scale = scales[group / 2];
    const int32_t scale =
        1 + 2 * ((group & 1) ? (packed_scale >> 4) : (packed_scale & 15));
    integer_sum += dot * scale;
  }
  return init + (float)integer_sum * weight_scale * activation_scale;
}
)c";
}

} // namespace weft::riscv_internal
