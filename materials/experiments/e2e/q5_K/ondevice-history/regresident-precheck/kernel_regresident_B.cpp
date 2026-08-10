// REGISTER-RESIDENT q5_K x q8_K vec_dot -- PROTOTYPE (Option B: m2 strip-resident decode).
//
// Same goal as Option A (kill aux8[256] round-trip, keep deferred 8-lane accumulate,
// stay bit-exact) but decodes the WHOLE 32-element sub-block strip ONCE at LMUL m2 into
// a vector register (like the factory's register-resident decode), then slice-extracts
// the four 8-element quarters (vslidedown + vlmul_trunc) to feed the 8-lane vwmacc.
//
// This is the HIGH-register-pressure form the spill pre-check targets: the decoded m2
// strip (2 vregs) AND the loaded q8 m2 strip (2 vregs) are held live across the 4-quarter
// MAC, on top of the i32m2 accumulator -- so it genuinely stresses the LMUL budget and is
// the honest test of "does holding a decoded strip in registers introduce a new spill?".
//
// Decoded values are byte-identical to the shipped aux8 decode (same nibble/qh-bit
// arithmetic); the 8-lane aux32 mapping and the fold are identical -> bit-exact.

#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

extern "C" void tcrv_emitc_ggml_vec_dot_q5_K_q8_K_kernel_rvv_q5_K_q8_K_block_dot(
    size_t v1, float* v2, const uint8_t* v3, const uint8_t* v4) {
  size_t nb = v1 / 256;

  uint32_t utmp[4];
  float sums8[8];

  vfloat32m2_t sums = __riscv_vfmv_v_f_f32m2(0.0f, 8);
  float sumf = 0.0f;

  for (size_t ib = 0; ib < nb; ib += 1) {
    const uint8_t* xb = v3 + ib * 176;
    const uint8_t* yb = v4 + ib * 292;

    const uint32_t* sc = (const uint32_t*)(xb + 4);
    uint32_t s0 = sc[0], s1 = sc[1], s2 = sc[2];
    utmp[0] = s0 & 0x3f3f3f3f;
    utmp[1] = (s2 & 0x0f0f0f0f) | (((s0 >> 6) & 0x03030303) << 4);
    utmp[2] = s1 & 0x3f3f3f3f;
    utmp[3] = ((s2 >> 4) & 0x0f0f0f0f) | (((s1 >> 6) & 0x03030303) << 4);
    const uint8_t* scales = (const uint8_t*)&utmp[0];
    const uint8_t* mins = (const uint8_t*)&utmp[2];

    const int8_t* q8 = (const int8_t*)(yb + 4);
    const uint8_t* qs_base = xb + 48;
    const uint8_t* qh_base = xb + 16;

    vint32m2_t aux32 = __riscv_vmv_v_x_i32m2(0, 8);

    for (size_t j = 0; j < 8; j += 1) {
      int scale = (int)scales[j];
      size_t p = j >> 1;

      // --- REGISTER-RESIDENT: decode the full 32-element strip ONCE at m2 (no store) ---
      size_t vl32 = __riscv_vsetvl_e8m2(32);
      vuint8m2_t vqs = __riscv_vle8_v_u8m2(qs_base + p * 32, vl32);
      vuint8m2_t vqh = __riscv_vle8_v_u8m2(qh_base, vl32);
      vuint8m2_t nib = (j & 1) ? __riscv_vsrl_vx_u8m2(vqs, 4, vl32)
                               : __riscv_vand_vx_u8m2(vqs, 0x0F, vl32);
      vuint8m2_t bit = __riscv_vand_vx_u8m2(__riscv_vsrl_vx_u8m2(vqh, j, vl32), 0x01, vl32);
      vuint8m2_t hi = __riscv_vsll_vx_u8m2(bit, 4, vl32);
      vint8m2_t a_strip = __riscv_vreinterpret_v_u8m2_i8m2(__riscv_vadd_vv_u8m2(nib, hi, vl32));
      // q8 strip stays register-resident too
      vint8m2_t q8_strip = __riscv_vle8_v_i8m2(q8 + j * 32, vl32);

      // --- 4 quarters sliced from the resident strips, fed into the 8-lane vwmacc ---
      for (size_t q = 0; q < 4; q += 1) {
        size_t vl8 = __riscv_vsetvl_e8mf2(8);
        vint8mf2_t a = __riscv_vlmul_trunc_v_i8m2_i8mf2(
            __riscv_vslidedown_vx_i8m2(a_strip, q * 8, vl32));
        vint8mf2_t q8q = __riscv_vlmul_trunc_v_i8m2_i8mf2(
            __riscv_vslidedown_vx_i8m2(q8_strip, q * 8, vl32));
        vint16m1_t pr = __riscv_vwmul_vv_i16m1(a, q8q, vl8);
        aux32 = __riscv_vwmacc_vx_i32m2(aux32, scale, pr, vl8);
      }
    }

    const float yd = *(const float*)yb;
    float xd = (float)*(const _Float16*)(xb);
    float d = xd * yd;
    vfloat32m2_t auxf = __riscv_vfcvt_f_x_v_f32m2(aux32, 8);
    vfloat32m2_t scaled = __riscv_vfmul_vf_f32m2(auxf, d, 8);
    sums = __riscv_vfadd_vv_f32m2(sums, scaled, 8);

    const int16_t* bsums = (const int16_t*)(yb + 260);
    int sumi = 0;
    for (int t = 0; t < 16; ++t) sumi += (int)bsums[t] * (int)mins[t / 2];
    float xdmin = (float)*(const _Float16*)(xb + 2);
    float dmin = xdmin * yd;
    sumf = sumf - dmin * (float)sumi;
  }

  __riscv_vse32_v_f32m2(sums8, sums, 8);
  float acc = sumf;
  for (int l = 0; l < 8; ++l) acc += sums8[l];
  v2[0] = acc;
}
