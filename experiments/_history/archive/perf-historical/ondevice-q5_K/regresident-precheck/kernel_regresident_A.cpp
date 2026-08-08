// REGISTER-RESIDENT q5_K x q8_K vec_dot -- PROTOTYPE (Option A: per-quarter inline decode).
//
// Machine-level pre-check prototype for GAP-1/q5_K-aux8-roundtrip (P3).
// Goal: eliminate the aux8[256] scratch round-trip (8x vse8 store + 32x vle8 read-back)
// that our shipped kernel inherited from mirroring the ggml `_generic` decode pass,
// WHILE keeping our structural win (deferred 8-lane vwmacc accumulate, ZERO in-loop
// vredsum) and staying BIT-EXACT to the ggml scalar q5_K generic oracle.
//
// HOW: our shipped kernel decodes all 256 q5 values into aux8[256] (m2, 8 vse8), then
// the MAC reads them back in 32 mf2 quarters (vle8 aux8+off). This prototype DELETES the
// aux8 pre-pass and instead decodes each 8-element quarter INLINE, immediately before the
// vwmul/vwmacc that consumes it -- so the decoded strip stays in a vector register and
// feeds the MAC directly. No aux8 store, no aux8 read-back.
//
// The decoded value a[pos] for aux8 strip s (== sub-block j), quarter q, lane l is,
// verbatim with the shipped decode:
//     p        = j >> 1                       // which 32-byte qs group
//     nib      = (j&1) ? (qs>>4) : (qs & 0xF) // high / low nibble
//     qh_bit   = (qh >> j) & 1                // 5th bit plane, bit index = j
//     a        = (int8_t)(nib + (qh_bit<<4))
// with qs element = x.qs[p*32 + q*8 + l], qh element = x.qh[q*8 + l], q8 element =
// y.qs[j*32 + q*8 + l]. Same products -> same 8-lane aux32 -> same fold -> bit-exact.
//
// Shared symbol + 4-role ABI identical to kernel_ours.cpp / kernel_factory.c so the
// SAME q5k_verify_driver.c links it.  Compile -x c++ (mlir-to-cpp path).  DO NOT commit.

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
    const uint8_t* xb = v3 + ib * 176;  // q5_K block base
    const uint8_t* yb = v4 + ib * 292;  // q8_K block base

    // --- scale/min 6-bit bit-dance (verbatim from shipped kernel) ---
    const uint32_t* sc = (const uint32_t*)(xb + 4);
    uint32_t s0 = sc[0], s1 = sc[1], s2 = sc[2];
    utmp[0] = s0 & 0x3f3f3f3f;
    utmp[1] = (s2 & 0x0f0f0f0f) | (((s0 >> 6) & 0x03030303) << 4);
    utmp[2] = s1 & 0x3f3f3f3f;
    utmp[3] = ((s2 >> 4) & 0x0f0f0f0f) | (((s1 >> 6) & 0x03030303) << 4);
    const uint8_t* scales = (const uint8_t*)&utmp[0];  // scales[0..8]
    const uint8_t* mins = (const uint8_t*)&utmp[2];    // mins[0..8]

    const int8_t* q8 = (const int8_t*)(yb + 4);   // q8_K quants
    const uint8_t* qs_base = xb + 48;             // q5_K low-nibble quants (128 bytes)
    const uint8_t* qh_base = xb + 16;             // q5_K high bit plane (32 bytes)

    // --- deferred 8-lane integer accumulator (our win: NO in-loop vredsum) ---
    vint32m2_t aux32 = __riscv_vmv_v_x_i32m2(0, 8);

    for (size_t j = 0; j < 8; j += 1) {
      int scale = (int)scales[j];
      size_t p = j >> 1;
      const uint8_t* qs_grp = qs_base + p * 32;
      for (size_t q = 0; q < 4; q += 1) {
        size_t qoff = q * 8;
        size_t yoff = j * 32 + qoff;
        size_t vl = __riscv_vsetvl_e8mf2(8);

        // --- REGISTER-RESIDENT inline decode of this 8-element quarter (no aux8) ---
        vuint8mf2_t vqs = __riscv_vle8_v_u8mf2(qs_grp + qoff, vl);
        vuint8mf2_t vqh = __riscv_vle8_v_u8mf2(qh_base + qoff, vl);
        vuint8mf2_t nib = (j & 1) ? __riscv_vsrl_vx_u8mf2(vqs, 4, vl)
                                  : __riscv_vand_vx_u8mf2(vqs, 0x0F, vl);
        vuint8mf2_t bit = __riscv_vand_vx_u8mf2(__riscv_vsrl_vx_u8mf2(vqh, j, vl), 0x01, vl);
        vuint8mf2_t hi = __riscv_vsll_vx_u8mf2(bit, 4, vl);
        vint8mf2_t a = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vadd_vv_u8mf2(nib, hi, vl));

        // --- MAC: same vwmul + deferred vwmacc as the shipped kernel ---
        vint8mf2_t q8q = __riscv_vle8_v_i8mf2(q8 + yoff, vl);
        vint16m1_t pr = __riscv_vwmul_vv_i16m1(a, q8q, vl);
        aux32 = __riscv_vwmacc_vx_i32m2(aux32, scale, pr, vl);
      }
    }

    // --- fold (verbatim order from shipped kernel): main term into sums (8-lane) ---
    const float yd = *(const float*)yb;
    float xd = (float)*(const _Float16*)(xb);
    float d = xd * yd;
    vfloat32m2_t auxf = __riscv_vfcvt_f_x_v_f32m2(aux32, 8);
    vfloat32m2_t scaled = __riscv_vfmul_vf_f32m2(auxf, d, 8);
    sums = __riscv_vfadd_vv_f32m2(sums, scaled, 8);

    // --- mins term into sumf (verbatim) ---
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
