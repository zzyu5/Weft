// HAND-WRITTEN e8m1 q8_0 x q8_0 vec_dot reference kernel -- measurement probe only.
//
// PURPOSE: hardware fact test. The typed region currently pins integer_core_lmul=m2
// (cannot express m1), so to measure whether an e8m1 core (which EXACTLY fills the
// vector register on VLEN256: VLMAX_e8m1 = 256/8 = 32 = one q8_0 block) beats the
// half-filled e8m2 core (vl=32 of VLMAX=64), we hand-write the m1 analog.
//
// It is byte-for-byte the compiler-EMITTED e8m2 core (experiments/.../ir/kernel_mbf1.c)
// with ONLY the LMUL swapped:
//   vsetvl_e8m2 -> vsetvl_e8m1 ; i8m2 load -> i8m1 ; vwmul i16m4 -> i16m2 ;
//   vwredsum i16m4_i32m1 -> i16m2_i32m1.
// Everything else identical: same QK=32, same block_count = n/32, same quant offset 2,
// same 34-byte block stride, same fp16->fp32 scale reads, same fold
//   sumf = sumf + (float)sumi * (dx * dy)   (contracts to fmaf at -O2, matching e8m2).
// sumi is an EXACT integer reduction, so this is bit-identical to the e8m2 core; the
// on-device driver confirms bit-exact vs the scalar q8_0 ref.
//
// One-vsetvl / NO strip loop (elided single-cover), matching the e8m2 elided core.
// NOTE: valid ONLY where VLMAX_e8m1 >= 32, i.e. VLEN>=256. On VLEN128
// (VLMAX_e8m1=16) this single-cover form would under-read a block; a strip loop
// would be needed there. This probe targets k1 (VLEN256).

#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>
extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(size_t v1, float* v2, size_t v3, const uint8_t* v4, size_t v5, const uint8_t* v6, size_t v7, int32_t v8, const int32_t* v9) {
  size_t v10 = __riscv_vsetvl_e8m1(v1);
  (void)v10; (void)v3; (void)v5; (void)v7; (void)v8; (void)v9;
  float v11;
  v11 = 0.0f;
  size_t v12 = v1 / 32;
  for (size_t v13 = 0; v13 < v12; v13 += 1) {
    size_t v14 = v13 * 34;
    const uint8_t* v15 = v4 + v14;
    size_t v16 = v13 * 34;
    const uint8_t* v17 = v6 + v16;
    float v18 = (float)*(const _Float16 *)(v15);
    float v19 = (float)*(const _Float16 *)(v17);
    int32_t v20;
    v20 = 0;
    size_t v21 = __riscv_vsetvl_e8m1(32);
    const uint8_t* v22 = v15 + 2;
    const uint8_t* v23 = v22 + 0;
    const int8_t* v24 = (const int8_t*) v23;
    vint8m1_t v25 = __riscv_vle8_v_i8m1(v24, v21);
    const uint8_t* v26 = v17 + 2;
    const uint8_t* v27 = v26 + 0;
    const int8_t* v28 = (const int8_t*) v27;
    vint8m1_t v29 = __riscv_vle8_v_i8m1(v28, v21);
    vint16m2_t v30 = __riscv_vwmul_vv_i16m2(v25, v29, v21);
    vint32m1_t v31 = __riscv_vmv_v_x_i32m1(0, 1);
    vint32m1_t v32 = __riscv_vwredsum_vs_i16m2_i32m1(v30, v31, v21);
    int32_t v33 = __riscv_vmv_x_s_i32m1_i32(v32);
    v20 = v33;
    int32_t v34 = v20;
    float v35 = v11;
    v11 = v35 + (float) v34 * (v18 * v19);
  }
  float v36 = v11;
  v2[0] = v36;
  return;
}
