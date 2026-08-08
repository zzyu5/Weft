// GGML FACTORY q8_0 x q8_0 vec_dot -- the REAL contribution baseline.
//
// Body copied VERBATIM from the __riscv_v branch of ggml_vec_dot_q8_0_q8_0 at
//   llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c:452-470
// (vl=qk=32 fixed, i8m2 loads, vwmul i16m4, vmv_v_x_i32m1(0,vl) seed,
//  vwredsum i16m4_i32m1, vmv.x.s, then sumf += sumi*(dx*dy)).
// Wrapped in the driver's 9-role ABI + exported symbol so the SAME
// q8_0_verify_driver.c can call it. GGML_CPU_FP16_TO_FP32 is realized with the
// same `(float)*(const _Float16*)` idiom our emitted kernels use (reads the SAME
// stored fp16 bytes -> bit-exact vs the scalar ref, matched fmaf fold).
//
// Built with the SAME local cross-clang as our e8m2/e8m1 cores
// (clang++-20 -target riscv64 -O2 -march=rv64gcv -x c++). On VLEN256 this is the
// vl=32-of-VLMAX64 HALF-filled i8m2 core -- structurally identical to our e8m2.

#include <stddef.h>
#include <stdint.h>
#include <riscv_vector.h>

#define QK8_0 32
typedef struct __attribute__((packed)) { uint16_t d; int8_t qs[QK8_0]; } block_q8_0;

static inline float g_f16_to_f32(uint16_t h) { return (float)*(const _Float16 *)&h; }

extern "C" void tcrv_emitc_ggml_vec_dot_q8_0_q8_0_kernel_rvv_q8_0_q8_0_block_dot(
    size_t n, float* s, size_t bs, const uint8_t* vx, size_t bx,
    const uint8_t* vy, size_t by, int32_t nrc, const int32_t* zero_seed) {
  (void)bs; (void)bx; (void)by; (void)nrc; (void)zero_seed;
  const int qk = QK8_0;
  const int nb = (int)(n / (size_t)qk);
  const block_q8_0* x = (const block_q8_0*)vx;
  const block_q8_0* y = (const block_q8_0*)vy;
  int ib = 0;
  float sumf = 0;
  size_t vl = qk;
  for (; ib < nb; ++ib) {
    vint8m2_t bx_0 = __riscv_vle8_v_i8m2(x[ib].qs, vl);
    vint8m2_t by_0 = __riscv_vle8_v_i8m2(y[ib].qs, vl);
    vint16m4_t vw_mul = __riscv_vwmul_vv_i16m4(bx_0, by_0, vl);
    vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t v_sum = __riscv_vwredsum_vs_i16m4_i32m1(vw_mul, v_zero, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(v_sum);
    sumf += sumi * (g_f16_to_f32(x[ib].d) * g_f16_to_f32(y[ib].d));
  }
  *s = sumf;
}
