#include "RISCVIntrinsicC.h"

#include "llvm/Support/raw_ostream.h"

namespace weft::riscv_internal {

void emitIMEIntrinsicCLeaves(llvm::raw_ostream &output,
                             bool usesIME1SymmetricI4I8,
                             bool usesIME1AffineI4I8,
                             bool usesIME1SymmetricI4I8M4,
                             bool usesIME1AffineI4I8M4) {
  if (usesIME1SymmetricI4I8 || usesIME1AffineI4I8) {
    output << R"ime(#define __WEFT_IME1_COMP_I4_I8_M1                                      \
  "vmadot       v16, v14, v0            \n\t"                         \
  "vmadot       v18, v14, v1            \n\t"                         \
  "vmadot       v20, v14, v2            \n\t"                         \
  "vmadot       v22, v14, v3            \n\t"                         \
  "vmadot       v16, v15, v4            \n\t"                         \
  "vmadot       v18, v15, v5            \n\t"                         \
  "vmadot       v20, v15, v6            \n\t"                         \
  "vmadot       v22, v15, v7            \n\t"

#define __WEFT_IME1_ACC_I4_I8_M1                                       \
  "vfcvt.f.x.v  v16, v16                \n\t"                         \
  "vfcvt.f.x.v  v18, v18                \n\t"                         \
  "vfcvt.f.x.v  v20, v20                \n\t"                         \
  "vfcvt.f.x.v  v22, v22                \n\t"                         \
  "addi         s2, s1, 8                \n\t"                         \
  "addi         s3, s1, 16               \n\t"                         \
  "addi         s4, s1, 24               \n\t"                         \
  "addi         s6, s5, 8                \n\t"                         \
  "vfmacc.vv    v28, v16, v24            \n\t"                         \
  "vfmacc.vv    v29, v18, v25            \n\t"                         \
  "vfmacc.vv    v30, v20, v26            \n\t"                         \
  "vfmacc.vv    v31, v22, v27            \n\t"

#define __WEFT_IME1_LOAD_I4_I8_M1                                      \
  "vle8.v       v4, (s1)                 \n\t"                         \
  "addi         s1, s1, 128              \n\t"                         \
  "vle8.v       v5, (s2)                 \n\t"                         \
  "addi         s2, s2, 128              \n\t"                         \
  "vle8.v       v6, (s3)                 \n\t"                         \
  "addi         s3, s3, 128              \n\t"                         \
  "vle8.v       v7, (s4)                 \n\t"                         \
  "addi         s4, s4, 128              \n\t"                         \
  "vsetvli      t0, zero, e8, mf4        \n\t"                         \
  "vle8.v       v14, (s5)                \n\t"                         \
  "addi         s5, s5, 16               \n\t"                         \
  "vle8.v       v15, (s6)                \n\t"                         \
  "addi         s6, s6, 16               \n\t"                         \
  "addi         t5, t5, -1               \n\t"                         \
  "vsetvli      t0, zero, e8, m1         \n\t"                         \
  "vand.vi      v0, v4, 15               \n\t"                         \
  "vand.vi      v1, v5, 15               \n\t"                         \
  "vand.vi      v2, v6, 15               \n\t"                         \
  "vand.vi      v3, v7, 15               \n\t"                         \
  "vsrl.vi      v4, v4, 4                \n\t"                         \
  "vsrl.vi      v5, v5, 4                \n\t"                         \
  "vsrl.vi      v6, v6, 4                \n\t"                         \
  "vsrl.vi      v7, v7, 4                \n\t"

#define __WEFT_IME1_LOAD_ZP_I4_I8_M1                                   \
  "vsetvli      t0, zero, e8, mf2       \n\t"                          \
  "vle8.v       v1, (s7)                 \n\t"                          \
  "vsetvli      t0, zero, e8, m1        \n\t"                          \
  "vrgather.vv  v8, v1, v13              \n\t"                          \
  "vadd.vi      v13, v13, 4              \n\t"                          \
  "vrgather.vv  v9, v1, v13              \n\t"                          \
  "vadd.vi      v13, v13, 4              \n\t"                          \
  "vrgather.vv  v10, v1, v13             \n\t"                          \
  "vadd.vi      v13, v13, 4              \n\t"                          \
  "vrgather.vv  v11, v1, v13             \n\t"                          \
  "vadd.vi      v13, v13, -12            \n\t"

)ime";
  }
  if (usesIME1SymmetricI4I8) {
    output << R"ime(
static inline __attribute__((unused)) void
__weft_ime1_symmetric_i4_i8_n16_k32(
    float activation_scale, const int8_t *activation_code,
    const uint8_t *packed_weight, float *accumulator) {
  __asm__ volatile(
        "vsetvli      t0, zero, e32, mf2      \n\t"
        "addi         s1, %[C], 16           \n\t"
        "addi         s2, %[C], 32           \n\t"
        "addi         s3, %[C], 48           \n\t"
        "vle32.v      v28, (%[C])            \n\t"
        "vle32.v      v29, (s1)              \n\t"
        "vle32.v      v30, (s2)              \n\t"
        "vle32.v      v31, (s3)              \n\t"
        "addi         s1, %[B], 0            \n\t"
        "addi         s2, %[B], 8            \n\t"
        "addi         s3, %[B], 16           \n\t"
        "addi         s4, %[B], 24           \n\t"
        "vsetvli      t0, zero, e16, mf4     \n\t"
        "vle16.v      v4, (s1)               \n\t"
        "vle16.v      v5, (s2)               \n\t"
        "vle16.v      v6, (s3)               \n\t"
        "vle16.v      v7, (s4)               \n\t"
        "vfwcvt.f.f.v v8, v4                 \n\t"
        "vfwcvt.f.f.v v9, v5                 \n\t"
        "vfwcvt.f.f.v v10, v6                \n\t"
        "vfwcvt.f.f.v v11, v7                \n\t"
        "vsetvli      t0, zero, e32, mf2     \n\t"
        "vxor.vv      v16, v16, v16          \n\t"
        "vxor.vv      v18, v18, v18          \n\t"
        "vxor.vv      v20, v20, v20          \n\t"
        "vxor.vv      v22, v22, v22          \n\t"
        "vfmul.vf     v24, v8, %[AS]         \n\t"
        "vfmul.vf     v25, v9, %[AS]         \n\t"
        "vfmul.vf     v26, v10, %[AS]        \n\t"
        "vfmul.vf     v27, v11, %[AS]        \n\t"
        "addi         s1, %[B], 32           \n\t"
        "addi         s2, %[B], 64           \n\t"
        "addi         s3, %[B], 96           \n\t"
        "addi         s4, %[B], 128          \n\t"
        "addi         s5, %[A], 0            \n\t"
        "addi         s6, %[A], 8            \n\t"
        "li           t5, 2                  \n\t"
        "vsetvli      t0, zero, e8, m1       \n\t"
        "LOOP_INNER%=:                       \n\t"
        __WEFT_IME1_LOAD_I4_I8_M1
        "vadd.vi      v0, v0, -8             \n\t"
        "vadd.vi      v1, v1, -8             \n\t"
        "vadd.vi      v2, v2, -8             \n\t"
        "vadd.vi      v3, v3, -8             \n\t"
        "vadd.vi      v4, v4, -8             \n\t"
        "vadd.vi      v5, v5, -8             \n\t"
        "vadd.vi      v6, v6, -8             \n\t"
        "vadd.vi      v7, v7, -8             \n\t"
        __WEFT_IME1_COMP_I4_I8_M1
        "bnez         t5, LOOP_INNER%=       \n\t"
        "vsetvli      t0, zero, e32, mf2     \n\t"
        __WEFT_IME1_ACC_I4_I8_M1
        "addi         s1, %[C], 16           \n\t"
        "addi         s2, %[C], 32           \n\t"
        "addi         s3, %[C], 48           \n\t"
        "vse32.v      v28, (%[C])            \n\t"
        "vse32.v      v29, (s1)              \n\t"
        "vse32.v      v30, (s2)              \n\t"
        "vse32.v      v31, (s3)              \n\t"
        :
        : [AS] "f"(activation_scale), [A] "r"(activation_code),
          [B] "r"(packed_weight), [C] "r"(accumulator)
        : "cc", "memory", "t0", "t5", "s1", "s2", "s3", "s4", "s5",
          "s6");
}

)ime";
  }
  if (usesIME1AffineI4I8) {
    output << R"ime(
static inline __attribute__((unused)) void __weft_ime1_affine_i4_i8_n16_k32(
    float activation_scale, const int8_t *activation_code,
    const uint8_t *packed_weight, float *accumulator) {
  const size_t inner = 2;
  const uint8_t *weight = packed_weight;

  __asm__ volatile(
        "vsetvli      t0, zero, e32, mf2      \n\t"
        "vle32.v      v28, (%[C])             \n\t"
        "addi         s1, %[C], 16            \n\t"
        "vle32.v      v29, (s1)               \n\t"
        "addi         s1, %[C], 32            \n\t"
        "vle32.v      v30, (s1)               \n\t"
        "addi         s1, %[C], 48            \n\t"
        "vle32.v      v31, (s1)               \n\t"
        "vsetvli      t0, zero, e32, m4       \n\t"
        "vsetvli      t0, zero, e8, m1        \n\t"
        "vmv.v.i      v13, 3                  \n\t"
        "li           s1, 24                  \n\t"
        "vsetvli      t0, s1, e8, m1          \n\t"
        "vmv.v.i      v13, 2                  \n\t"
        "vsetvli      t0, zero, e8, mf2       \n\t"
        "vmv.v.i      v13, 1                  \n\t"
        "vsetvli      t0, zero, e8, mf4       \n\t"
        "vmv.v.i      v13, 0                  \n\t"
        "addi         s1, %[B], 0             \n\t"
        "addi         s2, %[B], 8             \n\t"
        "addi         s3, %[B], 16            \n\t"
        "addi         s4, %[B], 24            \n\t"
        "addi         s7, %[B], 32            \n\t"
        "addi         s5, %[A], 0             \n\t"
        "addi         s6, %[A], 8             \n\t"
        "vsetvli      t0, zero, e16, mf4      \n\t"
        "vle16.v      v4, (s1)                \n\t"
        "addi         s1, s1, 48              \n\t"
        "vle16.v      v5, (s2)                \n\t"
        "addi         s2, s2, 72              \n\t"
        "vle16.v      v6, (s3)                \n\t"
        "addi         s3, s3, 96              \n\t"
        "vle16.v      v7, (s4)                \n\t"
        "addi         s4, s4, 120             \n\t"
        "fmv.s        f1, %[AS]                \n\t"
        "vfwcvt.f.f.v v8, v4                  \n\t"
        "vfwcvt.f.f.v v9, v5                  \n\t"
        "vfwcvt.f.f.v v10, v6                 \n\t"
        "vfwcvt.f.f.v v11, v7                 \n\t"
        "vsetvli      t0, zero, e32, mf2      \n\t"
        "addi         t5, %[INNER], 0         \n\t"
        "vxor.vv      v16, v16, v16           \n\t"
        "vxor.vv      v18, v18, v18           \n\t"
        "vxor.vv      v20, v20, v20           \n\t"
        "vxor.vv      v22, v22, v22           \n\t"
        "vfmul.vf     v24, v8, f1             \n\t"
        "vfmul.vf     v25, v9, f1             \n\t"
        "vfmul.vf     v26, v10, f1            \n\t"
        "vfmul.vf     v27, v11, f1            \n\t"
        __WEFT_IME1_LOAD_ZP_I4_I8_M1
        "LOOP_INNER%=:                        \n\t"
        __WEFT_IME1_LOAD_I4_I8_M1
        "vsub.vv      v0, v0, v8              \n\t"
        "vsub.vv      v4, v4, v8              \n\t"
        "vsub.vv      v1, v1, v9              \n\t"
        "vsub.vv      v5, v5, v9              \n\t"
        "vsub.vv      v2, v2, v10             \n\t"
        "vsub.vv      v6, v6, v10             \n\t"
        "vsub.vv      v3, v3, v11             \n\t"
        "vsub.vv      v7, v7, v11             \n\t"
        __WEFT_IME1_COMP_I4_I8_M1
        "bnez         t5, LOOP_INNER%=        \n\t"
        "vsetvli      t0, zero, e32, mf2      \n\t"
        __WEFT_IME1_ACC_I4_I8_M1
        "addi         s1, %[C], 16            \n\t"
        "addi         s2, %[C], 32            \n\t"
        "addi         s3, %[C], 48            \n\t"
        "vse32.v      v28, (%[C])             \n\t"
        "vse32.v      v29, (s1)               \n\t"
        "vse32.v      v30, (s2)               \n\t"
        "vse32.v      v31, (s3)               \n\t"
        :
        : [INNER] "r"(inner), [A] "r"(activation_code), [B] "r"(weight),
          [C] "r"(accumulator), [AS] "f"(activation_scale)
      : "cc", "memory", "t0", "t5", "f1", "s1", "s2", "s3",
        "s4", "s5", "s6", "s7");
}

)ime";
  }
  if (usesIME1SymmetricI4I8M4 || usesIME1AffineI4I8M4) {
    output << R"ime(
#define __WEFT_IME1_LOAD_B_M4                                            \
  "vsetvli      t0, zero, e8, m1        \n\t"                         \
  "vle8.v       v6, (s1)                 \n\t"                         \
  "addi         s1, s1, 128              \n\t"                         \
  "vle8.v       v7, (s2)                 \n\t"                         \
  "addi         s2, s2, 128              \n\t"                         \
  "vle8.v       v8, (s3)                 \n\t"                         \
  "addi         s3, s3, 128              \n\t"                         \
  "vle8.v       v9, (s4)                 \n\t"                         \
  "addi         s4, s4, 128              \n\t"                         \
  "vand.vi      v2, v6, 15               \n\t"                         \
  "vand.vi      v3, v7, 15               \n\t"                         \
  "vand.vi      v4, v8, 15               \n\t"                         \
  "vand.vi      v5, v9, 15               \n\t"                         \
  "vsrl.vi      v6, v6, 4                \n\t"                         \
  "vsrl.vi      v7, v7, 4                \n\t"                         \
  "vsrl.vi      v8, v8, 4                \n\t"                         \
  "vsrl.vi      v9, v9, 4                \n\t"

#define __WEFT_IME1_COMP_M4                                             \
  "vmadot       v16, v10, v2            \n\t"                         \
  "vmadot       v18, v10, v3            \n\t"                         \
  "vmadot       v20, v10, v4            \n\t"                         \
  "vmadot       v22, v10, v5            \n\t"                         \
  "vmadot       v16, v11, v6            \n\t"                         \
  "vmadot       v18, v11, v7            \n\t"                         \
  "vmadot       v20, v11, v8            \n\t"                         \
  "vmadot       v22, v11, v9            \n\t"

#define __WEFT_IME1_LOAD_SCALE_M4                                      \
  "addi         s2, s5, -8              \n\t"                         \
  "addi         s3, s5, 8               \n\t"                         \
  "addi         s4, s5, 16              \n\t"                         \
  "addi         s6, s5, 24              \n\t"                         \
  "li           t1, 0xf0                \n\t"                         \
  "vmv.s.x      v0, t1                  \n\t"                         \
  "vsetvli      t0, zero, e16, mf4      \n\t"                         \
  "vle16.v      v9, (s5)                \n\t"                         \
  "vle16.v      v11, (s3)               \n\t"                         \
  "vle16.v      v13, (s4)               \n\t"                         \
  "vle16.v      v15, (s6)               \n\t"                         \
  "vsetvli      t0, zero, e16, mf2      \n\t"                         \
  "vle16.v      v9, (s2), v0.t          \n\t"                         \
  "vle16.v      v11, (s5), v0.t         \n\t"                         \
  "vle16.v      v13, (s3), v0.t         \n\t"                         \
  "vle16.v      v15, (s4), v0.t         \n\t"                         \
  "vfwcvt.f.f.v v8, v9                  \n\t"                         \
  "vfwcvt.f.f.v v10, v11                \n\t"                         \
  "vfwcvt.f.f.v v12, v13                \n\t"                         \
  "vfwcvt.f.f.v v14, v15                \n\t"                         \
  "vsetvli      t0, zero, e32, m1       \n\t"                         \
  "vmv.v.v      v9, v8                  \n\t"                         \
  "vmv.v.v      v11, v10                \n\t"                         \
  "vmv.v.v      v13, v12                \n\t"                         \
  "vmv.v.v      v15, v14                \n\t"                         \
  "li           t1, 0xf0                \n\t"                         \
  "vmv.s.x      v0, t1                  \n\t"                         \
  "vsetvli      t0, zero, e32, mf2      \n\t"                         \
  "vfmul.vf     v8, v8, f1              \n\t"                         \
  "vfmul.vf     v10, v10, f1            \n\t"                         \
  "vfmul.vf     v12, v12, f1            \n\t"                         \
  "vfmul.vf     v14, v14, f1            \n\t"                         \
  "vfmul.vf     v9, v9, f3              \n\t"                         \
  "vfmul.vf     v11, v11, f3            \n\t"                         \
  "vfmul.vf     v13, v13, f3            \n\t"                         \
  "vfmul.vf     v15, v15, f3            \n\t"                         \
  "vsetvli      t0, zero, e32, m1       \n\t"                         \
  "vfmul.vf     v8, v8, f2, v0.t        \n\t"                         \
  "vfmul.vf     v10, v10, f2, v0.t      \n\t"                         \
  "vfmul.vf     v12, v12, f2, v0.t      \n\t"                         \
  "vfmul.vf     v14, v14, f2, v0.t      \n\t"                         \
  "vfmul.vf     v9, v9, f4, v0.t        \n\t"                         \
  "vfmul.vf     v11, v11, f4, v0.t      \n\t"                         \
  "vfmul.vf     v13, v13, f4, v0.t      \n\t"                         \
  "vfmul.vf     v15, v15, f4, v0.t      \n\t"

#define __WEFT_IME1_SAVE_ACC_M4                                        \
  "addi         a1, %[C], 0             \n\t"                         \
  "add          a2, %[C], %[LDC]        \n\t"                         \
  "add          a3, a2, %[LDC]          \n\t"                         \
  "add          a4, a3, %[LDC]          \n\t"                         \
  "addi         a2, a2, -16             \n\t"                         \
  "addi         a4, a4, -16             \n\t"                         \
  "li           t1, 0xf0                \n\t"                         \
  "vmv.s.x      v0, t1                  \n\t"                         \
  "vsetvli      t0, zero, e32, mf2      \n\t"                         \
  "vse32.v      v24, (a1)               \n\t"                         \
  "addi         a1, a1, 16              \n\t"                         \
  "vse32.v      v25, (a3)               \n\t"                         \
  "addi         a3, a3, 16              \n\t"                         \
  "vse32.v      v26, (a1)               \n\t"                         \
  "addi         a1, a1, 16              \n\t"                         \
  "vse32.v      v27, (a3)               \n\t"                         \
  "addi         a3, a3, 16              \n\t"                         \
  "vse32.v      v28, (a1)               \n\t"                         \
  "addi         a1, a1, 16              \n\t"                         \
  "vse32.v      v29, (a3)               \n\t"                         \
  "addi         a3, a3, 16              \n\t"                         \
  "vse32.v      v30, (a1)               \n\t"                         \
  "vse32.v      v31, (a3)               \n\t"                         \
  "vsetvli      t0, zero, e32, m1       \n\t"                         \
  "vse32.v      v24, (a2), v0.t         \n\t"                         \
  "addi         a2, a2, 16              \n\t"                         \
  "vse32.v      v25, (a4), v0.t         \n\t"                         \
  "addi         a4, a4, 16              \n\t"                         \
  "vse32.v      v26, (a2), v0.t         \n\t"                         \
  "addi         a2, a2, 16              \n\t"                         \
  "vse32.v      v27, (a4), v0.t         \n\t"                         \
  "addi         a4, a4, 16              \n\t"                         \
  "vse32.v      v28, (a2), v0.t         \n\t"                         \
  "addi         a2, a2, 16              \n\t"                         \
  "vse32.v      v29, (a4), v0.t         \n\t"                         \
  "addi         a4, a4, 16              \n\t"                         \
  "vse32.v      v30, (a2), v0.t         \n\t"                         \
  "vse32.v      v31, (a4), v0.t         \n\t"

#define __WEFT_IME1_LOAD_ZP_M4                                         \
  "vsetvli      t0, zero, e8, mf2       \n\t"                         \
  "vle8.v       v11, (s6)               \n\t"                         \
  "vsetvli      t0, zero, e8, m1        \n\t"                         \
  "vrgather.vv  v12, v11, v1            \n\t"                         \
  "vadd.vi      v1, v1, 4               \n\t"                         \
  "vrgather.vv  v13, v11, v1            \n\t"                         \
  "vadd.vi      v1, v1, 4               \n\t"                         \
  "vrgather.vv  v14, v11, v1            \n\t"                         \
  "vadd.vi      v1, v1, 4               \n\t"                         \
  "vrgather.vv  v15, v11, v1            \n\t"                         \
  "vadd.vi      v1, v1, -12             \n\t"

)ime";
  }
  if (usesIME1SymmetricI4I8M4) {
    output << R"ime(
static inline __attribute__((always_inline, unused)) void
__weft_ime1_symmetric_i4_i8_m4_n16_k32(
    const float *activation_scale, const int8_t *activation_code,
    const uint8_t *packed_weight, float *accumulator) {
  const size_t ldc = 16 * sizeof(float);
  float contribution[4 * 16];
  __asm__ volatile(
      "vsetvli      t0, zero, e32, m8       \n\t"
      "vxor.vv      v24, v24, v24           \n\t"
      "addi         s5, %[B], 0             \n\t"
      "addi         s1, %[B], 32            \n\t"
      "addi         s2, s1, 32              \n\t"
      "addi         s3, s1, 64              \n\t"
      "addi         s4, s1, 96              \n\t"
      "addi         a1, %[A], 0             \n\t"
      "flw          f1, 0(%[AS])            \n\t"
      "flw          f2, 4(%[AS])            \n\t"
      "flw          f3, 8(%[AS])            \n\t"
      "flw          f4, 12(%[AS])           \n\t"
      "vsetvli      t0, zero, e32, m8       \n\t"
      "vxor.vv      v16, v16, v16           \n\t"
      "li           t2, 2                   \n\t"
      "LOOP_M4_SYM%=:                       \n\t"
      __WEFT_IME1_LOAD_B_M4
      "vsetvli      t0, zero, e8, m1        \n\t"
      "vle8.v       v10, (a1)               \n\t"
      "addi         a1, a1, 32              \n\t"
      "vle8.v       v11, (a1)               \n\t"
      "addi         a1, a1, 32              \n\t"
      "vadd.vi      v2, v2, -8              \n\t"
      "vadd.vi      v3, v3, -8              \n\t"
      "vadd.vi      v4, v4, -8              \n\t"
      "vadd.vi      v5, v5, -8              \n\t"
      "vadd.vi      v6, v6, -8              \n\t"
      "vadd.vi      v7, v7, -8              \n\t"
      "vadd.vi      v8, v8, -8              \n\t"
      "vadd.vi      v9, v9, -8              \n\t"
      __WEFT_IME1_COMP_M4
      "addi         t2, t2, -1              \n\t"
      "bnez         t2, LOOP_M4_SYM%=       \n\t"
      __WEFT_IME1_LOAD_SCALE_M4
      "vsetvli      t0, zero, e32, m8       \n\t"
      "vfcvt.f.x.v  v16, v16                \n\t"
      "vfmacc.vv    v24, v16, v8            \n\t"
      __WEFT_IME1_SAVE_ACC_M4
      :
      : [AS] "r"(activation_scale), [A] "r"(activation_code),
        [B] "r"(packed_weight), [C] "r"(contribution), [LDC] "r"(ldc)
      : "cc", "memory", "t0", "t1", "t2", "a1", "a2", "a3", "a4",
        "f1", "f2", "f3", "f4", "s1", "s2", "s3", "s4", "s5",
        "s6");
  const size_t vl = __riscv_vsetvl_e32m8(4 * 16);
  const vfloat32m8_t current =
      __riscv_vle32_v_f32m8(accumulator, vl);
  const vfloat32m8_t update =
      __riscv_vle32_v_f32m8(contribution, vl);
  __riscv_vse32_v_f32m8(
      accumulator, __riscv_vfadd_vv_f32m8(current, update, vl), vl);
}

)ime";
  }
  if (usesIME1AffineI4I8M4) {
    output << R"ime(
static inline __attribute__((always_inline, unused)) void
__weft_ime1_affine_i4_i8_m4_n16_k32(
    const float *activation_scale, const int8_t *activation_code,
    const uint8_t *packed_weight, float *accumulator) {
  const size_t ldc = 16 * sizeof(float);
  float contribution[4 * 16];
  __asm__ volatile(
      "vsetvli      t0, zero, e32, m8       \n\t"
      "vxor.vv      v24, v24, v24           \n\t"
      "vsetvli      t0, zero, e8, m1        \n\t"
      "li           s1, 24                  \n\t"
      "vmv.v.i      v1, 3                   \n\t"
      "vsetvli      t0, s1, e8, m1          \n\t"
      "vmv.v.i      v1, 2                   \n\t"
      "vsetvli      t0, zero, e8, mf2       \n\t"
      "vmv.v.i      v1, 1                   \n\t"
      "vsetvli      t0, zero, e8, mf4       \n\t"
      "vmv.v.i      v1, 0                   \n\t"
      "addi         s5, %[B], 0             \n\t"
      "addi         s6, %[B], 32            \n\t"
      "addi         s1, %[B], 48            \n\t"
      "addi         s2, s1, 32              \n\t"
      "addi         s3, s1, 64              \n\t"
      "addi         s4, s1, 96              \n\t"
      "addi         a1, %[A], 0             \n\t"
      "flw          f1, 0(%[AS])            \n\t"
      "flw          f2, 4(%[AS])            \n\t"
      "flw          f3, 8(%[AS])            \n\t"
      "flw          f4, 12(%[AS])           \n\t"
      "vsetvli      t0, zero, e32, m8       \n\t"
      "vxor.vv      v16, v16, v16           \n\t"
      __WEFT_IME1_LOAD_ZP_M4
      "li           t2, 2                   \n\t"
      "LOOP_M4_AFFINE%=:                    \n\t"
      __WEFT_IME1_LOAD_B_M4
      "vsetvli      t0, zero, e8, m1        \n\t"
      "vle8.v       v10, (a1)               \n\t"
      "addi         a1, a1, 32              \n\t"
      "vle8.v       v11, (a1)               \n\t"
      "addi         a1, a1, 32              \n\t"
      "vsub.vv      v2, v2, v12             \n\t"
      "vsub.vv      v6, v6, v12             \n\t"
      "vsub.vv      v3, v3, v13             \n\t"
      "vsub.vv      v7, v7, v13             \n\t"
      "vsub.vv      v4, v4, v14             \n\t"
      "vsub.vv      v8, v8, v14             \n\t"
      "vsub.vv      v5, v5, v15             \n\t"
      "vsub.vv      v9, v9, v15             \n\t"
      __WEFT_IME1_COMP_M4
      "addi         t2, t2, -1              \n\t"
      "bnez         t2, LOOP_M4_AFFINE%=    \n\t"
      __WEFT_IME1_LOAD_SCALE_M4
      "vsetvli      t0, zero, e32, m8       \n\t"
      "vfcvt.f.x.v  v16, v16                \n\t"
      "vfmacc.vv    v24, v16, v8            \n\t"
      __WEFT_IME1_SAVE_ACC_M4
      :
      : [AS] "r"(activation_scale), [A] "r"(activation_code),
        [B] "r"(packed_weight), [C] "r"(contribution), [LDC] "r"(ldc)
      : "cc", "memory", "t0", "t1", "t2", "a1", "a2", "a3", "a4",
        "f1", "f2", "f3", "f4", "s1", "s2", "s3", "s4", "s5",
        "s6");
  const size_t vl = __riscv_vsetvl_e32m8(4 * 16);
  const vfloat32m8_t current =
      __riscv_vle32_v_f32m8(accumulator, vl);
  const vfloat32m8_t update =
      __riscv_vle32_v_f32m8(contribution, vl);
  __riscv_vse32_v_f32m8(
      accumulator, __riscv_vfadd_vv_f32m8(current, update, vl), vl);
}

)ime";
  }
  if (usesIME1SymmetricI4I8 || usesIME1AffineI4I8) {
    output << R"ime(
#undef __WEFT_IME1_COMP_I4_I8_M1
#undef __WEFT_IME1_ACC_I4_I8_M1
#undef __WEFT_IME1_LOAD_I4_I8_M1
#undef __WEFT_IME1_LOAD_ZP_I4_I8_M1

)ime";
  }
  if (usesIME1SymmetricI4I8M4 || usesIME1AffineI4I8M4) {
    output << R"ime(
#undef __WEFT_IME1_LOAD_B_M4
#undef __WEFT_IME1_COMP_M4
#undef __WEFT_IME1_LOAD_SCALE_M4
#undef __WEFT_IME1_SAVE_ACC_M4
#undef __WEFT_IME1_LOAD_ZP_M4

)ime";
  }
}

} // namespace weft::riscv_internal
