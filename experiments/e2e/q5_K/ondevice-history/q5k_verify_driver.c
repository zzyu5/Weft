// q5k_verify_driver.c -- on-device (ssh rvv / ssh k1) numerical + perf harness
// for the q5_K x q8_K super-block block-dot kernel.
//
// It links against a kernel that exports the shared symbol
//   tcrv_emitc_ggml_vec_dot_q5_K_q8_K_kernel_rvv_q5_K_q8_K_block_dot
//     (size_t n, float *s, const uint8_t *vx, const uint8_t *vy)
// which is EITHER our emitted kernel (kernel_ours.cpp, from
// tcrv-opt --...front-door --...lower-to-emitc | mlir-translate --mlir-to-cpp)
// OR the faithful ggml factory body (kernel_factory.c). The SAME driver drives both.
//
// CORRECTNESS GATE (ours): bit-exact vs an INDEPENDENT ggml scalar q5_K reference
//   -- the portable ggml_vec_dot_q5_K_q8_K_generic fold, transcribed verbatim
//   (quants.c:720): integer aux32[8] lane-accumulation (EXACT), then per-super-block
//     sums[l] += d*aux32[l]   (8 float lanes, SEPARATE mul+add, no FMA)
//     sumf   -= dmin*sumi     (mins term)
//   final  *s = sumf + sum_l sums[l]   (left-assoc).
//   Our emitted kernel folds in EXACTLY this order (sums8 vector lanes += d*aux32[l]
//   via explicit vfmul+vfadd; sumf -= dmin*sumi; then the 8-lane horizontal add),
//   so it must be BIT-EXACT to this oracle. REQUIRES -ffp-contract=off so the
//   oracle's `sums[l] += d*aux32[l]` stays a separate mul+add (matching the kernel's
//   explicit non-fused vfmul/vfadd) rather than contracting to fmaf.
//
// FACTORY DIAGNOSTIC: the factory riscv body uses a DIFFERENT (coarser) float fold
//   -- it collapses the whole super-block to one int32 aux32 (2 vredsum/sub-block),
//   then `sums += aux32*d` as ONE scalar accumulation. We compute that fold as
//   `oracle_factory` so the factory-linked driver also shows a clean VERDICT and we
//   can report the factory<->generic ULP gap (they are NOT bit-identical folds).
//
// The fp16 super-block scales d/dmin are read back from the SAME stored block bytes
// both the kernel and the reference dereference, and fp16->fp32 is EXACT, so the
// comparison is genuinely bit-for-bit.
//
// Board identity (VLEN, vlenb) is printed at runtime; march/compiler provenance is
// passed in by board_ab.sh and echoed. KERNEL microbench -- NOT e2e decode.

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__riscv_vector)
#include <riscv_vector.h>
#define HAVE_RVV_INTRINSICS 1
#else
#define HAVE_RVV_INTRINSICS 0
#endif

// ---- q5_K / q8_K formats ---------------------------------------------------
#define QK_K 256
#define K_SCALE_SIZE 12
#define NSUB (QK_K / 32) // 8 sub-blocks of 32

typedef struct {
  uint16_t d;                   // fp16 super-block scale for scales
  uint16_t dmin;                // fp16 super-block scale for mins
  uint8_t scales[K_SCALE_SIZE]; // 8 x (6-bit scale, 6-bit min) packed in 12 bytes
  uint8_t qh[QK_K / 8];         // 32 bytes: 5th (high) bit plane
  uint8_t qs[QK_K / 2];         // 128 bytes: low-4-bit quants
} block_q5_K;

typedef struct {
  float d;                  // q8_K delta (float, NOT fp16)
  int8_t qs[QK_K];          // 256 int8 quants
  int16_t bsums[QK_K / 16]; // 16 group-sums of 16 quants
} block_q8_K;

#define Q5K_BYTES 176
#define Q8K_BYTES 292

// ---- fp16 <-> fp32 (pure-integer IEEE, exact) ------------------------------
static float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t man = h & 0x3ffu;
  uint32_t out;
  if (exp == 0) {
    if (man == 0) {
      out = sign;
    } else {
      int e = 127 - 15 + 1;
      while (!(man & 0x400u)) { man <<= 1; e--; }
      man &= 0x3ffu;
      out = sign | ((uint32_t)e << 23) | (man << 13);
    }
  } else if (exp == 0x1f) {
    out = sign | 0x7f800000u | (man << 13);
  } else {
    out = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f;
  memcpy(&f, &out, sizeof f);
  return f;
}
static uint16_t fp32_to_fp16(float f) {
  uint32_t x;
  memcpy(&x, &f, sizeof x);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t e = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
  uint32_t man = x & 0x7fffffu;
  if (((x >> 23) & 0xffu) == 0xff)
    return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u));
  if (e >= 0x1f)
    return (uint16_t)(sign | 0x7c00u);
  if (e <= 0) {
    if (e < -10)
      return (uint16_t)sign;
    man |= 0x800000u;
    int sh = 14 - e;
    uint32_t h = man >> sh;
    uint32_t rem = man & ((1u << sh) - 1u);
    uint32_t half = 1u << (sh - 1);
    if (rem > half || (rem == half && (h & 1u)))
      h++;
    return (uint16_t)(sign | h);
  }
  uint16_t h = (uint16_t)(sign | ((uint32_t)e << 10) | (man >> 13));
  uint32_t rem = man & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1u)))
    h++;
  return h;
}

// ---- ggml nearest_int (bit-trick, matches ggml-quants.c) -------------------
static inline int nearest_int(float fval) {
  float val = fval + 12582912.f;
  int i;
  memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

// ---- get_scale_min_k4 (verbatim from ggml-quants.c): recover 6-bit sc/min --
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// ---- build a byte-valid q8_K block (with bsums) directly from int8 quants ---
static void build_q8k(block_q8_K *b, const int8_t qs[QK_K], float d) {
  b->d = d;
  for (int j = 0; j < QK_K; j++)
    b->qs[j] = qs[j];
  for (int j = 0; j < QK_K / 16; j++) {
    int sum = 0;
    for (int ii = 0; ii < 16; ii++)
      sum += b->qs[j * 16 + ii];
    b->bsums[j] = (int16_t)sum;
  }
}

// ---- build a byte-valid q5_K block directly from 5-bit quants L[256],
//      per-sub-block 6-bit ls/lm, and fp16 d/dmin. The scale/min packing is
//      the EXACT inverse of get_scale_min_k4 (verbatim from quantize_row_q5_K_ref),
//      the qh/qs packing is verbatim from quantize_row_q5_K_ref. So both the
//      kernel's utmp bit-dance AND the generic reference recover ls/lm/L exactly.
static void build_q5k(block_q5_K *y, const uint8_t ls[NSUB], const uint8_t lm[NSUB],
                      const uint8_t L[QK_K], uint16_t d_bits, uint16_t dmin_bits) {
  y->d = d_bits;
  y->dmin = dmin_bits;
  memset(y->scales, 0, K_SCALE_SIZE);
  for (int j = 0; j < NSUB; ++j) {
    uint8_t s = ls[j] & 63, mn = lm[j] & 63; // 6-bit
    if (j < 4) {
      y->scales[j] = s;
      y->scales[j + 4] = mn;
    } else {
      y->scales[j + 4] = (s & 0xF) | ((mn & 0xF) << 4);
      y->scales[j - 4] |= ((s >> 4) << 6);
      y->scales[j - 0] |= ((mn >> 4) << 6);
    }
  }
  uint8_t *qh = y->qh;
  uint8_t *ql = y->qs;
  memset(qh, 0, QK_K / 8);
  uint8_t m1 = 1, m2 = 2;
  for (int n = 0; n < QK_K; n += 64) {
    for (int j = 0; j < 32; ++j) {
      int l1 = L[n + j];
      if (l1 > 15) { l1 -= 16; qh[j] |= m1; }
      int l2 = L[n + j + 32];
      if (l2 > 15) { l2 -= 16; qh[j] |= m2; }
      ql[j] = (uint8_t)(l1 | (l2 << 4));
    }
    m1 <<= 2;
    m2 <<= 2;
    ql += 32;
  }
}

// ---- unpack scales/mins into 8+8 bytes via ggml's utmp bit-dance ------------
static void unpack_scales_mins(const block_q5_K *x, uint8_t scales[8], uint8_t mins[8]) {
  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;
  uint32_t utmp[4];
  memcpy(utmp, x->scales, 12);
  utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
  const uint32_t uaux = utmp[1] & kmask1;
  utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
  utmp[2] = uaux;
  utmp[0] &= kmask1;
  memcpy(scales, (const uint8_t *)&utmp[0], 8);
  memcpy(mins, (const uint8_t *)&utmp[2], 8);
}

// ---- decode q5_K super-block into aux8[256] (5-bit values, exact) ----------
static void decode_q5k(const block_q5_K *x, int8_t aux8[QK_K]) {
  const uint8_t *q4 = x->qs;
  const uint8_t *hm = x->qh;
  int8_t *a = aux8;
  uint8_t m = 1;
  for (int j = 0; j < QK_K / 64; ++j) {
    for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
    for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
    a += 32; m <<= 1;
    for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
    for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
    a += 32; m <<= 1;
    q4 += 32;
  }
}

// ===========================================================================
// ORACLE 1 (THE GATE): ggml generic scalar q5_K fold, transcribed verbatim.
// Per-lane aux32[8] integer accumulation (exact), then per-super-block
// sums[l] += d*aux32[l] and sumf -= dmin*sumi, final sumf += sum_l sums[l].
// ===========================================================================
__attribute__((noinline)) static float
ref_generic(size_t n, const block_q5_K *x, const block_q8_K *y) {
  const int nb = (int)(n / QK_K);
  int8_t aux8[QK_K];
  int16_t aux16[8];
  float sums[8];
  int32_t aux32[8];
  memset(sums, 0, 8 * sizeof(float));
  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    decode_q5k(&x[i], aux8);
    uint8_t scales[8], mins[8];
    unpack_scales_mins(&x[i], scales, mins);
    memset(aux32, 0, 8 * sizeof(int32_t));
    int sumi = 0;
    for (int j = 0; j < QK_K / 16; ++j)
      sumi += y[i].bsums[j] * mins[j / 2];
    const int8_t *a = aux8;
    const int8_t *q8 = y[i].qs;
    int is = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
      int32_t scale = scales[is++];
      for (int g = 0; g < 4; ++g) {
        for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
        for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
        q8 += 8; a += 8;
      }
    }
    const float d = fp16_to_fp32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    const float dmin = fp16_to_fp32(x[i].dmin) * y[i].d;
    sumf -= dmin * sumi;
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  return sumf;
}

// ===========================================================================
// ORACLE 2 (DIAGNOSTIC): factory riscv fold -- collapses the super-block to one
// int32 aux32, then `sums += aux32*d` as a single scalar accumulation. Matches
// kernel_factory.c bit-for-bit under -ffp-contract=off.
// ===========================================================================
__attribute__((noinline)) static float
ref_factory(size_t n, const block_q5_K *x, const block_q8_K *y) {
  const int nb = (int)(n / QK_K);
  int8_t aux8[QK_K];
  float sumf = 0, sums = 0.0f;
  for (int i = 0; i < nb; ++i) {
    decode_q5k(&x[i], aux8);
    uint8_t scales[8], mins[8];
    unpack_scales_mins(&x[i], scales, mins);
    int sumi = 0;
    for (int j = 0; j < QK_K / 16; ++j)
      sumi += y[i].bsums[j] * mins[j / 2];
    const float d = fp16_to_fp32(x[i].d) * y[i].d;
    const float dmin = fp16_to_fp32(x[i].dmin) * y[i].d;
    sumf -= dmin * sumi;
    int32_t aux32 = 0;
    const int8_t *a = aux8;
    const int8_t *q8 = y[i].qs;
    int is = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
      int32_t scale = scales[is++];
      for (int l = 0; l < 32; ++l)
        aux32 += scale * (q8[l] * a[l]);
      q8 += 32; a += 32;
    }
    sums += aux32 * d;
  }
  return sumf + sums;
}

// ---- MUTATION oracle: generic fold but FUSING sums[l] += d*aux32[l] into fmaf.
// Diverges from ref_generic on >=1 input (single-round vs double-round), proving
// the bit-exact gate has discriminating power.
static float mutated(size_t n, const block_q5_K *x, const block_q8_K *y) {
  const int nb = (int)(n / QK_K);
  int8_t aux8[QK_K];
  float sums[8];
  int32_t aux32[8];
  memset(sums, 0, 8 * sizeof(float));
  float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    decode_q5k(&x[i], aux8);
    uint8_t scales[8], mins[8];
    unpack_scales_mins(&x[i], scales, mins);
    memset(aux32, 0, 8 * sizeof(int32_t));
    int sumi = 0;
    for (int j = 0; j < QK_K / 16; ++j)
      sumi += y[i].bsums[j] * mins[j / 2];
    const int8_t *a = aux8;
    const int8_t *q8 = y[i].qs;
    int is = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
      int32_t scale = scales[is++];
      for (int g = 0; g < 4; ++g) {
        for (int l = 0; l < 8; ++l) aux32[l] += scale * (q8[l] * a[l]);
        q8 += 8; a += 8;
      }
    }
    const float d = fp16_to_fp32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] = fmaf(d, (float)aux32[l], sums[l]); // MUTATION
    const float dmin = fp16_to_fp32(x[i].dmin) * y[i].d;
    sumf -= dmin * sumi;
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  return sumf;
}

// ---- kernel under test (real exported / factory .o) ------------------------
extern void tcrv_emitc_ggml_vec_dot_q5_K_q8_K_kernel_rvv_q5_K_q8_K_block_dot(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

static void call_kernel(size_t n, float *s, const block_q5_K *x, const block_q8_K *y) {
  tcrv_emitc_ggml_vec_dot_q5_K_q8_K_kernel_rvv_q5_K_q8_K_block_dot(
      n, s, (const uint8_t *)x, (const uint8_t *)y);
}

#define CLOBBER() __asm__ __volatile__("" ::: "memory")

// ---- helpers ---------------------------------------------------------------
static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static int64_t f2key(float f) {
  uint32_t u = f2u(f);
  return (u & 0x80000000u) ? -(int64_t)(u & 0x7fffffffu) : (int64_t)u;
}
static uint64_t ulp_dist(float a, float b) {
  int64_t d = f2key(a) - f2key(b);
  return d < 0 ? (uint64_t)(-d) : (uint64_t)d;
}
static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static int rr(unsigned *seed, int lo, int hi) {
  return lo + (int)(rand_r(seed) % (unsigned)(hi - lo + 1));
}

// Fill one random super-block pair with realistic, non-degenerate content that
// exercises: ~50% qh (5th-bit) set, varied 6-bit scales AND mins, signed q8.
static void fill_random_pair(block_q5_K *bx, block_q8_K *by, unsigned *seed) {
  uint8_t L[QK_K], ls[NSUB], lm[NSUB];
  int8_t q8[QK_K];
  for (int j = 0; j < NSUB; j++) {
    ls[j] = (uint8_t)rr(seed, 1, 63);
    lm[j] = (uint8_t)rr(seed, 0, 63);
  }
  for (int j = 0; j < QK_K; j++) {
    L[j] = (uint8_t)rr(seed, 0, 31);   // full 5-bit range -> ~half set qh
    q8[j] = (int8_t)rr(seed, -127, 127);
  }
  uint16_t d_bits = fp32_to_fp16((float)rr(seed, 1, 400) / 20000.0f);   // ~[5e-5,0.02]
  uint16_t dmin_bits = fp32_to_fp16((float)rr(seed, 0, 200) / 20000.0f);
  build_q5k(bx, ls, lm, L, d_bits, dmin_bits);
  build_q8k(by, q8, (float)rr(seed, 1, 400) / 2000.0f); // q8_K d in ~[5e-4,0.2]
}

static int run_property_case(const char *name, size_t n, const block_q5_K *x,
                             const block_q8_K *y, int expect_nonzero) {
  float s_kernel = 12345.0f;
  call_kernel(n, &s_kernel, x, y);
  float r = ref_generic(n, x, y);
  int match = (f2u(s_kernel) == f2u(r));
  int nz_ok = !expect_nonzero || (s_kernel != 0.0f);
  printf("  [%-16s] kernel=%.9g (0x%08x)  oracle=%.9g (0x%08x)  ulp=%llu  %s%s\n",
         name, s_kernel, f2u(s_kernel), r, f2u(r),
         (unsigned long long)ulp_dist(s_kernel, r),
         match ? "MATCH" : "MISMATCH", nz_ok ? "" : " (VACUOUS-ZERO!)");
  return match && nz_ok;
}

int main(int argc, char **argv) {
  size_t n = 4096;
  int trials = 256;
  long perf_iters = 60000;
  const char *march = (argc > 1) ? argv[1] : "unknown";
  if (argc > 2) n = (size_t)strtoul(argv[2], NULL, 10);
  if (argc > 3) trials = atoi(argv[3]);
  if (argc > 4) perf_iters = atol(argv[4]);
  if (n % QK_K) n -= (n % QK_K);
  size_t nb = n / QK_K;

  // ---- board identity (VLEN printed FIRST; board_ab gate4 greps it) ----
#if HAVE_RVV_INTRINSICS
  size_t vlenb = __riscv_vlenb();
  size_t vlmax_e8m2 = __riscv_vsetvlmax_e8m2();
#else
  size_t vlenb = 0, vlmax_e8m2 = 0;
#endif
  printf("== board identity ==\n");
  printf("VLEN(bits)=%zu  vlenb=%zu  VLMAX_e8m2=%zu  march=%s  driver_has_rvv_intrinsics=%d\n",
         vlenb * 8, vlenb, vlmax_e8m2, march, HAVE_RVV_INTRINSICS);
  printf("n=%zu super-blocks/dot=%zu QK_K=%d sizeof(q5_K)=%zu sizeof(q8_K)=%zu\n",
         n, nb, QK_K, sizeof(block_q5_K), sizeof(block_q8_K));
  if (sizeof(block_q5_K) != Q5K_BYTES || sizeof(block_q8_K) != Q8K_BYTES) {
    printf("FATAL: block struct size wrong (q5_K=%zu want %d, q8_K=%zu want %d)\n",
           sizeof(block_q5_K), Q5K_BYTES, sizeof(block_q8_K), Q8K_BYTES);
    return 2;
  }
  if (nb == 0) {
    // VLEN-probe invocation (board_ab preflight gate 4 passes n=64 -> nb=0).
    printf("nb==0 (probe invocation); board identity printed, exiting clean.\n");
    return 0;
  }

  block_q5_K *bx = malloc(nb * sizeof(block_q5_K));
  block_q8_K *by = malloc(nb * sizeof(block_q8_K));
  if (!bx || !by) { printf("FATAL: alloc\n"); return 2; }

  // ---- VERIFY (gate = generic scalar oracle) ----
  printf("\n== numerical verify (%d trials, n=%zu) ==\n", trials, n);
  unsigned seed = 0x5c0ffeeu;
  int match_gen = 0, match_fac = 0, mut_diff = 0;
  uint64_t worst_ulp_gen = 0, worst_ulp_fac = 0;
  float sample_k = 0, sample_g = 0;
  for (int t = 0; t < trials; t++) {
    for (size_t i = 0; i < nb; i++)
      fill_random_pair(&bx[i], &by[i], &seed);
    float s_kernel = 12345.0f;
    call_kernel(n, &s_kernel, bx, by);
    float r_gen = ref_generic(n, bx, by);
    float r_fac = ref_factory(n, bx, by);
    float r_mut = mutated(n, bx, by);
    if (f2u(s_kernel) == f2u(r_gen)) match_gen++;
    if (f2u(s_kernel) == f2u(r_fac)) match_fac++;
    if (f2u(r_gen) != f2u(r_mut)) mut_diff++;
    uint64_t ug = ulp_dist(s_kernel, r_gen);
    if (ug > worst_ulp_gen) worst_ulp_gen = ug;
    uint64_t uf = ulp_dist(r_gen, r_fac);
    if (uf > worst_ulp_fac) worst_ulp_fac = uf;
    if (t == 0) { sample_k = s_kernel; sample_g = r_gen; }
  }
  printf("sample[0]: kernel=%.9g (0x%08x)  ref_generic=%.9g (0x%08x)\n",
         sample_k, f2u(sample_k), sample_g, f2u(sample_g));
  printf("[GATE] exact-match vs GENERIC q5_K oracle : %d/%d   worst ULP=%llu\n",
         match_gen, trials, (unsigned long long)worst_ulp_gen);
  printf("[diag] exact-match vs FACTORY-fold oracle : %d/%d\n", match_fac, trials);
  printf("[diag] generic<->factory fold worst ULP   : %llu (folds differ by design)\n",
         (unsigned long long)worst_ulp_fac);
  int bitexact = (match_gen == trials);
  printf("VERDICT (generic q5_K gate): %s\n", bitexact ? "BIT-EXACT" : "NOT bit-exact");

  // ---- property tests (hand-built regimes) ----
  printf("\n== property tests (kernel vs generic oracle) ==\n");
  int prop_pass = 0, prop_total = 0;
  uint8_t L[QK_K], ls[NSUB], lm[NSUB];
  int8_t q8[QK_K];

  // (a) all-zero 5-bit quants (aux32==0 => main term 0; mins term also 0 here).
  for (int j = 0; j < NSUB; j++) { ls[j] = 20; lm[j] = 0; }
  for (int j = 0; j < QK_K; j++) { L[j] = 0; q8[j] = (int8_t)((j & 1) ? -30 : 30); }
  for (size_t i = 0; i < nb; i++) {
    build_q5k(&bx[i], ls, lm, L, fp32_to_fp16(0.01f), fp32_to_fp16(0.0f));
    build_q8k(&by[i], q8, 0.05f);
  }
  prop_total++; prop_pass += run_property_case("all-zero-L", n, bx, by, 0);

  // (b) max 5-bit quants (31) everywhere => ALL qh bits set; VARIED scales and a
  // non-symmetric q8 so the per-lane terms do NOT cancel to a vacuous zero.
  for (int j = 0; j < NSUB; j++) { ls[j] = (uint8_t)(40 + 3 * j); lm[j] = 20; }
  for (int j = 0; j < QK_K; j++) { L[j] = 31; q8[j] = (int8_t)(60 + (j % 41)); }
  for (size_t i = 0; i < nb; i++) {
    build_q5k(&bx[i], ls, lm, L, fp32_to_fp16(0.02f), fp32_to_fp16(0.01f));
    build_q8k(&by[i], q8, 0.1f);
  }
  prop_total++; prop_pass += run_property_case("max-L-all-qh", n, bx, by, 1);

  // (c) mins-dominant: L=0 (no main term) but large mins + nonzero bsums.
  for (int j = 0; j < NSUB; j++) { ls[j] = 10; lm[j] = 63; }
  for (int j = 0; j < QK_K; j++) { L[j] = 0; q8[j] = (int8_t)100; }
  for (size_t i = 0; i < nb; i++) {
    build_q5k(&bx[i], ls, lm, L, fp32_to_fp16(0.01f), fp32_to_fp16(0.02f));
    build_q8k(&by[i], q8, 0.08f);
  }
  prop_total++; prop_pass += run_property_case("mins-dominant", n, bx, by, 1);

  // (d) extreme fp16 scales: max-normal d, min-normal dmin, full quants.
  for (int j = 0; j < NSUB; j++) { ls[j] = 63; lm[j] = 63; }
  for (int j = 0; j < QK_K; j++) { L[j] = (uint8_t)(j & 31); q8[j] = (int8_t)((j & 1) ? -127 : 127); }
  for (size_t i = 0; i < nb; i++) {
    build_q5k(&bx[i], ls, lm, L, 0x7bff /*65504*/, 0x0400 /*min normal*/);
    build_q8k(&by[i], q8, 1.0f);
  }
  prop_total++; prop_pass += run_property_case("extreme-fp16", n, bx, by, 1);
  printf("property: %d/%d passed\n", prop_pass, prop_total);

  // ---- mutation test ----
  printf("\n== mutation test (fmaf-fused fold vs separate mul+add oracle) ==\n");
  printf("random-trial divergences: %d/%d\n", mut_diff, trials);
  printf("mutation caught (oracle discriminates): %s\n",
         (mut_diff > 0) ? "YES" : "NO -- ORACLE HAS NO DISCRIMINATING POWER");

  // ---- PERF ----
  for (size_t i = 0; i < nb; i++)
    fill_random_pair(&bx[i], &by[i], &seed);
  volatile float sink = 0.0f;
  float s = 0;
  const int REPEATS = 7;
  for (int i = 0; i < 2000; i++) {
    call_kernel(n, &s, bx, by);
    sink += s;
    sink += ref_generic(n, bx, by);
  }
  double best_kernel = 1e30, best_ref = 1e30;
  for (int r = 0; r < REPEATS; r++) {
    double t0 = now_s();
    for (long i = 0; i < perf_iters; i++) { CLOBBER(); call_kernel(n, &s, bx, by); sink += s; }
    double tk = now_s() - t0;
    if (tk < best_kernel) best_kernel = tk;
    double t1 = now_s();
    for (long i = 0; i < perf_iters; i++) { CLOBBER(); sink += ref_generic(n, bx, by); }
    double tr = now_s() - t1;
    if (tr < best_ref) best_ref = tr;
  }
  double ns_kernel = best_kernel * 1e9 / (double)perf_iters;
  double ns_ref = best_ref * 1e9 / (double)perf_iters;
  double bytes = (double)nb * (double)(Q5K_BYTES + Q8K_BYTES);
  printf("\n== perf (%ld iters, n=%zu, march=%s) ==\n", perf_iters, n, march);
  printf("kernel   : %.1f ns/call   %.2f GB/s\n", ns_kernel, bytes / ns_kernel);
  printf("reference: %.1f ns/call   %.2f GB/s   (scalar generic baseline)\n",
         ns_ref, bytes / ns_ref);
  printf("speedup kernel vs scalar-generic reference: %.2fx\n", ns_ref / ns_kernel);
  printf("(sink=%.6g)\n", (double)sink);
  printf("NOTE: KERNEL microbench (q5_K x q8_K vec_dot), NOT e2e decode.\n");

  free(bx); free(by);
  return (bitexact && prop_pass == prop_total && mut_diff > 0) ? 0 : 1;
}
