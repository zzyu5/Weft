// q5k_byteexact_mirror.c -- LAPTOP byte-exact proof for the register-resident q5_K
// prototype, WITHOUT hardware/qemu (no qemu-riscv64 available; "别上硬件").
//
// Strategy: the register-resident kernels (kernel_regresident_A/B.cpp) change ONLY
// WHERE the q5 decode happens (fused inline into the MAC instead of a pre-pass into
// aux8[256]); the integer products, the 8-lane aux32 accumulation, and the float fold
// are unchanged. This program:
//   (1) DECODE-EQUIVALENCE: reconstructs each decoded value a[pos] via the exact
//       inline formula the register-resident kernel uses and asserts it is
//       BYTE-IDENTICAL to the ggml `_generic` decode_q5k(aux8) for random blocks.
//   (2) FOLD BIT-EXACT: a scalar mirror of the register-resident dataflow
//       (mirror_regres: inline decode -> 8-lane aux32 -> same fold) is compared
//       BIT-FOR-BIT (f2u equality) to the ggml generic oracle (ref_generic, verbatim
//       from q5k_verify_driver.c) over 256 random trials + 4 hand-built regimes.
// Together these prove the reordering is value-preserving. (On-device RVV bit-exact
// vs the same oracle remains the next step, per the P3 plan.)
//
// Compiled NATIVELY (x86) with -ffp-contract=off so `sums[l] += d*aux32[l]` stays a
// separate mul+add, matching the kernel's explicit vfmul/vfadd.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QK_K 256
#define K_SCALE_SIZE 12
#define NSUB (QK_K / 32)

typedef struct {
  uint16_t d, dmin;
  uint8_t scales[K_SCALE_SIZE];
  uint8_t qh[QK_K / 8];
  uint8_t qs[QK_K / 2];
} block_q5_K;
typedef struct {
  float d;
  int8_t qs[QK_K];
  int16_t bsums[QK_K / 16];
} block_q8_K;

// ---- fp16<->fp32 (exact), nearest_int, packers: verbatim from q5k_verify_driver.c ----
static float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t man = h & 0x3ffu;
  uint32_t out;
  if (exp == 0) {
    if (man == 0) out = sign;
    else { int e = 127 - 15 + 1; while (!(man & 0x400u)) { man <<= 1; e--; } man &= 0x3ffu; out = sign | ((uint32_t)e << 23) | (man << 13); }
  } else if (exp == 0x1f) out = sign | 0x7f800000u | (man << 13);
  else out = sign | ((exp - 15 + 127) << 23) | (man << 13);
  float f; memcpy(&f, &out, sizeof f); return f;
}
static uint16_t fp32_to_fp16(float f) {
  uint32_t x; memcpy(&x, &f, sizeof x);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t e = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
  uint32_t man = x & 0x7fffffu;
  if (((x >> 23) & 0xffu) == 0xff) return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u));
  if (e >= 0x1f) return (uint16_t)(sign | 0x7c00u);
  if (e <= 0) { if (e < -10) return (uint16_t)sign; man |= 0x800000u; int sh = 14 - e; uint32_t h = man >> sh; uint32_t rem = man & ((1u << sh) - 1u); uint32_t half = 1u << (sh - 1); if (rem > half || (rem == half && (h & 1u))) h++; return (uint16_t)(sign | h); }
  uint16_t h = (uint16_t)(sign | ((uint32_t)e << 10) | (man >> 13));
  uint32_t rem = man & 0x1fffu; if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++; return h;
}
static void build_q8k(block_q8_K *b, const int8_t qs[QK_K], float d) {
  b->d = d;
  for (int j = 0; j < QK_K; j++) b->qs[j] = qs[j];
  for (int j = 0; j < QK_K / 16; j++) { int sum = 0; for (int ii = 0; ii < 16; ii++) sum += b->qs[j * 16 + ii]; b->bsums[j] = (int16_t)sum; }
}
static void build_q5k(block_q5_K *y, const uint8_t ls[NSUB], const uint8_t lm[NSUB], const uint8_t L[QK_K], uint16_t d_bits, uint16_t dmin_bits) {
  y->d = d_bits; y->dmin = dmin_bits; memset(y->scales, 0, K_SCALE_SIZE);
  for (int j = 0; j < NSUB; ++j) {
    uint8_t s = ls[j] & 63, mn = lm[j] & 63;
    if (j < 4) { y->scales[j] = s; y->scales[j + 4] = mn; }
    else { y->scales[j + 4] = (s & 0xF) | ((mn & 0xF) << 4); y->scales[j - 4] |= ((s >> 4) << 6); y->scales[j - 0] |= ((mn >> 4) << 6); }
  }
  uint8_t *qh = y->qh, *ql = y->qs; memset(qh, 0, QK_K / 8);
  uint8_t m1 = 1, m2 = 2;
  for (int n = 0; n < QK_K; n += 64) {
    for (int j = 0; j < 32; ++j) { int l1 = L[n + j]; if (l1 > 15) { l1 -= 16; qh[j] |= m1; } int l2 = L[n + j + 32]; if (l2 > 15) { l2 -= 16; qh[j] |= m2; } ql[j] = (uint8_t)(l1 | (l2 << 4)); }
    m1 <<= 2; m2 <<= 2; ql += 32;
  }
}
static void unpack_scales_mins(const block_q5_K *x, uint8_t scales[8], uint8_t mins[8]) {
  static const uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
  uint32_t utmp[4]; memcpy(utmp, x->scales, 12);
  utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
  const uint32_t uaux = utmp[1] & kmask1;
  utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
  utmp[2] = uaux; utmp[0] &= kmask1;
  memcpy(scales, (const uint8_t *)&utmp[0], 8); memcpy(mins, (const uint8_t *)&utmp[2], 8);
}
static void decode_q5k(const block_q5_K *x, int8_t aux8[QK_K]) {
  const uint8_t *q4 = x->qs, *hm = x->qh; int8_t *a = aux8; uint8_t m = 1;
  for (int j = 0; j < QK_K / 64; ++j) {
    for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
    for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
    a += 32; m <<= 1;
    for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] >> 4);
    for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
    a += 32; m <<= 1; q4 += 32;
  }
}

// ---- ORACLE: ggml generic scalar q5_K fold, verbatim from q5k_verify_driver.c ----
static float ref_generic(size_t n, const block_q5_K *x, const block_q8_K *y) {
  const int nb = (int)(n / QK_K);
  int8_t aux8[QK_K]; int16_t aux16[8]; float sums[8]; int32_t aux32[8];
  memset(sums, 0, 8 * sizeof(float)); float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    decode_q5k(&x[i], aux8);
    uint8_t scales[8], mins[8]; unpack_scales_mins(&x[i], scales, mins);
    memset(aux32, 0, 8 * sizeof(int32_t));
    int sumi = 0;
    for (int j = 0; j < QK_K / 16; ++j) sumi += y[i].bsums[j] * mins[j / 2];
    const int8_t *a = aux8; const int8_t *q8 = y[i].qs; int is = 0;
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

// ---- MIRROR: scalar transcription of the REGISTER-RESIDENT dataflow.
// Decode is fused INLINE per (j,q,l) exactly as kernel_regresident_A/B.cpp; NO aux8.
// 8-lane aux32 accumulation + fold identical to the RVV kernel.
static float mirror_regres(size_t n, const block_q5_K *x, const block_q8_K *y) {
  const int nb = (int)(n / QK_K);
  float sums[8]; int32_t aux32[8];
  memset(sums, 0, 8 * sizeof(float)); float sumf = 0;
  for (int i = 0; i < nb; ++i) {
    uint8_t scales[8], mins[8]; unpack_scales_mins(&x[i], scales, mins);
    const uint8_t *qs_base = x[i].qs;   // == (xb+48)
    const uint8_t *qh_base = x[i].qh;   // == (xb+16)
    const int8_t  *q8 = y[i].qs;        // == (yb+4)
    memset(aux32, 0, 8 * sizeof(int32_t));
    for (int j = 0; j < 8; ++j) {
      int scale = (int)scales[j];
      int p = j >> 1;
      const uint8_t *qs_grp = qs_base + p * 32;
      for (int q = 0; q < 4; ++q) {
        for (int l = 0; l < 8; ++l) {
          uint8_t qs = qs_grp[q * 8 + l];
          uint8_t qh = qh_base[q * 8 + l];
          uint8_t nib = (j & 1) ? (uint8_t)(qs >> 4) : (uint8_t)(qs & 0x0F);
          uint8_t bit = (uint8_t)((qh >> j) & 1u);
          int8_t a = (int8_t)(uint8_t)(nib + (uint8_t)(bit << 4));   // matches u8 add then reinterpret
          int16_t prod = (int16_t)(q8[j * 32 + q * 8 + l] * a);      // vwmul i16
          aux32[l] += scale * (int)prod;                            // vwmacc i32, lane l
        }
      }
    }
    const float d = fp16_to_fp32(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    int sumi = 0;
    for (int t = 0; t < QK_K / 16; ++t) sumi += y[i].bsums[t] * mins[t / 2];
    const float dmin = fp16_to_fp32(x[i].dmin) * y[i].d;
    sumf -= dmin * sumi;
  }
  for (int l = 0; l < 8; ++l) sumf += sums[l];
  return sumf;
}

static uint32_t f2u(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static int rr(unsigned *seed, int lo, int hi) { return lo + (int)(rand_r(seed) % (unsigned)(hi - lo + 1)); }
static void fill_random_pair(block_q5_K *bx, block_q8_K *by, unsigned *seed) {
  uint8_t L[QK_K], ls[NSUB], lm[NSUB]; int8_t q8[QK_K];
  for (int j = 0; j < NSUB; j++) { ls[j] = (uint8_t)rr(seed, 1, 63); lm[j] = (uint8_t)rr(seed, 0, 63); }
  for (int j = 0; j < QK_K; j++) { L[j] = (uint8_t)rr(seed, 0, 31); q8[j] = (int8_t)rr(seed, -127, 127); }
  uint16_t d_bits = fp32_to_fp16((float)rr(seed, 1, 400) / 20000.0f);
  uint16_t dmin_bits = fp32_to_fp16((float)rr(seed, 0, 200) / 20000.0f);
  build_q5k(bx, ls, lm, L, d_bits, dmin_bits);
  build_q8k(by, q8, (float)rr(seed, 1, 400) / 2000.0f);
}

// (1) decode-equivalence: inline-decoded a[pos] vs decode_q5k(aux8) byte-for-byte
static int decode_equiv_check(const block_q5_K *x) {
  int8_t aux8[QK_K]; decode_q5k(x, aux8);
  const uint8_t *qs_base = x->qs, *qh_base = x->qh;
  int mism = 0;
  for (int j = 0; j < 8; ++j) {
    int p = j >> 1;
    for (int q = 0; q < 4; ++q)
      for (int l = 0; l < 8; ++l) {
        uint8_t qs = qs_base[p * 32 + q * 8 + l];
        uint8_t qh = qh_base[q * 8 + l];
        uint8_t nib = (j & 1) ? (uint8_t)(qs >> 4) : (uint8_t)(qs & 0x0F);
        uint8_t bit = (uint8_t)((qh >> j) & 1u);
        int8_t a_inline = (int8_t)(uint8_t)(nib + (uint8_t)(bit << 4));
        int pos = j * 32 + q * 8 + l;            // aux8 strip layout
        if (a_inline != aux8[pos]) mism++;
      }
  }
  return mism;
}

int main(void) {
  const size_t n = 4096; const size_t nb = n / QK_K; const int trials = 256;
  block_q5_K *bx = malloc(nb * sizeof(block_q5_K));
  block_q8_K *by = malloc(nb * sizeof(block_q8_K));

  // (1) DECODE EQUIVALENCE
  unsigned seed = 0x5c0ffeeu; int dec_mism = 0;
  for (int t = 0; t < trials; t++) {
    for (size_t i = 0; i < nb; i++) fill_random_pair(&bx[i], &by[i], &seed);
    for (size_t i = 0; i < nb; i++) dec_mism += decode_equiv_check(&bx[i]);
  }
  printf("[1] decode-equivalence (inline a[pos] vs ggml decode_q5k): %s (mismatched bytes=%d over %d blocks x256 vals)\n",
         dec_mism == 0 ? "BYTE-IDENTICAL" : "MISMATCH", dec_mism, (int)nb * trials);

  // (2) FOLD BIT-EXACT: mirror_regres vs ref_generic, bit-for-bit
  seed = 0x5c0ffeeu; int match = 0; uint32_t sk = 0, sg = 0;
  for (int t = 0; t < trials; t++) {
    for (size_t i = 0; i < nb; i++) fill_random_pair(&bx[i], &by[i], &seed);
    float r_reg = mirror_regres(n, bx, by);
    float r_gen = ref_generic(n, bx, by);
    if (f2u(r_reg) == f2u(r_gen)) match++;
    if (t == 0) { sk = f2u(r_reg); sg = f2u(r_gen); }
  }
  printf("[2] fold bit-exact (mirror_regres vs generic oracle): %d/%d exact   sample reg=0x%08x gen=0x%08x\n",
         match, trials, sk, sg);

  // property regimes (verbatim from q5k_verify_driver.c)
  uint8_t L[QK_K], ls[NSUB], lm[NSUB]; int8_t q8[QK_K]; int prop_pass = 0, prop_total = 0;
  struct { const char *nm; } names[4] = {{"all-zero-L"}, {"max-L-all-qh"}, {"mins-dominant"}, {"extreme-fp16"}};
  for (int c = 0; c < 4; c++) {
    if (c == 0) { for (int j=0;j<NSUB;j++){ls[j]=20;lm[j]=0;} for (int j=0;j<QK_K;j++){L[j]=0;q8[j]=(int8_t)((j&1)?-30:30);} for (size_t i=0;i<nb;i++){build_q5k(&bx[i],ls,lm,L,fp32_to_fp16(0.01f),fp32_to_fp16(0.0f));build_q8k(&by[i],q8,0.05f);} }
    if (c == 1) { for (int j=0;j<NSUB;j++){ls[j]=(uint8_t)(40+3*j);lm[j]=20;} for (int j=0;j<QK_K;j++){L[j]=31;q8[j]=(int8_t)(60+(j%41));} for (size_t i=0;i<nb;i++){build_q5k(&bx[i],ls,lm,L,fp32_to_fp16(0.02f),fp32_to_fp16(0.01f));build_q8k(&by[i],q8,0.1f);} }
    if (c == 2) { for (int j=0;j<NSUB;j++){ls[j]=10;lm[j]=63;} for (int j=0;j<QK_K;j++){L[j]=0;q8[j]=(int8_t)100;} for (size_t i=0;i<nb;i++){build_q5k(&bx[i],ls,lm,L,fp32_to_fp16(0.01f),fp32_to_fp16(0.02f));build_q8k(&by[i],q8,0.08f);} }
    if (c == 3) { for (int j=0;j<NSUB;j++){ls[j]=63;lm[j]=63;} for (int j=0;j<QK_K;j++){L[j]=(uint8_t)(j&31);q8[j]=(int8_t)((j&1)?-127:127);} for (size_t i=0;i<nb;i++){build_q5k(&bx[i],ls,lm,L,0x7bff,0x0400);build_q8k(&by[i],q8,1.0f);} }
    float r_reg = mirror_regres(n, bx, by), r_gen = ref_generic(n, bx, by);
    int ok = (f2u(r_reg) == f2u(r_gen));
    prop_total++; prop_pass += ok;
    printf("    property[%-14s]: reg=0x%08x gen=0x%08x  %s\n", names[c].nm, f2u(r_reg), f2u(r_gen), ok ? "EXACT" : "MISMATCH");
  }
  printf("[3] property regimes bit-exact: %d/%d\n", prop_pass, prop_total);

  int all_ok = (dec_mism == 0) && (match == trials) && (prop_pass == prop_total);
  printf("\nVERDICT (laptop byte-exact of register-resident reorder): %s\n", all_ok ? "BIT-EXACT" : "FAILED");
  free(bx); free(by);
  return all_ok ? 0 : 1;
}
