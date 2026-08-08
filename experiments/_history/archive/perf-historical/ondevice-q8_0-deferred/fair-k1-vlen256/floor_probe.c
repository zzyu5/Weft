// Q1-c FLOOR PROBE -- pure nb-length-dependent fadd latency floor for q8_0 vec_dot.
//
// The q8_0 vec_dot's cross-block accumulation is a STRICT serial fp32 fold
// (pinned §1 oracle: sumf = sumf + t_block, left-assoc, no-FMA, no reassoc).
// This is a data-dependent chain of length nb (=n/32). Each fadd cannot start
// until the previous retires, so nb * (fadd latency) is a HARD lower bound on
// ns/call that NO amount of vector cleverness in the per-block integer work can
// beat -- the integer work (loads, vwmul, vwredsum) pipelines UNDER this chain.
//
// This microbench isolates ONLY that dependent-fadd chain (t[] pre-filled, so
// the per-block product is NOT on the timed critical path), timed with the SAME
// methodology as q8_0_verify_driver.c (best-of-7, 60000 iters, CLOBBER, volatile
// sink). Compile with the SAME clang/flags (full march, -ffp-contract=off).
//
// Report: floor ns/call = pure serial-fold latency for nb blocks.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CLOBBER() __asm__ __volatile__("" ::: "memory")

static double now_s(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// Pure §1 fold accumulation: strict left-assoc dependent fadd chain.
// noinline + array input => clang keeps the true serial chain (fp add is not
// associative, so with -ffp-contract=off / no -ffast-math it will NOT reassociate
// or vectorize this reduction). Critical path = nb dependent fadds.
__attribute__((noinline)) static float fold_chain(int nb, const float *t) {
  float s = 0.0f;
  for (int i = 0; i < nb; i++) {
    s = s + t[i]; // the pinned §1 accumulation step, isolated
  }
  return s;
}

int main(int argc, char **argv) {
  size_t n = 4096;       // must match driver: n/32 = nb blocks
  long perf_iters = 60000;
  int REPEATS = 7;
  const char *march = (argc > 1) ? argv[1] : "unknown";
  if (argc > 2) n = (size_t)strtoul(argv[2], NULL, 10);
  if (argc > 3) perf_iters = atol(argv[3]);
  int nb = (int)(n / 32);

  // realistic per-block addends (magnitude ~ sumi*dx*dy for [-4,4] q8_0 data);
  // varied & finite so the chain is neither constant-foldable nor degenerate.
  float *t = malloc((size_t)nb * sizeof(float));
  unsigned seed = 0x1234567u;
  for (int i = 0; i < nb; i++) {
    float r = (float)((int)(rand_r(&seed) % 4001) - 2000) / 7.0f; // ~[-285,285]
    t[i] = r;
  }

  volatile float sink = 0.0f;
  // warmup
  for (int i = 0; i < 2000; i++) { CLOBBER(); sink += fold_chain(nb, t); }

  double best = 1e30;
  for (int r = 0; r < REPEATS; r++) {
    double t0 = now_s();
    for (long i = 0; i < perf_iters; i++) {
      CLOBBER();
      sink += fold_chain(nb, t);
    }
    double dt = now_s() - t0;
    if (dt < best) best = dt;
  }
  double ns = best * 1e9 / (double)perf_iters;
  printf("floor: nb=%d  floor_ns=%.1f ns/call  per_fadd=%.3f ns  march=%s (sink=%.6g)\n",
         nb, ns, ns / (double)nb, march, (double)sink);
  free(t);
  return 0;
}
