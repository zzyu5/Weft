#!/usr/bin/env bash
# run_fair.sh -- on-board (ssh rvv) fair re-test of q8_0 deferred vs per-block vs factory.
# Kernels are clang-20 CROSS-compiled .o shipped here (fixed instruction stream).
# Driver compiled on-board (clang-17, rv64gcv, fp16 = integer -> march-neutral; only
# TIMES the kernel .o). Confound = whether the KERNEL's scalar fp16 became a
# __extendhfsf2 libcall (no-zfh) vs hardware fcvt.s.h (zfh). A uses vector fp16 gather.
# no-zfh variants also link extendhfsf2_shim.o (board libgcc lacks the symbol).
set -u
cd "$(dirname "$0")"
CORE=8
DN=4096; DT=256; DI=100000   # n, verify-trials, perf-iters (driver internal best-of-7)
ROUNDS=11
C=$(command -v clang-17 || command -v clang)
OD=$(command -v llvm-objdump || command -v objdump)
MARCH_DRV=rv64gcv
FLAGS_DRV="-O2 -march=$MARCH_DRV -mabi=lp64d -ffp-contract=off"

echo "== ENV FINGERPRINT =="
echo "uname=$(uname -a)"
echo "clang_driver=$C  $($C --version | head -1)"
echo "isa=$(grep -m1 -i isa /proc/cpuinfo)"
echo "pinned_core=$CORE governor=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_governor 2>/dev/null) max_khz=$(cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/cpuinfo_max_freq 2>/dev/null)"
echo "driver_flags=$FLAGS_DRV  n=$DN verify=$DT perf_iters=$DI rounds=$ROUNDS"
echo "kernel_.o provenance=clang-20 cross -O2 -mabi=lp64d -ffp-contract=off (A/B_zfh/C_zfh=rv64gcv_zfh_zvfhmin; B_nozfh/C_nozfh=rv64gcv_zvfhmin)"

echo "== COMPILE DRIVER + SHIM =="
$C $FLAGS_DRV -x c q8_0_verify_driver.c -c -o driver.o 2>cc_driver.err && echo "driver.o OK" || { echo FAIL; cat cc_driver.err; exit 2; }
$C $FLAGS_DRV -x c extendhfsf2_shim.c   -c -o shim.o   2>cc_shim.err   && echo "shim.o OK"   || { echo FAIL; cat cc_shim.err; exit 2; }

freq() { cat /sys/devices/system/cpu/cpu${CORE}/cpufreq/scaling_cur_freq 2>/dev/null || echo NA; }

for K in A_zfh B_nozfh B_zfh C_nozfh C_zfh; do
  KO=k_${K}.o
  echo "############################## KERNEL=$K ($KO) ##############################"
  echo "-- objdump confound seal ($K) --"
  echo -n "   __extendhfsf2 libcall : "; $OD -dr $KO 2>/dev/null | grep -c "extendhfsf2"
  echo -n "   hw fcvt.s.h           : "; $OD -d  $KO 2>/dev/null | grep -c "fcvt.s.h"
  echo -n "   vlse16 (vec fp16 gather): "; $OD -d $KO 2>/dev/null | grep -c "vlse16"
  echo -n "   vfwcvt.f.f.v          : "; $OD -d $KO 2>/dev/null | grep -c "vfwcvt.f.f.v"
  # link (no-zfh variants need the shim for __extendhfsf2)
  case $K in
    *nozfh*) EXTRA=shim.o ;;
    *)       EXTRA= ;;
  esac
  $C $FLAGS_DRV driver.o $KO $EXTRA -lm -o driver_$K 2>ld_$K.err && echo "  link OK (extra=$EXTRA)" || { echo "  LINK FAIL"; cat ld_$K.err; continue; }
  # correctness gate (bit-exact vs pinned §1 oracle) -- one run
  echo "-- correctness ($K) --"
  taskset -c $CORE ./driver_$K $MARCH_DRV $DN $DT 2000 2>/dev/null | grep -E '\[GATE\]|VERDICT|property:|mutation caught'
  # warmup (dropped)
  taskset -c $CORE ./driver_$K $MARCH_DRV $DN $DT $DI >/dev/null 2>&1
  echo "-- perf rounds ($K): raw kernel_ns per round --"
  for r in $(seq 1 $ROUNDS); do
    f=$(freq)
    ns=$(taskset -c $CORE ./driver_$K $MARCH_DRV $DN $DT $DI 2>/dev/null | awk '/^kernel/{print $3}')
    echo "PERF kernel=$K round=$r kernel_ns=$ns freq_khz=$f"
  done
done
echo "== DONE =="
