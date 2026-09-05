# IQ2_S row-dequant physical binding refresh

## Scope

The comparison ledger still placed SG2044 `dequantize_row_iq2_s` at 84.24%
of source even though the shaped author tree and its typed entry/bitmask
memory relations were already present.  This report records a targeted
physical-parameter refresh; no compiler or DSL source changed.

## Work account

With the runner's implicit `LMUL=m1` binding, final RISC-V IR already had one
typed unit entry load and one typed bitmask-window load, but each eight-value
payload was converted into two four-lane carriers before f32 production.
Generated C consequently contained two tuple truncations, two
`vslidedown` operations, and two four-element stores per entry.  Final-IR
register peak was only 2 groups, so the split was not forced by the resource
contract.

The legal LMUL domain was measured on SG2044.  LMUL=m2 and m4 were slower;
LMUL=m8 kept each eight-element result in one f32 carrier.  At LMUL=m8 the two
layout conversions disappear, generated C has one eight-element store per
entry, and final-IR peak remains only 4 vector groups.

This is a parameter choice in the target runner.  It does not add a
format-dependent compiler path or change the canonical program.

## Results

All formal values below are ten-repetition runs on the current worktree and
passed the numerical tolerance check with zero reported absolute and relative
error.

| target | Weft | source | ratio | binding |
|---|---:|---:|---:|---|
| SG2044 | 673.905403 MEl/s | 426.506007 MEl/s | 158.01% | LMUL=m8 |
| K1/X60 | 225.564634 MEl/s | 146.691980 MEl/s | 153.77% | existing default |

The SG2044 comparison row is no longer a bottom-cluster optimization target.
Both rows in `kernel-performance-comparison.csv` now carry the current
targeted-rerun values; the formal full-run Weft CSV remains untouched.

