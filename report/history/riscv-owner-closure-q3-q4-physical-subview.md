# RISC-V physical owner closure, Q3_K, and Q4_K local subview report

## Scope

This report records one coherent compiler state after validating the physical
topology owner, closing the previously unsupported Q4_K local encoded subview,
and re-measuring Q3_K and Q4_K against same-turn GGML baselines. It does not
claim full-format completion.

The structural reference was Triton's split between typed memory views and
loads: `ttg.memdesc_subslice` preserves allocation/layout identity while
changing a logical subview, and `ttg.local_load` materializes the selected view.
The relevant implementation is in:

- `ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`
- `ref/triton/lib/Dialect/TritonGPU/IR/Ops.cpp`
- `ref/triton/lib/Dialect/TritonGPU/Transforms/Prefetch.cpp`
- `ref/triton/lib/Dialect/TritonGPU/Transforms/OptimizeDotOperands.cpp`

TileLang carries the analogous facts as `BufferRegion` minima/extents plus
underlying buffer strides. Weft cannot directly reuse either GPU representation:
the relevant physical relation here is one logical K point combined with an RVV
lane axis that traverses encoded CPU records.

## Structural closure: external checks

### Equivalent author spellings

The two Q4_K decompositions were compiled and run with identical physical
parameters (`LMUL=4`, `unroll=2`, `pipeline_depth=1`). Both remained numerical
within tolerance.

| Target | `mac_pairs` GOP/s | `mac_groups(4)` GOP/s |
|---|---:|---:|
| SG2044 | 8.686854 | 10.089278 |
| K1 | 3.035081 | 3.224756 |

Their final physical IRs contain the same operation kinds and counts: three
Levels, two `rvv_grouped_mac_reduce` operations, and two
`rvv_contract_step` operations. The authored grouping parameter differs, but it
does not select a second lowering path. Both final IRs pass `weft-opt`
round-trip parsing, verification, and canonicalization.

### Existing over-source paths

For eleven existing paths, the same canonical IR was compiled by clean commit
`22d296893` and by this compiler state. Final RISC-V IR was byte-identical on
both VLEN128 and VLEN256 for every path:

`Q2_K`, `Q6_K`, `TQ2_0`, `Q5_K`, `IQ4_XS`, `Q4_0`, `Q4_1`, `Q5_1`,
`IQ4_NL`, `Q1_0`, and persistent `Q4_K`.

All 22 one-repetition hardware runs remained numerical within tolerance. The
SG2044 GOP/s values were respectively 9.908, 9.251, 20.101, 7.754, 7.323,
10.516, 6.731, 6.332, 7.822, 8.759, and 10.147. The K1 values were 2.739,
2.650, 6.491, 2.461, 2.435, 2.238, 2.006, 1.861, 2.374, 3.648, and 3.221.

This check detects the failure mode previously seen when TQ2_0 changed from
6.42 to 2.13 GOP/s because two topology owners disagreed. No such topology or
instruction-count change occurred here.

### `weft-opt` observability

`weft-opt` parses and verifies the strong-test IRs and representative Q2_K,
TQ2_0, persistent Q4_K, and staged Q4_K final IRs. Two observable contracts are
still incomplete:

1. Generic canonicalization introduces one `arith.select` in each sampled
   quantization path. The result is valid MLIR, but `arith.select` is outside the
   intrinsic-C terminal operation contract.
2. `ime_fragment_mma` is declared `Pure` while also carrying a
   `memory_clobber` attribute. The effect contract is internally inconsistent.

Generic CSE left RVV operation counts unchanged in ten of the eleven regression
paths. In Q4_0 it merged one duplicate pure `rvv_splat`; no independent partial
operation was merged in this corpus.

## Q4_K local encoded subview

The staged author tree already expresses a local `[N,K]` materialization and a
K sub-Level. The unsupported edge was the child access that fixes one K
coordinate while keeping N in RVV lanes. Existing layered-window operations
required the storage/reduction axis itself to survive in the result, so no op
could represent this two-axis relation.

The new `rvv_layered_record_load` physical operation carries:

- the typed encoded field;
- the typed K origin point and scalar local K offset;
- the grouped/layered storage plan and access relation;
- a result type whose surviving free axis has the selected RVV time/lane layout;
- one exact RVV leaf and resource contract.

`ShareRISCVLayeredWindows` derives the operation only from nested typed
`domain`/`index` extracts, axis identities, grouped/layered geometry, and the
selected result layout. The verifier requires the reduction axis to disappear,
all other logical axes to be preserved, exactly one RVV lane axis, an exact
projection extent, and a closed byte-layer plan. The emitter mechanically emits
one record-strided byte load plus the selected shift/mask for each issue part.

The final staged IR contains four `rvv_layered_record_load` operations and
passes `weft-opt -verify-each -verify-roundtrip --canonicalize`. Real hardware
results are numerical within tolerance:

| Target | Repetitions | GOP/s |
|---|---:|---:|
| SG2044 | 1 | 1.618410 |
| K1 | 1 | 1.376037 |

The result closes an executable IR gap, not a performance gap. SG and K1
assembly contain 114 and 47 `vlse8` instructions respectively. The author-level
`materialize` lifetime is preserved, but no compiler pass currently converts
the canonical encoded records into a unit-stride local encoded panel.

Same-turn SG2044 Q4_K measurements make the boundary explicit:

| Program | Repetitions | GOP/s | Ratio to source |
|---|---:|---:|---:|
| GGML source RVV | 10 | 9.835053 | 1.000 |
| canonical row-by-column Weft | 10 | 5.005900 | 0.509 |
| persistent derived-layout Weft | 10 | 10.133713 | 1.030 |
| staged canonical-input Weft | 1 | 1.618410 | 0.165 |

The persistent program changes artifact layout and ABI, so it remains an
author/caller choice. A high-performance canonical-input program needs a typed
local encoded-pack/materialization decision; the compiler cannot silently turn
the canonical ABI into the persistent one.

## Q3_K

`PackedPlaneMergeOp` now preserves non-scale logical axes as scalar register
replicas, and its emitter indexes packed words by the full typed register
coordinate instead of assuming one isolated scale axis. Aligned packed words
are loaded as two halfwords, which reduced the scalar setup in the Q3_K path.

The remaining structural alternatives already tested were negative: larger
LMUL activation carriers caused spill/reload, standalone scale asm caused
vector spills and dynamic stack setup, and MR/NR/unroll/pipeline changes did not
improve the path. NC/MC were therefore measured as the remaining parameteric
space. `MC=2` was the SG2044 winner; K1 retains `MC=8`.

Same-turn ten-repetition SG2044 results:

| Program | GOP/s |
|---|---:|
| GGML source Q3_K | 7.170891 |
| Weft Q3_K (`NC=32, MC=2, MR=1, NR=1`) | 7.389963 |

The ratio is 1.0305. K1 Weft is 2.724846 GOP/s; the recorded source result is
2.449247 GOP/s, a ratio of 1.1125.

## Remaining boundary

This state does not complete the eight row-by-column IQ/TQ1 author trees. The
known common physical gap is still a typed codebook/index window that can be
shared by multiple output consumers. Existing `LookupOp` is a same-element-width
single-element gather and cannot represent a `u64` entry expanded into a typed
eight-byte trailing axis; `ConvertLayoutOp` cannot change element type or
logical shape. That work requires a separate typed memory operation rather than
an emitter condition or a format-specific branch.
