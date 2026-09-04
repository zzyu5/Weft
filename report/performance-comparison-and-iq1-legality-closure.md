# Performance comparison and IQ1 legality closure

## Scope

This snapshot covers three changes against the current 206-row production ledger:

- make the current Weft/source gap directly readable without configuration columns;
- close the Q6_K and IQ1_M quantized vec-dot legality failures;
- replace the IQ1_M scalar correction expansion and IQ1_S scalar prefill traversal with explicit shaped programs.

No full performance rerun was performed. Rows changed by this work use ten-repetition targeted runs on SG2044 and K1/X60; all other rows retain the last full-rerun value.

## Human-facing comparison ledger

[`kernel-performance-comparison.csv`](kernel-performance-comparison.csv) contains exactly:

```text
kernel,target,phase,weft,source,ratio,measurement
```

It contains 206 rows, has no duplicate visible key, and is sorted by failure first and then ascending ratio. Kernel names are qualified where the family is otherwise ambiguous: for example `mul_mat_q8_0`, `quantize_row_q8_0`, and `dequantize_row_q8_0` are different entries. `measurement=full-rerun` means the complete row is unchanged from the full-rerun anchor at `2ef33bd25`; a changed row is `targeted-rerun`. A later rerun that reproduces an anchor row byte-for-byte is not distinguishable from the anchor by this rule.

The current distribution is:

| scope | >=100% | 90-100% | 70-90% | 50-70% | <50% |
|---|---:|---:|---:|---:|---:|
| all | 132 | 28 | 27 | 6 | 13 |
| SG2044 | 57 | 19 | 15 | 4 | 8 |
| K1/X60 | 75 | 9 | 12 | 2 | 5 |
| MUL_MAT | 62 | 17 | 15 | 4 | 6 |
| vec-dot | 26 | 9 | 7 | 2 | 4 |
| row-dequant | 40 | 2 | 3 | 0 | 3 |
| activation quantize | 4 | 0 | 2 | 0 | 0 |

All 206 Weft rows join one-to-one with the fixed source ledger by family, kernel, phase, target, and shape. The source ledger has 38 additional kernels outside the current Weft measurement scope.

## Q6_K legality

`Q6_K.scales` is declared as signed bytes with `grouped(16) @ layered(16)`. For an 8-bit element and equal group/layer extents, its storage map is the affine identity:

```text
floor(i / group) * group + i % group == i
```

`PlanRISCVMemory` previously kept the nominal `grouped_layered` name, selected an indexed-entry load, and later rejected the edge because no byte-aligned record mapping closed. The pass now records the proven physical edge as `unit/natural`. When the result has register replicas, it also leaves the extract for the existing replica-storage owner instead of prematurely materializing one indexed-entry load per part.

This is a geometry legality rule, not a format rule. Q6_K scales are its only current production consumer, so this work does not claim a second performance input.

Both previously failing entries now compile, run, and remain within tolerance:

| entry | target | result | source | ratio |
|---|---|---:|---:|---:|
| `q6_k_q8_k` | SG2044 | 3.171970 | 4.936994 | 64.25% |
| `q6_k_q8_k` | K1/X60 | 2.236471 | 2.321293 | 96.35% |

The legality failure is closed, but the SG performance deficit is not. The generated hot body still supplies the 16 scales as four two-byte vector loads followed by eight scalar extracts/slides; it contains 2 widened multiplies, 8 widened reductions, and 8 `vslidedown` operations. The selected memory edge is now legal, but its supply width is still worse than the donor's wide scale supply. This is a concrete P3 memory-form/supply-width deficit.

The relevant reference mechanism is Triton's coalescing analysis: it derives order and per-thread width from typed contiguity and divisibility facts, then writes a new encoding into IR before lowering (`ref/triton/lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp:31-94`). Weft cannot copy the GPU thread distribution, but the same separation applies: legality and width belong to the selected memory edge, not to terminal C spelling.

## IQ1_M shaped correction

The previous correction was an `8 x 4 x 8` Python expansion. The new program explicitly carries:

- `group = 8`;
- `scale_part = 2`;
- `entry = 2`;
- `payload = 8`.

The main and correction paths both contract `entry x payload`, multiply by the per-part scale, and then reduce `scale_part` and `group`. The input ABI, persistent bytes, Level ownership, i32 accumulation boundary, and result are unchanged. All six production measurements are exact against the reference result.

The performance result is negative and replaces the earlier temporary sequential-tree numbers:

| entry | target | old recorded | shaped | source | shaped/source |
|---|---|---:|---:|---:|---:|
| vec-dot | SG2044 | 1.587860 | 0.537902 | 2.650472 | 20.29% |
| MUL_MAT decode | SG2044 | 1.609963 | 0.525254 | 2.639333 | 19.90% |
| MUL_MAT prefill | SG2044 | 1.588674 | 0.525161 | 2.634283 | 19.94% |
| vec-dot | K1/X60 | 0.684853 | 0.407818 | 1.435304 | 28.41% |
| MUL_MAT decode | K1/X60 | 0.685928 | 0.421319 | 1.424159 | 29.58% |
| MUL_MAT prefill | K1/X60 | 0.686422 | 0.421602 | 1.446879 | 29.14% |

The seven-step work ledger identifies one specific compiler gap:

1. The canonical contraction eliminates `entry,payload`; `scale_part,group` remain as shaped values until two later reductions.
2. Each 256-element record has 256 main products and 256 sign-correction products; the mathematical work did not increase.
3. The final SG C hot body contains 82 gather spellings and reconstructs the two operand paths independently.
4. The memory edges are legal, so this is not a hard alignment or address-form failure.
5. The hot body contains 32 `vwmul`, 32 `vwredsum`, and 39 `vslidedown` spellings, with no typed partial set.
6. Pipeline is not considered because the product/reduction topology is already over-sliced.
7. The donor and the author program both expose the same four logical axes; the difference is physical partial organization.

`matchReplicaScaledDotReduction` correctly observes two later reduction axes. The nested partial planner and materializer both require exactly one, so this program falls to sequential per-issue finalization. The missing mechanism is a topology that distinguishes the contraction's `entry x payload` product carrier from the outer `scale_part x group` reductions. Merely relaxing `reducedAxes.size() == 1` would be wrong: the plan and materializer currently store only one window axis and would lose the second reduction relation.

This matches the useful part of the reference designs. Triton first performs reduction over register bases and lane bases selected by the source layout, then removes only the actual reduction axis (`ref/triton/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp:228-351`). TileLang projects partial storage from the update site's reduction axes and requires structurally equal plans across update sites (`ref/tilelang/src/transform/reducer_plan_materialize.cc:559-733`). Their GPU execution objects are not reusable here; the reusable contract is that all surviving axes and all later reduction axes remain explicit in the frozen topology.

The source currently uses `payload * 0` to broadcast the sign delta over the payload axis before `contract`. A cleaner pointwise-broadcast plus explicit reduction was tested: it reaches terminal lowering without an exact wide-reduce leaf. The zero term is therefore retained as an explicit current frontend/physical-contract limitation, not reported as an optimization.

## IQ1_S decode and prefill

Decode and prefill are now two author programs:

- decode retains the single-output GEMV traversal;
- prefill uses `NC/MC` tiles, `MR/NR` accumulators, materialized weight/activation panels, and activation reuse across an output cohort.

The blocked prefill change is effective:

| target | old row-column | blocked | source | old/source | blocked/source |
|---|---:|---:|---:|---:|---:|
| SG2044 | 0.636727 | 4.442665 | 4.854846 | 13.12% | 91.51% |
| K1/X60 | 0.892714 | 1.731067 | 2.740584 | 32.58% | 63.16% |

The legal binding is currently `MR=1, NR=4`. `MR=2/4/8` fails because the activation MR axis and weight NR axis are independently assigned to RVV lane/reduction streams, while the compiler has no common `MR x NR` product/partial topology. In the legal `MR=1` program, the generated C still reconstructs qh/index metadata and performs the codebook gather once per NR output replica; the activation window is loaded once. The remaining K1 gap is therefore a concrete output-replica carrier and multi-consumer supply gap, not missing outer blocking.

The old shaped standalone/decode entry was also rerun instead of trusted from CSV. It produces grossly incorrect values on both machines: the logical `group` iota does not become a typed record point, while grouped storage addressing requires a scalar group index tied to the source `PhysicalPoint`. The correct entry therefore keeps group as an explicit `L.subs` issue-time loop until physical group-to-record point/window mapping exists. Its current results are:

| entry | target | result | source | ratio |
|---|---|---:|---:|---:|
| vec-dot | SG2044 | 1.972146 | 4.861243 | 40.57% |
| MUL_MAT decode | SG2044 | 2.020052 | 4.839919 | 41.74% |
| vec-dot | K1/X60 | 1.006267 | 2.711676 | 37.11% |
| MUL_MAT decode | K1/X60 | 0.978946 | 2.663380 | 36.76% |

These values are not a valid regression from the old higher rows: the old shaped program fails current numerical validation and cannot supply a performance number.

## Current bottom clusters

The nineteen entries below 70% now fall into these evidence-backed groups:

1. **IQ1_M, six entries at 19.90-29.58%.** One shared compiler defect: the partial topology cannot represent `entry x payload` contraction followed by two distinct shaped reductions. This is the largest exact cluster.
2. **IQ1_S standalone/decode, four entries at 36.76-41.74%.** The logical group axis lacks a typed group-to-record point/window relation, so the only correct program serializes the eight groups. This is P1 carrier plus P3 storage geometry.
3. **IQ1_S K1 prefill, 63.16%.** Blocking exists; missing `MR x NR` partial closure and per-replica qh/index reconstruction are distinct from the decode defect.
4. **SG row-dequant Q1_0/TQ1_0/TQ2_0, three entries at 21.40-42.75%.** They share the symptom of an over-small output carrier and excess conversion/slide/truncate work, but their bitmask and radix storage relations differ. Current evidence does not justify one common topology rule.
5. **TQ1_0 SG vec-dot/decode, two entries at 56.95-57.87%.** Standalone and decode remain synchronized; the deficit is inside the shared radix contraction, not the GEMV wrapper.
6. **Q2_K prefill, two entries at 59.37-64.02%.** The author declares materialized panels, but decoded scale/payload residency is recreated inside the output/K lifetime. This is a P2 residency placement deficit.
7. **Q6_K SG vec-dot, 64.25%.** Legality is fixed; scale supply remains four narrow loads plus eight extracts instead of one wide typed supply.

The table therefore changes the immediate diagnosis: the dominant new cluster is IQ1_M topology, not a family-wide codebook-reuse issue. The three low row-dequant formats are not yet one proven compiler relation.

## Mechanical and runtime verification

- Build completed with the default CMake target graph; `weft-compile` and `weft-opt` are current.
- 96/96 generated Physical IR inputs (24 vec-dot and 24 row-dequant formats on both targets) independently parse and verify.
- 96/96 safely run canonicalizer, CSE, layout canonicalization, Share twice, final verification, and round-trip verification twice; every second-run textual diff is zero.
- All ten previously over-source vec-dot entries plus Q4_K persistent prefill ran on both machines with one repetition, stayed within tolerance, and showed no catastrophic regression. These one-repetition checks are not written to the performance CSV.
- Every affected value written to the CSV is a ten-repetition run. There are no FAIL rows and no retained old value for a failed or numerically invalid program.
