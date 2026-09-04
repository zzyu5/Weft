# IQ1 reduction topology and record-window closure

## Scope and immutable pre-change ledger

This snapshot starts from clean revision `3d16f3700`. It addresses two clusters
already present in the 206-row comparison ledger; it does not rerun or optimize
unrelated formats.

| cluster | SG2044 | K1/X60 | fixed source ratio range |
|---|---:|---:|---:|
| IQ1_M vec-dot / MUL_MAT decode / MUL_MAT prefill | 3 entries | 3 entries | 19.90-29.58% |
| IQ1_S vec-dot / MUL_MAT decode | 2 entries | 2 entries | 36.76-41.74% |

The diagnostic procedure is the seven-step work ledger in
`doc/compiler/optimization-principles.md` section 9. The predictions below are
written before implementation and are not rewritten to fit the result.

## Prediction P-IQ1M: preserve the surviving axis between reductions

Author precondition: the canonical graph explicitly contains
`contract(entry, payload)`, then `reduce(scale_part)`, then `reduce(group)` for
both the main and sign-correction paths. The compiler must not reassociate these
operations.

Predicted primary deficit: P1/P4. The nested RVV planner treats the result of
the first reduction as terminal unless it has one scalar replica. It therefore
cannot keep the surviving `group` axis as eight register replicas for the next
real `ReduceOp`, and falls back to sequential per-issue finalization.

Expected visible change:

- each inner `scale_part` reduction acquires a typed nested partial plan whose
  `output_axes` contains `group` and whose `output_replicas` is eight;
- the materializer emits one assembled `[group]` result and leaves the outer
  `group` reduction as a real operation;
- both main and correction paths contain typed partial-set/repack/reduce/
  scale-combine/finalize operations;
- the SG hot body's pre-change 32 widened multiplies, 32 widened reductions and
  39 `vslidedown` spellings decrease. The 512 mathematical products per
  256-element record do not change.

Typed facts: ordered canonical `ReduceOp` use-def, axis identities and extents,
the contraction `over` axes, surviving result axes, lane/time/replica layout,
selected RVV widening leaf, and the target vector-register budget. The two
independent main/correction contractions are separate producer graphs. Q3_K
and Q6_K are regression inputs for the same per-reduction-axis composition:
their existing typed partial topology and final IR must remain unchanged.

This is a physical representation change only. It does not change logical
values, Level ownership, widening position, overflow boundary, or artifact ABI.
The planned full carrier must fit the target resource contract; otherwise the
binding is illegal rather than silently falling back.

If the typed partial operations and widened-reduce/slide counts do not change,
there is no performance reason to retain the implementation. A verifier-only
change would be reported separately and would not count as closing this
prediction.

Reference relation: Triton reduces register bases and lane bases for one source
axis and removes only that axis from the resulting layout
(`ref/triton/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp:228-351`).
TileLang projects every reduction dimension into the induced partial layout and
requires update sites to induce structurally equal plans
(`ref/tilelang/src/transform/reducer_plan_materialize.cc:559-733`). Weft keeps
its explicit sequential `ReduceOp`s; it does not import their SIMT execution
objects.

## Prediction P-IQ1S: preserve the affine entry base when slicing a window

Author precondition: one shaped `group=8` axis indexes natural record-local
fields `qh[group]` and `q[group*4+entry]`, and indexes the matching activation
region. This expresses the same values and integer association as the currently
correct eight-iteration `L.subs` program.

The previously generated shaped program supplies a direct falsifiable defect:
four two-group issues advance the activation bases and `qh` bases, but all four
`q` entry-window loads use byte base zero. Their required record-local bases are
`0, 8, 16, 24`. The missing relation is therefore not a new record dialect op;
it is the affine base projection contract of the existing typed
`rvv_unit_entry_window_load` when its outer entry axis is sliced.

Predicted primary deficit: P1/P3. The group axis already reaches a two-group RVV
carrier, but the selected unit memory edge loses the `group * 4` contribution
during issue-window materialization.

Expected visible change:

- cloned entry-window ops carry the four distinct bases `0, 8, 16, 24`;
- the shaped program passes numerical validation on both targets;
- the eight scalar group issues become four typed two-group product issues;
- standalone vec-dot and MUL_MAT decode change in the same direction.

Typed facts: `entry_axes=[group,entry]`, their extents, payload axis and extent,
entry stride, original affine entry base, selected issue axis/extent and issue
index, plus the natural field access. Q8 activation and IQ1_S `q` are two
independent unit-entry-window inputs in the same graph; Q3_K/Q6_K are external
regressions for shaped record-local coordinates.

The change must not alter the logical group axis, Level graph, widening, or
record ABI. Its extra live state is bounded by one two-group input window and
the already planned full-product carrier. If the emitted bases or dynamic work
counts do not change, there is no reason to implement it.

## Result

Both predicted defects were real, but neither was the complete performance
bottleneck.  The implementation closes the missing typed relations and removes
measurable physical work; it does not make all ten entries reach the source.

### IQ1_M: the complete reduction chain is now part of one frozen topology

`NestedPartialPlanAttr` now carries every post-contraction reduction axis and
extent, the typed set produced after each combine, the combine axis and arity,
and the joint resource peak.  Materialization consumes that complete contract:

```text
rvv_partial_set
  -> rvv_partial_repack
  -> rvv_partial_reduce
  -> rvv_partial_scale_combine logical_reduction_axes=[scale_part]
  -> rvv_partial_combine      logical_reduction_axes=[group]
  -> rvv_partial_finalize
```

The partial combine operations declare `logical_reduction_axes` in ODS and
verify their input/output slot relation.  Crucially, `rvv_partial_reduce`
finishes the canonical `contract(entry,payload,acc=i32)` before the i32 scale
combine.  The compiler does not distribute scale into that contraction.  The
canonical two-reduction chain is not rediscovered by the emitter; it is lowered
to real Physical IR use-def while preserving both logical axes and their order.

This differs from one detail of the pre-implementation prediction: the final
Physical IR does not leave an outer canonical `ReduceOp` over an assembled
`[group]` value.  It freezes both reductions in the typed topology and emits an
axis-labelled pairwise combine for `group`.  The predicted missing information
was correct; the predicted final op shape was not.

For the SG2044 standalone artifact, the visible static work changed as follows:

| spelling / Physical IR entity | before | after |
|---|---:|---:|
| typed partial set/repack/reduce/scale-combine/combine/finalize | 0 | 4 of each |
| `vwmul` | 32 | 4 |
| `vwredsum` | 32 | 16 |
| `vslidedown` | 39 | 0 |
| gather spellings | 82 | 12 `vrgather` + 2 `vluxei32` |

The 256 main products and 256 correction products per record are unchanged.
Both targets produce the exact reference result.

### IQ1_S: the shaped group axis now reaches a legal record window

The author program now exposes `group=8`, `entry=4`, and `payload=8` as logical
axes and contracts over `entry,payload` before the group reduction.  This is the
same integer program and ABI as the serial `L.subs` spelling.

The selected unit-entry memory edge already contained the complete affine
relation, but `PlanRISCVMemory` previously reconstructed only top-level scalar
terms and lost the nested `group * 4` contribution.  Its scalar affine
materializer now recursively preserves constants, iota starts, representation
conversions, casts, and `add/sub/mul`; unsupported expressions fail rather than
falling back to base zero.  The four two-group issues now carry record-local q
bases `0, 8, 16, 24`.  Numerical validation is exact on both targets.

The shaped program changes eight serial group issues into four typed two-group
product issues.  It does not yet unify all issue-local supply: SG2044 still has
one full qh carrier plus four qh subloads, and K1/X60 has one full carrier plus
two subloads.  The q and activation fields likewise retain one distinct window
per issue.  These are not CSE identities because their offsets differ.

The reference boundary was checked before changing the pass.  Triton's
coalescing reads pointer contiguity from `AxisInfo` and derives the memory order
and width before lowering
(`ref/triton/lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp:31-76`).
TileLang records each load/store index as an
explicit `BufferRegion` minimum and extent
(`ref/tilelang/src/transform/pipeline_planning.cc:145-173`).  Weft cannot reuse
their thread or buffer objects, but it follows the same relevant rule here: the
selected memory edge preserves the existing affine coordinate relation; the
emitter does not reconstruct or invent it.

### Ten-repetition target measurements

The source values remain the fixed baseline.  Every new Weft value below is a
ten-repetition targeted run using the production runner, valid quantized
records, Clang 18.1.8 and the common flags in the experiment protocol.  SG2044
uses VLEN128/core 48; K1/X60 uses VLEN256/core 3.  All twelve numerical checks
are exact.

| entry | target | before | after | source | after/source |
|---|---|---:|---:|---:|---:|
| IQ1_M standalone vec-dot | SG2044 | 0.537902 | 1.104445 | 2.650472 | 41.67% |
| IQ1_M MUL_MAT decode | SG2044 | 0.525254 | 2.032608 | 2.639333 | 77.01% |
| IQ1_M MUL_MAT prefill | SG2044 | 0.525161 | 2.035345 | 2.634283 | 77.26% |
| IQ1_M standalone vec-dot | K1/X60 | 0.407818 | 0.861163 | 1.435304 | 60.00% |
| IQ1_M MUL_MAT decode | K1/X60 | 0.421319 | 0.846250 | 1.424159 | 59.42% |
| IQ1_M MUL_MAT prefill | K1/X60 | 0.421602 | 0.846676 | 1.446879 | 58.52% |
| IQ1_S standalone vec-dot | SG2044 | 1.972146 | 3.419773 | 4.861243 | 70.35% |
| IQ1_S MUL_MAT decode | SG2044 | 2.020052 | 3.417835 | 4.839919 | 70.62% |
| IQ1_S MUL_MAT prefill | SG2044 | 4.442665 | 4.451287 | 4.854846 | 91.69% |
| IQ1_S standalone vec-dot | K1/X60 | 1.006267 | 1.997987 | 2.711676 | 73.68% |
| IQ1_S MUL_MAT decode | K1/X60 | 0.978946 | 2.015856 | 2.663380 | 75.69% |
| IQ1_S MUL_MAT prefill | K1/X60 | 1.731067 | 1.735747 | 2.740584 | 63.33% |

IQ1_M's SG standalone result is materially lower than the decode/prefill pair
despite sharing the contraction implementation.  K1's three values agree.  No
current static count establishes the cause of the SG runtime difference, so it
remains an observed, unassigned deficit rather than being folded into the
topology claim.

The human-facing comparison ledger was updated and re-sorted.  Its distribution
is now:

| scope | >=100% | 90-100% | 70-90% | 50-70% | <50% |
|---|---:|---:|---:|---:|---:|
| all 206 | 132 | 28 | 33 | 9 | 4 |
| SG2044 | 57 | 19 | 19 | 4 | 4 |
| K1/X60 | 75 | 9 | 14 | 5 | 0 |

The four entries still below 50% are now SG2044 row-dequant TQ2_0
(21.40%), row-dequant Q1_0 (31.75%), IQ1_M standalone vec-dot (41.67%), and
row-dequant TQ1_0 (42.75%).  The other nine entries below 70% are the
two SG2044 TQ1_0 contraction entries, four K1 IQ1_M/IQ1_S entries, the two
Q2_K prefill entries, SG2044 Q6_K vec-dot, and no additional hidden failure.
They do not currently share one proven physical relation.

This is a targeted update to `kernel-performance-comparison.csv`.  The formal
`weft-kernel-performance.csv` was not partially overwritten: its protocol
requires one clean-checkout, full-manifest run and atomic replacement.

### Negative experiments not retained

- A provisional `widen_scale_then_reduce` realization changed
  `scale * reduce(product)` into `reduce(scale * product)`.  It removed all 16
  remaining `vwredsum` spellings and measured 1.154/1.940/1.941 GOP/s on the
  three SG entries, but it crossed the author's i32 contraction boundary.
  Exact results on the sampled records do not authorize that reassociation;
  the path and its plan fields were removed before commit.
- A vector loop-carried partial accumulator reduced the SG static final
  reductions from four to two, but raised the live i32m4/i16m8 set, introduced
  vector spills, and changed provisional standalone throughput from 1.153 to
  0.904 GOP/s.  The experiment was reverted and is not evidence about the final
  legal path.
- Removing IQ1_M's `payload * 0` broadcast made the frontend reject the
  contraction because `payload` no longer occurred in both operands.  The term
  currently carries an explicit logical broadcast axis; it was retained.
- Narrowing indexed offsets from u32 to u16 did not remove a gather, address
  operation, or dynamic issue and added a conversion.  It was rejected before
  implementation; the existing independent IQ2_XXS measurement was unchanged
  within noise (SG 3.736 to 3.735, K1 1.689 to 1.688 GOP/s).
- Moving IQ1_S qh into an author `materialize` at the same Level is not legal:
  the admitted value is unavailable at that birth placement.  The experiment
  was reverted rather than weakening Level semantics.

### Mechanical and runtime acceptance

- 48 vec-dot/row-dequant entries on each target, 96 total, independently parse
  and verify with `weft-opt`, run canonicalizer and CSE safely, survive Share
  twice, and produce a zero-line second-pass diff.
- Eleven vec-dot regression formats ran once on each target and all remained
  within tolerance.  Q3_K produced 7.475503 GOP/s on SG2044 and 2.866423 on
  K1/X60; Q6_K produced 3.418121 and 2.228485 GOP/s.  These one-repetition
  runs are correctness and gross-regression checks, not replacements for the
  comparison ledger.

The result closes two missing typed relations, but not the two performance
clusters.  IQ1_M still lacks a proven explanation for the residual leaf/supply
work and the SG standalone discrepancy.  IQ1_S still reconstructs qh-derived
metadata per issue and K1 prefill still lacks the previously identified MR x NR
partial closure.  Those statements are current observable boundaries, not
claims that another pass would necessarily recover the remaining gap.
