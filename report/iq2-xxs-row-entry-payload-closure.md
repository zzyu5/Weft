# IQ2_XXS row-dequant entry/payload closure

## Scope

This report records the targeted closure of `dequantize_row_iq2_xxs` on
SG2044 and K1/X60.  It does not update the formal full-run Weft CSV.  The
human-facing comparison table is updated with ten-repetition targeted runs.

## Seven-step work account before changing the program

The previous source represented one 32-element group as one flat logical
axis and reconstructed

```text
entry = element / 8
payload = element % 8
```

inside that axis.  The source work remained 256 decoded elements per record,
but the flat spelling hid the author's known `entry=4 × payload=8`
coordinate relation.

On SG2044 with the best legal flat-tree parameter found before the rewrite
(`LMUL=m4`, `unroll=2`), the generated artifact had:

- a 32-lane storage/index carrier and a two-part 16-lane f32 result;
- one indexed q load, one indexed grid lookup, and one indexed sign lookup
  per 32-element issue;
- two `vslidedown` operations per issue to split grid and sign values before
  widening;
- four scalar metadata-byte loads per issue;
- final-IR register peak 30 groups.

The actual SG2044 assembly for the two-issue unrolled body contained six
`vluxei16.v`, four `vslidedown`, four vector stores, and a spill/reload of a
long-lived vector shift carrier.  Ten repetitions produced
432.365411 MElements/s, still below the fixed source value
443.356980 MElements/s.

The source donor in
`source/c/ggml/llama.cpp/ggml/src/ggml-quants.c:2416` instead preserves four
entry identities and processes each entry's eight contiguous payload values.
The SG2044 source object has no indexed vector gather and no slide in this
function.  Its physical program uses scalar entry selection followed by unit
payload loads.

This account ruled out another flat-tree LMUL or unroll experiment: LMUL=m2
was slower, LMUL=m8 exceeded the resource contract, and unroll=4 reduced the
three-repetition result to 214.568932 MElements/s.  The remaining discrepancy
was the missing logical entry/payload relation, not a parameter value.

## Reference mechanisms

Triton's coalescing path consumes axis contiguity and order facts when it
selects a memory layout; its general gather lowering does not recover a
hidden entry/payload decomposition.  TileLang similarly selects bulk/unit
copies from rectangular region, stride, and alignment facts; it does not
infer this radix relation from a flat arithmetic index.

Weft already had the corresponding typed physical mechanism:
`analyzeIndexedEntryRelation` and `RVVIndexedEntryLoadOp` can represent a
dynamic scalar entry base followed by a contiguous payload axis.  The missing
fact was therefore in the author program.  Adding a backend matcher or a new
physical op would have duplicated information the author already knows.

## Change

`examples/kernels/dequantize/iq2_xxs.py` now represents each 32-element group
as four eight-element entry sub-Levels.  The inner shaped value has one
explicit eight-element payload axis.  Metadata and scale remain at the
32-element group Level and are consumed by the four entry iterations.

The change preserves:

- the 256 output values and their coordinates;
- all lookup table bytes and indices;
- floating-point operation order and scale association;
- input/output encodings and C ABI.

It changes only the explicit Level/axis structure of the author's program.
No compiler pass or emitter branch was added.

The selected row-dequant parameter is `LMUL=m2` on both targets.  This lets
the eight f32 payload results remain in one carrier.  It is a measured
physical parameter binding in the runner, not a target-dependent source
tree.

## Observable physical result

For both targets, final RISC-V IR now contains two typed
`rvv_indexed_entry_load` operations per entry, each with selected `unit`
access.  One loads eight contiguous grid values and one loads eight
contiguous sign values.  There is no generic indexed lookup and no
`convert_layout` in this path.

The SG2044 final-IR register peak is 4 groups; K1/X60 is 2 groups.  Generated
C has, per entry:

```text
2 × vle8(payload=8)
1 × vector widen/multiply/convert/scale
1 × vse32(payload=8)
0 × vluxei
0 × vslidedown
```

The mathematical work remains 256 products per record.  The measured gain
comes from choosing the memory form and carrier made legal by the explicit
coordinates, not from deleting outputs or changing the numerical program.

## Targeted performance

All runs used the current worktree, the common fixed input seed, the same
source comparison bytes, and ten repetitions.  Both passed the numerical
tolerance check with zero reported absolute and relative error.

| target | previous Weft | current Weft | source | previous ratio | current ratio |
|---|---:|---:|---:|---:|---:|
| SG2044 | 345.542296 MEl/s | 692.394905 MEl/s | 443.356980 MEl/s | 77.94% | 156.17% |
| K1/X60 | 200.152757 MEl/s | 372.473967 MEl/s | 148.438660 MEl/s | 134.84% | 250.93% |

Standalone row-dequant is therefore above the fixed source implementation on
both targets.

## Mechanical acceptance

The changed entry was lowered independently for VLEN128 and VLEN256.  Both
physical IR files passed parsing, canonicalization, CSE, layout
canonicalization, two consecutive Share runs, final verification,
per-pass verification, and round-trip verification.  Running that complete
pipeline a second time produced a zero-line textual diff for both targets.

## Prediction account

The earlier family prediction assigned IQ2 primarily to P2/P4.  This result
does not revise that explanation: the observed blocker was an author-side
missing entry axis followed by P3 unit-payload selection and a smaller P1
carrier.  It remains part of the already recorded prediction MISS rather
than being relabeled after the result.

