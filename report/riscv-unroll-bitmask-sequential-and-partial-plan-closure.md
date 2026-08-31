# RISC-V Unroll Owner、Bitmask Decode、Sequential Partial 与剩余 Plan 收口

日期：2026-09-01  
实现提交：`d66aecd9e`、`f78e9940b`、`e841bd76d`、`2c1a26150`

## 1. 结论

本轮完成了四个可独立复现的节点：

1. partial planner 先于 generic unroll 冻结 issue program，nested partial 不再因为
   `unroll=2/4` 丢失 full-product carrier；但是 materialize 出来的仍是顺序复制的 issue chain，
   不是 donor 的跨 issue steady-state，实测没有收益。
2. Q1_0 row-dequantize 的 logical-u1 field 现在通过真实
   `rvv_bitmask_window_load` 进入 byte-aligned RVV carrier；双机均能生成、运行并通过容差检查。
3. TQ2_0 的 `36/32` 不是不可化解的 target 边界。旧 dot 把 34 groups 的临时 product 全部同时
   计入资源；显式 sequential program 改为
   `issue_slice → widen_accumulate → finalize` 后，只保留一个 loop-carried accumulator，双机
   standalone/decode 均超过 source。
4. nested、scaled/reduced-scaled 与 layered 的 planner/materializer 合同补齐。对所有当前可触发的
   代表入口，收口前后 final IR diff 为 0；Q4_K persistent 双机真机回归通过。当前固定语料没有
   触发完整 layered topology 的入口，因此 layered 只有类型、verifier 与构建证据，不能据此声称
   已有第二个性能正例。

## 2. 动手前的参考机制

这轮采用的是 reference 中已经存在的职责拆分，而不是把 unroll 或 partial 选择塞进 emitter：

- Triton 的 pipeline 先由 `AssignLatencies` / `ScheduleLoops` 决定 stage 与顺序，再由
  [`PipelineExpander.cpp`](../../ref/triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/PipelineExpander.cpp)
  机械生成 prologue、steady state 和 epilogue；expander 不重新决定 schedule。
- Triton 的 reduction lowering在 typed layout bases 已确定后，才由
  [`ReduceOpToLLVM.cpp`](../../ref/triton/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp)
  机械生成 local combine/tree reduction。
- TileLang 由
  [`pipeline_planning.cc`](../../ref/tilelang/src/transform/pipeline_planning.cc) 写 stage/order，
  [`inject_pipeline.cc`](../../ref/tilelang/src/transform/inject_pipeline.cc) 再做 buffer versioning 与展开；
  reducer 则由
  [`reducer_plan_materialize.cc`](../../ref/tilelang/src/transform/reducer_plan_materialize.cc)
  先固定 partial storage/steps，再物化 update/finalize。

Weft 与它们的 target operation 不同，但 owner 关系相同：planner 冻结 carrier、source-part map、
combine/final leaf 与资源，materializer 只能验证并实例化。generic loop rewrite 不得先改掉 planner
尚未观察的 program。

## 3. 第九节七步诊断得到的计数

### 3.1 IQ2_XS unroll

修改前，generic unroll 在 partial planning 之前复制 loop，`unroll=4` 会使 nested matcher 失效并
重新出现 sliced product 与 slide 拼装。修改后，同一 decode canonical entry 的 final IR 为：

| binding | final IR lines | `rvv_partial_set` | 结果 |
|---|---:|---:|---|
| unroll=1 | 188 | 1 | one planned issue chain |
| unroll=2 | 233 | 2 | carrier 保留，顺序复制两份 |
| unroll=4 | 316 | 4 | carrier 保留，issue loop 被机械展开 |

full-product carrier 不再退化，但 IR 中没有 stage/buffer version、prologue/steady/epilogue；静态动态
工作量没有减少，只把同一 issue program 顺序复制。因此第九节最后一问的答案是：这次改动值得
保留，因为它修正 owner 顺序并防止合法 carrier 被破坏；它不构成性能优化。

SG2044 10 次 cold median：

| unroll | standalone GOP/s | MUL_MAT decode GOP/s | correctness |
|---:|---:|---:|---|
| 1（正式 CSV） | 3.391627 | 3.121361 | within tolerance |
| 2 | 2.095853 | 0.551565 | within tolerance |
| 4 | 3.137960 | 0.542261 | within tolerance |

因此 donor 的 four-issue steady schedule 没有形成，SG decode 的 89.5% 没有提高。unroll=2/4 不写入
正式 CSV。这是尚未解决的 P5 cluster/pipeline 问题，不是再调 unroll 数字能解决的问题。

### 3.2 Q1_0 direct bitmask window

原 row-dequantize 在 logical u1 `grouped_layered` field 后没有 typed decode-side window，无法进入
terminal emission。第一次尝试直接让 4-lane carrier 读取任意 bit origin，在 element 5 发生错误：
RVV mask load 按 byte 起点读取，不能表达非 byte-aligned 任意 bit window。最终计划固定为
time16 × lane8 的 byte-aligned load carrier，再通过 typed conversion交给 consumer。

该 operation 删除的是每 consumer 重建的 scalar bit address/shift 链；第二个独立输入 Q5_0 的
logical-u1 high plane 也进入同一 typed window：

| entry | target | Weft | source | Weft/source |
|---|---|---:|---:|---:|
| Q1_0 row-dequantize | SG2044 | 239.948 MElements/s | 450.553 | 53.3% |
| Q1_0 row-dequantize | K1 | 585.780 MElements/s | 148.276 | 395.1% |
| Q5_0 row-dequantize | SG2044 | 981.358 MElements/s | — | — |
| Q5_0 row-dequantize | K1 | 173.124 MElements/s | — | — |

四条均为 10 repetitions 且数值 within tolerance。这个节点解决了“不能生成”；SG Q1_0 仍只有
source 的 53.3%，不能写成 row-dequant 性能已经收口。

### 3.3 TQ2_0 exact resource ledger

修改前不同 LMUL eighths `1/2/4/8/16/32` 的 peak 为
`132/68/36/36/38/42`。最低两档超过预算 4 groups；具体组成是 q operand 1 group、q8 operand
1 group，以及旧 composite dot 的 34-group product temporary。donor 每 issue 只保持一个 widened
accumulator，不需要 34 个独立临时 product。

新 `SequentialPartialPlanAttr` 冻结 issue count、lhs/rhs issue type、source parts、accumulator、
accumulate/final leaf 与同时存活资源。m4/m8 的 final IR 分别为：

| binding | issue slices | widened accumulate | final reduction | 旧 widen-dot |
|---|---:|---:|---:|---:|
| LMUL eighths=4 | 64 | 32 | 1 | 0 |
| LMUL eighths=8 | 32 | 16 | 1 | 0 |

真机 10 次结果：

| target | entry | Weft GOP/s | source GOP/s | Weft/source |
|---|---|---:|---:|---:|
| SG2044 | standalone | 9.994051 | 6.643537 | 150.4% |
| SG2044 | MUL_MAT decode | 9.918413 | 5.996033 | 165.4% |
| K1 | standalone | 6.250369 | 4.924574 | 126.9% |
| K1 | MUL_MAT decode | 6.242627 | 4.779641 | 130.6% |

standalone 与 decode 同步，说明收益进入共享 contraction 底座，而不是只进入 blocked prefill。

## 4. 剩余 materialization owner 收口

提交 `2c1a26150` 增加或补全以下 plan 合同：

- nested：issue/product/set types之外，冻结 reduce/finalize leaf，并验证 scale replica 是 signed i32；
- scaled/reduced-scaled：冻结 scale supply、narrow scale type、multiply/reduce/final leaf 与 resource；
- layered：冻结 lhs/rhs issue types、root window types/plans/instructions、decode instructions、accumulator、
  finalize leaf 与全部同时存活的 resource groups。

materializer 不再从当前 shape 或 defining-op spelling 重选上述事实。收口前后使用相同 target/meta
重新生成 final IR：

| entry/binding | representative path | diff lines |
|---|---|---:|
| IQ2_XS SG LMUL=m4 | reduced-scaled | 0 |
| TQ2_0 SG LMUL eighths=4 | sequential fused | 0 |
| TQ2_0 SG LMUL=m1 | sequential fused | 0 |
| Q3_K SG LMUL=m4 | computed-scale partial | 0 |
| Q6_K SG LMUL=m4, unroll=1 | scaled partial | 0 |
| Q5_0 standalone SG LMUL=m1 | sequential/per-stream | 0 |

`Q5_0` SG prefill 额外暴露了一个真实错误：planner 曾把带不同 free-axis replicas 的
outer-contract 选成 reduction-only `sequential_fused`，materializer 随后才报计划不闭合。现在
planner 在写 topology 前检查完整 issue/accumulator/source-part 合同；不满足时明确选择
`sequential_per_stream`，不是 materializer fallback。修后结果为：

```text
numeric=within-tolerance
max_absolute_error=0
max_relative_error=0
cold_gop_s=7.210352
source_gop_s=6.307497
ratio=1.143x
```

Q4_K persistent 的 nested path 双机 10 次回归：SG2044 `10.151793` GOP/s，K1
`3.218515` GOP/s，均数值 within tolerance。当前语料没有完整 layered topology 输入；它通过构建、
TableGen/verifier 与 `weft-opt` 合同，但没有真机动态证据。

## 5. 正反结果与未闭合边界

- 正：unroll 不再在 partial planning 之前破坏 full-product carrier。
- 负：unroll=2/4 仍未形成 steady-state，四 issue 只是顺序复制；paired decode 明显变慢。
- 正：Q1_0 与 Q5_0 两种 logical-u1 storage 都进入同一 typed bitmask window。
- 负：SG Q1_0 row-dequantize 仍为 source 的 53.3%。
- 正：TQ2_0 的 36/32 被证明是 composite temporary 表达错误，而非目标寄存器不足；显式
  sequential program 双机、standalone/decode 均过 source。
- 正：当前可触发的 nested/scaled/sequential 路径在 owner 收口前后 final IR 不变。
- 未证：完整 layered topology 没有当前 corpus 入口，不能据类型合同宣称泛化或性能。
- prediction ledger 本轮没有新增已决项，仍为 `0 HIT / 1 MISS / 5 pending`。

## 6. 手工复现

```bash
examples/run/weft-row-dequantize.sh sg2044 q1_0 10
examples/run/weft-row-dequantize.sh k1 q1_0 10

examples/run/weft-quantized-vec-dot.sh sg2044 tq2_0 10
examples/run/weft-mul-mat.sh sg2044 tq2_0 decode 10
examples/run/weft-quantized-vec-dot.sh k1 tq2_0 10
examples/run/weft-mul-mat.sh k1 tq2_0 decode 10

WEFT_AUTO_UNROLL=2 examples/run/weft-quantized-vec-dot.sh sg2044 iq2_xs 10
WEFT_AUTO_UNROLL=2 examples/run/weft-mul-mat.sh sg2044 iq2_xs decode 10
WEFT_AUTO_UNROLL=4 examples/run/weft-quantized-vec-dot.sh sg2044 iq2_xs 10
WEFT_AUTO_UNROLL=4 examples/run/weft-mul-mat.sh sg2044 iq2_xs decode 10

examples/run/weft-mul-mat.sh sg2044 q5_0 prefill 10
examples/run/weft-mul-mat.sh sg2044 q4_k_persistent prefill 10
examples/run/weft-mul-mat.sh k1 q4_k_persistent prefill 10
```

这些命令均由当前主链生成 intrinsic C，经相同 Clang 18.1.8 配置在目标机编译运行；没有增加
test 目录、fixture 或兼容路径。
