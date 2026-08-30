# RISC-V physical owner 外部复验与 Q3_K/Q4_K 边界

本报告记录提交前的外部验证事实。验证范围是等价作者树、十一条既有高性能路径的 physical IR 稳定性、`weft-opt` 对最终 IR 的可观测性，以及 Q3_K/Q4_K 当前仍未闭合的具体边界。没有运行 206 条全量语料，也没有把一轮诊断数字写入正式性能 CSV。

## 1. 参考机制

动手和判断前对照了以下参考机制：

- Triton 的 `memdesc_subslice` 在真实 typed op 中保存 offsets、sizes 与 source memdesc；`local_load` 消费已经确定的 memdesc，而不是在 LLVM emitter 中从周围 extract 重新猜 subview。对应 `include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`、`lib/Dialect/TritonGPU/IR/Ops.cpp` 以及 `test/TritonGPU/memdesc-subview-split.mlir`。
- Triton `RemoveLayoutConversions` 以真实 conversion op、typed layout、backward slice 与 dominance 为输入做 rematerialization；它不从等价源码闭包恢复 layout。对应 `lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp`。
- TileLang reducer plan 从 update site 的 reduction axes 投影 partial layout，并要求多个 update site 得到 structurally equal plan。对应 `src/transform/reducer_plan_materialize.cc`。

Weft 不能复制 GPU thread/warp、shared memory 或 TMEM 对象；可复用的是“物理决定进入 typed IR，后续 pass 和 emitter 消费同一事实”的机制。

## 2. 强检验：等价 Q4_K 作者写法

固定相同参数：LMUL eighths=32、unroll=2、pipeline depth=1。两份作者程序分别是 `mac_pairs` 等价的 group=2 写法和 `mac_groups(4)` 写法。

| Target | group=2 GOP/s | group=4 GOP/s | correctness |
|---|---:|---:|---|
| SG2044 VLEN128 | 4.978166 | 10.053780 | both within tolerance |
| K1 VLEN256 | 1.485669 | 3.218627 | both within tolerance |

这些数字使用 1 repetition，只用于结构回归，不是正式性能结果。两条路径均生成 `rvv_grouped_mac_reduce` 与 `rvv_contract_step`；不同之处保留作者真实写下的 group=2/group=4 partial topology，没有 compile failure、fallback 或第二个 emitter 入口。

## 3. 回归检验：旧 HEAD 与当前 physical IR

用干净提交 `6b82de95b` 单独构建旧 `weft-compile`，再让旧、新编译器消费完全相同的 canonical IR、target facts、meta 与 physical 参数。覆盖 11 条路径、两个 target，共 22 份最终 physical IR：

```text
Q2_K, Q6_K, TQ2_0, Q5_K, IQ4_XS,
Q4_0, Q4_1, Q5_1, IQ4_NL, Q1_0,
Q4_K persistent
```

结果：

- 22/22 旧、新编译器均成功 lower；
- 20/22 的 physical op 种类与数量完全相同；
- 两个 Q4_K persistent 结果把 2 个未类型化的 extract 替换成 4 个 `rvv_replica_storage_load`；`rvv_grouped_mac_reduce`、`rvv_contract_step`、partial topology 数量不变；
- 22/22 生成 C 的 RVV intrinsic 名称与静态调用点计数完全相同。

因此本轮改变的是 storage record/part 事实的承载位置，不是悄悄换一套机器程序。Q4_K persistent 的真实 1-repetition 回归为 SG2044 10.148405 GOP/s、K1 3.217222 GOP/s，均数值正确。

## 4. 当前 typed owner

### 4.1 Grouped MAC layout

`LowerRISCVComposites` 以 result layout 为 anchor，显式产生 load layout 与 partial layout，并写入 grouped-MAC physical op。dialect verifier 检查 axes、time/lane/replica factors、SEW、LMUL 与 resource groups；emitter 只用这些 layout 拼 intrinsic 类型。

### 4.2 Replica storage record coordinate

`RVVReplicaStorageLoadOp` 不再保存一个含义不足的 `record_replica_for_window`。它显式保存：

- `record_rank`；
- 每个 window 的完整 record coordinates；
- `window_for_part`；
- 每个 part 的 layer、physical layer、shift 与 mask。

`ShareRISCVLayeredWindows` 根据 typed axes/layout 产生坐标并以 `(window offset, record coordinates)` 去重；verifier 对坐标数量和 record extent 做边界检查；emitter 只按坐标与 record byte stride 拼地址。

该改动解决了一个真实失败：仅保存扁平 replica key 时，record lane 宽度与 record coordinate 混在一起，Q3/Q4 的多轴 value 会被 verifier 判为越界。

## 5. `weft-opt` 可观测检验

以下代表性最终 IR 均通过：

```bash
weft-opt -verify-each -verify-roundtrip --canonicalize --cse INPUT.mlir
```

输入包括 Q4_K persistent、Q2_K、IQ4_XS、Q5_K 与 Q3_K。当前 Q2_K 的 `rvv_partial_set`、`rvv_partial_combine` 和 `rvv_partial_finalize` 在 CSE 后数量不变。

独立重放也暴露了三项尚未闭合的合同，未在本轮伪装成已解决：

1. generic canonicalizer 会把一个 `scf.if` 折成 `arith.select`，但最终 intrinsic-C emitter 没有 `arith.select` 合同；因此 canonicalizer 目前只能作为观测工具，不能直接插入 production terminal pipeline。
2. `PartialSetType` 仍不携带 independent partial 的 cohort/birth/lifetime identity；当前代表输入没有被 CSE 错并，不等于所有输入都安全。
3. `ime_fragment_mma` 同时声明 `Pure`，又允许 `memory_clobber=true`，trait/effect 语义仍矛盾。

## 6. Q3_K：当前结果与确定阻塞

当前统一树已经形成：

```text
rvv_replica_storage_load
→ rvv_partial_set
→ rvv_partial_repack(split=4)
→ rvv_partial_reduce(8 slots per 128-K half)
→ rvv_partial_scale_combine
→ rvv_partial_finalize
```

SG2044 当前 10-repetition 结果为 6.844190 GOP/s；同口径现场重测 source 为 7.178531 GOP/s，ratio 95.34%。K1 最近一次正式 10-repetition 结果为 2.633117 GOP/s，对 source 2.449247 为 107.51%。

VLEN128 donor 与当前程序都是每个 128-K half 八次 reduction，随后消费八个 scale；把 computed scale 强行提前到 widening partial 内不是缺失的唯一合法推导，而会改变中间值、结合位置与溢出边界。

本轮静态计数后做了两个未保留实验：

- 给四个 kernel pointer 加 `restrict`：SG2044 6.884192 GOP/s，与原代码无变化；
- 把两个连续 m4 activation load 合成一个 typed m8 load：静态少一次 load，但 SG2044 降到 2.675548 GOP/s。汇编出现 8 次 accumulator `vl1r.v` reload、vector spill 和反复 `csrr vlenb`。

剩余差距已经落到一个具体 compiler contract：storage operand 的 consumer-local materialization/subview 必须与 partial-set 的 resource/lifetime 联合表达。RVV intrinsic C 单独产生 m8 value 再 `vget` 两个 m4 子值会扩大 live range；donor 的局部 asm把 load、两个 subregister use 和 widening multiply固定在一个寄存器组织中。要复现该形态，需要 typed fused local operation 与对应的 RVV local-asm leaf；不能由 emitter 根据 Q3 名称临时合并，也不需要修改作者 logical tree。

## 7. Q4_K 三条作者路径

真实 1-repetition 结果：

| Path | SG2044 GOP/s | K1 GOP/s | 状态 |
|---|---:|---:|---|
| canonical base ABI, row×column | 4.782537 | 1.954918 | within tolerance |
| persistent derived encoding | 10.148405 | 3.217222 | within tolerance |
| canonical-input staged blocked | — | — | compile failure |

SG source prefill 为 9.672 GOP/s；因此 persistent path 已超过同口径 source。K1 source Q4_K 使用 IME1 与不同 persistent repack，不能与当前 RVV 数字制造直接 ratio。

canonical row×column 没有 output/K/M blocking，compiler 按 spec 不得把它猜成 persistent 或 blocked 程序。staged tree 已写出 NC/KC/MC、MR×16 accumulator 和 encoded materialization，但失败位置是：

```text
grouped/layered vector access reached emission without a typed physical window/stream operation
```

具体缺口是 encoded `[N,K]` local materialization 的二维 physical subview：固定 K projection 时，N cohort 映射到 SIMD lane；同一个 value 还必须保留 K-origin、record coordinate 与 local-pack stride。现有 `field → domain extract → index extract` 没有对应 typed local-pack/subview op。它和 Triton `memdesc_subslice → local_load` 是同一层问题，属于编译器结构性物理选择；作者已经通过 `materialize` 与 Level 写出了 lifetime，不应再写物理 pack 指令。
