# RISC-V 联合 residency、供应 lifetime 与 physical contract 报告

## 1. 范围与结论

本轮没有铺新格式，也没有全量重测。实际改动集中在两条已知输入：

- Q3_K：让 typed replica storage supply 在其唯一 `rvv_partial_set` consumer 前就地发生，并给 partial root 加入显式 owner/birth/lifetime 身份；
- Q4_K staged：把 canonical encoded records 的 `materialize` 降为真实 local allocation、一次显式 interleave pack，以及从该 local view 读取的后续 physical program。

两条路径都发生了可见工作变化，但结果不支持“只要增加一个统一的大型联合 leaf 就能同时解决两者”的原假设：

- Q3_K 需要的是 consumer-local supply lifetime，而不是扩大 load。SG2044 的正式 10 次结果从 `6.895563` 提升到 `7.421357` GOP/s，并超过当前 source `7.218739`；K1 从 `2.633117` 到 `2.724853`，超过 source `2.449247`。
- Q4_K staged 需要真实 local pack。生成 C 中 `vlse8` 从历史 artifact 的 SG `114` 次降为 `0`，但当前 10 次结果只有 SG `3.508368`、K1 `2.002277` GOP/s。相对旧 staged 的一轮结果 `1.618410 / 1.376037` 分别提高约 `116.8% / 45.5%`，但 SG 仍只有当前 source `9.699175` 的 `36.17%`。

因此，本轮闭合了 explicit `materialize` 的 residency owner 和 partial root 的 supply lifetime；没有闭合任意 multi-consumer producer 的通用 P2 owner。后者仍是明确的编译器缺口，不能把本轮结果描述成 P2 已整体完成。

## 2. 动手前的七步诊断

本轮按 `doc/compiler/optimization-principles.md` 第九节执行，而不是从 pass 名称出发。

### 2.1 Q3_K

1. final IR 的 reduction axis 在 lane，free axis 在 register replica；旧正式配置是 LMUL m4、MR1×NR1。
2. work ledger 显示 Q8 storage load 和 `rvv_partial_set` 是两个 SSA 阶段。
3. m8 的静态目标是把两个较窄 supply 合并成一个较宽 supply；历史实测却从 `6.691` 降到 `2.679` GOP/s，并产生 vector spill。
4. 该改动不改变 logical value、Level、widening 或 artifact，归 physical compiler。
5. Triton 的 allocation 分析把 alias view 的 lifetime 合并到同一 allocation；但 Q3 没有 local allocation，不能照搬 shared-memory allocation。这里可复用的是“consumer 使用真实 SSA supply，并由 liveness 观察其区间”，不是 GPU memory space。
6. 静态收益应表现为 supply 与唯一 consumer 邻接、supplier interval 缩短；若扩大 LMUL 后 peak 或动态工作不降，就不采用更宽 load。
7. 修改后 m4/m8 final IR 的 register peak 都是 `17` groups，说明旧的提前存活问题消失；但 m8 的 SG/K1 一次结果为 `4.354518 / 2.723967`，m4 为 `7.406261 / 2.725192`。所以保留 m4，拒绝“更宽一定更快”的预测。

### 2.2 Q4_K staged

1. 作者树已有 NC/KC/MC、MR×16 accumulator、K block 和 `materialize` lifetime。
2. 旧 final program 对 canonical record 的每次使用重新做 strided byte load；历史 SG/K1 assembly 分别出现 `114 / 47` 条 `vlse8`。
3. 若 local pack 有效，静态计数应出现一个 allocation、一个 pack，热计算中的 `vlse8` 消失。
4. local pack 不改变 logical value、Level 或跨调用 artifact，归 physical compiler；persistent `Q4K_I16` 改 ABI，仍归作者/caller。
5. Triton 的 `MemDescType` 同时保存 shape、encoding、memory space 与 allocation shape，`local_alloc` 是新 storage 的唯一根，view 只传播 alias identity；Allocation analysis 再合并 alias liveness。TileLang 则以 BufferRegion 的范围/stride 和显式 allocation lifetime 建立同类事实。Weft 不能照搬 CTA/shared-memory ownership，但需要同样真实的 allocation、view 与 lifetime，而不是 side record。
6. local pack 增加固定 `18432` bytes local storage；收益必须在 load form 和动态 load 数上出现，否则不值得实现。
7. 当前 final IR 有一个 `local_alloc<18432>` 和一个 `encoded_local_pack<interleave rows=16, record=144 bytes/256 elements>`；生成 C 有 `0` 条 `vlse8`、`87` 条 `vle8`。双机性能均提高，因此 P2/P3 预测得到部分支持；但没有达到 source。

## 3. 参考实现与 Weft 的落点

Triton 的关键机制不是“给 local storage 起一个名字”：

- `TritonGPUTypes.td:23-40` 的 `MemDescType` 把 allocation shape、view shape、encoding 和 memory space 放在真实 type 中；
- `Alias.cpp:23-63` 规定只有 `LocalAllocOp` 创建新 buffer，带 `MemDescViewTrait` 的 op 传播同一 alias identity；
- `Allocation.cpp:303-379` 从真实 SSA value 收集 alias，并把所有 alias view 的 liveness 合并到 allocation。

TileLang 的 reducer planning 同样先遍历全部 update site，要求它们产生结构一致的 plan，再物化 partial storage；shared allocation 则以 linear-scan interval 分配和回收 arena。它没有把 lifetime 留给 terminal printer 猜。

Weft 的非 SIMT abstract machine 没有预先存在的 thread/warp owner，因此不能直接复用 Triton 的 ownership 或 TileLang 的 shared arena。这里采用的对应关系是：

```text
canonical materialize birth/lifetime
→ typed residency plan
→ LocalType allocation + EncodedLocalPackOp
→ alias-preserving local record binding
→ exact memory/window operations
```

对于 Q3，则是：

```text
typed replica storage load
→ unique RVVPartialSet consumer
→ consumer-local issue placement
→ partial owner/birth/lifetime identity
→ resource pass 观察最终 SSA interval
```

没有把两者硬塞成一个格式无关但不减少工作的“大 op”。

## 4. 实现事实

### 4.1 Explicit materialize 的 residency owner

`PlanRISCVResidency.cpp` 现在集中决定 unresolved `MaterializeOp` 的三种结果：

- encoded、两轴、所有真实 consumer 需要同一 output cohort：产生 `LocalPackPlanAttr`，placement 为 `local`；
- 已选 local carrier 的普通数值 value：普通 local materialize；
- dense load-backed、无法寄存器投影的值：`reload`；
- 其他显式 materialize 保留 shared register materialization。

该逻辑在 `PropagateRISCVLayouts` 的末尾调用，而没有单独再放一个 pass。负实验表明把它移到下一个 pass 会使 layout invariant 在 pass 边界先被 verifier 判为未闭合，Q2_K 报 `local materialize requires one closed RVV-to-local value mapping`。决策代码仍独立，但它必须在 representation propagation 的同一原子边界内闭合。

`LocalPackPlanAttr` 保存 row axis、record axis、interleave rows、record bytes、record elements 和 local bytes。`EncodedLocalPackOp` 消费 encoded value、typed local allocation 和两个 logical points，产生 local encoded value。verifier 现在强制：

- allocation bytes 精确等于完整 row×record 几何，不允许 OOB 或未初始化尾部；
- 乘法全部做溢出检查；
- dynamic record active 必须是完整 record。由于 K Level 可以是 runtime tail，生成 C 在除法前显式 `__builtin_trap()` 非整 record，而不是静默向下取整。

### 4.2 Partial supply lifetime

每个 `RVVPartialSetOp` / `RVVPartialCaptureOp` 现在带 `owner_domain_id`、`birth_id` 和 `lifetime_end_domain_id`。birth 是 producer identity，不是 reusable topology type 的组成部分，因此放在 op 而不是 `PartialSetType`；标准 CSE 会比较这些 attrs，final verifier 同时拒绝重复 `(owner,birth)`。

`MaterializeRISCVPartialAccumulators` 对单 use、同 block、仅跨 pure/read operation 的 `RVVReplicaStorageLoadOp` 做 consumer-local sink。它不跨任何 write 或 unknown effect。Q3 final IR 中 load 与 partial set 已直接邻接。

在 Q3 final IR 上运行 `weft-opt --canonicalize --cse --verify-each --verify-roundtrip` 后：

- partial root 仍为 1；
- set→repack→reduce→scale-combine→finalize 五节点链保持；
- 没有 independent partial 被 CSE 合并。

### 4.3 Effect 与 terminal contract

- `ime_fragment_mma` 不再声明 `Pure`，而是显式 `MemRead + MemWrite`；这与实际 volatile asm 和 `memory` clobber 一致。
- generic canonicalizer 在 Q3 physical IR 中产生一个 scalar `arith.select`。intrinsic-C terminal 现在只接受 scalar condition/scalar values，并机械发射 C ternary；shaped、fragment 和 local select 仍明确 unsupported。

### 4.4 零移位

本轮没有增加新的 zero-shift op。当前 dynamic shift 由 logical offset 决定；把同一公式换一个 op 名不会减少 shift、地址计算或 lifetime，无法回答“若计数不变为什么值得实现”，因此按 P0 不做。

## 5. 真实结果

统一 shape 为 `M=128,N=4096,K=4096`，两端使用 Clang 18.1.8、`-O3 -ffp-contract=fast`。Q3 为当前 CSV 的 10 次正式结果；Q4 staged 不是 production manifest 行，因此只记录在本报告。

| 入口 | Target | Reps | Weft GOP/s | source GOP/s | ratio | numeric |
|---|---|---:|---:|---:|---:|---|
| Q3_K prefill | SG2044 VLEN128 | 10 | 7.421357 | 7.218739 | 1.0281 | within tolerance |
| Q3_K prefill | K1 VLEN256 | 10 | 2.724853 | 2.449247 | 1.1125 | within tolerance |
| Q4_K staged prefill | SG2044 VLEN128 | 10 | 3.508368 | 9.699175 | 0.3617 | exact in this run |
| Q4_K staged prefill | K1 VLEN256 | 10 | 2.002277 | IME1 baseline | 不可直接比较 | exact in this run |

最终 verifier 加固后又各执行一次双机 repro：Q3 SG/K1 为 `7.389115 / 2.722405`，Q4 staged SG/K1 为 `3.481425 / 2.002910`，均通过数值检查。这些一次结果只作回归，不写 CSV。

此前已过线的 11 条路径也各跑过一次双机回归，全部在容差内：

| Target | Q2_K | Q6_K | TQ2_0 | Q5_K | IQ4_XS | Q4_0 | Q4_1 | Q5_1 | IQ4_NL | Q1_0 | Q4_K persistent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SG | 9.850 | 9.193 | 20.144 | 7.745 | 7.324 | 10.438 | 6.720 | 6.284 | 7.818 | 8.755 | 10.158 |
| K1 | 2.743 | 2.633 | 6.535 | 2.461 | 2.439 | 2.245 | 2.007 | 1.864 | 2.380 | 3.716 | 3.229 |

这些是 1 repetition 回归值，不替换正式 CSV。

## 6. 负结果

| 实验 | 结果 | 处理 |
|---|---|---|
| Q3 supply 扩大到 m8 | SG `4.354518`，显著慢于 m4 `7.406261`；K1 基本相同 | 不采用；说明 lifetime 闭合不等于宽载体有收益 |
| residency 作为独立后续 pass | Q2_K 在 pass 边界出现 local/layout invariant 未闭合 | 撤销独立 pass；planner 在 Propagate 原子边界内调用 |
| production pipeline 直接加入 generic CSE | 该尝试与独立 residency pass 同期，失败原因无法单独归因 | 不保留；只用 `weft-opt` 独立重放证明 partial identity |
| 把 Q4 runtime tail 强制成静态 exact | 合法 K=4096 程序在 physical verifier 被误拒 | 不采用；改为 typed whole-record contract 加运行时 trap |
| 为 dynamic zero shift 再造 typed op | 静态预测删除 0 条动态工作 | 按 P0 未实现 |

## 7. 预测表对答案

`optimization-principles.md` 对 Q4_K canonical 的预测是：作者先给出 blocked/local-materialize tree，随后主要看 P2/P3/P5。

本轮 staged tree 对答案：

- P2/P3：命中。真实 local pack 让 `vlse8` 归零，SG/K1 分别提高约 `2.17× / 1.46×`。
- P5：未验证。本轮没有修改 pipeline。
- “blocked tree 已经等同 persistent 数值树”：不成立。当前 staged tree在 `mul_mat.py:718-727` 先做 i32 `outer_contract` 再乘 scale；persistent tree在 `:687-692` 先做 `mac_groups(... into=i16)`，再 widen/reduce/scale。两者的 logical partial、widening 和结合位置不同。

因此该预测记为“主要类别部分命中”，不能把剩余约 `2.76×` SG 差距继续全归到 local residency。若要改 staged 数值树，按 2.2 必须由作者决定。

目前只回填了一个入口，尚不足以触发“错误超过一半则撤销整套方法”的证伪条件。

## 8. 尚未闭合的边界

1. **通用 P2 owner 未完成。** `PlanRISCVResidency` 只统一 explicit `MaterializeOp` placement；arbitrary multi-consumer producer 的 rematerialize/share/hoist 仍分别存在于 layout canonicalization、typed layered-window sharing 和 memory planning 中。本轮没有用 IQ1/IQ2/IQ3 证明它。
2. **Q4 staged 未过 source。** local pack 已真实减少工作，但当前作者数值树与 persistent donor 不同；P5 也未验证。
3. **local pack 仍是 scalar copy。** 它已经把热计算改成 unit-load local view，但 pack 本身没有 vectorized transfer。当前数据不能证明剩余差距中 pack、partial topology 和 pipeline 各占多少。
4. **canonicalized physical IR 无法直接重新交给 `weft-compile` 发 C。** `weft-opt` 已证明 parse/verifier/CSE 合同，terminal 的 scalar `arith.select` 分支由正常 compile path 覆盖；本轮没有把 canonicalized physical module 重新 runtime 发射，因为 driver 只接受 canonical Kernel IR。

