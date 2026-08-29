# RISC-V 物理决策 owner 收敛、physical IR 重放与 Q3_K 结果

本报告记录当前提交状态中的事实。验证范围是 physical decision owner、两种等价 Q4_K 作者写法、`weft-opt` physical dialect 重放，以及 Q3_K production MUL_MAT 的双机结果。没有进行 206 条全量重测。

## 1. 参考实现先行得到的机制边界

动手前核对了 Triton 与 TileLang 中对应机制。

Triton 的 `RemoveLayoutConversions` 以真实 conversion op 和 typed layout 为输入，沿当前 use 取得 backward slice；只有 slice 中的 op 可重物化、支配关系成立且代价允许时才重写 producer，而不是由 terminal emitter 从 operand 数量或源码邻接恢复布局。对应实现在：

- `/home/kingdom/phdworks/ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp` 的 backward-slice、dominance 与 rematerialization 路径；
- `/home/kingdom/phdworks/ref/triton/lib/Dialect/TritonNvidiaGPU/Transforms/OptimizeTMemLayouts.cpp` 的 typed subslice/load 重写；
- `/home/kingdom/phdworks/ref/triton/lib/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.cpp` 中通过 consumer-local rematerialization 缩短 live range 的逻辑。

TileLang 的 reducer plan 从 reduction update site 投影 partial layout，多个 update site 必须得到 structurally equal 的 plan；pipeline scheduler 与 expander分别决定 stage 和机械展开。对应实现在：

- `/home/kingdom/phdworks/ref/tilelang/src/transform/reducer_plan_materialize.cc`；
- `/home/kingdom/phdworks/ref/tilelang/src/transform/inject_software_pipeline.cc`。

Weft 不能直接复用 GPU thread/warp、shared memory、TMEM descriptor 或 barrier 语义；可以复用的机制是：决定进入 typed IR，后续 pass 消费同一份事实，conversion/subview 是真实 op，terminal emission 不重新选择。

## 2. 四处 owner 重叠的当前状态

### 2.1 Partial topology

旧的 `groupedLanePartials`、`plannedReplicaLanePartials` 和 `stream_reduction` 已从当前 `lib/`、`include/` 中消失。

当前唯一入口是 `PlanRISCVPartialTopologiesPass`。它根据 typed operands、reduction/free axes、layered root、slot 数和资源合同，写入：

- `partial_topology`；
- `partial_layout_plan`；
- `layered_partial_plan`；
- 各阶段的 `PartialSetType`。

`MaterializeRISCVPartialAccumulatorsPass` 只读取这些属性来创建 `rvv_partial_set`、`rvv_partial_repack`、`rvv_partial_reduce`、`rvv_partial_scale_combine` 和 finalize op；它不再从 lane/time/replica factors 重建另一套 topology。

静态检索：

```text
groupedLanePartials       0
plannedReplicaLanePartials 0
stream_reduction          0
```

### 2.2 Layout 与 partial materialization

`PropagateRISCVLayoutsPass` 是 value 的 time/lane/register-replica/fragment/local factors 的 owner。`PlanRISCVPartialTopologiesPass` 只把已确定的 value layout 投影成 partial set 的 typed contract；materializer 不再重新决定哪个 free/reduction axis 进 lane 或 replica。

这一区分在最终 IR 中可观察：value 的 `!weft_riscv.value<..., layout<...>>` 先存在，partial op 再引用 `PartialLayoutPlanAttr` 与 `PartialSetType`。没有 side-record 门控参与二选一。

### 2.3 Layered/projected geometry

`ShareRISCVLayeredWindowsPass` 现在生成 typed geometry：

- `StorageWindowPlanAttr`；
- `LayeredStreamGeometryAttr`；
- `LayeredPartialPlanAttr.rootStoragePlans`；
- `RVVLayeredWindowOp` 的 physical layer、shift、mask 数组；
- `RVVReplicaStorageLoadOp` 的 window、record replica、shift 与 mask 数组。

emitter 不再从 group/layer/stream axis 重建窗口，不再生成自己的 raw-window cache key，也不再根据 encoding order 私选 physical layer。它按 geometry 中的 `windowForStream`、`representativeStreamForWindow`、`shiftAmountForStream`、`maskValueForStream` 机械拼写 load/shift/mask。

### 2.4 零移位

旧的 `logical_base_multiple` 属性及 emitter 中 `% group` 的 constant-layer 判定已删除。是否需要 shift 由 pass 写入 typed shift amount 或由 pass 选定 `identity/mask/shift/shift-mask` leaf：

- `shift_amount == 0` 的 layered decode 选择 identity 或 mask leaf；
- `RVVReplicaStorageLoadOp` 直接携带每个 part 的 `shift_offset` 与 `shift_base_factor`；
- emitter 只在 typed 值非零时拼写 shift。

静态检索：

```text
logical_base_multiple 0
LogicalBaseMultiple   0
```

`RVVStorageWindowOp` 仍会机械写出 plan 中的动态 shift 公式；该公式是 `StorageWindowPlanAttr` 的显式结果，不是 emitter 自行选择 physical layer。它与已删除的 pass/emitter 双 owner 不是同一条路径。

## 3. 等价作者写法的强检验

检验了：

- `examples/kernels/quantization/q4_k_gemv.py`：`mac_groups(group=2)`，即原 mac-pairs 分解；
- `examples/kernels/quantization/q4_k_gemv_groups4.py`：`mac_groups(group=4)`。

两份 canonical IR 均能 lower 到 RISC-V physical IR，且都通过：

```bash
build/tools/weft-opt/weft-opt -verify-each -verify-roundtrip INPUT.mlir
```

两份最终 physical IR 的外围统计一致：

```text
weft_riscv.kernel                 1
weft_riscv.level                  3
weft_riscv.rvv_grouped_mac_reduce 1
weft_riscv.rvv_contract_step      1
weft_riscv.convert_layout         3
```

差异只保留作者真实写下的 group：group=2 的 `GroupedMacPlanAttr` 有 `[0,1]`，group=4 有 `[0,1,2,3]`。两者进入同一种 `rvv_grouped_mac_reduce` physical op 和同一种 RVV contract leaf；不存在 non-scalar lane-remap compile failure，也没有为第二种写法增加 matcher 或 emitter 分支。

## 4. `weft-opt` physical dialect 重放

`tools/weft-opt` 现在链接并注册 `WeftRISCVDialect`。实测：

```text
Available Dialects: arith,builtin,scf,weft_kernel,weft_riscv
```

canonical IR 与最终 RISC-V IR 均可独立 parse、dump、执行 verifier 和 round-trip。非法顶层 physical terminator 会由 dialect verifier 报错，不再需要通过真机运行间接发现 parser/verifier 缺口。

注册之后确认的两个语义缺口没有在本轮伪装成已解决：

- `PartialSetType` 仍没有 cohort owner/birth/lifetime，因此通用 CSE 不能仅凭当前 type 判断两个 independent partial 是否可合并；
- `ime_fragment_mma` 当前同时声明 `[Pure]` 与 `memory_clobber=true`，trait/effect 合同矛盾。

本轮没有启用可能错误改写 partial lifetime 的通用 CSE，也没有修改 IME effect 语义。

## 5. Q3_K 的实现与双机结果

### 5.1 有效改动

Q3 scale storage 是一条 typed relation：

```text
u4[16] grouped(16)/layered(8)
+ u2[16] grouped(16)/layered(4) << 4
```

新增 `PackedPlaneMergePlanAttr` 与 `PackedPlaneMergeOp` 后，physical IR 显式携带 logical axis、16 个 logical elements、low/high bits、两个 layer byte widths、两个 byte offsets 与 insert bit。`FuseRISCVBitplanesPass` 依据这些 typed geometry 形成 op；不读取 Q3_K 名称、kernel 名称或固定 op 数。

当前 scalar-word leaf 对每个 record 读取 3 个 32-bit packed word，形成 4 个 merged word，并把 16 个 logical scale 作为 scalar tuple 交给 partial topology。旧的逐 scale encoded access 已消失。

### 5.2 真实结果

统一命令：

```bash
bash examples/run/weft-mul-mat.sh sg2044 q3_k prefill 10
bash examples/run/weft-mul-mat.sh k1 q3_k prefill 10
```

统一口径：M=128、N=4096、K=4096；kernel 内含 activation quantize；Clang 18.1.8；`-O3 -ffp-contract=fast`；NC=32、MC=8、MR=1、NR=1、LMUL=m4、unroll=1、pipeline=1。

| Target | Weft GOP/s | source GOP/s | ratio | correctness |
|---|---:|---:|---:|---|
| SG2044 VLEN128 | 6.895563 | 7.218739 | 95.52% | within tolerance; max abs 0.01171875; max rel 2.00336409e-05 |
| K1 VLEN256 | 2.633117 | 2.449247 | 107.51% | within tolerance; max abs 0.01171875; max rel 2.00336409e-05 |

K1 已超过 source；SG 没有达到本轮“超过 source”的目标，差 4.48%。CSV 已如实更新，没有把 3-repetition 或负实验数字写入正式行。

## 6. 负实验及其静态/汇编证据

以下改动均未保留在仓库中。

| 实验 | 静态预期 | SG2044 结果 | 汇编事实 | 结论 |
|---|---|---:|---|---|
| computed shaped scale 强制锚到 RVV | 避免 scalar scale chain | 5.864 GOP/s | scale carrier 扩大 live set | 撤回 |
| 16-byte local array scale leaf | 用一次 decode 后的 byte loads 代替 GPR shift | 5.983 GOP/s | store/reload 和栈生命周期抵消收益 | 撤回 |
| direct-GPR scale asm leaf | 12 byte loads/约 45 条 scalar decode 缩短 | 6.428 GOP/s | asm clobber 导致 vector spill 与动态栈 | 撤回 |
| 一次 RVV load 后提取 3 个 scale words | 12 byte scalar load 变成一条 vector load | 6.691 GOP/s | 新增 VL 切换、slide 与 scalar extract | 撤回 |
| 完整 RVV gather scale decode | 直接形成 16 个 logical scale | 4.693 GOP/s | gather/shift/index 构造超过 scalar-word leaf | 撤回 |
| 两个 Q8 m4 load 合成一个独立 m8 value | 每 half 少一次 load | 2.679 GOP/s | 3 次 vector spill store、12 次 reload | 撤回 |
| 把 m8 load 移到两次 vwmul 前 | 缩短 m8 live range | 6.312 GOP/s | 每个 half 仍出现一组 `vs4r/vl4r` spill/reload | 撤回 |
| MR=2 | scale decode 跨两个 M 输出复用 | 5.901 GOP/s | live partial/operand 资源上升 | 不采用 |
| NR=2, LMUL=m4 | activation 跨两个 N 输出复用 | compile invalid | 峰值 36 vector groups，预算 32 | 不采用 |
| NR=2, LMUL=m2 | 降低资源后运行 | 4.774 GOP/s | 更多 time decomposition 抵消共享 | 不采用 |
| scale 在 i32 中直接减 32 | 每 scale 的 sign/bias 路径从 3 条降到 2 条 | 6.921 GOP/s | 总体无可复现提升 | 撤回 |

这些负结果把 SG 的剩余差距收窄到一个具体边界：当前 physical IR 把 Q8 storage load 与 partial-set consumption 表成两个独立 SSA 阶段。单独扩大 load 会使 m8 value 与 q operand/partial 同时存活，Clang 必须 spill；把 C 语句移动到 consumer 前也不能消除这个重叠。source donor 把 load、subview 与两次 widened multiply 固定在一个局部寄存器组织中。

因此尚缺的不是另一个 emitter 条件，而是一个 compiler-side typed contract：storage operand 的 consumer-local materialization/subview 必须和 partial set 的 resource/lifetime 一起表达，使 allocator 看见联合 leaf 的输入、临时和输出。没有这项 contract 时，emitter 私自把两个 load 合成 m8 会复现已经测到的 spill。该缺口不改变作者 logical value 或 Level，归 physical compiler；本轮没有用 Q3 名称分支或 whole-kernel asm 掩盖它。

## 7. 受影响路径回归

Q2_K、Q5_K、Q6_K、IQ4_XS、TQ2_0 的 SG2044/K1 production prefill 均重新执行真实 repro，十条均为 `numeric=within-tolerance`。这批回归使用 1 repetition，只用于确认 owner 收敛没有破坏已过线路径，不作为新的正式性能记录写入 CSV。

本轮最终构建命令：

```bash
cmake --build build --target weft-compile weft-opt -j2
```

最终静态检查：

```text
git diff --check: pass
old partial-owner symbols: 0
old coalesced-load experimental getters: 0
weft-opt physical IR verify/roundtrip: pass
Q4_K group=2/group=4 lowering: pass
```
