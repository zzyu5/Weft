# Q2_K regular-repeat gather 与 independent-partial topology

日期：2026-08-28

## 结论

Q2_K production prefill 在当前唯一 RISC-V 主链上已经超过同目标 source baseline：

| target | 本轮前 Weft | 本轮 Weft | source | 本轮/source | 相对本轮前 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SG2044 / VLEN128 | 4.845207 | 9.969884 GOP/s | 9.462 | 105.4% | +105.8% |
| K1/X60 / VLEN256 | 1.659511 | 2.508207 GOP/s | 2.410 | 104.1% | +51.1% |

两条正式结果均为 production shape `M=128,N=4096,K=4096`、10 次重复、Clang 18、`-O3 -ffp-contract=fast`，并通过现有浮点容差检查：max absolute error `3.57627869e-07`，max relative error `1.65838598e-05`。

性能变化来自三项相互闭合的 physical program：

1. regular-repeat encoded access 被 lowering 成 typed source load、typed index 和 typed gather；
2. 一个 contraction 的多个 reduction-time partial 被表示为 typed `partial_set`；
3. partial 的 pairwise combine、scale combine、final reduction 与 output replica assembly 成为真实 physical op。

K1 最终汇编中有 8 次 `vrgather`、0 次 `vslideup`；原先的 `vwmul + 3×vwmacc` 串行链变为独立 product、vector add 和 staged reduction。SG 每次 physical issue 同时形成 4 个 scale partial，再按两个 2-term scale chain 合并；不再对每个 16-element sub-Level 立即结束 reduction 后更新 scalar carry。

## 1. 起点：缺的不是一个参数

### 1.1 K1 regular-repeat

本轮前，K1 的 32-lane scale value 由 terminal emitter 临时构造：

```text
scalar load
→ scalar broadcast
→ second scalar broadcast
→ vslideup
```

同一个 Q2_K block 共出现 8 次 `vslideup`。source donor 在
[`quants.c:875-909`](../source/c/ggml/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c#L875) 对应的是：先载入 16 个 scale，再以 4 次 gather 构造每个 half 的 4 个 32-lane scale vector。

旧 physical IR 只保存 `regular-repeat` access facts；是 emitter 根据 `repeat > 1` 决定 broadcast/slide。这使 memory/layout 选择没有成为可验证的程序实体。

### 1.2 K1 partial

本轮前一个 half 的 reduction 形态是：

```text
vwmul(first plane)
→ vwmacc(second plane)
→ vwmacc(third plane)
→ vwmacc(fourth plane)
→ one final reduction
```

source donor 是 4 个独立 widened product、pairwise vector add、再做 staged reduction。两者不改变 canonical contraction 的 logical values 或 Level 归属，差异属于 contraction 的物理 partial topology。

### 1.3 SG partial

SG 的 `group_reduced` 作者树已经明确表达：

```text
reduce(q2 * q8) * scale
```

但旧 lowering 对每个 16-element sub-Level 都立即结束 reduction，再更新 scalar carry。`unroll=8` 只写进 operation-local schedule，没有产生 8-partial register program，因此此前从 1 改到 8 没有性能变化。

## 2. 与 Triton、TileLang 的机制对照

本轮采用的是它们对 reduction distribution 的组织方式，不复制 GPU 的 thread 或 shared-memory 模型。

### 2.1 Triton

[`ReduceOpToLLVM.cpp:230-351`](../../ref/triton/lib/Conversion/TritonGPUToLLVM/ReduceOpToLLVM.cpp) 先从 operand layout 得到 register bases、lane bases 与被消去的 reduction axis，再按 logical slot 组织 thread-local/lane partial 并 tree-reduce。关键机制是：

- reduction axis 的物理分布先存在于 typed layout；
- partial 按 logical slot 存在，而不是 emitter 看见若干 multiply 后临时拼链；
- 最终 reduction topology 由已确定的 bases/layout 生成。

这对应 Weft 的 `PartialSetType`：一个 slot 保存一份完整 lane partial，slot 数与 terms-per-slot 由 reduction-time decomposition 给出，free axes 由 output replica 单独投影。

### 2.2 TileLang

[`reducer_plan_materialize.cc:559-733`](../../ref/tilelang/src/transform/reducer_plan_materialize.cc) 从 update site 的 reduction axes 投影 partial storage；多个 update site 只有得到 structurally equal plan 才能共享同一 reducer plan。

Weft 的 SG scale-combine 路径采用同一约束：每个被合并 contribution 必须具有同一 reduction axis、同一 slot type、同一 time decomposition 和可投影的 output replica；不满足时不建立 partial set。

## 3. Typed regular-repeat memory program

### 3.1 Physical IR

[`RISCVOps.td:913-936`](../include/Weft/Dialect/RISCV/IR/RISCVOps.td) 新增：

```text
weft_riscv.rvv_regular_repeat_index
weft_riscv.rvv_regular_repeat_gather
```

index op 保存：

- reduction axis；
- repeat；
- 每个 result part 的相对 base；
- 已选 index value layout；
- 已选 `pow2` 或 `div` index leaf。

gather op 保存：

- typed encoded Field；
- typed index operands；
- source base/count；
- repeat 与 part bases；
- per-use access relation；
- 已选 broadcast/gather leaf；
- 每个 result 的完整 axis/layout。

verifier 位于 [`RISCVDialect.cpp:3034-3155`](../lib/Dialect/RISCV/IR/RISCVDialect.cpp)，检查 Field/access、result axes、index SEW/LMUL/VL、part bases、repeat 和 leaf 是否闭合。

### 3.2 Planning 与 emission

[`PlanRISCVMemory.cpp:650-736`](../lib/Target/PlanRISCVMemory.cpp) 从同一 encoded Field、reduction axis、source interval 和 regular-repeat geometry 形成一次 source load，并选择：

```text
lanes <= repeat      → regular-repeat-broadcast
repeat 为 2 的幂    → regular-repeat-gather.pow2
其他合法 repeat     → regular-repeat-gather.div
```

该选择写入 physical op 的 leaf。terminal emitter
[`RISCVIntrinsicC.cpp:7784-7928`](../lib/Target/RISCVIntrinsicC.cpp) 只按 leaf 拼写 index 与 `vrgather`，不再从 repeat/shape 重新选择 slide、broadcast 或 gather。

K1 最终 physical IR 中，每个 128-element half 各出现：

```text
rvv_regular_repeat_index
rvv_regular_repeat_gather
```

当前临时汇编统计为：

| instruction | count |
| --- | ---: |
| `vrgather` | 8 |
| `vslideup` | 0 |

## 4. Typed independent partial 与 combine topology

### 4.1 Type 与 operations

[`RISCVOps.td:360-371`](../include/Weft/Dialect/RISCV/IR/RISCVOps.td) 新增 `PartialSetType`，保存：

```text
partial ValueType
reduction axis
independent slot count
terms per slot
resource groups
```

[`RISCVOps.td:998-1057`](../include/Weft/Dialect/RISCV/IR/RISCVOps.td) 新增：

```text
rvv_partial_set
rvv_partial_reduce
rvv_partial_scale_combine
rvv_partial_combine
rvv_partial_finalize
rvv_assemble_replicas
```

`rvv_partial_set` 还保存每个 operand 的 output-replica projection 与已选 multiply instruction。unsigned logical values 以 physical SEW 保存时，reinterpret 关系同样在 leaf 中确定，emitter 不再根据逻辑位宽临时换 `vwmul/vwmulsu`。

verifier 位于 [`RISCVDialect.cpp:3566-3778`](../lib/Dialect/RISCV/IR/RISCVDialect.cpp)：它检查 reduction axis、lane/time topology、lhs/rhs replica、multiply instruction、slot/resource 关系、combine arity 和 final result type。

### 4.2 K1 的 independent topology

[`MaterializeRISCVPartialAccumulators.cpp:902-1031`](../lib/Target/MaterializeRISCVPartialAccumulators.cpp) 对 reduction-time decomposition 应用固定结构规则：

- slot 数至少为 4，且为 2 的幂；
- lhs/rhs 具有相同 reduction-time decomposition；
- output replica 能分别投影到 lhs/rhs；
- `operand groups + partial-set groups` 不超过目标 vector-register budget；
- multiply dtype/instruction 关系闭合。

满足后生成 pairwise combine tree，直到剩两个 slot，再由 typed finalize 完成最终 reduction。规则不读取格式名、kernel 名或 target 名。

K1 Q2_K 当前每个 half 的 final IR 是：

```text
regular_repeat_index
→ regular_repeat_gather
→ partial_set(slots=4)
→ partial_combine(arity=2)
→ partial_finalize
→ assemble_replicas
```

当前汇编的相关静态计数：

| instruction | count |
| --- | ---: |
| `vwmul*` | 17 |
| `vwmacc*` | 0 |
| `vadd.vv` | 4 |
| `vwredsum*` | 4 |
| `vredsum*` | 5 |

这些计数包含整个函数，不等同于单个 Q2 block 的精确 operation 数；它们用于确认串行 `vwmacc` 链已消失。

### 4.3 SG 的 scale topology

SG 作者树把 16-element partial 和对应 scale 显式分开。为了让同一 scale window 跨 sub-Level 使用，`low_scales` 在 16-element Level 外由 `materialize` 产生；这改变作者声明的 lifetime，因此写在 std 树中，不由 pass 猜测。

[`MaterializeRISCVPartialAccumulators.cpp:730-900`](../lib/Target/MaterializeRISCVPartialAccumulators.cpp) 沿真实 loop-carried integer state 收集 contribution，并要求它们具有同一：

```text
reduction axis
slot ValueType
multiply instruction
output replica relation
scale replica relation
```

`unroll=4` 让每次 physical Level issue 收集 4 个 contribution，`slot_order=[0,2,1,3]` 将它们分成两个 2-term chain。该 Level 继续遍历作者写出的 16 个 subgroups；编译器没有把它改成 source donor 的完整 8-product issue，也没有改变其 loop-carried integer state。

最终每个 output replica 形成：

```text
partial_set
→ partial_reduce
→ partial_scale_combine(two chains)
→ partial_finalize
```

SG 当前 final IR 有两套上述链，对应 `NR=2` 的两个 output replicas，最后以 `assemble_replicas` 恢复 canonical `[MR,NR]` result。

## 5. 同一机制在其他输入上的正反检查

### 5.1 TQ2_0：16 槽与资源边界

把 independent topology 从固定 4 槽推广为任意 2 的幂后，TQ2_0 暴露出两个独立事实：

1. `MR=1,NR=1` 时，16-slot tree 数值完全一致，说明 `PartialSetType + pairwise combine` 的语义不依赖 Q2_K；
2. production `MR=4,NR=2` 时，partial set 本身占 32 groups，再加 lhs/rhs 已超过 SG 的 32-group budget。

第一次完整 emission 还暴露了 spill 的 correctness bug：resource groups 只统计同时 live 的 register replicas，但显式 spill 必须保存所有 sequential time parts。旧 spill slot 只分配 `groups × VLEN` bytes，导致多个 time part 以 1-byte stride 相互覆盖。

[`MaterializeRISCVResources.cpp`](../lib/Target/MaterializeRISCVResources.cpp) 与
[`RISCVDialect.cpp:2292-2338`](../lib/Dialect/RISCV/IR/RISCVDialect.cpp) 现在将 spill storage 定义为：

```text
register groups × product(time factors) × VLEN bytes
```

emitter 遇到显式 spill 的 typed Slice/Field 时先按既定 layout 物化，再机械 store。修复后强制 16-slot 版本数值完全一致，但只有 1.078 GOP/s，因为它需要大量 spill。

最终结构规则因此不是“见到多个 time parts 就建立 partial set”，而是“至少 4 个 pairwise slots 且整个 leaf issue 在目标资源内”。TQ2_0 production 保留既有流式 partial program，最终一次回归为 20.483 GOP/s；本轮前为 20.475 GOP/s，没有观察到退化。

### 5.2 Q5_1 与 IQ4_NL：两槽负例

Q5_1 与 IQ4_NL 都只有 2 个 reduction-time slots。临时把它们改成 independent set 后，没有任何 pairwise combine 层可形成，却拆掉了更紧凑的 widening-dot：

| case | 错误结构的一次/三次观察 | 恢复固定规则后一次观察 |
| --- | ---: | ---: |
| Q5_1 | 4.464 GOP/s | 6.404 GOP/s |
| IQ4_NL | 6.355 GOP/s | 8.632 GOP/s |

因此“至少 4 槽”是 combine topology 的结构门槛，不是格式白名单。当前 Q5_1/IQ4_NL final IR 不再出现 `rvv_partial_set`。

### 5.3 SG 六条既有过线路径

以下是最终代码的一次退化检查，不写入 10-repetition CSV：

| case | GOP/s | correctness |
| --- | ---: | --- |
| Q4_0 | 8.456 | within tolerance |
| Q4_1 | 6.775 | within tolerance |
| Q5_1 | 6.404 | within tolerance |
| IQ4_NL | 8.632 | within tolerance |
| Q1_0 | 6.930 | within tolerance |
| Q4_K persistent | 10.154 | within tolerance |

六条均恢复到本轮前的性能类别。Q4_K persistent 仍高于其 source；Q4_0/Q4_1/Q5_1/IQ4_NL/Q1_0 也没有因 Q2 partial topology 获得格式分支或新的作者树。

## 6. 参数与代码形态实验

### 6.1 SG

以下候选均在同一 `group_reduced` 作者树与同一编译主链上生成：

| candidate | GOP/s | 结论 |
| --- | ---: | --- |
| LMUL m2, unroll 1 | 4.846 | register footprint 过大 |
| LMUL m2, unroll 2 | 6.010 | 仍低于 m1 topology |
| LMUL m2, unroll 4 | 6.640 | 拒绝 |
| MR2 | 7.812 | output lifetime 增大，拒绝 |
| NR3 | 7.638 | 三 output replicas 过重，拒绝 |
| NR4 | 6.038 | 四 output replicas 过重，拒绝 |
| MR1×NR2, MC32, m1, unroll4 | 9.969884 | 最终 10 次 winner |

exact Level 原先直接以逻辑 offset 作为 induction variable，地址表达式反复做除法/乘法。现在
[`LowerRISCVComposites.cpp:856-900`](../lib/Target/LowerRISCVComposites.cpp) 让 exact Level 以 physical ordinal `0..total/partition` 迭代，再显式计算 `logical_offset = ordinal × partition`。同一 NR2 candidate 的单次对照为：

```text
logical-offset loop  8.969 GOP/s
physical ordinal     9.482 GOP/s
```

MR1 时该修改没有可测收益；NR2 的正向结果说明它解决的是多 consumer 地址表达式，不是 Q2_K 名称触发的 shortcut。

### 6.2 K1

本轮前的 lane16/MR2 为 1.659511 GOP/s；单纯扩大到 lane32/MR2 曾降为 1.471145 GOP/s，因为每个 scale vector 新增 `vslideup`。typed regular-repeat gather 和 independent partial 同时生效后，最终 lane32/MR1 为 2.508207 GOP/s。

这组对照说明 VLEN256 不是只改变 lane 数：memory form、scale construction 与 reduction topology 都随 target facts 进入了不同的 physical program。

## 7. 正确性、资源与 pass 边界

### 7.1 Leaf resource contract

新 op 创建时携带局部临时资源；统一 resource pass
[`MaterializeRISCVResources.cpp:201-219`](../lib/Target/MaterializeRISCVResources.cpp) 在最终验证前按真实 operands/results 重写每个 leaf 的 operand/result groups。最终 verifier
[`VerifyFinalRISCV.cpp:708-727`](../lib/Target/VerifyFinalRISCV.cpp) 重新计算并逐 op 比较，正式双机 repro 均经过该检查。

### 7.2 SSA evaluation

本轮涉及的 typed vector load、encoded Field 和 local materialization 的数值结果写回 emitter 的 SSA binding。同一个 vector/Field physical SSA producer只求值一次；reload/rematerialize 必须由显式 physical op 表示。这个修复使 `NR=2` 的 activation value 能被两个 output consumers 共享，而不是每个 consumer 重新生成一份 load/decode。

### 7.3 spec 2.2

本轮没有发现需要 compiler 改写作者逻辑树才能达到两台 source 的证据：

- SG 与 K1 仍使用上一轮已经确定的两棵作者树；runner/caller 显式选择 entry；
- SG 的 `low_scales` lifetime 由作者在 std 树中外提，因为它改变 materialization 所属 Level；
- regular-repeat load/gather、lane width、output replicas、independent partial、combine topology 与 exact-Level ordinal 都不改变 canonical values 或 Level 归属，因而由 physical compiler 生成。

这次结果支持当前边界：作者写数值树与 lifetime，编译器求同一树的 memory、register 与 reduction representation。

## 8. 最终正式结果

正式数据已写入 [`weft-kernel-performance.csv`](weft-kernel-performance.csv)。

| target | configuration | median | throughput | source | ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| SG2044 | `NC32 MC32 MR1 NR2 LMUL=m1 unroll4 pipeline1` | 430.794110 ms | 9.969884 GOP/s | 9.462 | 1.0537× |
| K1/X60 | `NC32 MC16 MR1 NR1 LMUL=m2 unroll1 pipeline1` | 1712.365516 ms | 2.508207 GOP/s | 2.410 | 1.0407× |

source 数字来自固定 baseline 记录；Weft 与 source 使用同一目标、shape、phase、计时范围和 Clang 编译选项。

## 9. Repro

正式 Q2_K：

```bash
examples/run/weft-mul-mat.sh sg2044 q2_k prefill 10
examples/run/weft-mul-mat.sh k1 q2_k prefill 10
```

横向回归：

```bash
examples/run/weft-mul-mat.sh sg2044 tq2_0 prefill 1
examples/run/weft-mul-mat.sh sg2044 q4_0 prefill 1
examples/run/weft-mul-mat.sh sg2044 q4_1 prefill 1
examples/run/weft-mul-mat.sh sg2044 q5_1 prefill 1
examples/run/weft-mul-mat.sh sg2044 iq4_nl prefill 1
examples/run/weft-mul-mat.sh sg2044 q1_0 prefill 1
examples/run/weft-mul-mat.sh sg2044 q4_k_persistent prefill 1
```

生成代码检查时使用：

```bash
WEFT_KEEP_ARTIFACTS=1 examples/run/weft-mul-mat.sh k1 q2_k prefill 1
```

当轮用于检查的 generated C、RISC-V assembly、physical IR 和远端运行目录在统计后删除，不进入仓库。
