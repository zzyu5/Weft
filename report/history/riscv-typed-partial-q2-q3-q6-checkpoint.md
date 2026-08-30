# RISC-V typed partial、Q2/Q3/Q6 收口报告

日期：2026-08-29

范围：`c29d835ac` 至 `99585592d`。本报告记录这一段已经提交的作者树、物理 IR、pass、emitter 与真实双机结果；不把更早的全量结果重新解释为当前结果。

所有表内“当前”性能均使用 Clang 18、`-O3`、相同的浮点 contraction 配置和固定 production shape。除明确标出的历史探索值外，性能均为 10 repetitions 的冷启动中位数。source 数字来自 `report/baseline/ggml-riscv-kernel-performance.csv`。

## 一、按 commit 说明实际改动

| commit | 改动层 | 为什么改 | 当前状态 |
|---|---|---|---|
| `c29d835ac` | physical IR、pass、emitter | 新增 typed storage window、layered load/decode、widen-accumulate 和 finalize，使 partial accumulation 不再由 emitter 从闭包猜测。TQ2_0 首次形成完整 typed partial program。 | 已提交并保留。TQ2_0 获得大幅正收益；Q2_K 当时没有形成同样的 partial lifetime。 |
| `c996444a9` | report | 将已经结束的阶段报告移入 archive。 | 已提交；不改变编译结果。 |
| `8477f4113` | 作者树、physical memory pass、emitter、runner | 增加 Q4_K、IQ2_XXS、IQ4_XS、MXFP4、NVFP4 等 blocked/decode 入口，并把 regular index/repeat 关系写入 typed physical access。 | 已提交并保留。不同格式结果正负都有，未把负结果藏在旧路径后面。 |
| `fec0c9943` | physical IR、pass、emitter | 新增 projected layered stream/window，让 encoded field 经 sub-Level 投影后仍携带 point、projection 和 layered geometry。 | 已提交并保留。表示能力闭合，但 Q2_K 性能为负；它没有自动产生 donor 的 partial program。 |
| `ff89f1ec1` | report | 静态计数 Q2_K 的 16 个 window 是否能做 read-CSE。加入 logical offset 后，完全相同的 window 数量为 0。 | 已提交；没有据此添加假 CSE pass。 |
| `cf04c54fd` | 作者树、canonical IR 输入、runner | 为 SG2044 增加 Q2_K `group_reduced` 作者特化；K1 保留 scale 进入 reduction 的树。两棵树改变中间值、widening 与溢出边界，不能由编译器按 VLEN 暗改。 | 已提交并保留；调用方按 target 选择 entry。 |
| `fb7232002` | physical IR、memory/layout/resource pass、emitter | 新增 typed regular-repeat index/gather、independent partial set、scale combine、pairwise combine/finalize 和 replica assembly，闭合 Q2_K donor 所需的物理拓扑。 | 已提交并保留。Q2_K production prefill 双机超过 source。 |
| `ae3b378a6` | pass | 修正 topology priority：`stream_reduction="fused"` 的 layered stream 不再被 independent-partial materializer 二次解释。 | 已提交并保留。K1 TQ2_0 从错误路径的 2.132 恢复到 6.657 GOP/s；当前为 6.746。 |
| `7b680e4f4` | 作者树、canonical IR 输入、physical IR、layout/memory pass、emitter | 将 Q3_K 改成显式 half/plane/scale-part/lane shaped tree；增加 natural affine coalescing、wide operand slice carrier、partial capture/repack/merge 和 dead-layout elimination。 | 已提交并保留。Q3_K 已能双机运行，但 computed scale 尚不能进入新的 partial-widen-scale 路径，性能未闭合。 |
| `99585592d` | 作者树、Encoding、physical IR、layout/share/partial pass、emitter、runner、CSV | 完成 Q6_K shaped Encoding、row-dequant、GEMV decode 和 blocked prefill；新增 `RVVPartialWidenScaleOp`，把 4 个 i16 product 拆成 8 个 i32 scaled partial，再 pairwise combine/final reduce；grouped/layered load 可直接形成 32-lane window，并消除已证明为零的层移位。 | 已提交并保留。Q6_K vec-dot、MUL_MAT decode、MUL_MAT prefill 在 SG2044/K1 上均超过各自 source。 |

这一段没有提交后又整体撤回的 commit。被撤销的是 commit 形成过程中证伪的局部实现或参数尝试，列在第三节。

## 二、当前双机结果与 source 比值

### 2.1 本段主线入口

| 格式 / 入口 | SG2044 Weft / source | SG 比值 | K1 Weft / source | K1 比值 | repetitions |
|---|---:|---:|---:|---:|---:|
| Q2_K vec-dot | 3.063 / 9.308 | 32.9% | 1.367 / 2.446 | 55.9% | 10 |
| Q2_K MUL_MAT decode | 3.034 / 8.815 | 34.4% | 1.367 / 2.404 | 56.9% | 10 |
| Q2_K MUL_MAT prefill | 10.134 / 9.462 | 107.1% | 2.962 / 2.410 | 122.9% | 10 |
| Q3_K vec-dot | 4.200 / 7.389 | 56.8% | 1.134 / 2.441 | 46.5% | 10 |
| Q3_K MUL_MAT decode | 4.248 / 7.193 | 59.1% | 1.135 / 2.394 | 47.4% | 10 |
| Q3_K MUL_MAT prefill | 0.887 / 7.219 | 12.3% | 0.432 / 2.449 | 17.6% | 10 |
| Q6_K vec-dot | 6.448 / 4.895 | 131.7% | 2.487 / 2.384 | 104.3% | 10 |
| Q6_K MUL_MAT decode | 5.217 / 4.851 | 107.5% | 2.472 / 2.333 | 106.0% | 10 |
| Q6_K MUL_MAT prefill | 9.208 / 7.397 | 124.5% | 2.600 / 2.509 | 103.6% | 10 |
| TQ2_0 MUL_MAT prefill | 20.582 / 6.983 | 294.8% | 6.746 / 4.979 | 135.5% | 10 |

Q6_K row-dequant 也完成真实运行且逐元素误差为 0：SG2044 为 703.300 MElements/s，K1 为 145.301 MElements/s。固定 source 表没有同一条 row-dequant microbenchmark，因此不制造比值。

Q3_K 在 `7b680e4f4` 报告中的 6.234 / 2.089 GOP/s 是单次探索值，不是 10-repetition 正式数字。当前表使用本次正式 10-repetition 结果。Q3 的下降不是数值错误：双机均在容差内；差异来自新的 nested partial hierarchy 没有被 computed signed scale 的 physical lowering 接住。

### 2.2 同期铺开或作为回归控制的格式

| 格式 / 入口 | SG 比值 | K1 比值 | 当前事实 |
|---|---:|---:|---|
| Q4_K canonical decode | 52.7% | 20.1% | K1 source 为 IME1 asm，当前 Weft 为 RVV；该百分比不是同引擎比较。 |
| Q4_K canonical prefill | 52.6% | 8.0% | staged blocked 树为负后恢复 canonical row/column 树。K1 source 为 IME1 asm。 |
| Q5_K vec-dot | 314.6% | 127.9% | shaped high-plane 主路径已获得收益。 |
| Q5_K MUL_MAT decode | 292.8% | 129.0% | 10 repetitions。 |
| Q5_K MUL_MAT prefill | 257.8% | 188.8% | 10 repetitions。 |
| IQ4_XS vec-dot | 36.0% | 29.6% | lookup/window 仍未跨 output consumer 共享。 |
| IQ4_XS MUL_MAT prefill | 58.9% | 74.3% | blocked 树已存在，但 codebook physical organization 未闭合。 |
| MXFP4 MUL_MAT decode | 101.0% | 86.3% | 10 repetitions。 |
| MXFP4 MUL_MAT prefill | 119.7% | 95.6% | 10 repetitions。 |
| NVFP4 MUL_MAT decode | 390.3% | 395.4% | source 是 scalar baseline；该比值不能证明接近手写 RVV。 |
| NVFP4 MUL_MAT prefill | 521.4% | 470.9% | source 是 scalar baseline。 |
| Q5_1 MUL_MAT prefill | 108.6% | 99.4% | independent-partial 强制实验撤销后，当前固定规则结果。 |
| IQ4_NL MUL_MAT prefill | 124.1% | 85.9% | independent-partial 强制实验撤销后，当前固定规则结果。 |

同期最重要的“掉下去又恢复”是 TQ2_0：K1 先为 6.419，错误命中 independent partial 后降至 2.132，修正 fused topology owner 后为 6.657，本次回归为 6.746 GOP/s。SG 当前为 20.582 GOP/s。

## 三、负实验与被排除的方向

1. **projected window 本身不等于复用。** Q2_K 加入 projected layered window 后，SG 从 2.136 降到 1.763，K1 从 1.823 降到 1.660 GOP/s。IR 已形成 2 loads、16 windows、8 updates、2 finalizers，但每个 window 仍独立完成 scale、payload 和 activation 工作。physical entity 只给工作命名，没有减少工作。

2. **Q2_K window read-CSE 假设被静态否定。** 16 个 window 加入 logical offset 后，没有两个完整 identity 相同；合法 CSE 可删除的 op 数是 0。忽略 offset 会读错数据。该方向没有进入代码。

3. **单纯扩大 K1 lane 不会自动得到 donor topology。** Q2_K lane16/MR2 为 1.660，lane32/MR2 降到 1.471，MR1 仅 0.985 GOP/s；新增 8 个 `vslideup`，没有 spill。regular-repeat gather 和 independent partial set 完成后才超过 source。

4. **fused stream 不能再次物化成 independent partial。** 错误门控使 K1 TQ2_0 从 6.419 降至 2.132 GOP/s；修正 `stream_reduction="fused"` 的优先级后恢复到 6.657，本次为 6.746。这个负结果证明两套 topology 的 owner 不能重叠。

5. **canonical Q4_K 不能只换成 staged blocked 树。** 该树虽能编译并通过容差，但 SG/K1 分别降到 1.489 / 1.383 GOP/s；它没有命中 persistent derived encoding 的 grouped-window/partial program，因此恢复原 canonical 树。

6. **Q3_K 的局部改写负结果。** scale 提前到父 Level 后 SG vec-dot 约为 2.28；raw window 扩成 m2 后约为 2.57；普通 `or/shl` bitplane 拼写无明显收益；盲目加宽 free axis 略微下降。通用 CSE 曾把 TQ2_0 从约 20.47 降到约 10.13，因为它错误合并了具有不同 partial lifetime 的值，已撤销。让 canonicalizer 产生 emitter 合同外的 `arith.select` 也已撤销。

7. **Q6_K 的 typed partial 不是一加就快。** 只加入 i16 partial → i32 scale topology 时，K1 为 1.921 GOP/s，低于此前约 2.079；随后保留 partial hierarchy 但仍以窄 load + `vslideup` 组织时降到 1.804。把 grouped/layered access 变成 32-lane direct load 后为 2.310；消除每个 256-record 内 6 个已证明为零的 shift 后，最终 10-repetition 为 2.487。保留的是完整关系，不是早期低效拼法。

8. **Q6_K 参数负结果。** SG2044 的 LMUL=16 在当前 widening layout 下无合法实现；LMUL=32 时 unroll=4 只有 3.975，而 unroll=2 的正式结果为 6.448 GOP/s。没有把失败参数留作默认。

9. **恢复旧 contraction conflict 方向不能修复 Q3。** 临时恢复“总 lane 较窄的 operand 转到较宽 operand”后，Q3 prefill 仍然在 widening-dot stream contract 处失败，因此该临时修改没有保留。真正的 correctness 问题是 emitter 额外要求两个 operand 的 stream 数相等，而 `RVVWidenDotOp` 已经携带各自独立的 typed part/lane-offset plan；删除这条多余约束后 SG 能运行。K1 最终使用合法的 LMUL=16 参数实例。

## 四、当前仍开着的口子

### 4.1 作者树尚未写完

- 8 个 production MUL_MAT 仍是 row×column 调用 vec-dot：IQ1_S、IQ1_M、IQ2_S、IQ2_XS、IQ2_XXS、IQ3_S、IQ3_XXS、TQ1_0。它们没有 output cohort、blocked accumulator 和跨输出 activation/codebook reuse。
- 8 个 row-dequant 函数仍用 Python `for j in range(256)` 展开：IQ1_S/M、IQ2_XXS/XS/S、IQ3_XXS/S、TQ1_0。
- IQ1/IQ2/IQ3/TQ1 的 vec-dot 仍含 Python group/entry/lane 或 radix digit 展开。Q4_K/Q5_K 的主 contraction 已 shaped，但 minimum correction 仍保留小型 scalar sub-loop。

### 4.2 编译器仍缺的能力

- **结构性物理选择：computed-scale partial topology。** Q6 的 direct signed i8 scale 能进入 `RVVPartialWidenScaleOp`；Q3 的 `scale_low | (scale_high << 4) - 32` 是计算得到的 shaped scale，目前仍落回 `rvv_widen_dot` 的逐 partial reduction。Q3 donor 的 4 independent i16 products、8 scaled i32 partial 和 staged final reduction尚未由同一机制生成。
- **结构性物理选择：lookup-aware window 与跨 output reuse。** IQ4_XS/NL 的 codebook/index/window 尚未成为多个 output register consumer 共享的 physical value。
- **唯一合法推导：multi-root joined storage geometry。** Q5_K 的 q/qh、IQ 的 grid/sign 等多个 field root 尚不能总是合成一个闭合的 typed window；目前部分路径仍退回普通 extract。
- **结构性物理选择：Q4_K persistent reduction handoff。** `production_mul_mat_q4_k_persistent` 在 clean `7b680e4f4` 与当前节点都会报 non-scalar reduction lane remap；独立 clean-HEAD 构建确认它不是 Q6 提交引入的回归。
- **legality/representation contract：constant layer proof。** `RVVReplicaStorageLoadOp` 当前保存 `logical_base_multiple`，Share pass 产生该证明，emitter 用它判断 layer 是否恒为 0。当前唯一 creator 与 Q6/Q3 实例均正确，但 verifier 只检查它为正数，没有重新验证 proof 与 SSA base 的一致性。

### 4.3 已知但未继续调查的现象

- Q3 正式 10-repetition 性能明显低于 source；生成程序已经数值正确，具体缺口已缩小到 computed-scale partial materialization，而不是作者轴缺失。
- Q4_K persistent 仍不能 emit；本轮只确认它在 clean HEAD 已存在，没有把修复混入 Q6 收口提交。
- 本段只回归实际受影响入口，没有重新跑 206 条全量。
- K1 普通用户环境仍没有可用的 PMU/perf 计数器，因此这一段的指令归因来自生成 C、汇编和静态动态工作计数。

## 五、同一物理决策是否仍在两个地方实现

答案是：**有。** 当前没有把它们写成“已经统一”。

1. **partial topology 有两个识别入口。** `MaterializeRISCVPartialAccumulators` 同时保留 `groupedLanePartials` 与 `plannedReplicaLanePartials` 两套 lane-split 判定，最后汇合到同一个 partial-set/repack topology。`stream_reduction="fused"` 又在外围决定是否跳过 independent materialization。TQ2_0 的 6.42 → 2.13 → 6.66 正是这个 owner 重叠被错误门控的外部证据。

2. **layout owner 与 partial materializer 有重叠。** `PropagateRISCVLayouts` 已决定 reduction/free axis 的 lane、time、replica hierarchy；`MaterializeRISCVPartialAccumulators` 又从 lane/time/replica factors 重建 grouped/planned lane split。前者应提供 typed facts，后者应只消费，但当前后者仍含二次结构判定。

3. **layered/projected geometry 在 pass 与 emitter 中重复。** Share pass 选择 window、layer、part offsets；旧 layered/projected stream emitter 仍重新计算 group/layer/stream axis、physical layer 和 raw-window cache key。部分是终端合同检查，部分仍是物理组织重算。

4. **零移位消除目前也跨越 pass/emitter 边界。** Share pass 写入 `logical_base_multiple`，emitter 再用 `% group` 判断 constant layer 并决定是否发 shift。Q6 的收益已由双机结果和生成指令验证，但“是否需要 shift”尚未成为一个完全选定的 terminal attribute/op。

5. **regular-repeat、resource 与 conversion 三处没有发现同类重复。** Plan pass 已选择 broadcast/pow2/div regular-repeat leaf，emitter只拼写；resource pass只做 liveness/peak legality；layout conflict 插 typed conversion，Finalize 只闭合 leaf。这三处与上述 topology owner 重叠不同。

`RVVPartialWidenScaleOp` 与原有 reduce-then-`RVVPartialScaleCombineOp` 表达的是两种不同的合法 physical topology；当前由同一个 materializer 根据 typed element width、scale value 与资源合同决定，不是 emitter 内的第二套 Q6 路径。
