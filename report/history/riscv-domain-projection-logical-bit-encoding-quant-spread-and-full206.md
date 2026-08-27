# RISC-V 逻辑位轴、表示传播、量化铺开与 206 条全量回归

日期：2026-08-27

## 1. 本轮回答的问题

本轮没有以某一条性能数字为唯一目标，而是用新的输入形态检查四件事：

1. `production_mul_mat_q4_k_staged` 的 widening-dot 回归是否来自 pipeline；
2. Q1_0 的 128 个逻辑 bit 应由 Encoding 表达，还是应放宽普通索引规则；
3. 已经存在的 physical pass 是否依赖固定 SSA 拼写，换一个 typed storage geometry 后是否仍工作；
4. 量化 MUL_MAT、row-dequant 和 vec-dot 横向铺开后，双机 206 条真实 production 结果实际落在哪里。

参考实现不是用来证明 Weft 的形式合理，而是用来确定同类机制的落点：

- Triton 在 `ref/triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp` 中按每个 memory op 的 use、contiguity 和 order 选择访问 encoding，并插入真实 `convert_layout`；
- Triton 在 `ref/triton/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp` 中沿 uses、region branch 和 backward slice 做传播、rematerialization 与 dominance-order rewrite；
- Triton 的局部 canonical combine 也会使用 direct producer matcher，例如 `ref/triton/lib/Dialect/Triton/Transforms/Combine.cpp`，但这类 matcher 不承担 target physical representation 的唯一决定；
- TileLang 先运行 `PipelinePlanning`，再运行 `InjectSoftwarePipeline`，具体顺序见 `ref/tilelang/tilelang/cuda/pipeline.py`；规划实现位于 `ref/tilelang/src/transform/pipeline_planning.cc`。scheduler 和 expander 是不同职责。

本轮的判断标准因此不是“pass 存在”或“代码里没有格式名”，而是换成 Q1、Q5、IQ 等不同 storage geometry 后，表示传播、memory planning 和 lowering 是否仍得到正确程序与合理机器形态。

## 2. Q4_K staged 回归

### 2.1 现象

`production_mul_mat_q4_k_staged` 在当前主干上即使 `pipeline_depth=1` 也无法 lower：RVV widening-dot 的 operand/result shape 不闭合。因此错误不是 scheduler/expander 引入的。

### 2.2 根因和修改

`SelectRISCVOperations` 曾把 `OuterContractOp` 送入只适合单一 reduction result 的 widening-dot realization。该候选的 live operand、partial 和 result 合计需要 85 个 vector register groups，目标只有 32 个；它既不满足结果 shape，也不满足资源约束。

修改后：

- widening-dot 只服务具有相应结果关系的 `DotOp` / `ContractOp`；
- `OuterContractOp` 使用普通 local register microkernel；
- 资源不足不再由 emitter 缩小或改走隐藏路径。

双机单次真实 repro 均恢复数值正确。SG2044 和 K1 的代表结果分别约为 1.37 和 0.68 GOP/s；这里的意义是回归闭合，不把这两个单次并行运行数字当正式性能结论。

## 3. Q1_0：缺的是 Encoding 语义，不是放宽索引

### 3.1 语言结论

Q1_0 一条 record 包含 128 个逻辑一位值，storage 是 16 个 byte。原声明 `q: u8[16]` 只描述 storage array，真实的 logical-bit-to-byte projection 留在数值树的 `index/shift` 中。

最终声明改为：

```python
q: u1[128] @ grouped(8) @ layered(1, lo_first)
```

这项信息只说明 bytes 如何承载 128 个逻辑元素，不定义 radix、codebook 或任何数值解码。因此它属于 Encoding，不需要新的 sub-axis projection op，也不应放宽 frontend 让一个 storage byte 同时冒充 logical bit value。

同一结论也用于 Q2_K/TQ2_0 的二位字段：

```python
q: u2[256] @ grouped(128) @ layered(32, lo_first)
```

### 3.2 真正的数值错误在 emitter 的轴归属

第一版 blocked Q1_0 仍然数值错误。physical IR 已经包含正确事实：

```text
result logical axes: output-N × K-subdomain(32)
time factors:        1 × 32
lane factors:        4 × 1   (SG2044 representative)
```

但 intrinsic-C emitter 在处理 encoded Field 的 `domain` selector 时，无条件把 `logicalRank` 减一。`domain` 只把轴限制到当前 Level partition，并没有消去轴；因此 32 个 time part 都被当成同一个 logical index，生成 C 中 32 次 load 使用相同 byte 和 shift。

修改后只有真正消轴的 `index/group_index/gather` 会降低 Field logical rank。新的生成 C 明确出现 `+0, +1, ... +31`，Q1 blocked prefill 在两台机器上都恢复数值正确。

正式 10 次结果：

| target | 上一份快照 | 本轮 | 提升 | GGML baseline | 当前比例 |
|---|---:|---:|---:|---:|---:|
| SG2044 | 0.298 GOP/s | 0.762 GOP/s | 2.56x | 4.250 GOP/s | 17.9% |
| K1 | 0.141 GOP/s | 0.508 GOP/s | 3.60x | 3.793 GOP/s | 13.4% |

这是一项正确性与通用轴传播修复；它没有自动解决 blocked quant microkernel 的性能。

## 4. per-use storage geometry 和 conversion

### 4.1 typed `source_access`

IQ4_NL 在 SG2044 可运行、K1 首次失败的原因不是 codebook 数值语义，而是同一个 encoded Field 经 `ExtractOp -> ConvertLayoutOp` 后，memory pass 重新从 Field 的全局 access 推导访问方式，丢掉了当前 use 的 strided/indexed 关系。

本轮把 `source_access` 变成 `ConvertLayoutOp` 的正式 typed optional attribute，并调整 memory planning：

- 先为每个 `ExtractOp` 产生 per-use access；
- conversion 读取该 extract edge 的 access；
- representation conversion 可被穿透，但不会把 per-use access 降成字段级默认值；
- verifier 限制 access 只能是 unit/strided/indexed/segment 中的已选形式。

这使此前 SG 上 12 条 IQ conversion/lookup compile failures 全部能够生成并运行；K1 IQ4_NL prefill 也保持数值正确。这个结果说明 access 必须挂在具体 use edge，而不是 Encoding field 的单一全局记录上。

### 4.2 与 Triton 的机制差异

Triton 的 conversion 是真实 op，memory consumer 的 encoding 也是 IR type 的组成部分；冲突通过插 op 表示，消除通过改写 IR 表示。Weft 现在也有真实 `ConvertLayoutOp`，但若 pass 仍通过 direct defining op 找 provenance，typed op 本身不会自动带来传播能力。

本轮进一步让以下路径穿透 representation conversion：

- widening reduction 识别；
- ConvertLayout source memory access；
- bitplane relation 的 typed Field provenance；
- layered-window share、grouped MAC 和部分 composite lowering 已有的 source-field/source-load helper。

## 5. Q5：用另一种 logical bitplane 检查 Q1 结论

### 5.1 Encoding 边界

Q5_0/Q5_1/Q5_K 的 `qh` 同样不是独立数值 metadata，而是每个 q 元素的高一位。声明改为：

```python
Q5_0/Q5_1: qh: u1[32]  @ grouped(8)   @ layered(1,  lo_first)
Q5_K:       qh: u1[256] @ grouped(256) @ layered(32, lo_first)
```

row-dequant、vec-dot 和 MUL_MAT 不再自己写 `byte index + shift`。随机 bytes 直接与 GGML reference 比较，三种格式在两台机器上均数值正确。

### 5.2 typed logical-bit fusion 的正结果

当 bit axis 本身映到 RVV lane 时，pass 现在可从以下 typed facts 形成 mask bitplane merge：

- high operand 来自 logical `ui1` encoded Field；
- Field mapping 是 grouped(8)+layered(1)；
- low/result 是相同 axes/shape 的 unsigned RVV value；
- shift 的 insert bit 是常量；
- lane axis 就是 Field 的唯一 logical bit axis。

这条路径不需要恢复 `iota -> div/mod -> gather -> shift` 的原始源码形状。Q5_0/Q5_1 row-dequant 和 vec-dot 因此可以从新 Encoding 进入同一 typed leaf。

### 5.3 Q5_K shaped 改写

Q5_K row-dequant 和 vec-dot 原先在 Python 中逐 256 个元素展开。改写后显式使用 32 元素 `L.subs`：

```text
logical q[32] + logical high-bit[32]
→ 32-element shaped value
→ contract with Q8_K activation
```

正式结果：

| case | target | 旧 Weft | 本轮 Weft | GGML | 当前比例 |
|---|---|---:|---:|---:|---:|
| Q5_K vec-dot | SG2044 | 0.223 | 4.097 GOP/s | 1.302 | 314.6% |
| Q5_K vec-dot | K1 | 0.091 | 1.588 GOP/s | 1.242 | 127.9% |
| Q5_K row-dequant | SG2044 | 约 35.1 | 315.6 MEl/s | 441.1 | 71.5% |
| Q5_K row-dequant | K1 | — | 186.2 MEl/s | 112.2 | 165.9% |

这是本轮最清楚的“作者轴 + compiler representation”共同起效案例：编译器没有从 256 个标量 SSA 猜回轴，作者显式写出 32 元素 shaped value 后，同一个 Encoding 和 contract lowering 在两台机器上工作。

### 5.4 失败并被删除的 realization

本轮也尝试让 typed logical bitplane 在 output axis 进 lane、bit axis 留在 sequential time 时生成 strided merge。它能够生成正确代码，但正式/代表性能低于普通 typed field lowering：

- SG Q5_0/Q5_1 约 0.308/0.288 GOP/s；
- 未融合的全量快照为 0.465/0.584 GOP/s。

因此针对 typed logical plane 的该 realization 已从最终 std/matcher 路径删除；最终 logical-bit fusion 只保留被外部结果证明有用的 lane-mask 形态。代码中另有一条服务旧式 raw-byte plane 的 `rvv.bitplane-merge.strided` leaf，它不接受 `ui1` logical plane，也不是这次负实验保留下来的候选。

## 6. Q5_K blocked MUL_MAT：树成立，pass 没有兑现

Q5_K production MUL_MAT 已改为与 dense blocked tree 同类的结构：

- NC/MC output tiles；
- MR/NR accumulator；
- 256-element K block；
- 32-element logical sub-Level；
- activation 在多个 output 间共享；
- scale/minimum 两支在 block 内组合。

它在两台机器上都数值正确，但正式 prefill 只有：

| target | Weft | GGML | ratio |
|---|---:|---:|---:|
| SG2044 | 0.345 GOP/s | 2.702 GOP/s | 12.8% |
| K1 | 0.277 GOP/s | 1.251 GOP/s | 22.2% |

这不是 row×column tree 的问题。生成的 physical form 仍把 output-N 放入很窄的 lane value，把 K=32 展开为 sequential time parts；每个 part 重建 encoded loads/decode，未形成跨 K 的 shared window 或 steady-state cluster。Q5_0 的代表生成 C 中可直接看到 32 组 low/high load、shift、merge 与随后的小向量 MAC。

MR/NR 扫描还暴露一个 correctness 缺口：

- `NR=4` 数值正确；
- `NR=8/16` 被当前 candidate pipeline 接受，但生成错误数值。

因此这不是“较慢候选”，而是 candidate legality 尚未覆盖 lane-axis 多 time-part handoff。正式 runner 没有采用这些错误实例；该缺口仍存在。

## 7. 拼写依赖审计

### 7.1 已改为 typed relation 的部分

- Field/storage provenance 可穿透 pure/read representation conversion；
- per-use memory access 由 Extract edge 产生；
- logical-bit fusion 有一条不依赖 `iota/div/mod/gather` 的 typed path；
- layered-window share 已按 FieldFacts、point、access geometry 和 validity 判断；
- widening reduction 可穿透 conversion；
- pipeline expander 消费 scheduler 写入的 stage/order 和真实 crossing SSA value，不在 emitter 中识别 GEMM/quant 名称。

### 7.2 仍然依赖固定 producer chain 的部分

以下事实来自当前代码，而不是未来计划：

- raw-byte bitplane 兼容路径仍要求 `or -> shl -> and -> shr -> extract -> div/mod -> iota`；
- partitioned widen-reduce-store 仍要求固定 `Store -> Reduce -> Widen -> Extract -> PhysicalPoint`；
- stream-dot 仍要求 reduce 的直接语义 producer 是 multiply；
- grouped MAC 仍要求 direct `MacGroupsOp` relation；
- encoded-dot 仍要求 direct `Fold2Op` relation；
- contract init fusion 仍要求 one-use direct add；
- layout canonicalization 的 rematerialization 仍只看一个 direct producer，而不是 Triton 式 backward slice；
- scheduler 只接受 closed read-only `scf.for` nested region，尚未覆盖一般 region-branch/effect dependence；
- computed aggregate 超预算仍没有完整 spill，主要只支持 reload 可重读 admitted value。

这些路径没有格式名并不意味着已经泛化；改变等价 SSA 拼写仍可能使能力消失。

## 8. 双机 206 条正式结果

### 8.1 口径

- SG2044：RVV VLEN128；
- K1/X60：RVV VLEN256；本轮不接 production IME；
- 两侧 Weft runtime、GGML wrapper 都使用 Clang 18 与相同 `-O3 -ffp-contract=fast`；
- production shape：vec-dot `M=1,N=14336,K=4096`，MUL_MAT `N=4096,K=4096` 且 decode `M=1`、prefill `M=128`；
- 每条 10 次，记录 cold median；
- 浮点使用绝对/相对容差，量化 storage 输出使用 bit-exact；
- `source/` 只提供 baseline，不进入 Weft 编译或运行路径。

完整数据：`report/weft-kernel-performance.csv`。

### 8.2 正确性

| target | entries | return code 0 | floating within tolerance | discrete bit-exact |
|---|---:|---:|---:|---:|
| SG2044 | 103 | 103 | 100 | 3 |
| K1 | 103 | 103 | 100 | 3 |

### 8.3 性能分布

按 `Weft / same-target GGML`：

| target | >=90% | 70–90% | 50–70% | <50% | median |
|---|---:|---:|---:|---:|---:|
| SG2044 | 22 | 7 | 7 | 67 | 0.339 |
| K1 | 26 | 3 | 6 | 68 | 0.355 |
| total | 48 | 10 | 13 | 135 | 0.349 |

上一轮用户提供的参照是中位 0.264、128/206 低于 50%。本轮中位上升，但低于 50% 的数量增加到 135。这说明 Q1、Q5_K、F16 等少数路径改善明显，横向覆盖没有同步闭合。

K1 baseline 中有 6 条使用 IME1 asm，而本轮 Weft 明确只运行 RVV：Q4_0/Q4_1/Q4_K 的 decode 与 prefill。若隔离这 6 条不可比项，剩余 200 条为：

```text
>=90%: 48
70–90%: 10
50–70%: 13
<50%: 129
median: 0.355
```

因此 IME 口径只解释 6 条，不解释整体低性能分布。

## 9. 归因聚类

### 9.1 按程序和编译阶段聚类

| cluster | entries | <50% | 观察 |
|---|---:|---:|---|
| dense blocked prefill | 4 | 0 | F16 为 SG 96.0%、K1 106.6%；blocked dense 主线已可用 |
| blocked quant prefill | 20 | 19 | tree 已有 output cohort，但 decode/load/reuse/pipeline 未形成高质量 physical program |
| row×column quant prefill | 28 | 25 | M=128 基本重复 M=1；作者树仍缺 output blocking/reuse |
| MUL_MAT decode | 52 | 37 | 主要受 vec-dot/decode realization 限制 |
| vec-dot | 48 | 33 | 未 shaped 的 bitplane/codebook/scale 路径仍大量标量化 |
| row-dequant | 48 | 21 | 仍有 12 个函数逐元素 Python 展开 |
| activation quantize | 6 | 0 | 当前 shaped VLA/quantize 主线整体可用 |

当前仍是 row×column prefill 的 14 个格式：

```text
Q3_K, Q4_K(canonical), Q6_K,
IQ1_S, IQ1_M, IQ2_S, IQ2_XS, IQ2_XXS,
IQ3_S, IQ3_XXS, IQ4_XS, TQ1_0, MXFP4, NVFP4
```

当前仍逐元素展开的 row-dequant 有 12 个，vec-dot 有 9 个。Q5_K 的前后结果证明，这一类需要先由作者显式恢复 shaped logical axis；compiler 不应从 Python 展开的 SSA 中猜回轴。

### 9.2 blocked quant 的实体级原因

blocked quant 的 19/20 低于 50%，所以“只补树”不足。代表 physical facts 是：

- output free axis 被映成很窄的 lane value；
- K sub-axis 被展开成大量 sequential parts；
- packed Field 在每个 part 重新计算 byte/shift/window；
- load/decode/contract 没有形成跨 K 的共享 cluster；
- Level assignment 仍常见 `unroll=1, pipeline=1`；
- target 从 VLEN128 变 VLEN256 时，lane 数变化多于 memory/lookup/schedule 结构变化。

Q8_0 是 blocked quant 中唯一超过 50% 的 SG prefill（72.6%）；K1 同一程序只有 37.7%。这说明当前 target facts 能影响宽度，却还没有稳定地产生不同的目标相关数据搬运和局部调度。

### 9.3 明确的正例和反例

正例：

- F16 blocked prefill：SG 96.0%，K1 106.6%；
- Q5_K shaped vec-dot：SG 314.6%，K1 127.9%；
- Q5_K shaped row-dequant：K1 165.9%；
- SG Q8_0 decode/vec-dot 明显超过 baseline。

反例：

- SG/K1 TQ2_0 prefill：2.55% / 2.87%；
- SG/K1 Q6_K prefill：3.57% / 3.98%；
- K1 Q3_K vec-dot/prefill：约 4%；
- blocked IQ4_NL prefill：SG 9.48%、K1 15.58%；
- IQ2_XXS shaped codebook helper虽比旧 Weft 提升约 4–5 倍，仍只有 SG 16.7%、K1 18.0% baseline。

## 10. 当前结论

本轮证明了三项具体能力：

1. logical sub-byte axis 可以由 Encoding 表达，并在 domain projection、time part 和两种 VLEN 上保持正确；
2. per-use storage geometry 可以穿过 typed conversion，不再只能依赖字段级默认 access；
3. 作者把 scalar-expanded Q5_K 改成 shaped sub-Level 后，同一套 lowering 能在两台机器上显著改善。

但 206 条外部结果不支持“已经形成统一高性能算子编译器”的结论。更准确的状态是：

- dense blocked tree 已有成熟度；
- shaped row-dequant/vec-dot 的一部分可达到或超过手写；
- blocked quant tree 已能表达和运行，但 physical mapping、decode reuse、cluster/pipeline 仍普遍不成熟；
- 14 个 quant prefill 仍是作者侧 row×column tree；
- 多个 physical pass 仍以固定 producer chain 代替一般 use-def/typed relation；
- tuner/candidate legality 仍存在会接受错误 NR 实例的 correctness 缺口；
- production IME 没有在本轮接入，K1 六条 IME baseline 只能单列，不能作为 RVV 编译器性能结论。

因此当前性能差距既不是单一 DSL 问题，也不是单一 emitter 问题：row×column 族首先缺作者树；已经 blocked 的量化族则明确缺 compiler 的物理表示传播、跨输出/跨 K 复用和通用 local scheduling。
