# 轴、值与生命周期编译核心：两轮推进与性能分析

## 结论

这两轮对应两个连续提交：

| 轮次 | 提交 | 目标 |
|---|---|---|
| 第一轮 | `185be3e080789d35060731e37a01933eb44222f7` | 建立轴、值链、memory relation、lifetime 与 target resource 共同驱动的 RISC-V 物理编译核心 |
| 第二轮 | `4b31286d84193186fb116d5fd193595524c7cf62` | 查漏补缺，扩大真实候选，修复跨控制流组合与 lifetime 问题，并重新测量性能 |

当前已经成立的是：

```text
DSL kernel
→ canonical Kernel IR
→ axis / value / use / memory / lifetime facts
→ resource-legal physical decisions
→ intrinsic C / local asm
→ system compiler
→ object / executable
```

这不是把现有 examples 分类后分别发射。VLA、memory、state、dot/matmul、lookup、quant 与
extension 分别根据自身 typed facts 产生物理决定；生成代码阶段消费这些决定，不再拥有第二次
LMUL、microtile、memory form 或 fragment 选择。

但“性能闭合”目前只能评价为**部分完成**：

- Q4 N16×K32 的共享 RVV leaf 已经从低质量逐元素实现变成可复用的向量实现，并同时改善
  projection 与 MoE；
- K1 上相同 DSL primitive 能根据 target facts 选择 RVV 或 IME，IME 路径已进入接近 GGML
  production baseline 的区间；
- VLA reduction state placement、F32 dot、F16 matmul 等已有真实多候选；
- K1 F16/F32 prefill、CSR、transpose、IQ3，SG2044 IQ1_M，以及两台机器上的 flash attention
  仍有显著差距；quant 与 IME 的候选空间也仍比 dense matmul 窄。

因此，这两轮证明了编译器方向成立，但没有证明所有主要 kernel family 已达到 Triton 级的
成熟物理空间。

## 报告边界与证据

本文件是一次性工作快照，不定义 DSL 或 compiler。规范仍是 [`doc/index.md`](../doc/index.md)
及其链接的模块。性能原始记录是：

- 当前 Weft 数字：[`weft-kernel-performance.csv`](weft-kernel-performance.csv)；
- 固定 GGML 数字：[`baseline/ggml-riscv-kernel-performance.csv`](baseline/ggml-riscv-kernel-performance.csv)；
- GGML 测量说明：[`baseline/ggml-riscv-kernel-baseline.md`](baseline/ggml-riscv-kernel-baseline.md)。

轮间比较使用三个 Git 快照中的同一性能 CSV：

```text
45375e422  第一轮之前
185be3e08  第一轮结束
4b31286d8  第二轮结束
```

三个快照的 149 条记录具有相同的 kernel、shape、target、repetition、warmup、eviction、
correctness scope 与 preprocess scope，因此轮间变化可以直接按 `median_ms` 比较。它仍然会受
目标机状态影响，不能把小幅波动全部归因到某一行代码。

## 当前运行与计时范围

### 覆盖

| 目标 | 结构性真实运行 | 正式性能行 |
|---|---:|---:|
| SG2044 / RVV VLEN128 | 90/90 条命令 | 73 个 kernel/phase case |
| K1/X60 / RVV VLEN256 | 90/90 条命令 | 73 个 kernel/phase case |
| K1/X60 / IME1 VLEN256 | 3/3 条命令 | 3 个 kernel/phase case |

90 条命令包括 phase 展开、自然等价 entry 与 correctness-only quant probes；性能 CSV 不把
correctness-only 运行伪装成 timing。当前 CSV 共 149 个数据行：SG2044 73 行、K1 RVV 73 行、
K1 IME 3 行，所有行均有 `median_ms`。

### Weft 计时协议

- 每次样本前遍历 64 MiB eviction buffer；
- 只计 DSL kernel entry invocation，不计 DSL frontend、intrinsic C 生成、目标 C 编译和输入初始化；
- repetition 按 case 为 3、5、7、10 或 30；
- 140 行使用 `warmup=0`，9 行使用 `warmup=3`；后者包括 Q4 projection 的 6 个 target
  组合，以及 Q4 `mul_mat_id`/MoE 的 3 个 target 组合；
- correctness scope 为 `full` 112 行、`full_rows` 8 行、`sampled` 26 行、
  `activation_full_output_sampled` 3 行；
- 29 行将 activation quantization、staging 或其他预处理计入 timed region。

因此，`sampled` 只说明该 runtime 的抽样数值对照通过，`preprocess_in_timed_region=yes` 的吞吐
也不能解释成纯 microkernel 吞吐。

### 与 GGML baseline 的可比性

GGML baseline 是单线程、warmup 1 次、每次计时前 64 MiB eviction、10 次 invocation median；
编译、初始化、weight quantize/repack、warmup 与 eviction 不计时。Weft 大部分行没有 warmup，部分
case 的 repetition 和 preprocess scope 也不同。

所以本报告中的 Weft/GGML 比率是**同硬件、同算法、同 shape 下的方向性比较**，不是完全统一
协议后的最终 speedup。跨 SG2044 与 K1 的绝对时间不用于判断实现优劣。

下文“相对速度”统一定义为：

```text
吞吐型：Weft throughput / GGML throughput
时间型：GGML median_ms / Weft median_ms
```

大于 1 表示 Weft 更快，小于 1 表示 Weft 更慢。

## 第一轮：建立物理编译核心

### 从 per-op 参数转向连接的事实系统

第一轮在 [`RISCVLowering.cpp`](../lib/Target/RISCVLowering.cpp) 中建立了四类连接事实：

```text
LogicalAxisFact
ValueUseFact
MemoryAccessFact
KernelPhysicalFacts
```

它们把以下信息接到同一次 target lowering：

- block/VLA axis identity 与 extent；
- free、reduction、broadcast relation；
- producer、普通 consumers 与跨控制流 uses；
- pointer 对 logical coordinate 的 unit/strided/indexed/segment relation；
- predicate、validity 与 memory effect；
- VLA value 从 definition 到 last consumer 的 live interval；
- target VLEN、合法 SEW/LMUL、vector register budget 与 extension facts。

这一步的关键变化不是增加一张 Plan 表，而是让同一 value 在 load、cast、pointwise、state、
dot/matmul 与 store 之间使用一致的 physical shape 与 handoff。若一个 consumer 需要不同形态，
lowering 必须显式记录 share、reload、rematerialize 或 local conversion，不能因 closure 不匹配
而取消 primitive 的实现能力。

### 真实候选与资源合法性

第一轮形成的主要候选包括：

- F32 VLA/local dot：合法 LMUL × K-unroll；
- F16 matmul：input LMUL、row microtile、multiple accumulators、K-unroll 与 pipeline stages；
- VLA region：data/index/mask shape、memory mode、state 与 local temporary；
- target profile：SEW/LMUL、indexed memory、segment memory、widening 与 IME eligibility。

候选不只保存字段。每个 entity plan 记录 operand、accumulator、index、mask、state、temporary、
handoff 与 pipeline 所占 register groups；超过 target budget 的组合在生成代码前非法。

Emitter 保留复杂的 RVV intrinsic、C ABI 与 typed inline asm 拼写，但不再拥有 primitive 分类和
资源选择。Kernel IR 仍是唯一长期表示；这些 facts、候选和 winner 都只在一次 lowering 中存在。

### 第一轮性能结果

第一轮的主要目标是把编译决策接起来，而不是只优化一个 leaf。149 条同协议记录的整体变化为：

| 统计 | SG2044 | K1/X60（含 3 个 IME 行） |
|---|---:|---:|
| median time 几何平均变化 | 1.006× | 1.010× |
| 至少快 10% | 3 项 | 0 项 |
| 至少慢 10% | 4 项 | 2 项 |

`1.006×` 表示整体时间约慢 0.6%，即第一轮没有产生全局性能提升。它的价值主要是暴露哪些
共享能力仍不完整。

主要改善：

| target | kernel | 第一轮前 ms | 第一轮后 ms | 提升 |
|---|---|---:|---:|---:|
| SG2044 | interleaved complex segment2 | 3.655556 | 2.846432 | 1.28× |
| SG2044 | out product F32 | 237.683710 | 186.170411 | 1.28× |
| SG2044 | F32 prefill local dot | 1046.267654 | 923.112699 | 1.13× |
| SG2044 | dense conv2d | 864.176405 | 788.259252 | 1.10× |

主要退化：

| target | kernel | 第一轮前 ms | 第一轮后 ms | 退化 |
|---|---|---:|---:|---:|
| K1 | segmented inclusive scan | 21.321605 | 35.376641 | 1.66× |
| SG2044 | online flash attention | 23.371502 | 35.383774 | 1.51× |
| SG2044 | segmented inclusive scan | 9.746722 | 13.406417 | 1.38× |
| K1 | online flash attention | 52.776069 | 64.442771 | 1.22× |
| SG2044 | F16 GEMM decode | 14.376841 | 16.992434 | 1.18× |

第一轮因此得到的真实结论是：axis/value/lifetime 编译骨架已经形成，但 state/summary、attention
和部分 dense microkernel 的资源与 schedule 仍不足，不能把“有候选字段”当成性能成熟。

## 第二轮：查漏补缺与性能闭合

### 跨控制流与普通组合

第二轮补齐了第一轮遗漏的组合边界：

- `if` 两个 yield 的 logical extent 必须一致，并能为 result 推导唯一 extent；
- physical operation 收集递归进入 `for/if/while`；
- VLA vector、mask 与 predicate 可以通过 `if/while` carry；
- lookup decision 显式保存 code、index16、index32、table 与 result 的 RVV shape；
- block widening product 与 reduction result 拥有显式 input/result/seed handoff；
- block tuple closure 跨 region 时明确 unsupported，不把不同 block 的 operation 交给
  `isBeforeInBlock` 排序；
- structured result 增加普通 consumer、经过 control carry 或改变 SSA 临时组织后，不需要新的
  combination matcher。

本轮只修改了 `ssm_conv.py` 与 `col2im_1d.py` 两个 DSL 文件，用自然等价写法压力测试
`if/while` carry、predicate、pointer-base 与临时 SSA。其他 examples 没有为了留下修改痕迹而
改写。

例如 [`causal_mask.py`](../examples/kernels/attention/causal_mask.py) 四天未修改仍然正确，是因为
它只使用仍然有效的 `W.range + W.vla + compare + masked store`。当前 HEAD 已现场确认它生成
canonical `for/vla/compare/ptr_add/store`。正式性能 CSV 记录 SG2044 为 `6.346109 ms`；另一次
现场重跑得到 `6.313408 ms`，后者不属于正式 CSV。这不是旧 reader 或兼容路径。

### Reduction state placement

`reductionStatePlacement` 现在拥有两个真实实现：

```text
1 → scalar state
2 → vector state
0 → 根据 typed reduction facts、VLEN 与 resource budget 选择
```

ordered reduction 不允许偷偷改成 vector carry。选择进入 state register groups 与跨 strip handoff，
emitter 不再重新判断。候选试跑也说明它不是占位字段：SG2044 RMSNorm 的 vector state 更快，
K1 RMSNorm 的 scalar state更快，而 K1 LayerNorm 又更接近 vector state。

### 修复 VLA lifetime 失真

旧 lifetime snapshot 会把 descendant region 内定义的 value 同时计入 parent block snapshot，造成
并不存在的并发 live values，进而把一些 VLA 错压到较小 LMUL。第二轮改为：

- parent snapshot只统计该层定义或真正 captured/carry 的 value；
- descendant local definitions 在其自己的 block 中计数；
- materialized storage reuse 不重复占用 vector live range；
- mask/index/narrow/lookup temporary 继续进入资源预算。

这项修复让 ROIAlign 重新获得合法的宽 RVV shape，而不是因为虚假的 parent lifetime 被迫使用
窄 shape。正式数字为 SG2044 `51.946855 ms`、K1 `153.281305 ms`；它消除了中间实现中约
`171 ms` 的 SG2044 异常退化，但最终 CSV 相对上一轮只有正常运行波动。

### Q4 N16×K32 共享 RVV leaf

第二轮最大的确定性能收益来自 affine/symmetric i4×i8 local primitive 的实现重构。

旧实现的问题是：

- 每个 output column 建立 `decoded[32]` 临时数组；
- 对同一 activation block 为多个 column 重复加载；
- nibble decode、zero-point 与 scale 处理没有形成跨 N16 的寄存器组织；
- structured primitive 已经表达 N16×K32，但 emitter 输出仍接近逐元素实现。

新实现沿 N16 vectorize：

- stride-8 读取 packed weights；
- vector nibble decode 与 affine zero-point；
- 同一 activation scalar在 16 个输出之间复用；
- 使用一个 i32m4 accumulator覆盖 16 个输出；
- vector f16 scale widening 后累加到 f32 result；
- symmetric 与 affine 只在局部数值关系上不同，共享相同 physical organization；
- RVV eligibility 要求 e8m1/e16m2/e32m4 与 widening 能力，IME eligibility 由 K1 profile单独给出。

它仍然只是当前 `affine/symmetric i4×i8 N16×K32` primitive 的 leaf，不拥有 projection、MoE
grouping、outer traversal、activation quantization、workspace 或 ABI。因此同一项改动自然同时
改善 Q4 projection 与 `q4_k_mul_mat_id`，没有增加 kernel-name path。

### 哪些候选是真的，哪些仍不是

| 能力 | 当前状态 |
|---|---|
| VLA data LMUL、index/mask shape、memory mode | 多个合法 shape，由 target/resource facts 过滤 |
| F32 dot LMUL × K-unroll | 真实多候选 |
| F16 matmul input LMUL、row microtile、K-unroll、pipeline stages | 真实多候选 |
| reduction state scalar/vector placement | 真实多候选 |
| narrow LMUL、sort radix | 真实多候选 |
| RVV/IME Q4 realization | target-specific legality choice，不是同 target 上的实测调优维度 |
| Q4 N16×K32 microtile | 目前固定为该 primitive 的一个 leaf shape |
| E2M1/grouped affine VLEN128/scalable | 主要按 VLEN 阈值选择，候选空间仍窄 |
| sign-bit i8 dot | 当前只有一个 widening/sign-sum realization |
| prefetch | 没有第二个真实实现，已从 config 与文档删除 |

这一区分很重要：第二轮没有用“有字段”伪装已支持 tuning，也没有为了性能把 LMUL、fragment
或 backend 名写进 DSL。

### 第二轮相对第一轮的性能变化

149 条记录中：

| 统计 | 数量 |
|---|---:|
| 至少快 10% | 15 |
| 位于 ±5% | 102 |
| 至少慢 10% | 16 |
| SG2044 median time 几何平均 | 0.966×，约快 3.4% |
| K1/X60 median time 几何平均（RVV+IME 76 行） | 0.981×，约快 1.9% |

主要改善：

| target | kernel | 第一轮 ms | 第二轮 ms | 提升 |
|---|---|---:|---:|---:|
| SG2044 | Q4_K MoE `mul_mat_id` | 2654.656934 | 794.376330 | 3.34× |
| SG2044 | Q4_0 projection RVV | 21.510814 | 6.578790 | 3.27× |
| SG2044 | Q4_K projection RVV | 20.655649 | 6.458929 | 3.20× |
| K1 | Q4_0 projection RVV | 44.787575 | 21.055614 | 2.13× |
| K1 | Q4_K projection RVV | 42.869242 | 21.202069 | 2.02× |
| K1 | Q4_K MoE RVV | 5256.822103 | 2697.581750 | 1.95× |
| K1 | SSM convolution | 51.953546 | 30.041132 | 1.73× |
| K1 | RMSNorm | 1.485905 | 1.090921 | 1.36× |
| K1 | LayerNorm | 1.558717 | 1.221461 | 1.28× |
| K1 | transposed convolution | 257.098621 | 201.537436 | 1.28× |
| SG2044 | SSM convolution | 3.521075 | 2.857493 | 1.23× |

主要退化：

| target | kernel | 第一轮 ms | 第二轮 ms | 退化 |
|---|---|---:|---:|---:|
| K1 | F16 GEMM prefill | 2220.879517 | 3701.324668 | 1.67× |
| K1 | CSR SpMV | 97.693963 | 162.548515 | 1.66× |
| K1 | GroupNorm | 5.166033 | 7.872758 | 1.52× |
| SG2044 | Q8_0 activation quantize | 2.707192 | 3.805178 | 1.41× |
| SG2044 | col2im 1D | 21.625475 | 29.340333 | 1.36× |
| SG2044 | RWKV WKV7 | 32.417120 | 41.032526 | 1.27× |
| SG2044 | F32 prefill local dot | 923.112699 | 1155.398042 | 1.25× |
| K1 | Q8_0 activation quantize | 5.192659 | 6.392418 | 1.23× |
| K1 | codebook lookup | 4.415452 | 5.414964 | 1.23× |

这说明第二轮不是“所有东西都更快”。Q4 leaf 的改善可以直接归因到共享实现；其余大幅退化
必须作为下一轮 physical selection/resource/schedule 的压力点，不能用单 kernel branch 隐藏。

## 当前性能相对 GGML baseline

以下比率只用于定位差距。它们遵守同硬件、同算法、同 shape，但仍有前述 warmup、repetition、
preprocess scope 差异。

### SG2044 / RVV VLEN128

| kernel | Weft | GGML | 相对速度 |
|---|---:|---:|---:|
| F16 GEMM decode | 2.015 GOP/s | 4.542 GOP/s | 0.44× |
| F16 GEMM prefill | 6.140 GOP/s | 14.358 GOP/s | 0.43× |
| F32 local-dot decode | 2.299 GOP/s | 1.628 GOP/s | 1.41× |
| F32 local-dot prefill | 3.717 GOP/s | 7.480 GOP/s | 0.50× |
| Q4_K projection | 5.195 GOP/s | 8.923 GOP/s | 0.58× |
| Q4_0 projection | 5.100 GOP/s | 2.456 GOP/s | 2.08× |
| Q4_K×Q8_K FFN helper | 9.590 GOP/s | 9.028 GOP/s | 1.06× |
| Q1_0×Q8_0 | 2.063 GOP/s | 2.265 GOP/s | 0.91× |
| MXFP4×Q8_0 | 5.886 GOP/s | 6.702 GOP/s | 0.88× |
| Q6_K×Q8_K | 6.159 GOP/s | 4.895 GOP/s | 1.26× |
| IQ2_S×Q8_K | 1.756 GOP/s | 2.393 GOP/s | 0.73× |
| IQ3_S×Q8_K | 1.746 GOP/s | 0.823 GOP/s | 2.12× |
| IQ1_M×Q8_K | 0.477 GOP/s | 4.591 GOP/s | 0.10× |
| Q8_0 activation quantize | 482.240 MElements/s | 557.451 MElements/s | 0.87× |
| IQ4_NL row dequantize | 735.370 MElements/s | 161.431 MElements/s | 4.56× |
| SiLU | 4.167 ms | 4.113 ms | 0.99× |
| RMSNorm | 1.014 ms | 2.637 ms | 2.60× |
| Softmax | 4.520 ms | 4.170 ms | 0.92× |
| contiguous transpose | 4.128 ms | 5.574 ms | 1.35× |
| RoPE | 1.118 ms | 1.253 ms | 1.12× |
| flash attention | 38.478 ms | 21.667 ms | 0.56× |
| get_rows Q4_K | 0.907 ms | 0.825 ms | 0.91× |

SG2044 的结论不是简单的“RVV 做得好”或“不好”：F32 decode、Q4_0、Q6、IQ3、IQ4 decode、
RMSNorm 与 transpose 已有竞争力；F16 blocked GEMM、F32 prefill、Q4_K、IQ1_M 与 flash attention
仍缺少适合该结构的物理实现。

### K1/X60 / RVV VLEN256

| kernel | Weft | GGML | 相对速度 |
|---|---:|---:|---:|
| F16 GEMM decode | 0.831 GOP/s | 3.429 GOP/s | 0.24× |
| F16 GEMM prefill | 1.160 GOP/s | 5.488 GOP/s | 0.21× |
| F32 local-dot decode | 1.675 GOP/s | 2.007 GOP/s | 0.83× |
| F32 local-dot prefill | 0.791 GOP/s | 2.673 GOP/s | 0.30× |
| Q4_K projection RVV | 1.583 GOP/s | 9.766 GOP/s（GGML IME） | 0.16×† |
| Q4_0 projection RVV | 1.594 GOP/s | 11.409 GOP/s（GGML IME） | 0.14×† |
| Q4_K×Q8_K FFN helper | 0.911 GOP/s | 2.583 GOP/s | 0.35× |
| Q1_0×Q8_0 | 1.779 GOP/s | 3.773 GOP/s | 0.47× |
| MXFP4×Q8_0 | 0.649 GOP/s | 2.587 GOP/s | 0.25× |
| Q6_K×Q8_K | 2.231 GOP/s | 2.384 GOP/s | 0.94× |
| IQ2_S×Q8_K | 1.530 GOP/s | 1.461 GOP/s | 1.05× |
| IQ3_S×Q8_K | 0.122 GOP/s | 1.664 GOP/s | 0.07× |
| IQ1_M×Q8_K | 0.360 GOP/s | 1.436 GOP/s | 0.25× |
| Q8_0 activation quantize | 287.060 MElements/s | 330.016 MElements/s | 0.87× |
| IQ4_NL row dequantize | 270.630 MElements/s | 155.008 MElements/s | 1.75× |
| SiLU | 13.544 ms | 13.233 ms | 0.98× |
| RMSNorm | 1.091 ms | 1.699 ms | 1.56× |
| Softmax | 8.184 ms | 8.349 ms | 1.02× |
| contiguous transpose | 20.310 ms | 4.251 ms | 0.21× |
| RoPE | 4.230 ms | 5.237 ms | 1.24× |
| flash attention | 71.404 ms | 44.422 ms | 0.62× |
| get_rows F32 | 0.677 ms | 0.691 ms | 1.02× |
| get_rows Q4_K | 2.123 ms | 3.362 ms | 1.58× |

`†` 两行是 Weft RVV 与 GGML IME 的同算法/shape参考，不能解释成 RVV-vs-RVV。K1 的主要
问题不是 VLEN256 自动带来更高性能，而是当前 LMUL、register organization、memory schedule 与
quant leaf 没有充分利用 K1 的目标事实。F16/F32 prefill、transpose、IQ3 和 flash attention 是最
明显的例子。

### K1/X60 / IME1

严格可对应的 production projection 为：

| kernel | Weft IME | GGML IME | 相对速度 | Weft max abs error |
|---|---:|---:|---:|---:|
| Q4_K projection | 8.412 GOP/s | 9.766 GOP/s | 0.86× | `3.33786011e-06` |
| Q4_0 projection | 10.541 GOP/s | 11.409 GOP/s | 0.92× | `0.0312509537` |

同一 Weft DSL 与 shape 在 K1 上的 target realization差异为：

| kernel | RVV ms | IME ms | IME 相对 RVV |
|---|---:|---:|---:|
| Q4_K projection | 21.202069 | 3.988747 | 5.32× |
| Q4_0 projection | 21.055614 | 3.183107 | 6.62× |
| Q4_K MoE `mul_mat_id` | 2697.581750 | 518.030367 | 5.21× |

这三项证明 IME 是同一 local primitive 的 target realization，而不是另一套 kernel：outer
traversal、activation quantization、grouping、workspace、persistent weights 与 ABI 都仍来自 DSL。
但当前 fragment family、packing 与 resource model仍较固定，尚未形成多个 IME fragment candidate。

## 当前真正闭合的能力

1. **唯一主链**：Python DSL 与外部前端都只能进入 canonical Kernel IR，再进入同一个 RISC-V
   lowering；不存在 GGML/materials runtime 调用或旧 emitter fallback。
2. **逐实体事实**：VLA、memory、predicate、state、dot/matmul、lookup、quant 与 extension各自
   产生 typed facts，不按 kernel 名或 op 数量分类。
3. **连接的 value chain**：physical value、handoff、reload/rematerialize 与 lifetime 由普通
   use-def和control carry共同决定。
4. **目标相关合法性**：SG2044 与 K1 读取同一 target profile接口，根据 VLEN、SEW/LMUL、
   widening、indexed/segment memory 与 IME facts形成不同实现。
5. **若干真实 candidate family**：F32 dot、F16 matmul、VLA LMUL、reduction state、narrow 与
   sort已有多个能生成和执行的候选，而不是文档字段。
6. **共享高性能 leaf**：Q4 N16×K32 的一次局部改进同时服务 projection和MoE，没有恢复格式
   route或whole-kernel emitter。

## 仍未闭合的能力与后续判断依据

### 1. Dense microkernel 与 pipeline

F16/F32 prefill在两台机器均显著落后，K1尤其严重；同一轮中 SG2044 F16 prefill略有改善，
K1却出现大幅退化。说明 row microtile、LMUL、K-unroll与pipeline虽已有候选，但 resource/cost
排序和 target-specific winner仍不成熟。Dense conv、OutProd等局部改善不能替代这项闭合。

### 2. Quant candidate space

Q4 N16×K32 已有高质量 leaf，但仍固定一个 microtile；E2M1/grouped affine主要按 VLEN阈值
切换，sign-bit只有单一路径，若干 codebook leaf的resource budget仍是手工常数。IQ1_M、K1
IQ3与MXFP4的巨大差距说明 decode、gather、widen、accumulator与cross-output reuse尚未成为统一
候选空间。

### 3. State、summary 与 nested control resource model

reduction state placement已真实可选，但 segmented scan、flash attention、GroupNorm和RWKV的
退化说明 state/lookup/nested-control temporary还没有全部进入统一 live interval和schedule模型。
当前资源式对部分 state、lookup和primitive temporary仍使用汇总group数，缺少候选失败后的回溯
重选。

### 4. Indexed/irregular memory

unit/strided/indexed/segment memory已有统一事实和合法性，但K1 transpose、CSR与部分col2im仍明显
落后。这表明“识别出 indexed/strided”不等于拥有成熟的index资源、load ordering、prefetch、
coordinate hoist和window reuse空间。当前没有真实第二个prefetch实现，所以本轮正确地删除了
prefetch声明，而不是保留假能力。

### 5. Target profile粒度

当前profile能承载VLEN、SEW/LMUL、vector register、widening、indexed/segment与IME事实，但对
标准V子能力仍较粗；部分能力由full `V`整体打开。下一阶段若继续扩展target，应该细化同一
profile中的local legality，而不是为SG2044、K1或新扩展建立独立编译主链。

### 6. 性能协议

Weft当前表和固定GGML baseline的warmup/repetition不完全一致。当前比率足够发现数量级问题，
不足以为接近1.0的项目下最终胜负结论。后续若要评价“已经追平”，必须在不改变算法、shape、
线程数、preprocess scope和timed region的条件下统一协议；否则只能保留方向性表述。

## 关于未修改 examples 与 `/tmp/weft-kernel.*`

这两轮开始前，全仓已经完成 Core-local blocked DSL切换。后续compiler重构不要求无关DSL
文件产生mtime变化：例如causal mask的作者程序和语义没有变化，但相同Kernel IR会由新的
physical lowering重新生成不同的intrinsic C。

每次 `examples/run/weft.sh` 都会：

```text
当前 examples/kernels/*.py
→ 当前 Python frontend
→ 当前 weft-compile
→ 新建 /tmp/weft-kernel.XXXXXX
→ 写入本次 kernel.c/header/object/runtime
→ 目标机编译、执行
→ 退出时删除本次临时目录
```

`/tmp/weft-kernel.*` 不是第二份DSL kernel或缓存authority。旧runner使用`exec taskset`，导致
远端EXIT trap无法执行；本轮现场记录显示，历史运行曾留下SG2044 2358个、K1 1459个临时
目录，该计数不是仓库产物。本轮已改为普通`taskset`调用；修改后的causal mask现场重跑前后
SG2044目录数没有增加，说明本次新目录已在退出时删除。历史目录不参与当前编译。

## 总判断

两轮之后，Weft已经不再只是“多个kernel各有一段能跑的RVV发射代码”。它具备一套连接
logical axis、value use、memory relation、lifetime、resource和target extension的物理编译核心，
并且Q4、state placement、F32/F16 candidate等能力已经产生可测量的共享结果。

当前最重要的事实同样明确：

> **编译器结构已经跨过了模式库阶段，但物理候选空间与性能选择尚未完成闭合。**

下一阶段的判断不应是继续增加kernel数量，而应看同一项shared physical ability能否同时解决：

- F16/F32 GEMM、conv、OutProd的microtile与pipeline；
- IQ/Q1/MXFP4的decode reuse与register organization；
- summary/state的跨strip placement与nested-control lifetime；
- K1 indexed/transpose/CSR的memory schedule；
- IME fragment与RVV handoff的多候选资源模型。

任何改善如果只能通过kernel名、格式route、完整closure或whole-kernel leaf成立，都不算这套
编译核心继续成熟。
