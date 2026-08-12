# 十个事后选择的陌生 Kernel

## 结论

这一轮先锁定十个此前没有围绕 Weft lowering 设计过的算法，再接入编译器。十项均已通过同一条

```text
Python DSL -> canonical Kernel IR -> RISC-V physical decision -> intrinsic C / local asm -> system compiler
```

在 SG2044（RV64GCV，VLEN=128）上完成真实编译、全输出数值对照和冷缓存计时。新增能力没有按 kernel 名、量化格式或完整 region 形状接管 outer traversal；新发射入口分别由 typed `contract` facts 和四个局部 packed-dot primitive 驱动。

这轮最重要的变化不是多了十个可运行 case，而是两个此前过窄的边界被陌生程序实际打破：

1. `W.contract` 不再只接受 `[BM,K] x [VLA,K]`、零 init、固定 RHS-vector 的局部 closure；现在自然的 `[VLA,K] x [K] -> [VLA]`、Region init 与有序 outer context 也由同一 contract candidate family lowering。
2. IQ2_S、IQ3_S、IQ1_M、Q6_K 的 decode、lookup、scale 与 i8 dot 被建模为四个有明确 typed operand 合同的 256-element local semantic primitive；外层 block traversal、persistent packed layout 和 kernel ABI 仍由 DSL 拥有。

## 事后锁定的名单

名单在检查 Weft 能否处理之前确定，此后没有换题。

| Kernel | 真实来源类别 | 用来施压的组合 |
|---|---|---|
| Gated Linear Attention | 线性注意力模型 | ordered state + VLA projection |
| RWKV-WKV7 | RWKV-7 | 多项 sequential state + VLA |
| Gated Delta Net / KDA | delta-rule 模型 | gate + state + projection |
| Batched lower-triangular solve | CPU linear algebra | ordered traversal + contract in unfamiliar outer context |
| GroupNorm | 视觉模型 | 多层 reduction + VLA normalization |
| SAM relative-position add | Segment Anything | indexed relation + irregular scatter effect |
| IQ2_S x Q8_K | GGML quantized inference | packed codebook + sign + scale + compute |
| IQ3_S x Q8_K | GGML quantized inference | packed lookup + sign + compute |
| IQ1_M x Q8_K | GGML quantized inference | codebook + delta + packed scale |
| Q6_K x Q8_K | GGML quantized inference | narrow decode + grouped scale + local asm leaf |

前三个 kernel 的 token/head/width traversal、triangular solve 的 batch/row traversal、SAM 的 coordinate traversal，以及四个量化 kernel 的 256-element block traversal，全部保留在作者 DSL 中。Target 没有替换这些算法结构。

## 被迫扩大的共享编译能力

### Contract 的方向、init 和 candidate 不再由 emitter 猜

Triangular solve 自然写成：外层 batch 和 row 是作者指定的有序循环，`W.vla` 授权 RHS column 轴向量化，`W.block_axis(row)` 定义已经求解的前缀，`W.contract(previous, coefficient, init=rhs/diagonal)` 只授权局部 reduction domain。

新的 contract decision 在分析阶段一次性物化：

```text
free vector operand
blocked scalar operand
shared reduction axis and extent
Region init realization
unit/strided memory form
LMUL
K-unroll
output memory form
absorbed local closure
```

Emitter 只按照这些字段生成 vector load、scalar projection、FMA 和 store，不再从 operand 的周围 source closure 重新选择方向、memory mode 或 microtile。原来的 `[BM,K] x [VLA,K]` realization 仍由同一 `ContractOp` typed facts 进入；新增方向不是 triangular-solve special path。

同一个 triangular program 对合法候选做了真实构建期实测。LMUL4/K1 为本次默认最快结果；K2、K4 与 LMUL1/2/4 的其他组合也都通过数值对照，但更慢。候选维度因此是可执行的物理空间，而不是报告中的名义枚举；结果没有固化为 kernel 专用规则。

| contract config | median ms | GOP/s |
|---|---:|---:|
| LMUL4, K-unroll 1 | 5.854985 | 5.730917 |
| LMUL4, K-unroll 2 | 6.306948 | 5.320233 |
| LMUL4, K-unroll 4 | 6.863750 | 4.888644 |
| LMUL1, K-unroll 4 | 20.022708 | 1.675819 |
| LMUL2, K-unroll 4 | 10.624787 | 3.158127 |

最终采用严格双指标 gate 重测时，triangular solve 的记录值为 5.818646 ms、5.766708 GOP/s。

### Packed/lookup compute 成为局部 primitive

四个新增 Extension op 分别完整描述一块 packed weight 与一块 Q8_K activation 的数值关系：

```text
iq2_s_i8_dot
iq3_s_i8_dot
iq1_m_i8_dot
q6_k_i8_dot
```

它们的 verifier 固定的是局部格式本身所定义的字段类型、256-element extent，以及该 op 必须处在 canonical Kernel 中；它不读取 kernel 名、outer loop 数量或完整 use closure。Target decision 只读取这些 typed blocks、all-active local loads、显式 block axis、target endian/VLEN facts 和 scalar operands；emitter 消费已经选择的 local-block realization及其 op 对应的 typed intrinsic spelling。

IQ2_S、IQ1_M 使用 RVV gather/widen/dot，IQ3_S 使用局部 decoded buffer 加 RVV dot；Q6_K 使用一个只拥有单个 256-element block 的 inline-asm leaf。它们都不拥有 outer block loop、完整 kernel ABI 或 persistent packing，也不调用 GGML 或 `materials/`。代码表和 RVV/asm 组织是从成熟实现重新吸收进 production lowering 的本地实现知识。

## 最终有效重测

每个数字都来自对应 kernel 最后一次有效执行。每次样本前触碰 64 MiB eviction buffer；六个通用 kernel 取 10 次中位数，四个全投影量化 kernel 取 3 次中位数。Correctness 覆盖 runtime 声明的全部输出；量化四项比较全部 14336 rows，而不是抽样行。

| Kernel | shape | realization | error（abs / rel） | median | throughput |
|---|---|---|---:|---:|---:|
| Gated Linear Attention | tokens=128, heads=32, width=64 | RVV intrinsic C | 6.75209e-09 / 5.53982e-04 | 16.416653 ms | 5.109816 GOP/s |
| RWKV-WKV7 | tokens=128, heads=32, width=64 | RVV intrinsic C | 1.97906e-09 / 7.25251e-04 | 40.087257 ms | 4.185174 GOP/s |
| Gated Delta Net / KDA | tokens=128, heads=32, width=64 | RVV intrinsic C | 1.86265e-09 / 6.90221e-05 | 58.359477 ms | 3.737247 GOP/s |
| Batched triangular solve | batches=8, rows=256, rhs=64 | RVV contract vector-dot | 1.19209e-07 / 9.50217e-05 | 5.818646 ms | 5.766708 GOP/s |
| GroupNorm | N=1, C=320, H=W=64, groups=32 | RVV intrinsic C | 5.96046e-07 / 7.37177e-05 | 3.397995 ms | 385.733352 MElements/s |
| SAM relative-position add | patches=64, query=8x8, key=14 | RVV intrinsic C | 7.45058e-09 / 4.65661e-04 | 3.808197 ms | 1.806965 GB/s |
| IQ2_S x Q8_K | M=1, N=14336, K=4096 | RVV local codebook dot | 0 / 0 | 65.380747 ms | 1.796255 GOP/s |
| IQ3_S x Q8_K | M=1, N=14336, K=4096 | RVV local codebook dot | 0 / 0 | 68.639682 ms | 1.710971 GOP/s |
| IQ1_M x Q8_K | M=1, N=14336, K=4096 | RVV local codebook dot | 0 / 0 | 37.359844 ms | 3.143496 GOP/s |
| Q6_K x Q8_K | M=1, N=14336, K=4096 | RVV local asm block dot | 0 / 0 | 19.161684 ms | 6.128924 GOP/s |

四个量化 kernel 可与固定 GGML baseline 做同 shape、同 SG2044 比较：

| Kernel | Weft GOP/s | GGML GOP/s | Weft / GGML |
|---|---:|---:|---:|
| IQ2_S x Q8_K | 1.796255 | 2.393391 | 0.751x |
| IQ3_S x Q8_K | 1.710971 | 0.822512 | 2.080x |
| IQ1_M x Q8_K | 3.143496 | 4.591404 | 0.685x |
| Q6_K x Q8_K | 6.128924 | 4.895281 | 1.252x |

IQ2_S 和 IQ1_M 的差距仍分别指向共享的 codebook gather/scale accumulation 与 register organization，不是 correctness 或 fallback 问题；没有用 kernel-specific whole-loop code 掩盖它们。IQ3_S 和 Q6_K 已在本次口径下超过对应 GGML baseline。

## 架构检查

Production path 中没有新增以下入口：

```text
kernel-name matcher
q-format string route
exact op-count matcher
whole-region fast path
GGML/materials runtime call
legacy or scalar fallback
emitter-side physical reselection
```

四个量化 kernel 在手工 repro 层按名字选择要运行的 DSL entry 和 reference，这是运行入口，不进入 target lowering。GGML header 只被该独立 correctness harness 用来取得 reference 数据布局和静态 codebook；生成的 kernel object 不 include、link 或调用 GGML。

当前十项均没有 target unsupported 阻塞。性能仍未进入竞争区间的明确剩余项是 IQ2_S 与 IQ1_M；它们对应的是可复用 local packed-dot candidate 的实现质量，而不是缺少 production 主链。
