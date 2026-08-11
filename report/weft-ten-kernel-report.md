# 十个真实 Kernel 之后的 Weft

这份报告只描述当前 Weft 实现事实：十个真实 kernel 逼出了什么能力、重构收敛到什么
程度，以及本轮真机性能。它不是设计规范、性能门槛或自动检查输入。

`report/baseline/` 中的 GGML/RISC-V baseline 保持不变；本文只读取其中与当前 workload
同硬件、同算法和同 shape 的数字作为对照。Weft 自身的原始记录单独放在
[`weft-kernel-performance.csv`](weft-kernel-performance.csv)。

## 结论

十个 kernel 已经全部通过同一条实际执行链：

```text
Weft Python DSL
→ canonical worker-local Kernel IR
→ direct RISC-V target lowering
→ RVV intrinsic C / typed inline-asm leaf
→ target system compiler
→ SG2044 或 K1/X60 executable
```

这条链已经不再是图编译器，也不再是旧的 selected IR、provider、route 或 whole-kernel
emitter。算法的 outer loop、persistent layout、staging、state 和数值关系由 DSL/Kernel IR
显式拥有；target 只实现局部 primitive，并决定动态 `vl`、LMUL、register organization、
microtile 和指令形式。

从性能看，旧 Q4_K 路径约 76 倍的落后已经消失。当前十项在对应固定 GGML 数字附近，
最慢的一项是 F16 prefill GEMM，时间高 10.6%；其余慢项在 8.3% 内，同时 RMSNorm、
Softmax、RoPE 和 IME 路径更快。这里可以说“已经进入竞争区间”，还不能说 target
lowering 已经具备 Triton 级通用性。

## 十个 kernel 逼出的能力

| Kernel | DSL / Kernel IR 必须显式承载的算法 | 被逼出的共享 lowering 能力 | 当前边界 |
|---|---|---|---|
| SiLU F32 | VLA logical domain、显式公式和 fast-exp 数值选择 | runtime `vl`、RVV f32 pointwise、局部 exp realization、unit-stride memory | generic VLA 仍主要是 all-active f32，LMUL 选择较固定 |
| RMSNorm F32 | row loop、第一遍 sum-square reduce、scalar normalization state、第二遍 scale | 跨 strip RVV reduction、scalar/RVV state handoff、同一 kernel 中多个 VLA | reduce 只覆盖有限 dtype/op/axis 组合 |
| Online Softmax F32 | `(maximum, scaled_sum)` tuple、lift/merge、rescale 和 preserve-order summary algebra | summary-state lowering、跨 strip max/sum、局部 exp 与 normalize fusion | 当前 summary lowering 只完整覆盖 online-softmax algebra |
| F16 dense GEMM | M/N/K cache loops、BM/BN/BK、masked logical blocks、contract axes、carried accumulator | F16 load、F32 accumulate、RVV register microtile、widening FMA、tail realization | target 仍匹配固定 4×8 source slice和较精确的 contract closure |
| Q4_K×Q8_K projection dot | 144/292-byte persistent ABI、outer block traversal、显式 scales、minimum correction 和 init | typed grouped affine i4×i8 local primitive、VLEN128 RVV asm leaf | primitive 固定 256-element grouped format与 VLEN128，但没有接管 outer loop |
| Q4_K IME projection | activation max-reduce、F32→I8 quantize、scale/code scratch、N/K loops、persistent K32×N16 layout | narrow/round/saturate、local affine i4×i8 contract、IME1 fragment与 accumulator realization | IME lowering 仍只覆盖 K1/IME1、VLEN256、N16/K32 结构，并知道过多具体 layout 关系 |
| Q4_K GetRows | token→row 间接索引、packed scale/min decode、nibble 解码和输出关系 | irregular pointer chain、block producer closure、RVV packed decode/store | 真实 gather 已运行，但尚未使用并逼通通用 `W.lookup` / `W.decode` lowering |
| Contiguous transpose | row outer loop与显式 source/destination affine index relation | 同一 VLA memory lowering中的 unit-stride load与strided store | 仍限于 all-active f32 VLA；不是通用 permute primitive |
| RoPE NeoX | position/head/pair loops、angle cache、顺序 theta carry、旋转公式 | scalar sin/cos与RVV VLA组合、scratch reuse、half-dimension strided access | trig 仍为 scalar realization，masked/vector trig 尚未覆盖 |
| FlashAttention F32/F16 | GQA mapping、causal bound、query scratch、key loop、online max/sum state、accumulator rescale | F32→F16、widening F16 dot、weighted update、normalize等多个局部 realization在同一 source 中组合 | 这些 realization 仍依赖较精确的局部 op-count和use relation |

这十项真正形成的不是十个“算子后端”，而是五组可以组合的承重能力：

1. worker-local scalar control、pointer、effect与普通 C ABI；
2. VLA memory/pointwise/reduction和跨 strip state；
3. logical block、explicit contract和register microtile lowering；
4. packed decode、irregular access和短生命周期 producer-consumer fusion；
5. typed extension primitive到RVV/IME inline-asm leaf。

十个 source 之间没有 kernel-kind 枚举；target lowering 也不根据 symbol、模型名或
q-format 选择整段实现。Python CLI 的 `--kernel` 只选择一个待编译的 source definition。
Q4_K 和 IME 使用 extension op，是因为它们具有普通 contract 无法完整表达的可观察
block-scale、minimum correction与packing语义；extension op只覆盖局部计算，外层算法仍在
Kernel IR 中。

## 重构完成了什么

### 表示与主链已经收敛

旧的 `Selection`、Execution/Selected IR、SourceEmitter 和 RVV route/provider production
stack 已从当前 compiler core 删除。当前长期 authority 只有：

```text
Python DSL source
    ↓ 直接构造
canonical Kernel IR
    ↓ 单次 target lowering
generated source / object-side artifact
```

Target capability、legality、物理配置和 emission 都只是一次 lowering 内部机制，不再各自
拥有 IR schema、front door或 verifier。`materials/` 没有进入 CMake、include、Python import、
link或 runtime；GGML只在 repro 中作为数值/性能对照，不是 Weft fallback。

### 根程序模型已经从图变成 worker-local kernel

Weft 编译一份完整 worker-local callable，而不是导入 graph、识别 operator、划分 program
grid或创建线程。线程池、work partition与 affinity由调用者负责。DSL作者显式决定：

- outer traversal与cache blocking；
- staging、scratch和persistent packing；
- pointer/index/mask/effect；
- contraction axes、state algebra与numerical mode；
- 需要哪个具有可观察语义的extension primitive。

Target lowering可以决定动态 `vl`、LMUL、K-unroll、register microtile、fragment和局部纯融合，
但不能从普通 multiply/add graph猜出GEMM、Softmax、Attention或量化格式。

### 当前仍不是“完整通用”的部分

- `RISCVLowering.cpp` 仍是集中式 emitter；共享 scalar/VLA/block machinery 已存在，但很多
  realization 还没有整理成清晰的 primitive-local family。
- Softmax、F16 GEMM、FlashAttention 和 IME 已经结构驱动、与 kernel 名无关，但 matcher
  仍要求较精确的 source closure。它们证明了主链可行，尚未证明同一 primitive 的广泛组合。
- Generic VLA 主要覆盖 all-active f32/f16；predicate、masked memory、dtype、reduce/scan
  和 LMUL 空间仍不完整。`scan` 目前没有十项中的真实 consumer。
- Canonical IR 已声明 lookup/decode/permute 等语义锚点，但 GetRows/Transpose 当前通过普通
  pointer、block和strided memory实现，因此这些一等 primitive的通用lowering尚未被逼通。
- IME路径是一个真实可执行leaf，但仍绑定 K1/IME1、VLEN256和特定 N16/K32 envelope；
  target 对persistent layout关系的认识仍然偏多。
- `weft-compile` 当前直接产出 canonical IR 或 intrinsic C；object/executable由同一 repro
  中的目标system compiler继续形成，library/header还没有成为完整 CLI artifact界面。

所以，重构的准确判断是：**旧架构已经被替换，唯一主链已经成立；下一阶段的主要问题不再
是“路径混乱”，而是把当前较窄的 structural slice扩展成真正可组合的 primitive-local
lowering。**

## 真机性能

所有 Weft 数字来自本轮重新执行的单 kernel repro。SG2044 使用标准 RVV、VLEN128；IME
项使用 K1/X60、VLEN256与 SpacemiT IME1。每个样本在 64 MiB cache eviction 后计时并取
中位数；IME 项另有三次显式 warmup。`Weft / GGML` 是执行时间比，低于 1 表示 Weft 更快。

| Kernel | Hardware | Shape / scope | Weft ms | 固定 GGML ms | Weft / GGML | 读数 |
|---|---|---|---:|---:|---:|---|
| SiLU F32 | SG2044 | `ffn[128,14336]` | 4.197 | 4.113 | 1.021 | 同档，慢 2.1% |
| RMSNorm F32 | SG2044 | `hidden[128,4096]` | 1.601 | 2.637 | 0.607 | 快 39.3% |
| Softmax F32 | SG2044 | `heads=32,Q=128,K=128` | 2.646 | 4.170 | 0.634 | 快 36.6% |
| F16 GEMM decode | SG2044 | `M=1,N=4096,K=4096` | 7.222 | 7.387 | 0.978 | 同档，快 2.2% |
| F16 GEMM prefill | SG2044 | `M=128,N=4096,K=4096` | 330.751 | 299.138 | 1.106 | 慢 10.6% |
| Q4_K×Q8_K projection | SG2044 | `M=1,N=14336,K=4096` | 12.687 | 13.009 | 0.975 | 固定表快 2.5%；同进程 GGML 12.257 ms，Weft 慢 3.5%，属于同档波动 |
| Q4_K GetRows | SG2044 | `tokens=128,hidden=4096` | 0.891 | 0.825 | 1.080 | 慢 8.0% |
| Contiguous transpose | SG2044 | `128×4096` | 6.034 | 5.574 | 1.083 | 慢 8.3% |
| RoPE NeoX | SG2044 | `tokens=128,heads=32,D=128` | 0.995 | 1.253 | 0.794 | 快 20.6% |
| FlashAttention F32/F16 | SG2044 | `Q=128,KV=128,H=32,Hkv=8,D=128` | 23.462 | 21.667 | 1.083 | 慢 8.3% |
| Q4_K IME projection | K1/X60 | `M=1,N=4096,K=4096`，含activation quantize | 3.257 | 3.436 | 0.948 | 快 5.2% |

性能含义很明确：

- 十项已经没有数量级落后；RVV/IME codegen确实进入了成熟手写实现的性能区间。
- Q4_K×Q8_K和IME计时的Weft分支不调用GGML或materials；Q4_K runtime中的GGML调用
  只用于同进程 comparator，计时的Weft实现是生成的RVV/IME leaf。
- Dense prefill是当前相对差距最大的一项；其余慢项都在8.3%以内。
- RMSNorm、Softmax和RoPE的领先只说明当前shape与实现下的实际读数，不自动推出所有shape
  或所有数值模式都领先。
- F16 GEMM与FlashAttention的correctness runtime采用代表点采样；其他八项为全输出对照。

这张表和 CSV 都只是当前数字记录，不替代固定 GGML baseline，也不附带同步、版本跟踪、
阈值判断或任何校验逻辑。
