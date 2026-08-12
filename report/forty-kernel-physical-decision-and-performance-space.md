# 四十个 Kernel 的物理决定与性能空间

## 结论

本轮没有增加 kernel，也没有改写现有 DSL。固定语料仍是四十个 distinct kernel；F16 与
F32 GEMM 分别记录 decode、prefill，所以最终真机执行为四十二条 phase。

这轮完成了两件实际工作：第一，主要高性能路径的 physical decision 已在 C 发射之前由同一
次 target lowering 产生；第二，VLA 与 contract 开始拥有可枚举、受资源约束的物理候选，
并用真机数据修正了“合法最大 LMUL 就是最快”的错误假设。最终四十二条均由现有单-kernel
runtime 重新执行，数字写入 `weft-kernel-performance.csv`。

当前形态更接近：

```text
natural Weft DSL
→ canonical Kernel IR
→ typed facts + target facts + explicit backend config
→ transient physical decisions
→ intrinsic C / local asm spelling
→ system compiler
```

它仍不是完成态的 RISC-V Triton：若干 local fusion envelope 仍窄，contract、state placement、
memory scheduling 与 extension fragment 的候选空间也仍不足。

## 编译器责任如何收敛

`preparePhysicalDecisions()` 现在先遍历并准备：

- 每个 VLA region 的 predicate、access、cast、binary、state、narrow、contract 与 physical
  vector shape；
- local F32 contract 的 row/reduction axes、typed producers/consumer、LMUL、microtile 与
  K-unroll；
- F16 contract 与 affine IME source loop 所拥有的 local physical realization；
- symmetric i4×i8、grouped affine i4×i8、sign-bit×i8、E2M1/E8M0×i8 等 extension primitive
  的 operand/layout/fragment decision。

对应 emitter 只按 owning operation 查表。Online softmax 的 state-consumer decision 也不再
保存提前拼出的 C expression；它保存 canonical bound、pointer 与 coordinate value，发射时才
机械投影成 intrinsic-C spelling。

这没有新增 selected IR、Physical IR、provider registry、validator 或长期 pipeline stage。
Decision 仍是一次 target lowering 内的短生命周期 C++ 对象；长期算法 authority 仍只有
Kernel IR。

当前还没有全部前移的是 block decode、block store grouping 与 block reduce grouping。这三类
仍在 block emission 入口根据已经收集的 local block closure 产生 strip decision。它们没有
第二条 fallback，也不读 kernel 名，但代码责任边界尚未与 VLA、contract、extension primitive
完全一致。

## VLA 候选空间

普通 F32 VLA 不再固定 LMUL2，也不再默认从 LMUL8 开始。Lowering 先从每个 region 的实体
事实形成候选依据：

- load/store 的 element type、unit/strided memory relation 与 activity；
- reduce、scan、coordinate summary、online summary 的 state algebra；
- exp realization 对 F32m2 的当前约束；
- nested scalar loop 与每个 operation 的峰值 F32 vector footprint；
- XLEN、index LMUL、predicate、VLEN profile 与 vector-register resource。

随后在 LMUL1/2/4/8 中按语义类别排序，再用寄存器资源与 index/mask 合法性过滤。显式
`--vla-lmul` 走同一合法性检查，供构建期重复编译和实测；它不是另一条生产路径。

候选扫描证明不存在一个全局最优 LMUL：

| 结构事实 | 实测最优或当前选择 | 代表结果 |
|---|---:|---:|
| ordered inclusive scan | LMUL1 | Cumsum 0.371852 ms |
| coordinate argmax summary | LMUL4 | Argmax 16.376911 ms |
| exp / online summary | LMUL2 | Softmax 2.643862 ms |
| multiple reduction states | LMUL4 | RMSNorm backward 5.914966 ms |
| nested channel-window arithmetic | LMUL8 | Depthwise Conv2D 2.889792 ms |
| nested recurrent vector state | LMUL4/8 by region | RWKV 14.915945 ms |

这组选择不读取 symbol、算子名字或 q-format。它仍是有限 target heuristic，不是已经完成的
自动调优器；尤其普通无状态 VLA 还缺更好的 scalar-index provenance、cache locality 与真实
live-range cost，因此 ADD_ID 的默认值仍不是候选扫描中的最快值。

## Contract 与 extension 的候选边界

F32 local-row 与 VLA-free-axis contract 共用 register-resource selector。LMUL 候选会随 row
microtile 与 streamed operands 的寄存器组数变化；K-unroll 接受 1/2/4，但本轮真机扫描中
2 与 4 对现有 F32 contraction 都明显退化，所以默认仍选择 1。这个维度仍存在，失败候选没有
被改成另一套 emitter。

F16 GEMM 暴露 row microtile 与 input LMUL，但当前默认仍取 source row tile 与 LMUL1；实测
LMUL2、较小 row microtile 均未胜出。N/K blocking、BM/BN/BK 与 outer traversal继续属于作者
程序，target 只决定 contract 内的 register realization。

Narrow 支持 source LMUL4/8 的合法选择，但默认仍为 LMUL4。IME1、sign-bit 与 E2M1 local leaf
目前各只有一个合法 fragment family；这些已成为准备好的 physical decision，但还不能称为
多 fragment candidate space。Online softmax 的 state placement也仍只有 stack scratch。

## 四十二条最终执行

最终执行使用 CSV 已记录的模型级 shape、repetition、eviction 与 correctness scope。SG2044
执行 RV64GCV/VLEN128 项；K1/X60 执行 IME1/VLEN256 项。四十二条命令全部成功退出；runtime
输出中的 median、throughput 与显式 correctness 数值已逐行转录并复核。`sampled` scope 仍只
表示该 runtime 的抽样正确性，不应解释成 full-output 比较。

相对本轮开始时的 Weft 数字，42 条 phase 的几何平均耗时比为 0.958，即约 4.2% 改善；
10 条改善超过 5%，2 条回退超过 5%，其余 30 条在 ±5% 内。

| Kernel / phase | 之前 ms | 当前 ms | 变化 |
|---|---:|---:|---:|
| Depthwise Conv2D | 8.803279 | 2.889792 | 快 67.2% |
| MUL_MAT_ID | 1012.079609 | 792.053311 | 快 21.7% |
| RMSNorm | 1.608407 | 1.365566 | 快 15.1% |
| Q8_0 quantize | 4.140899 | 3.526595 | 快 14.8% |
| Bilinear upscale | 6.623429 | 5.897086 | 快 11.0% |
| LayerNorm | 1.601767 | 1.442806 | 快 9.9% |
| Top-K | 3.181154 | 2.905553 | 快 8.7% |
| RMSNorm backward | 6.363108 | 5.914966 | 快 7.0% |
| RWKV-WKV6 | 15.904670 | 14.915945 | 快 6.2% |
| Cumsum | 0.392702 | 0.371852 | 快 5.3% |
| F16 GEMM prefill | 341.452654 | 379.894115 | 慢 11.3% |
| ADD_ID | 6.612489 | 7.701514 | 慢 16.5% |

F16 prefill 在重复扫描中离散度较大，当前默认 MR4/LMUL1仍是已测合法候选中的最快组合，
所以没有为了单次数字改物理选择。ADD_ID 的回退则是可复现的默认 LMUL选择问题：显式
LMUL1约 6.33 ms，当前通用默认约 7.70 ms；现有 Kernel IR 已表达标量 expert index，但 VLA
access decision尚未把跨 scalar-loop 的 indexed-base provenance作为独立 memory fact。这项没有
被 kernel 特例掩盖。

与固定 GGML 表中可直接按相同 shape 解释的代表项相比：F16 GEMM decode/prefill分别为
6.630619/379.894115 ms；相对7.387/299.138 ms的F16参照，decode快约10.2%，prefill慢约
27.0%。Softmax 2.643862 ms、RMSNorm 1.365566 ms快于对应参照。Q4_K×Q8_K为13.462768 ms，
约慢3.5%；Q1与MXFP4分别为
57.104849与19.466505 ms，约慢10.1%与11.1%。IME上，Q4_K projection为3.231835 ms，快约
5.9%；Q4_0 projection为3.138545 ms，慢约6.7%。这些只用于理解当前差距，完整数字仍以两份
CSV各自记录为准。

## 仍然存在的窄边界

- F16 widening dot仍要求双unit-stride load→cast→multiply→reduce的single-use local closure；
- online softmax仍要求summary producer与normalize consumer满足精确use-def、noalias、同domain
  和effect closure；canonical地址比较目前是保守结构等价，等价但不同写法可能失去融合；
- F16 GEMM仍要求当前m→n→k carried contract envelope；
- affine Q4_K IME仍要求source显式写出N16/K32与304-byte persistent layout，当前helper leaf仍
  拥有一段局部N-tile traversal；
- generic VLA不支持lane-varying masked load，只递归支持的scalar `for`，不普遍覆盖`if/while`；
- contract尚缺成熟multi-axis microtile、load scheduling、prefetch、local reuse和software
  pipeline；
- state placement、unit/strided/indexed memory scheduling、vision window reuse与IME多fragment
  仍未形成宽候选空间；
- block decode/store/reduce的decision producer仍贴近block emission入口。

## 本轮判断

本轮已经把“analysis与emission各选一次”从主要VLA、contract和extension路径中清掉，并用
四十个固定kernel证明共享candidate不能等同于最大LMUL或固定常数。性能改善来自VLA state、
contract、narrow、memory与extension primitive的共同路径，没有新增kernel-name route、
q-format route、legacy fallback、GGML/materials runtime调用或whole-kernel emitter。

但报告也保留两个明确结论：物理候选空间目前只在VLA LMUL和F32 contract上开始成形，远未
覆盖用户要求的全部microtile/pipeline/prefetch/state/fragment维度；若干高性能leaf仍依赖窄
local closure。Weft已经进一步完成“编译器化”，但当前结果不能表述成高性能空间已经完整。
