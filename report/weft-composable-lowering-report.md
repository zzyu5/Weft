# 十个新增 Kernel 与可组合 Lowering 现状

这份报告只记录当前实现、真机运行结果与仍存在的架构边界。固定 GGML/RISC-V 数字继续来自
`report/baseline/`，没有修改；Weft 原始记录集中在
[`weft-kernel-performance.csv`](weft-kernel-performance.csv)。十个kernel中F32 GEMM同时记录
decode与prefill，所以CSV新增十一条性能行。

## 结论

十个新增kernel全部通过同一条主链：

```text
Weft Python DSL
→ canonical worker-local Kernel IR
→ RISC-V physical decisions / lowering
→ RVV intrinsic C / local IME asm fragment
→ target system compiler
→ SG2044 或 K1/X60 executable
```

本轮最重要的变化不是新增十个entry，而是四类decision开始从完整source closure中独立出来：

- VLA内每个predicate、memory access、reduce/scan/summary和narrow分别产生lane relation、
  activity、memory mode、state realization与LMUL；
- F32 contract保存operand axes、pointer/stride relation、row microtile与LMUL；
- codebook decode保存table extent、code/result physical vector shape与RVV gather realization；
- symmetric i4×i8 primitive保存local packed relation与IME1 N16×K32 realization。

作者的blocking、outer traversal、staging、persistent layout、predicate、state和numerical policy
没有经过“规范化作者程序”pass，也没有被target从use-count或kernel shape补猜。新增Q4_0路径
尤其保留了source中的128次K32 recurrence；asm leaf只处理一次N16×K32 primitive。

这仍不是完全摆脱closure的通用编译器。旧online-softmax、F16 GEMM、affine Q4_K IME和若干
F16 VLA fast path仍依赖精确region/use/loop closure。当前准确状态是：逐实体主干已经出现，
但较早的高性能slice尚未全部迁移到这套decision模型。

## 十个 Kernel 逼出的能力

| Kernel | Source / Kernel IR 明确拥有 | 新的共享 lowering 能力 | 当前边界 |
|---|---|---|---|
| Causal mask F32 | row→query relation、VLA key domain、causal predicate、masked `-inf` store | compare与store分别选择predicate mask、activity、LMUL和unit-stride memory | masked load与更复杂predicate组合仍未完整覆盖 |
| Cumsum F32 | row traversal、inclusive ordered scan、identity和每位置output | RVV slide/add prefix、strip末值到下一strip的显式carry | 当前scan realization只覆盖all-active ordered F32 add |
| LayerNorm F32 | 同一VLA中的sum与sum-square、mean/variance/eps、第二遍normalize | 一个region内多个reduction state独立决策并共享VLA traversal | state family仍只覆盖有限dtype/op |
| Argmax F32 | `(maximum,index)` state、显式logical coordinate、first-index tie break | max reduction、equal mask、first-set lane实现同一summary algebra | 当前coordinate summary realization只覆盖该argmax algebra |
| SwiGLU F16/F32 | 两个F16输入、显式F32 cast、gate→activation→up顺序、F32输出 | mixed-dtype VLA load/cast/math/store组合 | fast exp仍是窄的F32 realization |
| GetRows F32 | u32 row index、row/table stride、token与hidden traversal | scalar indexed row base与unit-stride RVV copy自然组合 | 当前真实shape在SG2044没有同硬件GGML直接对照 |
| GEMM F32 | 6-row blocking、column loop、dynamic K block、contract axes与masked tail | local `ContractDecision`、LMUL4、一个K vector复用6个row accumulator | decision仍有source-loop closure gate；prefill发射质量明显不足 |
| Q8_0 quantize | K32 max、zero-maximum branch、F16 metadata、RNE+saturating I8、34-byte layout | per-VLA narrow decision与F32→I16→I8 intrinsic chain | 当前比固定GGML RVV实现慢25.3% |
| IQ4_NL dequantize | 18-byte block、显式16-entry codebook、low/high nibble、scale和output relation | block-local decode decision、`vrgather`、signed widening、唯一physical vector shape来源 | 当前只实现all-active 16-entry I8 table realization |
| Q4_0 IME projection | activation quantize/scratch、288-byte N16×K32 layout、N/K loops、carried accumulator | symmetric local extension、block carried storage、单fragment IME1 asm leaf | 只覆盖K1/IME1 VLEN256；memory-resident carry带来额外开销 |

## DSL、IR 与发射边界的变化

### DSL / Kernel IR

- `summary_fold` 增加可选 `coordinate` operand；提供时lift签名是
  `lift(element, coordinate)`，argmax不再让target猜logical index。
- 新增 `weft_ext.symmetric_i4_i8_contract`。它只定义一个K32 activation block与16列packed
  i4的 `code - 8` numerical relation，不含Q4_0名称、288-byte pointer、outer loop或ABI。
- 其余八个kernel沿用原有VLA、memory、reduce/scan、contract、decode、cast/narrow等通用
  surface；DSL没有为了fast path改写成backend标准形状。

### Physical decisions

`VLARegionDecision`现在按operation保存predicate/access/state/narrow选择；`BlockDecodeDecision`
保存decode的table与physical vector形状；`SymmetricI4I8Decision`只消费extension operand的local
memory closure和target facts。Emitter可以厚在intrinsic C、typed asm、C ABI spelling上，但
不再为这些路径重新回答LMUL、memory mode、decode family或fragment类型。

F32 contract也有独立decision，但仍要求当前row/column source context；它是从whole-kernel
matcher走向local primitive family的中间状态，而不是任意context都已可用的contract lowering。

### 仍存在的 exact closure

当前集中式 `RISCVLowering` 中仍保留：

- F32→F16、F16 fill、widening F16 dot、weighted update与F16→F32 normalize的固定VLA closure；
- online-softmax summary producer与normalize consumer的完整envelope；
- `m → n → k` F16 GEMM nested-loop fast path；
- affine Q4_K N16/K32 IME whole-loop fast path；
- local F32 contract对source loop/access关系的严格gate。

这些路径不依赖kernel symbol，也没有legacy/GGML fallback，但尚不能作为“逐实体组合已经完成”
的证据。本轮新增kernel没有增加同类whole-kernel case。

## 真机性能

SG2044记录使用RVV VLEN128；Q4_0使用K1/X60、RVV VLEN256与SpacemiT IME1。每条记录都在
64 MiB eviction后计时并取中位数；IME项另有三次warmup。F32 GEMM correctness是代表点采样，
其余新增项是全输出或完整离散结果对照。下表先列全部Weft读数。

| Kernel | Hardware | Shape | Median | Throughput |
|---|---|---|---:|---:|
| Causal mask F32 | SG2044 | `heads=32,Q=128,K=2048` | 6.684719 ms | 1254.893 MElements/s |
| Cumsum F32 | SG2044 | `sequences=1,heads=64,tokens=4096` | 0.393851 ms | 665.592 MElements/s |
| LayerNorm F32 | SG2044 | `hidden[128,4096]` | 1.602827 ms | 327.102 MElements/s |
| Argmax F32 | SG2044 | `batch=128,vocab=128256` | 16.925466 ms | 969.945 MElements/s |
| SwiGLU F16/F32 | SG2044 | `ffn[128,14336]` | 5.844536 ms | 313.970 MElements/s |
| GetRows F32 | SG2044 | `TinyLlama vocab=32000,hidden=2048,tokens=128` | 0.403822 ms | 5.193 GB/s |
| GEMM F32 decode | SG2044 | `M=1,N=4096,K=4096` | 17.742197 ms | 1.891 GOP/s |
| GEMM F32 prefill | SG2044 | `M=128,N=4096,K=4096` | 854.746405 ms | 5.025 GOP/s |
| Q8_0 quantize | SG2044 | `M=128,K=14336` | 4.126398 ms | 444.700 MElements/s |
| IQ4_NL dequantize | SG2044 | `N=1024,K=4096` | 5.715355 ms | 733.866 MElements/s |
| Q4_0 IME projection | K1/X60 | `M=1,N=4096,K=4096`，含activation quantize | 3.188198 ms | 10.525 GOP/s |

Q8_0 metadata/code与IQ4_NL输出逐项一致；Cumsum、Argmax、GetRows和采样F32 GEMM为零误差。
LayerNorm与SwiGLU最大绝对误差分别为 `1.20e-6` 与 `8.72e-7`。Q4_0 activation code与
scale逐项一致，完整output的最大绝对/相对误差为 `0.0312509537 / 0.0312504768`，来自local
K32 fragment逐次F32累加与scalar reference的运算顺序差异。

只有同算法、同shape、同硬件的固定GGML行进入直接比值。`Weft / GGML` 是时间比，低于1
表示Weft更快。

| Kernel | Weft ms | 固定 GGML ms | Weft / GGML | 读数 |
|---|---:|---:|---:|---|
| LayerNorm F32 / GGML `norm` | 1.602827 | 3.507 | 0.457 | Weft快54.3% |
| GEMM F32 decode | 17.742197 | 20.612 | 0.861 | Weft快13.9% |
| GEMM F32 prefill | 854.746405 | 574.193 | 1.489 | Weft慢48.9% |
| Q8_0 quantize | 4.126398 | 3.292 | 1.253 | Weft慢25.3% |
| IQ4_NL dequantize | 5.715355 | 25.982 | 0.220 | Weft快78.0%，吞吐约4.55倍 |
| Q4_0 IME projection | 3.188198 | 2.941 | 1.084 | Weft慢8.4% |

Causal mask、Cumsum、Argmax和SwiGLU在固定表中没有同算法/shape项。TinyLlama GetRows F32
只有K1/X60的0.691 ms记录，而当前Weft数字来自SG2044，所以不做跨硬件比值。SwiGLU也不与
只有单输入F32 SiLU的4.113 ms数字混比。

性能结论不是“十项都追平”：IQ4_NL、LayerNorm和F32 decode明显领先，Q4_0 local IME已经
进入同档；Q8_0仍慢25.3%，F32 prefill慢48.9%，后者是当前最明显的发射质量缺口。这里的
数字只是一份现状记录，没有阈值、同步、版本跟踪或自动校验逻辑。
