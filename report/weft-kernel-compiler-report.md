# 四十个真实 Kernel 之后的 Weft

这份报告合并此前四轮 kernel 推进记录，统一说明 Weft 从重建主链到当前状态究竟完成了
什么。它记录实现事实、真机数字和仍存在的边界，不是设计规范、性能门槛或自动检查输入。

固定 GGML/RISC-V 参照仍在 [`baseline/`](baseline/)；Weft 当前所有原始性能记录仍在
[`weft-kernel-performance.csv`](weft-kernel-performance.csv)。二者都不由本文复制、同步或
校验。

## 结论

当前记录包含四轮共 **40 个 distinct kernel**。两个 GEMM 各自记录 decode 与 prefill，
所以性能 CSV 有 **42 条数据行**。四十项都沿同一条 production path 生成并在 SG2044 或
K1/X60 上执行：

```text
natural Weft Python DSL
→ canonical worker-local Weft Kernel IR
→ RISC-V primitive-local physical decisions
→ RVV intrinsic C / necessary local inline asm
→ target system compiler
→ ordinary executable
```

这条链已经是单 kernel 的 RISC-V 算子编译器，而不是图编译器或 kernel 模式库：

- 作者在 DSL 中写完整 worker-local 算法、outer traversal、blocking、staging、state、layout、
  predicate、effect 与 numerical relation；
- canonical Kernel IR 是唯一长期算法表示；
- target lowering 只从 typed primitive、logical axes、local use/access relation、target facts
  与显式 config 形成瞬态 physical decision；
- emitter 消费已经决定的 `vl`、LMUL、memory mode、microtile、decode family 与 extension
  fragment，不再拥有第二套算法或物理选择；
- GGML 和 `materials/` 没有进入 production link、runtime 或 fallback。

四轮推进的核心不是累计四十个 case，而是逐步建立并压力验证同一组可组合能力：VLA、
predicate/memory、reduce/scan/summary、ordered state、local contract、packed decode、irregular
access，以及 typed RVV/IME extension primitive。当前已经有真实外推能力，但 physical
candidate space、部分旧 source envelope 的接受范围，以及少数高性能路径仍不足以称为
“完整 Triton for RISC-V”。

## 统一口径

| 轮次 | Kernel 数 | 主要问题 | 得到的核心结果 |
|---|---:|---|---|
| 第一轮：主链成立 | 10 | 能否让结构差异很大的真实算法走一条可执行路径 | 建立 Python DSL、canonical Kernel IR、RVV/IME intrinsic/asm 发射和真实 runtime |
| 第二轮：逐实体决策 | 10 | lowering 能否把部分 whole-kernel/enclosing-loop closure 收缩为 primitive-local analysis | predicate/access/state/narrow、F32 contract、decode 与 symmetric IME decision 成为明确实体；若干旧 family 仍保留精确 local envelope |
| 第三轮：陌生组合 | 10 | 已有能力换到新的 control、state、indexed 和 vision 上下文能否复用 | VLA analysis 穿过支持的 nested `for`，F32 contract脱离固定外围loop形状，并得到等价source证据 |
| 第四轮：事后选择与性能修正 | 10 | 在 lowering 已形成后选择的新算法能否编译并跑到有意义区间 | radix selection、source-owned staging/traversal、backward、两种新量化 primitive，以及 decision/emission authority 收敛 |

这里的“40 个”只计算性能记录中的 distinct kernel。仓库中其他可执行 source 或同一
block-dot DSL 的额外格式入口不进入这项统计。

## 当前编译器形态

### 程序模型

一次 Weft 编译输入是一份完整 worker/hart-local callable。线程池、worker range、affinity、
多核调度与 graph orchestration 都在外部 runtime。Weft core 不导入或划分计算图，不从普通
multiply/add SSA 猜 GEMM、attention 或量化格式，也不根据 kernel 名选择整段实现。

作者与 Kernel IR 拥有：

- 普通 C ABI、scalar `if/for/while` 与 carried state；
- outer traversal、cache blocking、source meta 与算法 variant；
- VLA logical domain、logical block shape、pointer/index/predicate/effect；
- staging、scratch、persistent packing 与可观察 layout；
- reduce、scan、summary fold、sequential carry 与 contract axes；
- numerical mode 和具有独立可观察语义的 local extension primitive。

Target lowering 拥有：

- dynamic `vl`、SEW、LMUL、mask/index vector relation；
- unit-stride、strided、indexed 或 scalar memory realization；
- reduction realization、跨 strip physical state；
- register repeat、local microtile、K-unroll 与短生命周期 packing；
- RVV intrinsic、IME fragment 和必要的 typed inline-asm spelling；
- 不改变 source boundary 的 local pure producer/consumer fusion。

### 表示层次

旧 Selection、Execution/Selected IR、route/provider 与 whole-kernel SourceEmitter 已退出当前
production core。长期 authority 压缩为：

```text
Python DSL source
    ↓ 直接构造
canonical Kernel IR
    ↓ 一次 target lowering
intrinsic C / local asm artifact
```

Capability、legality、resource equation、candidate enumeration 与 selection 只存在于本次
target lowering 调用，不形成可单独输入的 IR、dialect、validator 或 provider registry。

### DSL 与 Kernel IR 已形成的承重语义

当前 DSL/IR 已经实际被四十个 kernel 使用，而不是只存在于接口文档：

- 普通 scalar/control：typed scalar、pointer、`if`、`for`、真正 loop-carried `while`、
  scalar `floor/log/exp/sin/cos` 等；
- VLA：一个 active logical VLA axis、runtime strip-mining、nested scalar control；
- validity/memory：logical predicate、filled masked load、unit/strided/indexed access、明确 effect；
- logical block：`block_axis`、broadcast/reshape/transpose、block load/store；
- state：`reduce`、inclusive ordered `scan`、带 `lift/merge/finalize` 的 `summary_fold`、普通
  sequential carry；
- structured compute：显式 `contract` axes、init、dtype、order、math 与 output relation；
- packed semantics：cast/widen/narrow、bitcast、decode、codebook 与 local packed relation；
- sibling extension：affine/grouped/symmetric i4×i8、sign-bit×i8、E2M1/E8M0×i8 等 local
  numerical primitive。

其中 `summary_fold(coordinate=...)` 让 argmax/selection 的 logical coordinate 由算法显式提供；
target 不再从 lane ID 或 pointer expression 猜 index。量化 extension 只表达一次 local numerical
relation，不拥有 public ABI、persistent pointer layout、outer loop 或 target fragment。

### 当前 physical decisions

RISC-V lowering 当前已经明确持有以下短生命周期 decision：

- 每个 VLA predicate、load/store、reduce/scan/summary 与 narrow 的 lane relation、activity、
  memory/state realization 与 LMUL；
- nested scalar control 内每个 VLA access 的独立 projection；
- local F32 contract 的 typed producers/consumer、row/reduction axes、extent、pointer/predicate
  projection、row microtile 与 LMUL；
- VLA contract 与 enclosing VLA region 共享同一个 LMUL authority，strip width、mask ratio、
  index LMUL 与 contract intrinsic 不再各自选择；
- block decode 的 table extent、code/result vector shape 与 gather family；
- symmetric i4×i8、sign-bit×i8、E2M1/E8M0×i8 的 typed operand closure、local layout relation
  与 RVV/IME realization；
- 较早 F16 VLA、online-softmax、F16 GEMM 与 affine Q4_K IME envelope 的 typed transient
  decision。

最后一组仍要求较精确的 region/use/loop envelope，但结构分析和代码发射已经分开：analysis
形成 pointer、bound、stride、tile、layout 与 realization，emitter 只消费，不再边打印边重新
匹配一次。

这些是已经落地的typed physical decisions，不等于成熟的多候选搜索。多数family目前仍只有
固定配置或少量资源规则；candidate enumeration、实测调优与更宽的realization space尚未完成。

## 第一轮：十个 Kernel 建立真实主链

| Kernel | Source / Kernel IR 明确拥有 | 逼出的可复用 lowering |
|---|---|---|
| SiLU F32 | VLA domain、显式公式与 fast-exp 数值选择 | runtime `vl`、RVV F32 pointwise、unit-stride memory 与局部 exp |
| RMSNorm F32 | row loop、sum-square reduction、scalar normalization state、第二遍 scale | 跨 strip reduction、scalar/RVV state handoff、同一 kernel 多个 VLA |
| Online Softmax F32 | `(maximum, scaled_sum)` state、lift/merge/rescale/finalize | summary-state、跨 strip max/sum、局部 exp 与 normalize realization |
| F16 dense GEMM | M/N/K loops、BM/BN/BK、masked blocks、contract axes、carried accumulator | F16 load、F32 accumulate、register microtile、widening FMA 与 tail |
| Q4_K×Q8_K projection | 144/292-byte ABI、outer blocks、scales、minimum correction、init | grouped affine i4×i8 primitive与 VLEN128 RVV asm leaf |
| Q4_K GetRows | row index、scale/min/nibble decode 与 output relation | irregular pointer chain、block producer closure、packed RVV decode/store |
| Contiguous transpose | row traversal和两侧 affine index | 同一 VLA 中 unit-stride load 与 strided store |
| RoPE NeoX | position/head/pair loops、theta carry、rotation公式与 scratch | scalar trig、RVV VLA、half-dimension strided access |
| FlashAttention F32/F16 | GQA/causal bound、query staging、key loop、online state与accumulator rescale | F32→F16、widening dot、weighted update、normalize组合 |
| Q4_K IME projection | activation quantize、scratch、N/K loops、persistent K32×N16 layout | narrow/round/saturate、affine i4×i8 local contract 与 IME1 fragment |

这一轮确定了五类核心机制：worker-local scalar/ABI、VLA memory/state、logical block与
contract、packed/irregular producer-consumer，以及 typed extension 到 RVV/IME leaf。它也暴露
了早期高性能路径对精确 source closure 的依赖，成为后续 decision 拆分的起点。

## 第二轮：十个 Kernel 将选择拆到实体

| Kernel | Source / Kernel IR 明确拥有 | 逼出的可复用 lowering |
|---|---|---|
| Causal mask F32 | query/key relation、causal predicate、masked `-inf` store | compare与store各自的predicate mask、activity、LMUL和memory decision |
| Cumsum F32 | row traversal、inclusive ordered scan、identity与逐位置output | RVV slide/add prefix与跨 strip carry |
| LayerNorm F32 | sum/sum-square、mean/variance/epsilon、第二遍normalize | 同一 VLA 中多个 reduction state 独立选择并共享 traversal |
| Argmax F32 | `(maximum,index)` state、logical coordinate、first-index tie break | coordinate summary、max/equal mask与first-set lane |
| SwiGLU F16/F32 | 两个 F16 input、F32 cast、gate→activation→up、F32 output | mixed-dtype VLA load/cast/math/store组合 |
| GetRows F32 | U32 row index、row/table stride、token/hidden traversal | scalar indexed base与unit-stride RVV copy |
| GEMM F32 | source-owned row blocking、column loop、dynamic K、contract axes和tail | operation-local `ContractDecision` 与多row accumulator组织 |
| Q8_0 quantize | K32 max、zero branch、F16 scale、RNE+saturating I8、34-byte layout | per-VLA narrow与F32→I16→I8 intrinsic chain |
| IQ4_NL dequantize | 18-byte block、16-entry codebook、nibble、scale/output relation | `BlockDecodeDecision`、`vrgather` 与唯一 vector shape来源 |
| Q4_0 IME projection | activation quantize、288-byte N16×K32 layout、N/K carry | symmetric local extension与单 N16×K32 IME fragment |

这轮最重要的重构不是 extension 数量，而是 decision ownership：VLA operation、F32 contract、
decode 和 symmetric IME 各自形成 typed physical fact；emitter 不再从完整 kernel 或 q-format
route 重建 memory mode、LMUL、decode family 或 fragment。

## 第三轮：十个陌生 Kernel 检查组合与外推

这十项是在检查已有 lowering 能力之前先选定的，目的是打破围绕成熟 family 选题的偏差。

| Kernel | 自然算法结构 | 复用/新增的局部能力 |
|---|---|---|
| MoE Router Top-K | row/rank traversal、previous selections、coordinate summary | VLA predicate、tuple summary、max/equal/first-lane |
| Mamba short convolution | sequence/channel traversal、token VLA、短 tap loop | physical scan穿过nested scalar loop，unit/strided memory |
| Mamba2 SSM scan | token/head/dimension/state、softplus、decay与ordered update | scalar math、ordered memory state、dimension VLA |
| RWKV-WKV6 | token/head/row/column顺序与matrix state recurrence | scalar carried traversal与column VLA memory |
| ADD_ID | token/slot、expert index与hidden-domain add | scalar indexed base、unit-stride RVV add/store |
| GET_ROWS_BACK | zero-fill、source order与duplicate scatter-add | indexed base、VLA load/add/store；冲突语义仍由source保留 |
| MUL_MAT_ID | expert count/offset/cursor/items grouping与local contraction | source grouping不变；F32 contract在expert→row→item上下文复用 |
| Depthwise Conv2D | spatial/window loops、padding、NHWC channel domain | scalar coordinate predicate、channel RVV memory/FMA |
| MaxPool2D | channel/row/window traversal与width domain | width VLA load/max/store |
| Bilinear upscale | half-pixel、floor/clamp、四邻域与C-fast layout | scalar coordinate math与channel RVV arithmetic |

这一轮得到三项关键架构结果：

1. VLA analysis 当前可以递归观察支持的nested `for`内的compare/load/store/narrow，而不把
   tap、rank或window loop规范化成标准region；`if/while`等其他scalar region尚未普遍递归；
2. local F32 contract不再依赖固定的外围row/column loop形状，同一primitive可以进入MoE
   grouping、卷积与其他上下文；局部axes、typed operands、use/access projection仍须满足
   当前family契约；
3. system compiler边界保持不变，Weft没有为了性能继续向register allocation、machine
   scheduling或peephole扩张。

还对两个 kernel 写了自然等价 source：Top-K 的pointer-base hoist/predicate operand换序，以及
SSM convolution 的二层坐标与线性 position decode。两组 source 选择了相同的 VLA memory、
LMUL和summary类别，intrinsic集合与计数一致；没有为此增加程序规范化 pass。

## 第四轮：十个事后选择 Kernel 与后续收敛

| Kernel | 自然算法结构 | 逼出的能力或修复 |
|---|---|---|
| F32 Argsort | four-pass radix、F32 sortable-key bit relation、histogram/prefix、ping-pong indices、scratch ABI | 真正 ordered scalar control、indexed histogram/scatter和通用F32→U32 bitcast |
| F32 SetRows | update order、indexed destination、共享row index与strides | scalar indexed destination加unit-stride VLA row update |
| Window Partition | window order、padding predicate、zero fill与output layout | scalar window control、logical predicate和VLA memory组合 |
| Dense Conv2D | explicit patch/weight staging、position blocking与local contraction | source-owned im2col加row-resource decision；row8选择LMUL1 |
| ConvTranspose2D | input-owned traversal、source/weight repack、non-overlap ownership | 普通local contract消费新traversal，不增加transpose-conv route |
| RMSNorm backward | 两次reduction、backward equation、epsilon与row traversal | 多state reduction后接VLA consumer的新组合 |
| OutProd F32 | samples/rows/columns relation、短reduction与output layout | VLA free-axis local F32 contract在新axes关系中复用 |
| Im2Col backward | output-owned dInput traversal、ordered window accumulation | scalar ordered window与strided VLA load组合 |
| Q1_0×Q8_0 | sign bits、128→4×32 relation、scales与outer blocks | `sign_bit_i8_dot`、mask/sign merge与widening reduction |
| MXFP4×Q8_0 | E2M1 codebook、E8M0 exponent、packed nibble与outer blocks | `e2m1_e8m0_i8_dot`、table gather与widening multiply/reduction |

这一轮先暴露了三个明显性能问题：Argsort 的 heapsort source、Dense Conv 的直接 strided
contraction、ConvTranspose 的 output-owned inverse traversal。修复遵守算法/物理边界：

- Argsort由作者显式改为radix source，target没有识别或替换“sort kernel”；
- Dense Conv的patch/weight staging与row8 blocking写进source，target只根据local contract的
  row resource选择LMUL1；
- ConvTranspose由source显式选择input-owned traversal和packing，当前`stride=kernel=2`的
  non-overlap ownership不需要target猜测或atomic fallback。

本轮之后还完成两组 lowering authority 收敛：

- F16 conversion/fill/dot/update/normalize、online-softmax、F16 GEMM与affine Q4_K IME从
  “matcher同时打印代码”拆成analysis decision加emitter消费；
- VLA contract的LMUL成为enclosing VLA strip、mask/index relation与contract emitter的唯一
  来源，消除同一region内两份物理宽度选择。

## 四十项真机性能

### 测量边界

- SG2044记录使用单 hart、RVV VLEN128与GCC 15；IME记录使用K1/X60、RVV VLEN256、
  SpacemiT IME1与目标Clang；
- 每条记录使用对应单-kernel runtime，在64 MiB cache eviction后计时并取中位数；IME项有
  显式warmup；
- Weft计generated kernel entry；GGML public-op repro计`graph_compute + synchronize`；Q1_0与
  MXFP4对照直接读取固定baseline中的同形single-thread手写RVV vec-dot记录；
- 表中的比值只在同算法、同shape、同硬件时给出。小于1表示Weft更快；
- 时间表是当前数字记录，不包含阈值、同步、版本跟踪或校验逻辑。

### 第一轮

| Kernel / phase | Hardware | Weft ms | 对照 ms | Weft / 对照 | 读数 |
|---|---|---:|---:|---:|---|
| SiLU F32 | SG2044 | 4.197469 | 4.113 | 1.021 | 同档，慢2.1% |
| RMSNorm F32 | SG2044 | 1.601028 | 2.637 | 0.607 | 快39.3% |
| Softmax F32 | SG2044 | 2.645742 | 4.170 | 0.634 | 快36.6% |
| F16 GEMM decode | SG2044 | 7.221713 | 7.387 | 0.978 | 快2.2% |
| F16 GEMM prefill | SG2044 | 330.751278 | 299.138 | 1.106 | 慢10.6% |
| Q4_K×Q8_K projection | SG2044 | 12.687497 | 13.009 | 0.975 | 固定表快2.5% |
| Q4_K GetRows | SG2044 | 0.890604 | 0.825 | 1.080 | 慢8.0% |
| Contiguous transpose | SG2044 | 6.034178 | 5.574 | 1.083 | 慢8.3% |
| RoPE NeoX | SG2044 | 0.994714 | 1.253 | 0.794 | 快20.6% |
| FlashAttention F32/F16 | SG2044 | 23.462027 | 21.667 | 1.083 | 慢8.3% |
| Q4_K IME projection | K1/X60 | 3.256607 | 3.436 | 0.948 | 快5.2% |

第一轮证明主链已经进入实际性能区间，不再存在旧Q4_K路径的数量级落后。F16 GEMM与
FlashAttention采用代表点correctness；其余八个kernel使用全输出或完整离散结果对照。

### 第二轮

| Kernel / phase | Hardware | Weft ms | 同硬件同shape对照 ms | 比值 | 读数 |
|---|---|---:|---:|---:|---|
| Causal mask F32 | SG2044 | 6.684719 | — | — | 当前性能记录 |
| Cumsum F32 | SG2044 | 0.393851 | — | — | 当前性能记录 |
| LayerNorm F32 | SG2044 | 1.602827 | 3.507 | 0.457 | 快54.3% |
| Argmax F32 | SG2044 | 16.925466 | — | — | 当前性能记录 |
| SwiGLU F16/F32 | SG2044 | 5.844536 | — | — | 当前性能记录 |
| GetRows F32 | SG2044 | 0.403822 | — | — | 固定表只有另一硬件，不跨硬件比较 |
| GEMM F32 decode | SG2044 | 17.742197 | 20.612 | 0.861 | 快13.9% |
| GEMM F32 prefill | SG2044 | 854.746405 | 574.193 | 1.489 | 慢48.9% |
| Q8_0 quantize | SG2044 | 4.126398 | 3.292 | 1.253 | 慢25.3% |
| IQ4_NL dequantize | SG2044 | 5.715355 | 25.982 | 0.220 | 快78.0% |
| Q4_0 IME projection | K1/X60 | 3.188198 | 2.941 | 1.084 | 慢8.4% |

这轮的主要性能缺口是F32 prefill与Q8_0；明显优势来自IQ4_NL、LayerNorm与F32 decode。
Q4_0 IME已进入同档，但memory-resident carried accumulator仍有开销。

### 第三轮

| Kernel | Weft ms | GGML public-op ms | Weft / GGML | 可比边界 |
|---|---:|---:|---:|---|
| Top-K | 3.184054 | 1.021364 | 3.117 | 同一selection集合；GGML使用C++ partial sort |
| SSM convolution | 2.747092 | 5.491962 | 0.500 | math、shape、layout相同 |
| SSM scan | 4.845841 | 9.114867 | 0.532 | math/layout相同，output/state ABI对应 |
| RWKV-WKV6 | 16.281330 | 16.707418 | 0.974 | math/layout相同 |
| ADD_ID | 6.572648 | 6.746318 | 0.974 | math、shape、layout相同 |
| GET_ROWS_BACK | 86.614495 | 89.849307 | 0.964 | duplicate scatter与zero-fill相同 |
| MUL_MAT_ID F32 | 939.236930 | 1134.325136 | 0.828 | grouping与完整contraction均计时 |
| MaxPool2D | 1.795128 | 9.226990 | 0.195 | math、shape、WHCN layout相同 |
| Depthwise Conv2D | 8.797379 | 113.517962 | 0.077 | 只作语义参照：两侧native layout不同 |
| Bilinear upscale | 6.557349 | 84.307306 | 0.078 | 只作语义参照：output layout不同 |

严格或近严格可比的前八项中七项更快，Top-K是唯一明显落后项。只在第三轮这十个GGML
public-op对照里，`MUL_MAT_ID F32`走真实RVV intrinsic，其余是C++或scalar path；因此该轮
不能推出“全面超过手写RVV”。Depthwise与Bilinear还混入native layout差异。

### 第四轮

| Kernel | Weft ms | 对照 ms | Weft / 对照 | 读数 |
|---|---:|---:|---:|---|
| Argsort F32 | 8.958199 | 21.902416 | 0.409 | 显式radix比GGML `std::sort` 快59.1% |
| SetRows F32 | 0.355782 | 0.432772 | 0.822 | 快17.8% |
| Window Partition | 5.165583 | 5.575184 | 0.927 | 快7.3% |
| Dense Conv2D | 779.800666 | 629.354844 | 1.239 | 6.196 GOP/s；按时间慢23.9% |
| ConvTranspose2D | 16.333010 | 17.179675 | 0.951 | 4.109 GOP/s；快4.9% |
| RMSNorm backward | 6.358188 | 10.622016 | 0.599 | 快40.1% |
| OutProd F32 | 128.860142 | 231.133128 | 0.558 | 快44.2% |
| Im2Col backward | 35.573495 | 374.418412 | 0.095 | 快90.5% |
| Q1_0×Q8_0 | 56.174805 | 51.859000 | 1.083 | 距手写RVV 8.3% |
| MXFP4×Q8_0 | 19.462305 | 17.522000 | 1.111 | 距手写RVV 11.1% |

第四轮十项中七项快于对应public-op；Q1_0与MXFP4是更严格的手写RVV对照，已经进入12%
以内。Dense Conv从最初约2.025提升到6.196 GOP/s，但仍未追平GGML；其剩余差距已从缺少
staging和错误LMUL收敛为local-contract microtile、unroll和pipeline候选不足。

## Correctness 记录边界

四十项都通过各自单-kernel runtime中已经实现的reference comparison后才记录性能，但不同
kernel的comparison scope不同：

- 第一轮F16 GEMM与FlashAttention、第二轮F32 GEMM使用代表点采样；第一轮其余项与第二轮
  大部分离散/张量结果做全输出或完整code/metadata对照；
- 第三轮CSV中只有Top-K没有记录数值误差字段，不能把空字段解释成零；MaxPool及其他已记录
  项为零，SSM/RWKV等
  记录的最大误差在`5.96e-8`到`5.66e-7`量级，其余已记录项为零；
- 第四轮Argsort、SetRows、Window、RMSNorm backward与Im2Col backward做全输出，另外五项
  做固定代表点；除RMSNorm backward最大误差`1.10268593e-05`外，记录值为零；
- Q4_0 IME完整output最大绝对/相对误差为`0.0312509537 / 0.0312504768`，来源是K32
  fragment逐次F32累加与scalar reference的运算顺序差异；activation code/scale逐项一致。

这些是现有runtime和CSV的事实边界，不构成额外测试体系、精度阈值或验证逻辑。

## 四十个 Kernel 共同逼出的能力

### 1. VLA 是逻辑轴，不是硬件 lane API

Source只写逻辑`[begin,end)`；exact `vl`、strip、LMUL、mask/index group由target决定。普通
scalar loop可以位于VLA body中，但第二个active VLA仍禁止。Causal、window、state、indexed
memory与vision workload证明VLA不是某一类kernel的入口标签。

### 2. State 语义没有被压成一个模糊 graph pattern

Reduce只观察最终值，scan输出prefixes，summary显式拥有lift/merge/finalize，普通sequential
carry保持严格顺序。RMSNorm、LayerNorm、Softmax、Argmax、Cumsum、SSM与RWKV分别使用这些
不同语义；target只能实现它们，不能从use graph补猜结合律或state boundary。

### 3. Contract 是算法授权的局部重组域

作者显式给出operands、paired axes、free axes、init、dtype、predicate与numerical mode。
F16/F32 GEMM、MUL_MAT_ID、Dense/Transpose Conv和OutProd证明同一contract anchor可以进入不同
outer contexts。Target lowering的授权范围包括microtile、LMUL、unroll、fragment和op-local
packing；当前只在部分contract/extension family中落地其中若干选择，不能创建source中不存在
的outer K loop、staging或persistent layout。

### 4. 新硬件语义通过local primitive进入

普通contract无法无损表达minimum correction、特定codebook、sign-bit mapping、block scale或
IME packed relation时，DSL/IR增加typed local extension。Outer block traversal、pointer ABI、
scratch与persistent format仍在source。Inline asm leaf实现一次明确primitive，不接管完整kernel。

### 5. 算法variant属于source，physical realization属于target

Argsort的radix、ConvTranspose的input-owned traversal、Dense Conv的staging都由source显式写出；
compiler没有把heapsort自动换成radix，也没有从output-owned卷积偷偷改变ownership。相反，
row8选择LMUL1、VLA strip width、decode gather和IME fragment属于target decision。

### 6. 等价source不需要规范化成作者模板

Top-K与SSM convolution的两组自然等价表达获得相同physical realization类别。当前判据是
axis、use-def、memory、state与primitive facts一致，而不是完整source文本形状相同；system
compiler仍可令最终C/机器码和时间略有差异。

## 当前仍存在的明确边界

### Physical candidate space仍窄

- F32 prefill GEMM比对照慢48.9%，Dense Conv按时间慢23.9%；local F32 contract虽然已有
  row6/LMUL4与row8/LMUL1的resource decision，但还没有成熟的multi-axis microtile、K-unroll、
  pointer scheduling和software-pipeline候选；
- Top-K每个rank重新扫描完整expert域，没有candidate reuse、rank-level组织或selection-local
  register strategy；
- Q8_0仍慢25.3%，Q1_0与MXFP4仍落后手写RVV 8.3%和11.1%；
- SSM/RWKV缺state tiling与persistent register placement，short convolution缺tap unroll/reuse；
- indexed memory仍缺prefetch、alias-aware scheduling与duplicate contention physical choice；
- vision window路径缺spatial/channel tiling、sliding-window reuse与coordinate-hoist candidate。

### 一些合法source envelope仍较窄

- F16 conversion/fill/dot/update/normalize、online-softmax producer/consumer、F16 GEMM与affine
  Q4_K IME已经有独立analysis decision，但仍只接受较精确的local region/use/loop envelope；
- generic VLA的masked load、predicate组合、dtype以及nested reduce/scan/summary覆盖还不完整；
- 第二个active VLA axis仍明确unsupported；
- intrinsic-C backend在module入口整体要求RVV；kernel内普通scalar C是正式realization，但
  当前没有独立scalar-only target backend；
- local F32 contract与VLA free-axis contract只覆盖当前少数rank/axes关系；
- scan当前只实现all-active ordered F32 add，coordinate summary也只有少数algebra realization。

### Extension和artifact边界仍然明确

- IME当前绑定K1/IME1、VLEN256与N16/K32 local envelope；这是已执行的真实leaf，不是通用
  matrix-extension abstraction已经完成；
- `weft-compile`当前public输出止于canonical Kernel IR或intrinsic C。Object、archive与
  executable由同一repro中的system compiler/archiver继续形成；
- runtime仍由应用维护普通C declaration并负责线程和work partition，Weft不生成调度runtime。

## 最终判断

四轮推进已经证明三件事：

1. **架构替换成立。** Weft当前只有一份worker-local Kernel IR和一条RISC-V主链，不再是图
   发射器、route/provider系统或新旧双路径仓库。
2. **可组合性开始成立。** 四十个结构不同的算法由同一组VLA、memory、state、contract、
   decode与extension anchors组合，第三、第四轮没有重新长出kernel-kind或whole-kernel
   fallback。
3. **性能只部分成立。** 多项已超过对应public-op，RVV/IME量化路径进入手写实现附近；但
   F32 prefill、Dense Conv、Top-K和若干量化/状态路径仍暴露明显physical freedom缺口。

因此当前最准确的描述是：**Weft已经从旧项目骨架长成一门真实可执行、具有初步外推能力的
RISC-V worker-local kernel DSL/compiler；它的算法/编译器ownership和唯一主链已经正确，但
primitive-local physical candidate space与一部分source-envelope通用性仍未完成。**
