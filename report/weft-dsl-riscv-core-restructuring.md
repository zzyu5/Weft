# Weft DSL 原语收缩与 RISC-V 编译核心重构

## 本轮结论

本轮没有增加 kernel，也没有重写固定六十个 kernel 的算法。工作的核心是把 Weft 的授权边界
真正落实到语言、Kernel IR 和 RISC-V lowering：删除一般化 `W.contract` 与未经真实需求证明的
generic summary fold，保留显式 `W.dot`、`W.matmul` 和具有完整局部语义的 structured primitive；
target 再从这些 primitive、typed operands、局部 axis/use/memory/state 事实与 target profile 产生
唯一物理决定。

当前唯一 production 主链仍是：

```text
Weft Python DSL
→ canonical worker-local Kernel IR
→ per-entity RISC-V physical decisions
→ intrinsic C / local inline asm
→ system C compiler
→ object / executable
```

本轮改动覆盖 30 个文件，增加 1839 行、删除 2302 行，净删除 463 行。主要变化集中在
`RISCVLowering.cpp`，同时删掉了 DSL、Kernel IR、frontend 和文档中的旧 general-contract authority。
整个过程没有增加 kernel-name、operator-name、q-format 或 whole-region route，也没有引入
normalization pass、legacy emitter、scalar fallback、GGML/materials production 调用。

## 一、DSL 与 Kernel IR 收缩

### 1. 一般化 contract 被删除

旧接口允许作者传入任意 `lhs_axes`、`rhs_axes` 和 `output_order`。现有真实语料并没有证明这种
任意轴 contraction 是 Weft 核心语言所需，因此 public DSL 和 canonical Kernel IR 已改成两个
明确授权点：

```python
W.dot(lhs, rhs, init=acc, acc_dtype=W.f32,
      order="relaxed", math="native")

W.matmul(lhs, rhs, init=acc, acc_dtype=W.f32,
         order="relaxed", math="native")
```

`W.dot` 固定收缩双方最后一个 logical block axis，当前承认的关系是：

```text
[R,K]   × [K]     → [R]
[VLA,K] × [K]     → [VLA]
[R,K]   × [VLA,K] → [VLA,R]
```

`W.matmul` 固定表达局部 `[M,K] × [K,N] → [M,N]` block relation。VLA axis 始终是 free/batch
axis，不能被 dot 或 matmul 隐式缩并；普通 scalar loop 中的 multiply/add 也不会被识别成
dot/matmul。

固定语料中只有七份 source 需要从 `W.contract` 机械迁移到 `W.dot/W.matmul`：blocked F16/F32
GEMM、indexed matmul、out product、triangular solve、dense conv 和 transposed conv。它们原有的
outer traversal、blocking、staging、pointer/index、predicate、init、carry 和 store 均保持不变；
删除的是任意轴元数据，不是算法结构。

### 2. generic summary fold 被删除

旧 `summary_fold(identity, lift, merge, finalize, ...)` 同时允许任意用户 algebra 和 helper closure，
但当前语料只证明了两个独立、完整、局部且可观察的 summary 语义：

```text
argmax(input, coordinate,
       tie=lowest_coordinate, order=relaxed)

online_softmax_summary(input,
                       math=native, order=preserve)
    → (maximum, scaled_sum)
```

因此 `SummaryFoldOp`、Python builtin/frontend 路径及其 verifier 已全部删除。Target 不再遍历
generic helper SSA closure 猜测 argmax 或 online softmax。`argmax` 明确拥有 value、logical
coordinate 与最低 coordinate tie；`online_softmax_summary` 只拥有稳定的 `(maximum,
scaled_sum)` state，后续 normalize VLA 独立 lowering。

### 3. structured primitive 的统一授权标准

当前保留的 ordering、quantization 和 extension primitive 均服从同一边界：

- 必须由作者显式写入 DSL/Kernel IR，不能从普通 SSA graph 猜出；
- 必须定义完整、可观察、局部的数值或 effect 语义；
- 可以拥有 primitive-local scratch、packing 或 asm leaf；
- 不能拥有外围 traversal、blocking、staging、persistent layout 或 kernel ABI；
- target 只能改变 primitive 内部 realization。

`sort_indices` 因此仍是显式 local ordering primitive；IQ2/IQ3/IQ1/Q6、sign-bit、E2M1 以及
affine/symmetric i4×i8 也仍是不同 typed local numerical relations。它们名字中出现的
`contract` 只表示一个明确量化 block 的局部乘加语义，不是已经删除的一般化 `W.contract`。

## 二、RISC-V lowering 的权责重构

### 1. 统一的 decision preparation

`preparePhysicalDecisions()` 现在在 emission 之前统一产生并保存各实体的 transient decision：

```text
VLA region
  ├─ predicate
  ├─ load / store memory relation
  ├─ binary / cast / narrow
  ├─ reduce / scan / argmax / online summary
  └─ local dot

dot / matmul
block decode / store / reduce
sort_indices
packed quant primitive
affine / symmetric i4×i8 primitive
```

这些 decision 都以 canonical operation 为 owner，只存在于一次 target lowering 中，不进入
Kernel IR，也不形成 provider/capability/selection 等第二套持久 authority。

### 2. 从整体 loop owner 改为局部 primitive owner

本轮删除了两类最明显的 source-closure authority：

- affine IME 不再向上寻找固定 N/K `for` loop、扫描整个 loop body、要求唯一 K loop 和唯一
  store；每个 affine i4×i8 primitive 独立产生 decision；
- F16 GEMM 不再以外围 N loop 为 decision key，也不再通过完整 M/N/K loop closure 识别整个
  GEMM；每个 `MatmulOp` 根据自己的 typed operands、block axes、load predicate 和 target
  resources 产生局部 decision。

Online-softmax summary 与 normalize consumer 的 343 行专用融合路径也已删除。Summary 只决定
自身 state realization，consumer 中的 memory、cast、binary 与 store 各自参与普通 VLA lowering。

Emitter 入口现在必须查到对应 decision；没有 decision 就明确报 unsupported/error。Emitter
仍负责地址/axis 投影、intrinsic C、ABI 和 typed asm spelling，但不再选择 LMUL、microtile、
memory mode、state kind、quant family 或 IME/RVV fragment。

### 3. VLA：逐实体事实组合 LMUL 与资源合法性

VLA 不再先判断“这个 region 属于哪类 kernel”。它先独立收集：

- unit/strided/indexed access 数量、element dtype 与 index width；
- predicate 类型与 mask footprint；
- reduction、scan、segmented scan、argmax、online summary 的 state footprint；
- narrow/cast intermediate；
- local dot 对 LMUL 的硬要求；
- region 内最大同时存活的 f32 vectors。

然后从 `{LMUL1, LMUL2, LMUL4, LMUL8}` 构造候选。VLEN128 与 VLEN256 只改变 candidate preference；
真正 legality 由 element shape、index/address vectors、predicate groups、state placement、narrow
intermediate 和 32 个 architectural vector registers 的资源关系决定。多个实体提出互相冲突的
硬 LMUL 要求时直接失败，不由 emitter 任选一个。

Memory decision 同时固定 `UnitStride/Strided/Indexed`、`AllActive/PredicateMask/ScalarPredicate`
和 scalar-broadcast/vector store；state decision 固定 reduce、scan、segmented scan、argmax 或
online-summary realization 以及跨 strip placement。后端拼写阶段只消费这些字段。

### 4. Dot/matmul 的物理空间

Local F32 dot 将 LMUL 与 `K-unroll={1,2,4}` 组合，按 accumulator groups、streamed operand
groups 和 vector-register headroom 过滤。VLEN128/VLEN256 与 row tile 影响 candidate 顺序，而
不是使用 kernel-specific winner。

F16 matmul 从 row-tile divisors 构造 row-microtile 候选，并与 input LMUL `{1,2,4}` 组合；
`computeLMUL=2×inputLMUL`，资源式为 `(3×rows+1)×inputLMUL`，只有保留 allocator headroom 的
组合合法。Accumulator 的 local storage 也先形成 materialization decision，再由 emitter 使用。

这里仍有明确边界：local F32 candidate 目前主要是单轴 microtile，load schedule、prefetch、
reuse 和 software pipeline 仍窄；F16 matmul 仍要求 f16 input、f32 accumulator、prefix predicate
和可证明的二维 block axes。它已经不再拥有完整 kernel，却还不是任意 source envelope 的通用
matmul lowering。

### 5. Quant 与 IME

Packed quant decision 不再依赖裸 base pointer 或 q-format route，而先建立 `LocalBlockMemoryFact`：
semantic/storage type、extent、base、all-active contiguous load 和 block-axis relation都在 selection
阶段固定。VLEN128 可以选择已经验证的 fixed-vector leaf；VLEN256 选择 scalable RVV realization。

Affine i4×i8 primitive 在相同 typed relation上拥有两种 target realization：

```text
matrix_extension = none          → RVV local N16×K32 fragment
matrix_extension = spacemit-ime1 → IME1 local N16×K32 asm fragment
```

IME leaf 只接管一个 16×32 fragment；activation quantization、outer N/K traversal、persistent
packed layout、accumulator carry 和 ABI 仍属于 source。当前 IME envelope明确要求 VLEN256、
little-endian、K32 activation、N16 zero point、16 个 fp16 scales和304-byte affine packed block；
没有把这些局部约束扩张成 whole-kernel IME emitter。

## 三、跨目标真实执行

### 1. 显式 target profiles

运行入口现在要求作者明确指定 profile：

```text
sg2044-rvv128 → SG2044, RV64GCV, VLEN128, matrix extension none
k1-rvv256     → K1/X60, RV64GCV, VLEN256, matrix extension none
k1-ime256     → K1/X60, RV64GCV, VLEN256, spacemit-ime1
```

Profile只提供 `march/ABI/VLEN/matrix-extension` target facts，不参与 DSL source 选择。相同 kernel
source、shape、runtime ABI、correctness范围、重复次数、warmup和cache-eviction协议进入同一
`DSL → Kernel IR → intrinsic C → system compiler` 主链。

标准 RVV 固定语料的真实结果为：

```text
SG2044 / VLEN128 : 62 / 62 phase 执行完成
K1/X60 / VLEN256: 62 / 62 phase 执行完成
K1/X60 / IME1   : 2 / 2 local fragment 执行完成
```

当前性能表共有126条记录：62条SG RVV、62条K1 RVV、2条K1 IME。104条使用full correctness
scope，22条使用sampled scope；20条明确把source-owned preprocessing纳入计时，不能将整张表
概括成“全部full”或“全部不含preprocess”。

### 2. 相同语义的 RVV 与 IME realization

K1 上两个量化 projection 使用相同 DSL source、M=1/N=4096/K=4096 shape 和计时协议：

| Local primitive | K1 RVV256 median | K1 RVV256 throughput | K1 IME1 median | K1 IME1 throughput | IME加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| affine q4_K×q8 | 44.164185 ms | 0.759766 GOP/s | 4.003037 ms | 8.382244 GOP/s | 11.03× |
| symmetric q4_0×q8 | 45.744584 ms | 0.733517 GOP/s | 3.168402 ms | 10.590333 GOP/s | 14.44× |

两项 activation code mismatch均为0；q4_K最大绝对误差为`3.33786011e-06`，q4_0为
`0.0312509537`。这证明target facts能够让同一个local semantic primitive选择RVV或IME
realization，而不是证明任意matmul都已自动映射到IME。

## 四、性能结果与真实缺口

所有当前数字以 `weft-kernel-performance.csv` 为准。跨设备数字用于观察target-specific
realization，不应被解释为同微架构的VLEN单因素实验：SG2044与K1的core、compiler和memory
system也不同。

SG与K1的62项完全对应，其中54项两边都有同单位throughput。按每项
`K1 throughput / SG throughput` 取几何平均，K1为SG的`0.326×`；这不是架构正确性的失败，
但明确说明“VLEN256已经自动得到更高性能”并不成立。

| 共享能力压力点 | SG2044/VLEN128 | K1/VLEN256 | K1 / SG |
| --- | ---: | ---: | ---: |
| F16 GEMM decode | 2.661416 GOP/s | 0.854541 GOP/s | 0.321× |
| F16 GEMM prefill | 5.868972 GOP/s | 1.796577 GOP/s | 0.306× |
| F32 GEMM decode | 2.264688 GOP/s | 1.773335 GOP/s | 0.783× |
| F32 GEMM prefill | 5.673938 GOP/s | 0.821950 GOP/s | 0.145× |
| q4_K×q8_K | 9.088889 GOP/s | 0.914794 GOP/s | 0.101× |
| IQ3_S×q8_K | 1.718327 GOP/s | 0.121601 GOP/s | 0.071× |
| Q6_K×q8_K | 6.142021 GOP/s | 0.241673 GOP/s | 0.039× |
| SSM convolution | 0.747420 GOP/s | 0.078850 GOP/s | 0.105× |
| ADD_ID | 6.563364 GB/s | 2.480678 GB/s | 0.378× |
| indexed row copy | 5.204219 GB/s | 3.314495 GB/s | 0.637× |

本轮开始前与结束后的SG共同60项median几何平均由`11.514579 ms`变为`12.176728 ms`，即整体
回退`5.75%`。这必须作为本轮结果的一部分，而不能只报告62/62通过。主要回退包括：

- F16 GEMM decode：`6.664249 → 12.607737 ms`；
- F16 GEMM prefill：`352.083699 → 731.809075 ms`；
- Softmax：`2.643301 → 4.584860 ms`；
- Depthwise Conv：`2.882452 → 5.747046 ms`；
- IQ1_M×Q8_K：`37.359844 → 77.098845 ms`。

当前性能缺口可以归因到共享 physical freedom，而不是缺少kernel fast path：

- F16/F32 matmul缺更成熟的multiple accumulators、load schedule、reuse、prefetch和pipeline；
- VLEN256 quant scalable leaf的decode/gather/widen/register organization明显不足；
- state convolution、indexed contraction与vision contraction缺target-aware local reuse和memory
  scheduling；
- VLEN256 candidate preference已经变化，但“合法候选”与“该微架构的高性能winner”仍不是一回事；
- 构建期实测选择尚未覆盖完整候选空间，当前默认顺序会导致SG若干已成熟路径回退。

因此，本轮在编译器结构上完成了关键收缩和跨目标可执行性，但没有完成“两个target都形成成熟
高性能物理实现”的性能目标。当前数字证明的是：同一语义能够由target facts选择不同合法
realization；它们也同时准确指出下一阶段需要继续扩大的共享candidate与selection能力。

## 五、当前成立的边界

本轮结束后，可以确认：

- Weft core仍是一门worker/hart-local kernel DSL，不是graph compiler；
- 一般化`W.contract`和generic summary fold已从语言与IR删除；
- dot/matmul、state、memory、ordering、quant和extension均以显式primitive授权；
- 主要物理决定在emission之前唯一产生，emitter不会按kernel/格式重选实现；
- 相同DSL可以在VLEN128、VLEN256和IME profile上沿唯一主链生成并真实执行；
- `source/`与`materials/`没有进入compiler production fallback。

仍不能声称：

- 任意dot/matmul source envelope都能获得高性能lowering；
- VLEN256量化、prefill GEMM、state/vision contraction已经具有竞争力；
- IME支持已超出两个明确N16×K32 local primitive；
- 当前candidate ordering等同于成熟的build-time tuning；
- 固定语料全部采用full correctness或统一不含preprocess的计时范围。

本轮真正得到的是一个更清楚、更难被whole-kernel特例污染的编译核心，以及一份没有掩盖跨目标
性能缺口的真实测量结果。结构重构已经成立，性能空间还需要继续成熟。
