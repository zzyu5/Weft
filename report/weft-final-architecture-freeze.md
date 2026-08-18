# Weft DSL、物理编译器与 RISC-V 扩展模型的最终定性

## 结论先行

Weft 应冻结为：

> **面向单个 RISC-V worker / hart 的 Core-local blocked kernel DSL 与 AOT 编译器。作者编写一份持续执行的有序 CPU kernel，显式拥有 outer traversal、blocking、staging、storage、state 与算法 variant；编译器只在显式 VLA、structured compute、state、memory、quant 和 extension primitive 的授权范围内，依据逻辑轴、值链、访存关系与生命周期，生成 sequential / lane / register / unroll / fragment 的物理布局、局部调度和 RVV / IME 实现。**

这个定义同时否定四种错误定位：

- Weft 不是 tensor graph compiler；
- Weft 不是按 kernel 或量化格式选择手写实现的高性能函数库；
- Weft 不是把 Triton 的 GPU grid 换成 CPU `for` 循环；
- Weft 也不是一个试图从任意 SSA 图自动发现算法的万能 layout solver。

当前方向不需要再次推翻 DSL 或 Kernel IR 根模型。真正需要继续完善的是物理编译器内部的两个承重部分：

1. 由目前的“每轴 factor 分解”细化为 **operand/result-aware 的瞬态 physical arrangement**；
2. 由目前的“单 primitive 局部 schedule”扩展为 **作者显式 loop 内、跨相连 operation 的依赖驱动局部调度与完整 live-resource 计算**。

因此，最终冻结不是“当前代码已经完成”，而是：

```text
冻结：程序模型、作者/编译器边界、op 驱动规则、决策分层、扩展接入方式

不冻结：当前 factor-only layout、当前局部 pipeline、当前候选质量、当前 cost heuristic
```

这与两份高级分析的核心结论一致：当前 Weft 已经是一个真实的、显式 op 约束下的 core-local 物理程序生成器，但还不是 Triton LinearLayout 成熟度的 layout compiler；高性能 schema 与更细的 layout 并不冲突，前者应当由后者实例化，而不能重新退化为 whole-kernel 模板。

---

## 一、证据边界：当前到底已经是什么

本报告同时依据：

- 现行 DSL、Kernel IR 与 RISC-V lowering 源码；
- 当前规范中的程序语义与职责边界；
- [`aichat1.md`](aichat1.md) 和 [`aichat2.md`](aichat2.md) 的独立架构分析；
- 现有真机结果与当前性能差距；
- Triton、TileLang 的官方文档与原始资料。

当前生产信息流已经是：

```text
Core-local blocked DSL kernel
→ canonical Kernel IR
→ RISCVKernelFacts / ordinary use-def / memory / lifetime facts
→ op-specific compile rules
→ shared axis mapping / reuse / schedule / resource planning
→ selected value shape / handoff / local implementation
→ intrinsic C / local inline asm
→ system C compiler
```

这条链已经不是一个概念图。源码中存在以下真实结构：

- `RISCVKernelFacts` 保存 axis identity、定义与最后使用、普通 consumer、control carry、pointer/index 与各轴的 memory relation；
- `RISCVReuseAnalysis` 推导 operand 是否随 reduction 推进、是否可 reload、address/predicate 是否依赖 accumulator、consumer 数和 control crossing；
- `CorePhysicalMapping` 保存 logical axis 到 sequential / lane / register / unroll / fragment 的分解；
- `LocalMicrokernelSchedule` 保存 iteration、accumulator、operand window、decode 与 pipeline action；
- 当前已有 VLA entity 与 local primitive 各自的资源计算，能够计入其作用域内的 value、memory、predicate、state、accumulator、handoff 及部分 temporary/fragment；两者尚未合并为覆盖外围 value 与 local primitive 同时存活关系的完整 global live-resource 计算；
- emitter 在 physical decision 完成后才生成 C，并只收集实际使用的 intrinsic / asm leaf。

当前也已经删除了下列错误 authority：

- 以 `RVVStrip`、`RegisterMicrokernel`、`IMEFragment` 作为完整结构入口；
- lhs-VLA、rhs-VLA、local-row 等 whole-structure 分类；
- kernel-name、format-name、VLEN-name 驱动的完整实现 route；
- legacy、scalar、GGML 或 materials fallback；
- emitter 阶段再次选择 LMUL、microtile、decode chunk 或 fragment。

所以现在的问题不是“有没有编译器”，而是这个编译器的物理表示和可生成实现空间是否足够成熟。

---

## 二、最终冻结的 DSL 编程模型

### 2.1 根执行单位

一份 Weft kernel 是由调用者在当前 hart 上调用的一段普通 AOT 函数。一次调用从入口持续执行到返回，可以顺序处理多个 block；block、accumulator、state、workspace 与 packed object 的 lifetime 由作者程序的 lexical/control/storage relation 决定。

```text
external runtime
→ 划分一个 worker-local work slice
→ 调用 Weft entry(args..., slice...)
→ 当前 hart 持续执行 scalar control + VLA + local primitive
```

Weft core 中没有：

- grid；
- `program_id`；
- 隐式 hart / worker identity；
- task launch；
- 多 hart 协作一个 tile 的同步语义。

调用者若需要 row range、tile coordinate、expert range 或 ragged descriptor，必须把它们作为普通参数传入。多线程、线程池、affinity 与 NUMA 属于外部 runtime。

### 2.2 作者真正编写的内容

作者编写的是完整的 worker-local CPU kernel，而不是只写一个数学算子：

- scalar `for / while / if` 与 effect 顺序；
- outer traversal 与 cache / algorithmic blocking；
- staging、workspace、persistent packed layout；
- pointer、index、predicate、mask 与 alias/alignment 承诺；
- block 与 accumulator 的创建、更新、多 use 和跨 loop 生命周期；
- reduce、scan、summary、sequential carry 的具体 state 语义；
- 一遍还是多遍、重算还是保存；
- 显式 `W.vla`、`W.dot/W.matmul`、lookup/decode/quant 与局部 extension semantic primitive；
- 算法 variant 和 author-visible numerical policy。

这些内容不能由 compiler 从普通 SSA graph 猜出来，也不能为了性能被 target 偷换。

### 2.3 DSL 中的两类一等逻辑数据域

#### VLA iteration region

`W.vla(begin, end)` 表示一个运行时长度的一维逻辑迭代域。它授权 target 把该域 strip-mine 成动态 `vl`，但作者不能观察 strip 数、当前 `vl`、physical lane、VLEN 或 LMUL。

VLA 是 pointwise、memory、reduce、scan 与跨 strip summary 的主要逻辑域。普通 scalar loop 不会因为看起来可并行就自动变成 VLA。

#### Logical block value

`W.block` 建立具有唯一 axis identity 的 logical block domain。Logical block：

- 是普通 SSA value；
- 可以有多个 consumer；
- 可以进入 pointwise、state、memory、control carry 或其他合法 structured primitive；
- 可以跨 `for / while / if` 存活；
- 不等于 cache block；
- 不等于 register microtile；
- 不等于 IME fragment；
- 不对应一个独立 worker 或 program instance。

`dot → add → store`、dot 的两个 consumer、`matmul → pointwise → reduce/store` 和 block/state loop carry 都必须通过普通 SSA/use-def 成立，不能为每种组合建立 closure matcher。

### 2.4 DSL 操作的最终分类

| 类别 | 作者写出的语义 | 对 compiler 的授权 |
|---|---|---|
| scalar control | 有序 `for/while/if`、carry、effect order | 保持语义；不自动交换 traversal，不自动 SIMD 化 |
| VLA | 一维 logical SIMD domain | strip-mining、`vl`、LMUL、mask、lane mapping |
| block/value | 带 axis identity 的 shaped SSA value | 决定 transient value/register arrangement 与 handoff |
| memory | pointer/index/predicate/effect、load/store、storage ownership | 选择 unit/strided/indexed/segment、hoist、reload、local pack |
| pointwise/cast | logical element correspondence、broadcast、SEW 变化 | 传播 layout，必要时产生 typed conversion |
| reduce | 只观察聚合结果的代数 | 在 order/math policy 允许范围内选择 sequential/lane/extension realization |
| scan | 每个 logical position 的 prefix 和顺序 | 只能选择保持 prefix/order 的实现 |
| typed summary | argmax、online summary 等完整局部 state algebra | 只重组 primitive 内部，不拥有 surrounding traversal |
| sequential carry | 作者写出的逐 iteration state transition | 不得被猜成 reduce/scan/summary |
| `W.dot` | 固定 local block reduction relation | 重组当前 reduction domain，生成 RVV/register microkernel |
| `W.matmul` | `[M,K] × [K,N] + init → [M,N]` | 生成当前 block product 的 RVV/IME 微内核 |
| lookup/decode/quant | 普通 SSA 无法无歧义恢复的局部 byte/numerical relation | 选择 decode/gather/widen/accumulate arrangement 与硬件 leaf |
| extension semantic op | 新的、局部且可观察的 target-independent semantic relation | 为该语义增加 target realization；不得拥有 kernel ABI/outer control |

表中是已经冻结的语义类别，不代表每个 shape/dtype 组合都已有 target artifact。当前正式范围仍是：`W.dot` 只接受已登记的三类 shape，multiplicand 与 accumulator 为 f32；`W.matmul` 当前只接受 local `[M,K] × [K,N] → [M,N]`，两侧同为 f16 或 f32，accumulator/result 为 f32。扩大 dtype、rank 或 domain 必须补齐 frontend、canonical verifier 与真实 RISC-V lowering，不能由这张分类表自动推定为已支持。

### 2.5 什么时候应增加一个 DSL primitive

一个新 primitive 必须同时满足：

1. 它具有完整、局部、可观察的语义；
2. 它的 operand、result、validity、order、effect 与 numerical policy 可以独立说明；
3. result 是普通 SSA value，而不是某条 emitter 的 terminal；
4. 它不拥有 outer traversal、blocking、staging、persistent storage 或 entry ABI；
5. 普通 SSA 无法无歧义恢复该语义，或者作者必须显式授予 compiler 该局部重组权；
6. 它不是某个 kernel 名、模型格式名或一份完整算法的黑盒包装。

由此得到两条扩展规则：

```text
已有语义 + 新硬件实现
→ 不改 DSL，只增加新的 local realization

新的可观察局部语义
→ 增加 typed DSL/IR primitive，再为其提供 target realization
```

IME MMA 若实现现有 `W.matmul` 语义，不应成为一份新的 GEMM DSL。某种 packed quant 若具有独立 byte layout、scale/correction 和数值关系，则可以是 typed local primitive，但“Q4 kernel”不能成为 whole-kernel route。

### 2.6 Storage 与 lifetime

最终 storage 模型只保留四类：

| Storage | 分配者 | 生命周期 | 是否进入 entry ABI |
|---|---|---|---|
| external | caller | 至少一次调用，可由应用长期保存 | 是 |
| persistent | caller/部署系统 | 按显式 format 跨调用复用 | 是 |
| workspace | caller | 当前 worker 的一次调用，可跨 loop/primitive 复用 | 是 |
| primitive-private temporary | target lowering | 单个局部 realization 内 | 否 |

编译器不能为了命中实现偷偷创建 workspace ABI、persistent repack 或 author-visible staging。反过来，作者不能命名寄存器 temporary、LMUL group 或 fragment register。

### 2.7 DSL 中永远不出现的内容

- LMUL、exact `vl`、fixed lane count；
- 物理 vector type 与 register number；
- register microtile、accumulator register group；
- IME fragment register 编号；
- RVV intrinsic 名、asm constraint；
- target 名、VLEN128/VLEN256 route；
- tuning winner；
- grid、warp、CTA、thread identity。

这些都是 target lowering 或 system compiler 的责任。

### 2.8 当前正式能力边界，不等于新的程序模型

以下是当前实现明确拒绝的程序，不应被“Core-local blocked model”这个总定义误读为已经可生成 artifact：

- 同一 lexical scope 只允许一个活跃 VLA；nested VLA 明确 unsupported；
- runtime-extent 二维 block 的通用 materialized load 当前没有 RISC-V artifact；
- 二维 block 到一维 block 的 ordered reduction 当前没有 RISC-V artifact；
- 部分 VLA transcendental、block dtype/rank 与 cast 组合仍受当前 target capability 限制。

这些边界必须明确失败，不能走 scalar 或旧 emitter fallback。未来若增加 nested/multi-dimensional VLA，需要先定义新的 source semantics；其余 shape/dtype artifact 可以在不改变根程序模型的前提下扩展 target lowering。

---

## 三、GEMM 是程序模型的中心判据

Weft GEMM 必须由作者显式表达：

```text
M/N/K outer traversal
BM/BN/BK 的使用位置
A/B block 的地址、mask 与 staging
accumulator 的创建及跨 K-loop 生命周期
persistent packing 是否存在
epilogue pointwise 与 store
```

`W.matmul` 只表达：

```text
[BM, BK] × [BK, BN] + [BM, BN] → [BM, BN]
```

Compiler 可以在这个 local product 内选择：

- N-lane 或 M-lane RVV 组织；
- 多行/多列 register accumulator；
- MR/NR、K-unroll、operand window；
- load/compute overlap 与 register buffering；
- 短生命周期 packing；
- RVV FMA 或 IME fragment。

Compiler 不能替作者创建 K loop、cache blocking、persistent repack 或另一种 GEMM traversal。

这说明高性能 local microkernel schema 并不是被禁止的“模板”。必须区分三件事：

| 层次 | 是否允许 | 原因 |
|---|---|---|
| 作者写出的 GEMM algorithm/blocking | 必须存在于 DSL | 决定可观察 traversal、reuse 与 storage lifetime |
| 参数化的 local matmul lowering schema | 必须由 compiler 提供 | 把相同语义映射到 RVV/IME、microtile 与 pipeline |
| 固定 whole-kernel GEMM 实现 | 禁止 | 会接管 outer control、ABI、staging 和算法 variant |

因此，“更细 layout”与“高性能 GEMM schema”不冲突：layout、reuse 与 resource facts 应实例化 schema；schema 不应绕过 layout 变成一段按 kernel 匹配的完整代码。

---

## 四、最终冻结的编译器抽象

### 4.1 三条边界

#### DSL → compiler

输入只有：

- canonical Kernel IR；
- bound constexpr/meta values；
- target facts；
- 显式 backend candidate config。

Kernel IR 固定作者语义。Compiler 可以分析它，但不能补写算法。

#### compiler 内部

Compiler 负责：

```text
facts
→ 唯一合法推导
→ structural choices
→ parameter instances
→ resource legality
→ selected physical decision
```

这部分是 Weft 的核心，不应退化为 leaf selector，也不应被 emitter 重做。

#### compiler → primitive / ISA

Selected decision 指定一个局部硬件 operation 及其完整 operand/result arrangement、schedule、resource 和 handoff。RVV intrinsic 或 IME asm 只机械实现它。

### 4.2 形式化决策空间

令：

```text
P = canonical Kernel IR program
T = RISC-V target profile
B = explicit backend candidate bounds/config
M = bound meta values
```

首先形成输入事实：

```text
F = Facts(P, T, B, M)
```

唯一合法推导为：

```text
D = Derive(F)
```

合法候选空间为：

```text
C(P,T,B,M) = {
    (L, S, H, θ)
    |
    SemanticLegal(P, L, S, H)
    ∧ DependencyLegal(P, S)
    ∧ MemoryLegal(P, L, S, T)
    ∧ TargetLegal(T, L, H, θ)
    ∧ HandoffConsistent(P, L)
    ∧ ResourceLegal(P, L, S, H, θ, T)
}
```

其中：

- `L`：值与 operand/result 的 physical layout / arrangement；
- `S`：primitive 内部及显式 loop 内的局部 schedule；
- `H`：RVV、IME 或其他 local hardware operation；
- `θ`：LMUL、microtile factor、unroll、buffer count 等参数。

构建期选择只允许在 `C` 中选择：

```text
q* = argmin(q ∈ C) measured_runtime(compile(q), run(q))
```

Measurement 不定义新结构，不把非法候选变合法，也不写回 Kernel IR。

### 4.3 输入事实不是编译决定

以下内容是事实：

- logical axis identity 与 extent；
- free / reduction / broadcast / packed / group / state role；
- scalar loop nesting 与 ordered parent；
- definition、consumer、last use、control carry；
- pointer/index 相对每条轴的 unit/strided/indexed/non-affine relation；
- predicate、validity、effect 与 alias；
- state algebra、order 与 numerical policy；
- explicit dot/matmul/decode/lookup/quant semantics；
- ISA、ABI、endianness、VLEN、SEW/LMUL legality、vector register 数、IME capability。

VLEN128、VLEN256、Q4 或 target 型号只是事实，不能成为 implementation identity。

### 4.4 唯一合法推导

唯一合法推导没有优化偏好。给定相同输入，结果必须唯一：

- dot/matmul 的 free、reduction 与 broadcast 轴；
- pointer 的 memory relation；
- validity、tail 与 mask relation；
- cast/widen/narrow 的 logical element correspondence；
- ordered state 允许或禁止的重排；
- use-def、lifetime、loop invariance 与真实 dependence；
- target 是否支持某个 dtype、memory form、fragment 或 instruction；
- 某候选是否违反 resource limit。

这一层算错会导致错误结果或非法机器代码，因此不能由 cost heuristic、tuner 或 emitter决定。

### 4.5 结构性选择

结构性选择决定“生成哪一种局部机器程序拓扑”，包括：

- 哪条 logical axis 进入 SIMD lanes；
- 哪些 free-axis factors 形成 register repetitions / multiple accumulators；
- reduction 使用顺序、lane collective、unroll 还是 fragment；
- RVV register microkernel 或 IME fragment；
- scalar state 或 vector state；
- decode materialize 或 decode-compute fusion；
- share、reload、rematerialize、shuffle 或 local pack；
- unit/strided/indexed/segment memory schedule；
- single-buffer、double-buffer 或更深的合法 local pipeline topology；
- operand/result 在 RVV 与 extension fragment 之间的 handoff topology。

结构性选择必须由 op-specific semantic constraints 与共享 physical planner 共同产生。它不能由 kernel 名、格式名或完整源码 closure 产生。

### 4.6 参数性选择

参数只实例化已经确定的结构拓扑：

- LMUL；
- lane factor；
- MR/NR 或 register factor；
- accumulator 数；
- K-unroll；
- pipeline buffer count；
- prefetch distance；
- 同一 instruction family 内的合法 fragment size；
- DSL 已显式暴露的 constexpr BM/BN/BK。

若改变某个值会改变 producer/consumer topology、memory ownership 或 pipeline dependence，它就不是普通参数，而是结构性选择。

### 4.7 资源合法性

资源必须从最终 mapping、layout 与 schedule 统一计算：

```text
operand values
+ accumulator tuple
+ mask / predicate
+ index / coordinate
+ state
+ cast / widen / narrow temporary
+ decode / gather temporary
+ pipeline buffers
+ extension fragments
+ value handoff
+ reserved target groups
```

候选超出目标资源时整体非法。Emitter 不得通过缩小 LMUL、减少 accumulator 或切换隐藏 leaf 来“修复”它。

### 4.8 选择与发射

Compiler 可以先用 deterministic target cost 排序，也可以由构建期 compile-and-measure 选 winner；两者都只消费合法候选。

Emitter 只负责：

- C ABI 与函数声明；
- ordinary C control/address expression；
- typed local variable；
- RVV intrinsic 拼写与 API 适配；
- `vsetvl`、mask/inactive policy；
- IME / extension asm、clobber 与 constraint；
- C header metadata。

Emitter 不得再次扫描 IR 决定 layout、LMUL、microtile、decode chunk、table width、fragment 或 pipeline。

System C compiler继续负责最终寄存器分配、最终机器调度、peephole、常量折叠和机器码生成。Weft 不应向 LLVM backend 的职责扩张。

---

## 五、统一的不是 op 语义，而是物理表示与组合方法

Weft 不应建设“所有 op 都交给一个万能约束求解器”的模型。每个显式 op 必须保留自己的编译规则，但这些规则只贡献语义约束和可用 local realization，不返回完整 kernel 模板。

| Op | 贡献的约束 | 共享 planner 可选择的内容 |
|---|---|---|
| pointwise | 相同 logical element 一一对应；broadcast axis 为 singleton | 共享 operand layout，必要时 handoff/convert |
| load/store | pointer 对各轴的 relation、alignment、effect、predicate | lane axis、unit/strided/indexed/segment、hoist/reload |
| cast/widen/narrow | logical element 不变、SEW 改变 | 新 LMUL footprint、convert/handoff |
| reduce | 被消去的轴、identity、order/math | sequential、lane collective、local extension realization |
| scan | prefix 与 ordered carry | 保序 schedule、state placement |
| summary/state | 完整 state algebra 与 carry | scalar/vector placement、跨 strip representation |
| dot/matmul | free/reduction axes、init/result、dtype | lane/register/unroll/fragment、microkernel、pipeline |
| lookup/decode/quant | packed/group axis、byte relation、scale/correction | decode arrangement、gather、widen、fusion、accumulator layout |
| extension op | 独立局部语义 | 支持该语义的 target-specific hardware operation |

共享的部分是：

- physical layout/arrangement 的表示；
- layout 沿普通 value-use chain 的传播；
- reuse 与 dependence；
- local loop/schedule generation；
- resource calculation；
- target capability 与 legality；
- selected decision 与 handoff；
- mechanical emission contract。

这就是 Weft “统一扩展”的准确含义。统一不等于把 dense、state、quant、memory 的数值语义抹平；也不等于每种格式保留一份完整 microkernel。

---

## 六、当前 layout 为什么更粗

### 6.1 当前真实表示

当前 `CorePhysicalMapping` 对每条 logical axis 只记录：

```text
sequentialFactor
laneFactor
registerFactor
unrollFactor
fragmentFactor
```

并额外记录一个 `laneAxis`、一个 RVV lane shape、pipeline buffer count 与 fixed fragment resource groups。

这能回答：

- 哪条轴进入 RVV lane；
- 每条轴有多少 register repetition；
- 哪条 reduction 轴展开多少；
- fragment 覆盖每条轴多少元素；
- 剩余多少局部顺序 iteration。

它已经是一个真实的 core-local layout 骨架，但不是完整的数据分布描述。

### 6.2 为什么它可以比 Triton 粗

Triton/Gluon 的 layout 显式描述逻辑 tensor 元素如何分布到 thread block、warp、lane 与每个 lane 的 registers；官方 layout 文档直接把 layout 定义为 register/lane/warp 所拥有 element 的映射。Triton 的根模型还包含 SPMD launch grid 和 `program_id`。[Triton Vector Add](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)、[Triton Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)、[Triton/Gluon Tensor Layouts](https://triton-lang.org/main/getting-started/tutorials/gluon/layouts.html)

Weft 的 target hierarchy 更简单：

```text
one ordered worker stream
+ one scalable RVV lane domain
+ multiple register values/accumulators
+ optional fixed extension fragment
```

它没有 CTA、warp、thread ownership、shared-memory bank mapping 或 inter-thread communication。因此 Weft 没有理由复制 Triton 完整的 CTA/warp/lane/register LinearLayout，也不应把 GPU execution hierarchy写进 DSL/Kernel IR。

RVV 的 exact `vl` 又是动态的；物理寄存器编号最终由 system compiler分配。一个静态 GPU-style per-thread/per-register map既不自然，也会错误侵入下层寄存器分配。

### 6.3 为什么当前又确实过粗

“目标层级更简单”不能成为隐藏真实布局信息的理由。当前 factor-only mapping 尚不能独立表达：

- operand 的 logical axes 分别如何对应 lane/register/fragment coordinates；
- 多个 accumulator 与 M/N free axes 的 correspondence；
- row-major 与 column-major register tuple；
- packed nibble、high-bit、bitplane decode 后的 element order；
- cross-axis register interleaving；
- codebook/gather index 与 decoded values 的 grouping；
- lhs/rhs/accumulator 的 IME fragment coordinates；
- RVV value 到 IME fragment、再回 RVV value 的具体 handoff arrangement。

这些信息目前部分藏在：

- `LocalOperationProjection` 的少数 enum；
- `valueShapes[]` 的位置约定；
- quant leaf 的 typed helper body；
- IME leaf 的固定寄存器约定；
- helper symbol suffix。

如果继续只扩充 projection enum 或 leaf helper，Weft 会再次变成“共享外壳 + family-specific microkernel 黑盒”。这是当前最大的架构风险。

---

## 七、layout 到底要细到哪里

### 7.1 最终决定：需要做细，但不做成通用 GPU LinearLayout

应保留现有 axis-factor decomposition 作为 mapping skeleton，同时在同一次 target lowering 中增加 typed、operand-aware 的 physical arrangement。它仍然是瞬态决定，不成为新的持久 IR。

最小完整表示应包含六部分。

#### 1. Axis placement

每条 logical axis 到：

```text
sequential / lane / register / unroll / fragment
```

的 factor 与顺序。当前模型基本已具备。

#### 2. Operand/result layout

对每个 value 明确：

- 参与哪些 logical axes；
- 各轴对 lane/register/fragment 的贡献；
- element ordering；
- memory order 与每 step 的 vectors；
- 哪些 coordinate 随 iteration 推进；
- 与 producer/consumer 的 arrangement compatibility。

现有 `LocalOperandWindow` 只能继续承担 schedule/reuse 信息，不能代替 layout。

#### 3. Accumulator layout

明确：

- 每个 accumulator 对应哪些 free-axis coordinates；
- accumulator tuple 的 row/column ordering；
- reduction update order；
- multiple consumer 如何读取同一个 result arrangement。

仅有 `accumulatorCount` 不足以描述这些关系。

#### 4. Packed/decode layout

明确：

- packed/group axis decomposition；
- nibble、high-bit、bitplane extraction order；
- decode chunk、group 与 reduction segment；
- decoded element 到 lane/register 的映射；
- sign、scale、minimum/correction 与 accumulator 的 correspondence。

Typed quant op 继续定义数值语义，layout 定义机器排列；两者不能混成一个格式 leaf。

#### 5. Fragment layout

扩展 capability 至少提供：

- M/N/K axis correspondence；
- lhs/rhs/accumulator fragment coordinates；
- dtype、mask/tail 与 shape constraints；
- fixed register groups；
- result arrangement。

当前只有 axis factor 与 fixed resource groups，不足以独立说明 fragment ABI。

#### 6. Handoff layout

每个 `(consumer, value)` 不只记录 source/result RVV shape，还应记录 arrangement conversion：

```text
share
convert
reload
rematerialize
shuffle
local pack
RVV ↔ fragment conversion
```

同一个 `e32m1` shape 可以具有不同 element arrangement，shape 相同不能再被视为必然可直接 share。

### 7.2 推荐的概念形式

对每个 transient value `v`，physical layout 可概念性表达为：

```text
L_v : logical coordinates
      → (sequential coordinate,
         dynamic lane coordinate,
         register-tuple coordinate,
         optional fragment coordinate)
```

这是一种 target-lowering relation，不要求持久化，也不要求指定物理寄存器编号。RVV lane coordinate 可以符号化依赖当前 `vl`；fragment coordinate 则由 target capability 给出固定局部关系。

### 7.3 明确不做的细化

不应加入：

- CTA/warp/thread hierarchy；
- 物理寄存器编号；
- system compiler 的最终 register allocation；
- 为了理论统一而设计任意 bit-level affine/linear map；
- 没有真实 consumer 的 swizzle algebra；
- 持久 Physical IR、Selected IR 或 layout dialect；
- DSL 中可观察的 lane/register/fragment coordinate。

只有真实 RVV/quant/IME consumer 需要的 arrangement 才进入这份瞬态 schema。目标是消除 leaf 私约定，不是复制一套 GPU compiler。

---

## 八、layout、schedule、hardware operation 必须分开

这三个概念现在容易混在 `LocalImplementation` 中，最终定义应当明确：

| 概念 | 回答的问题 | 例子 |
|---|---|---|
| physical layout | logical value 在 core 内怎样分布 | N→lane、M→accumulator tuple、K→unroll、decoded nibble→lane/register |
| local schedule | 这些 value 与 operation 何时产生和消费 | load order、prologue/steady/epilogue、double buffer、prefetch distance |
| hardware operation | 哪条局部机器能力执行 semantic primitive | RVV widening FMA、indexed gather、IME MMA |
| parameter | 同一结构实例的大小 | LMUL、MR、NR、unroll、buffer count |

Layout 不应偷偷编码 operation order；schedule 不应重新选择 value layout；hardware leaf 不应拥有 outer loop 或完整 local traversal。

### 自动局部流水的最终边界

Compiler 只能在作者已经写出的 loop 内分析：

```text
load / address / index
→ decode / lookup / widen / local pack
→ compute primitive
→ loop-carried accumulator/state
```

并生成：

```text
prologue
steady state
epilogue
single/double/deeper legal buffer
```

可移动性由 use-def、effect、alias、state order 与 lifetime 唯一决定；结构和参数由 planner 选择。

当前 pipeline 主要围绕单个 primitive 的 Load/Compute actions，作用域仍太小。FlashAttention 的 load→dot→summary→exp→V load→weighted update，或 quant 的 packed load→decode→widen→dot，需要显式 loop 内的跨 op scheduler。这个扩展不等于发明新的算法 pass，也不能越过作者 loop 或创建 workspace。

---

## 九、RISC-V 原语与新扩展如何统一接入

### 9.1 Local hardware operation 的合同

每个 RVV、IME 或未来扩展 realization 必须声明：

```text
实现哪个显式 semantic primitive
接受哪些 dtype / axis relation / validity
接受哪些 operand/result physical arrangements
支持哪些 memory relation、mask、tail、order 与 numerical policy
占用多少 operand / accumulator / temporary / fragment resources
允许哪些 unroll、buffering 与 local packing
产生何种 result arrangement 与 handoff
如何拼写 intrinsic C 或 typed local asm
```

它不能拥有：

- entry ABI；
- outer traversal；
- algorithmic/cache blocking；
- author workspace；
- persistent packing；
- 完整 state machine；
- 完整 kernel 名称或调度 route。

### 9.2 Target profile 只提供事实

统一 target profile 提供：

- triple、ISA、ABI、endianness；
- VLEN、SEW 与 LMUL legality；
- vector register budget；
- F16、indexed、segment、widening 等 RVV 能力；
- IME 或 vendor extension capability；
- fragment M/N/K factor 与固定资源限制。

SG2044 和 K1 读取同一套 compiler rules，只因 target facts 不同而得到不同合法 mappings。VLEN 或 target 型号不能成为代码路径身份。

### 9.3 IME 的正确位置

IME 是现有 `W.matmul` 或 typed quant primitive 的一种 local realization：

```text
same semantic op
→ same axis/value/reuse facts
→ RVV candidate or IME fragment candidate
→ same resource and handoff authority
→ intrinsic C or local asm
```

新增矩阵扩展应只增加 fragment constraint、resource、handoff 和 instruction spelling。GEMM、Conv、MoE、Attention、OutProd 若含相同 local matmul relation，应自动获得这项 candidate；不新增六条 extension-specific kernel route。

### 9.4 Quant 的正确位置

不同 quant primitive 可以保留不同的局部 byte/numerical semantics，但以下结构必须共享：

- packed/group axis mapping；
- cross-output reuse；
- decode/gather/widen arrangement；
- accumulator organization；
- operand window 与 pipeline；
- resource calculation；
- RVV/IME selection 与 handoff。

Leaf 只保留不可再分解的指令序列和具体 intrinsic/asm 拼写。

---

## 十、与 Triton、TileLang 的真正差异

### 10.1 对照表

| 维度 | Triton | TileLang 的主流 GPU 路径 | Weft |
|---|---|---|---|
| 根执行单位 | SPMD launch grid 中的 program instance | `T.Kernel(grid..., threads=...)` 中的 block/thread context | caller 调用的单 worker/hart 持续函数 |
| 工作身份 | `program_id` 可见 | block index、threads 可见 | 无隐式 identity；工作坐标是普通参数 |
| program owner | 一个 program/CTA 通常拥有一个结果 tile | 一个 thread block 拥有 tile 与 shared/fragment buffers | 一个 worker 可顺序拥有多个 block/state |
| outer traversal | program grid + kernel 内 K/control | grid、tile loops、`T.serial/Parallel/Pipelined` | 作者的普通 scalar `for/while/if` |
| tile/block lifetime | 通常围绕一次 program instance | 围绕 thread block 与显式 memory scopes | 可跨作者 loop、多个 block 与 primitive 持续存活 |
| storage | GPU global/shared/register hierarchy | 显式 `alloc_shared/alloc_fragment/copy` | caller-owned external/persistent/workspace；register/fragment 不进入 DSL |
| layout | logical element→CTA/warp/lane/register | layout inference + explicit hardware-aware scope/thread primitives | logical axis/value→sequential/lane/register/fragment transient arrangement |
| structured compute | `tl.dot` 等 op-specific lowering | `T.gemm`、tile library、thread primitives | `W.dot/W.matmul` 与 typed local primitives |
| pipeline | compiler passes + meta configuration | `T.Pipelined`，可显式 stage/order | 作者 loop 内由 dependence/resource 生成的 local pipeline |
| tuning | block size、warps、stages 等 | tile size、threads、stages、layout/schedule knobs | legal physical mapping/parameter + explicit constexpr candidate |
| extension | backend layout/instruction lowering | tile op/thread primitive/target backend | 同一 semantic primitive 的新 local realization |
| runtime | GPU launch/JIT ecosystem | JIT/target executable | AOT C/object/header，外部 CPU runtime 调用 |

### 10.2 Weft 与 Triton 的共同点

- 都让作者显式写 tile/block-level data relation，而不是输入完整 graph；
- 都需要 op-specific semantics，而不是纯通用 SSA 自动发现；
- 都需要 logical value 到 machine layout 的传播；
- dot/matmul 等 op 需要 target-specific microkernel/tensor instruction lowering；
- layout、memory coalescing/relation、resource 和 pipeline 共同决定性能；
- autotuning 只能在合法实现空间中选择。

### 10.3 Weft 与 Triton 的本质差异

差异不在“有没有 tile”，而在：

1. **owner**：Triton program instance 逻辑拥有 tile；Weft worker 持续拥有多个 block/state；
2. **control**：Triton 以 grid/program identity 组织外层并行；Weft 以作者显式的 CPU ordered control 组织；
3. **lifetime**：Weft block、state、workspace 可以跨 loop 和多个 primitive 存活；
4. **SIMD 语义**：Weft VLA 的 exact strip/lane 不可观察，适配 RVV scalable vectors；
5. **storage**：Weft 不向作者暴露 GPU shared/warp memory hierarchy；
6. **product boundary**：Weft 产出可被普通 CPU runtime 调用的 AOT worker-local entry。

这些差异足以构成独立编程模型，而不是 Triton CPU backend 的语法变体。

### 10.4 Weft 与 TileLang 的差异

TileLang 官方主流模型显式使用 `T.Kernel(grid, threads)`、shared/fragment allocation、`T.copy`、`T.gemm` 与 `T.Pipelined`。用户可以处于 hardware-unaware、tile-library 或 expert thread-primitive 层；layout inference 决定 fragment/register distribution。[TileLang Language Basics](https://www.tilelang.com/programming_guides/language_basics.html)、[TileLang Overview](https://www.tilelang.com/get_started/overview.html)、[TileLang Software Pipeline](https://www.tilelang.com/programming_guides/software_pipeline.html)

TileLang 更接近“显式 GPU tile dataflow + 可逐步暴露硬件 hierarchy”；Weft 更接近“显式 CPU worker algorithm + compiler-private core layout”。

Weft 可以借鉴 TileLang：

- op-specific lowering 与 layout inference 分工；
- effectful producer/consumer 与 replayable SSA 的 pipeline 区分；
- data movement、compute、layout 与 pipeline 的职责拆分；
- typed target operation 的扩展接口。

Weft 不应复制：

- grid/block/thread identity；
- shared/fragment 作为 canonical source storage scope；
- 用户可观察的 physical lane/register layout；
- GPU barrier/synchronization 根模型；
- whole-operator tile library 作为 production fallback。

### 10.5 CPU/RVV 差异不能成为哪些借口

“CPU 比 GPU 层级简单”不能用来解释：

- layout 信息继续藏在 quant/IME leaf；
- register accumulator correspondence 不被建模；
- value 多 consumer 依赖特殊 closure；
- scheduler 只能处理一个 Load/Compute pair；
- target cost 与资源模型长期只靠固定 heuristic；
- 每种格式重新长出完整 microkernel；
- 性能差时恢复 whole-kernel asm。

Weft 可以比 Triton layout 更窄，但不能比真实 RISC-V 高性能实现所需的信息更粗。

---

## 十一、Weft 的创新点应如何定性

以下内容可以作为项目的核心贡献方向，但不能在缺少文献与实验对照时直接宣称“首次”。

### 11.1 Core-local blocked programming model

Weft 把 CPU worker 的持续 ordered control、跨 loop block/state lifetime、caller-owned storage 与局部 structured operations 放在同一份 kernel DSL 中。它既不是 graph，又不是 GPU grid tile，也不是 intrinsic wrapper。

### 11.2 Op-constrained core physical mapping

每种 semantic op 保留自己的规则，但共同约束一份 core-local physical arrangement：

```text
logical axes / values / memory / lifetime
→ sequential / lane / register / unroll / fragment
```

这种设计避免 universal graph guessing，也避免 family template selection。

### 11.3 Ordinary structured SSA composition

Dot、matmul、state、decode 与 extension result 都是普通 SSA value。高性能能力依赖 semantic facts、layout、lifetime 与 handoff，而不是 one-use、direct-store 或 exact closure。

### 11.4 RISC-V extension as local realization

RVV、IME 与未来扩展以 local operation capability 扩张同一 physical candidate space，不新增完整算子 backend。这是 Weft 面向 RISC-V 多扩展生态最重要的可扩展性主张。

### 11.5 Single physical authority

Canonical Kernel IR 是唯一持久程序表示；layout、resource、candidate、winner 与 fragment 都是一次 lowering 的瞬态决定。每项决定只有一个 producer，emitter机械消费，避免新旧 route 和多份 IR authority。

### 11.6 创新不在什么地方

下列内容本身不是核心创新：

- Python 语法；
- 使用 MLIR；
- 生成 intrinsic C；
- 有 block type 或 `dot` op；
- 有 autotuning；
- 封装一段 IME asm；
- 支持若干量化格式。

真正需要由实验支撑的主张是：同一 DSL/IR 与同一物理编译方法，能让相同 semantic primitive 在不同控制、consumer、storage、target 与扩展上下文中组合地产生高质量实现，而不是累计更多 case。

---

## 十二、当前方案的好处、代价与风险

### 12.1 好处

#### 权责清楚

算法结构不会被 compiler 偷偷改变，物理机器细节也不会污染 DSL。

#### 可组合

Block、state 与 structured result 是普通 SSA；新增 consumer 只改变 lifetime、resource 与 handoff，不应改变 primitive 是否可 lowering。

#### 可扩展

新硬件能力只增加 local realization；相同 matmul/quant relation 的多个 kernel 都能获得候选。

#### 跨目标

SG2044 与 K1 共享 compiler 主干，目标差异只通过 VLEN、resource 与 extension facts体现。

#### 可嵌入

最终是普通 object/library/header，可进入 llama.cpp、GGML 或其他 CPU runtime，不需要 Weft runtime 接管多线程。

#### 避免图编译器回退

Compiler 不需要识别“这是 Attention/GEMM/Q4 kernel”，只编译作者明确写出的 local semantics 和 value relations。

### 12.2 代价

#### 作者负担更高

作者必须理解 CPU blocking、staging、persistent packing 与 state 算法。Weft 不承诺从纯数学表达自动发现高性能 CPU algorithm。

#### 性能依赖 DSL 与 compiler 双方正确

算法结构写错不能靠 target 自动补；算法写对后，compiler 仍必须有足够强的 layout/reuse/schedule/candidate。

#### Typed primitive 设计需要克制

太少会迫使 compiler 从 SSA 猜语义；太多会变成每个格式/算子一个黑盒 op。

#### 编译器实现复杂

Physical layout、use-chain propagation、cross-op schedule、resource 与 target leaf 必须协调，复杂度不会因为不做 graph compiler而消失。

### 12.3 最大风险

1. **过粗 layout**：真实 arrangement 继续泄漏到 leaf，最终重新形成 family microkernel 库；
2. **过细 layout**：复制 Triton/LLVM 全套抽象，侵入 register allocation 与无用 GPU hierarchy；
3. **primitive 膨胀**：每个 kernel 或格式变成一个 op；
4. **scheduler 越权**：为了性能发明作者未写出的 blocking/staging/algorithm pass；
5. **tuner 掩盖缺结构**：只调 LMUL/常数，却没有真正的 cross-output reuse 或 pipeline candidate；
6. **局部资源假象**：primitive 自己合法，但与外围 VLA/state/live values 合并后 spill；
7. **目标特例回生**：K1、VLEN256 或 IME 重新变成 route 名。

冻结后的判断标准很简单：新增代码若不能被描述为“op semantic rule、shared physical arrangement/schedule/resource、local hardware operation 或 mechanical spelling”中的一个，就很可能放错了层。

---

## 十三、后续性能到底应该做哪一部分

### 13.1 先做所有权判断

每个性能问题必须先问四个问题：

1. Baseline 是否使用了不同 outer traversal、blocking、staging、persistent packing 或算法 variant？
   - 是：修改 DSL kernel，必要时补最小语言语义。
2. DSL 是否已经写清算法，但 value layout、reuse、memory、pipeline、resource 不足？
   - 是：修改 shared compiler core。
3. Selected physical program 是否正确，但 intrinsic/asm 指令序列低效？
   - 是：修改 local leaf/spelling。
4. 只剩最终 register allocation、machine scheduling 或 peephole？
   - 是：交给 system compiler。

不能直接从“慢”跳到新增 leaf。

### 13.2 当前主要差距的归属

当前报告已经表明：F16 dense、Q2、IME prefill、FlashAttention 与 K1 contiguous transpose仍明显落后，而部分 F32、Q3/Q4、softmax、dequant/get_rows 已证明共享 mapping 能产生有效实现。Transpose 暂时只能作为该项 memory-permutation/data-layout 问题的证据，不能外推为所有 irregular-memory kernel 都同样落后。

| 性能族 | 先检查 DSL | compiler 的主要责任 | leaf 的责任 |
|---|---|---|---|
| F16/F32 GEMM、Conv、MoE、OutProd | BM/BN/BK、staging、persistent layout、epilogue是否与 baseline 同算法 | operand/result layout、cross-row/output reuse、multiple accumulators、load window、K pipeline、global live resource、target cost | widening FMA 与最小 RVV/IME sequence |
| quant Q2/Q4/Q6/IQ/MXFP4 | packed layout、group traversal、activation staging 是否显式 | packed/decode layout、cross-output reuse、gather/widen/scale/correction schedule、resource | bit extraction、codebook gather、widen/dot intrinsic |
| IME decode/prefill | outer traversal与persistent input format是否一致 | fragment layout、operand reuse、RVV↔IME handoff、fragment pipeline、resource | MMA asm与最小 pack/load instruction |
| FlashAttention | 作者是否写了正确 online algorithm、staging与 state lifetime | 跨 op loop-local scheduler、state placement、Q/K/V reuse、window/load ordering | exp/reduction/FMA 等局部序列 |
| reduce/scan/RWKV/SSM | state algebra和ordered carry是否正确 | scalar/vector state placement、跨 strip/loop lifetime、resource | local reduction/scan intrinsic |
| transpose/CSR/col2im/embedding | baseline 是否用了不同 blocked traversal/data layout | coordinate/index hoist、gather/segment choice、window reuse、prefetch、multi-consumer load | segment/gather intrinsic |

### 13.3 当前最重要的 compiler 工作顺序

#### 第一：补 physical arrangement，而不是继续加 LMUL 常数

优先完成 operand、accumulator、packed/decode、fragment 与 handoff layout。没有这些信息，F16 cross-output、quant decode 和 IME fragment 都只能继续藏在 leaf。

#### 第二：把 reuse 与 schedule 扩到显式 loop 内的 operation chain

从单 primitive Load/Compute 扩展到：

```text
load → decode/lookup/widen → compute → state/consumer
```

同时生成真实 prologue/steady/epilogue 和 buffer lifetime。

#### 第三：合并完整 live resource

把外围 VLA/value/state 与 local primitive temporary、pipeline、fragment 合成同一时间轴上的 resource peaks，避免“局部合法、整体 spill”。

#### 第四：扩充共享结构候选

重点是 cross-output/register microtile、operand panel reuse、state placement、indexed/window scheduling 与 fragment reuse，不是新增 kernel/family route。

#### 第五：完成 target cost 与 compile-and-measure

候选空间存在以后，再让 SG2044/K1 按各自 target facts 与实测选择 LMUL、microtile、unroll、buffer 与 fragment。Tuner 不能替代前四步。

#### 第六：最后才抠 leaf spelling

只有 selected arrangement/schedule 已完整时，才根据 materials 中的真实实现吸收最小 intrinsic/asm sequence。

### 13.4 对 DSL 的结论

后续性能工作的主体在 compiler core，而不是再次扩写 DSL 根模型。

只有当 baseline 的性能来自作者可观察的不同 blocking、staging、workspace、persistent layout、state algorithm 或 algorithm variant，而当前 DSL kernel没有写出时，才修改 DSL kernel；只有现有语言无法无歧义表达这种结构时，才修改 DSL/Kernel IR。

不能把 LMUL、microtile、pipeline stage、fragment 或 target hint 加进 DSL 来规避 compiler工作。

---

## 十四、最终冻结清单

### 可以现在冻结

1. **项目定位**：RISC-V Core-local blocked kernel DSL/compiler，不是 graph compiler；
2. **执行模型**：单 worker/hart 持续 ordered program，无 grid/implicit identity；
3. **作者 ownership**：outer control、blocking、staging、storage、state algorithm、persistent layout；
4. **授权点**：显式 VLA、state algebra、dot/matmul、lookup/decode/quant、local extension semantic op；
5. **普通 SSA composition**：structured result 多 use、control carry、pointwise/state/memory consumption；
6. **唯一持久表示**：canonical Kernel IR；
7. **编译决策分层**：facts → unique derivation → structural choice → parameter choice → resource legality → selection → emission；
8. **共享物理坐标**：sequential / lane / register / unroll / fragment；
9. **op-specific rule + shared planner**：不做 universal graph guessing，不做 family whole-kernel selector；
10. **local leaf 边界**：leaf 只实现一个 primitive-local hardware operation；
11. **扩展方式**：已有语义增加 realization，新语义才增加 typed primitive；
12. **跨目标方式**：target profile 是事实，VLEN/型号不是 route；
13. **发射边界**：emitter机械消费 selected decision；
14. **system compiler 边界**：最终 register allocation 与 machine scheduling 下放。

### 不能把当前实现冻结成最终形态

1. 只有 axis factor 的 `CorePhysicalMapping`；
2. 无 operand/result coordinate relation 的 `valueShapes[]`；
3. 由 `LocalOperationProjection` 承担 packed/quant layout ABI；
4. 只有 accumulator 数、没有 accumulator/free-axis correspondence；
5. 只有 fragment factor、没有 lhs/rhs/result fragment layout；
6. 只按 RVV shape 判断的 handoff；
7. 只覆盖单 primitive Load/Compute 的 pipeline；
8. local primitive 与外围 VLA/state 分离的 resource peak；
9. 固定 preferred row/column/unroll 的 heuristic；
10. 当前不足的 F16、quant、IME、attention candidate space。

这些是已冻结架构内必须继续完成的 compiler能力，不是下一次重新设计编程模型的理由。

---

## 十五、最终一句话

Weft 最终不是“更粗的 Triton”，也不是“RISC-V 高性能 kernel 模板库”。它应当是：

> **一门让作者显式写出单个 CPU core 的完整 blocked/streaming algorithm、同时让编译器在每个显式局部语义授权点内，统一生成 scalable SIMD、register microtile、memory schedule、state placement 与 extension fragment 的 RISC-V kernel DSL。**

后续若性能工作坚持这条边界，Weft 可以持续长出更强的 layout、reuse、pipeline 与扩展实现，而不再推翻 DSL；若重新按 kernel、格式或 whole-region 选择高性能主体，则无论数字多快，都会退回旧的算子模板系统。
