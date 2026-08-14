# Weft Core-Local Blocked Programming Model

> **状态：本轮设计输入快照。** 稳定语义已经吸收到 `doc/index.md` 所链接的现行规范；本文保留问题来源、审计反例与设计推导，不再作为第二份持续维护的规范 authority。

---

## 1. 项目目标

Weft 要解决的是一个具体问题：

> RISC-V CPU 需要一门真正的高性能算子 DSL/编译器；它既要适合单个 hart 上的 SIMD、状态、控制流、blocking 与 packing，又要能在 RVV、矩阵扩展、量化扩展及 vendor extension 持续碎片化时，局部接入新能力，而不是重新实现整套算子。

Weft 的正式产物是：

```text
Weft DSL kernel
→ canonical worker-local Kernel IR
→ RISC-V physical realization
→ intrinsic C / primitive-local inline asm
→ GCC / Clang
→ object / library
```

Weft 不是：

- 从算子图自动选择算法的图编译器；
- 根据 kernel 名或 q-format 调手写实现的 dispatch library；
- 只给 RVV intrinsic 套 Python 外壳；
- 多核线程池、模型图或设备 runtime；
- 让编译器从普通乘加、普通循环中猜出 matmul、scan、online softmax 或量化格式的模式机。

---

## 2. 核心编程模型：Core-local blocked program

Weft 的根对象不是 GPU grid 中的 program instance，也不是一个孤立的 VLA loop，而是：

> **由一个 CPU worker/hart 执行的一份持续存在的有序 kernel 程序。该程序可以顺序处理多个数据块，持有局部 block、state 与 storage，使它们跨循环存活、更新和复用。**

一个 Weft kernel 由以下结构共同组成：

```text
有序 scalar control
+ logical blocks and state
+ explicit memory/storage lifetime
+ explicit wide/local primitives
```

### 2.1 有序控制程序

作者显式写：

- `for / while / if`；
- worker 负责的数据范围或 descriptor；
- loop order；
- cache/algorithm blocking；
- 一遍或多遍算法；
- state 的更新顺序；
- staging、scratch 与 persistent packing 的位置和生命周期。

这些是 kernel 实现结构，不是后端可以从数据流中重新发明的内容。

### 2.2 Logical block 与 state

Block 是固定或参数化 shape 的逻辑 SSA 值。它可以：

- 从 pointer/index/load 构造；
- 被 pointwise、reduce、scan、dot/matmul、lookup/decode 等消费；
- 跨 loop iteration 作为 accumulator 或 state 存活；
- 被多个普通 SSA consumer 使用；
- 最终写回 memory，或进入另一个 structured primitive。

Block 不预先绑定：

- RVV lane；
- LMUL；
- 物理寄存器；
- IME fragment；
- 具体 intrinsic。

但 block 的逻辑 shape、axis、extent、validity、dtype 和数值关系必须明确。

### 2.3 Explicit local primitive

局部 structured primitive 是作者显式授予编译器物理重组权的边界，例如：

- `W.vla`：显式一维 SIMD/VLA logical domain；
- `W.reduce / W.scan / W.argmax / W.online_softmax_summary`；
- `W.dot / W.matmul`；
- `W.lookup / W.decode`；
- 具有独立可观察数值语义的 packed/quant/extension primitive。

没有显式 primitive，target 不得从普通 SSA 图中猜测它。

局部 primitive 可以拥有 primitive 内部、短生命周期、对外不可观察的 packing、temporary 和 asm sequence；它不能拥有完整 kernel 的 ABI、outer traversal、algorithmic staging、persistent layout 或完整状态机。

---

## 3. Weft 与 Triton 的真正区别

Weft 不以“我们不用 tile”区别 Triton。Tile 是强大且可跨硬件使用的思想；Weft 也使用 block/tile 和可调 BM/BN/BK。

真正差异是 **tile/block 的执行合同和生命周期**。

### 3.1 Triton 的合同

Triton 的典型模型是：

```text
grid
→ program instance
→ program 逻辑拥有一个结果 tile
→ compiler 将 tile 分布到 warp / lane / register / shared memory
```

作者写 block algorithm、pointer、mask、K loop 和 `tl.dot`；具体 tile layout、线程分布、shared-memory conversion、MMA 和 pipeline 由 GPU backend 决定。

### 3.2 Weft 的合同

Weft 的模型是：

```text
external runtime 将一段工作交给一个 worker
→ worker 顺序执行一份持久 control program
→ program 可连续处理多个 cache blocks
→ block/state/storage 可跨循环存活和复用
→ compiler 将 block operation 映射到 SIMD registers、多个 accumulators 或扩展 fragment
```

因此 Weft 与 Triton 的差异不是“都有无 tile”，而是：

| | Triton | Weft |
|---|---|---|
| 根执行单位 | grid 中的 blocked program instance | 一个 CPU worker/hart 上的持续有序程序 |
| block 的拥有者 | program/CTA 逻辑拥有 | worker control program 持有并跨循环复用 |
| 多个 output blocks | 通常由多个 program instances 分担 | 一个 worker 可顺序处理多个 blocks |
| storage 生命周期 | 主要围绕一个 program/tile 与 GPU memory hierarchy | caller/persistent/worker-local/primitive-local 生命周期必须明确 |
| 物理化 | tile → warp/lane/register/shared/MMA | block/state → SIMD/register microkernel/extension fragment |
| 多核 | grid/launch 属于模型 | 外部 runtime 负责 |

Weft 的独立价值不在于语法不同，而在于：

> **它让作者直接表达 CPU 高性能 kernel 中 backend 无法可靠恢复的 block/state/storage 生命周期与有序复用结构，同时让编译器隐藏 ISA、寄存器和扩展细节。**

如果 Weft 只保留静态 block + `W.matmul`，却不正式表达 worker-local storage/lifetime 和自然 SSA composition，它就不足以区别于 Triton CPU backend。

---

## 4. GEMM：当前 DSL 应表达什么

当前 `gemm_worker` 已经体现了 Core-local blocked program 的一部分：

```text
worker 接收 m_begin / m_end
→ 顺序遍历多个 M/N cache blocks
→ accumulator block 跨 K loop 存活
→ A/B block 每个 BK 被构造和消费
→ W.matmul 只负责当前局部 block product
→ 最后由同一个 worker 写回
```

### 4.1 作者在 GEMM 中控制

- worker 的 M 范围；
- M/N/K loop order；
- BM/BN/BK blocking 的存在、位置与 meta-parameter；
- A/B/C pointer 与 index relation；
- mask；
- accumulator 的 shape 与跨 K-loop 生命周期；
- algorithmic staging；
- persistent packed A/B layout；
- 算法 variant。

这些信息比数学上的 `C=A×B` 更低，已经是具体 CPU kernel 结构。

### 4.2 `W.matmul` 控制的边界

`W.matmul(a_blk, b_blk, init=acc)` 只声明当前：

```text
[BM, BK] × [BK, BN] + [BM, BN]
```

的局部 block product。

Realizer 可以在该 primitive 内决定：

- RVV register microkernel；
- vectorized 方向与合法 memory realization；
- register microtile；
- multiple accumulators；
- K-unroll；
- primitive-local load schedule、packing 与 software pipeline；
- RVV、IME 或未来 tensor extension realization。

Realizer不得决定：

- 是否存在 outer K loop；
- 是否增加 persistent repack；
- 是否把 GEMM 改成另一种 traversal；
- 是否增加跨 primitive 的 staging；
- 是否改变 BM/BN/BK 在算法中的角色。

### 4.3 与 Triton GEMM 的核心区别

Triton 与 Weft 都允许作者写 BM/BN/BK，也都可以 autotune 具体值。两者的差异不是参数。

Triton GEMM 通常让一个 program instance负责一个 output tile，后端围绕 GPU collective execution 分配 tile。

Weft GEMM 让一个 worker 在普通 control flow 中持有 accumulator、顺序遍历多个 blocks，并显式拥有 storage/reuse lifetime。后端只把 `W.matmul` 等局部 block operation 物理化成 SIMD/microkernel/extension。

因此 Weft 对 CPU 作者真正简化的不是“替他选择 GEMM 算法”，而是：

- 不手写 RVV strip 与 tail；
- 不手写每种 dtype/width 的 intrinsic 链；
- 不为每个 MR×NR、LMUL、unroll、pipeline 候选复制内核；
- 不为 RVV 与矩阵扩展复制完整 GEMM/Conv/Attention；
- 不手工维持 load、cast、accumulator、reduce、store 之间的物理兼容；
- 不自己承担最终寄存器分配和机器调度。

但作者仍需写出高性能所需的 blocking、staging、persistent packing 和算法 variant。

---

## 5. DSL、Realizer、Tuner、Leaf 的严格分工

### 5.1 DSL 作者拥有

```text
worker-local ABI
ordered control and traversal
algorithm/cache blocking
block/state shape and lifetime
VLA logical domain
pointer/index/predicate/effect
algorithmic staging and workspace
persistent packing/layout
state algebra and numerical policy
explicit local primitive
algorithm variant
```

判断标准：若优化改变 loop/pass、blocking、staging、persistent data organization、state algorithm 或 traversal，它属于 DSL。

### 5.2 Weft Realizer 拥有

```text
dynamic vl / SEW / LMUL
value/register physical shape
unit/strided/indexed/segment memory realization
state scalar/vector placement
block handoff and ordinary SSA composition
register microtile / multiple accumulators
K-unroll / prefetch / primitive-local pipeline
short-lived local packing
RVV / IME / vendor extension realization
resource legality
```

判断标准：若优化不改变作者程序，只改变显式 block/primitive 的机器执行，它属于 Realizer。

### 5.3 Tuner 拥有

Tuner 只在 Realizer 已证明合法的实现族中实测选择，例如：

```text
BM / BN / BK meta values
LMUL
microtile
multiple accumulator count
unroll
prefetch distance
pipeline depth
fragment family
local packing strategy
```

Tuner不创造结构，也不能让非法实现合法。

### 5.4 Primitive-local leaf 与 system compiler

手写 intrinsic/asm microkernel 可以存在，但只能实现明确的局部 primitive。

Leaf 不得拥有：

- public kernel ABI；
- outer loop；
-完整 blocking；
- persistent layout；
-完整 algorithm state。

GCC/Clang 继续负责：

- 物理寄存器分配；
- 最终 `vsetvli` 与机器调度；
- spill；
- peephole；
- 常量折叠与机器码。

---

## 6. Storage、workspace 与 lifetime 是核心，不是附件

`Core-local blocked program` 要成立，作者必须能表达“谁持有数据，以及持有多久”。至少需要以下语义类别：

### 6.1 External/public storage

输入、输出和必须保持调用 ABI 的 buffer。

### 6.2 Persistent packed storage

在构建期或模型加载阶段形成，并跨 kernel 调用复用的 target-compatible representation。

作者/上游 runtime拥有是否允许 repack、packed layout identity 和部署生命周期。Target 不得偷偷创造 persistent ABI。

### 6.3 Worker-local algorithmic workspace

作者显式使用、可跨 loop 或 primitive 存活的 staging/scratch，例如：

- attention query/accumulator scratch；
- convolution packed patch；
- invocation-local per-worker state；
- 算法可见的 ping-pong buffer。

它必须有 shape、alignment、alias 和 lifetime contract，并进入 artifact ABI/metadata。

### 6.4 Compiler-private primitive temporary

只在一个 primitive 内部存在、不可观察的 register/local pack、temporary 或 buffer。它不进入 DSL public ABI。

本轮实现将表面语法冻结为 caller-provided entry pointer：external为默认class；persistent使用
`W.persistent(format)`，workspace使用`W.workspace, W.noalias`；后二者各由entry body中唯一的
`W.storage(pointer, shape)`闭合。Primitive-private temporary不进入DSL/Kernel IR/artifact ABI。

---

## 7. Block 与 structured primitive 必须自然组合

这是当前模型是否真实成立的最硬判据。

一个合法 primitive result 必须是普通 SSA value，而不是 fast-path terminal。至少以下程序应由同一模型支持：

```text
dot → add → store
dot → two consumers
dot → state update
matmul → pointwise activation → store
block transform → pointwise → store
state → indexed memory
workspace value跨loop复用
persistent packed operand → RVV/IME local primitive
```

若在 primitive 后插入一个合法 pointwise consumer就失去 lowering，说明 backend仍依赖精确 source closure，而不是编译 block program。

不能通过再增加 closure matcher修复。正确修复位置是：

```text
producer physical result
→ explicit handoff
→ ordinary consumer realization
```

---

## 8. RISC-V 碎片化与新扩展

Weft 的扩展性遵守三条规则。

### 8.1 已有语义的新硬件实现

新 tensor/segment/quant/permute instruction如果实现已有 primitive，只增加：

```text
target capability facts
legality and resource equations
physical candidates
handoff / pipeline constraints
intrinsic C or typed asm leaf
```

例如新 tensor extension 实现 `W.matmul` 后，GEMM、Conv、MoE、Attention 等所有使用该局部 primitive 的 kernel都可获得新 candidate；不增加完整算子后端。

### 8.2 新的局部可观察语义

如果硬件引入普通 primitive无法表达的：

- block scale；
- minimum/zero correction；
- codebook；
- saturation/rounding；
- sign-bit mapping；

可以增加一个局部 typed semantic primitive，但它仍不得拥有 outer traversal、persistent layout或kernel ABI。

### 8.3 新的完整算法结构

若峰值性能需要新的 outer traversal、persistent packing、staging 或 state algorithm，作者显式提供新的 DSL kernel variant。Target不得暗中改写旧程序。

---

## 9. Intrinsic C 是正式后端边界

Weft 面向 RISC-V 及其碎片化扩展，正式生成：

```text
standard RVV intrinsic C
vendor intrinsic
necessary typed inline asm leaf
ordinary scalar C
```

然后交给系统 C 编译器。

这个边界使 Weft 能：

- 精确表达向量类型、mask、memory form和extension intrinsic；
- 保留可读、可审查、可嵌入的产物；
- 复用现有 RISC-V 编译器完成最终后端；
- 避免自己维护 LLVM machine backend。

它不意味着 Weft lowering 很薄。Block layout、state、microkernel、packing、pipeline和extension realization仍由 Weft决定。

---

## 10. `materials/`：高性能知识资产的学习与复用

`materials/` 不是旧 backend，也不是 fallback。它是经过真实硬件验证的高性能知识 donor。

每当现有 kernel出现性能瓶颈，应主动审查相关 materials，而不是从零猜测；但必须按所有权提取，不能复制完整旧路径。

### 10.1 算法/DSL 知识

进入 DSL kernel 或显式 variant：

- outer traversal；
- cache blocking；
- algorithmic staging；
- persistent repack；
- 跨 loop reuse；
- scale/min 分离累加等作者可观察的算法组织。

### 10.2 Realizer 物理知识

进入共享 candidate、legality、resource 和 schedule：

- MR×NR register blocking；
- multiple accumulators；
- LMUL/SEW/resource relation；
- 8×32 subblock 等局部组织；
- decode/widen/reduce 连接；
- codebook gather；
- load schedule、prefetch、pipeline；
- fragment 与 handoff。

### 10.3 Leaf 拼写知识

进入 primitive-local intrinsic/asm：

- RVV intrinsic sequence；
- `vmadot` 等 asm leaf；
- operand constraint / clobber；
- exp polynomial 等局部数学实现。

### 10.4 必须建立复用台账

每项吸收的知识应能回答：

```text
来自materials的哪项事实
→ 归属DSL / Realizer / leaf哪一层
→ 进入哪个共享能力
→ 被哪些不同kernel复用
→ 是否删除了对应旧生产authority
```

禁止：

- 直接链接 materials 函数；
- 恢复旧 route/variant；
- 用 kernel 名或 q-format 进入整段实现；
- 把旧常数原样搬到新位置并称为编译器能力。

---

## 11. 当前实现的真实状态

最新审计表明，当前 Weft 已经有真实 worker-local AOT compiler 主干，但编程模型尚未闭合。审计中的“支持”要求完整经过：DSL → Kernel IR → RISC-V lowering → intrinsic C/asm → system compiler → 真实执行；只到 Python、IR 或特定 closure 都不算支持。

### 已经成立

- Python AST → canonical Kernel IR；
- scalar ordered control；
- 一条清楚的 `W.vla` 路径；
- canonical type/control/value 骨架；
- 部分 VLA/state/dot/matmul/quant/IME physical decisions；
- intrinsic C / local asm 主链；
- unsupported fail-closed；
- `source/` 和 `materials/` 不进入 production fallback。

### 仍未闭合

- `dot → add → store` 等自然 SSA composition；
- block result 多 use、state/memory consumer；
- workspace、storage lifetime 与 persistent packed ABI；
- public DSL 与 canonical schema/target artifact 一致性；
- verifier 中 validity/extent/lookup/decode/masked-dot 合同；
- family-specific exact closure matcher；
- emitter 中残留的 shape/layout/resource决定；
- 真实 multiple accumulators、prefetch和software-pipeline candidate；
- 正式 target profile；
- object/header/workspace metadata 与通用 compiler front door；
- 与同 target、同算法、同 shape、同 preprocessing ownership 的 baseline闭合比较。

---

## 12. 不可妥协的关闭判据

后续重构必须满足：

1. **公开能力 = canonical语义 = 正式target artifact。**
2. **Structured primitive result 是普通可组合 SSA value。**
3. **Storage ownership、shape、alignment与lifetime从source/artifact唯一可知。**
4. **作者算法结构不被target猜测或替换。**
5. **每项physical fact只有一个producer，emitter只消费。**
6. **性能能力来自共享block/state/memory/primitive lowering，不来自精确closure。**
7. **新扩展增加local realization，不增加整kernel route。**
8. **性能瓶颈必须检查并分类吸收materials知识。**
9. **Tuner只在资源和语义均合法的候选中选点。**
10. **SG2044与K1分别和各自`source/` baseline比较；跨机只分析decision变化。**

---

## 13. 本轮冻结的单一决定

- Workspace不使用源级allocation；caller按generated header分配，作为`W.workspace,
  W.noalias` entry pointer传入，并由`W.storage`声明shape。
- Persistent packed object同样由caller提供；`W.persistent(format)`保存稳定format identity，
  `W.storage`保存shape。Builder若存在，是另一份显式kernel或应用构建步骤，consumer target不得
  偷偷创建。
- Local block继续由`W.block_axis`、`W.full/zeros`、load、broadcast/reshape/transpose与pointwise
  组成普通SSA value system，不增加opaque block object或第二条执行路径。
- Physical decisions只存在于一次lowering；可以打印非规范化诊断文本，但不提供可重输入、可
  验证或持久化的debug schema。
- Compile-and-measure tuner是外部build loop，重复调用唯一编译主链，不形成compiler stage。
- Public primitive必须同时具有Python surface、canonical op/schema与target-local capability入口；
  某target缺少合法realization时明确unsupported，不保留只到frontend的假能力，也不静默替换
  算法。

---

## 14. 最终定义

> **Weft 是一门面向 RISC-V 的 Core-local blocked kernel DSL。作者编写由一个 worker/hart 执行的完整有序程序，显式组织 cache blocking、局部 block、state、staging、workspace、persistent layout 与 local structured primitives；这些 block和state可以跨循环存活、更新和复用，并像普通SSA值一样组合。Weft compiler在不改变作者算法的前提下，将 block operations物理化为RVV SIMD、寄存器微内核、局部pipeline和RISC-V扩展，并生成intrinsic C或primitive-local asm。**

最简洁的 Triton 对照是：

> **Triton围绕GPU program/CTA拥有的tile组织程序；Weft围绕CPU worker的持久控制程序所持有的block、state和storage lifetime组织程序。**

这个差异只有在 storage/lifetime、自然 composition 和 extension-local realization 真正进入 DSL/IR/artifact 后才成立。否则 Weft 仍只是一个具有若干 RISC-V fast path 的原型，而不是一门完成的 CPU 算子编译语言。
