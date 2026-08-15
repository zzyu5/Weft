# Weft Core-local Blocked Programming Model

> **状态：当前唯一源语言与编程模型说明。** Canonical Kernel IR、RISC-V lowering、示例和其他文档只能投影本文定义的语义，不得另行发明第二套 block、state、storage 或 execution model。

## 1. 最终定义

Weft 是一门面向单个 RISC-V CPU worker/hart 的 AOT kernel DSL。

> 一个 Weft kernel 是由当前 worker 从入口到返回持续执行的一份有序程序。作者显式写出 traversal、blocking、staging、局部数据布局、state lifetime 与 local structured operations；编译器只在这些显式结构内部选择 SIMD、寄存器微内核和 RISC-V 扩展 realization。

正式主链只有：

```text
Weft Python DSL
→ canonical Core-local Kernel IR
→ one target-specific RISC-V lowering
→ intrinsic C / primitive-local inline asm + C header
→ system C compiler
→ object / library
```

Weft 不是：

- graph compiler；
- operator catalog 或按 kernel/q-format 选择手写函数的 dispatch library；
- GPU program grid 在 CPU 上的语法翻译；
- 自动从普通 loop/SSA 猜 GEMM、scan、summary 或量化格式的模式机；
- 多核线程池、work stealing、NUMA 或模型 runtime。

## 2. 根执行合同

一个 kernel entry 是普通 C ABI callable。外部 runtime 选择当前 worker 的 work slice，并将它作为普通 pointer/scalar 参数传入：

```text
runtime
  → 选择 row range / tile / expert range / descriptor
  → 在当前 hart 调用一个 Weft entry
  → entry 顺序执行 scalar control、VLA、block compute 和 memory effect
  → 返回 caller
```

Weft 不提供：

- grid rank；
- `program_id`；
- worker/hart ID；
- 隐式 launch coordinate；
- 多个 hart 协作同一 logical block 的 core-level barrier。

需要 work identity 时，caller 将它作为普通 ABI 参数传入。不同 kernel 可以使用不同 descriptor；语言不强制统一 `work_begin/work_end`。

一次 invocation 可以顺序处理多个 output/cache block。Block、accumulator、state 和 workspace 的 lifetime 由作者写下的 control/use/storage relation 决定，不随某个 VLA strip或primitive调用自动结束。

## 3. 与 Triton 的区别

区别不是 Weft “没有 tile”。两种语言都可以让作者写 block algorithm、BM/BN/BK 和 local product。

```text
Triton
  grid 中一个 program/CTA instance逻辑拥有一个或几个result tile
  compiler把tile分布到协作线程、warp、register和shared memory

Weft
  一个CPU worker持续执行普通ordered program
  同一worker可顺序处理多个block
  block、accumulator、state与workspace可跨作者loop存活和复用
  compiler把local block operation实现为SIMD register、microkernel或extension fragment
```

因此，Weft 的独立合同是：

- block 的 owner 是 worker control program；
- outer traversal、cache blocking、staging 和 persistent packing 是 source program；
- VLA、dot/matmul 和 extension op 只是局部 realization 授权点；
- block/state/storage lifetime 必须能从 DSL 与 canonical IR 本身读出。

如果一个实现仍需查看 emitter matcher 或 Physical Plan 才能解释“这个 block 属于谁、活多久”，它不符合本模型。

## 4. 一种程序，不是几种 kernel 的拼接

Weft kernel 统一由以下部分构成：

```text
ordered scalar control
+ scalar / block / VLA-region SSA values
+ explicit pointer, predicate and memory effects
+ explicit source-visible storage
+ explicit local semantic primitives
```

Scalar、block、VLA、state、quant 和 IME 不是不同的 kernel 类别。它们服从同一 lexical scope、SSA use-def、logical domain、validity、effect 与 control-carry 规则。

一个 ordinary loop 可以包含 VLA；VLA value 可以带额外 block axes；dot operands 可以来自 indexed memory；dot result 可以进入 pointwise、另一个 consumer、loop carry 或 store。出现新组合不会产生新的 whole-kernel 类别。

## 5. Ordered control 与 state

Python `for / while / if` 是当前 worker 的普通有序控制。

```python
state = initial
for step in W.range(begin, end):
    state = update(state, step)
```

这里的赋值按照普通 Python source semantics 形成 canonical loop-carried SSA。Frontend 将被更新的 local 投影成 region argument、yield 和 loop result；这是 source 到 SSA 的机械转换，不是 target 推测算法。

作者拥有：

- loop 是否存在；
- iteration order 与 step；
- 哪些 value/state 跨 iteration carry；
- blocking 所在的 loop level；
- branch、effect 和 state update 顺序。

Compiler不得把 scalar loop 改成新的 VLA domain，也不得把普通 carry 猜成 reduce、scan 或 summary。

Block state 直接作为普通 block SSA value跨 `for/while/if` carry，不包装进 opaque state object。`W.tuple` 只表示闭合的 scalar state tuple；block state 仍由普通 control carry表达，从而保留其 domain identity和consumer关系。

Reduce、scan、argmax、online summary 与 sequential carry 保持不同的可观察语义：

- `reduce` 只观察最终聚合值；
- `scan` 为每个 logical position产生有序prefix；
- `argmax` 显式定义coordinate与tie policy；
- `online_softmax_summary` 显式定义稳定 `(maximum, scaled_sum)` 合并代数；
- ordinary carry 按作者 loop order执行。

它们不能互相猜测或替换。

## 6. Logical block：显式 domain 的普通 SSA value

### 6.1 `W.block` 同时定义 axis identity 与坐标

每一次 `W.block(extent, offset=0)` 创建一个新的 source-owned logical block axis，并产生该轴上的 index block：

```python
m = W.block(BM)
n = W.block(BN)
k = W.block(BK)
```

这里 `m`、`n`、`k` 不只是三个长度；它们是三个不同的 logical axis identity。即使 `BM == BN == BK`，三个轴也不能互换。

同一个 axis value 可以被多个 operand共享：

```python
a_index = m[:, None] * lda + k[None, :]
b_index = k[:, None] * ldb + n[None, :]
```

因此，`W.matmul` 的 K relation来自两个 operand显式共享同一个 `k`，而不是 backend看到两个相等的静态维度后猜测它们是 contraction axis。

旧 `W.block_axis` 不再属于语言；它只有 shape 而没有稳定 axis identity，不能作为 canonical block 模型继续保留。

### 6.2 Block constructor 直接消费 axis values

Block constructor 不接受裸整数 shape。作者用实际 axis value定义 block domain：

```python
m = W.block(BM)
n = W.block(BN)
acc = W.zeros((m, n), dtype=W.f32)
bias = W.full((n,), W.f32(1.0))
```

这样，accumulator的 `[M,N]` identity 在创建时即已确定。Frontend、canonical verifier 与target都不能把另一个同长 axis替换进来。

### 6.3 Explicit singleton-axis broadcast

`None` slicing只插入一个显式 singleton broadcast dimension：

```python
m[:, None]   # [M, 1]
n[None, :]   # [1, N]
```

Singleton broadcast axis在 canonical type中使用专用 identity `0`，不等于任何真正的 source axis。Pointwise join只有在以下条件之一成立时合法：

- 两个位置携带同一个 source axis identity；
- 一侧是显式 singleton broadcast；
- 一侧是 scalar。

同 shape、不同 identity 的两个 block 不能 pointwise 合并。这使同形不同轴的错误在 source/canonical 边界失败，而不是到 emitter才暴露。

### 6.4 Canonical block type

Canonical type同时保存静态/动态 shape 与 logical axis identity：

```text
!weft_kernel.block<[D0, D1, ...], [a0, a1, ...], T>
!weft_kernel.region<[-1, D0, ...], [-1, a0, ...], T>
```

- positive `ak` 是由某个唯一 `weft_kernel.block_index` 定义的 block axis；
- `0` 是 explicit singleton broadcast；
- region首轴的 `-1` 是当前 lexical VLA axis；
- dynamic dimension仍使用 shape `-1`，但 axis identity不因动态长度而丢失。

Block value不绑定 RVV lane、VLEN、LMUL、register number、microtile或IME fragment。

### 6.5 Ordinary SSA composition

Block load、pointwise、state carry和structured result都进入同一 SSA value system：

```text
dot → add → store
dot → consumer A and consumer B
matmul → pointwise → store
block → loop carry → block
block + scalar state in one ordered program
```

同一个 value允许多个 consumer。One-use 可以影响是否复用 physical storage，但不能决定 source是否合法，也不能成为 primitive fast path入口。

## 7. VLA 是显式 SIMD logical domain，不是根模型

```python
with W.vla(begin, end) as i:
    value = W.load(source + i)
    W.store(output + i, value)
```

`i` 表示 `[begin,end)` 中每个逻辑位置，不是 lane ID。Target可以把它分成任意数量的动态 `vl` strip，但 source不能观察：

- exact `vl`；
- VLEN；
- strip ordinal；
- lane ID；
- LMUL。

一个 lexical scope 当前只允许一个 active VLA axis。普通 scalar loop可以出现在 VLA body内；第二个 VLA 不可以。

VLA body不能任意更新 outer state。跨完整 VLA domain 的状态必须由 reduce、scan 或显式 typed summary primitive产生；否则 strip choice会变成 source-observable。

VLA axis可以与 block axes组合：

```text
region<[VLA, K], [vla, k], T>
```

Dot/matmul不能缩并 VLA axis；跨 VLA 的聚合必须由相应 state primitive显式授权。

## 8. Pointer、predicate、memory 与 validity

作者显式写 pointer/index relation：

```python
pointer = base + row * stride_row + column * stride_column
```

Typed pointer expression传播完整 block/VLA domain。Load/store的 predicate、fill与value必须是 scalar，或能显式 broadcast到同一个 pointer domain；同 shape不同 axis不会被接受。

```python
x = W.load(ptr, where=valid)
```

当 `where` 非恒真且没有 `other` 时，结果携带 first-class logical validity。

```python
x = W.load(ptr, where=valid, other=0.0)
```

提供 `other` 后得到普通 filled value。未填充 masked value不能直接进入不理解 validity 的普通 consumer或store。

Target可以选择 unit/strided/indexed/segment memory、mask form、地址strength reduction和physical prefetch，但不能改变 source-visible pointer relation、effect order或storage ownership。

## 9. Storage、staging 与 lifetime

Source-visible memory object均由 caller分配并作为 entry pointer传入。语言只有四类 ownership：

| 类别 | Source表达 | Lifetime与owner |
| --- | --- | --- |
| external | 默认 pointer | caller持有的普通input/output/state |
| persistent | `W.persistent(format)` + `W.storage` | caller/model loader持有，按明确format跨调用复用 |
| workspace | `W.workspace, W.noalias` + `W.storage` | 当前worker本次invocation独占，可跨loop/primitive复用 |
| primitive-private temporary | 不进入DSL | target只在一个local primitive realization内部创建 |

Workspace示例：

```python
scratch: W.ptr[W.f32, W.workspace, W.noalias]
W.storage(scratch, shape=(rows, columns))
```

作者决定何时写入scratch、跨哪些loop复用、何时失效。Target不能因为某个fast path需要scratch就偷偷增加ABI参数。

Persistent示例：

```python
packed_weight: W.ptr[
    W.u8,
    W.readonly,
    W.noalias,
    W.persistent("q4_k_n16_k32_304b"),
]
W.storage(packed_weight, shape=(column_blocks, k_blocks, 304))
```

Format identity是caller和consumer共享的长期ABI，不是LMUL、register layout或IME fragment。Consumer lowering不得在内部创建persistent repack；需要builder时，builder是另一份显式kernel或应用构建步骤。

纯 SSA block/state不需要为了跨 loop存活而自动写进workspace。Register、rematerialize、reload与target-local spill是physical realization；只有作者明确选择algorithmic staging时才使用workspace。

## 10. Structured primitive 的授权边界

### 10.1 Dot

`W.dot`固定收缩双方共享的最后一个 block axis。当前公开数值域是F32 multiplicand/F32 accumulator，合法关系是：

```text
[R,K] × [K]       → [R]
[VLA,K] × [K]     → [VLA]
[R,K] × [VLA,K]   → [VLA,R]
```

两侧 K 不仅extent相同，还必须来自同一个 `W.block` identity。

### 10.2 Matmul

`W.matmul`固定表达：

```text
[M,K] × [K,N] + [M,N] → [M,N]
```

当前公开数值域是F16 multiplicand/F32 accumulator。M、N、K relationship由operand与init的axis identity完整定义。

### 10.3 Result 与 init

`init` 是必填 source semantics，可以是scalar或与result完全同domain的 accumulator。Dot/matmul result是普通 block/region SSA value，可以：

- 多次使用；
- 进入pointwise；
- 作为loop-carried accumulator；
- 进入state或memory；
- 成为另一个合法local primitive的operand。

Target只在当前 primitive 内决定：

- SIMD方向与LMUL；
- register microtile/multiple accumulators；
- K-unroll；
- primitive-local load schedule与packing；
- RVV/IME/vendor extension realization。

Target不得创建outer loop、改变blocking、增加source-visible staging、创造persistent packing，或选择另一种GEMM算法。

## 11. GEMM 中心判据

一份符合本模型的F16 blocked GEMM如下：

```python
for m0 in W.range(m_begin, m_end, BM):
    for n0 in W.range(0, n, BN):
        m = W.block(BM)
        n_block = W.block(BN)
        acc = W.zeros((m, n_block), dtype=W.f32)
        m_index = m0 + m[:, None]
        n_index = n0 + n_block[None, :]

        for k0 in W.range(0, k, BK):
            k_block = W.block(BK)

            k_lhs = k0 + k_block[None, :]
            k_rhs = k0 + k_block[:, None]

            a_block = W.load(
                a + m_index * lda + k_lhs,
                where=(m_index < m_end) & (k_lhs < k),
            )
            b_block = W.load(
                b + n_index * ldb + k_rhs,
                where=(k_rhs < k) & (n_index < n),
            )

            acc = W.matmul(
                a_block,
                b_block,
                init=acc,
                acc_dtype=W.f32,
                order="relaxed",
                math="native",
            )

        acc = acc + W.f32(0.0)
        W.store(c + m_index * ldc + n_index, acc,
                where=(m_index < m_end) & (n_index < n))
```

仅从source即可回答：

- 当前worker遍历`m0/n0/k0`；
- BM/BN/BK位于哪些loop；
- `acc`由当前worker持有并跨K-loop carry；
- A/B block如何构造和mask；
- M/N/K轴分别是谁，两个operand共享哪个K；
- source没有staging或persistent packing；
- `W.matmul`只覆盖当前local `[M,K]×[K,N]` product；
- matmul result在loop后仍是普通block value，可继续pointwise后store。

Realizer若只在M/N/K静态shape和direct-store closure成立时才能解释这个程序，错误在Realizer，不在source模型。

## 12. Quant、lookup 与 RISC-V extension

一个extension primitive只有在以下条件同时成立时才属于DSL：

- 它表达独立、完整、局部且可观察的数值关系；
- operand/result使用core block/region/validity/storage system；
- 它不拥有entry ABI、outer traversal、blocking、staging或persistent layout；
- 普通SSA图不会被target猜成该primitive。

Packed bit ordering、scale/minimum、zero-point、codebook、rounding/saturation等确实可观察的关系可以拥有typed local primitive。LMUL、register grouping、local narrow/widen sequence、fragment和asm spelling不进入DSL。

已有semantic primitive获得新的RVV/IME/vendor指令时，只增加target realization；不能增加完整kernel route。

## 13. Canonical IR 与 Realizer 边界

Canonical Kernel IR持久保存：

- ordered control与carry；
- scalar/block/VLA-region values；
- block axis identity、extent与singleton broadcast；
- pointer、predicate、validity与effects；
- external/persistent/workspace contracts；
- reduce/scan/summary/dot/matmul与local extension semantics。

它禁止保存：

- exact `vl`；
- VLEN/LMUL；
- register number或microtile；
- instruction/leaf identity；
- candidate winner或measurement；
- launch/thread policy。

一次 target lowering可以产生短生命周期physical facts与decisions，但它们只能实现上述source semantics，不能补写缺失算法。

Emitter负责intrinsic C、ABI与typed local asm拼写。System compiler负责最终register allocation、machine scheduling、spill、peephole和machine code。

## 14. 当前唯一 public model

核心surface按职责分为：

```text
entry/control
  @weft.kernel
  W.range
  Python if / while
  W.vla

block/domain
  W.block
  singleton slicing [:, None] / [None, :]
  W.full / W.zeros

memory/storage
  W.load / W.store
  W.storage
  W.external / W.workspace / W.persistent

state
  W.reduce / W.scan
  W.argmax / W.online_softmax_summary
  ordinary scalar or block control carry

structured local compute
  W.dot / W.matmul
  W.lookup / W.decode
  typed quant and extension primitives
```

核心surface没有：

- `W.block_axis`；
- arbitrary reshape/transpose/broadcast；
- source-visible lane/VL/LMUL；
- generic contraction；
- generic summary fold；
- atomic/prefetch/fence等尚未闭合的假能力；
- compatibility alias或legacy route。

## 15. 关闭判据

一个语言/IR实现只有在以下问题能由source和canonical IR直接回答时才符合本模型：

1. 当前kernel由谁执行；
2. 当前worker负责什么work slice；
3. 每个block axis在哪里创建，与哪些operand共享；
4. block/state跨哪些loop与branch存活；
5. staging/workspace/persistent object由谁持有、何时有效；
6. 哪个显式construct授权VLA、state合并、local product或extension realization；
7. 哪些内容仍属于target physical choice；
8. 为什么插入普通pointwise或增加第二个consumer不会改变primitive的语言类别。

答案不得依赖kernel名、完整source closure、Physical Plan字段或emitter内部假设。

最终定义是：

> **Weft 作者写一份由单个CPU worker/hart持续执行的Core-local blocked program。Ordered control、block domain与axis identity、state、staging、workspace、persistent layout和local semantic primitives均由作者显式拥有；compiler只把这些结构实现为RVV SIMD、register microkernel和primitive-local RISC-V extension。**
