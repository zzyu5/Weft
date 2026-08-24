# Weft 编程模型

## 1. Weft 承载哪一段程序

高层算子定义通常只规定结果，不规定高性能实现中实际存在的中间值、频率层次和有限位宽关系；手写 intrinsic 则同时固定了这些数值结构和具体机器表示。

Weft 承载两者之间的数值 realization：

```text
数学/算子语义
    ↓ 作者选择一种具体 realization
Weft：逻辑值、数值支路、层归属、有限位宽、供应与复用结构
    ↓ 目标编译器物理化
lane / register tuple / fragment / memory operation / intrinsic / local asm
```

作者写的是一份确定程序，而不是一个等待编译器发现算法的计算图。程序明确规定：

- 哪些逻辑值存在；
- 值在哪一层诞生、更新和交回外层；
- 代数支路如何组织；
- 中间类型与舍入位置；
- 哪些数据在某层供应一次并由后代复用；
- 哪些布局跨调用可见；
- 哪个局部 operation 要求哪类 engine。

作者不写：

- RVV SEW、LMUL、`vl` 和物理 lane；
- register tuple、fragment 和物理寄存器编号；
- load/gather/segment/unpack 指令；
- primitive 内部的物理 partial；
- invocation-local `pack` 的物理目标形状；
- 局部软件流水、spill、reload 和 rematerialization；
- intrinsic 名称与 inline asm spelling。

## 2. Q4_K 说明的缺口

设一条 Q4_K weight block 与一条 Q8_K activation block 的数学关系为：

```text
y = Σ_i dequant(q_i) · x_i
dequant(q_i) = d · sc_b · q_i - dmin · m_b
```

成熟实现通常计算：

```text
y = ds · [ d · Σ_b(sc_b · Σ_{i∈b} q_i x_i)
         - dmin · Σ_b(m_b · Σ_{i∈b} x_i) ]
```

这份 realization 包含三项源程序知识。

### 2.1 代数支路

min 支路不再读取 `q`，而是使用 Q8_K record 中预存的 `bsum`。作者明确写出 scale 支路和 min 支路，编译器不从原始 dequant 表达式发现分配律，也不证明两个式子等价。

### 2.2 频率分层

```text
element 层：q_i × x_i
sub 层：   partial × sc_b
block 层：d、dmin、ds 与两条支路合成
```

每项乘法出现在哪一层决定了执行次数、值的生命周期和复用关系。

### 2.3 有限位宽分层

```text
i8/u4 element products
→ i16 local partial
→ i32 sub/block accumulator
→ f32 block result
```

`into=i16` 和 widen 的位置是数值语义：移动它们会改变溢出、舍入和中间值集合。

在 Weft 的职责划分中，这三项不是目标编译器自动施加的 schedule，也不是 ISA 选择；它们是作者提供的数值 realization。传统编译器文献也可以把它们称为 algorithmic 或 computation-DAG rewrite——Weft 的关键不是否认这一名称，而是把这段 rewrite 变成源语言的一等结构。

## 3. 执行模型

一个 Weft kernel 是一次普通函数调用中的完整有序程序。调用创建一份局部 SSA 执行上下文；普通语句、`for`、`while` 和 `if` 按程序顺序执行。

语言没有隐式：

- launch grid；
- program id；
- worker/hart id；
- thread、warp 或 CTA ownership；
- “每个实例天然拥有一个 tile”的规则。

调用方可以在多个 hart 上调用同一 kernel，但 hart 数量、任务划分和线程运行时不属于 kernel 语义。并行调用的 buffer alias、同步和任务分配由调用方负责。

kernel 内部可以使用 [Level](../dsl/values-and-levels.md) 表达层级数值合成，也可以完全使用普通有序控制。Level 是可选构造，不是每个 kernel 必须套用的根对象。

## 4. 根抽象

Weft 的源语言不选择任何目标物理对象作为统一根类型。这是一项语言设计选择，不是“单控制器硬件没有特权对象”的硬件定理。

RVV register group、IME fragment、AMX/SME tile 和 VLIW register file 都是强物理对象；它们属于目标编译器的表示空间。源语言固定的是两端：

```text
内存端：Encoding 定义 logical field → storage bits
指令端：显式 operation 与 engine role 约束允许的 realization
```

两端之间的 lane、register、fragment 和 local memory representation 由目标编译器决定。它们共同构成[非 SIMT 物理抽象机器](../machine/physical-machine.md)；该机器是 lowering 的目标，不是新的源语言根类型。

源程序的组织骨架是：

```text
Encoding / View
        ↓ admit、materialize
shaped Value + explicit numerical operations
        ↓ Level births、carry、handoff
commit / returned effect
```

## 5. 唯一职责判据

> 对开放的 canonical SSA 程序，改变逻辑值集合或改变逻辑值的 Level 归属，属于作者程序；不改变这两者的机器实现，属于编译器。

“逻辑值”不是只有 dtype/shape 的匿名元素集合；它是 canonical Kernel IR 中由具体 operation contract 定义、带 typed operands、axes、effects 与硬 engine role 的 SSA value。改变 producing operation 的硬 role 或数值/effect contract，就是换了 canonical value definition，即使 result type 相同。closed primitive 内部为了实现同一 operand/result contract 而产生的 register partial、树归约节点、spill slot 或 fragment 临时值不是新的语言逻辑值。

### 5.1 作者决定

| 决定 | 为什么属于作者 |
|---|---|
| 是否拆出 min 支路 | 改变数值支路和逻辑值 |
| `bsum` 是否作为输入 | 改变读取的字段和数据依赖 |
| i16 partial 覆盖 2 项还是 4 项 | 改变 primitive、逻辑 partial 关系与溢出行为 |
| accumulator 是一份还是四份源级 partial | 改变 canonical state 数量与合并位置 |
| accumulator 跨 KC 还是跨整个 K | 改变状态诞生层、C 的读写频率与浮点累加顺序 |
| 是否 `pack`、`along`、位于 KC 还是 MC、绑定哪个 engine role | 改变 staged value 的逻辑域、诞生层、物化次数或允许的 engine |
| persistent interleave 是否存在 | 改变 artifact、ABI 和跨调用生命周期 |
| 普通标量循环还是显式 shaped axis | 改变 canonical 逻辑域 |

### 5.2 编译器决定的三种性质

判据只规定一项决定归作者还是编译器，并不意味着所有编译器决定都能从语义唯一算出。目标编译器内部必须区分下面三类；把它们都称为“推导”会把策略伪装成硬件定理。

#### 5.2.1 唯一合法推导

这类事实由语言语义、typed use-def、Encoding 和 target legality 唯一决定，不能使用启发式：

| 实体 | 必须唯一得到的事实 |
|---|---|
| shaped operation | free、reduction 与 broadcast axis；result 保留或消去哪些 logical axes |
| encoded field/use | logical coordinate 到 storage bit 的映射 |
| cast/widen/narrow chain | element width 与数值转换关系 |
| memory/effect edge | alias、order、validity、mask 与可移动性 |
| local instruction/fragment | dtype、shape、mask、tail、显式 role compatibility 与资源要求是否合法 |

同一事实只能有一个 producer。若 producer 和 consumer 的合法表示不一致，编译器插入保持 logical value 与 Level 归属不变的物理转换，不能用默认值掩盖冲突。

#### 5.2.2 结构性选择

这类决定不改变 canonical value 或 Level，但通常存在多个合法物理实现：

| 决定 | 约束来源 |
|---|---|
| logical axes 映射到 time、lane、register replica 或 fragment | value axes、producer/consumer、engine operand contract |
| 未绑定 role 且 op 允许多个 role 时选择 wide 或 matrix realization | op role 集合与 target profile；显式 `@wide/@matrix` 不参与选择 |
| unit/strided/indexed/segment memory form | encoding mapping、pointer relation 与 consumer mapping |
| invocation-local `pack` 的物理 schema：axis orientation、carrier、consumer handoff 与 local-storage/register/fragment 关系 | 作者给出的 `along`/role，加上 producer 连续性、多 consumer、widening、pipeline、资源与 handoff |
| shared、reload、rematerialize、spill 或 primitive-local pack | dominance、live interval、effect 与资源 |
| sequential 或 local pipelined schedule、decode materialize 或 decode-compute fusion | Level 内依赖、alias、engine 与 target 能力 |
| fragment family 与 value handoff 结构 | target operation contract 与跨 engine legality |

Weft 不为这些结构建立静态 cost model，也不生成多个合法结构后做性能竞赛。每个 target profile 提供确定的规则和优先级；编译器按规则选择第一个满足语义、target 和资源约束的结构。较低优先级结构只在较高优先级结构不合法时使用，不能因为“可能更快”而隐式改走另一条路径。

#### 5.2.3 参数性选择

结构固定后，仍有只改变物理实例、不改变 source tree 的数值参数：

| 参数 | 作用对象 |
|---|---|
| LMUL / physical lane factor | physical value |
| 已固定 schema 内的 physical microtile extent / register-replica count | local operation/result representation |
| unroll | Level/local cluster |
| pipeline depth / buffer count / prefetch distance | Level/local cluster 与 memory edge |

目标可以为这些参数给出有限合法域；构建期 tuner 对具体 target 和 shape 实测这些绑定并选择 winner。每个绑定重新经过合法性与资源检查，非法绑定在 emission 前拒绝。tuner 不生成新的结构，也不改变 target 的结构优先级。

源语言中的 `auto` 是另一侧的有限参数域：它由作者或 std 声明，可能实例化不同 cohort、Level extent 或其它 source candidate。目标编译器不得发明这个域。source `auto` 与 target physical 参数可以由同一构建过程测量，但二者的语义归属不能混淆。

### 5.3 编译器决定的实例

| 决定 | 为什么属于编译器 |
|---|---|
| shaped axis 映射到 lane/register/fragment | canonical value 与层归属不变 |
| SEW、LMUL、`vl` 与尾部执行 | 物理表示 |
| `reduce` 内部的物理 partial 与树归约 | primitive 的 observable result 不变 |
| unit/strided/indexed/segment load | memory edge realization |
| nibble 使用 shift/mask、gather 或专用 unpack | encoding mapping 不变 |
| loop-invariant 地址提升 | 逻辑诞生层本来就不依赖内层迭代 |
| local pipeline、spill、reload、rematerialize | canonical values、Level 和 handoff 不变 |

### 5.4 违反判据会怎样

如果编译器把 `group=16` 的一个逻辑 cohort 改成四个源级 `group=4`：

- `admit` 发生四次而不是一次；
- state 与 handoff 被复制四份；
- staged value 的复用域变化；
- effects 与浮点结合顺序可能变化。

这不是更换表示，而是生成了另一棵程序树。它必须作为另一个 std 实现或 `auto` 实例由作者声明，不能作为后端隐藏路径。

## 6. 与 Triton 的机制差异

Triton kernel 由 launch grid 中的独立 program instance 执行：

```python
pid = tl.program_id(0)
offsets = pid * BLOCK + tl.arange(0, BLOCK)
x = tl.load(ptr + offsets, mask=...)
```

`program_id` 和最多三维 grid 是公开语义。block tensor 通过 `arange`、广播、masked load/store、`dot` 和 `reduce` 构造。进入 TritonGPU IR 后，tensor type 的 encoding 显式描述 register/lane/warp/block/CTA ownership；blocked、slice、dot operand、shared 和 MMA layouts 都是该系统的一部分。具体机制可在 `ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td:738` 与 `ref/triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td:32` 中看到。

Triton 同样没有把全部高性能结构交给 compiler。FlashAttention 的 online-softmax recurrence、causal stage、QK/PV 顺序和 accumulator 更新由作者写在 `ref/triton/python/tutorials/06-fused-attention.py:48`；GEMM 的 tile、program mapping 和 K loop 也属于 source/meta。`@autotune` 只从作者提供的 `BLOCK_M/N/K`、`num_warps`、`num_stages` 等有限配置中实测选择。

目标内部的物理组织则由真实 IR pass 完成。NVIDIA pipeline 在 `ref/triton/third_party/nvidia/backend/compiler.py:297-360` 依次运行 coalescing、layout conversion elimination、thread-locality、matmul acceleration、loop scheduling、pipeline、prefetch 与 MMA lowering。`AccelerateMatmul` 在 `ref/triton/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp:42` 使用 target legality、固定优先级和 shape heuristic 选择 MMA version 与 warp tile；它不是硬件唯一映射。pipeline expander 在 `ref/triton/include/triton/Dialect/TritonGPU/Transforms/PipelineExpander.h:79` 机械生成 prologue/kernel/epilogue，但 operation-to-stage schedule 由前序分析提供。

Triton 可以手写 packed decode、有限位宽乘法和 correction 支路。差异不在表达能力，而在一等结构：

- Triton 的根执行分解是 program grid，物理 layout 围绕 SIMT ownership；
- Weft 的源程序没有 program grid 或 owner，Level 保存值的逻辑诞生层和 handoff；
- Triton packed record 通常由 pointer arithmetic 和 block operations 展开；Weft Encoding 直接保存 logical field 到 storage bits 的映射；
- Triton `tl.dot` 是语言 builtin；Weft 的 GEMM/GEMV 是由基本 op 和 Level 写成的普通 std 函数；
- Triton 用户可以用 `tl.range(num_stages=...)` 或 config 暴露 pipeline 参数；Weft 源程序表达 logical lifetime，target 的 local pipeline structure 不进入 DSL，physical depth 属 target 参数。

因此 Triton 省掉的是目标内部的 layout、MMA、搬运和大量合法循环变换，不是作者的算法 realization。Weft 不是把 Triton tile 换成 CPU loop；两者对 source machine 保存什么作了不同选择，而 typed layout、显式 conversion 和真实 pass 改写仍是 Weft 应采用的编译器机制。

## 7. 与 TileLang 的机制差异

TileLang 的编译输入可以来自 `@T.prim_func` 的 TIR/TIRX `PrimFunc`，也可以来自 `@tilelang.jit` 的 eager builder；`T.Kernel` 构造 launch frame，backend 再将其中的 thread-binding loops 物化为目标 launch。用户可以显式写：

- `T.Kernel(grid..., threads=...)` 的 grid/block/thread binding；
- global/shared/local/fragment/TMEM 等 buffer scope；
- `T.copy`、`T.async_copy` 和 `T.gemm` tile operation；
- `T.Parallel`、`T.Pipelined(num_stages=...)`、manual stage/order；
- buffer/parallel-loop layout annotations。

这些 source contract/annotation 使 GPU storage scope、launch/thread binding、copy intent 和 tensorization 成为源程序的一等结构；见 `ref/tilelang/tilelang/language/allocate.py:1`、`kernel.py:149`、`loop.py:13` 与 `copy_op.py:54`。其 compiler 仍运行 layout inference、pipeline planning 和 lower-tile-op，具体机制位于 `ref/tilelang/src/transform/layout_inference/layout_inference.cc:92`、`pipeline_planning.cc:36` 和 `lower_tile_op.cc:43`。TileLang 同样可以用这些构造和普通表达式写出 Q4 decode、online softmax、状态更新和标量控制。

TileLang 的 CPU pipeline 仍继承这套 IR，但在所参考版本中将 thread binding 串行化，CPU GEMM lowering 只选择 `cpu.scalar`；见 `ref/tilelang/tilelang/cpu/pipeline.py:15` 与 `ref/tilelang/src/cpu/op/gemm.cc:21`。因此直接复用 TileLang 的 CPU 表面会把 Weft 的 Level/Encoding 数值模型放进一个以 GPU storage/thread/tile 为中心的 source contract，再在 CPU 后端抹去其中一部分结构。

Weft 选择暴露另一组事实：

| TileLang 源程序显式表达 | Weft 源程序显式表达 |
|---|---|
| grid/thread binding | 无隐式 grid；普通调用中的逻辑 domain |
| shared/local/fragment allocation | state/staged value 的逻辑生命周期与 Level 归属 |
| copy source/destination storage scope | `admit/materialize/handoff/commit` 的供应关系 |
| tile GEMM primitive | 由基本 contract/outer-contract 与 Level 写成的 std 函数 |
| 可显式给出 pipeline `num_stages`、stage/order | 只给 logical use-def；物理 local pipeline 由编译器实现 |
| fragment/thread layout | shaped axis，无物理 owner |

这不是“TileLang 写不出来、Weft 写得出来”的区别；它是源语言保存的结构不同。TileLang 保存硬件感知的 buffer/copy/thread structure，Weft 保存 encoding-aware 数值树与 logical lifetime structure。

## 8. 正确性边界

Encoding 与浮点数值使用不同判据：

- 离散 storage bytes、packed field、bit layout 和整数解码关系可以要求逐字节或逐字段一致；
- 浮点 kernel 只要求符合 operation 的数值语义和声明的误差容差，不要求 bit-exact；
- 合法 contraction、vector reduction 和 FMA 可以改变最后几位；
- 编译器不能为了复刻某个 reference 的舍入位置而改变 canonical tree；
- 作者负责有限位宽选择和允许的 wrap/saturate 行为，编译器不证明不溢出。

数值容差是调用或验证协议的一部分，不是允许编译器任意重结合的全局权限位。每个 operation 的结合、顺序和精度语义必须由 operation 本身定义。
