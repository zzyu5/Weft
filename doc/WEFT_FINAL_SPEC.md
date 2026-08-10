# Weft 语言与编译器规范

> **状态：目标态规范（Normative Target-State Specification）**
> 本文件只定义 Weft 最终成品的语言、语义、编译边界与外部接口；不包含实施阶段、迁移步骤、工程排期或测试战役。
> 与旧设计文档冲突时，以本文件为准。

---

## 0. 最终定义

Weft 是一门面向**单个 RISC-V worker / hart** 的高性能 kernel DSL 与 AOT 编译器。

作者使用普通控制流、一等 VLA iteration region、显式指针与逻辑 predicate、局部 logical block value、结构化 state algebra，以及 contraction / scan / lookup / decode 等 primitive 编写完整的 worker-local kernel。编译器把这些结构映射到动态 `vl`、LMUL、寄存器组织、memory instruction、register microtile、矩阵 fragment 和 RISC-V 扩展指令。

多核线程创建、工作切分、线程池、OpenMP、affinity 与 NUMA 均由外部 runtime 负责，不属于 Weft core。

Weft 的扩展原则是：

> **已有语义增加局部 realization provider；新增可观察语义增加一个局部 primitive；永不为某个完整算子、模型格式或 kernel 名增加整段后端模板。**

Weft 不是 Tensor graph compiler，也不是 Triton CPU backend 的重写。它与 Triton 的关键区别是：

- Weft 没有 program grid、`program_id` 或隐式 launch identity；
- Weft 的入口是一个普通 worker-local callable kernel；
- VLA region 的 strip 边界与 `vl` 在 source 中不可观察；
- 跨 strip 的 reduce、scan、summary state 是一等语义；
- logical block 只作为局部数据域和 structured primitive 的 operand，不是整个程序的根执行单元；
- RISC-V 的 V、矩阵、量化、重排及 vendor extension 通过 primitive-local provider 组合接入。

---

## 1. 规范用词

本文件使用以下约束词：

- **必须 / MUST**：符合 Weft 的实现不可违反；
- **禁止 / MUST NOT**：符合 Weft 的实现不可提供该行为；
- **应该 / SHOULD**：除非有明确且可说明的原因，否则应遵守；
- **可以 / MAY**：可选能力，不影响核心语义。

---

## 2. 设计目标与非目标

### 2.1 设计目标

Weft 必须同时满足：

1. **worker-local kernel 可写性**：作者能直接表达循环、blocking、staging、pointer/index、mask、state 与 structured compute；
2. **VLA 原生性**：source 不观察固定 VLEN、exact `vl` 或 hardware lane count；
3. **高性能物理自由度**：编译器与 provider 能决定 LMUL、register tile、unroll、packing、fragment 与 instruction family；
4. **RISC-V 扩展局部接入**：新增扩展不要求新增完整 GEMM、Softmax、RMSNorm、量化算子模板；
5. **AOT 可嵌入性**：产物是普通 object / static library / header，运行期不依赖 Python、LLVM 或 JIT；
6. **外部 runtime 兼容性**：llama.cpp、ggml、框架线程池或应用自己的调度器可直接调用生成的 worker-local entry；
7. **算法与 realization 分离**：source 固定 worker-local 算法，selected execution 只保存物理选择。

### 2.2 非目标

Weft 不负责：

- framework graph 导入；
- 多算子 fusion、partition 或 end-to-end model compilation；
- 自动从任意 SSA 图发现 GEMM、attention、量化格式或完整算子；
- 自动发明作者未写出的 cache loop、staging skeleton 或算法 variant；
- 创建或管理 CPU 线程；
- OpenMP、pthread pool、work stealing、NUMA placement；
- 多个 hart 协作同一个 logical tile 的同步编程模型；
- 运行期 JIT；
- 通用正确性证明、实现审批或 certification framework；
- 用 kernel 名、格式名、route string 驱动 codegen。

---

## 3. 根程序模型

### 3.1 Worker-local kernel

一个 Weft kernel 是一个普通可调用函数。它接收显式 pointer / scalar / descriptor 参数，并在当前 worker 上完成一段工作。

```text
external runtime
    → 选择 worker-local work slice
    → 调用 Weft kernel(args..., slice descriptor...)
    → Weft kernel 在当前 hart 上执行 scalar + VLA + extension code
```

Weft 不规定 work slice 的统一形状。应用可以传入：

- `row_begin / row_end`；
- 一个 tile index；
- expert range；
- quant block range；
- ragged descriptor；
- 任意普通 scalar / pointer 描述符。

因此，Weft ABI **不得**强制所有 kernel 使用 `work_begin/work_end`，但允许库提供该常见约定的 helper。

### 3.2 无 program grid

Canonical Weft language 中不存在：

- `program_id`；
- grid rank；
- launch grid；
- hart ID；
- worker ID；
- 隐式 task identity。

需要工作坐标时，调用者必须将其作为普通参数传入。

### 3.3 普通控制流

作者显式拥有：

- scalar `for` / `while` / `if`；
- cache / algorithmic blocking 的位置；
- staging 与重算骨架；
- outer K loop；
- block 之间的顺序与状态；
- memory effect 与 atomic/fence。

编译器可以做保持语义的 canonicalization、unroll、hoist、rematerialization 和局部 scheduling，但不得创建一套 source 中不存在的算法骨架。

---

## 4. 两类一等数据域

Weft 不把所有计算统一成一个 Triton 风格 tile。语言有两类互补的一等数据域。

### 4.1 VLA iteration region

VLA region 表示一个运行时长度的一维逻辑迭代域：

```python
with W.vla(begin, end) as i:
    ...
```

语义是：

```text
i ∈ [begin, end)
```

该逻辑域由编译器和目标实现分解为任意数量的连续动态 `vl` strip。source 不得观察：

- strip 数量；
- 当前 `vl`；
- strip ordinal；
- physical lane ID；
- fixed VLEN；
- LMUL。

VLA region 是普通 pointwise、memory、reduction、scan 与跨 strip summary 的主要执行域。

### 4.2 Logical block value

Logical block 是具有显式 shape 的局部 SSA region value，例如：

```text
block<BM × BK, f16>
block<BK × BN, f16>
block<BM × BN, f32>
```

Logical block：

- 是局部数据域；
- 可以由 block axis、broadcast、load、reshape、transpose 和 pointwise 产生；
- 可以作为 `W.contract`、block reduction、permute、decode 等 primitive 的 operand；
- 不等于 cache block；
- 不等于 register tile；
- 不等于 IME fragment；
- 不对应独立 worker、program instance 或 launch task；
- 可以是编译器中的 lazy region value，不要求先物化为实际数组或寄存器集合。

### 4.3 组合形态

一个 region value 可以具有：

- 零个或一个 VLA axis；
- 零个或多个 specialization-time block axes。

Canonical 类型可概念性表示为：

```text
region<[* , D0, D1, ...], T>
```

其中 `*` 表示当前 VLA axis，`Dk` 是 compile-time / meta block extent。

同一 lexical scope 中最多只能有一个活跃 VLA axis。嵌套第二个 VLA region 必须被 verifier 拒绝，除非未来规范显式引入新的多维 VLA 语义。

`W.contract` 不得缩并 VLA axis；跨 VLA axis 的聚合必须使用 reduce、scan 或 summary fold。VLA axis 可以作为 contract 的 batch/free axis。

---

## 5. Tile 与分块层次

Weft 必须区分以下四层，不得混用同一个 `tile` 概念：

### 5.1 Algorithmic / cache block

由作者决定是否存在、位于哪个循环层、如何影响 memory reuse。典型参数为 `BM/BN/BK`。

这些参数可以是 build-time meta-parameter，但其**存在和使用位置**属于 source algorithm。

### 5.2 Logical operand block

由 source 构造并由 structured primitive 消费的 shaped semantic value。

它只描述局部坐标域与数据关系，不声明寄存器或 ISA fragment。

### 5.3 Register microtile

例如 `mr × nr` accumulator、register repeat、LMUL 组合及 K-unroll。

它属于 primitive realization provider 的物理调优空间，由 compiler 建立 legality，构建期 tuner 选择。

### 5.4 ISA fragment

例如某个矩阵扩展规定的 `4×4×8`、accumulator register class 或 encoded operand tile。

它是具体扩展的硬件叶子，只存在于 selected execution / provider lowering，不进入通用 source block 类型。

---

## 6. Python DSL 的地位

### 6.1 参考前端，不是运行时发射器

Weft Python DSL 是规范化参考 source frontend：

```text
Python Weft source
    → canonical Weft Kernel IR
    → selected RISC-V execution
    → object / static library / header
```

Python 不参与生成物运行。运行期不得依赖 Python interpreter、LLVM JIT 或 Weft Python package。

其他前端可以直接生成同一个 canonical Weft Kernel IR，不经过 Python。

### 6.2 Kernel 定义

```python
import weft
import weft.language as W

@weft.kernel
def saxpy(
    x: W.ptr[W.f32],
    y: W.ptr[W.f32],
    a: W.f32,
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        xv = W.load(x + i)
        yv = W.load(y + i)
        W.store(y + i, a * xv + yv)
```

`@weft.kernel` 定义一个 worker-local entry。它不是 Python callable 的 eager 执行语义。

### 6.3 Kernel 参数类型

核心参数类型：

```python
W.i1
W.i8, W.i16, W.i32, W.i64
W.u8, W.u16, W.u32, W.u64
W.f16, W.bf16, W.f32, W.f64
W.index
W.ptr[T]
W.constexpr[T]
```

实现可以增加扩展 scalar type，但不得改变核心类型语义。

`W.index` 是地址和逻辑坐标使用的整数类型，不等于 physical lane index。

`W.constexpr[T]` 是 specialization-time meta-parameter。它与普通 runtime scalar 必须在类型上区分。

### 6.4 Pointer qualifier

Pointer 参数可以声明：

- address space；
- readonly / writeonly；
- noalias；
- minimum alignment；
- restrict-like ownership facts。

这些 qualifier 是作者承诺。编译器可以用其进行 vector memory、hoist、prefetch 和 local fusion；运行时违反承诺属于调用方错误。

概念语法：

```python
x: W.ptr[W.f32, W.readonly, W.noalias, W.aligned(64)]
```

具体 Python typing 形式可以调整，但 canonical IR 必须保存相同事实。

### 6.5 Compile-time meta-parameter

```python
@weft.kernel
def gemm_worker(..., BM: W.constexpr[W.index], BN: W.constexpr[W.index], BK: W.constexpr[W.index]):
    ...
```

Source 只声明 meta-parameter 的语义位置。候选值集合由 build specification 提供，不要求写进 kernel body。

禁止在 runtime data-dependent branch 中把 `W.constexpr` 当作普通运行时输入。

---

## 7. Python 控制流

### 7.1 Scalar range

```python
for row in W.range(row_begin, row_end):
    ...
```

`W.range` 是普通有序 scalar loop。作者写出的 carried scalar / block state 必须保持逻辑迭代顺序，除非它被显式改写成 reduce、scan 或 summary fold。

### 7.2 Scalar condition

Python `if` 条件必须是 scalar `i1`。

对 VLA / block predicate 必须使用：

```python
W.select(predicate, true_value, false_value)
```

或 validity-aware memory / structured primitive，不能把 vector predicate 当作 Python 控制流。

### 7.3 While 与 early exit

`W.while_` 或受限 Python `while` 可以表达有序状态机。其 carried state 默认不可重排、不可跨 iteration 并行。

### 7.4 Helper function

纯 helper 可以使用：

```python
@W.pure
def merge(a, b):
    ...
```

Effectful helper 必须显式声明 effect，并遵守 kernel ABI / region 限制。

任意 Python reflection、动态对象、文件 I/O、异常、generator 和运行时 monkey-patching 不属于 kernel language。

---

## 8. VLA region 规范

### 8.1 Source 语法

```python
with W.vla(begin, end) as i:
    value = W.load(ptr + i)
    W.store(out + i, value * 2.0)
```

`i` 是 logical VLA index region value，不是 scalar loop induction variable，也不是 physical lane ID。

### 8.2 覆盖语义

一个 VLA region 必须覆盖 `[begin, end)` 中每个逻辑位置恰好一次。

目标 lowering 可以选择：

```text
remaining = end - begin
base = begin
while remaining > 0:
    vl = choose_vl(remaining, SEW, LMUL, selected policy)
    execute logical indices [base, base + vl)
    base += vl
    remaining -= vl
```

上述循环只是物理 realization，不是 canonical source 结构。

### 8.3 Strip 不可观察性

Source 与 canonical IR 禁止提供：

- `W.vl()`；
- `W.vlen()`；
- lane ID；
- strip number；
- first/last strip test；
- exact LMUL；
- vector register number。

需要根据 logical position 判断首尾时，作者使用 `i == begin`、`i + 1 == end` 等逻辑 predicate，而不是物理 strip 状态。

### 8.4 VLA region 中的状态限制

VLA region body 不得任意修改外层 scalar / block state。

跨整个 VLA domain 的状态必须通过以下一种结构产生：

- `W.reduce`；
- `W.scan`；
- `W.summary_fold`；
- effectful memory operation；
- atomic operation。

任意依赖上一逻辑元素的 recurrence 必须写成有序 scalar loop，或使用显式 `W.scan`。这样可以防止 source 行为依赖编译器选择的 strip 边界。

### 8.5 VLA effect independence

非 atomic 的 VLA memory effect 必须满足 lane independence：

- 两个 active logical indices 不得写同一地址；
- 一个 iteration 不得依赖另一个 active iteration 写入的结果；
- 若存在可能冲突，作者必须使用 atomic、scalar loop 或显式 ordered primitive。

该性质一般无法完全静态证明，属于作者义务；编译器可以利用 noalias、affine pointer 和 effect analysis 尽量检查。

### 8.6 VLA 与 block axis

VLA region 内可以构造额外的 specialization-time block axis，形成：

```text
[VLA, D0, D1, ...]
```

但物理硬件仍只有一个 RVV lane domain。静态 block axes 必须由 provider 映射到 serial loop、register repeat、register microtile 或 extension fragment，不能被伪装成第二个 VLA lane axis。

---

## 9. Logical predicate、masked value 与 physical tail

### 9.1 三个不同概念

Weft 必须严格区分：

1. **iteration extent**：例如 `i ∈ [0, N)`；
2. **logical predicate**：causal、padding、ragged validity、bounds 或用户条件；
3. **physical tail**：最后一个 strip 的 `remaining` 小于最大可用 `vl`。

Physical tail 不进入 canonical mask。它由 VLA lowering 通过缩短 `vl` 实现。

### 9.2 Masked load

```python
x = W.load(ptr, where=valid)
```

当 `other` 省略时，结果是一个带 logical validity 的 masked value。

```python
x = W.load(ptr, where=valid, other=0.0)
```

当提供 `other` 时，结果是普通 filled value；无效位置的值等于 `other`。

### 9.3 Masked value 传播

Pointwise op 对 masked operand 传播 validity。多个 masked operand 的默认 validity 是其 predicate 的逻辑与，除非 op 显式定义其他规则。

Masked value 只能：

- 由 validity-aware structured primitive 消费；
- 由 masked store 消费；
- 通过 `W.fill(masked, value)` 转成普通 value；
- 通过 `W.valid(masked)` 取出逻辑 predicate。

禁止将未填充 masked value 作为普通 scalar / block value逃逸到不理解 validity 的 op。

### 9.4 Consumer-specific invalid semantics

| Consumer | invalid element 的语义 |
|---|---|
| masked store | 不产生 store effect |
| reduce | 不贡献，等价于该 reduce 的 identity |
| summary fold | 不贡献，等价于 state identity |
| contract | 不贡献，等价于 contraction algebra 的语义零元素 |
| scan | 默认作为 identity；segment boundary 必须使用独立 `segment_start` |
| ordinary sequential carry | 作者必须显式分支或 fill，compiler 不猜 |

对于量化/编码 operand，contract 的“语义零”不一定是物理 bit pattern `0`。实现该 primitive 的 provider 必须根据 primitive 语义处理。

### 9.5 Predicate 到 AVL 的吸收

若 compiler 证明 logical predicate 恰好是 VLA domain 的连续后缀 bounds，例如：

```text
i < N
```

可以通过缩短 AVL / `vl` 实现，而不生成 mask register。

这是 lowering 优化，不能改变 canonical predicate 的语义身份。

---

## 10. Memory 与 effect

### 10.1 Pointer arithmetic

Source 显式表达 pointer/index arithmetic：

```python
ptr = base + row * stride_row + col * stride_col
```

Compiler / provider 决定：

- scalar、unit-stride、strided、indexed 或 segment memory；
- vector grouping；
- mask realization；
- prefetch instruction；
- local address strength reduction。

### 10.2 Memory API

核心 API：

```python
W.load(ptr, *, where=True, other=W.invalid, alignment=None)
W.store(ptr, value, *, where=True, alignment=None)
W.prefetch(ptr, *, where=True, locality="default")
W.atomic_add(ptr, value, *, where=True, order="relaxed")
W.fence(order="acq_rel")
```

`alignment` 是 source assertion，不是 physical layout 选择。

### 10.3 Local rematerialization

Compiler 可以重新计算纯 pointer/index/value producer，也可以在合法 alias/effect 条件下重新 load，而不是强制保存 register value。

若 rematerialization 跨越可能写入同一 memory 的 effect 且无 alias proof，必须拒绝。

### 10.4 Local fusion

Provider 可以吸收 structured primitive 附近的纯 producer / consumer，例如 cast、decode、scale 或 packing，前提是：

- 所有被吸收 value 的 use 都在该局部 envelope 内，或仍能正确 materialize；
- 不创建新的外层 loop；
- 不改变 source staging 与 memory effect；
- 不根据完整 kernel 名匹配；
- 选择仍绑定到明确 primitive/interface。

这允许实现 fused dequantize + contract，但不允许从任意 multiply/add graph 自动发现 GEMM。

---

## 11. Block value API

### 11.1 Block axis

```python
m = W.block_axis(BM)
k = W.block_axis(BK)
```

`W.block_axis(D, offset=0)` 产生 shape `[D]` 的 logical index block。`D` 必须在 specialization 时可知，或具有 provider 可接受的静态上界。

### 11.2 Broadcasting

Python frontend 支持：

```python
m[:, None]
k[None, :]
```

并提供等价显式 API：

```python
W.expand_dims(value, axis)
W.broadcast_to(value, shape)
```

### 11.3 Block constructors

```python
W.full(shape, value, dtype=...)
W.zeros(shape, dtype=...)
W.reshape(value, shape)
W.transpose(value, permutation)
```

这些构造只定义 logical value。是否 materialize、如何分寄存器、是否重新生成，由 selected realization 决定。

### 11.4 Block load/store

`W.load` 与 `W.store` 对 block-shaped pointer 自动产生 block value/effect，不需要独立 `block_load` 语义。

```python
m = W.block_axis(BM)
k = W.block_axis(BK)
a_blk = W.load(a + (m0 + m[:, None]) * lda + (k0 + k[None, :]), where=...)
```

---

## 12. Contraction 规范

### 12.1 定义

`W.contract` 表达当前 logical operand blocks 上的 tensor contraction：

```python
acc = W.contract(
    lhs,
    rhs,
    init=acc,
    lhs_axes=(...),
    rhs_axes=(...),
    acc_dtype=W.f32,
)
```

它显式建立：

- lhs 与 rhs 的 paired contracted axes；
- free axes；
- output axis order；
- accumulator/init；
- element/accumulator dtype；
- logical validity；
- numerical mode。

### 12.2 Operand 必须是 region value

`lhs` 与 `rhs` 必须是 logical block / region value。

以下 source：

```python
for k in W.range(...):
    acc += a[k] * b[k]
```

是普通顺序 carry，不会被自动提升为 `W.contract`，也不会获得 register microtile / matrix extension 的完整调优自由度。

作者必须显式使用 `W.contract` 才把局部 contraction decomposition 权交给 compiler/provider。

### 12.3 三层辖域

作者拥有：

- 外层 cache blocking；
- outer M/N/K loops；
- staging、double-buffering 或重算骨架；
- 当前 operand block 的 shape；
- pointer、mask 与 load/store；
- block 之间的 carried accumulator。

`W.contract` 拥有：

- 当前 blocks 的 contraction axes 与 output relation；
- 局部 accumulator 语义；
- 允许 provider 重组的局部计算域。

Compiler/provider 拥有：

- register microtile `mr × nr`；
- LMUL、register repeat；
- K 内部 unroll；
- RVV microkernel；
- matrix fragment；
- op-local packing/scratch；
- extension instruction family；
- op-local software pipeline。

### 12.4 Packing 边界

Provider 只能自动决定 contract-local、短生命周期且不会越过 primitive 边界的 packing/scratch。

以下行为必须进入 source 或独立显式 primitive：

- 改变 public ABI 的 packed format；
- 持久化 packed weight；
- 被多个 contract 共享的 packing；
- 跨 outer loop 保留的 transformed storage；
- 改变 source-visible memory effect 的 staging。

### 12.5 Shape 规则

Paired contraction axes 必须具有相同 logical extent identity。仅两个维度都打印为 `-1` 或运行时数值偶然相等，不构成相等证明。

默认 output axis order 是：

```text
lhs 未收缩轴，随后 rhs 未收缩轴
```

可以通过 `output_order` 显式重排。

`init` 必须是 scalar 或 output-shaped region value。

### 12.6 Python 签名

```python
W.contract(
    lhs,
    rhs,
    *,
    init=None,
    lhs_axes: tuple[int, ...],
    rhs_axes: tuple[int, ...],
    output_order: tuple[int, ...] | None = None,
    acc_dtype=None,
    out_dtype=None,
    where_lhs=True,
    where_rhs=True,
    order="relaxed",
    math="native",
)
```

`W.dot` 可以作为 rank-1/rank-2 convenience，但 canonical IR 必须统一为 contraction。

### 12.7 Provider 不得做的事

Contract provider 禁止：

- 创建 source 中不存在的 outer K loop；
- 修改 cache block 位置；
- 把普通 pointwise graph 改写成 contract；
- 跨多个 source contract 合并成新的算法；
- 改变 init、axes、output order、logical predicate 或 memory effect；
- 用 kernel 名或格式名选择实现。

---

## 13. State algebra

Weft 表面提供四种 state construct，但它们共享清楚的代数基础。

### 13.1 Reduce

```python
result = W.reduce(
    value,
    op="add",
    identity=0.0,
    where=True,
    axis=None,
    order="relaxed",
    acc_dtype=W.f32,
)
```

语义：只观察最终聚合状态。

对于当前 VLA region，`axis=None` 默认消去当前 VLA axis。对于 block value，必须显式给出 block axis 或使用无歧义默认。

Compiler 可以产生 lane-local reduction、tree reduction、cross-strip accumulator 与 extension reduction instruction。

### 13.2 Scan

```python
prefix = W.scan(
    value,
    op="add",
    identity=0.0,
    inclusive=True,
    where=True,
    segment_start=None,
)
```

语义：为每个逻辑位置产生前缀状态。

Scan 的 output order 是 source-observable，不能退化成只输出最终值的 reduce。

`segment_start` 是独立语义，不得由 logical mask 隐式猜测。

### 13.3 Summary fold

```python
state_or_result = W.summary_fold(
    value,
    *,
    identity=...,
    lift=...,
    merge=...,
    finalize=None,
    where=True,
    order="preserve",
)
```

语义：

```text
lift(element) -> state
merge(state_a, state_b) -> state
finalize(state) -> result
```

它用于 online softmax、stable weighted merge、统计 summary、分块归一化等需要重标定的 monoid-like state。

Online softmax 的 rescale 必须写在 `merge` 中，而不是隐藏在普通 carried loop 里。

### 13.4 Sequential carry

普通 scalar loop 中的 carried state：

```python
state = init
for i in W.range(begin, end):
    state = step(state, i)
```

默认严格按 logical iteration order 执行。Compiler 不得将其视为 associative fold，也不得根据代码形状猜测结合律。

### 13.5 共享 algebra interface

Reduce、scan 与 summary fold 都可以实现共同的 associative-state interface：

```text
state type
identity
lift
merge
optional finalize
observable mode: final | prefixes
```

但 canonical source 必须保留不同 observable contract：

- reduce 只输出 final；
- scan 输出 prefixes；
- summary fold 有显式 lift/merge/finalize；
- sequential carry 没有可重组 algebra。

不得为了复用底层实现而把四种 source 语义合并成一个模糊 op。

### 13.6 User obligation

对自定义 `merge`，作者必须保证：

- state type 闭合；
- `identity` 对空分区成立；
- `merge` 满足声明的结合性质；
- 若 `order="relaxed"` 允许交换分区，必须满足所需的交换性质；
- `lift/merge/finalize` 无未声明 side effect。

Compiler 必须检查类型、region capture 与 effect purity，但不要求证明结合律。代数性质属于显式 user obligation，不引入审批或证明系统。

---

## 14. Numerical semantics

### 14.1 最小数值接口

Weft 不建立全局 numerical contract framework。只有真正影响 observable semantics 与高性能 lowering 的属性进入 op：

- input / output dtype；
- accumulator dtype；
- `order` / reassociation policy；
- `math` mode；
- rounding；
- saturation；
- exceptional-value policy（仅相关 op）。

### 14.2 Reduction order

`order` 至少支持：

- `"ordered"`：保持 source 定义的逐元素顺序，不允许改变浮点 parenthesization；
- `"preserve"`：允许对连续分区重新结合，但保持逻辑分区顺序；
- `"relaxed"`：允许目标相关的 tree、lane 与 strip 重组。

Built-in floating reduce 的默认值是 `"relaxed"`，以允许高性能 VLA reduction。需要严格复现时，作者必须显式选择 `"ordered"`。

### 14.3 跨 VLEN 可复现性

对于 `order="relaxed"` 的 floating reduce、summary fold 或 contract：

- 同一 source 在不同 VLEN、LMUL 或 provider 上可以采用不同 parenthesization；
- 结果低位可以不同；
- Weft 不承诺 bitwise reproducibility；
- 该差异是语言允许的实现自由，而不只是测试策略。

### 14.4 Math mode

`math` 至少支持：

- `"strict"`：禁止未授权近似和 contraction；
- `"native"`：允许目标原生精度与已定义的 fused instruction；
- `"fast"`：允许显式文档化的近似数学 realization。

默认 `"native"`。

近似 `exp`、`rsqrt`、activation 等 provider 必须满足相应 primitive 定义的 value semantics；误差测试属于实现质量，不进入 source 认证流程。

### 14.5 Quantization

窄化、量化与饱和 op 必须显式定义：

- source / destination dtype；
- scale / zero-point / codebook relation；
- rounding mode；
- saturation / wrap policy。

如果某个 RISC-V 扩展改变这些 observable semantics，必须增加新的局部 primitive 或显式属性，不能伪装成普通 cast/contract 的无差别 lowering。

---

## 15. RISC-V target model

### 15.1 Target profile

Weft 编译输入必须绑定一个 RISC-V target profile。Profile 保存 architectural facts，例如：

```text
XLEN / ABI / endianness
ISA extension set
supported scalar and vector element widths
available LMUL set
architectural vector register count
VLEN fixed value、范围或运行时未知标记
mask/tail architectural能力
matrix/quant/vendor extension identity
fragment shapes 与 accumulator classes
rounding/saturation capability
memory instruction classes
```

### 15.2 Optional microarchitecture profile

可以另外提供非语义 microarchitecture hints：

```text
cache capacity / line size
preferred unroll
instruction throughput / latency hint
load-store bandwidth hint
```

这些 hint 只能用于候选排序或搜索缩减，不能让 architectural-illegal candidate 变合法。

### 15.3 Target profile 不是 cost model

Target profile 是 typed machine fact table。Provider 可以使用解析 resource constraints；构建期 tuner 使用实测处理未建模的 residual performance。

---

## 16. Primitive realization provider

### 16.1 Provider 的职责

一个 provider 为某类 semantic primitive 定义参数化物理实现族：

```text
supported semantic interface
capability predicate
physical parameter space
legality constraints
resource equations
selected record schema
lowering/emission
optional local fusion envelope
```

Provider 绑定：

```text
primitive/interface + typed operands + target facts
```

禁止绑定：

```text
kernel name + operator name + model format + route string
```

### 16.2 Core primitive interfaces

规范至少定义以下接口族：

- VLA pointwise；
- scalar / vector memory；
- reduce；
- scan；
- summary fold；
- contract；
- permute / gather / table lookup；
- widen / narrow / convert；
- decode / dequantize；
- math primitive；
- atomic / fence。

一个扩展只需要为它真正加速的接口注册 provider。

### 16.3 Baseline 与扩展组合

同一个 target 可以同时拥有：

```text
baseline scalar lowering
baseline RVV provider
Zvfh provider
matrix extension provider
vendor quant provider
vendor permute/lookup provider
```

它们是可组合的 primitive providers，不是互斥 whole-kernel backend。

普通 scalar control 直接由 scalar lowering 处理；普通 VLA pointwise/memory 通常由 RVV provider 处理；structured primitive 可以选择更专用 provider。

### 16.4 纯 realization 扩展

若扩展不改变 source-observable semantics，它可以实现现有 primitive，无需修改 Python DSL 或 canonical IR。

例如：

```text
W.contract
  → RVV FMA microkernel
  → RVV dot extension
  → IME fragment
  → future matrix extension
```

Provider 可以改变：

- LMUL；
- register microtile；
- fragment；
- packing；
- local scratch；
- instruction family；
- local unroll / pipeline。

不得改变 source axes、outer loops、state、mask、ABI 或 numerical mode。

### 16.5 新语义扩展

若扩展引入可观察的新语义，例如：

- block-scaled accumulation；
- 特有 codebook / decode；
- 特殊 saturation / rounding；
- persistent architectural state；
- 新的 lane permutation semantics；
- 不同 output relation；

则必须增加一个局部 canonical primitive，例如：

```python
W.block_scaled_contract(...)
W.table_decode(...)
W.saturating_dot(...)
```

它仍然必须是局部 primitive，不得增加 `softmax_kernel`、`q4_K_gemm_kernel` 等整算子 op。

### 16.6 算法 variant

若扩展要求改变：

- outer traversal；
- cache blocking 层次；
- persistent packed storage；
- 多阶段 staging；
- 跨 primitive 共享状态；

则作者或上游必须显式提供另一份 kernel variant。Compiler 不自动发明该 variant。

Build system 可以在作者提供的 variants 中选择，但每个 variant 都必须是独立完整的 Weft kernel。

---

## 17. Compiler、provider 与 tuner 的权限边界

### 17.1 作者/source 拥有

- worker-local ABI；
- scalar loop 与 control；
- VLA region 的逻辑范围；
- algorithm/cache blocking 的存在和位置；
- staging / recomputation skeleton；
- pointer/index/mask/effect；
- structured primitive 是否存在；
- contract outer loops 与 operand logical blocks；
- source meta-parameter；
- explicit algorithm variant；
- observable numerical policy。

### 17.2 Provider 拥有

- primitive-local physical parameter family；
- LMUL 候选；
- register tile / repeat；
- instruction / fragment family；
- local packing / scratch；
- local unroll / pipeline；
- capability 与 resource constraints；
- lowering。

### 17.3 Compiler 拥有

- 从 canonical facts 重算 shape、axis、stride 与 use relation；
- 绑定 target profile；
- 构造 provider 候选；
- 解析 legality；
- 删除寄存器、fragment、dtype、mask、memory 不合法实例；
- 推导动态 `vl` mechanics；
- 生成 selected execution；
- 进行 semantics-preserving local optimization；
- 生成 artifact。

### 17.4 Tuner 拥有

- 在 compiler 已证明合法的候选中测量；
- 为 source meta-parameter 与 provider physical knobs 选值；
- 选择作者提供的 kernel variant；
- 生成 specialization / dispatch decision。

Tuner 禁止：

- 创造 source 中不存在的 loop 或 primitive；
- 让非法 candidate 合法；
- 改变 numerical policy；
- 根据测量绕过 verifier。

---

## 18. Tuning 规范

### 18.1 三类变量

#### Source structural knobs

例如：

```text
BM / BN / BK
作者显式声明的 prefetch distance
作者显式声明的 staging depth
显式 algorithm variant
```

其存在与语义位置属于 source，候选值由 build specification 提供。

#### Provider physical knobs

例如：

```text
LMUL
register microtile mr×nr
register repeat
unroll
fragment strategy
local packing strategy
```

由 provider 声明参数空间与约束。

#### Compiler-derived mechanics

例如：

```text
每次 strip 的 actual vl
physical tail
vsetvl placement
已唯一决定的 mask realization
pointer strength reduction
```

这些不是 tuner knob。

### 18.2 AOT build-time search

Tuning 流程发生在构建期：

```text
候选绑定
  → selected execution
  → emit source/object
  → compile
  → benchmark
  → 固化最快合法变体
```

运行期不带 compiler，不即时生成代码。

### 18.3 Shape specialization

Tuning key 是：

```text
(target profile, specialization predicate)
```

Specialization predicate 可以是：

- exact shape；
- shape range；
- alignment / stride class；
- quant format；
- model-specific constant；
- generic fallback。

### 18.4 单变体与多变体

Weft artifact 必须支持两种 AOT 形态：

1. **generic entry**：runtime shape 动态，单个 VLEN-agnostic kernel；
2. **multiversion entry**：构建期生成多个 shape / target specialization，并生成轻量 runtime dispatch。

Dispatch 只在已生成的 AOT variants 中选择，不进行 JIT。

### 18.5 Build specification

Python source 不需要包含完整搜索集合。外部 build specification 提供：

```text
entry
source meta domains
target profile
specialization predicates
measurement harness
allowed providers
optional provider constraints
artifact options
```

概念示例：

```toml
[entry.gemm_worker]
meta.BM = [16, 32, 64]
meta.BN = [16, 32, 64]
meta.BK = [16, 32]

[[entry.gemm_worker.specialization]]
when = "N % 64 == 0"

[[entry.gemm_worker.specialization]]
when = "true" # fallback
```

具体配置格式可以变化，职责边界不得变化。

---

## 19. Canonical Weft Kernel IR

### 19.1 唯一算法真理

Canonical Kernel IR 保存：

- kernel ABI；
- scalar / pointer / constexpr types；
- scalar control flow；
- VLA region；
- logical block axis与 region value；
- pointer/index；
- logical predicate / masked value；
- load/store/atomic/fence；
- pointwise；
- reduce/scan/summary fold；
- contract 与 extension primitive；
- source meta-parameter；
- source location；
- numerical attributes。

它不得保存：

- exact `vl`；
- LMUL；
- register number；
- register microtile；
- provider ID；
- IME fragment；
- instruction spelling；
- build measurement；
- thread count或 launch policy。

### 19.2 核心 op 族

概念 op：

```text
weft_kernel.kernel
weft_kernel.range
weft_kernel.vla
weft_kernel.block_axis
weft_kernel.load / store / prefetch
weft_kernel.atomic / fence
weft_kernel.mask / fill
weft_kernel.reduce
weft_kernel.scan
weft_kernel.summary_fold
weft_kernel.contract
weft_kernel.reshape / transpose / broadcast
weft_kernel.meta_value
```

扩展可以注册 sibling dialect 的局部 semantic op，但 `weft-compile` 的正式输入必须能独立 parse、verify 并链接所需扩展 dialect。

### 19.3 Python frontend

Python frontend 必须直接生成 canonical Kernel IR。它不得维护另一套长期 typed Python IR、另一套 verifier 或另一份算法 authority。

---

## 20. Selected Execution IR

### 20.1 只保存物理选择

Selected IR 保存无法从 canonical IR 与固定 target facts 唯一重算的选择，例如：

```text
meta binding
VLA region → RVV/scalar realization
SEW / LMUL / repeat / unroll
memory strategy
reduction / scan strategy
contract provider
register microtile
fragment / packing / scratch
local fusion decision
math primitive strategy
multiversion specialization identity
```

### 20.2 不复制算法

Selected IR 禁止复制：

- source loop bounds；
- pointer expression；
- logical shape；
- contraction axes；
- logical predicate；
- reduce combiner；
- state identity；
- ABI；
- output order。

这些事实必须通过 canonical references 重算。

### 20.3 计划记录

概念记录：

```text
weft_execution.meta_binding
weft_execution.vla_plan
weft_execution.memory_plan
weft_execution.reduce_plan
weft_execution.scan_plan
weft_execution.summary_plan
weft_execution.contract_plan
weft_execution.scratch_plan
weft_execution.specialization
```

Extension provider 可以定义 sibling selected op，但必须实现统一 selected-provider interface，并只引用 canonical anchor。

---

## 21. Lowering 与 artifact

### 21.1 瞬态 owner IR

RVV、矩阵扩展、vendor extension 的 typed lowering IR 可以存在，但只能：

- 从 canonical + selected execution 单向生成；
- 不新增 selection；
- 不成为 source front door；
- 不反向驱动 canonical IR；
- 不成为第四份长期 authority。

### 21.2 Artifact 类型

Weft compiler 至少支持：

```text
canonical MLIR
selected MLIR
readable generated source（可选）
relocatable object
static library
C-compatible public header
```

### 21.3 Runtime ABI

生成 entry 必须具有普通 C-compatible ABI。

```c
void weft_rms_norm(
    const float *x,
    const float *weight,
    float *y,
    int64_t row_begin,
    int64_t row_end,
    int64_t cols,
    int64_t stride);
```

以上只是示例；实际 work descriptor 由 kernel source 决定。

Weft artifact 不创建线程。外部 runtime 可以：

```cpp
parallel_for(worker_ranges, [&](auto range) {
    weft_rms_norm(..., range.begin, range.end, ...);
});
```

### 21.4 Multiversion dispatch

可选 AOT dispatcher 可以检查：

- shape predicate；
- alignment / stride class；
- target extension；
- fixed VLEN guard；

并调用已编译 variant。

Dispatcher 不管理线程，也不调用 compiler。

---

## 22. Python DSL API 汇总

### 22.1 Kernel 与类型

```python
@weft.kernel
def kernel(...): ...

W.ptr[T]
W.constexpr[T]
W.index
W.i*/W.u*/W.f*/W.bf16
```

### 22.2 Control

```python
W.range(begin, end, step=1)
W.select(pred, a, b)
@W.pure
```

### 22.3 VLA

```python
with W.vla(begin, end) as i:
    ...
```

### 22.4 Memory

```python
W.load(ptr, where=True, other=W.invalid, alignment=None)
W.store(ptr, value, where=True, alignment=None)
W.prefetch(ptr, where=True, locality="default")
W.atomic_add(ptr, value, where=True, order="relaxed")
W.fence(order="acq_rel")
```

### 22.5 Block values

```python
W.block_axis(extent, offset=0)
W.full(shape, value, dtype=None)
W.zeros(shape, dtype)
W.expand_dims(value, axis)
W.broadcast_to(value, shape)
W.reshape(value, shape)
W.transpose(value, permutation)
```

### 22.6 Predicate

```python
W.valid(masked_value)
W.fill(masked_value, fill_value)
W.select(predicate, a, b)
```

### 22.7 State algebra

```python
W.reduce(...)
W.scan(...)
W.summary_fold(...)
```

普通 sequential carry 使用 scalar `W.range`。

### 22.8 Structured compute

```python
W.contract(...)
W.dot(...)  # sugar
W.permute(...)
W.lookup(...)
W.decode(...)
W.widen(...)
W.narrow(...)
```

语义不同的扩展 primitive 进入独立 namespace 或 extension dialect，但必须保持局部、typed、可组合。

---

## 23. 完整示例

### 23.1 Elementwise kernel

```python
import weft
import weft.language as W

@weft.kernel
def add_bias(
    x: W.ptr[W.f32],
    bias: W.ptr[W.f32],
    y: W.ptr[W.f32],
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        xv = W.load(x + i)
        bv = W.load(bias + i)
        W.store(y + i, xv + bv)
```

外部 runtime 决定每个 worker 的 `[begin,end)`。

### 23.2 RMSNorm worker

```python
@weft.kernel
def rms_norm_worker(
    x: W.ptr[W.f32],
    weight: W.ptr[W.f32],
    y: W.ptr[W.f32],
    row_begin: W.index,
    row_end: W.index,
    cols: W.index,
    stride: W.index,
    eps: W.f32,
) -> None:
    for row in W.range(row_begin, row_end):
        with W.vla(0, cols) as i:
            value = W.load(x + row * stride + i)
            sum_sq = W.reduce(
                value * value,
                op="add",
                identity=W.f32(0.0),
                order="relaxed",
                acc_dtype=W.f32,
            )

        scale = W.rsqrt(sum_sq / W.cast(cols, W.f32) + eps, math="native")

        with W.vla(0, cols) as i:
            value = W.load(x + row * stride + i)
            w = W.load(weight + i)
            W.store(y + row * stride + i, value * scale * w)
```

作者定义两遍算法与 row loop；compiler 决定每遍的 strip、LMUL、reduction tree 与 memory realization。

### 23.3 Predicate 与 reduce identity

```python
with W.vla(begin, end) as i:
    valid = token_index(i) < sequence_length
    value = W.load(x + address(i), where=valid)
    maximum = W.reduce(
        value,
        op="max",
        identity=W.neg_inf(W.f32),
        order="relaxed",
    )
```

`valid=false` 的位置不贡献，语义等同于 `-inf`，而不是无条件填 `0`。

### 23.4 Online softmax summary

```python
@W.pure
def softmax_lift(x):
    return W.tuple(x, W.f32(1.0))

@W.pure
def softmax_merge(a, b):
    ma, sa = a
    mb, sb = b
    m = W.maximum(ma, mb)
    s = sa * W.exp(ma - m, math="native") + sb * W.exp(mb - m, math="native")
    return W.tuple(m, s)

with W.vla(0, n) as i:
    valid = i < logical_n
    x = W.load(ptr + i, where=valid)
    state = W.summary_fold(
        x,
        identity=W.tuple(W.neg_inf(W.f32), W.f32(0.0)),
        lift=softmax_lift,
        merge=softmax_merge,
        finalize=None,
        order="preserve",
    )
```

编译器可以为每个 strip 形成局部 summary，再用 `softmax_merge` 合并。Rescale 是 merge 语义的一部分，不依赖源码中手写的 strip loop。

### 23.5 Worker-local blocked GEMM

```python
@weft.kernel
def gemm_worker(
    a: W.ptr[W.f16],
    b: W.ptr[W.f16],
    c: W.ptr[W.f32],
    m_begin: W.index,
    m_end: W.index,
    n: W.index,
    k: W.index,
    lda: W.index,
    ldb: W.index,
    ldc: W.index,
    BM: W.constexpr[W.index],
    BN: W.constexpr[W.index],
    BK: W.constexpr[W.index],
) -> None:
    for m0 in W.range(m_begin, m_end, BM):
        for n0 in W.range(0, n, BN):
            acc = W.zeros((BM, BN), dtype=W.f32)

            for k0 in W.range(0, k, BK):
                mi = W.block_axis(BM)
                ni = W.block_axis(BN)
                ki = W.block_axis(BK)

                m_idx = m0 + mi[:, None]
                n_idx = n0 + ni[None, :]
                k_lhs = k0 + ki[None, :]
                k_rhs = k0 + ki[:, None]

                a_valid = (m_idx < m_end) & (k_lhs < k)
                b_valid = (k_rhs < k) & (n_idx < n)

                a_blk = W.load(a + m_idx * lda + k_lhs, where=a_valid)
                b_blk = W.load(b + k_rhs * ldb + n_idx, where=b_valid)

                acc = W.contract(
                    a_blk,
                    b_blk,
                    init=acc,
                    lhs_axes=(1,),
                    rhs_axes=(0,),
                    acc_dtype=W.f32,
                    order="relaxed",
                    math="native",
                )

            mi = W.block_axis(BM)
            ni = W.block_axis(BN)
            m_idx = m0 + mi[:, None]
            n_idx = n0 + ni[None, :]
            out_valid = (m_idx < m_end) & (n_idx < n)
            W.store(c + m_idx * ldc + n_idx, acc, where=out_valid)
```

这里：

- `m0/n0/k0`、BM/BN/BK 与 staging skeleton 归作者；
- block values 与 contraction axes 归 canonical semantics；
- `mr×nr`、LMUL、RVV microkernel 或 IME fragment 归 provider/compiler/tuner；
- 外部 runtime 决定每个 worker 的 `[m_begin,m_end)`。

### 23.6 语义不同的扩展 primitive

假设某扩展提供带 block scale、固定 saturation 与特定 accumulator 语义的 dot，它不能作为普通 `W.contract` 的无差别 lowering。Source 使用显式 primitive：

```python
acc = W.block_scaled_contract(
    packed_a,
    packed_b,
    scale_a=scale_a,
    scale_b=scale_b,
    init=acc,
    rounding="rne",
    saturation=True,
)
```

不同硬件仍可以为该同一 semantic primitive 提供多个 provider。

---

## 24. Verifier 义务

Canonical verifier 必须检查：

- kernel ABI 与 type legality；
- pointer arithmetic type；
- VLA region lexical nesting；
- VLA value 不非法逃逸；
- VLA region 中无任意外层 state mutation；
- block shape 与 broadcast；
- dynamic extent identity；
- contract paired axes 与 output shape；
- reduce/scan/summary state type；
- custom lift/merge/finalize purity；
- masked value 只进入 validity-aware consumer；
- meta parameter 不进入非法 runtime role；
- extension primitive dialect 是否注册；
- effect 与 atomic/fence 基本规则。

Selected verifier 必须检查：

- 所有物理 plan 引用真实 canonical anchor；
- provider 与 target capability 匹配；
- LMUL、register、fragment、scratch legality；
- selected record 未复制或改变 canonical algorithm；
- 每个需要 realization 的 primitive 有唯一计划；
- local fusion envelope 合法；
- multiversion predicate 具有 fallback 或明确 no-match 行为。

---

## 25. 作者义务

下列性质通常不能完全静态证明，由作者承担：

- `noalias`、alignment 与 bounds assertion 真实成立；
- VLA non-atomic iteration effect independence；
- custom summary `merge` 的 identity / associativity / commutativity 声明；
- runtime work slices 之间无未同步数据竞争；
- 外部 runtime 传入的 work descriptor 合法；
- extension-specific semantic primitive 的参数满足其 source contract。

这些义务必须由文档和 diagnostics 明确，不建立额外审批系统。

---

## 26. 禁止退化方向

出现以下任一情况，说明实现偏离本规范：

1. 在 canonical language 中重新引入 `program_id` / grid；
2. Weft runtime 自行创建线程或把 OpenMP 变成语言语义；
3. 把整个 worker kernel 当成一个静态 Triton-style tile；
4. 暴露 `vl`、VLEN、LMUL、lane ID 或 vector register number 给普通 source；
5. 把 logical block 等同于 register tile 或 IME fragment；
6. 从普通 multiply/add graph 自动发现并替换成 contract；
7. 根据 GEMM、Softmax、q4_K 等名字选择整段实现；
8. 新增扩展时复制整算子 kernel 模板；
9. provider 修改 source outer loop、staging、ABI 或 logical predicate；
10. selected IR 与 canonical IR 同时拥有算法真理；
11. tuner 创造 candidate 或让非法 candidate 合法；
12. 把 physical tail 存成 canonical logical mask；
13. 用普通 carried loop 假装 summary fold，同时期待 compiler 猜出 merge；
14. 把所有 state construct 压成一个无 observable distinction 的 op；
15. 把 numerical semantics 降成“测试时用容差”，却不给 compiler 合法 reassociation 权；
16. 让 Python 成为运行时依赖或唯一可生成 IR 的入口；
17. 将 extension evidence、approval 或 certification 变成 DSL 核心抽象。

---

## 27. 最终架构总图

```text
Python Weft DSL / other frontend
                │
                ▼
Canonical worker-local Weft Kernel IR
  ├─ scalar control / pointer / effect
  ├─ VLA iteration regions
  ├─ logical predicates / masked values
  ├─ logical block values
  ├─ reduce / scan / summary fold
  ├─ contract / lookup / decode / extension primitives
  └─ source meta-parameters
                │
                ▼
RISC-V target profile
  architectural facts + optional microarchitecture hints
                │
                ▼
Primitive-local realization providers
  capability + parameter family + legality + resource + lowering
                │
                ▼
Compiler-derived legal execution space
                │
        build-time AOT tuner
                │
                ▼
Selected Execution IR
  vl mechanics / LMUL / memory plan / microtile / fragment / strategy
                │
                ▼
Mechanical scalar + RVV + extension lowering
                │
                ▼
object / static library / C header / optional AOT dispatcher
                │
                ▼
llama.cpp / ggml / framework / application runtime
  owns threads, work partition and multi-core scheduling
```

---

## 28. 定位一句话

> **Weft 是一门面向单个 RISC-V worker/hart 的 AOT kernel DSL：作者用普通控制流、一等 VLA iteration region、logical block、显式 predicate/state algebra 与 structured compute primitive 编写完整 worker-local 算法；compiler 从 target facts 与 primitive provider 中构造合法的动态 `vl`、LMUL、register microtile、memory 与扩展 realization，构建期 tuner 选择性能点，多核调度由外部 runtime 负责。**
