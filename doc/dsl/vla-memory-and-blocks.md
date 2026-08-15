# VLA、Predicate、Memory 与 Logical Block

## VLA region 规范

### Source 语法

```python
with W.vla(begin, end) as i:
    value = W.load(ptr + i)
    W.store(out + i, value * 2.0)
```

`i` 是 logical VLA index region value，不是 scalar loop induction variable，也不是 physical lane ID。

### 覆盖语义

一个 VLA region 必须覆盖 `[begin, end)` 中每个逻辑位置恰好一次。

目标 lowering 可以选择：

```text
remaining = end - begin
base = begin
while remaining > 0:
    vl = choose_vl(remaining, SEW, LMUL, backend config)
    execute logical indices [base, base + vl)
    base += vl
    remaining -= vl
```

上述循环只是物理 realization，不是 canonical source 结构。

### Strip 不可观察性

Source 与 canonical IR 禁止提供：

- `W.vl()`；
- `W.vlen()`；
- lane ID；
- strip number；
- first/last strip test；
- exact LMUL；
- vector register number。

需要根据 logical position 判断首尾时，作者使用 `i == begin`、`i + 1 == end` 等逻辑 predicate，而不是物理 strip 状态。

### VLA region 中的状态限制

VLA region body 不得任意修改外层 scalar / block state。

跨整个 VLA domain 的状态必须通过以下一种结构产生：

- `W.reduce`；
- `W.scan`；
- 显式typed summary primitive；
- 满足 lane independence 的 effectful memory operation。

任意依赖上一逻辑元素的 recurrence 必须写成有序 scalar loop，或使用显式 `W.scan`。这样可以防止 source 行为依赖编译器选择的 strip 边界。

普通有序scalar `for` / `while`可以出现在VLA body中；禁止的是第二个active VLA region，
不是scalar control。Target可以逐physical strip执行这些scalar loops，但不得把它们重排为
新的VLA axis，也不得改变loop-carried state或memory effect的逻辑顺序。

### VLA effect independence

VLA memory effect 必须满足 lane independence：

- 两个 active logical indices 不得写同一地址；
- 一个 iteration 不得依赖另一个 active iteration 写入的结果；
- 若存在可能冲突，作者必须使用 scalar ordered loop；当前 core surface 不提供 atomic VLA
  effect。未来只有在定义完整可观察 ordering/effect 语义后，才能增加相应 local primitive。

该性质一般无法完全静态证明，属于作者义务；编译器可以利用 noalias、affine pointer 和 effect analysis 尽量检查。

### VLA 与 block axis

VLA region内可以构造额外的logical block axis，形成：

```text
[VLA, D0, D1, ...]
```

Target lowering 可以把 logical block axes 映射到唯一 RVV lane domain、serial loop、register
repeat、register microtile、extension fragment或这些机制的合法组合，但不能把一个 block
axis 暴露成第二个独立、source-observable 的 physical lane domain。


## Logical predicate、masked value 与 physical tail

### 三个不同概念

Weft 必须严格区分：

1. **iteration extent**：例如 `i ∈ [0, N)`；
2. **logical predicate**：causal、padding、ragged validity、bounds 或用户条件；
3. **physical tail**：最后一个 strip 的 `remaining` 小于最大可用 `vl`。

Physical tail 不进入 canonical logical predicate。Target lowering可以使用缩短 AVL/`vl`、
内部 physical mask，或二者组合实现尾部；这些 mechanics 不得反向成为 source-observable
mask。

### Masked load

```python
x = W.load(ptr, where=valid)
```

当 `other` 省略且`where`不是静态`True`时，结果是一个带logical validity的masked value。
`where=True`的无填充值load覆盖整个逻辑domain，因此产生普通all-active value。

```python
x = W.load(ptr, where=valid, other=0.0)
```

当提供 `other` 时，结果是普通 filled value；无效位置的值等于 `other`。

### Masked value 传播

Pointwise op 对 masked operand 传播 validity。多个 masked operand 的默认 validity 是其 predicate 的逻辑与，除非 op 显式定义其他规则。

Masked value 只能：

- 由 validity-aware structured primitive 消费；
- 继续经过会传播相同 logical validity 的 pointwise chain。

禁止将未填充 masked value 作为普通 scalar / block value逃逸到不理解 validity 的 op。普通
consumer和store需要filled value时，作者必须在原始 `W.load` 提供 `other`。语言不提供从masked
value事后猜回predicate/fill的第二套API，当前public store也不接受masked value。

### Consumer-specific invalid semantics

| Consumer | invalid element 的语义 |
|---|---|
| store | 不接受masked value；作者使用独立`where`并在producer处给出fill |
| reduce | 不贡献，等价于该 reduce 的 identity |
| typed summary | 由该primitive自身定义 |
| dot / matmul | 不贡献，等价于乘加域的语义零元素 |
| scan | 默认作为 identity；segment boundary 必须使用独立 `segment_start` |
| ordinary sequential carry | 作者必须在 producer 处显式提供 filled value，compiler 不猜 |

对于量化/编码operand，局部dot primitive的“语义零”不一定是物理bit pattern `0`。Target lowering必须根据primitive语义处理。

### Predicate 到物理长度的吸收

若compiler证明logical predicate恰好等价于VLA domain的连续bounds restriction，例如：

```text
i < N
```

可以通过缩短 AVL / `vl` 实现，而不生成 mask register。

这是 lowering 优化，不能改变 canonical predicate 的语义身份。


## Memory 与 effect

### Pointer arithmetic

Source 显式表达 pointer/index arithmetic：

```python
ptr = base + row * stride_row + col * stride_col
```

Target lowering 决定：

- scalar、unit-stride、strided、indexed 或 segment memory；
- vector grouping；
- mask realization；
- source 不可观察的 physical prefetch instruction/schedule；
- local address strength reduction。

若access或predicate位于VLA内的nested scalar control中，target仍按每个实体的pointer、
lane relation、predicate与effect选择memory realization。该physical scan只为当前strip投影
地址与mask，不把作者的scalar loop、window或state改写成标准region形状。

### Memory API

核心 API：

```python
W.load(ptr, *, where=True, other=W.invalid, alignment=None)
W.store(ptr, value, *, where=True, alignment=None)
```

`alignment` 是 source assertion，不是 physical layout 选择。

### Local rematerialization

Compiler 可以重新计算纯 pointer/index/value producer，也可以在合法 alias/effect 条件下重新 load，而不是强制保存 register value。

若 rematerialization 跨越可能写入同一 memory 的 effect 且无 alias proof，必须拒绝。

### Local fusion

Target lowering 可以吸收 structured primitive 附近的纯 producer / consumer，例如 cast、decode、scale 或 packing，前提是：

- 所有被吸收 value 的 use 都在该局部 envelope 内，或仍能正确 materialize；
- 不发明新的算法级 loop、state 或 staging；
- 不改变 source staging 与 memory effect；
- 不根据完整 kernel 名匹配；
- 选择仍绑定到明确 primitive/interface。

这允许实现 fused dequantize + dot/matmul，也允许在 effect、alias 与 numerical legality 成立时
联合 lower 相邻 primitive、共享 load/register/schedule；每个 primitive 的 observable boundary
必须保留。它不允许从任意 multiply/add graph 自动发现 GEMM。


## Block value API

### Block domain与index

```python
m = W.block(BM)
k = W.block(BK)
```

`W.block(D, offset=0)`同时建立一个唯一logical axis identity并产生shape `[D]` 的index block。
每次调用都是不同axis；相同extent不代表相同domain。Source integer literal写成static dimension；
meta或runtime extent在canonical type中记为`-1`，真实长度由block_index operand保存。Target
不能从相等shape猜axis identity。

### Broadcasting

Python frontend 支持：

```python
m[:, None]
k[None, :]
```

这两种slicing只插入axis identity为`0`的singleton logical axis；它们不授权任意broadcast或
shape重写。Pointwise join只能合并同一source axis、显式singleton或scalar；同shape不同axis
在frontend/canonical边界拒绝。

### Block constructors

```python
W.full((axis0, axis1, ...), value, dtype=...)
W.zeros((axis0, axis1, ...), dtype=...)
```

Shape entry必须是direct `W.block` value，不接受裸整数。同一个block domain因此在constructor、
operand、result和loop carry中保持同一axis identity。当前public surface没有任意reshape/transpose；
算法需要的坐标变换由作者显式写入pointer/index relation。

### Block load/store

`W.load` 与 `W.store` 对 block-shaped pointer 自动产生 block value/effect，不需要独立 `block_load` 语义。

```python
m = W.block(BM)
k = W.block(BK)
a_blk = W.load(a + (m0 + m[:, None]) * lda + (k0 + k[None, :]), where=...)
```

Block load、pointwise、state 与 structured primitive 的结果都进入同一普通 SSA value system。
Block value可以有多个consumer，也可作为`for`/`while` carry跨iteration存活；它不必立即被某个
primitive消费或store。Target负责合法的register/memory handoff，但不得要求一个固定terminal
use closure来重新定义 source legality。
