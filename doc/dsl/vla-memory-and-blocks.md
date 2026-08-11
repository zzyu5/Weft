# VLA、Predicate、Memory 与 Logical Block

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
    vl = choose_vl(remaining, SEW, LMUL, backend config)
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

但物理硬件仍只有一个 RVV lane domain。静态 block axes 必须由 target lowering 映射到 serial loop、register repeat、register microtile 或 extension fragment，不能被伪装成第二个 VLA lane axis。

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

对于量化/编码 operand，contract 的“语义零”不一定是物理 bit pattern `0`。Target lowering 必须根据 primitive 语义处理。

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

Target lowering 决定：

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

Target lowering 可以吸收 structured primitive 附近的纯 producer / consumer，例如 cast、decode、scale 或 packing，前提是：

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

`W.block_axis(D, offset=0)` 产生 shape `[D]` 的 logical index block。`D` 必须在 specialization 时可知，或具有 target lowering 可接受的静态上界。

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

这些构造只定义 logical value。是否 materialize、如何分寄存器、是否重新生成，由 target lowering 决定。

### 11.4 Block load/store

`W.load` 与 `W.store` 对 block-shaped pointer 自动产生 block value/effect，不需要独立 `block_load` 语义。

```python
m = W.block_axis(BM)
k = W.block_axis(BK)
a_blk = W.load(a + (m0 + m[:, None]) * lda + (k0 + k[None, :]), where=...)
```

---
