# Contraction 与 State Algebra

## Contraction 规范

### 定义

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

Canonical `weft_kernel.contract` 保存 `lhs/rhs/init`、`where_lhs/where_rhs`、paired axes、
`output_order`、accumulator/output dtype、`order` 与 `math`。这些字段都是 source semantics，
不能由 target 从 shape 或 use context补出。

### Operand 必须是 region value

`lhs` 与 `rhs` 必须是 logical block / region value。

以下 source：

```python
for k in W.range(...):
    acc += a[k] * b[k]
```

是普通顺序 carry，不会被自动提升为 `W.contract`，也不会获得 register microtile / matrix extension 的完整调优自由度。

作者必须显式使用 `W.contract` 才把局部 contraction decomposition 权交给 target lowering。

### 三层辖域

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
- 允许 target lowering 重组的局部计算域。

Target lowering 拥有：

- register microtile `mr × nr`；
- LMUL、register repeat；
- K 内部 unroll；
- RVV microkernel；
- matrix fragment；
- op-local packing/scratch；
- extension instruction family；
- op-local software pipeline。

### Packing 边界

Target lowering 只能自动决定 contract-local、短生命周期且不会越过 primitive 边界的 packing/scratch。

以下行为必须进入 source 或独立显式 primitive：

- 改变 public ABI 的 packed format；
- 持久化 packed weight；
- 被多个 contract 共享的 packing；
- 跨 outer loop 保留的 transformed storage；
- 改变 source-visible memory effect 的 staging。

### Shape 规则

Paired contraction axes 必须具有相同 logical extent identity。仅两个维度都打印为 `-1` 或运行时数值偶然相等，不构成相等证明。

默认 output axis order 是：

```text
lhs 未收缩轴，随后 rhs 未收缩轴
```

可以通过 `output_order` 显式重排。若 operand 是 active VLA region，VLA axis 不能被收缩，
必须作为结果的第一个 free/batch axis保留；`output_order` 也不能把它移出首位。

`init` 必须是 scalar 或 output-shaped block/region value，element type 等于 `acc_dtype`。
`where_lhs` 与 `where_rhs` 是显式 logical predicate，并必须能广播到对应 operand footprint。

### Python 签名

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

### Target lowering 不得做的事

Contract lowering 禁止：

- 创建 source 中不存在的 outer K loop；
- 修改 cache block 位置；
- 把普通 pointwise graph 改写成 contract；
- 把多个 source contract 改写成语义不同的新算法；
- 改变 init、axes、output order、logical predicate 或 memory effect；
- 用 kernel 名或格式名选择实现。

Target 可以在 effect/alias/numerical legality成立时联合 lower相邻 contract、共享局部 loads、
packing或instruction schedule，但每个 contract 的 axes、init、output、predicate与state boundary
必须仍然成立。


## State algebra

Weft 表面提供四种 state construct，但它们共享清楚的代数基础。

### Reduce

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

对于当前 VLA region，`axis=None` 消去 active VLA axis；canonical 记为 `axis=-1`。对于 block
value必须显式给出有效 block axis。Built-in reduce kind 是 `add`、`max`、`min`、`mul`、
`and` 与 `or`；masked/`where=false` 元素等价于 identity，不把 validity带到结果。

Compiler 可以产生 lane-local reduction、tree reduction、cross-strip accumulator 与 extension reduction instruction。

### Scan

```python
prefix = W.scan(
    value,
    op="add",
    identity=0.0,
    inclusive=True,
    where=True,
    segment_start=None,
    order="ordered",
    acc_dtype=...,
)
```

语义：为每个逻辑位置产生前缀状态。

Scan 的 output order 是 source-observable，不能退化成只输出最终值的 reduce。

`segment_start` 是独立语义，不得由 logical mask 隐式猜测。

### Summary fold

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

`lift`、`merge` 以及可选 `finalize` 必须是 `@W.pure` helper。Canonical regions 的参数分别是
element、`(state,state)` 与 state；每个 region只返回一个闭合值，不能含 memory effect。

### Sequential carry

普通 scalar loop 中的 carried state：

```python
state = init
for i in W.range(begin, end):
    state = step(state, i)
```

默认严格按 logical iteration order 执行。Compiler 不得将其视为 associative fold，也不得根据代码形状猜测结合律。

### 共享 algebra interface

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

### User obligation

对自定义 `merge`，作者必须保证：

- state type 闭合；
- `identity` 对空分区成立；
- `merge` 满足声明的结合性质；
- 若 `order="relaxed"` 允许交换分区，必须满足所需的交换性质；
- `lift/merge/finalize` 无未声明 side effect。

Compiler 必须检查类型、region capture 与 effect purity，但不要求证明结合律。代数性质属于显式 user obligation，不引入审批或证明系统。
