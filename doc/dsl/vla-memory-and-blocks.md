# 逻辑轴、值与数据移动

## 逻辑域不是物理 layout

Weft source 描述逻辑元素身份、地址关系和生命周期。Lane、LMUL、vector type、register tuple
和 extension fragment 只存在于 target lowering。源程序中最重要的两个显式域构造是
`W.vla` 和 `W.axis`，但它们不是两套 kernel 模型。

## Scalar、memory view 与 engine value

- scalar value：控制、地址、loop counter 和 scalar state；
- memory view：typed pointer 与地址表达式形成的逻辑 footprint；
- engine value：带逻辑轴的 block/region SSA value，可被本 core 的 scalar/vector/matrix
  realization 消费。

Engine value 的名字不承诺常驻寄存器。普通 use-def 和 loop lifetime 决定编译器应共享、
reload、rematerialize、spill 或 local-pack。

## `W.vla`

`with W.vla(begin, end) as i` 创建一条动态 extent 的 region axis。作者明确授权其元素用 scalable
SIMD 实现；target 选择 strip 长度、SEW/LMUL、mask/tail 与 memory form。

VLA 的限制：

- region 不可嵌套；
- region value 不可逃出其词法作用域；
- VLA body 不可暗中修改 outer scalar state，应使用明确 reduce/scan/summary；
- VLA 不取得 kernel 的 outer traversal 或 ABI。

## `W.axis`

`W.axis(extent, offset=0)` 创建一条有唯一身份的逻辑轴。把轴组成 tuple 就得到 engine value 的
逻辑域：

```python
mi = W.axis(BM)
ni = W.axis(BN)
tile = W.zeros((mi, ni), dtype=W.f32)
```

切片 `mi[:, None]` 和 `ni[None, :]` 只表达广播与轴对应。相同 extent 但身份不同的两条轴不会
自动合并；dot/matmul 的 reduction axis 必须由 operand 显式共享。

## Pointwise 传播

Pointwise operation 保持逻辑元素映射。两个 engine value 只有在轴身份与 broadcast 关系兼容
时才能组合。一个 consumer 的物理需求不能改写 producer 的逻辑域；若多 consumer 需要不同
机器形态，target 在共享、转换、reload 或 rematerialize 中选择。

## Pointer relation

地址表达式把 pointer 对逻辑轴的变化写进 IR：

```text
base + i                    unit stride candidate
base + i * stride           strided candidate
base + index[i]             indexed/gather candidate
interleaved field relation  segment candidate
```

Memory form 是逻辑轴映射与 pointer relation 的共同结果，不是 source annotation。Alignment、
access、alias 和 storage class 来自 pointer type。

## Load、store 与 validity

`W.load(ptr, where, other)` 产生与 pointer footprint 相同的逻辑值。若 `where` 不是全真且没有
`other`，结果携带 validity；带 validity 的值必须先由显式 fill/select 闭合，不能直接 store 或
进入不接受 masked input 的命令。

`W.store(ptr, value, where)` 要求 value 与 pointer domain 相容，并把 effect 保留在 canonical
IR。Target 可选择 unit/strided/indexed/segment 指令，但不能扩大可见写集合。

## `W.transfer`

`W.transfer(source, destination, alignment=...)` 是可组合的数据移动命令。当前语义严格限定为：

- source 可读、destination 可写；
- element type、逻辑 shape 与轴身份相同；
- 全部逻辑元素有效；
- 等价于一次 canonical load 后一次 canonical store。

它不隐含 async、packing、transpose、cache level 或 storage allocation。需要这些可观察算法结构
时，作者写出相应 pointer/index、workspace 与 loop；primitive-private local pack 可由 target 在
命令内部生成。

## 跨控制流的值

Engine value 与 scalar state 都能作为 `for`/`while` carry 或 `if` merge value。类型、逻辑轴和
dynamic extent identity 必须在控制流边界一致。进入控制流前后的 consumer 变化不能决定某个
structured command 是否可 lowering。

## Canonical 表示

Canonical Kernel IR 使用 `!weft_kernel.block` 与 `!weft_kernel.region` 保存 engine value 的
shape、axis identity 和 element type，使用 `weft_kernel.block_index` 保存 `W.axis`。这些名字是
内部语义 schema，不是旧 source API，也不赋予任何物理 layout。
