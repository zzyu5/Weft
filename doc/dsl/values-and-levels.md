# Value、Level 与有序控制

## 1. Value

Value 是带 element type、逻辑 shape 和 axis identity 的局部 SSA 值：

```text
Value[element, shape, axes]
```

Value 没有 physical owner、物理 lane、LMUL、register group、fragment 或 hardware storage scope。它仍有由 SSA dominance 与 Level births/handoff 明确规定的逻辑作用域。标量是 rank-0 Value；正 rank Value 的 axis identity 是编译器理解 pointwise、broadcast、reduce、contract 和 memory relation 的依据。

Value 可以：

- 被多个 consumer 使用；
- 进入 pointwise、reduce、contract、lookup 或 memory operation；
- 作为 Level state 跨迭代更新；
- 通过 handoff 交回外层；
- 在普通 `for/while/if` 中作为 SSA carry；
- 被 materialize 为本层后代共享的 staged value。

structured operation 的结果仍是普通 Value，不形成只能 direct-store 的封闭路径。

## 2. 四个生命周期构造

### 2.1 `new`

```python
acc = new(f32, [MR, NR], init=0)
```

`new` 在当前逻辑作用域诞生一个可更新 state；该作用域可以是 kernel root，也可以是 Level。位于 Level 中时，它是该 Level 的 `births.state`，可被内层 Level 或普通控制更新，并在本层 handoff 时交回。Top-K 的 heap/indices 则是 kernel-root state。

`init` 是数值语义。`init=0`、`init=admit(C[...])` 和未初始化不是可互换实现。

### 2.2 `materialize`

```python
Bp = materialize(admit(B[kc, nc]))
```

`materialize` 在当前逻辑作用域形成一次 staged Value，后代只读复用；位于 Level 中时，它是该 Level 的 `births.staged`。staged Value 的 element type、logical shape、axes 与 Level 归属进入 canonical IR；它不表示某个具体 cache、栈、vector register、fragment、shared memory 或 packed panel。target 可以在保持这些 canonical facts 的前提下，为该 Value 选择原样引用、register window、local-storage panel、RVV tuple、IME operand 或其它 physical representation。

### 2.3 `admit`

```python
x = admit(X[kb])
```

`admit` 把一个 View region 在当前 kernel/Level 作用域供应一次。把同一 `admit` 移到更内层会增加逻辑供应次数；移到更外层会扩大 Value 生命周期。两者是不同作者程序。

### 2.4 `commit`

```python
commit(acc, C[mb, nb])
```

`commit` 将 Value 写回指定 View region。写回位置、覆盖域和 effect 顺序属于源程序。

## 3. Level

Level 表达一段层级数值合成程序。它至少包含：

```text
domain
    当前 Level 覆盖的逻辑域及 axis identity

partition
    当前实例与父域的划分关系，包括 tail

multiplicity
    每个父实例中有多少个当前实例

births.state
    在本层由 new 诞生的状态

births.staged
    在本层由 materialize 形成、供后代复用的值

body
    可以包含多棵数值子树、普通控制、多个 state update 和 effect

handoff
    body 的结果如何成为外层可见的 SSA value
```

Level 相比普通 `for` 的核心新增语义是 typed lifetime、层归属和 handoff；domain partition、multiplicity 与 tail 使这一层的逻辑覆盖范围可被精确保存。

Level 不是“一层一个 reduce”，也不是 target loop 的别名。同一 Level 可以包含两个 contract、若干 pointwise、多个状态和 memory effect。

## 4. Domain 与 partition

常见 domain 构造：

```python
with L.tiles(N, extent=NC) as nc:
with L.rows(M, group=MR) as mb:
with L.cols(N, group=NR) as nb:
with L.blocks(K, extent=KB) as kb:
with L.subs(extent=32) as s:
```

这些名字描述逻辑 relation，不指定 lane/thread/fragment。

`extent` 或 `group` 定义当前实例覆盖的逻辑宽度；`multiplicity` 由父域范围和 partition 得出；不能整除时，tail 仍属于同一逻辑 domain。

编译器可以 strip-mine 一个 Level 的物理执行、分多次 issue 或使用 tail mask，但不能把一个 source Level 改成多个独立 source Level，因为 births、admit 和 handoff 的次数会变化。

## 5. 逻辑 cohort

```python
with L.rows(M, group=16) as mb:
    acc = new(f32, [16], init=0)
```

一个 Level instance 逻辑上共同产生 16 个输出。这 16 个输出：

- 可以共同消费在本层发生的一次 `admit` 或 birth；
- 共享 births/staged scope；
- 具有同一 handoff 边界。

`group=16` 不表示 16 个输出必须同时驻留在 16 个物理寄存器中。目标编译器可以把 cohort 分成多次 issue、spill/reload 或多个 fragment；但它不能把源程序改写成四个 `group=4` Level。

## 6. Handoff

### 6.1 Level handoff

表面语法使用普通 SSA 赋值：

```python
acc += block_term
m = m_new
o = o * alpha + contract(p, v, over="tk")
```

在 canonical Kernel IR 中，Level body 的终结必须保存每个 carried/state result 的名称、类型和来源。作者不需要学习独立 handoff 代数，但层归属不能从 IR 中消失。

handoff 不限于 `+=`。attention 中：

```python
o = o * alpha + contribution
```

先重标定旧 state，再加入当前 block，是一个可观察的非归约 handoff。

### 6.2 跨 physical engine 的 value use

```python
def local_decode(
    packed: Value[u4, (K,), ("k",)]
) -> Value[i8, (K,), ("k",)]:
    return i8(packed) - i8(8)

panel = materialize(local_decode(packed))
acc += contract(panel, x, over="k")
```

`local_decode` 是保持 K axis 的普通 helper。`panel` 的 materialized birth 和普通 SSA use-def 已完整表达 source-visible value boundary；core DSL 不再增加 `stage_handoff` 或 source engine annotation。

若 target 让 producer 与 consumer 使用不同 physical engine，它在具体 use edge 上插入 register/fragment/local-storage conversion，并承担相应 ordering 与 resources。若同步、并发或 workspace 会改变 source-visible effects，则必须定义真实的 effect operation 或另一棵作者程序，不能由 engine convention 暗中补出。

## 7. 普通控制流

```python
for i in range(N):
    if predicate:
        ...

while condition:
    ...
```

普通 `for/while/if` 是有序标量控制：

- 迭代按程序顺序发生；
- 编译器不把迭代自动改成 shaped axis 或 SIMD domain；
- loop/branch carry 使用普通 SSA；
- effect、alias 和依赖顺序必须保留；
- 常量折叠、地址提升等不改变逻辑值集合的优化仍合法。

Top-K、heap update、依赖前一迭代的搜索和 CSR traversal 可以只用普通控制，不必使用 Level。

## 8. `iota` 与显式逻辑轴

```python
lane = iota(8, dtype=u32)
within_k = iota(32, dtype=u32, axis="k")
```

未指定 `axis` 时，`iota` 建立一条新的 logical axis。`axis="k"` 则让坐标值沿已经存在的 K axis 排列；它用于显式描述一个 K block 内的 field gather、bit-plane 组合或 lookup，不创建第二条同 extent 轴。两种形式都产生 shaped logical index Value，后续 gather、lookup、pointwise 与 reduce 按其 axis identity 组合。

它与下面程序不同：

```python
for lane in range(8):
    value0 = lookup(table, index0)
    ...
```

八次 Python/DSL 标量展开只产生八组独立 SSA operation；不存在一个 8-element logical axis。编译器不得根据相同 op count、同构 source closure 或相邻语句重新发明轴。

若算法要求 codebook 的 8 个元素成为一个 shaped value，作者必须用 `iota`、shaped indexing 或返回 shaped Value 的函数显式写出。

## 9. 顺序性

顺序性来自 ordinary control、SSA use-def 和 operation effect，不使用通用 `ordered=True` 权限位。

例如 online softmax：

```text
m_new depends on m
alpha depends on old m and m_new
l_new depends on old l and alpha
o_new depends on old o, alpha and current contribution
```

这些依赖已经唯一规定合法顺序。若一个 reduce/scan primitive需要额外结合或顺序语义，它必须写入该 primitive 的定义，而不是成为全局 hint。
