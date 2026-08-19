# Python DSL

Python 只用于构造 AOT Weft kernel。`@weft.kernel` 函数不会按普通 Python 执行；frontend 从
函数 AST 生成 canonical Kernel IR，未知语法或未闭合能力直接报错。

## Kernel 与参数

```python
import weft
import weft.language as W

@weft.kernel
def add(
    x: W.ptr[W.f32, W.readonly, W.noalias],
    y: W.ptr[W.f32, W.readonly, W.noalias],
    out: W.ptr[W.f32, W.writeonly, W.noalias],
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        W.store(out + i, W.load(x + i) + W.load(y + i))
```

参数必须使用正式 scalar、pointer 或 `constexpr` annotation。Pointer qualifier 描述 access、
alignment、alias 与 storage class，不描述 cache 或寄存器层级。

## 有序控制

普通 `if`、`while` 和以下两种 `for` 都由当前 worker 顺序执行：

```python
for i in W.range(begin, end, step):
    ...

for block_begin in W.blocks(begin, end, block_size):
    ...
```

`W.range` 表示一般有序遍历；`W.blocks` 表示作者选择的 algorithmic/cache block traversal。
二者当前都 lower 为真实 scalar loop，区别保存在 canonical IR，供 target reasoning 使用。

作者可以授权现有循环做局部软件流水：

```python
for k0 in W.pipeline(W.blocks(0, k, BK)):
    loaded = W.load(...)
    acc = W.gemm(..., init=acc, acc_dtype=W.f32)
```

`W.pipeline` 不指定 buffer 数、stage 或 prefetch distance；顺序执行始终是合法 realization。
它不能包裹 `while`，也不能创建作者没有写出的 staging 或 workspace。

## 显式 VLA 域

```python
with W.vla(begin, end) as i:
    value = W.load(input + i)
    W.store(output + i, value)
```

`W.vla` 是作者对一条运行时长度逻辑轴的 SIMD 授权。VLA region 不能嵌套，region value 不能
逃出词法作用域，普通 scalar loop 不会自动转换为 VLA。

## 逻辑轴与 engine value

```python
mi = W.axis(BM)
ki = W.axis(BK)
a_block = W.load(a + (m0 + mi[:, None]) * lda + (k0 + ki[None, :]))
acc = W.accumulator((mi, ni), W.f32, init=0.0)
```

`W.axis(extent, offset=0)` 创建有身份的逻辑轴。轴切片只构造广播关系，不指定 SIMD lane、
register 或 fragment。`W.full`、`W.zeros` 和 `W.accumulator` 创建普通 engine SSA value；
`W.accumulator` 只是把“这个输出域将被循环 carry”写得自然，canonical 语义仍是普通初始值。

## 数据移动

```python
value = W.load(ptr, where=predicate, other=fill, alignment=16)
W.store(ptr, value, where=predicate, alignment=16)
W.transfer(source_ptr, destination_ptr, alignment=16)
```

`W.transfer` 表示同一逻辑域、同一 element type 的全有效 source-to-destination copy，并直接
展开为 canonical load/store。它不表示异步 copy，不分配 storage，也不改变 layout。

## Buffer 合同

Workspace 或 persistent pointer 必须在入口显式绑定 shape：

```python
W.buffer(scratch, shape=(rows, columns))
```

这不是分配。调用者仍提供真实内存；详见
[Storage 与生命周期](storage-and-lifetime.md)。

## 局部命令

公开命令分为：

- collective：`W.vdot`、`W.gemm`、`W.reduce`、`W.scan`、`W.argmax`、summary、sort；
- data/transform：load、store、transfer、cast、select、lookup、decode、pointwise；
- typed packed compute：`W.quant.*`。

命令结果是普通 SSA value，可多 use、参与 pointwise、跨有序控制 carry 或进入另一个合法命令。

## Helper

`@W.pure` 定义纯 helper；`@W.helper(effects=("read", "write"))` 显式列出 effect。Helper 在
frontend 内联，只减少源码重复，不形成函数 ABI、第二份 IR 或特殊 lowering。Helper 的展开与
手写等价程序必须产生同类语义事实。

## 禁止的 Python 行为

Frontend 不接受运行时反射、动态对象、异常处理、未知 `weft_kernel.*` op 或任意 Python
library call。未实现构造直接报告 unsupported；不会执行 Python fallback 或生成假 artifact。
