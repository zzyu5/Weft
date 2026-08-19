# Storage 与生命周期

Weft 暴露 storage 的所有权和可观察生命周期，不暴露 cache、vector register 或 extension
fragment。任何作者可见内存都由调用者传入；DSL 不执行隐式 heap/stack allocation。

## 四类所有权

### External buffer

普通输入输出，使用 `W.external`（默认）pointer storage class。它在一次 kernel 调用期间有效，
shape 由参数、索引和算法关系使用，不需要 `W.buffer` 声明。

### Worker-local workspace

调用者为当前 worker 提供、在一次调用内可跨循环和多个命令复用的临时内存：

```python
scratch: W.ptr[W.f16, W.workspace, W.noalias]
W.buffer(scratch, shape=(rows, columns))
```

作者决定何时写入、何时读取以及如何分区；target 不得用另一块隐藏 persistent storage 替换它。

### Persistent packed storage

跨调用保留、具有显式格式 identity 的 caller-owned 数据：

```python
packed: W.ptr[W.u8, W.persistent("ggml.q4_0"), W.readonly, W.noalias]
W.buffer(packed, shape=(rows, blocks_per_row, bytes_per_block))
```

格式 identity 是 ABI/数据组织事实，不是后端 route 名。RVV 与 IME realization 可读取同一个
persistent view，但不能静默改写其格式。

### Primitive-private temporary

寄存器临时值、短生命周期 local packing、decode 中间值、pipeline buffer 与 fragment 只属于
一次局部 command realization。它们不可观察，不进入 kernel 参数、source storage 或 persistent
layout。

## `W.buffer`

`W.buffer(pointer, shape=...)` 只能直接出现在 kernel 入口 body，并只能绑定一个 entry
workspace/persistent pointer。它在 canonical IR 中记录：

- rank 与 static/dynamic extent；
- pointer element type、access、alignment、alias 与 storage format；
- 调用者必须提供的 storage contract。

它不返回值、不分配内存，也不表示硬件 buffer。一个 pointer 只能有一个 shape 声明。

## Lifetime

作者通过普通 control 和 use-def 决定 algorithmic lifetime：

- 在 loop 外创建并作为 carry 更新的 accumulator 跨 loop 存活；
- workspace 中写入的数据直到最后一次 read 前保持可见；
- persistent view 的内存跨调用存在，但一次 kernel 中产生的 SSA value 不跨调用；
- VLA region value 只能活在其词法 region；
- primitive-private temporary 不得逃出 command。

Target 可让一个 SSA value 保持在 scalar/vector register、重新加载或临时落栈，但必须保持上述
可观察 lifetime 与 effect。

## Alias 与 alignment

`W.noalias`、`W.restrict`、read/write qualifier 和 alignment 是编译器可用的事实。没有这些事实时，
pipeline、hoist 或 load reuse 必须保守；target 不能靠参数名称或 benchmark runner 私约定推断。

## ABI metadata

最终 header/artifact 必须让调用者知道 workspace/persistent 参数、element type、shape 所需的
dynamic extent、alignment 与 storage format。Primitive-private temporary 不进入 ABI。
