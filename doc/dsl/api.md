# Python DSL API 索引

本文件只汇总 source surface；各构造的规范语义由前述 DSL 文档定义。

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
integer `&`, `|`, `^`, `<<`, `>>`
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
W.bitcast(value, dtype)
```

语义不同的扩展 primitive 进入独立 namespace 或 extension dialect，但必须保持局部、typed、可组合。

---
