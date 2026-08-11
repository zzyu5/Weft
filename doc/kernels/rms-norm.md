# Normalization Workers

## 两遍 RMSNorm worker

```python
@weft.kernel
def rms_norm_worker(
    x: W.ptr[W.f32, W.readonly, W.noalias],
    y: W.ptr[W.f32, W.writeonly, W.noalias],
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
            W.store(y + row * stride + i, value * scale)
```

作者定义两遍算法与 row loop；compiler 决定每遍的 strip、LMUL、reduction tree 与 memory realization。

## 两遍 LayerNorm worker

LayerNorm在第一遍同一个VLA domain上显式产生两个独立state：

```python
with W.vla(0, cols) as i:
    value = W.load(x + row * stride + i)
    total = W.reduce(value, op="add", identity=W.f32(0.0),
                     order="relaxed", acc_dtype=W.f32)
    total_square = W.reduce(value * value, op="add",
                            identity=W.f32(0.0),
                            order="relaxed", acc_dtype=W.f32)

mean = total / W.cast(cols, W.f32)
variance = total_square / W.cast(cols, W.f32) - mean * mean
inverse = W.rsqrt(variance + eps, math="native")
```

第二遍显式写出 `(value - mean) * inverse`。两个reduce可以共享同一VLA traversal，但仍是
两个独立primitive与physical decision；compiler不能把它们折叠成隐藏的LayerNorm op。
