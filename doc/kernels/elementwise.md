# Elementwise Kernel

## VLA elementwise

```python
import weft
import weft.language as W

@weft.kernel
def add_bias(
    x: W.ptr[W.f32, W.readonly, W.noalias],
    bias: W.ptr[W.f32, W.readonly, W.noalias],
    y: W.ptr[W.f32, W.writeonly, W.noalias],
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        xv = W.load(x + i)
        bv = W.load(bias + i)
        W.store(y + i, xv + bv)
```

外部 runtime 决定每个 worker 的 `[begin,end)`。

## SiLU 与 local math realization

```python
@weft.kernel
def silu_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    destination: W.ptr[W.f32, W.writeonly, W.noalias],
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        value = W.load(source + i)
        denominator = W.f32(1.0) + W.exp(-value, math="fast")
        W.store(destination + i, value / denominator)
```

SiLU公式与 `math="fast"` 是source semantics。Target决定dynamic `vl`、LMUL、register reuse与
exp realization；它不能根据entry名把任意pointwise graph替换成SiLU模板。
