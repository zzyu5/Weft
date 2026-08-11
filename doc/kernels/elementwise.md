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

## Mixed F16/F32 SwiGLU

```python
with W.vla(begin, end) as i:
    gate_f32 = W.cast(W.load(gate_f16 + i), W.f32)
    activated = gate_f32 / (
        W.f32(1.0) + W.exp(-gate_f32, math="fast")
    )
    up_f32 = W.cast(W.load(up_f16 + i), W.f32)
    W.store(output_f32 + i, activated * up_f32)
```

两个F16 load、显式F32 cast、source声明的gate→activation→up顺序与F32 output都是算法事实。
Target分别为memory、cast、exp和arithmetic选择RVV realization；dtype不能从pointer use或
kernel名推断。
