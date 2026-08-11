# Online Softmax Summary

## Closed summary state

```python
@W.pure
def softmax_lift(x):
    return W.tuple(x, W.f32(1.0))

@W.pure
def softmax_merge(a, b):
    ma, sa = a
    mb, sb = b
    maximum = W.maximum(ma, mb)
    scaled_sum = sa * W.exp(ma - maximum, math="native") + sb * W.exp(
        mb - maximum, math="native"
    )
    return W.tuple(maximum, scaled_sum)
```

`(maximum, scaled_sum)`、identity与rescale都由source定义。Compiler可以在strip内或strip间
重新组合summary，但不能从一个普通carried loop猜出这个merge。

## Full row softmax

```python
@weft.kernel
def softmax_f32(
    x: W.ptr[W.f32, W.readonly],
    y: W.ptr[W.f32, W.writeonly],
    row_begin: W.index,
    row_end: W.index,
    cols: W.index,
    stride: W.index,
) -> None:
    for row in W.range(row_begin, row_end):
        row_offset = row * stride
        with W.vla(0, cols) as i:
            value = W.load(x + row_offset + i)
            state = W.summary_fold(
                value,
                identity=W.tuple(W.neg_inf(W.f32), W.f32(0.0)),
                lift=softmax_lift,
                merge=softmax_merge,
                finalize=None,
                order="preserve",
            )

        maximum, scaled_sum = state
        with W.vla(0, cols) as i:
            value = W.load(x + row_offset + i)
            probability = W.exp(value - maximum, math="fast") / scaled_sum
            W.store(y + row_offset + i, probability)
```

Row loop、第二遍normalize与store属于algorithm skeleton；summary lowering不能吸收并替换成
一个按`softmax_f32` symbol选择的whole-kernel emitter。
