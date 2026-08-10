# Online Softmax Summary

### 23.4 Online softmax summary

```python
@W.pure
def softmax_lift(x):
    return W.tuple(x, W.f32(1.0))

@W.pure
def softmax_merge(a, b):
    ma, sa = a
    mb, sb = b
    m = W.maximum(ma, mb)
    s = sa * W.exp(ma - m, math="native") + sb * W.exp(mb - m, math="native")
    return W.tuple(m, s)

with W.vla(0, n) as i:
    valid = i < logical_n
    x = W.load(ptr + i, where=valid)
    state = W.summary_fold(
        x,
        identity=W.tuple(W.neg_inf(W.f32), W.f32(0.0)),
        lift=softmax_lift,
        merge=softmax_merge,
        finalize=None,
        order="preserve",
    )
```

编译器可以为每个 strip 形成局部 summary，再用 `softmax_merge` 合并。Rescale 是 merge 语义的一部分，不依赖源码中手写的 strip loop。
