import weft
import weft.language as W


@W.pure
def softmax_lift(x):
    return W.tuple(x, W.f32(1.0))


@W.pure
def softmax_merge(a, b):
    ma, sa = a
    mb, sb = b
    m = W.maximum(ma, mb)
    s = sa * W.exp(ma - m, math="native") + sb * W.exp(
        mb - m, math="native"
    )
    return W.tuple(m, s)


@weft.kernel
def online_softmax_summary(
    x: W.ptr[W.f32, W.readonly],
    out: W.ptr[W.f32, W.writeonly],
    n: W.index,
    logical_n: W.index,
) -> None:
    with W.vla(0, n) as i:
        active = i < logical_n
        value = W.load(x + i, where=active)
        state = W.summary_fold(
            value,
            identity=W.tuple(W.neg_inf(W.f32), W.f32(0.0)),
            lift=softmax_lift,
            merge=softmax_merge,
            finalize=None,
            order="preserve",
        )
    maximum, scaled_sum = state
    W.store(out, maximum)
    W.store(out + 1, scaled_sum)
