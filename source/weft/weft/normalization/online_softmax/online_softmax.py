import weft
import weft.language as W


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


@weft.kernel
def online_softmax_f32(
    x: W.ptr[W.f32, W.readonly],
    y: W.ptr[W.f32, W.writeonly],
    n: W.index,
) -> None:
    with W.vla(0, n) as i:
        value = W.load(x + i)
        state = W.summary_fold(
            value,
            identity=W.tuple(W.neg_inf(W.f32), W.f32(0.0)),
            lift=softmax_lift,
            merge=softmax_merge,
            finalize=None,
            order="preserve",
        )

    maximum, scaled_sum = state
    with W.vla(0, n) as i:
        value = W.load(x + i)
        probability = W.exp(value - maximum, math="native") / scaled_sum
        W.store(y + i, probability)
