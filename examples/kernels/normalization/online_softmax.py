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
