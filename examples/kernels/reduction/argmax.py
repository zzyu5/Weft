import weft
import weft.language as W


@W.pure
def argmax_lift(value, index):
    return W.tuple(value, index)


@W.pure
def argmax_merge(a, b):
    value_a, index_a = a
    value_b, index_b = b
    take_b = (value_b > value_a) | (
        (value_b == value_a) & (index_b < index_a)
    )
    return W.tuple(
        W.select(take_b, value_b, value_a),
        W.select(take_b, index_b, index_a),
    )


@weft.kernel
def argmax_f32(
    x: W.ptr[W.f32, W.readonly, W.noalias],
    indices: W.ptr[W.u32, W.writeonly, W.noalias],
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
                identity=W.tuple(W.neg_inf(W.f32), W.index(0)),
                lift=argmax_lift,
                merge=argmax_merge,
                coordinate=i,
                order="relaxed",
            )
        _, best_index = state
        W.store(indices + row, W.cast(best_index, W.u32))
