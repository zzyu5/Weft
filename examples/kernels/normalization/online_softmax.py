import weft
import weft.language as W

@weft.kernel
def softmax_f32(
    x: W.ptr[W.f32, W.readonly, W.noalias],
    y: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    cols: W.index,
    stride: W.index,
) -> None:
    for row in W.range(row_begin, row_end):
        row_offset = row * stride
        with W.vla(0, cols) as i:
            value = W.load(x + row_offset + i)
            state = W.online_softmax_summary(
                value,
                math="native",
                order="preserve",
            )

        maximum, scaled_sum = state
        with W.vla(0, cols) as i:
            value = W.load(x + row_offset + i)
            probability = W.exp(value - maximum, math="fast") / scaled_sum
            W.store(y + row_offset + i, probability)
