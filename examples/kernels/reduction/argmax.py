import weft
import weft.language as W

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
            state = W.argmax(
                value,
                i,
                tie="lowest_coordinate",
                order="relaxed",
            )
        _, best_index = state
        W.store(indices + row, W.cast(best_index, W.u32))
