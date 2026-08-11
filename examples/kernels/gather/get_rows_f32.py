import weft
import weft.language as W


@weft.kernel
def get_rows_f32(
    table: W.ptr[W.f32, W.readonly, W.noalias],
    indices: W.ptr[W.u32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    token_begin: W.index,
    token_end: W.index,
    hidden: W.index,
    table_stride: W.index,
    output_stride: W.index,
) -> None:
    for token in W.range(token_begin, token_end):
        row = W.cast(W.load(indices + token, other=W.u32(0)), W.index)
        with W.vla(0, hidden) as i:
            value = W.load(table + row * table_stride + i)
            W.store(output + token * output_stride + i, value)
