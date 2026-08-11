import weft
import weft.language as W


@weft.kernel
def cumsum_f32(
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
            prefix = W.scan(
                value,
                op="add",
                identity=W.f32(0.0),
                inclusive=True,
                order="ordered",
                acc_dtype=W.f32,
            )
            W.store(y + row_offset + i, prefix)
