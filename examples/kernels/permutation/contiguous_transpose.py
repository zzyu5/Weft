import weft
import weft.language as W


@weft.kernel
def transpose_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    destination: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    columns: W.index,
    source_stride: W.index,
    destination_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        with W.vla(0, columns) as column:
            value = W.load(source + row * source_stride + column)
            W.store(destination + column * destination_stride + row, value)
