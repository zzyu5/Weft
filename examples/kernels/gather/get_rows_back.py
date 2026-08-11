import weft
import weft.language as W


@weft.kernel
def get_rows_back_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    indices: W.ptr[W.u32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.noalias],
    rows: W.index,
    items: W.index,
    hidden: W.index,
    source_stride: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        with W.vla(0, hidden) as column:
            W.store(output + row * output_stride + column, W.f32(0.0))

    for item in W.range(0, items):
        row = W.cast(W.load(indices + item, other=W.u32(0)), W.index)
        with W.vla(0, hidden) as column:
            destination = output + row * output_stride + column
            previous = W.load(destination)
            update = W.load(source + item * source_stride + column)
            W.store(destination, previous + update)
