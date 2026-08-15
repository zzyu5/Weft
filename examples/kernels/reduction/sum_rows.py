import weft
import weft.language as W


@weft.kernel
def sum_rows_f32(
    input: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    input_stride: W.index,
) -> None:
    for row in W.range(row_begin, row_end):
        with W.vla(0, columns) as column:
            value = W.load(input + row * input_stride + column)
            total = W.reduce(
                value,
                op="add",
                identity=W.f32(0.0),
                order="relaxed",
                acc_dtype=W.f32,
            )
        W.store(output + row, total)
