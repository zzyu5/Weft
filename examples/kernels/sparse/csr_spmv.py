import weft
import weft.language as W


@weft.kernel
def csr_spmv_f32(
    row_offsets: W.ptr[W.u32, W.readonly, W.noalias],
    column_indices: W.ptr[W.u32, W.readonly, W.noalias],
    values: W.ptr[W.f32, W.readonly, W.noalias],
    vector: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
) -> None:
    for row in W.range(row_begin, row_end):
        begin = W.cast(W.load(row_offsets + row, other=W.u32(0)), W.index)
        end = W.cast(W.load(row_offsets + row + 1, other=W.u32(0)), W.index)
        with W.vla(begin, end) as entry:
            column = W.cast(
                W.load(column_indices + entry, other=W.u32(0)), W.index
            )
            matrix_value = W.load(values + entry, other=W.f32(0.0))
            vector_value = W.load(vector + column, other=W.f32(0.0))
            total = W.reduce(
                matrix_value * vector_value,
                op="add",
                identity=W.f32(0.0),
                axis=None,
                order="relaxed",
                acc_dtype=W.f32,
            )
        W.store(output + row, total)
