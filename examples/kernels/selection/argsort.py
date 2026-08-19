import weft
import weft.language as W


@weft.kernel
def argsort_f32(
    values: W.ptr[W.f32, W.readonly, W.noalias],
    indices: W.ptr[W.u32, W.noalias],
    scratch: W.ptr[W.u32, W.workspace, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    value_row_stride: W.index,
    index_row_stride: W.index,
) -> None:
    W.buffer(scratch, (columns,))
    for row in W.range(row_begin, row_end):
        value_base = row * value_row_stride
        index_base = row * index_row_stride
        W.sort_indices(
            values + value_base,
            indices + index_base,
            scratch,
            columns,
            order="ascending",
            nan="last",
            tie="index_ascending",
        )
