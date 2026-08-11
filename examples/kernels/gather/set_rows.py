import weft
import weft.language as W


@weft.kernel
def set_rows_f32(
    destination: W.ptr[W.f32, W.writeonly, W.noalias],
    source: W.ptr[W.f32, W.readonly, W.noalias],
    indices: W.ptr[W.u32, W.readonly, W.noalias],
    groups: W.index,
    updates: W.index,
    width: W.index,
    destination_group_stride: W.index,
    destination_row_stride: W.index,
    source_group_stride: W.index,
    source_row_stride: W.index,
) -> None:
    for group in W.range(0, groups):
        for update in W.range(0, updates):
            row = W.cast(
                W.load(indices + update, other=W.u32(0)),
                W.index,
            )
            with W.vla(0, width) as column:
                value = W.load(
                    source
                    + group * source_group_stride
                    + update * source_row_stride
                    + column
                )
                W.store(
                    destination
                    + group * destination_group_stride
                    + row * destination_row_stride
                    + column,
                    value,
                )
