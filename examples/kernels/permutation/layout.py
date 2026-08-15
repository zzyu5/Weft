import weft
import weft.language as W


@weft.kernel
def copy_f32(
    input: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        W.store(output + i, W.load(input + i))


@weft.kernel
def repeat_rows_f32(
    input: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(row_begin, row_end):
        with W.vla(0, columns) as column:
            W.store(
                output + row * output_stride + column,
                W.load(input + column),
            )


@weft.kernel
def concat_rows_f32(
    lhs: W.ptr[W.f32, W.readonly, W.noalias],
    rhs: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    lhs_rows: W.index,
    columns: W.index,
) -> None:
    for row in W.range(row_begin, row_end):
        source = lhs
        source_row = row
        if row >= lhs_rows:
            source = rhs
            source_row = row - lhs_rows
        with W.vla(0, columns) as column:
            W.store(
                output + row * columns + column,
                W.load(source + source_row * columns + column),
            )


@weft.kernel
def concat_columns_f32(
    lhs: W.ptr[W.f32, W.readonly, W.noalias],
    rhs: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    lhs_columns: W.index,
    rhs_columns: W.index,
) -> None:
    output_columns = lhs_columns + rhs_columns
    for row in W.range(row_begin, row_end):
        with W.vla(0, lhs_columns) as column:
            W.store(
                output + row * output_columns + column,
                W.load(lhs + row * lhs_columns + column),
            )
        with W.vla(0, rhs_columns) as column:
            W.store(
                output + row * output_columns + lhs_columns + column,
                W.load(rhs + row * rhs_columns + column),
            )
