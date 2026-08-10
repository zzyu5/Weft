from __future__ import annotations

import weft
import weft.language as W


@weft.kernel(grid_rank=1)
def mixed_axis_sums(
    row_x: W.ptr[W.f32],
    row_out: W.ptr[W.f32],
    col_x: W.ptr[W.f32],
    col_out: W.ptr[W.f32],
    rows: W.index,
    cols: W.index,
    stride: W.index,
):
    task = W.task_id(0)

    row0 = W.expand_dims(W.arange(0, rows), 1)
    col0 = W.expand_dims(W.arange(0, cols), 0)
    offset0 = task * rows * stride + row0 * stride + col0
    value0 = W.load(row_x + offset0, mask=col0 < cols, other=0.0)
    row_sums = W.sum(value0, axis=1)
    row_index = W.arange(0, rows)
    W.store(
        row_out + task * rows + row_index,
        row_sums,
        mask=row_index < rows,
    )

    row1 = W.expand_dims(W.arange(0, rows), 1)
    col1 = W.expand_dims(W.arange(0, cols), 0)
    offset1 = task * rows * stride + row1 * stride + col1
    value1 = W.load(
        col_x + offset1,
        mask=(row1 < rows) & (col1 < cols),
        other=0.0,
    )
    col_sums = W.sum(value1, axis=0)
    col_index = W.arange(0, cols)
    W.store(
        col_out + task * cols + col_index,
        col_sums,
        mask=col_index < cols,
    )


if __name__ == "__main__":
    print(weft.lower_to_mlir(mixed_axis_sums), end="")
