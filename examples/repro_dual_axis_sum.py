from __future__ import annotations

import weft
import weft.language as W


@weft.kernel(grid_rank=1)
def dual_axis_sum(
    row_out: W.ptr[W.f32],
    col_out: W.ptr[W.f32],
    rows: W.index,
    cols: W.index,
    value: W.f32,
):
    task = W.task_id(0)
    row = W.expand_dims(W.arange(0, rows), 1)
    col = W.expand_dims(W.arange(0, cols), 0)
    tile = W.full_like(row + col, value)

    row_sums = W.sum(tile, axis=1)
    col_sums = W.sum(tile, axis=0)

    out_row = W.arange(0, rows)
    row_index = task * rows + out_row
    W.store(row_out + row_index, row_sums, mask=out_row < rows)

    out_col = W.arange(0, cols)
    col_index = task * cols + out_col
    W.store(col_out + col_index, col_sums, mask=out_col < cols)


if __name__ == "__main__":
    print(weft.lower_to_mlir(dual_axis_sum), end="")
