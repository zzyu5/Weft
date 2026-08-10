from __future__ import annotations

import weft
import weft.language as W


@weft.kernel(grid_rank=1)
def dual_axis_load_sum(
    x: W.ptr[W.f32],
    row_out: W.ptr[W.f32],
    col_out: W.ptr[W.f32],
    rows: W.index,
    cols: W.index,
    stride: W.index,
):
    task = W.task_id(0)
    row = W.expand_dims(W.arange(0, rows), 1)
    col = W.expand_dims(W.arange(0, cols), 0)
    offset = task * rows * stride + row * stride + col
    tile = W.load(
        x + offset,
        mask=(row < rows) & (col < cols),
        other=0.0,
    )

    row_sums = W.sum(tile, axis=1)
    col_sums = W.sum(tile, axis=0)

    out_row = W.arange(0, rows)
    W.store(
        row_out + task * rows + out_row,
        row_sums,
        mask=out_row < rows,
    )
    out_col = W.arange(0, cols)
    W.store(
        col_out + task * cols + out_col,
        col_sums,
        mask=out_col < cols,
    )


if __name__ == "__main__":
    print(weft.lower_to_mlir(dual_axis_load_sum), end="")
