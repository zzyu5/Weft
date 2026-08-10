from __future__ import annotations

import weft
import weft.language as W


@weft.kernel(grid_rank=1)
def columnwise_sum(
    x: W.ptr[W.f32],
    out: W.ptr[W.f32],
    rows: W.index,
    cols: W.index,
    stride: W.index,
):
    task = W.task_id(0)
    row = W.expand_dims(W.arange(0, rows), 1)
    col = W.expand_dims(W.arange(0, cols), 0)
    task_base = task * rows * stride
    offset = task_base + row * stride + col
    value = W.load(
        x + offset,
        mask=(row < rows) & (col < cols),
        other=0.0,
    )
    sums = W.sum(value, axis=0)

    out_col = W.arange(0, cols)
    out_index = task * cols + out_col
    W.store(out + out_index, sums, mask=out_col < cols)


if __name__ == "__main__":
    print(weft.lower_to_mlir(columnwise_sum), end="")
