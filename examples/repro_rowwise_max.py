from __future__ import annotations

import weft
import weft.language as W


@weft.kernel(grid_rank=1)
def rowwise_max(
    x: W.ptr[W.f32],
    out: W.ptr[W.f32],
    rows: W.index,
    cols: W.index,
    stride: W.index,
):
    task = W.task_id(0)
    row = W.expand_dims(W.arange(0, rows), 1)
    col = W.expand_dims(W.arange(0, cols), 0)
    offset = task * rows * stride + row * stride + col
    value = W.load(x + offset, mask=col < cols, other=-3.4028234663852886e38)
    maxima = W.reduce(
        value,
        -3.4028234663852886e38,
        axis=1,
        kind="max",
    )
    out_row = W.arange(0, rows)
    W.store(
        out + task * rows + out_row,
        maxima,
        mask=out_row < rows,
    )


if __name__ == "__main__":
    print(weft.lower_to_mlir(rowwise_max), end="")
