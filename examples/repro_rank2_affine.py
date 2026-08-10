from __future__ import annotations

import weft
import weft.language as W


@weft.kernel(grid_rank=1)
def rank2_affine(
    x: W.ptr[W.f32],
    out: W.ptr[W.f32],
    rows: W.index,
    cols: W.index,
    stride: W.index,
    alpha: W.f32,
    beta: W.f32,
):
    task = W.task_id(0)
    row = W.expand_dims(W.arange(0, rows), 1)
    col = W.expand_dims(W.arange(0, cols), 0)
    task_base = task * rows * stride
    offset = task_base + row * stride + col
    valid = col < cols
    value = W.load(x + offset, mask=valid, other=0.0)
    result = value * alpha + beta
    W.store(out + offset, result, mask=valid)


if __name__ == "__main__":
    print(weft.lower_to_mlir(rank2_affine), end="")
