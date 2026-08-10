from __future__ import annotations

import weft
import weft.language as W


@weft.kernel(grid_rank=1)
def rank3_affine(
    x: W.ptr[W.f32],
    out: W.ptr[W.f32],
    dim0: W.index,
    dim1: W.index,
    dim2: W.index,
    stride0: W.index,
    stride2: W.index,
    alpha: W.f32,
    beta: W.f32,
):
    task = W.task_id(0)

    axis0 = W.expand_dims(W.expand_dims(W.arange(0, dim0), 1), 2)
    axis1 = W.expand_dims(W.expand_dims(W.arange(0, dim1), 0), 2)
    axis2 = W.expand_dims(W.expand_dims(W.arange(0, dim2), 0), 0)

    offset = (
        task * dim0 * stride0
        + axis0 * stride0
        + axis2 * stride2
        + axis1
    )
    valid = (axis0 < dim0) & (axis1 < dim1) & (axis2 < dim2)
    value = W.load(x + offset, mask=valid, other=0.0)
    W.store(out + offset, value * alpha + beta, mask=valid)


if __name__ == "__main__":
    print(weft.lower_to_mlir(rank3_affine), end="")
