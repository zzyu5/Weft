from __future__ import annotations

import weft
import weft.language as W


@weft.kernel(grid_rank=1)
def contract(
    lhs: W.ptr[W.f32],
    rhs: W.ptr[W.f32],
    out: W.ptr[W.f32],
    m: W.index,
    n: W.index,
    k: W.index,
):
    task = W.task_id(0)

    lhs_row = W.expand_dims(W.arange(0, m), 1)
    lhs_k = W.expand_dims(W.arange(0, k), 0)
    lhs_offset = task * m * k + lhs_row * k + lhs_k
    lhs_value = W.load(
        lhs + lhs_offset,
        mask=(lhs_row < m) & (lhs_k < k),
        other=0.0,
    )

    rhs_k = W.expand_dims(W.arange(0, k), 1)
    rhs_col = W.expand_dims(W.arange(0, n), 0)
    rhs_offset = task * k * n + rhs_k * n + rhs_col
    rhs_value = W.load(
        rhs + rhs_offset,
        mask=(rhs_k < k) & (rhs_col < n),
        other=0.0,
    )

    result = W.contract(
        lhs_value,
        rhs_value,
        0.0,
        lhs_axes=(-1,),
        rhs_axes=(0,),
    )
    out_row = W.expand_dims(W.arange(0, m), 1)
    out_col = W.expand_dims(W.arange(0, n), 0)
    out_offset = task * m * n + out_row * n + out_col
    W.store(
        out + out_offset,
        result,
        mask=(out_row < m) & (out_col < n),
    )


if __name__ == "__main__":
    print(weft.lower_to_mlir(contract), end="")
