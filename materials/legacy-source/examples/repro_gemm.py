from __future__ import annotations

import weft
import weft.language as W


@weft.kernel(grid_rank=2)
def gemm(
    lhs: W.ptr[W.f32],
    rhs: W.ptr[W.f32],
    out: W.ptr[W.f32],
    m: W.index,
    n: W.index,
    k: W.index,
    BM: W.constexpr,
    BN: W.constexpr,
    BK: W.constexpr,
):
    task_m = W.task_id(0)
    task_n = W.task_id(1)
    row = task_m * BM + W.arange(0, BM)
    col = task_n * BN + W.arange(0, BN)
    out_row = W.expand_dims(row, 1)
    out_col = W.expand_dims(col, 0)
    out_offset = out_row * n + out_col
    accumulator = W.zeros_like(out_offset, W.f32)

    for k0 in W.range(0, k, BK):
        inner = k0 + W.arange(0, BK)
        lhs_row = W.expand_dims(row, 1)
        lhs_inner = W.expand_dims(inner, 0)
        lhs_offset = lhs_row * k + lhs_inner
        lhs_value = W.load(
            lhs + lhs_offset,
            mask=(lhs_row < m) & (lhs_inner < k),
            other=0.0,
        )

        rhs_inner = W.expand_dims(inner, 1)
        rhs_col = W.expand_dims(col, 0)
        rhs_offset = rhs_inner * n + rhs_col
        rhs_value = W.load(
            rhs + rhs_offset,
            mask=(rhs_inner < k) & (rhs_col < n),
            other=0.0,
        )
        accumulator = W.dot(lhs_value, rhs_value, accumulator)

    W.store(
        out + out_offset,
        accumulator,
        mask=(out_row < m) & (out_col < n),
    )


if __name__ == "__main__":
    print(weft.lower_to_mlir(gemm), end="")
