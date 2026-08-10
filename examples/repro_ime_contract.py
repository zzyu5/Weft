from __future__ import annotations

import weft
import weft.language as W


@weft.kernel(grid_rank=2)
def ime_contract(
    lhs: W.ptr[W.i8],
    rhs_t: W.ptr[W.i8],
    out: W.ptr[W.i32],
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
    inner = W.arange(0, BK)

    lhs_row = W.expand_dims(row, 1)
    lhs_inner = W.expand_dims(inner, 0)
    lhs_offset = lhs_row * k + lhs_inner
    lhs_value = W.load(
        lhs + lhs_offset,
        mask=(lhs_row < m) & (lhs_inner < k),
        other=W.cast(0, W.i8),
    )

    rhs_col = W.expand_dims(col, 1)
    rhs_inner = W.expand_dims(inner, 0)
    rhs_offset = rhs_col * k + rhs_inner
    rhs_value = W.load(
        rhs_t + rhs_offset,
        mask=(rhs_col < n) & (rhs_inner < k),
        other=W.cast(0, W.i8),
    )

    out_row = W.expand_dims(row, 1)
    out_col = W.expand_dims(col, 0)
    out_offset = out_row * n + out_col
    accumulator = W.zeros_like(out_offset, W.i32)
    result = W.contract(
        lhs_value,
        rhs_value,
        accumulator,
        lhs_axes=(1,),
        rhs_axes=(1,),
    )
    W.store(
        out + out_offset,
        result,
        mask=(out_row < m) & (out_col < n),
    )


if __name__ == "__main__":
    print(weft.lower_to_mlir(ime_contract), end="")
