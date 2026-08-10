from __future__ import annotations

import weft
import weft.language as W


@weft.kernel(grid_rank=1)
def rms_norm(
    x: W.ptr[W.f32],
    weight: W.ptr[W.f32],
    out: W.ptr[W.f32],
    cols: W.index,
    stride: W.index,
    BLOCK: W.constexpr,
):
    row = W.task_id(0)
    sum_sq = 0.0

    for base in W.range(0, cols, BLOCK):
        lanes = W.arange(0, BLOCK)
        index = base + lanes
        valid = index < cols
        values = W.load(
            x + row * stride + index,
            mask=valid,
            other=0.0,
        )
        sum_sq += W.sum(values * values, axis=0)

    scale = W.rsqrt(sum_sq / W.cast(cols, W.f32))

    for base in W.range(0, cols, BLOCK):
        lanes = W.arange(0, BLOCK)
        index = base + lanes
        valid = index < cols
        values = W.load(
            x + row * stride + index,
            mask=valid,
            other=0.0,
        )
        weights = W.load(weight + index, mask=valid, other=0.0)
        W.store(
            out + row * stride + index,
            values * scale * weights,
            mask=valid,
        )


if __name__ == "__main__":
    print(weft.lower_to_mlir(rms_norm), end="")
