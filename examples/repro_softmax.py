from __future__ import annotations

import weft
import weft.language as W


@weft.kernel(grid_rank=1)
def softmax(
    x: W.ptr[W.f32],
    out: W.ptr[W.f32],
    cols: W.index,
    stride: W.index,
    BLOCK: W.constexpr,
):
    row = W.task_id(0)
    row_max = -3.4028234663852886e38

    for base in W.range(0, cols, BLOCK):
        lanes = W.arange(0, BLOCK)
        index = base + lanes
        valid = index < cols
        values = W.load(
            x + row * stride + index,
            mask=valid,
            other=-3.4028234663852886e38,
        )
        row_max = W.maximum(
            row_max,
            W.max(values, -3.4028234663852886e38, axis=0),
        )

    row_sum = 0.0
    for base in W.range(0, cols, BLOCK):
        lanes = W.arange(0, BLOCK)
        index = base + lanes
        valid = index < cols
        values = W.load(
            x + row * stride + index,
            mask=valid,
            other=-3.4028234663852886e38,
        )
        exponentials = W.exp(values - row_max)
        row_sum += W.sum(exponentials, axis=0)

    for base in W.range(0, cols, BLOCK):
        lanes = W.arange(0, BLOCK)
        index = base + lanes
        valid = index < cols
        values = W.load(x + row * stride + index, mask=valid, other=0.0)
        exponentials = W.exp(values - row_max)
        W.store(
            out + row * stride + index,
            exponentials / row_sum,
            mask=valid,
        )


if __name__ == "__main__":
    print(weft.lower_to_mlir(softmax), end="")
