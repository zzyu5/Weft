from __future__ import annotations

import weft
import weft.language as W


@weft.kernel(grid_rank=1)
def vector_dot(
    x: W.ptr[W.f32],
    y: W.ptr[W.f32],
    out: W.ptr[W.f32],
    count: W.index,
):
    task = W.task_id(0)
    lanes = W.arange(0, count)
    offset = task * count + lanes
    valid = lanes < count
    x_value = W.load(x + offset, mask=valid, other=0.0)
    y_value = W.load(y + offset, mask=valid, other=0.0)
    result = W.dot(x_value, y_value)
    W.store(out + task, result, mask=True)


if __name__ == "__main__":
    print(weft.lower_to_mlir(vector_dot), end="")
