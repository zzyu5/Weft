import weft
import weft.language as W


@weft.kernel
def layer_norm_worker(
    x: W.ptr[W.f32, W.readonly, W.noalias],
    y: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    cols: W.index,
    stride: W.index,
    eps: W.f32,
) -> None:
    for row in W.range(row_begin, row_end):
        row_offset = row * stride
        with W.vla(0, cols) as i:
            value = W.load(x + row_offset + i)
            total = W.reduce(
                value,
                op="add",
                identity=W.f32(0.0),
                order="relaxed",
                acc_dtype=W.f32,
            )
            sum_sq = W.reduce(
                value * value,
                op="add",
                identity=W.f32(0.0),
                order="relaxed",
                acc_dtype=W.f32,
            )

        count = W.cast(cols, W.f32)
        mean = total / count
        variance = sum_sq / count - mean * mean
        scale = W.rsqrt(variance + eps, math="native")

        with W.vla(0, cols) as i:
            value = W.load(x + row_offset + i)
            W.store(y + row_offset + i, (value - mean) * scale)
