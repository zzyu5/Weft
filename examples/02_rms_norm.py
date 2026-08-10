import weft
import weft.language as W


@weft.kernel
def rms_norm_worker(
    x: W.ptr[W.f32, W.readonly, W.noalias],
    weight: W.ptr[W.f32, W.readonly],
    y: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    cols: W.index,
    stride: W.index,
    eps: W.f32,
) -> None:
    for row in W.range(row_begin, row_end):
        with W.vla(0, cols) as i:
            value = W.load(x + row * stride + i)
            sum_sq = W.reduce(
                value * value,
                op="add",
                identity=W.f32(0.0),
                order="relaxed",
                acc_dtype=W.f32,
            )

        scale = W.rsqrt(sum_sq / W.cast(cols, W.f32) + eps, math="native")

        with W.vla(0, cols) as i:
            value = W.load(x + row * stride + i)
            w = W.load(weight + i)
            W.store(y + row * stride + i, value * scale * w)
