import weft
import weft.language as W


@weft.kernel
def rms_norm_backward_f32(
    gradient: W.ptr[W.f32, W.readonly, W.noalias],
    source: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    row_stride: W.index,
    eps: W.f32,
) -> None:
    count = W.cast(columns, W.f32)
    for row in W.range(row_begin, row_end):
        with W.vla(0, columns) as column:
            x = W.load(source + row * row_stride + column)
            dz = W.load(gradient + row * row_stride + column)
            sum_xx = W.reduce(
                x * x,
                op="add",
                identity=W.f32(0.0),
                order="relaxed",
                acc_dtype=W.f32,
            )
            sum_xdz = W.reduce(
                x * dz,
                op="add",
                identity=W.f32(0.0),
                order="relaxed",
                acc_dtype=W.f32,
            )
        mean_eps = sum_xx / count + eps
        sum_eps = sum_xx + eps * count
        reciprocal_rms = W.rsqrt(mean_eps)
        source_scale = -sum_xdz / sum_eps
        with W.vla(0, columns) as column:
            x = W.load(source + row * row_stride + column)
            dz = W.load(gradient + row * row_stride + column)
            W.store(
                output + row * row_stride + column,
                (dz + x * source_scale) * reciprocal_rms,
            )
