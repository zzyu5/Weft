import weft
import weft.language as W


@weft.kernel
def rms_norm_mul_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    weight: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    source_stride: W.index,
    output_stride: W.index,
    epsilon: W.f32,
) -> None:
    for row in W.range(row_begin, row_end):
        source_row = source + row * source_stride
        output_row = output + row * output_stride
        with W.vla(0, columns) as column:
            value = W.load(source_row + column)
            sum_of_squares = W.reduce(
                value * value,
                op="add",
                identity=W.f32(0.0),
                order="relaxed",
                acc_dtype=W.f32,
            )

        inverse_norm = W.rsqrt(
            sum_of_squares / W.cast(columns, W.f32) + epsilon,
            math="native",
        )
        with W.vla(0, columns) as column:
            value = W.load(source_row + column)
            multiplier = W.load(weight + column)
            W.store(output_row + column, value * inverse_norm * multiplier)


@weft.kernel
def rms_norm_mul_f32_equivalent(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    weight: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    source_stride: W.index,
    output_stride: W.index,
    epsilon: W.f32,
) -> None:
    for row in W.range(row_begin, row_end):
        with W.vla(0, columns) as column:
            value = W.load(source + row * source_stride + column)
            square = value * value
            sum_of_squares = W.reduce(
                square,
                op="add",
                identity=W.f32(0.0),
                order="relaxed",
                acc_dtype=W.f32,
            )

        mean_square = sum_of_squares / W.cast(columns, W.f32)
        inverse_norm = W.rsqrt(mean_square + epsilon, math="native")
        with W.vla(0, columns) as column:
            normalized = (
                W.load(source + row * source_stride + column) * inverse_norm
            )
            W.store(
                output + row * output_stride + column,
                W.load(weight + column) * normalized,
            )
