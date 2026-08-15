import weft
import weft.language as W


@weft.kernel
def out_product_f32(
    lhs: W.ptr[W.f32, W.readonly, W.noalias],
    rhs: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    columns: W.index,
    samples: W.index,
    lhs_sample_stride: W.index,
    rhs_sample_stride: W.index,
    output_row_stride: W.index,
) -> None:
    for row in W.range(0, rows, 6):
        with W.vla(0, columns) as column:
            row_lane = W.block(6)
            sample_axis = W.block(samples)
            row_index = row + row_lane[:, None]
            sample_index = sample_axis[None, :]
            left = W.load(
                lhs + sample_index * lhs_sample_stride + row_index,
                where=row_index < rows,
                other=W.f32(0.0),
            )
            right = W.load(
                rhs
                + sample_axis[None, :] * rhs_sample_stride
                + column[:, None]
            )
            value = W.dot(
                left,
                right,
                init=W.f32(0.0),
                acc_dtype=W.f32,
                order="relaxed",
                math="native",
            )
            W.store(
                output
                + (row + row_lane[None, :]) * output_row_stride
                + column[:, None],
                value,
                where=row + row_lane[None, :] < rows,
            )
