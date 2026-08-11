import weft
import weft.language as W


@weft.kernel
def conv_transpose2d_p0_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    weight: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    batch: W.index,
    input_height: W.index,
    input_width: W.index,
    input_channels: W.index,
    output_channels: W.index,
    kernel_height: W.index,
    kernel_width: W.index,
    stride: W.index,
) -> None:
    output_height = (input_height - 1) * stride + kernel_height
    output_width = (input_width - 1) * stride + kernel_width
    output_plane = output_height * output_width
    positions = batch * output_plane
    kernel_plane = kernel_height * kernel_width
    reduction_extent = input_channels * kernel_plane

    for position_base in W.range(0, positions, 6):
        with W.vla(0, output_channels) as output_channel:
            position_lane = W.block_axis(6)
            reduction = W.block_axis(reduction_extent)

            position = position_base + position_lane[:, None]
            image = position / output_plane
            output_spatial = position % output_plane
            output_y = output_spatial / output_width
            output_x = output_spatial % output_width

            reduction_index = reduction[None, :]
            input_channel = reduction_index / kernel_plane
            kernel_spatial = reduction_index % kernel_plane
            kernel_y = kernel_spatial / kernel_width
            kernel_x = kernel_spatial % kernel_width
            relative_y = output_y - kernel_y
            relative_x = output_x - kernel_x
            input_y = relative_y / stride
            input_x = relative_x / stride
            input_valid = (
                (position < positions)
                & (output_y >= kernel_y)
                & (output_x >= kernel_x)
                & (relative_y % stride == 0)
                & (relative_x % stride == 0)
                & (input_y < input_height)
                & (input_x < input_width)
            )
            input_value = W.load(
                source
                + input_x
                + input_width
                * (
                    input_y
                    + input_height
                    * (input_channel + input_channels * image)
                ),
                where=input_valid,
                other=W.f32(0.0),
            )

            weight_channel = output_channel[:, None]
            weight_value = W.load(
                weight
                + kernel_x
                + kernel_width
                * (
                    kernel_y
                    + kernel_height
                    * (weight_channel + output_channels * input_channel)
                )
            )
            value = W.contract(
                input_value,
                weight_value,
                init=W.f32(0.0),
                lhs_axes=(1,),
                rhs_axes=(1,),
                acc_dtype=W.f32,
                order="relaxed",
                math="native",
            )

            output_position = position_base + position_lane[None, :]
            output_image = output_position / output_plane
            output_spatial = output_position % output_plane
            W.store(
                output
                + output_spatial
                + output_plane
                * (output_channel[:, None] + output_channels * output_image),
                value,
                where=output_position < positions,
            )
