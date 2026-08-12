import weft
import weft.language as W


@weft.kernel
def dense_conv2d_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    weight: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    packed_patches: W.ptr[W.f32, W.noalias],
    packed_weight: W.ptr[W.f32, W.noalias],
    batch: W.index,
    input_height: W.index,
    input_width: W.index,
    input_channels: W.index,
    output_channels: W.index,
    kernel_height: W.index,
    kernel_width: W.index,
    output_height: W.index,
    output_width: W.index,
    stride_y: W.index,
    stride_x: W.index,
    padding_y: W.index,
    padding_x: W.index,
    dilation_y: W.index,
    dilation_x: W.index,
) -> None:
    output_plane = output_height * output_width
    positions = batch * output_plane
    kernel_plane = kernel_height * kernel_width
    reduction_extent = input_channels * kernel_plane

    for position in W.range(0, positions):
        image = position / output_plane
        output_spatial = position % output_plane
        output_y = output_spatial / output_width
        output_x = output_spatial % output_width
        for kernel_y in W.range(0, kernel_height):
            for kernel_x in W.range(0, kernel_width):
                padded_y = output_y * stride_y + kernel_y * dilation_y
                padded_x = output_x * stride_x + kernel_x * dilation_x
                input_y = padded_y - padding_y
                input_x = padded_x - padding_x
                input_valid = (
                    (padded_y >= padding_y)
                    & (input_y < input_height)
                    & (padded_x >= padding_x)
                    & (input_x < input_width)
                )
                kernel_spatial = kernel_x + kernel_width * kernel_y
                with W.vla(0, input_channels) as input_channel:
                    value = W.load(
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
                    W.store(
                        packed_patches
                        + position * reduction_extent
                        + kernel_spatial * input_channels
                        + input_channel,
                        value,
                    )

    for output_channel in W.range(0, output_channels):
        for kernel_y in W.range(0, kernel_height):
            for kernel_x in W.range(0, kernel_width):
                kernel_spatial = kernel_x + kernel_width * kernel_y
                with W.vla(0, input_channels) as input_channel:
                    W.store(
                        packed_weight
                        + output_channel * reduction_extent
                        + kernel_spatial * input_channels
                        + input_channel,
                        W.load(
                            weight
                            + kernel_x
                            + kernel_width
                            * (
                                kernel_y
                                + kernel_height
                                * (
                                    input_channel
                                    + input_channels * output_channel
                                )
                            )
                        ),
                    )

    for position in W.range(0, positions, 6):
        for output_channel in W.range(0, output_channels):
            position_lane = W.block_axis(6)
            reduction = W.block_axis(reduction_extent)
            position_index = position + position_lane[:, None]
            patch = W.load(
                packed_patches + position_index * reduction_extent + reduction[None, :],
                where=position_index < positions,
                other=W.f32(0.0),
            )
            filter_value = W.load(
                packed_weight + output_channel * reduction_extent + reduction,
                other=W.f32(0.0),
            )
            value = W.contract(
                patch,
                filter_value,
                init=W.zeros((6,), dtype=W.f32),
                lhs_axes=(1,),
                rhs_axes=(0,),
                acc_dtype=W.f32,
                order="relaxed",
                math="native",
            )
            position_index = position + position_lane
            image = position_index / output_plane
            spatial = position_index % output_plane
            W.store(
                output
                + spatial
                + output_plane * (output_channel + output_channels * image),
                value,
                where=position_index < positions,
            )
