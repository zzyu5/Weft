import weft
import weft.language as W


@weft.kernel
def conv3d_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    weight: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    packed_patches: W.ptr[W.f32, W.workspace, W.noalias],
    packed_weight: W.ptr[W.f32, W.workspace, W.noalias],
    batch: W.index,
    input_depth: W.index,
    input_height: W.index,
    input_width: W.index,
    input_channels: W.index,
    output_channels: W.index,
    kernel_depth: W.index,
    kernel_height: W.index,
    kernel_width: W.index,
    output_depth: W.index,
    output_height: W.index,
    output_width: W.index,
    stride_z: W.index,
    stride_y: W.index,
    stride_x: W.index,
    padding_z: W.index,
    padding_y: W.index,
    padding_x: W.index,
    dilation_z: W.index,
    dilation_y: W.index,
    dilation_x: W.index,
) -> None:
    output_plane = output_height * output_width
    output_volume = output_depth * output_plane
    positions = batch * output_volume
    kernel_plane = kernel_height * kernel_width
    kernel_volume = kernel_depth * kernel_plane
    reduction_extent = input_channels * kernel_volume
    W.storage(packed_patches, (positions, reduction_extent))
    W.storage(packed_weight, (output_channels, reduction_extent))

    for position in W.range(0, positions):
        image = position / output_volume
        volume_coordinate = position % output_volume
        output_z = volume_coordinate / output_plane
        plane_coordinate = volume_coordinate % output_plane
        output_y = plane_coordinate / output_width
        output_x = plane_coordinate % output_width
        for kernel_z in W.range(0, kernel_depth):
            for kernel_y in W.range(0, kernel_height):
                for kernel_x in W.range(0, kernel_width):
                    padded_z = output_z * stride_z + kernel_z * dilation_z
                    padded_y = output_y * stride_y + kernel_y * dilation_y
                    padded_x = output_x * stride_x + kernel_x * dilation_x
                    input_z = padded_z - padding_z
                    input_y = padded_y - padding_y
                    input_x = padded_x - padding_x
                    input_valid = (
                        (padded_z >= padding_z)
                        & (input_z < input_depth)
                        & (padded_y >= padding_y)
                        & (input_y < input_height)
                        & (padded_x >= padding_x)
                        & (input_x < input_width)
                    )
                    kernel_spatial = (
                        kernel_x
                        + kernel_width * (kernel_y + kernel_height * kernel_z)
                    )
                    with W.vla(0, input_channels) as input_channel:
                        value = W.load(
                            source
                            + input_x
                            + input_width
                            * (
                                input_y
                                + input_height
                                * (
                                    input_z
                                    + input_depth
                                    * (
                                        input_channel
                                        + input_channels * image
                                    )
                                )
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
        for kernel_z in W.range(0, kernel_depth):
            for kernel_y in W.range(0, kernel_height):
                for kernel_x in W.range(0, kernel_width):
                    kernel_spatial = (
                        kernel_x
                        + kernel_width * (kernel_y + kernel_height * kernel_z)
                    )
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
                                        kernel_z
                                        + kernel_depth
                                        * (
                                            input_channel
                                            + input_channels * output_channel
                                        )
                                    )
                                )
                            ),
                        )

    for position in W.range(0, positions, 8):
        for output_channel in W.range(0, output_channels):
            position_lane = W.block_axis(8)
            reduction = W.block_axis(reduction_extent)
            position_index = position + position_lane[:, None]
            patch = W.load(
                packed_patches
                + position_index * reduction_extent
                + reduction[None, :],
                where=position_index < positions,
                other=W.f32(0.0),
            )
            filter_value = W.load(
                packed_weight + output_channel * reduction_extent + reduction,
                other=W.f32(0.0),
            )
            value = W.dot(
                patch,
                filter_value,
                init=W.zeros((8,), dtype=W.f32),
                acc_dtype=W.f32,
                order="relaxed",
                math="native",
            )
            position_index = position + position_lane
            W.store(
                output
                + position_index % output_volume
                + output_volume
                * (output_channel + output_channels * (position_index / output_volume)),
                value,
                where=position_index < positions,
            )
