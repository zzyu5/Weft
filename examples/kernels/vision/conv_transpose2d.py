import weft
import weft.language as W


@weft.kernel
def conv_transpose2d_p0_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    weight: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    packed_source: W.ptr[W.f32, W.workspace, W.noalias],
    packed_weight: W.ptr[W.f32, W.workspace, W.noalias],
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
    W.storage(
        packed_source,
        shape=(batch, input_height, input_width, input_channels),
    )
    W.storage(
        packed_weight,
        shape=(output_channels, kernel_height, kernel_width, input_channels),
    )

    for image in W.range(0, batch):
        for input_y in W.range(0, input_height):
            for input_x in W.range(0, input_width):
                with W.vla(0, input_channels) as input_channel:
                    W.store(
                        packed_source
                        + input_channel
                        + input_channels
                        * (input_x + input_width * (input_y + input_height * image)),
                        W.load(
                            source
                            + input_x
                            + input_width
                            * (
                                input_y
                                + input_height
                                * (input_channel + input_channels * image)
                            )
                        ),
                    )

    for output_channel in W.range(0, output_channels):
        for kernel_y in W.range(0, kernel_height):
            for kernel_x in W.range(0, kernel_width):
                with W.vla(0, input_channels) as input_channel:
                    W.store(
                        packed_weight
                        + input_channel
                        + input_channels
                        * (
                            kernel_x
                            + kernel_width
                            * (kernel_y + kernel_height * output_channel)
                        ),
                        W.load(
                            weight
                            + kernel_x
                            + kernel_width
                            * (
                                kernel_y
                                + kernel_height
                                * (
                                    output_channel
                                    + output_channels * input_channel
                                )
                            )
                        ),
                    )

    for image in W.range(0, batch):
        for input_y in W.range(0, input_height):
            for input_x in W.range(0, input_width):
                for kernel_y in W.range(0, kernel_height):
                    for kernel_x in W.range(0, kernel_width):
                        output_y = input_y * stride + kernel_y
                        output_x = input_x * stride + kernel_x
                        output_spatial = output_x + output_width * output_y
                        for output_channel_base in W.range(
                            0, output_channels, 6
                        ):
                            output_channel = W.block(6)
                            reduction = W.block(input_channels)
                            input_value = W.load(
                                packed_source
                                + reduction
                                + input_channels
                                * (input_x + input_width * (input_y + input_height * image))
                            )
                            weight_channel = (
                                output_channel_base + output_channel[:, None]
                            )
                            weight_value = W.load(
                                packed_weight
                                + reduction[None, :]
                                + input_channels
                                * (
                                    kernel_x
                                    + kernel_width
                                    * (kernel_y + kernel_height * weight_channel)
                                ),
                                where=weight_channel < output_channels,
                                other=W.f32(0.0),
                            )
                            value = W.dot(
                                weight_value,
                                input_value,
                                init=W.zeros((output_channel,), dtype=W.f32),
                                acc_dtype=W.f32,
                                order="relaxed",
                                math="native",
                            )
                            W.store(
                                output
                                + output_spatial
                                + output_width
                                * output_height
                                * (
                                    output_channel_base
                                    + output_channel
                                    + output_channels * image
                                ),
                                value,
                                where=output_channel_base + output_channel
                                < output_channels,
                            )
