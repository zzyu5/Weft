import weft
import weft.language as W


@weft.kernel
def im2col_backward_stride1_f32(
    gradient: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    batch: W.index,
    input_height: W.index,
    input_width: W.index,
    input_channels: W.index,
    kernel_height: W.index,
    kernel_width: W.index,
    output_height: W.index,
    output_width: W.index,
    padding_y: W.index,
    padding_x: W.index,
    dilation_y: W.index,
    dilation_x: W.index,
) -> None:
    input_plane = input_height * input_width
    kernel_plane = kernel_height * kernel_width
    column_stride = input_channels * kernel_plane

    for image in W.range(0, batch):
        for input_y in W.range(0, input_height):
            for input_x in W.range(0, input_width):
                with W.vla(0, input_channels) as input_channel:
                    padded_y = input_y + padding_y
                    padded_x = input_x + padding_x
                    first_output_y = padded_y
                    first_output_x = padded_x
                    first_valid = (first_output_y < output_height) & (
                        first_output_x < output_width
                    )
                    value = W.load(
                        gradient
                        + input_channel * kernel_plane
                        + column_stride
                        * (
                            first_output_x
                            + output_width
                            * (first_output_y + output_height * image)
                        ),
                        where=first_valid,
                        other=W.f32(0.0),
                    )

                    for window in W.range(1, kernel_plane):
                        kernel_y = window / kernel_width
                        kernel_x = window % kernel_width
                        kernel_offset_y = kernel_y * dilation_y
                        kernel_offset_x = kernel_x * dilation_x
                        sample_output_y = padded_y - kernel_offset_y
                        sample_output_x = padded_x - kernel_offset_x
                        sample_valid = (
                            (padded_y >= kernel_offset_y)
                            & (padded_x >= kernel_offset_x)
                            & (sample_output_y < output_height)
                            & (sample_output_x < output_width)
                        )
                        sample = W.load(
                            gradient
                            + input_channel * kernel_plane
                            + window
                            + column_stride
                            * (
                                sample_output_x
                                + output_width
                                * (sample_output_y + output_height * image)
                            ),
                            where=sample_valid,
                            other=W.f32(0.0),
                        )
                        value = value + sample

                    W.store(
                        output
                        + input_x
                        + input_width
                        * (
                            input_y
                            + input_height
                            * (input_channel + input_channels * image)
                        ),
                        value,
                    )
