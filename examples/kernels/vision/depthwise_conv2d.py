import weft
import weft.language as W


@weft.kernel
def depthwise_conv2d_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    weight: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    batch: W.index,
    input_height: W.index,
    input_width: W.index,
    channels: W.index,
    output_height: W.index,
    output_width: W.index,
    kernel_height: W.index,
    kernel_width: W.index,
    stride_y: W.index,
    stride_x: W.index,
    padding_y: W.index,
    padding_x: W.index,
    dilation_y: W.index,
    dilation_x: W.index,
) -> None:
    input_row_stride = input_width * channels
    input_batch_stride = input_height * input_row_stride
    output_row_stride = output_width * channels
    output_batch_stride = output_height * output_row_stride
    window_elements = kernel_height * kernel_width

    for image in W.range(0, batch):
        for output_y in W.range(0, output_height):
            for output_x in W.range(0, output_width):
                with W.vla(0, channels) as channel:
                    source_y_padded = output_y * stride_y
                    source_x_padded = output_x * stride_x
                    inside = (
                        (source_y_padded >= padding_y)
                        & (source_y_padded < input_height + padding_y)
                        & (source_x_padded >= padding_x)
                        & (source_x_padded < input_width + padding_x)
                    )
                    source_y = source_y_padded - padding_y
                    source_x = source_x_padded - padding_x
                    sample = W.load(
                        source
                        + image * input_batch_stride
                        + source_y * input_row_stride
                        + source_x * channels
                        + channel,
                        where=inside,
                        other=W.f32(0.0),
                    )
                    value = sample * W.load(weight + channel)
                    for window in W.range(1, window_elements):
                        kernel_y = window / kernel_width
                        kernel_x = window % kernel_width
                        loop_y_padded = (
                            output_y * stride_y + kernel_y * dilation_y
                        )
                        loop_x_padded = (
                            output_x * stride_x + kernel_x * dilation_x
                        )
                        loop_inside = (
                            (loop_y_padded >= padding_y)
                            & (loop_y_padded < input_height + padding_y)
                            & (loop_x_padded >= padding_x)
                            & (loop_x_padded < input_width + padding_x)
                        )
                        loop_y = loop_y_padded - padding_y
                        loop_x = loop_x_padded - padding_x
                        loop_sample = W.load(
                            source
                            + image * input_batch_stride
                            + loop_y * input_row_stride
                            + loop_x * channels
                            + channel,
                            where=loop_inside,
                            other=W.f32(0.0),
                        )
                        coefficient = W.load(
                            weight + window * channels + channel
                        )
                        value = value + loop_sample * coefficient
                    W.store(
                        output
                        + image * output_batch_stride
                        + output_y * output_row_stride
                        + output_x * channels
                        + channel,
                        value,
                    )
