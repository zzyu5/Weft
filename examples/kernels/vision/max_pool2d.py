import weft
import weft.language as W


@weft.kernel
def max_pool2d_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    batch: W.index,
    channels: W.index,
    input_height: W.index,
    input_width: W.index,
    output_height: W.index,
    output_width: W.index,
    kernel_height: W.index,
    kernel_width: W.index,
    stride_y: W.index,
    stride_x: W.index,
) -> None:
    input_plane_stride = input_height * input_width
    input_batch_stride = channels * input_plane_stride
    output_plane_stride = output_height * output_width
    output_batch_stride = channels * output_plane_stride
    window_elements = kernel_height * kernel_width

    for image in W.range(0, batch):
        for channel in W.range(0, channels):
            for output_y in W.range(0, output_height):
                with W.vla(0, output_width) as output_x:
                    source_y = output_y * stride_y
                    source_x = output_x * stride_x
                    value = W.load(
                        source
                        + image * input_batch_stride
                        + channel * input_plane_stride
                        + source_y * input_width
                        + source_x
                    )
                    for window in W.range(1, window_elements):
                        kernel_y = window / kernel_width
                        kernel_x = window % kernel_width
                        sample = W.load(
                            source
                            + image * input_batch_stride
                            + channel * input_plane_stride
                            + (source_y + kernel_y) * input_width
                            + source_x
                            + kernel_x
                        )
                        value = W.maximum(value, sample)
                    W.store(
                        output
                        + image * output_batch_stride
                        + channel * output_plane_stride
                        + output_y * output_width
                        + output_x,
                        value,
                    )
