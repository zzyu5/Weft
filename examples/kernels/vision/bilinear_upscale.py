import weft
import weft.language as W


@weft.kernel
def bilinear_upscale_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    batch: W.index,
    input_height: W.index,
    input_width: W.index,
    channels: W.index,
    output_height: W.index,
    output_width: W.index,
) -> None:
    input_row_stride = input_width * channels
    input_batch_stride = input_height * input_row_stride
    output_row_stride = output_width * channels
    output_batch_stride = output_height * output_row_stride
    scale_y = W.cast(output_height, W.f32) / W.cast(input_height, W.f32)
    scale_x = W.cast(output_width, W.f32) / W.cast(input_width, W.f32)
    maximum_y = W.cast(input_height - W.index(1), W.f32)
    maximum_x = W.cast(input_width - W.index(1), W.f32)

    for image in W.range(0, batch):
        for output_y in W.range(0, output_height):
            source_y = (
                (W.cast(output_y, W.f32) + W.f32(0.5)) / scale_y
                - W.f32(0.5)
            )
            floor_y = W.floor(source_y)
            y0 = W.cast(
                W.minimum(W.maximum(floor_y, W.f32(0.0)), maximum_y),
                W.index,
            )
            y1 = W.cast(
                W.minimum(
                    W.maximum(floor_y + W.f32(1.0), W.f32(0.0)),
                    maximum_y,
                ),
                W.index,
            )
            dy = W.minimum(
                W.maximum(source_y - W.cast(y0, W.f32), W.f32(0.0)),
                W.f32(1.0),
            )
            for output_x in W.range(0, output_width):
                source_x = (
                    (W.cast(output_x, W.f32) + W.f32(0.5)) / scale_x
                    - W.f32(0.5)
                )
                floor_x = W.floor(source_x)
                x0 = W.cast(
                    W.minimum(W.maximum(floor_x, W.f32(0.0)), maximum_x),
                    W.index,
                )
                x1 = W.cast(
                    W.minimum(
                        W.maximum(floor_x + W.f32(1.0), W.f32(0.0)),
                        maximum_x,
                    ),
                    W.index,
                )
                dx = W.minimum(
                    W.maximum(
                        source_x - W.cast(x0, W.f32), W.f32(0.0)
                    ),
                    W.f32(1.0),
                )
                with W.vla(0, channels) as channel:
                    top_left = W.load(
                        source
                        + image * input_batch_stride
                        + y0 * input_row_stride
                        + x0 * channels
                        + channel
                    )
                    top_right = W.load(
                        source
                        + image * input_batch_stride
                        + y0 * input_row_stride
                        + x1 * channels
                        + channel
                    )
                    bottom_left = W.load(
                        source
                        + image * input_batch_stride
                        + y1 * input_row_stride
                        + x0 * channels
                        + channel
                    )
                    bottom_right = W.load(
                        source
                        + image * input_batch_stride
                        + y1 * input_row_stride
                        + x1 * channels
                        + channel
                    )
                    one_minus_x = W.f32(1.0) - dx
                    one_minus_y = W.f32(1.0) - dy
                    value = (
                        top_left * one_minus_x * one_minus_y
                        + top_right * dx * one_minus_y
                        + bottom_left * one_minus_x * dy
                        + bottom_right * dx * dy
                    )
                    W.store(
                        output
                        + image * output_batch_stride
                        + output_y * output_row_stride
                        + output_x * channels
                        + channel,
                        value,
                    )
