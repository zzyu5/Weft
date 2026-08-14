import weft
import weft.language as W


@weft.kernel
def col2im_1d_f32(
    columns: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    input_steps: W.index,
    kernel: W.index,
    output_channels: W.index,
    stride: W.index,
    padding: W.index,
) -> None:
    kernel_channels = kernel * output_channels
    output_steps = (
        (input_steps - W.index(1)) * stride
        + kernel
        - W.index(2) * padding
    )
    for channel in W.range(0, output_channels):
        with W.vla(0, output_steps) as output_step:
            absolute_step = output_step + padding
            accumulator = W.cast(output_step, W.f32) * W.f32(0.0)
            for kernel_step in W.range(0, kernel):
                numerator = absolute_step - kernel_step
                candidate_input = numerator / stride
                valid = (
                    (absolute_step >= kernel_step)
                    & (numerator % stride == W.index(0))
                    & (candidate_input < input_steps)
                )
                safe_input = W.select(valid, candidate_input, W.index(0))
                value = W.load(
                    columns
                    + channel * kernel
                    + kernel_step
                    + safe_input * kernel_channels,
                    where=valid,
                    other=W.f32(0.0),
                )
                accumulator = accumulator + value
            W.store(output + channel * output_steps + output_step, accumulator)


@weft.kernel
def col2im_1d_f32_equivalent(
    columns: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    input_steps: W.index,
    kernel: W.index,
    output_channels: W.index,
    stride: W.index,
    padding: W.index,
) -> None:
    kernel_channels = kernel * output_channels
    output_steps = (
        (input_steps - W.index(1)) * stride
        + kernel
        - W.index(2) * padding
    )
    for channel in W.range(0, output_channels):
        channel_base = columns + channel * kernel
        with W.vla(0, output_steps) as output_step:
            absolute_step = output_step + padding
            accumulator = W.cast(output_step, W.f32) * W.f32(0.0)
            for kernel_step in W.range(0, kernel):
                numerator = absolute_step - kernel_step
                aligned = numerator % stride == W.index(0)
                candidate_input = numerator / stride
                valid = (
                    (candidate_input < input_steps)
                    & aligned
                    & (kernel_step <= absolute_step)
                )
                safe_input = W.select(valid, candidate_input, W.index(0))
                input_base = channel_base + safe_input * kernel_channels
                value = W.load(
                    input_base + kernel_step,
                    where=valid,
                    other=W.f32(0.0),
                )
                accumulator = accumulator + value
            W.store(output + output_step + channel * output_steps, accumulator)
