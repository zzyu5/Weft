import weft
import weft.language as W


@weft.kernel
def ssm_conv_f32(
    state: W.ptr[W.f32, W.readonly, W.noalias],
    weight: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    sequences: W.index,
    tokens: W.index,
    channels: W.index,
    taps: W.index,
    state_channel_stride: W.index,
    state_sequence_stride: W.index,
    weight_channel_stride: W.index,
    output_token_stride: W.index,
    output_sequence_stride: W.index,
) -> None:
    for sequence in W.range(0, sequences):
        for channel in W.range(0, channels):
            with W.vla(0, tokens) as token:
                value = W.load(
                    state
                    + sequence * state_sequence_stride
                    + channel * state_channel_stride
                    + token
                ) * W.load(weight + channel * weight_channel_stride)
                for tap in W.range(1, taps):
                    sample = W.load(
                        state
                        + sequence * state_sequence_stride
                        + channel * state_channel_stride
                        + token
                        + tap
                    )
                    coefficient = W.load(
                        weight + channel * weight_channel_stride + tap
                    )
                    value = value + sample * coefficient
                W.store(
                    output
                    + sequence * output_sequence_stride
                    + token * output_token_stride
                    + channel,
                    value,
                )


@weft.kernel
def ssm_conv_f32_equivalent(
    state: W.ptr[W.f32, W.readonly, W.noalias],
    weight: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    sequences: W.index,
    tokens: W.index,
    channels: W.index,
    taps: W.index,
    state_channel_stride: W.index,
    state_sequence_stride: W.index,
    weight_channel_stride: W.index,
    output_token_stride: W.index,
    output_sequence_stride: W.index,
) -> None:
    for position in W.range(0, sequences * channels):
        sequence = position / channels
        channel = position % channels
        state_position = (
            state
            + sequence * state_sequence_stride
            + channel * state_channel_stride
        )
        output_position = output + sequence * output_sequence_stride + channel
        with W.vla(0, tokens) as token:
            value = W.load(
                state_position + token
            ) * W.load(weight + channel * weight_channel_stride)
            for tap in W.range(1, taps):
                sample = W.load(
                    state_position + token + tap
                )
                coefficient = W.load(
                    weight + channel * weight_channel_stride + tap
                )
                value = value + sample * coefficient
            W.store(output_position + token * output_token_stride, value)
