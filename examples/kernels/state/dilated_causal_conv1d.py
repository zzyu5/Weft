import weft
import weft.language as W


@weft.kernel
def dilated_causal_conv1d_f32(
    source: W.ptr[W.f32, W.readonly, W.noalias],
    weight: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    tokens: W.index,
    channels: W.index,
    taps: W.index,
    dilation: W.index,
) -> None:
    padding = (taps - 1) * dilation
    for channel in W.range(0, channels):
        coefficient_base = weight + channel * taps
        with W.vla(0, tokens) as token:
            first = W.load(source + (token + padding) * channels + channel)
            value = first * W.load(coefficient_base)
            for tap in W.range(1, taps):
                distance = tap * dilation
                active = token >= distance
                sample = W.load(
                    source + (token + padding - distance) * channels + channel,
                    where=active,
                    other=W.f32(0.0),
                )
                coefficient = W.load(coefficient_base + tap)
                value = value + sample * coefficient
            W.store(output + token * channels + channel, value)
