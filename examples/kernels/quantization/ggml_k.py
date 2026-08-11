import weft.language as W


@W.helper(effects=("read",))
def load_f16_le(base):
    low = W.cast(W.load(base, other=W.u8(0)), W.u16)
    high = W.cast(W.load(base + 1, other=W.u8(0)), W.u16)
    bits = low | (high << W.u16(8))
    return W.cast(W.bitcast(bits, W.f16), W.f32)


@W.helper(effects=("read",))
def load_k4_scale_min(scales, scale_index):
    low_group = scale_index < W.index(4)
    high_group = scale_index >= W.index(4)

    low_scale = W.load(
        scales + scale_index,
        where=low_group,
        other=W.u8(0),
    ) & W.u8(63)
    low_minimum = W.load(
        scales + W.index(4) + scale_index,
        where=low_group,
        other=W.u8(0),
    ) & W.u8(63)

    high_scale_low = W.load(
        scales + W.index(4) + scale_index,
        where=high_group,
        other=W.u8(0),
    ) & W.u8(15)
    high_scale_high = (
        W.load(
            scales + (scale_index - W.index(4)),
            where=high_group,
            other=W.u8(0),
        )
        >> W.u8(6)
    ) << W.u8(4)
    high_scale = high_scale_low | high_scale_high

    high_minimum_low = W.load(
        scales + W.index(4) + scale_index,
        where=high_group,
        other=W.u8(0),
    ) >> W.u8(4)
    high_minimum_high = (
        W.load(
            scales + scale_index,
            where=high_group,
            other=W.u8(0),
        )
        >> W.u8(6)
    ) << W.u8(4)
    high_minimum = high_minimum_low | high_minimum_high

    scale = W.select(low_group, low_scale, high_scale)
    minimum = W.select(low_group, low_minimum, high_minimum)
    return W.tuple(scale, minimum)
