import weft
import weft.language as W


@W.helper(effects=("read",))
def load_f16_le(base):
    low = W.cast(W.load(base, other=W.u8(0)), W.u16)
    high = W.cast(W.load(base + 1, other=W.u8(0)), W.u16)
    bits = low | (high << W.u16(8))
    return W.cast(W.bitcast(bits, W.f16), W.f32)


@W.helper(effects=("read",))
def load_f32_le(base):
    byte0 = W.cast(W.load(base, other=W.u8(0)), W.u32)
    byte1 = W.cast(W.load(base + 1, other=W.u8(0)), W.u32)
    byte2 = W.cast(W.load(base + 2, other=W.u8(0)), W.u32)
    byte3 = W.cast(W.load(base + 3, other=W.u8(0)), W.u32)
    bits = byte0 | (byte1 << W.u32(8))
    bits = bits | (byte2 << W.u32(16))
    bits = bits | (byte3 << W.u32(24))
    return W.bitcast(bits, W.f32)


@W.helper(effects=("read",))
def load_i16_le(base):
    low = W.cast(W.load(base, other=W.u8(0)), W.u16)
    high = W.cast(W.load(base + 1, other=W.u8(0)), W.u16)
    bits = low | (high << W.u16(8))
    return W.cast(W.bitcast(bits, W.i16), W.i32)


@W.helper(effects=("read",))
def unpack_q4(qs, logical_index, zero):
    packed_index = logical_index % W.index(16)
    shift = W.cast((logical_index // W.index(16)) * W.index(4), W.u8)
    packed = W.load(qs + packed_index, other=W.u8(0))
    code = (packed >> shift) & W.u8(15)
    return W.cast(code, W.i32) + zero


@W.helper(effects=("read",))
def unpack_q5(qs, qh, logical_index, zero):
    packed_index = logical_index % W.index(16)
    nibble_shift = W.cast(
        (logical_index // W.index(16)) * W.index(4), W.u8
    )
    packed = W.load(qs + packed_index, other=W.u8(0))
    low_bits = (packed >> nibble_shift) & W.u8(15)

    # logical_index 16..31 selects qh bits 16..31, matching GGML's
    # ((qh & (1u << (j + 16))) >> (j + 12)) high-half extraction.
    high_byte = W.load(
        qh + logical_index // W.index(8), other=W.u8(0)
    )
    high_shift = W.cast(logical_index % W.index(8), W.u8)
    high_bit = ((high_byte >> high_shift) & W.u8(1)) << W.u8(4)
    return W.cast(low_bits | high_bit, W.i32) + zero


@W.helper(effects=("read",))
def load_q8(qs, logical_index):
    return W.cast(
        W.bitcast(W.load(qs + logical_index, other=W.u8(0)), W.i8),
        W.i32,
    )


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


@weft.kernel
def q4_0_q8_0(
    x: W.ptr[W.u8, W.readonly],
    y: W.ptr[W.u8, W.readonly],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x_block = x + block * W.index(18)
        y_block = y + block * W.index(34)
        logical_index = W.block_axis(32)
        x_values = unpack_q4(x_block + 2, logical_index, W.i32(-8))
        y_values = load_q8(y_block + 2, logical_index)
        integer_sum = W.reduce(
            x_values * y_values,
            identity=W.i32(0),
            axis=0,
            acc_dtype=W.i32,
            order="relaxed",
        )
        scale = load_f16_le(x_block) * load_f16_le(y_block)
        result += W.cast(integer_sum, W.f32) * scale
    return result


@weft.kernel
def q4_1_q8_1(
    x: W.ptr[W.u8, W.readonly],
    y: W.ptr[W.u8, W.readonly],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x_block = x + block * W.index(20)
        y_block = y + block * W.index(36)
        logical_index = W.block_axis(32)
        x_values = unpack_q4(x_block + 4, logical_index, W.i32(0))
        y_values = load_q8(y_block + 4, logical_index)
        integer_sum = W.reduce(
            x_values * y_values,
            identity=W.i32(0),
            axis=0,
            acc_dtype=W.i32,
            order="relaxed",
        )
        dot_scale = load_f16_le(x_block) * load_f16_le(y_block)
        correction = load_f16_le(x_block + 2) * load_f16_le(y_block + 2)
        result += W.cast(integer_sum, W.f32) * dot_scale + correction
    return result


@weft.kernel
def q5_0_q8_0(
    x: W.ptr[W.u8, W.readonly],
    y: W.ptr[W.u8, W.readonly],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x_block = x + block * W.index(22)
        y_block = y + block * W.index(34)
        logical_index = W.block_axis(32)
        x_values = unpack_q5(
            x_block + 6,
            x_block + 2,
            logical_index,
            W.i32(-16),
        )
        y_values = load_q8(y_block + 2, logical_index)
        integer_sum = W.reduce(
            x_values * y_values,
            identity=W.i32(0),
            axis=0,
            acc_dtype=W.i32,
            order="relaxed",
        )
        scale = load_f16_le(x_block) * load_f16_le(y_block)
        result += W.cast(integer_sum, W.f32) * scale
    return result


@weft.kernel
def q5_1_q8_1(
    x: W.ptr[W.u8, W.readonly],
    y: W.ptr[W.u8, W.readonly],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x_block = x + block * W.index(24)
        y_block = y + block * W.index(36)
        logical_index = W.block_axis(32)
        x_values = unpack_q5(
            x_block + 8,
            x_block + 4,
            logical_index,
            W.i32(0),
        )
        y_values = load_q8(y_block + 4, logical_index)
        integer_sum = W.reduce(
            x_values * y_values,
            identity=W.i32(0),
            axis=0,
            acc_dtype=W.i32,
            order="relaxed",
        )
        dot_scale = load_f16_le(x_block) * load_f16_le(y_block)
        correction = load_f16_le(x_block + 2) * load_f16_le(y_block + 2)
        result += W.cast(integer_sum, W.f32) * dot_scale + correction
    return result


@weft.kernel
def q8_0_q8_0(
    x: W.ptr[W.u8, W.readonly],
    y: W.ptr[W.u8, W.readonly],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x_block = x + block * W.index(34)
        y_block = y + block * W.index(34)
        logical_index = W.block_axis(32)
        x_values = load_q8(x_block + 2, logical_index)
        y_values = load_q8(y_block + 2, logical_index)
        integer_sum = W.reduce(
            x_values * y_values,
            identity=W.i32(0),
            axis=0,
            acc_dtype=W.i32,
            order="relaxed",
        )
        scale = load_f16_le(x_block) * load_f16_le(y_block)
        result += W.cast(integer_sum, W.f32) * scale
    return result


@weft.kernel
def q4_K_q8_K(
    x: W.ptr[W.u8, W.readonly],
    y: W.ptr[W.u8, W.readonly],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x_block = x + block * W.index(144)
        y_block = y + block * W.index(292)

        logical_index = W.block_axis(256)
        group = logical_index // W.index(64)
        offset = logical_index % W.index(64)
        packed_index = group * W.index(32) + offset % W.index(32)
        shift = W.cast((offset // W.index(32)) * W.index(4), W.u8)
        packed = W.load(
            x_block + W.index(16) + packed_index,
            other=W.u8(0),
        )
        x_values = W.cast((packed >> shift) & W.u8(15), W.i32)
        y_values = load_q8(y_block + 4, logical_index)
        scale_index = logical_index // W.index(32)
        scales, _ = load_k4_scale_min(x_block + 4, scale_index)
        scaled_products = (
            W.cast(scales, W.i32) * x_values * y_values
        )
        integer_sum = W.reduce(
            scaled_products,
            identity=W.i32(0),
            axis=0,
            acc_dtype=W.i32,
            order="relaxed",
        )

        bsum_index = W.block_axis(16)
        bsum_byte = y_block + W.index(260) + bsum_index * W.index(2)
        bsums = load_i16_le(bsum_byte)
        _, minimums = load_k4_scale_min(
            x_block + 4,
            bsum_index // W.index(2),
        )
        minimum_sum = W.reduce(
            bsums * W.cast(minimums, W.i32),
            identity=W.i32(0),
            axis=0,
            acc_dtype=W.i32,
            order="relaxed",
        )

        y_scale = load_f32_le(y_block)
        dot_scale = load_f16_le(x_block) * y_scale
        minimum_scale = load_f16_le(x_block + 2) * y_scale
        result += dot_scale * W.cast(integer_sum, W.f32)
        result -= minimum_scale * W.cast(minimum_sum, W.f32)
    return result
