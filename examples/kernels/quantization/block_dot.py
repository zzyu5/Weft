import weft
import weft.language as W

from examples.kernels.quantization.dequantize import _load_q3_k_scale
from examples.kernels.quantization.ggml_k import load_k4_scale_min



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
    high = W.cast(W.load(base + W.index(1), other=W.u8(0)), W.u16)
    return W.cast(W.bitcast(low | (high << W.u16(8)), W.i16), W.i32)


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
def q4_0_row_dot(x, y, blocks):
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x_block = x + block * W.index(18)
        y_block = y + block * W.index(34)
        logical_index = W.block(32)
        x_values = unpack_q4(x_block + 2, logical_index, W.i32(-8))
        y_values = load_q8(y_block + 2, logical_index)
        integer_sum = W.reduce(
            x_values * y_values,
            identity=W.i32(0),
            axis=0,
            acc_dtype=W.i32,
            order="relaxed",
        )
        scale = W.load_f16_le(x_block) * W.load_f16_le(y_block)
        result += W.cast(integer_sum, W.f32) * scale
    return result


@weft.kernel
def q4_0_q8_0(
    x: W.ptr[W.u8, W.readonly],
    y: W.ptr[W.u8, W.readonly],
    blocks: W.index,
) -> W.f32:
    return q4_0_row_dot(x, y, blocks)


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
        logical_index = W.block(32)
        x_values = unpack_q4(x_block + 4, logical_index, W.i32(0))
        y_values = load_q8(y_block + 4, logical_index)
        integer_sum = W.reduce(
            x_values * y_values,
            identity=W.i32(0),
            axis=0,
            acc_dtype=W.i32,
            order="relaxed",
        )
        dot_scale = W.load_f16_le(x_block) * W.load_f16_le(y_block)
        correction = W.load_f16_le(x_block + 2) * W.load_f16_le(y_block + 2)
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
        logical_index = W.block(32)
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
        scale = W.load_f16_le(x_block) * W.load_f16_le(y_block)
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
        logical_index = W.block(32)
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
        dot_scale = W.load_f16_le(x_block) * W.load_f16_le(y_block)
        correction = W.load_f16_le(x_block + 2) * W.load_f16_le(y_block + 2)
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
        logical_index = W.block(32)
        x_values = load_q8(x_block + 2, logical_index)
        y_values = load_q8(y_block + 2, logical_index)
        integer_sum = W.reduce(
            x_values * y_values,
            identity=W.i32(0),
            axis=0,
            acc_dtype=W.i32,
            order="relaxed",
        )
        scale = W.load_f16_le(x_block) * W.load_f16_le(y_block)
        result += W.cast(integer_sum, W.f32) * scale
    return result


@weft.kernel
def q4_K_q8_K(
    x: W.ptr[W.u8, W.readonly, W.noalias],
    y: W.ptr[W.u8, W.readonly, W.noalias],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x_block = x + block * W.index(144)
        y_block = y + block * W.index(292)
        scale_index = W.block(12)
        scale_min = W.load(x_block + W.index(4) + scale_index, other=W.u8(0))
        packed_index = W.block(128)
        packed_weight = W.load(
            x_block + W.index(16) + packed_index, other=W.u8(0)
        )
        activation_index = W.block(256)
        activation = W.bitcast(
            W.load(y_block + W.index(4) + activation_index, other=W.u8(0)),
            W.i8,
        )
        sum_byte = W.block(32)
        activation_sum_bytes = W.load(
            y_block + W.index(260) + sum_byte, other=W.u8(0)
        )
        y_scale = load_f32_le(y_block)
        dot_scale = W.load_f16_le(x_block) * y_scale
        minimum_scale = W.load_f16_le(x_block + 2) * y_scale
        result = W.grouped_affine_i4_i8_dot(
            packed_weight,
            scale_min,
            activation,
            activation_sum_bytes,
            dot_scale,
            minimum_scale,
            result,
        )
    return result


@weft.kernel
def tq2_0_q8_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x = weight + block * W.index(66)
        y = activation + block * W.index(292)
        integer_sum = W.i32(0)
        for half in W.range(0, 2):
            member = W.block(32)
            packed_codes = W.load(
                x + half * W.index(32) + member, other=W.u8(0)
            )
            for field in W.range(0, 4):
                shift = W.cast(field * W.index(2), W.u8)
                code = W.cast(
                    (packed_codes >> shift) & W.u8(3), W.i32
                ) - W.i32(1)
                activation_values = load_q8(
                    y
                    + W.index(4)
                    + half * W.index(128)
                    + field * W.index(32),
                    member,
                )
                integer_sum = integer_sum + W.reduce(
                    code * activation_values,
                    identity=W.i32(0),
                    axis=0,
                    acc_dtype=W.i32,
                    order="relaxed",
                )
        result = result + (
            W.load_f16_le(x + W.index(64))
            * load_f32_le(y)
            * W.cast(integer_sum, W.f32)
        )
    return result


@weft.kernel
def tq1_0_q8_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x = weight + block * W.index(54)
        y = activation + block * W.index(292)
        integer_sum = W.i32(0)

        power = W.u8(1)
        for digit in W.range(0, 5):
            member = W.block(32)
            encoded = W.load(x + member, other=W.u8(0)) * power
            ternary = (
                W.cast(encoded, W.u16) * W.u16(3)
            ) >> W.u16(8)
            code = W.cast(ternary, W.i32) - W.i32(1)
            activation_values = load_q8(
                y + W.index(4) + digit * W.index(32), member
            )
            integer_sum = integer_sum + W.reduce(
                code * activation_values,
                identity=W.i32(0),
                axis=0,
                acc_dtype=W.i32,
                order="relaxed",
            )
            power = power * W.u8(3)

        power = W.u8(1)
        for digit in W.range(0, 5):
            member = W.block(16)
            encoded = W.load(
                x + W.index(32) + member, other=W.u8(0)
            ) * power
            ternary = (
                W.cast(encoded, W.u16) * W.u16(3)
            ) >> W.u16(8)
            code = W.cast(ternary, W.i32) - W.i32(1)
            activation_values = load_q8(
                y + W.index(164) + digit * W.index(16), member
            )
            integer_sum = integer_sum + W.reduce(
                code * activation_values,
                identity=W.i32(0),
                axis=0,
                acc_dtype=W.i32,
                order="relaxed",
            )
            power = power * W.u8(3)

        power = W.u8(1)
        for digit in W.range(0, 4):
            member = W.block(4)
            encoded = W.load(
                x + W.index(48) + member, other=W.u8(0)
            ) * power
            ternary = (
                W.cast(encoded, W.u16) * W.u16(3)
            ) >> W.u16(8)
            code = W.cast(ternary, W.i32) - W.i32(1)
            activation_values = load_q8(
                y + W.index(244) + digit * W.index(4), member
            )
            integer_sum = integer_sum + W.reduce(
                code * activation_values,
                identity=W.i32(0),
                axis=0,
                acc_dtype=W.i32,
                order="relaxed",
            )
            power = power * W.u8(3)

        result = result + (
            W.load_f16_le(x + W.index(52))
            * load_f32_le(y)
            * W.cast(integer_sum, W.f32)
        )
    return result


@weft.kernel
def q2_K_q8_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x = weight + block * W.index(84)
        y = activation + block * W.index(292)
        weight_scale = W.load_f16_le(x + W.index(80))
        weight_minimum = W.load_f16_le(x + W.index(82))
        activation_scale = load_f32_le(y)
        for half in W.range(0, 2):
            for field in W.range(0, 4):
                shift = W.cast(field * W.index(2), W.u8)
                for lane_group in W.range(0, 2):
                    group = (
                        half * W.index(8)
                        + field * W.index(2)
                        + lane_group
                    )
                    metadata = W.load(x + group, other=W.u8(0))
                    member = W.block(16)
                    packed_codes = W.load(
                        x
                        + W.index(16)
                        + half * W.index(32)
                        + lane_group * W.index(16)
                        + member,
                        other=W.u8(0),
                    )
                    code = W.cast(
                        (packed_codes >> shift) & W.u8(3), W.i32
                    )
                    activation_values = load_q8(
                        y + W.index(4) + group * W.index(16), member
                    )
                    q_sum = W.reduce(
                        code * activation_values,
                        identity=W.i32(0),
                        axis=0,
                        acc_dtype=W.i32,
                        order="relaxed",
                    )
                    activation_sum = load_i16_le(
                        y + W.index(260) + group * W.index(2)
                    )
                    scale = W.cast(metadata & W.u8(15), W.f32)
                    minimum = W.cast(metadata >> W.u8(4), W.f32)
                    result = result + activation_scale * (
                        weight_scale * scale * W.cast(q_sum, W.f32)
                        - weight_minimum
                        * minimum
                        * W.cast(activation_sum, W.f32)
                    )
    return result


@weft.kernel
def q3_K_q8_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x = weight + block * W.index(110)
        y = activation + block * W.index(292)
        scale_product = W.load_f16_le(x + W.index(108)) * load_f32_le(y)
        for half in W.range(0, 2):
            for field in W.range(0, 4):
                low_shift = W.cast(field * W.index(2), W.u8)
                high_shift = W.cast(half * W.index(4) + field, W.u8)
                for lane_group in W.range(0, 2):
                    group = (
                        half * W.index(8)
                        + field * W.index(2)
                        + lane_group
                    )
                    local_scale = _load_q3_k_scale(x + W.index(96), group)
                    member = W.block(16)
                    packed_codes = W.load(
                        x
                        + W.index(32)
                        + half * W.index(32)
                        + lane_group * W.index(16)
                        + member,
                        other=W.u8(0),
                    )
                    high_bits = W.load(
                        x + lane_group * W.index(16) + member,
                        other=W.u8(0),
                    )
                    low = (packed_codes >> low_shift) & W.u8(3)
                    present = (high_bits >> high_shift) & W.u8(1)
                    correction = (present ^ W.u8(1)) << W.u8(2)
                    code = W.cast(low, W.i32) - W.cast(correction, W.i32)
                    activation_values = load_q8(
                        y + W.index(4) + group * W.index(16), member
                    )
                    q_sum = W.reduce(
                        code * activation_values,
                        identity=W.i32(0),
                        axis=0,
                        acc_dtype=W.i32,
                        order="relaxed",
                    )
                    result = result + (
                        scale_product
                        * (W.cast(local_scale, W.f32) - W.f32(32.0))
                        * W.cast(q_sum, W.f32)
                    )
    return result


@weft.kernel
def q5_K_q8_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    blocks: W.index,
) -> W.f32:
    result = W.f32(0.0)
    for block in W.range(0, blocks):
        x = weight + block * W.index(176)
        y = activation + block * W.index(292)
        weight_scale = W.load_f16_le(x)
        weight_minimum = W.load_f16_le(x + W.index(2))
        activation_scale = load_f32_le(y)
        for pair in W.range(0, 4):
            low_group = pair * W.index(2)
            high_group = low_group + W.index(1)
            low_scale, low_minimum = load_k4_scale_min(
                x + W.index(4), low_group
            )
            high_scale, high_minimum = load_k4_scale_min(
                x + W.index(4), high_group
            )
            low_shift = W.cast(pair * W.index(2), W.u8)
            high_shift = low_shift + W.u8(1)
            member = W.block(32)
            high_bits = W.load(x + W.index(16) + member, other=W.u8(0))
            packed_codes = W.load(
                x + W.index(48) + pair * W.index(32) + member,
                other=W.u8(0),
            )
            low_code = W.cast(
                (packed_codes & W.u8(15))
                | (((high_bits >> low_shift) & W.u8(1)) << W.u8(4)),
                W.i32,
            )
            high_code = W.cast(
                (packed_codes >> W.u8(4))
                | (((high_bits >> high_shift) & W.u8(1)) << W.u8(4)),
                W.i32,
            )
            activation_low = load_q8(
                y + W.index(4) + pair * W.index(64), member
            )
            activation_high = load_q8(
                y + W.index(36) + pair * W.index(64), member
            )
            low_q_sum = W.reduce(
                low_code * activation_low,
                identity=W.i32(0),
                axis=0,
                acc_dtype=W.i32,
                order="relaxed",
            )
            high_q_sum = W.reduce(
                high_code * activation_high,
                identity=W.i32(0),
                axis=0,
                acc_dtype=W.i32,
                order="relaxed",
            )
            low_activation_sum = load_i16_le(
                y + W.index(260) + pair * W.index(8)
            )
            low_activation_sum = low_activation_sum + load_i16_le(
                y + W.index(262) + pair * W.index(8)
            )
            high_activation_sum = load_i16_le(
                y + W.index(264) + pair * W.index(8)
            )
            high_activation_sum = high_activation_sum + load_i16_le(
                y + W.index(266) + pair * W.index(8)
            )
            result = result + activation_scale * (
                weight_scale
                * W.cast(low_scale, W.f32)
                * W.cast(low_q_sum, W.f32)
                - weight_minimum
                * W.cast(low_minimum, W.f32)
                * W.cast(low_activation_sum, W.f32)
                + weight_scale
                * W.cast(high_scale, W.f32)
                * W.cast(high_q_sum, W.f32)
                - weight_minimum
                * W.cast(high_minimum, W.f32)
                * W.cast(high_activation_sum, W.f32)
            )
    return result
