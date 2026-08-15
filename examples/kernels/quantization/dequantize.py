import weft
import weft.language as W

from examples.kernels.quantization.ggml_k import load_k4_scale_min


@W.helper(effects=("read",))
def _load_u16_le(pointer):
    low = W.cast(W.load(pointer, other=W.u8(0)), W.u16)
    high = W.cast(W.load(pointer + W.index(1), other=W.u8(0)), W.u16)
    return low | (high << W.u16(8))


@W.helper(effects=())
def _e8m0_half(exponent):
    bits = W.u32(0)
    if exponent < W.u8(2):
        bits = W.u32(0x00200000) << W.cast(exponent, W.u32)
    else:
        bits = (W.cast(exponent, W.u32) - W.u32(1)) << W.u32(23)
    return W.bitcast(bits, W.f32)


@W.helper(effects=())
def _ue4m3_half(encoded):
    result = W.f32(0.0)
    if encoded != W.u8(0):
        if encoded != W.u8(127):
            exponent = (encoded >> W.u8(3)) & W.u8(15)
            mantissa = encoded & W.u8(7)
            if exponent == W.u8(0):
                result = W.cast(mantissa, W.f32) * W.f32(0.0009765625)
            else:
                bits = (
                    (W.cast(exponent, W.u32) + W.u32(119)) << W.u32(23)
                ) | (W.cast(mantissa, W.u32) << W.u32(20))
                result = W.bitcast(bits, W.f32)
    return result


@weft.kernel
def dequantize_q4_0(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(18)
            output_block = output_row + block * W.index(32)
            scale = W.load_f16_le(input_block)
            member = W.block(16)
            codes = W.load(input_block + W.index(2) + member, other=W.u8(0))
            low = W.cast(codes & W.u8(15), W.f32) - W.f32(8.0)
            high = W.cast(codes >> W.u8(4), W.f32) - W.f32(8.0)
            W.store(output_block + member, scale * low)
            W.store(output_block + W.index(16) + member, scale * high)


@weft.kernel
def dequantize_q4_1(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(20)
            output_block = output_row + block * W.index(32)
            scale = W.load_f16_le(input_block)
            minimum = W.load_f16_le(input_block + W.index(2))
            member = W.block(16)
            codes = W.load(input_block + W.index(4) + member, other=W.u8(0))
            low = W.cast(codes & W.u8(15), W.f32)
            high = W.cast(codes >> W.u8(4), W.f32)
            W.store(output_block + member, scale * low + minimum)
            W.store(output_block + W.index(16) + member, scale * high + minimum)


@weft.kernel
def dequantize_q8_0(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(34)
            output_block = output_row + block * W.index(32)
            scale = W.load_f16_le(input_block)
            member = W.block(32)
            codes = W.bitcast(
                W.load(input_block + W.index(2) + member, other=W.u8(0)),
                W.i8,
            )
            W.store(output_block + member, scale * W.cast(codes, W.f32))


@W.helper(effects=("read",))
def _load_q5_code(pointer, logical_index):
    packed_index = logical_index % W.index(16)
    packed_shift = W.cast(
        (logical_index // W.index(16)) * W.index(4), W.u8
    )
    low = (W.load(pointer + W.index(4) + packed_index, other=W.u8(0)) >> packed_shift) & W.u8(15)
    high_byte = W.load(
        pointer + logical_index // W.index(8), other=W.u8(0)
    )
    high_shift = W.cast(logical_index % W.index(8), W.u8)
    high = ((high_byte >> high_shift) & W.u8(1)) << W.u8(4)
    return low | high


@weft.kernel
def dequantize_q5_0(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(22)
            output_block = output_row + block * W.index(32)
            scale = W.load_f16_le(input_block)
            logical_index = W.block(32)
            code = _load_q5_code(input_block + W.index(2), logical_index)
            value = scale * (W.cast(code, W.f32) - W.f32(16.0))
            W.store(output_block + logical_index, value)


@weft.kernel
def dequantize_q5_1(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(24)
            output_block = output_row + block * W.index(32)
            scale = W.load_f16_le(input_block)
            minimum = W.load_f16_le(input_block + W.index(2))
            logical_index = W.block(32)
            code = _load_q5_code(input_block + W.index(4), logical_index)
            value = scale * W.cast(code, W.f32) + minimum
            W.store(output_block + logical_index, value)


@weft.kernel
def dequantize_q1_0(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(18)
            output_block = output_row + block * W.index(128)
            scale = W.load_f16_le(input_block)
            for quarter in W.range(0, 4):
                logical_index = W.block(32)
                block_index = quarter * W.index(32) + logical_index
                sign_byte = W.load(
                    input_block
                    + W.index(2)
                    + block_index // W.index(8),
                    other=W.u8(0),
                )
                shift = W.cast(block_index % W.index(8), W.u8)
                bit = (sign_byte >> shift) & W.u8(1)
                value = scale * (
                    W.cast(bit, W.f32) * W.f32(2.0) - W.f32(1.0)
                )
                W.store(output_block + block_index, value)


@weft.kernel
def dequantize_tq2_0(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(66)
            output_block = output_row + block * W.index(256)
            scale = W.load_f16_le(input_block + W.index(64))
            for half in W.range(0, 2):
                member = W.block(32)
                codes = W.load(
                    input_block + half * W.index(32) + member,
                    other=W.u8(0),
                )
                for field in W.range(0, 4):
                    shift = W.cast(field * W.index(2), W.u8)
                    code = (codes >> shift) & W.u8(3)
                    value = scale * (
                        W.cast(code, W.f32) - W.f32(1.0)
                    )
                    W.store(
                        output_block
                        + half * W.index(128)
                        + field * W.index(32)
                        + member,
                        value,
                    )


@weft.kernel
def dequantize_tq1_0(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(54)
            output_block = output_row + block * W.index(256)
            scale = W.load_f16_le(input_block + W.index(52))

            power = W.u8(1)
            for digit in W.range(0, 5):
                member = W.block(32)
                encoded = W.load(input_block + member, other=W.u8(0)) * power
                ternary = (
                    W.cast(encoded, W.u16) * W.u16(3)
                ) >> W.u16(8)
                value = scale * (
                    W.cast(ternary, W.f32) - W.f32(1.0)
                )
                W.store(output_block + digit * W.index(32) + member, value)
                power = power * W.u8(3)

            power = W.u8(1)
            for digit in W.range(0, 5):
                member = W.block(16)
                encoded = W.load(
                    input_block + W.index(32) + member, other=W.u8(0)
                ) * power
                ternary = (
                    W.cast(encoded, W.u16) * W.u16(3)
                ) >> W.u16(8)
                value = scale * (
                    W.cast(ternary, W.f32) - W.f32(1.0)
                )
                W.store(
                    output_block + W.index(160) + digit * W.index(16) + member,
                    value,
                )
                power = power * W.u8(3)

            power = W.u8(1)
            for digit in W.range(0, 4):
                member = W.block(4)
                encoded = W.load(
                    input_block + W.index(48) + member, other=W.u8(0)
                ) * power
                ternary = (
                    W.cast(encoded, W.u16) * W.u16(3)
                ) >> W.u16(8)
                value = scale * (
                    W.cast(ternary, W.f32) - W.f32(1.0)
                )
                W.store(
                    output_block + W.index(240) + digit * W.index(4) + member,
                    value,
                )
                power = power * W.u8(3)


@weft.kernel
def dequantize_mxfp4(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    codebook: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(17)
            output_block = output_row + block * W.index(32)
            scale = _e8m0_half(W.load(input_block, other=W.u8(0)))
            member = W.block(16)
            codes = W.load(
                input_block + W.index(1) + member, other=W.u8(0)
            )
            table = W.bitcast(
                W.load(codebook + member, other=W.u8(0)), W.i8
            )
            low = W.decode(codes & W.u8(15), table, out_dtype=W.i8)
            high = W.decode(codes >> W.u8(4), table, out_dtype=W.i8)
            W.store(output_block + member, scale * W.cast(low, W.f32))
            W.store(
                output_block + W.index(16) + member,
                scale * W.cast(high, W.f32),
            )


@weft.kernel
def dequantize_nvfp4(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    codebook: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(36)
            output_block = output_row + block * W.index(64)
            for subblock in W.range(0, 4):
                scale = _ue4m3_half(
                    W.load(input_block + subblock, other=W.u8(0))
                )
                member = W.block(16)
                packed_index = member % W.index(8)
                shift = W.cast((member // W.index(8)) * W.index(4), W.u8)
                codes = (
                    W.load(
                        input_block
                        + W.index(4)
                        + subblock * W.index(8)
                        + packed_index,
                        other=W.u8(0),
                    )
                    >> shift
                ) & W.u8(15)
                table = W.bitcast(
                    W.load(codebook + member, other=W.u8(0)), W.i8
                )
                decoded = W.decode(codes, table, out_dtype=W.i8)
                W.store(
                    output_block + subblock * W.index(16) + member,
                    scale * W.cast(decoded, W.f32),
                )


@weft.kernel
def dequantize_q4_K(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(144)
            output_block = output_row + block * W.index(256)
            block_scale = W.load_f16_le(input_block)
            block_minimum = W.load_f16_le(input_block + W.index(2))
            for group_pair in W.range(0, 4):
                low_group = group_pair * W.index(2)
                high_group = low_group + W.index(1)
                low_scale, low_minimum = load_k4_scale_min(
                    input_block + W.index(4), low_group
                )
                high_scale, high_minimum = load_k4_scale_min(
                    input_block + W.index(4), high_group
                )
                member = W.block(32)
                codes = W.load(
                    input_block
                    + W.index(16)
                    + group_pair * W.index(32)
                    + member,
                    other=W.u8(0),
                )
                low_value = (
                    block_scale
                    * W.cast(low_scale, W.f32)
                    * W.cast(codes & W.u8(15), W.f32)
                    - block_minimum * W.cast(low_minimum, W.f32)
                )
                high_value = (
                    block_scale
                    * W.cast(high_scale, W.f32)
                    * W.cast(codes >> W.u8(4), W.f32)
                    - block_minimum * W.cast(high_minimum, W.f32)
                )
                output_group = output_block + group_pair * W.index(64)
                W.store(output_group + member, low_value)
                W.store(output_group + W.index(32) + member, high_value)


@weft.kernel
def dequantize_q2_K(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(84)
            output_block = output_row + block * W.index(256)
            block_scale = W.load_f16_le(input_block + W.index(80))
            block_minimum = W.load_f16_le(input_block + W.index(82))
            for half in W.range(0, 2):
                for field in W.range(0, 4):
                    shift = W.cast(field * W.index(2), W.u8)
                    for lane_group in W.range(0, 2):
                        metadata_index = (
                            half * W.index(8)
                            + field * W.index(2)
                            + lane_group
                        )
                        metadata = W.load(
                            input_block + metadata_index, other=W.u8(0)
                        )
                        scale = block_scale * W.cast(
                            metadata & W.u8(15), W.f32
                        )
                        minimum = block_minimum * W.cast(
                            metadata >> W.u8(4), W.f32
                        )
                        member = W.block(16)
                        packed_codes = W.load(
                            input_block
                            + W.index(16)
                            + half * W.index(32)
                            + lane_group * W.index(16)
                            + member,
                            other=W.u8(0),
                        )
                        code = (packed_codes >> shift) & W.u8(3)
                        value = scale * W.cast(code, W.f32) - minimum
                        W.store(
                            output_block
                            + half * W.index(128)
                            + field * W.index(32)
                            + lane_group * W.index(16)
                            + member,
                            value,
                        )


@W.helper(effects=("read",))
def _load_q3_k_scale(scales, scale_index):
    quarter = scale_index // W.index(4)
    lane = scale_index % W.index(4)
    base = W.load(
        scales + (quarter % W.index(2)) * W.index(4) + lane,
        other=W.u8(0),
    )
    low = base & W.u8(15)
    if quarter >= W.index(2):
        low = base >> W.u8(4)
    high = (
        W.load(scales + W.index(8) + lane, other=W.u8(0))
        >> W.cast(quarter * W.index(2), W.u8)
    ) & W.u8(3)
    return low | (high << W.u8(4))


@weft.kernel
def dequantize_q3_K(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(110)
            output_block = output_row + block * W.index(256)
            block_scale = W.load_f16_le(input_block + W.index(108))
            for half in W.range(0, 2):
                for field in W.range(0, 4):
                    shift = W.cast(field * W.index(2), W.u8)
                    high_shift = W.cast(
                        half * W.index(4) + field, W.u8
                    )
                    for lane_group in W.range(0, 2):
                        scale_index = (
                            half * W.index(8)
                            + field * W.index(2)
                            + lane_group
                        )
                        local_scale = _load_q3_k_scale(
                            input_block + W.index(96), scale_index
                        )
                        scale = block_scale * (
                            W.cast(local_scale, W.f32) - W.f32(32.0)
                        )
                        member = W.block(16)
                        packed_codes = W.load(
                            input_block
                            + W.index(32)
                            + half * W.index(32)
                            + lane_group * W.index(16)
                            + member,
                            other=W.u8(0),
                        )
                        high_bits = W.load(
                            input_block
                            + lane_group * W.index(16)
                            + member,
                            other=W.u8(0),
                        )
                        low = (packed_codes >> shift) & W.u8(3)
                        present = (high_bits >> high_shift) & W.u8(1)
                        correction = (present ^ W.u8(1)) << W.u8(2)
                        code = W.cast(low, W.f32) - W.cast(correction, W.f32)
                        W.store(
                            output_block
                            + half * W.index(128)
                            + field * W.index(32)
                            + lane_group * W.index(16)
                            + member,
                            scale * code,
                        )


@weft.kernel
def dequantize_q5_K(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(176)
            output_block = output_row + block * W.index(256)
            block_scale = W.load_f16_le(input_block)
            block_minimum = W.load_f16_le(input_block + W.index(2))
            for pair in W.range(0, 4):
                low_group = pair * W.index(2)
                high_group = low_group + W.index(1)
                low_scale, low_minimum = load_k4_scale_min(
                    input_block + W.index(4), low_group
                )
                high_scale, high_minimum = load_k4_scale_min(
                    input_block + W.index(4), high_group
                )
                low_shift = W.cast(pair * W.index(2), W.u8)
                high_shift = low_shift + W.u8(1)
                member = W.block(32)
                high_bits = W.load(
                    input_block + W.index(16) + member, other=W.u8(0)
                )
                packed_codes = W.load(
                    input_block
                    + W.index(48)
                    + pair * W.index(32)
                    + member,
                    other=W.u8(0),
                )
                low_code = (packed_codes & W.u8(15)) | (
                    ((high_bits >> low_shift) & W.u8(1)) << W.u8(4)
                )
                high_code = (packed_codes >> W.u8(4)) | (
                    ((high_bits >> high_shift) & W.u8(1)) << W.u8(4)
                )
                low_value = (
                    block_scale
                    * W.cast(low_scale, W.f32)
                    * W.cast(low_code, W.f32)
                    - block_minimum * W.cast(low_minimum, W.f32)
                )
                high_value = (
                    block_scale
                    * W.cast(high_scale, W.f32)
                    * W.cast(high_code, W.f32)
                    - block_minimum * W.cast(high_minimum, W.f32)
                )
                output_pair = output_block + pair * W.index(64)
                W.store(output_pair + member, low_value)
                W.store(output_pair + W.index(32) + member, high_value)


@weft.kernel
def dequantize_q6_K(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(210)
            output_block = output_row + block * W.index(256)
            block_scale = W.load_f16_le(input_block + W.index(208))
            for half in W.range(0, 2):
                member = W.block(32)
                low_a = W.load(
                    input_block + half * W.index(64) + member,
                    other=W.u8(0),
                )
                low_b = W.load(
                    input_block
                    + half * W.index(64)
                    + W.index(32)
                    + member,
                    other=W.u8(0),
                )
                high = W.load(
                    input_block
                    + W.index(128)
                    + half * W.index(32)
                    + member,
                    other=W.u8(0),
                )
                scale_index = member // W.index(16)
                scale_base = input_block + W.index(192) + half * W.index(8)

                scale_1 = W.cast(
                    W.bitcast(
                        W.load(scale_base + scale_index, other=W.u8(0)), W.i8
                    ),
                    W.f32,
                )
                scale_2 = W.cast(
                    W.bitcast(
                        W.load(
                            scale_base + W.index(2) + scale_index,
                            other=W.u8(0),
                        ),
                        W.i8,
                    ),
                    W.f32,
                )
                scale_3 = W.cast(
                    W.bitcast(
                        W.load(
                            scale_base + W.index(4) + scale_index,
                            other=W.u8(0),
                        ),
                        W.i8,
                    ),
                    W.f32,
                )
                scale_4 = W.cast(
                    W.bitcast(
                        W.load(
                            scale_base + W.index(6) + scale_index,
                            other=W.u8(0),
                        ),
                        W.i8,
                    ),
                    W.f32,
                )
                code_1 = W.cast(
                    (low_a & W.u8(15))
                    | (((high >> W.u8(0)) & W.u8(3)) << W.u8(4)),
                    W.f32,
                ) - W.f32(32.0)
                code_2 = W.cast(
                    (low_b & W.u8(15))
                    | (((high >> W.u8(2)) & W.u8(3)) << W.u8(4)),
                    W.f32,
                ) - W.f32(32.0)
                code_3 = W.cast(
                    (low_a >> W.u8(4))
                    | (((high >> W.u8(4)) & W.u8(3)) << W.u8(4)),
                    W.f32,
                ) - W.f32(32.0)
                code_4 = W.cast(
                    (low_b >> W.u8(4))
                    | (((high >> W.u8(6)) & W.u8(3)) << W.u8(4)),
                    W.f32,
                ) - W.f32(32.0)
                output_half = output_block + half * W.index(128)
                W.store(
                    output_half + member,
                    block_scale * scale_1 * code_1,
                )
                W.store(
                    output_half + W.index(32) + member,
                    block_scale * scale_2 * code_2,
                )
                W.store(
                    output_half + W.index(64) + member,
                    block_scale * scale_3 * code_3,
                )
                W.store(
                    output_half + W.index(96) + member,
                    block_scale * scale_4 * code_4,
                )


@weft.kernel
def dequantize_iq4_xs(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    codebook: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(136)
            output_block = output_row + block * W.index(256)
            block_scale = W.load_f16_le(input_block)
            scales_high = _load_u16_le(input_block + W.index(2))
            for group in W.range(0, 8):
                scales_low = W.load(
                    input_block + W.index(4) + group // W.index(2),
                    other=W.u8(0),
                )
                low_shift = W.cast((group % W.index(2)) * W.index(4), W.u8)
                low = (scales_low >> low_shift) & W.u8(15)
                high_shift = W.cast(group * W.index(2), W.u16)
                high = W.cast((scales_high >> high_shift) & W.u16(3), W.u8)
                local_scale = W.cast(low | (high << W.u8(4)), W.f32)
                scale = block_scale * (local_scale - W.f32(32.0))
                member = W.block(16)
                codes = W.load(
                    input_block + W.index(8) + group * W.index(16) + member,
                    other=W.u8(0),
                )
                table = W.bitcast(
                    W.load(codebook + member, other=W.u8(0)), W.i8
                )
                low_value = W.decode(
                    codes & W.u8(15), table, out_dtype=W.i8
                )
                high_value = W.decode(
                    codes >> W.u8(4), table, out_dtype=W.i8
                )
                output_group = output_block + group * W.index(32)
                W.store(
                    output_group + member,
                    scale * W.cast(low_value, W.f32),
                )
                W.store(
                    output_group + W.index(16) + member,
                    scale * W.cast(high_value, W.f32),
                )


@weft.kernel
def dequantize_iq4_nl(
    packed: W.ptr[W.u8, W.readonly, W.noalias],
    codebook: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    rows: W.index,
    blocks_per_row: W.index,
    input_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for row in W.range(0, rows):
        input_row = packed + row * input_stride_bytes
        output_row = output + row * output_stride
        for block in W.range(0, blocks_per_row):
            input_block = input_row + block * W.index(18)
            output_block = output_row + block * W.index(32)
            scale = W.load_f16_le(input_block)
            member = W.block(16)
            packed_codes = W.load(
                input_block + W.index(2) + member,
                other=W.u8(0),
            )
            table_bits = W.load(codebook + member, other=W.u8(0))
            table = W.bitcast(table_bits, W.i8)
            low_decoded = W.decode(
                packed_codes & W.u8(15),
                table,
                out_dtype=W.i8,
            )
            high_decoded = W.decode(
                packed_codes >> W.u8(4),
                table,
                out_dtype=W.i8,
            )
            W.store(
                output_block + member,
                scale * W.cast(low_decoded, W.f32),
            )
            W.store(
                output_block + W.index(16) + member,
                scale * W.cast(high_decoded, W.f32),
            )
