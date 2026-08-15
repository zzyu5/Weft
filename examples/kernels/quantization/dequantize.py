import weft
import weft.language as W

from examples.kernels.quantization.ggml_k import load_k4_scale_min


@W.helper(effects=("read",))
def _load_u16_le(pointer):
    low = W.cast(W.load(pointer, other=W.u8(0)), W.u16)
    high = W.cast(W.load(pointer + W.index(1), other=W.u8(0)), W.u16)
    return low | (high << W.u16(8))


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
