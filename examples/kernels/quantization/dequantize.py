import weft
import weft.language as W


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
