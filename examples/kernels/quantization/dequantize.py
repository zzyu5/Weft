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
