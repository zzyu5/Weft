import weft
import weft.language as W

from examples.kernels.quantization.ggml_k import load_f16_le


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
            scale = load_f16_le(input_block)

            member = W.block_axis(16)
            packed_codes = W.load(
                input_block + W.index(2) + member,
                other=W.u8(0),
            )
            table_bits = W.load(codebook + member, other=W.u8(0))
            table = W.bitcast(table_bits, W.i8)

            low_code = packed_codes & W.u8(15)
            low_decoded = W.decode(
                low_code,
                table,
                out_dtype=W.i8,
            )
            low_value = scale * W.cast(low_decoded, W.f32)
            W.store(output_block + member, low_value)

            high_code = packed_codes >> W.u8(4)
            high_decoded = W.decode(
                high_code,
                table,
                out_dtype=W.i8,
            )
            high_value = scale * W.cast(high_decoded, W.f32)
            W.store(output_block + W.index(16) + member, high_value)
