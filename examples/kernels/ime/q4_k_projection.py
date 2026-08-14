import weft
import weft.language as W

from examples.kernels.quantization.ggml_k import load_f16_le


@weft.kernel
def q4_k_projection_ime(
    activation: W.ptr[W.f32, W.readonly, W.noalias],
    packed_weight: W.ptr[
        W.u8, W.persistent("q4_k_n16_k32_304b"), W.readonly, W.noalias
    ],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    activation_scale: W.ptr[W.f32, W.workspace, W.noalias],
    activation_code: W.ptr[W.i8, W.workspace, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    inner: W.index,
) -> None:
    block_extent = W.index(32)
    packed_column_extent = W.index(16)
    blocks = inner // block_extent
    W.storage(
        packed_weight,
        shape=(columns // packed_column_extent, blocks, 304),
    )
    W.storage(activation_scale, shape=(row_end, blocks))
    W.storage(activation_code, shape=(row_end, inner))

    for row in W.range(row_begin, row_end):
        activation_row = activation + row * inner
        scale_row = activation_scale + row * blocks
        code_row = activation_code + row * inner
        output_row = output + row * columns

        for block in W.range(0, blocks):
            block_begin = block * block_extent
            block_end = block_begin + block_extent
            with W.vla(block_begin, block_end) as k:
                value = W.load(activation_row + k)
                magnitude = W.maximum(value, -value)
                maximum = W.reduce(
                    magnitude,
                    op="max",
                    identity=W.f32(0.0),
                    order="relaxed",
                    acc_dtype=W.f32,
                )

            scale = maximum / W.f32(127.0)
            W.store(scale_row + block, scale)
            inverse_scale = W.f32(1.0) / scale
            with W.vla(block_begin, block_end) as k:
                value = W.load(activation_row + k)
                quantized = W.narrow(
                    value * inverse_scale,
                    W.i8,
                    rounding="rne",
                    saturation=True,
                )
                W.store(code_row + k, quantized)

        for column_begin in W.range(0, columns, packed_column_extent):
            accumulator = W.zeros((16,), dtype=W.f32)
            for block in W.range(0, blocks):
                k = W.block_axis(32)
                activation_codes = W.load(
                    code_row + block * block_extent + k,
                    other=W.i8(0),
                )
                scale = W.load(scale_row + block, other=W.f32(0.0))

                column = W.block_axis(16)
                packed_byte = W.block_axis(16)
                packed_block = (
                    (column_begin // packed_column_extent) * blocks + block
                )
                packed_base = packed_weight + packed_block * W.index(304)
                weight_scale = load_f16_le(
                    packed_base + column * W.index(2)
                )
                weight_zero_point = W.load(
                    packed_base + W.index(32) + column,
                    other=W.u8(0),
                )
                packed_codes = W.load(
                    packed_base
                    + W.index(48)
                    + (packed_byte[None, :] // W.index(8)) * W.index(128)
                    + column[:, None] * W.index(8)
                    + packed_byte[None, :] % W.index(8),
                    other=W.u8(0),
                )
                accumulator = W.affine_i4_i8_contract(
                    activation_codes,
                    packed_codes,
                    activation_scale=scale,
                    weight_scale=weight_scale,
                    weight_zero_point=weight_zero_point,
                    init=accumulator,
                )

            column = W.block_axis(16)
            W.store(output_row + column_begin + column, accumulator)
