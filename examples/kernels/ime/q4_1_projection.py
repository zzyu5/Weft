import weft
import weft.language as W


@weft.kernel
def q4_1_projection_ime(
    activation: W.ptr[W.f32, W.readonly, W.noalias],
    packed_weight: W.ptr[
        W.u8, W.persistent("affine_i4_n16_k32_304b"), W.readonly, W.noalias
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
    W.buffer(
        packed_weight,
        shape=(columns // packed_column_extent, blocks, 304),
    )
    W.buffer(activation_scale, shape=(row_end, blocks))
    W.buffer(activation_code, shape=(row_end, inner))

    row_tile = W.index(4)
    full_row_end = row_begin + ((row_end - row_begin) // row_tile) * row_tile

    for row_group in W.blocks(row_begin, full_row_end, row_tile):
        for block in W.range(0, blocks):
            block_begin = block * block_extent
            block_end = block_begin + block_extent
            for row_lane in W.range(0, row_tile):
                row = row_group + row_lane
                activation_row = activation + row * inner
                scale_address = (
                    activation_scale
                    + row_group * blocks
                    + block * row_tile
                    + row_lane
                )
                with W.vla(block_begin, block_end) as k:
                    value = W.load(activation_row + k)
                    maximum = W.reduce(
                        W.maximum(value, -value),
                        op="max",
                        identity=W.f32(0.0),
                        order="relaxed",
                        acc_dtype=W.f32,
                    )
                scale = maximum / W.f32(127.0)
                W.store(scale_address, scale)
                inverse_scale = W.f32(1.0) / scale
                for fragment in W.range(0, 4):
                    with W.vla(0, 8) as lane:
                        k = block_begin + fragment * W.index(8) + lane
                        value = W.load(activation_row + k)
                        quantized = W.narrow(
                            value * inverse_scale,
                            W.i8,
                            rounding="rne",
                            saturation=True,
                        )
                        W.store(
                            activation_code
                            + row_group * inner
                            + block * W.index(128)
                            + fragment * W.index(32)
                            + row_lane * W.index(8)
                            + lane,
                            quantized,
                        )

        for column_begin in W.blocks(0, columns, packed_column_extent):
            row = W.axis(4)
            column = W.axis(16)
            accumulator = W.zeros((row, column), dtype=W.f32)
            for block in W.range(0, blocks):
                code = W.axis(128)
                activation_codes = W.load(
                    activation_code
                    + row_group * inner
                    + block * W.index(128)
                    + code,
                    other=W.i8(0),
                )
                scales = W.load(
                    activation_scale
                    + row_group * blocks
                    + block * row_tile
                    + row,
                    other=W.f32(0.0),
                )
                packed_block = (
                    (column_begin // packed_column_extent) * blocks + block
                )
                packed_base = packed_weight + packed_block * W.index(304)
                accumulator = W.quant.affine_i4_i8_dot(
                    activation_codes,
                    packed_base,
                    activation_scale=scales,
                    init=accumulator,
                )
            W.store(
                output
                + (row_group + row[:, None]) * columns
                + column_begin
                + column[None, :],
                accumulator,
            )

    for row in W.range(full_row_end, row_end):
        activation_row = activation + row * inner
        scale_row = activation_scale + row * blocks
        code_row = activation_code + row * inner
        output_row = output + row * columns

        for block in W.range(0, blocks):
            block_begin = block * block_extent
            block_end = block_begin + block_extent
            with W.vla(block_begin, block_end) as k:
                value = W.load(activation_row + k)
                maximum = W.reduce(
                    W.maximum(value, -value),
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

        for column_begin in W.blocks(0, columns, packed_column_extent):
            column = W.axis(16)
            accumulator = W.zeros((column,), dtype=W.f32)
            for block in W.range(0, blocks):
                k = W.axis(32)
                activation_codes = W.load(
                    code_row + block * block_extent + k,
                    other=W.i8(0),
                )
                scale = W.load(scale_row + block, other=W.f32(0.0))
                packed_block = (
                    (column_begin // packed_column_extent) * blocks + block
                )
                packed_base = packed_weight + packed_block * W.index(304)
                accumulator = W.quant.affine_i4_i8_dot(
                    activation_codes,
                    packed_base,
                    activation_scale=scale,
                    init=accumulator,
                )

            W.store(output_row + column_begin + column, accumulator)
