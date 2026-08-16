import weft
import weft.language as W

from examples.kernels.quantization.block_dot import q4_0_row_dot


@W.helper(effects=("read", "write"))
def quantize_q8_0_row(input_row, output_row, blocks):
    for block in W.range(0, blocks):
        input_base = input_row + block * W.index(32)
        output_base = output_row + block * W.index(34)

        with W.vla(0, 32) as lane:
            value = W.load(input_base + lane)
            maximum = W.reduce(
                W.maximum(value, -value),
                op="max",
                identity=W.neg_inf(W.f32),
                order="relaxed",
                acc_dtype=W.f32,
            )

        scale = maximum / W.f32(127.0)
        inverse = W.f32(0.0)
        if maximum != W.f32(0.0):
            inverse = W.f32(1.0) / scale

        scale_bits = W.bitcast(W.cast(scale, W.f16), W.u16)
        W.store(output_base, W.cast(scale_bits & W.u16(255), W.u8))
        W.store(
            output_base + W.index(1),
            W.cast(scale_bits >> W.u16(8), W.u8),
        )

        with W.vla(0, 32) as lane:
            value = W.load(input_base + lane)
            code = W.narrow(
                value * inverse,
                W.i8,
                rounding="rne",
                saturation=True,
            )
            W.store(output_base + W.index(2) + lane, W.bitcast(code, W.u8))
    return blocks


@weft.kernel
def mul_mat_q4_0(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    activation_q8: W.ptr[W.u8, W.workspace, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    inner: W.index,
) -> None:
    blocks = inner // W.index(32)
    activation_row_bytes = blocks * W.index(34)
    weight_row_bytes = blocks * W.index(18)
    W.storage(activation_q8, shape=(row_end, activation_row_bytes))

    for row in W.range(row_begin, row_end):
        quantize_q8_0_row(
            activation + row * inner,
            activation_q8 + row * activation_row_bytes,
            blocks,
        )

    for row in W.range(row_begin, row_end):
        activation_row = activation_q8 + row * activation_row_bytes
        output_row = output + row * columns
        for column in W.range(0, columns):
            result = q4_0_row_dot(
                weight + column * weight_row_bytes,
                activation_row,
                blocks,
            )
            W.store(output_row + column, result)
