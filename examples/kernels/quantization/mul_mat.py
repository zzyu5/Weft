import weft
import weft.language as W

from examples.kernels.quantization.block_dot import (
    q2_K_row_dot,
    q3_K_row_dot,
    q4_0_row_dot,
    q4_1_row_dot,
    q4_K_row_dot,
    q5_0_row_dot,
    q5_1_row_dot,
    q5_K_row_dot,
    q8_0_row_dot,
    tq1_0_row_dot,
    tq2_0_row_dot,
)
from examples.kernels.quantization.codebook_k import q6_K_row_dot
from examples.kernels.quantization.q1_0 import q1_0_row_dot


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


@W.helper(effects=("read", "write"))
def quantize_q8_1_row(input_row, output_row, blocks):
    for block in W.range(0, blocks):
        input_base = input_row + block * W.index(32)
        output_base = output_row + block * W.index(36)

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
            W.store(output_base + W.index(4) + lane, W.bitcast(code, W.u8))
            code_sum = W.reduce(
                code,
                op="add",
                identity=W.i32(0),
                order="relaxed",
                acc_dtype=W.i32,
            )

        sum_scale = W.cast(code_sum, W.f32) * scale
        sum_bits = W.bitcast(W.cast(sum_scale, W.f16), W.u16)
        W.store(
            output_base + W.index(2),
            W.cast(sum_bits & W.u16(255), W.u8),
        )
        W.store(
            output_base + W.index(3),
            W.cast(sum_bits >> W.u16(8), W.u8),
        )
    return blocks


@W.helper(effects=("read", "write"))
def quantize_q8_K_row(input_row, output_row, blocks):
    for block in W.range(0, blocks):
        input_base = input_row + block * W.index(256)
        output_base = output_row + block * W.index(292)

        with W.vla(0, 256) as lane:
            value = W.load(input_base + lane)
            maximum_state = W.argmax(
                W.maximum(value, -value),
                lane,
                tie="lowest_coordinate",
                order="relaxed",
            )
        maximum, maximum_index = maximum_state
        extreme = W.load(input_base + maximum_index)
        inverse = W.f32(0.0)
        scale = W.f32(0.0)
        if maximum != W.f32(0.0):
            inverse = W.f32(-127.0) / extreme
            scale = W.f32(1.0) / inverse

        scale_bits = W.bitcast(scale, W.u32)
        W.store(output_base, W.cast(scale_bits & W.u32(255), W.u8))
        W.store(
            output_base + W.index(1),
            W.cast((scale_bits >> W.u32(8)) & W.u32(255), W.u8),
        )
        W.store(
            output_base + W.index(2),
            W.cast((scale_bits >> W.u32(16)) & W.u32(255), W.u8),
        )
        W.store(
            output_base + W.index(3),
            W.cast(scale_bits >> W.u32(24), W.u8),
        )

        with W.vla(0, 256) as lane:
            value = W.load(input_base + lane)
            code = W.narrow(
                value * inverse,
                W.i8,
                rounding="rne",
                saturation=True,
            )
            W.store(output_base + W.index(4) + lane, W.bitcast(code, W.u8))

        for group in W.range(0, 16):
            group_base = group * W.index(16)
            with W.vla(0, 16) as lane:
                code = W.bitcast(
                    W.load(output_base + W.index(4) + group_base + lane),
                    W.i8,
                )
                code_sum = W.reduce(
                    code,
                    op="add",
                    identity=W.i32(0),
                    order="relaxed",
                    acc_dtype=W.i32,
                )

            sum_bits = W.bitcast(W.cast(code_sum, W.i16), W.u16)
            W.store(
                output_base + W.index(260) + group * W.index(2),
                W.cast(sum_bits & W.u16(255), W.u8),
            )
            W.store(
                output_base + W.index(261) + group * W.index(2),
                W.cast(sum_bits >> W.u16(8), W.u8),
            )
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


@weft.kernel
def mul_mat_q4_1(
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
    activation_row_bytes = blocks * W.index(36)
    weight_row_bytes = blocks * W.index(20)
    W.storage(activation_q8, shape=(row_end, activation_row_bytes))

    for row in W.range(row_begin, row_end):
        quantize_q8_1_row(
            activation + row * inner,
            activation_q8 + row * activation_row_bytes,
            blocks,
        )

    for row in W.range(row_begin, row_end):
        activation_row = activation_q8 + row * activation_row_bytes
        output_row = output + row * columns
        for column in W.range(0, columns):
            W.store(
                output_row + column,
                q4_1_row_dot(
                    weight + column * weight_row_bytes,
                    activation_row,
                    blocks,
                ),
            )


@weft.kernel
def mul_mat_q5_0(
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
    weight_row_bytes = blocks * W.index(22)
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
            W.store(
                output_row + column,
                q5_0_row_dot(
                    weight + column * weight_row_bytes,
                    activation_row,
                    blocks,
                ),
            )


@weft.kernel
def mul_mat_q5_1(
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
    activation_row_bytes = blocks * W.index(36)
    weight_row_bytes = blocks * W.index(24)
    W.storage(activation_q8, shape=(row_end, activation_row_bytes))

    for row in W.range(row_begin, row_end):
        quantize_q8_1_row(
            activation + row * inner,
            activation_q8 + row * activation_row_bytes,
            blocks,
        )

    for row in W.range(row_begin, row_end):
        activation_row = activation_q8 + row * activation_row_bytes
        output_row = output + row * columns
        for column in W.range(0, columns):
            W.store(
                output_row + column,
                q5_1_row_dot(
                    weight + column * weight_row_bytes,
                    activation_row,
                    blocks,
                ),
            )


@weft.kernel
def mul_mat_q8_0(
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
    row_bytes = blocks * W.index(34)
    W.storage(activation_q8, shape=(row_end, row_bytes))

    for row in W.range(row_begin, row_end):
        quantize_q8_0_row(
            activation + row * inner,
            activation_q8 + row * row_bytes,
            blocks,
        )

    for row in W.range(row_begin, row_end):
        activation_row = activation_q8 + row * row_bytes
        output_row = output + row * columns
        for column in W.range(0, columns):
            W.store(
                output_row + column,
                q8_0_row_dot(
                    weight + column * row_bytes,
                    activation_row,
                    blocks,
                ),
            )


@weft.kernel
def mul_mat_q1_0(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    activation_q8: W.ptr[W.u8, W.workspace, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    inner: W.index,
) -> None:
    activation_blocks = inner // W.index(32)
    weight_blocks = inner // W.index(128)
    activation_row_bytes = activation_blocks * W.index(34)
    weight_row_bytes = weight_blocks * W.index(18)
    W.storage(activation_q8, shape=(row_end, activation_row_bytes))

    for row in W.range(row_begin, row_end):
        quantize_q8_0_row(
            activation + row * inner,
            activation_q8 + row * activation_row_bytes,
            activation_blocks,
        )

    for row in W.range(row_begin, row_end):
        activation_row = activation_q8 + row * activation_row_bytes
        output_row = output + row * columns
        for column in W.range(0, columns):
            W.store(
                output_row + column,
                q1_0_row_dot(
                    weight + column * weight_row_bytes,
                    activation_row,
                    weight_blocks,
                ),
            )


@weft.kernel
def mul_mat_q2_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    activation_q8: W.ptr[W.u8, W.workspace, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    inner: W.index,
) -> None:
    blocks = inner // W.index(256)
    activation_row_bytes = blocks * W.index(292)
    weight_row_bytes = blocks * W.index(84)
    W.storage(activation_q8, shape=(row_end, activation_row_bytes))
    for row in W.range(row_begin, row_end):
        quantize_q8_K_row(
            activation + row * inner,
            activation_q8 + row * activation_row_bytes,
            blocks,
        )
    for row in W.range(row_begin, row_end):
        activation_row = activation_q8 + row * activation_row_bytes
        output_row = output + row * columns
        for column in W.range(0, columns):
            W.store(
                output_row + column,
                q2_K_row_dot(
                    weight + column * weight_row_bytes,
                    activation_row,
                    blocks,
                ),
            )


@weft.kernel
def mul_mat_q3_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    activation_q8: W.ptr[W.u8, W.workspace, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    inner: W.index,
) -> None:
    blocks = inner // W.index(256)
    activation_row_bytes = blocks * W.index(292)
    weight_row_bytes = blocks * W.index(110)
    W.storage(activation_q8, shape=(row_end, activation_row_bytes))
    for row in W.range(row_begin, row_end):
        quantize_q8_K_row(
            activation + row * inner,
            activation_q8 + row * activation_row_bytes,
            blocks,
        )
    for row in W.range(row_begin, row_end):
        activation_row = activation_q8 + row * activation_row_bytes
        output_row = output + row * columns
        for column in W.range(0, columns):
            W.store(
                output_row + column,
                q3_K_row_dot(
                    weight + column * weight_row_bytes,
                    activation_row,
                    blocks,
                ),
            )


@weft.kernel
def mul_mat_q4_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    activation_q8: W.ptr[W.u8, W.workspace, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    inner: W.index,
) -> None:
    blocks = inner // W.index(256)
    activation_row_bytes = blocks * W.index(292)
    weight_row_bytes = blocks * W.index(144)
    W.storage(activation_q8, shape=(row_end, activation_row_bytes))
    for row in W.range(row_begin, row_end):
        quantize_q8_K_row(
            activation + row * inner,
            activation_q8 + row * activation_row_bytes,
            blocks,
        )
    for row in W.range(row_begin, row_end):
        activation_row = activation_q8 + row * activation_row_bytes
        output_row = output + row * columns
        for column in W.range(0, columns):
            W.store(
                output_row + column,
                q4_K_row_dot(
                    weight + column * weight_row_bytes,
                    activation_row,
                    blocks,
                ),
            )


@weft.kernel
def mul_mat_q5_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    activation_q8: W.ptr[W.u8, W.workspace, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    inner: W.index,
) -> None:
    blocks = inner // W.index(256)
    activation_row_bytes = blocks * W.index(292)
    weight_row_bytes = blocks * W.index(176)
    W.storage(activation_q8, shape=(row_end, activation_row_bytes))
    for row in W.range(row_begin, row_end):
        quantize_q8_K_row(
            activation + row * inner,
            activation_q8 + row * activation_row_bytes,
            blocks,
        )
    for row in W.range(row_begin, row_end):
        activation_row = activation_q8 + row * activation_row_bytes
        output_row = output + row * columns
        for column in W.range(0, columns):
            W.store(
                output_row + column,
                q5_K_row_dot(
                    weight + column * weight_row_bytes,
                    activation_row,
                    blocks,
                ),
            )


@weft.kernel
def mul_mat_q6_K(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    activation_q8: W.ptr[W.u8, W.workspace, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    inner: W.index,
) -> None:
    blocks = inner // W.index(256)
    activation_row_bytes = blocks * W.index(292)
    weight_row_bytes = blocks * W.index(210)
    W.storage(activation_q8, shape=(row_end, activation_row_bytes))
    for row in W.range(row_begin, row_end):
        quantize_q8_K_row(
            activation + row * inner,
            activation_q8 + row * activation_row_bytes,
            blocks,
        )
    for row in W.range(row_begin, row_end):
        activation_row = activation_q8 + row * activation_row_bytes
        output_row = output + row * columns
        for column in W.range(0, columns):
            W.store(
                output_row + column,
                q6_K_row_dot(
                    weight + column * weight_row_bytes,
                    activation_row,
                    blocks,
                ),
            )


@weft.kernel
def mul_mat_tq1_0(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    activation_q8: W.ptr[W.u8, W.workspace, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    inner: W.index,
) -> None:
    blocks = inner // W.index(256)
    activation_row_bytes = blocks * W.index(292)
    weight_row_bytes = blocks * W.index(54)
    W.storage(activation_q8, shape=(row_end, activation_row_bytes))
    for row in W.range(row_begin, row_end):
        quantize_q8_K_row(
            activation + row * inner,
            activation_q8 + row * activation_row_bytes,
            blocks,
        )
    for row in W.range(row_begin, row_end):
        activation_row = activation_q8 + row * activation_row_bytes
        output_row = output + row * columns
        for column in W.range(0, columns):
            W.store(
                output_row + column,
                tq1_0_row_dot(
                    weight + column * weight_row_bytes,
                    activation_row,
                    blocks,
                ),
            )


@weft.kernel
def mul_mat_tq2_0(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    activation_q8: W.ptr[W.u8, W.workspace, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    columns: W.index,
    inner: W.index,
) -> None:
    blocks = inner // W.index(256)
    activation_row_bytes = blocks * W.index(292)
    weight_row_bytes = blocks * W.index(66)
    W.storage(activation_q8, shape=(row_end, activation_row_bytes))
    for row in W.range(row_begin, row_end):
        quantize_q8_K_row(
            activation + row * inner,
            activation_q8 + row * activation_row_bytes,
            blocks,
        )
    for row in W.range(row_begin, row_end):
        activation_row = activation_q8 + row * activation_row_bytes
        output_row = output + row * columns
        for column in W.range(0, columns):
            W.store(
                output_row + column,
                tq2_0_row_dot(
                    weight + column * weight_row_bytes,
                    activation_row,
                    blocks,
                ),
            )
