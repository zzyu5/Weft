import weft
import weft.language as W


@weft.kernel
def quantize_q8_0(
    input: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.i8, W.writeonly, W.noalias],
    block_begin: W.index,
    block_end: W.index,
) -> None:
    for block in W.range(block_begin, block_end):
        input_base = block * W.index(32)
        output_base = block * W.index(34)

        with W.vla(0, 32) as i:
            value = W.load(input + input_base + i)
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
        W.store(
            output + output_base,
            W.bitcast(W.cast(scale_bits & W.u16(255), W.u8), W.i8),
        )
        W.store(
            output + output_base + W.index(1),
            W.bitcast(W.cast(scale_bits >> W.u16(8), W.u8), W.i8),
        )

        with W.vla(0, 32) as i:
            value = W.load(input + input_base + i)
            code = W.narrow(
                value * inverse,
                W.i8,
                rounding="rne",
                saturation=True,
            )
            W.store(output + output_base + W.index(2) + i, code)


@weft.kernel
def quantize_q8_1(
    input: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.u8, W.writeonly, W.noalias],
    block_begin: W.index,
    block_end: W.index,
) -> None:
    for block in W.range(block_begin, block_end):
        input_base = block * W.index(32)
        output_base = block * W.index(36)

        with W.vla(0, 32) as i:
            value = W.load(input + input_base + i)
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
        W.store(output + output_base, W.cast(scale_bits & W.u16(255), W.u8))
        W.store(
            output + output_base + W.index(1),
            W.cast(scale_bits >> W.u16(8), W.u8),
        )

        with W.vla(0, 32) as i:
            value = W.load(input + input_base + i)
            code = W.narrow(
                value * inverse,
                W.i8,
                rounding="rne",
                saturation=True,
            )
            W.store(
                output + output_base + W.index(4) + i,
                W.bitcast(code, W.u8),
            )
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
            output + output_base + W.index(2),
            W.cast(sum_bits & W.u16(255), W.u8),
        )
        W.store(
            output + output_base + W.index(3),
            W.cast(sum_bits >> W.u16(8), W.u8),
        )


@weft.kernel
def quantize_q8_K(
    input: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.u8, W.noalias],
    block_begin: W.index,
    block_end: W.index,
) -> None:
    for block in W.range(block_begin, block_end):
        input_base = block * W.index(256)
        output_base = block * W.index(292)

        with W.vla(0, 256) as i:
            value = W.load(input + input_base + i)
            state = W.argmax(
                W.maximum(value, -value),
                i,
                tie="lowest_coordinate",
                order="relaxed",
            )
        maximum, maximum_index = state
        extreme = W.load(input + input_base + maximum_index)
        inverse = W.f32(0.0)
        scale = W.f32(0.0)
        if maximum != W.f32(0.0):
            inverse = W.f32(-127.0) / extreme
            scale = W.f32(1.0) / inverse

        scale_bits = W.bitcast(scale, W.u32)
        W.store(output + output_base, W.cast(scale_bits & W.u32(255), W.u8))
        W.store(
            output + output_base + W.index(1),
            W.cast((scale_bits >> W.u32(8)) & W.u32(255), W.u8),
        )
        W.store(
            output + output_base + W.index(2),
            W.cast((scale_bits >> W.u32(16)) & W.u32(255), W.u8),
        )
        W.store(
            output + output_base + W.index(3),
            W.cast(scale_bits >> W.u32(24), W.u8),
        )

        with W.vla(0, 256) as i:
            value = W.load(input + input_base + i)
            code = W.narrow(
                value * inverse,
                W.i8,
                rounding="rne",
                saturation=True,
            )
            W.store(
                output + output_base + W.index(4) + i,
                W.bitcast(code, W.u8),
            )

        for group in W.range(0, 16):
            group_base = group * W.index(16)
            with W.vla(0, 16) as i:
                code = W.bitcast(
                    W.load(
                        output + output_base + W.index(4) + group_base + i
                    ),
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
                output + output_base + W.index(260) + group * W.index(2),
                W.cast(sum_bits & W.u16(255), W.u8),
            )
            W.store(
                output + output_base + W.index(261) + group * W.index(2),
                W.cast(sum_bits >> W.u16(8), W.u8),
            )
