import weft
import weft.language as W


@weft.kernel
def quantize_q8_0(
    x: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.i8, W.writeonly, W.noalias],
    block_begin: W.index,
    block_end: W.index,
) -> None:
    for block in W.range(block_begin, block_end):
        input_base = block * W.index(32)
        output_base = block * W.index(34)

        with W.vla(0, 32) as i:
            value = W.load(x + input_base + i)
            magnitude = W.maximum(value, -value)
            maximum = W.reduce(
                magnitude,
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
        low_byte = W.bitcast(
            W.cast(scale_bits & W.u16(255), W.u8), W.i8
        )
        high_byte = W.bitcast(W.cast(scale_bits >> W.u16(8), W.u8), W.i8)
        W.store(output + output_base, low_byte)
        W.store(output + output_base + W.index(1), high_byte)

        with W.vla(0, 32) as i:
            value = W.load(x + input_base + i)
            code = W.narrow(
                value * inverse,
                W.i8,
                rounding="rne",
                saturation=True,
            )
            W.store(output + output_base + W.index(2) + i, code)
