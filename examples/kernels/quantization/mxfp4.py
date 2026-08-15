import weft
import weft.language as W

from examples.kernels.quantization.dequantize import _ue4m3_half



@weft.kernel
def mxfp4_q8_0(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    elements: W.index,
) -> None:
    blocks = elements / W.index(32)
    weight_row_stride = blocks * W.index(17)

    for row in W.range(row_begin, row_end):
        result = W.f32(0.0)
        for block in W.range(0, blocks):
            weight_block = weight + row * weight_row_stride + block * W.index(17)
            exponent = W.load(weight_block, other=W.u8(0))
            code_lane = W.block(16)
            packed_codes = W.load(
                weight_block + W.index(1) + code_lane,
                other=W.u8(0),
            )
            activation_block = activation + block * W.index(34)
            activation_scale = W.load_f16_le(activation_block)
            activation_lane = W.block(32)
            activation_values = W.bitcast(
                W.load(
                    activation_block + W.index(2) + activation_lane,
                    other=W.u8(0),
                ),
                W.i8,
            )
            result = W.e2m1_e8m0_i8_dot(
                packed_codes,
                exponent,
                activation_values,
                activation_scale,
                result,
            )
        W.store(output + row, result)


@weft.kernel
def nvfp4_q8_0(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    codebook: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    elements: W.index,
) -> None:
    superblocks = elements / W.index(64)
    weight_row_stride = superblocks * W.index(36)

    for row in W.range(row_begin, row_end):
        result = W.f32(0.0)
        for superblock in W.range(0, superblocks):
            weight_block = (
                weight + row * weight_row_stride + superblock * W.index(36)
            )
            for subblock in W.range(0, 4):
                member = W.block(16)
                packed_index = member % W.index(8)
                shift = W.cast(
                    (member // W.index(8)) * W.index(4), W.u8
                )
                packed_codes = W.load(
                    weight_block
                    + W.index(4)
                    + subblock * W.index(8)
                    + packed_index,
                    other=W.u8(0),
                )
                codes = (packed_codes >> shift) & W.u8(15)
                table = W.bitcast(
                    W.load(codebook + member, other=W.u8(0)), W.i8
                )
                decoded = W.decode(codes, table, out_dtype=W.i8)
                activation_block = (
                    activation
                    + (
                        superblock * W.index(2)
                        + subblock // W.index(2)
                    )
                    * W.index(34)
                )
                activation_values = W.bitcast(
                    W.load(
                        activation_block
                        + W.index(2)
                        + (subblock % W.index(2)) * W.index(16)
                        + member,
                        other=W.u8(0),
                    ),
                    W.i8,
                )
                integer_sum = W.reduce(
                    W.cast(decoded, W.i32) * W.cast(activation_values, W.i32),
                    identity=W.i32(0),
                    axis=0,
                    acc_dtype=W.i32,
                    order="relaxed",
                )
                scale = _ue4m3_half(
                    W.load(weight_block + subblock, other=W.u8(0))
                ) * W.load_f16_le(activation_block)
                result = result + scale * W.cast(integer_sum, W.f32)
        W.store(output + row, result)
