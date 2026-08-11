import weft
import weft.language as W

from examples.kernels.quantization.ggml_k import load_f16_le


@weft.kernel
def mxfp4_q8_0_rows(
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
            code_lane = W.block_axis(16)
            packed_codes = W.load(
                weight_block + W.index(1) + code_lane,
                other=W.u8(0),
            )
            activation_block = activation + block * W.index(34)
            activation_scale = load_f16_le(activation_block)
            activation_lane = W.block_axis(32)
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
