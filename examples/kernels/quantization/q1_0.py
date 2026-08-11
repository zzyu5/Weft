import weft
import weft.language as W

from examples.kernels.quantization.ggml_k import load_f16_le


@weft.kernel
def q1_0_q8_0_rows(
    weight: W.ptr[W.u8, W.readonly, W.noalias],
    activation: W.ptr[W.u8, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    row_begin: W.index,
    row_end: W.index,
    elements: W.index,
) -> None:
    blocks = elements / W.index(128)
    weight_row_stride = blocks * W.index(18)

    for row in W.range(row_begin, row_end):
        result = W.f32(0.0)
        for block in W.range(0, blocks):
            weight_block = weight + row * weight_row_stride + block * W.index(18)
            sign_scale = load_f16_le(weight_block)
            for sub_block in W.range(0, 4):
                sign_byte = W.block_axis(4)
                sign_bits = W.load(
                    weight_block
                    + W.index(2)
                    + sub_block * W.index(4)
                    + sign_byte,
                    other=W.u8(0),
                )
                activation_block = (
                    activation + (block * W.index(4) + sub_block) * W.index(34)
                )
                activation_scale = load_f16_le(activation_block)
                activation_lane = W.block_axis(32)
                activation_values = W.bitcast(
                    W.load(
                        activation_block + W.index(2) + activation_lane,
                        other=W.u8(0),
                    ),
                    W.i8,
                )
                result = W.sign_bit_i8_dot(
                    sign_bits,
                    activation_values,
                    activation_scale,
                    sign_scale,
                    result,
                )
        W.store(output + row, result)
