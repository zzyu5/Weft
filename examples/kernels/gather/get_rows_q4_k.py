import weft
import weft.language as W

from examples.kernels.quantization.ggml_k import load_f16_le
from examples.kernels.quantization.ggml_k import load_k4_scale_min


@weft.kernel
def get_rows_q4_k(
    packed_rows: W.ptr[W.u8, W.readonly, W.noalias],
    row_indices: W.ptr[W.i32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    tokens: W.index,
    blocks_per_row: W.index,
    row_stride_bytes: W.index,
    output_stride: W.index,
) -> None:
    for token in W.range(0, tokens):
        row_index = W.cast(
            W.load(row_indices + token, other=W.i32(0)), W.index
        )
        row_base = packed_rows + row_index * row_stride_bytes
        output_row = output + token * output_stride

        for block in W.range(0, blocks_per_row):
            packed_block = row_base + block * W.index(144)
            output_block = output_row + block * W.index(256)
            block_scale = load_f16_le(packed_block)
            block_minimum = load_f16_le(packed_block + W.index(2))

            for group_pair in W.range(0, 4):
                low_group = group_pair * W.index(2)
                high_group = low_group + W.index(1)
                low_scale, low_minimum = load_k4_scale_min(
                    packed_block + W.index(4), low_group
                )
                high_scale, high_minimum = load_k4_scale_min(
                    packed_block + W.index(4), high_group
                )
                output_group = output_block + group_pair * W.index(64)
                high_output_group = output_group + W.index(32)

                member = W.block_axis(32)
                packed = W.load(
                    packed_block
                    + W.index(16)
                    + group_pair * W.index(32)
                    + member,
                    other=W.u8(0),
                )
                low_code = packed & W.u8(15)
                low_value = (
                    block_scale
                    * W.cast(low_scale, W.f32)
                    * W.cast(low_code, W.f32)
                    - block_minimum * W.cast(low_minimum, W.f32)
                )
                W.store(output_group + member, low_value)

                high_code = packed >> W.u8(4)
                high_value = (
                    block_scale
                    * W.cast(high_scale, W.f32)
                    * W.cast(high_code, W.f32)
                    - block_minimum * W.cast(high_minimum, W.f32)
                )
                W.store(high_output_group + member, high_value)
