import weft
import weft.language as W



@weft.kernel
def q4_k_mul_mat_id(
    activation: W.ptr[W.f32, W.readonly, W.noalias],
    packed_weight: W.ptr[
        W.u8, W.persistent("affine_i4_n16_k32_304b"), W.readonly, W.noalias
    ],
    ids: W.ptr[W.u32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    activation_scale: W.ptr[W.f32, W.workspace, W.noalias],
    activation_code: W.ptr[W.i8, W.workspace, W.noalias],
    expert_counts: W.ptr[W.u32, W.workspace, W.noalias],
    expert_offsets: W.ptr[W.u32, W.workspace, W.noalias],
    expert_cursors: W.ptr[W.u32, W.workspace, W.noalias],
    expert_items: W.ptr[W.u32, W.workspace, W.noalias],
    experts: W.index,
    tokens: W.index,
    slots: W.index,
    rows: W.index,
    inner: W.index,
    weight_expert_stride: W.index,
    activation_slot_stride: W.index,
    activation_token_stride: W.index,
    id_slot_stride: W.index,
    id_token_stride: W.index,
    output_slot_stride: W.index,
    output_token_stride: W.index,
) -> None:
    block_extent = W.index(32)
    column_extent = W.index(16)
    packed_block_bytes = W.index(304)
    blocks = inner / block_extent
    items = tokens * slots
    W.buffer(
        packed_weight,
        (experts, rows / column_extent, blocks, packed_block_bytes),
    )
    W.buffer(activation_scale, (tokens, slots, blocks))
    W.buffer(activation_code, (tokens, slots, inner))
    W.buffer(expert_counts, (experts,))
    W.buffer(expert_offsets, (experts + W.index(1),))
    W.buffer(expert_cursors, (experts,))
    W.buffer(expert_items, (tokens, slots))

    for item in W.range(0, items):
        token = item / slots
        slot = item % slots
        activation_row = (
            activation
            + token * activation_token_stride
            + slot * activation_slot_stride
        )
        scale_row = activation_scale + item * blocks
        code_row = activation_code + item * inner
        for block in W.range(0, blocks):
            block_begin = block * block_extent
            with W.vla(block_begin, block_begin + block_extent) as k:
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
            inverse_scale = W.f32(0.0)
            if maximum != W.f32(0.0):
                inverse_scale = W.f32(1.0) / scale
            with W.vla(block_begin, block_begin + block_extent) as k:
                W.store(
                    code_row + k,
                    W.narrow(
                        W.load(activation_row + k) * inverse_scale,
                        W.i8,
                        rounding="rne",
                        saturation=True,
                    ),
                )

    for expert in W.range(0, experts):
        W.store(expert_counts + expert, W.u32(0))
    for item in W.range(0, items):
        token = item / slots
        slot = item % slots
        expert = W.cast(
            W.load(
                ids + token * id_token_stride + slot * id_slot_stride,
                other=W.u32(0),
            ),
            W.index,
        )
        count = W.load(expert_counts + expert, other=W.u32(0))
        W.store(expert_counts + expert, count + W.u32(1))

    running = W.index(0)
    for expert in W.range(0, experts):
        W.store(expert_offsets + expert, W.cast(running, W.u32))
        running = running + W.cast(
            W.load(expert_counts + expert, other=W.u32(0)), W.index
        )
    W.store(expert_offsets + experts, W.cast(running, W.u32))

    for expert in W.range(0, experts):
        W.store(
            expert_cursors + expert,
            W.load(expert_offsets + expert, other=W.u32(0)),
        )
    for item in W.range(0, items):
        token = item / slots
        slot = item % slots
        expert = W.cast(
            W.load(
                ids + token * id_token_stride + slot * id_slot_stride,
                other=W.u32(0),
            ),
            W.index,
        )
        position = W.cast(
            W.load(expert_cursors + expert, other=W.u32(0)), W.index
        )
        W.store(expert_items + position, W.cast(item, W.u32))
        W.store(expert_cursors + expert, W.cast(position + W.index(1), W.u32))

    for expert in W.range(0, experts):
        begin = W.cast(
            W.load(expert_offsets + expert, other=W.u32(0)), W.index
        )
        end = W.cast(
            W.load(expert_offsets + expert + W.index(1), other=W.u32(0)),
            W.index,
        )
        expert_weight = packed_weight + expert * weight_expert_stride
        for row_begin in W.blocks(0, rows, column_extent):
            for position in W.range(begin, end):
                item = W.cast(
                    W.load(expert_items + position, other=W.u32(0)), W.index
                )
                token = item / slots
                slot = item % slots
                scale_row = activation_scale + item * blocks
                code_row = activation_code + item * inner
                output_row = (
                    output
                    + token * output_token_stride
                    + slot * output_slot_stride
                )
                column = W.axis(16)
                accumulator = W.zeros((column,), dtype=W.f32)
                for block in W.range(0, blocks):
                    k = W.axis(32)
                    activation_codes = W.load(
                        code_row + block * block_extent + k,
                        other=W.i8(0),
                    )
                    activation_block_scale = W.load(
                        scale_row + block, other=W.f32(0.0)
                    )
                    packed_base = (
                        expert_weight
                        + (
                            (row_begin / column_extent) * blocks + block
                        )
                        * packed_block_bytes
                    )
                    accumulator = W.quant.affine_i4_i8_dot(
                        activation_codes,
                        packed_base,
                        activation_scale=activation_block_scale,
                        init=accumulator,
                    )
                W.store(output_row + row_begin + column, accumulator)
