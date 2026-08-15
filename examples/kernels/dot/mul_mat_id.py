import weft
import weft.language as W


@weft.kernel
def mul_mat_id_f32(
    weights: W.ptr[W.f32, W.readonly, W.noalias],
    activations: W.ptr[W.f32, W.readonly, W.noalias],
    ids: W.ptr[W.u32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
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
    weight_row_stride: W.index,
    activation_slot_stride: W.index,
    activation_token_stride: W.index,
    id_token_stride: W.index,
    output_slot_stride: W.index,
    output_token_stride: W.index,
) -> None:
    W.storage(expert_counts, (experts,))
    W.storage(expert_offsets, (experts + 1,))
    W.storage(expert_cursors, (experts,))
    W.storage(expert_items, (tokens, slots))

    for expert in W.range(0, experts):
        W.store(expert_counts + expert, W.u32(0))

    for item in W.range(0, tokens * slots):
        token = item / slots
        slot = item % slots
        expert = W.cast(
            W.load(ids + token * id_token_stride + slot, other=W.u32(0)),
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

    for item in W.range(0, tokens * slots):
        token = item / slots
        slot = item % slots
        expert = W.cast(
            W.load(ids + token * id_token_stride + slot, other=W.u32(0)),
            W.index,
        )
        position = W.cast(
            W.load(expert_cursors + expert, other=W.u32(0)), W.index
        )
        W.store(expert_items + position, W.cast(item, W.u32))
        W.store(expert_cursors + expert, W.cast(position + 1, W.u32))

    for expert in W.range(0, experts):
        begin = W.cast(
            W.load(expert_offsets + expert, other=W.u32(0)), W.index
        )
        end = W.cast(
            W.load(expert_offsets + expert + 1, other=W.u32(0)), W.index
        )
        weight_base = weights + expert * weight_expert_stride
        for row in W.range(0, rows, 6):
            for position in W.range(begin, end):
                item = W.cast(
                    W.load(expert_items + position, other=W.u32(0)), W.index
                )
                token = item / slots
                slot = item % slots
                activation_base = (
                    activations
                    + token * activation_token_stride
                    + slot * activation_slot_stride
                )
                output_base = (
                    output
                    + token * output_token_stride
                    + slot * output_slot_stride
                )
                row_lane = W.block(6)
                reduction = W.block(inner)
                row_index = row + row_lane[:, None]
                reduction_index = reduction[None, :]
                row_valid = row_index < rows
                lhs = W.load(
                    weight_base
                    + row_index * weight_row_stride
                    + reduction_index,
                    where=row_valid,
                    other=W.f32(0.0),
                )
                rhs = W.load(
                    activation_base + reduction,
                    other=W.f32(0.0),
                )
                value = W.dot(
                    lhs,
                    rhs,
                    init=W.zeros((row_lane,), dtype=W.f32),
                    acc_dtype=W.f32,
                    order="relaxed",
                    math="native",
                )
                W.store(
                    output_base + row + row_lane,
                    value,
                    where=row + row_lane < rows,
                )
