import weft
import weft.language as W


@weft.kernel
def ssm_scan_f32(
    initial_state: W.ptr[W.f32, W.readonly, W.noalias],
    state_ids: W.ptr[W.u32, W.readonly, W.noalias],
    x: W.ptr[W.f32, W.readonly, W.noalias],
    dt: W.ptr[W.f32, W.readonly, W.noalias],
    decay: W.ptr[W.f32, W.readonly, W.noalias],
    b: W.ptr[W.f32, W.readonly, W.noalias],
    c: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    final_state: W.ptr[W.f32, W.noalias],
    sequences: W.index,
    tokens: W.index,
    heads: W.index,
    groups: W.index,
    dimensions: W.index,
    state_width: W.index,
) -> None:
    state_head_stride = dimensions * state_width
    state_sequence_stride = heads * state_head_stride
    token_head_stride = dimensions
    token_stride = heads * dimensions
    sequence_stride = tokens * token_stride
    parameter_group_stride = state_width
    parameter_token_stride = groups * state_width
    parameter_sequence_stride = tokens * parameter_token_stride

    for sequence in W.range(0, sequences):
        source_sequence = W.cast(
            W.load(state_ids + sequence, other=W.u32(0)), W.index
        )
        for head in W.range(0, heads):
            for state_index in W.range(0, state_width):
                with W.vla(0, dimensions) as dimension:
                    value = W.load(
                        initial_state
                        + source_sequence * state_sequence_stride
                        + head * state_head_stride
                        + dimension * state_width
                        + state_index
                    )
                    W.store(
                        final_state
                        + sequence * state_sequence_stride
                        + head * state_head_stride
                        + dimension * state_width
                        + state_index,
                        value,
                    )

        for token in W.range(0, tokens):
            for head in W.range(0, heads):
                dt_value = W.load(
                    dt + (sequence * tokens + token) * heads + head
                )
                positive_dt = W.log(W.f32(1.0) + W.exp(dt_value))
                decay_value = W.exp(
                    positive_dt * W.load(decay + head)
                )
                group = head / (heads / groups)
                parameter_base = (
                    sequence * parameter_sequence_stride
                    + token * parameter_token_stride
                    + group * parameter_group_stride
                )
                with W.vla(0, dimensions) as dimension:
                    x_value = W.load(
                        x
                        + sequence * sequence_stride
                        + token * token_stride
                        + head * token_head_stride
                        + dimension
                    )
                    x_dt = x_value * positive_dt
                    state_base = (
                        final_state
                        + sequence * state_sequence_stride
                        + head * state_head_stride
                        + dimension * state_width
                    )
                    old = W.load(state_base)
                    next_state = (
                        old * decay_value
                        + W.load(b + parameter_base) * x_dt
                    )
                    W.store(state_base, next_state)
                    value = next_state * W.load(c + parameter_base)
                    for state_index in W.range(1, state_width):
                        loop_old = W.load(state_base + state_index)
                        loop_next = (
                            loop_old * decay_value
                            + W.load(b + parameter_base + state_index) * x_dt
                        )
                        W.store(state_base + state_index, loop_next)
                        value = (
                            value
                            + loop_next
                            * W.load(c + parameter_base + state_index)
                        )
                    W.store(
                        output
                        + sequence * sequence_stride
                        + token * token_stride
                        + head * token_head_stride
                        + dimension,
                        value,
                    )
