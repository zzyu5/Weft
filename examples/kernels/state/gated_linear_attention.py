import weft
import weft.language as W


@weft.kernel
def gated_linear_attention_f32(
    key: W.ptr[W.f32, W.readonly, W.noalias],
    value: W.ptr[W.f32, W.readonly, W.noalias],
    query: W.ptr[W.f32, W.readonly, W.noalias],
    gate: W.ptr[W.f32, W.readonly, W.noalias],
    initial_state: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.noalias],
    final_state: W.ptr[W.f32, W.noalias],
    tokens: W.index,
    heads: W.index,
    width: W.index,
    scale: W.f32,
) -> None:
    token_stride = heads * width
    state_head_stride = width * width

    for head in W.range(0, heads):
        for row in W.range(0, width):
            with W.vla(0, width) as column:
                W.store(
                    final_state + head * state_head_stride + row * width + column,
                    W.load(
                        initial_state
                        + head * state_head_stride
                        + row * width
                        + column
                    ),
                )

    for token in W.range(0, tokens):
        for head in W.range(0, heads):
            token_head = token * token_stride + head * width
            state_head = final_state + head * state_head_stride

            with W.vla(0, width) as column:
                W.store(output + token_head + column, W.f32(0.0))

            for row in W.range(0, width):
                key_row = W.load(key + token_head + row)
                query_row = W.load(query + token_head + row) * scale
                gate_row = W.load(gate + token_head + row)
                with W.vla(0, width) as column:
                    next_state = (
                        W.load(state_head + row * width + column) * gate_row
                        + W.load(value + token_head + column) * key_row
                    )
                    current = W.load(output + token_head + column)
                    W.store(
                        output + token_head + column,
                        current + next_state * query_row,
                    )
                    W.store(
                        state_head + row * width + column,
                        next_state,
                    )
