import weft
import weft.language as W


@weft.kernel
def rwkv_wkv7_f32(
    key: W.ptr[W.f32, W.readonly, W.noalias],
    value: W.ptr[W.f32, W.readonly, W.noalias],
    receptance: W.ptr[W.f32, W.readonly, W.noalias],
    decay: W.ptr[W.f32, W.readonly, W.noalias],
    a: W.ptr[W.f32, W.readonly, W.noalias],
    b: W.ptr[W.f32, W.readonly, W.noalias],
    initial_state: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    final_state: W.ptr[W.f32, W.noalias],
    tokens: W.index,
    heads: W.index,
    width: W.index,
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
            for row in W.range(0, width):
                with W.vla(0, width) as column:
                    state_value = W.load(state_head + row * width + column)
                    a_state = W.reduce(
                        W.load(a + token_head + column) * state_value,
                        op="add",
                        identity=W.f32(0.0),
                        order="relaxed",
                        acc_dtype=W.f32,
                    )

                with W.vla(0, width) as column:
                    next_state = (
                        W.load(state_head + row * width + column)
                        * W.load(decay + token_head + column)
                        + W.load(value + token_head + row)
                        * W.load(key + token_head + column)
                        + a_state * W.load(b + token_head + column)
                    )
                    W.store(
                        state_head + row * width + column,
                        next_state,
                    )
                    result = W.reduce(
                        next_state * W.load(receptance + token_head + column),
                        op="add",
                        identity=W.f32(0.0),
                        order="relaxed",
                        acc_dtype=W.f32,
                    )
                W.store(output + token_head + row, result)
