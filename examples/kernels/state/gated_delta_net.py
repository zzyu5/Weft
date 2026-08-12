import weft
import weft.language as W


@weft.kernel
def gated_delta_net_kda_f32(
    query: W.ptr[W.f32, W.readonly, W.noalias],
    key: W.ptr[W.f32, W.readonly, W.noalias],
    value: W.ptr[W.f32, W.readonly, W.noalias],
    gate: W.ptr[W.f32, W.readonly, W.noalias],
    beta: W.ptr[W.f32, W.readonly, W.noalias],
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
                    gated_state = (
                        W.load(state_head + row * width + column)
                        * W.exp(W.load(gate + token_head + column), math="native")
                    )
                    W.store(
                        state_head + row * width + column,
                        gated_state,
                    )

            for row in W.range(0, width):
                with W.vla(0, width) as column:
                    old_state = W.load(state_head + row * width + column)
                    projection = W.reduce(
                        old_state * W.load(key + token_head + column),
                        op="add",
                        identity=W.f32(0.0),
                        order="relaxed",
                        acc_dtype=W.f32,
                    )

                delta = (
                    W.load(value + token_head + row) - projection
                ) * W.load(beta + token * heads + head)

                with W.vla(0, width) as column:
                    next_state = (
                        W.load(state_head + row * width + column)
                        + delta * W.load(key + token_head + column)
                    )
                    W.store(
                        state_head + row * width + column,
                        next_state,
                    )
                    result = W.reduce(
                        next_state * W.load(query + token_head + column),
                        op="add",
                        identity=W.f32(0.0),
                        order="relaxed",
                        acc_dtype=W.f32,
                    )
                W.store(
                    output + token_head + row,
                    result * W.rsqrt(W.cast(width, W.f32), math="native"),
                )
