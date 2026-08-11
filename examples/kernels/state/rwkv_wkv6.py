import weft
import weft.language as W


@weft.kernel
def rwkv_wkv6_f32(
    k: W.ptr[W.f32, W.readonly, W.noalias],
    v: W.ptr[W.f32, W.readonly, W.noalias],
    r: W.ptr[W.f32, W.readonly, W.noalias],
    time_decay: W.ptr[W.f32, W.readonly, W.noalias],
    time_first: W.ptr[W.f32, W.readonly, W.noalias],
    initial_state: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    final_state: W.ptr[W.f32, W.noalias],
    tokens: W.index,
    heads: W.index,
    width: W.index,
) -> None:
    head_stride = width
    token_stride = heads * width
    state_row_stride = width
    state_head_stride = width * width

    for head in W.range(0, heads):
        for row in W.range(0, width):
            with W.vla(0, width) as column:
                value = W.load(
                    initial_state
                    + head * state_head_stride
                    + row * state_row_stride
                    + column
                )
                W.store(
                    final_state
                    + head * state_head_stride
                    + row * state_row_stride
                    + column,
                    value,
                )

    for token in W.range(0, tokens):
        for head in W.range(0, heads):
            token_head = token * token_stride + head * head_stride
            state_head = final_state + head * state_head_stride
            with W.vla(0, width) as column:
                v_column = W.load(v + token_head + column)
                old = W.load(state_head + column)
                kv = v_column * W.load(k + token_head)
                value = (
                    old
                    + kv * W.load(time_first + head * head_stride)
                ) * W.load(r + token_head)
                W.store(
                    state_head + column,
                    old * W.load(time_decay + token_head) + kv,
                )
                for row in W.range(1, width):
                    loop_old = W.load(
                        state_head + row * state_row_stride + column
                    )
                    loop_kv = v_column * W.load(k + token_head + row)
                    value = value + (
                        loop_old
                        + loop_kv
                        * W.load(time_first + head * head_stride + row)
                    ) * W.load(r + token_head + row)
                    W.store(
                        state_head + row * state_row_stride + column,
                        loop_old
                        * W.load(time_decay + token_head + row)
                        + loop_kv,
                    )
                W.store(output + token_head + column, value)
