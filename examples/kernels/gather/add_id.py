import weft
import weft.language as W


@weft.kernel
def add_id_f32(
    a: W.ptr[W.f32, W.readonly, W.noalias],
    b: W.ptr[W.f32, W.readonly, W.noalias],
    ids: W.ptr[W.u32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    token_begin: W.index,
    token_end: W.index,
    slots: W.index,
    hidden: W.index,
    a_slot_stride: W.index,
    a_token_stride: W.index,
    b_expert_stride: W.index,
    id_token_stride: W.index,
    output_slot_stride: W.index,
    output_token_stride: W.index,
) -> None:
    for token in W.range(token_begin, token_end):
        for slot in W.range(0, slots):
            expert = W.cast(
                W.load(ids + token * id_token_stride + slot, other=W.u32(0)),
                W.index,
            )
            with W.vla(0, hidden) as column:
                lhs = W.load(
                    a
                    + token * a_token_stride
                    + slot * a_slot_stride
                    + column
                )
                rhs = W.load(b + expert * b_expert_stride + column)
                W.store(
                    output
                    + token * output_token_stride
                    + slot * output_slot_stride
                    + column,
                    lhs + rhs,
                )
