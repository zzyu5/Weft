import weft
import weft.language as W


@weft.kernel
def swiglu_f16_f32(
    gate: W.ptr[W.f16, W.readonly, W.noalias],
    up: W.ptr[W.f16, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        gate_value = W.cast(W.load(gate + i), W.f32)
        activated = gate_value / (
            W.f32(1.0) + W.exp(-gate_value, math="fast")
        )
        up_value = W.cast(W.load(up + i), W.f32)
        W.store(output + i, activated * up_value)
