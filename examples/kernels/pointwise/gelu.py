import weft
import weft.language as W


@weft.kernel
def gelu_f32(
    input: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        value = W.load(input + i)
        cubic = value * value * value
        argument = W.f32(0.7978845608028654) * (
            value + W.f32(0.044715) * cubic
        )
        result = W.f32(0.5) * value * (
            W.f32(1.0) + W.tanh(argument, math="native")
        )
        W.store(output + i, result)
