import weft
import weft.language as W


@weft.kernel
def silu_f32(
    x: W.ptr[W.f32, W.readonly, W.noalias],
    y: W.ptr[W.f32, W.writeonly, W.noalias],
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        value = W.load(x + i)
        result = value / (W.f32(1.0) + W.exp(-value, math="fast"))
        W.store(y + i, result)
