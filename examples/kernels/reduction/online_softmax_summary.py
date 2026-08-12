import weft
import weft.language as W

@weft.kernel
def online_softmax_summary(
    x: W.ptr[W.f32, W.readonly],
    out: W.ptr[W.f32, W.writeonly],
    n: W.index,
    logical_n: W.index,
) -> None:
    with W.vla(0, n) as i:
        active = i < logical_n
        value = W.load(x + i, where=active)
        state = W.online_softmax_summary(
            value,
            math="native",
            order="preserve",
        )
    maximum, scaled_sum = state
    W.store(out, maximum)
    W.store(out + 1, scaled_sum)
