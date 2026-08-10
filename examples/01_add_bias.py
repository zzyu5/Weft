import weft
import weft.language as W


@weft.kernel
def add_bias(
    x: W.ptr[W.f32],
    bias: W.ptr[W.f32],
    y: W.ptr[W.f32],
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        xv = W.load(x + i)
        bv = W.load(bias + i)
        W.store(y + i, xv + bv)
