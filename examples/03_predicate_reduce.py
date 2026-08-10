import weft
import weft.language as W


@W.pure
def token_index(i):
    return i


@W.pure
def address(i):
    return i


@weft.kernel
def predicate_reduce(
    x: W.ptr[W.f32, W.readonly],
    out: W.ptr[W.f32, W.writeonly],
    begin: W.index,
    end: W.index,
    sequence_length: W.index,
) -> None:
    with W.vla(begin, end) as i:
        active = token_index(i) < sequence_length
        value = W.load(x + address(i), where=active)
        maximum = W.reduce(
            value,
            op="max",
            identity=W.neg_inf(W.f32),
            order="relaxed",
        )
    W.store(out, maximum)
