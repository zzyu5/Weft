import weft
import weft.language as W


@weft.kernel
def add_f32(
    lhs: W.ptr[W.f32, W.readonly, W.noalias],
    rhs: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        W.store(output + i, W.load(lhs + i) + W.load(rhs + i))


@weft.kernel
def sub_f32(
    lhs: W.ptr[W.f32, W.readonly, W.noalias],
    rhs: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        W.store(output + i, W.load(lhs + i) - W.load(rhs + i))


@weft.kernel
def mul_f32(
    lhs: W.ptr[W.f32, W.readonly, W.noalias],
    rhs: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        W.store(output + i, W.load(lhs + i) * W.load(rhs + i))


@weft.kernel
def div_f32(
    lhs: W.ptr[W.f32, W.readonly, W.noalias],
    rhs: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        W.store(output + i, W.load(lhs + i) / W.load(rhs + i))


@weft.kernel
def scale_f32(
    input: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    scale: W.f32,
    begin: W.index,
    end: W.index,
) -> None:
    with W.vla(begin, end) as i:
        W.store(output + i, W.load(input + i) * scale)
