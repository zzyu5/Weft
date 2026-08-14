import weft
import weft.language as W


@weft.kernel
def interleaved_complex_mul_f32(
    lhs: W.ptr[W.f32, W.readonly, W.noalias],
    rhs: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    pairs: W.index,
) -> None:
    with W.vla(0, pairs) as pair:
        left_base = lhs + pair * 2
        right_base = rhs + pair * 2
        left_real = W.load(left_base)
        left_imag = W.load(left_base + 1)
        right_real = W.load(right_base)
        right_imag = W.load(right_base + 1)
        out_base = output + pair * 2
        W.store(out_base, left_real * right_real - left_imag * right_imag)
        W.store(out_base + 1, left_real * right_imag + left_imag * right_real)


@weft.kernel
def interleaved_complex_mul_f32_equivalent(
    lhs: W.ptr[W.f32, W.readonly, W.noalias],
    rhs: W.ptr[W.f32, W.readonly, W.noalias],
    output: W.ptr[W.f32, W.writeonly, W.noalias],
    pairs: W.index,
) -> None:
    with W.vla(0, pairs) as pair:
        left_base = lhs + 2 * pair
        right_base = rhs + 2 * pair
        left_imag = W.load(left_base + 1)
        left_real = W.load(left_base)
        right_imag = W.load(right_base + 1)
        right_real = W.load(right_base)
        real = left_real * right_real - left_imag * right_imag
        imag = left_real * right_imag + left_imag * right_real
        destination = output + 2 * pair
        W.store(destination + 1, imag)
        W.store(destination, real)
