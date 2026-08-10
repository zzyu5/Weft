import weft
import weft.language as W


@weft.kernel
def block_scaled_dot(
    packed_a_ptr: W.ptr[W.i8, W.readonly],
    packed_b_ptr: W.ptr[W.i8, W.readonly],
    scale_a_ptr: W.ptr[W.f32, W.readonly],
    scale_b_ptr: W.ptr[W.f32, W.readonly],
    out: W.ptr[W.f32, W.writeonly],
    K: W.constexpr[W.index],
) -> None:
    k = W.block_axis(K)
    packed_a = W.load(packed_a_ptr + k)
    packed_b = W.load(packed_b_ptr + k)
    scale_a = W.load(scale_a_ptr, other=W.f32(0.0))
    scale_b = W.load(scale_b_ptr, other=W.f32(0.0))
    init = W.f32(0.0)
    acc = W.block_scaled_contract(
        packed_a,
        packed_b,
        scale_a=scale_a,
        scale_b=scale_b,
        init=init,
        rounding="rne",
        saturation=True,
    )
    W.store(out, acc)
