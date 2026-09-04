from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(W: wl.View[ggml.Q2_K, (K,)], X: wl.View[ggml.Q8_K, (K,)]):
    result = wl.f32(0.0)
    with wl.L.blocks(K, extent=256) as kb:
        w = wl.admit(W[kb])
        x = wl.admit(X[kb])
        integer = wl.i32(0)
        with wl.L.subs(kb, extent=16) as sub:
            partial = wl.contract(w.q[sub], x.q[sub], over="k", acc=wl.i32)
            scale = wl.i32(wl.u8(w.scales[sub]) & wl.u8(15))
            integer += partial * scale
        mins = wl.widen(wl.u8(w.scales) >> wl.u8(4), wl.i16)
        minimum = wl.contract(x.bsum, mins, over="k", acc=wl.i32)
        result += wl.f32(x.ds) * (
            wl.f32(w.d) * wl.f32(integer) - wl.f32(w.dmin) * wl.f32(minimum)
        )
    return result


@weft.kernel
def quantized_vec_dot_q2_k_q8_k(
    W: wl.View[ggml.Q2_K, (K,)], X: wl.View[ggml.Q8_K, (K,)], Y: wl.View[wl.f32, (1,)]
):
    wl.commit(compute(W, X), Y[0])
