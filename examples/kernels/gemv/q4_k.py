from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


@weft.kernel
def q4_k_q8_k_gemv(
    W: wl.View[ggml.Q4K_I[16], (M, K)],
    X: wl.View[ggml.Q8_K, (K,)],
    Y: wl.View[wl.f32, (M,)],
):
    with wl.L.rows(M, group=16) as mb:
        f32_acc = wl.new(wl.f32, [16], init=0)
        with wl.L.blocks(K, extent=256) as kb:
            w = wl.admit(W[mb, kb])
            x = wl.admit(X[kb])
            i32_acc = wl.new(wl.i32, [16], init=0)
            with wl.L.subs(extent=32) as s:
                p16 = wl.mac_pairs(w.q[s], x.q[s], into=wl.i16)
                i32_acc += wl.reduce(wl.widen(p16, wl.i32)) * w.sc[s]
            mins = wl.fold2(x.bsum)
            min_term = wl.dot(w.m, mins)
            f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)
        wl.commit(f32_acc, Y[mb])
