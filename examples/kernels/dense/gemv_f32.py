from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel
def gemv_f32(
    W: wl.View[wl.f32, (M, K)], X: wl.View[wl.f32, (K,)], Y: wl.View[wl.f32, (M,)]
):
    with wl.L.rows(M, group=wl.auto("MR")) as mb:
        acc = wl.new(wl.f32, [MR], init=0)
        with wl.L.blocks(K, extent=wl.auto("KB")) as kb:
            x = wl.admit(X[kb])
            acc += wl.contract(wl.admit(W[mb, kb]), x, over="k")
        wl.commit(acc, Y[mb])
