from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel
def gemm_f32(
    X: wl.View[wl.f32, (M, K)], W: wl.View[wl.f32, (N, K)], Y: wl.View[wl.f32, (M, N)]
):
    with wl.L.tiles(N, extent=wl.auto("NC")) as nc:
        with wl.L.tiles(M, extent=wl.auto("MC")) as mc:
            with wl.L.cols(nc, group=wl.auto("NR")) as nb:
                with wl.L.rows(mc, group=wl.auto("MR")) as mb:
                    product = wl.contract(wl.admit(X[mb]), wl.admit(W[nb]), over="k")
                    wl.commit(product, Y[mb, nb])
