from __future__ import annotations

import weft
from weft.language import L, View, admit, auto, commit, contract, f32


@weft.kernel
def gemm_f32(
    X: View[f32, (M, K)],
    W: View[f32, (N, K)],
    Y: View[f32, (M, N)],
):
    with L.tiles(N, extent=auto("NC")) as nc:
        with L.tiles(M, extent=auto("MC")) as mc:
            with L.cols(nc, group=auto("NR")) as nb:
                with L.rows(mc, group=auto("MR")) as mb:
                    product = contract(
                        admit(X[mb]), admit(W[nb]), over="k"
                    )
                    commit(product, Y[mb, nb])
