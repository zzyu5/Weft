from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_f16(
    W: wl.View[wl.f16, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xh: wl.View[wl.f16, (M, K)],
    Y: wl.View[wl.f32, (M, N)],
):
    for row in range(M):
        with wl.L.blocks(K, extent=1) as kb:
            value = wl.admit(X[row, kb])
            wl.commit(wl.f16(value), Xh[row, kb])
    for row in range(M):
        for column in range(N):
            wl.commit(wl.f32(0.0), Y[row, column])
    with wl.L.tiles(N, extent=wl.auto("NC")) as nc:
        with wl.L.tiles(K, extent=wl.auto("KC")) as kc:
            wp = wl.materialize(wl.admit(W[nc, kc]))
            with wl.L.tiles(M, extent=wl.auto("MC")) as mc:
                xp = wl.materialize(wl.admit(Xh[mc, kc]))
                with wl.L.rows(mc, group=wl.auto("MR")) as mb:
                    with wl.L.cols(nc, group=wl.auto("NR")) as nb:
                        acc = wl.new(wl.f32, [MR, NR], init=wl.admit(Y[mb, nb]))
                        with wl.L.blocks(kc, extent=wl.auto("KB")) as kb:
                            weights = wp[nb, kb]
                            activations = xp[mb, kb]
                            acc += wl.outer_contract(
                                activations, weights, over="k", acc=wl.f32
                            )
                        wl.commit(acc, Y[mb, nb])
