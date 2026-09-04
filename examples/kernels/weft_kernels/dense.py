from __future__ import annotations

from weft.language import (
    L,
    View,
    admit,
    auto,
    commit,
    contract,
    f32,
    new,
    reduce,
)


def gemm(
    X: View[f32, (M, K)],
    W: View[f32, (N, K)],
    Y: View[f32, (M, N)],
):
    with L.tiles(N, extent=auto("NC")) as nc:
        with L.tiles(M, extent=auto("MC")) as mc:
            with L.cols(nc, group=auto("NR")) as nb:
                with L.rows(mc, group=auto("MR")) as mb:
                    product = contract(admit(X[mb]), admit(W[nb]), over="k")
                    commit(product, Y[mb, nb])


def gemv(
    W: View[f32, (M, K)],
    X: View[f32, (K,)],
    Y: View[f32, (M,)],
):
    with L.rows(M, group=auto("MR")) as mb:
        acc = new(f32, [MR], init=0)
        with L.blocks(K, extent=auto("KB")) as kb:
            x = admit(X[kb])
            acc += contract(admit(W[mb, kb]), x, over="k")
        commit(acc, Y[mb])
