from __future__ import annotations

from weft.language import (
    L,
    View,
    admit,
    auto,
    commit,
    contract,
    f32,
    materialize,
    new,
    outer_contract,
)


def gemm(
    A: View[f32, (M, K)],
    B: View[f32, (K, N)],
    C: View[f32, (M, N)],
):
    with L.tiles(N, extent=auto("NC")) as nc:
        with L.tiles(K, extent=auto("KC")) as kc:
            Bp = materialize(admit(B[kc, nc]))
            with L.tiles(M, extent=auto("MC")) as mc:
                Ap = materialize(admit(A[mc, kc]))
                with L.rows(mc, group=auto("MR")) as mb:
                    with L.cols(nc, group=auto("NR")) as nb:
                        acc = new(f32, [MR, NR], init=admit(C[mb, nb]))
                        with L.blocks(kc, extent=auto("KB")) as kb:
                            a = Ap[mb, kb]
                            b = Bp[kb, nb]
                            acc += outer_contract(a, b, over="k")
                        commit(acc, C[mb, nb])


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
