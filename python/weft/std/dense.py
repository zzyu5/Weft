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
    pack,
    transfer,
    wide,
)


def gemm(
    A: View[f32, (M, K)],
    B: View[f32, (K, N)],
    C: View[f32, (M, N)],
):
    with L.tiles(N, extent=auto("NC")) as nc:
        with L.tiles(K, extent=auto("KC")) as kc:
            Bp = materialize(pack(B[kc, nc], along="k")) @ transfer
            with L.tiles(M, extent=auto("MC")) as mc:
                Ap = materialize(pack(A[mc, kc], along="k")) @ transfer
                with L.rows(mc, group=auto("MR")) as mb:
                    with L.cols(nc, group=auto("NR")) as nb:
                        acc = new(f32, [MR, NR], init=admit(C[mb, nb]))
                        with L.blocks(kc, extent=auto("KB")) as kb:
                            a = admit(Ap[mb, kb]) @ transfer
                            b = admit(Bp[kb, nb]) @ transfer
                            acc += outer_contract(a, b, over="k") @ wide
                        commit(acc, C[mb, nb])


def gemv(
    W: View[f32, (M, K)],
    X: View[f32, (K,)],
    Y: View[f32, (M,)],
):
    with L.rows(M, group=auto("MR")) as mb:
        acc = new(f32, [MR], init=0)
        with L.blocks(K, extent=auto("KB")) as kb:
            x = admit(X[kb]) @ transfer
            acc += contract(admit(W[mb, kb]), x, over="k") @ wide
        commit(acc, Y[mb])
