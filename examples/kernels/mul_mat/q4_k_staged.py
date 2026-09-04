from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_q4_k_staged(
    W: wl.View[ggml.Q4_K, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        for column in range(N):
            wl.commit(wl.f32(0.0), Y[row, column])
    with wl.L.tiles(N, extent=wl.auto("NC")) as nc:
        with wl.L.tiles(K, extent=wl.auto("KC")) as kc:
            wp = wl.materialize(wl.admit(W[nc, kc]))
            with wl.L.tiles(M, extent=wl.auto("MC")) as mc:
                with wl.L.rows(mc, group=wl.auto("MR")) as mb:
                    with wl.L.cols(nc, group=16) as nb:
                        f32_acc = wl.new(wl.f32, [MR, 16], init=wl.admit(Y[mb, nb]))
                        with wl.L.blocks(kc, extent=256) as kb:
                            w = wp[nb, kb]
                            x = wl.admit(Xq[mb, kb])
                            i32_acc = wl.new(wl.i32, [MR, 16], init=0)
                            with wl.L.subs(extent=32) as s:
                                p16 = wl.mac_groups(x.q[s], w.q[s], n=4, into=wl.i16)
                                i32_acc += wl.reduce(wl.widen(p16, wl.i32)) * w.sc[s]
                            mins = wl.fold2(x.bsum)
                            min_term = wl.outer_contract(
                                mins, w.m, over="k", acc=wl.i32
                            )
                            f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)
                        wl.commit(f32_acc, Y[mb, nb])
