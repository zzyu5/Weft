from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_q5_k(
    W: wl.View[ggml.Q5_K, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    with wl.L.tiles(N, extent=wl.auto("NC")) as nc:
        wp = wl.materialize(wl.admit(W[nc]))
        with wl.L.tiles(M, extent=wl.auto("MC")) as mc:
            xp = wl.materialize(wl.admit(Xq[mc]))
            with wl.L.cols(nc, group=wl.auto("NR")) as nb:
                with wl.L.rows(mc, group=wl.auto("MR")) as mb:
                    acc = wl.new(wl.f32, [MR, NR], init=wl.f32(0.0))
                    with wl.L.blocks(K, extent=256) as kb:
                        w = wp[nb, kb]
                        x = xp[mb, kb]
                        integer_acc = wl.new(wl.i32, [MR, NR], init=wl.i32(0))
                        with wl.L.subs(kb, extent=32) as sub:
                            q = wl.u8(w.q[:, sub]) | wl.u8(w.qh[:, sub]) << wl.u8(4)
                            integer = wl.outer_contract(
                                x.q[:, sub], q, over="k", acc=wl.i32
                            )
                            integer_acc += integer * wl.i32(w.sc[:, sub])
                        min_term = wl.outer_contract(
                            wl.fold2(x.bsum), w.m, over="k", acc=wl.i32
                        )
                        acc += wl.f32(x.ds) * (
                            wl.f32(w.d) * wl.widen(integer_acc, wl.f32)
                            - wl.f32(w.dmin) * wl.widen(min_term, wl.f32)
                        )
                    wl.commit(acc, Y[mb, nb])
