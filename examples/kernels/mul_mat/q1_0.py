from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_0 as quant_q8_0
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_q1_0(
    W: wl.View[ggml.Q1_0, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_0, (M, K)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_0.quantize_matrix(X, Xq)
    with wl.L.tiles(N, extent=wl.auto("NC")) as nc:
        wp = wl.materialize(wl.admit(W[nc]))
        with wl.L.tiles(M, extent=wl.auto("MC")) as mc:
            xp = wl.materialize(wl.admit(Xq[mc]))
            with wl.L.cols(nc, group=wl.auto("NR")) as nb:
                with wl.L.rows(mc, group=wl.auto("MR")) as mb:
                    acc = wl.new(wl.f32, [MR, NR], init=wl.f32(0.0))
                    with wl.L.blocks(K, extent=128) as wb:
                        w = wp[nb, wb]
                        with wl.L.subs(wb, extent=32) as xb:
                            x = xp[mb, xb]
                            centered = wl.i8(w.q[:, xb]) * wl.i8(2) - wl.i8(1)
                            integer = wl.outer_contract(
                                x.q, centered, over="k", acc=wl.i32
                            )
                            acc += wl.f32(w.d) * wl.f32(x.d) * wl.widen(integer, wl.f32)
                    wl.commit(acc, Y[mb, nb])
