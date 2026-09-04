from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_iq4_xs(
    W: wl.View[ggml.IQ4_XS, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    codebook: wl.View[wl.i8, (16,)],
    Y: wl.View[wl.f32, (M, N)],
):
    table = wl.materialize(wl.admit(codebook))
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
                        sub_index = wl.index(0)
                        with wl.L.subs(kb, extent=32) as sub:
                            low = wl.u32(w.scales_l[:, sub_index // wl.index(2)])
                            shift = sub_index % wl.index(2) * wl.index(4)
                            scale_low = low >> wl.u32(shift) & wl.u32(15)
                            scale_high = wl.u32(w.scales_h) >> wl.u32(
                                sub_index * wl.index(2)
                            ) & wl.u32(3)
                            scale = wl.i32(
                                scale_low | scale_high << wl.u32(4)
                            ) - wl.i32(32)
                            q = wl.lookup(table, wl.u8(w.q[:, sub]), bounds="in_bounds")
                            integer = wl.outer_contract(
                                x.q[:, sub], q, over="k", acc=wl.i32
                            )
                            acc += (
                                wl.f32(w.d)
                                * wl.f32(x.ds)
                                * wl.widen(scale, wl.f32)
                                * wl.widen(integer, wl.f32)
                            )
                            sub_index += wl.index(1)
                    wl.commit(acc, Y[mb, nb])
