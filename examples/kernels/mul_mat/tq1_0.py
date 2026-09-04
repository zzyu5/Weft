from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_tq1_0(
    W: wl.View[ggml.TQ1_0, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    powers: wl.View[wl.u32, (5,)],
    Y: wl.View[wl.f32, (M, N)],
):
    power_table = wl.materialize(wl.admit(powers))
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
                        integer = wl.new(wl.i32, [MR, NR], init=wl.i32(0))
                        lane0 = wl.iota(32, axis="k")
                        digit0 = wl.iota(5, dtype=wl.u32, axis="digit0")
                        q0 = qf.radix3_digit_i8(power_table, w.q[:, lane0], digit0)
                        partial0 = wl.contract(
                            x.q[:, digit0 * wl.u32(32) + lane0],
                            q0,
                            over="k",
                            acc=wl.i32,
                        )
                        integer += wl.reduce(partial0, axis="digit0")
                        lane1 = wl.iota(16, axis="k")
                        digit1 = wl.iota(5, dtype=wl.u32, axis="digit1")
                        q1 = qf.radix3_digit_i8(
                            power_table, w.q[:, wl.u32(32) + lane1], digit1
                        )
                        partial1 = wl.contract(
                            x.q[:, wl.u32(160) + digit1 * wl.u32(16) + lane1],
                            q1,
                            over="k",
                            acc=wl.i32,
                        )
                        integer += wl.reduce(partial1, axis="digit1")
                        lane2 = wl.iota(16, axis="k")
                        q2 = qf.radix3_digit_i8(
                            power_table, w.qh[:, lane2 // wl.u32(4)], lane2 % wl.u32(4)
                        )
                        integer += wl.outer_contract(
                            x.q[
                                :,
                                wl.u32(240)
                                + lane2 % wl.u32(4) * wl.u32(4)
                                + lane2 // wl.u32(4),
                            ],
                            q2,
                            over="k",
                            acc=wl.i32,
                        )
                        acc += wl.f32(w.d) * wl.f32(x.ds) * wl.widen(integer, wl.f32)
                    wl.commit(acc, Y[mb, nb])
