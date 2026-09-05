from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_iq1_s(
    W: wl.View[ggml.IQ1_S, (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    grid: wl.View[ggml.I8X8, (2048, 8)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    grid_values = grid.values
    with wl.level.tiles(N, extent=wl.auto("NC")) as nc:
        wp = wl.stage(wl.load(W[nc]))
        with wl.level.tiles(M, extent=wl.auto("MC")) as mc:
            xp = wl.stage(wl.load(Xq[mc]))
            with wl.level.cols(nc, group=wl.auto("NR")) as nb:
                with wl.level.rows(mc, group=wl.auto("MR")) as mb:
                    acc = wl.state(wl.f32, [MR, NR], init=wl.f32(0.0))
                    with wl.level.blocks(K, extent=256) as kb:
                        w = wp[nb, kb]
                        x = xp[mb, kb]
                        entry = wl.arange(0, 4, dtype=wl.u32, axis="entry")
                        payload = wl.arange(0, 8, dtype=wl.u16, axis="payload")
                        main = wl.state(wl.i32, [MR, NR], init=wl.i32(0))
                        correction = wl.state(wl.i32, [MR, NR], init=wl.i32(0))
                        group = wl.arange(0, 8, dtype=wl.u32, axis="group")
                        metadata = wl.widen(w.qh[:, group], wl.u32)
                        scale = wl.i32(
                            (metadata >> wl.u32(12) & wl.u32(7)) * wl.u32(2)
                            + wl.u32(1)
                        )
                        grid_index = wl.widen(
                            w.q[:, group * wl.u32(4) + entry], wl.u32
                        ) | (metadata >> entry * wl.u32(3) & wl.u32(7)) << wl.u32(8)
                        weight = wl.lookup(
                            grid_values,
                            grid_index * wl.u32(8) + payload,
                            bounds="in_bounds",
                        )
                        activation = x.q[
                            :,
                            group * wl.u32(32)
                            + entry * wl.u32(8)
                            + wl.u32(payload),
                        ]
                        group_sum = wl.reduce_dot(
                            activation,
                            weight,
                            over=("entry", "payload"),
                            acc_dtype=wl.i32,
                        )
                        main += wl.reduce(scale * group_sum, axis="group")
                        delta = wl.i32(1) - wl.i32(
                            metadata >> wl.u32(15) & wl.u32(1)
                        ) * wl.i32(2)
                        bsum = wl.i32(x.bsum[:, group * wl.u32(2)]) + wl.i32(
                            x.bsum[:, group * wl.u32(2) + wl.u32(1)]
                        )
                        correction += wl.reduce(bsum * scale * delta, axis="group")
                        combined = wl.f32(main) + wl.f32(0.125) * wl.f32(correction)
                        acc += wl.f32(w.d) * wl.f32(x.ds) * combined
                    wl.store(Y[mb, nb], acc)
