from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.IQ1_S, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (2048, 8)],
):
    result = wl.f32(0.0)
    grid_values = grid.values
    with wl.L.blocks(K, extent=256) as kb:
        w = wl.admit(W[kb])
        x = wl.admit(X[kb])
        entry = wl.iota(4, dtype=wl.u32, axis="entry")
        payload = wl.iota(8, dtype=wl.u16, axis="payload")
        main = wl.i32(0)
        correction = wl.i32(0)
        group_index = wl.index(0)
        with wl.L.subs(kb, extent=32) as group:
            metadata = wl.widen(w.qh[group_index], wl.u32)
            scale = wl.i32(
                (metadata >> wl.u32(12) & wl.u32(7)) * wl.u32(2) + wl.u32(1)
            )
            grid_index = wl.widen(
                w.q[group_index * wl.index(4) + entry], wl.u32
            ) | (metadata >> entry * wl.u32(3) & wl.u32(7)) << wl.u32(8)
            weight = wl.lookup(
                grid_values, grid_index * wl.u32(8) + payload, bounds="in_bounds"
            )
            activation = x.q[
                group_index * wl.index(32) + entry * wl.u32(8) + wl.u32(payload)
            ]
            main += wl.contract(
                activation, weight, over=("entry", "payload"), acc=wl.i32
            ) * scale
            delta = wl.i32(1) - wl.i32(
                metadata >> wl.u32(15) & wl.u32(1)
            ) * wl.i32(2)
            correction += (
                scale
                * delta
                * (
                    wl.i32(x.bsum[group_index * wl.index(2)])
                    + wl.i32(x.bsum[group_index * wl.index(2) + wl.index(1)])
                )
            )
            group_index += wl.index(1)
        combined = wl.f32(main) + wl.f32(0.125) * wl.f32(correction)
        result += wl.f32(w.d) * wl.f32(x.ds) * combined
    return result


@weft.kernel
def quantized_vec_dot_iq1_s_q8_k(
    W: wl.View[ggml.IQ1_S, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (2048, 8)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.commit(compute(W, X, grid), Y[0])
