from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.IQ3_XXS, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X4, (256, 4)],
    signs: wl.View[ggml.I8X4, (256, 4)],
):
    sumf = wl.f32(0.0)
    grid_values = grid.values
    sign_values = signs.values
    with wl.L.blocks(K, extent=256) as kb:
        w = wl.admit(W[kb])
        x = wl.admit(X[kb])
        half = wl.iota(2, dtype=wl.u32, axis="half")
        entry = wl.iota(4, dtype=wl.u32, axis="entry")
        code = wl.iota(2, dtype=wl.u32, axis="code")
        payload = wl.iota(4, dtype=wl.u16, axis="payload")
        block_sum = wl.i32(0)
        group_index = wl.index(0)
        with wl.L.subs(kb, extent=64) as group:
            metadata = w.metadata[wl.u32(group_index) * wl.u32(2) + half]
            grid_index = wl.widen(
                w.q[
                    wl.u32(group_index) * wl.u32(16)
                    + half * wl.u32(8)
                    + entry * wl.u32(2)
                    + code
                ],
                wl.u32,
            )
            sign_index = metadata >> entry * wl.u32(7) & wl.u32(127)
            sign_entry = sign_index * wl.u32(2) + code
            weight = wl.lookup(
                grid_values, grid_index * wl.u32(4) + payload, bounds="in_bounds"
            )
            sign = wl.lookup(
                sign_values, sign_entry * wl.u32(4) + payload, bounds="in_bounds"
            )
            activation = x.q[
                wl.u32(group_index) * wl.u32(64)
                + half * wl.u32(32)
                + entry * wl.u32(8)
                + code * wl.u32(4)
                + wl.u32(payload)
            ]
            local = wl.contract(
                activation, weight * sign, over=("entry", "code", "payload"), acc=wl.i32
            )
            scale = wl.i32((metadata >> wl.u32(28)) * wl.u32(2) + wl.u32(1))
            block_sum += wl.reduce(local * scale, axis="half")
            group_index += wl.index(1)
        sumf += wl.f32(w.d) * wl.f32(x.ds) * wl.f32(block_sum)
    return wl.f32(0.25) * sumf


@weft.kernel
def quantized_vec_dot_iq3_xxs_q8_k(
    W: wl.View[ggml.IQ3_XXS, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X4, (256, 4)],
    signs: wl.View[ggml.I8X4, (256, 4)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.commit(compute(W, X, grid, signs), Y[0])
