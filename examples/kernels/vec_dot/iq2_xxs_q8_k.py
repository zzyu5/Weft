from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.IQ2_XXS, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (256, 8)],
    signs: wl.View[ggml.I8X8, (128, 8)],
):
    sumf = wl.f32(0.0)
    grid_values = grid.values
    sign_values = signs.values
    with wl.L.blocks(K, extent=256) as kb:
        w = wl.admit(W[kb])
        x = wl.admit(X[kb])
        group = wl.iota(8, dtype=wl.u32, axis="group")
        entry = wl.iota(4, dtype=wl.u32, axis="entry")
        payload = wl.iota(8, dtype=wl.u16, axis="payload")
        group_byte = group * wl.u32(8)
        grid_index = wl.widen(w.q[group_byte + entry], wl.u32)
        word1 = wl.widen(w.q[group_byte + wl.u32(4)], wl.u32) | wl.widen(
            w.q[group_byte + wl.u32(5)], wl.u32
        ) << wl.u32(8)
        word1 = word1 | wl.widen(w.q[group_byte + wl.u32(6)], wl.u32) << wl.u32(16)
        word1 = word1 | wl.widen(w.q[group_byte + wl.u32(7)], wl.u32) << wl.u32(24)
        sign_index = word1 >> entry * wl.u32(7) & wl.u32(127)
        weight = wl.lookup(
            grid_values, grid_index * wl.u32(8) + payload, bounds="in_bounds"
        )
        sign = wl.lookup(
            sign_values, sign_index * wl.u32(8) + payload, bounds="in_bounds"
        )
        activation = x.q[group * wl.u32(32) + entry * wl.u32(8) + wl.u32(payload)]
        partial = wl.contract(
            activation, weight * sign, over=("entry", "payload"), acc=wl.i32
        )
        scale = wl.i32((word1 >> wl.u32(28)) * wl.u32(2) + wl.u32(1))
        block_sum = wl.reduce(partial * scale, axis="group")
        sumf += wl.f32(w.d) * wl.f32(x.ds) * wl.f32(block_sum)
    return wl.f32(0.125) * sumf


@weft.kernel
def quantized_vec_dot_iq2_xxs_q8_k(
    W: wl.View[ggml.IQ2_XXS, (K,)],
    X: wl.View[ggml.Q8_K, (K,)],
    grid: wl.View[ggml.I8X8, (256, 8)],
    signs: wl.View[ggml.I8X8, (128, 8)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.commit(compute(W, X, grid, signs), Y[0])
