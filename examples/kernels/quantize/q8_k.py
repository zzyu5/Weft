from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


def quantize_row(X: wl.View[wl.f32, (K,)], Y: wl.View[ggml.Q8_K, (K,)]):
    with wl.L.blocks(K, extent=256) as kb:
        block = wl.admit(X[kb])
        maximum = wl.reduce(block, op="max")
        minimum = wl.reduce(block, op="min")
        extreme = minimum
        if wl.abs(maximum) > wl.abs(minimum):
            extreme = maximum
        inverse = wl.f32(0.0)
        d = wl.f32(0.0)
        if extreme != wl.f32(0.0):
            inverse = wl.f32(-127.0) / extreme
            d = wl.f32(1.0) / inverse
        with wl.L.subs(kb, extent=64) as chunk:
            group_values = wl.admit(X[chunk])
            q = wl.narrow(
                group_values * inverse, wl.i8, rounding="rne", saturation=True
            )
            wl.commit(q, Y[kb].q[chunk])
            with wl.L.subs(chunk, extent=16) as group:
                bsum = wl.reduce(wl.widen(q[group], wl.i16), op="add")
                wl.commit(wl.i16(bsum), Y[kb].bsum[group])
        wl.commit(d, Y[kb].ds)


def quantize_matrix(X: wl.View[wl.f32, (M, K)], Y: wl.View[ggml.Q8_K, (M, K)]):
    for row in range(M):
        quantize_row(X[row], Y[row])


@weft.kernel
def q8_K_quantize(X: wl.View[wl.f32, (K,)], Y: wl.View[ggml.Q8_K, (K,)]):
    quantize_row(X, Y)
