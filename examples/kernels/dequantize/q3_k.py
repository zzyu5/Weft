from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


@weft.kernel
def row_dequantize_q3_k(W: wl.View[ggml.Q3_K, (K,)], Y: wl.View[wl.f32, (K,)]):
    with wl.L.blocks(K, extent=256) as kb:
        w = wl.admit(W[kb])
        with wl.L.subs(kb, extent=16) as sub:
            scale = (
                wl.i32(w.scale_low[sub]) | wl.i32(w.scale_high[sub]) << wl.u32(4)
            ) - wl.i32(32)
            q = wl.i32(w.q[sub]) + wl.i32(w.hmask[sub]) * wl.i32(4) - wl.i32(4)
            wl.commit(wl.f32(w.d) * wl.f32(scale) * wl.f32(q), Y[kb][sub])
