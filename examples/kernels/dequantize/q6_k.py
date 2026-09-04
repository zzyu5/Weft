from __future__ import annotations

import formats.ggml as ggml
import weft
import weft.language as wl


@weft.kernel
def row_dequantize_q6_k(W: wl.View[ggml.Q6_K, (K,)], Y: wl.View[wl.f32, (K,)]):
    with wl.L.blocks(K, extent=256) as kb:
        w = wl.admit(W[kb])
        with wl.L.subs(kb, extent=16) as sub:
            q = wl.i32(w.ql[sub]) | wl.i32(w.qh[sub]) << wl.u32(4)
            q -= wl.i32(32)
            wl.commit(wl.f32(w.d) * wl.f32(w.scales[sub]) * wl.f32(q), Y[kb][sub])
