from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


@weft.kernel
def row_dequantize_nvfp4(
    W: wl.View[ggml.NVFP4, (K,)],
    codebook: wl.View[wl.i8, (16,)],
    scale: wl.View[wl.f32, (256,)],
    Y: wl.View[wl.f32, (K,)],
):
    table = wl.materialize(wl.admit(codebook))
    with wl.L.blocks(K, extent=64) as kb:
        w = wl.admit(W[kb])
        sub_index = wl.index(0)
        with wl.L.subs(kb, extent=16) as sub:
            decoded_scale = qf.exponent_scale(scale, w.d[sub_index])
            q = qf.small_nonlinear_lookup(table, w.q[sub])
            wl.commit(qf.fp4_codebook(q, decoded_scale), Y[kb][sub])
            sub_index += wl.index(1)
