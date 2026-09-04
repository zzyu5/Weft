from __future__ import annotations

import formats.ggml as ggml
import functions.quantization as qf
import weft
import weft.language as wl


def compute(
    W: wl.View[ggml.IQ4_NL, (K,)],
    X: wl.View[ggml.Q8_0, (K,)],
    codebook: wl.View[wl.i8, (16,)],
):
    table = wl.materialize(wl.admit(codebook))
    result = wl.f32(0.0)
    with wl.L.blocks(K, extent=32) as kb:
        w = wl.admit(W[kb])
        x = wl.admit(X[kb])
        integer = qf.dot_codebook32(w.q, x.q, table)
        result += wl.f32(x.d) * wl.f32(w.d) * wl.f32(integer)
    return result


@weft.kernel
def quantized_vec_dot_iq4_nl_q8_0(
    W: wl.View[ggml.IQ4_NL, (K,)],
    X: wl.View[ggml.Q8_0, (K,)],
    codebook: wl.View[wl.i8, (16,)],
    Y: wl.View[wl.f32, (1,)],
):
    wl.commit(compute(W, X, codebook), Y[0])
