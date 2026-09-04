from __future__ import annotations

import weft
from weft.language import (
    L,
    View,
    admit,
    commit,
    dot,
    f32,
    fold2,
    i16,
    i32,
    mac_pairs,
    new,
    reduce,
    widen,
)

from quantization.encodings import Q4K_I, Q8_K


@weft.kernel
def q4_k_q8_k_gemv(
    W: View[Q4K_I[16], (M, K)],
    X: View[Q8_K, (K,)],
    Y: View[f32, (M,)],
):
    with L.rows(M, group=16) as mb:
        f32_acc = new(f32, [16], init=0)
        with L.blocks(K, extent=256) as kb:
            w = admit(W[mb, kb])
            x = admit(X[kb])
            i32_acc = new(i32, [16], init=0)
            with L.subs(extent=32) as s:
                p16 = mac_pairs(w.q[s], x.q[s], into=i16)
                i32_acc += reduce(widen(p16, i32)) * w.sc[s]
            mins = fold2(x.bsum)
            min_term = dot(w.m, mins)
            f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)
        commit(f32_acc, Y[mb])
