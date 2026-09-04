from __future__ import annotations

import weft
from weft.language import L, View, admit, auto, commit, contract, f32, new


@weft.kernel
def gemv_f32(
    W: View[f32, (M, K)],
    X: View[f32, (K,)],
    Y: View[f32, (M,)],
):
    with L.rows(M, group=auto("MR")) as mb:
        acc = new(f32, [MR], init=0)
        with L.blocks(K, extent=auto("KB")) as kb:
            x = admit(X[kb])
            acc += contract(admit(W[mb, kb]), x, over="k")
        commit(acc, Y[mb])
