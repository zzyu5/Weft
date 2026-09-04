from __future__ import annotations

import weft
import weft.language as wl


@weft.kernel
def ime_i8_contract(
    A: wl.View[wl.i8, (M, K)], B: wl.View[wl.i8, (K, N)], C: wl.View[wl.i32, (M, N)]
):
    with wl.L.rows(M, group=wl.auto("MR")) as mb:
        with wl.L.cols(N, group=wl.auto("NR")) as nb:
            acc = wl.new(wl.i32, [MR, NR], init=0)
            with wl.L.blocks(K, extent=wl.auto("KB")) as kb:
                acc += wl.outer_contract(
                    wl.admit(A[mb, kb]), wl.admit(B[kb, nb]), over="k", acc=wl.i32
                )
            wl.commit(acc, C[mb, nb])
