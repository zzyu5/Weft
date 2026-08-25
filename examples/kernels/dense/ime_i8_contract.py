from __future__ import annotations

import weft
from weft.language import L, View, admit, auto, commit, i8, i32, new, outer_contract


@weft.kernel
def ime_i8_contract(
    A: View[i8, (M, K)],
    B: View[i8, (K, N)],
    C: View[i32, (M, N)],
):
    with L.rows(M, group=auto("MR")) as mb:
        with L.cols(N, group=auto("NR")) as nb:
            acc = new(i32, [MR, NR], init=0)
            with L.blocks(K, extent=auto("KB")) as kb:
                acc += outer_contract(
                    admit(A[mb, kb]), admit(B[kb, nb]), over="k", acc=i32
                )
            commit(acc, C[mb, nb])
