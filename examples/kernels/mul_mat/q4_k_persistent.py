from __future__ import annotations

import formats.ggml as ggml
import kernels.quantize.q8_k as quant_q8_k
import weft
import weft.language as wl


@weft.kernel
def production_mul_mat_q4_k_persistent(
    W: wl.View[ggml.Q4K_I[16], (N, K)],
    X: wl.View[wl.f32, (M, K)],
    Xq: wl.View[ggml.Q8_K, (M, K)],
    Y: wl.View[wl.f32, (M, N)],
):
    quant_q8_k.quantize_matrix(X, Xq)
    for row in range(M):
        with wl.L.rows(N, group=16) as nb:
            f32_acc = wl.new(wl.f32, [16], init=0)
            with wl.L.blocks(K, extent=256) as kb:
                w = wl.admit(W[nb, kb])
                x = wl.admit(Xq[row, kb])
                i32_acc = wl.new(wl.i32, [16], init=0)
                with wl.L.subs(extent=32) as s:
                    p16 = wl.mac_groups(w.q[s], x.q[s], n=4, into=wl.i16)
                    i32_acc += wl.reduce(wl.widen(p16, wl.i32)) * w.sc[s]
                mins = wl.fold2(x.bsum)
                min_term = wl.dot(w.m, mins)
                f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)
            wl.commit(f32_acc, Y[row, nb])
