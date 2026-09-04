from __future__ import annotations

import weft
from weft.language import View, f32
from weft_kernels import Q4K_I, Q8_K, q4k_gemv


@weft.kernel
def q4_k_q8_k_gemv(
    W: View[Q4K_I[16], (M, K)],
    X: View[Q8_K, (K,)],
    Y: View[f32, (M,)],
):
    q4k_gemv(W, X, Y)
