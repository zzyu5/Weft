from __future__ import annotations

import weft
from weft.language import View, f32
from weft_kernels import gemv


@weft.kernel
def gemv_f32(
    W: View[f32, (M, K)],
    X: View[f32, (K,)],
    Y: View[f32, (M,)],
):
    gemv(W, X, Y)
