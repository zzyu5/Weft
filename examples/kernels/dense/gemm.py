from __future__ import annotations

import weft
from weft.language import View, f32
from weft_kernels import gemm


@weft.kernel
def gemm_f32(
    X: View[f32, (M, K)],
    W: View[f32, (N, K)],
    Y: View[f32, (M, N)],
):
    gemm(X, W, Y)
