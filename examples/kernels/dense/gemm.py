from __future__ import annotations

import weft
from weft.language import View, f32
from weft.std import gemm


@weft.kernel
def gemm_f32(
    A: View[f32, (M, K)],
    B: View[f32, (K, N)],
    C: View[f32, (M, N)],
):
    gemm(A, B, C)
