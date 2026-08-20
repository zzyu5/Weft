from __future__ import annotations

import weft
from weft.language import View, f32, i32
from weft.std import topk


@weft.kernel
def top_k_f32(
    X: View[f32, (N,)],
    out: View[i32, (K_,)],
):
    topk(X, out)
