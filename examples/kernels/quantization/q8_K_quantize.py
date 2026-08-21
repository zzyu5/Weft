from __future__ import annotations

import weft
from weft.language import View, f32
from weft.std import Q8_K, quantize_q8_K


@weft.kernel
def q8_K_quantize(
    X: View[f32, (M, K)],
    Y: View[Q8_K, (M, K)],
):
    quantize_q8_K(X, Y)
