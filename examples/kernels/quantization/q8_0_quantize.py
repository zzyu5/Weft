from __future__ import annotations

import weft
from weft.language import View, f32
from weft.std import Q8_0, quantize_q8_0


@weft.kernel
def q8_0_quantize(
    X: View[f32, (M, K)],
    Y: View[Q8_0, (M, K)],
):
    quantize_q8_0(X, Y)
