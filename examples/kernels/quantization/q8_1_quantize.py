from __future__ import annotations

import weft
from weft.language import View, f32
from weft.std import Q8_1, quantize_q8_1


@weft.kernel
def q8_1_quantize(
    X: View[f32, (M, K)],
    Y: View[Q8_1, (M, K)],
):
    quantize_q8_1(X, Y)
