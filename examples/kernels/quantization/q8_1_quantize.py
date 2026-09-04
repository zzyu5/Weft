from __future__ import annotations

import weft
from weft.language import View, f32
from weft_kernels import Q8_1, quantize_row_q8_1


@weft.kernel
def q8_1_quantize(
    X: View[f32, (K,)],
    Y: View[Q8_1, (K,)],
):
    quantize_row_q8_1(X, Y)
