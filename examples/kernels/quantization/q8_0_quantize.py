from __future__ import annotations

import weft
from weft.language import (
    L,
    View,
    abs as absolute,
    admit,
    commit,
    f16,
    f32,
    i8,
    narrow,
    reduce,
)

from quantization.encodings import Q8_0


def _quantize_row_q8_0(X: View[f32, (K,)], Y: View[Q8_0, (K,)]):
    with L.blocks(K, extent=32) as kb:
        x = admit(X[kb])
        amax = reduce(absolute(x), op="max")
        d = amax / f32(127.0)
        inverse = f32(0.0)
        if amax != f32(0.0):
            inverse = f32(1.0) / d
        q = narrow(
            x * inverse,
            i8,
            rounding="dynamic",
            saturation=False,
        )
        commit(f16(d), Y[kb].d)
        commit(q, Y[kb].q)


@weft.kernel
def q8_0_quantize(X: View[f32, (K,)], Y: View[Q8_0, (K,)]):
    _quantize_row_q8_0(X, Y)
