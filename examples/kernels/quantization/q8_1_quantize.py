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
    i16,
    narrow,
    reduce,
    widen,
)

from quantization.encodings import Q8_1


def _quantize_row_q8_1(X: View[f32, (K,)], Y: View[Q8_1, (K,)]):
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
        qsum = reduce(widen(q, i16), op="add")
        commit(f16(d), Y[kb].d)
        commit(f16(f32(qsum) * d), Y[kb].s)
        commit(q, Y[kb].q)


@weft.kernel
def q8_1_quantize(X: View[f32, (K,)], Y: View[Q8_1, (K,)]):
    _quantize_row_q8_1(X, Y)
