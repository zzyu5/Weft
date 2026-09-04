from __future__ import annotations

import weft
from weft.language import (
    L,
    View,
    abs as absolute,
    admit,
    commit,
    f32,
    i8,
    i16,
    narrow,
    reduce,
    widen,
)

from quantization.encodings import Q8_K


def _quantize_row_q8_K(X: View[f32, (K,)], Y: View[Q8_K, (K,)]):
    with L.blocks(K, extent=256) as kb:
        block = admit(X[kb])
        maximum = reduce(block, op="max")
        minimum = reduce(block, op="min")
        extreme = minimum
        if absolute(maximum) > absolute(minimum):
            extreme = maximum

        inverse = f32(0.0)
        d = f32(0.0)
        if extreme != f32(0.0):
            inverse = f32(-127.0) / extreme
            d = f32(1.0) / inverse

        with L.subs(kb, extent=64) as chunk:
            group_values = admit(X[chunk])
            q = narrow(
                group_values * inverse,
                i8,
                rounding="rne",
                saturation=True,
            )
            commit(q, Y[kb].q[chunk])
            with L.subs(chunk, extent=16) as group:
                bsum = reduce(widen(q[group], i16), op="add")
                commit(i16(bsum), Y[kb].bsum[group])

        commit(d, Y[kb].ds)


@weft.kernel
def q8_K_quantize(X: View[f32, (K,)], Y: View[Q8_K, (K,)]):
    _quantize_row_q8_K(X, Y)
