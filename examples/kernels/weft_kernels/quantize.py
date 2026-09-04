from __future__ import annotations

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
    i32,
    narrow,
    reduce,
    widen,
)

from .encodings import Q8_0, Q8_1, Q8_K


def quantize_row_q8_0(X: View[f32, (K,)], Y: View[Q8_0, (K,)]):
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


def quantize_q8_0(X: View[f32, (M, K)], Y: View[Q8_0, (M, K)]):
    for row in range(M):
        quantize_row_q8_0(X[row], Y[row])


def quantize_row_q8_1(X: View[f32, (K,)], Y: View[Q8_1, (K,)]):
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


def quantize_q8_1(X: View[f32, (M, K)], Y: View[Q8_1, (M, K)]):
    for row in range(M):
        quantize_row_q8_1(X[row], Y[row])


def quantize_row_q8_K(X: View[f32, (K,)], Y: View[Q8_K, (K,)]):
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


def quantize_q8_K(X: View[f32, (M, K)], Y: View[Q8_K, (M, K)]):
    for row in range(M):
        quantize_row_q8_K(X[row], Y[row])
