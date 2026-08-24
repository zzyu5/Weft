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


def quantize_q8_0(
    X: View[f32, (M, K)],
    Y: View[Q8_0, (M, K)],
):
    for row in range(M):
        with L.blocks(K, extent=32) as kb:
            x = admit(X[row, kb])
            amax = reduce(absolute(x), op="max")
            d = amax / f32(127.0)
            inverse = f32(0.0)
            if amax != f32(0.0):
                inverse = f32(1.0) / d
            q = narrow(
                x * inverse,
                i8,
                rounding="rne",
                saturation=True,
            )
            commit(f16(d), Y[row, kb].d)
            commit(q, Y[row, kb].q)


def quantize_q8_1(
    X: View[f32, (M, K)],
    Y: View[Q8_1, (M, K)],
):
    for row in range(M):
        with L.blocks(K, extent=32) as kb:
            x = admit(X[row, kb])
            amax = reduce(absolute(x), op="max")
            d = amax / f32(127.0)
            inverse = f32(0.0)
            if amax != f32(0.0):
                inverse = f32(1.0) / d
            q = narrow(
                x * inverse,
                i8,
                rounding="rne",
                saturation=True,
            )
            qsum = reduce(widen(q, i16), op="add")
            commit(f16(d), Y[row, kb].d)
            commit(f16(f32(qsum) * d), Y[row, kb].s)
            commit(q, Y[row, kb].q)


def quantize_q8_K(
    X: View[f32, (M, K)],
    Y: View[Q8_K, (M, K)],
):
    for row in range(M):
        with L.blocks(K, extent=256) as kb:
            block = admit(X[row, kb])
            extreme = f32(0.0)
            magnitude = f32(0.0)
            with L.subs(kb, extent=1) as element:
                sample = f32((admit(X[row, element]))[0])
                sample_magnitude = absolute(sample)
                if sample_magnitude > magnitude:
                    magnitude = sample_magnitude
                    extreme = sample

            inverse = f32(0.0)
            d = f32(0.0)
            if extreme != f32(0.0):
                inverse = f32(-127.0) / extreme
                d = f32(1.0) / inverse

            q = narrow(
                block * inverse,
                i8,
                rounding="rne",
                saturation=True,
            )
            commit(q, Y[row, kb].q)

            with L.subs(kb, extent=16) as group:
                q_group = q[group]
                bsum = reduce(widen(q_group, i16), op="add")
                commit(i16(bsum), Y[row, kb].bsum[group])

            commit(d, Y[row, kb].ds)
