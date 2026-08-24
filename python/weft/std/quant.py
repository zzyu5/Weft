from __future__ import annotations

from weft.language import (
    L,
    View,
    admit,
    commit,
    contract,
    dot,
    f32,
    fold2,
    i16,
    i32,
    mac_groups,
    mac_pairs,
    new,
    reduce,
    widen,
)

from .encodings import Q4K_I, Q8_K


def q4k_gemv(
    W: View[Q4K_I[16], (M, K)],
    X: View[Q8_K, (K,)],
    Y: View[f32, (M,)],
):
    with L.rows(M, group=16) as mb:
        f32_acc = new(f32, [16], init=0)
        with L.blocks(K, extent=256) as kb:
            w = admit(W[mb, kb])
            x = admit(X[kb])
            i32_acc = new(i32, [16], init=0)
            with L.subs(extent=32) as s:
                p16 = mac_pairs(w.q[s], x.q[s], into=i16)
                i32_acc += reduce(widen(p16, i32)) * w.sc[s]
            mins = fold2(x.bsum)
            min_term = dot(w.m, mins)
            f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)
        commit(f32_acc, Y[mb])


def q4k_gemv_groups4(
    W: View[Q4K_I[16], (M, K)],
    X: View[Q8_K, (K,)],
    Y: View[f32, (M,)],
):
    with L.rows(M, group=16) as mb:
        f32_acc = new(f32, [16], init=0)
        with L.blocks(K, extent=256) as kb:
            w = admit(W[mb, kb])
            x = admit(X[kb])
            i32_acc = new(i32, [16], init=0)
            with L.subs(extent=32) as s:
                p16 = mac_groups(w.q[s], x.q[s], n=4, into=i16)
                i32_acc += reduce(widen(p16, i32)) * w.sc[s]
            mins = fold2(x.bsum)
            min_term = dot(w.m, mins)
            f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)
        commit(f32_acc, Y[mb])


def q4k_gemv_contract(
    W: View[Q4K_I[16], (M, K)],
    X: View[Q8_K, (K,)],
    Y: View[f32, (M,)],
):
    with L.rows(M, group=16) as mb:
        f32_acc = new(f32, [16], init=0)
        with L.blocks(K, extent=256) as kb:
            w = admit(W[mb, kb])
            x = admit(X[kb])
            i32_acc = new(i32, [16], init=0)
            with L.subs(extent=32) as s:
                p32 = contract(w.q[s], x.q[s], over="k", acc=i32)
                i32_acc += p32 * w.sc[s]
            mins = fold2(x.bsum)
            min_term = dot(w.m, mins)
            f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)
        commit(f32_acc, Y[mb])
