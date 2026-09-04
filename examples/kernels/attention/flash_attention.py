from __future__ import annotations

from math import inf

import weft
import weft.language as wl


def _rowmax(value):
    return wl.reduce(value, op="max", axis="tk")


def _rowsum(value):
    return wl.reduce(value, op="add", axis="tk")


@weft.kernel
def online_flash_attention(
    Q: wl.View[wl.f16, (Tq, D)],
    K: wl.View[wl.f16, (Tk, D)],
    V: wl.View[wl.f16, (Tk, D)],
    O: wl.View[wl.f16, (Tq, D)],
):
    with wl.L.rows(Tq, group=wl.auto("BQ")) as qb:
        q = wl.materialize(wl.admit(Q[qb, :]))
        m = wl.new(wl.f32, [BQ], init=-inf)
        l = wl.new(wl.f32, [BQ], init=0)
        o = wl.new(wl.f32, [BQ, D], init=0)
        with wl.L.blocks(Tk, extent=wl.auto("BK")) as kb:
            k = wl.admit(K[kb, :])
            v = wl.admit(V[kb, :])
            s = wl.contract(q, k, over="d", acc=wl.f32)
            m_new = wl.maximum(m, _rowmax(s))
            p = wl.exp(s - m_new)
            alpha = wl.exp(m - m_new)
            l = l * alpha + _rowsum(p)
            o = o * alpha + wl.contract(p, v, over="tk")
            m = m_new
        wl.commit(wl.narrow(o / l, wl.f16, rounding="rne", saturation=False), O[qb, :])
