from __future__ import annotations

from math import inf

from weft.language import (
    L,
    View,
    admit,
    auto,
    commit,
    contract,
    exp,
    f16,
    f32,
    materialize,
    maximum,
    new,
    reduce,
    transfer,
    wide,
)


def _rowmax(value):
    return reduce(value, op="max", axis="tk")


def _rowsum(value):
    return reduce(value, op="add", axis="tk")


def flash_attention(
    Q: View[f16, (Tq, D)],
    K: View[f16, (Tk, D)],
    V: View[f16, (Tk, D)],
    O: View[f16, (Tq, D)],
):
    with L.rows(Tq, group=auto("BQ")) as qb:
        q = materialize(admit(Q[qb, :])) @ transfer
        m = new(f32, [BQ], init=-inf)
        l = new(f32, [BQ], init=0)
        o = new(f32, [BQ, D], init=0)
        with L.blocks(Tk, extent=auto("BK")) as kb:
            k = admit(K[kb, :]) @ transfer
            v = admit(V[kb, :]) @ transfer
            s = contract(q, k, over="d", acc=f32) @ wide
            m_new = maximum(m, _rowmax(s)) @ wide
            p = exp(s - m_new) @ wide
            alpha = exp(m - m_new) @ wide
            l = l * alpha + _rowsum(p) @ wide
            o = o * alpha + contract(p, v, over="tk") @ wide
            m = m_new
        commit(o / l, O[qb, :])
