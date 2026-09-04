from __future__ import annotations

from math import inf

import weft
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
    narrow,
    new,
    reduce,
)


def _rowmax(value):
    return reduce(value, op="max", axis="tk")


def _rowsum(value):
    return reduce(value, op="add", axis="tk")


@weft.kernel
def online_flash_attention(
    Q: View[f16, (Tq, D)],
    K: View[f16, (Tk, D)],
    V: View[f16, (Tk, D)],
    O: View[f16, (Tq, D)],
):
    with L.rows(Tq, group=auto("BQ")) as qb:
        q = materialize(admit(Q[qb, :]))
        m = new(f32, [BQ], init=-inf)
        l = new(f32, [BQ], init=0)
        o = new(f32, [BQ, D], init=0)
        with L.blocks(Tk, extent=auto("BK")) as kb:
            k = admit(K[kb, :])
            v = admit(V[kb, :])
            s = contract(q, k, over="d", acc=f32)
            m_new = maximum(m, _rowmax(s))
            p = exp(s - m_new)
            alpha = exp(m - m_new)
            l = l * alpha + _rowsum(p)
            o = o * alpha + contract(p, v, over="tk")
            m = m_new
        commit(
            narrow(o / l, f16, rounding="rne", saturation=False), O[qb, :]
        )
