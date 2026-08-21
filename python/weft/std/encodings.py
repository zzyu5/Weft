from __future__ import annotations

import weft
from weft.language import (
    View,
    bitorder,
    byteorder,
    f16,
    f32,
    grouped,
    i8,
    i16,
    interleave,
    joined,
    layered,
    lo_first,
    u4,
    u6,
)


@weft.encoding
class Q4_K:
    layout = bitorder.lsb_first, byteorder.little
    d: f16
    dmin: f16
    sc: u6[8] @ joined(4, 2, 4, lo_first)
    m: u6[8] @ joined(4, 2, 4, lo_first)
    q: u4[256] @ grouped(64) @ layered(32, lo_first)


@weft.encoding
class Q4_0:
    layout = bitorder.lsb_first, byteorder.little
    d: f16
    q: u4[32] @ grouped(32) @ layered(16, lo_first)


@weft.encoding
class Q8_0:
    layout = bitorder.lsb_first, byteorder.little
    d: f16
    q: i8[32]


@weft.encoding
class Q8_1:
    layout = bitorder.lsb_first, byteorder.little
    d: f16
    s: f16
    q: i8[32]


@weft.encoding
class Q8_K:
    layout = bitorder.lsb_first, byteorder.little
    ds: f32
    q: i8[256]
    bsum: i16[16]


@weft.derive
def Q4K_I16(W: View[Q4_K, (M, K)]) -> View[Q4K_I16, (M, K)]:
    return interleave(W, rows=16)
