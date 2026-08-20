from __future__ import annotations

import weft
from weft.language import (
    View,
    bitorder,
    byteorder,
    f16,
    f32,
    i8,
    i16,
    interleave,
    lo_first,
    nibble,
    packed,
    u4,
    u6,
)


@weft.encoding
class Q4_K:
    layout = bitorder.lsb_first, byteorder.little
    d: f16
    dmin: f16
    sc: u6[8] @ packed(12)
    m: u6[8] @ packed(12)
    q: u4[256] @ nibble(lo_first)


@weft.encoding
class Q8_K:
    layout = bitorder.lsb_first, byteorder.little
    ds: f32
    q: i8[256]
    bsum: i16[16]


@weft.derive
def Q4K_I16(W: View[Q4_K, (M, K)]) -> View[Q4K_I16, (M, K)]:
    return interleave(W, rows=16)
