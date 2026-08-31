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
    static,
    u1,
    u2,
    u4,
    u6,
    u8,
    u16,
)


@weft.encoding
class Q1_0:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 2
    elements = 128
    d: f16
    q: u1[128] @ grouped(8) @ layered(1, lo_first)


@weft.encoding
class Q4_0:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 2
    elements = 32
    d: f16
    q: u4[32] @ grouped(32) @ layered(16, lo_first)


@weft.encoding
class Q4_1:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 2
    elements = 32
    d: f16
    m: f16
    q: u4[32] @ grouped(32) @ layered(16, lo_first)


@weft.encoding
class Q5_0:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 2
    elements = 32
    d: f16
    qh: u1[32] @ grouped(8) @ layered(1, lo_first)
    q: u4[32] @ grouped(32) @ layered(16, lo_first)


@weft.encoding
class Q5_1:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 2
    elements = 32
    d: f16
    m: f16
    qh: u1[32] @ grouped(8) @ layered(1, lo_first)
    q: u4[32] @ grouped(32) @ layered(16, lo_first)


@weft.encoding
class Q8_0:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 2
    elements = 32
    d: f16
    q: i8[32]


@weft.encoding
class Q2_K:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 2
    elements = 256
    scales: u8[16]
    q: u2[256] @ grouped(128) @ layered(32, lo_first)
    d: f16
    dmin: f16


@weft.encoding
class Q3_K:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 2
    elements = 256
    hmask: u1[256] @ grouped(256) @ layered(32, lo_first)
    q: u2[256] @ grouped(128) @ layered(32, lo_first)
    scale_low: u4[16] @ grouped(16) @ layered(8, lo_first)
    scale_high: u2[16] @ grouped(16) @ layered(4, lo_first)
    d: f16


@weft.encoding
class Q4_K:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 2
    elements = 256
    d: f16
    dmin: f16
    sc: u6[8] @ joined(4, 2, 4, lo_first)
    m: u6[8] @ joined(4, 2, 4, lo_first)
    q: u4[256] @ grouped(64) @ layered(32, lo_first)


@weft.encoding
class Q5_K:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    d: f16
    dmin: f16
    sc: u6[8] @ joined(4, 2, 4, lo_first)
    m: u6[8] @ joined(4, 2, 4, lo_first)
    qh: u1[256] @ grouped(256) @ layered(32, lo_first)
    q: u4[256] @ grouped(64) @ layered(32, lo_first)


@weft.encoding
class Q6_K:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    ql: u4[256] @ grouped(128) @ layered(64, lo_first)
    qh: u2[256] @ grouped(128) @ layered(32, lo_first)
    scales: i8[16] @ grouped(16) @ layered(16, lo_first)
    d: f16


@weft.encoding
class IQ1_S:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    d: f16
    q: u8[32]
    qh: u16[8]


@weft.encoding
class IQ1_M:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    q: u8[32]
    qh: u8[16]
    scales: u8[8]


@weft.encoding
class IQ2_S:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    d: f16
    q: u8[32]
    signs: u1[256] @ grouped(8) @ layered(1, lo_first)
    qh: u8[8]
    scales: u8[8]


@weft.encoding
class IQ2_XS:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    d: f16
    q: u16[32]
    scales: u8[8]


@weft.encoding
class IQ2_XXS:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    d: f16
    q: u16[32]


@weft.encoding
class I8X8:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 8
    elements = 8
    values: i8[8]


@weft.encoding
class I8X4:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 4
    elements = 4
    values: i8[4]


@weft.encoding
class IQ3_S:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    d: f16
    q: u8[64]
    qh: u1[64] @ grouped(8) @ layered(1, lo_first)
    signs: u1[256] @ grouped(8) @ layered(1, lo_first)
    scales: u4[8] @ grouped(2) @ layered(1, lo_first)


@weft.encoding
class IQ3_XXS:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    d: f16
    q: u8[96]


@weft.encoding
class IQ4_NL:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 2
    elements = 32
    d: f16
    q: u4[32] @ grouped(32) @ layered(16, lo_first)


@weft.encoding
class IQ4_XS:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 2
    elements = 256
    d: f16
    scales_h: u16
    scales_l: u8[4]
    q: u4[256] @ grouped(32) @ layered(16, lo_first)


@weft.encoding
class TQ1_0:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    q: u8[48]
    qh: u8[4]
    d: f16


@weft.encoding
class TQ2_0:
    layout = bitorder.lsb_first, byteorder.little
    elements = 256
    q: u2[256] @ grouped(128) @ layered(32, lo_first)
    d: f16


@weft.encoding
class MXFP4:
    layout = bitorder.lsb_first, byteorder.little
    elements = 32
    e: u8
    q: u4[32] @ grouped(32) @ layered(16, lo_first)


@weft.encoding
class NVFP4:
    layout = bitorder.lsb_first, byteorder.little
    elements = 64
    d: u8[4]
    q: u4[64] @ grouped(16) @ layered(8, lo_first)


@weft.encoding
class Q8_1:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 2
    elements = 32
    d: f16
    s: f16
    q: i8[32]


@weft.encoding
class Q8_K:
    layout = bitorder.lsb_first, byteorder.little
    alignment = 4
    elements = 256
    ds: f32
    q: i8[256]
    bsum: i16[16]


@weft.derive
def Q4K_I(
    W: View[Q4_K, (M, K)], *, rows: static[int]
) -> View[Q4K_I[rows], (M, K)]:
    return interleave(W, rows=rows)
