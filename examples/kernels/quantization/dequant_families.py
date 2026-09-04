from __future__ import annotations

from weft.language import f32, i32


def symmetric_integer(q, d, zero: int = 0):
    return f32(i32(q) - i32(zero)) * f32(d)


def min_affine(q, d, minimum):
    return f32(i32(q)) * f32(d) + f32(minimum)


def k_superblock(q, scale, d, minimum, dmin):
    scaled = f32(d) * f32(i32(scale))
    minimum_scaled = f32(dmin) * f32(i32(minimum))
    return scaled * f32(i32(q)) - minimum_scaled


def ternary_radix(q, d):
    return f32(i32(q)) * f32(d)


def iq_codebook(q, scale):
    return f32(q) * f32(scale)


def fp4_codebook(q, scale):
    return f32(i32(q)) * f32(scale)
