from __future__ import annotations

from weft.language import f32, i32, lookup, u32


def extract_bits(value, shift, width: int = 1):
    mask = (1 << width) - 1
    return (u32(value) >> u32(shift)) & u32(mask)


def high_bit_plane(low, plane, position, bit: int = 4):
    return i32(low) | i32(extract_bits(plane, position, 1) << u32(bit))


def signed_scale(raw, zero: int = 0):
    return i32(raw) - i32(zero)


def grid_sign(grid, signs, grid_index, sign_index, lane, lanes: int = 8):
    grid_value = lookup(grid, u32(grid_index) * u32(lanes) + u32(lane))
    sign_value = lookup(signs, u32(sign_index) * u32(lanes) + u32(lane))
    return grid_value * sign_value


def grid_delta(grid, grid_index, lane, delta):
    return lookup(grid, u32(grid_index) * u32(8) + u32(lane)) + f32(delta)


def nonlinear_lookup(codebook, code):
    return lookup(codebook, u32(code))


def exponent_scale(table, raw):
    return lookup(table, u32(raw))


def radix3_digit(powers, packed, digit):
    power = lookup(powers, u32(digit))
    wrapped = (u32(packed) * u32(power)) & u32(255)
    return i32((wrapped * u32(3)) >> u32(8)) - i32(1)
