from __future__ import annotations

from weft.language import f32, i8, i32, lookup, narrow, u8, u16, u32, widen


def extract_bits(value, shift, width: int = 1):
    mask = (1 << width) - 1
    return (u32(value) >> u32(shift)) & u32(mask)


def signed_scale(raw, zero: int = 0):
    return i32(raw) - i32(zero)


def grid_sign(grid, signs, grid_index, sign_index, lane, lanes: int = 8):
    grid_value = lookup(
        grid, u32(grid_index) * u32(lanes) + u32(lane), bounds="in_bounds"
    )
    sign_value = lookup(
        signs, u32(sign_index) * u32(lanes) + u32(lane), bounds="in_bounds"
    )
    return i32(grid_value) * i32(sign_value)


def grid_delta(grid, grid_index, lane, delta):
    return (
        lookup(
            grid, u32(grid_index) * u32(8) + u32(lane), bounds="in_bounds"
        )
        + f32(delta)
    )


def nonlinear_lookup(codebook, code):
    return lookup(codebook, u32(code), bounds="in_bounds")


def small_nonlinear_lookup(codebook, code):
    return lookup(codebook, u8(code), bounds="in_bounds")


def exponent_scale(table, raw):
    return lookup(table, u32(raw), bounds="in_bounds")


def radix3_digit(powers, packed, digit):
    power = lookup(powers, u32(digit), bounds="in_bounds")
    wrapped = (u32(packed) * u32(power)) & u32(255)
    return i32((wrapped * u32(3)) >> u32(8)) - i32(1)


def radix3_digit_i8(powers, packed, digit):
    power = narrow(
        lookup(powers, u32(digit), bounds="in_bounds"),
        u16,
        rounding="rtz",
        saturation=False,
    )
    wrapped = (widen(u8(packed), u16) * power) & u16(255)
    decoded = narrow(
        (wrapped * u16(3)) >> u16(8),
        u8,
        rounding="rtz",
        saturation=False,
    )
    return i8(decoded) - i8(1)
