from __future__ import annotations

from weft.language import (
    L,
    View,
    admit,
    contract,
    f32,
    i8,
    i16,
    i32,
    iota,
    index,
    lookup,
    mac_groups,
    materialize,
    reduce,
    u8,
    u16,
    u32,
    widen,
)

from .encodings import (
    I8X4,
    IQ1_M,
    IQ1_S,
    IQ2_S,
    IQ2_XS,
    IQ2_XXS,
    IQ3_S,
    IQ3_XXS,
    IQ4_NL,
    IQ4_XS,
    I8X8,
    MXFP4,
    NVFP4,
    Q1_0,
    Q2_K,
    Q3_K,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_0,
    Q8_1,
    Q8_K,
    TQ1_0,
    TQ2_0,
)
from .quant_fragments import (
    exponent_scale,
    nonlinear_lookup,
    radix3_digit_i8,
    small_nonlinear_lookup,
)


def _dot_q4_values(q, x):
    return contract(q, x, over="k", acc=i32)


def _dot_codebook32(q, x, codebook):
    return contract(small_nonlinear_lookup(codebook, q), x, over="k", acc=i32)


def _signed_grid8(grid, signs, grid_index, sign_index, x, base):
    lane = iota(8, dtype=u32, axis="k")
    value = nonlinear_lookup(grid, u32(grid_index) * u32(8) + lane)
    sign = nonlinear_lookup(signs, u32(sign_index) * u32(8) + lane)
    product = widen(value, i32) * widen(sign, i32) * widen(x[base + lane], i32)
    return reduce(product, axis=0)


def _grid8(grid, grid_index, x, base):
    lane = iota(8, dtype=u32, axis="k")
    value = nonlinear_lookup(grid, u32(grid_index) * u32(8) + lane)
    product = widen(value, i32) * widen(x[base + lane], i32)
    return reduce(product, axis=0)


def _iq2_xs_entry_reduce(
    code,
    metadata,
    activation,
    grid,
    signs,
    scale_group,
    payload,
):
    code_bits = u16(code)
    grid_index = code_bits & u16(511)
    sign_index = code_bits >> u16(9)
    weight = lookup(
        grid, grid_index * u16(8) + payload, bounds="in_bounds"
    )
    sign = lookup(
        signs, sign_index * u16(8) + payload, bounds="in_bounds"
    )
    partial = contract(
        activation,
        weight * sign,
        over=("entry", "payload"),
        acc=i32,
    )
    scale = i32(
        (
            (widen(metadata, u32) >> ((scale_group % u32(2)) * u32(4)))
            & u32(15)
        )
        * u32(2)
        + u32(1)
    )
    return reduce(partial * scale, axis="scale_group")


def vec_dot_q1_0_q8_0(
    W: View[Q1_0, (K,)],
    X: View[Q8_0, (K,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=128) as wb:
        w = admit(W[wb])
        subtotal = f32(0.0)
        with L.subs(wb, extent=32) as xb:
            x = admit(X[xb])
            centered = i8(w.q[xb]) * i8(2) - i8(1)
            integer = contract(x.q, centered, over="k", acc=i32)
            subtotal += f32(x.d) * f32(integer)
        result += f32(w.d) * subtotal
    return result


def vec_dot_q4_0_q8_0(
    W: View[Q4_0, (K,)],
    X: View[Q8_0, (K,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=32) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        centered = i8(w.q) - i8(8)
        integer = contract(x.q, centered, over="k", acc=i32)
        result += (f32(w.d) * f32(x.d)) * f32(integer)
    return result


def vec_dot_q4_1_q8_1(
    W: View[Q4_1, (K,)],
    X: View[Q8_1, (K,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=32) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        integer = _dot_q4_values(w.q, x.q)
        result += (f32(w.d) * f32(x.d)) * f32(integer) + f32(w.m) * f32(x.s)
    return result


def vec_dot_q5_0_q8_0(
    W: View[Q5_0, (K,)],
    X: View[Q8_0, (K,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=32) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        q = u8(w.q) | (u8(w.qh) << u8(4))
        centered = i8(q) - i8(16)
        integer = contract(centered, x.q, over="k", acc=i32)
        result += (f32(w.d) * f32(x.d)) * f32(integer)
    return result


def vec_dot_q5_1_q8_1(
    W: View[Q5_1, (K,)],
    X: View[Q8_1, (K,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=32) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        q = u8(w.q) | (u8(w.qh) << u8(4))
        integer = contract(q, x.q, over="k", acc=i32)
        result += (f32(w.d) * f32(x.d)) * f32(integer) + f32(w.m) * f32(x.s)
    return result


def vec_dot_q8_0_q8_0(
    W: View[Q8_0, (K,)],
    X: View[Q8_0, (K,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=32) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        integer = contract(w.q, x.q, over="k", acc=i32)
        result += f32(integer) * (f32(w.d) * f32(x.d))
    return result


def vec_dot_q2_k_q8_k(
    W: View[Q2_K, (K,)],
    X: View[Q8_K, (K,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        integer = i32(0)
        with L.subs(kb, extent=16) as sub:
            partial = contract(
                w.q[sub],
                x.q[sub],
                over="k",
                acc=i32,
            )
            scale = i32(u8(w.scales[sub]) & u8(15))
            integer += partial * scale
        mins = widen(u8(w.scales) >> u8(4), i16)
        minimum = contract(x.bsum, mins, over="k", acc=i32)
        result += f32(x.ds) * (
            f32(w.d) * f32(integer) - f32(w.dmin) * f32(minimum)
        )
    return result


def vec_dot_q3_k_q8_k(
    W: View[Q3_K, (K,)],
    X: View[Q8_K, (K,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        decoded_scales = i8(
            (u8(w.scale_low) | (u8(w.scale_high) << u8(4))) - u8(32)
        )
        integer_acc = i32(0)
        half_index = index(0)
        with L.subs(kb, extent=128) as half:
            plane = iota(4, dtype=u8)
            scale_part = iota(2, dtype=u8)
            lane = iota(16, dtype=u8, axis="k")
            coordinate = (
                u8(half_index) * u8(128)
                + plane * u8(32)
                + scale_part * u8(16)
                + lane
            )
            q = (
                i8(
                    u8(w.q[coordinate])
                    | (u8(w.hmask[coordinate]) << u8(2))
                )
                - i8(4)
            )
            products = contract(x.q[coordinate], q, over="k", acc=i32)
            scale_coordinate = (
                u8(half_index) * u8(8)
                + plane * u8(2)
                + scale_part
            )
            scales = i32(decoded_scales[scale_coordinate])
            integer_acc += reduce(
                reduce(products * scales, axis=1), axis=0
            )
            half_index += index(1)
        result += (f32(w.d) * f32(x.ds)) * f32(integer_acc)
    return result


def vec_dot_q3_k_q8_k_predecoded_scales(
    W: View[Q3_K, (K,)],
    X: View[Q8_K, (K,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        decoded_scales = materialize(
            i8((u8(w.scale_low) | (u8(w.scale_high) << u8(4))) - u8(32))
        )
        integer_acc = i32(0)
        half_index = index(0)
        with L.subs(kb, extent=128) as half:
            plane = iota(4, dtype=u8)
            scale_part = iota(2, dtype=u8)
            lane = iota(16, dtype=u8, axis="k")
            coordinate = (
                u8(half_index) * u8(128)
                + plane * u8(32)
                + scale_part * u8(16)
                + lane
            )
            q = (
                i8(
                    u8(w.q[coordinate])
                    | (u8(w.hmask[coordinate]) << u8(2))
                )
                - i8(4)
            )
            products = contract(x.q[coordinate], q, over="k", acc=i32)
            scale_coordinate = (
                u8(half_index) * u8(8)
                + plane * u8(2)
                + scale_part
            )
            scales = i32(decoded_scales[scale_coordinate])
            integer_acc += reduce(
                reduce(products * scales, axis=1), axis=0
            )
            half_index += index(1)
        result += (f32(w.d) * f32(x.ds)) * f32(integer_acc)
    return result


def vec_dot_q4_k_q8_k(
    W: View[Q4_K, (K,)],
    X: View[Q8_K, (K,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        i32_acc = i32(0)
        with L.subs(extent=32) as sub:
            partial = contract(w.q[sub], x.q[sub], over="k", acc=i32)
            i32_acc += partial * i32(w.sc[sub])
        minimum = i32(0)
        for sub in range(8):
            minimum += i32(w.m[sub]) * (
                i32(x.bsum[sub * 2]) + i32(x.bsum[sub * 2 + 1])
            )
        result += f32(x.ds) * (
            f32(w.d) * f32(i32_acc) - f32(w.dmin) * f32(minimum)
        )
    return result


def vec_dot_q5_k_q8_k(
    W: View[Q5_K, (K,)],
    X: View[Q8_K, (K,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        integer = i32(0)
        with L.subs(extent=32) as sub:
            q = u8(w.q[sub]) | (u8(w.qh[sub]) << u8(4))
            integer += contract(q, x.q[sub], over="k", acc=i32) * i32(w.sc[sub])
        minimum = i32(0)
        for sub in range(8):
            minimum += i32(w.m[sub]) * (
                i32(x.bsum[sub * 2]) + i32(x.bsum[sub * 2 + 1])
            )
        result += f32(x.ds) * (
            f32(w.d) * f32(integer) - f32(w.dmin) * f32(minimum)
        )
    return result


def vec_dot_q6_k_q8_k(
    W: View[Q6_K, (K,)],
    X: View[Q8_K, (K,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        integer_acc = i32(0)
        half_index = index(0)
        with L.subs(kb, extent=128) as half:
            plane = iota(4, dtype=u8)
            scale_part = iota(2, dtype=u8)
            lane = iota(16, dtype=u8, axis="k")
            coordinate = (
                u8(half_index) * u8(128)
                + plane * u8(32)
                + scale_part * u8(16)
                + lane
            )
            q = i8(w.ql[coordinate]) | (i8(w.qh[coordinate]) << u8(4))
            q -= i8(32)
            products = contract(x.q[coordinate], q, over="k", acc=i32)
            scale_coordinate = (
                u8(half_index) * u8(8)
                + plane * u8(2)
                + scale_part
            )
            integer_acc += reduce(
                reduce(products * i32(w.scales[scale_coordinate]), axis=1), axis=0
            )
            half_index += index(1)
        result += f32(w.d) * f32(x.ds) * f32(integer_acc)
    return result


def vec_dot_iq1_s_q8_k(
    W: View[IQ1_S, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[I8X8, (2048, 8)],
):
    result = f32(0.0)
    grid_values = grid.values
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        entry = iota(4, dtype=u32, axis="entry")
        payload = iota(8, dtype=u16, axis="payload")
        main = i32(0)
        correction = i32(0)
        group_index = index(0)
        with L.subs(kb, extent=32) as group:
            metadata = widen(w.qh[group_index], u32)
            scale = i32(((metadata >> u32(12)) & u32(7)) * u32(2) + u32(1))
            grid_index = widen(
                w.q[u32(group_index) * u32(4) + entry], u32
            ) | (((metadata >> (entry * u32(3))) & u32(7)) << u32(8))
            weight = lookup(
                grid_values,
                grid_index * u32(8) + payload,
                bounds="in_bounds",
            )
            activation = x.q[
                u32(group_index) * u32(32)
                + entry * u32(8)
                + u32(payload)
            ]
            group_sum = contract(
                activation,
                weight,
                over=("entry", "payload"),
                acc=i32,
            )
            delta = i32(1) - i32((metadata >> u32(15)) & u32(1)) * i32(2)
            main += scale * group_sum
            correction += scale * delta * (
                i32(x.bsum[group_index * index(2)])
                + i32(x.bsum[group_index * index(2) + index(1)])
            )
            group_index += index(1)
        combined = f32(main) + f32(0.125) * f32(correction)
        result += f32(w.d) * f32(x.ds) * combined
    return result


def vec_dot_iq1_m_q8_k(
    W: View[IQ1_M, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[i8, (16384,)],
    f16_bits: View[f32, (65536,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        sc0 = u32(w.scales[0]) | (u32(w.scales[1]) << u32(8))
        sc1 = u32(w.scales[2]) | (u32(w.scales[3]) << u32(8))
        sc2 = u32(w.scales[4]) | (u32(w.scales[5]) << u32(8))
        sc3 = u32(w.scales[6]) | (u32(w.scales[7]) << u32(8))
        scale_bits = (sc0 >> u32(12)) | ((sc1 >> u32(8)) & u32(0x00F0))
        scale_bits = scale_bits | ((sc2 >> u32(4)) & u32(0x0F00))
        scale_bits = scale_bits | (sc3 & u32(0xF000))
        block_scale = nonlinear_lookup(f16_bits, scale_bits)
        main = i32(0)
        correction = i32(0)
        for group in range(8):
            sum10 = i32(0)
            sum11 = i32(0)
            sum20 = i32(0)
            sum21 = i32(0)
            for entry in range(4):
                qh = u32(w.qh[group * 2 + entry // 2])
                high_shift = i32(0)
                delta_shift = i32(3)
                if entry % 2 == 1:
                    high_shift = i32(4)
                    delta_shift = i32(7)
                grid_index = u32(w.q[group * 4 + entry]) | (
                    ((qh >> u32(high_shift)) & u32(7)) << u32(8)
                )
                dot = _grid8(grid, grid_index, x.q, group * 32 + entry * 8)
                qsum = i32(0)
                for lane in range(8):
                    qsum += i32(x.q[group * 32 + entry * 8 + lane])
                delta = i32(1)
                if ((qh >> u32(delta_shift)) & u32(1)) != u32(0):
                    delta = i32(-1)
                if entry < 2:
                    sum10 += dot
                    sum20 += qsum * delta
                if entry >= 2:
                    sum11 += dot
                    sum21 += qsum * delta
            word = u32(w.scales[2 * (group // 2)]) | (
                u32(w.scales[2 * (group // 2) + 1]) << u32(8)
            )
            shift = (group % 2) * 6
            ls1 = i32(((word >> u32(shift)) & u32(7)) * u32(2) + u32(1))
            ls2 = i32(((word >> u32(shift + 3)) & u32(7)) * u32(2) + u32(1))
            main += sum10 * ls1 + sum11 * ls2
            correction += sum20 * ls1 + sum21 * ls2
        combined = f32(main) + f32(0.125) * f32(correction)
        result += f32(block_scale) * f32(x.ds) * combined
    return result


def vec_dot_iq2_xxs_q8_k(
    W: View[IQ2_XXS, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[I8X8, (256, 8)],
    signs: View[I8X8, (128, 8)],
):
    sumf = f32(0.0)
    grid_values = grid.values
    sign_values = signs.values
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        entry = iota(4, dtype=u32, axis="entry")
        payload = iota(8, dtype=u16, axis="payload")
        block_sum = i32(0)
        group_index = index(0)
        with L.subs(kb, extent=32) as group:
            word0 = widen(w.q[group_index * index(4)], u32) | (
                widen(w.q[group_index * index(4) + index(1)], u32)
                << u32(16)
            )
            word1 = widen(w.q[group_index * index(4) + index(2)], u32) | (
                widen(w.q[group_index * index(4) + index(3)], u32)
                << u32(16)
            )
            grid_index = (word0 >> (entry * u32(8))) & u32(255)
            sign_index = (word1 >> (entry * u32(7))) & u32(127)
            weight = lookup(
                grid_values, grid_index * u32(8) + payload, bounds="in_bounds"
            )
            sign = lookup(
                sign_values, sign_index * u32(8) + payload, bounds="in_bounds"
            )
            activation = x.q[
                u32(group_index) * u32(32)
                + entry * u32(8)
                + u32(payload)
            ]
            partial = contract(
                activation,
                weight * sign,
                over=("entry", "payload"),
                acc=i32,
            )
            scale = i32((word1 >> u32(28)) * u32(2) + u32(1))
            block_sum += partial * scale
            group_index += index(1)
        sumf += f32(w.d) * f32(x.ds) * f32(block_sum)
    return f32(0.125) * sumf


def vec_dot_iq2_xs_q8_k(
    W: View[IQ2_XS, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[I8X8, (512, 8)],
    signs: View[I8X8, (128, 8)],
):
    sumf = f32(0.0)
    grid_values = grid.values
    sign_values = signs.values
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        scale_group = iota(16, dtype=u32, axis="scale_group")
        entry = iota(2, dtype=u32, axis="entry")
        payload = iota(8, dtype=u16, axis="payload")
        linear_entry = scale_group * u32(2) + entry
        entry_offset = scale_group * u32(16) + entry * u32(8)
        block_sum = _iq2_xs_entry_reduce(
            w.q[linear_entry],
            w.scales[scale_group // u32(2)],
            x.q[entry_offset + u32(payload)],
            grid_values,
            sign_values,
            scale_group,
            payload,
        )
        sumf += f32(w.d) * f32(x.ds) * f32(block_sum)
    return f32(0.125) * sumf


def vec_dot_iq2_s_q8_k(
    W: View[IQ2_S, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[I8X8, (1024, 8)],
):
    sumf = f32(0.0)
    grid_values = grid.values
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        scale_group = iota(2, dtype=u32, axis="scale_group")
        entry = iota(2, dtype=u32, axis="entry")
        payload = iota(8, dtype=u16, axis="payload")
        block_sum = i32(0)
        group_index = index(0)
        with L.subs(kb, extent=32) as group:
            linear_entry = scale_group * u32(2) + entry
            grid_index = widen(
                w.q[u32(group_index) * u32(4) + linear_entry], u32
            ) | (
                (
                    widen(w.qh[group_index], u32)
                    >> (linear_entry * u32(2))
                )
                & u32(3)
            ) << u32(8)
            sign_bit = w.signs[
                u32(group_index) * u32(32)
                + linear_entry * u32(8)
                + u32(payload)
            ]
            weight = lookup(
                grid_values,
                grid_index * u32(8) + payload,
                bounds="in_bounds",
            )
            signed_weight = weight * (i8(1) - i8(sign_bit) * i8(2))
            activation = x.q[
                u32(group_index) * u32(32)
                + linear_entry * u32(8)
                + u32(payload)
            ]
            partial = contract(
                activation,
                signed_weight,
                over=("entry", "payload"),
                acc=i32,
            )
            metadata = widen(w.scales[group_index], u32)
            scale = i32(
                (
                    (metadata >> (scale_group * u32(4)))
                    & u32(15)
                )
                * u32(2)
                + u32(1)
            )
            block_sum += reduce(partial * scale, axis="scale_group")
            group_index += index(1)
        sumf += f32(w.d) * f32(x.ds) * f32(block_sum)
    return f32(0.125) * sumf


def vec_dot_iq3_xxs_q8_k(
    W: View[IQ3_XXS, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[I8X4, (256, 4)],
    signs: View[I8X4, (256, 4)],
):
    sumf = f32(0.0)
    grid_values = grid.values
    sign_values = signs.values
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        entry = iota(4, dtype=u32, axis="entry")
        code = iota(2, dtype=u32, axis="code")
        payload = iota(4, dtype=u16, axis="payload")
        block_sum = i32(0)
        group_index = index(0)
        with L.subs(kb, extent=32) as group:
            metadata_base = index(64) + group_index * index(4)
            metadata = u32(w.q[metadata_base]) | (u32(w.q[metadata_base + 1]) << u32(8))
            metadata = metadata | (u32(w.q[metadata_base + 2]) << u32(16))
            metadata = metadata | (u32(w.q[metadata_base + 3]) << u32(24))
            grid_index = widen(
                w.q[
                    u32(group_index) * u32(8)
                    + entry * u32(2)
                    + code
                ],
                u32,
            )
            sign_index = (metadata >> (entry * u32(7))) & u32(127)
            sign_entry = sign_index * u32(2) + code
            weight = lookup(
                grid_values,
                grid_index * u32(4) + payload,
                bounds="in_bounds",
            )
            sign = lookup(
                sign_values,
                sign_entry * u32(4) + payload,
                bounds="in_bounds",
            )
            activation = x.q[
                u32(group_index) * u32(32)
                + entry * u32(8)
                + code * u32(4)
                + u32(payload)
            ]
            local = contract(
                activation,
                weight * sign,
                over=("entry", "code", "payload"),
                acc=i32,
            )
            scale = i32((metadata >> u32(28)) * u32(2) + u32(1))
            block_sum += local * scale
            group_index += index(1)
        sumf += f32(w.d) * f32(x.ds) * f32(block_sum)
    return f32(0.25) * sumf


def vec_dot_iq3_s_q8_k(
    W: View[IQ3_S, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[I8X4, (512, 4)],
):
    result = f32(0.0)
    grid_values = grid.values
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        entry = iota(4, dtype=u32, axis="entry")
        code = iota(2, dtype=u32, axis="code")
        payload = iota(4, dtype=u16, axis="payload")
        block_sum = i32(0)
        group_index = index(0)
        with L.subs(kb, extent=32) as group:
            storage_coordinate = (
                u32(group_index) * u32(8)
                + entry * u32(2)
                + code
            )
            grid_index = widen(w.q[storage_coordinate], u32) | (
                widen(w.qh[storage_coordinate], u32) << u32(8)
            )
            weight = lookup(
                grid_values,
                grid_index * u32(4) + payload,
                bounds="in_bounds",
            )
            sign_bit = w.signs[
                u32(group_index) * u32(32)
                + entry * u32(8)
                + code * u32(4)
                + u32(payload)
            ]
            signed_weight = weight * (i8(1) - i8(sign_bit) * i8(2))
            activation = x.q[
                u32(group_index) * u32(32)
                + entry * u32(8)
                + code * u32(4)
                + u32(payload)
            ]
            local = contract(
                activation,
                signed_weight,
                over=("entry", "code", "payload"),
                acc=i32,
            )
            scale = i32(w.scales[group_index]) * i32(2) + i32(1)
            block_sum += local * scale
            group_index += index(1)
        result += f32(w.d) * f32(x.ds) * f32(block_sum)
    return result


def vec_dot_iq4_nl_q8_0(
    W: View[IQ4_NL, (K,)],
    X: View[Q8_0, (K,)],
    codebook: View[i8, (16,)],
):
    table = materialize(admit(codebook))
    result = f32(0.0)
    with L.blocks(K, extent=32) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        integer = _dot_codebook32(w.q, x.q, table)
        result += (f32(x.d) * f32(w.d)) * f32(integer)
    return result


def vec_dot_iq4_xs_q8_k(
    W: View[IQ4_XS, (K,)],
    X: View[Q8_K, (K,)],
    codebook: View[i8, (16,)],
):
    table = materialize(admit(codebook))
    result = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        sub_index = index(0)
        with L.subs(kb, extent=32) as sub:
            low = u32(w.scales_l[sub_index // index(2)])
            shift = (sub_index % index(2)) * index(4)
            scale_low = (low >> u32(shift)) & u32(15)
            scale_high = (u32(w.scales_h) >> u32(sub_index * index(2))) & u32(3)
            scale = i32(scale_low | (scale_high << u32(4))) - i32(32)
            q = small_nonlinear_lookup(table, w.q[sub])
            integer = contract(q, x.q[sub], over="k", acc=i32)
            block_scale = f32(w.d) * f32(x.ds) * f32(scale)
            result += block_scale * f32(integer)
            sub_index += index(1)
    return result


def vec_dot_tq1_0_q8_k(
    W: View[TQ1_0, (K,)],
    X: View[Q8_K, (K,)],
    powers: View[u32, (5,)],
):
    power_table = materialize(admit(powers))
    result = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        lane0 = iota(32, axis="k")
        q0 = radix3_digit_i8(power_table, w.q[lane0], 0)
        partial0 = widen(q0, i16) * widen(x.q[lane0], i16)
        for digit0 in range(1, 5):
            q0 = radix3_digit_i8(power_table, w.q[lane0], digit0)
            partial0 += widen(q0, i16) * widen(
                x.q[u32(digit0 * 32) + lane0], i16
            )
        integer = reduce(widen(partial0, i32), axis=0)

        lane1 = iota(16, axis="k")
        q1 = radix3_digit_i8(power_table, w.q[u32(32) + lane1], 0)
        partial1 = widen(q1, i16) * widen(x.q[u32(160) + lane1], i16)
        for digit1 in range(1, 5):
            q1 = radix3_digit_i8(
                power_table, w.q[u32(32) + lane1], digit1
            )
            partial1 += widen(q1, i16) * widen(
                x.q[u32(160 + digit1 * 16) + lane1], i16
            )
        integer += reduce(widen(partial1, i32), axis=0)

        lane2 = iota(16, axis="k")
        q2 = radix3_digit_i8(
            power_table, w.qh[lane2 // u32(4)], lane2 % u32(4)
        )
        integer += contract(
            q2,
            x.q[u32(240) + (lane2 % u32(4)) * u32(4) + lane2 // u32(4)],
            over="k",
            acc=i32,
        )
        result += f32(integer) * (f32(w.d) * f32(x.ds))
    return result


def vec_dot_tq2_0_q8_k(
    W: View[TQ2_0, (K,)],
    X: View[Q8_K, (K,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        q = i8(w.q) - i8(1)
        integer = contract(x.q, q, over="k", acc=i32)
        scale = f32(x.ds) * f32(w.d)
        result += widen(integer, f32) * scale
    return result


def vec_dot_mxfp4_q8_0(
    W: View[MXFP4, (K,)],
    X: View[Q8_0, (K,)],
    codebook: View[i8, (16,)],
    e8m0_scale: View[f32, (256,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=32) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        integer = _dot_codebook32(w.q, x.q, codebook)
        scale = f32(x.d) * exponent_scale(e8m0_scale, w.e)
        result += scale * f32(integer)
    return result


def vec_dot_nvfp4_q8_0(
    W: View[NVFP4, (K,)],
    X: View[Q8_0, (K,)],
    codebook: View[i8, (16,)],
    ue4m3_scale: View[f32, (256,)],
):
    table = materialize(admit(codebook))
    result = f32(0.0)
    with L.blocks(K, extent=64) as wb:
        w = admit(W[wb])
        with L.subs(wb, extent=32) as xb:
            x = admit(X[xb])
            with L.subs(xb, extent=16) as sub:
                q = small_nonlinear_lookup(table, w.q[sub])
                integer = contract(q, x.q[sub], over="k", acc=i32)
                scale = f32(x.d) * exponent_scale(ue4m3_scale, w.d[sub])
                result += scale * f32(integer)
    return result
