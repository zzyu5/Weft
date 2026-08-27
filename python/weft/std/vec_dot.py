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
    mac_groups,
    materialize,
    reduce,
    u8,
    u32,
    widen,
)

from .encodings import (
    IQ1_M,
    IQ1_S,
    IQ2_S,
    IQ2_XS,
    IQ2_XXS,
    IQ3_S,
    IQ3_XXS,
    IQ4_NL,
    IQ4_XS,
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
    radix3_digit,
    small_nonlinear_lookup,
)


def _dot_q4_values(q, x):
    return contract(q, x, over="k", acc=i32)


def _dot_codebook32(q, x, codebook):
    return contract(small_nonlinear_lookup(codebook, q), x, over="k", acc=i32)


def _q3_scale(scales, sub):
    quarter = sub // 4
    position = sub % 4
    source = u32(scales[position])
    if quarter == 1:
        source = u32(scales[4 + position])
    if quarter == 2:
        source = u32(scales[position])
    if quarter == 3:
        source = u32(scales[4 + position])
    shift = i32(0)
    if quarter >= 2:
        shift = i32(4)
    low = (source >> u32(shift)) & u32(15)
    high = (u32(scales[8 + position]) >> u32(quarter * 2)) & u32(3)
    return i32(low | (high << u32(4))) - i32(32)


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
        coordinates = iota(128, dtype=u32, axis="k")
        first_scale = widen(
            u8(w.scales[coordinates / u32(16)]) & u8(15), i16
        )
        first_weight = widen(w.q[coordinates], i16) * first_scale
        first = contract(
            widen(x.q[coordinates], i16), first_weight, over="k", acc=i32
        )

        second_coordinates = coordinates + u32(128)
        second_scale = widen(
            u8(w.scales[coordinates / u32(16) + u32(8)]) & u8(15), i16
        )
        second_weight = widen(w.q[second_coordinates], i16) * second_scale
        second = contract(
            widen(x.q[second_coordinates], i16),
            second_weight,
            over="k",
            acc=i32,
        )

        mins = widen(u8(w.scales) >> u8(4), i16)
        minimum = contract(x.bsum, mins, over="k", acc=i32)
        scaled = first + second
        result += f32(x.ds) * (
            f32(w.d) * f32(scaled) - f32(w.dmin) * f32(minimum)
        )
    return result


def vec_dot_q3_k_q8_k(
    W: View[Q3_K, (K,)],
    X: View[Q8_K, (K,)],
):
    lane0 = f32(0.0)
    lane1 = f32(0.0)
    lane2 = f32(0.0)
    lane3 = f32(0.0)
    lane4 = f32(0.0)
    lane5 = f32(0.0)
    lane6 = f32(0.0)
    lane7 = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        acc0 = i32(0)
        acc1 = i32(0)
        acc2 = i32(0)
        acc3 = i32(0)
        acc4 = i32(0)
        acc5 = i32(0)
        acc6 = i32(0)
        acc7 = i32(0)
        for sub in range(16):
            scale = _q3_scale(w.scales, sub)
            for lane in range(16):
                j = sub * 16 + lane
                within = j % 128
                packed = w.q[(j // 128) * 32 + within % 32]
                low = (u32(packed) >> u32((within // 32) * 2)) & u32(3)
                high = (u32(w.hmask[j % 32]) >> u32(j // 32)) & u32(1)
                q = i32(low) + i32(high) * i32(4) - i32(4)
                product = scale * q * i32(x.q[j])
                which = lane % 8
                if which == 0:
                    acc0 += product
                if which == 1:
                    acc1 += product
                if which == 2:
                    acc2 += product
                if which == 3:
                    acc3 += product
                if which == 4:
                    acc4 += product
                if which == 5:
                    acc5 += product
                if which == 6:
                    acc6 += product
                if which == 7:
                    acc7 += product
        scale = f32(w.d) * f32(x.ds)
        lane0 += scale * f32(acc0)
        lane1 += scale * f32(acc1)
        lane2 += scale * f32(acc2)
        lane3 += scale * f32(acc3)
        lane4 += scale * f32(acc4)
        lane5 += scale * f32(acc5)
        lane6 += scale * f32(acc6)
        lane7 += scale * f32(acc7)
    result = f32(0.0)
    result += lane0
    result += lane1
    result += lane2
    result += lane3
    result += lane4
    result += lane5
    result += lane6
    result += lane7
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
    lane0 = f32(0.0)
    lane1 = f32(0.0)
    lane2 = f32(0.0)
    lane3 = f32(0.0)
    lane4 = f32(0.0)
    lane5 = f32(0.0)
    lane6 = f32(0.0)
    lane7 = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        acc0 = i32(0)
        acc1 = i32(0)
        acc2 = i32(0)
        acc3 = i32(0)
        acc4 = i32(0)
        acc5 = i32(0)
        acc6 = i32(0)
        acc7 = i32(0)
        for sub in range(16):
            scale = i32(w.scales[sub])
            for lane in range(16):
                j = sub * 16 + lane
                within = j % 128
                ql = w.ql[(j // 128) * 64 + within % 64]
                low = (u32(ql) >> u32((within // 64) * 4)) & u32(15)
                qh = w.qh[(j // 128) * 32 + within % 32]
                high = (u32(qh) >> u32((within // 32) * 2)) & u32(3)
                q = i32(low | (high << u32(4))) - i32(32)
                product = scale * q * i32(x.q[j])
                which = lane % 8
                if which == 0:
                    acc0 += product
                if which == 1:
                    acc1 += product
                if which == 2:
                    acc2 += product
                if which == 3:
                    acc3 += product
                if which == 4:
                    acc4 += product
                if which == 5:
                    acc5 += product
                if which == 6:
                    acc6 += product
                if which == 7:
                    acc7 += product
        scale = f32(w.d) * f32(x.ds)
        lane0 += scale * f32(acc0)
        lane1 += scale * f32(acc1)
        lane2 += scale * f32(acc2)
        lane3 += scale * f32(acc3)
        lane4 += scale * f32(acc4)
        lane5 += scale * f32(acc5)
        lane6 += scale * f32(acc6)
        lane7 += scale * f32(acc7)
    result = f32(0.0)
    result += lane0
    result += lane1
    result += lane2
    result += lane3
    result += lane4
    result += lane5
    result += lane6
    result += lane7
    return result


def vec_dot_iq1_s_q8_k(
    W: View[IQ1_S, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[i8, (16384,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        main = i32(0)
        correction = i32(0)
        for group in range(8):
            metadata = u32(w.qh[group])
            scale = i32(((metadata >> u32(12)) & u32(7)) * u32(2) + u32(1))
            group_sum = i32(0)
            for entry in range(4):
                grid_index = u32(w.q[group * 4 + entry]) | (
                    ((metadata >> u32(entry * 3)) & u32(7)) << u32(8)
                )
                group_sum += _grid8(grid, grid_index, x.q, group * 32 + entry * 8)
            delta = i32(1)
            if ((metadata >> u32(15)) & u32(1)) != u32(0):
                delta = i32(-1)
            main += scale * group_sum
            correction += scale * delta * (i32(x.bsum[group * 2]) + i32(x.bsum[group * 2 + 1]))
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
    grid: View[i8, (2048,)],
    signs: View[i8, (1024,)],
):
    sumf = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        block_sum = i32(0)
        for group in range(8):
            word0 = u32(w.q[group * 4]) | (u32(w.q[group * 4 + 1]) << u32(16))
            word1 = u32(w.q[group * 4 + 2]) | (u32(w.q[group * 4 + 3]) << u32(16))
            local = i32(0)
            for entry in range(4):
                grid_index = (word0 >> u32(entry * 8)) & u32(255)
                sign_index = (word1 >> u32(entry * 7)) & u32(127)
                local += _signed_grid8(
                    grid, signs, grid_index, sign_index, x.q, group * 32 + entry * 8
                )
            scale = i32((word1 >> u32(28)) * u32(2) + u32(1))
            block_sum += local * scale
        sumf += f32(w.d) * f32(x.ds) * f32(block_sum)
    return f32(0.125) * sumf


def vec_dot_iq2_xs_q8_k(
    W: View[IQ2_XS, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[i8, (4096,)],
    signs: View[i8, (1024,)],
):
    sumf = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        block_sum = i32(0)
        for group in range(8):
            first = i32(0)
            second = i32(0)
            for entry in range(4):
                code = u32(w.q[group * 4 + entry])
                dot = _signed_grid8(
                    grid,
                    signs,
                    code & u32(511),
                    code >> u32(9),
                    x.q,
                    group * 32 + entry * 8,
                )
                if entry < 2:
                    first += dot
                if entry >= 2:
                    second += dot
            metadata = u32(w.scales[group])
            ls1 = i32((metadata & u32(15)) * u32(2) + u32(1))
            ls2 = i32((metadata >> u32(4)) * u32(2) + u32(1))
            block_sum += first * ls1 + second * ls2
        sumf += f32(w.d) * f32(x.ds) * f32(block_sum)
    return f32(0.125) * sumf


def vec_dot_iq2_s_q8_k(
    W: View[IQ2_S, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[i8, (8192,)],
):
    sumf = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        block_sum = i32(0)
        for group in range(8):
            first = i32(0)
            second = i32(0)
            for entry in range(4):
                grid_index = u32(w.q[group * 4 + entry]) | (
                    ((u32(w.qh[group]) >> u32(entry * 2)) & u32(3)) << u32(8)
                )
                local = i32(0)
                sign_byte = u32(w.q[32 + group * 4 + entry])
                for lane in range(8):
                    sign = i32(1)
                    if ((sign_byte >> u32(lane)) & u32(1)) != u32(0):
                        sign = i32(-1)
                    value = i32(
                        nonlinear_lookup(grid, grid_index * u32(8) + u32(lane))
                    )
                    local += i32(x.q[group * 32 + entry * 8 + lane]) * value * sign
                if entry < 2:
                    first += local
                if entry >= 2:
                    second += local
            metadata = u32(w.scales[group])
            ls1 = i32((metadata & u32(15)) * u32(2) + u32(1))
            ls2 = i32((metadata >> u32(4)) * u32(2) + u32(1))
            block_sum += ls1 * first + ls2 * second
        sumf += f32(w.d) * f32(x.ds) * f32(block_sum)
    return f32(0.125) * sumf


def vec_dot_iq3_xxs_q8_k(
    W: View[IQ3_XXS, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[i8, (1024,)],
    signs: View[i8, (1024,)],
):
    sumf = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        block_sum = i32(0)
        for group in range(8):
            metadata_base = 64 + group * 4
            metadata = u32(w.q[metadata_base]) | (u32(w.q[metadata_base + 1]) << u32(8))
            metadata = metadata | (u32(w.q[metadata_base + 2]) << u32(16))
            metadata = metadata | (u32(w.q[metadata_base + 3]) << u32(24))
            local = i32(0)
            for entry in range(4):
                sign_index = (metadata >> u32(entry * 7)) & u32(127)
                for lane in range(4):
                    q0 = i32(
                        nonlinear_lookup(
                            grid,
                            u32(w.q[group * 8 + entry * 2]) * u32(4) + u32(lane),
                        )
                    )
                    q1 = i32(
                        nonlinear_lookup(
                            grid,
                            u32(w.q[group * 8 + entry * 2 + 1]) * u32(4) + u32(lane),
                        )
                    )
                    s0 = i32(
                        nonlinear_lookup(signs, sign_index * u32(8) + u32(lane))
                    )
                    s1 = i32(
                        nonlinear_lookup(signs, sign_index * u32(8) + u32(lane + 4))
                    )
                    base = group * 32 + entry * 8
                    local += q0 * i32(x.q[base + lane]) * s0
                    local += q1 * i32(x.q[base + lane + 4]) * s1
            scale = i32((metadata >> u32(28)) * u32(2) + u32(1))
            block_sum += local * scale
        sumf += f32(w.d) * f32(x.ds) * f32(block_sum)
    return f32(0.25) * sumf


def vec_dot_iq3_s_q8_k(
    W: View[IQ3_S, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[i8, (2048,)],
):
    result = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        block_sum = i32(0)
        for group in range(8):
            local = i32(0)
            for entry in range(4):
                qh = u32(w.qh[group])
                grid0 = u32(w.q[group * 8 + entry * 2]) | (
                    ((qh >> u32(entry * 2)) & u32(1)) << u32(8)
                )
                grid1 = u32(w.q[group * 8 + entry * 2 + 1]) | (
                    ((qh >> u32(entry * 2 + 1)) & u32(1)) << u32(8)
                )
                sign_byte = u32(w.signs[group * 4 + entry])
                for lane in range(4):
                    sign0 = i32(1)
                    sign1 = i32(1)
                    if ((sign_byte >> u32(lane)) & u32(1)) != u32(0):
                        sign0 = i32(-1)
                    if ((sign_byte >> u32(lane + 4)) & u32(1)) != u32(0):
                        sign1 = i32(-1)
                    q0 = i32(nonlinear_lookup(grid, grid0 * u32(4) + u32(lane)))
                    q1 = i32(nonlinear_lookup(grid, grid1 * u32(4) + u32(lane)))
                    base = group * 32 + entry * 8
                    local += q0 * i32(x.q[base + lane]) * sign0
                    local += q1 * i32(x.q[base + lane + 4]) * sign1
            metadata = u32(w.scales[group // 2])
            shift = (group % 2) * 4
            scale = i32(((metadata >> u32(shift)) & u32(15)) * u32(2) + u32(1))
            block_sum += local * scale
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
    result = f32(0.0)
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        x = admit(X[kb])
        integer = i32(0)
        for j in range(32):
            for digit in range(5):
                q = radix3_digit(powers, w.q[j], digit)
                integer += q * i32(x.q[digit * 32 + j])
        for j in range(16):
            for digit in range(5):
                q = radix3_digit(powers, w.q[32 + j], digit)
                integer += q * i32(x.q[160 + digit * 16 + j])
        for digit in range(4):
            for j in range(4):
                q = radix3_digit(powers, w.qh[j], digit)
                integer += q * i32(x.q[240 + digit * 4 + j])
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
