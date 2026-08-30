from __future__ import annotations

from weft.language import (
    L,
    View,
    admit,
    commit,
    f32,
    i8,
    i32,
    iota,
    index,
    materialize,
    u8,
    u32,
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
    Q4_K,
    Q5_0,
    Q5_1,
    Q5_K,
    Q6_K,
    Q8_0,
    TQ1_0,
    TQ2_0,
)
from .quant_families import (
    fp4_codebook,
    iq_codebook,
    k_superblock,
    min_affine,
    symmetric_integer,
    ternary_radix,
)
from .quant_fragments import (
    exponent_scale,
    extract_bits,
    grid_delta,
    grid_sign,
    nonlinear_lookup,
    radix3_digit,
    signed_scale,
    small_nonlinear_lookup,
)


def dequantize_q1_0(W: View[Q1_0, (K,)], Y: View[f32, (K,)]):
    with L.blocks(K, extent=128) as kb:
        w = admit(W[kb])
        q = i32(i8(w.q) * i8(2) - i8(1))
        commit(ternary_radix(q, w.d), Y[kb])


def dequantize_q4_0(W: View[Q4_0, (K,)], Y: View[f32, (K,)]):
    with L.blocks(K, extent=32) as kb:
        w = admit(W[kb])
        commit(symmetric_integer(w.q, w.d, zero=8), Y[kb])


def dequantize_q4_1(W: View[Q4_1, (K,)], Y: View[f32, (K,)]):
    with L.blocks(K, extent=32) as kb:
        w = admit(W[kb])
        commit(min_affine(w.q, w.d, w.m), Y[kb])


def dequantize_q5_0(W: View[Q5_0, (K,)], Y: View[f32, (K,)]):
    with L.blocks(K, extent=32) as kb:
        w = admit(W[kb])
        q = i32(u8(w.q) | (u8(w.qh) << u8(4))) - i32(16)
        commit(symmetric_integer(q, w.d), Y[kb])


def dequantize_q5_1(W: View[Q5_1, (K,)], Y: View[f32, (K,)]):
    with L.blocks(K, extent=32) as kb:
        w = admit(W[kb])
        q = i32(u8(w.q) | (u8(w.qh) << u8(4)))
        commit(min_affine(q, w.d, w.m), Y[kb])


def dequantize_q8_0(W: View[Q8_0, (K,)], Y: View[f32, (K,)]):
    with L.blocks(K, extent=32) as kb:
        w = admit(W[kb])
        commit(symmetric_integer(w.q, w.d), Y[kb])


def dequantize_q2_k(W: View[Q2_K, (K,)], Y: View[f32, (K,)]):
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        with L.subs(kb, extent=16) as sub:
            metadata = w.scales[sub]
            scale = extract_bits(metadata, 0, 4)
            minimum = extract_bits(metadata, 4, 4)
            commit(
                k_superblock(w.q[sub], scale, w.d, minimum, w.dmin),
                Y[kb][sub],
            )


def dequantize_q3_k(W: View[Q3_K, (K,)], Y: View[f32, (K,)]):
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        with L.subs(kb, extent=16) as sub:
            scale = (
                i32(w.scale_low[sub])
                | (i32(w.scale_high[sub]) << u32(4))
            ) - i32(32)
            q = i32(w.q[sub]) + i32(w.hmask[sub]) * i32(4) - i32(4)
            commit(f32(w.d) * f32(scale) * f32(q), Y[kb][sub])


def dequantize_q4_k(W: View[Q4_K, (K,)], Y: View[f32, (K,)]):
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        with L.subs(kb, extent=32) as sub:
            commit(
                k_superblock(w.q[sub], w.sc[sub], w.d, w.m[sub], w.dmin),
                Y[kb][sub],
            )


def dequantize_q5_k(W: View[Q5_K, (K,)], Y: View[f32, (K,)]):
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        with L.subs(kb, extent=32) as sub:
            q = i32(w.q[sub]) | (i32(w.qh[sub]) << u32(4))
            commit(
                k_superblock(q, w.sc[sub], w.d, w.m[sub], w.dmin),
                Y[kb][sub],
            )


def dequantize_q6_k(W: View[Q6_K, (K,)], Y: View[f32, (K,)]):
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        with L.subs(kb, extent=16) as sub:
            q = i32(w.ql[sub]) | (i32(w.qh[sub]) << u32(4))
            q -= i32(32)
            commit(f32(w.d) * f32(w.scales[sub]) * f32(q), Y[kb][sub])


def dequantize_iq1_s(
    W: View[IQ1_S, (K,)],
    grid: View[i8, (16384,)],
    Y: View[f32, (K,)],
):
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        for j in range(256):
            group = j // 32
            entry = (j % 32) // 8
            lane = j % 8
            metadata = w.qh[group]
            grid_index = u32(w.q[group * 4 + entry]) | (
                extract_bits(metadata, entry * 3, 3) << u32(8)
            )
            delta = f32(0.125)
            if extract_bits(metadata, 15) != u32(0):
                delta = f32(-0.125)
            scale = f32(w.d) * f32(i32(extract_bits(metadata, 12, 3)) * i32(2) + i32(1))
            commit(iq_codebook(grid_delta(grid, grid_index, lane, delta), scale), Y[kb][j])


def dequantize_iq1_m(
    W: View[IQ1_M, (K,)],
    grid: View[i8, (16384,)],
    f16_bits: View[f32, (65536,)],
    Y: View[f32, (K,)],
):
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        sc0 = u32(w.scales[0]) | (u32(w.scales[1]) << u32(8))
        sc1 = u32(w.scales[2]) | (u32(w.scales[3]) << u32(8))
        sc2 = u32(w.scales[4]) | (u32(w.scales[5]) << u32(8))
        sc3 = u32(w.scales[6]) | (u32(w.scales[7]) << u32(8))
        scale_bits = (sc0 >> u32(12)) | ((sc1 >> u32(8)) & u32(0x00F0))
        scale_bits = scale_bits | ((sc2 >> u32(4)) & u32(0x0F00))
        scale_bits = scale_bits | (sc3 & u32(0xF000))
        d = nonlinear_lookup(f16_bits, scale_bits)
        for j in range(256):
            group = j // 32
            entry = (j % 32) // 8
            lane = j % 8
            qh = w.qh[group * 2 + entry // 2]
            high_shift = i32(0)
            delta_shift = i32(3)
            if entry % 2 == 1:
                high_shift = i32(4)
                delta_shift = i32(7)
            grid_index = u32(w.q[group * 4 + entry]) | (
                extract_bits(qh, high_shift, 3) << u32(8)
            )
            delta = f32(0.125)
            if extract_bits(qh, delta_shift) != u32(0):
                delta = f32(-0.125)
            word = u32(w.scales[2 * (group // 2)]) | (
                u32(w.scales[2 * (group // 2) + 1]) << u32(8)
            )
            subscale_shift = (group % 2) * 6
            if entry >= 2:
                subscale_shift = subscale_shift + 3
            subscale = i32(extract_bits(word, subscale_shift, 3)) * i32(2) + i32(1)
            commit(iq_codebook(grid_delta(grid, grid_index, lane, delta), d * f32(subscale)), Y[kb][j])


def dequantize_iq2_xxs(
    W: View[IQ2_XXS, (K,)],
    grid: View[i8, (2048,)],
    signs: View[i8, (1024,)],
    Y: View[f32, (K,)],
):
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        for j in range(256):
            group = j // 32
            entry = (j % 32) // 8
            lane = j % 8
            word = w.q[group * 4 + entry // 2]
            grid_index = extract_bits(word, (entry % 2) * 8, 8)
            metadata = u32(w.q[group * 4 + 2]) | (u32(w.q[group * 4 + 3]) << u32(16))
            sign_index = extract_bits(metadata, entry * 7, 7)
            subscale = f32(0.5) + f32(extract_bits(metadata, 28, 4))
            scale = f32(w.d) * subscale * f32(0.25)
            commit(iq_codebook(grid_sign(grid, signs, grid_index, sign_index, lane), scale), Y[kb][j])


def dequantize_iq2_xs(
    W: View[IQ2_XS, (K,)],
    grid: View[i8, (4096,)],
    signs: View[i8, (1024,)],
    Y: View[f32, (K,)],
):
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        for j in range(256):
            group = j // 32
            entry = (j % 32) // 8
            lane = j % 8
            word = w.q[group * 4 + entry]
            grid_index = extract_bits(word, 0, 9)
            sign_index = extract_bits(word, 9, 7)
            subscale = extract_bits(w.scales[group], (entry // 2) * 4, 4)
            scale = f32(w.d) * (f32(0.5) + f32(subscale)) * f32(0.25)
            commit(iq_codebook(grid_sign(grid, signs, grid_index, sign_index, lane), scale), Y[kb][j])


def dequantize_iq2_s(
    W: View[IQ2_S, (K,)],
    grid: View[i8, (8192,)],
    Y: View[f32, (K,)],
):
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        for j in range(256):
            group = j // 32
            entry = (j % 32) // 8
            lane = j % 8
            grid_index = u32(w.q[group * 4 + entry]) | (
                extract_bits(w.qh[group], entry * 2, 2) << u32(8)
            )
            value = nonlinear_lookup(grid, grid_index * u32(8) + u32(lane))
            if u32(w.signs[group * 32 + entry * 8 + lane]) != u32(0):
                value = -value
            subscale = extract_bits(w.scales[group], (entry // 2) * 4, 4)
            scale = f32(w.d) * (f32(0.5) + f32(subscale)) * f32(0.25)
            commit(iq_codebook(value, scale), Y[kb][j])


def dequantize_iq3_xxs(
    W: View[IQ3_XXS, (K,)],
    grid: View[i8, (1024,)],
    signs: View[i8, (1024,)],
    Y: View[f32, (K,)],
):
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        for j in range(256):
            group = j // 32
            entry = (j % 32) // 8
            lane = j % 8
            grid_base = group * 8
            metadata_base = 64 + group * 4
            grid_index = w.q[grid_base + entry * 2 + lane // 4]
            metadata = u32(w.q[metadata_base]) | (u32(w.q[metadata_base + 1]) << u32(8))
            metadata = metadata | (u32(w.q[metadata_base + 2]) << u32(16))
            metadata = metadata | (u32(w.q[metadata_base + 3]) << u32(24))
            sign_index = extract_bits(metadata, entry * 7, 7)
            grid_value = nonlinear_lookup(grid, u32(grid_index) * u32(4) + u32(lane % 4))
            sign = nonlinear_lookup(signs, u32(sign_index) * u32(8) + u32(lane))
            scale = f32(w.d) * (f32(0.5) + f32(extract_bits(metadata, 28, 4))) * f32(0.5)
            commit(iq_codebook(grid_value * sign, scale), Y[kb][j])


def dequantize_iq3_s(
    W: View[IQ3_S, (K,)],
    grid: View[i8, (2048,)],
    Y: View[f32, (K,)],
):
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        for j in range(256):
            group = j // 32
            entry = (j % 32) // 8
            lane = j % 8
            grid_index = u32(w.q[group * 8 + entry * 2 + lane // 4]) | (
                extract_bits(w.qh[group], entry * 2 + lane // 4) << u32(8)
            )
            value = nonlinear_lookup(grid, grid_index * u32(4) + u32(lane % 4))
            if extract_bits(w.signs[group * 4 + entry], lane) != u32(0):
                value = -value
            subscale = extract_bits(w.scales[group // 2], (group % 2) * 4, 4)
            scale = f32(w.d) * f32(i32(1) + i32(2) * i32(subscale))
            commit(iq_codebook(value, scale), Y[kb][j])


def dequantize_iq4_xs(
    W: View[IQ4_XS, (K,)],
    codebook: View[i8, (16,)],
    Y: View[f32, (K,)],
):
    table = materialize(admit(codebook))
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        sub_index = index(0)
        with L.subs(kb, extent=32) as sub:
            low = extract_bits(
                w.scales_l[sub_index // index(2)],
                (sub_index % index(2)) * index(4),
                4,
            )
            high = extract_bits(w.scales_h, sub_index * index(2), 2)
            scale = signed_scale(low | (high << u32(4)), zero=32)
            q = small_nonlinear_lookup(table, w.q[sub])
            commit(iq_codebook(i32(q), f32(w.d) * f32(scale)), Y[kb][sub])
            sub_index += index(1)


def dequantize_tq1_0(
    W: View[TQ1_0, (K,)],
    powers: View[u32, (5,)],
    Y: View[f32, (K,)],
):
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        for j in range(256):
            packed = u32(w.q[0])
            digit = j // 32
            if j < 160:
                packed = u32(w.q[j % 32])
            if j >= 160:
                packed = u32(w.q[32 + ((j - 160) % 16)])
                digit = (j - 160) // 16
            if j >= 240:
                packed = u32(w.qh[(j - 240) % 4])
                digit = (j - 240) // 4
            q = radix3_digit(powers, packed, digit)
            commit(ternary_radix(q, w.d), Y[kb][j])


def dequantize_iq4_nl(
    W: View[IQ4_NL, (K,)],
    codebook: View[i8, (16,)],
    Y: View[f32, (K,)],
):
    table = materialize(admit(codebook))
    with L.blocks(K, extent=32) as kb:
        w = admit(W[kb])
        q = small_nonlinear_lookup(table, w.q)
        commit(fp4_codebook(q, w.d), Y[kb])


def dequantize_tq2_0(W: View[TQ2_0, (K,)], Y: View[f32, (K,)]):
    with L.blocks(K, extent=256) as kb:
        w = admit(W[kb])
        q = i32(i8(w.q) - i8(1))
        commit(ternary_radix(q, w.d), Y[kb])


def dequantize_mxfp4(
    W: View[MXFP4, (K,)],
    codebook: View[i8, (16,)],
    e8m0_scale: View[f32, (256,)],
    Y: View[f32, (K,)],
):
    with L.blocks(K, extent=32) as kb:
        w = admit(W[kb])
        scale = exponent_scale(e8m0_scale, w.e)
        q = nonlinear_lookup(codebook, w.q)
        commit(fp4_codebook(q, scale), Y[kb])


def dequantize_nvfp4(
    W: View[NVFP4, (K,)],
    codebook: View[i8, (16,)],
    ue4m3_scale: View[f32, (256,)],
    Y: View[f32, (K,)],
):
    table = materialize(admit(codebook))
    with L.blocks(K, extent=64) as kb:
        w = admit(W[kb])
        sub_index = index(0)
        with L.subs(kb, extent=16) as sub:
            scale = exponent_scale(ue4m3_scale, w.d[sub_index])
            q = small_nonlinear_lookup(table, w.q[sub])
            commit(fp4_codebook(q, scale), Y[kb][sub])
            sub_index += index(1)
