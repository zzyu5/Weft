from __future__ import annotations

from weft.language import (
    L,
    View,
    admit,
    auto,
    commit,
    contract,
    dot,
    f16,
    f32,
    fold2,
    i8,
    i16,
    i32,
    iota,
    lookup,
    mac_groups,
    materialize,
    new,
    outer_contract,
    reduce,
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
    Q4_K,
    Q4K_I,
    Q5_0,
    Q5_1,
    Q5_K,
    Q6_K,
    Q8_0,
    Q8_1,
    Q8_K,
    TQ1_0,
    TQ2_0,
)
from .quantize import quantize_q8_0, quantize_q8_1, quantize_q8_K
from .vec_dot import (
    vec_dot_iq1_m_q8_k,
    vec_dot_iq1_s_q8_k,
    vec_dot_iq2_s_q8_k,
    vec_dot_iq2_xs_q8_k,
    vec_dot_iq2_xxs_q8_k,
    vec_dot_iq3_s_q8_k,
    vec_dot_iq3_xxs_q8_k,
    vec_dot_iq4_nl_q8_0,
    vec_dot_iq4_xs_q8_k,
    vec_dot_mxfp4_q8_0,
    vec_dot_nvfp4_q8_0,
    vec_dot_q1_0_q8_0,
    vec_dot_q2_k_q8_k,
    vec_dot_q3_k_q8_k,
    vec_dot_q4_0_q8_0,
    vec_dot_q4_1_q8_1,
    vec_dot_q4_k_q8_k,
    vec_dot_q5_0_q8_0,
    vec_dot_q5_1_q8_1,
    vec_dot_q5_k_q8_k,
    vec_dot_q6_k_q8_k,
    vec_dot_q8_0_q8_0,
    vec_dot_tq1_0_q8_k,
    vec_dot_tq2_0_q8_k,
)


def _iq2_xxs_entry_products(
    word0,
    word1,
    x,
    grid,
    signs,
    codebook_lane,
    group,
    entry,
):
    grid_index = (word0 >> u32(entry * 8)) & u32(255)
    sign_index = (word1 >> u32(entry * 7)) & u32(127)
    weight = lookup(
        grid, grid_index * u32(8) + codebook_lane, bounds="in_bounds"
    )
    sign = lookup(
        signs, sign_index * u32(8) + codebook_lane, bounds="in_bounds"
    )
    signed_weight = weight * sign
    activation = x.q[:, group * 32 + entry * 8 + codebook_lane]
    return widen(activation, i16) * widen(signed_weight, i16)


def _iq2_xxs_group_products(
    w,
    x,
    grid,
    signs,
    entry_lane,
    codebook_lane,
    group,
):
    word0 = widen(w.q[:, group * 4], u32) | (
        widen(w.q[:, group * 4 + 1], u32) << u32(16)
    )
    word1 = widen(w.q[:, group * 4 + 2], u32) | (
        widen(w.q[:, group * 4 + 3], u32) << u32(16)
    )
    local = _iq2_xxs_entry_products(
        word0,
        word1,
        x,
        grid,
        signs,
        codebook_lane,
        group,
        entry_lane,
    )
    scale = i32((word1 >> u32(28)) * u32(2) + u32(1))
    return widen(local, i32) * scale


def mul_mat_q1_0(
    W: View[Q1_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_0(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_q1_0_q8_0(W[column], Xq[row]), Y[row, column])


def mul_mat_q4_0(
    W: View[Q4_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_0(X, Xq)
    for column in range(N):
        with L.tiles(M, extent=auto("MC")) as mc:
            with L.rows(mc, group=auto("MR")) as mb:
                acc = new(f32, [MR], init=f32(0.0))
                with L.blocks(K, extent=32) as kb:
                    w = admit(W[column, kb])
                    x = admit(Xq[mb, kb])
                    centered = i8(w.q) - i8(8)
                    integer = contract(x.q, centered, over="k", acc=i32)
                    acc += (f32(w.d) * f32(x.d)) * widen(integer, f32)
                commit(acc, Y[mb, column])


def mul_mat_q4_0_decode(
    W: View[Q4_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_0(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_q4_0_q8_0(W[column], Xq[row]), Y[row, column])


def mul_mat_q4_1(
    W: View[Q4_1, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_1, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_1(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_q4_1_q8_1(W[column], Xq[row]), Y[row, column])


def mul_mat_q5_0(
    W: View[Q5_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_0(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_q5_0_q8_0(W[column], Xq[row]), Y[row, column])


def mul_mat_q5_1(
    W: View[Q5_1, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_1, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_1(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_q5_1_q8_1(W[column], Xq[row]), Y[row, column])


def mul_mat_q8_0(
    W: View[Q8_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_0(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_q8_0_q8_0(W[column], Xq[row]), Y[row, column])


def mul_mat_q2_k(
    W: View[Q2_K, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_q2_k_q8_k(W[column], Xq[row]), Y[row, column])


def mul_mat_q3_k(
    W: View[Q3_K, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_q3_k_q8_k(W[column], Xq[row]), Y[row, column])


def mul_mat_q4_k(
    W: View[Q4_K, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_q4_k_q8_k(W[column], Xq[row]), Y[row, column])


def mul_mat_q4_k_persistent(
    W: View[Q4K_I[16], (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        with L.rows(N, group=16) as nb:
            f32_acc = new(f32, [16], init=0)
            with L.blocks(K, extent=256) as kb:
                w = admit(W[nb, kb])
                x = admit(Xq[row, kb])
                i32_acc = new(i32, [16], init=0)
                with L.subs(extent=32) as s:
                    p16 = mac_groups(w.q[s], x.q[s], n=4, into=i16)
                    i32_acc += reduce(widen(p16, i32)) * w.sc[s]
                mins = fold2(x.bsum)
                min_term = dot(w.m, mins)
                f32_acc += x.ds * (w.d * i32_acc - w.dmin * min_term)
            commit(f32_acc, Y[row, nb])


def mul_mat_q4_k_staged(
    W: View[Q4_K, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(f32(0.0), Y[row, column])
    with L.tiles(N, extent=auto("NC")) as nc:
        with L.tiles(K, extent=auto("KC")) as kc:
            wp = materialize(admit(W[nc, kc]))
            with L.tiles(M, extent=auto("MC")) as mc:
                with L.rows(mc, group=auto("MR")) as mb:
                    with L.cols(nc, group=16) as nb:
                        f32_acc = new(f32, [MR, 16], init=admit(Y[mb, nb]))
                        with L.blocks(kc, extent=256) as kb:
                            w = wp[nb, kb]
                            x = admit(Xq[mb, kb])
                            i32_acc = new(i32, [MR, 16], init=0)
                            with L.subs(extent=32) as s:
                                p32 = outer_contract(
                                    x.q[s], w.q[s], over="k", acc=i32
                                )
                                i32_acc += p32 * w.sc[s]
                            mins = fold2(x.bsum)
                            min_term = outer_contract(
                                mins, w.m, over="k", acc=i32
                            )
                            f32_acc += x.ds * (
                                w.d * i32_acc - w.dmin * min_term
                            )
                        commit(f32_acc, Y[mb, nb])


def mul_mat_q5_k(
    W: View[Q5_K, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_q5_k_q8_k(W[column], Xq[row]), Y[row, column])


def mul_mat_q6_k(
    W: View[Q6_K, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_q6_k_q8_k(W[column], Xq[row]), Y[row, column])


def mul_mat_iq1_s(
    W: View[IQ1_S, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    grid: View[i8, (16384,)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_iq1_s_q8_k(W[column], Xq[row], grid), Y[row, column])


def mul_mat_iq1_m(
    W: View[IQ1_M, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    grid: View[i8, (16384,)],
    f16_bits: View[f32, (65536,)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vec_dot_iq1_m_q8_k(W[column], Xq[row], grid, f16_bits)
            commit(value, Y[row, column])


def mul_mat_iq2_s(
    W: View[IQ2_S, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    grid: View[i8, (8192,)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_iq2_s_q8_k(W[column], Xq[row], grid), Y[row, column])


def mul_mat_iq2_xs(
    W: View[IQ2_XS, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    grid: View[i8, (4096,)],
    signs: View[i8, (1024,)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vec_dot_iq2_xs_q8_k(W[column], Xq[row], grid, signs)
            commit(value, Y[row, column])


def mul_mat_iq2_xxs(
    W: View[IQ2_XXS, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    grid: View[i8, (2048,)],
    signs: View[i8, (1024,)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vec_dot_iq2_xxs_q8_k(W[column], Xq[row], grid, signs)
            commit(value, Y[row, column])


def mul_mat_iq2_xxs_staged(
    W: View[IQ2_XXS, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    grid: View[i8, (2048,)],
    signs: View[i8, (1024,)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(f32(0.0), Y[row, column])
    with L.tiles(N, extent=auto("NC")) as nc:
        with L.tiles(K, extent=auto("KC")) as kc:
            wp = materialize(admit(W[nc, kc]))
            with L.tiles(M, extent=auto("MC")) as mc:
                with L.rows(mc, group=auto("MR")) as mb:
                    with L.cols(nc, group=auto("NR")) as nb:
                        f32_acc = new(f32, [MR, NR], init=admit(Y[mb, nb]))
                        with L.blocks(kc, extent=256) as kb:
                            w = wp[nb, kb]
                            x = admit(Xq[mb, kb])
                            entry_lane = iota(4)
                            codebook_lane = iota(8)
                            block_partial = _iq2_xxs_group_products(
                                w,
                                x,
                                grid,
                                signs,
                                entry_lane,
                                codebook_lane,
                                0,
                            )
                            for group in range(1, 8):
                                group_partial = _iq2_xxs_group_products(
                                    w,
                                    x,
                                    grid,
                                    signs,
                                    entry_lane,
                                    codebook_lane,
                                    group,
                                )
                                block_partial = (
                                    block_partial + group_partial
                                )
                            entry_sum = reduce(block_partial, axis=1)
                            block_sum = reduce(entry_sum, axis=1)
                            f32_acc += (
                                f32(w.d)
                                * f32(x.ds)
                                * widen(block_sum, f32)
                            )
                        commit(f32_acc, Y[mb, nb])
    with L.tiles(N, extent=auto("NC")) as nc:
        with L.tiles(M, extent=auto("MC")) as mc:
            with L.rows(mc, group=auto("MR")) as mb:
                with L.cols(nc, group=auto("NR")) as nb:
                    value = admit(Y[mb, nb])
                    commit(f32(0.125) * value, Y[mb, nb])


def mul_mat_iq3_s(
    W: View[IQ3_S, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    grid: View[i8, (2048,)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_iq3_s_q8_k(W[column], Xq[row], grid), Y[row, column])


def mul_mat_iq3_xxs(
    W: View[IQ3_XXS, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    grid: View[i8, (1024,)],
    signs: View[i8, (1024,)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vec_dot_iq3_xxs_q8_k(W[column], Xq[row], grid, signs)
            commit(value, Y[row, column])


def mul_mat_iq4_nl(
    W: View[IQ4_NL, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    codebook: View[i8, (16,)],
    Y: View[f32, (M, N)],
):
    quantize_q8_0(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vec_dot_iq4_nl_q8_0(W[column], Xq[row], codebook)
            commit(value, Y[row, column])


def mul_mat_iq4_xs(
    W: View[IQ4_XS, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    codebook: View[i8, (16,)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vec_dot_iq4_xs_q8_k(W[column], Xq[row], codebook)
            commit(value, Y[row, column])


def mul_mat_tq1_0(
    W: View[TQ1_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    powers: View[u32, (5,)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_tq1_0_q8_k(W[column], Xq[row], powers), Y[row, column])


def mul_mat_tq2_0(
    W: View[TQ2_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            commit(vec_dot_tq2_0_q8_k(W[column], Xq[row]), Y[row, column])


def mul_mat_mxfp4(
    W: View[MXFP4, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    codebook: View[i8, (16,)],
    scale: View[f32, (256,)],
    Y: View[f32, (M, N)],
):
    quantize_q8_0(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vec_dot_mxfp4_q8_0(W[column], Xq[row], codebook, scale)
            commit(value, Y[row, column])


def mul_mat_nvfp4(
    W: View[NVFP4, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    codebook: View[i8, (16,)],
    scale: View[f32, (256,)],
    Y: View[f32, (M, N)],
):
    quantize_q8_0(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vec_dot_nvfp4_q8_0(W[column], Xq[row], codebook, scale)
            commit(value, Y[row, column])


def mul_mat_f16(
    W: View[f16, (N, K)],
    X: View[f32, (M, K)],
    Xh: View[f16, (M, K)],
    Y: View[f32, (M, N)],
):
    for row in range(M):
        with L.blocks(K, extent=1) as kb:
            value = admit(X[row, kb])
            commit(f16(value), Xh[row, kb])
    for row in range(M):
        for column in range(N):
            commit(f32(0.0), Y[row, column])
    with L.tiles(N, extent=auto("NC")) as nc:
        with L.tiles(K, extent=auto("KC")) as kc:
            wp = materialize(admit(W[nc, kc]))
            with L.tiles(M, extent=auto("MC")) as mc:
                xp = materialize(admit(Xh[mc, kc]))
                with L.rows(mc, group=auto("MR")) as mb:
                    with L.cols(nc, group=auto("NR")) as nb:
                        acc = new(f32, [MR, NR], init=admit(Y[mb, nb]))
                        with L.blocks(kc, extent=auto("KB")) as kb:
                            weights = wp[nb, kb]
                            activations = xp[mb, kb]
                            acc += outer_contract(
                                activations, weights, over="k", acc=f32
                            )
                        commit(acc, Y[mb, nb])
