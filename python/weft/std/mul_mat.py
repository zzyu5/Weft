from __future__ import annotations

from weft.language import L, View, admit, commit, f16, f32, transfer, u32

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
    for row in range(M):
        for column in range(N):
            value = vec_dot_q4_0_q8_0(W[column], Xq[row])
            commit(value, Y[row, column])


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
    grid: View[f32, (16384,)],
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
    grid: View[f32, (16384,)],
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
    grid: View[f32, (8192,)],
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
    grid: View[f32, (4096,)],
    signs: View[f32, (1024,)],
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
    grid: View[f32, (2048,)],
    signs: View[f32, (1024,)],
    Y: View[f32, (M, N)],
):
    quantize_q8_K(X, Xq)
    for row in range(M):
        for column in range(N):
            value = vec_dot_iq2_xxs_q8_k(W[column], Xq[row], grid, signs)
            commit(value, Y[row, column])


def mul_mat_iq3_s(
    W: View[IQ3_S, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    grid: View[f32, (2048,)],
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
    grid: View[f32, (1024,)],
    signs: View[f32, (1024,)],
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
    codebook: View[f32, (16,)],
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
    codebook: View[f32, (16,)],
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
    codebook: View[f32, (16,)],
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
    codebook: View[f32, (16,)],
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
            value = admit(X[row, kb]) @ transfer
            commit(f16(value), Xh[row, kb]) @ transfer
    for row in range(M):
        for column in range(N):
            accumulator = f32(0.0)
            with L.blocks(K, extent=1) as kb:
                weight = f32((admit(W[column, kb]) @ transfer)[0])
                activation = f32((admit(Xh[row, kb]) @ transfer)[0])
                accumulator += weight * activation
            commit(accumulator, Y[row, column])
