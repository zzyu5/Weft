from __future__ import annotations

import weft
from weft.language import View, commit, f32, i8, u8, u32
from weft.std.encodings import (
    I8X8,
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
from weft.std.vec_dot import (
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
    vec_dot_q5_0_q8_0,
    vec_dot_q5_1_q8_1,
    vec_dot_q4_k_q8_k,
    vec_dot_q5_k_q8_k,
    vec_dot_q6_k_q8_k,
    vec_dot_q8_0_q8_0,
    vec_dot_tq1_0_q8_k,
    vec_dot_tq2_0_q8_k,
)


@weft.kernel
def quantized_vec_dot_q1_0_q8_0(
    W: View[Q1_0, (K,)], X: View[Q8_0, (K,)], Y: View[f32, (1,)]
):
    commit(vec_dot_q1_0_q8_0(W, X), Y[0])


@weft.kernel
def quantized_vec_dot_q4_0_q8_0(
    W: View[Q4_0, (K,)],
    X: View[Q8_0, (K,)],
    Y: View[f32, (1,)],
):
    commit(vec_dot_q4_0_q8_0(W, X), Y[0])


@weft.kernel
def quantized_vec_dot_q4_1_q8_1(
    W: View[Q4_1, (K,)], X: View[Q8_1, (K,)], Y: View[f32, (1,)]
):
    commit(vec_dot_q4_1_q8_1(W, X), Y[0])


@weft.kernel
def quantized_vec_dot_q5_0_q8_0(
    W: View[Q5_0, (K,)], X: View[Q8_0, (K,)], Y: View[f32, (1,)]
):
    commit(vec_dot_q5_0_q8_0(W, X), Y[0])


@weft.kernel
def quantized_vec_dot_q5_1_q8_1(
    W: View[Q5_1, (K,)], X: View[Q8_1, (K,)], Y: View[f32, (1,)]
):
    commit(vec_dot_q5_1_q8_1(W, X), Y[0])


@weft.kernel
def quantized_vec_dot_q8_0_q8_0(
    W: View[Q8_0, (K,)], X: View[Q8_0, (K,)], Y: View[f32, (1,)]
):
    commit(vec_dot_q8_0_q8_0(W, X), Y[0])


@weft.kernel
def quantized_vec_dot_q2_k_q8_k(
    W: View[Q2_K, (K,)], X: View[Q8_K, (K,)], Y: View[f32, (1,)]
):
    commit(vec_dot_q2_k_q8_k(W, X), Y[0])


@weft.kernel
def quantized_vec_dot_q3_k_q8_k(
    W: View[Q3_K, (K,)], X: View[Q8_K, (K,)], Y: View[f32, (1,)]
):
    commit(vec_dot_q3_k_q8_k(W, X), Y[0])


@weft.kernel
def quantized_vec_dot_q4_k_q8_k(
    W: View[Q4_K, (K,)], X: View[Q8_K, (K,)], Y: View[f32, (1,)]
):
    commit(vec_dot_q4_k_q8_k(W, X), Y[0])


@weft.kernel
def quantized_vec_dot_q5_k_q8_k(
    W: View[Q5_K, (K,)], X: View[Q8_K, (K,)], Y: View[f32, (1,)]
):
    commit(vec_dot_q5_k_q8_k(W, X), Y[0])


@weft.kernel
def quantized_vec_dot_q6_k_q8_k(
    W: View[Q6_K, (K,)], X: View[Q8_K, (K,)], Y: View[f32, (1,)]
):
    commit(vec_dot_q6_k_q8_k(W, X), Y[0])


@weft.kernel
def quantized_vec_dot_iq4_nl_q8_0(
    W: View[IQ4_NL, (K,)],
    X: View[Q8_0, (K,)],
    codebook: View[i8, (16,)],
    Y: View[f32, (1,)],
):
    commit(vec_dot_iq4_nl_q8_0(W, X, codebook), Y[0])


@weft.kernel
def quantized_vec_dot_iq1_s_q8_k(
    W: View[IQ1_S, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[i8, (16384,)],
    Y: View[f32, (1,)],
):
    commit(vec_dot_iq1_s_q8_k(W, X, grid), Y[0])


@weft.kernel
def quantized_vec_dot_iq1_m_q8_k(
    W: View[IQ1_M, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[I8X8, (2048, 8)],
    f16_bits: View[f32, (65536,)],
    Y: View[f32, (1,)],
):
    commit(vec_dot_iq1_m_q8_k(W, X, grid, f16_bits), Y[0])


@weft.kernel
def quantized_vec_dot_iq2_s_q8_k(
    W: View[IQ2_S, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[I8X8, (1024, 8)],
    Y: View[f32, (1,)],
):
    commit(vec_dot_iq2_s_q8_k(W, X, grid), Y[0])


@weft.kernel
def quantized_vec_dot_iq2_xs_q8_k(
    W: View[IQ2_XS, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[I8X8, (512, 8)],
    signs: View[I8X8, (128, 8)],
    Y: View[f32, (1,)],
):
    commit(vec_dot_iq2_xs_q8_k(W, X, grid, signs), Y[0])


@weft.kernel
def quantized_vec_dot_iq2_xxs_q8_k(
    W: View[IQ2_XXS, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[I8X8, (256, 8)],
    signs: View[I8X8, (128, 8)],
    Y: View[f32, (1,)],
):
    commit(vec_dot_iq2_xxs_q8_k(W, X, grid, signs), Y[0])


@weft.kernel
def quantized_vec_dot_iq3_s_q8_k(
    W: View[IQ3_S, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[i8, (2048,)],
    Y: View[f32, (1,)],
):
    commit(vec_dot_iq3_s_q8_k(W, X, grid), Y[0])


@weft.kernel
def quantized_vec_dot_iq3_xxs_q8_k(
    W: View[IQ3_XXS, (K,)],
    X: View[Q8_K, (K,)],
    grid: View[i8, (1024,)],
    signs: View[i8, (1024,)],
    Y: View[f32, (1,)],
):
    commit(vec_dot_iq3_xxs_q8_k(W, X, grid, signs), Y[0])


@weft.kernel
def quantized_vec_dot_iq4_xs_q8_k(
    W: View[IQ4_XS, (K,)],
    X: View[Q8_K, (K,)],
    codebook: View[i8, (16,)],
    Y: View[f32, (1,)],
):
    commit(vec_dot_iq4_xs_q8_k(W, X, codebook), Y[0])


@weft.kernel
def quantized_vec_dot_tq1_0_q8_k(
    W: View[TQ1_0, (K,)],
    X: View[Q8_K, (K,)],
    powers: View[u32, (5,)],
    Y: View[f32, (1,)],
):
    commit(vec_dot_tq1_0_q8_k(W, X, powers), Y[0])


@weft.kernel
def quantized_vec_dot_tq2_0_q8_k(
    W: View[TQ2_0, (K,)], X: View[Q8_K, (K,)], Y: View[f32, (1,)]
):
    commit(vec_dot_tq2_0_q8_k(W, X), Y[0])


@weft.kernel
def quantized_vec_dot_mxfp4_q8_0(
    W: View[MXFP4, (K,)],
    X: View[Q8_0, (K,)],
    codebook: View[i8, (16,)],
    scale: View[f32, (256,)],
    Y: View[f32, (1,)],
):
    commit(vec_dot_mxfp4_q8_0(W, X, codebook, scale), Y[0])


@weft.kernel
def quantized_vec_dot_nvfp4_q8_0(
    W: View[NVFP4, (K,)],
    X: View[Q8_0, (K,)],
    codebook: View[i8, (16,)],
    scale: View[f32, (256,)],
    Y: View[f32, (1,)],
):
    commit(vec_dot_nvfp4_q8_0(W, X, codebook, scale), Y[0])
