from __future__ import annotations

import weft
from weft.language import View, f16, f32, i8, u8, u32
from weft.std.encodings import (
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
from weft.std.mul_mat import (
    mul_mat_f16,
    mul_mat_iq1_m,
    mul_mat_iq1_s,
    mul_mat_iq2_s,
    mul_mat_iq2_xs,
    mul_mat_iq2_xxs,
    mul_mat_iq2_xxs_decode,
    mul_mat_iq2_xxs_staged,
    mul_mat_iq3_s,
    mul_mat_iq3_xxs,
    mul_mat_iq4_nl,
    mul_mat_iq4_nl_decode,
    mul_mat_iq4_xs,
    mul_mat_iq4_xs_decode,
    mul_mat_mxfp4,
    mul_mat_mxfp4_decode,
    mul_mat_nvfp4,
    mul_mat_nvfp4_decode,
    mul_mat_q1_0,
    mul_mat_q1_0_decode,
    mul_mat_q2_k,
    mul_mat_q2_k_decode,
    mul_mat_q2_k_group_reduced,
    mul_mat_q2_k_group_reduced_decode,
    mul_mat_q3_k,
    mul_mat_q3_k_decode,
    mul_mat_q4_0,
    mul_mat_q4_0_decode,
    mul_mat_q4_1,
    mul_mat_q4_1_decode,
    mul_mat_q4_k,
    mul_mat_q4_k_decode,
    mul_mat_q4_k_persistent,
    mul_mat_q4_k_staged,
    mul_mat_q5_0,
    mul_mat_q5_0_decode,
    mul_mat_q5_1,
    mul_mat_q5_1_decode,
    mul_mat_q5_k,
    mul_mat_q5_k_decode,
    mul_mat_q6_k,
    mul_mat_q6_k_decode,
    mul_mat_q8_0,
    mul_mat_q8_0_decode,
    mul_mat_tq1_0,
    mul_mat_tq2_0,
    mul_mat_tq2_0_decode,
)


@weft.kernel
def production_mul_mat_q1_0(
    W: View[Q1_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q1_0(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q1_0_decode(
    W: View[Q1_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q1_0_decode(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q4_0(
    W: View[Q4_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q4_0(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q4_0_decode(
    W: View[Q4_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q4_0_decode(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_f16(
    W: View[f16, (N, K)],
    X: View[f32, (M, K)],
    Xh: View[f16, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_f16(W, X, Xh, Y)


@weft.kernel
def production_mul_mat_q4_1(
    W: View[Q4_1, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_1, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q4_1(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q4_1_decode(
    W: View[Q4_1, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_1, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q4_1_decode(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q5_0(
    W: View[Q5_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q5_0(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q5_0_decode(
    W: View[Q5_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q5_0_decode(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q5_1(
    W: View[Q5_1, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_1, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q5_1(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q5_1_decode(
    W: View[Q5_1, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_1, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q5_1_decode(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q8_0(
    W: View[Q8_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q8_0(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q8_0_decode(
    W: View[Q8_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_0, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q8_0_decode(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q2_k(
    W: View[Q2_K, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)], Y: View[f32, (M, N)]
):
    mul_mat_q2_k(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q2_k_decode(
    W: View[Q2_K, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)], Y: View[f32, (M, N)]
):
    mul_mat_q2_k_decode(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q2_k_group_reduced(
    W: View[Q2_K, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)], Y: View[f32, (M, N)]
):
    mul_mat_q2_k_group_reduced(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q2_k_group_reduced_decode(
    W: View[Q2_K, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)], Y: View[f32, (M, N)]
):
    mul_mat_q2_k_group_reduced_decode(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q3_k(
    W: View[Q3_K, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)], Y: View[f32, (M, N)]
):
    mul_mat_q3_k(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q3_k_decode(
    W: View[Q3_K, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)], Y: View[f32, (M, N)]
):
    mul_mat_q3_k_decode(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q4_k(
    W: View[Q4_K, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)], Y: View[f32, (M, N)]
):
    mul_mat_q4_k(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q4_k_decode(
    W: View[Q4_K, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)], Y: View[f32, (M, N)]
):
    mul_mat_q4_k_decode(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q4_k_persistent(
    W: View[Q4K_I[16], (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q4_k_persistent(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q4_k_staged(
    W: View[Q4_K, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q4_k_staged(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q5_k(
    W: View[Q5_K, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)], Y: View[f32, (M, N)]
):
    mul_mat_q5_k(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q5_k_decode(
    W: View[Q5_K, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q5_k_decode(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q6_k(
    W: View[Q6_K, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)], Y: View[f32, (M, N)]
):
    mul_mat_q6_k(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_q6_k_decode(
    W: View[Q6_K, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_q6_k_decode(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_iq1_s(
    W: View[IQ1_S, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)],
    grid: View[i8, (16384,)], Y: View[f32, (M, N)]
):
    mul_mat_iq1_s(W, X, Xq, grid, Y)


@weft.kernel
def production_mul_mat_iq1_m(
    W: View[IQ1_M, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)],
    grid: View[i8, (16384,)], f16_bits: View[f32, (65536,)], Y: View[f32, (M, N)]
):
    mul_mat_iq1_m(W, X, Xq, grid, f16_bits, Y)


@weft.kernel
def production_mul_mat_iq2_s(
    W: View[IQ2_S, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)],
    grid: View[i8, (8192,)], Y: View[f32, (M, N)]
):
    mul_mat_iq2_s(W, X, Xq, grid, Y)


@weft.kernel
def production_mul_mat_iq2_xs(
    W: View[IQ2_XS, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)],
    grid: View[i8, (4096,)], signs: View[i8, (1024,)], Y: View[f32, (M, N)]
):
    mul_mat_iq2_xs(W, X, Xq, grid, signs, Y)


@weft.kernel
def production_mul_mat_iq2_xxs(
    W: View[IQ2_XXS, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)],
    grid: View[i8, (2048,)], signs: View[i8, (1024,)], Y: View[f32, (M, N)]
):
    mul_mat_iq2_xxs(W, X, Xq, grid, signs, Y)


@weft.kernel
def production_mul_mat_iq2_xxs_decode(
    W: View[IQ2_XXS, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)],
    grid: View[i8, (2048,)], signs: View[i8, (1024,)], Y: View[f32, (M, N)]
):
    mul_mat_iq2_xxs_decode(W, X, Xq, grid, signs, Y)


@weft.kernel
def production_mul_mat_iq2_xxs_staged(
    W: View[IQ2_XXS, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    grid: View[i8, (2048,)],
    signs: View[i8, (1024,)],
    Y: View[f32, (M, N)],
):
    mul_mat_iq2_xxs_staged(W, X, Xq, grid, signs, Y)


@weft.kernel
def production_mul_mat_iq3_s(
    W: View[IQ3_S, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)],
    grid: View[i8, (2048,)], Y: View[f32, (M, N)]
):
    mul_mat_iq3_s(W, X, Xq, grid, Y)


@weft.kernel
def production_mul_mat_iq3_xxs(
    W: View[IQ3_XXS, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)],
    grid: View[i8, (1024,)], signs: View[i8, (1024,)], Y: View[f32, (M, N)]
):
    mul_mat_iq3_xxs(W, X, Xq, grid, signs, Y)


@weft.kernel
def production_mul_mat_iq4_nl(
    W: View[IQ4_NL, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_0, (M, K)],
    codebook: View[i8, (16,)], Y: View[f32, (M, N)]
):
    mul_mat_iq4_nl(W, X, Xq, codebook, Y)


@weft.kernel
def production_mul_mat_iq4_nl_decode(
    W: View[IQ4_NL, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_0, (M, K)],
    codebook: View[i8, (16,)], Y: View[f32, (M, N)]
):
    mul_mat_iq4_nl_decode(W, X, Xq, codebook, Y)


@weft.kernel
def production_mul_mat_iq4_xs(
    W: View[IQ4_XS, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)],
    codebook: View[i8, (16,)], Y: View[f32, (M, N)]
):
    mul_mat_iq4_xs(W, X, Xq, codebook, Y)


@weft.kernel
def production_mul_mat_iq4_xs_decode(
    W: View[IQ4_XS, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)],
    codebook: View[i8, (16,)], Y: View[f32, (M, N)]
):
    mul_mat_iq4_xs_decode(W, X, Xq, codebook, Y)


@weft.kernel
def production_mul_mat_tq1_0(
    W: View[TQ1_0, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)],
    powers: View[u32, (5,)], Y: View[f32, (M, N)]
):
    mul_mat_tq1_0(W, X, Xq, powers, Y)


@weft.kernel
def production_mul_mat_tq2_0(
    W: View[TQ2_0, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_K, (M, K)], Y: View[f32, (M, N)]
):
    mul_mat_tq2_0(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_tq2_0_decode(
    W: View[TQ2_0, (N, K)],
    X: View[f32, (M, K)],
    Xq: View[Q8_K, (M, K)],
    Y: View[f32, (M, N)],
):
    mul_mat_tq2_0_decode(W, X, Xq, Y)


@weft.kernel
def production_mul_mat_mxfp4(
    W: View[MXFP4, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_0, (M, K)],
    codebook: View[i8, (16,)], scale: View[f32, (256,)], Y: View[f32, (M, N)]
):
    mul_mat_mxfp4(W, X, Xq, codebook, scale, Y)


@weft.kernel
def production_mul_mat_mxfp4_decode(
    W: View[MXFP4, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_0, (M, K)],
    codebook: View[i8, (16,)], scale: View[f32, (256,)], Y: View[f32, (M, N)]
):
    mul_mat_mxfp4_decode(W, X, Xq, codebook, scale, Y)


@weft.kernel
def production_mul_mat_nvfp4(
    W: View[NVFP4, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_0, (M, K)],
    codebook: View[i8, (16,)], scale: View[f32, (256,)], Y: View[f32, (M, N)]
):
    mul_mat_nvfp4(W, X, Xq, codebook, scale, Y)


@weft.kernel
def production_mul_mat_nvfp4_decode(
    W: View[NVFP4, (N, K)], X: View[f32, (M, K)], Xq: View[Q8_0, (M, K)],
    codebook: View[i8, (16,)], scale: View[f32, (256,)], Y: View[f32, (M, N)]
):
    mul_mat_nvfp4_decode(W, X, Xq, codebook, scale, Y)
