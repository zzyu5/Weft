from __future__ import annotations

import weft
from weft.language import View, f32, i8, u8, u32
from weft_kernels.encodings import (
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
from weft_kernels.row_dequant import (
    dequantize_iq1_m,
    dequantize_iq1_s,
    dequantize_iq2_s,
    dequantize_iq2_xs,
    dequantize_iq2_xxs,
    dequantize_iq3_s,
    dequantize_iq3_xxs,
    dequantize_iq4_nl,
    dequantize_iq4_xs,
    dequantize_mxfp4,
    dequantize_nvfp4,
    dequantize_q1_0,
    dequantize_q2_k,
    dequantize_q3_k,
    dequantize_q4_0,
    dequantize_q4_1,
    dequantize_q4_k,
    dequantize_q5_0,
    dequantize_q5_1,
    dequantize_q5_k,
    dequantize_q6_k,
    dequantize_q8_0,
    dequantize_tq1_0,
    dequantize_tq2_0,
)


@weft.kernel
def row_dequantize_q1_0(W: View[Q1_0, (K,)], Y: View[f32, (K,)]):
    dequantize_q1_0(W, Y)


@weft.kernel
def row_dequantize_q4_0(W: View[Q4_0, (K,)], Y: View[f32, (K,)]):
    dequantize_q4_0(W, Y)


@weft.kernel
def row_dequantize_q4_1(W: View[Q4_1, (K,)], Y: View[f32, (K,)]):
    dequantize_q4_1(W, Y)


@weft.kernel
def row_dequantize_q5_0(W: View[Q5_0, (K,)], Y: View[f32, (K,)]):
    dequantize_q5_0(W, Y)


@weft.kernel
def row_dequantize_q5_1(W: View[Q5_1, (K,)], Y: View[f32, (K,)]):
    dequantize_q5_1(W, Y)


@weft.kernel
def row_dequantize_q8_0(W: View[Q8_0, (K,)], Y: View[f32, (K,)]):
    dequantize_q8_0(W, Y)


@weft.kernel
def row_dequantize_q2_k(W: View[Q2_K, (K,)], Y: View[f32, (K,)]):
    dequantize_q2_k(W, Y)


@weft.kernel
def row_dequantize_q3_k(W: View[Q3_K, (K,)], Y: View[f32, (K,)]):
    dequantize_q3_k(W, Y)


@weft.kernel
def row_dequantize_q4_k(W: View[Q4_K, (K,)], Y: View[f32, (K,)]):
    dequantize_q4_k(W, Y)


@weft.kernel
def row_dequantize_q5_k(W: View[Q5_K, (K,)], Y: View[f32, (K,)]):
    dequantize_q5_k(W, Y)


@weft.kernel
def row_dequantize_q6_k(W: View[Q6_K, (K,)], Y: View[f32, (K,)]):
    dequantize_q6_k(W, Y)


@weft.kernel
def row_dequantize_iq1_s(
    W: View[IQ1_S, (K,)], grid: View[i8, (16384,)], Y: View[f32, (K,)]
):
    dequantize_iq1_s(W, grid, Y)


@weft.kernel
def row_dequantize_iq1_m(
    W: View[IQ1_M, (K,)],
    grid: View[i8, (16384,)],
    f16_bits: View[f32, (65536,)],
    Y: View[f32, (K,)],
):
    dequantize_iq1_m(W, grid, f16_bits, Y)


@weft.kernel
def row_dequantize_iq2_s(
    W: View[IQ2_S, (K,)], grid: View[i8, (8192,)], Y: View[f32, (K,)]
):
    dequantize_iq2_s(W, grid, Y)


@weft.kernel
def row_dequantize_iq2_xs(
    W: View[IQ2_XS, (K,)],
    grid: View[i8, (4096,)],
    signs: View[i8, (1024,)],
    Y: View[f32, (K,)],
):
    dequantize_iq2_xs(W, grid, signs, Y)


@weft.kernel
def row_dequantize_iq2_xxs(
    W: View[IQ2_XXS, (K,)],
    grid: View[i8, (2048,)],
    signs: View[i8, (1024,)],
    Y: View[f32, (K,)],
):
    dequantize_iq2_xxs(W, grid, signs, Y)


@weft.kernel
def row_dequantize_iq3_s(
    W: View[IQ3_S, (K,)], grid: View[i8, (2048,)], Y: View[f32, (K,)]
):
    dequantize_iq3_s(W, grid, Y)


@weft.kernel
def row_dequantize_iq3_xxs(
    W: View[IQ3_XXS, (K,)],
    grid: View[i8, (1024,)],
    signs: View[i8, (1024,)],
    Y: View[f32, (K,)],
):
    dequantize_iq3_xxs(W, grid, signs, Y)


@weft.kernel
def row_dequantize_iq4_nl(
    W: View[IQ4_NL, (K,)], codebook: View[i8, (16,)], Y: View[f32, (K,)]
):
    dequantize_iq4_nl(W, codebook, Y)


@weft.kernel
def row_dequantize_iq4_xs(
    W: View[IQ4_XS, (K,)], codebook: View[i8, (16,)], Y: View[f32, (K,)]
):
    dequantize_iq4_xs(W, codebook, Y)


@weft.kernel
def row_dequantize_tq1_0(
    W: View[TQ1_0, (K,)], powers: View[u32, (5,)], Y: View[f32, (K,)]
):
    dequantize_tq1_0(W, powers, Y)


@weft.kernel
def row_dequantize_tq2_0(W: View[TQ2_0, (K,)], Y: View[f32, (K,)]):
    dequantize_tq2_0(W, Y)


@weft.kernel
def row_dequantize_mxfp4(
    W: View[MXFP4, (K,)],
    codebook: View[i8, (16,)],
    scale: View[f32, (256,)],
    Y: View[f32, (K,)],
):
    dequantize_mxfp4(W, codebook, scale, Y)


@weft.kernel
def row_dequantize_nvfp4(
    W: View[NVFP4, (K,)],
    codebook: View[i8, (16,)],
    scale: View[f32, (256,)],
    Y: View[f32, (K,)],
):
    dequantize_nvfp4(W, codebook, scale, Y)
