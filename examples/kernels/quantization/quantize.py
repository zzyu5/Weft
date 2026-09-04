from __future__ import annotations

from weft.language import View, f32

from quantization.encodings import Q8_0, Q8_1, Q8_K
from quantization.q8_0_quantize import _quantize_row_q8_0
from quantization.q8_1_quantize import _quantize_row_q8_1
from quantization.q8_K_quantize import _quantize_row_q8_K


def quantize_q8_0(X: View[f32, (M, K)], Y: View[Q8_0, (M, K)]):
    for row in range(M):
        _quantize_row_q8_0(X[row], Y[row])


def quantize_q8_1(X: View[f32, (M, K)], Y: View[Q8_1, (M, K)]):
    for row in range(M):
        _quantize_row_q8_1(X[row], Y[row])


def quantize_q8_K(X: View[f32, (M, K)], Y: View[Q8_K, (M, K)]):
    for row in range(M):
        _quantize_row_q8_K(X[row], Y[row])
