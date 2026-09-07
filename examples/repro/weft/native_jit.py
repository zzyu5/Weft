from __future__ import annotations

import argparse
import array
import ctypes
import json
import math
from pathlib import Path

import weft
from kernels.dense.gemm_f32 import gemm_f32
from kernels.vec_dot.q4_0_q8_0 import quantized_vec_dot_q4_0_q8_0
from weft.runtime import Buffer, CompileOptions, Toolchain


def require_numeric(actual: float, expected: float) -> float:
    error = abs(actual - expected)
    if not math.isfinite(actual) or not math.isfinite(expected) or error > 1e-4 + 2e-3 * abs(expected):
        raise RuntimeError(f"numeric mismatch: actual={actual}, expected={expected}")
    return error


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real dense and encoded kernels through native Weft JIT")
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--cc", required=True)
    parser.add_argument("--cflag", action="append", default=[])
    parser.add_argument("--ggml-lib-dir", required=True, type=Path)
    args = parser.parse_args()
    toolchain = Toolchain(args.compiler, (args.cc,), tuple(args.cflag))

    m, n, k = 16, 32, 256
    lhs = array.array("f", [1.0 / k]) * (m * k)
    rhs = array.array("f", [1.0]) * (n * k)
    output = array.array("f", [0.0]) * (m * n)
    views = (Buffer(lhs, shape=(m, k)), Buffer(rhs, shape=(n, k)),
             Buffer(output, shape=(m, n)))
    options = CompileOptions(meta={"MC": 64, "NC": 16, "MR": 2, "NR": 2}, lmul_eighths=16)
    dense = weft.jit(gemm_f32, options=options, toolchain=toolchain)
    dense(*views)
    dense(*views)
    dense_error = max(require_numeric(value, 1.0) for value in output)
    if weft.compile(gemm_f32, options=options, toolchain=toolchain) is not dense.compiled:
        raise RuntimeError("same-binding JIT invocation did not reuse its artifact")
    alternative = weft.compile(gemm_f32, options=CompileOptions(
        meta=options.meta, lmul_eighths=16, unroll=2), toolchain=toolchain)
    if alternative is dense.compiled:
        raise RuntimeError("different compile bindings reused the same artifact")
    alternative(*views)
    dense_error = max(dense_error, *(require_numeric(value, 1.0) for value in output))

    base = ctypes.CDLL(str(args.ggml_lib_dir / "libggml-base.so"), mode=ctypes.RTLD_GLOBAL)
    cpu = ctypes.CDLL(str(args.ggml_lib_dir / "libggml-cpu.so"), mode=ctypes.RTLD_GLOBAL)
    quantize_weight = base.quantize_row_q4_0_ref
    quantize_activation = base.quantize_row_q8_0_ref
    for quantize in (quantize_weight, quantize_activation):
        quantize.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        quantize.restype = None
    reference = cpu.ggml_vec_dot_q4_0_q8_0_generic
    reference.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                         ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                         ctypes.c_size_t, ctypes.c_int]
    reference.restype = None
    weights = array.array("f", [((index * 19 + 7) % 127 - 63) / 13.0 for index in range(32)])
    activation = array.array("f", [((index * 37) % 251 - 125) / 17.0 for index in range(32)])
    weight_record, activation_record = bytearray(18), bytearray(34)
    weight_input, activation_input = Buffer(weights), Buffer(activation)
    weight_storage, activation_storage = Buffer(weight_record), Buffer(activation_record)
    quantize_weight(weight_input.address, weight_storage.address, 32)
    quantize_activation(activation_input.address, activation_storage.address, 32)
    w = Buffer(weight_record * (k // 32), shape=(k,), encoding="Q4_0")
    x = Buffer(activation_record * (k // 32), shape=(k,), encoding="Q8_0")
    y = array.array("f", [0.0])
    expected = ctypes.c_float()
    reference(k, ctypes.byref(expected), 0, w.address, 0, x.address, 0, 1)
    encoded = weft.jit(quantized_vec_dot_q4_0_q8_0, toolchain=toolchain)
    encoded(w, x, y)
    encoded(w, x, y)
    encoded_error = require_numeric(y[0], expected.value)
    print(json.dumps({
        "execution": "native-jit", "march": dense.compiled.target.march,
        "vlen_bits": dense.compiled.target.vlen_bits, "shape": [m, n, k],
        "dense_numeric": "within-tolerance", "dense_max_absolute_error": dense_error,
        "encoded_numeric": "within-tolerance", "encoded_max_absolute_error": encoded_error,
        "same_binding_reused": True, "different_binding_specialized": True,
    }, sort_keys=True))
    dense.compiled.close()
    alternative.close()
    encoded.compiled.close()


if __name__ == "__main__":
    main()
