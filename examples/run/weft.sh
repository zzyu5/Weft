#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <kernel> [kernel-specific arguments]" >&2
  exit 2
fi

kernel=$1
shift
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
compiler="${project_root}/build/tools/weft-compile/weft-compile"
local_root=$(mktemp -d /tmp/weft-kernel.XXXXXX)
cleanup_local() {
  status=$?
  trap - EXIT
  if ! find "${local_root}" -depth -delete; then
    echo "failed to remove local temporary directory: ${local_root}" >&2
    if [[ ${status} -eq 0 ]]; then
      status=1
    fi
  fi
  exit "${status}"
}
trap cleanup_local EXIT

quant=0
runtime_arguments=()
case "${kernel}" in
  add_bias)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 add_bias" >&2
      exit 2
    fi
    dsl=examples/kernels/elementwise/add_bias.py
    runtime=examples/repro/weft/elementwise/add_bias_runtime.cpp
    ;;
  silu)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 silu" >&2
      exit 2
    fi
    dsl=examples/kernels/pointwise/silu.py
    runtime=examples/repro/weft/pointwise/silu_runtime.cpp
    ;;
  rms_norm)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 rms_norm" >&2
      exit 2
    fi
    dsl=examples/kernels/normalization/rms_norm.py
    runtime=examples/repro/weft/normalization/rms_norm_runtime.cpp
    ;;
  softmax)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 softmax" >&2
      exit 2
    fi
    dsl=examples/kernels/normalization/online_softmax.py
    runtime=examples/repro/weft/normalization/online_softmax_runtime.cpp
    ;;
  blocked_gemm)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 blocked_gemm" >&2
      exit 2
    fi
    dsl=examples/kernels/contraction/blocked_gemm.py
    runtime=examples/repro/weft/contraction/blocked_gemm_runtime.cpp
    ;;
  q4_0_q8_0 | q4_1_q8_1 | q5_0_q8_0 | q5_1_q8_1 | q8_0_q8_0)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 ${kernel}" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/block_dot.py
    quant=1
    ;;
  q4_K_q8_K)
    if [[ $# -ne 3 ]]; then
      echo "usage: $0 q4_K_q8_K <attn_q|attn_k|attn_output|ffn_gate|ffn_up> <decode|prefill> <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/block_dot.py
    quant=1
    runtime_arguments=("$1" "$2" "$3")
    ;;
  *)
    echo "unsupported Weft kernel: ${kernel}" >&2
    exit 2
    ;;
esac

runtime_command=./weft_runtime
for argument in "${runtime_arguments[@]}"; do
  printf -v escaped_argument '%q' "${argument}"
  runtime_command+=" ${escaped_argument}"
done

if [[ ! -x "${compiler}" ]]; then
  echo "missing compiler executable: ${compiler}" >&2
  exit 1
fi

if [[ ${quant} -eq 1 ]]; then
  PYTHONPATH="${project_root}/python" python3 -m weft \
    "${project_root}/${dsl}" --kernel "${kernel}" |
    "${compiler}" --emit=intrinsic-c --march=rv64gcv --abi=lp64d \
      -o "${local_root}/kernel.c"
  mkdir -p "${local_root}/leaf" "${local_root}/common"
  cp "${project_root}/examples/repro/weft/quantization/block_dot/${kernel}/runtime.c" \
    "${local_root}/leaf/runtime.c"
  cp "${project_root}/examples/repro/weft/quantization/block_dot/common/ggml_quant.h" \
    "${local_root}/common/ggml_quant.h"
else
  if [[ ${kernel} == blocked_gemm ]]; then
    PYTHONPATH="${project_root}/python" python3 -m weft "${project_root}/${dsl}" |
      "${compiler}" --emit=intrinsic-c --march=rv64gcv --abi=lp64d \
        --meta=BM=16 --meta=BN=16 --meta=BK=16 -o "${local_root}/kernel.c"
  else
    PYTHONPATH="${project_root}/python" python3 -m weft "${project_root}/${dsl}" |
      "${compiler}" --emit=intrinsic-c --march=rv64gcv --abi=lp64d \
        -o "${local_root}/kernel.c"
  fi
  cp "${project_root}/${runtime}" "${local_root}/runtime.cpp"
fi

tar -C "${local_root}" -cf - . |
  ssh rvv "
    set -eu
    remote_root=\$(mktemp -d /tmp/weft-kernel.XXXXXX)
    cleanup() {
      status=\$?
      trap - EXIT
      if ! find \"\${remote_root}\" -depth -delete; then
        echo \"failed to remove remote temporary directory: \${remote_root}\" >&2
        if [ \"\${status}\" -eq 0 ]; then
          status=1
        fi
      fi
      exit \"\${status}\"
    }
    trap cleanup EXIT
    tar -C \"\${remote_root}\" -xf -
    cd \"\${remote_root}\"
    cc=/opt/tcrv-toolchains/gcc-15.2.0/bin/gcc
    cxx=/opt/tcrv-toolchains/gcc-15.2.0/bin/g++
    if [ '${quant}' -eq 1 ]; then
      \"\${cc}\" -O3 -std=c11 -Wall -Wextra -Werror \
        -march=rv64gcv -mabi=lp64d -c kernel.c -o kernel.o
      ar rcs libweft_kernel.a kernel.o
      \"\${cc}\" -O3 -std=c11 -Wall -Wextra -Werror \
        -march=rv64gcv -mabi=lp64d \
        -c leaf/runtime.c -o runtime.o
      if [ '${kernel}' = q4_K_q8_K ]; then
        ggml_build=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
        \"\${cxx}\" -march=rv64gcv -mabi=lp64d runtime.o libweft_kernel.a \
          -L/opt/tcrv-toolchains/gcc-15.2.0/lib \
          -L\"\${ggml_build}\" -Wl,-rpath,\"\${ggml_build}\" \
          -Wl,--no-as-needed -lggml-cpu -lggml-base -lgomp -lm -ldl -pthread \
          -o weft_runtime
      else
        \"\${cxx}\" -march=rv64gcv -mabi=lp64d runtime.o libweft_kernel.a \
          -L/opt/tcrv-toolchains/gcc-15.2.0/lib -lm -o weft_runtime
      fi
    else
      \"\${cc}\" -O3 -std=c11 -Wall -Wextra -Werror \
        -march=rv64gcv -mabi=lp64d -c kernel.c -o kernel.o
      ar rcs libweft_kernel.a kernel.o
      \"\${cxx}\" -O3 -std=c++17 -Wall -Wextra -Werror \
        -march=rv64gcv -mabi=lp64d -c runtime.cpp -o runtime.o
      \"\${cxx}\" -march=rv64gcv -mabi=lp64d runtime.o libweft_kernel.a \
        -L/opt/tcrv-toolchains/gcc-15.2.0/lib -lm \
        -o weft_runtime
    fi
    exec taskset -c 8 ${runtime_command}
  "
