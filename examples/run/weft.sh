#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <add_bias|rms_norm|online_softmax|blocked_gemm|q4_0_q8_0|q4_1_q8_1|q5_0_q8_0|q5_1_q8_1|q8_0_q8_0|q4_K_q8_K>" >&2
  exit 2
fi

kernel=$1
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
case "${kernel}" in
  add_bias)
    dsl=examples/kernels/elementwise/add_bias.py
    runtime=examples/kernels/elementwise/add_bias_runtime.cpp
    ;;
  rms_norm)
    dsl=examples/kernels/normalization/rms_norm.py
    runtime=examples/kernels/normalization/rms_norm_runtime.cpp
    ;;
  online_softmax)
    dsl=examples/kernels/normalization/online_softmax.py
    runtime=examples/kernels/normalization/online_softmax_runtime.cpp
    ;;
  blocked_gemm)
    dsl=examples/kernels/contraction/blocked_gemm.py
    runtime=examples/kernels/contraction/blocked_gemm_runtime.cpp
    ;;
  q4_0_q8_0 | q4_1_q8_1 | q5_0_q8_0 | q5_1_q8_1 | q8_0_q8_0 | q4_K_q8_K)
    dsl=examples/kernels/quantization/block_dot.py
    quant=1
    ;;
  *)
    echo "unsupported Weft kernel: ${kernel}" >&2
    exit 2
    ;;
esac

if [[ ! -x "${compiler}" ]]; then
  echo "missing compiler executable: ${compiler}" >&2
  exit 1
fi

if [[ ${quant} -eq 1 ]]; then
  PYTHONPATH="${project_root}/python" python3 -m weft \
    "${project_root}/${dsl}" --kernel "${kernel}" |
    "${compiler}" --emit=source --march=rv64gcv --abi=lp64d \
      -o "${local_root}/kernel.cpp"
  mkdir -p "${local_root}/leaf" "${local_root}/common"
  cp "${project_root}/source/c/ggml/quantization/block_dot/${kernel}/${kernel}_runtime.c" \
    "${local_root}/leaf/runtime.c"
  cp "${project_root}/source/c/ggml/quantization/block_dot/common/ggml_quant.h" \
    "${local_root}/common/ggml_quant.h"
else
  if [[ ${kernel} == blocked_gemm ]]; then
    PYTHONPATH="${project_root}/python" python3 -m weft "${project_root}/${dsl}" |
      "${compiler}" --emit=source --march=rv64gcv --abi=lp64d \
        --meta=BM=16 --meta=BN=16 --meta=BK=16 -o "${local_root}/kernel.cpp"
  else
    PYTHONPATH="${project_root}/python" python3 -m weft "${project_root}/${dsl}" |
      "${compiler}" --emit=source --march=rv64gcv --abi=lp64d \
        -o "${local_root}/kernel.cpp"
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
    if [ '${quant}' -eq 1 ]; then
      g++ -O2 -std=c++17 -Wall -Wextra -Werror -c kernel.cpp -o kernel.o
      clang -O2 -std=c11 -Wall -Wextra -Werror \
        -Dggml_source_${kernel}=${kernel} -c leaf/runtime.c -o runtime.o
      g++ kernel.o runtime.o -lm -o weft_runtime
    else
      g++ -O2 -std=c++17 -Wall -Wextra -Werror \
        kernel.cpp runtime.cpp -lm -o weft_runtime
    fi
    ./weft_runtime
  "
