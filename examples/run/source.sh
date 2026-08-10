#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <add_bias|rms_norm|online_softmax|blocked_gemm|q4_0_q8_0|q4_1_q8_1|q5_0_q8_0|q5_1_q8_1|q8_0_q8_0|q4_K_q8_K>" >&2
  exit 2
fi

kernel=$1
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

case "${kernel}" in
  add_bias)
    leaf=elementwise/add
    source_file=add.cpp
    runtime_file=add_runtime.cpp
    compiler=g++
    standard=c++17
    ;;
  rms_norm)
    leaf=normalization/rms_norm
    source_file=rms_norm.cpp
    runtime_file=rms_norm_runtime.cpp
    compiler=g++
    standard=c++17
    ;;
  online_softmax)
    leaf=normalization/online_softmax
    source_file=online_softmax.cpp
    runtime_file=online_softmax_runtime.cpp
    compiler=g++
    standard=c++17
    ;;
  blocked_gemm)
    leaf=gemm/blocked
    source_file=mul_mat.cpp
    runtime_file=mul_mat_runtime.cpp
    compiler=g++
    standard=c++17
    ;;
  q4_0_q8_0 | q4_1_q8_1 | q5_0_q8_0 | q5_1_q8_1 | q8_0_q8_0 | q4_K_q8_K)
    leaf=quantization/block_dot/${kernel}
    source_file=${kernel}.c
    runtime_file=${kernel}_runtime.c
    compiler=clang
    standard=c11
    ;;
  *)
    echo "unsupported source kernel: ${kernel}" >&2
    exit 2
    ;;
esac

tar -C "${project_root}/source/c" -cf - ggml |
  ssh rvv "
    set -eu
    remote_root=\$(mktemp -d /tmp/ggml-source.XXXXXX)
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
    cd \"\${remote_root}/ggml/${leaf}\"
    ${compiler} -O2 -std=${standard} -Wall -Wextra -Werror \
      '${source_file}' '${runtime_file}' -lm -o source_runtime
    ./source_runtime
  "
