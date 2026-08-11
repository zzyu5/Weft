#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <vec_dot|quantize|dequantize|forward> <kernel> <repetitions>" >&2
  exit 2
fi

family=$1
kernel=$2
repetitions=$3
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

case "${family}" in
  vec_dot)
    runtime=examples/repro/ggml/vec_dot_runtime.cpp
    ;;
  quantize)
    runtime=examples/repro/ggml/quantize_runtime.cpp
    ;;
  dequantize)
    runtime=examples/repro/ggml/dequantize_runtime.cpp
    ;;
  forward)
    runtime=examples/repro/ggml/forward_runtime.cpp
    ;;
  *)
    echo "unsupported GGML kernel family: ${family}" >&2
    exit 2
    ;;
esac

local_root=$(mktemp -d /tmp/ggml-kernel.XXXXXX)
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

cp "${project_root}/${runtime}" "${local_root}/runtime.cpp"
printf -v kernel_argument '%q' "${kernel}"
printf -v repetitions_argument '%q' "${repetitions}"

tar -C "${local_root}" -cf - runtime.cpp |
  ssh rvv "
    set -eu
    remote_root=\$(mktemp -d /tmp/ggml-kernel.XXXXXX)
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

    source_root=/home/ubuntu/llama.cpp-upstream-native
    build_root=\"\${source_root}/build-gcc15-rv64gcv\"
    cxx=/opt/tcrv-toolchains/gcc-15.2.0/bin/g++
    march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause

    \"\${cxx}\" -O3 -std=c++17 -Wall -Wextra -Werror \
      -march=\"\${march}\" -mabi=lp64d \
      -I\"\${source_root}/ggml/include\" \
      -I\"\${source_root}/ggml/src\" \
      -I\"\${source_root}/ggml/src/ggml-cpu\" \
      runtime.cpp -L\"\${build_root}/bin\" \
      -L/opt/tcrv-toolchains/gcc-15.2.0/lib \
      -Wl,-rpath,\"\${build_root}/bin:/opt/tcrv-toolchains/gcc-15.2.0/lib\" \
      -Wl,--no-as-needed \
      -lggml -lggml-cpu -lggml-base -lgomp -lm -ldl -pthread \
      -o ggml_kernel_runtime

    exec taskset -c 8 ./ggml_kernel_runtime ${kernel_argument} ${repetitions_argument}
  "
