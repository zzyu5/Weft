#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <vec_dot|quantize|dequantize|forward|spacemit_rvv|ime1> <kernel> <repetitions>" >&2
  exit 2
fi

family=$1
kernel=$2
repetitions=$3
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
remote_host=rvv
remote_source=/home/ubuntu/llama.cpp-upstream-native
remote_build=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv
remote_cxx=/opt/tcrv-toolchains/gcc-15.2.0/bin/g++
remote_cpu=8
remote_march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause
remote_extra_cxx_flags=
remote_extra_define=

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
  spacemit_rvv | ime1)
    if [[ ${family} == spacemit_rvv ]]; then
      runtime=examples/repro/ggml/forward_runtime.cpp
      remote_extra_cxx_flags=-fno-integrated-as
      remote_extra_define=-DGGML_BASELINE_SPACEMIT=1
    else
      runtime=examples/repro/ggml/ime1_runtime.cpp
      remote_extra_cxx_flags=-fno-integrated-as
    fi
    remote_host=k1
    remote_source=/home/bianbu/tcrv-k1-llama
    remote_build=/home/bianbu/tcrv-k1-llama/build-ime
    remote_cxx=/usr/bin/clang++-18
    remote_cpu=3
    remote_march=rv64gcv_zfh_zvfh_zicbop_zihintpause_zba
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
printf -v source_argument '%q' "${remote_source}"
printf -v build_argument '%q' "${remote_build}"
printf -v cxx_argument '%q' "${remote_cxx}"
printf -v cpu_argument '%q' "${remote_cpu}"
printf -v march_argument '%q' "${remote_march}"
printf -v extra_cxx_flags_argument '%q' "${remote_extra_cxx_flags}"
printf -v extra_define_argument '%q' "${remote_extra_define}"

tar -C "${local_root}" -cf - runtime.cpp |
  ssh "${remote_host}" "
    set -eu
    remote_root=\$(mktemp -d /tmp/ggml-kernel.XXXXXX)
    cleanup() {
      exit_status=\$?
      trap - EXIT
      if ! find \"\${remote_root}\" -depth -delete; then
        echo \"failed to remove remote temporary directory: \${remote_root}\" >&2
        if [ \"\${exit_status}\" -eq 0 ]; then
          exit_status=1
        fi
      fi
      exit \"\${exit_status}\"
    }
    trap cleanup EXIT
    tar -C \"\${remote_root}\" -xf -
    cd \"\${remote_root}\"

    source_root=${source_argument}
    build_root=${build_argument}
    cxx=${cxx_argument}
    cpu=${cpu_argument}
    march=${march_argument}
    extra_cxx_flags=${extra_cxx_flags_argument}
    extra_define=${extra_define_argument}

    \"\${cxx}\" -O3 -std=c++17 -Wall -Wextra -Werror \${extra_cxx_flags} \${extra_define} \
      -march=\"\${march}\" -mabi=lp64d \
      -I\"\${source_root}/ggml/include\" \
      -I\"\${source_root}/ggml/src\" \
      -I\"\${source_root}/ggml/src/ggml-cpu\" \
      -I\"\${source_root}/ggml/src/ggml-cpu/spacemit\" \
      runtime.cpp -L\"\${build_root}/bin\" \
      -L/opt/tcrv-toolchains/gcc-15.2.0/lib \
      -Wl,-rpath,\"\${build_root}/bin:/opt/tcrv-toolchains/gcc-15.2.0/lib\" \
      -Wl,--no-as-needed \
      -lggml -lggml-cpu -lggml-base -lgomp -lm -ldl -pthread \
      -o ggml_kernel_runtime

    exec taskset -c \"\${cpu}\" ./ggml_kernel_runtime ${kernel_argument} ${repetitions_argument}
  "
