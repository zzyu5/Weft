#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
compiler=${CC:-cc}
out_dir=$(mktemp -d "${TMPDIR:-/tmp}/weft-source-repro.XXXXXX")
trap 'rm -rf "${out_dir}"' EXIT

baselines=(
  source/c/weft/elementwise/add_bias
  source/c/weft/normalization/rms_norm
  source/c/weft/normalization/online_softmax
  source/c/weft/gemm/blocked
  source/c/ggml/quantization/block_dot/q4_0_q8_0
  source/c/ggml/quantization/block_dot/q4_1_q8_1
  source/c/ggml/quantization/block_dot/q5_0_q8_0
  source/c/ggml/quantization/block_dot/q5_1_q8_1
  source/c/ggml/quantization/block_dot/q8_0_q8_0
  source/c/ggml/quantization/block_dot/q4_K_q8_K
)

for baseline in "${baselines[@]}"; do
  name=${baseline##*/}
  output="${out_dir}/${name}"
  echo "[c-source] ${baseline}"
  "${compiler}" -std=c11 -O2 -Wall -Wextra -Werror \
    "${repo_root}/${baseline}/reference.c" \
    "${repo_root}/${baseline}/runtime.c" \
    -lm -o "${output}"
  "${output}"
done
