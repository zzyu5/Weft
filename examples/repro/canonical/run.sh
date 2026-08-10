#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
weft_opt="${repo_root}/build/tools/weft-opt/weft-opt"

if [[ ! -x "${weft_opt}" ]]; then
  echo "missing ${weft_opt}; build the weft-opt target first" >&2
  exit 1
fi

sources=(
  source/weft/weft/elementwise/add_bias/add_bias.py
  source/weft/weft/normalization/rms_norm/rms_norm.py
  source/weft/weft/reduction/predicate_reduce/predicate_reduce.py
  source/weft/weft/reduction/online_softmax_summary/online_softmax_summary.py
  source/weft/weft/gemm/blocked/blocked_gemm.py
  source/weft/weft/extension/block_scaled_contract/block_scaled_contract.py
  source/weft/weft/normalization/online_softmax/online_softmax.py
)

for source in "${sources[@]}"; do
  echo "[canonical] ${source}"
  PYTHONPATH="${repo_root}/python" \
    python3 -m weft "${repo_root}/${source}" | "${weft_opt}" >/dev/null
done
