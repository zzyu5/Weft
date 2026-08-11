#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <remote-model.gguf> <prompt-tokens> <generation-tokens> <threads> <repetitions>" >&2
  exit 2
fi

remote_model=$1
prompt_tokens=$2
generation_tokens=$3
threads=$4
repetitions=$5
remote_bench=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin/llama-bench

printf -v model_arg '%q' "${remote_model}"
printf -v prompt_arg '%q' "${prompt_tokens}"
printf -v generation_arg '%q' "${generation_tokens}"
printf -v threads_arg '%q' "${threads}"
printf -v repetitions_arg '%q' "${repetitions}"
printf -v bench_arg '%q' "${remote_bench}"

ssh rvv "
  set -eu
  model=${model_arg}
  bench=${bench_arg}
  if [ ! -f \"\${model}\" ]; then
    echo \"missing GGUF model: \${model}\" >&2
    exit 1
  fi
  if [ ! -x \"\${bench}\" ]; then
    echo \"missing llama-bench: \${bench}\" >&2
    exit 1
  fi
  exec taskset -c 8-15 \"\${bench}\" \
    -m \"\${model}\" \
    -p ${prompt_arg} \
    -n ${generation_arg} \
    -b 2048 \
    -ub 512 \
    -t ${threads_arg} \
    -r ${repetitions_arg} \
    -o json
"
