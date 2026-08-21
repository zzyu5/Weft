#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <sg2044|k1> <kernel> <repetitions>" >&2
  exit 2
fi

target=$1
kernel=$2
repetitions=$3
run_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

IFS=',' read -r -a unrolls <<< "${WEFT_TUNE_UNROLLS:-1,2,4}"
IFS=',' read -r -a pipelines <<< "${WEFT_TUNE_PIPELINE_DEPTHS:-1,2}"
IFS=',' read -r -a prefetches <<< "${WEFT_TUNE_PREFETCH_DISTANCES:-0,1,2}"

meta_configs=("")
if [[ -n ${WEFT_TUNE_META_CHOICES:-} ]]; then
  IFS=';' read -r -a dimensions <<< "${WEFT_TUNE_META_CHOICES}"
  for dimension in "${dimensions[@]}"; do
    name=${dimension%%=*}
    values=${dimension#*=}
    if [[ -z ${name} || ${values} == "${dimension}" ]]; then
      echo "invalid WEFT_TUNE_META_CHOICES dimension: ${dimension}" >&2
      exit 2
    fi
    IFS=',' read -r -a choices <<< "${values}"
    expanded=()
    for existing in "${meta_configs[@]}"; do
      for choice in "${choices[@]}"; do
        if [[ -z ${choice} ]]; then
          echo "empty meta choice for ${name}" >&2
          exit 2
        fi
        if [[ -n ${existing} ]]; then
          expanded+=("${existing};${name}=${choice}")
        else
          expanded+=("${name}=${choice}")
        fi
      done
    done
    meta_configs=("${expanded[@]}")
  done
fi

results=$(mktemp /tmp/weft-tune.XXXXXX)
cleanup() {
  status=$?
  trap - EXIT
  rm -f -- "${results}"
  exit "${status}"
}
trap cleanup EXIT

for meta in "${meta_configs[@]}"; do
  for unroll in "${unrolls[@]}"; do
    for pipeline in "${pipelines[@]}"; do
      for prefetch in "${prefetches[@]}"; do
        output=$(
          WEFT_META_BINDINGS="${meta}" \
          WEFT_AUTO_UNROLL="${unroll}" \
          WEFT_AUTO_PIPELINE_DEPTH="${pipeline}" \
          WEFT_AUTO_PREFETCH_DISTANCE="${prefetch}" \
            "${run_dir}/weft-kernel.sh" "${target}" "${kernel}" "${repetitions}"
        )
        numeric=$(sed -n 's/^numeric=//p' <<< "${output}")
        metric_name=cold_gop_s
        metric=$(sed -n 's/^cold_gop_s=//p' <<< "${output}")
        if [[ -z ${metric} ]]; then
          metric_name=melements_s
          metric=$(sed -n 's/^melements_s=//p' <<< "${output}")
        fi
        if [[ ${numeric} != bit-exact || -z ${metric} ]]; then
          echo "candidate produced no bit-exact measurable result" >&2
          exit 1
        fi
        printf 'unroll=%s pipeline_depth=%s prefetch_distance=%s meta=%s %s=%s\n' \
          "${unroll}" "${pipeline}" "${prefetch}" "${meta:-none}" \
          "${metric_name}" "${metric}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${metric}" "${unroll}" "${pipeline}" "${prefetch}" \
          "${meta:-none}" "${metric_name}" >> "${results}"
      done
    done
  done
done

winner=$(sort -t $'\t' -k1,1gr "${results}" | head -n 1)
IFS=$'\t' read -r metric unroll pipeline prefetch meta metric_name <<< "${winner}"
printf 'winner unroll=%s pipeline_depth=%s prefetch_distance=%s meta=%s %s=%s\n' \
  "${unroll}" "${pipeline}" "${prefetch}" "${meta}" "${metric_name}" "${metric}"
