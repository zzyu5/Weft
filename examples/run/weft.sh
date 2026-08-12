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
target_march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause
target_vlen_bits=128
matrix_extension=none
remote_host=rvv
remote_cpu=8
remote_cc=/opt/tcrv-toolchains/gcc-15.2.0/bin/gcc
remote_cxx=/opt/tcrv-toolchains/gcc-15.2.0/bin/g++
remote_compile_flags=
remote_link_flags="-L/opt/tcrv-toolchains/gcc-15.2.0/lib -lm"
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
k_quant_dot=0
k_quant_kind=0
multi=0
multi_primary=
multi_equivalent=
runtime_arguments=()
backend_arguments=()
if [[ -n ${WEFT_BACKEND_CONFIG:-} ]]; then
  read -r -a backend_arguments <<< "${WEFT_BACKEND_CONFIG}"
fi
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
  swiglu)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 swiglu" >&2
      exit 2
    fi
    dsl=examples/kernels/pointwise/swiglu.py
    runtime=examples/repro/weft/pointwise/swiglu_runtime.cpp
    ;;
  rms_norm)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 rms_norm" >&2
      exit 2
    fi
    dsl=examples/kernels/normalization/rms_norm.py
    runtime=examples/repro/weft/normalization/rms_norm_runtime.cpp
    ;;
  layer_norm)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 layer_norm" >&2
      exit 2
    fi
    dsl=examples/kernels/normalization/layer_norm.py
    runtime=examples/repro/weft/normalization/layer_norm_runtime.cpp
    ;;
  rms_norm_backward)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 rms_norm_backward" >&2
      exit 2
    fi
    dsl=examples/kernels/normalization/rms_norm_backward.py
    runtime=examples/repro/weft/normalization/rms_norm_backward_runtime.cpp
    ;;
  cumsum)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 cumsum" >&2
      exit 2
    fi
    dsl=examples/kernels/reduction/cumsum.py
    runtime=examples/repro/weft/reduction/cumsum_runtime.cpp
    ;;
  segmented_scan)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 segmented_scan" >&2
      exit 2
    fi
    dsl=examples/kernels/reduction/segmented_scan.py
    runtime=examples/repro/weft/reduction/segmented_scan_runtime.cpp
    multi=1
    multi_primary=segmented_inclusive_scan_f32
    multi_equivalent=segmented_inclusive_scan_f32_equivalent
    ;;
  argmax)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 argmax" >&2
      exit 2
    fi
    dsl=examples/kernels/reduction/argmax.py
    runtime=examples/repro/weft/reduction/argmax_runtime.cpp
    ;;
  top_k)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 top_k" >&2
      exit 2
    fi
    dsl=examples/kernels/selection/top_k.py
    runtime=examples/repro/weft/selection/top_k_runtime.cpp
    multi=1
    multi_primary=top_k_f32
    multi_equivalent=top_k_f32_equivalent
    ;;
  argsort)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 argsort" >&2
      exit 2
    fi
    dsl=examples/kernels/selection/argsort.py
    runtime=examples/repro/weft/selection/argsort_runtime.cpp
    ;;
  ssm_conv)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 ssm_conv" >&2
      exit 2
    fi
    dsl=examples/kernels/state/ssm_conv.py
    runtime=examples/repro/weft/state/ssm_conv_runtime.cpp
    multi=1
    multi_primary=ssm_conv_f32
    multi_equivalent=ssm_conv_f32_equivalent
    ;;
  ssm_scan)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 ssm_scan" >&2
      exit 2
    fi
    dsl=examples/kernels/state/ssm_scan.py
    runtime=examples/repro/weft/state/ssm_scan_runtime.cpp
    ;;
  rwkv_wkv6)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 rwkv_wkv6" >&2
      exit 2
    fi
    dsl=examples/kernels/state/rwkv_wkv6.py
    runtime=examples/repro/weft/state/rwkv_wkv6_runtime.cpp
    ;;
  gated_linear_attention)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 gated_linear_attention" >&2
      exit 2
    fi
    dsl=examples/kernels/state/gated_linear_attention.py
    runtime=examples/repro/weft/state/gated_linear_attention_runtime.cpp
    ;;
  rwkv_wkv7)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 rwkv_wkv7" >&2
      exit 2
    fi
    dsl=examples/kernels/state/rwkv_wkv7.py
    runtime=examples/repro/weft/state/rwkv_wkv7_runtime.cpp
    ;;
  gated_delta_net)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 gated_delta_net" >&2
      exit 2
    fi
    dsl=examples/kernels/state/gated_delta_net.py
    runtime=examples/repro/weft/state/gated_delta_net_runtime.cpp
    ;;
  solve_triangular)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 solve_triangular" >&2
      exit 2
    fi
    dsl=examples/kernels/linear_algebra/solve_triangular.py
    runtime=examples/repro/weft/linear_algebra/solve_triangular_runtime.cpp
    ;;
  group_norm)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 group_norm" >&2
      exit 2
    fi
    dsl=examples/kernels/normalization/group_norm.py
    runtime=examples/repro/weft/normalization/group_norm_runtime.cpp
    ;;
  sam_relative_position)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 sam_relative_position" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/sam_relative_position.py
    runtime=examples/repro/weft/vision/sam_relative_position_runtime.cpp
    ;;
  depthwise_conv2d)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 depthwise_conv2d" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/depthwise_conv2d.py
    runtime=examples/repro/weft/vision/depthwise_conv2d_runtime.cpp
    ;;
  max_pool2d)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 max_pool2d" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/max_pool2d.py
    runtime=examples/repro/weft/vision/max_pool2d_runtime.cpp
    ;;
  bilinear_upscale)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 bilinear_upscale" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/bilinear_upscale.py
    runtime=examples/repro/weft/vision/bilinear_upscale_runtime.cpp
    ;;
  window_partition)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 window_partition" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/window_partition.py
    runtime=examples/repro/weft/vision/window_partition_runtime.cpp
    ;;
  dense_conv2d)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 dense_conv2d" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/dense_conv2d.py
    runtime=examples/repro/weft/vision/dense_conv2d_runtime.cpp
    ;;
  conv_transpose2d)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 conv_transpose2d" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/conv_transpose2d.py
    runtime=examples/repro/weft/vision/conv_transpose2d_runtime.cpp
    ;;
  im2col_backward)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 im2col_backward" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/im2col_backward.py
    runtime=examples/repro/weft/vision/im2col_backward_runtime.cpp
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
    if [[ $# -ne 2 ]]; then
      echo "usage: $0 blocked_gemm <decode|prefill> <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/contraction/blocked_gemm.py
    runtime=examples/repro/weft/contraction/blocked_gemm_runtime.cpp
    runtime_arguments=("$1" "$2")
    ;;
  blocked_gemm_f32)
    if [[ $# -ne 2 ]]; then
      echo "usage: $0 blocked_gemm_f32 <decode|prefill> <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/contraction/blocked_gemm_f32.py
    runtime=examples/repro/weft/contraction/blocked_gemm_f32_runtime.cpp
    runtime_arguments=("$1" "$2")
    ;;
  mul_mat_id)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 mul_mat_id" >&2
      exit 2
    fi
    dsl=examples/kernels/contraction/mul_mat_id.py
    runtime=examples/repro/weft/contraction/mul_mat_id_runtime.cpp
    ;;
  out_product)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 out_product" >&2
      exit 2
    fi
    dsl=examples/kernels/contraction/out_product.py
    runtime=examples/repro/weft/contraction/out_product_runtime.cpp
    ;;
  contiguous_transpose)
    if [[ $# -ne 1 ]]; then
      echo "usage: $0 contiguous_transpose <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/permutation/contiguous_transpose.py
    runtime=examples/repro/weft/permutation/contiguous_transpose_runtime.cpp
    runtime_arguments=("$1")
    ;;
  get_rows_q4_k)
    if [[ $# -ne 1 ]]; then
      echo "usage: $0 get_rows_q4_k <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/gather/get_rows_q4_k.py
    runtime=examples/repro/weft/gather/get_rows_q4_k_runtime.cpp
    runtime_arguments=("$1")
    ;;
  get_rows_f32)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 get_rows_f32" >&2
      exit 2
    fi
    dsl=examples/kernels/gather/get_rows_f32.py
    runtime=examples/repro/weft/gather/get_rows_f32_runtime.cpp
    ;;
  weighted_embedding_bag)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 weighted_embedding_bag" >&2
      exit 2
    fi
    dsl=examples/kernels/gather/weighted_embedding_bag.py
    runtime=examples/repro/weft/gather/weighted_embedding_bag_runtime.cpp
    multi=1
    multi_primary=weighted_embedding_bag_f32
    multi_equivalent=weighted_embedding_bag_f32_equivalent
    ;;
  csr_spmv)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 csr_spmv" >&2
      exit 2
    fi
    dsl=examples/kernels/sparse/csr_spmv.py
    runtime=examples/repro/weft/sparse/csr_spmv_runtime.cpp
    ;;
  add_id)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 add_id" >&2
      exit 2
    fi
    dsl=examples/kernels/gather/add_id.py
    runtime=examples/repro/weft/gather/add_id_runtime.cpp
    ;;
  get_rows_back)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 get_rows_back" >&2
      exit 2
    fi
    dsl=examples/kernels/gather/get_rows_back.py
    runtime=examples/repro/weft/gather/get_rows_back_runtime.cpp
    ;;
  set_rows)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 set_rows" >&2
      exit 2
    fi
    dsl=examples/kernels/gather/set_rows.py
    runtime=examples/repro/weft/gather/set_rows_runtime.cpp
    ;;
  rope_neox)
    if [[ $# -ne 1 ]]; then
      echo "usage: $0 rope_neox <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/rotation/rope_neox.py
    runtime=examples/repro/weft/rotation/rope_neox_runtime.cpp
    runtime_arguments=("$1")
    ;;
  flash_attention)
    if [[ $# -ne 1 ]]; then
      echo "usage: $0 flash_attention <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/attention/online_flash_attention.py
    runtime=examples/repro/weft/attention/online_flash_attention_runtime.cpp
    runtime_arguments=("$1")
    ;;
  causal_mask)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 causal_mask" >&2
      exit 2
    fi
    dsl=examples/kernels/attention/causal_mask.py
    runtime=examples/repro/weft/attention/causal_mask_runtime.cpp
    ;;
  q4_k_projection_ime)
    if [[ $# -ne 1 ]]; then
      echo "usage: $0 q4_k_projection_ime <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/ime/q4_k_projection.py
    runtime=examples/repro/weft/ime/q4_k_projection_runtime.cpp
    runtime_arguments=("$1")
    target_march=rv64gcv_zfh_zvfh_zicbop_zihintpause_zba
    target_vlen_bits=256
    matrix_extension=spacemit-ime1
    remote_host=k1
    remote_cpu=3
    remote_cc=/usr/bin/clang-18
    remote_cxx=/usr/bin/clang++-18
    remote_compile_flags=-fno-integrated-as
    remote_link_flags=-lm
    ;;
  q4_0_projection_ime)
    if [[ $# -ne 1 ]]; then
      echo "usage: $0 q4_0_projection_ime <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/ime/q4_0_projection.py
    runtime=examples/repro/weft/ime/q4_0_projection_runtime.cpp
    runtime_arguments=("$1")
    target_march=rv64gcv_zfh_zvfh_zicbop_zihintpause_zba
    target_vlen_bits=256
    matrix_extension=spacemit-ime1
    remote_host=k1
    remote_cpu=3
    remote_cc=/usr/bin/clang-18
    remote_cxx=/usr/bin/clang++-18
    remote_compile_flags=-fno-integrated-as
    remote_link_flags=-lm
    ;;
  q4_0_q8_0 | q4_1_q8_1 | q5_0_q8_0 | q5_1_q8_1 | q8_0_q8_0)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 ${kernel}" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/block_dot.py
    quant=1
    ;;
  q1_0_rows)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 q1_0_rows" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/q1_0.py
    runtime=examples/repro/weft/quantization/q1_0_runtime.cpp
    ;;
  mxfp4_rows)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 mxfp4_rows" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/mxfp4.py
    runtime=examples/repro/weft/quantization/mxfp4_runtime.cpp
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
  quantize_q8_0)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 quantize_q8_0" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/q8_0.py
    runtime=examples/repro/weft/quantization/q8_0_runtime.cpp
    ;;
  dequantize_iq4_nl)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 dequantize_iq4_nl" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/iq4_nl.py
    runtime=examples/repro/weft/quantization/iq4_nl_runtime.cpp
    ;;
  iq2_S_q8_K | iq3_S_q8_K | iq1_M_q8_K | q6_K_q8_K)
    if [[ $# -ne 0 ]]; then
      echo "usage: $0 ${kernel}" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/codebook_k.py
    k_quant_dot=1
    case "${kernel}" in
      iq1_M_q8_K) k_quant_kind=1 ;;
      iq2_S_q8_K) k_quant_kind=2 ;;
      iq3_S_q8_K) k_quant_kind=3 ;;
      q6_K_q8_K) k_quant_kind=6 ;;
    esac
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

if [[ ${k_quant_dot} -eq 1 ]]; then
  PYTHONPATH="${project_root}/python" python3 -m weft \
    "${project_root}/${dsl}" --kernel "${kernel}" |
    "${compiler}" --emit=intrinsic-c --march="${target_march}" --abi=lp64d \
      --vlen-bits="${target_vlen_bits}" --matrix-extension="${matrix_extension}" \
      "${backend_arguments[@]}" -o "${local_root}/kernel.c"
  cp "${project_root}/examples/repro/weft/quantization/codebook_k_runtime.cpp" \
    "${local_root}/runtime.cpp"
  cp "${project_root}/source/c/ggml/llama.cpp/ggml/src/ggml-common.h" \
    "${local_root}/ggml-common.h"
elif [[ ${quant} -eq 1 ]]; then
  PYTHONPATH="${project_root}/python" python3 -m weft \
    "${project_root}/${dsl}" --kernel "${kernel}" |
    "${compiler}" --emit=intrinsic-c --march="${target_march}" --abi=lp64d \
      --vlen-bits="${target_vlen_bits}" --matrix-extension="${matrix_extension}" \
      "${backend_arguments[@]}" -o "${local_root}/kernel.c"
  mkdir -p "${local_root}/leaf" "${local_root}/common"
  cp "${project_root}/examples/repro/weft/quantization/block_dot/${kernel}/runtime.c" \
    "${local_root}/leaf/runtime.c"
  cp "${project_root}/examples/repro/weft/quantization/block_dot/common/ggml_quant.h" \
    "${local_root}/common/ggml_quant.h"
else
  if [[ ${kernel} == blocked_gemm ]]; then
    PYTHONPATH="${project_root}/python" python3 -m weft "${project_root}/${dsl}" |
      "${compiler}" --emit=intrinsic-c --march="${target_march}" --abi=lp64d \
        --vlen-bits="${target_vlen_bits}" --matrix-extension="${matrix_extension}" \
        --meta=BM=4 --meta=BN=8 --meta=BK=64 "${backend_arguments[@]}" \
        -o "${local_root}/kernel.c"
  elif [[ ${multi} -eq 1 ]]; then
    PYTHONPATH="${project_root}/python" python3 -m weft "${project_root}/${dsl}" \
      --kernel "${multi_primary}" |
      "${compiler}" --emit=intrinsic-c --march="${target_march}" --abi=lp64d \
        --vlen-bits="${target_vlen_bits}" --matrix-extension="${matrix_extension}" \
        "${backend_arguments[@]}" -o "${local_root}/kernel.c"
    PYTHONPATH="${project_root}/python" python3 -m weft "${project_root}/${dsl}" \
      --kernel "${multi_equivalent}" |
      "${compiler}" --emit=intrinsic-c --march="${target_march}" --abi=lp64d \
        --vlen-bits="${target_vlen_bits}" --matrix-extension="${matrix_extension}" \
        "${backend_arguments[@]}" -o "${local_root}/kernel_equivalent.c"
  else
    PYTHONPATH="${project_root}/python" python3 -m weft "${project_root}/${dsl}" |
      "${compiler}" --emit=intrinsic-c --march="${target_march}" --abi=lp64d \
      --vlen-bits="${target_vlen_bits}" --matrix-extension="${matrix_extension}" \
        "${backend_arguments[@]}" -o "${local_root}/kernel.c"
  fi
  cp "${project_root}/${runtime}" "${local_root}/runtime.cpp"
fi

tar -C "${local_root}" -cf - . |
  ssh "${remote_host}" "
    set -eu
    remote_root=\$(mktemp -d /tmp/weft-kernel.XXXXXX)
    cleanup() {
      cleanup_code=\$?
      trap - EXIT
      if ! find \"\${remote_root}\" -depth -delete; then
        echo \"failed to remove remote temporary directory: \${remote_root}\" >&2
        if [ \"\${cleanup_code}\" -eq 0 ]; then
          cleanup_code=1
        fi
      fi
      exit \"\${cleanup_code}\"
    }
    trap cleanup EXIT
    tar -C \"\${remote_root}\" -xf -
    cd \"\${remote_root}\"
    cc=${remote_cc}
    cxx=${remote_cxx}
    if [ '${k_quant_dot}' -eq 1 ]; then
      "\${cc}" -O3 ${remote_compile_flags} -funroll-loops -std=c11 -Wall -Wextra -Werror \
        -march=${target_march} -mabi=lp64d -c kernel.c -o kernel.o
      "\${cxx}" -O3 ${remote_compile_flags} -funroll-loops -std=c++17 -Wall -Wextra -Werror \
        -DWEFT_QUANT_KIND=${k_quant_kind} -march=${target_march} -mabi=lp64d \
        -c runtime.cpp -o runtime.o
      "\${cxx}" -march=${target_march} -mabi=lp64d runtime.o kernel.o \
        -L/opt/tcrv-toolchains/gcc-15.2.0/lib -lm -o weft_runtime
    elif [ '${quant}' -eq 1 ]; then
      \"\${cc}\" -O3 ${remote_compile_flags} -funroll-loops -std=c11 -Wall -Wextra -Werror \
        -march=${target_march} -mabi=lp64d -c kernel.c -o kernel.o
      ar rcs libweft_kernel.a kernel.o
      \"\${cc}\" -O3 ${remote_compile_flags} -funroll-loops -std=c11 -Wall -Wextra -Werror \
        -march=${target_march} -mabi=lp64d \
        -c leaf/runtime.c -o runtime.o
      if [ '${kernel}' = q4_K_q8_K ]; then
        ggml_build=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
        \"\${cxx}\" -march=${target_march} -mabi=lp64d runtime.o libweft_kernel.a \
          -L/opt/tcrv-toolchains/gcc-15.2.0/lib \
          -L\"\${ggml_build}\" -Wl,-rpath,\"\${ggml_build}\" \
          -Wl,--no-as-needed -lggml-cpu -lggml-base -lgomp -lm -ldl -pthread \
          -o weft_runtime
      else
        \"\${cxx}\" -march=${target_march} -mabi=lp64d runtime.o libweft_kernel.a \
          -L/opt/tcrv-toolchains/gcc-15.2.0/lib -lm -o weft_runtime
      fi
    else
      \"\${cc}\" -O3 ${remote_compile_flags} -std=c11 -Wall -Wextra -Werror \
        -march=${target_march} -mabi=lp64d -c kernel.c -o kernel.o
      if [ '${multi}' -eq 1 ]; then
        \"\${cc}\" -O3 ${remote_compile_flags} -std=c11 -Wall -Wextra -Werror \
          -march=${target_march} -mabi=lp64d -c kernel_equivalent.c \
          -o kernel_equivalent.o
        ar rcs libweft_kernel.a kernel.o kernel_equivalent.o
      else
        ar rcs libweft_kernel.a kernel.o
      fi
      \"\${cxx}\" -O3 ${remote_compile_flags} -std=c++17 -Wall -Wextra -Werror \
        -march=${target_march} -mabi=lp64d -c runtime.cpp -o runtime.o
      \"\${cxx}\" -march=${target_march} -mabi=lp64d runtime.o libweft_kernel.a \
        ${remote_link_flags} \
        -o weft_runtime
    fi
    exec taskset -c ${remote_cpu} ${runtime_command}
  "
