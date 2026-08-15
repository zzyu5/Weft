#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <sg2044-rvv128|k1-rvv256|k1-ime256> <kernel> [kernel-specific arguments]" >&2
  exit 2
fi

profile=$1
kernel=$2
shift 2
usage_prefix="$0 ${profile}"
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
compiler="${project_root}/build/tools/weft-compile/weft-compile"
case "${profile}" in
  sg2044-rvv128)
    target_march=rv64gcv_zfh_zfhmin_zvfh_zvfhmin_zfa_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zicond_zawrs_zihintpause
    target_vlen_bits=128
    matrix_extension=none
    remote_host=rvv
    remote_cpu=8
    remote_cc=/opt/tcrv-toolchains/gcc-15.2.0/bin/gcc
    remote_cxx=/opt/tcrv-toolchains/gcc-15.2.0/bin/g++
    remote_compile_flags=
    remote_link_flags="-L/opt/tcrv-toolchains/gcc-15.2.0/lib -lm"
    remote_ggml_build=/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin
    remote_ggml_toolchain_lib=/opt/tcrv-toolchains/gcc-15.2.0/lib
    runtime_hardware=SG2044
    projection_scope=activation-quantize-plus-local-N16-K32-dot
    ;;
  k1-rvv256)
    target_march=rv64gcv_zfh_zvfh_zicbop_zihintpause_zba
    target_vlen_bits=256
    matrix_extension=none
    remote_host=k1
    remote_cpu=3
    remote_cc=/usr/bin/clang-18
    remote_cxx=/usr/bin/clang++-18
    remote_compile_flags=
    remote_link_flags=-lm
    remote_ggml_build=/data/build-k1-q4k/bin
    remote_ggml_toolchain_lib=
    runtime_hardware=K1/X60
    projection_scope=activation-quantize-plus-local-N16-K32-dot
    ;;
  k1-ime256)
    target_march=rv64gcv_zfh_zvfh_zicbop_zihintpause_zba
    target_vlen_bits=256
    matrix_extension=spacemit-ime1
    remote_host=k1
    remote_cpu=3
    remote_cc=/usr/bin/clang-18
    remote_cxx=/usr/bin/clang++-18
    remote_compile_flags=-fno-integrated-as
    remote_link_flags=-lm
    remote_ggml_build=/data/build-k1-q4k/bin
    remote_ggml_toolchain_lib=
    runtime_hardware=K1/X60
    projection_scope=production-activation-quantize-plus-ime1-gemm
    ;;
  *)
    echo "unsupported Weft target profile: ${profile}" >&2
    exit 2
    ;;
esac
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
      echo "usage: ${usage_prefix} add_bias" >&2
      exit 2
    fi
    dsl=examples/kernels/elementwise/add_bias.py
    runtime=examples/repro/weft/elementwise/add_bias_runtime.cpp
    ;;
  silu)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} silu" >&2
      exit 2
    fi
    dsl=examples/kernels/pointwise/silu.py
    runtime=examples/repro/weft/pointwise/silu_runtime.cpp
    ;;
  swiglu)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} swiglu" >&2
      exit 2
    fi
    dsl=examples/kernels/pointwise/swiglu.py
    runtime=examples/repro/weft/pointwise/swiglu_runtime.cpp
    ;;
  rms_norm)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} rms_norm" >&2
      exit 2
    fi
    dsl=examples/kernels/normalization/rms_norm.py
    runtime=examples/repro/weft/normalization/rms_norm_runtime.cpp
    ;;
  rms_norm_mul)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} rms_norm_mul" >&2
      exit 2
    fi
    dsl=examples/kernels/normalization/rms_norm_mul.py
    runtime=examples/repro/weft/normalization/rms_norm_mul_runtime.cpp
    multi=1
    multi_primary=rms_norm_mul_f32
    multi_equivalent=rms_norm_mul_f32_equivalent
    ;;
  layer_norm)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} layer_norm" >&2
      exit 2
    fi
    dsl=examples/kernels/normalization/layer_norm.py
    runtime=examples/repro/weft/normalization/layer_norm_runtime.cpp
    ;;
  cross_entropy)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} cross_entropy" >&2
      exit 2
    fi
    dsl=examples/kernels/normalization/cross_entropy.py
    runtime=examples/repro/weft/normalization/cross_entropy_runtime.cpp
    ;;
  rms_norm_backward)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} rms_norm_backward" >&2
      exit 2
    fi
    dsl=examples/kernels/normalization/rms_norm_backward.py
    runtime=examples/repro/weft/normalization/rms_norm_backward_runtime.cpp
    ;;
  cumsum)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} cumsum" >&2
      exit 2
    fi
    dsl=examples/kernels/reduction/cumsum.py
    runtime=examples/repro/weft/reduction/cumsum_runtime.cpp
    ;;
  segmented_scan)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} segmented_scan" >&2
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
      echo "usage: ${usage_prefix} argmax" >&2
      exit 2
    fi
    dsl=examples/kernels/reduction/argmax.py
    runtime=examples/repro/weft/reduction/argmax_runtime.cpp
    ;;
  online_softmax_summary)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} online_softmax_summary" >&2
      exit 2
    fi
    dsl=examples/kernels/reduction/online_softmax_summary.py
    runtime=examples/repro/weft/reduction/online_softmax_summary_runtime.cpp
    ;;
  predicate_reduce)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} predicate_reduce" >&2
      exit 2
    fi
    dsl=examples/kernels/reduction/predicate_reduce.py
    runtime=examples/repro/weft/reduction/predicate_reduce_runtime.cpp
    ;;
  top_k)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} top_k" >&2
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
      echo "usage: ${usage_prefix} argsort" >&2
      exit 2
    fi
    dsl=examples/kernels/selection/argsort.py
    runtime=examples/repro/weft/selection/argsort_runtime.cpp
    ;;
  nms)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} nms" >&2
      exit 2
    fi
    dsl=examples/kernels/selection/nms.py
    runtime=examples/repro/weft/selection/nms_runtime.cpp
    ;;
  top_p)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} top_p" >&2
      exit 2
    fi
    dsl=examples/kernels/selection/top_p.py
    runtime=examples/repro/weft/selection/top_p_runtime.cpp
    ;;
  ssm_conv)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} ssm_conv" >&2
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
      echo "usage: ${usage_prefix} ssm_scan" >&2
      exit 2
    fi
    dsl=examples/kernels/state/ssm_scan.py
    runtime=examples/repro/weft/state/ssm_scan_runtime.cpp
    ;;
  rwkv_wkv6)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} rwkv_wkv6" >&2
      exit 2
    fi
    dsl=examples/kernels/state/rwkv_wkv6.py
    runtime=examples/repro/weft/state/rwkv_wkv6_runtime.cpp
    ;;
  gated_linear_attention)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} gated_linear_attention" >&2
      exit 2
    fi
    dsl=examples/kernels/state/gated_linear_attention.py
    runtime=examples/repro/weft/state/gated_linear_attention_runtime.cpp
    ;;
  rwkv_wkv7)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} rwkv_wkv7" >&2
      exit 2
    fi
    dsl=examples/kernels/state/rwkv_wkv7.py
    runtime=examples/repro/weft/state/rwkv_wkv7_runtime.cpp
    ;;
  gated_delta_net)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} gated_delta_net" >&2
      exit 2
    fi
    dsl=examples/kernels/state/gated_delta_net.py
    runtime=examples/repro/weft/state/gated_delta_net_runtime.cpp
    ;;
  solve_triangular)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} solve_triangular" >&2
      exit 2
    fi
    dsl=examples/kernels/linear_algebra/solve_triangular.py
    runtime=examples/repro/weft/linear_algebra/solve_triangular_runtime.cpp
    ;;
  group_norm)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} group_norm" >&2
      exit 2
    fi
    dsl=examples/kernels/normalization/group_norm.py
    runtime=examples/repro/weft/normalization/group_norm_runtime.cpp
    ;;
  sam_relative_position)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} sam_relative_position" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/sam_relative_position.py
    runtime=examples/repro/weft/vision/sam_relative_position_runtime.cpp
    ;;
  depthwise_conv2d)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} depthwise_conv2d" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/depthwise_conv2d.py
    runtime=examples/repro/weft/vision/depthwise_conv2d_runtime.cpp
    ;;
  max_pool2d)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} max_pool2d" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/max_pool2d.py
    runtime=examples/repro/weft/vision/max_pool2d_runtime.cpp
    ;;
  bilinear_upscale)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} bilinear_upscale" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/bilinear_upscale.py
    runtime=examples/repro/weft/vision/bilinear_upscale_runtime.cpp
    ;;
  roi_align)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} roi_align" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/roi_align.py
    runtime=examples/repro/weft/vision/roi_align_runtime.cpp
    multi=1
    multi_primary=roi_align_f32
    multi_equivalent=roi_align_f32_equivalent
    ;;
  window_partition)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} window_partition" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/window_partition.py
    runtime=examples/repro/weft/vision/window_partition_runtime.cpp
    ;;
  dense_conv2d)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} dense_conv2d" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/dense_conv2d.py
    runtime=examples/repro/weft/vision/dense_conv2d_runtime.cpp
    ;;
  conv3d)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} conv3d" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/conv3d.py
    runtime=examples/repro/weft/vision/conv3d_runtime.cpp
    ;;
  conv_transpose2d)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} conv_transpose2d" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/conv_transpose2d.py
    runtime=examples/repro/weft/vision/conv_transpose2d_runtime.cpp
    ;;
  im2col_backward)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} im2col_backward" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/im2col_backward.py
    runtime=examples/repro/weft/vision/im2col_backward_runtime.cpp
    ;;
  col2im_1d)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} col2im_1d" >&2
      exit 2
    fi
    dsl=examples/kernels/vision/col2im_1d.py
    runtime=examples/repro/weft/vision/col2im_1d_runtime.cpp
    multi=1
    multi_primary=col2im_1d_f32
    multi_equivalent=col2im_1d_f32_equivalent
    ;;
  softmax)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} softmax" >&2
      exit 2
    fi
    dsl=examples/kernels/normalization/online_softmax.py
    runtime=examples/repro/weft/normalization/online_softmax_runtime.cpp
    ;;
  blocked_gemm)
    if [[ $# -ne 2 ]]; then
      echo "usage: ${usage_prefix} blocked_gemm <decode|prefill> <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/dot/blocked_gemm.py
    runtime=examples/repro/weft/dot/blocked_gemm_runtime.cpp
    runtime_arguments=("$1" "$2")
    ;;
  blocked_gemm_f32)
    if [[ $# -ne 2 ]]; then
      echo "usage: ${usage_prefix} blocked_gemm_f32 <decode|prefill> <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/dot/blocked_gemm_f32.py
    runtime=examples/repro/weft/dot/blocked_gemm_f32_runtime.cpp
    runtime_arguments=("$1" "$2")
    ;;
  mul_mat_id)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} mul_mat_id" >&2
      exit 2
    fi
    dsl=examples/kernels/dot/mul_mat_id.py
    runtime=examples/repro/weft/dot/mul_mat_id_runtime.cpp
    ;;
  out_product)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} out_product" >&2
      exit 2
    fi
    dsl=examples/kernels/dot/out_product.py
    runtime=examples/repro/weft/dot/out_product_runtime.cpp
    ;;
  contiguous_transpose)
    if [[ $# -ne 1 ]]; then
      echo "usage: ${usage_prefix} contiguous_transpose <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/permutation/contiguous_transpose.py
    runtime=examples/repro/weft/permutation/contiguous_transpose_runtime.cpp
    runtime_arguments=("$1")
    ;;
  fwht)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} fwht" >&2
      exit 2
    fi
    dsl=examples/kernels/permutation/fwht.py
    runtime=examples/repro/weft/permutation/fwht_runtime.cpp
    ;;
  adamw)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} adamw" >&2
      exit 2
    fi
    dsl=examples/kernels/optimization/adamw.py
    runtime=examples/repro/weft/optimization/adamw_runtime.cpp
    multi=1
    multi_primary=adamw_f32
    multi_equivalent=adamw_f32_equivalent
    ;;
  get_rows_q4_k)
    if [[ $# -ne 1 ]]; then
      echo "usage: ${usage_prefix} get_rows_q4_k <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/gather/get_rows_q4_k.py
    runtime=examples/repro/weft/gather/get_rows_q4_k_runtime.cpp
    runtime_arguments=("$1")
    ;;
  get_rows_f32)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} get_rows_f32" >&2
      exit 2
    fi
    dsl=examples/kernels/gather/get_rows_f32.py
    runtime=examples/repro/weft/gather/get_rows_f32_runtime.cpp
    ;;
  weighted_embedding_bag)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} weighted_embedding_bag" >&2
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
      echo "usage: ${usage_prefix} csr_spmv" >&2
      exit 2
    fi
    dsl=examples/kernels/sparse/csr_spmv.py
    runtime=examples/repro/weft/sparse/csr_spmv_runtime.cpp
    ;;
  add_id)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} add_id" >&2
      exit 2
    fi
    dsl=examples/kernels/gather/add_id.py
    runtime=examples/repro/weft/gather/add_id_runtime.cpp
    ;;
  get_rows_back)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} get_rows_back" >&2
      exit 2
    fi
    dsl=examples/kernels/gather/get_rows_back.py
    runtime=examples/repro/weft/gather/get_rows_back_runtime.cpp
    ;;
  set_rows)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} set_rows" >&2
      exit 2
    fi
    dsl=examples/kernels/gather/set_rows.py
    runtime=examples/repro/weft/gather/set_rows_runtime.cpp
    ;;
  rope_neox)
    if [[ $# -ne 1 ]]; then
      echo "usage: ${usage_prefix} rope_neox <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/rotation/rope_neox.py
    runtime=examples/repro/weft/rotation/rope_neox_runtime.cpp
    runtime_arguments=("$1")
    ;;
  qwen3vl_mrope)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} qwen3vl_mrope" >&2
      exit 2
    fi
    dsl=examples/kernels/rotation/qwen3vl_mrope.py
    runtime=examples/repro/weft/rotation/qwen3vl_mrope_runtime.cpp
    ;;
  interleaved_complex)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} interleaved_complex" >&2
      exit 2
    fi
    dsl=examples/kernels/rotation/interleaved_complex.py
    runtime=examples/repro/weft/rotation/interleaved_complex_runtime.cpp
    multi=1
    multi_primary=interleaved_complex_mul_f32
    multi_equivalent=interleaved_complex_mul_f32_equivalent
    ;;
  interleaved_rope)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} interleaved_rope" >&2
      exit 2
    fi
    dsl=examples/kernels/rotation/interleaved_rope.py
    runtime=examples/repro/weft/rotation/interleaved_rope_runtime.cpp
    ;;
  dilated_causal_conv1d)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} dilated_causal_conv1d" >&2
      exit 2
    fi
    dsl=examples/kernels/state/dilated_causal_conv1d.py
    runtime=examples/repro/weft/state/dilated_causal_conv1d_runtime.cpp
    ;;
  codebook_lookup_affine)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} codebook_lookup_affine" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/codebook_lookup_affine.py
    runtime=examples/repro/weft/quantization/codebook_lookup_affine_runtime.cpp
    multi=1
    multi_primary=codebook_lookup_affine_f32
    multi_equivalent=codebook_lookup_affine_f32_equivalent
    ;;
  flash_attention)
    if [[ $# -ne 1 ]]; then
      echo "usage: ${usage_prefix} flash_attention <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/attention/online_flash_attention.py
    runtime=examples/repro/weft/attention/online_flash_attention_runtime.cpp
    runtime_arguments=("$1")
    ;;
  flash_attn_ext)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} flash_attn_ext" >&2
      exit 2
    fi
    dsl=examples/kernels/attention/flash_attn_ext.py
    runtime=examples/repro/weft/attention/flash_attn_ext_runtime.cpp
    ;;
  csr_sparse_attention)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} csr_sparse_attention" >&2
      exit 2
    fi
    dsl=examples/kernels/attention/csr_sparse_attention.py
    runtime=examples/repro/weft/attention/csr_sparse_attention_runtime.cpp
    ;;
  causal_mask)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} causal_mask" >&2
      exit 2
    fi
    dsl=examples/kernels/attention/causal_mask.py
    runtime=examples/repro/weft/attention/causal_mask_runtime.cpp
    ;;
  q4_k_projection_ime)
    if [[ $# -ne 1 ]]; then
      echo "usage: ${usage_prefix} q4_k_projection_ime <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/ime/q4_k_projection.py
    runtime=examples/repro/weft/ime/q4_k_projection_runtime.cpp
    runtime_arguments=("$1" "${runtime_hardware}" "${projection_scope}")
    ;;
  q4_k_mul_mat_id)
    if [[ $# -ne 1 ]]; then
      echo "usage: ${usage_prefix} q4_k_mul_mat_id <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/dot/q4_k_mul_mat_id.py
    runtime=examples/repro/weft/dot/q4_k_mul_mat_id_runtime.cpp
    runtime_arguments=("$1" "${runtime_hardware}" "${projection_scope}")
    ;;
  timestep_embedding)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} timestep_embedding" >&2
      exit 2
    fi
    dsl=examples/kernels/embedding/timestep_embedding.py
    runtime=examples/repro/weft/embedding/timestep_embedding_runtime.cpp
    ;;
  q4_0_projection_ime)
    if [[ $# -ne 1 ]]; then
      echo "usage: ${usage_prefix} q4_0_projection_ime <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/ime/q4_0_projection.py
    runtime=examples/repro/weft/ime/q4_0_projection_runtime.cpp
    runtime_arguments=("$1" "${runtime_hardware}" "${projection_scope}")
    ;;
  q4_0_q8_0 | q4_1_q8_1 | q5_0_q8_0 | q5_1_q8_1 | q8_0_q8_0)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} ${kernel}" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/block_dot.py
    quant=1
    ;;
  q1_0_rows)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} q1_0_rows" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/q1_0.py
    runtime=examples/repro/weft/quantization/q1_0_runtime.cpp
    ;;
  mxfp4_rows)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} mxfp4_rows" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/mxfp4.py
    runtime=examples/repro/weft/quantization/mxfp4_runtime.cpp
    ;;
  q4_K_q8_K)
    if [[ $# -ne 3 ]]; then
      echo "usage: ${usage_prefix} q4_K_q8_K <attn_q|attn_k|attn_output|ffn_gate|ffn_up> <decode|prefill> <repetitions>" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/block_dot.py
    quant=1
    runtime_arguments=("$1" "$2" "$3")
    ;;
  quantize_q8_0)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} quantize_q8_0" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/q8_0.py
    runtime=examples/repro/weft/quantization/q8_0_runtime.cpp
    ;;
  dequantize_iq4_nl)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} dequantize_iq4_nl" >&2
      exit 2
    fi
    dsl=examples/kernels/quantization/iq4_nl.py
    runtime=examples/repro/weft/quantization/iq4_nl_runtime.cpp
    ;;
  iq2_S_q8_K | iq3_S_q8_K | iq1_M_q8_K | q6_K_q8_K)
    if [[ $# -ne 0 ]]; then
      echo "usage: ${usage_prefix} ${kernel}" >&2
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
runtime_header_flags="-include kernel.h"
if [[ ${multi} -eq 1 ]]; then
  runtime_header_flags+=" -include kernel_equivalent.h"
fi
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
      "${backend_arguments[@]}" --header="${local_root}/kernel.h" \
      -o "${local_root}/kernel.c"
  cp "${project_root}/examples/repro/weft/quantization/codebook_k_runtime.cpp" \
    "${local_root}/runtime.cpp"
  cp "${project_root}/source/c/ggml/llama.cpp/ggml/src/ggml-common.h" \
    "${local_root}/ggml-common.h"
elif [[ ${quant} -eq 1 ]]; then
  PYTHONPATH="${project_root}/python" python3 -m weft \
    "${project_root}/${dsl}" --kernel "${kernel}" |
    "${compiler}" --emit=intrinsic-c --march="${target_march}" --abi=lp64d \
      --vlen-bits="${target_vlen_bits}" --matrix-extension="${matrix_extension}" \
      "${backend_arguments[@]}" --header="${local_root}/kernel.h" \
      -o "${local_root}/kernel.c"
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
        --header="${local_root}/kernel.h" -o "${local_root}/kernel.c"
  elif [[ ${multi} -eq 1 ]]; then
    PYTHONPATH="${project_root}/python" python3 -m weft "${project_root}/${dsl}" \
      --kernel "${multi_primary}" |
      "${compiler}" --emit=intrinsic-c --march="${target_march}" --abi=lp64d \
        --vlen-bits="${target_vlen_bits}" --matrix-extension="${matrix_extension}" \
        "${backend_arguments[@]}" --header="${local_root}/kernel.h" \
        -o "${local_root}/kernel.c"
    PYTHONPATH="${project_root}/python" python3 -m weft "${project_root}/${dsl}" \
      --kernel "${multi_equivalent}" |
      "${compiler}" --emit=intrinsic-c --march="${target_march}" --abi=lp64d \
        --vlen-bits="${target_vlen_bits}" --matrix-extension="${matrix_extension}" \
        "${backend_arguments[@]}" \
        --header="${local_root}/kernel_equivalent.h" \
        -o "${local_root}/kernel_equivalent.c"
  else
    PYTHONPATH="${project_root}/python" python3 -m weft "${project_root}/${dsl}" |
      "${compiler}" --emit=intrinsic-c --march="${target_march}" --abi=lp64d \
      --vlen-bits="${target_vlen_bits}" --matrix-extension="${matrix_extension}" \
        "${backend_arguments[@]}" --header="${local_root}/kernel.h" \
        -o "${local_root}/kernel.c"
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
        -march=${target_march} -mabi=lp64d -include kernel.h \
        -c kernel.c -o kernel.o
      "\${cxx}" -O3 ${remote_compile_flags} -funroll-loops -std=c++17 -Wall -Wextra -Werror \
        -DWEFT_QUANT_KIND=${k_quant_kind} -march=${target_march} -mabi=lp64d \
        -include kernel.h \
        -c runtime.cpp -o runtime.o
      "\${cxx}" -march=${target_march} -mabi=lp64d runtime.o kernel.o \
        ${remote_link_flags} -o weft_runtime
    elif [ '${quant}' -eq 1 ]; then
      \"\${cc}\" -O3 ${remote_compile_flags} -funroll-loops -std=c11 -Wall -Wextra -Werror \
        -march=${target_march} -mabi=lp64d -include kernel.h \
        -c kernel.c -o kernel.o
      ar rcs libweft_kernel.a kernel.o
      \"\${cc}\" -O3 ${remote_compile_flags} -funroll-loops -std=c11 -Wall -Wextra -Werror \
        -D_POSIX_C_SOURCE=200809L \
        -march=${target_march} -mabi=lp64d \
        -include kernel.h \
        -c leaf/runtime.c -o runtime.o
      if [ '${kernel}' = q4_K_q8_K ]; then
        ggml_build=${remote_ggml_build}
        ggml_toolchain_lib=${remote_ggml_toolchain_lib}
        \"\${cxx}\" -march=${target_march} -mabi=lp64d runtime.o libweft_kernel.a \
          \${ggml_toolchain_lib:+-L\"\${ggml_toolchain_lib}\"} \
          -L\"\${ggml_build}\" -Wl,-rpath,\"\${ggml_build}\" \
          -Wl,--no-as-needed -lggml-cpu -lggml-base -lgomp -lm -ldl -pthread \
          -o weft_runtime
      else
        \"\${cxx}\" -march=${target_march} -mabi=lp64d runtime.o libweft_kernel.a \
          ${remote_link_flags} -o weft_runtime
      fi
    else
      \"\${cc}\" -O3 ${remote_compile_flags} -std=c11 -Wall -Wextra -Werror \
        -march=${target_march} -mabi=lp64d -include kernel.h \
        -c kernel.c -o kernel.o
      if [ '${multi}' -eq 1 ]; then
        \"\${cc}\" -O3 ${remote_compile_flags} -std=c11 -Wall -Wextra -Werror \
          -march=${target_march} -mabi=lp64d -include kernel_equivalent.h \
          -c kernel_equivalent.c \
          -o kernel_equivalent.o
        ar rcs libweft_kernel.a kernel.o kernel_equivalent.o
      else
        ar rcs libweft_kernel.a kernel.o
      fi
      \"\${cxx}\" -O3 ${remote_compile_flags} -std=c++17 -Wall -Wextra -Werror \
        -march=${target_march} -mabi=lp64d ${runtime_header_flags} \
        -c runtime.cpp -o runtime.o
      \"\${cxx}\" -march=${target_march} -mabi=lp64d runtime.o libweft_kernel.a \
        ${remote_link_flags} \
        -o weft_runtime
    fi
    exec taskset -c ${remote_cpu} ${runtime_command}
  "
