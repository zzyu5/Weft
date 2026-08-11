# GGML RISC-V LLM source

本目录只保存 llama.cpp 中真实参与 CPU/RISC-V inference 的 GGML source。文件保持上游路径
与内容，不包含生成物、函数摘录、独立改写、测试、日志或其他架构 backend。

## 模型到 kernel 的真实路径

```text
GGUF tensor type
  → ggml-cpu/ggml-cpu.c::type_traits_cpu
  → GGML_OP_MUL_MAT / GGML_OP_MUL_MAT_ID
  → target repack GEMV/GEMM（若该格式和 target 有实现）
  → 否则 ggml_compute_forward_mul_mat + 对应 ggml_vec_dot_* entry
```

因此量化格式不是独立 graph op。`quantize_row_*`、`dequantize_row_*` 和 `vec_dot` 是真实
matmul 路径内部的 helper/microkernel；不能把历史实验里的每一行都冒充一个 llama.cpp op。

## 量化 source

`ggml/src/ggml-cpu/arch/riscv/quants.c` 保存 23 个真实 RISC-V vec-dot entry：

```text
q1_0
q4_0 q4_1 q5_0 q5_1 q8_0
q2_K q3_K q4_K q5_K q6_K
iq1_s iq1_m iq2_s iq2_xs iq2_xxs iq3_s iq3_xxs iq4_nl iq4_xs
tq1_0 tq2_0 mxfp4
```

这些 entry 使用 RVV intrinsic、按 VLEN 选择的 body 或 XTheadVector 路径；不满足 target
条件时才进入同名 generic implementation。`nvfp4` 当前没有 RISC-V 专用 entry，只保留
`ggml/src/ggml-cpu/quants.c` 中真实 generic scalar implementation。

同一文件还实现 activation 侧的 `quantize_row_q8_0`、`quantize_row_q8_1`、
`quantize_row_q8_K`。所有 GGML row-dequant 与 reference quantize 位于
`ggml/src/ggml-quants.c`。

`ggml/src/ggml-cpu/arch/riscv/repack.cpp` 保存当前真实 RISC-V packed path：

```text
ggml_quantize_mat_q8_0_4x8
ggml_gemv_q4_0_8x8_q8_0       ggml_gemm_q4_0_8x8_q8_0
ggml_gemv_q4_0_16x1_q8_0      ggml_gemm_q4_0_16x1_q8_0
ggml_gemv_q4_K_16x1_q8_K      ggml_gemm_q4_K_16x1_q8_K
ggml_gemv_q2_K_16x1_q8_K      ggml_gemm_q2_K_16x1_q8_K
ggml_gemv_iq4_nl_16x1_q8_0    ggml_gemm_iq4_nl_16x1_q8_0
ggml_gemv_q8_0_16x1_q8_0      ggml_gemm_q8_0_16x1_q8_0
```

没有出现在这张表中的格式并不存在一个可伪造的 `ggml_gemm_*` 对手；其模型路径是 generic
mul-mat traversal 加真实 vec-dot entry。

## 非量化与扩展 source

- `ggml-cpu/ops.cpp`、`binary-ops.cpp`、`unary-ops.cpp`、`vec.cpp`/`vec.h`：add、mul、
  scale、copy、GELU、SiLU、softmax、RoPE、Norm、RMSNorm 及其 generic scalar/autovec/RVV
  helper；
- `ggml-cpu/spacemit/rvv_kernels.cpp`：FlashAttention、Norm/RMSNorm、F32/F16 binary、
  permuted copy/cont、repeat、sum-rows、get-rows、concat；
- `ggml-cpu/spacemit/ime1_kernels.cpp`、`ime2_kernels.cpp` 与 `repack.cpp`：真实 IME
  quantized GEMM 与布局转换。

`include/`、GGML common files、CPU traits 与 threading 文件只保留上述 source 的 ABI/依赖
闭包。`source/` 不是第二份完整 llama.cpp 仓库；完整模型 graph、loader 与 benchmark 由目标
机上的真实 llama.cpp build 提供。

模型级执行入口：

```bash
./examples/run/ggml.sh <remote-model.gguf> <prompt-tokens> <generation-tokens> <threads> <repetitions>
```

该入口在 `ssh rvv` 上用真实 GGUF 运行 `llama-bench`，同时测 prompt/prefill 与 token
generation/decode；不再使用小固定数组作为 GGML source baseline。
